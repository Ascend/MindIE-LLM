# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import gc
import copy
import math
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_npu

from .generator_backend import GeneratorBackend, MAX_KEY_LENGTH
from .torch_utils.kvcache_pool import KVCachePool
from ..utils.model_input import ModelInput
from ...utils.decorators.time_decorator import timer
from ...utils.log.error_code import ErrorCode
from ...utils.log.logging import logger
from ...utils.file_utils import standardize_path, check_file_safety
from ...utils.env import ENV

ASCEND_310B = 240


class ExpertParallelDegree(int, Enum):
    NO_EP = 0
    STATIC_EP = 1
    DYNAMIC_EP = 2
    MIX_EP = 3


class DapStreamId(int, Enum):
    BASE_STREAM = 0
    DAP_STREAM = 1
    STREAM_NUM = 2


@dataclass
class DapReqInfo:
    origin_index: int
    context_length: int
    dap_stream_id: int = DapStreamId.BASE_STREAM


@dataclass
class DapPartitionInfo:
    context_length: List[int] = field(default_factory=list)
    req_info_list: List[DapReqInfo] = field(default_factory=list)


def reorder_array(array: np.ndarray, order: List[int], axis: int = 0, position: List[int] = None):
    if array is None or len(array) == 0:
        return array
    if position is None:
        position = []
    pos_len = len(position)
    order_len = len(order)
    group_count = pos_len if pos_len > 0 else order_len
    if order_len == 0 or order == list(range(group_count)):
        return array
    tensor_group = np.array_split(array, position if pos_len > 0 else group_count, axis=axis)
    tensor_group = [tensor_group[idx] for idx in order]
    return np.concatenate(tensor_group, dtype=array.dtype, axis=axis)


def reorder_tensor(tensor: torch.Tensor, order: List[int], dim: int = 0, position: List[int] = None):
    if position is None:
        position = []
    pos_len = len(position)
    order_len = len(order)
    group_count = pos_len if pos_len > 0 else order_len
    if order_len == 0 or order == list(range(group_count)):
        return tensor
    tensor_group = torch.split(tensor, position if pos_len > 0 else tensor.size(dim) // order_len)
    tensor_group = [tensor_group[idx] for idx in order]
    return torch.concat(tensor_group).reshape_as(tensor)


def check_model_config(model_config):
    model_name = model_config.get('model_name')
    if model_name is not None and (len(model_name) < 1 or len(model_name) > MAX_KEY_LENGTH):
        message = "The length of `model_name` should be in range of [1, 256]. " \
                  "If you are using MindIE as a service framework, " \
                  "`model_name` is loaded from $BackendConfig.ModelDeployConfig.ModelConfig.modelName " \
                  "in ${MINDIE_LLM_HOME_PATH}/conf/config.json."
        logger.error(message, ErrorCode.TEXT_GENERATOR_PARAM_OUT_OF_RANGE)
        raise ValueError(message)

    max_position_embeddings = model_config.get('max_position_embeddings')
    if max_position_embeddings is not None and max_position_embeddings <= 0:
        message = "`max_position_embeddings` must be greater than 0. " \
                  "If you are using MindIE as a service framework, " \
                  "`max_position_embeddings` is derived from " \
                  "$BackendConfig.ModelDeployConfig.ModelConfig.max_position_embeddings " \
                  "in ${MINDIE_LLM_HOME_PATH}/conf/config.json."
        logger.error(message, ErrorCode.TEXT_GENERATOR_PARAM_OUT_OF_RANGE)
        raise ValueError(message)

    num_lccl_comm_shards = model_config.get("num_lccl_comm_shards", 1)
    if num_lccl_comm_shards > 65536 or num_lccl_comm_shards < 0:
        message = "`num_lccl_comm_shards` must be in the range of [0, 65536]."
        logger.error(message, ErrorCode.TEXT_GENERATOR_PARAM_OUT_OF_RANGE)
        raise ValueError(message)

    lccl_comm_shard_id = model_config.get("lccl_comm_shard_id", 0)
    if lccl_comm_shard_id >= num_lccl_comm_shards or lccl_comm_shard_id < 0:
        message = "`lccl_comm_shard_id` must be in the range of [0, `num_lccl_comm_shards`)."
        logger.error(message, ErrorCode.TEXT_GENERATOR_PARAM_OUT_OF_RANGE)
        raise ValueError(message)


STABLE_SORT_TYPE = "stable"


class GeneratorTorch(GeneratorBackend):
    """The interface class for using `torch` backend.

    The interface class exposed to third-party service frameworks in scenarios where the `torch` backend is used. It
    mainly provides forward inference and sampling functions. Its sampling function is implemented by the `sample`
    method of the base class `GeneratorBackend`.

    Attributes:
        cache_pool: A pool used for storing the kv cache.

    Args:
        model_config: A dictionary containing the model configuration as detailed in
            `mindie_llm.text_generator.utils.config.ModelConfig`.
    """
    cache_pool: KVCachePool = None

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__(model_config)
        check_model_config(model_config)

        self.tokenizer = self.model_wrapper.tokenizer
        self.device = self.model_wrapper.device
        self.rank = self.model_wrapper.rank
        self.adapter_manager = self.model_wrapper.adapter_manager
        self.mapping = self.model_wrapper.mapping
        self.llm_config = self.model_wrapper.model_runner.llm_config
        self.enable_dap = False if self.llm_config is None else self.llm_config.llm.stream_options.micro_batch
        self.enable_kv_pool = len(self.kv_pool_backend) != 0 and len(self.kv_pool_config_path) != 0

        if self.llm_config is not None and \
            self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir is not None:
            self.obfuscation_func = self._get_obfuscation_func()

        self.cp_rank = 0
        self.sp_rank = 0
        self.sp_size = 1
        self.cp_size = 1
        if self.mapping.has_attn_inner_sp():
            self.sp_size = self.mapping.attn_inner_sp.group_size
            self.sp_rank = self.mapping.attn_inner_sp.rank
        if self.mapping.has_attn_cp():
            self.cp_rank = self.mapping.attn_cp.rank
            self.cp_size = self.mapping.attn_cp.group_size

    def set_device(self):
        torch_npu.npu.set_device(self.npu_device_id)

    def to_tensor(self, data):
        return torch.tensor(data, device=self.device)

    def forward_tensor(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: bool,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_seq_len: int,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Call the `forward_tensor` of `model_wrapper`."""
        adapter_ids = kwargs.get("adapter_ids")
        batch_size = input_lengths.shape[0]
        if adapter_ids is not None and len(adapter_ids) > batch_size:
            message = "The length of `adapter_ids` should not be larger than batch size."
            logger.error(message, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            raise ValueError(message)
        logits = self.model_wrapper.forward_tensor(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=is_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_seq_len=max_seq_len,
            lm_head_indices=lm_head_indices,
            **kwargs,
        )
        return logits

    @timer.track_time('forward')
    def forward(self, model_inputs: ModelInput, **kwargs):
        if self.mapping.has_dp() and model_inputs.dp_rank_ids is None:
            error_msg = "`dp_rank_ids` is not given when data parallel size > 1."
            logger.error(error_msg, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            raise AssertionError(error_msg)

        is_dap, dap_stream_id_mask = self._partition_dap_stream_by_rank(model_inputs, **kwargs)
        if is_dap:
            logits = self._dap_forward(dap_stream_id_mask, model_inputs, **kwargs)
        else:
            logits = self._forward(model_inputs, **kwargs)
        
        return logits

    def update_cache_policy(self, kvcache_settings, sepd_worker=None):
        self.cache_pool = KVCachePool(kvcache_settings, self.device, enable_kv_pool=self.enable_kv_pool)
        self.cache_pool.allocate_cpu_kvcache()
        self.cache_pool.allocate_npu_kvcache()
        if sepd_worker:
            if self.cache_pool.use_mb_swapper:
                sepd_worker.set_npu_cache(model_id=0, npu_addrs=self.cache_pool.npu_blocks_addrs)
            else:
                if self.cache_pool.kvcache_settings.k_head_size != self.cache_pool.kvcache_settings.v_head_size:
                    if self.cache_pool.kvcache_settings.k_head_size != 0:
                        if len(self.cache_pool.k_blocks_addrs) > 0:
                            sepd_worker.set_npu_cache(model_id=0, npu_addrs=self.cache_pool.k_blocks_addrs)
                        if len(self.cache_pool.k_blocks_quant_addrs) > 0:
                            sepd_worker.set_npu_cache(model_id=2, npu_addrs=self.cache_pool.k_blocks_quant_addrs)
                    if self.cache_pool.kvcache_settings.v_head_size != 0:
                        sepd_worker.set_npu_cache(model_id=1, npu_addrs=self.cache_pool.v_blocks_addrs)
                else:
                    sepd_worker.set_npu_cache(model_id=0, npu_addrs=self.cache_pool.npu_blocks_addrs)

    def clear_kv_cache(self):
        del self.cache_pool
        torch.npu.empty_cache()
        gc.collect()

    def swap_cache(self, swap_decision):
        swap_decision_tensor = torch.tensor(swap_decision, dtype=torch.int64, device=self.device)
        self.cache_pool.swap_kvcache_method(swap_decision_tensor)

    def update_cache_after_switch_pd_role(self):
        self.cache_pool.allocate_npu_cache()

    def execute_recover_command(self, command: str) -> dict:
        '''
        Execute recover related command.
        Args:
            command (str): recover command, including "CMD_PAUSE_ENGINE".
        Returns:    
            Tuple[int, str]: (return code, error message). return code: 1 for success, 0 for failure.
        '''
        error_msg = ""
        # Recover command execution result, 0 for success, 1 for failure.
        command_result = 1
        try:
            if (command == "CMD_PAUSE_ENGINE"):
                command_result = torch_npu.npu.stop_device(self.npu_device_id)
            elif (command == "CMD_REINIT_NPU"):
                torch_npu.npu.restart_device(self.npu_device_id)
                command_result = 0
        except Exception as e:
            error_msg = f"Execute recover command {command} failed, exception msg: {e}"
            logger.error(error_msg, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            error_msg = str(e)
        ret_dict = {"command_result": command_result, "error_msg": error_msg, "npu_device_id": self.npu_device_id}
        return ret_dict

    def _sort_model_inputs_by_adapter_ids(self, model_inputs):
        adapter_ids = model_inputs.adapter_ids
        batch_size = model_inputs.context_length.shape[0]
        if len(adapter_ids) > batch_size:
            message = "The length of `adapter_ids` should not be larger than batch size."
            logger.error(message, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            raise ValueError(message)
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(
            adapter_ids, model_name=self.model_wrapper.model_name, model_id=self.model_wrapper.model_id)
        if len(effective_adapter_ids) == 1:
            model_inputs.adapter_ids = effective_adapter_ids
            return False, []
        if self.adapter_manager.check_adapter_ids_is_sorted(effective_adapter_ids):
            return False, []
        sorted_adapter_idx, revert_adapter_idx = self.adapter_manager.sort_adapter_ids(effective_adapter_ids)

        input_lengths_origin = model_inputs.context_length.cumsum()
        split_batch_group = input_lengths_origin.tolist() if model_inputs.is_prefill else []
        model_inputs.adapter_ids = [effective_adapter_ids[idx] for idx in sorted_adapter_idx]
        model_inputs.input_ids = reorder_array(
            model_inputs.input_ids, sorted_adapter_idx, position=split_batch_group)
        model_inputs.position_ids = reorder_array(
            model_inputs.position_ids, sorted_adapter_idx, position=split_batch_group)
        model_inputs.slots = reorder_array(
            model_inputs.slots, sorted_adapter_idx, position=split_batch_group)
        model_inputs.block_tables = reorder_array(model_inputs.block_tables, sorted_adapter_idx)
        model_inputs.context_length = reorder_array(model_inputs.context_length, sorted_adapter_idx)
        if model_inputs.prefill_head_indices is not None:
            lm_head_indices_check = input_lengths_origin - 1
            if np.array_equal(model_inputs.prefill_head_indices, lm_head_indices_check):
                model_inputs.prefill_head_indices = model_inputs.context_length.cumsum() - 1
            else:
                logger.warning("prefill_head_indices passed through model_inputs "
                               "is not consistent with context_length")

        return True, revert_adapter_idx

    def _lm_head_local_dp_calc(self, dp_rank_ids, dp_rank_ids_per_token, prefill_head_indices_alias):
        if prefill_head_indices_alias is None:
            return None
        num_speculative_tokens = self.model_wrapper.model_runner.num_speculative_tokens
        local_dp_idx = np.where(dp_rank_ids == self.mapping.attn_dp.rank)[0]

        # 陪跑直接返回 [0]
        if len(local_dp_idx) == 0:
            return np.array([0], dtype=np.int32)

        # 提取当前dp组的lmhead
        lm_head_tmp = prefill_head_indices_alias[local_dp_idx]

        # 更新当前dp组的lmhead
        adjusted_values = lm_head_tmp - local_dp_idx * (num_speculative_tokens + 1)
        k = np.arange(len(local_dp_idx))
        lm_head_local_dp = adjusted_values + k * (num_speculative_tokens + 1)

        return lm_head_local_dp

    def _partition_data(self, slot_dp_rank_id, dp_rank_ids_per_token, model_inputs,
                        token_size_per_request, output_token_num_per_batch):
        dp_rank_ids = model_inputs.dp_rank_ids
        cur_dp_rank_id_per_token_mask = dp_rank_ids_per_token == self.mapping.attn_dp.rank
        cur_dp_slot_mask = slot_dp_rank_id == self.mapping.attn_dp.rank
        cur_dp_rank_id_mask = dp_rank_ids == self.mapping.attn_dp.rank
        shard_input_ids = model_inputs.input_ids[cur_dp_rank_id_per_token_mask]

        if shard_input_ids.size == 0:
            # dp组未分配数据，构造陪跑数据
            shard_input_ids = np.array([1], dtype=np.int64)
            shard_position_ids = np.array([0], dtype=np.int32)
            is_prefill_with_dp = model_inputs.is_prefill
            shard_block_tables = np.array([[self.cache_pool.kvcache_settings.num_npu_blocks - 1]], dtype=np.int32)
            shard_slots = np.array(
                [(self.cache_pool.kvcache_settings.num_npu_blocks - 1) * self.cache_pool.kvcache_settings.block_size],
                dtype=np.int32,
            )
            shard_input_lengths = np.array([1], dtype=np.int32)
            shard_max_seq_len = 1
        else:
            # 获取当前dp组的数据
            shard_position_ids = model_inputs.position_ids[cur_dp_rank_id_per_token_mask]
            is_prefill_with_dp = model_inputs.is_prefill
            shard_block_tables = model_inputs.block_tables[cur_dp_rank_id_mask]
            shard_slots = model_inputs.slots[cur_dp_slot_mask]
            shard_input_lengths = model_inputs.context_length[cur_dp_rank_id_mask]
            shard_max_seq_len = max(shard_input_lengths.tolist())

        sorted_indices = np.argsort(dp_rank_ids, kind=STABLE_SORT_TYPE)
        sorted_output_token_num_per_batch = output_token_num_per_batch[sorted_indices]
        sorted_token_size_per_request = token_size_per_request[sorted_indices]

        delta_offset = np.zeros(len(model_inputs.input_ids), dtype=np.int32)
        input_lm_head = model_inputs.prefill_head_indices
        if all(x == 1 for x in sorted_output_token_num_per_batch):
            lm_head_indices = np.cumsum(sorted_token_size_per_request) - 1
            lm_head_indices_alias = np.cumsum(token_size_per_request) - 1
            delta_offset = (lm_head_indices_alias - input_lm_head) if input_lm_head is not None else delta_offset
        elif (sorted_output_token_num_per_batch == sorted_token_size_per_request).all():
            delta_offset = np.zeros(len(model_inputs.input_ids), dtype=np.int32)
            lm_head_indices = np.arange(sum(sorted_output_token_num_per_batch))
            sorted_indices = np.argsort(dp_rank_ids_per_token, kind=STABLE_SORT_TYPE)
        else:
            lm_head_indices = np.arange(sum(sorted_output_token_num_per_batch))
            lm_head_indices_alias = np.cumsum(token_size_per_request) - 1
            delta_offset = (lm_head_indices_alias - input_lm_head) if input_lm_head is not None else delta_offset
            cur_len = 0
            for i, token_num in enumerate(sorted_output_token_num_per_batch):
                cur_len += token_num
                lm_head_indices[i] = cur_len - 1

        # 基于当前的lm_head来插入陪跑数据
        lm_head_indices_with_dp = lm_head_indices
        if lm_head_indices_with_dp is None:
            lm_head_indices_with_dp = np.arange(len(model_inputs.input_ids))
        cur_idx = 0
        for i in range(self.mapping.attn_dp.group_size):
            cur_shard_input_lengths = token_size_per_request[dp_rank_ids == i]
            cur_shared_output_lengths = output_token_num_per_batch[dp_rank_ids == i]
            cur_num_request = cur_shard_input_lengths.shape[0]
            if cur_num_request == 0: # 当前进程上没有这个dp组，会构造一个陪跑数据，那么从这个位置开始往后的所有lm_head都+1
                lm_head_indices_with_dp[cur_idx:] += 1
            else:
                cur_idx += sum(cur_shared_output_lengths)
        
        # 刷新后重新反排
        reverse_indices = np.argsort(sorted_indices, kind=STABLE_SORT_TYPE)
        lm_head_indices_with_dp = lm_head_indices_with_dp[reverse_indices]

        # 基于原有输入长度计算都在最后一个输出的lmhead，和model_inputs中的原始lm_head做差值，用差值更新最新的lmhead
        # 这一步主要用于lmhead不是固定取最后一个输出的情况
        lm_head_indices_with_dp = lm_head_indices_with_dp - delta_offset

        # 额外计算一个当前dp组的lmhead输出（用于MTP的多轮抽取）
        lm_head_local_dp = self._lm_head_local_dp_calc(dp_rank_ids, dp_rank_ids_per_token,
                                                       model_inputs.prefill_head_indices)

        model_inputs.input_ids = shard_input_ids
        model_inputs.position_ids = shard_position_ids
        model_inputs.is_prefill = is_prefill_with_dp
        model_inputs.block_tables = shard_block_tables
        model_inputs.slots = shard_slots
        model_inputs.context_length = shard_input_lengths
        model_inputs.max_seq_len = shard_max_seq_len
        model_inputs.prefill_head_indices = lm_head_indices_with_dp
        if model_inputs.sp_tokens is not None:
            model_inputs.sp_tokens = model_inputs.sp_tokens[cur_dp_rank_id_mask]
        if model_inputs.cp_tokens is not None:
            model_inputs.cp_tokens = model_inputs.cp_tokens[cur_dp_rank_id_mask]

        return lm_head_local_dp

    def _gather_dp_data(self, dp_rank_ids_per_token):
        # attn dp + mlp tp场景下，构造额外输入用于tp前收集所有dp组的输入，dp前进行数据切分
        token_size_per_dp_group = np.bincount(dp_rank_ids_per_token, minlength=self.mapping.attn_dp.group_size)
        token_size_per_dp_group = np.where(token_size_per_dp_group == 0, 1, token_size_per_dp_group)

        # 用于dp前数据切分：从所有请求按dp组排列后的token index中，选取当前dp组的token index，包含陪跑数据
        start_indices = np.cumsum(token_size_per_dp_group, axis=0) - token_size_per_dp_group
        end_indices = np.cumsum(token_size_per_dp_group, axis=0)
        shard_effective_token_indices = np.arange(
            start_indices[self.mapping.attn_dp.rank], end_indices[self.mapping.attn_dp.rank],
            dtype=np.int64
        )

        max_token_size_per_dp_group = token_size_per_dp_group.max().item()
        skip_padding_token_indices = np.arange(
            self.mapping.attn_dp.group_size * max_token_size_per_dp_group, dtype=np.int64
        ).reshape((self.mapping.attn_dp.group_size, max_token_size_per_dp_group))
        token_offset_per_dp_group = np.arange(
            0, (self.mapping.attn_dp.group_size - 1) * max_token_size_per_dp_group + 1,
            step=max_token_size_per_dp_group,
            dtype=np.int64)
        token_index_with_padding = skip_padding_token_indices - token_offset_per_dp_group.reshape((-1, 1))
        padding_mask = token_index_with_padding >= token_size_per_dp_group.reshape((-1, 1))

        # 用于tp前数据汇总：包含padding token的token index（每个dp组的请求统一padding到所有请求的最大的输入长度，padding token index使用0表示）
        token_index_with_padding = token_index_with_padding[self.mapping.attn_dp.rank]
        token_index_with_padding = np.where(padding_mask[self.mapping.attn_dp.rank], 0, token_index_with_padding)

        # 用于跳过padding token的token index
        skip_padding_token_indices = skip_padding_token_indices[~padding_mask]

        return {
            "token_size_per_dp_group": token_size_per_dp_group,
            "shard_effective_token_indices": shard_effective_token_indices,
            "token_index_with_padding": token_index_with_padding,
            "skip_padding_token_indices": skip_padding_token_indices,
        }
    
    def _update_qlen_and_mask_dp_data(self, model_inputs, q_lens, spec_mask):
        filtered_mask = None
        dp_rank_ids = model_inputs.dp_rank_ids
        cur_dp_rank_id_mask = dp_rank_ids == self.mapping.attn_dp.rank
        q_lens_np = np.array(q_lens)
        q_lens_tmp = q_lens_np[cur_dp_rank_id_mask]
        if q_lens_tmp.size == 0:
            q_lens_new = np.array([1], dtype=np.int32) # q_len为1的list
            filtered_mask = spec_mask[0] if spec_mask is not None else None
        else:
            # 获取当前dp组的数据
            q_lens_new = q_lens_np[cur_dp_rank_id_mask]
            filtered_mask_rows = []
            current_row = 0
            if spec_mask is not None:
                for qlen, t in zip(q_lens, cur_dp_rank_id_mask):
                    if t:
                        filtered_mask_rows.append(spec_mask[current_row:current_row + qlen])
                    current_row += qlen
                filtered_mask = torch.cat(filtered_mask_rows, dim=0)
        q_lens_new = q_lens_new.tolist()
        return q_lens_new, filtered_mask
    
    def _update_hidden_states_dp_data(self, model_inputs, q_lens, hidden_states):
        dp_rank_ids = model_inputs.dp_rank_ids
        cur_dp_rank_id_mask = dp_rank_ids == self.mapping.attn_dp.rank
        q_lens_np = np.array(q_lens)
        q_lens_tmp = q_lens_np[cur_dp_rank_id_mask]
        if q_lens_tmp.size == 0:
            filtered_hidden_states = hidden_states[0:1]
        else:
            # 获取当前dp组的数据
            filtered_hidden_states_rows = []
            current_row = 0
            for qlen, t in zip(q_lens, cur_dp_rank_id_mask):
                if t:
                    filtered_hidden_states_rows.append(hidden_states[current_row:current_row + qlen])
                current_row += qlen
            filtered_hidden_states = torch.cat(filtered_hidden_states_rows, dim=0)

        return filtered_hidden_states
    
    def _dp_partition_data(self, model_inputs, kwargs):
        dp_rank_ids = model_inputs.dp_rank_ids
        q_lens = kwargs.get("q_lens", None)
        spec_mask = kwargs.get("spec_mask", None)
        hidden_states = kwargs.get("hidden_states", None)
        is_sub_model = kwargs.get("sub_model", False)
        if model_inputs.is_prefill:
            input_token_num_per_batch = q_lens if q_lens is not None else model_inputs.context_length
            output_token_num_per_batch = [1 for _ in range(len(model_inputs.context_length))]
        else:
            input_token_num_per_batch = \
                q_lens if q_lens is not None else [1 for _ in range(len(model_inputs.input_ids))]
            batch_size = len(model_inputs.context_length)
            output_token_num_per_batch = [1] * batch_size if is_sub_model else input_token_num_per_batch
        input_token_num_per_batch = np.array(input_token_num_per_batch)
        output_token_num_per_batch = np.array(output_token_num_per_batch)
        slot_num_per_batch = input_token_num_per_batch if not is_sub_model else (2 * (input_token_num_per_batch - 1))
        dp_rank_ids_per_token = np.repeat(dp_rank_ids, input_token_num_per_batch)
        slot_dp_rank_id = np.repeat(dp_rank_ids, slot_num_per_batch)
        lm_head_local_dp = self._partition_data(slot_dp_rank_id, dp_rank_ids_per_token, model_inputs,
                                                input_token_num_per_batch, output_token_num_per_batch)
        additional_kwargs = self._gather_dp_data(dp_rank_ids_per_token)
        kwargs.update(additional_kwargs)

        if ENV.enable_dp_move_up:
            self._get_dp_ep_padding_inputs(model_inputs.is_prefill, kwargs)

        if q_lens is not None:
            q_lens_new, spec_mask_new = self._update_qlen_and_mask_dp_data(model_inputs, q_lens, spec_mask)
            kwargs['q_lens'] = q_lens_new
            kwargs['spec_mask'] = spec_mask_new
        if hidden_states is not None:
            hidden_states_new = self._update_hidden_states_dp_data(model_inputs, q_lens, hidden_states)
            kwargs['hidden_states'] = hidden_states_new
        if not is_sub_model or model_inputs.is_prefill:
            kwargs['lm_head_indices_dp_rank_ids'] = dp_rank_ids
            kwargs['dp_rank_ids'] = torch.tensor(dp_rank_ids).to(self.device)
        if is_sub_model:
            kwargs['lm_head_local_dp'] = lm_head_local_dp
    
    def _generate_lm_head_for_cp(self, context_lens):
        # 每个seq对应的token个数
        # 单bs，不做负载均衡时的lmhead indices
        ori_prefill_head_indices_single_batch = context_lens - 1
        # 每个seq按照cp切分后的chunk的长度（切分2 * cp份）
        chunk_lens = np.ceil(context_lens / (2 * self.cp_size)).astype(np.int64)
        # pad之后每条seq的长度
        seq_lens = chunk_lens * (2 * self.cp_size)
        # 每个seq中的最后一个token在对应的chunk中的偏移量
        offsets = ori_prefill_head_indices_single_batch % chunk_lens
        # 每个seq中的最后一个token所在的chunk段
        chunk_ids = np.floor(ori_prefill_head_indices_single_batch / chunk_lens)
        # 按照cp负载均衡重新排序后的token id对应的cp rank下标
        rank_ids = np.where(
            chunk_ids >= self.cp_size,
            2 * self.cp_size - chunk_ids - 1,  # 满足条件时的计算
            chunk_ids  # 不满足条件时的计算
        )
        # 前缀和，表示相对于seq的偏移（第i+1个batch从前i个batch的总token个数处开始）
        sum_prefix_lens_per_rank = np.insert(np.cumsum(seq_lens / self.cp_size)[:-1], 0, 0)
        # lmhead indices计算结果：rank_id * rank_len + sum(prefix_seq_lens) + chunk_id * chunk_len + offset
        prefill_head_indices = np.sum(seq_lens) / self.cp_size * rank_ids \
            + sum_prefix_lens_per_rank + \
            (chunk_ids / self.cp_size).astype(np.int64) * chunk_lens + offsets
        return prefill_head_indices
    
    def _cp_partition_data(self, model_inputs: ModelInput):
        # CP 参数切分
        cp_size = self.mapping.attn_cp.group_size
        cp_rank = self.mapping.attn_cp.rank
        num_chunks = cp_size * 2 # 负载均衡下，需要将序列切 2*cp 份
        input_ids_list = []
        position_ids_list = []
        batch_seq_len = model_inputs.context_length
        cumulative_seq_len = 0
        for input_length in batch_seq_len:
            start_idx = cumulative_seq_len
            end_idx = cumulative_seq_len + input_length
            chunk_length = math.ceil(input_length / num_chunks)  # 负载均衡下，每一份子序列的长度
            former_st_idx = chunk_length * cp_rank
            former_end_idx = chunk_length * (cp_rank + 1)
            latter_st_idx = chunk_length * (num_chunks - 1 - cp_rank)
            latter_end_idx = chunk_length * (num_chunks - cp_rank)
 
            input_ids_list.extend(model_inputs.input_ids[start_idx:end_idx][former_st_idx:former_end_idx])
            input_ids_list.extend(model_inputs.input_ids[start_idx:end_idx][latter_st_idx:latter_end_idx])
 
            position_ids_list.extend(model_inputs.position_ids[start_idx:end_idx][former_st_idx:former_end_idx])
            position_ids_list.extend(model_inputs.position_ids[start_idx:end_idx][latter_st_idx:latter_end_idx])
            
            cumulative_seq_len += input_length
        
        context_length_without_padding = model_inputs.context_length - model_inputs.pad_token_count
        model_inputs.prefill_head_indices = self._generate_lm_head_for_cp(context_length_without_padding)
        model_inputs.input_ids = np.array(input_ids_list, dtype=np.int64)
        model_inputs.position_ids = np.array(position_ids_list, dtype=np.int32)
        model_inputs.max_seq_len = model_inputs.cp_tokens.max()
        model_inputs.context_length = model_inputs.cp_tokens[:, cp_rank].astype(np.int32)

    def _cp_update_context_length(self, model_inputs: ModelInput):
        all_sp_tokens = model_inputs.sp_tokens[:, self.cp_rank * self.sp_size: (self.cp_rank + 1) * self.sp_size]
        model_inputs.context_length = all_sp_tokens.sum(axis=1).astype(np.int32)

    def _sp_partition_data(self, model_inputs: ModelInput):
        all_sp_tokens = model_inputs.sp_tokens[:, self.cp_rank * self.sp_size: (self.cp_rank + 1) * self.sp_size]

        if len(all_sp_tokens) == 0:
            input_lengths_sp = np.array([1], dtype=np.int32)
        else:
            input_lengths_sp = all_sp_tokens[:, self.sp_rank]
        return input_lengths_sp

    def _warm_up(self, model_inputs: ModelInput, **kwargs) -> None:
        inference_mode = kwargs.get('inference_mode', None)

        if inference_mode.enable_prefill_pa and model_inputs.is_prefill:
            q_lens = list(map(lambda x: int(x / self.mapping.attn_cp.group_size), model_inputs.context_length.tolist()))
            kv_device = self.model_wrapper.device
            kv_dtype = self.model_wrapper.model_info.dtype
            # 300I DUO doesn't support splitfusepa
            if self.model_wrapper.model_runner.soc_info.is_300i():
                attn_mask = self.model_wrapper.model_runner.attn_mask.get_attn_mask(
                    model_inputs.max_seq_len, kv_dtype, kv_device)
                if attn_mask.numel() > 0 and len(attn_mask[0]) > 1 and attn_mask[0][1] > 0:
                    attn_mask = attn_mask * - 10000.0
                req_mask_list = []
                for _, q_len in enumerate(q_lens):
                    req_mask = attn_mask[0: q_len]
                    req_mask_list.append(req_mask)
                spec_mask = torch.cat(req_mask_list, dim=0)
            else:
                spec_mask = self.model_wrapper.model_runner.attn_mask.get_splitfuse_mask(kv_device)
            super()._warm_up(model_inputs, q_lens=q_lens, attn_mask=spec_mask, **kwargs)
        elif inference_mode.enable_decode_pa and not model_inputs.is_prefill:
            q_lens = model_inputs.context_length.tolist()
            kv_device = self.model_wrapper.device
            kv_dtype = self.model_wrapper.model_info.dtype
            model_inputs.is_need_mask = [0] * len(q_lens)
            attn_mask = self.model_wrapper.model_runner.attn_mask.get_attn_mask(
                model_inputs.max_seq_len, kv_dtype, kv_device)
            super()._warm_up(model_inputs, q_lens=q_lens, attn_mask=attn_mask, **kwargs)
        else:
            super()._warm_up(model_inputs, **kwargs)
        self.model_wrapper.model_runner.model.warmup_is_end = True

    def _update_lm_head_indices_dp_rank_ids(self, model_inputs, kwargs):
        is_sub_model = kwargs.get("sub_model", False)
        if not is_sub_model or model_inputs.is_prefill:
            dp_rank_ids = model_inputs.dp_rank_ids
            output_token_num_per_batch = np.array([1 for _ in range(len(model_inputs.context_length))])
            sorted_indices = np.argsort(dp_rank_ids, kind="stable")
            sorted_dp_rank_ids = dp_rank_ids[sorted_indices]
            sorted_output_token_num_per_batch = output_token_num_per_batch[sorted_indices]
            lm_head_indices_dp_rank_ids = np.repeat(sorted_dp_rank_ids, sorted_output_token_num_per_batch)
            additional_kwargs = {"lm_head_indices_dp_rank_ids": lm_head_indices_dp_rank_ids}
            kwargs.update(additional_kwargs)
    
    def _update_shard_effective_token_indices(self, model_inputs, kwargs):
        token_size_per_dp_group = kwargs.get("token_size_per_dp_group")
        start_indices = np.cumsum(token_size_per_dp_group, axis=0) - token_size_per_dp_group
        end_indices = np.cumsum(token_size_per_dp_group, axis=0)
        shard_effective_token_indices = np.arange(
            start_indices[self.mapping.attn_dp.rank], end_indices[self.mapping.attn_dp.rank],
            dtype=np.int64
        )
        additional_kwargs = {"shard_effective_token_indices": shard_effective_token_indices}
        kwargs.update(additional_kwargs)

    def _update_token_size_per_dp_group(self, model_inputs, kwargs):
        len_first = len(model_inputs.position_ids)
        seq_lens_flattened = [len_first] * self.mapping.attn_dp.group_size
        if model_inputs.seq_lens is not None:
            seq_lens_flattened = [max(1, sum(dp_seq_len)) for dp_seq_len in model_inputs.seq_lens]
        additional_kwargs = {"token_size_per_dp_group": np.array(seq_lens_flattened)}
        kwargs.update(additional_kwargs)

    def _update_lm_head_indices(self, model_inputs: ModelInput, model_kwargs: Dict[str, Any]):
        if model_inputs.seq_lens is None:
            return
        flatten_seq_len = []
        dp_logits_num = []

        for dp_seq_len in model_inputs.seq_lens:
            if len(dp_seq_len) == 0:
                flatten_seq_len.append(1)
                dp_logits_num.append(1)
            else:
                flatten_seq_len.extend(dp_seq_len)
                dp_logits_num.append(len(dp_seq_len))

        flatten_seq_len = np.array(flatten_seq_len)
        flatten_seq_len[0] -= 1
        model_inputs.prefill_head_indices = np.cumsum(flatten_seq_len)

        dp_logits_num = np.cumsum(dp_logits_num)
        additional_kwargs = {"dp_logits_num": dp_logits_num}
        model_kwargs.update(additional_kwargs)

    def _prepare_model_inputs(self, model_inputs: ModelInput, model_kwargs: Dict[str, Any]):
        # sort input by adapter_ids
        do_reorder_requests = False
        revert_adapter_idx = []
        adapter_ids = model_inputs.adapter_ids
        if adapter_ids is not None and self.adapter_manager is not None:
            do_reorder_requests, revert_adapter_idx = self._sort_model_inputs_by_adapter_ids(model_inputs)

        if self.mapping.has_dp() and not self.distributed_enable:
            tmp_dict = {'q_lens': model_kwargs.get('q_lens', None),
                            'sub_model': True}
            self._dp_partition_data(model_inputs, model_kwargs)
            sub_model_inputs = model_kwargs.get('sub_model_inputs', None)
            if sub_model_inputs is not None:
                self._dp_partition_data(sub_model_inputs, tmp_dict)
                model_kwargs['sub_model_inputs'] = sub_model_inputs
                lm_head_local_dp = tmp_dict.get('lm_head_local_dp', None)
                model_kwargs['lm_head_local_dp'] = lm_head_local_dp

        if self.mapping.has_attn_cp():
            if model_inputs.is_prefill:
                self._cp_partition_data(model_inputs)
            else:
                self._cp_update_context_length(model_inputs)

        if self.mapping.has_attn_inner_sp() and not model_inputs.is_prefill:
            model_kwargs['input_lengths_sp'] = self._sp_partition_data(model_inputs)
            sub_model_inputs = model_kwargs.get('sub_model_inputs', None)
            if sub_model_inputs is not None:
                model_kwargs['sub_input_lengths_sp'] = self._sp_partition_data(sub_model_inputs)
        
        has_tp = self.mapping.has_attn_tp() or self.mapping.has_attn_o_proj_tp() or \
                                self.mapping.lm_head_tp.group_size > 1

        if model_inputs.is_prefill:
            if has_tp and self.distributed_enable:
                batch_size = len(model_inputs.context_length)
                logger.debug(f'Prefill Distributed scenario and TP > 1, generate dp ep padding. '
                            f'The batch size is {batch_size}')
                self._update_lm_head_indices(model_inputs, model_kwargs)
                self._update_token_size_per_dp_group(model_inputs, model_kwargs)
                self._update_lm_head_indices_dp_rank_ids(model_inputs, model_kwargs)
                self._update_shard_effective_token_indices(model_inputs, model_kwargs)
                self._get_dp_ep_padding_inputs(model_inputs.is_prefill, model_kwargs)
        else:
            if has_tp and self.distributed_enable:
                batch_size = len(model_inputs.context_length)
                token_num_per_batch = self.num_speculative_tokens + 1

                logger.debug(f'Decode Distributed scenario and TP > 1, generate dp ep padding. '
                            f'The batch size is {batch_size}')
                dep_inputs = self._get_dp_ep_inputs(batch_size * token_num_per_batch, model_inputs.input_ids)
                data = {
                    "dep_inputs": dep_inputs
                }
                model_kwargs.update(data)

        return do_reorder_requests, revert_adapter_idx

    def _forward(self, model_inputs: ModelInput, **kwargs):
        """
        Preprocess the inputs involving multi-lora and data parallelism, and pass the processed inputs to the model
        wrapper for forward inference.
        """
        do_reorder_requests, revert_adapter_idx = self._prepare_model_inputs(model_inputs, kwargs)
        
        if not hasattr(self, 'soc_version') or self.soc_version != ASCEND_310B:
            logits = self.model_wrapper.forward(model_inputs, self.cache_pool.npu_cache, **kwargs)
            
            # sort logits back to the original order (related to lm_head_indices)
            if do_reorder_requests and ( \
                model_inputs.layerwise_disaggregated_exe_stage is None \
                or model_inputs.layerwise_disaggregated_exe_stage.end_of_generate_token):
                logits = reorder_tensor(logits, revert_adapter_idx)
            return logits
        else:
            logits, past_key_values = self.model_wrapper.forward(model_inputs, self.cache_pool.npu_cache, **kwargs)
            if (
                do_reorder_requests
                and (model_inputs.layerwise_disaggregated_exe_stage is None
                or model_inputs.layerwise_disaggregated_exe_stage.end_of_generate_token
                )
            ):
                logits = reorder_tensor(logits, revert_adapter_idx)
            return logits, past_key_values

    def _get_dp_ep_inputs(self, batch_size, input_ids):
        if self.mapping.enable_lm_head_local_tp or self.mapping.enable_o_proj_local_tp or \
                self.model_wrapper.config.ep_level == 1:
            max_decode_dp_token_size = self.max_batch_size * (self.num_speculative_tokens + 1)
            padding_batch_size = max_decode_dp_token_size - batch_size
        elif batch_size % self.mapping.attn_tp.group_size != 0:   
            padding_batch_size = self.mapping.attn_tp.group_size - batch_size % self.mapping.attn_tp.group_size
        else:
            padding_batch_size = 0
        
        attn_padding_idx = np.concatenate([
                np.arange(batch_size, dtype=np.int32),
                np.zeros(padding_batch_size, dtype=np.int32)
            ]).reshape(-1)
        if self.model_wrapper.config.ep_level == 1:
            attn_unpadding_idx = np.arange((batch_size + padding_batch_size) * \
                            self.mapping.attn_dp.group_size, dtype=np.int32)
        else:
            attn_unpadding_idx = np.arange((batch_size + padding_batch_size) // \
                            self.mapping.attn_tp.group_size, dtype=np.int32)
        ffn_padding_idx = attn_unpadding_idx
        ffn_unpadding_idx = np.arange(batch_size, dtype=np.int32)
        lm_head_skip_padding_token_indices = ffn_unpadding_idx
        if self.mapping.enable_lm_head_local_tp:
            lm_head_skip_padding_token_indices = \
                            np.arange(max_decode_dp_token_size * self.mapping.lm_head_tp.group_size, dtype=np.int32)

        tp_rank = self.mapping.attn_tp.rank
        token_size_per_tp_rank = (batch_size + padding_batch_size) // self.mapping.attn_tp.group_size
        gather_prenorm_idx = attn_padding_idx[tp_rank * token_size_per_tp_rank: (tp_rank + 1) * token_size_per_tp_rank]
        if self.mapping.enable_o_proj_local_tp:
            gather_prenorm_idx = np.arange(len(input_ids), dtype=np.int32)

        dynamic_ep_idx = np.array([1], dtype=np.int32)
        moe_idx = np.array([1], dtype=np.int32)
        post_lmhead_gather_indices = np.array([1], dtype=np.int32)
        if self.mapping.enable_lm_head_local_tp:
            post_lmhead_gather_indices = np.arange(
                len(input_ids), 
                dtype=np.int32
            )

        dep_inputs = [attn_padding_idx, attn_unpadding_idx, ffn_padding_idx,
            ffn_unpadding_idx, lm_head_skip_padding_token_indices, gather_prenorm_idx,
            dynamic_ep_idx, moe_idx, post_lmhead_gather_indices]

        return dep_inputs

    def _partition_dap_stream_by_rank(self, model_inputs, **kwargs):
        # 仅在DAP使能，且为prefill阶段，且该batch总输入长度超过2k时进行DAP双流并发
        q_lens = kwargs.get('q_lens')
        if q_lens:
            input_lengths = np.array(q_lens)
        else:
            input_lengths = np.array(model_inputs.context_length)
        is_prefill_stage = model_inputs.is_prefill
        total_input_length_exceeds_2k = np.sum(input_lengths) > 2048
        is_dap_disabled = not self.enable_dap or not is_prefill_stage or \
            not total_input_length_exceeds_2k
        if is_dap_disabled:
            return False, np.array([])
        
        if self.mapping.has_dp():
            dap_rank_ids = model_inputs.dp_rank_ids
        else:
            dap_rank_ids = np.zeros(input_lengths.size)
        batch_size = dap_rank_ids.size

        dap_partition_map = {} # key: rank_id, value: DapPartitionInfo
        for i in range(batch_size):
            rank_id = dap_rank_ids[i]
            if rank_id not in dap_partition_map:
                dap_partition_map[rank_id] = DapPartitionInfo()
                dap_partition_map[rank_id].context_length = [0] * DapStreamId.STREAM_NUM
            context_length = input_lengths[i]
            req_info = DapReqInfo(origin_index=i, context_length=context_length)
            dap_partition_map[rank_id].req_info_list.append(req_info)

        sub_stream_data_count = 0
        dap_stream_id_mask = np.zeros(batch_size)
        swap_map = {DapStreamId.BASE_STREAM: DapStreamId.DAP_STREAM, DapStreamId.DAP_STREAM: DapStreamId.BASE_STREAM}
        for dap_partition_info in dap_partition_map.values():
            dap_partition_info.req_info_list.sort(key=lambda x: x.context_length, reverse=True)
            for req_info in dap_partition_info.req_info_list:
                if dap_partition_info.context_length[DapStreamId.BASE_STREAM] <= \
                    dap_partition_info.context_length[DapStreamId.DAP_STREAM]:
                    req_info.dap_stream_id = DapStreamId.BASE_STREAM
                    dap_stream_id_mask[req_info.origin_index] = DapStreamId.BASE_STREAM
                    dap_partition_info.context_length[DapStreamId.BASE_STREAM] += req_info.context_length
                else:
                    req_info.dap_stream_id = DapStreamId.DAP_STREAM
                    dap_stream_id_mask[req_info.origin_index] = DapStreamId.DAP_STREAM
                    dap_partition_info.context_length[DapStreamId.DAP_STREAM] += req_info.context_length
            
            if dap_partition_info.context_length[DapStreamId.BASE_STREAM] < \
                dap_partition_info.context_length[DapStreamId.DAP_STREAM]:
                for req_info in dap_partition_info.req_info_list:
                    dap_stream_id_mask[req_info.origin_index] = swap_map.get(req_info.dap_stream_id)

            # 只要有一个rank无法分双流，就不分双流
            if dap_partition_info.context_length[DapStreamId.DAP_STREAM] != 0:
                sub_stream_data_count += 1

        if sub_stream_data_count == 0 or (sub_stream_data_count < math.ceil(self.mapping.attn_dp.group_size / 2)):
            return False, np.array([])

        return True, dap_stream_id_mask
    
    def _dap_partition_data(self, partition_id, partition_mask, dap_model_inputs, dap_kwargs):
        model_inputs = dap_model_inputs[partition_id]
        dap_kwargs = dap_kwargs[partition_id]
        q_lens = dap_kwargs.get("q_lens", None) 
        if q_lens:
            partition_mask_per_token = np.repeat(partition_mask, q_lens)
        else:
            partition_mask_per_token = np.repeat(partition_mask, model_inputs.context_length)
        cur_partition_id_mask = partition_mask == partition_id
        cur_partition_id_mask_per_token = partition_mask_per_token == partition_id

        # 切分dap_kwargs[q_lens]
        if q_lens:
            q_lens = np.array(q_lens, dtype=np.int64)[cur_partition_id_mask].tolist()
            dap_kwargs["q_lens"] = q_lens
        model_inputs.input_ids = model_inputs.input_ids[cur_partition_id_mask_per_token]
        model_inputs.position_ids = model_inputs.position_ids[cur_partition_id_mask_per_token]
        model_inputs.block_tables = model_inputs.block_tables[cur_partition_id_mask]
        model_inputs.slots = model_inputs.slots[cur_partition_id_mask_per_token]
        model_inputs.context_length = model_inputs.context_length[cur_partition_id_mask]
        model_inputs.max_seq_len = max(model_inputs.context_length.tolist())

        if q_lens:
            model_inputs.prefill_head_indices = np.cumsum(np.array(q_lens, dtype=np.int64)) - 1
        else:
            model_inputs.prefill_head_indices = np.cumsum(model_inputs.context_length) - 1
        if self.mapping.has_dp():
            model_inputs.dp_rank_ids = model_inputs.dp_rank_ids[cur_partition_id_mask]
        if model_inputs.adapter_ids is not None and self.adapter_manager is not None:
            model_inputs.adapter_ids = model_inputs.adapter_ids[cur_partition_id_mask]
    
    def _dap_reorder_tensor(self, dap_stream_id_mask, dap_logits):
        base_stream_batch_size = (dap_stream_id_mask == DapStreamId.BASE_STREAM).sum()
        base_stream_index = 0
        dap_stream_index = base_stream_batch_size
        need_reorder = False
        reorder_list = []
        for origin_index, dap_stream_id in enumerate(dap_stream_id_mask):
            if dap_stream_id == DapStreamId.BASE_STREAM:
                reorder_index = base_stream_index
                base_stream_index += 1
            else:
                reorder_index = dap_stream_index
                dap_stream_index += 1
            reorder_list.append(reorder_index)
            if reorder_index != origin_index:
                need_reorder = True
        
        logits = torch.cat(dap_logits, dim=0)
        if not need_reorder:
            return logits
        reordered_logits = logits[reorder_list]
        return reordered_logits
    
    def _dap_forward(self, dap_stream_id_mask, model_inputs, **kwargs):
        num_speculative_tokens = self.model_wrapper.model_runner.num_speculative_tokens
        dap_kwargs = [kwargs]
        dap_kwargs.append(copy.deepcopy(kwargs))
        dap_model_inputs = [model_inputs]
        dap_model_inputs.append(copy.deepcopy(model_inputs))
        self._dap_partition_data(DapStreamId.BASE_STREAM, dap_stream_id_mask, dap_model_inputs, dap_kwargs)
        self._dap_partition_data(DapStreamId.DAP_STREAM, dap_stream_id_mask, dap_model_inputs, dap_kwargs)

        # sort input by adapter_ids
        dap_do_reorder_requests = []
        dap_revert_adapter_idx = []
        for i, model_inputs in enumerate(dap_model_inputs):
            kwargs = dap_kwargs[i]
            adapter_ids = model_inputs.adapter_ids
            if adapter_ids is not None and self.adapter_manager is not None:
                do_reorder_requests, revert_adapter_idx = self._sort_model_inputs_by_adapter_ids(model_inputs)
                dap_do_reorder_requests.append(do_reorder_requests)
                dap_revert_adapter_idx.append(revert_adapter_idx)

            if self.mapping.has_dp() and not self.distributed_enable:
                self._dp_partition_data(model_inputs, kwargs)
        outputs = self.model_wrapper.dap_forward(dap_model_inputs, dap_kwargs, self.cache_pool.npu_cache)
        if num_speculative_tokens > 0:
            dap_logits, hidden_states = outputs
        else:
            dap_logits = outputs
        # sort logits back to the original order (related to lm_head_indices)
        for i, do_reorder_requests in enumerate(dap_do_reorder_requests):
            if do_reorder_requests:
                dap_logits[i] = reorder_tensor(dap_logits[i], dap_revert_adapter_idx[i])
                if num_speculative_tokens > 0:
                    hidden_states[i] = reorder_tensor(hidden_states[i], dap_revert_adapter_idx[i])
        
        reordered_logits = self._dap_reorder_tensor(dap_stream_id_mask, dap_logits)
        if num_speculative_tokens > 0:
            reordered_hidden_states = self._dap_reorder_tensor(dap_stream_id_mask, hidden_states)
            return (reordered_logits, reordered_hidden_states)
        return reordered_logits

    def _get_dp_ep_padding_inputs(self, is_prefill, kwargs):
        token_size_per_dp_group = kwargs['token_size_per_dp_group']
        input_ids_len = token_size_per_dp_group[self.mapping.attn_dp.rank]
        max_dp_batch_size = max(token_size_per_dp_group)
        local_token_size = token_size_per_dp_group[self.mapping.attn_dp.rank]

        if self.mapping.has_moe_ep():
            expert_parallel_degree = self.model_wrapper.config.ep_level
        else:
            expert_parallel_degree = 0

        group_size = self.mapping.attn_tp.group_size
        max_token_size_per_dp_group = np.max(token_size_per_dp_group)

        if max_token_size_per_dp_group % group_size != 0:
            padding_tmp = group_size - max_token_size_per_dp_group % group_size
            max_token_size_per_dp_group += padding_tmp
        token_size_per_dp_group_startid = np.cumsum(token_size_per_dp_group)
        token_size_per_dp_group_startid[-1] = 0

        lm_head_skip_padding_token_indices = np.concatenate([
            np.arange(j, dtype=np.int32) + max_token_size_per_dp_group * rank_id
            for rank_id, j in enumerate(token_size_per_dp_group)
        ])

        atom_dp_size = max_token_size_per_dp_group // self.mapping.attn_tp.group_size
        input_length_padding = max_token_size_per_dp_group - local_token_size
        attn_padding_idx = np.concatenate([
                np.arange(local_token_size, dtype=np.int32),
                np.zeros(input_length_padding, dtype=np.int32)
            ]).reshape(-1)

        if self.mapping.attn_o_proj_tp.group_size > 1:
            gather_prenorm_idx = np.arange(atom_dp_size, dtype=np.int32)
        else:
            gather_prenorm_idx = \
                        attn_padding_idx[self.mapping.attn_tp.rank * atom_dp_size: \
                        (self.mapping.attn_tp.rank + 1) * atom_dp_size]

        if expert_parallel_degree == ExpertParallelDegree.DYNAMIC_EP or \
        (expert_parallel_degree == ExpertParallelDegree.MIX_EP and is_prefill):
            attn_unpadding_idx = np.arange(atom_dp_size, dtype=np.int32)
            ffn_padding_idx = attn_unpadding_idx
        else:
            attn_unpadding_idx = np.concatenate(
                [np.arange(s) + max_token_size_per_dp_group * i
                    for i, s in enumerate(token_size_per_dp_group)])
            ffn_padding_idx = np.concatenate([
                np.concatenate([np.arange(j) + token_size_per_dp_group_startid[rank_id - 1],
                np.zeros(max_token_size_per_dp_group - j, dtype=np.int32)]) \
                    for rank_id, j in enumerate(token_size_per_dp_group)], axis=0)

        ffn_unpadding_idx = np.arange(token_size_per_dp_group[self.mapping.attn_dp.rank], dtype=np.int32)

        dynamic_ep_idx = np.zeros(1, dtype=np.int32)
        moe_idx = np.zeros(1, dtype=np.int32)

        if expert_parallel_degree == ExpertParallelDegree.DYNAMIC_EP or \
            (expert_parallel_degree == ExpertParallelDegree.MIX_EP and is_prefill):
            if self.mapping.attn_tp.group_size == 1:
                dynamic_ep_idx = np.arange(input_ids_len * self.model_wrapper.config.num_experts_per_tok, 
                                            dtype=np.int32)
                dynamic_ep_idx_padding = \
                    np.arange(attn_unpadding_idx.shape[0] * self.model_wrapper.config.num_experts_per_tok, 
                                dtype=np.int32)
            else:
                dynamic_ep_idx = np.arange(attn_unpadding_idx.shape[0] * self.model_wrapper.config.num_experts_per_tok, 
                                            dtype=np.int32)
                dynamic_ep_idx_padding = dynamic_ep_idx

            ep_input_length = \
                int(dynamic_ep_idx_padding.shape[0] * self._get_all2all_buffer_factor(dynamic_ep_idx_padding.shape[0]))
            all2all_padding = ep_input_length % self.mapping.moe_ep.group_size
            ep_input_length_padding = (self.mapping.moe_ep.group_size - all2all_padding) if all2all_padding != 0 else 0
            ep_input_length_padding += ep_input_length
            moe_idx = np.arange(1, ep_input_length_padding + 1, dtype=np.int32)

        post_lmhead_gather_indices = np.array([1], dtype=np.int32)
        dep_inputs = [attn_padding_idx, attn_unpadding_idx, ffn_padding_idx,
                ffn_unpadding_idx, lm_head_skip_padding_token_indices, gather_prenorm_idx,
                dynamic_ep_idx, moe_idx, post_lmhead_gather_indices]

        additional_kwargs = dict(dep_inputs=dep_inputs, max_dp_batch_size=max_dp_batch_size)

        kwargs.update(additional_kwargs)

    def _get_all2all_buffer_factor(self, length):
        all2all_buffer_factor = float(self.mapping.moe_ep.group_size)
        length = length * self.mapping.attn_dp.group_size
        if hasattr(self.model_wrapper.config, "alltoall_ep_buffer_scale_factors"):
            length_thresholds = self.model_wrapper.config.alltoall_ep_buffer_scale_factors
        else:
            length_thresholds = \
                [[1048576, 1.32], [524288, 1.4], [262144, 1.53],
                 [131072, 1.8], [32768, 3.0], [8192, 5.2], [0, 8.0]]
        for threshold in length_thresholds:
            if length >= threshold[0]:
                all2all_buffer_factor = threshold[1]
                break
        return all2all_buffer_factor

    def _get_obfuscation_func(self):
        try:
            from ai_asset_obfuscate import data_asset_obfuscation
            obfuscation_func = data_asset_obfuscation.DataAssetObfuscation(self.config.vocab_size)
            file_name_list = ["kms_ca.pem", "kms_cfs.pem", "kms_cfs.key", "kms_client_key_enc.txt", \
                "aiguard_psk", "aiguard_psk_enc.txt"]
            file_list = []
            for name in file_name_list:
                file_path = standardize_path(
                    os.path.join(self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir, name))
                check_file_safety(file_path)
                file_list.append(file_path)

            tls_info = (
                file_list[0], file_list[1], file_list[2],
                self.llm_config.llm.pmcc_obfuscation_options.kms_agent_port,
                self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir,
                file_list[3],
            )
            psk_info = (
                file_list[4],
                self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir,
                file_list[5],
            )
            obfuscation_func.set_seed_safer(tls_info, psk_info)
            return obfuscation_func
        except Exception as e:
            error_msg = f"If you enable data obfuscation, please make sure you have installed ai_asset_obfuscate, " \
                        f"and use it properly. Exception msg: {e}"
            logger.error(error_msg, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            raise RuntimeError(error_msg) from e
