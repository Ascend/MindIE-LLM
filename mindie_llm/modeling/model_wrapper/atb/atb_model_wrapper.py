# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from importlib import import_module
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import numpy as np

from atb_llm.models import InferenceMode
from atb_llm.runner.model_runner import ModelRunner
from atb_llm.utils.env import ENV
from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from atb_llm.utils.moe_utils import EPLBType, save_eplb_data
from ..model_info import ModelInfo
from ..wrapper import ModelWrapper
from ....utils.log.logging import logger

ASCEND_310B = 240


class ATBModelWrapper(ModelWrapper):
    def __init__(self,
                 rank, local_rank, world_size, npu_device_id,
                 model_id: str,
                 **kwargs,
                 ):
        self.model_id = model_id
        self.model_name = kwargs.get('model_name')
        self.soc_version = kwargs.get('soc_version', None)
        if self.soc_version is not None and self.soc_version == ASCEND_310B:
            is_flash_causal_lm = False
            enable_edge = True
        else:
            is_flash_causal_lm = True
            enable_edge = False

        self.layerwise_disaggregated = kwargs.get("layerwise_disaggregated", False)

        self.model_runner = ModelRunner(
            model_name_or_path=model_id,
            rank=rank,
            local_rank=local_rank,
            npu_id=npu_device_id,
            world_size=world_size,
            is_flash_causal_lm=is_flash_causal_lm,
            enable_edge=enable_edge,
            trust_remote_code=kwargs.get('trust_remote_code', False),
            load_tokenizer=kwargs.get('load_tokenizer', True),
            tokenizer_path=kwargs.get('tokenizer_path', None),
            inference_mode=kwargs.get("inference_mode", InferenceMode.REGRESSION),
            max_position_embeddings=kwargs.get("max_position_embeddings", None),
            lora_modules=kwargs.get("lora_modules", None),
            max_loras=kwargs.get('max_loras', 0),
            max_lora_rank=kwargs.get('max_lora_rank', 0),
            plugin_params=kwargs.get('plugin_params', None),
            num_speculative_tokens=kwargs.get('num_speculative_tokens', None),
            dp=kwargs.get("dp", -1),
            tp=kwargs.get("tp", -1),
            sp=kwargs.get("attn_inner_sp", -1),
            cp=kwargs.get("cp", -1),
            moe_tp=kwargs.get("moe_tp", -1),
            moe_ep=kwargs.get("moe_ep", -1),
            distributed_enable=kwargs.get("distributed_enable", False),
            max_batch_size=kwargs.get("max_batch_size", -1),
            models_dict=kwargs.get("models", None),
            num_lccl_comm_shards=kwargs.get("num_lccl_comm_shards", 1),
            lccl_comm_shard_id=kwargs.get("lccl_comm_shard_id", 0),
            model_role=kwargs.get('role', 'standard'),
            layerwise_disaggregated=self.layerwise_disaggregated,
            layerwise_disaggregated_role_type=kwargs.get("layerwise_disaggregated_role_type", ""),
            tls_enable=kwargs.get("interNodeTLSEnabled", '0'),
            tls_ca_path=kwargs.get("interNodeTlsCaPath", ""),
            tls_ca_file=kwargs.get("interNodeTlsCaFiles", ""),
            tls_cert=kwargs.get("interNodeTlsCert", ""),
            tls_pk=kwargs.get("interNodeTlsPk", ""),
            tls_crl_path=kwargs.get("interNodeTlsCrlPath", ""),
            tls_crl_files=kwargs.get("interNodeTlsCrlFiles", ""),
            batch_p_num=2 if kwargs.get('lwdNextPHeadPrior', False) else 1
        )
        self.config = self.model_runner.config
        self.config_dict = self.model_runner.config_dict
        self.tokenizer = self.model_runner.tokenizer
        self.process_group = self.model_runner.process_group
        self.rank = rank
        self.mapping = self.model_runner.mapping
        self.dp_size = self.mapping.attn_dp.group_size
        self.sp_size = self.mapping.attn_inner_sp.group_size
        self.cp_size = self.mapping.attn_cp.group_size

        if self.layerwise_disaggregated:
            self.in_holder = torch.tensor([0.], device='npu')

        logger.debug(f"Enter ATBModelWrapper initialization. The current rank is {self.rank}. "
                     f"The size of process group is {self.process_group.size()}.")
        self.device = self.model_runner.device

        logger.debug(f"[Config]\t>>> rank:{self.rank} load weight start...")
        try:
            self.model_runner.load_weights()
        except Exception as e:
            logger.error(f"Error in load_weights: {e}")
            raise e
        logger.debug(f"[Config]\t>>> rank:{self.rank} load weight finish")

        enable_nz = self.model_runner.enable_nz
        self.model_info = ModelInfo(self.model_runner.device,
                                    self.model_runner.kv_cache_dtype,
                                    torch.tensor([], dtype=self.model_runner.kv_cache_dtype).element_size(),
                                    self.model_runner.num_layers,
                                    self.model_runner.num_kv_heads,
                                    self.model_runner.head_size,
                                    k_head_size=self.model_runner.k_head_size,
                                    v_head_size=self.model_runner.v_head_size,
                                    enable_nz=enable_nz,
                                    kvcache_quant_layers=self.model_runner.kvcache_quant_layers)
        self.max_position_embeddings = self.model_runner.max_position_embeddings
        self.soc_info = self.model_runner.soc_info
        self.adapter_manager = self.model_runner.adapter_manager
        eplb_level = getattr(self.model_runner.model, "eplb_level", 0)
        if eplb_level == EPLBType.DYNAMIC_EPLB:
            module = import_module('atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker')
            EplbWorker = getattr(module, 'EplbWorker')
            self.eplb_worker = EplbWorker(self.model_runner, self.rank, self.model_id, self.device)

        self.model = self.model_runner.model
        self.is_multimodal = getattr(self.model, "is_multimodal", False)

    def forward(self, model_inputs, npu_cache=None, **kwargs):
        """
        Refers to a single forward pass through the network.

        Args:
            model_inputs: Inference inputs.
            npu_cache: KV cache tensor on NPU.
            kwargs: If data parallel is enabled, kwargs contains the following additional inputs.
                shard_effective_token_indices: Used for data segmentation. From the token index of all requests
                    arranged by DP group (including dummy data), select token indices for the current DP group.
                token_index_with_padding: Used for data summarization.
                    Token index of the current DP group (containing padding token).
                    Requests of each DP group are uniformly padded to the maximum input length of all requests.
                    Padding token index is represented by 0.
                skip_padding_token_indices: Used to filter padding tokens.
                    Token index of all dp groups (excluding dummy data).

        Returns:
            torch.Tensor: logits.
        """
        kwargs['layerwise_disaggregated_exe_stage'] = model_inputs.layerwise_disaggregated_exe_stage
        model_inputs, kwargs = self.prepare_model_inputs(model_inputs, **kwargs)
        result = self.forward_from_model_inputs(model_inputs, npu_cache, **kwargs)
        return result

    def prepare_model_inputs(self, model_inputs, **kwargs):
        """Do operations like H2D to prepare for the forward function."""
        model_inputs.block_tables_array = model_inputs.block_tables
        if not self.layerwise_disaggregated \
           or model_inputs.layerwise_disaggregated_exe_stage is None \
           or model_inputs.layerwise_disaggregated_exe_stage.start_exec_layer == 0:
            input_ids = torch.tensor(model_inputs.input_ids).to(self.device)
            position_ids = torch.tensor(model_inputs.position_ids, dtype=torch.int64).to(self.device)
            block_tables = torch.tensor(model_inputs.block_tables, dtype=torch.int32).to(self.device)
            slots = torch.tensor(model_inputs.slots).to(self.device)
            input_lengths = torch.tensor(model_inputs.context_length).to(self.device)
            lm_head_indices = torch.tensor(
                model_inputs.prefill_head_indices, dtype=torch.int32).to(self.device) \
                if model_inputs.prefill_head_indices is not None else None
        elif self.layerwise_disaggregated and model_inputs.layerwise_disaggregated_exe_stage.start_exec_layer != 0:
            input_ids, position_ids, block_tables, slots, input_lengths, lm_head_indices = \
                self.in_holder, self.in_holder, self.in_holder, self.in_holder, self.in_holder, self.in_holder

        token_size_per_dp_group = kwargs.get("token_size_per_dp_group")
        if token_size_per_dp_group is not None:
            kwargs["token_size_per_dp_group"] = torch.tensor(token_size_per_dp_group)
        shard_effective_token_indices = kwargs.get("shard_effective_token_indices")
        if shard_effective_token_indices is not None:
            kwargs["shard_effective_token_indices"] = torch.tensor(shard_effective_token_indices).to(self.device)
        token_index_with_padding = kwargs.get("token_index_with_padding")
        if token_index_with_padding is not None:
            kwargs["token_index_with_padding"] = torch.tensor(token_index_with_padding).to(self.device)
        skip_padding_token_indices = kwargs.get("skip_padding_token_indices")
        if skip_padding_token_indices is not None:
            kwargs["skip_padding_token_indices"] = torch.tensor(skip_padding_token_indices).to(self.device)
        k_sp_gather_indices = kwargs.get("k_sp_gather_indices")
        if k_sp_gather_indices is not None:
            kwargs["k_sp_gather_indices"] = torch.tensor(k_sp_gather_indices).to(self.device)
        input_lengths_sp = kwargs.get("input_lengths_sp")
        if input_lengths_sp is not None:
            kwargs["input_lengths_sp"] = torch.tensor(input_lengths_sp, dtype=torch.int32).to(self.device)
        sub_input_lengths_sp = kwargs.get("sub_input_lengths_sp")
        if sub_input_lengths_sp is not None:
            kwargs["sub_input_lengths_sp"] = torch.tensor(sub_input_lengths_sp, dtype=torch.int32).to(self.device)
        dep_inputs = kwargs.get("dep_inputs")
        if dep_inputs is not None:
            kwargs["dep_inputs"] = [torch.tensor(arr).to(self.device) for arr in dep_inputs]
        max_dp_batch_size = kwargs.get("max_dp_batch_size")
        if max_dp_batch_size is not None:
            kwargs["max_dp_batch_size"] = torch.tensor(max_dp_batch_size).to(self.device)


        if model_inputs.sp_computed_slots_padding_idx is not None:
            kwargs["sp_computed_slots_padding_idx"] = \
                torch.tensor(model_inputs.sp_computed_slots_padding_idx).to(self.device)
        if model_inputs.sp_computed_slots_order is not None:
            kwargs["sp_computed_slots_order"] = torch.tensor(model_inputs.sp_computed_slots_order).to(self.device)
        if model_inputs.all_rank_prefix_lens is not None:
            kwargs["all_rank_prefix_lens"] = model_inputs.all_rank_prefix_lens.tolist()
        if model_inputs.per_rank_prefix_lens is not None:
            kwargs["per_rank_prefix_lens"] = \
                torch.tensor(model_inputs.per_rank_prefix_lens, dtype=torch.int32).to(self.device)

        sub_model_inputs = kwargs.get("sub_model_inputs")
        if sub_model_inputs is not None:
            sub_input_ids = torch.tensor(sub_model_inputs.input_ids).to(self.device)
            sub_position_ids = torch.tensor(sub_model_inputs.position_ids, dtype=torch.int64).to(self.device)
            sub_slots = torch.tensor(sub_model_inputs.slots).to(self.device)
            sub_input_lengths = torch.tensor(sub_model_inputs.context_length).to(self.device)
            sub_lm_head_indices = torch.tensor(
                sub_model_inputs.prefill_head_indices, dtype=torch.int32).to(self.device) \
                if sub_model_inputs.prefill_head_indices is not None else None
            sub_block_tables = torch.tensor(sub_model_inputs.block_tables, dtype=torch.int32).to(self.device)
            sub_model_inputs.input_ids = sub_input_ids
            sub_model_inputs.position_ids = sub_position_ids
            sub_model_inputs.slots = sub_slots
            sub_model_inputs.context_length = sub_input_lengths
            sub_model_inputs.prefill_head_indices = sub_lm_head_indices
            sub_model_inputs.block_tables = sub_block_tables
            kwargs["sub_model_inputs"] = sub_model_inputs

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is not None:
            kwargs["hidden_states"] = hidden_states.to(self.device)

        lm_head_local_dp = kwargs.get("lm_head_local_dp")
        if lm_head_local_dp is not None:
            kwargs["lm_head_local_dp"] = torch.tensor(lm_head_local_dp).to(self.device)

        model_inputs.input_ids = input_ids
        model_inputs.position_ids = position_ids
        model_inputs.block_tables = block_tables
        model_inputs.slots = slots
        model_inputs.input_lengths = input_lengths
        model_inputs.lm_head_indices = lm_head_indices
        return model_inputs, kwargs

    def forward_from_model_inputs(self, model_inputs, npu_cache=None, **kwargs):
        result = self.forward_tensor(
            input_ids=model_inputs.input_ids,
            position_ids=model_inputs.position_ids,
            is_prefill=model_inputs.is_prefill,
            kv_cache=npu_cache,
            block_tables=model_inputs.block_tables,
            slots=model_inputs.slots,
            input_lengths=model_inputs.input_lengths,
            max_seq_len=model_inputs.max_seq_len,
            lm_head_indices=model_inputs.lm_head_indices,
            adapter_ids=model_inputs.adapter_ids,
            is_need_mask=model_inputs.is_need_mask,
            **kwargs,
        )
        return result

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
            is_need_mask: Optional[List[int]] = None,
            **kwargs):
        from mindie_llm.utils.prof.profiler import span_start, span_end, span_attr, tensor_attr, Level
        prof = span_start(name="forward_tensor", level=Level.DETAILED)
        prof = span_attr(prof, "input_ids", lambda: tensor_attr(input_ids, False))
        prof = span_attr(prof, "position_ids", lambda: tensor_attr(position_ids, False))
        prof = span_attr(prof, "is_prefill", is_prefill)
        prof = span_attr(prof, "block_tables", lambda: tensor_attr(block_tables, False))
        prof = span_attr(prof, "slots", lambda: tensor_attr(slots, False))
        prof = span_attr(prof, "input_lengths", lambda: tensor_attr(input_lengths, False))
        prof = span_attr(prof, "max_seq_len", int(max_seq_len))
        prof = span_attr(prof, "lm_head_indices", None if lm_head_indices is None else tensor_attr(lm_head_indices, False))

        if self.soc_version is not None and self.soc_version == ASCEND_310B:
            attention_mask = kwargs.get('spec_mask', None)
            past_key_values = kwargs.get('past_key_values', None)
            if is_prefill:
                past_key_values = None
            try:
                result, past_key_values = self.model_runner.forward(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids.unsqueeze(0),
                    past_key_values=past_key_values
                )
            except Exception as e:
                logger.error(f"Error in forward_tensor: {e}")
                raise e

            span_end(prof)
            return result, past_key_values
        else:
            try:
                result = self.model_runner.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    is_prefill=is_prefill,
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                    max_seq_len=max_seq_len,
                    lm_head_indices=lm_head_indices,
                    is_need_mask=is_need_mask,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Error in forward_tensor: {e}")
                raise e
            eplb_level = getattr(self.model_runner.model, "eplb_level", 0)
            if ENV.enable_expert_hotpot_gather or eplb_level == EPLBType.DYNAMIC_EPLB:
                EplbExpertDataCollect().accumulation_expert_cumsum(is_prefill=is_prefill)
                if is_prefill:
                    save_eplb_data(
                                self.rank,
                                EplbExpertDataCollect().get_prefill_token_num_per_expert(),
                                "prefill", EplbExpertDataCollect().prefill_forward_count
                                )
                else:
                    save_eplb_data(
                                self.rank,
                                EplbExpertDataCollect().get_decode_token_num_per_expert(),
                                "decode", EplbExpertDataCollect().decode_forward_count
                                )
            topk_output = getattr(self.model_runner.model, "topk_output", False)
            if topk_output:
                if is_prefill:
                    save_eplb_data(
                                self.rank,
                                EplbExpertDataCollect().get_topk(),
                                "prefill", EplbExpertDataCollect().prefill_forward_count, True
                                )
                else:
                    save_eplb_data(
                                self.rank,
                                EplbExpertDataCollect().get_topk(),
                                "decode", EplbExpertDataCollect().decode_forward_count, True
                                )
            if eplb_level == EPLBType.DYNAMIC_EPLB:
                self.eplb_worker.eplb_forwarder.do_aggregate()

            span_end(prof)
            return result

    def dap_forward(self, dap_model_inputs, dap_kwargs, npu_cache=None):
        input_ids = []
        position_ids = []
        is_prefill = []
        block_tables = []
        slots = []
        input_lengths = []
        max_seq_len = []
        lm_head_indices = []

        for i, model_inputs in enumerate(dap_model_inputs):
            kwargs = dap_kwargs[i]
            input_ids.append(torch.tensor(model_inputs.input_ids).to(self.device))
            position_ids.append(torch.tensor(model_inputs.position_ids, dtype=torch.int64).to(self.device))
            is_prefill.append(model_inputs.is_prefill)
            block_tables.append(torch.tensor(model_inputs.block_tables, dtype=torch.int32).to(self.device))
            slots.append(torch.tensor(model_inputs.slots).to(self.device))
            input_lengths.append(torch.tensor(model_inputs.context_length).to(self.device))
            max_seq_len.append(model_inputs.max_seq_len)
            lm_head_indices.append(torch.tensor(
                model_inputs.prefill_head_indices, dtype=torch.int32).to(self.device) \
                if model_inputs.prefill_head_indices is not None else None)
            kwargs['adapter_ids'] = model_inputs.adapter_ids

            token_size_per_dp_group = kwargs.get("token_size_per_dp_group")
            if token_size_per_dp_group is not None:
                kwargs["token_size_per_dp_group"] = torch.tensor(token_size_per_dp_group)
            shard_effective_token_indices = kwargs.get("shard_effective_token_indices")
            if shard_effective_token_indices is not None:
                kwargs["shard_effective_token_indices"] = torch.tensor(shard_effective_token_indices).to(self.device)
            token_index_with_padding = kwargs.get("token_index_with_padding")
            if token_index_with_padding is not None:
                kwargs["token_index_with_padding"] = torch.tensor(token_index_with_padding).to(self.device)
            skip_padding_token_indices = kwargs.get("skip_padding_token_indices")
            if skip_padding_token_indices is not None:
                kwargs["skip_padding_token_indices"] = torch.tensor(skip_padding_token_indices).to(self.device)

            dep_inputs = kwargs.get("dep_inputs")
            if dep_inputs is not None:
                kwargs["dep_inputs"] = [torch.tensor(arr).to(self.device) for arr in dep_inputs]
            max_dp_batch_size = kwargs.get("max_dp_batch_size")
            if max_dp_batch_size is not None:
                kwargs["max_dp_batch_size"] = torch.tensor(max_dp_batch_size).to(self.device)

        try:
            dap_logits = self.model_runner.dap_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                is_prefill=is_prefill,
                kv_cache=npu_cache,
                block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                max_seq_len=max_seq_len,
                lm_head_indices=lm_head_indices,
                dap_kwargs=dap_kwargs,
            )
        except Exception as e:
            logger.error(f"Error in dap_forward: {e}")
            raise e
        return dap_logits

    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        try:
            position_ids = self.model_runner.generate_position_ids(input_ids)
        except Exception as e:
            logger.error(f"Error in generate_position_ids: {e}")
            raise e
        return position_ids

    def make_context(self, conversation: List[Dict[str, str]], **kwargs):
        try:
            context = self.model_runner.input_builder.make_context(self.rank, conversation, **kwargs)
        except Exception as e:
            logger.error(f"Error in make_context: {e}")
            raise e
        return context

    def resume_hccl_comm(self):
        self.model_runner.resume_hccl_comm()