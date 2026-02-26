# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import math
import os
from pathlib import Path
import queue
import time
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch

import acl
from mindie_llm.connector.common.model_execute_data_pb2 import LoraOperationStatus
from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
from mindie_llm.text_generator.utils.npu_mem_tool import (
    calc_npu_mem,
    calc_block_mem,
    gb,
    NpuMemoryWatcher
)
from mindie_llm.text_generator.utils.tg_infer_context_store import TGInferContextStore
from mindie_llm.modeling.backend_type import BackendType

from .adapter import get_generator_backend, parse_config, ParseType
from .plugins import get_plugin, InferenceMode, PluginParameterValidator
from .utils.block_copy import BlockCopy
from .utils.config import CacheConfig, ModelConfig, ContextParams
from .utils.input_metadata import InputMetadata
from .utils.output_filter import OutputFilter
from .utils.request import Request
from .utils.generation_output import GenerationOutput
from ..utils.decorators.time_decorator import timer
from ..utils.env import ENV
from ..utils.log.error_code import ErrorCode
from ..utils.log.logging import logger, print_log
from ..utils.status import MindieLlmStatusCode
from ..utils.tensor import npu
from .utils.separate_deployment_engine import SeparateDeploymentWorker, LinkParams, DmiModeNodeRole
from ..utils import file_utils

DECODER_TAG = 'decoder'
PREFILL_TAG = 'prefill'
STANDARD_TAG = 'standard'
NPU_OUT_OF_MEMORY_TAG = 'NPU out of memory'
EOS_TOKEN_ID = 'eos_token_id'
KV_HALF_BYTE = 4
MEM_POOL_WORKER_ROLE = "worker"


class PDModelConfig:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.model_role = config.get('role', STANDARD_TAG)
        self.local_cluster_id = int(config.get('local_instance_id', 0))
        self.local_device_ip = config.get('local_device_ip', None)
        self.local_logic_device_id = int(config.get('npu_device_id', 0))
        self.local_physical_device_id = int(config.get('local_physical_device_id', 0))
        self.local_host_ip = config.get('local_host_ip', None)
        self.remote_device_ips = config.get('remote_device_ips', '').split(',')
        self.local_super_pod_id = config.get('local_super_pod_id', None)
        self.local_super_device_id = config.get('local_super_device_id', None)
        self.kv_trans_timeout = int(config.get('kv_trans_timeout', 1))
        self.kv_rdma_sl = int(config.get('kv_rdma_sl', -1))
        self.kv_rdma_tc = int(config.get('kv_rdma_tc', -1))
        self.kv_link_timeout = int(config.get('kv_link_timeout', 1080))
        if self.local_super_device_id is not None:
            self.local_super_device_id = int(self.local_super_device_id)
            self.local_super_pod_id = int(self.local_super_pod_id)


class PDInterface:
    """
    PDInterface 作为具体实现的基类，将 PD 分离相关的接口和参数抽取在此，
    包括 _init_sepd_engine、link、unlink、switch_role 以及 pull_kv 接口。
    """

    def __init__(self, pd_config: PDModelConfig) -> None:
        self.pd_config = pd_config
        self.separate_deployment_worker = None
        self.device_inited = False
        self.input_metadata_queue = queue.Queue()
        self.dst_block_table = []

    def link(self, **kwargs) -> List[Tuple[str, ErrorCode]]:
        """创建与远端设备的链接。"""
        params = LinkParams(**kwargs)
        return self.separate_deployment_worker.link(remote_cluster_ids=params.remote_cluster_ids, 
                                                    remote_physical_device_ids=params.remote_physical_device_ids, 
                                                    remote_device_ips=params.remote_device_ips, 
                                                    host_ips=params.host_ips, 
                                                    remote_super_device_ids=params.remote_super_device_ids,
                                                    remote_super_pod_ids=params.remote_super_pod_ids
                                                    )

    def unlink(self, remote_cluster_id: int) -> Union[MindieLlmStatusCode, ErrorCode]:
        """断开与远端设备的链接。"""
        return self.separate_deployment_worker.unlink(remote_cluster_id)

    def switch_role(self, role: str) -> None:
        self.pd_config.model_role = role

    def pull_kv(self, input_metadata: InputMetadata,
                p_d_infos: List[Tuple[int, List[int], List[int]]]
                ) -> Tuple[Union[MindieLlmStatusCode, ErrorCode], int]:
        """
        从远程模型实例拉取 kv cache，并将 input_metadata 放入队列。
        """
        if not self.device_inited:
            npu.set_device(int(self.pd_config.local_logic_device_id))
            self.device_inited = True
        for x in p_d_infos:
            rt = self.separate_deployment_worker.pull_blocks(
                remote_model_instance_id=x[0],
                src_block_table=x[1],
                dst_block_table=x[2]
            )
            self.dst_block_table = x[2]
            if rt != MindieLlmStatusCode.SUCCESS:
                logger.error(f"Pull blocks from remote cluster id: {x[0]} failed, error code is {rt}")
                return rt, x[0]
        self.input_metadata_queue.put(input_metadata)
        return MindieLlmStatusCode.SUCCESS, 0

    def _init_sepd_engine(self) -> None:
        """根据角色初始化分离部署引擎。"""
        role_info = [DmiModeNodeRole.DECODER, DmiModeNodeRole.PREFILL, DmiModeNodeRole.FLEX]
        if self.pd_config.model_role in role_info:
            self.separate_deployment_worker = SeparateDeploymentWorker(
                role=self.pd_config.model_role,
                local_logic_device_id=self.pd_config.local_logic_device_id,
                local_physical_device_id=self.pd_config.local_physical_device_id,
                local_cluster_id=self.pd_config.local_cluster_id,
                local_device_ip=self.pd_config.local_device_ip,
                local_host_ip=self.pd_config.local_host_ip,
                local_super_pod_id=self.pd_config.local_super_pod_id,
                local_super_device_id=self.pd_config.local_super_device_id,
                kv_trans_timeout=self.pd_config.kv_trans_timeout,
                kv_link_timeout=self.pd_config.kv_link_timeout,
                kv_rdma_sl=self.pd_config.kv_rdma_sl,
                kv_rdma_tc=self.pd_config.kv_rdma_tc
            )


class Generator(PDInterface):
    """A class for generating tokens using a large language model.

    The primary interface class. Upper-level applications can create instances of the `Generator` to perform actions
    such as warm-up and token generation.

    Args:
        model_config: A dictionary or `ModelConfig` instance containing model configuration parameters. Refer to the
            definition of `ModelConfig` for the specific meaning of each parameter.
    """

    def __init__(self, model_config: Union[Dict[str, Any], ModelConfig]) -> None:
        if isinstance(model_config, ModelConfig):
            model_config = vars(model_config)
        self.model_config = model_config
        self.cpu_mem = parse_config(model_config, 'cpu_mem', required=True, parse_type=ParseType.TO_INT)
        self.npu_mem = parse_config(model_config, 'npu_mem', required=True, parse_type=ParseType.TO_INT)
        self.block_size = parse_config(model_config, 'block_size', required=True, parse_type=ParseType.TO_INT)
        max_seq_len = parse_config(model_config, 'max_seq_len', required=True, parse_type=ParseType.TO_INT)
        max_iter_times = parse_config(model_config, 'max_iter_times', required=True, parse_type=ParseType.TO_INT)
        max_input_len = parse_config(model_config, 'max_input_len', required=True, parse_type=ParseType.TO_INT)
        self.max_batch_size = parse_config(model_config, 'max_batch_size', required=True, parse_type=ParseType.TO_INT)
        self.max_prefill_batch_size = parse_config(model_config, 'max_prefill_batch_size',
                                                   required=True, parse_type=ParseType.TO_INT)
        max_prefill_tokens = parse_config(model_config, 'max_prefill_tokens', required=True,
                                          parse_type=ParseType.TO_INT)
        self.soc_version = parse_config(model_config, 'soc_version', required=False,
                                        parse_type=ParseType.TO_INT)
        self.max_seq_len = max_seq_len
        self.max_iter_times = max_iter_times
        self.max_input_len = max_input_len
        self.max_prefill_tokens = max_prefill_tokens
        ignore_eos = parse_config(model_config, 'ignore_eos', parse_type=ParseType.TO_BOOL)
        self.tokenizer_sliding_window_size = parse_config(model_config, 'tokenizer_sliding_window_size',
                                                     parse_type=ParseType.TO_INT, default_value=3)
        self.trust_remote_code = parse_config(model_config, 'trust_remote_code', required=True,
                                              parse_type=ParseType.TO_BOOL, default_value=False)
        self.distributed_enable = parse_config(model_config, 'distributed_enable', required=False, 
                                                parse_type=ParseType.TO_BOOL, default_value=False)

        self.pd_config = PDModelConfig(model_config)
        super().__init__(self.pd_config)

        self.is_mix_model = parse_config(model_config, 'enable_split', parse_type=ParseType.TO_BOOL)
        self.device_inited = False
        self.separate_deployment_worker = None

        self.watcher = NpuMemoryWatcher()
        self.input_metadata_queue = queue.Queue()

        plugin_params = parse_config(model_config, 'plugin_params')
        speculation_gamma = parse_config(model_config, 'speculation_gamma', 
                                         parse_type=ParseType.TO_INT, default_value=0)
        self.max_generated_tokens = speculation_gamma + 1 if speculation_gamma > 0 else 1
        validator = PluginParameterValidator(speculation_gamma)
        plugin_config, self.is_mix_model, plugin_list = validator.validate(plugin_params)
        self.num_speculative_tokens = plugin_config.get('num_speculative_tokens', 0)
        self.enable_mtp = 'mtp' in plugin_list
        self.enable_warmup_with_sampling = parse_config(model_config, 'enable_warmup_with_sampling',
                                                    parse_type=ParseType.TO_BOOL, default_value=True)
        model_config['num_speculative_tokens'] = self.num_speculative_tokens
        model_config['inference_mode'] = InferenceMode(plugin_list, plugin_config, self.is_mix_model)
        self.inference_mode = model_config['inference_mode']

        self.backend_type = parse_config(model_config, 'backend_type', required=True, default_value='atb')
        self.rank = parse_config(model_config, 'rank', required=True, parse_type=ParseType.TO_INT)
        self.world_size = parse_config(model_config, 'world_size', required=True, parse_type=ParseType.TO_INT)
        self.local_rank = parse_config(model_config, 'local_rank', required=True, parse_type=ParseType.TO_INT)
        self.npu_device_id = parse_config(model_config, 'npu_device_id', required=True, parse_type=ParseType.TO_INT)

        async_inference_key = 'async_inference'
        model_config[async_inference_key] = ENV.async_inference
        self.async_inference = model_config[async_inference_key]
        if self.async_inference:
            logger.info('Async inference is activated.')
            validator.check_async_inference_and_plugin_type(True, plugin_config.get("plugin_type"))
        model_config['splitfuse_enabled'] = self.is_mix_model

        self.layerwise_disaggregated = parse_config(model_config, 'layerwiseDisaggregated', required=False,
            parse_type=ParseType.TO_BOOL, default_value=False)
        self.layerwise_disaggregated_role_type = parse_config(model_config, 'layerwiseDisaggregatedRoleType',
            default_value="")
        self.lwd_multi_nodes_enable = parse_config(model_config, 'lwd_multi_nodes_enable', required=False,
            parse_type=ParseType.TO_BOOL, default_value=False)

        model_config["layerwise_disaggregated"] = self.layerwise_disaggregated
        model_config["layerwise_disaggregated_role_type"] = self.layerwise_disaggregated_role_type

        self.generator_backend = get_generator_backend(model_config)
        self.model_wrapper = self.generator_backend.model_wrapper
        self.sampler = self.generator_backend.sampler
        self.model_info = self.generator_backend.model_info
        self.tokenizer = self.generator_backend.tokenizer
        self.max_position_embeddings = self.generator_backend.max_position_embeddings
        self.to_tensor = self.generator_backend.to_tensor
        self.vocab_size = self.model_wrapper.config.vocab_size
        self.warmup_topk_size = getattr(self.model_wrapper.config, 'top_k', 1000)
        self.enable_dap = self.generator_backend.enable_dap
        self.obfuscation_func = self.generator_backend.obfuscation_func

        self.dp_size = self.model_wrapper.dp_size
        self.sp_size = self.model_wrapper.sp_size
        self.cp_size = self.model_wrapper.cp_size
        self.scp_size = self.cp_size * self.sp_size

        self.cache_config = CacheConfig(
            ignore_eos=ignore_eos,
            max_block_size=self.block_size,
            max_gen_len=min(max_iter_times, max_seq_len),
            max_seq_len=max_seq_len,
            model_wrapper_config=self.model_wrapper.config,
            rank=self.rank,
            tokenizer_sliding_window_size=self.tokenizer_sliding_window_size,
            vocab_size=self.vocab_size
        )

        # remove the benchmark_file first every time when service starts
        if self.rank == 0 and os.path.exists(ENV.benchmark_filepath):
            ENV.benchmark_filepath = file_utils.standardize_path(ENV.benchmark_filepath)
            file_utils.check_path_permission(ENV.benchmark_filepath)
            Path(ENV.benchmark_filepath).unlink(missing_ok=True)
        
        if getattr(self.model_wrapper.config, EOS_TOKEN_ID, None) is not None:
            self.cache_config.set_eos_token_id(self.model_wrapper.config.eos_token_id)
        elif getattr(self.tokenizer, EOS_TOKEN_ID, None) is not None:
            self.cache_config.set_eos_token_id(self.tokenizer.eos_token_id)
        if getattr(self.cache_config, EOS_TOKEN_ID, None) is not None and self.obfuscation_func is not None:
            if isinstance(self.cache_config.eos_token_id, list):
                eos_token_id_obf = [self.obfuscation_func.token_obf(eos) for eos in self.cache_config.eos_token_id]
            else:
                eos_token_id_obf = self.obfuscation_func.token_obf(self.cache_config.eos_token_id)
            self.cache_config.set_eos_token_id(eos_token_id_obf)
        print_log(self.rank, logger.info, f'The effective eos_token_id is `{self.cache_config.eos_token_id}`.')

        if getattr(self.model_wrapper.config, 'pad_token_id', None) is not None:
            self.cache_config.set_pad_token_id(self.model_wrapper.config.pad_token_id)
        elif getattr(self.tokenizer, 'pad_token_id', None) is not None:
            self.cache_config.set_pad_token_id(self.tokenizer.pad_token_id)
        if self.obfuscation_func is not None:
            self.cache_config.set_pad_token_id(self.obfuscation_func.token_obf(self.cache_config.pad_token_id))
        print_log(self.rank, logger.info, f'The effective pad_token_id is `{self.cache_config.pad_token_id}`.')

        if getattr(self.model_wrapper.config, 'bos_token_id', None) is not None:
            self.cache_config.set_bos_token_id(self.model_wrapper.config.bos_token_id)
        elif getattr(self.tokenizer, 'bos_token_id', None) is not None:
            self.cache_config.set_bos_token_id(self.tokenizer.bos_token_id)
        if self.obfuscation_func is not None:
            self.cache_config.set_bos_token_id(self.obfuscation_func.token_obf(self.cache_config.bos_token_id))
        print_log(self.rank, logger.info, f'The effective bos_token_id is `{self.cache_config.bos_token_id}`.')

        self.generator_backend.init_sampler(self.cache_config.eos_token_id)

        if self.backend_type == 'atb':
            lora_adapter = self.model_wrapper.model_runner.lora_adapter
            if lora_adapter is not None and self.is_mix_model:
                message = ("SplitFuse is not supported when LoRA is enabled!"
                           " If you want to enable SplitFuse only, please ensure that there is no file"
                           " named 'lora_adapter.json' in the model path. Or if LoRA is exactly what you need,"
                           " try to clear 'plugin_params' and to make sure 'templateType' is 'Standard'.")
                logger.error(message, ErrorCode.TEXT_GENERATOR_FEAT_COMPAT_INVALID)
                raise NotImplementedError(message)

        self.hidden_size = getattr(self.model_wrapper.config, 'hidden_size', 0)

        mapping = None
        spcp_parallel_info = None
        if hasattr(self.model_wrapper, "mapping"):
            mapping = self.model_wrapper.mapping
            spcp_parallel_info = (
                mapping.attn_inner_sp,
                mapping.attn_cp,
            )

        self.is_multimodal = self.model_wrapper.is_multimodal
        if self.is_multimodal and "memory_decoding" in plugin_list:
            message = ("Memory decoding is not supported when the model type is multimodal.")
            logger.error(message, ErrorCode.TEXT_GENERATOR_FEAT_COMPAT_INVALID)
            raise NotImplementedError(message)

        logger.debug(f"[Config]\t>>> rank:{self.rank} Warm up inference start...")
        self.is_separated_pd = self.pd_config.model_role in [PREFILL_TAG, DECODER_TAG]
        self.context_params = ContextParams(
            self.is_separated_pd,
            self.num_speculative_tokens,
            self.hidden_size,
            self.model_wrapper.model_info.dtype,
            self.enable_mtp,
            self.async_inference,
            self.distributed_enable,
            self.max_generated_tokens,
            {"atb": BackendType.ATB, "ms": BackendType.MS}.get(self.backend_type, BackendType.TORCH),
            self.layerwise_disaggregated,
            self.layerwise_disaggregated_role_type
        )
        self.kvcache_settings = self.warm_up(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
        logger.debug(f"[Config]\t>>> rank:{self.rank} Warm up inference finish")

        self.copy_blocks_ops = None
        self.infer_context = TGInferContextStore(
            self.kvcache_settings,
            self.cache_config,
            spcp_parallel_info,
            self.model_wrapper.model_info.device,
            self.context_params,
            self.tokenizer,
            self.tokenizer_sliding_window_size,
            self.model_wrapper.generate_position_ids
        )
        self.output_filter = OutputFilter(self.cache_config, self.infer_context, self.tokenizer, self.async_inference)
        plugin_utils = (
            self.generator_backend,
            self.kvcache_settings,
            self.infer_context,
            self.output_filter,
            self.pd_config.model_role,
        )
        plugin_config['eos_token_id'] = self.cache_config.eos_token_id
        plugin_config['cache_size'] = self.cache_config.cache_size
        plugin_config['max_gen_len'] = self.cache_config.max_gen_len
        plugin_config['max_block_size'] = self.cache_config.max_block_size
        plugin_config['max_seq_len'] = self.cache_config.max_seq_len
        plugin_config['device'] = self.model_wrapper.model_info.device
        plugin_config["layerwise_disaggregated"] = self.layerwise_disaggregated
        plugin_config["layerwise_disaggregated_role_type"] = self.layerwise_disaggregated_role_type

        total_mem, warmup_mem = self.watcher.watch_npu_mem(self.rank, f'After warmup', 
                                                           self.is_multimodal, max_input_len)
        self.watcher._set_warmup_mem_(warmup_mem)

        self.plugin = get_plugin(plugin_list, plugin_config, plugin_utils, self.is_mix_model, self.watcher)
        self.plugin.initialize()
        self.is_inference_pause = False

        if self.pd_config.model_role == DmiModeNodeRole.DECODER and \
            len(self.generator_backend.kv_pool_backend) != 0 and len(self.generator_backend.kv_pool_config_path) != 0:
            from mindie_llm.text_generator.mempool import MemPool
            self.m_store = MemPool.create_pool(
                backend=self.generator_backend.kv_pool_backend,
                config_path=self.generator_backend.kv_pool_config_path,
                role=MEM_POOL_WORKER_ROLE,
                device_id=self.model_wrapper.device.index,
                kv_caches=self.generator_backend.cache_pool.npu_cache
            )

    def __del__(self):
        if self.separate_deployment_worker is not None:
            self.separate_deployment_worker.finalize()

    def build_inputs(self, conversations: List[List[Dict[str, str]]], **kwargs) -> List[List[int]]:
        """Convert the multi-turn dialogues of all requests in a batch into token ids based on the template."""
        return [self.model_wrapper.make_context(conversation, **kwargs) for conversation in conversations]

    def clear_cache(self, sequence_ids: npt.NDArray[np.int64]):
        """Clear the cache of certain `Request` objects."""
        logger.debug(f"Rank-{self.rank} is clearing the cache of {sequence_ids}.")
        self.infer_context.clear_context_by_seq_ids(sequence_ids)

    def copy_blocks(self, src_dst_map):
        if self.copy_blocks_ops is None:
            self.copy_blocks_ops = BlockCopy(
                self.backend_type, self.generator_backend.cache_pool.npu_cache, self.to_tensor)
        self.copy_blocks_ops.copy_blocks(src_dst_map)

    def check_batch_size_limit(self, is_prefill: bool, batch_size: int) -> None:
        if is_prefill:
            max_allowed = self.max_prefill_batch_size
            stage_name = "prefill"
        else:
            max_allowed = self.max_batch_size
            stage_name = "Decode"

        if batch_size > max_allowed:
            message = (
                f"The `batch_size` is {batch_size} but 'max_{stage_name}_batch_size' is {max_allowed}. "
                f"The `batch_size` should be less than 'max_{stage_name}_batch_size'."
            )
            logger.warning(message)

    def generate_token(self, input_metadata: InputMetadata, warmup=False) -> GenerationOutput:
        """The core method for generating token ids.

        The key method called by the `BatchScheduler`, used for one iteration of generating token ids for a batch. Each
        iteration can be divided into four stages: preprocessing, forward propagation, sampling, and stop-checking. The
        handling of these stages may vary depending on the different plugins configured.

        Args:
            input_metadata: The input metadata constructed by the `BatchScheduler` includes request data such as input
                ids, post-processing parameters, etc.
        """
        if self.backend_type == 'atb' and not warmup and input_metadata.batch_seq_len.any():
            cur_dp_rank_id_mask = self.model_wrapper.mapping.attn_dp.rank
            mask = input_metadata.batch_dp_rank_ids == cur_dp_rank_id_mask
            if len(input_metadata.batch_seq_len) != len(input_metadata.batch_dp_rank_ids): 
                seq_tensor = input_metadata.batch_seq_len
                comp_tensor = input_metadata.computed_blocks
            else:
                seq_tensor = input_metadata.batch_seq_len[mask]
                comp_tensor = (input_metadata.computed_blocks[mask]
                               if input_metadata.computed_blocks is not None
                               else None)
            input_ids_length = seq_tensor.sum().item()
            if self.scp_size == 1 and comp_tensor is not None: 
                input_ids_length -= comp_tensor.sum().item() * self.generator_backend.block_size
            batch_size = np.asarray(mask).sum()
            if input_ids_length > self.max_prefill_tokens:
                message = (
                        f"`input_id` is {input_ids_length} but 'max_prefill_token' is {self.max_prefill_tokens}. "
                        f"`input_id` should be less than 'max_prefill_tokens'." 
                )
                logger.warning(message)
            self.check_batch_size_limit(input_metadata.is_prefill, batch_size)

        from ..utils.prof.profiler import span_start, span_end, span_attr, Level
        prof = span_start(name="generate_token", level=Level.DETAILED)
        prof = span_attr(prof, "input_metadata", lambda: str(input_metadata))
        try:
            if (self.pd_config.model_role in [DmiModeNodeRole.DECODER, DmiModeNodeRole.FLEX] and
                    not input_metadata.is_prefill and not self.input_metadata_queue.empty()):
                while not self.input_metadata_queue.empty():
                    input_metadata_pass = self.input_metadata_queue.get()
                    cache_ids = self.infer_context.get_batch_context_handles(input_metadata_pass)
                    _, sampling_metadata, _ = self.infer_context.compose_model_inputs(
                        input_metadata_pass, cache_ids, warmup=False, is_pd_separate=True)
                    if not input_metadata_pass.is_dummy_batch and sampling_metadata.do_sample_array is not None:
                        self.generator_backend.configure_sampler(sampling_metadata)

            if self.plugin:
                if self.async_inference:
                    if self.layerwise_disaggregated and self.pd_config.model_role != STANDARD_TAG:
                        raise RuntimeError(f'Disaggregated-pd circumstance is not supported \
                        when `splitInference = true` in `config.json`. \
                        {self.pd_config.model_role} is not compatible with split inference settings. \
                        Please check pd config.')

                    generation_output = self.plugin.generate_token_async(input_metadata, warmup)
                else:
                    generation_output = self.plugin.generate_token(input_metadata, warmup)
            else:
                raise NotImplementedError('plugin not implemented')
            if generation_output is None:
                return generation_output

            if ENV.benchmark_enable and generation_output.trace_ids is not None:
                timer.log_time(
                    self.rank, generation_output.trace_ids, generation_output.current_token_indices)
            if ENV.benchmark_enable_async and generation_output.simulator_ids is not None:
                timer.log_time_async(
                    self.rank, generation_output.simulator_ids, generation_output.current_token_indices, input_metadata)

            generation_output.collate()
        except NotImplementedError as e:
            print_log(self.rank, logger.error, f'Something not implemented: {e}')
            if input_metadata.all_sequence_ids is not None:
                self.clear_cache(input_metadata.all_sequence_ids)
            raise e
        except Exception as e:
            print_log(self.rank, logger.error, f'Unknown exception: {e}')
            if self.is_inference_pause:
                return GenerationOutput.make_empty()
            raise e

        prof = span_attr(prof, "generation_output", lambda: str(generation_output))
        span_end(prof)
        return generation_output

    def generate(self, requests: List[Request], is_prefill: bool = False
                 ) -> GenerationOutput:
        """Generate token ids using `Request`."""
        block_len_max = 0
        for request in requests:
            for sequence in request.sequences.values():
                block_len_max = max(block_len_max, len(sequence.block_tables))
        block_tables = []
        for request in requests:
            for sequence in request.sequences.values():
                if len(sequence.block_tables) < block_len_max:
                    pad_len = block_len_max - len(sequence.block_tables)
                    block_table = np.pad(sequence.block_tables, (0, pad_len), 'constant', constant_values=(-1,))
                    sequence.block_tables = block_table
                block_tables.append(sequence.block_tables)
        block_tables = np.asarray(block_tables)
        input_metadata = InputMetadata.from_requests(requests, block_tables, is_prefill)
        generation_output = self.generate_token(input_metadata)
        return generation_output

    def prefill(self, requests: List[Request]) -> GenerationOutput:
        """Generate token ids using `Request` of prefilling stage."""
        return self.generate(requests, is_prefill=True)

    def decode(self, requests: List[Request]) -> GenerationOutput:
        """Generate token ids using `Request` of decoding stage."""
        return self.generate(requests, is_prefill=False)

    def generate_mix(self, requests: List[Request], is_prefill_batch: np.ndarray
                     ) -> GenerationOutput:
        """Generate token ids using `Request` under splitfuse scenario."""
        block_tables = np.asarray([request.block_tables for request in requests])
        input_metadata = InputMetadata.from_requests(requests, block_tables, is_prefill_batch)
        res_and_stop = self.generate_token(input_metadata)
        return res_and_stop

    def warm_up(
        self,
        max_prefill_tokens: int = 4096,
        max_seq_len: int = 2560,
        max_input_len: int = 1,
        max_iter_times: int = 2560,
    ) -> KVCacheSettings:
        """Warm-up to ensure normal performance.

        Since the performance during the first inference may be impacted by the initialization of certain objects, we
        simulate an inference as a warm-up. In addition, when `npu_mem` is set to -1, the warm-up will also calculate an
        optimal npu memory size for the kv cache based on the memory usage of the NPU during the simulated inference.

        Args:
            max_prefill_tokens: The maximum limit of the sum of tokens in a batch to prefill.
            max_seq_len: The maximum sequence length of the model.
            max_input_len: The maximum input length set by configuration.
            max_iter_times: The maximum number of iterations. It can also be considered as the max output length.

        Returns:
            KVCacheSettings: The kv cache settings constructed based on the specified or calculated npu memory size.

        Raises:
            RuntimeError: Setting the above-mentioned upper limit parameters too large, or setting `NPU_MEMORY_FRACTION`
                too large, could lead to an OOM (Out of Memory) exception.
        """
        if self.npu_mem != -1:
            self._init_sepd_engine()
            kvcache_settings = self.__warmup_specified(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times,
                                                       self.npu_mem)
        else:
            if self.is_multimodal:
                print_log(self.rank, logger.warning, 'When `npuMemSize` set to -1, the multimodal model may exhibit '
                                                     'poor performance or may be out of memory during inference. Please'
                                                     ' manually configure `npuMemSize` according to the document.')
                num_hidden_layers = getattr(self.model_wrapper.config, 'num_hidden_layers', 28)
                num_key_value_heads = getattr(self.model_wrapper.config, 'num_key_value_heads', 4)
                hidden_size = getattr(self.model_wrapper.config, 'hidden_size', 3584)
                num_attention_heads = getattr(self.model_wrapper.config, 'num_attention_heads', 28)
                kv_hidden_dim = num_key_value_heads * hidden_size // num_attention_heads
                total_tokens = max_prefill_tokens + self.max_batch_size * max_iter_times
                npu_mem = KV_HALF_BYTE * num_hidden_layers * kv_hidden_dim * total_tokens / self.world_size
                npu_mem = gb(npu_mem)
                npu_mem = math.ceil(npu_mem)
                kvcache_settings = self.__warmup_specified(max_prefill_tokens, max_seq_len, max_input_len,
                                                           max_iter_times, npu_mem)
                return kvcache_settings

            if self.pd_config.model_role == STANDARD_TAG:
                npu_mem = self.__warmup_standard(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
            elif self.pd_config.model_role == DmiModeNodeRole.PREFILL:
                npu_mem = self.__warmup_prefill(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
            elif self.pd_config.model_role == DmiModeNodeRole.DECODER:
                npu_mem = self.__warmup_decode(max_iter_times)
            elif self.pd_config.model_role == DmiModeNodeRole.FLEX:
                npu_mem = self.__warmup_standard(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
            else:
                raise RuntimeError(f'{self.pd_config.model_role} not defined. Please check pd config')
            self._init_sepd_engine()
            kvcache_settings = self.__update_kvcache_settings(npu_mem)
        return kvcache_settings

    def swap(self, block_operation: Any) -> None:
        """Operate cache according to the swap decision.

        Args:
            block_operation: Any type can be transferred to the numpy array.
                Its first element should be a swap decision like this: [[decision_id, src_block, dst_block] * n].
                The map of decision ids is: 0 -> swap from cpu to npu; 1 -> swap from npu to cpu.
        """
        swap_decision = np.array(block_operation, copy=False)[0]
        self.generator_backend.swap_cache(swap_decision)

    def load_lora(self, lora_name: str, lora_path: str):
        if self.model_wrapper.adapter_manager is None:
            response = LoraOperationStatus.UNSUPPORT_CMD
        else:
            try:
                self.model_wrapper.adapter_manager.load_adapter({lora_name: lora_path})
                response = LoraOperationStatus.LORA_CMD_SUCCESS
            except (FileNotFoundError, PermissionError):
                response = LoraOperationStatus.INVALID_LORA_PATH
            except Exception as e:
                err_msg = str(e)
                if "LORA MEMORY ERROR" in err_msg:
                    response = LoraOperationStatus.SLOTS_FULL
                elif "DUPLICATED LORA ID" in err_msg:
                    response = LoraOperationStatus.DUPLICATED_LORA_ID
                elif "INVALID LORA ID" in err_msg:
                    response = LoraOperationStatus.INVALID_LORA_ID
                elif "INVALID LORA RANK" in err_msg:
                    response = LoraOperationStatus.INVALID_LORA_RANK
                else:
                    response = LoraOperationStatus.UNSUPPORT_CMD
        return response

    def unload_lora(self, lora_name: str):
        if self.model_wrapper.adapter_manager is None:
            response = LoraOperationStatus.UNSUPPORT_CMD
        else:
            try:
                self.model_wrapper.adapter_manager.unload_adapter(lora_name)
                response = LoraOperationStatus.LORA_CMD_SUCCESS
            except Exception as e:
                err_msg = str(e)
                if "INVALID LORA ID" in err_msg:
                    response = LoraOperationStatus.INVALID_lora_name
                else:
                    response = LoraOperationStatus.UNSUPPORT_CMD
        return response

    def execute_recover_command(self, command: str) -> dict:
        """Execute a recovery command for the backend.

        Only the 'atb' backend supports recovery commands. Supported commands:
        - CMD_REINIT_NPU: Reinitialize NPU and resume HCCL communication.
        - CMD_START_ENGINE: Resume inference engine.
        - CMD_PAUSE_ENGINE: Pause inference engine (delegated to generator backend).

        Args:
            command: The recovery command string (e.g., "CMD_REINIT_NPU").

        Returns:
            A dictionary with keys:
                - "command_result": 0 for success, 1 for failure.
                - "error_msg": Empty string if successful, otherwise an error message.
                - "npu_device_id": The NPU device ID used in the operation.
        """
        error_msg_key = "error_msg"
        command_res_key = "command_result"
        # Standard response template
        ret_dict = {
            command_res_key: 1,  # 1 = failure by default
            error_msg_key: "",
            "npu_device_id": self.npu_device_id
        }

        # Only 'atb' backend supports recovery commands
        if self.backend_type != 'atb':
            error_msg = f"Recovery commands are only supported by 'atb' backend, got: {self.backend_type!r}"
            logger.error(error_msg)
            ret_dict[error_msg_key] = error_msg
            return ret_dict

        logger.info(f"Executing recover command {command} on NPU device {self.npu_device_id}.")
        # Dispatch by command
        if command == "CMD_REINIT_NPU":
            try:
                self.infer_context.reset_all_context()
                ret_dict = self.generator_backend.execute_recover_command(command)
                if ret_dict[command_res_key] == 0:
                    acl.rt.set_device(self.npu_device_id)
                    self.model_wrapper.resume_hccl_comm()
                    ret_dict[command_res_key] = 0  # success
            except Exception as e:
                error_msg = f"Failed to execute recovery command {command!r}: {e}"
                logger.exception(error_msg)
                ret_dict[error_msg_key] = error_msg

        elif command == "CMD_START_ENGINE":
            # Resume engine: clear state and signal ready
            time.sleep(1)
            self.plugin.last_sequence_ids = None
            self.plugin.is_inference_pause = False
            self.is_inference_pause = False

            # If the plugin supports asynchronous inference (indicated by the presence of 'output_queue'),
            # enqueue an empty ModelOutputWrapper to flush the pipeline when the output_queue is empty.
            if hasattr(self.plugin, 'output_queue') and self.plugin.output_queue is not None and \
                self.plugin.output_queue.empty():
                from .utils.model_output import ModelOutputWrapper
                self.plugin.output_queue.put(ModelOutputWrapper.make_empty())

            ret_dict[command_res_key] = 0

        elif command == "CMD_PAUSE_ENGINE":
            # Delegate pause to generator backend
            self.is_inference_pause = True
            self.plugin.is_inference_pause = True
            time.sleep(20)
            ret_dict = self.generator_backend.execute_recover_command(command)

        elif command == "CMD_CLEAR_TRANSER":
            ret_dict[command_res_key] = 0

        else:
            # Unknown command
            error_msg = f"Unknown recovery command: {command!r}"
            logger.error(error_msg)
            ret_dict[error_msg_key] = error_msg

        result_status = "failure" if ret_dict[command_res_key] == 1 else "success"
        logger.info(
            f"Execute recover command '{command}' completed: "
            f"command_result={result_status}, "
            f"error_msg='{ret_dict[error_msg_key]}', "
            f"npu_device_id={self.npu_device_id}"
        )
        return ret_dict

    def __execute_warm_up(
        self,
        kvcache_settings: KVCacheSettings,
        input_metadata: InputMetadata,
        dummy: bool = False
    ) -> None:
        mapping = None
        spcp_parallel_info = None
        if hasattr(self.model_wrapper, "mapping"):
            mapping = self.model_wrapper.mapping
            spcp_parallel_info = (mapping.attn_inner_sp, mapping.attn_cp)
        infer_context = TGInferContextStore(
            kvcache_settings,
            self.cache_config,
            spcp_parallel_info,
            self.model_wrapper.model_info.device,
            self.context_params,
            self.tokenizer,
            self.tokenizer_sliding_window_size,
            self.model_wrapper.generate_position_ids,
        )
        decode_input_len = self.num_speculative_tokens + 1 if self.num_speculative_tokens > 0 else 1
        warmup_cache_ids = np.zeros(input_metadata.batch_size, dtype=np.int32)
        model_inputs, sampling_metadata, _ = infer_context.compose_model_inputs(
            input_metadata, warmup_cache_ids, warmup=True, dummy=dummy, decode_input_len=decode_input_len,
            dp_size=self.model_wrapper.mapping.attn_dp.group_size if hasattr(self.model_wrapper, "mapping") else 1
        )
        try:
            self.generator_backend._warm_up(model_inputs, inference_mode=self.inference_mode,
                                            sampling_metadata=sampling_metadata)
        except RuntimeError as e:
            if str(e).startswith(NPU_OUT_OF_MEMORY_TAG):
                print_log(self.rank,
                          logger.error,
                          f'Warmup failed, '
                          f'because of model inference out of memory when `npuMemSize` set to {self.npu_mem}. '
                          f'please try to decrease `npuMemSize`')
            raise e

    def __get_warm_up_params(self,
                             max_prefill_tokens=4096,
                             max_seq_len=2560,
                             max_input_len=1,
                             max_iter_times=2560):
        params_checklist = (max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
        if not all(isinstance(param, int) for param in params_checklist):
            raise ValueError(f'Warmup need all params int type, but got {max_prefill_tokens=}, '
                             f'{max_seq_len=}, {max_input_len=}, {max_iter_times=}')
        print_log(self.rank, logger.info, f'warmup params: {max_prefill_tokens=}, {max_seq_len=}, '
                                          f'{max_input_len=}, {max_iter_times=}')
        max_len = min(max_seq_len, max_input_len + max_iter_times)
        if max_prefill_tokens < max_len:
            max_prefill_tokens = max_len
            self.max_prefill_tokens = max_len
            print_log(self.rank, logger.warning, '`max_prefill_tokens` is smaller than `max_len`, '
                                                 ' and it will be replaced by `max_len`, '
                                                 '`max_len` = min(max_seq_len, max_input_len + max_iter_times)')
        max_input_length = min(max_seq_len, max_input_len)
        if max_input_length <= 0:
            raise ValueError("max_input_len and max_seq_len should be greater than 0")
        return max_prefill_tokens, max_input_length, max_len

    def __calc_block_tables(self, batch_size, block_nums):
        if self.scp_size > 1:
            block_tables = np.zeros(batch_size * self.scp_size * block_nums[0],
                                    dtype=np.int32).reshape([batch_size, self.sp_size * self.cp_size, -1])
        else:
            block_tables = np.zeros(batch_size * block_nums[0],
                                    dtype=np.int32).reshape(batch_size, -1)
        if block_nums[-1] != block_nums[0]:
            block_tables[-1, block_nums[-1]:] = -1
        return block_tables

    def __get_warm_up_reqs(
        self,
        num_blocks,
        warm_up_params=(4096, 2560, 1, 2560),
        is_prefill=True
    ) -> Tuple[List[Request], np.ndarray, List]:
        max_prefill_tokens, max_seq_len, max_input_len, max_iter_times = warm_up_params
        try:
            max_prefill_tokens, _, max_len = \
                self.__get_warm_up_params(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
        except ValueError as e:
            print_log(self.rank, logger.error, f'ValueError: {e}')
            raise e

        prefill_reqs = []
        prefill_blocks = []

        prefill_blocks_per_dp_group = [0 for _ in range(self.dp_size if not self.distributed_enable else 1)]
        if self.lwd_multi_nodes_enable and self.dp_size > 1 and is_prefill: # 多机多dp时显存优化
            max_prefill_tokens_per_dp_group = [max_prefill_tokens, 1]
        else:
            max_prefill_tokens_per_dp_group = \
                        [max_prefill_tokens for _ in range(self.dp_size if not self.distributed_enable else 1)]
        while max(max_prefill_tokens_per_dp_group) > 0:
            dp_idx = np.argmin(prefill_blocks_per_dp_group)
            input_len = min(max_prefill_tokens_per_dp_group[dp_idx], max_len)
            load_balance_cp_size = self.cp_size * 2  # 负载均衡
            if self.cp_size > 1 and input_len % load_balance_cp_size != 0:
                base = input_len // load_balance_cp_size
                input_len = (base + 1) * load_balance_cp_size
            output_len = 0
            if self.block_size == 0:
                raise ZeroDivisionError('self.kvcache_settings.block_size should be greater than 0')
            num_required_block = math.ceil(input_len / self.block_size)
            if prefill_blocks_per_dp_group[dp_idx] + num_required_block > num_blocks * self.scp_size:
                print_log(self.rank, logger.warning, 'The `max_prefill_tokens` exceeds what the npu mem can hold.')
                break
            request = Request.from_warmup(input_len, output_len, self.warmup_topk_size, self.enable_warmup_with_sampling)
            request.dp_rank_id = dp_idx
            request.sp_tokens = None

            if self.scp_size > 1:
                num_full_blocks = request.input_length // self.block_size
                remainder = request.input_length % self.block_size

                base = num_full_blocks // self.scp_size
                extra = num_full_blocks % self.scp_size

                input_len_per_sp = np.full(self.scp_size, base * self.block_size, dtype=int)
                input_len_per_sp[:extra] += self.block_size
                
                remainder_rank = num_full_blocks % self.scp_size  # 下一个轮询位置
                input_len_per_sp[remainder_rank] += remainder
                request.sp_tokens = np.array(input_len_per_sp, dtype=np.int32)
                request.sp_rank_id = remainder_rank

            max_prefill_tokens_per_dp_group[dp_idx] -= input_len
            prefill_blocks_per_dp_group[dp_idx] += num_required_block
            prefill_blocks.append(num_required_block)
            prefill_reqs.append(request)
        batch_size = len(prefill_reqs)
        if batch_size == 0:
            message = ('Warmup failed. This issue could be caused by setting both `max_prefill_tokens` and '
                       '`max_input_length` to very large values, or setting insufficient `npu_mem`. Reducing either '
                       '`max_prefill_tokens` or `max_input_length` may help resolve this issue. If `npu_mem` is -1, try'
                       ' to increase the environment value `NPU_MEMORY_FRACTION` or `npu_mem` in configuration '
                       'directly. Increase `world_size` can be another choice.')
            logger.error(message)
            raise RuntimeError(message)
        block_tables = self.__calc_block_tables(batch_size, prefill_blocks)
        return prefill_reqs, block_tables, prefill_blocks

    def __warmup_prefill(self, max_prefill_tokens, max_seq_len, max_input_len, max_iter_times):
        if self.enable_dap:
            npu_mem = self.__auto_warmup(
                max_prefill_tokens, math.ceil(max_prefill_tokens / 2), max_input_len, max_iter_times, is_prefill=True
            )
        ori_enable_dap = self.generator_backend.enable_dap
        self.generator_backend.enable_dap = False
        npu_mem = self.__auto_warmup(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times, is_prefill=True)
        self.generator_backend.enable_dap = ori_enable_dap
        return npu_mem

    def __warmup_decode(self, max_iter_times):
        max_prefill_tokens = (self.num_speculative_tokens + 1) * self.max_batch_size
        max_seq_len = self.num_speculative_tokens + 1
        max_input_len = (self.num_speculative_tokens + 1) * self.max_batch_size
        npu_mem = self.__auto_warmup(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times, is_prefill=False)
        return npu_mem

    def __warmup_standard(self, max_prefill_tokens, max_seq_len, max_input_len, max_iter_times):
        npu_mem = self.__warmup_prefill(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
        if self.backend_type == 'atb':
            npu_mem = self.__warmup_decode(max_iter_times)
        return npu_mem

    def __warmup_specified(self, max_prefill_tokens, max_seq_len, max_input_len, max_iter_times, npu_mem):
        kvcache_settings = self.__update_kvcache_settings(npu_mem)
        prefill_token_length = max_seq_len
        if self.enable_dap:
            prefill_token_length = math.ceil(max_prefill_tokens / 2)
        try:
            warm_up_params = (max_prefill_tokens, prefill_token_length, max_input_len, max_iter_times)
            prefill_reqs, block_tables, _ = self.__get_warm_up_reqs(kvcache_settings.num_npu_blocks, warm_up_params)
        except Exception as e:
            print_log(self.rank, logger.error, f'Error: {e}')
            raise e
        input_metadata = InputMetadata.from_requests(prefill_reqs, block_tables, True, self.block_size)
        if self.pd_config.model_role == DmiModeNodeRole.DECODER:
            input_metadata.max_seq_len = max_seq_len
            input_metadata.max_batch_size = self.max_batch_size
            input_metadata.is_prefill = False
        self.__execute_warm_up(kvcache_settings, input_metadata, self.inference_mode)
        if self.enable_dap and self.pd_config.model_role != DmiModeNodeRole.DECODER:
            self.generator_backend.enable_dap = False
            try:
                warm_up_params = (max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
                prefill_reqs, block_tables, _ = self.__get_warm_up_reqs(kvcache_settings.num_npu_blocks, warm_up_params)
            except Exception as e:
                print_log(self.rank, logger.error, f'Error: {e}')
                raise e
            input_metadata = InputMetadata.from_requests(prefill_reqs, block_tables, True, self.block_size)
            self.__execute_warm_up(kvcache_settings, input_metadata, self.inference_mode)
            self.generator_backend.enable_dap = True
        _, _ = self.watcher.watch_npu_mem(self.rank, 'warmup specified success', self.is_multimodal, self.max_input_len)
        return kvcache_settings

    def __update_kvcache_settings(self, npu_mem):
        kvcache_settings = KVCacheSettings(
            self.rank,
            self.model_info,
            self.cpu_mem,
            npu_mem,
            self.block_size,
            self.backend_type,
            self.separate_deployment_worker is not None,
            self.num_speculative_tokens,
        )
        if self.separate_deployment_worker is not None:
            # K npu_cache model_id = 0, V npu_cache model_id = 1, K int8 npu_cache model_id = 2
            if kvcache_settings.k_head_size != kvcache_settings.v_head_size:
                num_quant_layers = 0
                if kvcache_settings.kvcache_quant_layers:
                    num_quant_layers = kvcache_settings.kvcache_quant_layers.count(True)
                if kvcache_settings.k_head_size > 0:
                    if kvcache_settings.num_layers - num_quant_layers > 0:
                        self.separate_deployment_worker.build(
                            model_id=0,
                            num_tensors=kvcache_settings.num_layers - num_quant_layers,
                            num_blocks=kvcache_settings.num_npu_blocks,
                            blockshape=kvcache_settings.k_block_shape,
                            dtype=kvcache_settings.dtype_str,
                        )
                    if num_quant_layers > 0:
                        self.separate_deployment_worker.build(
                            model_id=2,
                            num_tensors=num_quant_layers,
                            num_blocks=kvcache_settings.num_npu_blocks,
                            blockshape=kvcache_settings.k_block_quant_shape,
                            dtype=KVCacheSettings.dtype_to_str(kvcache_settings.backend_type, torch.int8),
                        )
                if kvcache_settings.v_head_size > 0:
                    self.separate_deployment_worker.build(
                        model_id=1,
                        num_tensors=kvcache_settings.num_layers,
                        num_blocks=kvcache_settings.num_npu_blocks,
                        blockshape=kvcache_settings.v_block_shape,
                        dtype=kvcache_settings.dtype_str,
                    )
                if kvcache_settings.index_head_dim is not None:
                    self.separate_deployment_worker.build(
                        model_id=3,
                        num_tensors=kvcache_settings.num_layers,
                        num_blocks=kvcache_settings.num_npu_blocks,
                        blockshape=kvcache_settings.index_block_shape,
                        dtype=kvcache_settings.dtype_str,
                    )
            else:
                num_tensors = 2 * kvcache_settings.num_layers
                self.separate_deployment_worker.build(
                    model_id=0,
                    num_tensors=num_tensors,
                    num_blocks=kvcache_settings.num_npu_blocks,
                    blockshape=kvcache_settings.k_block_shape,
                    dtype=kvcache_settings.dtype_str,
                )
        try:
            self.generator_backend.update_cache_policy(kvcache_settings, self.separate_deployment_worker)
        except RuntimeError as e:
            if str(e).startswith('Trying to create tensor with negative dimension'):
                print_log(self.rank,
                          logger.error,
                          'Warmup failed, because of model inference out of memory when `npuMemSize` set to -1.'
                          'please try to increase the environment value `NPU_MEMORY_FRACTION`'
                          'or decrease `max_prefill_tokens`')
                raise e
        return kvcache_settings

    def __auto_warmup(self, max_prefill_tokens, max_seq_len, max_input_len, max_iter_times, is_prefill=True):
        total_mem, peak_mem = self.watcher.watch_npu_mem(self.rank, 
            f'Before {"prefill" if is_prefill else "Decode"} warmup', self.is_multimodal, self.max_input_len)
        block_mem_size = calc_block_mem(self.model_info, self.block_size, self.num_speculative_tokens)
        total_blocks = int((total_mem * ENV.memory_fraction - peak_mem) // block_mem_size)

        try:
            warm_up_params = (max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
            prefill_reqs, block_tables, _ = self.__get_warm_up_reqs(total_blocks, warm_up_params, is_prefill)
        except Exception as e:
            print_log(self.rank, logger.error, f'Error: {e}')
            raise e

        dp_group_size = self.dp_size if not self.distributed_enable else 1

        input_metadata = InputMetadata.from_requests(
            prefill_reqs[:self.max_batch_size * dp_group_size],
            block_tables[:self.max_batch_size * dp_group_size],
            True,
            self.block_size
        )

        if self.pd_config.model_role == DmiModeNodeRole.DECODER or (not is_prefill and self.pd_config.model_role == STANDARD_TAG):
            input_metadata.max_seq_len = max_seq_len
            input_metadata.max_batch_size = self.max_batch_size * dp_group_size
            input_metadata.is_prefill = False

        num_prefill_blocks = 1
        prefill_npu_mem = calc_npu_mem(num_prefill_blocks, self.model_info,
                                       self.block_size, self.num_speculative_tokens)
        prefill_npu_mem_gb = prefill_npu_mem / 1024 / 1024 / 1024
        print_log(self.rank, logger.info,
                  f'`{self.pd_config.model_role} blocks` during warmup needs npu memory(GB): {prefill_npu_mem_gb}')
        kvcache_settings = self.__update_kvcache_settings(prefill_npu_mem_gb)

        self.__execute_warm_up(kvcache_settings, input_metadata, dummy=True)
        total_mem, peak_mem = self.watcher.watch_npu_mem(self.rank, 
            f'After {"prefill" if is_prefill else "Decode"} warmup', self.is_multimodal, self.max_input_len)
        total_blocks = int(total_mem * ENV.memory_fraction - peak_mem) // block_mem_size + num_prefill_blocks

        try:
            warm_up_params = (max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
            prefill_reqs, block_tables, _ = self.__get_warm_up_reqs(total_blocks, warm_up_params, is_prefill)
        except Exception as e:
            print_log(self.rank, logger.error, f'Error: {e}')
            raise e
        _ = InputMetadata.from_requests(prefill_reqs, block_tables, True, self.block_size)
        _, _ = self.watcher.watch_npu_mem(self.rank, 'After check warmup', self.is_multimodal, self.max_input_len)

        npu_mem = calc_npu_mem(total_blocks, self.model_info, self.block_size) / 1024 / 1024 / 1024
        if self.soc_version is not None and self.soc_version == 240:
            npu_mem = 5

        print_log(self.rank, logger.info,
                  f'`Total blocks` needs npu memory(GB): {npu_mem}')
        return npu_mem