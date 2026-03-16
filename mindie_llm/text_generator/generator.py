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
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt
import torch

import acl
from mindie_llm.connector.common.model_execute_data_pb2 import LoraOperationStatus
from mindie_llm.text_generator.utils.npu_mem_tool import (
    calc_block_mem,
    gb,
    memory_profiling,
    NpuMemoryWatcher,
    WeightMemoryProfiler
)
from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.text_generator.adapter import get_generator_backend
from mindie_llm.utils.validation import parse_config, ParseType
from mindie_llm.text_generator.plugins import get_plugin
from mindie_llm.text_generator.plugins.plugin_utils import (
    InferenceMode, PluginParameterValidator
)
from mindie_llm.text_generator.utils import (
    TGInferContextStore, KVCacheSettings, InputMetadata,
    GenerationOutput, OutputFilter
)
from mindie_llm.text_generator.utils.block_copy import BlockCopy
from mindie_llm.text_generator.utils.config import (
    CacheConfig, ModelConfig, ContextParams, ResponseConfig
)
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.utils.decorators.time_decorator import timer
from mindie_llm.utils.env import ENV
from mindie_llm.utils.log import ErrorCode, logger, print_log
from mindie_llm.utils.log.error_code import ErrorCodeException, convert_exception_to_error_code
from mindie_llm.utils.status import MindieLlmStatusCode
from mindie_llm.utils.tensor import npu
from mindie_llm.text_generator.utils.separate_deployment_engine import (
    SeparateDeploymentWorker, LinkParams, DmiModeNodeRole
)
from mindie_llm.utils import file_utils

STANDARD_TAG = 'standard'
NPU_OUT_OF_MEMORY_TAG = 'NPU out of memory'
EOS_TOKEN_ID = 'eos_token_id'
KV_HALF_BYTE = 4
MEM_POOL_WORKER_ROLE = "worker"


@dataclass
class WarmupParams:
    max_prefill_tokens: int = 4096
    max_seq_len: int = 2560
    max_input_len: int = 1
    max_iter_times: int = 2560

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer, got {value}")


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
        从远程模型实例拉取 kv cache, 并将 input_metadata 放入队列。
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
        self.enable_warmup_with_sampling = parse_config(model_config, 'enable_warmup_with_sampling',
                                                    parse_type=ParseType.TO_BOOL, default_value=True)

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
        self.enable_prefix_cache = 'prefix_cache' in plugin_list

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

        with WeightMemoryProfiler() as prof:
            self.generator_backend = get_generator_backend(model_config)
    
        self.model_memory_usage = prof.model_weight
        print_log(self.rank, logger.info, f'Model loading took {gb(self.model_memory_usage):.2f} GiB memory.')

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

        self.is_multimodal = self.model_wrapper.is_multimodal
        if self.is_multimodal and "memory_decoding" in plugin_list:
            message = ("Memory decoding is not supported when the model type is multimodal.")
            logger.error(message, ErrorCode.TEXT_GENERATOR_FEAT_COMPAT_INVALID)
            raise NotImplementedError(message)
        self.is_separated_pd = self.pd_config.model_role in [DmiModeNodeRole.PREFILL, DmiModeNodeRole.DECODER]

        self.copy_blocks_ops = None
        self.is_inference_pause = False

        plugin_config['eos_token_id'] = self.cache_config.eos_token_id
        plugin_config['cache_size'] = self.cache_config.cache_size
        plugin_config['max_gen_len'] = self.cache_config.max_gen_len
        plugin_config['max_block_size'] = self.cache_config.max_block_size
        plugin_config['max_seq_len'] = self.cache_config.max_seq_len
        plugin_config['device'] = self.model_wrapper.model_info.device
        plugin_config["layerwise_disaggregated"] = self.layerwise_disaggregated
        plugin_config["layerwise_disaggregated_role_type"] = self.layerwise_disaggregated_role_type
        
        # NOTE: Warmup async inference will lead to lower mtp acceptance rate with unknown reason,
        # so we disable async inference here.
        with self._temporarily_disable(async_inference=self.async_inference):
            block_mem_size_gb = gb(calc_block_mem(self.model_info, self.block_size, self.num_speculative_tokens))
            print_log(self.rank, logger.info,
                    f'One block during warmup needs npu memory(GiB): {block_mem_size_gb}')
            kvcache_settings = self._update_kvcache_settings(block_mem_size_gb)
            self._init_plugin_manager(kvcache_settings, plugin_list, plugin_config)
            warmup_param = WarmupParams(max_prefill_tokens, max_seq_len, max_input_len, max_iter_times)
            npu_mem = self.warm_up(warmup_param)

        self._init_sepd_engine()
        self.kvcache_settings = self._update_kvcache_settings(npu_mem)

        _, warmup_mem = self.watcher.watch_npu_mem(self.rank, f'After warmup', 
                                                        self.is_multimodal, max_input_len)
        self.watcher._set_warmup_mem(warmup_mem)

        self._init_plugin_manager(self.kvcache_settings, plugin_list, plugin_config)

        if (
            self.pd_config.model_role == DmiModeNodeRole.DECODER
            and len(self.generator_backend.kv_pool_backend) != 0
            and len(self.generator_backend.kv_pool_config_path) != 0
        ):
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
                        input_metadata_pass, cache_ids, warmup=warmup, is_pd_separate=True)
                    if not input_metadata_pass.is_dummy_batch and sampling_metadata.do_sample_array is not None:
                        self.generator_backend.configure_sampler(sampling_metadata)

            if self.plugin_manager:
                if self.async_inference:
                    if self.layerwise_disaggregated and self.pd_config.model_role != STANDARD_TAG:
                        raise RuntimeError(f'Disaggregated-pd circumstance is not supported \
                        when `splitInference = true` in `config.json`. \
                        {self.pd_config.model_role} is not compatible with split inference settings. \
                        Please check pd config.')

                    generation_output = self.plugin_manager.generate_token_async(input_metadata, warmup=warmup)
                else:
                    generation_output = self.plugin_manager.generate_token(input_metadata, warmup=warmup)
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
        except ErrorCodeException as e:
            if warmup:
                print_log(self.rank, logger.error, f'Out-of-memory exception occurred during warmup')
                raise RuntimeError from e
            else:
                raise e
        except Exception as e:
            error_code = convert_exception_to_error_code(str(e))
            if isinstance(e, RuntimeError) and error_code is not None:
                message = (
                    f'{error_code.name} fault happened in generate_token, error code: {error_code.value}.'
                )
                logger.error(message)
                raise ErrorCodeException(error_code) from e
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
        warmup_params: WarmupParams
    ) -> float | int:
        """Warm-up to determine the available npu memory for KVcache.

        Since the performance during the first inference may be impacted by the initialization of certain objects, we
        simulate an inference as a warm-up. In addition, when `npu_mem` is set to -1, the warm-up will also calculate an
        optimal npu memory size for the kv cache based on the memory usage of the NPU during the simulated inference.

        Args:
            warmup_params: The parameters for warmup.
                - max_prefill_tokens: The maximum limit of the sum of tokens in a batch to prefill.
                - max_seq_len: The maximum sequence length of the model.
                - max_input_len: The maximum input length set by configuration.
                - max_iter_times: The maximum number of iterations. It can also be considered as the max output length.

        Returns:
            Available npu memory(GiB) for KVcache.

        Raises:
            RuntimeError: Setting the above-mentioned upper limit parameters too large, or setting `NPU_MEMORY_FRACTION`
                too large, could lead to an OOM (Out of Memory) exception.
        """
        npu_mem: float | int = self.npu_mem
        if self.soc_version is not None and self.soc_version == 240:
            npu_mem = 5

        if npu_mem != -1:
            self._warmup_specified(warmup_params)
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
                total_tokens = warmup_params.max_prefill_tokens + self.max_batch_size * warmup_params.max_iter_times
                npu_mem = KV_HALF_BYTE * num_hidden_layers * kv_hidden_dim * total_tokens / self.world_size
                return math.ceil(gb(npu_mem))

            with memory_profiling(
                baseline_non_torch=0,
                weights_memory=self.model_memory_usage
            ) as profile_result:
                if self.pd_config.model_role in {STANDARD_TAG, DmiModeNodeRole.FLEX}:
                    self._warmup_standard(warmup_params)
                elif self.pd_config.model_role == DmiModeNodeRole.PREFILL:
                    self._warmup_prefill(warmup_params)
                elif self.pd_config.model_role == DmiModeNodeRole.DECODER:
                    self._warmup_decode(warmup_params)
            requested_memory = profile_result.total_memory * ENV.memory_fraction
            npu_mem = requested_memory - profile_result.non_kv_cache_memory
            print_log(self.rank, logger.info,
                      f'Requested memory: {ENV.memory_fraction} (util), {gb(requested_memory):.2f} GiB')
            print_log(self.rank, logger.info, profile_result)
            npu_mem = self._validate_warmup_memory(warmup_params, npu_mem)
        return npu_mem

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
                self.plugin_manager.error_code_collected_in_async = None
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
            self.plugin_manager.last_sequence_ids = None
            self.plugin_manager.is_inference_pause = False
            self.is_inference_pause = False

            # If the plugin supports asynchronous inference (indicated by the presence of 'output_queue'),
            # enqueue an empty ModelOutputWrapper to flush the pipeline when the output_queue is empty.
            if hasattr(self.plugin_manager, 'output_queue') and self.plugin_manager.output_queue is not None and \
                self.plugin_manager.output_queue.empty():
                from .utils.model_output import ModelOutputWrapper
                self.plugin_manager.output_queue.put(ModelOutputWrapper.make_empty())

            ret_dict[command_res_key] = 0

        elif command == "CMD_PAUSE_ENGINE":
            # Delegate pause to generator backend
            self.is_inference_pause = True
            self.plugin_manager.is_inference_pause = True
            time.sleep(20)
            
            ret_dict = self.generator_backend.execute_recover_command(command)

        elif command == "CMD_PAUSE_ENGINE_ROCE":
            # Delegate pause to generator backend
            self.is_inference_pause = True
            self.plugin.is_inference_pause = True
            ret_dict[command_res_key] = 0

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

    @contextmanager
    def _temporarily_disable(self, dap: bool = False, async_inference: bool = False):
        origin_enable_dap = self.generator_backend.enable_dap
        origin_async_inference = self.async_inference
        try:
            if dap:
                self.generator_backend.enable_dap = False
            if async_inference:
                self.async_inference = False
            yield
        finally:
            if dap:
                self.generator_backend.enable_dap = origin_enable_dap
            if async_inference:
                self.async_inference = origin_async_inference

    def _init_plugin_manager(
        self,
        kvcache_settings: KVCacheSettings,
        plugin_list: List[str],
        plugin_config: dict
    ):
        mapping = None
        spcp_parallel_info = None
        if hasattr(self.model_wrapper, "mapping"):
            mapping = self.model_wrapper.mapping
            spcp_parallel_info = (
                mapping.attn_inner_sp,
                mapping.attn_cp,
            )
        context_params = ContextParams(
            self.is_separated_pd,
            self.num_speculative_tokens,
            self.hidden_size,
            self.model_wrapper.model_info.dtype,
            self.enable_mtp,
            self.async_inference,
            self.distributed_enable,
            self.max_generated_tokens,
            BackendType.from_string(self.backend_type),
            self.layerwise_disaggregated,
            self.layerwise_disaggregated_role_type
        )
        self.infer_context = TGInferContextStore(
            kvcache_settings,
            self.cache_config,
            spcp_parallel_info,
            self.model_wrapper.model_info.device,
            context_params,
            self.tokenizer,
            self.tokenizer_sliding_window_size,
            self.model_wrapper.generate_position_ids,
        )
        output_filter = OutputFilter(self.cache_config, self.infer_context, self.tokenizer, self.async_inference)
        plugin_utils = (
            self.generator_backend,
            kvcache_settings,
            self.infer_context,
            output_filter,
            self.pd_config.model_role,
        )

        self.plugin_manager = get_plugin(plugin_list,
                                         plugin_config,
                                         plugin_utils,
                                         self.is_mix_model,
                                         self.watcher)
        self.plugin_manager.initialize()

    def _execute_warm_up(
        self,
        requests: List[Request],
        is_prefill: bool
    ) -> GenerationOutput:
        try:
            if self.is_mix_model:
                is_prefill_flag = (
                    np.ones(len(requests), dtype=bool)
                    if is_prefill
                    else np.zeros(len(requests), dtype=bool)
                )
            else:
                is_prefill_flag = is_prefill
            input_metadata = InputMetadata.from_requests(requests, None, is_prefill_flag, self.block_size)
            generation_output = self.generate_token(input_metadata, warmup=True)
            return generation_output
        except RuntimeError as e:
            if str(e).startswith(NPU_OUT_OF_MEMORY_TAG):
                print_log(self.rank,
                          logger.error,
                          f'Warmup failed due to NPU out of memory when `npuMemSize` set to {self.npu_mem}. '
                          f'Try to decrease `npuMemSize` or `maxPrefillTokens`')
            raise e

    def _get_warm_up_params(self, warmup_params: WarmupParams):
        message = (
            'warmup params: '
            f'max_prefill_tokens={warmup_params.max_prefill_tokens}, '
            f'max_seq_len={warmup_params.max_seq_len}, '
            f'max_input_len={warmup_params.max_input_len}, '
            f'max_iter_times={warmup_params.max_iter_times}'
        )
        print_log(self.rank, logger.info, message)
        max_len = min(warmup_params.max_seq_len, warmup_params.max_input_len + warmup_params.max_iter_times)
        return max_len

    def _get_warm_up_reqs(
        self,
        warmup_params: WarmupParams,
        max_output_len: int = 1,
        do_prefix_cache_warmup: bool = False,
        do_dap_warmup: bool = False
    ) -> list[Request]:
        
        max_len = self._get_warm_up_params(warmup_params)

        prefill_reqs = []

        dp_group_size = self.dp_size if not self.distributed_enable else 1

        request_lengths_per_dp_group = self._get_request_lengths_by_dp(
            max_len=max_len,
            do_prefix_cache_warmup=do_prefix_cache_warmup,
            do_dap_warmup=do_dap_warmup
        )

        max_placeholder_num = self.max_generated_tokens if self.max_generated_tokens > 1 else 1

        for dp_idx in range(dp_group_size):
            for input_len in request_lengths_per_dp_group:
                request = Request.from_warmup(
                    input_len,
                    max_output_len=max_output_len,
                    max_placeholder_num=max_placeholder_num,
                    warmup_topk_size=self.warmup_topk_size,
                    enable_warmup_sampling=self.enable_warmup_with_sampling
                )
                request.build(
                    dp_rank_id=dp_idx,
                    scp_size=self.scp_size,
                    block_size=self.block_size,
                    is_mix_model=self.is_mix_model
                )
                prefill_reqs.append(request)

        if do_prefix_cache_warmup:
            self._update_request_for_prefix_cache(prefill_reqs)

        return prefill_reqs

    def _warmup_prefill(self, warmup_params: WarmupParams):
        self._auto_warmup_prefill(warmup_params)
        if self.enable_prefix_cache:
            self._auto_warmup_prefill(warmup_params, do_prefix_cache_warmup=True)

    def _warmup_decode(self, warmup_params: WarmupParams):
        self._auto_warmup_decode(warmup_params)

    def _warmup_standard(self, warmup_params: WarmupParams):
        if self.enable_dap:
            self._auto_warmup(warmup_params, do_dap_warmup=True)
        with self._temporarily_disable(
            dap=self.enable_dap
        ):
            self._auto_warmup(warmup_params)
        if self.enable_prefix_cache:
            self._auto_warmup(warmup_params, do_prefix_cache_warmup=True)

    def _warmup_specified(self, warmup_params: WarmupParams):
        prefill_reqs = self._get_warm_up_reqs(warmup_params)

        _ = self._execute_warm_up(
            requests=prefill_reqs,
            is_prefill=True
        )

        self.watcher.watch_npu_mem(self.rank, 'warmup specified success', self.is_multimodal, self.max_input_len)

    def _update_kvcache_settings(self, npu_mem):
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
                          'Warmup failed due to NPU out of memory when `npuMemSize` set to -1.'
                          'Try to increase the environment value `NPU_MEMORY_FRACTION`'
                          'or decrease `max_prefill_tokens`')
                raise e
        return kvcache_settings

    def _auto_warmup(
        self,
        warmup_params: WarmupParams,
        do_prefix_cache_warmup: bool = False,
        do_dap_warmup: bool = False
    ):
        prefill_reqs = self._generate_warmup_requests(
            warmup_params,
            do_prefix_cache_warmup=do_prefix_cache_warmup,
            do_dap_warmup=do_dap_warmup
        )

        prefill_output = self._execute_warm_up(
            requests=prefill_reqs,
            is_prefill=True
        )
        
        decode_reqs, reqs_out_token_num = self._filter_end_reqs(prefill_reqs, prefill_output)
        print_log(self.rank, logger.info, f"After prefill, number of decode requests: {len(decode_reqs)}")

        if decode_reqs and self.generator_backend.backend_type != 'ms':
            self._warmup_decode_iteration(decode_reqs, reqs_out_token_num)
    
    def _warmup_decode_iteration(
        self,
        decode_reqs: list[Request],
        reqs_out_token_num: list[int]
    ):
        for req, num_new_token in zip(decode_reqs, reqs_out_token_num):
            req.step(
                num_new_token,
                self.scp_size,
                self.block_size,
                self.is_mix_model
            )

        generation_output = self._execute_warm_up(
            requests=decode_reqs,
            is_prefill=False
        )
        remaining_reqs, _ = self._filter_end_reqs(decode_reqs, generation_output)
        if remaining_reqs:
            raise RuntimeError(
                f"Decode warmup did not finish all requests: {len(remaining_reqs)} remaining."
            )

    def _auto_warmup_prefill(self, warmup_params: WarmupParams, do_prefix_cache_warmup=False):
        prefill_reqs = self._generate_warmup_requests(warmup_params, do_prefix_cache_warmup)
        
        prefill_output = self._execute_warm_up(
            requests=prefill_reqs,
            is_prefill=True
        )
        
        remaining_reqs, _ = self._filter_end_reqs(prefill_reqs, prefill_output)
        if len(remaining_reqs) != 0:
            raise ValueError(f"Expected 0 decode requests, got {len(remaining_reqs)}")

    def _auto_warmup_decode(self, warmup_params: WarmupParams):
        requests = self._generate_warmup_requests(warmup_params)
        prefill_input_metadata = InputMetadata.from_requests(
            requests, None, True, self.block_size
        )
        self.input_metadata_queue.put(prefill_input_metadata)

        decode_output = self._execute_warm_up(
            requests=requests,
            is_prefill=False
        )
        
        remaining_reqs, _ = self._filter_end_reqs(requests, decode_output)
        if remaining_reqs:
            raise RuntimeError(
                f"Decode warmup did not finish all requests: {len(remaining_reqs)} remaining."
            )
    
    def _generate_warmup_requests(
        self, 
        warmup_params: WarmupParams,
        do_prefix_cache_warmup: bool = False,
        do_dap_warmup: bool = False
    ) -> list[Request]:
        # 根据模式确定 max_output_len
        max_output_len = 1
        if self.pd_config.model_role == STANDARD_TAG or self.pd_config.model_role == DmiModeNodeRole.FLEX:
            max_output_len = 2

        prefill_reqs = self._get_warm_up_reqs(
            warmup_params,
            max_output_len=max_output_len,
            do_prefix_cache_warmup=do_prefix_cache_warmup,
            do_dap_warmup=do_dap_warmup
        )

        total_request_length = sum(prefill_req.input_length for prefill_req in prefill_reqs)
        message = (
            f"Number of prefill requests (all dp groups): {len(prefill_reqs)}. "
            f"First request length: {prefill_reqs[0].input_length}. "
            f"Last request length: {prefill_reqs[-1].input_length}. "
            f"Total request length (all dp groups): {total_request_length}. "
        )
        print_log(self.rank, logger.info, message)
        return prefill_reqs
    
    def _validate_warmup_memory(self, warmup_params: WarmupParams, npu_mem):
        block_mem_size = calc_block_mem(self.model_info, self.block_size, self.num_speculative_tokens)
        total_blocks = max(math.floor(npu_mem / block_mem_size), 0)
        
        try:
            max_len = self._get_warm_up_params(warmup_params)
            num_required_block = math.ceil(max_len / self.block_size)
            if num_required_block > total_blocks * self.scp_size:
                message = (
                    'Warmup failed. This could be caused by setting both `max_prefill_tokens` and '
                    '`max_input_length` to very large values, or setting insufficient `npu_mem`. Reducing either '
                    '`max_prefill_tokens` or `max_input_length` may resolve this issue. If `npu_mem` is -1, try '
                    'to increase the environment value `NPU_MEMORY_FRACTION`. '
                    f'Required block number: {num_required_block}. '
                    f'Left block number: {total_blocks * self.scp_size}.'
                )
                logger.error(message)
                raise RuntimeError(message)
        except Exception as e:
            print_log(self.rank, logger.error, f'Error: {e}')
            raise e

        npu_mem = gb(npu_mem)
        message = (
            f'Allocated {npu_mem} (GiB) NPU memory for KVcache. '
            f'Model role: {self.pd_config.model_role}'
        )
        print_log(self.rank, logger.info, message)
        return npu_mem
    
    def _filter_end_reqs(
        self,
        requests: list[Request],
        generation_output: GenerationOutput
    ) -> tuple[list[Request], list[int]]:
        unfinished_requests = []
        out_token_num = []
        for i, req in enumerate(requests):
            finish_reason = generation_output.finish_reason[i]
            if finish_reason.item() == ResponseConfig.CONTINUE:
                unfinished_requests.append(req)
                out_token_num.append(generation_output.num_new_tokens[i].item())
        return unfinished_requests, out_token_num
    
    def _get_request_lengths_by_dp(
        self,
        max_len: int,
        do_prefix_cache_warmup: bool = False,
        do_dap_warmup: bool = False
    ) -> list[int]:
        """Calculate request lengths for data parallel groups"""

        def align_for_context_parallel(length: int) -> int:
            """Align length for context parallel"""
            if self.cp_size <= 1:
                return length
            alignment = self.cp_size * 2
            return ((length + alignment - 1) // alignment) * alignment

        if self.pd_config.model_role == DmiModeNodeRole.PREFILL:
            max_batch_size_per_dp = self.max_prefill_batch_size
        else:
            max_batch_size_per_dp = self.max_batch_size

        # Max prefill tokens has been aligned in llm_manager for context parallel.
        aligned_prefill_tokens = (
            math.ceil(self.max_prefill_tokens / 2)
            if do_dap_warmup
            else self.max_prefill_tokens
        )

        result: list[int] = []

        # Case 1: Can fit all requests with max_len
        if max_len * max_batch_size_per_dp <= aligned_prefill_tokens:
            aligned_len = align_for_context_parallel(max_len) if self.cp_size > 1 else max_len
            full_count = aligned_prefill_tokens // aligned_len
            remainder = aligned_prefill_tokens % aligned_len
            result = [aligned_len] * full_count + ([remainder] if remainder > 0 else [])
        else:
            # Case 2: Need to distribute tokens among requests
            base_len = aligned_prefill_tokens // max_batch_size_per_dp
            aligned_base_len = align_for_context_parallel(base_len) if self.cp_size > 1 else base_len

            full_count = aligned_prefill_tokens // aligned_base_len
            remainder = aligned_prefill_tokens % aligned_base_len
            batch_size = full_count + (1 if remainder > 0 else 0)

            if batch_size <= max_batch_size_per_dp:
                result = [aligned_base_len] * full_count + ([remainder] if remainder > 0 else [])
            else:
                # Case 3: Need to distribute remainder among all requests
                request_lens = [aligned_base_len] * max_batch_size_per_dp
                remaining_tokens = aligned_prefill_tokens - max_batch_size_per_dp * aligned_base_len

                # Distribute remaining tokens round-robin
                alignment_size = self.cp_size * 2 if self.cp_size > 1 else 1
                idx = 0
                while remaining_tokens > 0:
                    add_amount = min(alignment_size if self.cp_size > 1 else 1, remaining_tokens)
                    request_lens[idx] += add_amount
                    remaining_tokens -= add_amount
                    idx = (idx + 1) % max_batch_size_per_dp
                result = request_lens

        if do_prefix_cache_warmup:
            result[0] = min(result[0] + self.block_size, max_len)
        return result[:max_batch_size_per_dp]

    def _update_request_for_prefix_cache(self, prefill_reqs: list[Request]):
        if len(prefill_reqs) % self.dp_size != 0:
            raise ValueError(
                f"Request count ({len(prefill_reqs)}) must be divisible by dp_size ({self.dp_size})"
            )

        request_num_per_dp = len(prefill_reqs) // self.dp_size
        
        if self.scp_size > 1:
            warmup_blocks = [0] * self.scp_size
            warmup_blocks[0] = 1
        else:
            warmup_blocks = 1

        for dp_rank in range(self.dp_size):
            req = prefill_reqs[dp_rank * request_num_per_dp]
            req.computed_blocks = warmup_blocks
            req.remote_computed_blocks = warmup_blocks
