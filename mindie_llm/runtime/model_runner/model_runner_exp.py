# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Iterable, Optional
import os
from tqdm.auto import tqdm

import torch
import numpy as np

from mindie_llm.utils.env import ENV as ENV_utils
from mindie_llm.runtime.utils.helpers.env import ENV
from mindie_llm.runtime.utils.cpu.affinity import bind_cpus
from mindie_llm.runtime.utils.npu.device_utils import get_npu_hbm_info
from mindie_llm.runtime.models import get_router_ins
from mindie_llm.runtime.model_runner.forward_context_exp import create_forward_context, set_forward_context, \
    get_forward_context, BatchDescriptor, ForwardContext
from mindie_llm.runtime.model_runner.forward_metadata.attn_metadata import AttentionMetadata
from mindie_llm.runtime.model_runner.forward_metadata.dp_metadata import DPMetadata
from mindie_llm.runtime.utils.torch_utils import set_default_torch_dtype
from mindie_llm.runtime.utils.loader.default_model_loader import DefaultModelLoader
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager, init_distributed
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.distributed.utils import set_device
from mindie_llm.runtime.layers.attention import get_global_attn_dict
from mindie_llm.runtime.compilation.aclgraph_backend import set_aclgraph_capturing_enabled, AclGraphBackend
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig
from mindie_llm.runtime.model_runner.input_buffer import input_buffer
from mindie_llm.runtime.config.load_config import LoadConfig
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.text_generator.utils.model_input import ModelInput
from mindie_llm.runtime.layers.attention.sparse_attention_layer import SFA
from mindie_llm.runtime.config.mindie_llm_config import SpeculativeConfig
from mindie_llm.runtime.layers.sampling.sampler import Sampler

# Allow tensor initialization and casting with internal format(e.g., NZ)
torch.npu.config.allow_internal_format = True


class KVCacheInfo:
    """Information about KV cache state.
    
    Attributes:
        kcache_id: ID of the key cache tensor.
        vcache_id: ID of the value cache tensor.
        kcache_shape: Shape of the key cache tensor.
        vcache_shape: Shape of the value cache tensor.
    """
    kcache_id: Optional[int] = None
    vcache_id: Optional[int] = None
    kcache_shape: Optional[torch.Size] = None
    vcache_shape: Optional[torch.Size] = None

    def check_diff(self, kv_caches: list) -> bool:
        """Check if KV cache has changed.
        
        Args:
            kv_caches: List of KV cache tuples.
            
        Returns:
            True if cache has changed, False otherwise.
        """
        kcache_id_diff = self.kcache_id != id(kv_caches[0][0])
        vcache_id_diff = self.vcache_id != id(kv_caches[0][1])
        kcache_shape_diff = self.kcache_shape != kv_caches[0][0].shape
        vcache_shape_diff = self.vcache_shape != kv_caches[0][1].shape
        kcache_diff = not self.kcache_id or kcache_id_diff or kcache_shape_diff
        vcache_diff = not self.vcache_id or vcache_id_diff or vcache_shape_diff
        return kcache_diff or vcache_diff


class ModelRunnerExp:
    """Experimental model runner for inference.
    
    This class handles model loading, initialization, and forward passes
    for experimental features.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        rank: int,
        world_size: int,
        npu_id: Optional[int] = None,
        local_rank: Optional[int] = None,
        load_tokenizer: bool = True,
        max_position_embeddings: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        llm_config_path: Optional[str] = None,
        models_dict: Optional[dict] = None,
        **kwargs
    ) -> None:
        """Initialize the model runner.
        
        Args:
            model_name_or_path: Path or name of the model.
            rank: Process rank in distributed setup.
            world_size: Total number of processes.
            npu_id: NPU device ID.
            local_rank: Local rank in distributed setup.
            load_tokenizer: Whether to load tokenizer.
            max_position_embeddings: Maximum position embeddings.
            tokenizer_path: Path to tokenizer.
            llm_config_path: Path to LLM configuration.
            models_dict: Dictionary of model configurations.
            **kwargs: Additional keyword arguments.
        """
        self._model_name_or_path = model_name_or_path
        self._max_batch_size = kwargs.get("max_batch_size", -1)
        self._max_num_token = self._max_batch_size
        local_rank = local_rank if local_rank is not None else rank
        self.device = set_device(rank, npu_id if npu_id is not None else local_rank)
        self._max_seq_len = kwargs.get("max_seq_len", -1)

        # bin cpus to the NUMA
        if ENV.bind_cpu:
            try:
                bind_cpus(ratio=1.0)
            except RuntimeError as e:
                print_log(rank, logger.info, e)
            except ValueError as e:
                print_log(rank, logger.info, e)
            except Exception as _:
                print_log(rank, logger.info, 'Skip binding cpu.')

        # NOTE: `max_position_embeddings` should be deleted later
        load_config_dict = {
            "model_name_or_path": model_name_or_path,
            "load_tokenizer": load_tokenizer,
            "max_position_embeddings": max_position_embeddings,
            "tokenizer_path": tokenizer_path,
            "trust_remote_code": kwargs.get("trust_remote_code", False),
            "llm_config_path": llm_config_path,
            "models_dict": models_dict,
        }
        
        load_config = LoadConfig.from_dict(load_config_dict)
        router_ins = get_router_ins(load_config)
        self._model_cls = router_ins.model_cls

        # NOTE: These attributes maybe depreciated after TG is refactored.
        self.config = router_ins.config
        self.kv_cache_dtype = self.config.torch_dtype
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.config_dict = router_ins.config_dict
        self.llm_config = router_ins.llm_config
        self.enable_nz = self.llm_config.llm.kv_cache_options.enable_nz
        init_distributed(rank, world_size, local_rank, llm_config=self.llm_config, server_config=kwargs)
        self.head_size = self.config.head_dim
        self.num_heads = self.config.get_num_attention_heads_per_rank()
        self.num_kv_heads = self.config.get_num_kv_heads_per_rank()
        self.num_layers = self.config.num_hidden_layers
        self.k_head_size = self.head_size
        self.v_head_size = self.head_size
        self.kvcache_quant_layers = []
        self.index_head_dim = None
        self.num_index_heads = None
        self.max_position_embeddings = self.config.max_position_embeddings
        self.adapter_manager = None
        self.lora_adapter = None
        self.model = None
        self.num_speculative_tokens = kwargs.get('num_speculative_tokens', 0)

        self._mindie_llm_config = MindIELLMConfig(
            self._model_name_or_path,
            self.config, self.llm_config,
            router_ins.generation_config,
            speculative_config=SpeculativeConfig(self.num_speculative_tokens),
        )

        # NOTE: default mask and rotary_emb will be refactored later.
        self._mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).npu()

        self._is_warmup_completed = False

        self._kv_cache_info = KVCacheInfo()

        role = kwargs.get('role', 'standard')
        # False: eager mode, True: AclGraph mode; 
        # prefill node in pd disaggregated mode do not need acl graph currently.
        self._is_aclgraph_enabled = (role != 'prefill')

        self.is_draft_model = kwargs.get("is_draft_model", False)

        sampler_config = kwargs.get('sampler_config', None)
        if ENV_utils.async_inference:
            self.sampler = Sampler(sampler_config)

    def load_weights(self) -> None:
        """Load model weights and initialize rotary embeddings."""
        if "OMP_NUM_THREADS" not in os.environ and get_parallel_info_manager().world_size > 1:
            os.environ["OMP_NUM_THREADS"] = "1"

        # load model
        if self._max_seq_len > self._mindie_llm_config.hf_config.rope_scaling.max_position_embeddings:
            _msg = (f"`max_seq_len` cannot be larger than `max_position_embeddings` "
                "or `max_position_embeddings`*`scaling_factor` when scaling. "
                f"max_seq_len={self._max_seq_len} but max_position_embeddings"
                f"={self._mindie_llm_config.hf_config.rope_scaling.max_position_embeddings}")
            logger.error(_msg)
            raise ValueError(_msg)
        with set_default_torch_dtype(self.config.torch_dtype):
            with self.device:
                self.model = self._model_cls(self._mindie_llm_config)
        logger.info("Initialize model cls done.")
        DefaultModelLoader().load_weights(self.model, self._model_name_or_path)
        logger.info("Load weight done.")

        # NOTE: These attributes maybe depreciated after TG is refactored.
        self.index_head_dim = getattr(self._mindie_llm_config.hf_config, "index_head_dim", None)
        self.num_index_heads = getattr(self._mindie_llm_config.hf_config, "index_n_heads", None)
        # not equal k v length for mla
        if hasattr(self.model, 'kv_lora_rank') and hasattr(self.model, 'qk_rope_head_dim'):   # deepseekv2/v3/r1
            self.num_kv_heads = 1
            if self.index_head_dim is not None:
                self.num_index_heads = 1
            self.k_head_size = self.model.kv_lora_rank
            self.v_head_size = self.model.qk_rope_head_dim
        else:
            self.k_head_size = self.head_size
            self.v_head_size = self.head_size

        logger.info(f'model:\n {self.model}')

        if self._is_aclgraph_enabled:
            self.model = AclGraphBackend(self.model, self._max_batch_size)
            self._max_num_token = self.model.capture_sizes[-1]
            logger.info(f"AclGraph is enabled. Graph batch sizes contains {self.model.capture_sizes}.")

    def forward(
        self,
        kv_caches: list,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_context: 'ForwardContext'
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass through the model.
        
        Args:
            kv_caches: List of KV cache tuples.
            input_ids: Input token IDs.
            position_ids: Position IDs.
            forward_context: Forward context containing metadata.
            
        Returns:
            Logits tensor, or tuple of (logits, hidden_states) if speculative tokens enabled.
        """
        if self._kv_cache_info.check_diff(kv_caches):
            self._bind_kv_cache(kv_caches)

        if not self._is_warmup_completed:
            self._init_buffer()
            if self._is_aclgraph_enabled:
                self._warm_up_and_compile()
            self._is_warmup_completed = True

        set_forward_context(forward_context)
        if forward_context.dp_metadata is not None:
            forward_context.dp_metadata.get_num_tokens_across_dp_cpu()
        # Do operations like D2D to prepare for the forward function
        if self._is_aclgraph_enabled and not forward_context.is_prefill:
            input_ids, position_ids = self._padding_forward_context(input_ids, position_ids)

        # build layerwise attn_metadata for eager mode
        attn_metadata = forward_context.attn_metadata
        attn_metadata_dict = build_layerwise_attn_metadata(attn_metadata)
        forward_context.attn_metadata_dict = attn_metadata_dict

        torch.npu.current_stream().synchronize()

        hidden_states = self.model(input_ids, position_ids)
        if forward_context.dp_metadata is not None:
            dp_metadata = forward_context.dp_metadata
            dp_metadata.num_tokens_across_dp_cpu = dp_metadata.num_actual_tokens_across_dp_cpu
        hidden_states = self.model.maybe_gather_and_unpad_for_flashcomm(hidden_states)
        hidden_states = self.model.maybe_pad_and_gather_cross_dp_and_unpad(hidden_states)
        if not self.is_draft_model and get_parallel_info_manager().get(ParallelType.ATTN_CP).is_enabled():
            hidden_states_cp = self.model.maybe_all_gather_cp(hidden_states)
            logits = self.model.compute_logits(hidden_states_cp)
        else:
            logits = self.model.compute_logits(hidden_states)
        if self.num_speculative_tokens > 0:
            return logits, hidden_states
        return logits

    def build_forward_context(self, model_inputs: ModelInput) -> 'ForwardContext':
        """Build forward context from model inputs.
        
        Do operations like H2D to prepare for the forward function.
        It will be used by eager mode and aclgraph mode.
        
        Args:
            model_inputs: Model input data.
            
        Returns:
            Forward context.
        """
        forward_context = create_forward_context(
            model_inputs, self._mask, self.num_speculative_tokens)
        forward_context.to_device(self.device)
        return forward_context

    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        """Generate position ids.
        
        Args:
            input_ids: Input token IDs.
            
        Returns:
            Position IDs.
        """
        position_ids = self.input_builder.generate_position_ids(input_ids)
        return position_ids

    def clear_internal_tensors(self) -> None:
        """Clear internal tensors.
        
        NOTE: This function is only implemented in atbgraph and will be depreciated.
        """
        pass

    def _padding_forward_context(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad forward context for graph mode.
        
        Do operations like D2D to prepare for the forward function.
        It will be used by aclgraph mode only.
        
        Args:
            input_ids: Input token IDs.
            position_ids: Position IDs.
            
        Returns:
            Tuple of (padded_input_ids, padded_position_ids).
        """
        # get global forward_context
        forward_context = get_forward_context()

        # find num_tokens and check if num_actual_tokens larger than biggest size
        num_actual_tokens = input_ids.shape[0]
        forward_context.num_actual_tokens = num_actual_tokens

        padding_tokens = num_actual_tokens
        if forward_context.dp_metadata is not None:
            padding_tokens = forward_context.dp_metadata.num_tokens_across_dp_cpu.max().item()
        num_tokens = self.model.get_padded_graph_size(padding_tokens)

        if num_tokens > self._max_num_token:
            logger.info(f"Current batch size {num_tokens} is larger than {self._max_num_token},"
                        " using eager mode.")
            return input_ids, position_ids

        # do d2d operation for data not in atten_metadata
        input_buffer.get("input_ids")[:num_actual_tokens].copy_(input_ids[:num_actual_tokens])
        input_buffer.get("position_ids")[:num_actual_tokens].copy_(position_ids[:num_actual_tokens])

        forward_context.batch_descriptor = BatchDescriptor(num_tokens,
            forward_context.batch_descriptor.is_flash_comm_enabled)
        forward_context.attn_metadata.num_tokens = num_tokens

        # do d2d operation
        input_ids = input_buffer.get("input_ids")[:num_tokens]
        position_ids = input_buffer.get("position_ids")[:num_tokens]
        forward_context.copy(num_actual_tokens, num_tokens)

        return input_ids, position_ids

    def _init_buffer(self) -> None:
        """Initialize input buffers for graph mode."""
        ForwardContext.register(self._max_num_token, self.device)
        input_buffer.register("input_ids", torch.zeros(self._max_num_token, dtype=torch.int32, device=self.device))
        input_buffer.register("position_ids", torch.zeros(self._max_num_token, dtype=torch.int64, device=self.device))

    def _warm_up_and_compile(self) -> None:
        """Warm up and compile graphs for different batch sizes."""
        max_memory = get_npu_hbm_info().get_hbm_capacity()
        current_memory = int(max_memory * get_npu_hbm_info().get_hbm_usage()) / (1024 ** 3)
        logger.info(f"Before capturing, {current_memory=}")
        set_aclgraph_capturing_enabled(True)
        for num_tokens in tqdm(
            list(reversed(self.model.capture_sizes)),
            desc="Capturing acl graph",
            disable=get_parallel_info_manager().rank
        ):
            self._dummy_run(num_tokens)
        set_aclgraph_capturing_enabled(False)

        current_memory = int(max_memory * get_npu_hbm_info().get_hbm_usage()) / (1024 ** 3)
        logger.info(f"After capturing, {current_memory=}")

        output_buffer_hbm = 0
        for k in self.model.output_buffer:
            v = self.model.output_buffer[k]
            if torch.is_tensor(v):
                tmp = v.numel() * v.element_size() / (1024 ** 2)
                output_buffer_hbm += tmp
        logger.info(f"After capturing, {output_buffer_hbm=} MB")

    def _dummy_run(self, num_tokens: int) -> None:
        """Run dummy forward pass for graph compilation.
        
        Args:
            num_tokens: Number of tokens for dummy run.
            
        Raises:
            ValueError: If num_tokens exceeds max_num_token.
        """
        if num_tokens > self._max_num_token:
            raise ValueError("Dummy run failed for capture batch size is larger than max input_len.")
        input_ids, position_ids = self._generate_dummy_inputs(num_tokens)
        _ = self.model(input_ids, position_ids)

    def _generate_dummy_inputs(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate dummy inputs for graph compilation.
        
        Args:
            num_tokens: Number of tokens.
            
        Returns:
            Tuple of (input_ids, position_ids).
        """
        is_prefill = False
        input_ids = input_buffer.get("input_ids")[:num_tokens]
        position_ids = input_buffer.get("position_ids")[:num_tokens]
        slot_mapping = input_buffer.get("slot_mapping")[:num_tokens]
        seq_lens = input_buffer.get("seq_lens")[:num_tokens]
        block_tables = input_buffer.get("block_tables")[:num_tokens, :]
        seq_lens_list = seq_lens.cpu().tolist()

        model_inputs = ModelInput(
            is_prefill=is_prefill,   
            input_ids=input_ids,
            position_ids=position_ids,
            block_tables=block_tables,
            slots=slot_mapping,
            context_length=seq_lens,
            max_seq_len=max(seq_lens_list),
            cached_context_length=[1],
            prefill_head_indices=None
        )

        # capturing is True to set address for capturing
        forward_context = create_forward_context(
            model_inputs, self._mask, self.num_speculative_tokens)
        if DPMetadata.is_enabled():
            forward_context.dp_metadata.get_num_tokens_across_dp_cpu()
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens, get_parallel_info_manager().get(ParallelType.ATTN_DP).is_enabled())
        # This parameter will be calculated during D2D operation in the formal inference.
        forward_context.attn_metadata.seq_lens_list = seq_lens_list
        if hasattr(forward_context.attn_metadata, 'prepare_dummy_input'): 
            forward_context.attn_metadata.prepare_dummy_input(num_tokens)
        if forward_context.dp_metadata is not None:
            forward_context.dp_metadata.num_tokens_across_dp_cpu = torch.tensor(
                [num_tokens] * get_parallel_info_manager().get(ParallelType.ATTN_DP).group_size)

        attn_metadata_dict = build_layerwise_attn_metadata(forward_context.attn_metadata)
        forward_context.attn_metadata_dict = attn_metadata_dict
        set_forward_context(forward_context)

        return input_ids, position_ids

    def _bind_kv_cache(self, kv_caches: list) -> None:
        """Bind KV cache to attention layers.
        
        NOTE: The location of this function will be adjusted in the future.
        
        Args:
            kv_caches: List of KV cache tuples.
        """
        self._kv_cache_info.kcache_id = id(kv_caches[0][0])
        self._kv_cache_info.vcache_id = id(kv_caches[0][1])
        self._kv_cache_info.kcache_shape = kv_caches[0][0].shape
        self._kv_cache_info.vcache_shape = kv_caches[0][1].shape

        attns = get_global_attn_dict()
        for i, prefix in enumerate(attns):
            attn_layer = attns[prefix]
            attn_layer.key_cache = kv_caches[i][0]
            attn_layer.value_cache = kv_caches[i][1]
            if isinstance(attn_layer, SFA):
                attn_layer.index_cache = kv_caches[i][2]


def build_layerwise_attn_metadata(
    attn_metadata: AttentionMetadata
) -> dict[str, AttentionMetadata]:
    """Build layerwise attention metadata dictionary.
    
    NOTE: extra_metadata is to input new attribute.
    
    Args:
        attn_metadata: Attention metadata.
        
    Returns:
        Dictionary mapping layer prefixes to attention metadata.
    """
    attns = get_global_attn_dict()
    attn_metadata_dict = {}
    for prefix in attns:
        attn_metadata_dict[prefix] = attn_metadata
    return attn_metadata_dict
