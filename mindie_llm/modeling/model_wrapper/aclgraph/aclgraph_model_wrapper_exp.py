# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Dict, List, Iterable, Optional, Any
import torch
import numpy as np

from mindie_llm.modeling.model_wrapper.model_info import ModelInfo
from mindie_llm.modeling.model_wrapper.wrapper import ModelWrapper
from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.model_runner.model_runner_exp import ModelRunnerExp
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.text_generator.utils.model_input import ModelInput
from mindie_llm.runtime.model_runner.forward_context_exp import ForwardContext


class AclGraphModelWrapperExp(ModelWrapper):
    """Experimental ACL graph model wrapper.
    
    This wrapper provides an interface for running models with ACL graph backend.
    It handles model initialization, input preparation, and forward passes.
    """
    
    def __init__(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        npu_device_id: int,
        model_id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the ACL graph model wrapper.
        
        Args:
            rank: Process rank in distributed setup.
            local_rank: Local rank in distributed setup.
            world_size: Total number of processes.
            npu_device_id: NPU device ID.
            model_id: Model identifier or path.
            **kwargs: Additional keyword arguments including:
                - trust_remote_code: Whether to trust remote code.
                - load_tokenizer: Whether to load tokenizer.
                - tokenizer_path: Path to tokenizer.
                - max_position_embeddings: Maximum position embeddings.
                - num_speculative_tokens: Number of speculative tokens.
                - max_batch_size: Maximum batch size.
                - models: Dictionary of model configurations.
                - tp: Tensor parallelism size.
                - dp: Data parallelism size.
                - cp: Context parallelism size.
                - moe_tp: MoE tensor parallelism size.
                - moe_ep: MoE expert parallelism size.
                - role: Model role.
                - sampler_config: Sampler configuration.
                - distributed_enable: Whether distributed mode is enabled.
        """
        logger.debug(f"Enter AclGraphModelWrapper initialization.")
        self.model_runner = ModelRunnerExp(
            model_name_or_path=model_id,
            rank=rank,
            local_rank=local_rank,
            npu_id=npu_device_id,
            world_size=world_size,
            trust_remote_code=kwargs.get('trust_remote_code', False),
            load_tokenizer=kwargs.get('load_tokenizer', True),
            tokenizer_path=kwargs.get('tokenizer_path', None),
            max_position_embeddings=kwargs.get("max_position_embeddings"),
            num_speculative_tokens=kwargs.get('num_speculative_tokens'),
            max_batch_size=kwargs.get("max_batch_size", -1),
            models_dict=kwargs.get("models", None),
            tp=kwargs.get('tp', -1),
            dp=kwargs.get('dp', -1),
            cp=kwargs.get('cp', -1),
            moe_tp=kwargs.get('moe_tp', -1),
            moe_ep=kwargs.get('moe_ep', -1),
            role=kwargs.get('role', 'standard'),
            max_seq_len=kwargs.get("max_seq_len", -1),
            sampler_config=kwargs.get('sampler_config', None),
            distributed_enable=kwargs.get("distributed_enable", False),
        )
        # NOTE: These attributes maybe depreciated after TG is refactored.
        self.config = self.model_runner.config
        self.config_dict = self.model_runner.config_dict
        self.tokenizer = self.model_runner.tokenizer
        self.device = self.model_runner.device
        self.rank = rank
        self.mapping = get_parallel_info_manager()
        self.dp_size = get_parallel_info_manager().get(ParallelType.ATTN_DP).group_size
        self.sp_size = get_parallel_info_manager().get(ParallelType.ATTN_INNER_SP).group_size
        self.cp_size = get_parallel_info_manager().get(ParallelType.ATTN_CP).group_size

        logger.debug("Start to load weights.")
        self.model_runner.load_weights()
        logger.debug("Weights are loaded successfully.")

        # NOTE: These attributes maybe depreciated after TG is refactored.
        self.model_info = ModelInfo(
            self.device,
            self.model_runner.kv_cache_dtype,
            torch.tensor([], dtype=self.model_runner.kv_cache_dtype).element_size(),
            self.model_runner.num_layers,
            self.model_runner.num_kv_heads,
            self.model_runner.head_size,
            k_head_size=self.model_runner.k_head_size,
            v_head_size=self.model_runner.v_head_size,
            enable_nz=self.model_runner.enable_nz,
            kvcache_quant_layers=self.model_runner.kvcache_quant_layers,
            index_head_dim=self.model_runner.index_head_dim,
            num_index_heads=self.model_runner.num_index_heads
        )
        self.max_position_embeddings = self.model_runner.max_position_embeddings
        self.adapter_manager = self.model_runner.adapter_manager
        self.is_multimodal = False
        self.model = self.model_runner.model
        self.is_multimodal = getattr(self.model, "is_multimodal", False)

    def forward(
        self,
        model_inputs: ModelInput,
        npu_cache: Optional[Any] = None,
        **kwargs: Any
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass through the model.
        
        Args:
            model_inputs: Model input data.
            npu_cache: Optional NPU cache for KV cache.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Logits tensor, or tuple of (logits, hidden_states) if speculative tokens enabled.
        """
        model_inputs, kwargs = self.prepare_model_inputs(model_inputs, **kwargs)
        result = self.forward_from_model_inputs(
            npu_cache, model_inputs.input_ids, model_inputs.position_ids, model_inputs.forward_context, **kwargs)
        return result

    def prepare_model_inputs(self, model_inputs: ModelInput, **kwargs) -> ModelInput:
        """Prepare model inputs for forward pass.
        
        Performs H2D (Host to Device) operations and builds forward context.
        NOTE: `PluginManager` will modify `model_inputs.input_lengths`.
        Assign `forward_context.seq_lens` to `model_inputs.input_lengths` so that
        they can share same address.
        
        Args:
            model_inputs: Model input data.
            
        Returns:
            Prepared model inputs with forward context.
        """
        # do h2d operation
        model_inputs.input_ids = torch.tensor(model_inputs.input_ids, dtype=torch.int64, device=self.device)
        model_inputs.position_ids = torch.tensor(model_inputs.position_ids, dtype=torch.int64, device=self.device)
        model_inputs.block_tables_array = model_inputs.block_tables

        q_lens = kwargs.get("q_lens", None)
        if q_lens is not None:
            model_inputs.q_lens = torch.tensor(q_lens).to(self.device)
            kwargs["q_lens"] = model_inputs.q_lens
        mtp_logits_gather_indices = kwargs.get('mtp_logits_gather_indices', None)
        if mtp_logits_gather_indices is not None:
            kwargs["mtp_logits_gather_indices"] = mtp_logits_gather_indices.to(self.device)
        shard_effective_token_indices = kwargs.get("shard_effective_token_indices", None)
        if shard_effective_token_indices is not None:
            kwargs["shard_effective_token_indices"] = torch.tensor(shard_effective_token_indices).to(self.device)
        lm_head_local_dp = kwargs.get("lm_head_local_dp", None)
        if lm_head_local_dp is not None:
            kwargs["lm_head_local_dp"] = torch.tensor(lm_head_local_dp).to(self.device)

        sub_model_inputs = kwargs.get("sub_model_inputs", None)
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

        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is not None:
            model_inputs.last_hidden_states = hidden_states.to(self.device)
            kwargs["hidden_states"] = model_inputs.last_hidden_states

        forward_context = self.model_runner.build_forward_context(model_inputs, **kwargs)
        # NOTE: `PluginManager` will modify `model_inputs.input_lengths`.
        # Assign `forward_context.seq_lens` to `model_inputs.input_lengths` so that they can share same address.
        model_inputs.input_lengths = forward_context.attn_metadata.seq_lens
        model_inputs.forward_context = forward_context
        return model_inputs, kwargs

    def forward_from_model_inputs(
        self,
        npu_cache: Optional[Any] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        forward_context: Optional[ForwardContext] = None,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass from prepared model inputs.
        
        Args:
            npu_cache: Optional NPU cache for KV cache.
            input_ids: Input token IDs tensor.
            position_ids: Position IDs tensor.
            forward_context: Forward context containing metadata.
            
        Returns:
            Logits tensor, or tuple of (logits, hidden_states) if speculative tokens enabled.
            
        Raises:
            Exception: If forward pass fails.
        """
        try:
            result = self.model_runner.forward(
                npu_cache,
                input_ids,
                position_ids,
                forward_context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in `forward_tensor`: {e}")
            raise e

        return result

    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        """Generate position IDs from input token IDs.
        
        Args:
            input_ids: Input token IDs array.
            
        Returns:
            Position IDs iterable.
            
        Raises:
            Exception: If position ID generation fails.
        """
        try:
            position_ids = self.model_runner.generate_position_ids(input_ids)
        except Exception as e:
            logger.error(f"Error in generate_position_ids: {e}")
            raise e
        return position_ids

    def make_context(
        self,
        conversation: List[Dict[str, str]],
        **kwargs: Any
    ) -> Any:
        """Make context from conversation.
        
        Args:
            conversation: List of conversation dictionaries with role and content.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Generated context.
            
        Raises:
            Exception: If context generation fails.
        """
        try:
            context = self.model_runner.input_builder.make_context(self.rank, conversation, **kwargs)
        except Exception as e:
            logger.error(f"Error in make_context: {e}")
            raise e
        return context

    def resume_hccl_comm(self) -> None:
        """Resume HCCL communication.
        
        NOTE: Aclgraph currently doesn't support this function.
        
        Raises:
            NotImplementedError: Always raised as this function is not supported.
        """
        # NOTE: Aclgraph currently doesn't support this function
        raise NotImplementedError
