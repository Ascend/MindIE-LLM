# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from typing import NamedTuple, Optional
import torch

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig
from mindie_llm.runtime.layers.attention import get_global_attn_dict
from mindie_llm.runtime.model_runner.forward_metadata.module_metadata import (
    ModuleMetadata,
)
from mindie_llm.runtime.model_runner.forward_metadata.dp_metadata import DPMetadata
from mindie_llm.runtime.model_runner.forward_metadata.attn_metadata import (
    AttentionMetadata,
)
from mindie_llm.runtime.model_runner.forward_metadata.mtp_metadata import MtpMetadata
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType


_MC2_TOKEN_CAPACITY = None
MAX_MC2_OPERATOR_CAPACITY = 512


def set_mc2_token_capacity(max_num_reqs, uniform_decode_query_len):
    parallel_mgr = get_parallel_info_manager()
    max_num_tokens = max_num_reqs * uniform_decode_query_len
    max_num_tokens_per_device = max_num_tokens
    # CP
    if parallel_mgr.get(ParallelType.ATTN_CP).is_enabled():
        cp_size = parallel_mgr.get(ParallelType.ATTN_CP).group_size
        max_num_tokens_per_device = (max_num_tokens_per_device + cp_size - 1) // cp_size
    # TP + flash_comm
    if (
        parallel_mgr.get(ParallelType.ATTN_DP).is_enabled()  # check if flash_comm enabled
        and parallel_mgr.get(ParallelType.ATTN_TP).is_enabled()
    ):
        tp_size = parallel_mgr.get(ParallelType.ATTN_TP).group_size
        max_num_tokens_per_device = (max_num_tokens_per_device + tp_size - 1) // tp_size

    # Num tokens of DP group or CP group
    global _MC2_TOKEN_CAPACITY
    # Use integer arithmetic for ceiling division.
    _MC2_TOKEN_CAPACITY = min(max_num_tokens_per_device, MAX_MC2_OPERATOR_CAPACITY)


def get_mc2_token_capacity():
    return _MC2_TOKEN_CAPACITY


class BatchDescriptor(NamedTuple):
    """Descriptor for batch information.

    Attributes:
        num_tokens: Number of tokens in the batch.
        is_flash_comm_enabled: Whether flash communication is enabled.
    """

    num_tokens: int
    is_flash_comm_enabled: bool = False


@dataclass
class ForwardContext:
    """Context information for forward pass.

    Attributes:
        is_prefill: Whether this is a prefill stage.
        lm_head_indices: Indices for language model head.
        attn_metadata: Attention metadata.
        attn_metadata_dict: Dictionary of attention metadata per layer.
        dp_metadata: Data parallel metadata.
        capturing: Whether capturing mode is enabled.
        num_actual_tokens: Actual number of tokens.
        batch_descriptor: Batch descriptor.
        mc2_mask: MC2 mask tensor.
    """

    is_prefill: Optional[bool] = None
    lm_head_indices: Optional[torch.Tensor] = None

    attn_metadata: Optional[AttentionMetadata] = None
    attn_metadata_dict: Optional[dict[str, AttentionMetadata]] = None

    dp_metadata: Optional[DPMetadata] = None

    capturing: bool = False
    num_actual_tokens: int = 0
    batch_descriptor: Optional[BatchDescriptor] = None
    mc2_mask: Optional[torch.Tensor] = None

    mtp_metadata: MtpMetadata = None
    sub_forward_context = None

    @staticmethod
    def register(
        max_num_token: int,
        device: torch.device,
        hf_config: HuggingFaceConfig,
        max_block_per_seq: int,
    ) -> None:
        """Register buffer for attention metadata.

        Args:
            max_num_token: Maximum number of tokens.
            device: Target device.
            hf_config: HuggingFaceConfig of model.
            max_block_per_seq: Maximum number of blocks per sequence.
        """
        attns = get_global_attn_dict()
        attn_layer = attns[next(iter(attns))]
        attn_backend = attn_layer.get_attn_backend()
        metadata_cls = attn_backend.get_builder_cls().get_metadata_cls()
        metadata_cls.register_buffer(max_num_token, device, max_block_per_seq)
        MtpMetadata.register_buffer(max_num_token, device, hf_config)

    def to_device(self, device: torch.device) -> None:
        """Move context tensors to the specified device.

        Args:
            device: Target device.
        """
        if self.attn_metadata is not None:
            self.attn_metadata.to_device(device)
        if self.dp_metadata is not None:
            self.dp_metadata.to_device(device)
        self.lm_head_indices = torch.tensor(self.lm_head_indices, dtype=torch.int64, device=device)

    def record_stream(self, stream: torch.npu.Stream) -> None:
        """Record the stream for all tensors in the context.

        Args:
            stream: The stream to record.
        """

        def _record(obj):
            if isinstance(obj, torch.Tensor) and obj.is_npu:
                obj.record_stream(stream)
            elif isinstance(obj, list) or isinstance(obj, tuple):
                for item in obj:
                    _record(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _record(v)
            elif isinstance(obj, ModuleMetadata):
                obj.record_stream(stream)

        for value in self.__dict__.values():
            _record(value)

    def copy(self, num_actual_tokens: int, num_tokens: int) -> None:
        """Copy metadata with new token counts.

        Args:
            num_actual_tokens: Actual number of tokens.
            num_tokens: Total number of tokens (including padding).
        """
        self.attn_metadata.copy(num_actual_tokens, num_tokens)
        self.mtp_metadata.copy(num_actual_tokens, num_tokens)
        if hasattr(self.attn_metadata, "mc2_mask"):
            self.mc2_mask = self.attn_metadata.mc2_mask
        if self.dp_metadata is not None:
            self.dp_metadata.copy(num_actual_tokens, num_tokens)


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context.

    Returns:
        The current forward context.

    Raises:
        RuntimeError: If forward context is not set.
    """
    if _forward_context is None:
        raise RuntimeError("Forward context is not set. Please use `set_forward_context` to set the forward context.")
    return _forward_context


def set_forward_context(context: ForwardContext) -> None:
    """Set the current forward context.

    Args:
        context: The forward context to set.
    """
    global _forward_context
    _forward_context = context


# NOTE: This API will be refactored
def create_forward_context(model_inputs: any, mask: torch.Tensor, num_speculative_tokens: int = 0) -> ForwardContext:
    """Create a forward context from model inputs.

    Args:
        model_inputs: Model input data.
        mask: Attention mask.
        num_speculative_tokens: Number of speculative tokens.

    Returns:
        Created forward context.
    """
    attns = get_global_attn_dict()
    attn_layer = attns[next(iter(attns))]
    attn_backend = attn_layer.get_attn_backend()
    metadata_cls = attn_backend.get_builder_cls().get_metadata_cls()
    # NOTE: This API will be refactored
    attn_metadata = metadata_cls.from_model_input(model_inputs, mask, num_speculative_tokens)

    dp_metadata = DPMetadata.from_model_input(model_inputs)

    num_actual_tokens = len(model_inputs.input_ids)
    batch_descriptor = BatchDescriptor(
        num_actual_tokens,
        get_parallel_info_manager().get(ParallelType.ATTN_TP).is_enabled(),
    )

    if model_inputs.prefill_head_indices is not None:
        lm_head_indices = model_inputs.prefill_head_indices
    else:
        lm_head_indices = torch.arange(model_inputs.input_ids.shape[0], device=model_inputs.input_ids.device)

    return ForwardContext(
        is_prefill=model_inputs.is_prefill,
        lm_head_indices=lm_head_indices,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        num_actual_tokens=num_actual_tokens,
        batch_descriptor=batch_descriptor,
        mtp_metadata=MtpMetadata.from_model_input(model_inputs),
    )
