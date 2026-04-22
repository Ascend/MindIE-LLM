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
import torch
from mindie_llm.runtime.model_runner.forward_context_exp import BatchDescriptor
from mindie_llm.runtime.model_runner.forward_metadata.module_metadata import ModuleMetadata
from mindie_llm.runtime.model_runner.forward_metadata.attn_metadata import AttentionMetadata
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType


@dataclass
class MtpMetadata(ModuleMetadata):
    last_hidden_states: torch.Tensor = None


@dataclass
class ForwardContext:
    attn_metadata: AttentionMetadata
    lm_head_indices: torch.Tensor
    mtp_metadata: MtpMetadata
    is_prefill: bool
    num_tokens_across_dp_cpu: torch.Tensor
    # following ones for aclgraph
    capturing: bool = False  # default eager mode
    num_tokens: int = 0
    num_actual_tokens: int = 0
    # NOTE: rope generate in DeepSeekV3Model;
    seq_lens: torch.Tensor = None
    batch_descriptor: BatchDescriptor = None
    attn_metadata_dict: dict[str, AttentionMetadata] = None
    mc2_mask: torch.Tensor = None


_forward_context: ForwardContext | None = None


def get_forward_context():
    """Lazy proxy to avoid import-time circular dependency."""
    from mindie_llm.runtime.model_runner.forward_context_exp import (
        get_forward_context as _get_forward_context,
    )

    return _get_forward_context()


def set_forward_context(context):
    """Lazy proxy to avoid import-time circular dependency."""
    from mindie_llm.runtime.model_runner.forward_context_exp import (
        set_forward_context as _set_forward_context,
    )

    _set_forward_context(context)


def create_forward_context(input_metadata: dict, capturing: bool = False):
    attn_metadata = AttentionMetadata.from_dict(input_metadata)
    lm_head_indices = input_metadata.get("lm_head_indices", None)
    mtp_metadata = MtpMetadata.from_dict(input_metadata)
    is_prefill = input_metadata["is_prefill"]
    num_tokens = input_metadata.get("num_tokens", 0)
    num_actual_tokens = input_metadata.get("num_actual_tokens", 0)
    seq_lens = input_metadata["seq_lens"]
    num_tokens_across_dp_cpu = input_metadata["num_tokens_across_dp_cpu"]
    batch_descriptor = BatchDescriptor(num_tokens, get_parallel_info_manager().get(ParallelType.ATTN_DP).is_enabled())
    mc2_mask = input_metadata.get("mc2_mask", None)

    return ForwardContext(
        attn_metadata=attn_metadata,
        lm_head_indices=lm_head_indices,
        mtp_metadata=mtp_metadata,
        is_prefill=is_prefill,
        num_tokens=num_tokens,
        num_actual_tokens=num_actual_tokens,
        capturing=capturing,
        seq_lens=seq_lens,
        num_tokens_across_dp_cpu=num_tokens_across_dp_cpu,
        batch_descriptor=batch_descriptor,
        mc2_mask=mc2_mask,
    )
