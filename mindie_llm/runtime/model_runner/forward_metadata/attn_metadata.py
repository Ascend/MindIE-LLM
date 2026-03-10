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
from typing import Optional, List

import torch

from mindie_llm.runtime.model_runner.forward_metadata.module_metadata import ModuleMetadata


@dataclass
class AttentionMetadata(ModuleMetadata):
    """Metadata for attention operations.
    
    Attributes:
        slot_mapping: Slot mapping tensor for KV cache.
        seq_lens: Sequence lengths tensor.
        block_tables: Block tables tensor for paged attention.
        attn_mask: Attention mask tensor.
        cos_table: Cosine table for rotary embedding.
        sin_table: Sine table for rotary embedding.
        seq_lens_list: List of sequence lengths (optional).
        max_seq_len: Maximum sequence length.
    """
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    attn_mask: torch.Tensor
    seq_lens_list: Optional[List[int]] = None
    max_seq_len: int = 0
