# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Part of this file implemented based on vllm project.
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch.nn as nn
import torch_npu

from mindie_llm.utils.log.logging import logger


class RotaryEmbedding(nn.Module):
    """Base class for Rotary Positional Embedding (RoPE) with NPU acceleration support.

    This module precomputes cosine and sine caches for rotary position embeddings
    and applies them to query and key tensors during the forward pass using an
    optimized NPU kernel (`torch_npu._npu_rotary_embedding`).

    Supports both full and partial rotary embedding (via `rotary_dim < head_size`)
    and both Neox-style (LLaMA) and GPT-J-style interleaved rotation.

    Buffers:
        cos_cache: Precomputed cosine values of shape `[max_position_embeddings, rotary_dim // 2]`.
        sin_cache: Precomputed sine values of shape `[max_position_embeddings, rotary_dim // 2]`.

    Note:
        All caches are registered as non-persistent buffers to avoid saving them in checkpoints.
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        dtype=None
    ) -> None:
        """Initialize the RotaryEmbedding module.

        Args:
            head_size: Dimension of each attention head.
            rotary_dim: Number of dimensions to apply RoPE on. Must be <= head_size.
                        If less than head_size, only the first `rotary_dim` dimensions are rotated.
            max_position_embeddings: Maximum sequence length supported. Determines cache size.
            base: Base value for frequency computation (e.g., 10000.0).
            is_neox_style: If True, uses Neox-style (chunked) rotation.
                           If False, uses GPT-J-style (interleaved) rotation.
            dtype: Data type for internal caches (e.g., torch.float16, torch.bfloat16).
            device: device for rope module
        """
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype if dtype else torch.get_default_dtype()

        self._compute_inv_freq()
        self._compute_cos_sin_cache()

    def get_cos_sin_for_positions(self, positions) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.index_select(self.cos_cache, dim=0, index=positions)
        sin = torch.index_select(self.sin_cache, dim=0, index=positions)
        
        return cos, sin # [q_len, rotary_dim // 2]
    
    def set_cos_sin_indexed_cache(self, positions) -> None:
        """Create position-indexed cosine/sine caches with dimension doubling.

        Extracts position-specific rotary values from precomputed caches and
        duplicates them across the last dimension to match attention head layout.

        Args:
            postions: 1D tensor of position indices to index into the cache.
        """
        cos = torch.index_select(self.cos_cache, dim=0, index=positions)
        sin = torch.index_select(self.sin_cache, dim=0, index=positions)
        cos_indexed_cache = cos.repeat(1, 2).view(1, -1, 1, self.head_size).contiguous()
        sin_indexed_cache = sin.repeat(1, 2).view(1, -1, 1, self.head_size).contiguous()
        self.register_buffer("cos_indexed_cache", cos_indexed_cache, persistent=False) # [seq_len, 1, 1, rotary_dim]
        self.register_buffer("sin_indexed_cache", sin_indexed_cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to query and key tensors.

        Uses the Ascend NPU-optimized kernel `torch_npu.npu_apply_rotary_pos_emb` for high performance.

        Args:
            positions: Position indices, shape `[num_tokens]`.
            query: Query tensor, shape `[num_tokens, num_heads * head_size]`.
            key: Key tensor, shape `[num_tokens, num_kv_heads * head_size]`.

        Returns:
            A tuple of (rotated_query, rotated_key) with same shapes as inputs.
        """
        query_shape = query.shape
        key_shape = key.shape
        cos = self.cos_indexed_cache
        sin = self.sin_indexed_cache
        if self._npu_apply_rotary_pos_emb_support():
            # If cos and sin are generated outside, use npu_apply_rotary_pos_emb to avoid redundant calculation.
            # This method requires head_size and rotary_dim equal 128 and neox_style is True
            query = query.contiguous().view(1, query_shape[0], -1,
                                            self.head_size) # [1, B*S, N_head, D]
            key = key.contiguous().view(1, key_shape[0], -1,
                                            self.head_size) # [1, B*S, N_head, D]
            # Although this function modifies in-place, please retain the function's return value.
            # Otherwise, the graph fusion operation may fail.
            
            query, key = torch_npu.npu_apply_rotary_pos_emb(
                query, key, cos, sin)
            return query, key
        else:
            _msg = ("This method requires head_size and rotary_dim equal 128 and neox_style is True. "
                    f"{self.is_neox_style=}, "
                    f"{self.head_size=}, "
                    f"{self.cos_indexed_cache.shape[-1]=}, "
                    f"{self.sin_indexed_cache.shape[-1]=}.")
            logger.error(_msg)
            raise ValueError(_msg)

    def _npu_apply_rotary_pos_emb_support(self) -> bool:
        return (
            self.is_neox_style
            and self.head_size == 128
            and self.cos_indexed_cache.shape[-1] == 128
            and self.sin_indexed_cache.shape[-1] == 128
        )

    def _compute_inv_freq(self):
        """Compute inverse frequencies for RoPE.

        Returns:
            inv_freq: Tensor of shape `[rotary_dim // 2]` containing 1 / (base^(i / rotary_dim)).
        """
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2).to(torch.float32) / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False) # [rotary_dim // 2, ]

    def _compute_cos_sin_cache(self) -> None:
        """Precompute cosine, sine, and concatenated caches for all positions.

        The frequency matrix is computed as `freqs[i, j] = t[i] * inv_freq[j]`,
        where `t` is the position index and `inv_freq` is the inverse frequency.

        The shape of three tensors:
            - cos: Cosine values, shape `[max_position_embeddings, rotary_dim // 2]`
            - sin: Sine values, shape `[max_position_embeddings, rotary_dim // 2]`
        """
        t = torch.arange(self.max_position_embeddings).to(torch.float32) # [max_position_embeddings, ]
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        cos = freqs.cos().to(self.dtype)
        sin = freqs.sin().to(self.dtype)
        self.register_buffer("cos_cache", cos, persistent=False) # [max_position_embeddings, rotary_dim // 2]
        self.register_buffer("sin_cache", sin, persistent=False) # [max_position_embeddings, rotary_dim // 2]
