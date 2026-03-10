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
import torch_npu

from mindie_llm.runtime.layers.embedding.rotary_embedding.yarn_scaling_rope import (
    YarnScalingRotaryEmbedding,
    yarn_get_mscale
)


class DeepseekV3YarnRotaryEmbedding(YarnScalingRotaryEmbedding):
    """DeepSeek-V3 specialized YaRN rotary embedding with mscale_all_dim scaling.

    Extends standard YaRN scaling with DeepSeek-V3's additional magnitude scaling
    parameter (mscale_all_dim) for fine-grained attention magnitude control.
    """
    def __init__(
        self,
        dim,
        original_max_position_embeddings=4096,
        base=10000,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        is_neox_style=True,
        dtype=None,
        mscale=1.0,
        mscale_all_dim=1.0,
    ) -> None:
        """Initialize DeepSeek-V3 YaRN rotary embedding.

        Args:
            dim: Rotary embedding dimension (applied to both head and rotary dims).
            original_max_position_embeddings: Original context length before scaling.
            base: Base frequency for rotary embedding (theta).
            factor: Context extension scaling factor (>1.0 for extrapolation).
            beta_fast: YaRN fast decay window parameter.
            beta_slow: YaRN slow decay window parameter.
            is_neox_style: Use NeoX-style interleaved rotation (default: True).
            dtype: Data type for embedding tensors (e.g., torch.float16).
            mscale: Base magnitude scaling factor for attention preservation.
            mscale_all_dim: DeepSeek-V3 specific scaling factor applied across all dimensions.
        """
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, dim, original_max_position_embeddings, base,
        dtype=dtype,
            is_neox_style=is_neox_style,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            mscale=mscale
        )
    
    def set_cos_sin_indexed_cache(self, postions) -> None:
        """Create position-indexed cosine/sine caches with dimension doubling.

        Extracts position-specific rotary values from precomputed caches and
        duplicates them across the last dimension to match attention head layout.

        Args:
            postions: 1D tensor of position indices to index into the cache.
        """
        cos_indexed_cache = torch.index_select(self.cos_cache, dim=0, index=postions.view(-1)).unsqueeze(1).unsqueeze(1)
        sin_indexed_cache = torch.index_select(self.sin_cache, dim=0, index=postions.view(-1)).unsqueeze(1).unsqueeze(1) 
        cos_indexed_cache = torch.cat((cos_indexed_cache, cos_indexed_cache), dim=-1)
        sin_indexed_cache = torch.cat((sin_indexed_cache, sin_indexed_cache), dim=-1)
        self.register_buffer("cos_indexed_cache", cos_indexed_cache, persistent=False) # [seq_len, 1, 1, rotary_dim]
        self.register_buffer("sin_indexed_cache", sin_indexed_cache, persistent=False)

    def _compute_cos_sin_cache(self) -> None:
        """Precompute cosine/sine caches with DeepSeek-V3 specific magnitude scaling.

        Applies dual scaling factors (mscale and mscale_all_dim) to preserve attention
        magnitude during context extrapolation. The effective scale is mscale/mscale_all_dim.
        """
        t = torch.arange(
            self.max_position_embeddings
        ).to(torch.float32)
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        cos = freqs.cos().to(self.dtype) * _mscale
        sin = freqs.sin().to(self.dtype) * _mscale
        self.register_buffer("cos_cache", cos, persistent=False) # [max_position_embeddings, rotary_dim // 2]
        self.register_buffer("sin_cache", sin, persistent=False) # [max_position_embeddings, rotary_dim // 2]
