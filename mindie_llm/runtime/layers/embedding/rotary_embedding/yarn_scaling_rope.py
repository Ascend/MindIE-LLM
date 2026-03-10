# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on aiter.
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math

import torch

from .base import RotaryEmbedding


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0):
    """Compute YaRN magnitude scaling factor."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    """Compute dimension index for given number of rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float | int, float | int]:
    """Compute [low, high] dimension range for frequency blending."""
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Linear ramp mask from 1 (at low) to 0 (at high)."""
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class YarnScalingRotaryEmbedding(RotaryEmbedding):
    """YaRN (Yet another RoPE extensioN) rotary embedding with context scaling.

    Extends standard RoPE to support long-context inference by blending interpolated
    and extrapolated frequency components, along with attention magnitude scaling.
    Based on the method proposed by Peng et al. (https://github.com/jquesnelle/yarn).

    Key features:
      - Smooth transition between interpolation (low frequencies) and extrapolation (high frequencies)
      - Attention magnitude compensation via `mscale`
      - Configurable correction boundaries via `beta_fast` and `beta_slow`

    Inherits from `RotaryEmbedding` and overrides cache computation to support scaled context.
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        original_max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype = None,
        *,
        factor=1,
        extrapolation_factor: float = 1,
        attention_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        apply_yarn_scaling: bool = True,
        truncate: bool = True,
        mscale: float | None = 1.0
    ) -> None:
        """Initialize YaRN rotary embedding.

        Args:
            head_size: Dimension of each attention head.
            rotary_dim: Number of dimensions to apply RoPE on.
            original_max_position_embeddings: Original maximum sequence length used during pretraining.
            base: Base period for RoPE frequencies.
            is_neox_style: Rotation style (Neox vs GPT-J).
            dtype: Data type for internal caches.
            factor: Context scaling factor (e.g., 2.0 extends context to 2x original length).
            extrapolation_factor: Weight for extrapolated frequencies in the blend.
            attention_factor: Base attention magnitude scaling factor.
            beta_fast: Number of rotations for the high-frequency (extrapolation) boundary.
            beta_slow: Number of rotations for the low-frequency (interpolation) boundary.
            apply_yarn_scaling: Whether to apply the YaRN-specific magnitude scaling (`mscale`).
            truncate: Whether to floor/ceil correction bounds to integer dimensions.
        """
        self.scaling_factor = factor
        self.extrapolation_factor = extrapolation_factor
        self.attention_factor = attention_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate
        self.mscale = mscale if mscale else (
            float(yarn_get_mscale(self.scaling_factor) * self.attention_factor)
            if apply_yarn_scaling
            else float(self.attention_factor)
        )
        self.original_max_position_embeddings = original_max_position_embeddings
        max_position_embeddings = int(self.scaling_factor * original_max_position_embeddings)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

    def _compute_inv_freq(self) -> None:
        """Compute blended inverse frequencies for YaRN.

        Combines interpolated frequencies (scaled by `factor`) and extrapolated frequencies
        using a linear ramp mask that depends on `beta_fast` and `beta_slow`.

        Returns:
            inv_freq: Blended inverse frequencies of shape `[rotary_dim // 2]`.
        """
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2).to(torch.float32) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.original_max_position_embeddings,
            self.truncate,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float32)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin_cache(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute cosine and sine caches for the extended context length.

        The cache length is `max_position_embeddings`, and all values
        are scaled by `mscale` to preserve attention magnitude.

        Returns:
            A tuple of (cos, sin, cos_sin) tensors, where:
                - cos/sin: Shape `[max_position_embeddings, rotary_dim // 2]`
        """
        t = torch.arange(
            self.max_position_embeddings
        ).to(torch.float32)
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        cos = freqs.cos().to(self.dtype) * self.mscale
        sin = freqs.sin().to(self.dtype) * self.mscale
        self.register_buffer("cos_cache", cos, persistent=False) # [max_position_embeddings, rotary_dim // 2]
        self.register_buffer("sin_cache", sin, persistent=False) # [max_position_embeddings, rotary_dim // 2]
