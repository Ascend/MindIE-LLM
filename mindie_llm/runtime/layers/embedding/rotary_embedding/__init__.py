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

from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.config.huggingface_config import BaseRopeScaling

from .base import RotaryEmbedding
from .registry import (
    cached_rope_factory,
    get_registered_rope_types,
    get_rope_factory,
    register_rope_type,
)
from .yarn_scaling_rope import YarnScalingRotaryEmbedding
from .deepseek_v3_yarn_scaling_rope import DeepseekV3YarnRotaryEmbedding


@register_rope_type("default")
@cached_rope_factory
def _create_default_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
) -> RotaryEmbedding:
    """Factory function for creating the default RotaryEmbedding implementation.

    This implementation follows the original RoPE formulation without scaling extensions.

    Args:
        head_size: Dimension of each attention head.
        rotary_dim: Dimensionality of the rotary embedding subspace.
        max_position: Maximum sequence length supported by the embedding.
        base: Base value for frequency computation (theta).
        is_neox_style: Whether to use NeoX-style interleaved rotation (default: True).
        dtype: Data type for embedding tensors (e.g., torch.float16, torch.bfloat16).

    Returns:
        Initialized RotaryEmbedding instance.
    """
    
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
    )


@register_rope_type("yarn")
@cached_rope_factory
def _create_yarn_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    rope_config: BaseRopeScaling,
) -> RotaryEmbedding:
    """Factory function for creating YaRN-scaled RotaryEmbedding.

    Implements YaRN (Yet another RoPE extensioN) scaling with configurable parameters
    for extrapolation control and attention scaling.

    Args:
    head_size: Dimension of each attention head.
        rotary_dim: Dimensionality of the rotary embedding subspace.
        max_position: Target maximum sequence length after scaling.
        base: Base value for frequency computation (theta).
        is_neox_style: Whether to use NeoX-style interleaved rotation.
        dtype: Data type for embedding tensors.
        rope_config: Configuration object containing YaRN-specific parameters:
            - original_max_position_embeddings: Original context length before scaling
            - factor: Scaling factor (>1.0 for context extension)
            - extrapolation_factor: Controls extrapolation behavior
            - beta_fast/beta_slow: YaRN attention window parameters
            - mscale: Magnitude scaling factor
            - apply_yarn_scaling: Whether to apply YaRN-specific modifications
            - truncate: Whether to truncate frequencies beyond original context

    Returns:
        Initialized YarnScalingRotaryEmbedding instance.
    """
    original_max_position = getattr(rope_config, 'original_max_position_embeddings', max_position)
    return YarnScalingRotaryEmbedding(
        head_size,
        rotary_dim,
        original_max_position,
        base,
        is_neox_style,
        dtype,
        factor=getattr(rope_config, 'factor', 1.0),
        extrapolation_factor=getattr(rope_config, 'extrapolation_factor', 1.0),
        beta_fast=getattr(rope_config, 'beta_fast', 32),
        beta_slow=getattr(rope_config, 'beta_slow', 1),
        mscale=getattr(rope_config, 'mscale', 1.0),
        apply_yarn_scaling=getattr(rope_config, 'apply_yarn_scaling', True),
        truncate=getattr(rope_config, 'truncate', True),
    )


@register_rope_type("deepseek_yarn")
@cached_rope_factory
def _create_deepseek_scaling_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    rope_config: BaseRopeScaling,
) -> RotaryEmbedding:
    """Factory function for creating DeepSeek-V3 YaRN-scaled RotaryEmbedding.

    Specialized implementation for DeepSeek-V3 architecture with YaRN scaling
    and DeepSeek-specific parameters like mscale_all_dim.

    Args:
        head_size: Dimension of each attention head.
        rotary_dim: Dimensionality of the rotary embedding subspace.
        max_position: Target maximum sequence length after scaling.
        base: Base value for frequency computation (theta).
        is_neox_style: Whether to use NeoX-style interleaved rotation.
        dtype: Data type for embedding tensors.
        rope_config: Configuration object containing DeepSeek-specific parameters:
            - original_max_position_embeddings: Original context length before scaling
            - factor: Scaling factor for context extension
            - beta_fast/beta_slow: YaRN attention window parameters
            - mscale: Magnitude scaling factor
            - mscale_all_dim: DeepSeek-specific magnitude scaling dimension parameter

    Returns:
        Initialized DeepseekV3YarnRotaryEmbedding instance.
    """
    original_max_position = getattr(rope_config, 'original_max_position_embeddings', max_position)
    return DeepseekV3YarnRotaryEmbedding(
        rotary_dim,
        original_max_position,
        base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        factor=getattr(rope_config, 'factor', 1.0),
        beta_fast=getattr(rope_config, 'beta_fast', 32),
        beta_slow=getattr(rope_config, 'beta_slow', 1),
        mscale=getattr(rope_config, 'mscale', 1.0),
        mscale_all_dim=getattr(rope_config, 'mscale_all_dim', None),
    )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    rope_config: BaseRopeScaling,
    is_neox_style: bool = True,
    dtype: torch.dtype | None = None,
) -> RotaryEmbedding:
    """Retrieve or create a RotaryEmbedding instance with automatic caching.

    This function serves as the main entry point for RoPE instantiation:
        1. Selects the appropriate implementation based on rope_config.rope_type
        2. Applies partial rotary embedding if partial_rotary_factor < 1.0
        3. Leverages factory registration and caching mechanisms for efficiency
        4. Handles unknown rope types with informative error messages

    Args:
        head_size: Dimension of each attention head.
        rotary_dim: Base dimensionality of the rotary embedding subspace.
        max_position: Target maximum sequence length after scaling.
        rope_config: Configuration object containing RoPE parameters including:
            - rope_type: Identifier for RoPE variant ('default', 'yarn', etc.)
            - rope_theta: Base frequency value
            - original_max_position_embeddings: Original context length
            - partial_rotary_factor: Fraction of dimensions to apply rotation to
            - Additional variant-specific parameters (e.g., YaRN parameters)
        is_neox_style: Whether to use NeoX-style interleaved rotation (default: True).
        dtype: Data type for embedding tensors. If None, uses torch.get_default_dtype().

    Returns:
        Cached or newly created RotaryEmbedding instance matching the configuration.

    Raises:
        ValueError: If the specified rope_type is not registered.

    Example:
        >>> config = Qwen2RopeScaling(rope_type="yarn", factor=2.0, ...)
        >>> rope = get_rope(128, 128, 8192, config, dtype=torch.bfloat16)
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    rope_type = getattr(rope_config, 'rope_type', 'default')
    base = rope_config.rope_theta
    
    factory_func = get_rope_factory(rope_type)
    
    if factory_func is None:
        _msg = (
            f"Unknown RoPE scaling type '{rope_type}'. "
            f"Available types: {', '.join(get_registered_rope_types())}"
        )
        logger.error(_msg)
        raise ValueError(_msg)

    partial_rotary_factor = getattr(rope_config, "partial_rotary_factor", 1.0)
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    
    return factory_func(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
        rope_config,
    )
