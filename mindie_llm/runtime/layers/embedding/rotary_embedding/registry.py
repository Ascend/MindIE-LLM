# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from functools import wraps
from typing import Any, Callable
from dataclasses import asdict

import torch
import torch_npu

from mindie_llm.runtime.config.huggingface_config import BaseRopeScaling
from .base import RotaryEmbedding


# Global cache dictionary for RoPE instances.
# Key: tuple of (head_size, rotary_dim, max_position, is_neox_style, rope_parameters_tuple, dtype)
# Value: Cached RotaryEmbedding instance
_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}

# Registry mapping rope_type strings to factory functions.
# Key: rope_type identifier (e.g., 'dynamic', 'yarn')
# Value: Factory function with signature:
#   (head_size: int, rotary_dim: int, max_position: int, base: float,
#    is_neox_style: bool, dtype: torch.dtype, rope_config: BaseRopeScaling) -> RotaryEmbedding
_ROPE_REGISTRY: dict[str, Callable] = {}


def register_rope_type(rope_type: str) -> Callable:
    """Decorator to register a RoPE implementation factory function.

    Factory functions must accept the following parameters:
        head_size: int
        rotary_dim: int
        max_position: int
        base: float
        is_neox_style: bool
        dtype: torch.dtype
        rope_config: BaseRopeScaling

    And return a RotaryEmbedding instance.

    Args:
        rope_type: Unique identifier for the RoPE variant (e.g., 'dynamic', 'yarn').

    Returns:
        The original factory function, now registered in _ROPE_REGISTRY.

    Raises:
        ValueError: If the rope_type is already registered.
    """
    def decorator(factory_func: Callable) -> Callable:
        if rope_type in _ROPE_REGISTRY:
            raise ValueError(
                f"RoPE type '{rope_type}' is already registered. "
                f"Use a different name or unregister the existing one."
            )
        _ROPE_REGISTRY[rope_type] = factory_func
        return factory_func
    return decorator


def unregister_rope_type(rope_type: str) -> None:
    """Unregister a RoPE implementation factory function.

    Args:
        rope_type: Identifier of the RoPE variant to remove from registry.

    Note:
        No exception is raised if the rope_type is not currently registered.
    """
    if rope_type in _ROPE_REGISTRY:
        del _ROPE_REGISTRY[rope_type]


def get_registered_rope_types() -> list[str]:
    """Retrieve a list of all registered RoPE type identifiers.

    Returns:
        List of registered rope_type strings (e.g., ['default', 'dynamic', 'yarn']).
    """
    return list(_ROPE_REGISTRY.keys())


def cached_rope_factory(factory_func: Callable) -> Callable:
    """Decorator that adds caching to a RoPE factory function.

    Caches RotaryEmbedding instances based on a composite key derived from:
        - head_size
        - rotary_dim (adjusted by partial_rotary_factor if present)
        - max_position
        - is_neox_style
        - rope_config parameters (converted to hashable tuple)
        - dtype
    
    Automatically handles:
        - partial_rotary_factor adjustment to rotary_dim
        - Conversion of BaseRopeScaling to hashable format
        - Conditional argument passing based on rope_type ('default' vs others)

    Args:
        factory_func: RoPE factory function to wrap. Must return a RotaryEmbedding instance.

    Returns:
        Wrapped factory function with integrated caching logic.

    """
    @wraps(factory_func)
    def wrapper(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        rope_config: BaseRopeScaling,
    ) -> RotaryEmbedding:
        # Convert BaseRopeScaling to hashable format for cache key
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in asdict(rope_config).items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
        
        # Construct composite cache key
        key = (
            head_size,
            rotary_dim,
            max_position,
            is_neox_style,
            rope_parameters_args,
            dtype,
        )
        
        # Return cached instance if available
        if key in _ROPE_DICT:
            return _ROPE_DICT[key]
        
        # Create new instance with appropriate arguments
        if rope_config.rope_type != "default":
            rotary_emb = factory_func(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                rope_config,
            )
        else:
            # 'default' RoPE implementations typically don't require rope_config
            rotary_emb = factory_func(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
            )
        
        # Cache and return the new instance
        _ROPE_DICT[key] = rotary_emb
        return rotary_emb
    
    return wrapper


def clear_rope_cache() -> None:
    """Clear all cached RoPE instances.

    Use this when:
        - Model configuration changes dynamically
        - Memory pressure requires cache eviction
        - Testing different RoPE configurations in the same runtime
    """
    _ROPE_DICT.clear()


def get_rope_factory(rope_type: str) -> Callable | None:
    """Retrieve the factory function for a registered RoPE type.

    Args:
        rope_type: Identifier of the desired RoPE variant.

    Returns:
        Factory function if registered, otherwise None.

    Example:
        factory = get_rope_factory("dynamic")
        if factory:
            rope = factory(head_size, rotary_dim, ...)
    """
    return _ROPE_REGISTRY.get(rope_type)
