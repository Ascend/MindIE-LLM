# get_rope Usage Guide
<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-28T08:44:42.860Z pushedAt=2026-05-29T06:25:44.192Z -->

## Overview

`get_rope` provides a flexible registration mechanism for creating and managing different types of Rotary Position Embedding (RoPE) instances. Through this registration mechanism, model-specific RoPE implementations can be placed in their respective model files instead of being centralized in a factory class.

## Core Features

1. **Registration Mechanism**: Register custom RoPE types using the `@register_rope_type` decorator.

2. **Automatic Caching**: Automatically cache RoPE instances with the same configuration to avoid redundant creation.

3. **Model-Specific Support**: Register model-specific extrapolation methods (such as DeepseekV3YarnRotaryEmbedding) in the model file.

## Usage

### 1. Using the Default or Already Registered RoPE

```python
from mindie_llm.runtime.layers.embedding.rotary_embedding import get_rope

self.rope_emb = get_rope(
            self.head_dim,
            self.head_dim,
            self.config.rope_scaling.max_position_embeddings,
            is_neox_style=True,
            rope_config=config.rope_scaling,
        )

 ...
 # Usage
 # Set up cos_sin_indexed_cache based on positions.
self.layers[0].self_attn.rope_emb.set_cos_sin_indexed_cache(positions)
...
 # 1. Apply RoPE to query and key directly via forward.
query, key = self.rope_emb(positions, query, key)
...
# 2. Retrieve cos and sin for use in attention backend.
return self.attn(hidden_states,
                        cos=self.rope_emb.cos_indexed_cache,
                        sin=self.rope_emb.sin_indexed_cache)
```

### 2. Implementing a Model-Specific RoPE Type (Using DeepseekV3 as an Example)

#### 2.1 Rope Module Implementation

Define your RoPE implementation in the model directory (for example, `mindie_llm/runtime/layers/embedding/rotary_embedding/deepseek_v3_yarn_scaling_rope.py`):
> You can choose to inherit from `RotaryEmbedding` under `mindie_llm/runtime/layers/embedding/rotary_embedding/base.py` or inherit from `YarnScalingRotaryEmbedding` under `mindie_llm/runtime/layers/embedding/rotary_embedding/yarn_scaling_rope.py` for extrapolation

```python
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

    def set_cos_sin_indexed_cache(self, positions) -> None:
        """Create position-indexed cosine/sine caches with dimension doubling.

        Extracts position-specific rotary values from precomputed caches and
        duplicates them across the last dimension to match attention head layout.

        Args:
            positions: 1D tensor of position indices to index into the cache.
        """
        cos_indexed_cache = torch.index_select(self.cos_cache, dim=0, index=positions.view(-1)).unsqueeze(1).unsqueeze(1)
        sin_indexed_cache = torch.index_select(self.sin_cache, dim=0, index=positions.view(-1)).unsqueeze(1).unsqueeze(1)
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

```

#### 2.2 Implementation and Registration of a Custom RoPE Constructor

The registration function must use the decorator `@register_rope_type("xxxx")`
`@cached_rope_factory`:

```python
@register_rope_type("deepseek_yarn")
@cached_rope_factory
def _create_deepseek_scaling_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    rope_config: RopeScaling,
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
    ds_yarn_extra_keys = (
        "factor",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim"
        )
    extra_kwargs = {
        k: getattr(rope_config, k)
        for k in ds_yarn_extra_keys
    }
    return DeepseekV3YarnRotaryEmbedding(
        rotary_dim,
        rope_config.original_max_position_embeddings,
        base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        **extra_kwargs,
    )
```

## Caching Mechanism

`get_rope` automatically caches RoPE instances with the same configuration. The cache key is based on:

- `head_size`

- `rotary_dim` (calculated as `head_size * partial_rotary_factor`)

- `max_position`

- `is_neox_style`

- `base`

- `rope_config` (Lists are converted to tuples to ensure stability.)

- `dtype`

This means multiple calls with the same configuration will return the same instance, saving memory and computational resources.

## Registered Types

- `default`: Standard RotaryEmbedding (default)

- `yarn`: YarnScalingRotaryEmbedding

- `deepseek_yarn`: DeepseekV3YarnRotaryEmbedding

## Precautions

1. **Model-specific RoPE implementations should be placed in separate files within the `mindie_llm/runtime/layers/embedding/rotary_embedding` directory**: For example, `DeepseekV3YarnRotaryEmbedding` should be registered in a file under the `mindie_llm/runtime/layers/embedding/rotary_embedding` directory, not in `rotary_embedding/__init__.py`.

2. **Registration timing**: Ensure registration is completed before using `get_rope`. This typically occurs during module import.

3. **Parameter extraction**: The registration function should retrieve required parameters from `rope_config`, rather than passing all parameters as positional arguments.

4. **Backward compatibility**: Existing code continues to work without modification. If a new model requires a custom RoPE implementation, implement and register a separate RoPE module as an incremental addition—do not modify existing code. Any changes to existing code must be tested against relevant scenarios.
