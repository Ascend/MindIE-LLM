# get_rope 使用指南

## 概述

`get_rope` 提供了一个灵活的注册机制来创建和管理不同类型的 Rotary Position Embedding (RoPE) 实例。通过注册机制，模型特定的 RoPE 实现可以放在各自的模型文件中，而不是集中在工厂类中。

## 核心特性

1. **注册机制**：通过 `@register_rope_type` 装饰器注册自定义 RoPE 类型
2. **自动缓存**：相同配置的 RoPE 实例会被自动缓存，避免重复创建
3. **模型特定支持**：模型特定的外推方式（如 DeepseekV3YarnRotaryEmbedding）可以在模型文件中注册

## 使用方式

### 1. 使用默认或者已经注册 RoPE

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
 # 使用方式
 # 根据postions设置cos_sin_indexed_cache
self.layers[0].self_attn.rope_emb.set_cos_sin_indexed_cache(positions)
...
 # 1. 调用forward直接对query,key进行rope变换
query, key = self.rope_emb(positions, query, key) 
...
# 2. 直接拿出cos, sin交给attention 后端使用
return self.attn(hidden_states,
                        cos=self.rope_emb.cos_indexed_cache,
                        sin=self.rope_emb.sin_indexed_cache)
```

### 2. 实现模型特定的 RoPE 类型 （以deepseekv3为例）

#### 2.1 rope模块实现

在模型目录下定义自己的rope实现（例如 `mindie_llm/runtime/layers/embedding/rotary_embedding/deepseek_v3_yarn_scaling_rope.py`）：
> 可以选择继承mindie_llm/runtime/layers/embedding/rotary_embedding/base.py下的RotaryEmbedding
> 或者继承mindie_llm/runtime/layers/embedding/rotary_embedding/yarn_scaling_rope.py 下的YarnScalingRotaryEmbedding用于外推

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

```

#### 2.2  实现自定义的rope构造函数并注册

注册函数必须使用装饰器@register_rope_type("xxxx")
@cached_rope_factory：

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

## 缓存机制

`get_rope` 会自动缓存相同配置的 RoPE 实例。缓存键基于：

- `head_size`
- `rotary_dim` (由 `head_size * partial_rotary_factor` 计算)
- `max_position`
- `is_neox_style`
- `base`
- `rope_config` (列表会被转换为元组以确保稳定性)
- `dtype`

这意味着相同配置的多次调用会返回同一个实例，节省内存和计算资源。

## 已注册的类型

- `default`: 标准 RotaryEmbedding（默认）
- `yarn`: YarnScalingRotaryEmbedding
- `deepseek_yarn`: DeepseekV3YarnRotaryEmbedding

## 注意事项

1. **模型特定rope实现应在mindie_llm/runtime/layers/embedding/rotary_embedding目录下单独文件**：如 `DeepseekV3YarnRotaryEmbedding` 应该在 `mindie_llm/runtime/layers/embedding/rotary_embedding` 目录下的文件中注册，而不是在 `rotary_embedding/__init__.py` 中。

2. **注册时机**：确保在使用 `get_rope` 之前完成注册。通常这发生在模块导入时。

3. **参数提取**：注册函数应该从 `rope_config` 中提取所需的参数，而不是期望所有参数都通过位置参数传递。

4. **向后兼容**：现有的代码无需修改即可继续工作，如果新增模型需要新的rope，请单独实现自己的rope模块并注册使用，做增量修改，不要修改原有代码，否则须测试相关场景。
