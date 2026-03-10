# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any

import torch
import torch_npu
from torch import nn

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig
from mindie_llm.runtime.layers.normalization import RMSNorm
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig

from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.models.qwen2.qwen2 import (
    Qwen2Attention, Qwen2Mlp, Qwen2Layer, Qwen2Model, Qwen2ForCausalLM
)


class Qwen3Attention(Qwen2Attention):
    """
    Qwen3 attention module that handles multi-head attention with rotary position embeddings.

    This class implements the attention mechanism for Qwen3 model, including:
    - QKV projection with parallel processing
    - Rotary position embedding application
    - Attention computation
    - Output projection

    Key features:
    - Support for grouped-query attention (GQA)
    - Optional QK normalization
    - NPU-optimized rotary position embedding
    - Parallel processing for distributed training

    Attributes:
        config: Model configuration object containing hyperparameters
        prefix: Prefix for parameter naming in the model
        quant_config: Quantization configuration (if applicable)
        head_dim: Dimension of each attention head
        num_heads_per_rank: Number of attention heads per device/rank
        num_key_value_heads_per_rank: Number of key/value heads per device/rank
        q_size: Total dimension for query projections
        kv_size: Total dimension for key/value projections
        scale: Scaling factor for attention scores (1/sqrt(head_dim))
        qkv_proj: Parallel linear layer for QKV projection
        o_proj: Parallel linear layer for output projection
        q_norm: RMS normalization for queries (if use_qk_norm is True)
        k_norm: RMS normalization for keys (if use_qk_norm is True)
        attn: Attention computation module
    """

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str,
            quant_config: QuantizationConfigBase = None,
    ):
        """
        Initialize the Qwen3 attention module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__(config, prefix, quant_config)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.q_norm")
            self.k_norm = RMSNorm(
                self.head_dim, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.k_norm")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            positions: Position indices for rotary embeddings
            hidden_states: Input hidden states

        Returns:
            torch.Tensor: Output hidden states after attention
        """
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.config.use_qk_norm:
            q_by_head = query.view(*query.shape[:-1], query.shape[-1] // self.head_dim, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            query = q_by_head.view(query.shape)

            k_by_head = key.view(*key.shape[:-1], key.shape[-1] // self.head_dim, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            key = k_by_head.view(key.shape)
        query, key = self.rope_emb(positions, query, key)
        attn_output = self.attn(query, key, value)

        output = self.o_proj(attn_output)
        return output


class Qwen3Mlp(Qwen2Mlp):
    """
    Qwen3 MLP (feed-forward) module.

    This class implements the MLP component of the Qwen3 transformer block, including:
    - Gate and up projections
    - SwiGLU activation
    - Down projection

    Key features:
    - Merged column parallel linear layers for gate/up projections
    - Row parallel linear layer for down projection
    - NPU-optimized SwiGLU activation

    Attributes:
        config: Model configuration object
        prefix: Prefix for parameter naming
        quant_config: Quantization configuration (if applicable)
        gate_up_proj: Merged parallel linear layer for gate and up projections
        down_proj: Parallel linear layer for down projection
    """

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str,
            quant_config: QuantizationConfigBase = None,
    ):
        """
        Initialize the Qwen3 MLP module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__(config, prefix, quant_config)
 

class Qwen3Layer(Qwen2Layer):
    """
    Qwen3 transformer layer.

    This class implements a single transformer layer of the Qwen3 model, including:
    - Input layer normalization
    - Self-attention module
    - Post-attention layer normalization
    - MLP module
    - Residual connections

    Key features:
    - Residual stream with optional pre-normalization
    - RMS normalization
    - Parallel processing support

    Attributes:
        config: Model configuration object
        prefix: Prefix for parameter naming
        layer_idx: Index of this layer in the model
        quant_config: Quantization configuration (if applicable)
        self_attn_prefix: Prefix for self-attention parameters
        self_attn: Self-attention module
        mlp: MLP module
        input_layernorm: Input layer normalization
        post_attention_layernorm: Post-attention layer normalization
    """
    attn_cls = Qwen3Attention
    mlp_cls = Qwen3Mlp

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str,
            layer_idx: int,
            quant_config: QuantizationConfigBase = None,
    ) -> None:
        """
        Initialize the Qwen3 transformer layer.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            layer_idx: Index of this layer
            quant_config: Quantization configuration (optional)
        """
        super().__init__(config, prefix, layer_idx, quant_config)

        
class Qwen3Model(Qwen2Model):
    """
    Qwen3 base model.

    This class implements the complete transformer model for Qwen3, including:
    - Token embeddings
    - Multiple transformer layers
    - Final layer normalization

    Key features:
    - Parallel embedding for large vocabularies
    - Configurable number of transformer layers
    - RMS normalization
    - Support for model parallelism

    Attributes:
        config: Model configuration object
        prefix: Prefix for parameter naming
        quant_config: Quantization configuration (if applicable)
        embed_tokens: Token embedding layer
        layers: List of transformer layers
        norm: Final layer normalization
    """
    layer_cls = Qwen3Layer

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str = "model",
            quant_config: QuantizationConfigBase = None
    ) -> None:
        """
        Initialize the Qwen3 base model.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__(config, prefix, quant_config)


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    model_cls = Qwen3Model
    """
    Qwen3 model for causal language modeling.

    This class extends the base Qwen3 model with a language modeling head for:
    - Text generation
    - Next token prediction
    - Causal language modeling tasks

    Key features:
    - LM head for token prediction
    - Support for tied or untied word embeddings
    - Integration with model runner for inference

    Attributes:
        hf_config: Hugging Face configuration object
        quant_config: Quantization configuration
        parallel_info_manager: Manager for parallelism information
        model_status: Model status information
        model: Base Qwen3 model
        lm_head: Language modeling head
    """
    model_cls = Qwen3Model

    def __init__(self, mindie_llm_config: MindIELLMConfig):
        """
        Initialize the Qwen3 causal language model.

        Args:
            mindie_llm_config: MindIE LLM configuration object containing:
                - hf_config: Hugging Face configuration
                - quant_config: Quantization configuration
        """
        super().__init__(mindie_llm_config)
