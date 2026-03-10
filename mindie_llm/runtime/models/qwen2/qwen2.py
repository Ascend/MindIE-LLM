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
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig
from mindie_llm.runtime.layers.normalization import RMSNorm
from mindie_llm.runtime.layers.linear.linear import RowParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead

from mindie_llm.runtime.layers.attention.attention_layer import Attention

from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.models.base.model import BaseModelForCausalLM
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.layers.embedding.rotary_embedding import get_rope


class Qwen2Attention(nn.Module):
    """
    Qwen2 attention module that handles multi-head attention with rotary position embeddings.

    This class implements the attention mechanism for Qwen2 model, including:
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
        q_norm: RMS normalization for queries
        k_norm: RMS normalization for keys 
        attn: Attention computation module
    """

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str,
            quant_config: QuantizationConfigBase = None,
    ):
        """
        Initialize the Qwen2 attention module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config
        self.head_dim = config.head_dim
        self.num_heads_per_rank = config.get_num_attention_heads_per_rank()
        self.num_key_value_heads_per_rank = config.get_num_kv_heads_per_rank()
        
        self.q_size = self.num_heads_per_rank * self.head_dim
        self.kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.scale = self.head_dim**-0.5
        attn_tp = get_parallel_info_manager().get(ParallelType.ATTN_TP)
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            parallel_info=attn_tp
        )

        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            parallel_info=attn_tp
        )

        self.rope_emb = get_rope(
            self.head_dim,
            self.head_dim,
            self.config.rope_scaling.max_position_embeddings,
            is_neox_style=True,
            rope_config=config.rope_scaling,
        )

        self.attn = Attention(
            head_size=self.head_dim,
            num_heads=self.num_heads_per_rank,
            scale=self.scale,
            num_kv_heads=self.num_key_value_heads_per_rank,
            num_kv_heads_replicas=attn_tp.group_size // config.num_key_value_heads,
            weight_dtype=config.torch_dtype,
            quant_config=quant_config,
            prefix=self.prefix,
        )

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
        query, key = self.rope_emb(positions, query, key)
        attn_output = self.attn(query, key, value)

        output = self.o_proj(attn_output)
        return output


class Qwen2Mlp(nn.Module):
    """
    Qwen2 MLP (feed-forward) module.

    This class implements the MLP component of the Qwen2 transformer block, including:
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
        Initialize the Qwen2 MLP module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config
        mlp_tp = get_parallel_info_manager().get(ParallelType.MLP_TP)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            parallel_info=mlp_tp
        )

        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            parallel_info=mlp_tp
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x: Input hidden states

        Returns:
            torch.Tensor: Output hidden states after MLP
        """
        gateup = self.gate_up_proj(x)
        x = torch_npu.npu_swiglu(gateup)
        x = self.down_proj(x)
        return x


class Qwen2Layer(nn.Module):
    attn_cls = Qwen2Attention
    mlp_cls = Qwen2Mlp
    """
    Qwen2 transformer layer.

    This class implements a single transformer layer of the Qwen2 model, including:
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
    attn_cls = Qwen2Attention
    mlp_cls = Qwen2Mlp

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str,
            layer_idx: int,
            quant_config: QuantizationConfigBase = None,
    ) -> None:
        """
        Initialize the Qwen2 transformer layer.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            layer_idx: Index of this layer
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        
        self.config = config
        self.prefix = f"{prefix}.layers.{layer_idx}"
        self.layer_idx = layer_idx
        self.quant_config = quant_config

        self.self_attn_prefix = f"{self.prefix}.self_attn"
        self.self_attn = self.attn_cls(config, self.self_attn_prefix, quant_config=quant_config)
        self.mlp = self.mlp_cls(config, f"{self.prefix}.mlp", quant_config=quant_config)

        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            quant_config=quant_config, prefix=f"{self.prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            quant_config=quant_config, prefix=f"{self.prefix}.post_attention_layernorm")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer layer.

        Args:
            positions: Position indices for rotary embeddings
            hidden_states: Input hidden states
            residual: Residual connection from previous layer

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - Output hidden states
                - Updated residual for next layer
        """
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    layer_cls = Qwen2Layer
    """
    Qwen2 base model.

    This class implements the complete transformer model for Qwen2, including:
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
    layer_cls = Qwen2Layer

    def __init__(
            self,
            config: HuggingFaceConfig,
            prefix: str = "model",
            quant_config: QuantizationConfigBase = None
    ) -> None:
        """
        Initialize the Qwen2 base model.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        
        self.config = config
        self.prefix = prefix
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{prefix}.embed_tokens",
            partition_weights=True,
        )

        self.layers = nn.ModuleList(
            [
                self.layer_cls(config, self.prefix, layer_idx, quant_config=self.quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, quant_config=quant_config, prefix=f"{prefix}.norm")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the base model.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Intermediate tensors
            inputs_embeds: Embedding vectors for input tokens

        Returns:
            torch.Tensor: Final hidden states
        """
        residual = None
        hidden_states = self.embed_tokens(input_ids)
        self.layers[0].self_attn.rope_emb.set_cos_sin_indexed_cache(positions)
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen2ForCausalLM(BaseModelForCausalLM):
    """
    Qwen2 model for causal language modeling.

    This class extends the base Qwen2 model with a language modeling head for:
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
        model: Base Qwen2 model
        lm_head: Language modeling head
    """
    model_cls = Qwen2Model

    def __init__(self, mindie_llm_config: MindIELLMConfig):
        """
        Initialize the Qwen2 causal language model.

        Args:
            mindie_llm_config: MindIE LLM configuration object containing:
                - hf_config: Hugging Face configuration
                - quant_config: Quantization configuration
        """
        super().__init__(mindie_llm_config)

        self.hf_config = mindie_llm_config.hf_config
        self.quant_config = mindie_llm_config.quant_config
        self.parallel_info_manager = get_parallel_info_manager()
        self.model = self.model_cls(
            config=mindie_llm_config.hf_config,
            prefix="model",
            quant_config=self.quant_config
        )

        self.lm_head = ParallelLMHead(
            self.hf_config.vocab_size,
            self.hf_config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"lm_head",
        )

        if self.hf_config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None
    ):
        """
        Forward pass of the model (without LM head).

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Intermediate tensors
            inputs_embeds: Embedding vectors for input tokens

        Returns:
            torch.Tensor: Hidden states before LM head
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Forward pass through the LM head.

        Args:
            hidden_states: Hidden states from the base model

        Returns:
            torch.Tensor: Logits for token prediction
        """
        forward_context = get_forward_context()
        lm_head_indices = forward_context.lm_head_indices
        return self.lm_head.forward(hidden_states, lm_head_indices)
