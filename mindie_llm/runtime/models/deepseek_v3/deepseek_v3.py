# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Any, NamedTuple, Optional, Tuple, Dict
import math
import torch
import torch_npu
import torch.distributed as dist
from torch import nn
from mindie_llm.runtime.models.base.model import BaseModelForCausalLM
from mindie_llm.runtime.layers.normalization import RMSNorm, LayerNorm
from mindie_llm.runtime.layers.linear.linear import RowParallelLinear, MergedColumnParallelLinear, \
    ColumnParallelLinear, ReplicatedLinear
from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead
from mindie_llm.runtime.layers.attention.sparse_attention_layer import SFA
from mindie_llm.runtime.layers.fused_moe.experts_selector import select_experts
from mindie_llm.runtime.layers.fused_moe.fused_moe import FusedMoE, assign_experts
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.layers.attention import get_global_attn_dict
from mindie_llm.runtime.layers.parameter import BaseParameter


# (Note): rewrite to a public function
def bind_cos_sin_table(cos_sin) -> None:
    attns = get_global_attn_dict()
    for _, prefix in enumerate(attns):
        attn_layer = attns[prefix]
        attn_layer.cos_sin = cos_sin


def init_rope(self) -> None:
    if self.config.rope_scaling_dict is None:
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.config.qk_rope_head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling_dict["type"]
        scaling_factor = self.config.rope_scaling_dict["factor"]
        if scaling_type == "yarn":
            kwargs = {}
            important_keys = [
                "original_max_position_embeddings",
                "beta_fast", 
                "beta_slow",
                "mscale",
                "mscale_all_dim"
            ]
            for key in important_keys:
                if key in self.config.rope_scaling_dict:
                    kwargs[key] = self.config.rope_scaling_dict[key]
            self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                self.config.qk_rope_head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.config.rope_theta,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


def _yarn_get_mscale(scale=1, mscale=1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
) -> float:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max:
        max_val += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3Moe(nn.Module):
    """
    DeepSeek V3 Mixture of Experts (MoE) module.
    
    This class represents  in the DeepseekV3 model. It includes:
    - Experts with fused kernel FusedMoE
    - Gating Network
    - Shared experts
    """
    def __init__(self, config, prefix, quant_config) -> None:
        """
        Initialize the DeepseekV3Moe module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.prefix = prefix
        self.config = config
        parallel_info = get_parallel_info_manager()
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.topk_num = config.num_experts_per_tok
        self.expert_num = config.n_routed_experts
        self.expert_list = assign_experts(config.n_routed_experts,
            parallel_info.moe_ep.group_size)[parallel_info.moe_ep.rank]

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            topk_num=self.topk_num,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            suffix=["gate_proj", "down_proj", "up_proj"])

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts = DeepseekV3MLP(
            config,
            f"{self.prefix}.shared_experts",
            quant_config=quant_config,
            intermediate_size=config.moe_intermediate_size,
            is_moe=True
        )
        self.gate.e_score_correction_bias = BaseParameter(torch.empty(config.n_routed_experts))
        self.gate.e_score_correction_bias.add_attrs({"weight_loader": self.weight_loader})

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        """
        Load weight into a parameter.

        Args:
            param (BaseParameter): Target parameter to load the weight into.
            loaded_weight (torch.Tensor): Weight to be loaded.
        """
        param.load_weight(loaded_weight)

    def forward(self, hidden_states) -> torch.Tensor:
        """
        Forward pass of the DeepseekV3Moe module.

        Args:
            hidden_states (torch.Tensor): Input hidden states

        Returns:
            torch.Tensor: Output combining the results of the shared expert and the selected experts.
        """
        # shared expert 
        shared_expert_out = self.shared_experts(hidden_states)
        # moe gate
        router_logits = self.gate(hidden_states)
        # expert selector
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.topk_num,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
            scoring_func="",
            routed_scaling_factor=1.0,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            global_num_experts=self.expert_num
        )

        final_hidden_states = self.experts(hidden_states, topk_weights, topk_ids)
        return final_hidden_states + shared_expert_out


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def forward(self, x, position_ids, kv_len, max_seq_len=None):
        if max_seq_len is None:
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)
        cos = torch.index_select(self.cos_cached, dim=0, index=position_ids.view(-1)).unsqueeze(1).unsqueeze(1)
        sin = torch.index_select(self.sin_cached, dim=0, index=position_ids.view(-1)).unsqueeze(1).unsqueeze(1)        
        return (
            cos.to(dtype=x.dtype),
            sin.to(dtype=x.dtype),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            _yarn_get_mscale(self.scaling_factor, self.mscale)
            / _yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


class Indexer(nn.Module):
    """
    Indexer module for Deepseek V3.2
    """
    def __init__(self, config, prefix) -> None:
        """
        Initialize the Indexer module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
        """
        super().__init__()
        self.dim: int = config.hidden_size  # 7168
        self.n_heads: int = config.index_n_heads  # 64
        self.head_dim: int = config.index_head_dim  # 128
        self.rope_head_dim: int = config.qk_rope_head_dim  # 64
        self.q_lora_rank: int = config.q_lora_rank  # 1536
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wq_b"
        )
        self.wk = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wk"
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None, 
            prefix=f"{prefix}.weights_proj"
        )
        self.k_norm = LayerNorm(
            self.head_dim,
            config.rms_norm_eps,
            quant_config=None,
            prefix=f"{prefix}.k_norm"
        )


class DeepseekV3Attention(nn.Module):
    """
    Multi-Head Latent Attention module for Deepseek V3
    """
    def __init__(self,
                config,
                prefix: str,
                quant_config,
                enable_mlapo,
                input_layernorm: RMSNorm = None,) -> None:
        """
        Initialize the Deepseek V3 MLA module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
            enable_mlapo: Enable the fused kernel npu_mla_process
            input_layernorm: Weight process for the fused kernel npu_mla_process
        """
        super().__init__()
        parallel_info = get_parallel_info_manager()
        self.config = config
        self.attn_tp_size = parallel_info.attn_tp.group_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.num_key_value_heads_per_rank = 1
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 64
        self.scale = self.qk_head_dim ** -0.5

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj"
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_layernorm"
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
            parallel_info=parallel_info.attn_tp
        )
        self.q_proj = (self.q_a_proj, self.q_a_layernorm, self.q_b_proj)

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,                   
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa"
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_layernorm"
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.v_head_dim + self.qk_nope_head_dim),                  
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
            parallel_info=parallel_info.attn_tp
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.qk_nope_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            parallel_info=parallel_info.attn_tp,
            reduce_results=True
        )

        self.softmax_scale = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling_dict is not None:
            mscale_all_dim = self.config.rope_scaling_dict.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling_dict["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        ## v32
        self.indexer = Indexer(
            self.config,
            prefix=f"{prefix}.indexer"
        )

        self.attn = SFA(
            head_size=self.qk_nope_head_dim,
            num_heads=self.num_heads,
            scale=self.scale,
            prefix=prefix,
            num_kv_heads=self.num_key_value_heads_per_rank,
            #SFA Para
            softmax_scale=self.softmax_scale,
            num_heads_per_rank=self.num_heads_per_rank,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            q_proj=self.q_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=self.indexer,
            #SFA switch
            enable_mlapo=enable_mlapo,
            input_layernorm=input_layernorm,
        )

    def forward(self, hidden_states) -> torch.Tensor:
        """
        Forward pass through the Deepseek V3 attention module.

        Args:
            hidden_states (torch.Tensor): Input hidden states

        Returns:
            torch.Tensor: Output hidden states after attention
        """
        return self.attn(hidden_states)


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, prefix: str, quant_config, intermediate_size, is_moe=False) -> None:
        """
        Initialize the Deepseek MLP module.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
            intermediate_size: Size of the intermediate layer in the MLP.
        """
        super().__init__()
        parallel_info = get_parallel_info_manager()
        self.config = config
        self.prefix = prefix
        cur_parallel_info = parallel_info.mlp_tp if not is_moe else parallel_info.moe_tp        
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            parallel_info=cur_parallel_info
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            parallel_info=cur_parallel_info,
            reduce_results=True
        )
    
    def forward(self, x) -> torch.Tensor:
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


class DeepseekV3Layer(nn.Module):
    """
    DeepseekV3 Layer Module

    This class represents a single deepseek layer. It includes:
    - Normalization before attention
    - Self-attention
    - Normalization after attention
    - Mixture-of-Experts (MoE) or standard MLP
    """
    def __init__(self, config, prefix: str, layer_idx: int, quant_config) -> None:
        """
        Initialize the MTP layer.

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
        # (NOTE): the fused kernel switch can be removed in the future
        self.enable_mlapo = True
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.prefix}.input_layernorm"
        )
        self.self_attn = DeepseekV3Attention(
            config,
            f"{self.prefix}.self_attn",
            quant_config,
            enable_mlapo=self.enable_mlapo,
            input_layernorm=self.input_layernorm if self.enable_mlapo else None
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.prefix}.post_attention_layernorm"
        )
        self.is_moe = config.n_routed_experts is not None and \
                layer_idx >= config.first_k_dense_replace and \
                (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0

        if not self.is_moe:
            self.mlp = DeepseekV3MLP(
                config,
                f"{self.prefix}.mlp",
                quant_config,
                config.intermediate_size
            )
        else:
            self.mlp = DeepseekV3Moe(
                config,
                f"{self.prefix}.mlp",
                quant_config
            )

    def forward(self, hidden_states, past_residual) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the deepseek layer.

        Args:
            positions: Position indices
            hidden_states: Input hidden states
            residual: Residual connection from previous layer

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - Output hidden states [seq_len, hidden_size]
                - Updated residual for next layer
        """
        forward_context = get_forward_context()
        is_prefill = forward_context.is_prefill

        if self.enable_mlapo and not is_prefill:
            if past_residual is not None:
                hidden_states = hidden_states + past_residual
            residual = hidden_states
        else:
            if past_residual is not None:
                hidden_states, residual = self.input_layernorm(hidden_states, past_residual)
            else:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        outputs = (residual, hidden_states)
        return outputs


class DeepseekV3Model(nn.Module):
    """
    Deepseek3 base model.

    This class implements the complete model for Deepseekv3, including:
    - Token embeddings
    - Multiple layers
    - Final layer normalization
    """
    def __init__(self, config, prefix, quant_config) -> None:
        """
        Initialize the Deepseekv3 model.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV3Layer(config, self.prefix, layer_idx, quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.prefix}.norm"
        )
        init_rope(self)

    def forward(self, input_ids, positions) -> torch.Tensor:
        """
        Forward pass of the model for multi-token prediction.

        Args:
            input_ids (torch.Tensor): Input token IDs tensor
            positions (torch.Tensor): Position indices

        Returns:
            torch.Tensor: Final hidden states after processing
        """
        hidden_states = self.embed_tokens(input_ids)

        forward_context = get_forward_context()    
        cos_sin = self.rotary_emb(
            hidden_states, 
            positions, forward_context.attn_metadata.seq_lens, 
            self.config.max_position_embeddings
        )
        bind_cos_sin_table(cos_sin)

        residual = None
        for layer in self.layers:
            residual, hidden_states = layer(hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class DeepseekV3ForCausalLM(BaseModelForCausalLM):
    """
    Deepseek V3 Model for causal language modeling.
    """
    def __init__(self, mindie_llm_config) -> None:
        """
        Initializes the DeepseekV3ForCausalLM model.

        Args:
            mindie_llm_config: MindIE LLM configuration object containing:
                - hf_config: Hugging Face configuration
                - quant_config: Quantization configuration
        """
        super().__init__(mindie_llm_config)
        self.config = mindie_llm_config.hf_config
        self.quant_config = mindie_llm_config.quant_config
        self.parallel_info = get_parallel_info_manager()
        self.ds_config = mindie_llm_config.llm_config.llm
        self.model = DeepseekV3Model(
            config=mindie_llm_config.hf_config,
            prefix="model",
            quant_config=self.quant_config
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"lm_head",
        )
        self.softmax_scale = (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim) ** (-0.5)
        self.kv_lora_rank = self.config.kv_lora_rank
        self.qk_rope_head_dim = self.config.qk_rope_head_dim

    def forward(self, input_ids, positions) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs tensor
            positions (torch.Tensor): Position indices

        Returns:
            torch.Tensor: The output tensor from the model before the LM head.
        """
        return self.model(input_ids, positions)
    
    def compute_logits(self, hidden_states) -> torch.Tensor:
        """
        Computes the final logits using the language model head (LM head).

        Args:
            hidden_states (torch.Tensor): Hidden states from the main model.

        Returns:
            torch.Tensor: Logits for token prediction.
        """
        forward_context = get_forward_context()
        lm_head_indices = forward_context.lm_head_indices
        return self.lm_head.forward(hidden_states, lm_head_indices)