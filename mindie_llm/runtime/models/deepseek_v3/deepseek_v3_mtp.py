# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
import torch.distributed as dist
from torch import nn
from mindie_llm.runtime.models.base.model import BaseModelForCausalLM
from mindie_llm.runtime.layers.normalization import RMSNorm
from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear
from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead, maybe_slice_cross_tp
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from .deepseek_v3 import DeepseekV3Layer


class SharedHead(nn.Module):
    """
    A shared head module used deepseek mtp model
    This module includes a normalization layer (`RMSNorm`) with a parallel language model head (`ParallelLMHead`).
    """
    def __init__(
        self,
        config,
        prefix,
        quant_config,
    ) -> None:
        """
        Initializes the SharedHead module

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.prefix = prefix
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.prefix}.norm"
        )
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{self.prefix}.head",
        )

    def forward(self, hidden_states, residual=None) -> torch.Tensor:
        """
        Forward pass the normalization layer

        Args:
            hidden_states (Tensor): Input hidden states [seq_len, hidden_size]
            residual (Tensor, optional): An optional residual tensor to be added before normalization.

        Returns:
            Tensor: The output tensor after normalization.
        """
        return self.norm(hidden_states, residual)


class DeepseekV3MtpLayer(DeepseekV3Layer):
    """
    Deepseek Multi-Token Prediction Layer, extending the base DeepseekV3Layer with 
    additional normalization, projection, embed and sharedhead modules
    """
    def __init__(self, config, prefix: str, layer_idx: int, quant_config) -> None:
        """
        Initialize the MTP layer.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            layer_idx: fix 61 for current deepseekv3.2 model
            quant_config: Quantization configuration (optional)
        """
        super().__init__(config, prefix, layer_idx, quant_config)
        self.mtp_prefix = f"{prefix}.layers.{self.layer_idx}"
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=f"{self.prefix}.embed_tokens",
        )
        self.enorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.mtp_prefix}.enorm"
        )
        self.hnorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{self.mtp_prefix}.hnorm"
        )
        self.shared_head = SharedHead(config, prefix=f"{self.mtp_prefix}.shared_head", quant_config=quant_config)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size, 
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{self.mtp_prefix}.eh_proj"
        )


class DeepseekV3MtpModel(nn.Module):
    """
    Deepseek Multi-Token Prediction Model
    """
    def __init__(self, config, prefix, quant_config) -> None:
        """
        Initialize the Deepseek MTP model.

        Args:
            config: Model configuration object
            prefix: Parameter naming prefix
            quant_config: Quantization configuration (optional)
        """
        super().__init__()
        self.config = config       
        self.parallel_info = get_parallel_info_manager()
        self.layer_idx = 61 # current ds model has fixed 61 layers
        self.layers = nn.ModuleDict(
            {
                str(self.layer_idx):
                    DeepseekV3MtpLayer(config, prefix, self.layer_idx, quant_config)
            }
        )

    def forward(self, input_ids, positions) -> torch.Tensor:
        """
        Forward pass of the model for multi-token prediction.

        Args:
            input_ids (torch.Tensor): Input token IDs tensor
            positions (torch.Tensor): Position indices

        Returns:
            torch.Tensor: Final hidden states after processing
        """
        mtp_layer = self.layers[str(self.layer_idx)]
        forward_context = get_forward_context()
        last_hidden_states = forward_context.mtp_metadata.last_hidden_states

        last_hidden_states = maybe_slice_cross_tp(last_hidden_states, self.parallel_info.attn_tp)

        hidden_states = mtp_layer.embed_tokens(input_ids)
        mtp_layer.self_attn.rope_emb.set_cos_sin_indexed_cache(positions)
        hidden_states = mtp_layer.enorm(hidden_states)
        last_hidden_states = mtp_layer.hnorm(last_hidden_states)
        hidden_states = torch.concat([hidden_states, last_hidden_states], dim=-1)
        hidden_states = mtp_layer.eh_proj(hidden_states)

        residual = None
        residual, hidden_states = mtp_layer(hidden_states, residual)
        hidden_states, _ = mtp_layer.shared_head(hidden_states, residual)
        return hidden_states


class DeepseekV3MTP(BaseModelForCausalLM):
    """
    A Deepseek V3 MTP causal language model class, extending BaseModelForCausalLM. 
    This module includs the draft model and the language model head (LM head), 
    """
    def __init__(self, mindie_llm_config) -> None:
        """
        Initializes the DeepseekV3MTP model with the provided configuration.

        Args:
            mindie_llm_config: MindIE LLM configuration object containing:
                - hf_config: Hugging Face configuration
                - quant_config: Quantization configuration
        """
        super().__init__(mindie_llm_config)
        self.config = mindie_llm_config.hf_config
        self.quant_config = mindie_llm_config.quant_config
        self.model = DeepseekV3MtpModel(
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
            hidden_states (torch.Tensor): Hidden states from the draft model.

        Returns:
            torch.Tensor: Logits for token prediction.
        """
        forward_context = get_forward_context()
        lm_head_indices = forward_context.lm_head_indices
        return self.lm_head.forward(hidden_states, lm_head_indices)