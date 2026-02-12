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
import torch.distributed as dist
import torch.nn.functional as F

from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.quantization.unquantized import UnquantizedEmbeddingMethod, UnquantizedLinearMethod
from mindie_llm.runtime.layers.parameter import BaseParameter, ColumnParameter
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.utils import even_divide
from mindie_llm.runtime.model_runner.forward_context import get_forward_context


class VocabParallelEmbedding(CustomLayer):
    """Vocabulary parallel embedding layer.

    This layer implements embedding with support for vocabulary parallelism,
    where the embedding table can be partitioned across the hidden size dimension.

    Args:
        num_embeddings: Size of the vocabulary.
        embedding_dim: Dimension of the embedding vectors.
        params_dtype: Data type for the embedding parameters. If None,
            uses default dtype. Defaults to None.
        quant_config: Quantization configuration for the layer. If None,
            uses unquantized method. Defaults to None.
        prefix: Prefix string used to construct the full layer name in the state dictionary. Defaults to "".
        partition_weights: Whether to partition the embedding weights.
            Defaults to False.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str = "",
        partition_weights: bool | None = False,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prefix = prefix

        self.num_embeddings_per_partition = self.embedding_dim
        self.params_dtype = params_dtype or torch.get_default_dtype()

        self.parallel_info = get_parallel_info_manager().word_embed_tp

        self.tp_rank = self.parallel_info.rank
        self.tp_size = self.parallel_info.group_size

        self.is_parallel = partition_weights and self.tp_size > 1

        if self.is_parallel:
            self.output_partition_size = even_divide(self.embedding_dim, self.tp_size)
        else:
            self.output_partition_size = self.embedding_dim

        self.quant_config = quant_config
        if self.quant_config is None:
            self.quant_method: QuantizationMethodBase | None = UnquantizedEmbeddingMethod()
        else:
            self.quant_method = self.quant_config.get_quant_method(self, prefix=self.prefix)

        self._post_init()
        self._create_weights()

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        """Load weight into a parameter with parallel support.

        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor read from file to be loaded into the parameter.
        """
        if self.is_parallel:
            param.load_row_parallel_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
        else:
            param.load_weight(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the vocabulary parallel embedding layer.

        Args:
            x: Input tensor of token indices.

        Returns:
            Output tensor with embedded representations.
        """
        embed_out = self.quant_method.apply(self, x)

        if self.is_parallel:
            hidden_state = \
                torch.zeros_like(embed_out).repeat(self.tp_size, 1, 1)
            dist.all_gather_into_tensor(hidden_state,
                                        embed_out,
                                        group=self.parallel_info.process_group)
            hidden_state = hidden_state.permute(1, 0, 2).contiguous()
            hidden_state = hidden_state.view(hidden_state.shape[0], -1)
        else:
            hidden_state = embed_out

        hidden_state = maybe_slice_cross_tp(hidden_state, self.parallel_info)
        return hidden_state

    def extra_repr(self) -> str:
        """Return a string representation of the layer configuration.

        Returns:
            A string containing the layer configuration details.
        """
        s = f"num_embeddings={self.num_embeddings}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", dtype={self.params_dtype}"
        s += f", quant_method={self.quant_method.__class__.__name__}"
        s += f", tp_size={self.tp_size}"
        return s

    def _post_init(self) -> None:
        """Hook method called after initialization, allowing subclasses to perform
        additional setup operations before weight creation.
        """
        pass

    def _create_weights(self) -> None:
        """Create weights for the embedding layer using the quantization method."""
        self.quant_method.create_weights(
            self,
            self.num_embeddings,
            [self.output_partition_size],
            self.num_embeddings,
            self.embedding_dim,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )


class ParallelLMHead(VocabParallelEmbedding):
    """Parallel language model head layer.

    This layer implements the output projection for language models with
    support for tensor parallelism and optional bias.

    Args:
        num_embeddings: Size of the vocabulary (output size).
        embedding_dim: Dimension of the input hidden states.
        bias: Whether to include a bias term. Defaults to False.
        weight_dtype: Data type for weight tensor. If None, uses default dtype.
            Defaults to None.
        bias_dtype: Data type for bias tensor. If None, uses default dtype.
            Defaults to None.
        quant_config: Quantization configuration for the layer. If None,
            uses unquantized method. Defaults to None.
        prefix: Prefix string used to construct the full layer name in the state dictionary. Defaults to "".
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str = "",
    ) -> None:
        self.has_bias = bias
        self.weight_dtype = weight_dtype
        self.bias_dtype = bias_dtype
        self.skip_bias_add = False

        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=weight_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        """Load weight into a parameter with column-parallel support.

        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor read from file to be loaded into the parameter.
        """
        if isinstance(param, ColumnParameter):
            param.load_column_parallel_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
        else:
            param.load_weight(loaded_weight=loaded_weight)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding) -> "ParallelLMHead":
        """Tie weights with an embedding layer.

        Args:
            embed_tokens: The embedding layer to share weights with.

        Returns:
            Self for method chaining.
        """
        self.prefix = embed_tokens.prefix
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the parallel language model head.

        Args:
            hidden_states: Input hidden states tensor.
            lm_head_indices: Optional indices for selectively retrieving specific outputs
                to improve performance during the prefill phase. Defaults to None.

        Returns:
            Output logits tensor.
        """
        if lm_head_indices is not None:
            lm_head_indices = lm_head_indices.unsqueeze(1).expand(-1, hidden_states.shape[1])
            hidden_states = torch.gather(hidden_states, dim=0, index=lm_head_indices)
            lm_head_out = self.quant_method.apply(self, hidden_states)
        else:
            lm_head_out = self.quant_method.apply(self, hidden_states)
        
        if self.tp_size > 1:
            logits = torch.zeros_like(lm_head_out).repeat(self.tp_size, 1, 1)
            dist.all_gather_into_tensor(logits, lm_head_out, group=self.parallel_info.process_group)
            logits = logits.permute(1, 0, 2).contiguous()
            logits = logits.view(logits.shape[0], -1)
        else:
            logits = lm_head_out
        return logits

    def _post_init(self) -> None:
        """Post-initialization: configure layer-specific parallel info and quantization method.
        
        Note that although `ParallelLMHead inherits from VocabParallelEmbedding, it uses
        different parallel info (lm_head_tp) and quantization method type (linear) compared
        to the parent class.
        """
        self.parallel_info = get_parallel_info_manager().lm_head_tp

        self.tp_rank = self.parallel_info.rank
        self.tp_size = self.parallel_info.group_size
        if self.quant_config is None:
            self.quant_method: QuantizationMethodBase | None = UnquantizedLinearMethod()
        else:
            self.quant_method = self.quant_config.get_quant_method(self, prefix=self.prefix)

    def _create_weights(self) -> None:
        """Create weights for the language model head using the linear quantization method."""
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.embedding_dim,
            output_partition_sizes=[even_divide(self.num_embeddings, self.tp_size)],
            input_size=self.embedding_dim,
            output_size=self.num_embeddings,
            bias=self.has_bias,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype,
            weight_loader=self.weight_loader,
        )

        if self.has_bias:
            self.bias = BaseParameter(
                torch.empty(even_divide(self.num_embeddings, self.tp_size), dtype=self.bias_dtype))
            self.bias.add_attrs({
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)


def maybe_slice_cross_tp(input_, parallel_info):
    # After enabling flash_comm, slice hidden_states from the attn_tp range onto each device.
    forward_context = get_forward_context()
    if not forward_context.batch_descriptor.is_flash_comm_enabled:
        return input_

    world_size = parallel_info.group_size
    if world_size <= 1:
        return input_
    
    pad_size = (world_size - (input_.shape[0] % world_size)) % world_size
    if pad_size > 0:
        input_ = F.pad(input_, (0, 0, 0, pad_size))
    output_ = input_.view(world_size, -1, *input_.shape[1:])[parallel_info.rank]

    return output_