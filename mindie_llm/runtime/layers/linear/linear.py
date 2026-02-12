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
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelInfo
from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.layers.parameter import BaseParameter, RowParameter, ColumnParameter, BiasParameter
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.quantization_config_base import QuantizationConfigBase
from mindie_llm.runtime.layers.linear.linear_method_base import LinearMethodBase
from mindie_llm.runtime.layers.linear.linear_op import get_linear_custom_op
from mindie_llm.runtime.layers.quantization.unquantized import UnquantizedLinearMethod
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.utils import even_divide
from mindie_llm.utils.log.logging import logger


class LinearBase(CustomLayer):
    """Base class for linear layers with support for weight initialization, quantization and tensor parallelism.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str | list[str] = "",
        parallel_info: ParallelInfo | None = None
    ):
        """Initialize the LinearBase layer.

        Args:
            input_size: Dimensionality of the input features.
            output_size: Dimensionality of the output features.
            bias: If True, adds a learnable bias to the output. Default is True.
            skip_bias_add: If True, bias is not added to the output during forward pass.
                This is useful for fused kernels. Default is False.
            return_bias: If True, the forward pass returns the bias tensor along with the output.
                Default is False.
            weight_dtype: Data type for the layer weights. If None, uses torch default.
            bias_dtype: Data type for the layer bias. If None, uses torch default.
            quant_config: Configuration object for weight quantization.
            prefix: Prefix for parameter names, used for weight loading/checkpointing.
            parallel_info: Object containing tensor parallelism information (rank, group size).
        """
        super().__init__()

        if (not bias) and skip_bias_add:
            raise ValueError(
                f"Cannot set `bias` to False and `skip_bias_add` to True simultaneously for Linear layers.")

        if (not bias) and return_bias:
            raise ValueError(
                f"Cannot set `bias` to False and `return_bias` to True simultaneously for Linear layers.")

        self.input_size = input_size
        self.input_size_per_partition = input_size
        self.output_size = output_size
        self.output_partition_sizes = [self.output_size]
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.weight_dtype = weight_dtype or torch.get_default_dtype()
        self.bias_dtype = bias_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix
        self.return_bias = return_bias

        if parallel_info is not None:
            self.tp_rank = parallel_info.rank
            self.tp_size = parallel_info.group_size
        else:
            self.tp_rank = get_parallel_info_manager().rank
            self.tp_size = get_parallel_info_manager().world_size

        if self.quant_config is None:
            self.quant_method: LinearMethodBase | None = UnquantizedLinearMethod()
        else:
            self.quant_method = self.quant_config.get_quant_method(self, prefix=self.prefix)

        self._post_init()
        self._create_weights()

    def __call__(self, *args, **kwargs):
        """
        Select an appropriate method to replace the forward implementation,
        typically related to features.

        Note: should be optimized, the op_cls object could be only init once.
        """
        custom_op = get_linear_custom_op(self)
        if custom_op is not None:
            custom_op.update_attrs()
            return custom_op(*args, **kwargs)
        
        return super().__call__(*args, **kwargs)

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor, **kwargs) -> None:
        param.load_weight(loaded_weight=loaded_weight)

    def _post_init(self) -> None:
        pass

    def _create_weights(self) -> None:
        """Initializes weights and bias based on quantization method."""
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            bias=self.has_bias,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype,
            weight_loader=self.weight_loader,
        )


class ReplicatedLinear(LinearBase):
    """
    Standard Linear layer where weights are replicated across all devices.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str | list[str] = "",
    ):
        if skip_bias_add:
            logger.warning("Set `skip_bias_add` to True is not tested in `ReplicatedLinear`.")

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            return_bias=return_bias,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, BaseParameter | None]:
        """
        Args:
            x: Input tensor.

        Returns:
            - If return_bias is False: Output tensor.
            - If return_bias is True: Tuple of (Output tensor, Bias parameter).
        """
        output = self.quant_method.apply(self, x)
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", weight_dtype={self.weight_dtype}"
        s += f", bias={getattr(self, 'bias', None) is not None}"
        s += f", bias_dtype={self.bias_dtype}"
        s += f", skip_bias_add={self.skip_bias_add}"
        s += f", quant_method={self.quant_method.__class__.__name__}"
        return s


class RowParallelLinear(LinearBase):
    """
    Linear layer where the weight matrix is split along the row dimension (input dimension).
    An all-reduce operation is performed on the output to aggregate results.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str | list[str] = "",
        parallel_info: ParallelInfo | None = None,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
    ):
        """
        Args:
            input_size: Total input dimension before splitting.
            output_size: Output dimension (not split).
            input_is_parallel: If True, input is assumed to be already split across devices.
            reduce_results: If True, performs all-reduce on the output to sum partial results.
        """
        if not input_is_parallel:
            raise NotImplementedError("`RowParallelLinear` doesn't support setting `input_is_parallel` to False.")

        if skip_bias_add:
            logger.warning("Set `skip_bias_add` to True is not tested in `RowParallelLinear`.")

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results in `RowParallelLinear`, adding bias to the "
                "results can lead to incorrect results"
            )

        self.parallel_info = parallel_info
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            return_bias=return_bias,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            quant_config=quant_config,
            prefix=prefix,
            parallel_info=parallel_info
        )

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        if isinstance(param, RowParameter):
            param.load_row_parallel_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
        else:
            param.load_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, BaseParameter | None]:
        """
        Args:
            input_: Input tensor.

        Returns:
            - Output tensor (aggregated if reduce_results is True).
            - Optionally, the bias tensor if return_bias is True.
        """
        if self.input_is_parallel:
            input_parallel = input_

        output_parallel = self.quant_method.apply(self, input_parallel)
        if self.reduce_results:
            dist.all_reduce(output_parallel, group=self.parallel_info.process_group)
            
        output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", weight_dtype={self.weight_dtype}"
        s += f", bias={getattr(self, 'bias', None) is not None}"
        s += f", bias_dtype={self.bias_dtype}"
        s += f", quant_method={self.quant_method.__class__.__name__}"
        s += f", tp_rank={self.tp_rank}"
        s += f", tp_size={self.tp_size}"
        return s

    def _post_init(self) -> None:
        self.input_size_per_partition = even_divide(self.input_size, self.tp_size)
        self.output_partition_sizes = [self.output_size]

        if self.parallel_info is not None:
            self.tp_rank = self.parallel_info.rank
            self.tp_size = self.parallel_info.group_size


class ColumnParallelLinear(LinearBase):
    """
    Linear layer where the weight matrix is split along the column dimension (output dimension).
    Outputs are partial results corresponding to the split.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str | list[str] = "",
        parallel_info: ParallelInfo | None = None,
        gather_output: bool = False,
    ):
        """
        Args:
            input_size: Input dimension (not split).
            output_size: Total output dimension before splitting.
            gather_output: If True, gathers outputs from all devicess. Currently not supported.
        """
        if gather_output:
            raise NotImplementedError("`ColumnParallelLinear` doesn't support setting `gather_output` to True.")
        
        if skip_bias_add:
            logger.warning("Set `skip_bias_add` to True is not tested in `ColumnParallelLinear`.")
        self.parallel_info = parallel_info
        self.gather_output = gather_output
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            return_bias=return_bias,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            quant_config=quant_config,
            prefix=prefix,
            parallel_info=parallel_info
        )

    def weight_loader(self, param: BaseParameter, loaded_weight: torch.Tensor) -> None:
        if isinstance(param, ColumnParameter):
            param.load_column_parallel_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
        else:
            param.load_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, BaseParameter | None]:
        """
        Args:
            input_: Input tensor.

        Returns:
            - Output tensor (partial result).
            - Optionally, the bias tensor if return_bias is True.
        """
        output_parallel = self.quant_method.apply(self, input_)
        output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features=sum({self.output_partition_sizes})={sum(self.output_partition_sizes)}"
        s += f", weight_dtype={self.weight_dtype}"
        s += f", bias={getattr(self, 'bias', None) is not None}"
        s += f", bias_dtype={self.bias_dtype}"
        s += f", quant_method={self.quant_method.__class__.__name__}"
        s += f", tp_rank={self.tp_rank}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s

    def _post_init(self) -> None:
        self.output_partition_sizes = [even_divide(self.output_size, self.tp_size)]

        if self.parallel_info is not None:
            self.tp_rank = self.parallel_info.rank
            self.tp_size = self.parallel_info.group_size

    def _create_weights(self) -> None:
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            bias=self.has_bias,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.bias_dtype,
            weight_loader=self.weight_loader,
        )


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    Merges multiple ColumnParallelLinear layers into a single layer for efficiency.
    The weight matrix is formed by concatenating the weights of the original layers.
    """
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str | list[str] = "",
        parallel_info: ParallelInfo | None = None,
        gather_output: bool = False,
    ):
        """
        Args:
            input_size: Input dimension (shared by all merged layers).
            output_sizes: List of output dimensions for each of the merged layers.
        """
        if gather_output:
            raise NotImplementedError("`MergedColumnParallelLinear` doesn't support setting `gather_output` to True.")
        
        if skip_bias_add:
            logger.warning("Set `skip_bias_add` to True is not tested in `MergedColumnParallelLinear`.")

        self.output_sizes = output_sizes

        super().__init__(
            input_size,
            sum(output_sizes),
            bias=bias,
            skip_bias_add=skip_bias_add,
            return_bias=return_bias,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            quant_config=quant_config,
            prefix=prefix,
            parallel_info=parallel_info,
            gather_output=gather_output
        )

    def weight_loader(
        self,
        param: BaseParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        """
        Loads weights for a specific shard of the merged linear layer.
        
        Args:
            param: The parameter to load weights into.
            loaded_weight: The weight tensor to load.
            loaded_shard_id: The index of the shard (corresponding to the index in output_sizes).
        """

        # Validate shard ID
        if loaded_shard_id >= len(self.output_sizes):
            raise ValueError(
                f"The parameter `loaded_shard_id` {loaded_shard_id} exceeds the valid range of "
                f"indices for the `output_sizes` array {self.output_sizes} defined in `MergedColumnParallelLinear`.")

        # Obtain a tensor slice of size `shard_size` starting from `shard_offset` in self.output_sizes
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        if isinstance(param, ColumnParameter):
            param.load_merged_column_weight(
                loaded_weight=loaded_weight,
                tp_rank=self.tp_rank,
                shard_offset=shard_offset,
                shard_size=shard_size,
            )
        else:
            param.load_weight(loaded_weight=loaded_weight)

    def _post_init(self) -> None:
        for output_size in self.output_sizes:
            if output_size % self.tp_size != 0:
                raise ValueError(f"All `output_sizes` {self.output_sizes} in `MergedColumnParallelLinear` "
                                 f"must be multiples of the tensor parallel size {self.tp_size}.")

        if self.output_sizes is not None:
            self.output_partition_sizes = [even_divide(output_size, self.tp_size) for output_size in self.output_sizes]


class QKVParallelLinear(ColumnParallelLinear):
    """
    Fused Query-Key-Value projection layer for Attention mechanisms.
    Handles splitting of Q, K, and V heads across devices.
    """
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        return_bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str | list[str] = "",
        parallel_info: ParallelInfo | None = None,
    ):
        """
        Args:
            hidden_size: Input hidden dimension.
            head_size: Dimension of each attention head.
            total_num_heads: Total number of query heads.
            total_num_kv_heads: Total number of key/value heads. If None, defaults to total_num_heads.
        """
        if skip_bias_add:
            logger.warning("Set `skip_bias_add` to True is not tested in `RowParallelLinear`.")

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        if parallel_info is None:
            tp_size = get_parallel_info_manager().world_size
        else:
            tp_size = parallel_info.group_size
        self.num_heads = even_divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = even_divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = even_divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        # 2: K and V each need their own heads (K has self.num_kv_heads, V has self.num_kv_heads)
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )

        super().__init__(
            self.hidden_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            return_bias=return_bias,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            quant_config=quant_config,
            prefix=prefix,
            parallel_info=parallel_info,
            gather_output=False,
        )

    def weight_loader(
        self,
        param: BaseParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        """
        Loads weights for Q, K, or V projections.
        
        Args:
            param: The parameter to load weights into.
            loaded_weight: The weight tensor to load.
            loaded_shard_id: Identifier for the projection type: 0 for Q, 1 for K, 2 for V.
        """
        if loaded_shard_id not in [0, 1, 2]:
            raise ValueError(f"Invalid `loaded_shard_id` passing to the `QKVParallelLinear` module. "
                             f"Expected an integer in the range of [0, 2], but got {loaded_shard_id}.")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if isinstance(param, ColumnParameter):
            param.load_qkv_weight(
                loaded_weight=loaded_weight,
                shard_offset=shard_offset,
                shard_size=shard_size,
                tp_rank=self.tp_rank,
                shard_id=loaded_shard_id,
                num_kv_head_replicas=self.num_kv_head_replicas,
            )
        else:
            param.load_weight(loaded_weight=loaded_weight)

    def _post_init(self):
        self.output_partition_sizes = [
            self.num_heads * self.head_size,  # q
            self.num_kv_heads * self.head_size,  # k
            self.num_kv_heads * self.head_size,  # v
        ]

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int:
        """Returns the offset in the weight matrix for a given shard ID."""
        shard_offset_mapping = {
            0: 0,
            1: self.num_heads * self.head_size,
            2: (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int:
        """Returns the size of the shard for a given shard ID."""
        shard_size_mapping = {
            0: self.num_heads * self.head_size,
            1: self.num_kv_heads * self.head_size,
            2: self.num_kv_heads * self.head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)
