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

from mindie_llm.runtime.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.linear.linear_method_base import LinearMethodBase
from mindie_llm.runtime.layers.parameter import (
    BaseParameter,
    ModelWeightParameter,
    BiasParameter,
    ColumnParameter,
    RowParameter
)


class UnquantizedLinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=weight_dtype,
            ),
        )
        weight.add_attrs({"input_dim": 1, "output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight", weight)

        # Determine if the anti-outlier feature is active by checking whether "norm.bias" parameters are present
        # in the weight files. If the bias tensor is added in the normalization module, it must be subtracted 
        # from the subsequent linear module to ensure precision.
        if layer.quant_config is not None:
            enable_anti_outlier = True
            try:
                layer.quant_config.get_quant_type_by_weight_name(layer.prefix, self.BIAS)
            except ValueError:
                enable_anti_outlier = False
        else:
            enable_anti_outlier = False
        setattr(layer, "enable_anti_outlier", enable_anti_outlier)

        if bias or enable_anti_outlier:
            bias = BiasParameter(torch.zeros(sum(output_partition_sizes), dtype=bias_dtype))
            bias.add_attrs({"output_dim": 0, **extra_weight_attrs})
            layer.register_parameter(self.BIAS, bias)
        else:
            layer.register_parameter(self.BIAS, None)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.nn.functional.linear(x, layer.weight.data)

        if layer.bias is not None and (not layer.skip_bias_add or layer.enable_anti_outlier):
            out = out + layer.bias.data
        return out


class UnquantizedEmbeddingMethod(QuantizationMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight = ModelWeightParameter(
            torch.empty(
                input_size_per_partition,
                sum(output_partition_sizes),
                dtype=params_dtype,
            ),
        )
        weight.add_attrs({"input_dim": 1, "output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight", weight)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.embedding(x, layer.weight.data)


class UnquantizedNormMethod(QuantizationMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        hidden_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight = BaseParameter(torch.ones(hidden_size, dtype=params_dtype))
        weight.add_attrs(extra_weight_attrs)
        layer.register_parameter("weight", weight)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, layer.weight.data, layer.variance_epsilon)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, layer.weight.data, layer.variance_epsilon)
        return x


class UnquantizedLayerNormBiasMethod(QuantizationMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        hidden_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight = BaseParameter(torch.ones(hidden_size, dtype=params_dtype))
        weight.add_attrs(extra_weight_attrs)
        layer.register_parameter("weight", weight)

        bias = BaseParameter(torch.zeros(hidden_size, dtype=params_dtype))
        bias.add_attrs(extra_weight_attrs)
        layer.register_parameter("bias", bias)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        dim
    ) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, (dim,), layer.weight.data, layer.bias.data, layer.variance_epsilon)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    def create_weights(
            self,
            layer: torch.nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            weight_dtype: torch.dtype,
            bias_dtype: torch.dtype,
            **extra_weight_attrs,
    ):
        gate_up_weight = ColumnParameter(
            torch.empty(num_experts,
                        2 *
                        intermediate_size_per_partition,
                        hidden_size,
                        dtype=weight_dtype,
                        ),
        )
        gate_up_weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs
        })
        layer.register_parameter("gate_up_weight", gate_up_weight)

        down_weight = RowParameter(
            torch.empty(num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        dtype=weight_dtype,
                        ),
        )
        down_weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs
        })
        layer.register_parameter("down_weight", down_weight)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              group_list: torch.Tensor,
              group_list_type: int = 1,
              dynamic_scale: torch.Tensor = None) -> torch.Tensor:
        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[layer.gate_up_weight],
            split_item=2,                     # output a single tensor
            group_list_type=group_list_type,
            group_type=0,                     # the axis to group
            group_list=group_list,
        )[0]

        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[layer.down_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
        )[0]
        return hidden_states

    def process_weights_after_loading(self, layer):
        layer.gate_up_weight.data = layer.gate_up_weight.data.transpose(-2, -1).contiguous()
        layer.down_weight.data = layer.down_weight.data.transpose(-2, -1).contiguous()
