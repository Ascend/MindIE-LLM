# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import List

import torch
import torch_npu
from torch import nn

from mindie_llm.runtime.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from mindie_llm.runtime.layers.linear.linear_method_base import LinearMethodBase
from mindie_llm.runtime.layers.parameter import (
    BiasParameter,
    ModelWeightParameter,
    ScalerParameter,
    PerTensorScaleParameter,
    ColumnParameter,
    RowParameter,
    ExpertsParameter
)
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import InferenceMode
from mindie_llm.runtime.utils.npu.device_utils import get_npu_node_info
from mindie_llm.runtime.utils.distributed.utils import even_divide
from mindie_llm.utils.log.logging import logger


SUPPORT_NZ_NPU_LIST = ("Ascend910B3", "Ascend910B4-1", "Ascend910_9382", "Ascend910_9362")
MXFP8_GROUP_SIZE = 32


class W8A8PerTensorLinearMethod(LinearMethodBase):
    """
    Implements per-tensor weight and activation quantization (W8A8).
    """
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
        """
        Creates and registers quantized weights and scales.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
        )
        weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs,
        })

        input_scale = ScalerParameter(data=torch.empty(1, dtype=weight_dtype))
        input_scale.add_attrs(extra_weight_attrs)

        input_offset = ScalerParameter(data=torch.empty(1, dtype=torch.int8))
        input_offset.add_attrs(extra_weight_attrs)

        deq_scale_dtype = weight_dtype
        if weight_dtype == torch.float16:
            deq_scale_dtype = torch.int64
        elif weight_dtype == torch.bfloat16:
            deq_scale_dtype = torch.float32
        else:
            raise ValueError(f"Dtype {weight_dtype} is not supported in `W8A8PerTensorLinearMethod`.")

        deq_scale = PerTensorScaleParameter(data=torch.empty(sum(output_partition_sizes), dtype=deq_scale_dtype))
        deq_scale.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})

        quant_bias = BiasParameter(data=torch.empty(sum(output_partition_sizes), dtype=torch.int32))
        quant_bias.add_attrs({self.INPUT_DIM: 0, self.OUTPUT_DIM: 0, **extra_weight_attrs})

        layer.register_parameter("weight", weight)
        layer.register_parameter("input_scale", input_scale)
        layer.register_parameter("input_offset", input_offset)
        layer.register_parameter("deq_scale", deq_scale)
        layer.register_parameter("quant_bias", quant_bias)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies per-tensor quantization and matrix multiplication.

        Args:
            layer: The layer containing quantization parameters.
            x: Input tensor.
        Returns:
            Output tensor after quantized matmul.
        """

        # Quantize input tensor to 8-bit signed integer (qint8) using per-tensor non-symmetric quantization.
        # Parameters:
        #   layer.input_scale.data: Scale factor for quantization (per-tensor)
        #   layer.input_offset.data: Zero-point offset for non-symmetric quantization
        #   torch.qint8: Target quantization data type (8-bit signed integer)
        #   axis=-1: Quantize along the LAST dimension (last axis) of the input tensor
        #   div_mode=False: Use MULTIPLICATION (not division) for scale application in quantization.
        input_tensor_quant = torch_npu.npu_quantize(
            input=x, scales=layer.input_scale.data,
            zero_points=layer.input_offset.data,
            dtype=torch.qint8, axis=-1, div_mode=False)
        out = torch_npu.npu_quant_matmul(
            input_tensor_quant, layer.weight.data, layer.deq_scale.data,
            bias=layer.quant_bias.data, output_dtype=x.dtype)
        return out

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]
        layer.input_scale.data = \
            1 / layer.input_scale.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.input_offset.data = \
            layer.input_offset.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()

        soc_name = get_npu_node_info().soc_name
        if soc_name in SUPPORT_NZ_NPU_LIST: 
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
            logger.debug("Convert weight to FRACTAL_NZ done, current format is %s", 
                       torch_npu.get_npu_format(layer.weight.data))


class W8A8PerTokenLinearMethod(LinearMethodBase):
    """
    Implements per-token activation quantization with per-tensor weight quantization (W8A8).
    """
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """
        Creates and registers quantized weights and scales.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
        )
        weight.add_attrs({self.INPUT_DIM: 1, self.OUTPUT_DIM: 0, **extra_weight_attrs})

        weight_scale_type = torch.float32 if weight_dtype == torch.float16 else torch.bfloat16
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(sum(output_partition_sizes), 1, dtype=weight_scale_type),
        )
        weight_scale.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})

        weight_offset = PerTensorScaleParameter(
            data=torch.empty(sum(output_partition_sizes), 1, dtype=torch.float16),
        )
        weight_offset.add_attrs({self.OUTPUT_DIM: 0, **extra_weight_attrs})

        enable_anti_outlier = True
        try:
            layer.quant_config.get_quant_type_by_weight_name(layer.prefix, self.BIAS)
        except ValueError:
            enable_anti_outlier = False
        if enable_anti_outlier:
            bias = BiasParameter(torch.zeros(sum(output_partition_sizes), dtype=bias_dtype))
            bias.add_attrs({"output_dim": 0, **extra_weight_attrs})
            layer.register_parameter(self.BIAS, bias)
        else:
            layer.register_parameter(self.BIAS, None)

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_offset", weight_offset)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_tensor_quant, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        out = torch_npu.npu_quant_matmul(
            input_tensor_quant, layer.weight.data, layer.weight_scale.data,
            pertoken_scale=pertoken_scale, bias=None, output_dtype=x.dtype)
        if layer.bias is not None:
            out = out + layer.bias.data
        return out

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()

        soc_name = get_npu_node_info().soc_name
        if soc_name in SUPPORT_NZ_NPU_LIST:
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
            logger.debug("Convert weight to FRACTAL_NZ done, current format is %s", 
                       torch_npu.get_npu_format(layer.weight.data))


class W8A8MixLinearMethod(LinearMethodBase):
    """
    Implements mixed W8A8 quantization using per-tensor for decode and per-token for prefill.
    """
    quant_method = {
        InferenceMode.PREFILL: W8A8PerTokenLinearMethod(),
        InferenceMode.DECODE: W8A8PerTensorLinearMethod(),
    }

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """
        Creates and registers quantized weights for both prefill and decode modes.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        for quant_method in self.quant_method.values():
            quant_method.create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                bias,
                weight_dtype,
                bias_dtype,
                **extra_weight_attrs,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: Temprorarily set `is_prefill` to true, wait for `get_forward_context` ready to replace
        is_prefill = True
        if is_prefill:
            result = self.quant_method[InferenceMode.PREFILL].apply(layer, x)
        else:
            result = self.quant_method[InferenceMode.DECODE].apply(layer, x)
        return result

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]
        layer.input_scale.data = \
            1 / layer.input_scale.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.input_offset.data = \
            layer.input_offset.data.repeat(expanding_factor).to(layer.weight_dtype).contiguous().npu()
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()

        soc_name = get_npu_node_info().soc_name
        if soc_name in SUPPORT_NZ_NPU_LIST:
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 29)
            logger.debug("Convert weight to FRACTAL_NZ done, current format is %s", 
                       torch_npu.get_npu_format(layer.weight.data))


class W8A8MXFP8PerGroupLinearMethod(LinearMethodBase):
    """
    Implements per-token activation quantization with per-group weight quantization (W8A8_MXFP8).
    """
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        bias: bool,
        weight_dtype: torch.dtype,
        bias_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Creates and registers quantized weights for both prefill and decode modes.

        Args:
            layer: The layer to register parameters in.
            input_size_per_partition: Input dimension for this partition.
            output_partition_sizes: List of output dimensions for each partition.
            bias: Whether to create bias parameters.
            weight_dtype: Data type for weights.
            bias_dtype: Data type for bias.
            **extra_weight_attrs: Additional attributes for parameters.
        """
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.float8_e4m3fn),
        )
        weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs,
        })

        weight_scale = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes),
                             even_divide(input_size_per_partition, MXFP8_GROUP_SIZE),
                             dtype=torch.uint8),
        )
        weight_scale.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs,
        })

        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("bias", None)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        pertoken_scale = dynamic_scale
        output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=layer.bias,
            output_dtype=output_dtype,
            group_sizes=[1, 1, MXFP8_GROUP_SIZE]
        )

        return output

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, even_divide(k_dim, 2), 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)


class W8A8PerTokenFusedMoEMethod(FusedMoEMethodBase):
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
                        dtype=torch.int8,
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
                        dtype=torch.int8,
                        ),
        )
        down_weight.add_attrs({
            self.INPUT_DIM: 1,
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs
        })
        layer.register_parameter("down_weight", down_weight)

        weight_scale_type = torch.float32 if weight_dtype == torch.float16 else torch.bfloat16
        layer.output_dtype = torch.bfloat16 if weight_scale_type == torch.bfloat16 else torch.float16
        gate_up_weight_scale = ColumnParameter(
            torch.empty(num_experts,
                        2 * intermediate_size_per_partition,
                        1,
                        dtype=weight_scale_type,
                        ),
        )
        gate_up_weight_scale.add_attrs({
            self.OUTPUT_DIM: 0,
            **extra_weight_attrs
        })
        layer.register_parameter("gate_up_weight_scale", gate_up_weight_scale)

        down_weight_scale = ExpertsParameter(
            torch.empty(num_experts,
                        hidden_size,
                        1,
                        dtype=weight_scale_type,
                        ),
        )
        down_weight_scale.add_attrs({
            self.INPUT_DIM: 0,
            **extra_weight_attrs
        })
        layer.register_parameter("down_weight_scale", down_weight_scale)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              group_list: torch.Tensor,
              group_list_type: int = 1,
              dynamic_scale: torch.Tensor = None) -> torch.Tensor:
        if dynamic_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        else:
            pertoken_scale = dynamic_scale

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[layer.gate_up_weight],
            split_item=2,                     # output a single tensor
            group_list_type=group_list_type,
            group_type=0,                     # the axis to group
            group_list=group_list,
            output_dtype=torch.int32)[0]

        hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=layer.gate_up_weight_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,  # whether to left-activate
            quant_mode=1,        # 0: static quant, 1: dynamic quant
        )

        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.down_weight],
            scale=[layer.down_weight_scale],
            bias=None,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=layer.output_dtype)[0]

        return hidden_states

    def process_weights_after_loading(self, layer):
        layer.gate_up_weight.data = layer.gate_up_weight.data.transpose(-2, -1).contiguous()
        layer.down_weight.data = layer.down_weight.data.transpose(-2, -1).contiguous()
        layer.gate_up_weight_scale.data = layer.gate_up_weight_scale.data.squeeze(-1).to(torch.float32)
        layer.down_weight_scale.data = layer.down_weight_scale.data.squeeze(-1)
