#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABC, abstractmethod

import torch
import torch_npu
from torch import nn

from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.quantize.quant_type import LinearTypeV2, QuantType
from atb_llm.utils.env import ENV
from atb_llm.utils.initial import NPUSocInfo
from mindie_llm.runtime.layers.linear.linear import LinearBase
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import InferenceMode
from mindie_llm.runtime.layers.quantization.unquantized import (
    UnquantizedLinearMethod as UnquantizedLinearMethodAdaptee,
    UnquantizedEmbeddingMethod as UnquantizedEmbeddingMethodAdaptee,
    UnquantizedNormMethod as UnquantizedNormMethodAdaptee,
)
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import \
    QuantizationConfig as QuantizationConfigAdaptee
from mindie_llm.runtime.layers.quantization.ms_model_slim.w8a8 import (
    W8A8PerTensorLinearMethod as W8A8PerTensorLinearMethodAdaptee,
    W8A8PerTokenLinearMethod as W8A8PerTokenLinearMethodAdaptee,
    W8A8MixLinearMethod as W8A8MixLinearMethodAdaptee,
)
from mindie_llm.runtime.layers.quantization.ms_model_slim.anti_outlier import \
    AntiOutlierNormMethod as AntiOutlierNormMethodAdaptee
from mindie_llm.utils.log.logging import logger


class QuantizationConfig:
    def __init__(self, adaptee: QuantizationConfigAdaptee, default_quant_method_cls):
        self.adaptee = adaptee
        self.default_quant_method_cls = default_quant_method_cls

    @staticmethod
    def _get_method_adapter(quant_method: QuantizationMethodBase) -> QuantizationMethodBase | None:
        quant_method_cls_adapter_map = {
            UnquantizedLinearMethodAdaptee: UnquantizedLinearMethod,
            UnquantizedEmbeddingMethodAdaptee: UnquantizedEmbeddingMethod,
            UnquantizedNormMethodAdaptee: UnquantizedNormMethod,
            W8A8PerTensorLinearMethodAdaptee: W8A8PerTensorLinearMethod,
            W8A8PerTokenLinearMethodAdaptee: W8A8PerTokenLinearMethod,
            W8A8MixLinearMethodAdaptee: W8A8MixLinearMethod,
            AntiOutlierNormMethodAdaptee: AntiOutlierNormMethod,
        }
        for quant_method_cls, adapter_cls in quant_method_cls_adapter_map.items():
            if isinstance(quant_method, quant_method_cls):
                return adapter_cls(quant_method)
        raise NotImplementedError(f"Cannot found the adapter class for `{quant_method}`")

    def get_quant_type_by_weight_name(self, *args, **kwargs) -> str:
        if self.adaptee is None:
            raise ValueError(f"NoneType adaptee doesn't support `get_quant_type_by_weight_name`.")
        return self.adaptee.get_quant_type_by_weight_name(*args, **kwargs)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str | list[str] = ""
    ) -> QuantizationMethodBase | None:
        if self.adaptee is not None:
            quant_method = self.adaptee.get_quant_method(layer, prefix)
        else:
            quant_method = self.default_quant_method_cls()
        return self._get_method_adapter(quant_method)


class MethodSupportAtbGraph(ABC):
    _PLACEHOLDER = torch.tensor([1], dtype=torch.get_default_dtype(), device='npu')

    def __init__(self, adaptee: QuantizationMethodBase):
        self.adaptee = adaptee

    def create_weights(self, *args, **kwargs) -> None:
        self.adaptee.create_weights(*args, **kwargs)

    def apply(self, *args, **kwargs) -> None:
        self.adaptee.apply(*args, **kwargs)

    def process_weights_after_loading(self, *args, **kwargs) -> None:
        self.adaptee.process_weights_after_loading(*args, **kwargs)

    @abstractmethod
    def get_weights_for_atb_graph(self, layer: torch.nn.Module, padding: bool = True) -> list[torch.Tensor]:
        pass


class LinearMethodSupportAtbGraph(MethodSupportAtbGraph):
    _soc_info = None

    def __init__(self, adaptee: QuantizationMethodBase):
        super().__init__(adaptee)

    @classmethod
    def set_soc_info(cls, soc_info: NPUSocInfo):
        cls._soc_info = soc_info

    @abstractmethod
    def get_weights_for_atb_graph(
        self, layer: torch.nn.Module, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def get_linear_descs(self, layer: LinearBase) -> list[LinearTypeV2]:
        pass

    @abstractmethod
    def get_weight_transpose_type(self, layer: LinearBase) -> list[TransposeType]:
        pass

    def _check_transpose(self, weight_shape: torch.Size) -> TransposeType:
        if self._soc_info is None:
            raise ValueError("``NPUSocInfo` is not set in `LinearMethodSupportAtbGraph`.")

        if self._soc_info.need_nz:
            return TransposeType.TRANSPOSE

        if not ENV.auto_transpose_enable:
            if self._soc_info.matmul_nd_nz:
                logger.warning("NZ weight format is enabled. To ensure hardware compatibility, "
                               "weights must be transposed to [k, n]. The environment variable "
                               "`ATB_LLM_ENABLE_AUTO_TRANSPOSE=0` is being ignored.")
                ENV.auto_transpose_enable = True
                return TransposeType.NOT_TRANSPOSE
            return TransposeType.TRANSPOSE

        if self._soc_info.matmul_nd_nz:
            # transpose weights to [k, n] when using nz format
            return TransposeType.NOT_TRANSPOSE

        is_k_divisible = weight_shape[-1] % 256 == 0  # Input dimension (k) alignment check
        is_n_divisible = weight_shape[-2] % 256 == 0  # Output dimension (n) alignment check
        if not is_k_divisible and is_n_divisible and len(weight_shape) != 3:
            return TransposeType.NOT_TRANSPOSE
        return TransposeType.TRANSPOSE


class UnquantizedLinearMethod(LinearMethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_transpose_type = TransposeType.TRANSPOSE

    def get_weights_for_atb_graph(
        self, layer: torch.nn.Module, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        weight_list = [layer.weight.data]
        if layer.bias is not None:
            weight_list.append(layer.bias.data)
        if not padding:
            return weight_list
        if layer.bias is None:
            weight_list.append(self._PLACEHOLDER)
        weight_list.extend([self._PLACEHOLDER] * 4)
        return weight_list

    def get_linear_descs(self, layer: LinearBase) -> list[LinearTypeV2]:
        if layer.weight.data.dtype == torch.bfloat16:
            return LinearTypeV2.BFLOAT16
        else:
            return LinearTypeV2.FLOAT16

    def get_weight_transpose_type(self, layer: LinearBase) -> list[TransposeType]:
        return self._weight_transpose_type

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight_shape = layer.weight.data.shape
        self._weight_transpose_type = self._check_transpose(weight_shape)
        if self._weight_transpose_type == TransposeType.NOT_TRANSPOSE:
            layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()
        if self._soc_info.need_nz or self._soc_info.matmul_nd_nz:
            layer.weight.data = torch_npu.npu_format_cast_(layer.weight.data, 29)


class W8A8PerTensorLinearMethod(LinearMethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, layer: torch.nn.Module, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        if is_swiglu_quant_enabled:
            layer.input_offset.data = layer.input_offset.data.to(torch.float32)
            layer.input_scale.data = layer.input_scale.data.to(torch.float32)
        weight_list = [
            layer.weight.data, layer.quant_bias.data, layer.deq_scale.data,
            layer.input_offset.data, layer.input_scale.data
        ]
        if not padding:
            return weight_list
        weight_list.append(self._PLACEHOLDER)
        return weight_list

    def get_linear_descs(self, layer: LinearBase) -> list[LinearTypeV2]:
        return LinearTypeV2.W8A8

    def get_weight_transpose_type(self, layer: LinearBase) -> list[TransposeType]:
        return self._weight_transpose_type

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight_shape = layer.weight.data.shape
        self._weight_transpose_type = self._check_transpose(weight_shape)
        if self._weight_transpose_type == TransposeType.NOT_TRANSPOSE:
            layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()
        if self._soc_info.need_nz or self._soc_info.matmul_nd_nz:
            layer.weight.data = torch_npu.npu_format_cast_(layer.weight.data, 29)


class W8A8PerTokenLinearMethod(LinearMethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(
        self, layer: torch.nn.Module, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        weight_list = [layer.weight.data]
        if layer.bias is not None:
            weight_list.append(layer.bias.data)
        else:
            weight_list.append(self._PLACEHOLDER)
        weight_list.append(self._PLACEHOLDER)
        weight_list.extend([layer.weight_offset.data, layer.weight_scale.data])
        if not padding:
            return weight_list
        weight_list.append(self._PLACEHOLDER)
        return weight_list

    def get_linear_descs(self, layer: LinearBase) -> list[LinearTypeV2]:
        return LinearTypeV2.W8A8_DYNAMIC

    def get_weight_transpose_type(self, layer: LinearBase) -> list[TransposeType]:
        return self._weight_transpose_type

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight_shape = layer.weight.data.shape
        self._weight_transpose_type = self._check_transpose(weight_shape)
        if self._weight_transpose_type == TransposeType.NOT_TRANSPOSE:
            layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = -layer.weight_offset.data.flatten()
        if self._soc_info.need_nz or self._soc_info.matmul_nd_nz:
            layer.weight.data = torch_npu.npu_format_cast_(layer.weight.data, 29)


class W8A8MixLinearMethod(LinearMethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prefill_quant_method = self.adaptee.quant_method.get(InferenceMode.PREFILL)
        if prefill_quant_method is not None:
            self.adaptee.quant_method[InferenceMode.PREFILL] = \
                W8A8PerTokenLinearMethod(self.adaptee.quant_method[InferenceMode.PREFILL])
        decode_quant_method = self.adaptee.quant_method.get(InferenceMode.DECODE)
        if decode_quant_method is not None:
            self.adaptee.quant_method[InferenceMode.DECODE] = \
                W8A8PerTensorLinearMethod(self.adaptee.quant_method[InferenceMode.DECODE])

    def get_weights_for_atb_graph(
        self, layer: torch.nn.Module, padding: bool = True,
        is_swiglu_quant_enabled: bool = False,
        quant_type: QuantType = None
    ) -> list[torch.Tensor]:
        if quant_type == QuantType.W8A8_DYNAMIC:
            quant_method = self.adaptee.quant_method[InferenceMode.PREFILL]
        else:
            quant_method = self.adaptee.quant_method[InferenceMode.DECODE]
        return quant_method.get_weights_for_atb_graph(
            layer=layer, padding=padding,
            is_swiglu_quant_enabled=is_swiglu_quant_enabled
        )

    def get_linear_descs(self, layer: LinearBase) -> list[LinearTypeV2]:
        return LinearTypeV2.W8A8_PDMIX

    def get_weight_transpose_type(self, layer: LinearBase) -> list[TransposeType]:
        return self._weight_transpose_type

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight_shape = layer.weight.data.shape
        self._weight_transpose_type = self._check_transpose(weight_shape)
        if self._weight_transpose_type == TransposeType.NOT_TRANSPOSE:
            layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = -layer.weight_offset.data.flatten()
        if self._soc_info.need_nz or self._soc_info.matmul_nd_nz:
            layer.weight.data = torch_npu.npu_format_cast_(layer.weight.data, 29)


class UnquantizedEmbeddingMethod(MethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(self, layer: torch.nn.Module, padding: bool = True) -> list[torch.Tensor]:
        if not padding:
            raise NotImplementedError(
                f"The `padding` parameter for `get_weights_for_atb_graph` "
                f"in `UnquantizedEmbeddingMethod` cannot be set to False.")
        return [layer.weight.data]


class UnquantizedNormMethod(MethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(self, layer: torch.nn.Module, padding: bool = True) -> list[torch.Tensor]:
        if not padding:
            return [layer.weight.data]
        bias_placeholder = torch.tensor([0] * layer.weight.data.shape[0], dtype=torch.get_default_dtype(), device='npu')
        return [layer.weight.data, bias_placeholder, self._PLACEHOLDER, self._PLACEHOLDER]


class AntiOutlierNormMethod(MethodSupportAtbGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_weights_for_atb_graph(self, layer: torch.nn.Module, padding: bool = True) -> list[torch.Tensor]:
        if not padding:
            weights = [layer.weight.data]
            if not torch.all(layer.bias.data == 0):
                weights.append(layer.bias.data)
            return weights
        return [layer.weight.data, layer.bias.data, self._PLACEHOLDER, self._PLACEHOLDER]
