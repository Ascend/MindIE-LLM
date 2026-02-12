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

import sys
import unittest
from unittest.mock import Mock, patch

import torch

from atb_llm.utils.data.quant_method_adapter import (
    UnquantizedLinearMethod,
    W8A8PerTensorLinearMethod,
    W8A8PerTokenLinearMethod,
    W8A8MixLinearMethod,
    UnquantizedEmbeddingMethod,
    UnquantizedNormMethod,
    AntiOutlierNormMethod,
    LinearMethodSupportAtbGraph,
)
from atb_llm.utils.quantize.quant_type import LinearTypeV2, QuantType
from atb_llm.utils.quantize.pack_type import TransposeType
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import InferenceMode


# Mock torch_npu to avoid NPU dependency
class MockTorchNpu:
    @staticmethod
    def npu_format_cast_(x, fmt):
        return x  # no-op


# Mock ENV and NPUSocInfo
class MockNPUSocInfo:
    def __init__(self):
        self.need_nz = False
        self.matmul_nd_nz = False


class MockENV:
    auto_transpose_enable = True


def make_fake_layer(
    weight_shape=(64, 128),
    bias_shape=(64,),
    has_bias=True,
    weight_dtype=torch.float16,
    use_int8_weight=False
):
    layer = Mock()
    if use_int8_weight:
        layer.weight.data = torch.randint(0, 256, weight_shape, dtype=torch.uint8, device='cpu')
    else:
        layer.weight.data = torch.randn(weight_shape, dtype=weight_dtype, device='cpu')
    if has_bias:
        layer.bias = Mock()
        layer.bias.data = torch.randn(bias_shape, dtype=torch.float32, device='cpu')
    else:
        layer.bias = None
    # For W8A8PerToken
    layer.weight_offset = Mock()
    layer.weight_offset.data = torch.tensor([10.0], device='cpu')
    layer.weight_scale = Mock()
    layer.weight_scale.data = torch.tensor([0.5], device='cpu')
    # For W8A8PerTensor
    layer.quant_bias = Mock()
    layer.quant_bias.data = torch.randn(bias_shape, device='cpu')
    layer.deq_scale = Mock()
    layer.deq_scale.data = torch.tensor(0.1, device='cpu')
    layer.input_offset = Mock()
    layer.input_offset.data = torch.tensor(128.0, device='cpu')
    layer.input_scale = Mock()
    layer.input_scale.data = torch.tensor(0.01, device='cpu')
    return layer


class TestQuantMethodAdapter(unittest.TestCase):

    @patch.dict("sys.modules", {"torch_npu": MockTorchNpu()})
    @patch("atb_llm.utils.data.quant_method_adapter.ENV", MockENV())
    def test_unquantized_linear_method(self):
        LinearMethodSupportAtbGraph.set_soc_info(MockNPUSocInfo())

        adaptee = Mock()
        method = UnquantizedLinearMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([999.0], device='cpu')

        # Test with bias, padding=True
        layer = make_fake_layer(has_bias=True)
        weights = method.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 6)

        # Test padding=False
        weights = method.get_weights_for_atb_graph(layer, padding=False)
        self.assertEqual(len(weights), 2)

        # Test dtype
        descs = method.get_linear_descs(layer)
        self.assertEqual(descs, LinearTypeV2.FLOAT16)

        layer_bf16 = make_fake_layer(weight_dtype=torch.bfloat16)
        descs_bf16 = method.get_linear_descs(layer_bf16)
        self.assertEqual(descs_bf16, LinearTypeV2.BFLOAT16)

        # Test process
        method.process_weights_after_loading(layer)

        LinearMethodSupportAtbGraph._soc_info = None

    @patch.dict("sys.modules", {"torch_npu": MockTorchNpu()})
    @patch("atb_llm.utils.data.quant_method_adapter.ENV", MockENV())
    def test_w8a8_per_tensor(self):
        LinearMethodSupportAtbGraph.set_soc_info(MockNPUSocInfo())

        adaptee = Mock()
        method = W8A8PerTensorLinearMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([888.0], device='cpu')

        layer = make_fake_layer(use_int8_weight=True, has_bias=True)
        weights = method.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 6)

        weights = method.get_weights_for_atb_graph(layer, padding=False)
        self.assertEqual(len(weights), 5)

        self.assertEqual(method.get_linear_descs(layer), LinearTypeV2.W8A8)
        method.process_weights_after_loading(layer)

        LinearMethodSupportAtbGraph._soc_info = None

    @patch.dict("sys.modules", {"torch_npu": MockTorchNpu()})
    @patch("atb_llm.utils.data.quant_method_adapter.ENV", MockENV())
    def test_w8a8_per_token(self):
        LinearMethodSupportAtbGraph.set_soc_info(MockNPUSocInfo())

        adaptee = Mock()
        method = W8A8PerTokenLinearMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([777.0], device='cpu')

        layer = make_fake_layer(use_int8_weight=True, has_bias=False)
        weights = method.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 6)

        self.assertEqual(method.get_linear_descs(layer), LinearTypeV2.W8A8_DYNAMIC)
        method.process_weights_after_loading(layer)

        LinearMethodSupportAtbGraph._soc_info = None

    @patch.dict("sys.modules", {"torch_npu": MockTorchNpu()})
    @patch("atb_llm.utils.data.quant_method_adapter.ENV", MockENV())
    def test_w8a8_mix_linear_method(self):
        LinearMethodSupportAtbGraph.set_soc_info(MockNPUSocInfo())

        prefill_adaptee = Mock()
        decode_adaptee = Mock()
        adaptee = Mock()
        adaptee.quant_method = {
            InferenceMode.PREFILL: prefill_adaptee,
            InferenceMode.DECODE: decode_adaptee,
        }

        with patch("atb_llm.utils.data.quant_method_adapter.W8A8PerTokenLinearMethod") as mock_prefill, \
             patch("atb_llm.utils.data.quant_method_adapter.W8A8PerTensorLinearMethod") as mock_decode:
            mock_prefill_instance = Mock()
            mock_prefill_instance.get_weights_for_atb_graph.return_value = ["prefill"]
            mock_prefill.return_value = mock_prefill_instance

            mock_decode_instance = Mock()
            mock_decode_instance.get_weights_for_atb_graph.return_value = ["decode"]
            mock_decode.return_value = mock_decode_instance

            method = W8A8MixLinearMethod(adaptee)

        layer = Mock()
        layer.weight = Mock()
        layer.weight.data = torch.randint(0, 256, (64, 128), dtype=torch.uint8, device='cpu')
        layer.weight_scale = Mock()
        layer.weight_scale.data = torch.tensor([0.5, 0.6], device='cpu')
        layer.weight_offset = Mock()
        layer.weight_offset.data = torch.tensor([10.0, 12.0], device='cpu')

        weights = method.get_weights_for_atb_graph(layer, quant_type=QuantType.W8A8_DYNAMIC)
        self.assertEqual(weights, ["prefill"])

        weights = method.get_weights_for_atb_graph(layer, quant_type=QuantType.W8A8)
        self.assertEqual(weights, ["decode"])

        self.assertEqual(method.get_linear_descs(layer), LinearTypeV2.W8A8_PDMIX)
        method.process_weights_after_loading(layer)

        LinearMethodSupportAtbGraph._soc_info = None

    def test_unquantized_embedding_method(self):
        adaptee = Mock()
        method = UnquantizedEmbeddingMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([666.0], device='cpu')

        layer = Mock()
        layer.weight.data = torch.randn(100, 64, device='cpu')

        weights = method.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 1)
        self.assertTrue(torch.allclose(weights[0], layer.weight.data))

        with self.assertRaises(NotImplementedError) as cm:
            method.get_weights_for_atb_graph(layer, padding=False)
        self.assertIn("cannot be set to False", str(cm.exception))

    def test_unquantized_norm_method(self):
        adaptee = Mock()
        method = UnquantizedNormMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([555.0], device='cpu')

        layer = Mock()
        layer.weight.data = torch.randn(64, device='cpu')

        weights = method.get_weights_for_atb_graph(layer, padding=False)
        self.assertEqual(len(weights), 1)
        self.assertTrue(torch.allclose(weights[0], layer.weight.data))

        weights = method.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 4)
        self.assertTrue(torch.equal(weights[0], layer.weight.data))

        self.assertTrue(torch.all(weights[1] == 0).item())
        self.assertEqual(weights[1].shape, torch.Size([64]))

        for w in weights[2:]:
            self.assertEqual(w.item(), 555.0)

    def test_anti_outlier_norm_method(self):
        adaptee = Mock()
        method = AntiOutlierNormMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([444.0], device='cpu')

        layer = Mock()
        layer.weight.data = torch.randn(64, device='cpu')
        layer.bias = Mock()
        layer.bias.data = torch.randn(64, device='cpu')

        weights = method.get_weights_for_atb_graph(layer, padding=False)
        self.assertEqual(len(weights), 2)

        weights = method.get_weights_for_atb_graph(layer, padding=True)
        self.assertEqual(len(weights), 4)
        self.assertTrue(torch.allclose(weights[0], layer.weight.data))
        self.assertTrue(torch.allclose(weights[1], layer.bias.data))
        for w in weights[2:]:
            self.assertEqual(w.item(), 444.0)

        layer_zero_bias = Mock()
        layer_zero_bias.weight.data = torch.randn(64, device='cpu')
        layer_zero_bias.bias = Mock()
        layer_zero_bias.bias.data = torch.zeros(64, device='cpu')
        weights = method.get_weights_for_atb_graph(layer_zero_bias, padding=False)
        self.assertEqual(len(weights), 1)

    @patch.dict("sys.modules", {"torch_npu": MockTorchNpu()})
    @patch("atb_llm.utils.data.quant_method_adapter.ENV", MockENV())
    def test_unquantized_linear_method_full_coverage(self):
        """Cover missing branches in UnquantizedLinearMethod"""
        LinearMethodSupportAtbGraph.set_soc_info(MockNPUSocInfo())

        adaptee = Mock()
        method = UnquantizedLinearMethod(adaptee)
        method._PLACEHOLDER = torch.tensor([1.0], device='cpu')

        # 1. bias=None + padding=True
        layer_no_bias = Mock()
        layer_no_bias.weight.data = torch.randn(128, 64, dtype=torch.float16, device='cpu')
        layer_no_bias.bias = None
        weights = method.get_weights_for_atb_graph(layer_no_bias, padding=True)
        self.assertEqual(len(weights), 6)
        self.assertEqual(weights[1].item(), 1.0)

        # 2. process with NOT_TRANSPOSE
        with patch.object(method, '_check_transpose', return_value=TransposeType.NOT_TRANSPOSE):
            layer = Mock()
            layer.weight.data = torch.randn(128, 64, device='cpu')
            original = layer.weight.data.clone()
            method.process_weights_after_loading(layer)
            self.assertEqual(layer.weight.data.shape, (64, 128))
            self.assertTrue(torch.allclose(layer.weight.data, original.t()))

        # 3. get_weight_transpose_type after update
        layer = Mock()
        layer.weight.data = torch.randn(128, 64, device='cpu')
        method._weight_transpose_type = TransposeType.TRANSPOSE
        self.assertEqual(method.get_weight_transpose_type(layer), TransposeType.TRANSPOSE)

        with patch.object(method, '_check_transpose', return_value=TransposeType.NOT_TRANSPOSE):
            method.process_weights_after_loading(layer)
            self.assertEqual(method.get_weight_transpose_type(layer), TransposeType.NOT_TRANSPOSE)

        LinearMethodSupportAtbGraph._soc_info = None


if __name__ == '__main__':
    unittest.main()
