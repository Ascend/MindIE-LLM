# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import unittest
from unittest.mock import patch, MagicMock, call

import torch

from atb_llm.models.base.graph_manager.layerwise_decode_graph_wrapper import LayerwiseDecodeGraphWrapper, \
    LayerwiseEdgeDecodeGraphWrapper, LayerwiseCloudDecodeGraphWrapper
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM, LayerWiseAttr, LwdLayerStatus, DistributedType
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses



class TestLayerwiseDecodeGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.layerwise = LayerWiseAttr(edge_start_layer_count=1, edge_end_layer_count=1, split_type=DistributedType.CLOUD)
        self.graph_wrapper = LayerwiseDecodeGraphWrapper(self.layerwise)
        self.model_type = "test_class"

    def test_active(self):
        context = MagicMock()
        runtime_params = MagicMock()
        self.assertFalse(self.graph_wrapper.activate(context, runtime_params, is_prefill=True))
        self.assertTrue(self.graph_wrapper.activate(context, runtime_params, is_prefill=False))



class TestLayerwiseEdgeDecodeGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.layerwise = LayerWiseAttr(edge_start_layer_count=1, edge_end_layer_count=1, split_type=DistributedType.CLOUD)
        self.graph_wrapper = LayerwiseEdgeDecodeGraphWrapper(self.layerwise)
        self.model_type = "test_class"

    @patch('torch.classes.ModelTorch.ModelTorch')
    def test_set_param(self, mock_model_torch):
        mock_model_torch.side_effect = lambda *args, **kwargs: MagicMock()

        test_params = {
            'head': {},
            'tail': {}
        }

        self.graph_wrapper.set_param(self.model_type, test_params)
        self.graph_wrapper.head_graph.set_param.assert_called_once_with(
                        json.dumps({**test_params['head'], **self.graph_wrapper.feature_params})
                        )

        self.graph_wrapper.tail_graph.set_param.assert_called_once_with(
                        json.dumps({**test_params['tail'], **self.graph_wrapper.feature_params})
                        )

    @patch('torch.classes.ModelTorch.ModelTorch')
    def test_set_weight(self, mock_model_torch):
        mock_model_torch.side_effect = lambda *args, **kwargs: MagicMock()
        test_weights = {
            'head': {},
            'tail': {}
        }
        self.graph_wrapper.head_graph = MagicMock()
        self.graph_wrapper.tail_graph = MagicMock()
        self.graph_wrapper.set_weight(test_weights)
        self.graph_wrapper.head_graph.set_weight.assert_called_once_with(test_weights['head'])
        self.graph_wrapper.tail_graph.set_weight.assert_called_once_with(test_weights['tail'])

    @patch('torch.classes.ModelTorch.ModelTorch')
    def test_set_kv_cache(self, mock_model_torch):
        mock_model_torch.side_effect = lambda *args, **kwargs: MagicMock()
        test_k_caches = {
            'head': {},
            'tail': {}
        }

        test_v_caches = {
            'head': {},
            'tail': {}
        }

        self.graph_wrapper.head_graph = MagicMock()
        self.graph_wrapper.tail_graph = MagicMock()
        self.graph_wrapper.set_kv_cache(test_k_caches, test_v_caches)
        self.graph_wrapper.head_graph.set_kv_cache.assert_called_once_with(
            test_k_caches['head'], test_v_caches['head']
        )

        self.graph_wrapper.tail_graph.set_kv_cache.assert_called_once_with(
            test_k_caches['tail'], test_v_caches['tail']
        )

    @patch('torch.classes.ModelTorch.ModelTorch')
    def test_execute(self, mock_model_torch):
        mock_model_torch.side_effect = lambda *args, **kwargs: MagicMock()

        inputs = [10]
        runtime_params = MagicMock()

        self.graph_wrapper.head_graph = MagicMock()
        self.graph_wrapper.tail_graph = MagicMock()
        out = self.graph_wrapper.execute(inputs, runtime_params)
        self.graph_wrapper.head_graph.execute.assert_called_once_with(
            inputs, runtime_params
        )
        self.graph_wrapper.tail_graph.execute.assert_called_once_with(
            inputs, runtime_params
        )

        self.graph_wrapper.head_graph.reset_mock()
        self.graph_wrapper.tail_graph.reset_mock()
        
        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.start_exec_layer = 0
        out = self.graph_wrapper.execute(inputs, runtime_params, 
                layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage)
        
        self.graph_wrapper.head_graph.execute.assert_called_once_with(
            inputs, runtime_params
        )
        

        self.graph_wrapper.head_graph.reset_mock()
        self.graph_wrapper.tail_graph.reset_mock()

        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.end_exec_layer = 1
        out = self.graph_wrapper.execute(inputs, runtime_params, 
                layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage)
        self.graph_wrapper.tail_graph.execute.assert_called_once_with(
            inputs, runtime_params
        )