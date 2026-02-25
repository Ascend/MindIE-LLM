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

from atb_llm.models.base.graph_manager.layerwise_prefill_graph_wrapper import LayerwisePrefillGraphWrapper, \
    LayerwiseEdgePrefillGraphWrapper, LayerwiseCloudPrefillGraphWrapper
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM, LayerWiseAttr, LwdLayerStatus, DistributedType
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses



class TestLayerwisePrefillGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.layerwise = LayerWiseAttr(edge_start_layer_count=1, edge_end_layer_count=1, split_type=DistributedType.CLOUD)
        self.graph_wrapper = LayerwisePrefillGraphWrapper(self.layerwise)
        self.model_type = "test_class"

    def test_active(self):
        context = MagicMock()
        runtime_params = MagicMock()
        self.assertTrue(self.graph_wrapper.activate(context, runtime_params, is_prefill=True))
        self.assertFalse(self.graph_wrapper.activate(context, runtime_params, is_prefill=False))


class TestLayerwiseEdgePrefillGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.layerwise = LayerWiseAttr(edge_start_layer_count=1, edge_end_layer_count=1, split_type=DistributedType.CLOUD)
        self.graph_wrapper = LayerwiseEdgePrefillGraphWrapper(self.layerwise)
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

  
class TestLayerwiseCloudPrefillGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_torch_classes = MockTorchClasses()
        torch.classes = self.mock_torch_classes
        self.layerwise = LayerWiseAttr(edge_start_layer_count=1, edge_end_layer_count=1, split_type=DistributedType.CLOUD)
        self.layerwise.num_hidden_layers = 4
        self.config = Qwen2Config(
        )
        self.graph_wrapper = LayerwiseCloudPrefillGraphWrapper(self.config, self.layerwise)
        self.model_type = "test_class"

    @patch('torch.classes.ModelTorch.ModelTorch')
    def test_set_param(self, mock_model_torch):

        mock_model_torch.side_effect = lambda *args, **kwargs: MagicMock()

        test_params = {
            'layers': [
                {'test1': {}},
                {'test2': {}}
            ],
        }

        self.graph_wrapper.set_param(self.model_type, test_params)
        self.assertEqual(len(self.graph_wrapper.graph_list), 2)


    def test_set_param(self):
        test_params = {
            'layers': [
                {'test1': {}}
            ],
        }

        self.graph_wrapper.graph_list = [MagicMock()]
        self.graph_wrapper.set_weight(test_params)
        self.assertEqual(len(self.graph_wrapper.graph_list), 1)

    
    def test_set_kv_cache(self):
        k_params = {
            'layers': [
                {'test1': {}}
            ],
        }

        v_params = {
            'layers': [
                {'test1': {}}
            ],
        }

        self.graph_wrapper.graph_list = [MagicMock()]
        self.graph_wrapper.set_kv_cache(k_params, v_params)
        self.assertEqual(len(self.graph_wrapper.graph_list), 1)

    def test_execute(self):
        
        inputs = [10]
        runtime_params = MagicMock()

        mock_graph1 = MagicMock()
        mock_graph1.execute.return_value = [0]

        mock_graph2 = MagicMock()
        mock_graph2.execute.return_value = [0]

        self.graph_wrapper.graph_list = [mock_graph1, mock_graph2]
        self.graph_wrapper.execute(inputs, runtime_params)

        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.start_exec_layer = 0
        layerwise_disaggregated_exe_stage.end_exec_layer = 2
        self.graph_wrapper.execute(inputs, runtime_params,
                                layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage
        )
