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
from unittest.mock import MagicMock

import torch

from atb_llm.models.base.graph_manager.graph_wrapper import CombinedATBGraphWrapper
from atb_llm.models.base.graph_manager.prefill_graph_wrapper import PrefillGraphWrapper
from atb_llm.models.base.graph_manager.decode_graph_wrapper import DecodeGraphWrapper
from atb_llm.models.base.graph_manager.dap_graph_wrapper import DapGraphWrapper
from atb_llm.models.base.graph_manager.flashcomm_graph_wrapper import FlashCommGraphWrapper
from atb_llm.models.base.graph_manager.single_lora_graph_wrapper import SingleLoraGraphWrapper
from atb_llm.models.base.graph_manager.multi_lora_graph_wrapper import MultiLoraGraphWrapper
from atb_llm.models.base.graph_manager.speculate_graph_wrapper import SpeculateGraphWrapper
from atb_llm.models.base.graph_manager.splitfuse_graph_wrapper import SplitFuseGraphWrapper
from atb_llm.models.base.graph_manager.mem_pool_graph_wrapper import MemPoolGraphWrapper
from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType
from tests.pythontest.atb_llm.models.base.graph_manager.test_graph_manager import MockATBGraphWrapper


class MockOperation:
    def __init__(self):
        pass

    def set_param(self, params):
        pass

    def set_weight(self, weights):
        pass

    def set_kv_cache(self, k_caches, v_caches):
        pass
    
    def execute(self, inputs, runtime_params):
        return 10


class TestCombinedATBGraphWrapper(unittest.TestCase):
    def setUp(self):
        self.graph_wrapper = CombinedATBGraphWrapper([MockATBGraphWrapper("A"), MockATBGraphWrapper("B")])
        self.model_type = "test_class"

        torch.classes = MagicMock()
        torch.classes.ModelTorch = MagicMock()
        torch.classes.ModelTorch.ModelTorch = MagicMock()
        torch.classes.ModelTorch.ModelTorch.return_value = MockOperation()

    def test_set_param(self):
        self.graph_wrapper.set_param(self.model_type, {})
        self.graph_wrapper.set_weight(1)
        self.graph_wrapper.set_kv_cache(3, 4)
        out = self.graph_wrapper.execute(None, {})
        self.assertEqual(out, 10)


class TestATBGraphWrapper(unittest.TestCase):
    def test_prefill(self):
        graph_wrapper = PrefillGraphWrapper()
        self.assertTrue(graph_wrapper.activate(None, {}, is_prefill=True))
        
    def test_decode(self):
        graph_wrapper = DecodeGraphWrapper()
        self.assertFalse(graph_wrapper.activate(None, {}, is_prefill=True))

    def test_dap(self):
        graph_wrapper = DapGraphWrapper()
        self.assertTrue(graph_wrapper.activate(None, {}, is_prefill=True, enable_dap=True))
        self.assertFalse(graph_wrapper.activate(None, {}, is_prefill=True, enable_dap=False))
    
    def test_flashcomm(self):
        graph_wrapper = FlashCommGraphWrapper()
        mock_context = MagicMock()
        mock_context.flash_comm_modifier = MagicMock()
        mock_context.flash_comm_modifier.active = True
        self.assertTrue(graph_wrapper.activate(mock_context, {}, is_prefill=True))
        mock_context.flash_comm_modifier.active = False
        self.assertFalse(graph_wrapper.activate(mock_context, {}, is_prefill=True))

    def test_single_lora(self):
        graph_wrapper = SingleLoraGraphWrapper()
        mock_context = MagicMock()
        mock_context.lora_modifier = MagicMock()
        mock_context.lora_modifier.use_single_adapter = MagicMock()
        mock_context.lora_modifier.use_single_adapter.return_value = True
        self.assertTrue(graph_wrapper.activate(mock_context, {}))

    def test_multi_lora(self):
        graph_wrapper = MultiLoraGraphWrapper()
        mock_context = MagicMock()
        mock_context.lora_modifier = MagicMock()
        mock_context.lora_modifier.use_multi_adapters = MagicMock()
        mock_context.lora_modifier.use_multi_adapters.return_value = True
        self.assertTrue(graph_wrapper.activate(mock_context, {}))

    def test_speculate(self):
        graph_wrapper = SpeculateGraphWrapper()
        mock_context = MagicMock()
        mock_context.inference_mode = MagicMock()
        mock_context.inference_mode.enable_decode_pa = True
        self.assertTrue(graph_wrapper.activate(mock_context, json.dumps({"qLen": 7}), is_prefill=False))
        self.assertFalse(graph_wrapper.activate(mock_context, json.dumps({"qLen": 8}), is_prefill=True))
        self.assertFalse(graph_wrapper.activate(mock_context, json.dumps({"seqLen": 9}), is_prefill=False))

        mock_context.inference_mode = None
        self.assertFalse(graph_wrapper.activate(mock_context, json.dumps({"qLen": 7}), is_prefill=False))

    def test_splitfuse(self):
        graph_wrapper = SplitFuseGraphWrapper()
        mock_context = MagicMock()
        mock_context.inference_mode = MagicMock()
        mock_context.inference_mode.enable_prefill_pa = True
        self.assertTrue(graph_wrapper.activate(mock_context, json.dumps({"qLen": 17}), is_prefill=True))
        self.assertFalse(graph_wrapper.activate(mock_context, json.dumps({"qLen": 18}), is_prefill=False))
        self.assertFalse(graph_wrapper.activate(mock_context, json.dumps({"seqLen": 19}), is_prefill=True))
        
        mock_context.inference_mode = None
        self.assertFalse(graph_wrapper.activate(mock_context, json.dumps({"qLen": 17}), is_prefill=True))

    def test_mem_pool(self):
        graph_wrapper = MemPoolGraphWrapper()
        mock_context = MagicMock()
        self.assertTrue(graph_wrapper.activate(mock_context, {}, mempool_type=MemPoolType.ASYNC_WRITE))
        self.assertFalse(graph_wrapper.activate(mock_context, {}, mempool_type=MemPoolType.DISABLED))
        self.assertFalse(graph_wrapper.activate(mock_context, {}, mempool_type=MemPoolType.SYNC_WRITE))