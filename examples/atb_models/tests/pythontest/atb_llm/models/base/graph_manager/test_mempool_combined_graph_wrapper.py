# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch

import json
import torch

from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from atb_llm.models.qwen2.flash_causal_qwen2 import FlashQwen2ForCausalLM
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.quantize.pack_type import TransposeType

from mindie_llm.runtime.utils.distributed import set_parallel_info_manager
from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses


LOAD_ATB_SPEED = "atb_llm.models.base.flash_causal_lm.load_atb_speed"
FLASH_QWEN2 = "atb_llm.models.qwen2.flash_causal_qwen2"


class TestMempoolCombinedATBGraphWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_classes = MockTorchClasses()
        torch.classes = self.torch_classes

        self.config = Qwen2Config(
            model_type="qwen2",
            hidden_size=1024,
            max_position_embeddings=1024,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_hidden_layers=28,
            rms_norm_eps=1e-6,
            torch_dtype=torch.float16,
            vocab_size=125696,
            tie_word_embeddings=False,
        )

        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.mapping = Mapping(world_size=2, rank=0)
        self.weights.mapping.attn_tp.rank = 1
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 0
        self.weights.process_group.size.return_value = 2
        self.weights.quant_desc = None

        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.process_group = None

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.world_size = 2
        self.mock_parallel_info_manager.word_embed_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.attn_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.lm_head_tp = self.mock_parallel_info

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.Qwen2Model", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.TensorHead")
    def test_init(
            self,
            mock_tensor_head,
            mock_load_column_multi,
            mock_new_qwen_model,
            mock_qwen_model,
            _mock_init_so
    ) -> None:
        FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        mock_new_qwen_model.assert_called_once_with(
            self.config, "model", quant_config=None
        )

        self.config.quantize = "w8a8sc"
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_tensor_head.load_weight.assert_called_once_with(
            self.config,
            prefix="lm_head",
            weights=self.weights,
            is_norm=False,
        )

        self.config.quantize = ""
        self.config.tie_word_embeddings = True
        FlashQwen2ForCausalLM(self.config, self.weights)
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["model.embed_tokens"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2}.load_column_multi")
    @patch(f"{FLASH_QWEN2}.WeightWrapper")
    def test_init_ascend_weight(
        self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        mock_weight_wrapper_ins = mock_weight_wrapper.return_value
        mock_weight_wrapper_ins.register_embedding = MagicMock()
        mock_weight_wrapper_ins.register_layer = MagicMock()
        mock_weight_wrapper_ins.register_model_norm = MagicMock()
        mock_weight_wrapper_ins.register_model_lmhead = MagicMock()
        mock_weight_wrapper_ins.weights = []
        mock_weight_wrapper_ins.linear_type = {}
        mock_weight_wrapper_ins.pack_quant_type = {}
        mock_weight_wrapper_ins.linear_transpose_types = {}
        mock_weight_wrapper_ins.linear_descs = []

        # prefill_splitfuse
        ins = FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        ins.lm_head.get_weight_transpose_type()[0] = TransposeType.NOT_TRANSPOSE
        ins.prefix_cache_enable = True
        ins.mempool_type = MemPoolType.ASYNC_WRITE
        ins.init_ascend_weight()
        mock_context = MagicMock()
        mock_context.inference_mode = MagicMock()
        mock_context.inference_mode.enable_prefill_pa = True
        selected_graph = None
        for graph in ins.graph_manager._graph_list:
            if graph.activate(mock_context, json.dumps({"qLen": 9}), is_prefill=True, mempool_type=MemPoolType.ASYNC_WRITE):
                selected_graph = graph
                break
        self.assertEqual(selected_graph.feature_name, "prefill_splitfuse")

        # prefill_mempool
        ins = FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        ins.lm_head.get_weight_transpose_type()[0] = TransposeType.NOT_TRANSPOSE
        ins.mempool_type = MemPoolType.ASYNC_WRITE
        ins.init_ascend_weight()
        mock_context = MagicMock()
        selected_graph = None
        for graph in ins.graph_manager._graph_list:
            if graph.activate(mock_context, {}, is_prefill=True, mempool_type=MemPoolType.ASYNC_WRITE):
                selected_graph = graph
                break
        self.assertEqual(selected_graph.feature_name, "prefill_mempool")

        # base_prefill
        ins = FlashQwen2ForCausalLM(self.config, self.weights, prealloc_weight_mem_on_npu=True)
        ins.lm_head.get_weight_transpose_type()[0] = TransposeType.NOT_TRANSPOSE
        ins.mempool_type = MemPoolType.SYNC_WRITE
        ins.init_ascend_weight()
        mock_context = MagicMock()
        selected_graph = None
        for graph in ins.graph_manager._graph_list:
            if graph.activate(mock_context, {}, is_prefill=True, mempool_type=MemPoolType.SYNC_WRITE):
                selected_graph = graph
                break
        self.assertEqual(str(selected_graph.feature_name), "FeatureType.PREFILL")
