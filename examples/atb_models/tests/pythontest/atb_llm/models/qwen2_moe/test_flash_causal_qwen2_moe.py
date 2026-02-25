# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from atb_llm.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from atb_llm.models.qwen2_moe.flash_causal_qwen2_moe import FlashQwen2moeForCausalLM
from atb_llm.utils.mapping import Mapping
from atb_llm.utils.moe_utils import EPLBType, ExpertParallelDegree
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.quantize.quant_type import QuantType
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager
from tests.pythontest.atb_llm.models.base.mock_class import MockTorchClasses

LOAD_ATB_SPEED = "atb_llm.models.base.flash_causal_lm.load_atb_speed"
FLASH_QWEN2_MOE = "atb_llm.models.qwen2_moe.flash_causal_qwen2_moe"


class TestFlashQwen2moeForCausalLM(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_classes = MockTorchClasses()
        torch.classes = self.torch_classes

        self.config = Qwen2MoeConfig(
            vocab_size=1024,
            hidden_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            num_experts=8,
            num_experts_per_tok=2,
            rope_scaling={
                "type": "yarn",
                "factor": 2.0
            }
        )
        self.config.torch_dtype = torch.float16
        self.config.attention_bias = False
        self.config.is_dense_layer = False
        self.config.has_shared_expert = False
        self.config.use_qk_norm = False
        self.config.norm_topk_prob = False
        self.config.quantization_config = MagicMock()
        self.config.quantization_config.kv_quant_type = None

        self.weights = MagicMock()
        self.weights.device = torch.device("npu")
        self.weights.dtype = torch.float16
        self.weights.mapping = Mapping(world_size=2, rank=0)
        self.weights.mapping.attn_tp.rank = 1
        self.weights.mapping.rank_table_file = "valid_file"
        self.weights.mapping.moe_tp = MagicMock()
        self.weights.mapping.moe_tp.group_size = 1
        self.weights.mapping.attn_dp = MagicMock()
        self.weights.mapping.attn_dp.rank = 0
        self.weights.process_group = MagicMock()
        self.weights.process_group.rank.return_value = 0
        self.weights.process_group.size.return_value = 2
        self.weights.quant_desc = None
        self.weights.expert_routing_map = {0: [0, 1, 2, 3]}
        self.weights.switch_process_group = MagicMock()

        # Create mock llm_config
        self.llm_config = MagicMock()
        self.llm_config.models.qwen_moe.ep_level = ExpertParallelDegree.DYNAMIC_EP
        self.llm_config.models.qwen_moe.eplb = MagicMock()
        self.llm_config.models.qwen_moe.eplb.level = EPLBType.NO_EPLB
        self.llm_config.models.qwen_moe.enable_aclnn_rope = False

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
        self.mock_parallel_info_manager.moe_ep = self.mock_parallel_info

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_init(self, mock_load_column_multi, mock_qwen_model, _mock_init_so):
        """Test model initialization with various configurations."""
        # Basic initialization
        FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        mock_qwen_model.assert_called_once()

        # Test with different quantize
        self.config.quantize = "w8a8sc"
        FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        mock_load_column_multi.assert_called_with(
            self.config,
            prefixes=["lm_head"],
            weights=self.weights,
            head_size=1,
            lm_head=True
        )

        # Test with distributed_enable
        ins = FlashQwen2moeForCausalLM(
            self.config, self.weights,
            llm_config=self.llm_config,
            distributed_enable=True,
            max_batch_size=32
        )
        self.assertEqual(ins.max_batch_size, 32)
        self.assertTrue(ins.distributed_enable)

        # Test with attn_quantize
        self.config.attn_quantize = "w8a8"
        ins2 = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        self.assertEqual(ins2.attn_quantize, "w8a8")

        # Test with w8a8_dynamic quantize
        # Delete attn_quantize attribute instead of setting to None
        if hasattr(self.config, 'attn_quantize'):
            delattr(self.config, 'attn_quantize')
        self.config.quantize = "w8a8_dynamic"
        ins3 = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        self.assertEqual(ins3.attn_quantize, "w8a8")

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    @patch(f"{FLASH_QWEN2_MOE}.MoeWeightWrapper")
    def test_init_ascend_weight(self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test ascend weight initialization."""
        mock_weight_wrapper_ins = mock_weight_wrapper.return_value
        mock_weight_wrapper_ins.register_embedding = MagicMock()
        mock_weight_wrapper_ins.register_moe_layer = MagicMock()
        mock_weight_wrapper_ins.register_model_norm = MagicMock()
        mock_weight_wrapper_ins.register_model_lmhead = MagicMock()
        mock_weight_wrapper_ins.weights = []
        mock_weight_wrapper_ins.pack_quant_type = {}
        mock_weight_wrapper_ins.attn_linear_types = {}
        mock_weight_wrapper_ins.mlp_linear_types = {}
        mock_weight_wrapper_ins.moe_linear_types = {}
        mock_weight_wrapper_ins.attn_linear_transpose_types = {}
        mock_weight_wrapper_ins.mlp_linear_transpose_types = {}
        mock_weight_wrapper_ins.moe_linear_transpose_types = {}

        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ins.graph_manager = MagicMock()
        ins.graph_manager.set_param.return_value = True
        ins.init_ascend_weight()
        ins.graph_manager.set_param.assert_called()

        # Test with prefix_cache_enable
        ins.prefix_cache_enable = True
        ins.speculate_enable = False
        with patch.object(ins, 'get_weights', return_value=mock_weight_wrapper_ins):
            ins.graph_manager = MagicMock()
            ins.init_ascend_weight()
            self.assertTrue(ins.graph_manager.register_graph.called)

        # Test with speculate_enable
        ins.prefix_cache_enable = False
        ins.speculate_enable = True
        with patch.object(ins, 'get_weights', return_value=mock_weight_wrapper_ins):
            ins.graph_manager = MagicMock()
            ins.init_ascend_weight()
            self.assertTrue(ins.graph_manager.register_graph.called)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_init_eplb_config(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test EPLB configuration initialization."""
        config = Qwen2MoeConfig(
            vocab_size=1024,
            hidden_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            num_experts=8,
            num_experts_per_tok=2,
        )
        config.ep_level = ExpertParallelDegree.DYNAMIC_EP

        ins = FlashQwen2moeForCausalLM(config, self.weights, llm_config=self.llm_config)

        # Test NO_EPLB
        llm_config = MagicMock()
        llm_config.models.qwen_moe.eplb.level = EPLBType.NO_EPLB.value
        result = ins.init_eplb_config(llm_config, config, ep_level=ExpertParallelDegree.DYNAMIC_EP)
        self.assertEqual(ins.eplb_level, EPLBType.NO_EPLB)

        # Test invalid EPLB level
        llm_config.models.qwen_moe.eplb.level = 999
        with self.assertRaises(ValueError):
            ins.init_eplb_config(llm_config, config, ep_level=ExpertParallelDegree.DYNAMIC_EP)

        # Test STATIC_EPLB with valid file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump({"expert_map": [[0, 1], [2, 3]]}, f)
            temp_file = f.name

        try:
            llm_config.models.qwen_moe.eplb.level = EPLBType.STATIC_EPLB.value
            llm_config.models.qwen_moe.eplb.expert_map_file = temp_file
            with patch('atb_llm.models.qwen2_moe.flash_causal_qwen2_moe.calculate_eplb_param',
                    return_value=(True, 2, 1)):
                result = ins.init_eplb_config(llm_config, config, ep_level=ExpertParallelDegree.DYNAMIC_EP)
                self.assertEqual(ins.eplb_level, EPLBType.STATIC_EPLB)
                self.assertEqual(ins.num_redundant_experts, 1)
        finally:
            os.unlink(temp_file)

        # Test with None llm_config
        result = ins.init_eplb_config(None, config, ep_level=ExpertParallelDegree.NO_EP)
        self.assertEqual(ins.eplb_level, 0)
        self.assertIsNone(result)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_init_ep_level(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test expert parallel level initialization."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)

        # Test DYNAMIC_EP without rank table
        ins.mapping.rank_table_file = ""
        ins.ep = True
        with self.assertRaises(RuntimeError):
            ins._init_ep_level()

        # Test DYNAMIC_EP with invalid moe_tp
        ins.mapping.rank_table_file = "valid_file"
        ins.mapping.moe_tp.group_size = 2
        with self.assertRaises(ValueError):
            ins._init_ep_level()

        # Test NO_EP
        ins.ep = False
        ins._init_ep_level()
        self.assertEqual(ins.ep_level, ExpertParallelDegree.NO_EP)

        # Test DYNAMIC_EP with ep=True
        ins.ep = True
        ins.mapping.rank_table_file = "valid_file"
        ins.mapping.moe_tp.group_size = 1
        llm_config = MagicMock()
        llm_config.models.qwen_moe.ep_level = ExpertParallelDegree.DYNAMIC_EP
        ins.qwen_moe_config = llm_config.models.qwen_moe
        ins._init_ep_level()
        self.assertEqual(ins.ep_level, ExpertParallelDegree.DYNAMIC_EP)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_build_dep_inputs(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test dependency inputs building."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)

        with patch.object(ins.mapping, 'has_dp', return_value=False):
            input_ids = torch.tensor([1, 2, 3])

            # Test without DP
            dep_inputs = ins.build_dep_inputs(
                input_ids=input_ids,
                is_prefill=True,
                lm_head_indices=None
            )
            self.assertEqual(len(dep_inputs), 11)

            # Test with STATIC_EPLB
            ins.eplb_level = EPLBType.STATIC_EPLB
            ins.expert_routing_map = torch.tensor([[0, 1], [2, 3]])
            dep_inputs = ins.build_dep_inputs(
                input_ids=input_ids,
                is_prefill=True,
                lm_head_indices=None
            )
            self.assertEqual(len(dep_inputs), 12)

        # Test with DP enabled
        with patch(f"{FLASH_QWEN2_MOE}.ENV") as mock_env:
            mock_env.enable_dp_move_up = True
            mock_dep_inputs = [torch.tensor([1])] * 10
            mock_dep_inputs.append(torch.tensor([1, 2, 3]))
            with patch.object(ins.mapping, 'has_dp', return_value=True):
                with patch.object(ins.mapping, 'has_attn_tp', return_value=False):
                    ins.mapping.lm_head_tp.group_size = 1
                    ins.distributed_enable = False
                    dep_inputs = ins.build_dep_inputs(
                        input_ids=input_ids,
                        is_prefill=True,
                        lm_head_indices=None,
                        dep_inputs=mock_dep_inputs
                    )
                    self.assertIsNotNone(ins.expert_array)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_forward(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test forward pass."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)

        mock_mapping = MagicMock()
        mock_attn_dp = MagicMock()
        mock_attn_dp.group_size = 1
        mock_mapping.attn_dp = mock_attn_dp
        mock_mapping.has_dp.return_value = False
        mock_mapping.world_size = 1
        mock_mapping.rank = 0

        ins.mapping = mock_mapping

        with patch.object(ins, 'init_ascend_weight') as mock_init_weight:
            mock_init_weight.return_value = None

            ins.graph_manager = MagicMock()
            output_tensor = torch.zeros([3, 1000], dtype=torch.float16).npu()
            ins.graph_manager.select_and_execute.return_value = [output_tensor]

            with patch.object(ins, 'prepare_inputs_for_ascend') as mock_prepare_inputs:
                mock_prepare_inputs.return_value = (MagicMock(), MagicMock())

                input_ids = torch.tensor([1, 2, 3])
                position_ids = torch.tensor([0, 1, 2])
                kv_cache = [(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1))]
                block_tables = torch.tensor([[0]])
                slots = torch.tensor([0])
                input_lengths = torch.tensor([3])
                max_seq_len = 512

                output = ins.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    is_prefill=True,
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    input_lengths=input_lengths,
                    max_seq_len=max_seq_len
                )

                mock_init_weight.assert_called_once()
                mock_prepare_inputs.assert_called_once()
                ins.graph_manager.select_and_execute.assert_called_once()
                self.assertIsNotNone(output)

        # Test with distributed_enable
        ins2 = FlashQwen2moeForCausalLM(
            self.config, self.weights,
            llm_config=self.llm_config,
            distributed_enable=True
        )
        ins2.mapping = mock_mapping
        ins2.ascend_weight = MagicMock()
        ins2.graph_manager = MagicMock()
        ins2.graph_manager.select_and_execute.return_value = [output_tensor]

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_init_kvcache(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test KV cache initialization."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ins.graph_manager = MagicMock()
        ins.graph_manager.set_kv_cache = MagicMock()
        ins.soc_info.need_nz = False

        cache = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache, cache)])
        self.assertEqual(ins.ascend_kcache_id, id(cache))

        # Test with cache changes
        cache2 = torch.zeros([1, 128, 1, 128], dtype=torch.float16).npu()
        ins.init_kvcache([(cache2, cache2)])
        self.assertEqual(ins.ascend_kcache_id, id(cache2))

        # Test with need_nz=True
        ins.soc_info.need_nz = True
        ins.ascend_kcache_id = None
        ins.ascend_vcache_id = None
        with patch('atb_llm.models.qwen2_moe.flash_causal_qwen2_moe.torch_npu') as mock_torch_npu:
            mock_torch_npu.npu_format_cast_.side_effect = lambda x, y: x
            ins.init_kvcache([(cache, cache)])
            self.assertEqual(ins.ascend_kcache_id, id(cache))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_init_rope(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test RoPE initialization."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ins._init_rope()
        self.assertEqual(ins.rope_backend, OpBackend.ATB)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_init_yarn(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test YARN position embedding initialization."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)

        # Test with YARN scaling
        ins.config.rope_scaling_dict = {
            "type": "yarn",
            "factor": 2.0
        }
        ins._init_yarn()
        self.assertTrue(hasattr(ins, 'rotary_embedding'))

        # Test with invalid scaling type
        ins.config.rope_scaling_dict = {
            "type": "invalid_type"
        }
        with self.assertRaises(ValueError):
            ins._init_yarn()

        # Test with None
        ins.config.rope_scaling_dict = None
        ins._init_yarn()  # Should not raise error

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_update_matmul_params(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test matmul parameters update."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)

        # Test with A2 SOC
        with patch('atb_llm.models.qwen2_moe.flash_causal_qwen2_moe.A2_SOCS', (220,)):
            with patch.object(ins.soc_info, 'soc_version', 220):
                ins._update_matmul_params(quantize=QuantType.FLOAT)
                self.assertFalse(ins.matmul_nd_nz)

                ins._update_matmul_params(quantize=QuantType.W8A8)
                self.assertTrue(ins.matmul_nd_nz)

        # Test with A3 SOC
        with patch('atb_llm.models.qwen2_moe.flash_causal_qwen2_moe.A3_SOCS', (250,)):
            with patch.object(ins.soc_info, 'soc_version', 250):
                ins._update_matmul_params(quantize=QuantType.FLOAT)
                self.assertFalse(ins.matmul_nd_nz)

        # Test with non-A2/A3 SOC
        with patch.object(ins.soc_info, 'soc_version', 100):
            ins._update_matmul_params(quantize=QuantType.FLOAT)
            self.assertFalse(ins.matmul_nd_nz)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_select_logits(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test logits selection."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        logits = torch.randn(10, 1000)

        # Test without dp_logits_num
        output = ins.select_logits(logits)
        self.assertEqual(output.shape, logits.shape)

        # Test with dp_logits_num, rank 0
        ins.mapping.attn_dp.rank = 0
        output = ins.select_logits(logits, dp_logits_num=[5, 10])
        self.assertEqual(output.shape[0], 5)

        # Test with dp_logits_num, rank > 0
        ins.mapping.attn_dp.rank = 1
        output = ins.select_logits(logits, dp_logits_num=[5, 10])
        self.assertEqual(output.shape[0], 5)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_register_layer_weights(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test register_layer_weights with shared expert."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ins.config.has_shared_expert = True

        mock_layer = MagicMock()
        mock_layer.state_dict.return_value = {
            "mlp.shared_expert.gate_up_proj.linear.weight": torch.tensor([1.0]),
            "mlp.shared_expert.down_proj.linear.weight": torch.tensor([2.0]),
            "mlp.shared_expert_gate.weight": torch.tensor([3.0])
        }

        mock_weight_wrapper = MagicMock()
        mock_weight_wrapper.weights = []
        mock_weight_wrapper.soc_info = MagicMock()

        ins.register_layer_weights(mock_weight_wrapper, mock_layer)
        self.assertEqual(len(mock_weight_wrapper.weights), 3)
        mock_weight_wrapper.register_moe_layer.assert_called_once()

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    @patch(f"{FLASH_QWEN2_MOE}.MoeWeightWrapper")
    def test_get_weights(self, mock_weight_wrapper, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test get_weights method."""
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.register_embedding = MagicMock()
        mock_wrapper_instance.register_moe_layer = MagicMock()
        mock_wrapper_instance.register_model_norm = MagicMock()
        mock_wrapper_instance.register_model_lmhead = MagicMock()
        mock_weight_wrapper.return_value = mock_wrapper_instance

        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ins.model.embed_tokens = MagicMock()
        ins.model.layers = [MagicMock() for _ in range(2)]
        ins.model.norm = MagicMock()
        ins.lm_head = MagicMock()

        result = ins.get_weights()
        self.assertEqual(mock_wrapper_instance.register_embedding.call_count, 1)
        self.assertEqual(mock_wrapper_instance.register_moe_layer.call_count, 2)
        self.assertEqual(mock_wrapper_instance.register_model_norm.call_count, 1)
        self.assertEqual(mock_wrapper_instance.register_model_lmhead.call_count, 1)

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_prepare_inputs_for_ascend(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test prepare_inputs_for_ascend method."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)

        # Mock rotary_embedding 的方法
        with patch.object(ins.rotary_embedding, 'update_cos_sin_cache_total') as mock_update_cos_sin:
            mock_update_cos_sin.return_value = None
            
            # Mock attn_mask 的方法，而不是替换整个对象
            with patch.object(ins.attn_mask, 'get_rope_prefill_mask') as mock_get_prefill_mask:
                mock_get_prefill_mask.return_value = torch.tensor([[1.0]])
                
                with patch.object(ins.attn_mask, 'get_rope_decode_mask') as mock_get_decode_mask:
                    mock_get_decode_mask.return_value = torch.tensor([[1.0]])
                    
                    # Mock 其他属性和方法
                    ins.transdata_operation = MagicMock()
                    ins.transdata_operation.execute.return_value = [torch.tensor([[1.0]])]
                    ins.soc_info.need_nz = False
                    ins.qlen_modifier = MagicMock()
                    ins.qlen_modifier.modify_inputs = MagicMock()
                    
                    # Mock mapping 对象
                    mock_mapping = MagicMock()
                    mock_mapping.has_attn_tp.return_value = False
                    mock_mapping.has_dp.return_value = False
                    mock_mapping.world_size = 1
                    mock_mapping.rank = 0
                    ins.mapping = mock_mapping

                    with patch.object(ins, 'build_dep_inputs', return_value=[torch.tensor([1])] * 11):
                        input_ids = torch.tensor([1, 2, 3])
                        position_ids = torch.tensor([0, 1, 2])
                        kv_cache = [(torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1))]
                        block_tables = torch.tensor([[0]])
                        slots = torch.tensor([0])
                        input_lengths = torch.tensor([3])

                        # Test prefill
                        acl_inputs, acl_param = ins.prepare_inputs_for_ascend(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            is_prefill=True,
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            slots=slots,
                            input_lengths=input_lengths,
                            max_seq_len=512
                        )
                        self.assertIsNotNone(acl_inputs)
                        self.assertIsNotNone(acl_param)
                        
                        # 验证 prefill 调用了正确的方法
                        mock_get_prefill_mask.assert_called()
                        
                        # Test decode
                        # 重置 mock 调用计数
                        mock_get_prefill_mask.reset_mock()
                        mock_get_decode_mask.reset_mock()
                        
                        acl_inputs, acl_param = ins.prepare_inputs_for_ascend(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            is_prefill=False,
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            slots=slots,
                            input_lengths=input_lengths,
                            max_seq_len=512
                        )
                        self.assertIsNotNone(acl_inputs)
                        
                        # 验证 decode 调用了正确的方法
                        mock_get_decode_mask.assert_called()

                        # Test with attn_mask provided
                        attn_mask = torch.tensor([[1.0]])
                        acl_inputs, acl_param = ins.prepare_inputs_for_ascend(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            is_prefill=True,
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            slots=slots,
                            input_lengths=input_lengths,
                            max_seq_len=512,
                            attn_mask=attn_mask
                        )
                        self.assertIsNotNone(acl_inputs)

                        # Test with need_nz=True
                        ins.soc_info.need_nz = True
                        acl_inputs, acl_param = ins.prepare_inputs_for_ascend(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            is_prefill=True,
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            slots=slots,
                            input_lengths=input_lengths,
                            max_seq_len=512
                        )
                        ins.transdata_operation.execute.assert_called()

                        # Test with FORCE_EPLB
                        ins.eplb_level = EPLBType.FORCE_EPLB
                        ins.fake_topk = torch.tensor([[0, 1], [2, 3], [4, 5]])
                        ins.soc_info.need_nz = False
                        mock_dep_inputs = [torch.tensor([1])] * 11
                        mock_dep_inputs[1] = torch.tensor([1, 2, 3])
                        with patch.object(ins, 'build_dep_inputs', return_value=mock_dep_inputs):
                            acl_inputs, acl_param = ins.prepare_inputs_for_ascend(
                                input_ids=input_ids,
                                position_ids=position_ids,
                                is_prefill=True,
                                kv_cache=kv_cache,
                                block_tables=block_tables,
                                slots=slots,
                                input_lengths=input_lengths,
                                max_seq_len=512
                            )
                            self.assertGreater(len(acl_inputs), len(mock_dep_inputs))

    @patch(f"{LOAD_ATB_SPEED}")
    @patch(f"{FLASH_QWEN2_MOE}.FlashQwenModel", return_value=MagicMock())
    @patch(f"{FLASH_QWEN2_MOE}.load_column_multi")
    def test_execute_ascend_operator(self, _mock_load_column_multi, _mock_qwen_model, _mock_init_so):
        """Test execute_ascend_operator with IndexError."""
        ins = FlashQwen2moeForCausalLM(self.config, self.weights, llm_config=self.llm_config)
        ins.graph_manager = MagicMock()
        ins.graph_manager.select_and_execute.return_value = []

        with self.assertRaises(RuntimeError):
            ins.execute_ascend_operator([], "{}", True)


if __name__ == "__main__":
    unittest.main()
