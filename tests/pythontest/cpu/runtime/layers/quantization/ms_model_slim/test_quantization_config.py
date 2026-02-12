
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
import torch

from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
from mindie_llm.runtime.layers.quantization.ms_model_slim.w8a8 import (
    W8A8PerTensorLinearMethod,
    W8A8PerTokenLinearMethod,
    W8A8MixLinearMethod,
)
from mindie_llm.runtime.layers.quantization.ms_model_slim.anti_outlier import AntiOutlierNormMethod
from mindie_llm.runtime.layers.quantization.unquantized import (
    UnquantizedLinearMethod,
    UnquantizedEmbeddingMethod,
    UnquantizedNormMethod,
)
from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear, LinearBase
from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead
from mindie_llm.runtime.layers.normalization import RMSNorm
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager, set_parallel_info_manager


class TestQuantizationConfig(unittest.TestCase):
    """Test cases for QuantizationConfig."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,
            "layer1.weight": QuantType.W8A8,
            "layer1.bias": QuantType.W8A8,
            "layer2.weight": QuantType.W8A8_DYNAMIC,
            "norm.weight": QuantType.FLOAT,
            "norm.bias": QuantType.W8A8,
        }

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def test_init(self):
        """Test initialization."""
        config = QuantizationConfig(self.base_config)

        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.model_quant_type, QuantType.W8A8)
        self.assertEqual(config.quant_descs, self.base_config)

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        minimal_config = {
            "layer1.weight": QuantType.W8A8,
        }

        config = QuantizationConfig(minimal_config)

        self.assertEqual(config.version, "0.0.0")
        self.assertIsNone(config.model_quant_type)
        self.assertEqual(config.quant_descs, minimal_config)

    def test_get_config_filenames(self):
        """Test get_config_filenames static method."""
        filenames = QuantizationConfig.get_config_filenames()

        self.assertEqual(filenames, ["quant_model_description.json"])

    def test_from_config(self):
        """Test from_config class method."""
        config = QuantizationConfig.from_config(self.base_config)

        self.assertIsInstance(config, QuantizationConfig)
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.model_quant_type, QuantType.W8A8)


class TestGetQuantTypeByWeightName(unittest.TestCase):
    """Test cases for get_quant_type_by_weight_name method."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,
            "layer1.weight": QuantType.W8A8,
            "layer1.bias": QuantType.W8A8,
            "layer2.weight": QuantType.W8A8_DYNAMIC,
            "qkv.weight": QuantType.W8A8,
            "k.weight": QuantType.W8A8,
            "v.weight": QuantType.W8A8,
            "gate.weight": QuantType.W8A8,
            "up.weight": QuantType.W8A8,
        })

    def test_get_quant_type_by_weight_name_string_prefix(self):
        """Test get_quant_type_by_weight_name with string prefix."""
        quant_type = self.config.get_quant_type_by_weight_name("layer1", "weight")

        self.assertEqual(quant_type, QuantType.W8A8)

    def test_get_quant_type_by_weight_name_string_prefix_bias(self):
        """Test get_quant_type_by_weight_name with string prefix for bias."""
        quant_type = self.config.get_quant_type_by_weight_name("layer1", "bias")

        self.assertEqual(quant_type, QuantType.W8A8)

    def test_get_quant_type_by_weight_name_list_prefix_same_types(self):
        """Test get_quant_type_by_weight_name with list prefix (same types)."""
        quant_type = self.config.get_quant_type_by_weight_name(["qkv", "k", "v"], "weight")

        self.assertEqual(quant_type, QuantType.W8A8)

    def test_get_quant_type_by_weight_name_list_prefix_gateup(self):
        """Test get_quant_type_by_weight_name with list prefix for gate/up."""
        quant_type = self.config.get_quant_type_by_weight_name(["gate", "up"], "weight")

        self.assertEqual(quant_type, QuantType.W8A8)

    def test_get_quant_type_by_weight_name_error_missing_key(self):
        """Test get_quant_type_by_weight_name error when key is missing."""
        with self.assertRaises(ValueError) as context:
            self.config.get_quant_type_by_weight_name("nonexistent", "weight")

        self.assertIn("not found", str(context.exception))
        self.assertIn("quantization descriptions", str(context.exception))

    def test_get_quant_type_by_weight_name_error_empty_string(self):
        """Test get_quant_type_by_weight_name error when quant_type is empty string."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "layer1.weight": "",  # Empty string
        })

        with self.assertRaises(ValueError) as context:
            config.get_quant_type_by_weight_name("layer1", "weight")

        self.assertIn("not found", str(context.exception))

    def test_get_quant_type_by_weight_name_error_inconsistent_types(self):
        """Test get_quant_type_by_weight_name error when types are inconsistent."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "layer1.weight": QuantType.W8A8,
            "layer2.weight": QuantType.W8A8_DYNAMIC,
        })

        with self.assertRaises(ValueError) as context:
            config.get_quant_type_by_weight_name(["layer1", "layer2"], "weight")

        self.assertIn("multiple quantization types", str(context.exception))
        self.assertIn("W8A8", str(context.exception))
        self.assertIn("W8A8_DYNAMIC", str(context.exception))

    def test_get_quant_type_by_weight_name_list_prefix_missing_one(self):
        """Test get_quant_type_by_weight_name error when one key in list is missing."""
        with self.assertRaises(ValueError) as context:
            self.config.get_quant_type_by_weight_name(["qkv", "nonexistent"], "weight")

        self.assertIn("not found", str(context.exception))


class TestGetQuantMethod(unittest.TestCase):
    """Test cases for get_quant_method method."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_w8a8 = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,
            "layer.weight": QuantType.W8A8,
            "embed_tokens.weight": QuantType.W8A8,
            "lm_head.weight": QuantType.W8A8,
        })

        self.config_w8a8_dynamic = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8_DYNAMIC,
            "layer.weight": QuantType.W8A8_DYNAMIC,
        })

        self.config_w8a8_mix = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8_MIX,
            "layer.weight": QuantType.W8A8_MIX,
        })

        self.config_float = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.FLOAT,
            "layer.weight": QuantType.FLOAT,
        })

    def mock_set_parallel_info_manager(self):
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 1
        self.mock_parallel_info.world_size = 1
        
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def test_get_quant_method_rmsnorm_without_bias(self):
        """Test get_quant_method for RMSNorm without bias."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "norm.weight": QuantType.W8A8,
        })

        self.mock_set_parallel_info_manager()

        layer = RMSNorm(hidden_size=512, prefix="norm")
        quant_method = config.get_quant_method(layer, prefix="norm")

        self.assertIsInstance(quant_method, UnquantizedNormMethod)

    def test_get_quant_method_rmsnorm_with_bias(self):
        """Test get_quant_method for RMSNorm with bias."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "norm.weight": QuantType.W8A8,
            "norm.bias": QuantType.W8A8,
        })

        self.mock_set_parallel_info_manager()
        layer = RMSNorm(hidden_size=512, prefix="norm")
        quant_method = config.get_quant_method(layer, prefix="norm")

        self.assertIsInstance(quant_method, AntiOutlierNormMethod)

    def test_get_quant_method_linear_w8a8(self):
        """Test get_quant_method for LinearBase with W8A8."""
        self.mock_set_parallel_info_manager()
        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = self.config_w8a8.get_quant_method(layer, prefix="layer")

        self.assertIsInstance(quant_method, W8A8PerTensorLinearMethod)

    def test_get_quant_method_linear_w8a8_dynamic(self):
        """Test get_quant_method for LinearBase with W8A8_DYNAMIC."""
        self.mock_set_parallel_info_manager()
        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = self.config_w8a8_dynamic.get_quant_method(layer, prefix="layer")

        self.assertIsInstance(quant_method, W8A8PerTokenLinearMethod)

    def test_get_quant_method_linear_w8a8_mix(self):
        """Test get_quant_method for LinearBase with W8A8_MIX."""
        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = self.config_w8a8_mix.get_quant_method(layer, prefix="layer")

        self.assertIsInstance(quant_method, W8A8MixLinearMethod)

    def test_get_quant_method_linear_float(self):
        """Test get_quant_method for LinearBase with FLOAT."""
        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = self.config_float.get_quant_method(layer, prefix="layer")

        self.assertIsInstance(quant_method, UnquantizedLinearMethod)

    def test_get_quant_method_linear_w8a8_mix_with_different_model_type(self):
        """Test get_quant_method for LinearBase with W8A8_MIX but different model_quant_type."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,  # Different from layer type
            "layer.weight": QuantType.W8A8_MIX,
        })

        # Create mock parallel info
        self.mock_set_parallel_info_manager()      

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = config.get_quant_method(layer, prefix="layer")

        # Should use model_quant_type instead of layer type
        self.assertIsInstance(quant_method, W8A8PerTensorLinearMethod)

    def test_get_quant_method_linear_w8a8_mix_with_same_model_type(self):
        """Test get_quant_method for LinearBase with W8A8_MIX and same model_quant_type."""
        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = self.config_w8a8_mix.get_quant_method(layer, prefix="layer")

        # Should use W8A8_MIX when model_quant_type matches
        self.assertIsInstance(quant_method, W8A8MixLinearMethod)

    
    @patch('mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type.QuantType')
    @patch('mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config.logger')
    def test_get_quant_method_linear_unsupported_type(self, mock_logger, mock_quant_type):
        """Test get_quant_method for LinearBase with unsupported quant type."""
        mock_quant_type.W8A8_MIX = MagicMock(spec=QuantType, value="W8A8_MIX")
        mock_quant_type.W8A8 = MagicMock(spec=QuantType, value="W8A8")

        config = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": "INVALID_TYPE",  # Unsupported
            "layer.weight": "INVALID_TYPE",
        })

        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")
        quant_method = config.get_quant_method(layer, prefix="layer")

        # Should fall back to UnquantizedLinearMethod
        self.assertIsInstance(quant_method, UnquantizedLinearMethod)
        # Should log a warning
        mock_logger.warning.assert_called_once()
        warning_message = str(mock_logger.warning.call_args)
        self.assertIn("not found", warning_message)
        self.assertIn("UnquantizedLinearMethod", warning_message)

    def test_get_quant_method_parallel_lm_head(self):
        """Test get_quant_method for ParallelLMHead."""
        # Set up mock parallel info manager
        mock_parallel_info_manager = MagicMock()
        mock_parallel_info_manager.rank = 0
        mock_parallel_info_manager.world_size = 1
        mock_parallel_info_manager.word_embed_tp = MagicMock()
        mock_parallel_info_manager.word_embed_tp.rank = 0
        mock_parallel_info_manager.word_embed_tp.group_size = 1
        mock_parallel_info_manager.lm_head_tp = MagicMock()
        mock_parallel_info_manager.lm_head_tp.rank = 0
        mock_parallel_info_manager.lm_head_tp.group_size = 1
        set_parallel_info_manager(mock_parallel_info_manager)

        try:
            layer = ParallelLMHead(num_embeddings=1000, embedding_dim=512, prefix="lm_head")
            quant_method = self.config_w8a8.get_quant_method(layer, prefix="lm_head")

            # Should always return UnquantizedLinearMethod for ParallelLMHead
            self.assertIsInstance(quant_method, UnquantizedLinearMethod)
        finally:
            set_parallel_info_manager(None)

    def test_get_quant_method_vocab_parallel_embedding(self):
        """Test get_quant_method for VocabParallelEmbedding."""
        # Set up mock parallel info manager
        mock_parallel_info_manager = MagicMock()
        mock_parallel_info_manager.rank = 0
        mock_parallel_info_manager.world_size = 1
        mock_parallel_info_manager.word_embed_tp = MagicMock()
        mock_parallel_info_manager.word_embed_tp.rank = 0
        mock_parallel_info_manager.word_embed_tp.group_size = 1
        mock_parallel_info_manager.lm_head_tp = MagicMock()
        mock_parallel_info_manager.lm_head_tp.rank = 0
        mock_parallel_info_manager.lm_head_tp.group_size = 1
        set_parallel_info_manager(mock_parallel_info_manager)

        try:
            layer = VocabParallelEmbedding(num_embeddings=1000, embedding_dim=512, prefix="embed_tokens")
            quant_method = self.config_w8a8.get_quant_method(layer, prefix="embed_tokens")

            # Should always return UnquantizedEmbeddingMethod for VocabParallelEmbedding
            self.assertIsInstance(quant_method, UnquantizedEmbeddingMethod)
        finally:
            set_parallel_info_manager(None)

    def test_get_quant_method_unknown_layer(self):
        """Test get_quant_method for unknown layer type."""
        # Create a mock layer that doesn't match any known types
        mock_layer = MagicMock(spec=torch.nn.Module)

        quant_method = self.config_w8a8.get_quant_method(mock_layer, prefix="layer")

        # Should return None for unknown layer types
        self.assertIsNone(quant_method)

    def test_get_quant_method_linear_with_list_prefix(self):
        """Test get_quant_method for LinearBase with list prefix."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,
            "qkv.weight": QuantType.W8A8,
            "k.weight": QuantType.W8A8,
            "v.weight": QuantType.W8A8,
        })

        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix=["qkv", "k", "v"])
        quant_method = config.get_quant_method(layer, prefix=["qkv", "k", "v"])

        self.assertIsInstance(quant_method, W8A8PerTensorLinearMethod)

    def test_get_quant_method_error_missing_weight_key(self):
        """Test get_quant_method error when weight key is missing."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "model_quant_type": QuantType.W8A8,
            # Missing "layer.weight"
        })

        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        layer = ReplicatedLinear(input_size=512, output_size=1024, prefix="layer")

        with self.assertRaises(ValueError) as context:
            config.get_quant_method(layer, prefix="layer")

        self.assertIn("not found", str(context.exception))

    def test_get_quant_method_rmsnorm_with_bias_key_but_no_bias_in_layer(self):
        """Test get_quant_method for RMSNorm when bias key exists but layer has no bias."""
        config = QuantizationConfig({
            "version": "1.0.0",
            "norm.weight": QuantType.W8A8,
            "norm.bias": QuantType.W8A8,
        })

        # Create mock parallel info
        self.mock_set_parallel_info_manager()

        # RMSNorm doesn't have bias by default, but config has bias key
        layer = RMSNorm(hidden_size=512, prefix="norm")
        quant_method = config.get_quant_method(layer, prefix="norm")

        # Should return AntiOutlierNormMethod because bias key exists in config
        self.assertIsInstance(quant_method, AntiOutlierNormMethod)


if __name__ == '__main__':
    unittest.main()
