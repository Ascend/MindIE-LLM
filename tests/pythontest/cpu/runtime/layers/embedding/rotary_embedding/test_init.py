# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, MagicMock
from dataclasses import asdict
import torch
from mindie_llm.runtime.config.huggingface_config import RopeScaling
from mindie_llm.runtime.layers.embedding.rotary_embedding import get_rope
from mindie_llm.runtime.layers.embedding.rotary_embedding.base import RotaryEmbedding
from mindie_llm.runtime.layers.embedding.rotary_embedding.yarn_scaling_rope import (
    YarnScalingRotaryEmbedding,
)
from mindie_llm.runtime.layers.embedding.rotary_embedding.deepseek_v3_yarn_scaling_rope import (
    DeepseekV3YarnRotaryEmbedding
)
from mindie_llm.runtime.layers.embedding.rotary_embedding.registry import clear_rope_cache


class TestGetRoPE(unittest.TestCase):

    def setUp(self):
        clear_rope_cache()

    def _create_config(self, **kwargs):
        defaults = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "original_max_position_embeddings": 2048,
            "partial_rotary_factor": 1.0,
        }
        defaults.update(kwargs)
        return RopeScaling(**defaults)

    def test_routes_to_default_factory(self):
        config = self._create_config(rope_type="default")
        
        with patch("mindie_llm.runtime.layers.embedding.rotary_embedding.registry._ROPE_REGISTRY") as mock_registry:
            mock_factory = MagicMock(return_value=RotaryEmbedding(64, 64, 128, 10000.0, True, torch.float16))
            mock_registry.get.return_value = mock_factory
            
            rope = get_rope(
                head_size=64,
                rotary_dim=64,
                max_position=128,
                rope_config=config,
                is_neox_style=True,
                dtype=torch.float16
            )
            
            mock_registry.get.assert_called_once_with("default")
            mock_factory.assert_called_once()
            self.assertIsInstance(rope, RotaryEmbedding)

    def test_routes_to_yarn_factory(self):
        config = self._create_config(
            rope_type="yarn",
            factor=2.0,
            beta_fast=32,
            beta_slow=1,
            extrapolation_factor=1.0,
            apply_yarn_scaling=True,
            truncate=True,
            mscale=1.0
        )
        
        with patch("mindie_llm.runtime.layers.embedding.rotary_embedding.registry._ROPE_REGISTRY") as mock_registry:
            mock_factory = MagicMock(return_value=YarnScalingRotaryEmbedding(64, 64, 128, 10000.0, True, torch.bfloat16))
            mock_registry.get.return_value = mock_factory
            
            rope = get_rope(
                head_size=64,
                rotary_dim=64,
                max_position=128,
                rope_config=config,
                is_neox_style=True,
                dtype=torch.bfloat16
            )
            
            mock_registry.get.assert_called_once_with("yarn")
            mock_factory.assert_called_once()
            self.assertIsInstance(rope, YarnScalingRotaryEmbedding)

    def test_routes_to_deepseek_yarn_factory(self):
        config = self._create_config(
            rope_type="deepseek_yarn",
            factor=8.0,
            beta_fast=64,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=0.0
        )
        
        with patch("mindie_llm.runtime.layers.embedding.rotary_embedding.registry._ROPE_REGISTRY") as mock_registry:
            mock_factory = MagicMock(return_value=DeepseekV3YarnRotaryEmbedding(128, 2048, 10000.0, is_neox_style=True, dtype=torch.float32))
            mock_registry.get.return_value = mock_factory
            
            rope = get_rope(
                head_size=128,
                rotary_dim=128,
                max_position=2048,
                rope_config=config,
                is_neox_style=True,
                dtype=torch.float32
            )
            
            mock_registry.get.assert_called_once_with("deepseek_yarn")
            mock_factory.assert_called_once()
            self.assertIsInstance(rope, DeepseekV3YarnRotaryEmbedding)

    def test_applies_partial_rotary_factor_before_factory_call(self):
        config = self._create_config(rope_type="default", partial_rotary_factor=0.5)
        
        with patch("mindie_llm.runtime.layers.embedding.rotary_embedding.registry._ROPE_REGISTRY") as mock_registry:
            mock_factory = MagicMock(return_value=RotaryEmbedding(64, 32, 128, 10000.0, True, torch.float16))
            mock_registry.get.return_value = mock_factory
            
            rope = get_rope(
                head_size=64,
                rotary_dim=64,  # Original rotary_dim
                max_position=128,
                rope_config=config,
                is_neox_style=True,
                dtype=torch.float16
            )
            
            # Verify factory was called with scaled rotary_dim (64 * 0.5 = 32)
            call_args = mock_factory.call_args[0]
            called_rotary_dim = call_args[1]  # Second positional arg is rotary_dim
            self.assertEqual(called_rotary_dim, 32)
            self.assertEqual(rope.rotary_dim, 32)

    def test_cache_reuse_for_identical_configurations(self):
        config = self._create_config(rope_type="yarn", factor=2.0, beta_fast=32, beta_slow=1, mscale=1.0)
        
        # First call creates instance
        rope1 = get_rope(64, 64, 128, config, dtype=torch.float16)
        
        # Second call with identical params should return same instance
        rope2 = get_rope(64, 64, 128, config, dtype=torch.float16)
        
        self.assertIs(rope1, rope2)

    def test_raises_value_error_for_unknown_rope_type(self):
        config = self._create_config(rope_type="unknown_type")
        
        with self.assertRaises(ValueError) as cm:
            get_rope(
                head_size=64,
                rotary_dim=64,
                max_position=128,
                rope_config=config,
                dtype=torch.float16
            )
        
        self.assertIn("unknown_type", str(cm.exception))
        self.assertIn("default", str(cm.exception))  # Should list available types

    def test_factory_not_called_for_cached_instances(self):
        config = self._create_config(rope_type="default")
        
        with patch("mindie_llm.runtime.layers.embedding.rotary_embedding.registry._ROPE_REGISTRY") as mock_registry:
            mock_factory = MagicMock(return_value=RotaryEmbedding(64, 64, 128, 10000.0, True, torch.float16))
            mock_registry.get.return_value = mock_factory
            
            rope1 = get_rope(64, 64, 128, config, dtype=torch.float16)
            self.assertEqual(mock_factory.call_count, 1)


if __name__ == '__main__':
    unittest.main()