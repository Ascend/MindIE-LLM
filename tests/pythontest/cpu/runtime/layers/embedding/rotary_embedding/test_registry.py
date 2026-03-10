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
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass, asdict
from functools import wraps
from typing import Callable
import torch
from mindie_llm.runtime.config.huggingface_config import RopeScaling
from mindie_llm.runtime.layers.embedding.rotary_embedding.registry import (
    register_rope_type,
    unregister_rope_type,
    get_registered_rope_types,
    cached_rope_factory,
    clear_rope_cache,
    get_rope_factory,
    _ROPE_REGISTRY,
    _ROPE_DICT
)


class TestRegistry(unittest.TestCase):

    def setUp(self):
        # Clear registry and cache before each test
        _ROPE_REGISTRY.clear()
        _ROPE_DICT.clear()

    def tearDown(self):
        # Ensure cleanup after each test
        _ROPE_REGISTRY.clear()
        _ROPE_DICT.clear()

    def test_register_rope_type_registers_factory(self):
        @register_rope_type("test_type")
        def dummy_factory(*args, **kwargs):
            return Mock()

        self.assertIn("test_type", _ROPE_REGISTRY)
        self.assertEqual(_ROPE_REGISTRY["test_type"], dummy_factory)

    def test_register_rope_type_rejects_duplicate_registration(self):
        @register_rope_type("duplicate_type")
        def factory1(*args, **kwargs):
            return Mock()

        with self.assertRaises(ValueError) as cm:
            @register_rope_type("duplicate_type")
            def factory2(*args, **kwargs):
                return Mock()
        
        self.assertIn("duplicate_type", str(cm.exception))

    def test_unregister_rope_type_removes_from_registry(self):
        @register_rope_type("removable_type")
        def dummy_factory(*args, **kwargs):
            return Mock()

        self.assertIn("removable_type", _ROPE_REGISTRY)
        unregister_rope_type("removable_type")
        self.assertNotIn("removable_type", _ROPE_REGISTRY)

    def test_unregister_rope_type_no_error_for_missing_type(self):
        # Should not raise when unregistering non-existent type
        unregister_rope_type("non_existent_type")
        self.assertNotIn("non_existent_type", _ROPE_REGISTRY)

    def test_get_registered_rope_types_returns_correct_list(self):
        @register_rope_type("type_a")
        def factory_a(*args, **kwargs): return Mock()
        
        @register_rope_type("type_b")
        def factory_b(*args, **kwargs): return Mock()
        
        @register_rope_type("type_c")
        def factory_c(*args, **kwargs): return Mock()

        registered = get_registered_rope_types()
        self.assertEqual(set(registered), {"type_a", "type_b", "type_c"})

    def test_cached_rope_factory_caches_instances(self):
        mock_instance = Mock()
        call_count = [0]  # Mutable counter

        @cached_rope_factory
        def test_factory(head_size, rotary_dim, max_position, base, is_neox_style, dtype, rope_config):
            call_count[0] += 1
            return mock_instance

        config = RopeScaling(
            rope_type="test",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0
        )

        # First call - should create instance
        result1 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        self.assertEqual(call_count[0], 1)
        self.assertIs(result1, mock_instance)

        # Second call with identical parameters - should return cached instance
        result2 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        self.assertEqual(call_count[0], 1)  # No additional call
        self.assertIs(result2, mock_instance)
        self.assertIs(result1, result2)

    def test_cached_rope_factory_different_params_create_new_instances(self):
        instance1 = Mock()
        instance2 = Mock()
        call_sequence = [instance1, instance2]
        call_index = [0]

        @cached_rope_factory
        def test_factory(head_size, rotary_dim, max_position, base, is_neox_style, dtype, rope_config):
            result = call_sequence[call_index[0]]
            call_index[0] += 1
            return result

        config = RopeScaling(
            rope_type="test",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0
        )

        # First call
        result1 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        
        # Second call with different head_size
        result2 = test_factory(128, 64, 128, 10000.0, True, torch.float16, config)
        
        self.assertIsNot(result1, result2)
        self.assertEqual(call_index[0], 2)  # Two factory calls

    def test_cached_rope_factory_handles_rope_config_hashing(self):
        mock_instance = Mock()
        call_count = [0]

        @cached_rope_factory
        def test_factory(head_size, rotary_dim, max_position, base, is_neox_style, dtype, rope_config):
            call_count[0] += 1
            return mock_instance

        # Create two configs with same values but different objects
        config1 = RopeScaling(
            rope_type="test",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=0.5,
            factor=2.0
        )
        config2 = RopeScaling(
            rope_type="test",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=0.5,
            factor=2.0
        )

        # Should be treated as same key (hashable conversion works)
        result1 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config1)
        result2 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config2)
        
        self.assertEqual(call_count[0], 1)  # Only one factory call
        self.assertIs(result1, result2)

    def test_cached_rope_factory_passes_correct_args_to_default_factory(self):
        mock_factory = Mock(return_value=Mock())
        wrapped_factory = cached_rope_factory(mock_factory)

        config = RopeScaling(
            rope_type="default",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0
        )

        wrapped_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        
        # For 'default' type, should NOT pass rope_config
        mock_factory.assert_called_once_with(
            64, 64, 128, 10000.0, True, torch.float16
        )

    def test_cached_rope_factory_passes_correct_args_to_non_default_factory(self):
        mock_factory = Mock(return_value=Mock())
        wrapped_factory = cached_rope_factory(mock_factory)

        config = RopeScaling(
            rope_type="yarn",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0,
            factor=2.0
        )

        wrapped_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        
        # For non-'default' types, should pass rope_config
        mock_factory.assert_called_once_with(
            64, 64, 128, 10000.0, True, torch.float16, config
        )

    def test_clear_rope_cache_removes_all_cached_instances(self):
        mock_instance = Mock()

        @cached_rope_factory
        def test_factory(head_size, rotary_dim, max_position, base, is_neox_style, dtype, rope_config):
            return mock_instance

        config = RopeScaling(
            rope_type="test",
            rope_theta=10000.0,
            original_max_position_embeddings=2048,
            partial_rotary_factor=1.0
        )

        # Cache an instance
        result1 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        self.assertIn((64, 64, 128, True, tuple(asdict(config).items()), torch.float16), _ROPE_DICT)

        # Clear cache
        clear_rope_cache()
        self.assertEqual(len(_ROPE_DICT), 0)

        # Next call should create new instance (though we return same mock, cache was cleared)
        result2 = test_factory(64, 64, 128, 10000.0, True, torch.float16, config)
        self.assertIs(result1, result2)  # Same mock object, but cache was cleared and recreated

    def test_get_rope_factory_returns_registered_factory(self):
        mock_factory = Mock()
        _ROPE_REGISTRY["test_type"] = mock_factory

        result = get_rope_factory("test_type")
        self.assertIs(result, mock_factory)

    def test_get_rope_factory_returns_none_for_unregistered_type(self):
        result = get_rope_factory("unregistered_type")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()