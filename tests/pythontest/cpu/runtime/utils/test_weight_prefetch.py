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
from unittest.mock import patch, MagicMock, PropertyMock

import torch
import mindie_llm.runtime.utils.weight_prefetcher as weight_prefetcher
# -----------------------------------------------------------------------------
# Pre-import Mocking
# -----------------------------------------------------------------------------
# Mock 'torch_npu' before importing the target module to prevent 
# ModuleNotFoundError on CPU-only environments.
if 'torch_npu' not in sys.modules:
    sys.modules['torch_npu'] = MagicMock()

# Import the module under test using the full package path.


class TestWeightPrefetcher(unittest.TestCase):
    """
    Test cases for WeightPrefetchMethod and helper functions.
    
    Ensures all NPU interactions are properly mocked and logic flows 
    correctly under simulated conditions.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures before each test method.
        
        Resets global state and configures mocks for NPU dependencies.
        """
        # Reset global prefetcher instance state to ensure test isolation
        weight_prefetcher.weight_prefetcher.disable_weight_prefetch()
        weight_prefetcher.weight_prefetcher.prefetch_weights = {}

        # Mock torch_npu used within the target module
        self.patcher_torch_npu = patch(
            'mindie_llm.runtime.utils.weight_prefetcher.torch_npu'
        )
        self.mock_torch_npu = self.patcher_torch_npu.start()

        # Mock torch.npu namespace used within the target module
        self.patcher_torch_npu_ns = patch(
            'mindie_llm.runtime.utils.weight_prefetcher.torch.npu'
        )
        self.mock_torch_npu_ns = self.patcher_torch_npu_ns.start()

        # Configure mock stream objects
        self.mock_stream = MagicMock()
        self.mock_torch_npu_ns.Stream.return_value = self.mock_stream
        self.mock_torch_npu_ns.current_stream.return_value = MagicMock()

    def tearDown(self) -> None:
        """
        Tear down test fixtures after each test method.
        
        Stops all active patches to prevent side effects on other tests.
        """
        self.patcher_torch_npu.stop()
        self.patcher_torch_npu_ns.stop()

    def test_enable_disable_prefetch(self) -> None:
        """
        Test enabling and disabling the prefetch mechanism.
        
        Verifies flag updates and stream initialization logic.
        """
        prefetcher = weight_prefetcher.WeightPrefetchMethod()

        # Initially disabled
        self.assertFalse(prefetcher.is_prefetch_enabled())

        # Enable prefetch
        prefetcher.enable_weight_prefetch()
        self.assertTrue(prefetcher.is_prefetch_enabled())
        # Verify Stream was called with current device
        self.mock_torch_npu_ns.Stream.assert_called_once()

        # Disable prefetch
        prefetcher.disable_weight_prefetch()
        self.assertFalse(prefetcher.is_prefetch_enabled())
        self.assertIsNone(prefetcher.prefetch_stream)

    def test_maybe_npu_prefetch_logic(self) -> None:
        """
        Test the internal helper function _maybe_npu_prefetch.
        
        Verifies size calculation and parameter passing to NPU interface.
        """
        dummy_tensor = torch.randn(5, 5)
        dummy_dep = torch.randn(1)
        element_size = dummy_tensor.element_size()
        numel = dummy_tensor.numel()
        total_size = element_size * numel

        # Case 1: max_size <= 0, should use input_size
        weight_prefetcher._maybe_npu_prefetch(dummy_tensor, dummy_dep, max_size=0)
        self.mock_torch_npu.npu_prefetch.assert_called_with(
            dummy_tensor, dummy_dep, total_size, 0
        )

        # Reset mock for next case
        self.mock_torch_npu.npu_prefetch.reset_mock()

        # Case 2: max_size > input_size, should use input_size
        weight_prefetcher._maybe_npu_prefetch(dummy_tensor, dummy_dep, max_size=total_size + 100)
        self.mock_torch_npu.npu_prefetch.assert_called_with(
            dummy_tensor, dummy_dep, total_size, 0
        )

        # Reset mock for next case
        self.mock_torch_npu.npu_prefetch.reset_mock()

        # Case 3: max_size valid, should use max_size
        valid_size = total_size - 10
        weight_prefetcher._maybe_npu_prefetch(dummy_tensor, dummy_dep, max_size=valid_size)
        self.mock_torch_npu.npu_prefetch.assert_called_with(
            dummy_tensor, dummy_dep, valid_size, 0
        )

    def test_prefetch_preprocess_stream_logic(self) -> None:
        """
        Test _prefetch_preprocess function.
        
        Verifies stream waiting and context manager usage.
        """
        dummy_weight = torch.randn(10, 10)
        dummy_flag = torch.randn(1)
        calc_stream = MagicMock()
        prefetch_stream = MagicMock()

        # Configure current_stream to return calc_stream
        self.mock_torch_npu_ns.current_stream.return_value = calc_stream
        
        # Configure context manager behavior for stream
        prefetch_stream.__enter__ = MagicMock(return_value=prefetch_stream)
        prefetch_stream.__exit__ = MagicMock(return_value=None)

        weight_prefetcher._prefetch_preprocess(
            weight=dummy_weight,
            start_flag=dummy_flag,
            max_weight_size=1024,
            weight_prefetch_stream=prefetch_stream
        )

        # Verify calculation stream wait
        prefetch_stream.wait_stream.assert_called_once_with(calc_stream)
        
        # Verify npu_prefetch was called within the context
        self.mock_torch_npu.npu_prefetch.assert_called()

    def test_prefetch_weight_preprocess_integration(self) -> None:
        """
        Test WeightPrefetchMethod.prefetch_weight_preprocess.
        
        Verifies integration of size calculation and helper call.
        """
        prefetcher = weight_prefetcher.WeightPrefetchMethod()
        prefetcher.enable_weight_prefetch()
        
        dummy_weight = torch.randn(10, 10)
        dummy_flag = torch.randn(1)
        ratio = 0.5
        
        expected_size = int(dummy_weight.element_size() * dummy_weight.numel() * ratio)

        prefetcher.prefetch_weight_preprocess(
            weight=dummy_weight,
            start_flag=dummy_flag,
            ratio=ratio
        )

        # Verify npu_prefetch was called with the calculated size
        self.mock_torch_npu.npu_prefetch.assert_called()
        
        # Check the last call arguments for size correctness
        last_call_args = self.mock_torch_npu.npu_prefetch.call_args
        # max_size is the 3rd argument (index 2)
        called_max_size = last_call_args[0][2] 
        self.assertEqual(called_max_size, expected_size)

    def test_prefetch_weight_postprocess(self) -> None:
        """
        Test WeightPrefetchMethod.prefetch_weight_postprocess.
        
        Verifies stream synchronization logic.
        """
        prefetcher = weight_prefetcher.WeightPrefetchMethod()
        prefetcher.enable_weight_prefetch()
        
        calc_stream = MagicMock()
        self.mock_torch_npu_ns.current_stream.return_value = calc_stream

        prefetcher.prefetch_weight_postprocess()

        # Verify calculation stream waits for prefetch stream
        calc_stream.wait_stream.assert_called_once_with(prefetcher.prefetch_stream)

    def test_global_instance_state(self) -> None:
        """
        Test the global weight_prefetcher instance behavior.
        
        Ensures global state management works as expected across calls.
        """
        # Initially disabled (ensured by setUp)
        self.assertFalse(weight_prefetcher.weight_prefetcher.is_prefetch_enabled())

        # Enable via global instance
        weight_prefetcher.weight_prefetcher.enable_weight_prefetch()
        self.assertTrue(weight_prefetcher.weight_prefetcher.is_prefetch_enabled())

        # Clean up
        weight_prefetcher.weight_prefetcher.disable_weight_prefetch()


if __name__ == '__main__':
    unittest.main()
