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

import numpy as np
import torch

from mindie_llm.text_generator.samplers.token_selectors.cpu_selectors import CPU_SELECTOR_REGISTRY
from mindie_llm.text_generator.samplers.sampler_params import HandlerParams, SelectorParams


class TestCpuHandlers(unittest.TestCase):
    def setUp(self):
        self.test_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, 0.01, 0.2, 0.02], [-0.3, -0.2, -0.1, 0.01]])
        self.params = HandlerParams(backend_type="atb", rank=0)
        self.selector_params = SelectorParams()
        # Create a mock SamplingMetadata object
        self.metadata = MagicMock()
        self.metadata.batch_size = 3
        self.metadata.all_sequence_ids = np.array([0, 1, 2])
        self.metadata.is_prefill = False
        self.metadata.is_mix = False
        self.metadata.num_top_tokens = np.array([0, 0, 0])
        self.metadata.parent_sequence_ids = np.array([0, 1, 2])
        self.metadata.group_indices = [(0, 1), (1, 2), (2, 3)]
        self.speed_mode = 2
        self.use_approx = 1

    def assert_almost_equal_1d(self, res_list, exp_list):
        for r, e in zip(res_list, exp_list):
            self.assertAlmostEqual(r, e, places=6, msg=f"\nres:\n{res_list}\nexp:\n{exp_list}\n")

    def test_top_k_top_p_sampling(self):
        # Create a mock CPU selector instance
        mock_selector = MagicMock()
        # Create a mock sampling output with expected token IDs
        mock_output = MagicMock()
        mock_output.token_ids = np.array([3, 2, 2])
        # Make the mock selector return our mock output
        mock_selector.return_value = mock_output

        # Mock the CPU_SELECTOR_REGISTRY to return our mock selector
        with patch.dict(
            "mindie_llm.text_generator.samplers.token_selectors.cpu_selectors.CPU_SELECTOR_REGISTRY",
            {"top_k_top_p_sampling": lambda *args, **kwargs: mock_selector},
        ):
            self.params.num_threads = 16
            self.params.npu_id = self.params.rank
            self.params.request_ids = np.array([0, 1, 2])
            self.params.sampling_method = "multinomial"

            topk_topp_sampling_lh = CPU_SELECTOR_REGISTRY["top_k_top_p_sampling"](self.selector_params)
            topk_topp_sampling_lh.configure(self.metadata)
            sampling_output = topk_topp_sampling_lh(self.test_logits, self.metadata)
            expected_tokens = [3, 2, 2]
            self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)

    def test_top_approx(self):
        # Create a mock CPU selector instance
        mock_selector = MagicMock()
        # Create a mock sampling output with expected token IDs
        mock_output = MagicMock()
        mock_output.token_ids = np.array([3, 2, 2])
        # Make the mock selector return our mock output
        mock_selector.return_value = mock_output

        # Mock the CPU_SELECTOR_REGISTRY to return our mock selector
        with patch.dict(
            "mindie_llm.text_generator.samplers.token_selectors.cpu_selectors.CPU_SELECTOR_REGISTRY",
            {"top_k_top_p_sampling": lambda *args, **kwargs: mock_selector},
        ):
            self.params.num_threads = 16
            self.params.npu_id = self.params.rank
            self.params.request_ids = np.array([0, 1, 2])
            self.params.sampling_method = "multinomial"

            topk_topp_sampling_lh = CPU_SELECTOR_REGISTRY["top_k_top_p_sampling"](self.selector_params)
            topk_topp_sampling_lh.speed_mode = self.speed_mode
            topk_topp_sampling_lh.use_approx = self.use_approx
            topk_topp_sampling_lh.configure(self.metadata)
            sampling_output = topk_topp_sampling_lh(self.test_logits, self.metadata)
            expected_tokens = [3, 2, 2]
            self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)


if __name__ == "__main__":
    unittest.main()
