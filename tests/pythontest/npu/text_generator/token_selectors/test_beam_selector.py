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

import numpy as np
import torch

from mindie_llm.text_generator.samplers.sampler_params import SelectorParams
from mindie_llm.text_generator.samplers.token_selectors.pta_selectors import PTA_SELECTOR_REGISTRY
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestBeamSearchTokenSelector(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.selector_params = SelectorParams()
        setattr(self.selector_params, "eos_token_id", [2])
        setattr(self.selector_params, "candidate_expansion_factor", 1)
        setattr(self.selector_params, "device", self.device)
        self.beam_search_selector = PTA_SELECTOR_REGISTRY["beam_search"](self.selector_params)

    def test_prefill(self):
        logits = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.11, 0.22, 0.33, 0.44],
                [-0.1, 0.01, 0.2, 0.02, -0.11, 0.011, 0.22, 0.022],
                [-0.3, -0.2, -0.1, 0.01, -0.33, -0.22, -0.11, 0.011],
            ]
        ).to(self.device)
        metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([100]), np.array([102]), np.array([105])],
            reserved_sequence_ids=[np.array([101]), np.array([103, 104]), np.array([])],
            is_prefill=True,
            to_tensor=torch.tensor,
        )
        metadata.max_beam_width = 3
        metadata.beam_width_array = np.array([2, 3, 1])
        metadata.use_beam_search_array = np.array([1, 1, 1])
        metadata.cumulative_logprobs = np.array([0, 0, 0])
        metadata.output_lengths = np.array([1, 1, 1])
        output = self.beam_search_selector(logits, metadata)
        print(f"output: {output}")

    def test_decode(self):
        logits = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.11, 0.22, 0.33, 0.44],
                [0.11, 0.22, 0.33, 0.44, 0.111, 0.222, 0.333, 0.444],
                [-0.1, 0.01, 0.2, 0.02, -0.11, 0.011, 0.22, 0.022],
                [-0.11, 0.011, 0.22, 0.022, -0.111, 0.0111, 0.222, 0.0222],
                [-0.13, 0.013, 0.23, 0.023, -0.134, 0.0134, 0.234, 0.0234],
                [-0.3, -0.2, -0.1, 0.01, -0.33, -0.22, -0.11, 0.011],
            ]
        ).to(self.device)
        metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([100, 101]), np.array([102, 103, 104]), np.array([105])],
            is_prefill=False,
            to_tensor=torch.tensor,
        )
        metadata.max_beam_width = 3
        metadata.beam_width_array = np.array([2, 2, 3, 3, 3, 1])
        metadata.use_beam_search_array = np.array([1, 1, 1, 1, 1, 1])
        metadata.cumulative_logprobs = np.array([-1.1, -0.9, -2.1, -2.0, -1.9, -0.3])
        metadata.output_lengths = np.array([3, 3, 3, 3, 3, 3])
        output = self.beam_search_selector(logits, metadata)
        print(f"output: {output}")


if __name__ == "__main__":
    unittest.main()
