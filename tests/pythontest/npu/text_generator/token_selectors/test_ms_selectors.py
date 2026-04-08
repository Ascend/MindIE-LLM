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

import mindspore as ms
import numpy as np

from mindie_llm.text_generator.samplers.token_selectors.ms_selectors import MS_SELECTOR_REGISTRY
from mindie_llm.text_generator.samplers.sampler_params import HandlerParams, SelectorParams
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestMsHandlers(unittest.TestCase):
    def setUp(self):
        self.params = HandlerParams(backend_type='ms', rank=0)
        self.selector_params = SelectorParams()
        self.params.batch_size = 3
        self.params.vocab_size = 4
        self.res_logits = None

    def assert_almost_equal_1d(self, res_list, exp_list):
        for r, e in zip(res_list, exp_list):
            self.assertAlmostEqual(r, e, places=6, msg=f'\nres:\n{res_list}\nexp:\n{exp_list}\n')

    def test_top_k_top_p_sampling(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        self.params.num_threads = 16
        self.params.npu_id = self.params.rank
        self.params.request_ids = np.array([0, 1, 2])
        self.params.sampling_method = 'multinomial'

        top_k_array = np.array([1, 2, 0])
        top_k_tensor = ms.tensor(top_k_array - 1)
        top_p_array = np.array([0.8, 0.1, 0.5])
        top_p_tensor = ms.tensor(top_p_array).unsqueeze(1)
        do_sample_array = np.array([True, False, True])
        top_logprobs_array = np.array([1, 1, 1])

        metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            is_prefill=True,
            to_tensor=ms.tensor
        )
        metadata.top_k_array = top_k_array
        metadata.top_k_idx = top_k_tensor.unsqueeze(1)
        metadata.top_k_disabled_mask = ms.tensor(top_k_array < 0).unsqueeze(1)
        metadata.max_top_k = 2
        metadata.top_p_array = top_p_array
        metadata.top_p = top_p_tensor
        metadata.do_sample_array = do_sample_array
        metadata.seed_array = np.array([10, 1, 1024])
        metadata.all_token_ids = None
        metadata.top_logprobs_array = top_logprobs_array

        topk_topp_sampling_lh = MS_SELECTOR_REGISTRY['top_k_top_p_sampling'](self.selector_params)
        sampling_output = topk_topp_sampling_lh(test_logits, metadata)
        expected_tokens = [3, 2, 2]
        self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)
        metadata.top_p = None
        sampling_output = topk_topp_sampling_lh(test_logits, metadata)
        expected_tokens = [3, 2, 2]
        self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)

    def test_fusion_top_k_top_p_sampling(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        self.params.num_threads = 16
        self.params.npu_id = self.params.rank
        self.params.request_ids = np.array([0, 1, 2])
        self.params.sampling_method = 'multinomial'

        top_k_array = np.array([1, 2, 3])
        top_k_tensor = ms.tensor(top_k_array - 1)
        top_p_array = np.array([0.8, 0.1, 0.5])
        top_p_tensor = ms.tensor(top_p_array).unsqueeze(1)
        do_sample_array = np.array([True, False, True])
        top_logprobs_array = np.array([1, 1, 1])

        metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            is_prefill=False,
            to_tensor=ms.tensor
        )
        metadata.top_k_array = top_k_array
        metadata.top_k_idx = top_k_tensor.unsqueeze(1)
        metadata.top_k_disabled_mask = None
        metadata.max_top_k = 2
        metadata.top_p_array = top_p_array
        metadata.top_p = top_p_tensor
        metadata.do_sample_array = do_sample_array
        metadata.seed_array = np.array([10, 1, 1024])
        metadata.all_token_ids = None
        metadata.top_logprobs_array = top_logprobs_array
        metadata.max_logprobs = top_logprobs_array.max()

        topk_topp_sampling_lh = MS_SELECTOR_REGISTRY['top_k_top_p_sampling'](self.selector_params)
        metadata.all_sequence_ids = None
        topk_topp_sampling_lh.configure(metadata)
        sampling_output = topk_topp_sampling_lh(test_logits, metadata)
        topk_topp_sampling_lh.clear(np.arange(3))
        expected_tokens = [3, 2, 3]
        self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)

    def test_greedy_search(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )

        metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            is_prefill=False,
            to_tensor=ms.tensor
        )
        greedy_search_lh = MS_SELECTOR_REGISTRY['greedy_search'](self.selector_params)
        expected_tokens = [3, 2, 3]
        sampling_output = greedy_search_lh(test_logits, metadata)

        self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)


if __name__ == '__main__':
    unittest.main()