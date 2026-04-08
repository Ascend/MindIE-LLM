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
from unittest.mock import patch, MagicMock

import mindspore as ms
import numpy as np

from mindie_llm.text_generator.samplers.logits_handlers.ms_handlers import MS_HANDLER_REGISTRY
from mindie_llm.text_generator.samplers.sampler_params import HandlerParams
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestMsHandlers(unittest.TestCase):
    def setUp(self):
        self.params = HandlerParams(backend_type='ms', rank=0)
        self.params.batch_size = 3
        self.params.vocab_size = 4
        self.metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            is_prefill=False,
            to_tensor=ms.tensor
        )
        self.res_logits = None

    def assert_almost_equal_2d(self, res_list, exp_list, places=6, delta=None):
        for res, exp in zip(res_list, exp_list):
            for r, e in zip(res, exp):
                self.assertAlmostEqual(
                    r, e, places=places, delta=delta, msg=f'\nres:\n{res_list}\nexp:\n{exp_list}\n')

    def test_repetition_penalty(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        repetition_penalty_tensor = ms.Tensor([1.5, 2.5, 0.5], dtype=ms.float32).unsqueeze(1)
        seq_ids_tensor = ms.Tensor(
            [[0, 0, 0],
             [1, 1, 1],
             [0, 2, 1]], dtype=ms.int32
        )
        repetition_penalty_lh = MS_HANDLER_REGISTRY['repetition_penalty'](self.params)
        self.metadata.repetition_penalty = repetition_penalty_tensor
        self.metadata.all_token_ids = seq_ids_tensor
        self.res_logits = repetition_penalty_lh(test_logits, self.metadata)
        expected_logits = [
            [0.1 / 1.5, 0.2, 0.3, 0.4],
            [-0.1, 0.01 / 2.5, 0.2, 0.02],
            [-0.3 * 0.5, -0.2 * 0.5, -0.1 * 0.5, 0.01]
        ]
        self.assert_almost_equal_2d(self.res_logits.asnumpy().tolist(), expected_logits)

    def test_repetition_penalty_with_outliers_ids(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        repetition_penalty_tensor = ms.Tensor([1.5, 2.5, 0.5], dtype=ms.float32).unsqueeze(1)
        seq_ids_tensor = ms.Tensor(
            [[0, -5, 3],
             [1, 12, 1],
             [-1, 2, 9]], dtype=ms.int32
        )
        repetition_penalty_lh = MS_HANDLER_REGISTRY['repetition_penalty'](self.params)
        self.metadata.repetition_penalty = repetition_penalty_tensor
        self.metadata.all_token_ids = seq_ids_tensor
        self.res_logits = repetition_penalty_lh(test_logits, self.metadata)
        expected_logits = [
          [0.1 / 1.5, 0.2, 0.3, 0.4 / 1.5],
          [-0.1, 0.01 / 2.5, 0.2, 0.02],
          [-0.3, -0.2, -0.1 * 0.5, 0.01]
        ]
        self.assert_almost_equal_2d(self.res_logits.asnumpy().tolist(), expected_logits)

    def test_frequency_penalty(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        frequency_penalty_tensor = ms.Tensor([1.5, 2.5, 0.5], dtype=ms.float32).unsqueeze(1)
        self.metadata.output_token_ids = ms.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3],
             [2, 3, 3, 3, 3, 3, 4, 4],
             [4, 4, 4, 4, 4, 4, 4, 4]], dtype=ms.int32
        )
        frequency_penalty_lh = MS_HANDLER_REGISTRY['frequency_penalty'](self.params)
        self.metadata.frequency_penalty = frequency_penalty_tensor
        self.res_logits = frequency_penalty_lh(test_logits, self.metadata)

    def test_presence_penalty(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        presence_penalty_tensor = ms.tensor([1.5, 2.5, 0.5]).unsqueeze(1)
        self.metadata.presence_penalty = presence_penalty_tensor
        self.metadata.output_token_ids = ms.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3],
             [2, 3, 3, 3, 3, 3, 4, 4],
             [4, 4, 4, 4, 4, 4, 4, 4]], dtype=ms.int32
        )
        presence_penalty_lh = MS_HANDLER_REGISTRY['presence_penalty'](self.params)
        self.res_logits = presence_penalty_lh(test_logits, self.metadata)

    def test_temperature(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        temperature_tensor = ms.tensor([1.5, 1, 0.5]).unsqueeze(1)
        self.metadata.temperature = temperature_tensor
        temperature_lh = MS_HANDLER_REGISTRY['temperature'](self.params)
        self.res_logits = temperature_lh(test_logits, self.metadata)
        expected_logits = [
            [0.1 / 1.5, 0.2 / 1.5, 0.3 / 1.5, 0.4 / 1.5],
            [-0.1 / 1, 0.01 / 1, 0.2 / 1, 0.02 / 1],
            [-0.3 / 0.5, -0.2 / 0.5, -0.1 / 0.5, 0.01 / 0.5]
        ]
        self.assert_almost_equal_2d(self.res_logits.asnumpy().tolist(), expected_logits)

    def test_guided_decoding_bitmask_none(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        guided_decoding_lh = MS_HANDLER_REGISTRY['guided_decoding'](self.params)
        self.metadata.guided_bitmask = None
        self.res_logits = guided_decoding_lh(test_logits, self.metadata)
        self.assertTrue(np.array_equal(self.res_logits.asnumpy(), test_logits.asnumpy()))

    def test_guided_decoding_import_success(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        guided_decoding_lh = MS_HANDLER_REGISTRY['guided_decoding'](self.params)
        bitmask = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
        self.metadata.guided_bitmask = bitmask
        mock_apply = MagicMock()
        guided_decoding_lh._apply_token_bitmask_inplace = mock_apply
        guided_decoding_lh._import_attempted = True
        self.res_logits = guided_decoding_lh(test_logits, self.metadata)
        mock_apply.assert_called_once()

    def test_guided_decoding_import_failure(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        guided_decoding_lh = MS_HANDLER_REGISTRY['guided_decoding'](self.params)
        bitmask = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
        self.metadata.guided_bitmask = bitmask
        guided_decoding_lh._import_attempted = True
        guided_decoding_lh._apply_token_bitmask_inplace = None
        self.res_logits = guided_decoding_lh(test_logits, self.metadata)
        self.assertTrue(np.array_equal(self.res_logits.asnumpy(), test_logits.asnumpy()))

    def test_guided_decoding_apply_exception(self):
        test_logits = ms.Tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        )
        guided_decoding_lh = MS_HANDLER_REGISTRY['guided_decoding'](self.params)
        bitmask = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32)
        self.metadata.guided_bitmask = bitmask
        guided_decoding_lh._apply_token_bitmask_inplace = MagicMock(side_effect=RuntimeError("test error"))
        guided_decoding_lh._import_attempted = True
        self.res_logits = guided_decoding_lh(test_logits, self.metadata)
        self.assertTrue(np.array_equal(self.res_logits.asnumpy(), test_logits.asnumpy()))

    def test_guided_decoding_lazy_import_first_attempt(self):
        guided_decoding_lh = MS_HANDLER_REGISTRY['guided_decoding'](self.params)
        mock_func = MagicMock()
        with patch.dict('sys.modules', {'mindie_llm.text_generator.plugins.structured_output': MagicMock(apply_token_bitmask_inplace=mock_func)}):
            result = guided_decoding_lh._lazy_import()
            self.assertTrue(guided_decoding_lh._import_attempted)

    def test_guided_decoding_lazy_import_already_attempted(self):
        guided_decoding_lh = MS_HANDLER_REGISTRY['guided_decoding'](self.params)
        guided_decoding_lh._import_attempted = True
        guided_decoding_lh._apply_token_bitmask_inplace = MagicMock()
        result = guided_decoding_lh._lazy_import()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()