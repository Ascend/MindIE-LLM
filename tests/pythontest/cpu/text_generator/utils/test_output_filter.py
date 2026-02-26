#!/usr/bin/env python
# coding=utf-8
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
from functools import partial
from unittest.mock import MagicMock

import numpy as np

from mindie_llm.text_generator.utils.output_filter import OutputFilter
from mindie_llm.text_generator.utils.stopping_criteria import make_mixed_eos, strings_eos
from mindie_llm.text_generator.utils.config import ResponseConfig
from mindie_llm.text_generator.utils.tg_decode_util import decode_one


class TestOutputFilter(unittest.TestCase):
    def setUp(self):
        cache_config = MagicMock()
        tokenizer = MagicMock()
        infer_context = MagicMock()
        self.output_filter = OutputFilter(cache_config, infer_context, tokenizer)
        return super().setUp()

    def test_decode_one(self):
        self.output_filter.tokenizer_sliding_window_size = 3
        self.output_filter.tokenizer.decode = MagicMock(return_value='')
        text = decode_one(self.output_filter.tokenizer, np.array([111]), True, 3)
        self.assertEqual(text, '')

        def mock_decode(input_ids, skip_special_tokens=True):
            if len(input_ids) == 1:
                return 'my'
            elif len(input_ids) > 1:
                return 'my word'
            return None

        self.output_filter.tokenizer.decode = mock_decode
        text = decode_one(self.output_filter.tokenizer, np.array([111, 222]), True, 3)
        self.assertEqual(text, ' word')

    def test_filter_by_async(self):
        self.output_filter.tg_infer_context.get_once_end_flag.return_value = np.array([True])
        filtered_ids = self.output_filter.filter_by_async(
            cache_ids=np.array([0]),
            filter_ids_arr=np.array([]),
            end_reason=np.array([0])
        )
        self.assertTrue((filtered_ids == np.array([0])).all())

    def test_filter_by_eos(self):
        self.output_filter.eos_token_id = [1, [1, 2]]
        self.output_filter.tg_infer_context.get_output_len_count.return_value = np.array([1])
        self.output_filter.tg_infer_context.get_all_output_ids.return_value = np.array([[0]])
        self.output_filter.tg_infer_context.get_ignore_eos.return_value = np.array([False])
        filtered_ids = self.output_filter.filter_by_eos(
            cache_ids=np.array([0]),
            next_token_ids=np.array([[1]]),
            num_new_tokens=np.array([1]),
            filter_ids_arr=np.array([]),
            end_reason=np.array([0])
        )
        self.assertTrue((filtered_ids == np.array([0])).all())

    def test_filter_by_stop(self):
        self.output_filter.tokenizer_sliding_window_size = 3
        self.output_filter.string_stopping_criteria = {0: partial(strings_eos, stop_strings=['<stop>'])}
        self.output_filter.stopping_criteria = {0: make_mixed_eos([2])}
        self.output_filter.decode_one = MagicMock(return_value='word')
        self.output_filter.tg_infer_context._batch_context.all_ndarray_context.output_texts = {0: 'my '}
        filtered_ids, _ = self.output_filter.filter_by_stop(
            cache_ids=np.array([0]),
            next_token_ids=np.array([[2]]),
            num_new_tokens=np.array([1]),
            filter_ids_arr=np.array([]),
            end_reason=np.array([0])
        )
        self.assertTrue((filtered_ids == np.array([0])).all())

    def test_filter_by_stop_empty_criteria(self):
        self.output_filter.tg_infer_context.is_empty_string_stopping_criteria.return_value = True
        self.output_filter.tg_infer_context.is_empty_stopping_criteria.return_value = True

        cache_ids = np.array([0, 1])
        filter_ids_arr = np.array([])
        end_reason = np.array([0, 0])

        result_ids, trunc_indices = self.output_filter.filter_by_stop(
            cache_ids=cache_ids,
            next_token_ids=np.array([[100], [200]]),
            num_new_tokens=np.array([1, 1]),
            filter_ids_arr=filter_ids_arr,
            end_reason=end_reason
        )

        self.assertTrue((result_ids == filter_ids_arr).all())
        self.assertTrue((trunc_indices == np.zeros(len(cache_ids))).all())

    def test_filter_by_stop_string_match(self):
        cache_ids = np.array([0])
        self.output_filter.tg_infer_context.is_empty_string_stopping_criteria.return_value = False
        self.output_filter.tg_infer_context.is_empty_stopping_criteria.return_value = True

        def mock_string_criterion(text, new_token, include_stop):
            return len(text) - 2 if text.endswith("stop") else None

        self.output_filter.tg_infer_context.get_string_stopping_criteria.return_value = [mock_string_criterion]
        self.output_filter.tg_infer_context.get_stopping_criteria.return_value = [None]
        self.output_filter.tg_infer_context.get_include_stop.return_value = False

        self.output_filter.tg_infer_context.get_output_len_count.return_value = 2
        self.output_filter.tg_infer_context.get_all_output_ids = MagicMock(return_value=np.array([1, 2]))
        self.output_filter.tg_infer_context.get_skip_special_tokens.return_value = True

        self.output_filter.decode_one = MagicMock()
        self.output_filter.decode_one.return_value = "stop"

        self.output_filter.tg_infer_context.append_and_return_output_text = MagicMock(return_value="teststop")

        filter_ids_arr = np.array([])
        end_reason = np.array([0])

        result_ids, trunc_indices = self.output_filter.filter_by_stop(
            cache_ids=cache_ids,
            next_token_ids=np.array([[3, 4]]),
            num_new_tokens=np.array([2]),
            filter_ids_arr=filter_ids_arr,
            end_reason=end_reason
        )

        self.assertTrue((result_ids == np.array([0])).all())
        self.assertEqual(trunc_indices[0], 6)
        self.assertEqual(end_reason[0], ResponseConfig.STOP_STRINGS)

    def test_filter_by_stop_id_match_no_include(self):
        cache_id = 1
        cache_ids = np.array([cache_id])
        self.output_filter.tg_infer_context.is_empty_string_stopping_criteria.return_value = True
        self.output_filter.tg_infer_context.is_empty_stopping_criteria.return_value = False

        def mock_id_criterion(token_ids):
            return token_ids[-1] == 999

        self.output_filter.tg_infer_context.get_stopping_criteria.return_value = [mock_id_criterion]
        self.output_filter.tg_infer_context.get_string_stopping_criteria.return_value = [None]
        self.output_filter.tg_infer_context.get_include_stop.return_value = False

        self.output_filter.tg_infer_context.get_output_len_count.return_value = 1
        self.output_filter.tg_infer_context.get_all_output_ids = MagicMock(return_value=np.array([5]))
        self.output_filter.tg_infer_context.get_skip_special_tokens.return_value = True

        self.output_filter.decode_one = MagicMock()
        self.output_filter.decode_one.return_value = "end"

        filter_ids_arr = np.array([])
        end_reason = np.array([0])

        result_ids, trunc_indices = self.output_filter.filter_by_stop(
            cache_ids=cache_ids,
            next_token_ids=np.array([[6, 999]]),
            num_new_tokens=np.array([2]),
            filter_ids_arr=filter_ids_arr,
            end_reason=end_reason
        )

        self.assertFalse((result_ids == np.array([cache_id])).all())
        self.assertEqual(trunc_indices[0], -3)
        self.assertEqual(end_reason[0], ResponseConfig.STOP_TOKEN_IDS)

    def test_filter_by_stop_empty_new_token(self):
        self.output_filter.tg_infer_context.is_empty_string_stopping_criteria.return_value = False
        self.output_filter.tg_infer_context.is_empty_stopping_criteria.return_value = False
        self.output_filter.tg_infer_context.get_string_stopping_criteria.return_value = [MagicMock()]
        self.output_filter.tg_infer_context.get_stopping_criteria.return_value = [MagicMock()]

        with self.assertRaises(RuntimeError) as ctx:
            self.output_filter.filter_by_stop(
                cache_ids=np.array([0]),
                next_token_ids=np.array([[777]]),
                num_new_tokens=np.array([0]),
                filter_ids_arr=np.array([]),
                end_reason=np.array([0])
            )

        self.assertEqual(str(ctx.exception), 'Empty `token_ids` generated!')

    def test_filter_finished_sequences_with_repeating_indices(self):
        sampling_output = MagicMock()
        sampling_output.repeating_indices = np.array([1, 0])
        sampling_output.token_ids = np.array([[999], [100]])
        sampling_output.num_new_tokens = np.array([1, 1])

        metadata = MagicMock()
        metadata.batch_max_output_lens = np.array([10, 10])
        metadata.is_mix = None

        self.output_filter.async_inference = False
        self.output_filter.ignore_eos = False
        self.output_filter.filter_by_eos = MagicMock(return_value=np.array([0]))
        self.output_filter.filter_by_stop = MagicMock(return_value=(np.array([]), np.array([0, 0])))
        self.output_filter.filter_by_length = MagicMock(return_value=np.array([]))

        end_reason, filter_ids_arr, trunc_indices = self.output_filter.filter_finished_sequences(
            cache_ids=np.array([10, 20]),
            metadata=metadata,
            sampling_output=sampling_output
        )

        actual_call = self.output_filter.filter_by_eos.call_args
        expected_cache_ids = np.array([20, 10])
        expected_filter_ids = np.array([], dtype=np.int32)

        self.assertTrue(np.array_equal(actual_call[0][0], expected_cache_ids))
        self.assertTrue(np.array_equal(actual_call[0][1], sampling_output.token_ids))
        self.assertTrue(np.array_equal(actual_call[0][2], sampling_output.num_new_tokens))
        self.assertTrue(np.array_equal(actual_call[0][3], expected_filter_ids))
        self.assertTrue(np.array_equal(actual_call[0][4], end_reason))


if __name__ == '__main__':
    unittest.main()
