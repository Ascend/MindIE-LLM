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
import torch
from mindie_llm.runtime.layers.attention.backend.sparse_attention import SfaMetadata


class TestSfaMetadata(unittest.TestCase):

    def setUp(self):
        self.dummy_mask = torch.ones(1, 1)
        self.dummy_cos = torch.randn(1, 1)
        self.dummy_sin = torch.randn(1, 1)

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.get_parallel_info_manager')
    def test_prefill_without_q_lens(self, mock_get_parallel_info_manager):
        mock_manager = MagicMock()
        mock_manager.attn_cp = MagicMock()
        mock_manager.attn_cp.group_size = 1
        mock_get_parallel_info_manager.return_value = mock_manager

        model_inputs = MagicMock()
        model_inputs.is_prefill = True
        model_inputs.context_length = [3, 5, 2]  # batch of 3 sequences
        model_inputs.q_lens = None

        metadata = SfaMetadata.from_model_input(
            model_inputs, self.dummy_cos, self.dummy_sin, self.dummy_mask
        )

        expected_kv = torch.tensor([3, 5, 2], dtype=torch.int32)
        expected_query = torch.tensor([3, 8, 10], dtype=torch.int32)

        self.assertTrue(torch.equal(metadata.actual_seq_lengths_kv, expected_kv))
        self.assertTrue(torch.equal(metadata.actual_seq_lengths_query, expected_query))

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.get_parallel_info_manager')
    def test_prefill_with_q_lens(self, mock_get_parallel_info_manager):
        mock_manager = MagicMock()
        mock_manager.attn_cp = MagicMock()
        mock_manager.attn_cp.group_size = 1
        mock_get_parallel_info_manager.return_value = mock_manager

        model_inputs = MagicMock()
        model_inputs.is_prefill = True
        model_inputs.context_length = [10, 20]
        model_inputs.q_lens = torch.tensor([4, 6], dtype=torch.int32)

        metadata = SfaMetadata.from_model_input(
            model_inputs, self.dummy_cos, self.dummy_sin, self.dummy_mask
        )

        expected_kv = torch.tensor([10, 20], dtype=torch.int32)
        expected_query = torch.tensor([4, 10], dtype=torch.int32)

        self.assertTrue(torch.equal(metadata.actual_seq_lengths_kv, expected_kv))
        self.assertTrue(torch.equal(metadata.actual_seq_lengths_query, expected_query))

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.get_parallel_info_manager')
    def test_prefill_single_sequence(self, mock_get_parallel_info_manager):
        mock_manager = MagicMock()
        mock_manager.attn_cp = MagicMock()
        mock_manager.attn_cp.group_size = 1
        mock_get_parallel_info_manager.return_value = mock_manager

        model_inputs = MagicMock()
        model_inputs.is_prefill = True
        model_inputs.context_length = [7]
        model_inputs.q_lens = None

        metadata = SfaMetadata.from_model_input(
            model_inputs, self.dummy_cos, self.dummy_sin, self.dummy_mask
        )

        expected_kv = torch.tensor([7], dtype=torch.int32)
        expected_query = torch.tensor([7], dtype=torch.int32)

        self.assertTrue(torch.equal(metadata.actual_seq_lengths_kv, expected_kv))
        self.assertTrue(torch.equal(metadata.actual_seq_lengths_query, expected_query))
    
    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.get_parallel_info_manager')
    def test_decode_mode(self, mock_get_parallel_info_manager):
        mock_manager = MagicMock()
        mock_manager.attn_cp = MagicMock()
        mock_manager.attn_cp.group_size = 1
        mock_get_parallel_info_manager.return_value = mock_manager

        batch_size = 4
        model_inputs = MagicMock()
        model_inputs.is_prefill = False
        model_inputs.block_tables = torch.zeros(batch_size, 8)

        metadata = SfaMetadata.from_model_input(
            model_inputs, self.dummy_cos, self.dummy_sin, self.dummy_mask
        )

        expected_kv = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        expected_query = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

        self.assertTrue(torch.equal(metadata.actual_seq_lengths_kv, expected_kv))
        self.assertTrue(torch.equal(metadata.actual_seq_lengths_query, expected_query))

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.get_parallel_info_manager')
    def test_decode_single_sequence(self, mock_get_parallel_info_manager):
        mock_manager = MagicMock()
        mock_manager.attn_cp = MagicMock()
        mock_manager.attn_cp.group_size = 1
        mock_get_parallel_info_manager.return_value = mock_manager

        model_inputs = MagicMock()
        model_inputs.is_prefill = False
        model_inputs.block_tables = torch.zeros(1, 8)

        metadata = SfaMetadata.from_model_input(
            model_inputs, self.dummy_cos, self.dummy_sin, self.dummy_mask
        )

        expected_kv = torch.tensor([1], dtype=torch.int32)
        expected_query = torch.tensor([1], dtype=torch.int32)

        self.assertTrue(torch.equal(metadata.actual_seq_lengths_kv, expected_kv))
        self.assertTrue(torch.equal(metadata.actual_seq_lengths_query, expected_query))


if __name__ == "__main__":
    unittest.main()
