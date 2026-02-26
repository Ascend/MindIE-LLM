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
from unittest.mock import patch, MagicMock, Mock
import torch
import numpy as np

from mindie_llm.text_generator.utils.batch_context import DictContext, NdarrayContext, BatchContext
from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
from mindie_llm.text_generator.utils.config import ContextParams, CacheConfig, SpCpParallelInfo, DEFAULT_SAMPLING_PARAMS
from mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess import SplitFusePreprocess
from mindie_llm.text_generator.utils.tg_infer_context_store import TGInferContextStore


class TestSplitFusePreprocess(unittest.TestCase):
    
    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_init(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        kvcache_settings = None
        
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)

        self.assertIsNone(splitfuse_preprocess.model_wrapper.device)
        self.assertIsNone(splitfuse_preprocess.device)
    
    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_make_attention_mask(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        kvcache_settings = None
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)
        model_inputs = None
        input_metadata = None
        q_lens = None
        hit_mask = None

        req_mask = splitfuse_preprocess.make_attention_mask(model_inputs, input_metadata, q_lens, hit_mask)

        self.assertIsNone(req_mask)

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='atb')
    def test_make_attention_mask_is300i(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        model_wrapper.model_runner = MagicMock()
        model_wrapper.model_runner.attn_mask = MagicMock()
        model_wrapper.model_runner.attn_mask.get_attn_mask.return_value = torch.ones(5, 5)
        kvcache_settings = MagicMock()
        kvcache_settings.dtype = torch.float16
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)
        splitfuse_preprocess.is_300i = True
        splitfuse_preprocess.async_inference = False
        model_inputs = MagicMock()
        model_inputs.max_seq_len = 1
        model_inputs.context_length = [3, 3]
        input_metadata = MagicMock()
        input_metadata.is_prefill = True
        q_lens = [2, 2]
        hit_mask = None

        req_mask = splitfuse_preprocess.make_attention_mask(model_inputs, input_metadata, q_lens, hit_mask)

        golden_mask = torch.ones(4, 5)
        self.assertTrue(torch.allclose(req_mask, golden_mask))

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_get_mix_decode_cache_without_hit_mask(self):
        tokenizer = MagicMock()
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.block_size = 4
        self.batch_config = CacheConfig(
            cache_size=4,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = SpCpParallelInfo(
            sp_parallel_info=Mock(group_size=1, rank=0),
            cp_parallel_info=Mock(group_size=1, rank=0)
        )
        self.context_params = ContextParams(distributed=False)
        self.batch_ctx = BatchContext(
            kvcache_settings=self.kvcache_settings,
            context_params=self.context_params,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device,
            tokenizer=tokenizer,
            tokenizer_sliding_window_size=3
        )
        cache_ids = torch.tensor([0, 2])
        decode_idx = 1
        self.batch_ctx.all_ndarray_context.last_input_ids = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        self.batch_ctx.all_ndarray_context.seq_lens = torch.tensor([2, 3, 1])
        self.batch_ctx.all_ndarray_context.last_position_ids = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.batch_ctx.all_ndarray_context.used_block_idx = torch.tensor([0, 1, 2])
        self.batch_ctx.all_ndarray_context.used_block_offset = torch.tensor([1, 0, 2])
        self.batch_ctx.kv_slots = torch.arange(500).reshape(50, 10)
        metadata = MagicMock()
        metadata.batch_block_tables = torch.tensor([
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33]
        ])
        result = self.batch_ctx.get_mix_decode_cache_for_splitfuse(cache_ids, decode_idx, metadata, hit_mask=None)
        input_ids, max_seq_len, position_ids, input_lengths, slots = result
        torch.testing.assert_close(input_ids, torch.tensor([[10, 11, 12], [30, 31, 32]]))
        self.assertEqual(max_seq_len, 2)
        torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        torch.testing.assert_close(input_lengths, torch.tensor([2, 1]))
        expected_slots = torch.tensor([
            self.batch_ctx.kv_slots[20, 1],
            self.batch_ctx.kv_slots[22, 2]
        ])
        torch.testing.assert_close(slots, expected_slots)

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_get_mix_decode_cache_with_hit_mask(self):
        tokenizer = MagicMock()
        cache_ids = torch.tensor([0, 1])
        decode_idx = 0
        hit_mask = torch.tensor([[True, False]])
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.block_size = 4
        self.batch_config = CacheConfig(
            cache_size=4,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = SpCpParallelInfo(
            sp_parallel_info=Mock(group_size=1, rank=0),
            cp_parallel_info=Mock(group_size=1, rank=0)
        )
        self.context_params = ContextParams(distributed=False)
        self.batch_ctx = BatchContext(
            kvcache_settings=self.kvcache_settings,
            context_params=self.context_params,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device,
            tokenizer=tokenizer,
            tokenizer_sliding_window_size=3
        )
        self.batch_ctx.spcp_parallel_info.scp_rank = 0

        self.batch_ctx.all_ndarray_context.last_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.batch_ctx.all_ndarray_context.seq_lens = torch.tensor([3, 2])
        self.batch_ctx.all_ndarray_context.last_position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]])
        self.batch_ctx.all_ndarray_context.cpu_cached_seq_idx = torch.tensor([[4], [5]])
        self.batch_ctx.batch_context_config.max_block_size = 4
        self.batch_ctx.kv_slots = torch.arange(1000).reshape(100, 10)
        metadata = MagicMock()
        metadata.batch_block_tables = torch.tensor([
            [50, 51, 52, 53],
            [60, 61, 62, 63]
        ])

        result = self.batch_ctx.get_mix_decode_cache_for_splitfuse(cache_ids, decode_idx, metadata, hit_mask=hit_mask)
        input_ids, max_seq_len, position_ids, input_lengths, slots = result
        torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(max_seq_len, 3)
        torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2], [0, 1, 2]]))
        torch.testing.assert_close(input_lengths, torch.tensor([3, 2]))
        expected_slots = torch.tensor([self.batch_ctx.kv_slots[51, 1], self.batch_ctx.kv_slots[51, 1]])
        torch.testing.assert_close(slots, expected_slots)

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_concatenate_mix_decode_only(self):
        tokenizer = MagicMock()
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.block_size = 4
        self.batch_config = CacheConfig(
            cache_size=4,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = [Mock(group_size=1, rank=0), Mock(group_size=1, rank=0)]
        self.context_params = ContextParams(distributed=False)
        store = TGInferContextStore(
            kvcache_settings=self.kvcache_settings,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device,
            context_params=self.context_params,
            tokenizer=tokenizer,
            tokenizer_sliding_window_size=3
        )
        store._batch_context = MagicMock(spec=BatchContext)

        metadata = MagicMock()
        metadata.input_ids = [100, 200]  # two decode tokens
        metadata.total_seq_num = 2
        metadata.batch_is_prefill = np.array([False, False])
        metadata.batch_seq_len = np.array([1, 1])
        metadata.max_seq_len = 5
        metadata.has_sampling = False
        metadata.all_sequence_ids = np.array([10, 11])
        metadata.batch_dp_rank_ids = np.array([0, 0])
        metadata.split_start_position = np.array([50, 60])
        metadata.split_end_position = np.array([51, 61])

        context_handles = np.array([0, 1], dtype=np.int64)
        hit_mask = None

        mock_decode_result = (                      # Mock decode cache return
            np.array([100, 200], dtype=np.int64),   # input_ids_decode
            3,                                      # max_seq_len_decode
            np.array([2, 1], dtype=np.int32),       # position_ids_decode
            np.array([3, 2], dtype=np.int32),       # input_lengths_decode
            np.array([1000, 2000], dtype=np.int32)  # slots_decode
        )
        store._batch_context.get_mix_decode_cache_for_splitfuse.return_value = mock_decode_result

        model_inputs, sampling_metadata, q_lens, trace_ids = store.concatenate_mix(metadata, context_handles, hit_mask)

        np.testing.assert_array_equal(model_inputs.input_ids, [100, 200])
        np.testing.assert_array_equal(model_inputs.position_ids, [2, 1])
        np.testing.assert_array_equal(model_inputs.slots, [1000, 2000])
        np.testing.assert_array_equal(model_inputs.context_length, [3, 2])
        self.assertEqual(model_inputs.max_seq_len, max(5, 3))  # 5 vs 3 → 5
        np.testing.assert_array_equal(q_lens, [1, 1])  # all decode → q_len=1
        self.assertIsNone(sampling_metadata)

        store._batch_context.update_context_for_splitfuse.assert_called_once()

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_concatenate_mix_prefill_only(self):
        tokenizer = MagicMock()
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 50
        self.kvcache_settings.block_size = 128
        self.batch_config = CacheConfig(
            cache_size=50,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = [Mock(group_size=1, rank=0), Mock(group_size=1, rank=0)]
        self.context_params = ContextParams(distributed=False)
        store = TGInferContextStore(
            kvcache_settings=self.kvcache_settings,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device,
            context_params=self.context_params,
            tokenizer=tokenizer,
            tokenizer_sliding_window_size=3
        )
        
        mock_batch_ctx = MagicMock(spec=BatchContext)
        store._batch_context = mock_batch_ctx

        def mock_b2s(block_table):
            block_id = block_table[0]
            return np.arange(block_id * 128, block_id * 128 + 128, dtype=np.int32).reshape(1, 128)
        
        mock_batch_ctx.block_table_to_slots.side_effect = mock_b2s

        metadata = MagicMock()
        metadata.input_ids = [1, 2, 3, 4, 5]
        metadata.total_seq_num = 5
        metadata.batch_is_prefill = np.array([True, True])
        metadata.batch_seq_len = np.array([3, 2])
        metadata.max_seq_len = 4
        metadata.has_sampling = False
        metadata.split_start_position = np.array([0, 0])
        metadata.split_end_position = np.array([3, 2])
        metadata.batch_block_tables = np.array([[10], [20]])
        metadata.batch_dp_rank_ids = np.array([0, 0])
        metadata.all_sequence_ids = np.array([100, 101])
        metadata.batch_sequence_ids = np.array([[100], [101]])
        context_handles = np.array([10, 11], dtype=np.int64)

        model_inputs, _, q_lens, _ = store.concatenate_mix(metadata, context_handles, hit_mask=None)

        expected_slots = np.array([
            10*128 + 0, 10*128 + 1, 10*128 + 2,   # req0: positions 0,1,2 in block 10
            20*128 + 0, 20*128 + 1                # req1: positions 0,1 in block 20
        ], dtype=np.int32)

        np.testing.assert_array_equal(model_inputs.slots, expected_slots)
        np.testing.assert_array_equal(model_inputs.position_ids, [0, 1, 2, 0, 1])
        np.testing.assert_array_equal(model_inputs.context_length, [3, 2])
        np.testing.assert_array_equal(q_lens, [3, 2])
        self.assertEqual(model_inputs.max_seq_len, 4)

        mock_batch_ctx.block_table_to_slots.assert_any_call(np.array([10]))
        mock_batch_ctx.block_table_to_slots.assert_any_call(np.array([20]))

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='ms')
    def test_concatenate_mix_mixed_prefill_decode(self):
        tokenizer = MagicMock()
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 100
        self.kvcache_settings.block_size = 128
        self.batch_config = CacheConfig(
            cache_size=100,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10
        )
        self.spcp_info = [Mock(group_size=1, rank=0), Mock(group_size=1, rank=0)]
        self.context_params = ContextParams(distributed=False)
        store = TGInferContextStore(
            kvcache_settings=self.kvcache_settings,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device,
            context_params=self.context_params,
            tokenizer=tokenizer,
            tokenizer_sliding_window_size=3
        )
        
        mock_batch_ctx = MagicMock(spec=BatchContext)
        store._batch_context = mock_batch_ctx

        # Mock block_table_to_slots for prefill requests
        def mock_b2s(block_table):
            block_id = block_table[0]
            return np.arange(block_id * 128, block_id * 128 + 128, dtype=np.int32).reshape(1, 128)
        mock_batch_ctx.block_table_to_slots.side_effect = mock_b2s

        # Mock decode cache: for 3 decode requests (indices 0,1,3)
        mock_decode_result = (
            np.array([990, 991, 993], dtype=np.int64),     # input_ids_decode
            7,                                             # max_seq_len_decode
            np.array([10, 11, 13], dtype=np.int32),        # position_ids_decode
            np.array([15, 16, 18], dtype=np.int32),        # input_lengths_decode
            np.array([770, 771, 773], dtype=np.int32)      # slots_decode
        )
        mock_batch_ctx.get_mix_decode_cache_for_splitfuse.return_value = mock_decode_result

        # Total tokens:
        # - DP0: decode(1) + decode(1) + prefill(2) → 1+1+2 = 4 tokens
        # - DP1: decode(1) + prefill(1) + prefill(3) → 1+1+3 = 5 tokens
        # Total = 9 tokens
        metadata = MagicMock()
        metadata.input_ids = list(range(9))  # [0,1,2,3,4,5,6,7,8] — placeholders, decode will be overwritten
        metadata.total_seq_num = 9
        metadata.batch_dp_rank_ids = np.array([0, 0, 0, 1, 1, 1])                       # 2 DP domains
        metadata.batch_is_prefill = np.array([False, False, True, False, True, True])   # decode before prefill in each DP
        metadata.batch_seq_len = np.array([1, 1, 2, 1, 1, 3])                           # token count per request
        metadata.max_seq_len = 8
        metadata.has_sampling = False
        metadata.all_sequence_ids = np.array([400, 401, 402, 403, 404, 405])
        metadata.batch_sequence_ids = np.array([[400], [401], [402], [403], [404], [405]])

        # split_start/end: only used for prefill; decode values are ignored (set to 0)
        metadata.split_start_position = np.array([0, 0, 0, 0, 0, 0])
        metadata.split_end_position = np.array([0, 0, 2, 0, 1, 3])      # prefill lengths: req2=2, req4=1, req5=3

        # Block tables: one block per request (for simplicity)
        metadata.batch_block_tables = np.array([
            [10],  # req0 (decode) — used in get_mix_decode_cache
            [11],  # req1 (decode)
            [12],  # req2 (prefill) — used in block_table_to_slots
            [13],  # req3 (decode)
            [14],  # req4 (prefill)
            [15],  # req5 (prefill)
        ])

        context_handles = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
        hit_mask = None

        # Act
        model_inputs, sampling_metadata, q_lens, trace_ids = store.concatenate_mix(metadata, context_handles, hit_mask)

        # Assert
        # Expected token layout by request:
        # req0 (decode): token 0 → input=990, pos=10, slot=770
        # req1 (decode): token 1 → input=991, pos=11, slot=771
        # req2 (prefill): tokens 2-3 → input=[2,3], pos=[0,1], slots=[12*128+0, 12*128+1]
        # req3 (decode): token 4 → input=993, pos=13, slot=773
        # req4 (prefill): token 5 → input=5, pos=0, slot=14*128+0
        # req5 (prefill): tokens 6-8 → input=[6,7,8], pos=[0,1,2], slots=[15*128+0, +1, +2]

        expected_input_ids = np.array([
            990, 991, 2, 3, 993, 5, 6, 7, 8
        ], dtype=np.int64)

        expected_position_ids = np.array([
            10, 11, 0, 1, 13, 0, 0, 1, 2
        ], dtype=np.int32)

        expected_slots = np.array([
            770,                          # req0 decode
            771,                          # req1 decode
            12*128 + 0, 12*128 + 1,      # req2 prefill
            773,                          # req3 decode
            14*128 + 0,                   # req4 prefill
            15*128 + 0, 15*128 + 1, 15*128 + 2  # req5 prefill
        ], dtype=np.int32)

        expected_context_length = np.array([
            15, 16, 2, 18, 1, 3
        ], dtype=np.int32)  # from mock decode (15,16,18) and prefill seq_len (2,1,3)

        expected_q_lens = np.array([
            1, 1, 2, 1, 1, 3
        ], dtype=np.int32)  # decode → 1; prefill → seq_len

        # Assertions
        np.testing.assert_array_equal(model_inputs.input_ids, expected_input_ids)
        np.testing.assert_array_equal(model_inputs.position_ids, expected_position_ids)
        np.testing.assert_array_equal(model_inputs.slots, expected_slots)
        np.testing.assert_array_equal(model_inputs.context_length, expected_context_length)
        np.testing.assert_array_equal(q_lens, expected_q_lens)
        self.assertEqual(model_inputs.max_seq_len, max(8, 7))  # 8
        self.assertIsNone(sampling_metadata)

        # Verify calls
        # - block_table_to_slots called for prefill requests: req2 ([12]), req4 ([14]), req5 ([15])
        mock_batch_ctx.block_table_to_slots.assert_any_call(np.array([12]))
        mock_batch_ctx.block_table_to_slots.assert_any_call(np.array([14]))
        mock_batch_ctx.block_table_to_slots.assert_any_call(np.array([15]))
        assert mock_batch_ctx.block_table_to_slots.call_count == 3

        calls = mock_batch_ctx.get_mix_decode_cache_for_splitfuse.call_args_list
        self.assertEqual(len(calls), 1, "Expected exactly one call to get_mix_decode_cache_for_splitfuse")
        (actual_cache_ids, actual_decode_idx, actual_metadata, actual_hit_mask), _ = calls[0]

        np.testing.assert_array_equal(actual_cache_ids, context_handles[np.array([0, 1, 3])])
        np.testing.assert_array_equal(actual_decode_idx, np.array([0, 1, 3]))
        self.assertIs(actual_metadata, metadata)
        self.assertIs(actual_hit_mask, hit_mask)

if __name__ == '__main__':
    unittest.main()
