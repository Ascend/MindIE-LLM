# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mindie_llm.text_generator.utils.batch_context import DictContext, NdarrayContext, BatchContext
from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
from mindie_llm.text_generator.utils.config import ContextParams, CacheConfig, SpCpParallelInfo, DEFAULT_SAMPLING_PARAMS
from mindie_llm.text_generator.utils.input_metadata import SAMPLING_DTYPE, InputMetadata
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata
from mindie_llm.text_generator.utils.sampling_output import SamplingOutput


class TestDictContext(unittest.TestCase):
    def setUp(self):
        self.dict_ctx = DictContext()
        self.metadata = Mock()
        self.metadata.adapter_ids = ["adapter1", None, "adapter2"]
        self.metadata.trace_ids = ["trace1", None, "trace3"]
        self.metadata.batch_request_ids = ["req1", "req2", "req3"]
        self.context_handles = [0, 1, 2]

    def test_initialization(self):
        """测试初始化后context字典为空"""
        self.assertEqual(self.dict_ctx.stopping_criteria, {})
        self.assertEqual(self.dict_ctx.string_stopping_criteria, {})
        self.assertEqual(self.dict_ctx.output_texts, {})
        self.assertEqual(self.dict_ctx.trace_ids, {})

    def test_add_context(self):
        """测试添加context数据"""
        self.dict_ctx.add_context(self.context_handles, self.metadata)

        self.assertEqual(self.dict_ctx.trace_ids[0], "trace1")
        self.assertEqual(self.dict_ctx.trace_ids[1], "req2")
        self.assertEqual(self.dict_ctx.trace_ids[2], "trace3")

    def test_clear_context(self):
        """测试清除指定context"""
        self.dict_ctx.add_context(self.context_handles, self.metadata)
        self.dict_ctx.clear_context([1])

        self.assertNotIn(1, self.dict_ctx.trace_ids)
        self.assertIn(0, self.dict_ctx.trace_ids)
        self.assertIn(2, self.dict_ctx.trace_ids)


class TestNdarrayContext(unittest.TestCase):
    def setUp(self):
        self.context_params = ContextParams(
            distributed=False,
            mtp_num_speculative_tokens=2,
            mtp_hidden_size=64,
        )
        self.spcp_info = SpCpParallelInfo(
            sp_parallel_info=Mock(group_size=1, rank=0),
            cp_parallel_info=Mock(group_size=1, rank=0)
        )
        self.default_sampling_params = np.array(tuple([1.0, 0.0, 0.0, 1.0, 1000, 1.0, False, 0]), dtype=SAMPLING_DTYPE)
        self.cache_config = CacheConfig()
        self.ndarray_ctx = NdarrayContext(
            context_params=self.context_params,
            default_sampling_params=self.default_sampling_params,
            cache_config=self.cache_config,
            spcp_parallel_info=self.spcp_info,
            capacity=self.cache_config.cache_size
        )

    def test_initialization(self):
        """测试初始化时数组容量和池状态"""
        self.assertEqual(len(self.ndarray_ctx.pool), self.cache_config.cache_size - 1)  # slot 0 is reserved
        self.assertEqual(self.ndarray_ctx.last_input_ids.shape, (self.cache_config.cache_size,))
        self.assertEqual(self.ndarray_ctx.seq_lens.dtype, np.int32)
        self.assertEqual(self.ndarray_ctx.sampling_params.shape, (self.cache_config.cache_size,))

    def test_allocate_slot(self):
        """测试分配slot"""
        slot1 = self.ndarray_ctx.allocate_slot()
        self.assertEqual(slot1, self.cache_config.cache_size - 1)
        self.assertEqual(len(self.ndarray_ctx.pool), self.cache_config.cache_size - 2)  # slot 0 is reserved

        for _ in range(self.cache_config.cache_size - 2):
            self.ndarray_ctx.allocate_slot()
        self.assertEqual(len(self.ndarray_ctx.pool), 0)

        slot_new = self.ndarray_ctx.allocate_slot()
        self.assertEqual(slot_new, self.cache_config.cache_size * 2 - 1)
        self.assertEqual(self.ndarray_ctx.cache_config.cache_size, self.cache_config.cache_size)

    def test_free_slot(self):
        """测试释放slot"""
        slot = self.ndarray_ctx.allocate_slot()
        self.ndarray_ctx._free_slot(slot)
        self.assertIn(slot, self.ndarray_ctx.pool)

    def test_grow_capacity(self):
        """测试容量扩容"""
        original_capacity = self.ndarray_ctx.cache_config.cache_size
        self.ndarray_ctx._grow_capacity()

        self.assertEqual(self.ndarray_ctx.last_input_ids.shape, (original_capacity * 2,))
        self.assertEqual(self.ndarray_ctx.seq_lens.shape, (original_capacity * 2,))
        self.assertEqual(self.ndarray_ctx.mtp_hidden_states.shape,
                         (original_capacity * 2, 3, 64))

    def test_clear_context(self):
        """测试清除context"""
        slot = self.ndarray_ctx.allocate_slot()
        self.ndarray_ctx.clear_context(slot)
        self.assertIn(slot, self.ndarray_ctx.pool)


class TestBatchContext(unittest.TestCase):
    def setUp(self):
        self.device = "npu"
        self.kvcache_settings = Mock(spec=KVCacheSettings)
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.block_size = 4
        self.batch_config = CacheConfig(
            cache_size=4,
            pad_token_id=0,
            max_seq_len=10,
            max_gen_len=10,
            vocab_size=10000
        )
        self.spcp_info = SpCpParallelInfo(
            sp_parallel_info=Mock(group_size=1, rank=0),
            cp_parallel_info=Mock(group_size=1, rank=0)
        )
        self.context_params = ContextParams(distributed=False)
        tokenizer = Mock()
        self.batch_ctx = BatchContext(
            kvcache_settings=self.kvcache_settings,
            context_params=self.context_params,
            batch_context_config=self.batch_config,
            spcp_parallel_info=self.spcp_info,
            device=self.device,
            tokenizer=tokenizer,
            tokenizer_sliding_window_size=3
        )

    def test_initialization(self):
        """测试BatchContext初始化"""
        self.assertEqual(self.batch_ctx.kv_slots.shape, (2, 4))
        self.assertTrue(np.array_equal(
            self.batch_ctx.kv_slots,
            np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        ))

        self.assertIsInstance(self.batch_ctx.all_ndarray_context, NdarrayContext)

    def test_allocate_context_slot(self):
        """测试分配context_slot"""
        seq_id = 123
        slot = self.batch_ctx.allocate_context_slot(seq_id)

        self.assertIn(seq_id, self.batch_ctx.sequence_context_slot_map)
        self.assertEqual(self.batch_ctx.sequence_context_slot_map[seq_id], slot)
        self.assertNotIn(slot, self.batch_ctx.all_ndarray_context.pool)

    def test_get_context_slot(self):
        """测试获取context_slot"""
        seq_id = 456
        slot1 = self.batch_ctx.get_context_slot(seq_id, is_prefill=True)
        self.assertEqual(self.batch_ctx.sequence_context_slot_map[seq_id], slot1)

        slot2 = self.batch_ctx.get_context_slot(seq_id, is_prefill=False)
        self.assertEqual(slot1, slot2)

        with self.assertRaises(RuntimeError):
            self.batch_ctx.get_context_slot("invalid_seq", is_prefill=False)

    @patch.object(NdarrayContext, 'clear_context')
    @patch.object(DictContext, 'clear_context')
    def test_clear_context_by_handles(self, mock_dict_clear, mock_ndarray_clear):
        """测试通过handle清除context"""
        handle = np.array([0, 1])
        self.batch_ctx.clear_context_by_handles(handle)

        mock_ndarray_clear.assert_called_once_with(handle)
        mock_dict_clear.assert_called_once_with(handle)

    def test_block_mapping_methods(self):
        """测试block_to_slots和block_table_to_slots"""
        block_id = np.array([0, 1])
        offset = np.array([2, 3])
        slots = self.batch_ctx.block_to_slots(block_id, offset)
        self.assertTrue(np.array_equal(slots, [2, 7]))

        block_table = np.array([0, 1])
        slots = self.batch_ctx.block_table_to_slots(block_table)
        self.assertTrue(np.array_equal(slots, [[0, 1, 2, 3], [4, 5, 6, 7]]))

    def test_sync_sampling_token_ids(self):
        """测试同步采样的token ids"""
        seq_id = "seq_sync"
        cache_id = self.batch_ctx.allocate_context_slot(seq_id)
        context_handles = np.array([cache_id])

        self.batch_ctx.all_ndarray_context.all_input_ids[cache_id, :2] = [100, 200]
        self.batch_ctx.all_ndarray_context.all_output_ids[cache_id, :1] = [300]
        self.batch_ctx.all_ndarray_context.output_len_count[cache_id] = 1

        sampling_meta = Mock(spec=SamplingMetadata)
        updated_meta = self.batch_ctx.sync_sampling_token_ids(
            context_handles=context_handles,
            sampling_metadata=sampling_meta,
            max_seq_len=2
        )

        # 验证是否调用了update_token_ids
        updated_meta.update_token_ids.assert_called_once()
        args, _ = updated_meta.update_token_ids.call_args
        self.assertTrue(np.array_equal(args[0], [[100, 200]]))  # all_input_ids
        self.assertTrue(np.array_equal(args[1], [[300]]))  # all_output_ids

    def test_init_default_sampling_params(self):
        """测试默认采样参数初始化"""
        self.assertEqual(self.batch_ctx.default_sampling_params['temperature'], 1)
        self.assertEqual(self.batch_ctx.default_sampling_params['top_k'], 0)
        self.assertEqual(self.batch_ctx.default_sampling_params['top_p'], 1.0)

        self.assertEqual(
            self.batch_ctx.default_sampling_params['repetition_penalty'],
            DEFAULT_SAMPLING_PARAMS['repetition_penalty']
        )

    def test_first_update_context_given_base_request(self):
        updated_ndarrays = (
            np.array([2]),  # last_position_ids
            np.array([3]),  # input_lengths
            np.array([0]),  # prefill_new_tokens
        )
        input_metadata = Mock(spec=InputMetadata)
        input_metadata.batch_seeds = np.array(None)
        input_metadata.batch_n = np.array([1.0])
        input_metadata.batch_best_of = np.array([1.0])
        input_metadata.batch_use_beam_search = np.array([0.0])
        input_metadata.batch_ignore_eos = np.array([None])
        input_metadata.batch_skip_special_tokens = np.array([True])
        input_metadata.batch_include_stop = np.array([None])
        input_metadata.batch_stop_strings = np.array([None])
        input_metadata.batch_stop_token_ids = np.array([None])
        input_metadata.batch_adapter_ids = np.array([None])
        input_metadata.trace_ids = [None]
        input_metadata.batch_request_ids = np.array(["0"])
        input_metadata.is_dummy_batch = False
        input_metadata.adapter_ids = None

        self.batch_ctx.update_context(
            context_handles=np.array([0]),
            updated_ndarrays=updated_ndarrays,
            input_metadata=input_metadata,
            sampling_args=None,
            is_pd_separate=False,
            is_first_update=True,
        )

        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.last_position_ids, np.array([2, 0, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.seq_lens, np.array([3, 0, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.cpu_cached_seq_idx, np.array([[2], [0], [0], [0]])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.seeds, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.n, np.array([1, 1, 1, 1])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.best_of, np.array([1, 1, 1, 1])))
        self.assertTrue(
            np.array_equal(self.batch_ctx.all_ndarray_context.use_beam_search, np.array([False, False, False, False]))
        )
        self.assertTrue(
            np.array_equal(self.batch_ctx.all_ndarray_context.ignore_eos, np.array([False, False, False, False]))
        )
        self.assertTrue(
            np.array_equal(self.batch_ctx.all_ndarray_context.skip_special_tokens, np.array([True, True, True, True]))
        )
        self.assertTrue(
            np.array_equal(self.batch_ctx.all_ndarray_context.include_stop, np.array([False, False, False, False]))
        )
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.last_input_ids, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.used_block_idx, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.used_block_offset, np.array([0, 0, 0, 0])))

        self.assertDictEqual(self.batch_ctx.all_dict_context.output_texts, {})
        self.assertDictEqual(self.batch_ctx.all_dict_context.string_stopping_criteria, {})
        self.assertDictEqual(self.batch_ctx.all_dict_context.stopping_criteria, {})
        self.assertDictEqual(self.batch_ctx.all_dict_context.lora_adapter_id, {})
        self.assertDictEqual(self.batch_ctx.all_dict_context.trace_ids, {0: "0"})

    def test_update_context_given_base_request(self):
        input_metadata = Mock(spec=InputMetadata)
        input_metadata.is_dummy_batch = False
        input_metadata.batch_is_prefill = None
        sampling_metadata = Mock(spec=SamplingMetadata)
        sampling_metadata.is_prefill = True
        sampling_output = Mock(spec=SamplingOutput)
        sampling_output.token_ids = np.array([[30]])
        sampling_output.num_new_tokens = np.array([1])
        sampling_output.logprobs = np.array([[-9999.0]])
        sampling_output.repetition_indices = np.array([0])
        sampling_output.seeds = None
        self.batch_ctx.all_ndarray_context.output_len_count = np.array([0, 0, 0, 0])
        self.batch_ctx.all_ndarray_context.last_position_ids = np.array([0, 2, 0, 0])
        self.batch_ctx.all_ndarray_context.seq_lens = np.array([0, 3, 0, 0])
        self.batch_ctx.all_ndarray_context.use_beam_search = np.array([False, False, False, False])
        self.batch_ctx.all_ndarray_context.cpu_cached_seq_idx = np.array([[0], [2], [0], [0]])
        self.batch_ctx.all_ndarray_context.all_output_ids = np.array(
            [
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
            ]
        )
        self.batch_ctx.all_ndarray_context.all_input_ids = np.array(
            [
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                [14623, 525, 498, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
            ]
        )

        self.batch_ctx.update_context(
            context_handles=np.array([1]),
            updated_ndarrays=None,
            input_metadata=input_metadata,
            sampling_args=(sampling_metadata, sampling_output),
            is_pd_separate=False,
            is_first_update=False,
        )

        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.last_input_ids, np.array([0, 30, 0, 0])))
        self.assertTrue(
            np.array_equal(
                self.batch_ctx.all_ndarray_context.all_input_ids,
                np.array(
                    [
                        [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                        [14623, 525, 498, 30, 151936, 151936, 151936, 151936, 151936, 151936],
                        [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                        [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.array_equal(
                self.batch_ctx.all_ndarray_context.all_output_ids,
                np.array(
                    [
                        [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                        [30, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                        [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                        [151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936, 151936],
                    ]
                ),
            )
        )
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.seeds, np.array([0, 0, 0, 0])))
        self.assertTrue(
            np.array_equal(self.batch_ctx.all_ndarray_context.cumulative_logprobs, np.array([0.0, 0.0, 0.0, 0.0]))
        )
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.output_len_count, np.array([0, 1, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.seq_lens, np.array([0, 4, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.cpu_cached_seq_idx, np.array([[0], [3], [0], [0]])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.last_position_ids, np.array([0, 3, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.used_block_idx, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(self.batch_ctx.all_ndarray_context.used_block_offset, np.array([0, 3, 0, 0])))

    def test_join_context_given_base_context(self):
        metadata = Mock(spec=InputMetadata)
        metadata.all_sequence_ids = np.array([100])
        metadata.batch_block_tables = np.array([[0]])
        self.batch_ctx.all_ndarray_context.used_block_idx = np.array([0, 0, 0, 0])
        self.batch_ctx.all_ndarray_context.used_block_offset = np.array([3, 0, 0, 0])
        self.batch_ctx.all_ndarray_context.last_input_ids = np.array([198, 0, 0, 0])
        self.batch_ctx.all_ndarray_context.last_position_ids = np.array([3, 0, 0, 0])
        self.batch_ctx.all_ndarray_context.seq_lens = np.array([4, 0, 0, 0])
        self.batch_ctx.all_dict_context.lora_adapter_id = {}
        ret = self.batch_ctx.join_context(
            context_handles=np.array([0]),
            metadata=metadata,
            hit_mask=None
        )
        self.assertTrue(np.array_equal(ret[0], np.array([198])))
        self.assertTrue(np.array_equal(ret[1], np.array([3])))
        self.assertTrue(np.array_equal(ret[2], np.array([3])))
        self.assertTrue(np.array_equal(ret[3], np.array([4])))
        self.assertEqual(ret[4], 4)
        self.assertListEqual(ret[5], [None])

    def test_fork_context_basic(self):
        """测试 fork_context 正确复制父 context 的数据到子 context"""
        parent_handles = np.array([0, 1], dtype=np.int32)
        child_handles = np.array([2, 3], dtype=np.int32)

        nd = self.batch_ctx.all_ndarray_context
        dc = self.batch_ctx.all_dict_context

        # === 设置 parent 数据 ===
        nd.last_input_ids[parent_handles] = [50256, 100]
        nd.last_position_ids[parent_handles] = [10, 20]
        nd.seq_lens[parent_handles] = [5, 8]
        nd.cpu_cached_seq_idx[parent_handles] = [[0], [1]]
        nd.output_len_count[parent_handles] = [1, 2]
        nd.used_block_idx[parent_handles] = [0, 1]
        nd.used_block_offset[parent_handles] = [2, 3]
        nd.cumulative_logprobs[parent_handles] = [0.1, 0.3]
        nd.num_top_tokens[parent_handles] = [1, 2]

        # all_input_ids / all_output_ids
        nd.all_input_ids[0, :5] = [1, 2, 3, 4, 5]
        nd.all_input_ids[1, :8] = [10, 20, 30, 40, 50, 60, 70, 80]
        nd.all_output_ids[0, :1] = [100]
        nd.all_output_ids[1, :2] = [200, 300]

        # sampling_params
        temp_params = np.array(
            [(1.0, 0.0, 0.0, 1.0, 1000, 1.0, False, 0), (1.2, 0.8, 30, 0.9, 800, 1.0, True, 1)], dtype=SAMPLING_DTYPE
        )
        nd.sampling_params[parent_handles] = temp_params


        # seeds, best_of, n 等
        nd.seeds[parent_handles] = [123, 456]
        nd.best_of[parent_handles] = [1, 3]
        nd.n[parent_handles] = [1, 2]
        nd.use_beam_search[parent_handles] = [False, True]
        nd.ignore_eos[parent_handles] = [True, False]
        nd.include_stop[parent_handles] = [False, True]
        nd.skip_special_tokens[parent_handles] = [True, False]

        # DictContext
        dc.output_texts[0] = "Hello"
        dc.output_texts[1] = "World"
        dc.trace_ids[0] = "trace-A"
        dc.trace_ids[1] = "trace-B"
        dc.lora_adapter_id[0] = "lora-A"
        dc.lora_adapter_id[1] = "lora-B"
        dc.stopping_criteria[0] = "stop1"
        dc.string_stopping_criteria[1] = "str_stop2"

        # === 执行 fork ===
        self.batch_ctx.fork_context(child_handles, parent_handles)

        # === 验证 NdarrayContext ===
        np.testing.assert_array_equal(nd.seq_lens[child_handles], [5, 8])
        np.testing.assert_array_equal(nd.used_block_idx[child_handles], [0, 1])
        np.testing.assert_array_equal(nd.all_input_ids[2, :5], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(nd.sampling_params[child_handles], temp_params)
        np.testing.assert_array_equal(nd.seeds[child_handles], [123, 456])
        np.testing.assert_array_equal(nd.best_of[child_handles], [1, 3])
        np.testing.assert_array_equal(nd.n[child_handles], [1, 2])


        # === 验证 DictContext ===
        self.assertEqual(dc.output_texts.get(2), "Hello")
        self.assertEqual(dc.output_texts.get(3), "World")
        self.assertEqual(dc.trace_ids.get(2), "trace-A")
        self.assertEqual(dc.trace_ids.get(3), "trace-B")
        self.assertEqual(dc.lora_adapter_id.get(2), "lora-A")
        self.assertEqual(dc.lora_adapter_id.get(3), "lora-B")
        self.assertEqual(dc.stopping_criteria.get(2), "stop1")
        self.assertIsNone(dc.stopping_criteria.get(3))
        self.assertEqual(dc.string_stopping_criteria.get(3), "str_stop2")
        self.assertIsNone(dc.string_stopping_criteria.get(2))

        # === 验证索引映射正确性 ===
        self.assertEqual(nd.seq_lens[2], 5)  # child[2] ← parent[0]
        self.assertEqual(nd.seq_lens[3], 8)  # child[3] ← parent[1]
        self.assertEqual(dc.output_texts.get(2), "Hello")
        self.assertEqual(dc.trace_ids.get(3), "trace-B")


if __name__ == "__main__":
    unittest.main()
