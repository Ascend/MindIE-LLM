import unittest
from unittest.mock import Mock, patch
import numpy as np

from mindie_llm.text_generator.utils.tg_infer_context_store import TGInferContextStore
from mindie_llm.text_generator.utils.sampling_output import SamplingOutput


class TestTGInferContextStoreForkContext(unittest.TestCase):

    def setUp(self):
        self.mock_kvcache_settings = Mock()
        self.mock_batch_context_config = Mock()
        self.mock_context_params = Mock()

        self.mock_sp_info = Mock()
        self.mock_sp_info.group_size = 1
        self.mock_sp_info.local_rank = 0

        self.mock_cp_info = Mock()
        self.mock_cp_info.group_size = 1
        self.mock_cp_info.local_rank = 0

        self.mock_spcp_parallel_info = (self.mock_sp_info, self.mock_cp_info)
        self.mock_device = "mock_device"
        tokenizer = Mock()

        with patch("mindie_llm.text_generator.utils.tg_infer_context_store.BatchContext"):
            self.store = TGInferContextStore(
                kvcache_settings=self.mock_kvcache_settings,
                batch_context_config=self.mock_batch_context_config,
                spcp_parallel_info=self.mock_spcp_parallel_info,
                device=self.mock_device,
                context_params=self.mock_context_params,
                tokenizer=tokenizer,
                tokenizer_sliding_window_size=3
            )

        self.mock_batch_context = Mock()
        self.store._batch_context = self.mock_batch_context

        # 使用可调用函数作为 side_effect，根据 (sequence_id, is_prefill) 返回 handle
        self.handle_map = {}

        def mock_get_context_slot(sequence_id, is_prefill):
            if sequence_id in self.handle_map:
                return self.handle_map[sequence_id]
            if is_prefill:
                handle = sequence_id + 1000
                self.handle_map[sequence_id] = handle
                return handle
            raise RuntimeError(f"no prefilled context for {sequence_id}")

        self.mock_batch_context.get_context_slot = Mock(side_effect=mock_get_context_slot)
        self.mock_batch_context.fork_context = Mock()  # 保持为 Mock

    def make_sampling_output(self, sequence_ids_list, parent_sequence_ids_list):
        batch_size = len(sequence_ids_list)
        return SamplingOutput(
            sequence_ids=np.array(sequence_ids_list, dtype=np.int32),
            parent_sequence_ids=np.array(parent_sequence_ids_list, dtype=np.int32),
            group_indices=[(i, i) for i in range(batch_size)],
            repeating_indices=np.zeros(batch_size, dtype=np.int32),
            token_ids=np.zeros((batch_size, 0), dtype=np.int32),
            logprobs=np.zeros((batch_size, 0), dtype=np.float32),
            top_token_ids=np.zeros((batch_size, 0), dtype=np.int32),
            top_logprobs=np.zeros((batch_size, 0), dtype=np.float32),
            num_new_tokens=np.ones(batch_size, dtype=np.int32),
            num_top_tokens=np.zeros(batch_size, dtype=np.int32),
            seeds=np.zeros(batch_size, dtype=np.uint64),
            cumulative_logprobs=np.zeros(batch_size, dtype=np.float32),
        )

    def test_fork_context_normal_case(self):
        """测试正常 fork 场景"""
        self.mock_batch_context.get_context_slot(1, True)  # 确保 parent 1 存在

        sampling_output = self.make_sampling_output([2, 3, 4, -1, 5], [1, 1, 2, 1, 3])
        result = self.store.fork_context(sampling_output)

        expected = np.array([1002, 1003, 1004, -1, 1005], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

        self.mock_batch_context.fork_context.assert_called_once()
        kwargs = self.mock_batch_context.fork_context.call_args[1]
        np.testing.assert_array_equal(
            kwargs["children_context_handles"], np.array([1002, 1003, 1004, 1005], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            kwargs["parents_context_handles"], np.array([1001, 1001, 1002, 1003], dtype=np.int32)
        )

    def test_fork_context_parent_not_exists_raises(self):
        """测试 parent 不存在时报错"""
        sampling_output = self.make_sampling_output([99], [999])

        with self.assertRaises(RuntimeError) as ctx:
            self.store.fork_context(sampling_output)
        self.assertIn("has no context", str(ctx.exception))

    def test_fork_context_padding_skipped(self):
        """测试 padding 被跳过"""
        sampling_output = self.make_sampling_output([-1, 8, -1, 9], [1, 1, 1, 8])

        self.mock_batch_context.get_context_slot(1, True)  # parent 1
        self.mock_batch_context.get_context_slot(8, True)  # parent 8

        result = self.store.fork_context(sampling_output)

        expected = np.array([-1, 1008, -1, 1009], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

        self.mock_batch_context.fork_context.assert_called_once()
        kwargs = self.mock_batch_context.fork_context.call_args[1]
        np.testing.assert_array_equal(kwargs["children_context_handles"], np.array([1008, 1009], dtype=np.int32))
        np.testing.assert_array_equal(kwargs["parents_context_handles"], np.array([1001, 1008], dtype=np.int32))

    def test_fork_context_multiple_children_same_parent(self):
        """测试多个 child 共享 parent"""
        sampling_output = self.make_sampling_output([2, 3, 4], [1, 1, 1])
        self.mock_batch_context.get_context_slot(1, True)  # parent 1 存在

        result = self.store.fork_context(sampling_output)

        expected = np.array([1002, 1003, 1004], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

        self.mock_batch_context.fork_context.assert_called_once()
        kwargs = self.mock_batch_context.fork_context.call_args[1]
        np.testing.assert_array_equal(kwargs["children_context_handles"], np.array([1002, 1003, 1004], dtype=np.int32))
        np.testing.assert_array_equal(kwargs["parents_context_handles"], np.array([1001, 1001, 1001], dtype=np.int32))

    def test_fork_context_self_continuation_no_fork(self):
        """测试 self-continuation 不 fork"""
        sampling_output = self.make_sampling_output([5], [5])
        result = self.store.fork_context(sampling_output)
        np.testing.assert_array_equal(result, np.array([1005], dtype=np.int32))
        self.mock_batch_context.fork_context.assert_not_called()

    def test_fork_context_reuse_existing_sequence(self):
        """测试复用已存在的 sequence"""
        output1 = self.make_sampling_output([88], [77])
        self.mock_batch_context.get_context_slot(77, True)
        result1 = self.store.fork_context(output1)
        np.testing.assert_array_equal(result1, np.array([1088], dtype=np.int32))

        output2 = self.make_sampling_output([88], [77])
        result2 = self.store.fork_context(output2)
        np.testing.assert_array_equal(result2, np.array([1088], dtype=np.int32))

        self.assertEqual(self.mock_batch_context.fork_context.call_count, 2)