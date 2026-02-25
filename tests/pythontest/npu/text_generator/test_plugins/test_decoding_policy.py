# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
Test cases for simulate inference (虚推) logic in DecodingPolicy
"""
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from mindie_llm.text_generator.plugins.mtp.decoding_policy import DecodingPolicy
from mindie_llm.text_generator.utils.input_metadata import SIMULATE_SEQUENCE_ID


class TestDecodingPolicySimulateInference(unittest.TestCase):
    """Test cases for simulate inference (虚推) logic in DecodingPolicy"""
    
    def setUp(self):
        self.num_speculative_tokens = 2
        self.block_size = 128
        self.num_npu_blocks = 10
        
        self.generator_backend = MagicMock()
        self.generator_backend.cache_pool.kvcache_settings.num_npu_blocks = self.num_npu_blocks
        
        self.infer_context = MagicMock()
        self.infer_context.spcp_parallel_info.scp_size = 1
        self.infer_context.spcp_parallel_info.scp_rank = 0
        self.infer_context.get_mtp_last_token_num.return_value = np.array([1, 1], dtype=np.int32)
        self.infer_context.get_output_len_count.return_value = np.array([1, 1], dtype=np.int32)
        
        kv_slots = np.arange(self.num_npu_blocks * self.block_size).reshape(self.num_npu_blocks, -1)
        self.infer_context._batch_context = MagicMock()
        self.infer_context._batch_context.kv_slots = kv_slots
        self.infer_context.block_table_to_slots.side_effect = lambda x: kv_slots[x]
        self.infer_context.block_to_slots.side_effect = lambda block, offset: block * self.block_size + offset
        
        self.decoding_policy = DecodingPolicy(
            generator_backend=self.generator_backend,
            infer_context=self.infer_context,
            model_wrapper=MagicMock(),
            num_speculative_tokens=self.num_speculative_tokens,
            device_and_type=('cpu', torch.float16),
            plugin_data_param=MagicMock(),
            model_role='standard',
            eos_token_id=2,
            max_block_size=self.block_size
        )
    
    def test_get_mtp_draft_model_inputs_standard_with_simulate_request(self):
        """Test get_mtp_draft_model_inputs_standard with simulate inference request
        
        验证虚推请求时，slots 被正确设置为 -1
        """
        batch_size = 2
        speculative_len = self.num_speculative_tokens + 1
        slots_num_per_batch = 2 * self.num_speculative_tokens
        
        input_metadata = MagicMock()
        input_metadata.batch_size = batch_size
        input_metadata.batch_block_tables = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
        input_metadata.all_sequence_ids = [SIMULATE_SEQUENCE_ID, 12345]
        
        model_inputs = MagicMock()
        model_inputs.block_tables = np.array([[0, 1], [2, 3]], dtype=np.int32)
        model_inputs.context_length = np.array([10, 10], dtype=np.int32)
        model_inputs.position_ids = np.array([9, 9], dtype=np.int32)
        model_inputs.is_prefill = False
        model_inputs.adapter_ids = []
        model_inputs.dp_rank_ids = np.array([0, 0])
        
        cached_idx = np.array([0, 1], dtype=np.int32)
        
        self.infer_context.get_all_input_ids.return_value = np.full(200, 3, dtype=np.int32)
        self.infer_context.get_seq_lens.return_value = 10  # 返回标量整数
        self.infer_context.get_mtp_hidden_states.return_value = torch.randn(speculative_len, 768)
        
        mtp_model_inputs, _ = self.decoding_policy.get_mtp_draft_model_inputs_standard(
            model_inputs, input_metadata, cached_idx, hit_mask=None
        )
        
        # 验证虚推请求的 slots 被设置为 -1
        simulate_slots = mtp_model_inputs.slots[0:slots_num_per_batch]
        self.assertTrue(np.all(simulate_slots == -1), 
                        f"Simulate request slots should be -1, but got {simulate_slots}")
        
        # 验证虚推请求的 context_length 被正确设置
        self.assertEqual(mtp_model_inputs.context_length[0], self.num_speculative_tokens)
        
        # 验证虚推请求的 position_ids 从 0 开始
        simulate_position_ids = mtp_model_inputs.position_ids[0:speculative_len]
        expected_position_ids = np.arange(speculative_len)
        self.assertTrue(np.array_equal(simulate_position_ids, expected_position_ids))
    
    def test_get_mtp_main_model_inputs_with_simulate_request(self):
        """Test get_mtp_main_model_inputs with simulate inference request
        
        验证虚推请求在主模型输入中 slots 被正确设置为 -1
        """
        batch_size = 2
        input_len_per_batch = self.num_speculative_tokens + 1
        
        input_metadata = MagicMock()
        input_metadata.batch_size = batch_size
        input_metadata.batch_block_tables = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
        input_metadata.all_sequence_ids = [SIMULATE_SEQUENCE_ID, 12345]
        
        model_inputs = MagicMock()
        model_inputs.input_ids = np.array([100, 200], dtype=np.int64)
        model_inputs.block_tables = np.array([[0, 1], [2, 3]], dtype=np.int32)
        model_inputs.context_length = np.array([10, 10], dtype=np.int32)
        model_inputs.position_ids = np.array([9, 9], dtype=np.int32)
        model_inputs.max_seq_len = 20
        model_inputs.prefill_head_indices = np.array([0, 1])
        model_inputs.is_prefill = False
        model_inputs.adapter_ids = []
        model_inputs.dp_rank_ids = np.array([0, 0])
        
        cached_idx = np.array([0, 1], dtype=np.int32)
        
        new_model_inputs = self.decoding_policy.get_mtp_main_model_inputs(
            model_inputs, input_metadata, cached_idx
        )
        
        # 验证虚推请求的 slots 被设置为 -1
        simulate_slots = new_model_inputs.slots[0:input_len_per_batch]
        self.assertTrue(np.all(simulate_slots == -1))
        
        # 验证虚推请求的 input_ids 第一个为 0 (跳过了赋值逻辑)
        self.assertEqual(new_model_inputs.input_ids[0], 0)
    
    def test_get_mtp_draft_model_inputs_all_simulate_requests(self):
        """Test when all requests are simulate inference requests
        
        验证所有请求都是虚推时，所有 slots 都被设置为 -1
        """
        batch_size = 2
        speculative_len = self.num_speculative_tokens + 1
        
        input_metadata = MagicMock()
        input_metadata.batch_size = batch_size
        input_metadata.batch_block_tables = [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)]
        input_metadata.all_sequence_ids = [SIMULATE_SEQUENCE_ID, SIMULATE_SEQUENCE_ID]
        
        model_inputs = MagicMock()
        model_inputs.block_tables = np.array([[0, 1], [2, 3]], dtype=np.int32)
        model_inputs.context_length = np.array([10, 10], dtype=np.int32)
        model_inputs.position_ids = np.array([9, 9], dtype=np.int32)
        model_inputs.is_prefill = False
        model_inputs.adapter_ids = []
        model_inputs.dp_rank_ids = np.array([0, 0])
        
        cached_idx = np.array([0, 1], dtype=np.int32)
        
        self.infer_context.get_mtp_hidden_states.return_value = torch.randn(speculative_len, 768)
        
        mtp_model_inputs, _ = self.decoding_policy.get_mtp_draft_model_inputs_standard(
            model_inputs, input_metadata, cached_idx, hit_mask=None
        )
        
        self.assertTrue(np.all(mtp_model_inputs.slots == -1))
    
    def test_mixed_batch_simulate_in_middle(self):
        """Test batch with simulate request in the middle position
        
        验证虚推请求在批次中间位置时的正确处理
        """
        batch_size = 3
        speculative_len = self.num_speculative_tokens + 1
        slots_num_per_batch = 2 * self.num_speculative_tokens
        
        self.infer_context.get_mtp_last_token_num.return_value = np.array([1, 1, 1], dtype=np.int32)
        self.infer_context.get_output_len_count.return_value = np.array([1, 1, 1], dtype=np.int32)
        
        input_metadata = MagicMock()
        input_metadata.batch_size = batch_size
        input_metadata.batch_block_tables = [
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32)
        ]
        input_metadata.all_sequence_ids = [12345, SIMULATE_SEQUENCE_ID, 67890]
        
        model_inputs = MagicMock()
        model_inputs.block_tables = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
        model_inputs.context_length = np.array([10, 10, 10], dtype=np.int32)
        model_inputs.position_ids = np.array([9, 9, 9], dtype=np.int32)
        model_inputs.is_prefill = False
        model_inputs.adapter_ids = []
        model_inputs.dp_rank_ids = np.array([0, 0, 0])
        
        cached_idx = np.array([0, 1, 2], dtype=np.int32)
        
        self.infer_context.get_all_input_ids.return_value = np.full(200, 3, dtype=np.int32)
        self.infer_context.get_seq_lens.return_value = 10  # 返回标量整数
        self.infer_context.get_mtp_hidden_states.return_value = torch.randn(speculative_len, 768)
        
        mtp_model_inputs, _ = self.decoding_policy.get_mtp_draft_model_inputs_standard(
            model_inputs, input_metadata, cached_idx, hit_mask=None
        )
        
        # 只有中间批次 (index 1) 的 slots 应该是 -1
        simulate_start = 1 * slots_num_per_batch
        simulate_end = 2 * slots_num_per_batch
        simulate_slots = mtp_model_inputs.slots[simulate_start:simulate_end]
        self.assertTrue(np.all(simulate_slots == -1))
        
        # 第一个批次不应该全是 -1
        first_slots = mtp_model_inputs.slots[0:slots_num_per_batch]
        self.assertFalse(np.all(first_slots == -1))


if __name__ == "__main__":
    unittest.main()
