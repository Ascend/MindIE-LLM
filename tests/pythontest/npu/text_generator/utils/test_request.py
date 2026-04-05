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

from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams


class TestRequest(unittest.TestCase):
    
    def setUp(self):
        # 创建测试所需的 GenerationParams 对象
        self.generation_params = GenerationParams(max_new_tokens=50, best_of=3, ignore_eos=True)
        self.seq_id = 1
        self.req_id = np.array([1])
        self.sampling_params = np.array([0.5, 0.5])  # 例如一些采样参数

    def test_initialization(self):
        # 测试初始化后的属性值
        request = Request(
            req_id=1,
            seq_id=self.seq_id,
            input_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
            generation_params=self.generation_params,
            has_sampling=True,
            sampling_params=self.sampling_params
        )
        self.assertEqual(request.req_id, 1)
        self.assertEqual(request.sequences[self.seq_id].seq_id, self.seq_id)
        self.assertEqual(request.best_of, 3)
        self.assertEqual(request.max_new_tokens, 50)
        self.assertTrue(request.ignore_eos)
        self.assertEqual(request.skip_special_tokens, self.generation_params.skip_special_tokens)
        self.assertTrue(np.array_equal(request.sampling_params, self.sampling_params))

    def test_from_warmup(self):
        # 测试 from_warmup 类方法
        input_len = 5
        max_output_len = 10
        warmup_request = Request.from_warmup(input_len, max_output_len)
        
        self.assertEqual(warmup_request.req_id, 0)
        self.assertEqual(warmup_request.input_ids.tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(warmup_request.max_new_tokens, 10)
        self.assertEqual(warmup_request.input_length, input_len)

    def test_request_from_token(self):
        # 测试 request_from_token 类方法
        input_ids = [10, 20, 30]
        sampling_params = np.array([0.7, 0.3])
        generation_params = GenerationParams(max_new_tokens=15, best_of=2)
        
        token_request = Request.request_from_token(
            input_ids, sampling_params, generation_params, req_id=self.req_id, seq_id=self.seq_id)
        
        self.assertEqual(token_request.req_id, 1)
        self.assertEqual(token_request.sequences[self.seq_id].seq_id, self.req_id)
        self.assertTrue(np.array_equal(token_request.input_ids, np.array([10, 20, 30])))
        self.assertEqual(token_request.max_new_tokens, 15)
        self.assertEqual(token_request.best_of, 2)
        self.assertTrue(np.array_equal(token_request.sampling_params, sampling_params))
    
    def test_request_lifecycle(self):
        input_len = 128 * 6
        max_output_len = 2
        max_placeholder_num = 1
        request = Request.from_warmup(input_len, max_output_len, max_placeholder_num)
        request.build(dp_rank_id=0, scp_size=4, block_size=128, is_mix_model=False)
        gold_block_table = np.array([[0, 0, -1, -1, -1, -1],
                                     [0, -1, -1, -1, -1, -1],
                                     [0, -1, -1, -1, -1, -1],
                                     [0, 0, -1, -1, -1, -1]])
        
        self.assertEqual(request.dp_rank_id, 0)
        self.assertEqual(request.sp_tokens.size, 4)
        self.assertEqual(request.sp_rank_id, 3)
        self.assertEqual(request.prefill_block_rank_id.size, 6)
        self.assertTrue(np.array_equal(request.block_tables, gold_block_table))
        self.assertEqual(len(request.computed_blocks), 4)
        self.assertEqual(len(request.remote_computed_blocks), 4)

        request.step(num_new_token=1, scp_size=4, block_size=128, is_mix_model=False)

        gold_block_table = np.array([[0, 0, 0, -1, -1, -1],
                                     [0, -1, -1, -1, -1, -1],
                                     [0, -1, -1, -1, -1, -1],
                                     [0, 0, -1, -1, -1, -1]])
        
        self.assertEqual(request.output_len, 1)
        self.assertEqual(request.max_placeholder_num, 1)
        self.assertTrue(request.is_append_block)
        self.assertEqual(request.block_rank_id, 0)
        self.assertTrue(np.array_equal(request.block_tables, gold_block_table))

    def test_update_with_chunkprefill_features(self):
        request = Request.from_warmup(100, 1)
        
        request.update_with_features(
            is_mix_model=True,
            scp_size=1,
            is_prefill=True
        )

        self.assertEqual(request.split_start_position, 0)
        self.assertEqual(request.split_end_position, 100)
        self.assertTrue(request.last_prompt)

if __name__ == '__main__':
    unittest.main()