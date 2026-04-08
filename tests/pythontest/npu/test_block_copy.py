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

import torch
import numpy as np

from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.text_generator.utils.block_copy import BlockCopy

BACKEND_TYPE = BackendType.ATB


class TestBlockCopy(unittest.TestCase):
    def setUp(self):
        self.backend_type = BACKEND_TYPE
        num_layer = 100
        num_block = 20
        block_size = 128
        num_head = 8
        head_size = 8
        self.kv_cache = None
        if self.backend_type == BackendType.ATB:
            def to_tensor_torch(data_):
                return torch.tensor(data_).npu()

            self.to_tensor = to_tensor_torch
            k_cache = self.to_tensor(torch.randn(num_block, block_size, num_head, head_size).to(torch.bfloat16))
            v_cache = self.to_tensor(torch.randn(num_block, block_size, num_head, head_size).to(torch.bfloat16))
            self.kv_cache = [(k_cache, v_cache) for _ in range(num_layer)]
        elif self.backend_type == BackendType.MS:
            import mindspore as ms

            def to_tensor_ms(data_):
                return ms.Tensor(data_) if data_.size > 0 else None

            self.to_tensor = to_tensor_ms
        else:
            raise ValueError('No such backend type.')
        self.block_copy_ops = BlockCopy('atb', self.kv_cache, self.to_tensor)
        self.block_copy_ops.block_copy = self.block_copy_ops.golden_copy_block
        self.src_dst_map = np.array([[1, 2]], dtype=np.int64)

    def test_copy_blocks_init(self):
        block_copy_ops = BlockCopy('atb', self.kv_cache, self.to_tensor)
        self.assertTrue(block_copy_ops.block_copy_init)

    def test_copy_blocks_ops(self):
        self.block_copy_ops.copy_blocks(self.src_dst_map)
        for _, (key, value) in enumerate(self.kv_cache):
            self.assertTrue(torch.allclose(key[1, ...], key[2, ...]))
            self.assertTrue(torch.allclose(value[1, ...], value[2, ...]))

if __name__ == '__main__':
    unittest.main()