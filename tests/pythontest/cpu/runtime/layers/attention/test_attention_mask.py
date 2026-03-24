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
import torch
from mindie_llm.runtime.layers.attention.attention_mask import AttentionMask


class TestAttentionMask(unittest.TestCase):

    def setUp(self):
        self.attention_mask = AttentionMask()

    def test_initial_state(self):
        self.assertIsNone(self.attention_mask.atten_splitfuse_mask)

    def test_get_splitfuse_mask_default(self):
        device = torch.device("cpu")
        result = self.attention_mask.get_splitfuse_mask(device)
        self.assertIsNone(result)

    def test_get_splitfuse_mask_with_mock_tensor(self):
        mock_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8)
        self.attention_mask.atten_splitfuse_mask = mock_mask

        device = torch.device("cpu")
        result = self.attention_mask.get_splitfuse_mask(device)
        
        self.assertIs(result, mock_mask)
        self.assertTrue(torch.equal(result, mock_mask))


if __name__ == "__main__":
    unittest.main()
