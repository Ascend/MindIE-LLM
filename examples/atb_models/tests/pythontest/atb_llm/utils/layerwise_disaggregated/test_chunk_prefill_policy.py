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
from unittest.mock import Mock, patch
from atb_llm.utils.layerwise_disaggregated.chunk_prefill_policy import ChunkPrefilPolicy


class TestChunkPrefilPolicy(unittest.TestCase):
    @patch('atb_llm.utils.layerwise_disaggregated.chunk_prefill_policy.acl')
    def setUp(self, mock_acl):
        mock_acl.get_soc_name = Mock()
        mock_acl.get_soc_name.return_value = 'Ascend910B4'
        self.qwen_policy = ChunkPrefilPolicy()
        self.deepseek_policy = ChunkPrefilPolicy('deepseek', 1, None)

    @classmethod
    def setUpClass(cls):
        print("TestChunkPrefilPolicy start")

    @classmethod
    def tearDownClass(cls):
        print("TestChunkPrefilPolicy end")
        
    def test_map_prefill_chunk_num(self):
        self.assertEqual(self.deepseek_policy.map_prefill_chunk_num(8192), 2)