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

import time
import struct
from mindie_llm.utils.layerwise.share_memory import SharedMemoryManager

IS_PRODUCER_ = True
CONSUMER_NUM_ = 7 # 8 cards 1 master
SHARE_MEM_TYPE_ = 'i'
MEM_TYPE_SIZE_ = struct.calcsize(SHARE_MEM_TYPE_)


class TestSharedMemoryManager(unittest.TestCase):
    def setUp(self):
        self.share_mem_manager = SharedMemoryManager('0')
        self.share_mem_manager.initialize(IS_PRODUCER_, CONSUMER_NUM_)

    @classmethod
    def setUpClass(cls):
        print("TestSharedMemoryManager start")

    @classmethod
    def tearDownClass(cls):
        print("TestSharedMemoryManager end")

    def test_init(self):
        self.assertEqual(self.share_mem_manager._is_producer, IS_PRODUCER_)
        self.assertEqual(self.share_mem_manager._consumer_num, CONSUMER_NUM_)
        self.assertEqual(self.share_mem_manager._share_mem_type, SHARE_MEM_TYPE_)
        self.assertEqual(self.share_mem_manager._mem_type_size, MEM_TYPE_SIZE_)


    def test_share_mem_write_read(self):
        # test write success and read success but the 2nd time read fail
        list_a = [5, -1, 2, 4, 8, -5]
        self.share_mem_manager.write_list_memory(list_a)
        list_1b = self.share_mem_manager.read_list_memory(1)
        self.assertEqual(list_1b, list_a)
        list_1c = self.share_mem_manager.read_list_memory(1)
        self.assertIsNone(list_1c)

        list_2b = self.share_mem_manager.read_list_memory(2)
        self.assertEqual(list_2b, list_a)
        list_2c = self.share_mem_manager.read_list_memory(2)
        self.assertIsNone(list_2c)

        list_6b = self.share_mem_manager.read_list_memory(6)
        self.assertEqual(list_6b, list_a)
        list_6c = self.share_mem_manager.read_list_memory(6)
        self.assertIsNone(list_6c)

        # test not CONSUMER_NUM_ cards read completed, write success after 5s 
        list_d = [7, -3, 4, 6, 10, -7]
        before_write_time = time.time()
        self.share_mem_manager.write_list_memory(list_d)
        after_write_time = time.time()
        self.assertGreaterEqual(after_write_time - before_write_time, 5.0)

        list_7b = self.share_mem_manager.read_list_memory(7)
        self.assertEqual(list_7b, list_d)
        list_7c = self.share_mem_manager.read_list_memory(7)
        self.assertIsNone(list_7c)

    def test_share_mem_write_read_dict(self):
        npu_smi_info = {'hbm_capacity': 65452113920, 'soc_name': 'Ascend910B3'}
        self.share_mem_manager.write_dict_memory(npu_smi_info)

        npu_smi_info_read = self.share_mem_manager.read_dict_memory(1)
        self.assertEqual(npu_smi_info_read, npu_smi_info)
        npu_smi_info_read = self.share_mem_manager.read_dict_memory(1)
        self.assertIsNone(npu_smi_info_read)

if __name__ == "__main__":
    unittest.main()