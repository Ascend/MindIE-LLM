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
import time
from collections import deque
from atb_llm.utils.layerwise_disaggregated.cloud_cut_policy import\
    CloudCutState, CloudCutInputData, CloudCutPolicy, CloudCutModelType


class TestCloudCutPolicy(unittest.TestCase):
    @patch('atb_llm.utils.layerwise_disaggregated.cloud_cut_policy.acl')
    def setUp(self, mock_acl):
        mock_acl.get_soc_name = Mock()
        mock_acl.get_soc_name.return_value = 'Ascend910B4'
        self.cloud_cut_policy = CloudCutPolicy("slave")
        self.cloud_cut_policy.initialize("slave", 0, (2, 62), False)
        pass

    @classmethod
    def setUpClass(cls):
        print("TestCloudCutPolicy start")

    @classmethod
    def tearDownClass(cls):
        print("TestCloudCutPolicy end")

    def test_init(self):
        self.assertIsInstance(self.cloud_cut_policy.decode_time_queue, deque)
        self.assertEqual(self.cloud_cut_policy.queue_max_len, 10)
        self.assertNotEqual(self.cloud_cut_policy.prefill_cut_num_max, 62)
        self.assertNotEqual(self.cloud_cut_policy.prefill_cut_num_min, 2)
        self.assertTrue(self.cloud_cut_policy.initialized)
        self.assertIsNotNone(self.cloud_cut_policy.prefill_default_cut_map)
        self.assertIsNotNone(self.cloud_cut_policy.prefill_seq_k_len_default_forward_time)
    
    def test_get_prefill_cut_num(self):
        seq_len_list = [256, 1024, 2048, 3072, 4096, 8192, 8888, 16000, 19000]
        gap_time_list = [[50], [100], [200], [300, 360], [400], [500, 550], [600], [700], [800]]
        decode_time_list = [300.000, 20.000, 20.000, 20.000, 20.000, 20.000, 28.145, 26.833, 23.843, 21.680, 19.680, 19.680, 18.787, 19.720]
        for i, seq_len in enumerate(seq_len_list):
            gap_time_list_tmp = gap_time_list[i]
            for decode_time in decode_time_list:
                self.cloud_cut_policy.set_decode_start_time(False, time.time())
                self.cloud_cut_policy.set_decode_end_time(False, time.time() + decode_time / 1000)
            cut_num = self.cloud_cut_policy.get_cut_num(CloudCutInputData(seq_len, gap_time_list_tmp))
            print(f"i: {i}, seq_len: {seq_len} cut_num: {cut_num}")
            self.cloud_cut_policy.calc_avg_request_rate(CloudCutInputData(seq_len, gap_time_list_tmp))
            self.cloud_cut_policy.cut_state = CloudCutState.FIND_IN_TBL
            self.cloud_cut_policy.decode_time_queue.clear()
    
    def test_get_in_range_cut_num(self):
        self.cloud_cut_policy.max_cut_num = 62
        cut_num = self.cloud_cut_policy.get_in_range_cut_num(10)
        self.assertEqual(cut_num, 21)
        self.cloud_cut_policy.max_cut_num = 1
        cut_num = self.cloud_cut_policy.get_in_range_cut_num(10)
        self.assertEqual(cut_num, 21)
    
    def test_cal_cut_num_by_decode(self):
        self.cloud_cut_policy.decode_avg_time = 1
        cut_num = self.cloud_cut_policy.cal_cut_num_by_decode()
        self.assertEqual(cut_num, 21)
        
        self.cloud_cut_policy.decode_avg_time = 0
        cut_num = self.cloud_cut_policy.cal_cut_num_by_decode()
        self.assertEqual(cut_num, 21)
        
    def test_ajust_prefill_cut_num_for_diff_npu_soc(self):
        self.cloud_cut_policy.soc_name = 'Ascend910B2'
        self.cloud_cut_policy.batch_p_num = 2
        self.cloud_cut_policy._CloudCutPolicy__ajust_prefill_cut_num_for_diff_npu_soc()
        self.assertEqual(self.cloud_cut_policy.prefill_default_cut_map.get(32), 100)
        
        self.cloud_cut_policy.soc_name = 'Ascend910B3'
        self.cloud_cut_policy.batch_p_num = 1
        self.cloud_cut_policy._CloudCutPolicy__ajust_prefill_cut_num_for_diff_npu_soc()
        self.assertEqual(self.cloud_cut_policy.prefill_default_cut_map.get(32), 70)
        self.cloud_cut_policy.batch_p_num = 2
        self.cloud_cut_policy._CloudCutPolicy__ajust_prefill_cut_num_for_diff_npu_soc()
        self.assertEqual(self.cloud_cut_policy.prefill_default_cut_map.get(32), 70)

        self.cloud_cut_policy.model_type = CloudCutModelType.DEEP_SEEK
        self.cloud_cut_policy.soc_name = 'Ascend910_9362'
        self.cloud_cut_policy._CloudCutPolicy__ajust_prefill_cut_num_for_diff_npu_soc()
        self.assertEqual(self.cloud_cut_policy.prefill_default_cut_map.get(7.5), 50)

    def test_ajust_prefill_cut_num_for_multi_nodes(self):
        self.cloud_cut_policy.moe_quantize = 'w4a8_dynamic'
        self.cloud_cut_policy._CloudCutPolicy__ajust_prefill_cut_num_for_multi_nodes()
        self.assertEqual(self.cloud_cut_policy.prefill_default_cut_map.get(7.5), 50)

        self.cloud_cut_policy.moe_quantize = None
        self.cloud_cut_policy._CloudCutPolicy__ajust_prefill_cut_num_for_multi_nodes()
        self.assertEqual(self.cloud_cut_policy.prefill_default_cut_map.get(7.5), 50)
        
if __name__ == "__main__":
    unittest.main()