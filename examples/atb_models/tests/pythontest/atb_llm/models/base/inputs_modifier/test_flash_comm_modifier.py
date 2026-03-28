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
from unittest.mock import Mock

import torch

from atb_llm.utils.weights import Weights
from atb_llm.models.base.inputs_modifier.flash_comm_modifier import FlashCommModifier
from atb_llm.utils.dist import initialize_distributed
from atb_llm.utils.log import logger
import atb_llm.utils.weights as weights_module


class TestFlashCommModifier(unittest.TestCase):
    def setUp(self):
        world_size = 4 
        rank = 1
        npu_id = 0
        self.process_group, self.device = initialize_distributed(rank, npu_id, world_size)
        self.model_name_or_path = "test_flash_comm_model"
        self.dtype = torch.float16

        mock_weight_files = Mock(return_value=[])
        weights_module.weight_files = mock_weight_files

        self.weights = Weights(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
        )

        self.hidden_size = 5120

        logger.debug = Mock()

    def test_init(self):
        modifier = FlashCommModifier(self.weights, self.hidden_size, True)
        self.assertEqual(modifier.tp_rank, 1)
        self.assertEqual(modifier.tp_world_size, 4)
        self.assertEqual(modifier.hidden_size, 5120)
        self.assertFalse(modifier.active)

    def test_pass_flash_comm_threshold_below(self):
        self.assertFalse(FlashCommModifier.pass_flash_comm_threshold(1024, self.hidden_size))
        self.assertFalse(FlashCommModifier.pass_flash_comm_threshold(1023, self.hidden_size))

    def test_pass_flash_comm_threshold_above(self):
        self.assertTrue(FlashCommModifier.pass_flash_comm_threshold(1025, self.hidden_size))
        self.assertTrue(FlashCommModifier.pass_flash_comm_threshold(2048, self.hidden_size))

    def test_modify_inputs_inactive_due_to_flash_comm_disabled(self):
        modifier = FlashCommModifier(self.weights, self.hidden_size, False)

        engine_inputs = [torch.randint(0, 100, (2048,), dtype=torch.int64).npu()] 
        runtime_param = {}

        modifier.modify_inputs(engine_inputs, is_prefill=True, runtime_param=runtime_param)

        self.assertFalse(modifier.active)
        self.assertDictEqual(runtime_param, {})
        self.assertEqual(len(engine_inputs), 1) 
    
    def test_modify_inputs_inactive_when_not_in_prefill(self):
        modifier = FlashCommModifier(self.weights, self.hidden_size, False)

        engine_inputs = [torch.randint(0, 100, (2048,), dtype=torch.int64).npu()] 
        runtime_param = {}

        modifier.modify_inputs(engine_inputs, is_prefill=False, runtime_param=runtime_param)

        self.assertFalse(modifier.active)
        self.assertDictEqual(runtime_param, {})
        self.assertEqual(len(engine_inputs), 1) 

    def test_modify_inputs_inactive_due_to_below_threshold(self):
        modifier = FlashCommModifier(self.weights, self.hidden_size, True)

        engine_inputs = [torch.randint(0, 100, (1024,), dtype=torch.int64).npu()]
        runtime_param = {}

        modifier.modify_inputs(engine_inputs, is_prefill=True, runtime_param=runtime_param)

        self.assertFalse(modifier.active)
        self.assertDictEqual(runtime_param, {})
        self.assertEqual(len(engine_inputs), 1)

    def test_modify_inputs_active(self):
        modifier = FlashCommModifier(self.weights, self.hidden_size, True)

        input_ids = torch.randint(0, 100, (2048,), dtype=torch.int64).npu()
        engine_inputs = [input_ids]
        runtime_param = {}

        modifier.modify_inputs(engine_inputs, is_prefill=True, runtime_param=runtime_param)

        expected_send_counts = [512 * 5120] * 4  
        expected_sdispls = [0, 512 * 5120, 2 * 512 * 5120, 3 * 512 * 5120]
        expected_send_count = [512]  
        expected_recv_counts = [512] * 4  
        expected_rdispls = [0, 512, 1024, 1536]
        expected_recv_count = [512 * 5120]  

        self.assertListEqual(runtime_param.get("sendCounts", []), expected_send_counts)
        self.assertListEqual(runtime_param.get("sdispls", []), expected_sdispls)
        self.assertListEqual(runtime_param.get("sendCount", []), expected_send_count)  
        self.assertListEqual(runtime_param.get("recvCounts", []), expected_recv_counts)
        self.assertListEqual(runtime_param.get("rdispls", []), expected_rdispls)
        self.assertListEqual(runtime_param.get("recvCount", []), expected_recv_count)  

        self.assertEqual(len(engine_inputs), 9)

        self.assertTrue(torch.equal(engine_inputs[1], torch.tensor(expected_send_counts, dtype=torch.int64).npu()))
        self.assertTrue(torch.equal(engine_inputs[2], torch.tensor(expected_sdispls, dtype=torch.int64).npu()))
        self.assertTrue(
            torch.equal(engine_inputs[3], torch.tensor(expected_send_count, dtype=torch.int64).npu())) 
        self.assertTrue(torch.equal(engine_inputs[4], torch.tensor(expected_recv_counts,
                                                                   dtype=torch.int64).npu()))  
        self.assertTrue(torch.equal(engine_inputs[5], torch.tensor(expected_rdispls, dtype=torch.int64).npu()))
        self.assertTrue(
            torch.equal(engine_inputs[6], torch.tensor(expected_recv_count, dtype=torch.int64).npu())) 
        self.assertEqual(engine_inputs[7].shape[0], 512) 
        self.assertEqual(engine_inputs[8].shape[0], 2048) 
        self.assertEqual(engine_inputs[7].device.type, "npu")
        self.assertEqual(engine_inputs[8].device.type, "npu")


    def test_modify_inputs_with_remainder(self):
        modifier = FlashCommModifier(self.weights, self.hidden_size, True)

        input_ids = torch.randint(0, 100, (2049,), dtype=torch.int64).npu()
        engine_inputs = [input_ids]
        runtime_param = {}

        modifier.modify_inputs(engine_inputs, is_prefill=True, runtime_param=runtime_param)

        expected_send_bs = [513, 512, 512, 512]
        expected_send_counts = [513 * 5120, 512 * 5120, 512 * 5120, 512 * 5120]

        self.assertListEqual(runtime_param.get("recvCounts", []), expected_send_bs)
        self.assertListEqual(runtime_param.get("sendCounts", []), expected_send_counts)


if __name__ == '__main__':
    unittest.main()