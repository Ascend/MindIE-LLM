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
from unittest.mock import patch, MagicMock, call
import torch
from atb_llm.models.base.inputs_modifier.layerwise_modifier import LayerwiseModifier
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM, LayerWiseAttr, LwdLayerStatus, DistributedType


class TestLayerwiseModifier(unittest.TestCase):
    def setUp(self):
        self.layerwise = LayerWiseAttr(edge_start_layer_count=1, edge_end_layer_count=1, split_type=DistributedType.CLOUD)
        self.layerwise_modifier = LayerwiseModifier(self.layerwise)


    def test_to_index(self):
        ret = self.layerwise_modifier.to_index(is_prefill=True)
        self.assertEqual(ret, 1)
        ret = self.layerwise_modifier.to_index(is_prefill=False)
        self.assertEqual(ret, 0)

    def test_modify_inputs(self):
        self.layerwise_modifier.active = False
        outputs = [0]
        runtime_param = MagicMock()
        position_ids = MagicMock()
        input_lengths = MagicMock()
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths
        )
        self.assertEqual(outputs, [0])

        self.layerwise_modifier.active = True
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths
        )
        self.assertIsNone(outputs[0])

        self.layerwise_modifier.active = True
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths,
            out_hidden=3
        )
        self.assertEqual(outputs, [3])

        self.layerwise_modifier.active = True
        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.start_exec_layer = 0
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths,
            out_hidden=3,
            layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage
        )
        self.assertEqual(outputs, [3])


        self.layerwise_modifier.active = True
        layerwise_disaggregated_exe_stage = MagicMock()
        outputs = [0,1]
        self.layerwise_modifier.acl_cloud_inputs = [1, 2]
        self.layerwise_modifier.acl_cloud_inner_hidden = 3
        layerwise_disaggregated_exe_stage.start_exec_layer = 1
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths,
            out_hidden=3,
            layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage
        )
        self.assertEqual(outputs, [3, 2])

        #warm-up
        self.layerwise_modifier.active = True
        self.layerwise_modifier.attr.split_type = DistributedType.EDGE
        outputs = [0]
        runtime_param = MagicMock()
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths
        )
        self.assertEqual(outputs, [0])

        self.layerwise_modifier.active = True
        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.start_exec_layer = 0
        outputs = [0, 1]
        self.layerwise_modifier.acl_edge_inputs = [[0],[0]]
        self.layerwise_modifier.acl_edge_params = [[0],[0]]
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths,
            layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage
        )

        self.layerwise_modifier.active = True
        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.end_exec_layer = 1
        outputs = [0, 1]
        self.layerwise_modifier.acl_edge_inputs = [[0],[0]]
        self.layerwise_modifier.acl_edge_params = [[0],[0]]
        self.layerwise_modifier.modify_inputs(
            outputs, True, runtime_param, position_ids, input_lengths,
            layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage
        )

    def test_process_out(self):
        self.layerwise_modifier.active = False
        outputs = [0]
        res_output = self.layerwise_modifier.process_out(outputs, is_prefill=False)
        self.assertEqual(res_output, outputs)

        self.layerwise_modifier.active = True
        res_output = self.layerwise_modifier.process_out(outputs, is_prefill=False)
        self.assertEqual(res_output, outputs)

        self.layerwise_modifier.active = True
        self.layerwise_modifier.attr.split_type = DistributedType.CLOUD
        layerwise_disaggregated_exe_stage = MagicMock()
        layerwise_disaggregated_exe_stage.end_of_generate_token = False
        res_output = self.layerwise_modifier.process_out(outputs, is_prefill=True,
                layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage
        )
        self.assertEqual(self.layerwise_modifier.acl_cloud_inner_hidden, 0)