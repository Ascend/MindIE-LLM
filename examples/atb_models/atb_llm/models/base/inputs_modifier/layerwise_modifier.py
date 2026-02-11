# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import queue
from typing import List
from atb_llm.models.base.flash_causal_lm import LayerWiseAttr, LwdLayerStatus, DistributedType

import torch


class LayerwiseModifier:
    """
    This class contains methods and attributes required to enable long sequences functionality in a model.
    """
    def __init__(self, attr: LayerWiseAttr):
        self.attr = attr
        self.acl_edge_decode_input = None
        self.acl_edge_prefill_input = None
        self.acl_edge_prefill_input_queue = queue.Queue()
        self.acl_edge_decode_param = None
        self.acl_edge_prefill_param = None
        self.acl_edge_prefill_param_queue = queue.Queue()
        self.acl_cloud_inputs = None
        self.acl_cloud_params = None
        self.acl_cloud_inner_hidden = None 
        self.acl_edge_inputs_prefill_pre = None
        self.acl_edge_params_prefill_pre = None

        if self.attr is not None:
            self.active = True
        else:
            self.active = False
      
    @staticmethod      
    def to_index(is_prefill):
        return 1 if is_prefill else 0
    
    def get_input_param(self, is_prefill, is_end_layer):
        if is_prefill:
            if self.acl_edge_prefill_input is None:
                self.acl_edge_prefill_input = self.acl_edge_prefill_input_queue.get(timeout=900)
                self.acl_edge_prefill_param = self.acl_edge_prefill_param_queue.get(timeout=900)
            prefill_input = self.acl_edge_prefill_input
            prefill_param = self.acl_edge_prefill_param
            if is_end_layer:
                self.acl_edge_prefill_input = None
                self.acl_edge_prefill_param = None
            return prefill_input, prefill_param
        else:
            decode_input = self.acl_edge_decode_input
            decode_param = self.acl_edge_decode_param
            if is_end_layer:
                self.acl_edge_decode_input = None
                self.acl_edge_decode_param = None
            return decode_input, decode_param

    def save_input_param(self, inputs, runtime_param, is_prefill):
        if is_prefill:
            self.acl_edge_prefill_input_queue.put([None] + inputs[1:])
            self.acl_edge_prefill_param_queue.put(runtime_param)
        else:
            # input[0] is hidden and needs to be replaced each time; no caching is required.
            self.acl_edge_decode_input = [None] + inputs[1:]
            self.acl_edge_decode_param = runtime_param

    def modify_inputs(
            self,
            inputs: List[torch.Tensor],
            is_prefill: bool,
            runtime_param,
            position_ids,
            input_lengths,
            **kwargs
        ):
        if not self.active:
            return
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage", None)
        out_hidden = kwargs.get("out_hidden", None)
        if self.attr.split_type == DistributedType.EDGE:
            if exe_stage is None:
                return
            if exe_stage.start_exec_layer == 0:
                # 首层需要缓存输入
                self.save_input_param(inputs, runtime_param, is_prefill)
            if exe_stage.end_exec_layer == 1:
                # 尾层需要替换输入
                last_input, last_param = self.get_input_param(is_prefill, True)
                last_input[0] = out_hidden
                inputs[:] = last_input
                runtime_param.clear()
                runtime_param.update(last_param)
        else:
            if exe_stage is None or not is_prefill:
                inputs[0] = out_hidden
            elif exe_stage.start_exec_layer == 0:
                # 首层需要缓存输入
                inputs[0] = out_hidden
                cos_list = kwargs.get("cos_list", None)
                sin_list = kwargs.get("sin_list", None)
                if cos_list is not None:
                    inputs[2] = cos_list
                    inputs[3] = sin_list
                    inputs[1] = torch.arange(
                    input_lengths.sum(), dtype=position_ids.dtype, device=position_ids.device
                    )
                self.acl_cloud_inputs = inputs
                self.acl_cloud_params = runtime_param
            else:
                inputs[:] = self.acl_cloud_inputs
                runtime_param.clear()
                runtime_param.update(self.acl_cloud_params)
                inputs[0] = self.acl_cloud_inner_hidden
                self.acl_cloud_inner_hidden = None
                if exe_stage.end_of_generate_token:
                    self.acl_cloud_inputs = None
                    self.acl_cloud_params = None

    
    def process_out(
        self,
        outputs,
        is_prefill: bool,
        **kwargs
    ):
        if not self.active:
            return outputs
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage", None)

        is_cloud_split = self.attr.split_type == DistributedType.CLOUD
        has_exe_stage = exe_stage is not None
        is_not_end_of_token = not exe_stage.end_of_generate_token if exe_stage else False
        is_cloud_prefill_processing = is_cloud_split and has_exe_stage and is_prefill and is_not_end_of_token
        if is_cloud_prefill_processing:
            self.acl_cloud_inner_hidden = outputs[0]

        is_edge_split = self.attr.split_type == DistributedType.EDGE
        is_edge_prefill_processing = is_edge_split and has_exe_stage and is_prefill and is_not_end_of_token
        if is_edge_prefill_processing:
            if exe_stage.start_exec_layer == 0:
                if kwargs.get("cos_list") is not None:
                    outputs = [[outputs[0], kwargs.get("cos_list"), kwargs.get("sin_list")]]
        return outputs