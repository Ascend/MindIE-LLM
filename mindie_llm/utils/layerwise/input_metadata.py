#!/usr/bin/env python
# coding=utf-8
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
from mindie_llm.connector.common.input_metadata_composite import InputMetadataComposite
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata


class EdgeCloudInputMetadata:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EdgeCloudInputMetadata, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.decode_input_metadata_composite = None
        self.prefill_input_metadata_composite = None
        self.prefill_input_metadata_composite_queue = queue.Queue()

    @staticmethod
    def have_input_metadata(exe_stage: LwdMetadata):
        is_last_layer = (
            exe_stage.start_exec_layer == 1 
            and exe_stage.end_exec_layer == 1
        )
        is_prefill_with_offset = (
            exe_stage.is_prefill 
            and (
                exe_stage.start_exec_layer != 0 
                or exe_stage.long_seq_start_idx != 0
            )
        )
        return is_last_layer or is_prefill_with_offset
    
    @staticmethod
    def need_storage_input_metadata(exe_stage: LwdMetadata):
        stage = exe_stage
        is_start_layer_0 = stage.start_exec_layer == 0
        is_end_layer_0 = stage.end_exec_layer == 0

        is_prefill = stage.is_prefill
        is_not_prefill_chunk = not stage.is_long_seq
        is_prefill_offset_0 = stage.long_seq_start_idx == 0
        
        is_valid_prefill = is_prefill and (is_not_prefill_chunk or is_prefill_offset_0)
        return is_start_layer_0 and ((is_end_layer_0 and not is_prefill) or is_valid_prefill)

    def get_input_metadata(self, is_prefill, exe_stage: LwdMetadata):
        # prefill的metadata缓存，如当前为None，则从队列中获取一个；当执行到当前P的最后一次，则讲缓存置为None，待下次从队列获取
        if is_prefill:
            if self.prefill_input_metadata_composite is None:
                self.prefill_input_metadata_composite = self.prefill_input_metadata_composite_queue.get(block=False)
            
            input_metadata_composite = self.prefill_input_metadata_composite
            if exe_stage.end_of_generate_token:
                self.prefill_input_metadata_composite = None
            return input_metadata_composite
        else:
            return self.decode_input_metadata_composite

    def set_input_metadata(self, input_metadata_composite: InputMetadataComposite, is_prefill):
        if is_prefill:
            self.prefill_input_metadata_composite_queue.put(input_metadata_composite)
        else:
            self.decode_input_metadata_composite = input_metadata_composite

pd_exec_matadata_instance = EdgeCloudInputMetadata()