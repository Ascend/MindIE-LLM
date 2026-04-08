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

import acl
from atb_llm.utils.layerwise_disaggregated.cloud_cut_policy import CloudCutModelType


class ChunkPrefilPolicy():
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChunkPrefilPolicy, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name_or_path='qwen', batch_p_num=1, moe_quantize=None):
        self.soc_name = acl.get_soc_name()
        if not hasattr(self, 'initialized'):
            self.model_type = self.__get_model_name(model_name_or_path)
            self.batch_p_num = batch_p_num
            self.multi_nodes_enable = False
            self.moe_quantize = moe_quantize
            # the map means to {prefill len(K) : chunk num}
            self.prefill_chunk_map = {125: 33, 64: 20, 32: 10, 16: 6, 8: 2}
            if self.model_type == CloudCutModelType.DEEP_SEEK:
                self.prefill_chunk_map = {31.5: 20, 15.5: 5, 7.5: 2}

    @staticmethod
    def __get_model_name(model_name_or_path):
        model_name_or_path_ = model_name_or_path.lower()
        if 'qwen' in model_name_or_path_:
            return CloudCutModelType.QWEN
        elif 'deepseek' in model_name_or_path_ or 'ds' in model_name_or_path_:
            return CloudCutModelType.DEEP_SEEK
        return CloudCutModelType.QWEN
    
    def initialize(self, multi_nodes_enable):
        self.multi_nodes_enable = multi_nodes_enable
        self.__ajust_prefill_chunk_map_for_multi_nodes()

    def map_prefill_chunk_num(self, prefill_seq_len):
        tmp_k_len = round(prefill_seq_len / 1024)
        prefill_chunk_num = 2
        for key, value in self.prefill_chunk_map.items(): 
            if tmp_k_len >= key:
                prefill_chunk_num = value
                return prefill_chunk_num
        return prefill_chunk_num

    def __ajust_prefill_chunk_map_for_multi_nodes(self):
        if not self.multi_nodes_enable:
            return
        
        if self.model_type == CloudCutModelType.DEEP_SEEK and self.moe_quantize == 'w4a8_dynamic':
            self.prefill_chunk_map = {31.5: 20, 15.5: 2, 7.5: 2}
