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

import math
import re
from enum import IntEnum, Enum
from dataclasses import dataclass
from collections import deque
import acl
from atb_llm.utils.log import logger

PREFILL_DEFAULT_CUT_MAP = "prefill_default_cut_map"
PREFILL_CUT_NUM_MAX = "prefill_cut_num_max"
PREFILL_CUT_NUM_MIN = "prefill_cut_num_min"


class CloudCutState(IntEnum):
    FIND_IN_TBL = 0
    CAL_BY_DECODE = 1


class CloudCutClassType(IntEnum):
    CLOUD = 0
    OTHER = 1


class CloudCutModelType(IntEnum):
    QWEN = 0
    DEEP_SEEK = 1


class CardType(Enum):
    ASCEND_910B4 = "ASCEND910B4"
    ASCEND_910B3 = "ASCEND910B3"
    ASCEND_910C = "Ascend910C" 


class NodeMode(Enum):
    SINGLE_NODE = "SINGLE_NODE"
    MULTI_NODE = "MULTI_NODE"


class MemoryType(Enum):
    MEM_32G = "32G"
    MEM_64G = "64G"
    MEM_128G = "128G"


class PolicyParserUtils:

    @staticmethod
    def merge_nested_dicts(dict1: dict, dict2: dict) -> dict:
        merged = dict1.copy()
        for key, value in dict2.items():
            # 如果key在两个字典中都存在，且都是字典，则递归合并；否则直接覆盖/新增
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = PolicyParserUtils.merge_nested_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def parse_card_type(card_str: str) -> CardType:
        normalized = re.sub(r'[^A-Z0-9]', '', card_str.strip().upper()) 
        match_map = {
            "910B4": CardType.ASCEND_910B4,
            "B4": CardType.ASCEND_910B4,
            "910B3": CardType.ASCEND_910B3,
            "B3": CardType.ASCEND_910B3,
            "910C": CardType.ASCEND_910C,
            "C": CardType.ASCEND_910C,   
            "9362": CardType.ASCEND_910C
        }

        for key, card_type in match_map.items():
            if key in normalized:
                return card_type
        
        supported = [c.value for c in CardType]
        raise ValueError(f"not support {card_str}，support：{supported}")

    @staticmethod
    def parse_memory_type(mem_str: str) -> MemoryType:
        try:
            return MemoryType(f"{mem_str}G")
        except ValueError as e:
            supported = [m.value for m in MemoryType]
            raise ValueError(f"not support memory {mem_str}G，support：{supported}") from e

    @staticmethod
    def parse_deploy_mode(deploy_str: str) -> NodeMode:
        try:
            return NodeMode(deploy_str.upper())
        except ValueError as e:
            supported = [m.value for m in NodeMode]
            raise ValueError(f"not support memory {deploy_str}，support：{supported}") from e


@dataclass
class CloudCutInputData():
    seq_len: int = 0
    prefill_gap_time: list = None


class CloudCutPolicy():
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CloudCutPolicy, cls).__new__(cls)
        return cls._instance

    def __init__(self, name="", model_name_or_path='qwen', batch_p_num=1, moe_quantize=None):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.model_type = self.__get_model_name(model_name_or_path)
            self.role_type = CloudCutClassType.CLOUD if name == "slave" else CloudCutClassType.OTHER
            self.rank_id = None
            self.soc_name = acl.get_soc_name()
            self.batch_p_num = batch_p_num
            self.multi_nodes_enable = False
            self.moe_quantize = moe_quantize
            self.initialized = False

            # Predict the number of chunks, hardcoded based on empirical values: n(K): [cut_num, cut_num_max];
            # if less than 1, set to 1 chunk (one additional chunk will be added by default, resulting in cut_num + 1
            # chunks); if greater than 8, set to 31 chunks.
            # For NPU Soc is Ascend910B2 or other models, use the following default prefill_cut_num
            self.prefill_default_cut_map = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
            self.prefill_cut_num_max = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
            self.prefill_cut_num_min = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 4, 3: 6, 2: 5, 1: 5, 0: 8}
            # The default inference time of the Qwen3-32B model across various sequence lengths, which requires
            #adjustment depending on the hardware card.
            self.prefill_seq_k_len_default_forward_time = {16: 100000000000,
                                                            8: 610.166,
                                                            4: 432,
                                                            3: 380,
                                                            2: 243,
                                                            1: 71.434,
                                                            0: 71.434
                                                            }

            self.last_cut_num = 8
            self.max_cut_num = 62
            self.min_cut_num = 2
            self.seq_k_len = 0
            self.prefill_exec_time = 0
            self.gap_time_list = None
            self.avg_gap_time = 0
            self.decode_avg_time = 0
            self.decode_start_time = None
            self.queue_max_len = 10
            self.request_queue_max_len = 50
            self.decode_time_queue = deque(maxlen=10)
            self.prefill_gap_list = deque(maxlen=50)
            self.request_rate = []
            self.moe_quantize = None
            self.hierarchical_policy_config = {}

            if self.model_type == CloudCutModelType.DEEP_SEEK:
                self.prefill_default_cut_map = {31.5: 80, 15.5: 40, 7.5: 62, 3.8: 30, 3.3: 36, 1.8: 30, 0.8: 17, 0: 17}
                self.prefill_cut_num_max = {31.5: 80, 15.5: 40, 7.5: 62, 3.8: 30, 3.3: 36, 1.8: 30, 0.8: 17, 0: 17}
                self.prefill_cut_num_min = {31.5: 80, 15.5: 40, 7.5: 62, 3.8: 30, 3.3: 36, 1.8: 30, 0.8: 17, 0: 17}
                self.prefill_seq_k_len_default_forward_time = {31.5: 10000, 15.5: 10000, 7.5: 10000, 3.8: 10000,
                    3.3: 1000, 1.8: 10000, 0.8: 10000, 0: 10000}

            self.cut_state = CloudCutState.FIND_IN_TBL

            self.state_process_func = {
                CloudCutState.FIND_IN_TBL: self.find_cut_num_in_tbl,
                CloudCutState.CAL_BY_DECODE: self.cal_cut_num_by_decode,
            }

    @staticmethod
    def __get_model_name(model_name_or_path):
        model_name_or_path_ = model_name_or_path.lower()
        if 'qwen' in model_name_or_path_:
            return CloudCutModelType.QWEN
        elif 'deepseek' in model_name_or_path_ or 'ds' in model_name_or_path_:
            return CloudCutModelType.DEEP_SEEK
        return CloudCutModelType.QWEN

    @staticmethod
    def __get_closest_standard(mem_bytes: int) -> int:
        standard_gbs = [32, 64, 128, 256]
        mem_gb = mem_bytes / (1024 ** 3)
        closest_gb = min(standard_gbs, key=lambda gb: abs(mem_gb - gb))
        return closest_gb

    def initialize(self, name, rank_id, cut_num_range, multi_nodes_enable):
        self.role_type = CloudCutClassType.CLOUD if name == "slave" else CloudCutClassType.OTHER
        self.rank_id = rank_id
        self.min_cut_num, self.max_cut_num = cut_num_range
        self.multi_nodes_enable = multi_nodes_enable
        self.__ajust_prefill_cut_num_for_diff_npu_soc()
        self.__ajust_prefill_cut_num_for_multi_nodes()
        logger.info(f"[layerwiseDisaggregated] cut policy init, model_type: {self.model_type.name} role_type: "
            f"{self.role_type.name} soc_name: {self.soc_name} moe_quantize: {self.moe_quantize} "
            f"default_cut_map: {self.prefill_default_cut_map} cut_num_max: {self.prefill_cut_num_max} "
            f"cut_num_min: {self.prefill_cut_num_min} ")
        self.initialized = True

    def initialize_standard_card(self, peer_soc_name, peer_mem_size):
        self.__ajust_prefill_cut_num_for_standard_card()

        edge_card_type = PolicyParserUtils.parse_card_type(peer_soc_name)
        peer_mem_size_gb = self.__get_closest_standard(peer_mem_size)
        edge_card_mem = PolicyParserUtils.parse_memory_type(peer_mem_size_gb)
        cloud_card_type = PolicyParserUtils.parse_card_type(self.soc_name)
        cloud_deploy_mode = "single_node"
        if self.multi_nodes_enable:
            cloud_deploy_mode = "multi_node"
        cloud_deploy_mode = PolicyParserUtils.parse_deploy_mode(cloud_deploy_mode)
        hardware_combo = (edge_card_type, edge_card_mem, cloud_card_type)
        logger.info(f"[layerwiseDisaggregated]hardware_combo is {hardware_combo}, {cloud_deploy_mode}")
        try:
            hw_config = self.hierarchical_policy_config[cloud_deploy_mode][hardware_combo]
            model_policy = hw_config[self.model_type]
            self.prefill_default_cut_map = model_policy["prefill_default_cut_map"]
            self.prefill_cut_num_max = model_policy["prefill_cut_num_max"]
            self.prefill_cut_num_min = model_policy["prefill_cut_num_min"]
            logger.info(f"[layerwiseDisaggregated]prefill default cut map is {self.prefill_default_cut_map}")
            logger.info(f"[layerwiseDisaggregated]prefill_cut_num_max is {self.prefill_cut_num_max}")
            logger.info(f"[layerwiseDisaggregated]prefill_cut_num_min is {self.prefill_cut_num_min}")

        except KeyError:
            logger.info(f"[layerwiseDisaggregated]not found hardware combo: {hardware_combo}, use default cut policy")

    def get_cut_num(self, input_data: CloudCutInputData):
        if not self.initialized:
            raise RuntimeError('The current CloudCutPolicy is not initialize.')

        if self.rank_id != 0:
            return self.last_cut_num

        batch_size = len(input_data.prefill_gap_time)
        if self.multi_nodes_enable and self.moe_quantize == 'w4a8_dynamic':
            tmp_k_len = input_data.seq_len / 1024
        else:
            tmp_k_len = input_data.seq_len / 1024 / batch_size
        # DS暂时保留一位小数
        self.seq_k_len = round(tmp_k_len, 1) if self.model_type == CloudCutModelType.DEEP_SEEK else round(tmp_k_len)
        self.adjust_ds_1k_cut_map(self.seq_k_len, batch_size)
        self.adjust_ds_8k_cut_map(self.seq_k_len, batch_size)
        self.calc_avg_request_rate(input_data)
        self.gap_time_list = self.request_rate

        cut_state = self.get_cut_state()

        func = self.state_process_func.get(cut_state)
        if func is None:
            return self.last_cut_num
        cut_num = func()
        logger.info(f"[layerwiseDisaggregated] prefill len:{input_data.seq_len} batch_size:{batch_size} gap_time_list: "
                    f"{self.gap_time_list} prefill_exec_time: {self.prefill_exec_time} decode_avg_time: "
                    f"{self.decode_avg_time} cut_num: {cut_num}")
        return cut_num

    def set_decode_start_time(self, is_prefill, curr_time):
        if is_prefill or self.rank_id != 0 or self.role_type != CloudCutClassType.CLOUD:
            return

        self.decode_start_time = curr_time

    def set_decode_end_time(self, is_prefill, curr_time):
        if is_prefill or self.rank_id != 0 or self.role_type != CloudCutClassType.CLOUD:
            return

        if self.decode_start_time is None:
            return

        decode_time = curr_time - self.decode_start_time
        if decode_time > 2: # A decode time exceeding 2 seconds is considered an abnormal value.
            self.decode_start_time = None
            return

        decode_time *= 1000 
        self.decode_time_queue.append(decode_time)

    # At least 10 data sets are required to compute an average; the return value is in milliseconds.
    def get_decode_avg_time(self):
        if len(self.decode_time_queue) < self.queue_max_len:
            return None

        avg_time_ms = sum(self.decode_time_queue) / len(self.decode_time_queue)
        return avg_time_ms
    
    def get_prefill_exec_time_for_tbl(self):
        for seq_k_len, exec_time in self.prefill_seq_k_len_default_forward_time.items():
            if self.seq_k_len >= seq_k_len:
                return exec_time
        return 1000000 

    def get_cut_state(self):
        decode_avg_time = self.get_decode_avg_time()
        if decode_avg_time is None:  # A minimum of 10 decode samples is required before the resulting data can be used.
            return CloudCutState.FIND_IN_TBL

        sum_gap_time = 0
        for gap_time in self.gap_time_list:
            if gap_time != -1: 
                sum_gap_time += gap_time

        seq_num = len(self.gap_time_list)
        if seq_num == 0:
            return CloudCutState.FIND_IN_TBL

        avg_gap_time = sum_gap_time / seq_num
        prefill_exec_time = self.get_prefill_exec_time_for_tbl()
        self.decode_avg_time = decode_avg_time
        self.avg_gap_time = avg_gap_time
        self.prefill_exec_time = prefill_exec_time

        if prefill_exec_time + decode_avg_time > avg_gap_time:
            return CloudCutState.FIND_IN_TBL

        return CloudCutState.CAL_BY_DECODE

    def find_cut_num_in_tbl(self):
        for key, value in self.prefill_default_cut_map.items(): 
            if self.seq_k_len >= key:
                cut_num = value
                return cut_num

        return self.last_cut_num

    def get_in_range_cut_num(self, cut_num):
        max_cut_num = self.max_cut_num
        for key, value in self.prefill_cut_num_max.items():
            if self.seq_k_len >= key:
                max_cut_num = value
                break

        if cut_num > max_cut_num:
            return max_cut_num

        min_cut_num = self.min_cut_num
        for key, value in self.prefill_cut_num_min.items():
            if self.seq_k_len >= key:
                min_cut_num = value
                break 

        if cut_num < min_cut_num:
            return min_cut_num
        return cut_num

    def cal_cut_num_by_decode(self):
        # Theoretically, there is no scenario in which the average decoding processing time is less than 1ms.
        if self.decode_avg_time < 1:
            return self.find_cut_num_in_tbl()

        cut_num = math.floor((self.avg_gap_time - self.prefill_exec_time) / self.decode_avg_time)
        cut_num = self.get_in_range_cut_num(cut_num)

        return cut_num
    
    def calc_avg_request_rate(self, input_data: CloudCutInputData):
        for gap_time in input_data.prefill_gap_time:
            self.prefill_gap_list.append(gap_time)
            
        if len(self.prefill_gap_list) < self.request_queue_max_len:
            self.request_rate = [-1]
        else:
            self.request_rate = [sum(self.prefill_gap_list) / self.request_queue_max_len]

    def adjust_ds_1k_cut_map(self, k_len, batch_size):
        if self.model_type != CloudCutModelType.DEEP_SEEK or k_len > 1.8 or k_len < 0.8:
            return
        if not self.multi_nodes_enable:
            cut_num_1k = 17 if batch_size == 2 else 10
        else:
            cut_num_1k = 21 if batch_size == 2 else 15
        self.prefill_default_cut_map[0.8] = cut_num_1k
        self.prefill_cut_num_max[0.8] = cut_num_1k
        self.prefill_cut_num_min[0.8] = cut_num_1k

    def adjust_ds_8k_cut_map(self, k_len, batch_size):
        if self.model_type != CloudCutModelType.DEEP_SEEK or self.moe_quantize != 'w4a8_dynamic' or \
            not self.multi_nodes_enable:
            return

        if k_len > 8.5 or k_len < 7.5:
            return

        cut_num_8k = 20 if batch_size == 2 else 38
        self.prefill_default_cut_map[7.5] = cut_num_8k
        self.prefill_cut_num_max[7.5] = cut_num_8k
        self.prefill_cut_num_min[7.5] = cut_num_8k

    def __ajust_prefill_cut_num_for_diff_npu_soc_qwen(self):
        if self.soc_name.startswith('Ascend910B2'):
            if self.batch_p_num != 1:
                self.prefill_default_cut_map = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
                self.prefill_cut_num_max = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
                self.prefill_cut_num_min = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 5, 3: 6, 2: 5, 1: 5, 0: 8}
            return
        if self.soc_name.startswith('Ascend910B3'):
            if self.batch_p_num == 1:
                self.prefill_default_cut_map = {125: 330, 64: 120, 32: 70, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
                self.prefill_cut_num_max = {125: 330, 64: 120, 32: 70, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
                self.prefill_cut_num_min = {125: 330, 64: 120, 32: 70, 16: 24, 8: 10, 4: 4, 3: 6, 2: 5, 1: 5, 0: 8}
            else:
                self.prefill_default_cut_map = {125: 330, 64: 120, 32: 70, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
                self.prefill_cut_num_max = {125: 330, 64: 120, 32: 70, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
                self.prefill_cut_num_min = {125: 330, 64: 120, 32: 70, 16: 24, 8: 10, 4: 5, 3: 6, 2: 5, 1: 5, 0: 8}
            return
        if self.soc_name.startswith('Ascend910B4'):
            if self.batch_p_num == 1:
                self.prefill_default_cut_map = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
                self.prefill_cut_num_max = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
                self.prefill_cut_num_min = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 4, 3: 6, 2: 5, 1: 5, 0: 8}
            else:
                self.prefill_default_cut_map = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
                self.prefill_cut_num_max = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
                self.prefill_cut_num_min = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 5, 3: 6, 2: 5, 1: 5, 0: 8}
            return
        if self.soc_name == 'Ascend910_9362':   # 910C
            self.prefill_default_cut_map = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 6, 0: 8}
            self.prefill_cut_num_max = {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6, 3: 8, 2: 9, 1: 5, 0: 8}
            self.prefill_cut_num_min = {125: 330, 64: 120, 32: 100, 16: 23, 8: 9, 4: 5, 3: 5, 2: 4, 1: 4, 0: 8}
            return

    def __ajust_prefill_cut_num_for_diff_npu_soc_deepseek(self):
        if self.soc_name == 'Ascend910_9362':   # 910C
            self.prefill_default_cut_map = {31.5: 80, 15.5: 40, 7.5: 50, 3.8: 23, 3.3: 34, 1.8: 32, 0.8: 21, 0: 21}
            self.prefill_cut_num_max = {31.5: 80, 15.5: 40, 7.5: 50, 3.8: 23, 3.3: 34, 1.8: 32, 0.8: 21, 0: 21}
            self.prefill_cut_num_min = {31.5: 80, 15.5: 40, 7.5: 50, 3.8: 23, 3.3: 34, 1.8: 32, 0.8: 21, 0: 21}
            return

    def __ajust_prefill_cut_num_for_diff_npu_soc(self):
        if self.model_type == CloudCutModelType.QWEN:
            self.__ajust_prefill_cut_num_for_diff_npu_soc_qwen()
            return

        if self.model_type == CloudCutModelType.DEEP_SEEK:
            self.__ajust_prefill_cut_num_for_diff_npu_soc_deepseek()
            return

    def __ajust_prefill_cut_num_for_multi_nodes(self):
        if not self.multi_nodes_enable or self.model_type != CloudCutModelType.DEEP_SEEK:
            return

        if self.moe_quantize == 'w4a8_dynamic':
            self.prefill_default_cut_map = {31.5: 59, 15.5: 60, 7.5: 32, 3.8: 17, 3.3: 17, 1.8: 17, 0.8: 21, 0: 21}
            self.prefill_cut_num_max = {31.5: 59, 15.5: 60, 7.5: 32, 3.8: 17, 3.3: 17, 1.8: 17, 0.8: 21, 0: 21}
            self.prefill_cut_num_min = {31.5: 59, 15.5: 60, 7.5: 32, 3.8: 17, 3.3: 17, 1.8: 17, 0.8: 21, 0: 21}
        else:
            self.prefill_default_cut_map = {31.5: 59, 15.5: 59, 7.5: 45, 3.8: 21, 3.3: 20, 1.8: 20, 0.8: 21, 0: 21}
            self.prefill_cut_num_max = {31.5: 59, 15.5: 59, 7.5: 45, 3.8: 21, 3.3: 20, 1.8: 20, 0.8: 21, 0: 21}
            self.prefill_cut_num_min = {31.5: 59, 15.5: 59, 7.5: 45, 3.8: 21, 3.3: 20, 1.8: 20, 0.8: 21, 0: 21}

    def __ajust_prefill_cut_num_for_standard_card(self):
        self.prefill_seq_k_len_default_forward_time = {16: 100000000000,
                                                            8: 610.166,
                                                            4: 432,
                                                            3: 380,
                                                            2: 195,
                                                            1: 71.434,
                                                            0: 71.434
                                                            }
        self.__ajust_prefill_cut_num_for_standard_card_edge_b4_32g()
        self.__ajust_prefill_cut_num_for_standard_card_edge_b4_64g()

    def __ajust_prefill_cut_num_for_standard_card_edge_b4_32g(self):
        hierarchical_policy_config_edge_b4_32g = {
            NodeMode.SINGLE_NODE: {
                (CardType.ASCEND_910B4, MemoryType.MEM_32G, CardType.ASCEND_910B3): {
                    CloudCutModelType.QWEN: {
                        PREFILL_DEFAULT_CUT_MAP: {125: 330, 64: 120, 32: 50, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 6, 0: 8},
                        PREFILL_CUT_NUM_MAX: {125: 330, 64: 120, 32: 50, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 5, 0: 8},
                        PREFILL_CUT_NUM_MIN: {125: 330, 64: 120, 32: 50, 16: 24, 8: 10, 4: 6,
                                                    3: 6, 2: 3, 1: 5, 0: 8},
                    },
                    CloudCutModelType.DEEP_SEEK: {
                        PREFILL_DEFAULT_CUT_MAP: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30,
                                                    3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MAX: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MIN: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                    }
                },
                (CardType.ASCEND_910B4, MemoryType.MEM_32G, CardType.ASCEND_910B4): {
                    CloudCutModelType.QWEN: {
                        PREFILL_DEFAULT_CUT_MAP: {125: 330, 64: 120, 32: 100, 16: 24, 8: 12, 4: 6,
                                                    3: 6, 2: 9, 1: 6, 0: 8},
                        PREFILL_CUT_NUM_MAX: {125: 330, 64: 120, 32: 100, 16: 24, 8: 12, 4: 6,
                                                    3: 6, 2: 9, 1: 6, 0: 8},
                        PREFILL_CUT_NUM_MIN: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 6, 2: 3, 1: 6, 0: 8},
                    },
                    CloudCutModelType.DEEP_SEEK: {
                        PREFILL_DEFAULT_CUT_MAP: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30,
                                                     3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MAX: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MIN: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                    }
                },
               
            },
            NodeMode.MULTI_NODE: {
                (CardType.ASCEND_910B4, MemoryType.MEM_32G, CardType.ASCEND_910C): {
                    CloudCutModelType.QWEN: {
                        PREFILL_DEFAULT_CUT_MAP: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 6, 0: 8},
                        PREFILL_CUT_NUM_MAX: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 5, 0: 8},
                        PREFILL_CUT_NUM_MIN: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 4,
                                                    3: 6, 2: 5, 1: 5, 0: 8},
                    }
                }
            }
        }
        merged_ = PolicyParserUtils.merge_nested_dicts(self.hierarchical_policy_config, \
                                                       hierarchical_policy_config_edge_b4_32g)
        self.hierarchical_policy_config = merged_.copy()

    def __ajust_prefill_cut_num_for_standard_card_edge_b4_64g(self):
        hierarchical_policy_config_edge_b4_64g = {
            NodeMode.SINGLE_NODE: {
                (CardType.ASCEND_910B4, MemoryType.MEM_64G, CardType.ASCEND_910B3): {
                    CloudCutModelType.QWEN: {
                        PREFILL_DEFAULT_CUT_MAP: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 6, 0: 8},
                        PREFILL_CUT_NUM_MAX: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 5, 0: 8},
                        PREFILL_CUT_NUM_MIN: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 4,
                                                    3: 6, 2: 5, 1: 5, 0: 8},
                    },
                    CloudCutModelType.DEEP_SEEK: {
                        PREFILL_DEFAULT_CUT_MAP: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30,
                                                    3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MAX: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MIN: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                    }
                },
                (CardType.ASCEND_910B4, MemoryType.MEM_64G, CardType.ASCEND_910B4): {
                    CloudCutModelType.QWEN: {
                        PREFILL_DEFAULT_CUT_MAP: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 6, 0: 8},
                        PREFILL_CUT_NUM_MAX: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 8, 2: 9, 1: 5, 0: 8},
                        PREFILL_CUT_NUM_MIN: {125: 330, 64: 120, 32: 100, 16: 24, 8: 10, 4: 6,
                                                    3: 6, 2: 5, 1: 5, 0: 8},
                    },
                    CloudCutModelType.DEEP_SEEK: {
                        PREFILL_DEFAULT_CUT_MAP: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30,
                                                    3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MAX: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                        PREFILL_CUT_NUM_MIN: {31.5: 80, 15.5: 30, 7.5: 50, 3.8: 30, 3.3: 28, 1.8: 38, 0.8: 18, 0: 18},
                    }
                },
               
            },
            NodeMode.MULTI_NODE: {
                (CardType.ASCEND_910B4, MemoryType.MEM_64G, CardType.ASCEND_910C): {
                    CloudCutModelType.DEEP_SEEK: {
                        PREFILL_DEFAULT_CUT_MAP: {31.5: 59, 15.5: 55, 7.5: 45,
                                                    3.8: 23, 3.3: 20, 1.8: 20, 0.8: 21, 0: 21},
                        PREFILL_CUT_NUM_MAX: {31.5: 59, 15.5: 55, 7.5: 45, 3.8: 23, 3.3: 20, 1.8: 20, 0.8: 21, 0: 21},
                        PREFILL_CUT_NUM_MIN: {31.5: 59, 15.5: 55, 7.5: 45, 3.8: 23, 3.3: 20, 1.8: 20, 0.8: 21, 0: 21},
                    }
                }
            }
        }
        merged_ = PolicyParserUtils.merge_nested_dicts(self.hierarchical_policy_config, \
                                                       hierarchical_policy_config_edge_b4_64g)
        self.hierarchical_policy_config = merged_.copy()

