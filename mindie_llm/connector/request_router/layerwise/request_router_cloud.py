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
import os
import sys
import time
import json
import threading
from enum import IntEnum
from pathlib import Path

from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end
from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.utils.layerwise.cloud_cut_inputdata import CloudCutInputData
from mindie_llm.connector.request_router.layerwise.request_router_lwd import DecisionMetadata, RequestInfo, \
    RequestRouterLwd, LastExecType, DecisionType, MASTER_ID, REQUEST_KEY_MAX, REQUEST_KEY_PREFILL, REQUEST_KEY_DECODE

sys.path.append(str(Path(__file__).parent / "sync"))

LAYERS_DIVI_MIN_NUM = 2


class CtrlTypePos(IntEnum):
    DECISION_TYPE = 0
    SHAPE_START = 1
    SHAPE_END = 2
    DIVI_NUM = 3
    CHUNK_NUM = 4
    REQUEST_KEY = 5
    PREFILL_DP_SEQ_LEN = 6
    MAX_NUM = 7


class RequestRouterCloud(RequestRouterLwd):
    def __init__(self, parent_pid):
        self.last_execute_type = None
        self.before_last_execute_type = None
        self.prefill_exec_last_time = None

        self.prefill_layers_divi_switch = False if os.getenv("PREFILL_LAYERS_DIVI_SWITCH", "on") == "off" else True
        
        self.is_doing_decode = False
        self.is_next_decode_arrived = False
        self.is_wait_prefill = False
        self.lock = threading.Lock()

        self.cloud_layer_num = 62

        self.isqwenvl = False   # 多模态
        self.lwd_multi_nodes_is_master = False # 多机

        self.process_func = {
            DecisionType.DO_PREFILL: self.do_prefill,
            DecisionType.DO_DECODE: self.do_decode,
            DecisionType.DO_CLEAN_UP: self.do_clean_up,
            DecisionType.DO_CLEAN_EOS: self.do_clean_eos,
        }

        super().__init__(parent_pid)

    def initialize_diff(self, model_config, models_config_dict):
        self.isqwenvl = True if model_config.get("model_name") == "qwen2_vl" else False
        self.lwd_multi_nodes_is_master = True if model_config.get('lwd_multi_nodes_is_master', 'false') == 'true' \
                                              else False

        start_layer_num = models_config_dict.get('startLayerNum', 1)
        end_layer_num = models_config_dict.get('endLayerNum', 1)
        model_runner_config = self.router_impl.generator.model_wrapper.model_runner.config
        total_layer_num = (64 if not hasattr(model_runner_config, 'num_hidden_layers')
            else model_runner_config.num_hidden_layers)
        self.cloud_layer_num = total_layer_num - start_layer_num - end_layer_num

        cloud_cut_instance = self.router_impl.generator.model_wrapper.model_runner.time_counter
        cut_num_range = (LAYERS_DIVI_MIN_NUM, self.cloud_layer_num)
        cloud_cut_instance.initialize("slave", self.rank, cut_num_range, self.lwd_multi_nodes_enable)

        #standard card
        edge_is_standard_card = self.ctrl_comm.edge_npu_smi_info.get("communication_backend") == 'hccl' if \
            self.ctrl_comm is not None and self.ctrl_comm.edge_npu_smi_info is not None else False
        if edge_is_standard_card:
            edge_soc_name = self.ctrl_comm.edge_npu_smi_info.get("soc_name")
            edge_hbm_cap = self.ctrl_comm.edge_npu_smi_info.get("hbm_capacity")
            if edge_soc_name and edge_hbm_cap:
                cloud_cut_instance.initialize_standard_card(edge_soc_name, edge_hbm_cap)
     
        prefill_layers_divi_num = int(os.getenv("PREFILL_CUT_NUM", "5"))
        prefill_layers_divi_policy = self.prepare_prefill_cut_policy(prefill_layers_divi_num)

        logger.info(f"[layerwiseDisaggregated] cloud initliaze ok rank:{self.rank}, "
            f"lwd_multi_nodes_is_master:{self.lwd_multi_nodes_is_master}, "
            f"start_layer_num:{start_layer_num}, end_layer_num:{end_layer_num}, "
            f"total_layer_num:{total_layer_num}, prefill_layers_divi_num:{prefill_layers_divi_num}, "
            f"prefill_layers_divi_policy:{prefill_layers_divi_policy}")

    def prepare_prefill_cut_policy(self, divi_num):
        divi_average_layer_num = int(self.cloud_layer_num / divi_num)
        mod = self.cloud_layer_num % divi_num
        prefill_layers_divi_policy = ([divi_average_layer_num] * divi_num if mod == 0 else 
            [divi_average_layer_num + 1] * mod + [divi_average_layer_num] * (divi_num - mod)
        )
        return prefill_layers_divi_policy
    
    def prepare_prefill_chunk_policy(self, prefill_seq_len, prefill_chunk_num):
        average_prefill_chunk_seq_len = int(prefill_seq_len / prefill_chunk_num)
        mod = prefill_seq_len % prefill_chunk_num
        prefill_chunk_policy = ([average_prefill_chunk_seq_len] * prefill_chunk_num if mod == 0 else 
            [average_prefill_chunk_seq_len + 1] * mod + [average_prefill_chunk_seq_len] * (prefill_chunk_num - mod)
        )
        return prefill_chunk_policy

    def get_prefill_gap_time_list(self, prefill_request):
        gap_time_list = []
        for seq_group_metadata in prefill_request.execute_model_request.seq_group_metadata_list:
            gap_time_list.append(seq_group_metadata.request_gap)
        return gap_time_list

    def update_prefill_layers_divi_num(self, request_key):
        prefill_request = self.request_map[REQUEST_KEY_PREFILL][request_key].request
        if not self.prefill_layers_divi_switch:
            return self.prepare_prefill_cut_policy(5)   # 关闭动态切块, 默认切5份

        cloud_cut_instance = self.router_impl.generator.model_wrapper.model_runner.time_counter
        gap_list = self.get_prefill_gap_time_list(prefill_request)
        prefill_dp_max_seq_len = self.request_map[REQUEST_KEY_PREFILL][request_key].prefill_dp_max_seq_len
        cut_data = CloudCutInputData(prefill_dp_max_seq_len, gap_list)
        prefill_layers_divi_num = cloud_cut_instance.get_cut_num(cut_data)
        self.request_map[REQUEST_KEY_PREFILL][request_key].layers_divi_num = prefill_layers_divi_num
        prefill_layers_divi_policy = self.prepare_prefill_cut_policy(prefill_layers_divi_num)
        return prefill_layers_divi_policy
    
    def update_prefill_long_seq_data(self, request_key, prefill_dp_seq_len):
        prefill_chunk_policy = self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, False)
        prefill_layers_divi_num = self.request_map[REQUEST_KEY_PREFILL][request_key].layers_divi_num
        self.request_map[REQUEST_KEY_PREFILL][request_key].layers_divi_num = round(
            prefill_layers_divi_num / len(prefill_chunk_policy))
        prefill_layers_divi_policy = self.prepare_prefill_cut_policy(
            self.request_map[REQUEST_KEY_PREFILL][request_key].layers_divi_num)

        edge_chunk_len_policy = self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, True)
        for chunk_idx in range(len(prefill_chunk_policy)):
            exec_start_layer = 0
            for layer in prefill_layers_divi_policy:
                exec_end_layer = exec_start_layer + layer
                metadata, need_recv_num = self.get_prefill_exec_metadata(exec_start_layer, exec_end_layer, request_key, 
                    chunk_policy=prefill_chunk_policy, chunk_idx=chunk_idx, 
                    edge_chunk_len_policy=edge_chunk_len_policy, prefill_dp_seq_len=prefill_dp_seq_len)
                long_seq_decision_meta = DecisionMetadata(1, 0, need_recv_num, 0)

                self.prefill_metadata_queue.append((request_key, metadata, long_seq_decision_meta))
                exec_start_layer = exec_end_layer

    def set_pd_curr_request(self):
        if not self.request_map[REQUEST_KEY_PREFILL] and not self.prefill_queue.empty():
            prefill_request = self.prefill_queue.get()
            prefill_request_info = RequestInfo(request=prefill_request)
            
            prefill_dp_seq_len = self.calc_curr_dp_seq_len(prefill_request)
            prefill_dp_max_seq_len = self.calc_max_seq_len(prefill_request)
            if prefill_dp_seq_len == 0:    # 当前batch是空
                prefill_dp_seq_len = 1     # 长度至少为1, 构造陪跑时的长度
                prefill_request_info.dp_empty = True
            if self.cp_size > 1:    # cp开启时, 每个cp域的hidden大小会不一样, 此时dp=1, 因此要修改
                prefill_dp_seq_len = self.calc_cp_seq_len(prefill_request)
            if self.isqwenvl:
                prefill_dp_seq_len = prefill_dp_seq_len * 2

            prefill_request_info.prefill_dp_len = prefill_dp_seq_len
            prefill_request_info.prefill_dp_max_seq_len = prefill_dp_max_seq_len
            self.request_map[REQUEST_KEY_PREFILL][self.request_map_prefill_key] = prefill_request_info

            recv_len = prefill_dp_seq_len
            recv_edge_list = []
            if prefill_dp_max_seq_len > self.get_long_seq_len_min():
                chunk_len_policy = self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, False)
                edge_chunk_len_policy = self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, True)
                recv_edge_list, _ = self.long_seq_get_recv_list(edge_chunk_len_policy, 0, chunk_len_policy[0])
                recv_len = recv_edge_list[0][1]
            p_comm_index = self.request_map_prefill_key % self.batch_p_num
            if len(recv_edge_list) > 1:
                self.data_comm.prefill_chunk_recv(recv_edge_list, p_comm_index)
            else:
                self.data_comm.p_shape[p_comm_index] = recv_len 
                self.data_comm.recv_hidden('p', self.data_comm.p_shape, recv_index=p_comm_index)    
            logger.info(f"[layerwiseDisaggregated] prefill recv_len {recv_len}, "
                        f"recv_edge_list {recv_edge_list} p_comm_index:{p_comm_index}, rank{self.rank}")
            
            if self.rank == MASTER_ID:
                self.prepare_prefill_request(self.request_map_prefill_key, prefill_dp_seq_len)
                self.request_map[REQUEST_KEY_PREFILL][self.request_map_prefill_key].prefill_dp_len = \
                    int(prefill_dp_seq_len)
            self.request_map_prefill_key = (self.request_map_prefill_key + 1) % REQUEST_KEY_MAX
            
        if not self.request_map[REQUEST_KEY_DECODE] and not self.decode_queue.empty():
            if not self.clean_eos_queue.empty():    # 下一个D需要等clean_eos做完, 否则可能出现prefill before decode
                return
            decode_request = self.decode_queue.get()
            decode_request_info = RequestInfo(request=decode_request)
            self.request_map[REQUEST_KEY_DECODE][self.request_map_decode_key] = decode_request_info
            self.prepare_decode_request(self.request_map_decode_key)
            self.request_map_decode_key = (self.request_map_decode_key + 1) % REQUEST_KEY_MAX

    def prepare_prefill_request(self, request_key, prefill_dp_seq_len):
        prefill_layers_divi_policy = self.update_prefill_layers_divi_num(request_key)
        prefill_dp_max_seq_len = self.request_map[REQUEST_KEY_PREFILL][request_key].prefill_dp_max_seq_len
        if prefill_dp_max_seq_len <= self.get_long_seq_len_min():
            exec_start_layer = 0
            for layer in prefill_layers_divi_policy:
                exec_end_layer = exec_start_layer + layer
                metadata, need_recv_num = self.get_prefill_exec_metadata(exec_start_layer, exec_end_layer, request_key)
                long_seq_decision_meta = DecisionMetadata(1, 0, need_recv_num, 0)
                self.prefill_metadata_queue.append((request_key, metadata, long_seq_decision_meta)) 
                exec_start_layer = exec_end_layer
        else:
            self.update_prefill_long_seq_data(request_key, prefill_dp_seq_len)
        logger.info(f"[layerwiseDisaggregated] prepare_prefill_request request_key {request_key}, rank{self.rank}, "
            f"prefill_metadata_queue {self.prefill_metadata_queue}")
        
    def prepare_decode_request(self, request_key):
        metadata = LwdMetadata(request_key, 0, self.cloud_layer_num, True, False, False, False,
                                   self.cloud_layer_num, False, 0, 0, 0, 0, False)
        self.decode_metadata_queue.append((request_key, metadata))

    def recv_prefill(self):
        if self.prefill_comm_finish or len(self.prefill_metadata_queue) == 0:
            return

        # 云侧一个大chunk等于边侧几个小chunk
        chunk_index_size = self.prefill_metadata_queue[0][2].chunk_index_size
        if self.prefill_comm_tcp_finish_count < chunk_index_size:
            self.ctrl_comm.recv_prefill()
            self.prefill_comm_tcp_finish_count += self.ctrl_comm.prefill_comm_finish_tcp_count
            self.ctrl_comm.prefill_comm_finish_tcp_count = 0

        if self.prefill_comm_tcp_finish_count >= chunk_index_size: 
            self.prefill_comm_finish = True
            logger.info(f"[layerwiseDisaggregated] recv_prefill tcp comm finish, prefill_comm_tcp_finish_count:"
                f"{self.prefill_comm_tcp_finish_count}, chunk_index_size:{chunk_index_size}") 
            self.prefill_comm_tcp_finish_count -= chunk_index_size

    def recv_ctrl_msg(self):
        if self.rank == MASTER_ID:
            self.recv_prefill() # 接收对方发来的prefill tcp控制信号
            self.recv_decode()  # 接收对方发来的decode tcp控制信号

    def master_rank_make_decision(self):
        # 双机云侧的slave节点只接收来自云master节点的决策
        if self.lwd_multi_nodes_enable and not self.lwd_multi_nodes_is_master:
            return self.recv_decision_from_master()
        else:
            # Determine the decision type for executing P/D based on the current information.
            decision_type = self.calc_decision_type()
            self.send_decision_to_slave(decision_type)
            return decision_type

    def send_decision_to_slave(self, decision_type):
        if self.lwd_multi_nodes_enable:
            send_message = {"decision_type": decision_type.value}
            send_str = json.dumps(send_message)
            self.ctrl_comm.broadcast_multi_nodes_decision(send_str)

    def recv_decision_from_master(self):
        recv_str = self.ctrl_comm.recv_multi_nodes_decision()
        recv_message = json.loads(recv_str)
        decision_type = DecisionType(recv_message["decision_type"])

        # 需要先准备好请求, 再准备metadata
        while (decision_type == DecisionType.DO_PREFILL and not self.request_map[REQUEST_KEY_PREFILL]) or \
              (decision_type == DecisionType.DO_DECODE and not self.request_map[REQUEST_KEY_DECODE]):
            self.get_all_request()
        return decision_type

    def decision_do_clean_eos_type(self):
        decision_type = DecisionType.WAIT_COMM
        if not self.clean_eos_queue.empty() and not self.decode_metadata_queue:
            decision_type = DecisionType.DO_CLEAN_EOS
            return decision_type

        return decision_type

    def decision_do_prefill_or_decode_type(self):
        # If both d and p are present, interleave p/d execution based on the previous execution type; when both are
        # present by default, prioritize p.
        decision_type = DecisionType.WAIT_COMM
        has_prefill_finish = self.prefill_metadata_queue and self.prefill_comm_finish
        has_decode_finish = self.decode_metadata_queue and self.decode_comm_finish
        
        if has_prefill_finish and has_decode_finish:
            if self.last_execute_type == LastExecType.DECODE:
                decision_type = DecisionType.DO_PREFILL
            elif self.last_execute_type == LastExecType.PREFILL:
                decision_type = DecisionType.DO_DECODE
            else:
                decision_type = DecisionType.DO_PREFILL
            return decision_type

        return decision_type

    def decision_do_decode_type(self):
        decision_type = DecisionType.WAIT_COMM
        has_decode_finish = self.decode_metadata_queue and self.decode_comm_finish
        if has_decode_finish and not self.is_wait_prefill:
            decision_type = DecisionType.DO_DECODE
            return decision_type

        return decision_type

    def decision_do_prefill_type(self):
        # If only p is present, execute p: the first two p executions require a 10ms delay to allow decode an
        # opportunity for interleaving.
        #  p p p d p d p   ^  p d d d d d p d p d p   ^  p p p d p d
        #                 10ms                       10ms
        decision_type = DecisionType.WAIT_COMM
        has_prefill_finish = self.prefill_metadata_queue and self.prefill_comm_finish
        if has_prefill_finish:
            if self.last_execute_type == LastExecType.PREFILL and \
                self.before_last_execute_type == LastExecType.DECODE:
                curr_time = time.time()
                if self.prefill_exec_last_time and \
                    curr_time - self.prefill_exec_last_time < 0.01 and self.clean_up_queue.empty():
                    decision_type = DecisionType.WAIT_DECODE
                else:
                    decision_type = DecisionType.DO_PREFILL
                return decision_type

            decision_type = DecisionType.DO_PREFILL
            return decision_type    

        return decision_type

    def calc_decision_type(self):
        decision_type = DecisionType.WAIT_COMM
        if self.decision_do_clean_eos_type() == DecisionType.DO_CLEAN_EOS:
            decision_type = DecisionType.DO_CLEAN_EOS
            return decision_type

        if self.decision_do_clean_up_type() == DecisionType.DO_CLEAN_UP:
            decision_type = DecisionType.DO_CLEAN_UP
            return decision_type

        decision_type = self.decision_do_prefill_or_decode_type()
        if decision_type != DecisionType.WAIT_COMM:
            return decision_type

        decision_type = self.decision_do_decode_type()
        if decision_type != DecisionType.WAIT_COMM:
            return decision_type
        
        decision_type = self.decision_do_prefill_type()
        if decision_type != DecisionType.WAIT_COMM:
            return decision_type

        return decision_type

    def broadcast_decision_type(self, decision_type, request_key):
        if self.process_func.get(decision_type) is None:   # 无需广播的决策
            return
        prefill_request_info = self.request_map[REQUEST_KEY_PREFILL][request_key] \
            if request_key in self.request_map[REQUEST_KEY_PREFILL] else None
        
        prefill_dp_seq_len = prefill_request_info.prefill_dp_len if prefill_request_info else -1 
        layers_divi_num = prefill_request_info.layers_divi_num if prefill_request_info else -1
        prefill_dp_max_seq_len = prefill_request_info.prefill_dp_max_seq_len if prefill_request_info else -1
        prefill_chunk_num = len(self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, False)) \
                if decision_type == DecisionType.DO_PREFILL and prefill_dp_max_seq_len > self.get_long_seq_len_min() \
                    else 1
        
        ctrl_tensor = [0] * CtrlTypePos.MAX_NUM
        ctrl_tensor[CtrlTypePos.DECISION_TYPE] = decision_type
        ctrl_tensor[CtrlTypePos.SHAPE_START] = -1
        ctrl_tensor[CtrlTypePos.SHAPE_END] = -1
        ctrl_tensor[CtrlTypePos.DIVI_NUM] = layers_divi_num
        ctrl_tensor[CtrlTypePos.CHUNK_NUM] = prefill_chunk_num
        ctrl_tensor[CtrlTypePos.REQUEST_KEY] = request_key
        ctrl_tensor[CtrlTypePos.PREFILL_DP_SEQ_LEN] = prefill_dp_seq_len
        self.mem_manager.write_list_memory(ctrl_tensor)

    def recv_do_prefill_type_update_policy(self, ctrl_tensor):
        prefill_layers_divi_num = int(ctrl_tensor[CtrlTypePos.DIVI_NUM])
        request_key = int(ctrl_tensor[CtrlTypePos.REQUEST_KEY])
        prefill_layers_divi_policy = self.prepare_prefill_cut_policy(prefill_layers_divi_num)
        prefill_chunk_num = int(ctrl_tensor[CtrlTypePos.CHUNK_NUM])
        prefill_dp_seq_len = int(ctrl_tensor[CtrlTypePos.PREFILL_DP_SEQ_LEN])
        if prefill_chunk_num > 1:  # is_long_seq
            prefill_chunk_policy = self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, False)
            edge_chunk_len_policy = self.prefill_chunk_instance.get_chunk_len_policy(prefill_dp_seq_len, True)
            for chunk_idx in range(len(prefill_chunk_policy)):
                exec_start_layer = 0
                for layer in prefill_layers_divi_policy:
                    exec_end_layer = exec_start_layer + layer
                    metadata, need_recv_num = self.get_prefill_exec_metadata(exec_start_layer, exec_end_layer,
                        request_key, chunk_policy=prefill_chunk_policy, chunk_idx=chunk_idx, 
                        edge_chunk_len_policy=edge_chunk_len_policy, prefill_dp_seq_len=prefill_dp_seq_len)
                    long_seq_decision_meta = DecisionMetadata(1, 0, need_recv_num, 0)
                    self.prefill_metadata_queue.append((request_key, metadata, long_seq_decision_meta)) 
                    exec_start_layer = exec_end_layer
        else:
            exec_start_layer = 0
            for layer in prefill_layers_divi_policy:
                exec_end_layer = exec_start_layer + layer
                metadata, need_recv_num = self.get_prefill_exec_metadata(exec_start_layer, exec_end_layer, request_key)
                long_seq_decision_meta = DecisionMetadata(1, 0, need_recv_num, 0)
                self.prefill_metadata_queue.append((request_key, metadata, long_seq_decision_meta))
                exec_start_layer = exec_end_layer
        logger.info(f"[layerwiseDisaggregated] prepare_prefill_request request_key {request_key}, rank{self.rank}, "
            f"prefill_metadata_queue {self.prefill_metadata_queue}")

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv decision type, ctrl_tensor:{ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            decision_type = DecisionType.DO_NOTHING
            return decision_type

        decision_type = DecisionType(ctrl_tensor[CtrlTypePos.DECISION_TYPE])
        shape = ctrl_tensor[CtrlTypePos.SHAPE_START: CtrlTypePos.SHAPE_END + 1]
        if decision_type == DecisionType.DO_DECODE:
            while not self.request_map[REQUEST_KEY_DECODE]:
                self.get_all_request()
            self.ctrl_comm.decode_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        elif decision_type == DecisionType.DO_PREFILL:
            while not self.request_map[REQUEST_KEY_PREFILL]:
                self.get_all_request()
            if not self.prefill_metadata_queue:
                self.recv_do_prefill_type_update_policy(ctrl_tensor)
                
            self.ctrl_comm.prefill_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        return decision_type

    def do_prefill_end_clear_data(self, request_key, metadata):
        if metadata.end_exec_layer == self.cloud_layer_num:
            self.prefill_comm_finish = False
        if metadata.end_of_generate_token:
            self.is_wait_prefill = False
            del self.request_map[REQUEST_KEY_PREFILL][request_key]

    def do_prefill(self, request_key):
        prof = span_start("Prefill")
        logger.info(f"[layerwiseDisaggregated] execute do_prefill before, rank:{self.rank}.")

        prefill_start_time = time.time()
        request = self.request_map[REQUEST_KEY_PREFILL][request_key].request
        metadata = lwd_metadata_manager.get_metadata()
        if metadata.is_dummy_batch:
            request.execute_model_request.forward_type = ForwardType.DUMMY
            self.router_impl.execute(request)
            logger.info(f"[layerwiseDisaggregated] execute do_prefill dummy end, rank:{self.rank}.")
        else:
            self.router_impl.execute(request)
        prefill_end_time = time.time()
        logger.info(f"[layerwiseDisaggregated] execute do_prefill "
                    f"exec layer:{metadata.start_exec_layer} - {metadata.end_exec_layer}, "
                    f"prefill_layers_divi_num:{self.request_map[REQUEST_KEY_PREFILL][request_key].layers_divi_num}, "
                    f"time exec cost {1000 * (prefill_end_time - prefill_start_time)}ms, rank{self.rank}.")
        self.do_prefill_end_clear_data(request_key, metadata)
        self.before_last_execute_type = self.last_execute_type
        self.last_execute_type = LastExecType.PREFILL
        self.prefill_exec_last_time = prefill_end_time
        span_end(prof)
        return

    def do_decode(self, request_key):
        prof = span_start("Decode")
        with self.lock:
            self.is_doing_decode = True
            self.is_next_decode_arrived = False
        logger.info(f"[layerwiseDisaggregated] execute do_decode before, rank:{self.rank}.")

        request = self.request_map[REQUEST_KEY_DECODE][request_key].request
        metadata = lwd_metadata_manager.get_metadata()
        decode_start_time = time.time()
        self.router_impl.execute(request)
        decode_end_time = time.time()
        self.before_last_execute_type = self.last_execute_type
        self.last_execute_type = LastExecType.DECODE
        self.decode_comm_finish = False 
        self.ctrl_comm.decode_comm_finish = False
        logger.info(f"[layerwiseDisaggregated] execute do_decode, time exec cost "
            f"{1000 * (decode_end_time - decode_start_time)}ms, rank{self.rank}.")
        with self.lock:
            self.is_doing_decode = False
        if metadata.end_of_generate_token:
            del self.request_map[REQUEST_KEY_DECODE][request_key]
        span_end(prof)
        return

    def long_seq_get_recv_list(self, edge_chunk_len_policy, start_offset, end_offset):
        tmp_edge_len = 0
        recv_list = []
        recv_hidden_pos = 0
        for chunk_len in edge_chunk_len_policy:
            if tmp_edge_len >= end_offset:
                break

            recv_hidden_pos = tmp_edge_len - start_offset
            if tmp_edge_len >= start_offset:
                recv_list.append((recv_hidden_pos, chunk_len))

            tmp_edge_len += chunk_len

        # 这个P首需要接收N次tcp的数量才能执行一次P
        return recv_list, recv_hidden_pos

    def get_prefill_exec_metadata(self, exec_start_layer, exec_end_layer, request_key, **kwargs):
        chunk_policy = kwargs['chunk_policy'] if 'chunk_policy' in kwargs else None
        chunk_idx = kwargs['chunk_idx'] if 'chunk_idx' in kwargs else None
        edge_chunk_len_policy = kwargs['edge_chunk_len_policy'] if 'edge_chunk_len_policy' in kwargs else None
        prefill_dp_seq_len = kwargs['prefill_dp_seq_len'] if 'prefill_dp_seq_len' in kwargs else 0
        
        is_long_seq = self.request_map[REQUEST_KEY_PREFILL][request_key].prefill_dp_max_seq_len > \
            self.get_long_seq_len_min()
        prefill_dp_seq_len = self.request_map[REQUEST_KEY_PREFILL][request_key].prefill_dp_len
        curr_dp_empty = self.request_map[REQUEST_KEY_PREFILL][request_key].dp_empty
        end_of_generate_token = exec_end_layer >= self.cloud_layer_num
        metadata = LwdMetadata(request_key, exec_start_layer, exec_end_layer, end_of_generate_token, 
            True, False, curr_dp_empty, self.cloud_layer_num, is_long_seq, 0, 0, 0, prefill_dp_seq_len, False)
        
        need_recv_tcp_num_finish = 1
        if is_long_seq:
            chunk_start_offset = sum(chunk_policy[:chunk_idx])
            chunk_end_offset = chunk_start_offset + chunk_policy[chunk_idx]
            prefill_chunk_num = len(chunk_policy)
            
            metadata.long_seq_start_idx = chunk_start_offset
            metadata.long_seq_end_idx = chunk_end_offset

            if chunk_idx == prefill_chunk_num - 1:
                metadata.is_last_chunk = True

            if exec_end_layer >= self.cloud_layer_num:
                # 长序列的某个域是短序列, 就只做一段chunk
                if prefill_dp_seq_len <= self.get_long_seq_len_min() / 2:
                    metadata.end_of_generate_token = True if chunk_idx == 0 else False
                    metadata.is_dummy_batch = True if chunk_idx > 0 else False
                else:
                    metadata.end_of_generate_token = True if metadata.is_last_chunk else False

            # 云的第一层, 需要计算下一段的接收长度
            if exec_start_layer == 0:
                curr_recv_list, curr_recv_hidden_pos = self.long_seq_get_recv_list(edge_chunk_len_policy,
                    chunk_start_offset, chunk_end_offset)
                next_recv_list = []
                if not metadata.is_last_chunk:
                    next_chunk_end_offset = chunk_end_offset + chunk_policy[chunk_idx + 1]
                    next_recv_list, _ = self.long_seq_get_recv_list(edge_chunk_len_policy, 
                                                                    chunk_end_offset, next_chunk_end_offset)
                logger.info(f"[layerwiseDisaggregated]curr_recv_list:{curr_recv_list} ,"
                            f" next_recv_list:{next_recv_list}, chunk_idx:{chunk_idx}, "
                            f"prefill_chunk_num:{prefill_chunk_num}, rank{self.rank}")

                # 这个P首需要接收N次tcp的数量才能执行一次P
                need_recv_tcp_num_finish = len(curr_recv_list)
                # 使用最后一小段的接收起点
                metadata.hidden_start_pos = curr_recv_hidden_pos
                # 非首段chunk的接收, 因为第一段chunk已经在set_pd_curr_request接收了; 最后一段chunk无需再提前挂接收
                metadata.long_seq_recv_list = next_recv_list

        return metadata, need_recv_tcp_num_finish

    def arrange_exec_stage(self, decision_type: DecisionType):
        metadata = None
        request_key = -1
        if decision_type == DecisionType.DO_DECODE:
            request_key, metadata = self.decode_metadata_queue.popleft()
        elif decision_type == DecisionType.DO_PREFILL:
            request_key, metadata, _ = self.prefill_metadata_queue.popleft()
        if metadata:
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated]exec {decision_type.name}, rank{self.rank} set metadata: {metadata}")
        return request_key

    def accept_decode_prepare(self, execute_request: ExecuteRequest):
        batch_size = self.calc_curr_dp_batch_size(execute_request)  # 兼容单dp, 当前dp域是空时, BS返回1
        if self.isqwenvl:
            batch_size = batch_size + 1
        self.data_comm.decode_batch_size_queue.put(batch_size)

        with self.data_comm.lock:
            logger.info(f"[layerwiseDisaggregated] req_router, pre recv "
                f"{self.data_comm.decode_batch_size_queue.qsize()} {self.data_comm.flag_pre_recv}")
            if self.data_comm.flag_pre_recv:
                self.data_comm.d_shape = self.data_comm.decode_batch_size_queue.get()
                self.data_comm.recv_hidden('d', self.data_comm.d_shape)
                self.data_comm.flag_pre_recv = False

        logger.info(f"[layerwiseDisaggregated]cloud decode batch size putted, rank{self.rank} batch_size: {batch_size}")

    def accept(self, execute_request: ExecuteRequest):
        if execute_request.execute_type == ExecuteType.MODEL_INFER:
            forward_type = execute_request.execute_model_request.forward_type
            if forward_type == ForwardType.DECODE:
                with self.lock:
                    self.is_next_decode_arrived = True
                self.accept_decode_prepare(execute_request)
            elif forward_type == ForwardType.PREFILL:
                with self.lock:
                    self.is_wait_prefill = self.is_wait_prefill or \
                        self.is_doing_decode and not self.is_next_decode_arrived
            self.inference_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.PD_LINK:
            self.link_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.KV_TRANSFER:
            self.transfer_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.LORA_OPERATION:
            self.command_queue.put(execute_request)
        else:
            self.inference_queue.put(execute_request)