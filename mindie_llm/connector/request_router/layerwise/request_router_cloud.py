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
from mindie_llm.connector.request_router.layerwise.request_router_lwd import RequestRouterLwd, LastExecType, \
    DecisionType, MASTER_ID

sys.path.append(str(Path(__file__).parent / "sync"))

LAYERS_DIVI_MIN_NUM = 2


class CtrlTypePos(IntEnum):
    DECISION_TYPE = 0
    SHAPE_START = 1
    SHAPE_END = 2
    DIVI_NUM = 3
    CHUNK_NUM = 4
    MAX_NUM = 5


class RequestRouterCloud(RequestRouterLwd):
    def __init__(self, parent_pid):
        self.last_execute_type = None
        self.before_last_execute_type = None

        self.prefill_exec_last_time = None
        self.prefill_exec_start_layer = 0
        self.prefill_exec_cnt = 0

        self.prefill_layers_divi_policy = [13] * 2 + [12] * 3
        self.prefill_layers_divi_num = int(os.getenv("PREFILL_CUT_NUM", "5"))
        self.prefill_layers_divi_switch = False if os.getenv("PREFILL_LAYERS_DIVI_SWITCH", "on") == "off" else True

        self.is_long_seq = False
        self.prefill_exec_chunk_cnt = 0
        self.prefill_chunk_policy = None
        self.prefill_chunk_num = 1
        self.long_seq_start_idx = 0
        
        self.is_doing_decode = False
        self.is_next_decode_arrived = False
        self.is_wait_prefill = False
        self.lock = threading.Lock()

        self.total_layer_num = 64
        self.start_layer_num = 1
        self.end_layer_num = 1
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

        self.start_layer_num = models_config_dict.get('startLayerNum', 1)
        self.end_layer_num = models_config_dict.get('endLayerNum', 1)
        model_runner_config = self.router_impl.generator.model_wrapper.model_runner.config
        self.total_layer_num = (64 if not hasattr(model_runner_config, 'num_hidden_layers')
            else model_runner_config.num_hidden_layers)
        self.cloud_layer_num = self.total_layer_num - self.start_layer_num - self.end_layer_num

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
     
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)

        logger.info(f"[layerwiseDisaggregated] cloud initliaze ok rank:{self.rank}, "
            f"lwd_multi_nodes_is_master:{self.lwd_multi_nodes_is_master}, "
            f"start_layer_num:{self.start_layer_num}, end_layer_num:{self.end_layer_num}, "
            f"total_layer_num:{self.total_layer_num}, prefill_layers_divi_num:{self.prefill_layers_divi_num}, "
            f"prefill_layers_divi_policy:{self.prefill_layers_divi_policy}")

    def prepare_prefill_cut_policy(self, divi_num):
        self.prefill_layers_divi_num = divi_num
        divi_average_layer_num = int(self.cloud_layer_num / self.prefill_layers_divi_num)
        mod = self.cloud_layer_num % self.prefill_layers_divi_num
        self.prefill_layers_divi_policy = ([divi_average_layer_num] * self.prefill_layers_divi_num if mod == 0 else 
            [divi_average_layer_num + 1] * mod + [divi_average_layer_num] * (self.prefill_layers_divi_num - mod)
        )
    
    def prepare_prefill_chunk_policy(self, prefill_seq_len, prefill_chunk_num):
        if self.cp_size > 1:
            align_size = 8 * self.router_impl.block_size        # 云侧8卡
            average_prefill_chunk_seq_len = int(prefill_seq_len / prefill_chunk_num) // align_size * align_size
            mod = prefill_seq_len - prefill_chunk_num * average_prefill_chunk_seq_len
            self.prefill_chunk_policy = ([average_prefill_chunk_seq_len] * prefill_chunk_num 
                if mod == 0 else [average_prefill_chunk_seq_len] * (prefill_chunk_num - 1) +
                [average_prefill_chunk_seq_len + mod]
            )
        else:
            average_prefill_chunk_seq_len = int(prefill_seq_len / prefill_chunk_num)
            mod = prefill_seq_len % prefill_chunk_num
            self.prefill_chunk_policy = ([average_prefill_chunk_seq_len] * prefill_chunk_num
                if mod == 0 else [average_prefill_chunk_seq_len + 1] * mod +
                [average_prefill_chunk_seq_len] * (prefill_chunk_num - mod)
            )

    def get_prefill_gap_time_list(self, prefill_request):
        gap_time_list = []
        for seq_group_metadata in prefill_request.execute_model_request.seq_group_metadata_list:
            gap_time_list.append(seq_group_metadata.request_gap)
        return gap_time_list

    def update_prefill_layers_divi_num(self):
        if not self.prefill_layers_divi_switch:
            return

        cloud_cut_instance = self.router_impl.generator.model_wrapper.model_runner.time_counter
        gap_list = self.get_prefill_gap_time_list(self.prefill_request)
        self.prefill_layers_divi_num = cloud_cut_instance.get_cut_num(
                                                            CloudCutInputData(self.prefill_dp_max_seq_len, gap_list))
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)

    def update_prefill_long_seq_data(self):
        if self.prefill_dp_max_seq_len <= self.get_long_seq_len_min():
            return

        self.is_long_seq = True
        self.prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(self.prefill_dp_max_seq_len)
        self.prepare_prefill_chunk_policy(self.prefill_dp_seq_len, self.prefill_chunk_num)
        self.prefill_layers_divi_num = round(self.prefill_layers_divi_num / self.prefill_chunk_num)
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)

    def set_pd_curr_request(self):
        if self.prefill_request is None and not self.prefill_queue.empty():
            self.prefill_request = self.prefill_queue.get()

            self.prefill_dp_seq_len = self.calc_curr_dp_seq_len(self.prefill_request)
            self.prefill_dp_max_seq_len = self.calc_max_seq_len(self.prefill_request)
            self.prefill_dp_empty = False
            if self.prefill_dp_seq_len == 0:    # 当前batch是空
                self.prefill_dp_seq_len = 1     # 长度至少为1, 构造陪跑时的长度
                self.prefill_dp_empty = True
                
            self.prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(self.prefill_dp_max_seq_len) \
                if self.prefill_dp_max_seq_len > self.get_long_seq_len_min() else 1
            if self.isqwenvl:
                self.prefill_dp_seq_len = self.prefill_dp_seq_len * 2
            if self.cp_size > 1:    # cp开启时, 每个cp域的hidden大小会不一样, 此时dp=1, 因此要修改
                self.prefill_dp_seq_len = self.calc_cp_seq_len(self.prefill_request)
            self.data_comm.p_shape[self.data_comm.recv_index] = \
                math.ceil(self.prefill_dp_seq_len / self.prefill_chunk_num) 
            self.data_comm.recv_hidden('p', self.data_comm.p_shape)

            if self.rank == MASTER_ID:
                self.update_prefill_layers_divi_num()
                self.update_prefill_long_seq_data()

        if self.decode_request is None and not self.decode_queue.empty():
            if not self.clean_eos_queue.empty():    # 下一个D需要等clean_eos做完, 否则可能出现prefill before decode
                return
            self.decode_request = self.decode_queue.get()

    def recv_ctrl_msg(self):
        if self.rank == MASTER_ID:
            self.recv_prefill() # 接收对方发来的prefill tcp控制信号
            self.recv_decode()  # 接收对方发来的decode tcp控制信号

    def master_rank_make_decision(self):
        # 双机云侧的slave节点只接收来自云master节点的决策
        if self.lwd_multi_nodes_enable and not self.lwd_multi_nodes_is_master:
            self.recv_decision_from_master()
        else:
            # Determine the decision type for executing P/D based on the current information.
            self.calc_decision_type()
            self.send_decision_to_slave()

    def send_decision_to_slave(self):
        if self.lwd_multi_nodes_enable:
            send_message = {"decision_type": self.decision_type}
            send_str = json.dumps(send_message)
            self.ctrl_comm.broadcast_multi_nodes_decision(send_str)

    def recv_decision_from_master(self):
        recv_str = self.ctrl_comm.recv_multi_nodes_decision()
        recv_message = json.loads(recv_str)
        self.decision_type = DecisionType(recv_message["decision_type"])
        while self.decision_type == DecisionType.DO_PREFILL and self.prefill_request is None:
            self.get_all_request()

    def decision_do_clean_eos_type(self):
        if not self.clean_eos_queue.empty() and self.decode_request is None:
            self.decision_type = DecisionType.DO_CLEAN_EOS
            return True

        return False

    def decision_do_prefill_or_decode_type(self):
        # If both d and p are present, interleave p/d execution based on the previous execution type; when both are
        # present by default, prioritize p.
        has_prefill_finish = self.prefill_request and self.prefill_comm_finish
        has_decode_finish = self.decode_request and self.decode_comm_finish
        if has_prefill_finish and has_decode_finish:
            if self.last_execute_type == LastExecType.DECODE:
                self.decision_type = DecisionType.DO_PREFILL
            elif self.last_execute_type == LastExecType.PREFILL:
                self.decision_type = DecisionType.DO_DECODE
            else:
                self.decision_type = DecisionType.DO_PREFILL
            return True

        return False

    def decision_do_decode_type(self):
        has_decode_finish = self.decode_request and self.decode_comm_finish
        if has_decode_finish and not self.is_wait_prefill:
            self.decision_type = DecisionType.DO_DECODE
            return True

        return False

    def decision_do_prefill_type(self):
        # If only p is present, execute p: the first two p executions require a 10ms delay to allow decode an
        # opportunity for interleaving.
        #  p p p d p d p   ^  p d d d d d p d p d p   ^  p p p d p d
        #                 10ms                       10ms
        has_prefill_finish = self.prefill_request and self.prefill_comm_finish
        if has_prefill_finish:
            if self.last_execute_type == LastExecType.PREFILL and \
                self.before_last_execute_type == LastExecType.DECODE:
                curr_time = time.time()
                if self.prefill_exec_last_time and \
                    curr_time - self.prefill_exec_last_time < 0.01 and self.clean_up_queue.empty():
                    self.decision_type = DecisionType.WAIT_DECODE
                else:
                    self.decision_type = DecisionType.DO_PREFILL
                return True

            self.decision_type = DecisionType.DO_PREFILL
            return True

        return False

    def calc_decision_type(self):
        if self.decision_do_clean_eos_type():
            return

        if self.decision_do_clean_up_type():
            return

        if self.decision_do_prefill_or_decode_type():
            return

        if self.decision_do_decode_type():
            return
        
        if self.decision_do_prefill_type():
            return

        self.decision_type = DecisionType.WAIT_COMM

    def broadcast_decision_type(self):
        if self.process_func.get(self.decision_type) is None:   # 无需广播的决策
            return

        ctrl_tensor = [0] * CtrlTypePos.MAX_NUM
        ctrl_tensor[CtrlTypePos.DECISION_TYPE] = self.decision_type
        ctrl_tensor[CtrlTypePos.SHAPE_START] = -1
        ctrl_tensor[CtrlTypePos.SHAPE_END] = -1
        ctrl_tensor[CtrlTypePos.DIVI_NUM] = self.prefill_layers_divi_num
        ctrl_tensor[CtrlTypePos.CHUNK_NUM] = self.prefill_chunk_num
        self.mem_manager.write_list_memory(ctrl_tensor)

    def recv_do_prefill_type_update_policy(self, ctrl_tensor):
        self.prefill_layers_divi_num = int(ctrl_tensor[CtrlTypePos.DIVI_NUM])
        self.prepare_prefill_cut_policy(self.prefill_layers_divi_num)
        self.prefill_chunk_num = int(ctrl_tensor[CtrlTypePos.CHUNK_NUM])
        if self.prefill_chunk_num > 1:
            self.is_long_seq = True
            self.prepare_prefill_chunk_policy(self.prefill_dp_seq_len, self.prefill_chunk_num)

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv decision type, ctrl_tensor:{ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            self.decision_type = DecisionType.DO_NOTHING
            return

        self.decision_type = DecisionType(ctrl_tensor[CtrlTypePos.DECISION_TYPE])
        shape = ctrl_tensor[CtrlTypePos.SHAPE_START: CtrlTypePos.SHAPE_END + 1]
        if self.decision_type == DecisionType.DO_DECODE:
            self.ctrl_comm.decode_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        elif self.decision_type == DecisionType.DO_PREFILL:
            while self.prefill_request is None:
                self.get_all_request()

            if self.prefill_exec_cnt == 0:
                self.recv_do_prefill_type_update_policy(ctrl_tensor)
            self.ctrl_comm.prefill_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        return

    def do_prefill_end_clear_data(self):
        if self.prefill_exec_cnt >= self.prefill_layers_divi_num and \
            (not self.is_long_seq or self.prefill_exec_chunk_cnt >= self.prefill_chunk_num):
            self.prefill_exec_cnt = 0
            self.prefill_exec_start_layer = 0
            self.prefill_request = None
            self.prefill_comm_finish = False
            self.is_long_seq = False
            self.prefill_exec_chunk_cnt = 0
            self.prefill_chunk_num = 0
            self.long_seq_start_idx = 0
            self.is_wait_prefill = False

    def do_prefill(self):
        prof = span_start("Prefill")
        logger.info(f"[layerwiseDisaggregated] execute do_prefill before, rank:{self.rank}.")
        while self.prefill_request is None:
            self.get_all_request()
        prefill_start_time = time.time()
        if not self.prefill_dp_empty and self.prefill_exec_chunk_cnt > self.prefill_dp_seq_len:
            self.prefill_request.execute_model_request.forward_type = ForwardType.DUMMY
            self.router_impl.execute(self.prefill_request)
            logger.info(f"[layerwiseDisaggregated] execute do_prefill dummy end, rank:{self.rank}.")
        else:
            self.router_impl.execute(self.prefill_request)
        prefill_end_time = time.time()
        logger.info(f"[layerwiseDisaggregated] execute do_prefill exec layer cnt:{self.prefill_exec_cnt}, "
            f"prefill_layers_divi_num:{self.prefill_layers_divi_num}, "
            f"exec chunk cnt:{self.prefill_exec_chunk_cnt} prefill_chunk_num:{self.prefill_chunk_num}, "
            f"time exec cost {1000 * (prefill_end_time - prefill_start_time)}ms, rank{self.rank}.")
        self.do_prefill_end_clear_data()
        self.before_last_execute_type = self.last_execute_type
        self.last_execute_type = LastExecType.PREFILL
        self.prefill_exec_last_time = prefill_end_time
        span_end(prof)
        return

    def do_decode(self):
        prof = span_start("Decode")
        with self.lock:
            self.is_doing_decode = True
            self.is_next_decode_arrived = False
        logger.info(f"[layerwiseDisaggregated] execute do_decode before, rank:{self.rank}.")
        while self.decode_request is None:
            self.get_all_request()
        decode_start_time = time.time()
        self.router_impl.execute(self.decode_request)
        decode_end_time = time.time()
        self.before_last_execute_type = self.last_execute_type
        self.last_execute_type = LastExecType.DECODE
        self.decode_comm_finish = False 
        self.ctrl_comm.decode_comm_finish = False
        self.decode_request = None
        logger.info(f"[layerwiseDisaggregated] execute do_decode, time exec cost "
            f"{1000 * (decode_end_time - decode_start_time)}ms, rank{self.rank}.")
        with self.lock:
            self.is_doing_decode = False
        span_end(prof)
        return

    def get_prefill_exec_metadata(self):
        chunk_end_input_offset = 0
        chunk_next_end_input_offset = 0
        if self.is_long_seq:
            chunk_end_input_offset = self.long_seq_start_idx + self.prefill_chunk_policy[self.prefill_exec_chunk_cnt]
            chunk_next_end_input_offset = (
                chunk_end_input_offset + self.prefill_chunk_policy[self.prefill_exec_chunk_cnt + 1] 
                if chunk_end_input_offset < self.prefill_dp_seq_len else 0
            )

        curr_dp_empty = True if self.prefill_dp_empty else False
        exec_end_layer = self.prefill_exec_start_layer + self.prefill_layers_divi_policy[self.prefill_exec_cnt]
        metadata = LwdMetadata(self.prefill_exec_start_layer, exec_end_layer, False, True, False, curr_dp_empty,
            self.cloud_layer_num, self.is_long_seq, self.long_seq_start_idx,
            chunk_end_input_offset, chunk_next_end_input_offset, self.prefill_dp_seq_len, False)

        self.prefill_exec_start_layer = exec_end_layer
        self.prefill_exec_cnt += 1

        # 最后一段序列, 要放在这里判断, 因为序列在最后一层的时候 + 1, 比如执行4chunk会出现cnt 从0->4
        if self.is_long_seq and self.prefill_exec_chunk_cnt == self.prefill_chunk_num - 1:
            metadata.is_last_chunk = True

        # 处理执行结束层超过云层数量的情况
        if exec_end_layer >= self.cloud_layer_num:
            # 长短序列都先重置层的执行状态
            self.prefill_exec_start_layer = 0
            metadata.end_of_generate_token = True

            # 长序列处理逻辑, 覆盖部分赋值
            if self.is_long_seq:
                self.prefill_exec_chunk_cnt += 1    # 执行chunk + 1
                if chunk_end_input_offset < self.prefill_dp_seq_len:
                    # 长序列且未处理完所有分块
                    self.prefill_exec_cnt = 0   # 重置执行层
                    self.long_seq_start_idx = chunk_end_input_offset    # 序列长度往下走
                    metadata.end_of_generate_token = False
                else:
                    # 长序列且已处理完当前分块
                    self.long_seq_start_idx = 0
                    metadata.end_of_generate_token = True   # 在最后一段真序列结束要生产token

                    # 处理长序列特殊情况：分块计数超过序列长度
                    if (self.prefill_exec_chunk_cnt >= self.prefill_dp_seq_len and 
                        self.prefill_exec_chunk_cnt < self.prefill_chunk_num):
                        self.long_seq_start_idx = chunk_end_input_offset    # 保持下发的起点
                        metadata.end_of_generate_token = False              # 超过序列长度不需要再生产token了
                        self.prefill_exec_cnt = 0

        return metadata

    def arrange_exec_stage(self):
        if self.decision_type == DecisionType.DO_DECODE:
            metadata = LwdMetadata(0, self.cloud_layer_num, True, False, False, False,
                                   self.cloud_layer_num, False, 0, 0, 0, 0, False)
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated]exec decode, rank{self.rank} set metadata: {metadata}")
        elif self.decision_type == DecisionType.DO_PREFILL:
            metadata = self.get_prefill_exec_metadata()
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated]exec prefill, rank{self.rank} set metadata: {metadata}")

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
