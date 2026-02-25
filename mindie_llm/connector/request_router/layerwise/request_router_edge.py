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
import sys
import time
import itertools
from enum import IntEnum
from pathlib import Path

from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end
from mindie_llm.connector.common.model_execute_data_pb2 import ForwardType
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.connector.request_router.layerwise.request_router_lwd import RequestRouterLwd, DecisionType, \
    MASTER_ID

sys.path.append(str(Path(__file__).parent / "sync"))


class CtrlTypePos(IntEnum):
    DECISION_TYPE = 0
    SHAPE_START = 1
    SHAPE_END = 2
    MAX_NUM = 3


class RequestRouterEdge(RequestRouterLwd):
    def __init__(self, parent_pid):
        self.prefill_first = False  
        self.decode_first = False 
        
        self.prefill_last_queue = queue.Queue()
        self.prefill_last_request = None
        self.wait_prefill_last = False
        self.wait_decode = False
        self.is_all_last = False
        

        '''
        Wait for the next Decode with timing; 
        non-empty indicates the previous execution was a D-end task.
        '''
        self.force_wait_d_time = None
        
        self.prefill_chunk_metadata_queue = queue.Queue()
        self.prefill_chunk_all_last_metadata_queue = queue.Queue()
        self.is_last_do_chunk_prefill = False

        self.process_func = {
            DecisionType.DO_PREFILL_FIRST: self.do_prefill_first,
            DecisionType.DO_PREFILL_LAST: self.do_prefill_last,
            DecisionType.DO_DECODE_FIRST: self.do_decode_first,
            DecisionType.DO_DECODE_LAST: self.do_decode_last,
            DecisionType.DO_CLEAN_UP: self.do_clean_up,
            DecisionType.DO_CLEAN_EOS: self.do_clean_eos,
        }
        
        super().__init__(parent_pid)

    def do_prefill_first(self):
        prof = span_start("Prefill_first")
        logger.info(f"[layerwiseDisaggregated] execute prefill_first before, rank:{self.rank}.")
        metadata = lwd_metadata_manager.get_metadata()
        if metadata.is_dummy_batch:  # 如果取出来的是结束了的, 就跑一个空batch, 否则执行prefill
            self.prefill_request.execute_model_request.forward_type = ForwardType.DUMMY
            self.router_impl.execute(self.prefill_request)
            self.prefill_request.execute_model_request.forward_type = ForwardType.PREFILL
            logger.info(f"[layerwiseDisaggregated] execute prefill_first dummy end, rank:{self.rank}.")
        else:
            self.router_impl.execute(self.prefill_request)
            logger.info(f"[layerwiseDisaggregated] execute prefill_first end, rank:{self.rank}.")
        if not self.is_request_long_seq(self.prefill_request):
            self.prefill_last_queue.put(self.prefill_request)
            self.prefill_request = None
            self.prefill_first = False
            self.is_last_do_chunk_prefill = False
        else:
            self.prefill_chunk_variable_policy(metadata)
        span_end(prof)

    def do_prefill_last(self):
        prof = span_start("Prefill_last")
        logger.info(f"[layerwiseDisaggregated] execute prefill_last before, rank:{self.rank}.")
        metadata = lwd_metadata_manager.get_metadata()
        if metadata.is_dummy_batch:  # 如果取出来的是结束了的, 就跑一个空batch, 否则执行prefill
            self.prefill_last_request.execute_model_request.forward_type = ForwardType.DUMMY
            self.router_impl.execute(self.prefill_last_request)
            self.prefill_last_request.execute_model_request.forward_type = ForwardType.PREFILL
            logger.info(f"[layerwiseDisaggregated] execute prefill_last dummy end, rank:{self.rank}.")
        else:
            self.router_impl.execute(self.prefill_last_request)
            logger.info(f"[layerwiseDisaggregated] execute prefill_last end, rank:{self.rank}.")
        if not self.is_request_long_seq(self.prefill_last_request):
            self.prefill_comm_finish = False
            self.prefill_last_request = None
            self.is_last_do_chunk_prefill = False
        else:
            self.prefill_chunk_variable_policy(metadata)
        span_end(prof)

    def do_decode_first(self):
        prof = span_start("Decode_first")
        logger.info(f"[layerwiseDisaggregated] execute decode_first before, rank:{self.rank}.")
        self.router_impl.execute(self.decode_request)
        logger.info(f"[layerwiseDisaggregated] execute decode_first end, rank:{self.rank}.")
        self.decode_first = False
        self.is_last_do_chunk_prefill = False
        self.wait_decode = False
        span_end(prof)
    
    def do_decode_last(self):
        prof = span_start("Decode_last")
        logger.info(f"[layerwiseDisaggregated] execute decode_last before, rank:{self.rank}.")
        self.router_impl.execute(self.decode_request)
        if self.rank == MASTER_ID:
            self.force_wait_d_time = time.time()     # Wait for the next Decode with timing.
        logger.info(f"[layerwiseDisaggregated] execute decode_last end, rank:{self.rank}.")
        self.decode_comm_finish = False
        self.ctrl_comm.decode_comm_finish = False
        self.decode_request = None
        self.is_last_do_chunk_prefill = False
        self.wait_decode = False
        span_end(prof)

    def do_clean_eos(self):
        while self.clean_eos_queue.empty():
            time.sleep(0.001)
            self.get_all_request()

        self.clean_eos_queue.get()  # 边侧推出eos会自动清理cache, 下eos请求下给云侧清理cache使用
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean eos, rank{self.rank}.")

    def recv_decode(self):
        self.ctrl_comm.recv_decode()
        self.decode_comm_finish = self.ctrl_comm.decode_comm_finish
        logger.info(f"[layerwiseDisaggregated] decode_comm_finish = {self.decode_comm_finish}")

    def recv_ctrl_msg(self):
        # 这里应该是多DP的主节点需要接收, 其他卡无需接收
        is_need_recv = self.lwd_multi_nodes_enable or (not self.lwd_multi_nodes_enable and self.rank == MASTER_ID)

        if is_need_recv:
            self.recv_prefill() # 接收对方发来的prefill tcp控制信号
            self.recv_decode()  # 接收对方发来的decode tcp控制信号

    def check_10ms_for_next_decode(self):
        if self.force_wait_d_time is None:
            return
        self.wait_d_time_gap = time.time() - self.force_wait_d_time
        if self.wait_d_time_gap > 0.01:
            logger.info(f"[layerwiseDisaggregated] Force wait decode state exit, "
                f"wait time: {self.wait_d_time_gap * 1000} ms.")
            self.force_wait_d_time = None
            self.wait_d_time_gap = 0

    def calc_prefill_priority_decision_type(self):
        is_prefill_first_ready = self.prefill_request and self.prefill_first
        if self.force_wait_d_time and self.decode_request and self.decode_first:
            self.decision_type = DecisionType.DO_DECODE_FIRST
            self.force_wait_d_time = None
            logger.info(f"[layerwiseDisaggregated] Force to do decode first, "
                f"wait time: {self.wait_d_time_gap * 1000} ms.")
            self.wait_d_time_gap = 0
        elif self.force_wait_d_time:
            self.decision_type = DecisionType.WAIT_DECODE
        elif is_prefill_first_ready and not self.wait_prefill_last and not self.wait_decode:
            self.decision_type = DecisionType.DO_PREFILL_FIRST
        elif self.prefill_last_request and self.prefill_comm_finish:
            self.decision_type = DecisionType.DO_PREFILL_LAST
        elif self.decode_request and self.decode_first:
            self.decision_type = DecisionType.DO_DECODE_FIRST
        elif self.decode_request and self.decode_comm_finish:
            self.decision_type = DecisionType.DO_DECODE_LAST

    def calc_decode_priority_decision_type(self):
        is_prefill_first_ready = self.prefill_request and self.prefill_first
        if self.decode_request and self.decode_first:
            self.decision_type = DecisionType.DO_DECODE_FIRST
        elif self.decode_request and self.decode_comm_finish:
            self.decision_type = DecisionType.DO_DECODE_LAST
        elif is_prefill_first_ready and not self.wait_prefill_last and not self.wait_decode:
            self.decision_type = DecisionType.DO_PREFILL_FIRST
        elif self.prefill_last_request and self.prefill_comm_finish:
            self.decision_type = DecisionType.DO_PREFILL_LAST

    def decision_do_clean_eos_type(self):
        if not self.clean_eos_queue.empty():
            self.decision_type = DecisionType.DO_CLEAN_EOS
            return True

        return False

    def calc_decision_type(self):
        self.check_10ms_for_next_decode()
        self.decision_type = DecisionType.WAIT_COMM

        if self.decision_do_clean_eos_type():
            return

        if self.decision_do_clean_up_type():
            return
        '''
        If the previous task was a decode-last and a new decode-first task arrives within 10ms, 
        execute the decode-first task.
        '''
        if not self.is_last_do_chunk_prefill:
            self.calc_prefill_priority_decision_type()
        else:
            self.calc_decode_priority_decision_type()
    
    def prepare_chunk_prefill_metadata_queue(self, curr_dp_seq_len, prefill_chunk_num, dp_empty):
        if self.prefill_chunk_metadata_queue.qsize() > 0:
            return

        average_prefill_chunk_seq_len = int(curr_dp_seq_len / prefill_chunk_num)
        mod = curr_dp_seq_len % prefill_chunk_num
        prefill_chunk_policy: list = ([0] + [average_prefill_chunk_seq_len] * prefill_chunk_num if mod == 0 else [0] + 
            [average_prefill_chunk_seq_len + 1] * mod + [average_prefill_chunk_seq_len] * (prefill_chunk_num - mod)
        )
        prefill_chunk_policy = list(itertools.accumulate(prefill_chunk_policy))
        logger.info(f"[layerwiseDisaggregated] prefill_chunk_num is {prefill_chunk_num}, curr_dp_seq_len: "
                    f"{curr_dp_seq_len} prefill_chunk_policy: {prefill_chunk_policy} "
                    f"prefill_dp_max_seq_len: {self.prefill_dp_max_seq_len} rank {self.rank}")

        # 如果当前dp域是空, 用于后面判断陪跑,
        is_dummy_batch = False  # 当前dp域执行到了末尾, 这个时候使用dummy batch进行下发
        is_last_chunk = False
        # 实现逻辑为: 将一个P切成n段, 执行顺序为: P0首 -> P1首 P0尾 P2首 P1尾 ... Pn-1首 Pn-2尾 -> Pn-1尾
        for i in range(prefill_chunk_num):
            if i >= curr_dp_seq_len and not dp_empty:
                is_dummy_batch = True

            if i == prefill_chunk_num - 1:
                is_last_chunk = True

            if i == 0:
                start_offset = 0
                end_offset = prefill_chunk_policy[i + 1]
                metadata = LwdMetadata(0, 0, False, True, is_dummy_batch, dp_empty, 0, True,
                                       start_offset, end_offset, 0, curr_dp_seq_len, is_last_chunk)
                self.prefill_chunk_metadata_queue.put(metadata)

            if i > 0 and i <= prefill_chunk_num - 1:
                start_offset = prefill_chunk_policy[i]
                end_offset = prefill_chunk_policy[i + 1]
                metadata = LwdMetadata(0, 0, False, True, is_dummy_batch, dp_empty, 0, True,
                                       start_offset, end_offset, 0, curr_dp_seq_len, is_last_chunk)
                self.prefill_chunk_metadata_queue.put(metadata)
                # 超过了真正的序列长度, 记录下来, 需要使用dummy_batch进行下发
                tmp_is_dummy_batch = True if (i - 1 >= curr_dp_seq_len and not dp_empty) else False 
                # 如果是最后一段真实的序列, 这个时候要在尾生成token
                start_offset = prefill_chunk_policy[i - 1]
                end_offset = prefill_chunk_policy[i]
                metadata = LwdMetadata(1, 1, False, True, tmp_is_dummy_batch, dp_empty, 0, True,
                                       start_offset, end_offset, 0, curr_dp_seq_len, False)
                self.prefill_chunk_metadata_queue.put(metadata)

            if i == prefill_chunk_num - 1:
                start_offset = prefill_chunk_policy[i]
                end_offset = prefill_chunk_policy[i + 1]
                # 如果已经大于了序列长度, 需要使用dummy_batch进行下发, 不需要生产token
                metadata = LwdMetadata(1, 1, True, True, is_dummy_batch, dp_empty, 0, True,
                                       start_offset, end_offset, 0, curr_dp_seq_len, True)
                self.prefill_chunk_all_last_metadata_queue.put(metadata)

    # prefill请求执行结束之后, 处理下一次调度需要的变量           
    def prefill_chunk_variable_policy(self, metadata: LwdMetadata):
        self.is_last_do_chunk_prefill = True
        start_exec_layer = metadata.start_exec_layer
        end_exec_layer = metadata.end_exec_layer
        long_seq_start_idx = metadata.long_seq_start_idx

        # 因为一开始是连续发两个P首, 所以执行完第一个P首之后, 然后要标记为P首
        if start_exec_layer == 0 and end_exec_layer == 0 and long_seq_start_idx == 0:
            self.prefill_first = True
            self.wait_prefill_last = self.prefill_last_request is not None
            self.wait_decode = self.decode_request is not None
            self.prefill_last_queue.put(self.prefill_request)
        elif self.decision_type == DecisionType.DO_PREFILL_LAST and self.is_all_last:
            # 如果数据已经取完了, 说明这个请求做完了, 就可以清除数据了
            self.wait_prefill_last = False
            self.prefill_comm_finish = False
            self.prefill_last_request = None
            self.is_all_last = False
            self.is_last_do_chunk_prefill = False
        elif self.prefill_chunk_metadata_queue.qsize() == 0:    # 倒数第二个P尾
            self.prefill_first = False
            self.prefill_comm_finish = False
            self.prefill_request = None     # 放行下一个P请求
            self.is_all_last = True
        elif start_exec_layer == 0 and end_exec_layer == 0:     # P首执行完了, 下一个就是P尾
            self.prefill_first = False
        elif start_exec_layer == 1 and end_exec_layer == 1:     # P尾执行完了, 没到结束, 下一个必然是P首
            self.prefill_first = True
            self.prefill_comm_finish = False

    def is_do_prefill_decision_type(self, decision_type):
        return decision_type == DecisionType.DO_PREFILL_FIRST or decision_type == DecisionType.DO_PREFILL_LAST

    def arrange_exec_stage(self):
        metadata = None
        if self.decision_type == DecisionType.DO_DECODE_FIRST:
            metadata = LwdMetadata(0, 0, False, False, False, False, 0, False, 0, 0, 0, 0, False)
        elif self.decision_type == DecisionType.DO_DECODE_LAST:
            metadata = LwdMetadata(1, 1, True, False, False, False, 0, False, 0, 0, 0, 0, False)
        elif self.decision_type == DecisionType.DO_PREFILL_FIRST:
            if not self.is_request_long_seq(self.prefill_request):
                metadata = LwdMetadata(0, 0, False, True, False, False, 0, False, 0, 0, 0, 0, False)
            else:
                metadata = self.prefill_chunk_metadata_queue.get(block=False)
        elif self.decision_type == DecisionType.DO_PREFILL_LAST:
            if not self.is_request_long_seq(self.prefill_last_request):
                metadata = LwdMetadata(1, 1, True, True, False, False, 0, False, 0, 0, 0, 0, False)
            else:
                metadata = self.prefill_chunk_all_last_metadata_queue.get(block=False) if self.is_all_last \
                    else self.prefill_chunk_metadata_queue.get(block=False)
        if metadata:
            lwd_metadata_manager.set_metadata(metadata)
            logger.info(f"[layerwiseDisaggregated] arrange exec stage decision_type:{self.decision_type.name} \
                metadata{metadata}, rank{self.rank}")

    def broadcast_decision_type(self):
        if self.process_func.get(self.decision_type) is None:   # 无需广播的决策
            return

        ctrl_tensor = [0] * CtrlTypePos.MAX_NUM
        ctrl_tensor[CtrlTypePos.DECISION_TYPE] = self.decision_type
        ctrl_tensor[CtrlTypePos.SHAPE_START] = -1
        ctrl_tensor[CtrlTypePos.SHAPE_END] = -1
        self.mem_manager.write_list_memory(ctrl_tensor)

    def recv_decision_type(self):
        ctrl_tensor = self.mem_manager.read_list_memory(self.rank)
        logger.info(f"[layerwiseDisaggregated] recv_decision_type ctrl_tensor {ctrl_tensor}, rank{self.rank}")
        if ctrl_tensor is None:
            self.decision_type = DecisionType.WAIT_COMM
            return

        self.decision_type = DecisionType(int(ctrl_tensor[CtrlTypePos.DECISION_TYPE]))
        while (self.decision_type == DecisionType.DO_PREFILL_FIRST and self.prefill_request is None) or \
            (self.decision_type == DecisionType.DO_DECODE_FIRST and self.decode_request is None):
            self.get_all_request()
        
        if self.decision_type == DecisionType.DO_PREFILL_FIRST:
            self.prepare_prefill_request(self.prefill_request)

        shape = ctrl_tensor[CtrlTypePos.SHAPE_START:CtrlTypePos.SHAPE_END + 1]
        if self.decision_type == DecisionType.DO_DECODE_LAST:
            self.ctrl_comm.decode_recv_msg = self.ctrl_comm.shape_to_msg(shape)
        elif self.decision_type == DecisionType.DO_PREFILL_LAST:
            self.ctrl_comm.prefill_recv_msg = self.ctrl_comm.shape_to_msg(shape)

    def set_pd_curr_request(self):
        if self.prefill_last_request is None and not self.prefill_last_queue.empty():
            self.prefill_last_request = self.prefill_last_queue.get(timeout=900)        
        
        if self.prefill_request is None and not self.prefill_queue.empty():
            self.prefill_request = self.prefill_queue.get()
            self.prefill_first = True
            if self.rank == MASTER_ID:
                self.prepare_prefill_request(self.prefill_request)

        if self.decode_request is None and not self.decode_queue.empty():
            self.decode_request = self.decode_queue.get()
            self.decode_first = True

    def curr_no_request(self):
        return self.prefill_queue.empty() and self.decode_queue.empty() and self.clean_up_queue.empty() and\
            self.prefill_last_queue.empty() and self.prefill_request is None and self.decode_request is None and \
            self.prefill_last_request is None 

    def decision_do_clean_up_type(self):
        has_clean_up = not self.clean_up_queue.empty()
        no_running_requests = self.prefill_request is None and \
            self.decode_request is None and self.prefill_last_request is None
        no_queued_requests = self.prefill_queue.empty() and \
            self.decode_queue.empty() and self.prefill_last_queue.empty()
        if has_clean_up and no_running_requests and no_queued_requests:
            self.decision_type = DecisionType.DO_CLEAN_UP
            return True

        return False
    
    def prepare_prefill_request(self, prefill_request):
        prefill_dp_seq_len = self.calc_curr_dp_seq_len(prefill_request)
        prefill_dp_max_seq_len = self.calc_max_seq_len(prefill_request)
        prefill_dp_empty = False
        if prefill_dp_seq_len == 0:    # 当前batch是空
            prefill_dp_seq_len = 1     # 长度至少为1, 构造陪跑时的长度
            prefill_dp_empty = True
        if prefill_dp_max_seq_len > self.get_long_seq_len_min():
            prefill_chunk_num = self.prefill_chunk_instance.map_prefill_chunk_num(prefill_dp_max_seq_len)
            self.prepare_chunk_prefill_metadata_queue(prefill_dp_seq_len, prefill_chunk_num, prefill_dp_empty)

    def print_do_inference_log(self):
        logger.info(f"[layerwiseDisaggregated] decision_type:{self.decision_type.name}, "
            f"has prefill_first:{self.prefill_request is not None}, "
            f"has prefill_last:{self.prefill_last_request is not None}, "
            f"prefill_comm_finish:{self.prefill_comm_finish}, "
            f"has decode:{self.decode_request is not None}, decode_comm_finish:{self.decode_comm_finish}, "
            f"clean_up_queue size:{self.clean_up_queue.qsize()}, "
            f"clean_eos_queue size:{self.clean_eos_queue.qsize()}.")
