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

import json
import queue
import sys
import time
from enum import IntEnum
from pathlib import Path
from itertools import chain
import numpy as np

from mindie_llm.connector.common import send_model_execute_response
from mindie_llm.connector.common.input_metadata_builder import parse_all_dp_batches_seq_lens
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder
from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType
from mindie_llm.connector.request_router.request_router import RequestRouter
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager
from mindie_llm.utils.layerwise.share_memory import SharedMemoryManager
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.prof.profiler import span_start, span_end

sys.path.append(str(Path(__file__).parent / "sync"))


MASTER_ID = 0   # 接收通信, 广播决策, 创建共享内存的主rank号
LONG_SEQ_LEN_MIN = 7500
MULTI_NODES_LONG_SEQ_LEN_MIN = 150000
MULTI_NODES_LONG_SEQ_LEN_MIN_INT4 = 15000


class LastExecType(IntEnum):
    PREFILL = 0
    DECODE = 1


class DecisionType(IntEnum):
    DO_NOTHING = 0
    DO_PREFILL_FIRST = 1
    DO_PREFILL = 2
    DO_PREFILL_LAST = 3
    DO_DECODE_FIRST = 4
    DO_DECODE = 5
    DO_DECODE_LAST = 6
    DO_CLEAN_UP = 7
    DO_CLEAN_EOS = 8
    WAIT_COMM = 9
    WAIT_DECODE = 10


class RequestRouterLwd(RequestRouter):
    def __init__(self, parent_pid):
        self.rank = None
        self.parent_pid = parent_pid

        self.prefill_queue = queue.Queue()
        self.decode_queue = queue.Queue()
        self.clean_up_queue = queue.Queue() 
        self.clean_eos_queue = queue.Queue()

        self.decision_type = DecisionType.DO_NOTHING

        self.prefill_request = None
        self.decode_request = None

        self.ctrl_comm = None
        self.data_comm = None

        self.mem_manager = None

        self.decode_comm_finish = False
        self.prefill_comm_finish = False
        self.prefill_comm_tcp_finish_count = 0

        self.prefill_chunk_instance = None

        # 多机新增, 多dp的变量可适配单dp
        self.lwd_multi_nodes_enable = False
        self.prefill_dp_seq_len = 0             # 自身dp域的长度, 如果dp=1就直接得到这个dp的总长度; 如果dp域是空, 则赋值为1
        self.prefill_dp_max_seq_len = 0         # 多dp中的最大长度, 如果只有一个dp就是本身的长度
        self.prefill_dp_empty = False           # dp域中本身是否为empty
        self.cp_size = 1
        self.moe_quantize = None

        super().__init__()

    @staticmethod
    def get_lwd_models_config_dict(model_config):
        model_config_dict_json = model_config.get("models")
        models_config_dict = None
        if isinstance(model_config_dict_json, str):
            try:
                models_config_dict = json.loads(model_config_dict_json)
            except json.JSONDecodeError as e:
                message = "The 'models' field does not conform to JSON format. Please check."
                logger.warning(f'{message}, exception info: {e}')
                models_config_dict = None

        if models_config_dict is None:
            logger.info("[layerwiseDisaggregated] request router lwd initliaze error.")
            raise ValueError("request router init error, layerwiseDisaggregated models is None")

        return models_config_dict

    def initialize_diff(self, model_config, models_config_dict):
        pass

    def process_sync_npu_smi_info(self, edge_cloud_comm):
        if not edge_cloud_comm.need_sync_npu_smi_info():
            return

        if self.rank == MASTER_ID:
            self.mem_manager.write_dict_memory(edge_cloud_comm.get_peer_npu_smi_info())
        else:
            while True:
                npu_smi_info = self.mem_manager.read_dict_memory(self.rank)
                if npu_smi_info is not None:
                    edge_cloud_comm.set_peer_npu_smi_info(npu_smi_info)
                    logger.info(f"[layerwiseDisaggregated] rank {self.rank} "
                                f"edge_npu_smi_info {self.ctrl_comm.edge_npu_smi_info} "
                                f"cloud_npu_smi_info {self.ctrl_comm.cloud_npu_smi_info}")
                    break
                else:
                    time.sleep(0.5)

    def initialize(self, request_config):
        model_config = self.get_config_dict(request_config)
        config = self.get_model_impl_config(model_config)
        self.enable_dp_distributed = config.distributed_enable

        edge_cloud_comm = LwdCommunicationManager()
        if edge_cloud_comm.communication_config_verify(config):
            initialize_result = self.initialize_impl(config)
            self.cp_size = config.cp_size
            self.sp_size = config.sp_size
            edge_cloud_comm.initialize(config, initialize_result, self.router_impl.generator)
            proto = ExecuteResponseBuilder.build_from_init_result(initialize_result)
            send_model_execute_response(proto)
        else:
            initialize_result = {
                "status": "error"
            }
            proto = ExecuteResponseBuilder.build_from_init_result(initialize_result)
            send_model_execute_response(proto)
            return

        self.rank = config.local_rank
        self.ctrl_comm = edge_cloud_comm.ctrl_comm
        self.data_comm = edge_cloud_comm.data_comm

        self.mem_manager = SharedMemoryManager(self.parent_pid)
        logger.info(f"[layerwiseDisaggregated] initliaze share mem ok rank:{self.rank}, parent_pid:{self.parent_pid}")

        models_config_dict = self.get_lwd_models_config_dict(model_config)
        npu_device_ids = config.model_config.get('npu_device_ids', '0,1').split(',')
        self.lwd_multi_nodes_enable = True if model_config.get('lwd_multi_nodes_enable', 'false') == 'true' else False

        is_producer = True if self.rank == MASTER_ID else False
        card_num = len(npu_device_ids)
        self.mem_manager.initialize(is_producer, card_num - 1)
        self.process_sync_npu_smi_info(edge_cloud_comm)

        model_runner_config = self.router_impl.generator.model_wrapper.model_runner.config
        self.moe_quantize = getattr(model_runner_config, 'moe_quantize', None)
        self.prefill_chunk_instance = self.router_impl.generator.model_wrapper.model_runner.chunk_prefill_manager
        self.prefill_chunk_instance.initialize(self.lwd_multi_nodes_enable)

        #standard card
        edge_is_standard_card = self.ctrl_comm.edge_npu_smi_info.get("communication_backend") == 'hccl' if \
            self.ctrl_comm is not None and self.ctrl_comm.edge_npu_smi_info is not None else False
        if edge_is_standard_card:
            self.prefill_chunk_instance.initialize_standard_card()

        logger.info(f"[layerwiseDisaggregated] mem_manager initliaze ok rank:{self.rank}, is_producer:{is_producer}, "
            f"card_num:{card_num} lwd_multi_nodes_enable:{self.lwd_multi_nodes_enable}")

        self.initialize_diff(model_config, models_config_dict)

    def final_cleanup(self):
        if self.router_impl is not None and self.router_impl.generator is not None:
            self.router_impl.generator.model_wrapper.model_runner.data_comm.final_cleanup()

    def get_long_seq_len_min(self):
        if self.lwd_multi_nodes_enable:
            if self.moe_quantize == 'w4a8_dynamic':
                return MULTI_NODES_LONG_SEQ_LEN_MIN_INT4
            return MULTI_NODES_LONG_SEQ_LEN_MIN

        return LONG_SEQ_LEN_MIN

    def curr_no_request(self):
        return self.prefill_queue.empty() and self.decode_queue.empty() and self.clean_up_queue.empty() and\
            self.prefill_request is None and self.decode_request is None

    def is_pd_inference_request(self, execute_type):
        return execute_type == ExecuteType.MODEL_INFER or execute_type == ExecuteType.MODEL_INFER_SECOND

    def save_pd_inference_request(self, execute_request: ExecuteRequest):
        forward_type = execute_request.execute_model_request.forward_type
        if forward_type == ForwardType.PREFILL:
            self.prefill_queue.put(execute_request)
        elif forward_type == ForwardType.DECODE:
            self.decode_queue.put(execute_request)
        else:
            self.router_impl.execute(execute_request)
        logger.info(f"[layerwiseDisaggregated] save pd request forward_type: {forward_type} rank: {self.rank}.")

    def do_other_request_now(self, execute_request: ExecuteRequest):
        execute_type = execute_request.execute_type
        if execute_type == ExecuteType.MODEL_INIT:
            self.initialize(execute_request.config)
            logger.info("[layerwiseDisaggregated][python thread: infer] model initialized.")
        elif execute_type == ExecuteType.MODEL_FINALIZE:
            self.router_impl.finalize()
            logger.info("[layerwiseDisaggregated][python thread: infer] model finalized.")
        else:
            logger.error(f"[layerwiseDisaggregated] Unknown execute_type {execute_type} now")
            raise RuntimeError("python doing an unknown execute inference type now")

    def save_inference_request(self, execute_request: ExecuteRequest):
        execute_type = execute_request.execute_type
        if self.is_pd_inference_request(execute_type):
            self.save_pd_inference_request(execute_request)
            return
        elif execute_type == ExecuteType.TEXT_GENERATOR_CLEANUP:
            self.clean_up_queue.put(execute_request)
        elif execute_type == ExecuteType.EOS_CLEANUP:
            self.clean_eos_queue.put(execute_request)
        else:
            self.do_other_request_now(execute_request)
        logger.info(f"[layerwiseDisaggregated] save other request execute_type: {execute_type} rank: {self.rank}.")

    def set_pd_curr_request(self):
        if self.prefill_request is None and not self.prefill_queue.empty():
            self.prefill_request = self.prefill_queue.get()
            self.prefill_dp_seq_len = self.calc_curr_dp_seq_len(self.prefill_request)

        if self.decode_request is None and not self.decode_queue.empty():
            self.decode_request = self.decode_queue.get()

    def get_all_request(self):
        while not self.inference_queue.empty() or self.curr_no_request():
            if not self.clean_up_queue.empty():
                logger.info(f"[layerwiseDisaggregated] curr has clean up request, wait to clean up before "
                    f"get next request, rank{self.rank}.")
                break

            if self.inference_queue.empty():
                logger.info(f"[layerwiseDisaggregated] inference_queue is empty: {self.inference_queue.empty()}, "
                    f"rank:{self.rank}.")
            execute_request: ExecuteRequest = self.inference_queue.get()
            self.save_inference_request(execute_request)
        self.set_pd_curr_request()

    def recv_prefill(self):
        if self.prefill_comm_finish:	 
            return

        if self.prefill_comm_tcp_finish_count == 0: 
            self.ctrl_comm.recv_prefill() 
            self.prefill_comm_tcp_finish_count = self.ctrl_comm.prefill_comm_finish_tcp_count 
          
        if self.prefill_comm_tcp_finish_count > 0: 
            self.prefill_comm_finish = True 
            self.prefill_comm_tcp_finish_count -= 1 
            self.ctrl_comm.prefill_comm_finish_tcp_count -= 1
            logger.info("[layerwiseDisaggregated] recv_prefill tcp comm finish.") 

    def recv_decode(self):
        self.ctrl_comm.recv_decode()
        self.decode_comm_finish = True

    def calc_decision_type(self):
        pass

    def arrange_exec_stage(self):
        metadata = LwdMetadata(0, 0, True, True, False, False, 62, False, 0, 0, 0, 0, False)
        lwd_metadata_manager.set_metadata(metadata)

    def decision_do_clean_up_type(self):
        has_clean_up = not self.clean_up_queue.empty()
        no_running_requests = self.prefill_request is None and self.decode_request is None
        no_queued_requests = self.prefill_queue.empty() and self.decode_queue.empty()
        if has_clean_up and no_running_requests and no_queued_requests:
            self.decision_type = DecisionType.DO_CLEAN_UP
            return True

        return False

    def broadcast_decision_type(self):
        if self.decision_type == DecisionType.DO_NOTHING:
            return

        self.mem_manager.write_list_memory([self.decision_type])

    def recv_decision_type(self):
        pass

    def exceute_inference_request(self):
        func = self.process_func.get(self.decision_type)
        if func:
            func()
        else:
            time.sleep(0.001)

    def recv_ctrl_msg(self):
        self.recv_prefill() # 接收对方发来的prefill tcp控制信号
        self.recv_decode()  # 接收对方发来的decode tcp控制信号

    def master_rank_make_decision(self):
        # Determine the decision type for executing P/D based on the current information.
        self.calc_decision_type()

    def print_do_inference_log(self):
        logger.info(f"[layerwiseDisaggregated] decision_type:{self.decision_type.name}, "
            f"has prefill:{self.prefill_request is not None}, prefill_comm_finish:{self.prefill_comm_finish}, "
            f"has decode:{self.decode_request is not None}, decode_comm_finish:{self.decode_comm_finish}, "
            f"clean_up_queue size:{self.clean_up_queue.qsize()}, "
            f"clean_eos_queue size:{self.clean_eos_queue.qsize()}.")

    def do_inference(self):
        while True:
            prof = span_start("get_request")
            self.get_all_request() # 将inference_queue里的所有请求, 存到新增的三个队列中, 用于调度
            span_end(prof)

            self.recv_ctrl_msg()   # 接收对方发来的tcp控制信号

            if self.rank == MASTER_ID:
                self.master_rank_make_decision()
                self.print_do_inference_log()
                prof = span_start("broadcast_decision")
                self.broadcast_decision_type()
                span_end(prof)
                self.arrange_exec_stage()
            else:
                prof = span_start("recv_decision")
                self.recv_decision_type()
                span_end(prof)
                self.arrange_exec_stage()

            self.exceute_inference_request()

    def do_clean_up(self):
        while self.clean_up_queue.empty():
            self.get_all_request()

        execute_request: ExecuteRequest = self.clean_up_queue.get()
        self.router_impl.seq_ctrl(execute_request)
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean up, rank{self.rank}.")

    def do_clean_eos(self):
        while self.clean_eos_queue.empty():
            self.get_all_request()

        execute_request: ExecuteRequest = self.clean_eos_queue.get()
        self.router_impl.seq_ctrl(execute_request)
        logger.info(f"[layerwiseDisaggregated][python thread: infer] text generator clean eos, rank{self.rank}.")

    def sum_nested(self, lst):
        total = 0
        for item in lst:
            if isinstance(item, list):
                total += self.sum_nested(item)
            else:
                total += item
        return total

    def calc_batch_size(self, execute_request: ExecuteRequest):
        return len(execute_request.execute_model_request.seq_group_metadata_list)

    def calc_curr_dp_seq_len(self, execute_request: ExecuteRequest):
        curr_dp_rank = self.router_impl.generator.plugin_manager.model_wrapper.mapping.attn_dp.rank
        all_batch_dp_rank_ids = []
        for seq_group_metadata in execute_request.execute_model_request.seq_group_metadata_list:
            all_batch_dp_rank_ids.append(seq_group_metadata.dp_rank_id)
        all_batch_dp_rank_ids = list(dict.fromkeys(all_batch_dp_rank_ids))
        pos = -1
        if curr_dp_rank in all_batch_dp_rank_ids:
            pos = all_batch_dp_rank_ids.index(curr_dp_rank)
        if pos == -1:
            return 0
        else:
            seq_lens = parse_all_dp_batches_seq_lens(execute_request.execute_model_request.all_dp_batches_seq_lens)[pos]
            return self.sum_nested(seq_lens)

    def calc_curr_dp_batch_size(self, execute_request: ExecuteRequest):
        curr_dp_rank = self.router_impl.generator.plugin_manager.model_wrapper.mapping.attn_dp.rank
        all_batch_dp_rank_ids = []
        for seq_group_metadata in execute_request.execute_model_request.seq_group_metadata_list:
            all_batch_dp_rank_ids.append(seq_group_metadata.dp_rank_id)
        if curr_dp_rank in all_batch_dp_rank_ids:
            return all_batch_dp_rank_ids.count(curr_dp_rank)
        else:
            return 1
    
    def max_nested(self, lst):
        max_val = 0
        for item in lst:
            if isinstance(item, list):
                item_max = self.max_nested(item)
                max_val = max(max_val, item_max)
            else:
                max_val = max(max_val, item)
        return max_val

    def calc_max_seq_len(self, execute_request: ExecuteRequest):
        seq_lens = parse_all_dp_batches_seq_lens(execute_request.execute_model_request.all_dp_batches_seq_lens)
        return self.max_nested(seq_lens)

    def is_request_long_seq(self, execute_request: ExecuteRequest):
        return self.calc_max_seq_len(execute_request) > self.get_long_seq_len_min()

    def calc_cp_seq_len(self, execute_request: ExecuteRequest):
        seq_lens = parse_all_dp_batches_seq_lens(execute_request.execute_model_request.all_dp_batches_seq_lens)
        seq_lens_list = list(chain.from_iterable(seq_lens))
        pad_token_count = (-np.array(seq_lens_list)) % (2 * self.cp_size)
        seq_lens_list += pad_token_count
        total_seq_num = seq_lens_list.sum()
        return total_seq_num // self.cp_size
