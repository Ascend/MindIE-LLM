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
import threading
import traceback
import os
import time

from mindie_llm.connector.common import send_model_execute_response
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder
from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType
from mindie_llm.connector.request_router.router_impl import RouterImpl
from mindie_llm.model_wrapper.utils.config import BaseConfig, DmiConfig
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.status import CoreThread
try:
    from ms_service_profiler.profiler import prof_step
except ImportError:
    def default_prof_step(stop_check=False):
        pass
    prof_step = default_prof_step


class RequestRouter:
    __slots__ = (
        "inference_queue",
        "transfer_queue",
        "pdlink_queue",
        "command_queue",
        "query_queue",
        "inference_related_thread",
        "trans_related_thread",
        "pdlink_related_thread",
        "command_related_thread",
        "query_related_thread",
        "enable_dp_distributed",
        "router_impl",
    )

    def __init__(self):
        self.inference_queue = queue.Queue()
        self.transfer_queue = queue.Queue()
        self.pdlink_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.query_queue = queue.Queue()
 
        self.inference_related_thread = CoreThread(target=self.do_inference, name="inference")
        self.trans_related_thread = CoreThread(target=self.do_transfer, name="transfer")
        self.pdlink_related_thread = CoreThread(target=self.do_pdlink, name="link")
        self.command_related_thread = CoreThread(target=self.do_command, name="command")
        self.query_related_thread = CoreThread(target=self.do_query, name="query")
        self.inference_related_thread.start()
        self.trans_related_thread.start()
        self.pdlink_related_thread.start()
        self.command_related_thread.start()
        self.query_related_thread.start()

        self.enable_dp_distributed = None
        self.router_impl = RouterImpl()

    @staticmethod
    def get_config_dict(config):
        config_dict = dict()
        for item in config.items():
            config_dict[item[0]] = item[1]
        return config_dict

    @staticmethod
    def get_model_impl_config(model_config):
        infer_mode = model_config.get("infer_mode", "standard")
        config = None
        if infer_mode == "standard":
            # standard是pd混部场景的部署型态
            config = BaseConfig(model_config)
            logger.info("init base config")
        elif infer_mode == "dmi":
            # DMI是pd分离场景的部署型态
            config = DmiConfig(model_config)
            logger.info("init dmi config")
        else:
            logger.error(
                """[MIE04E13030A] [Model]\t>>>
                infer mode %s is not supported""",
                infer_mode,
            )
        return config

    def initialize_impl(self, config):
        self.router_impl = RouterImpl()
        initialize_result = self.router_impl.initialize(config)
        return initialize_result

    def initialize(self, request_config):
        model_config = self.get_config_dict(request_config)
        config = self.get_model_impl_config(model_config)
        self.enable_dp_distributed = config.distributed_enable
        initialize_result = self.initialize_impl(config)
        proto = ExecuteResponseBuilder.build_from_init_result(initialize_result)
        send_model_execute_response(proto)

    def do_inference(self):
        while True:
            try:
                execute_request: ExecuteRequest = self.inference_queue.get(timeout=1)
                execute_type = execute_request.execute_type
                if execute_type == ExecuteType.MODEL_INFER:
                    # 这是一个prefill请求或者decode请求
                    self.router_impl.execute(execute_request)
                elif execute_type == ExecuteType.MODEL_INFER_SECOND:
                    # 这是一个prefill请求或者decode请求
                    self.router_impl.execute(execute_request)
                elif execute_type == ExecuteType.MODEL_INIT:
                    self.initialize(execute_request.config)
                    logger.info("[python thread: infer] model initialized.")
                elif execute_type == ExecuteType.MODEL_FINALIZE:
                    self.router_impl.finalize()
                    logger.info("[python thread: infer] model finalized.")
                    break
                elif execute_type == ExecuteType.TEXT_GENERATOR_CLEANUP:
                    self.router_impl.seq_ctrl(execute_request)
                elif execute_request.execute_type == ExecuteType.RECOVER_COMMAND_EXEC:
                    self.router_impl.recover_command_exec(execute_request)
                elif execute_type == ExecuteType.START_COMMAND_EXEC:
                    self.router_impl.recover_command_exec(execute_request)
                    self.router_impl.is_inference_pause = False
                else:
                    logger.error(f"[MIE04E13030A] Unknown execute_type {execute_type}")
                prof_step()
            except queue.Empty:
                prof_step(stop_check=True)
                continue

    def do_pdlink(self):
        while True:
            execute_request: ExecuteRequest = self.pdlink_queue.get()
            if execute_request.execute_type == ExecuteType.PD_LINK:
                self.router_impl.pd_role(execute_request)
            elif execute_request.execute_type == ExecuteType.MODEL_FINALIZE:
                break
            else:
                logger.error(
                    f"[MIE04E13030A] Unknown link type {execute_request.execute_type}, "
                    f"Expected PD_LINK type is {ExecuteType.PD_LINK}."
                )
            
    def do_transfer(self):
        while True:
            execute_request: ExecuteRequest = self.transfer_queue.get()
            execute_type = execute_request.execute_type
            if execute_type == ExecuteType.KV_TRANSFER:
                self.router_impl.transfer_data(execute_request)
            elif execute_type == ExecuteType.CLEAR_COMMAND_EXEC:
                self.router_impl.recover_command_exec(execute_request)
            elif execute_type == ExecuteType.MODEL_FINALIZE:
                break
            else:
                logger.error(f"[MIE04E13030A] Unknown transfer type {execute_type}, \
                    Expected transfer_data type is {ExecuteType.KV_TRANSFER}")
            
    def do_command(self):
        """
        This func will be used for command related tasks, including LoRA_load/unload, etc.
        If others exec_type is added later, please add validation branch here.
        """
        while True:
            execute_request: ExecuteRequest = self.command_queue.get()
            execute_type = execute_request.execute_type
            if execute_type == ExecuteType.LORA_OPERATION:
                self.router_impl.process_lora_operation(execute_request)
            elif execute_type == ExecuteType.PAUSE_COMMAND_EXEC or execute_type == ExecuteType.PAUSE_COMMAND_EXEC_ROCE:
                self.router_impl.is_inference_pause = True
                self.router_impl.recover_command_exec(execute_request)
            elif execute_type == ExecuteType.MODEL_FINALIZE:
                break
            else:
                logger.error(f"[MIE04E13030A] Unknown command type {execute_type}, \
                    Expected command type is {ExecuteType.LORA_OPERATION}")

    # 新增处理链路状态查询请求        
    def do_query(self):
        while True:
            execute_request: ExecuteRequest = self.query_queue.get()
            execute_type = execute_request.execute_type
            if execute_type == ExecuteType.PD_LINK_STATUS_QUERY:
                self.router_impl.query_link_status(execute_request)
            elif execute_type == ExecuteType.MODEL_FINALIZE:
                break
            else:
                logger.error(
                    f"[MIE04E13030A] Unknown query type {execute_request.execute_type}, "
                    f"Expected type is {ExecuteType.PD_LINK_STATUS_QUERY}."
                )

    def accept(self, execute_request: ExecuteRequest):
        if execute_request.execute_type == ExecuteType.MODEL_INFER or \
            execute_request.execute_type == ExecuteType.START_COMMAND_EXEC or \
            execute_request.execute_type == ExecuteType.RECOVER_COMMAND_EXEC:
            self.inference_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.PD_LINK:
            self.pdlink_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.KV_TRANSFER or \
            execute_request.execute_type == ExecuteType.CLEAR_COMMAND_EXEC:
            self.transfer_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.LORA_OPERATION or \
            execute_request.execute_type == ExecuteType.PAUSE_COMMAND_EXEC or \
            execute_request.execute_type == ExecuteType.PAUSE_COMMAND_EXEC_ROCE:
            self.command_queue.put(execute_request)
        # 新增：查询请求分发到 query_queue
        elif execute_request.execute_type == ExecuteType.PD_LINK_STATUS_QUERY:
            self.query_queue.put(execute_request)
        elif execute_request.execute_type == ExecuteType.MODEL_FINALIZE:
            self.inference_queue.put(execute_request)
            self.pdlink_queue.put(execute_request)
            self.transfer_queue.put(execute_request)
            self.command_queue.put(execute_request)
            self.query_queue.put(execute_request)
        else:
            self.inference_queue.put(execute_request)
