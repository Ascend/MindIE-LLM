#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import json
from threading import Lock, Condition
from typing import Union
from llm_manager_python_api_demo.dtype import get_infer_datatype_by_dtype
from llm_manager_python_api_demo.data import infer_tensor_to_data
from llm_manager_python_api_demo.status import Status, Code
from llm_manager_python_api_demo.request import Request
from llm_manager_python_api_demo.response import Response
from llm_manager_python_api_demo import llm_manager_python
from mindie_llm.utils.file_utils import safe_open

VOCAB_SIZE_DEFAULT = 1024


class Engine:
    def __init__(self):
        """
        创建一个infer_engine对象
        """
        self.response_callback = None
        self.llm_manager = None

        self.forward_mutex = Lock()
        self.stop_mutex = Lock()
        self.stats_mutex = Lock()

        self.request_queue = []
        self.stop_id_queue = []

        self.callback_map = {}

        self.forward_cv_map = {}
        self.forward_status_map = {}

        self.cv_map = {}
        self.ctrl_status_map = {}
        
        self.remain_blocks = 0
        self.remain_prefill_slots = 0
        self.remain_prefill_tokens = 0
        self.processing_request_num = 0

        self.slaves_status = {}

        self.max_position_embeddings = 0
        self.mutex_start = Lock()
        self.started = False
    
    @staticmethod
    def get_config_path():
        config_path = os.getenv('MIES_CONFIG_JSON_PATH')
        if config_path:
            return config_path

        home_path = os.getenv('MINDIE_LLM_HOME_PATH')
        if not home_path:
            home_path = os.getenv('MINDIE_LLM_HOME_PATH')
        if home_path:
            return os.path.join(home_path, "conf/config.json")
        else:
            raise RuntimeError("no config path found")
    
    @staticmethod
    def get_scheduler_config(config_path):
        with safe_open(config_path, 'r') as file:
            config = json.load(file)
            return config['BackendConfig']['ScheduleConfig']

    @staticmethod
    def convert_request_id(req_id: llm_manager_python.InferRequestId):
        if req_id.type() == llm_manager_python.DataType.STRING:
            return req_id.string_value()
        else:
            return req_id.unsigned_int_value()

    @staticmethod
    def construct_response_by_tensor_map(llm_req_id: Union[str, int],
                                         output: llm_manager_python.TensorMap,
                                         is_final: bool, err_msg: str) -> Response:
        response = Response(llm_req_id)

        for _, tensor in output.items():
            response_data = infer_tensor_to_data(tensor)
    
            ret = response.add_output(response_data)
            if not ret.is_ok():
                continue

        response.set_flags(int(err_msg))
        response.set_eos(is_final)

        return response

    @staticmethod
    def valid_request_input_ids(request, vocab_size):
        try:
            inputs = request.get_immutable_inputs()
        except Exception:
            return Status(Code.ERROR, "Invalid request: missing inputs")
        if not inputs:
            return Status(Code.ERROR, "Missing inputs in request")
        return Status(Code.OK)
 
    def init(self, config_path=None, response_callback=None, load_all_data=False, data_len=0) -> Status:
        """
        初始化engine对象
        :return: status
        """
        if config_path is None:
            config_path = self.get_config_path()
        
        if response_callback is not None:
            self.response_callback = response_callback
        
        def get_requests_callback_inner():
            requests: list[llm_manager_python.InferRequest] = []
            with self.forward_mutex:
                while (len(self.request_queue) != 0):
                    requests.append(self.request_queue.pop(0))
            return requests

        def get_requests_callback():
            if load_all_data:
                if len(self.request_queue) == data_len:
                    return get_requests_callback_inner()  
            else:
                return get_requests_callback_inner()
            return []

        def send_responses_callback(req_id: llm_manager_python.InferRequestId, output: llm_manager_python.TensorMap,
                                    is_final: bool, err_msg: str):
            with self.forward_mutex:
                if req_id.string_value() in self.callback_map:
                    self.callback_map[req_id.string_value()](req_id, output, is_final, err_msg)
                    if is_final:
                        del self.callback_map[req_id.string_value()]
                    return
            if self.response_callback is not None:
                self.response_callback(req_id, output, is_final, err_msg)
        
        def stop_signal_callback():
            stop_list = []
            with self.stop_mutex:
                while (len(self.stop_id_queue) != 0):
                    stop_list.append(self.stop_id_queue.pop(0))
            return stop_list

        def stats_callback(status: str):
            received_json = json.loads(status)
            with self.stats_mutex:
                self.slaves_status = received_json.get("slaves_status")
                self.remain_blocks = received_json.get("remain_blocks")
                self.remain_prefill_slots = received_json.get("free_npu_block_num")
                self.remain_prefill_tokens = received_json.get("free_npu_block_num")
                self.processing_request_num = received_json.get("processing_request_num")

        def send_status_callback(req_id: llm_manager_python.InferRequestId,
                                 status: llm_manager_python.Status, status_response_type: str):
            req_id_str = req_id.string_value()
            if status_response_type == llm_manager_python.StatusResponseType.CONTROL_SIGNAL_STATUS:
                with self.stop_mutex:
                    if req_id_str in self.cv_map:
                        self.ctrl_status_map[req_id_str] = status
                        self.cv_map[req_id_str].notify()
            elif status_response_type == llm_manager_python.StatusResponseType.REQUEST_ENQUEUE_STATUS:
                with self.forward_mutex:
                    if req_id_str in self.forward_cv_map:
                        cv = self.forward_cv_map[req_id_str]
                        with cv:
                            self.forward_status_map[req_id_str] = status
                            cv.notify()
            else:
                raise RuntimeError("SendStatusResponseCallback type invalid!")
        
        self.llm_manager = llm_manager_python.LlmManager(config_path, get_requests_callback, send_responses_callback,
                                             stop_signal_callback, stats_callback, send_status_callback)
        with safe_open(config_path, 'r') as f:
            config_data = json.load(f)
            model_instance_id = 0
            npu_device_id = set(config_data["BackendConfig"]["npuDeviceIds"][model_instance_id])
            status = self.llm_manager.init(model_instance_id, npu_device_id)
            if not status.is_ok():
                return Status(Code.ERROR, status.status_msg())
        return Status(Code.OK, "Success")
    
    def add_request_to_queue(self, runtime_req_id, send_response_callback, runtime_request) -> Status:
        runtime_req_id_str = runtime_req_id.string_value()
        with self.forward_mutex:
            if runtime_req_id_str in self.callback_map:
                return Status(Code.ERROR, "Request id has been used before!")

            self.callback_map[runtime_req_id_str] = send_response_callback
            self.request_queue.append(runtime_request)

        cv = Condition()
        self.forward_cv_map[runtime_req_id_str] = cv
        with cv:
            while self.forward_status_map.get(runtime_req_id_str, None) is None:
                cv.wait()
        return self.forward_status_map.get(runtime_req_id_str)

    def forward(self, llm_infer_request: Request, valid_input=True):
        if self.llm_manager is None:
            return Status(Code.ERROR, "RuntimeEngine not init!")

        if valid_input:
            result = self.valid_request_input_ids(llm_infer_request, VOCAB_SIZE_DEFAULT)
            if not result.is_ok():
                return result

        llm_req_id = llm_infer_request.get_request_id()
        runtime_req_id = llm_manager_python.InferRequestId(llm_req_id.id)

        runtime_request = llm_manager_python.InferRequest(runtime_req_id)
        rets = runtime_request.set_max_output_len(llm_infer_request.get_max_output_len())
        if not rets.is_ok():
            return Status(Code.ERROR, "Set maxOutputLen for runtimeRequest error")

        input_tensors = llm_infer_request.get_immutable_inputs()
        for input_tensor_name, input_tensor in input_tensors.items():
            runtime_tensor = llm_manager_python.InferTensor(
                input_tensor_name,
                get_infer_datatype_by_dtype(input_tensor.get_type()),
                input_tensor.get_shape()
            )
            runtime_tensor.set_buffer(
                input_tensor.get_data(),
                False
            )
            runtime_request.add_tensor(
                input_tensor_name,
                runtime_tensor
            )

        if llm_infer_request.get_send_response_callback() is None:
            llm_infer_request.set_send_response_callback(self.response_callback)

        send_response_callback = llm_infer_request.get_send_response_callback()
        res = self.add_request_to_queue(runtime_req_id, send_response_callback, runtime_request)
        return res
        

    def finalize(self) -> Status:
        """
        析构engine对象
        :return: status
        """
        self.llm_manager.shutdown()
        return Status(Code.OK, "Success")

    def async_forward(self, request: Request) -> Status:
        """
        执行请求推理
        :param request: 需要推理的请求
        :return: status
        """
        ret = self.forward(request)
        return ret

    def get_request_block_quotas(self):
        """
        返回 remainBlocks, remainPrefillSlots, remainPrefillTokens
        :return:
        """
        if self.llm_manager is None:
            return Status(Code.ERROR, "LLMInferEngine is not initialized!")

        with self.stats_mutex:
            return self.remain_blocks, self.remain_prefill_slots, self.remain_prefill_tokens

    def get_processing_request(self) -> int:
        """
        返回 processing num
        :return:
        """
        if self.llm_manager is None:
            return Status(Code.ERROR, "LLMInferEngine is not initialized!")

        with self.stats_mutex:
            return self.processing_request_num
