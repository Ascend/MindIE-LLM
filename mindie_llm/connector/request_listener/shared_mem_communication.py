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

import multiprocessing.shared_memory as shm
import os
import posix_ipc
from mindie_llm.connector.request_router.request_router import RequestRouter
from mindie_llm.connector.request_router.layerwise.request_router_edge import RequestRouterEdge
from mindie_llm.connector.request_router.layerwise.request_router_cloud import RequestRouterCloud
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.status import CoreThread
from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteRequest,
    ExecuteModelResponse,
    ExecuteResponse,
    MODEL_INIT,
    ExecuteType,
)
from mindie_llm.connector.common.adaptive_garbage_collector import AdaptiveGarbageCollector
from mindie_llm.utils.prof.profiler import span_start, span_end, span_attr


def check_owner_and_permission(path: str, current_uid: int) -> None:
    """
    校验 POSIX 信号量或共享内存所有者 UID 与当前进程一致，且权限为600
    """
    stat_info = os.stat(path)
    if stat_info.st_uid != current_uid:
        raise PermissionError(
            f"'{path}' is owned by user ID {stat_info.st_uid}, "
            f"but current process is running as user ID {current_uid}. Access denied."
        )
    if (stat_info.st_mode & 0o777) != 0o600:
        raise PermissionError(
            f"'{path}' permission is {oct(stat_info.st_mode & 0o777)}, expected 0o600."
        )


class SharedMemoryChannel:
    # Default size of shared memory buffer (8 MB) - must match C++ implementation
    DEFAULT_SHARED_MEMORY_SIZE = 1024 * 1024 * 8
    # Maximum buffer size for model initialization response (0.5 MB) - must match C++ setting
    MODEL_INIT_RESP_SIZE = 1024 * 512
    RECOVER_COMMAND_RESP_SIZE = 1024 * 512
    EXECUTE_RESP_SLOT_SIZE = 1024 * 512

    def __init__(self, name_prefix: str, local_rank_id: int):
        self.name_prefix = name_prefix
        self.local_rank_id = local_rank_id
        self.shared_memory = None
        self.producer_semaphore = None
        self.consumer_semaphore = None

    def open_channel(self, role: str):
        """
        role: "request" or "response"
        """
        sem_producer_name = f"{self.name_prefix}_{role}_produce_{self.local_rank_id}"
        sem_consumer_name = f"{self.name_prefix}_{role}_consume_{self.local_rank_id}"
        current_uid = os.getuid()
        
        check_owner_and_permission(f"/dev/shm/sem.{sem_producer_name.lstrip('/')}", current_uid)
        check_owner_and_permission(f"/dev/shm/sem.{sem_consumer_name.lstrip('/')}", current_uid)
        check_owner_and_permission(f"/dev/shm/{self.name_prefix}_{role}", current_uid)

        logger.debug(f"open semaphores {sem_producer_name}, {sem_consumer_name}")
        self.producer_semaphore = posix_ipc.Semaphore(sem_producer_name, flags=0)
        self.consumer_semaphore = posix_ipc.Semaphore(sem_consumer_name, flags=0)

        logger.debug(f"Open shared memory:{self.name_prefix}_{role}")
        self.shared_memory = shm.SharedMemory(f"{self.name_prefix}_{role}")

    def receive_message(self, message_class):
        self.consumer_semaphore.acquire()
        prof_deserialize = span_start("DeserializeRequests", domain="Connector")
        mem = self.shared_memory.buf
        if not mem or len(mem) == 0:
            raise ValueError("Shared memory buffer is empty or invalid.")

        msg_length_bytes = 4  # First 4 bytes represent the message length
        msg_length = int.from_bytes(mem[:msg_length_bytes], "little")
        if msg_length <= 0:
            raise ValueError(f"Invalid message length: {msg_length}")
        proto_data = bytes(mem[msg_length_bytes: msg_length_bytes + msg_length])
        self.producer_semaphore.release()

        message = message_class()
        message.ParseFromString(proto_data)
        span_attr(prof_deserialize, "execute_type", message.execute_type)
        span_end(prof_deserialize)
        return message

    def send_message(self, message, buffer_offset: int = 0):
        prof_serialize = span_start("SerializeResponses", domain="Connector")
        proto_data = message.SerializeToString()
        msg_length_bytes = 4  # First 4 bytes represent the message length
        msg_length = len(proto_data)
        # Ensure the message length does not exceed the shared memory size
        if buffer_offset + msg_length_bytes + msg_length > SharedMemoryChannel.DEFAULT_SHARED_MEMORY_SIZE:
            raise ValueError(f"Message size {msg_length} exceeds shared memory size limit at offset {buffer_offset}")

        self.producer_semaphore.acquire()
        self.shared_memory.buf[buffer_offset: buffer_offset + msg_length_bytes] = msg_length.to_bytes(
            msg_length_bytes, "little"
        )
        self.shared_memory.buf[buffer_offset + msg_length_bytes: buffer_offset + msg_length_bytes + msg_length] = (
            proto_data
        )
        span_end(prof_serialize)
        self.consumer_semaphore.release()

    def send_binary_data(self, byte_message):
        msg_length_bytes = 4  # First 4 bytes represent the message length
        msg_length = len(byte_message)
        # Ensure the message length does not exceed the shared memory size
        if msg_length_bytes + msg_length > SharedMemoryChannel.DEFAULT_SHARED_MEMORY_SIZE:
            raise ValueError(f"Message size {msg_length} exceeds shared memory size limit")

        self.producer_semaphore.acquire()
        self.shared_memory.buf[0:msg_length_bytes] = msg_length.to_bytes(
            msg_length_bytes, "little"
        )
        self.shared_memory.buf[msg_length_bytes: msg_length_bytes + msg_length] = (
            byte_message
        )
        self.consumer_semaphore.release()


class SharedMemCommunication:
    _instance = None
    # shared_sync_link: shared by pd_link, lora_load and lora_unload, which is not thread-safe and cannot be reentrant.
    CHANNEL_NAMES = ["execute", "shared_sync_link", "transfer", "execute_error"]
    CHANNELS_WITH_REQUEST_LISTENER = ["execute", "shared_sync_link", "transfer"]

    __slots__ = [
        "config",
        "_init_request_config",
        "_initial_config_map",
        "_channels",
        "_threads",
        "_is_running",
        "request_router",
    ]

    def __init__(self, config):
        self.config = config
        self._initial_config_map = {}
        self._init_request_config = {}
        self._channels = {}
        self._threads = []
        self._is_running = False
        self.request_router = None
        # 根据边云角色创建对应的类
        layerwise_disaggregated = config.layerwise_disaggregated
        if layerwise_disaggregated == "true":
            role_type = config.layerwise_disaggregated_role_type
            logger.info(f"[layerwiseDisaggregated] role_type is {role_type}")
            if role_type == "master":
                self.request_router = RequestRouterEdge(config.parent_pid)
            elif role_type == "slave":
                self.request_router = RequestRouterCloud(config.parent_pid)
        else:
            self.request_router = RequestRouter()

        for channel_name in self.CHANNEL_NAMES:
            # Initialize shared memory channels for each communication type, with names generated by the executor
            # npu_num_per_dp is equal to tp * cp
            prefix = f"{config.shm_name_prefix}_{channel_name}"
            request_channel = SharedMemoryChannel(prefix, config.local_rank % config.npu_num_per_dp)
            response_channel = SharedMemoryChannel(prefix, config.local_rank % config.npu_num_per_dp)

            request_channel.open_channel("request")
            response_channel.open_channel("response")

            self._channels[channel_name] = {"request": request_channel, "response": response_channel}

    @classmethod
    def get_instance(cls, config):
        if cls._instance is None:
            cls._instance = SharedMemCommunication(config)
        return cls._instance

    @classmethod
    def send_model_execute_response_cls(cls, response: ExecuteModelResponse):
        cls._instance.send_response(response)

    @classmethod
    def send_model_execute_binary_data_cls(cls, binary_response):
        cls._instance.send_model_execute_binary_data(binary_response)

    @classmethod
    def send_transfer_response_cls(cls, response: ExecuteModelResponse):
        cls._instance.send_response(response, is_transfer=True)

    @classmethod
    def send_command_response_cls(cls, response: ExecuteModelResponse):
        """
        Send command request response, including lora_load, lora_unload, etc.
        """
        cls._instance.send_response(response, is_command=True)

    @classmethod
    def send_recover_command_response_cls(cls, response: ExecuteModelResponse):
        cls._instance.send_response(response, is_recover_command=True)

    def start(self):
        self._is_running = True
        for channel_name in self.CHANNELS_WITH_REQUEST_LISTENER:
            thread = CoreThread(
                target=self._process_incoming_requests, args=(channel_name,), daemon=True, name=channel_name
            )
            self._threads.append(thread)
            thread.start()
        for thread in self._threads:
            thread.join()

    def send_response(self,
        response: ExecuteResponse,
        is_transfer: bool = False,
        is_command: bool = False,
        is_recover_command: bool = False
        ):
        shared_sync_link_key = "shared_sync_link"
        response_key = "response"

        # 故障码通过execute_error通道发送
        has_err_msg = (response.HasField("execute_model_response") and
                       response.execute_model_response.err_msg != "")
        if has_err_msg:
            error_channel = self._channels["execute_error"][response_key]
            offset = (self.config.local_rank % self.config.npu_num_per_dp) * \
                SharedMemoryChannel.EXECUTE_RESP_SLOT_SIZE
            error_channel.send_message(response, buffer_offset=offset)
            return

        if response.HasField("init_results"):
            # For initialization results, each rank writes to its own offset
            execute_channel = self._channels["execute"][response_key]
            offset = (self.config.local_rank % self.config.npu_num_per_dp) * SharedMemoryChannel.MODEL_INIT_RESP_SIZE
            execute_channel.send_message(response, buffer_offset=offset)
            return

        # Recover commands reuse the shared_sync_link channel.
        if is_recover_command:
            response_channel = self._channels[shared_sync_link_key][response_key]
            offset = (self.config.local_rank % self.config.npu_num_per_dp) * \
                SharedMemoryChannel.RECOVER_COMMAND_RESP_SIZE
            response_channel.send_message(response, buffer_offset=offset)
            return

        # Determine the correct channel based on response content
        if is_transfer:
            target_channel = shared_sync_link_key if response.HasField("pd_link_response") else "transfer"
        elif is_command:
            target_channel = shared_sync_link_key
        else:
            target_channel = "execute"
        response_channel = self._channels[target_channel][response_key]

        if self.config.local_rank % self.config.npu_num_per_dp == 0:
            # Only rankInDP 0 sends the actual response payload
            response_channel.send_message(response, buffer_offset=0)
        else:
            # Other ranks only toggle semaphores for sync
            response_channel.producer_semaphore.acquire()
            response_channel.consumer_semaphore.release()

    def send_model_execute_binary_data(self, binary_response):
        response_channel = self._channels["execute"]["response"]
        if self.config.local_rank % self.config.npu_num_per_dp == 0:
            response_channel.send_binary_data(binary_response)
        else:
            response_channel.producer_semaphore.acquire()
            response_channel.consumer_semaphore.release()

    def stop(self):
        self._is_running = False
        if self.config.layerwise_disaggregated == "true":
            self.request_router.final_cleanup()

    def is_running(self) -> bool:
        return self._is_running

    def _process_incoming_requests(self, channel_name: str):
        channel = self._channels[channel_name]["request"]
        while self._is_running:
            request = channel.receive_message(ExecuteRequest)
            self._apply_config_to_request(request)
            self.request_router.accept(request)

            if request.execute_type in [
                ExecuteType.MODEL_INFER,
                ExecuteType.KV_TRANSFER,
                ExecuteType.TEXT_GENERATOR_CLEANUP,
            ]:
                AdaptiveGarbageCollector.get_instance().request_counter_increase()

    def _apply_config_to_request(self, request: ExecuteRequest):
        if request.execute_type == MODEL_INIT:
            self._init_request_config = request.config
        else:
            for key, value in self._init_request_config.items():
                request.config[key] = value

        # Add distributed training parameters to the config
        if self.config.global_rank is not None:
            request.config["rank"] = str(self.config.global_rank)
            request.config["global_rank"] = str(self.config.global_rank)
        else:
            request.config["rank"] = str(self.config.local_rank)

        if self.config.global_world_size is not None:
            request.config["world_size"] = str(self.config.global_world_size)
            request.config["global_world_size"] = str(self.config.global_world_size)
        else:
            request.config["world_size"] = str(self.config.local_world_size)

        request.config["local_rank"] = str(self.config.local_rank)
        request.config["local_world_size"] = str(self.config.local_world_size)
        request.config["npu_device_id"] = str(self.config.npu_device_id)
