# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import array
import unittest
from unittest.mock import patch, MagicMock, call
import multiprocessing.shared_memory as shm
import posix_ipc
import numpy as np

from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteRequest,
    ExecuteResponse,
    ExecuteModelResponse,
    MODEL_INIT,
    ExecuteType,
    SequenceGroupMetadata,
)
from mindie_llm.connector.request_listener.shared_mem_communication import (
    SharedMemoryChannel,
    SharedMemCommunication,
    check_owner_and_permission,
)
from mindie_llm.connector.request_router.request_router import RequestRouter
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder


class TestCheckOwnerAndPermission(unittest.TestCase):
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.os.stat")
    def test_check_owner_uid_mismatch(self, mock_stat):
        mock_stat.return_value = MagicMock(st_uid=999, st_mode=0o600)
        with self.assertRaises(PermissionError) as ctx:
            check_owner_and_permission("/dev/shm/sem.test", current_uid=1000)
        self.assertIn("owned by user ID 999", str(ctx.exception))
        self.assertIn("current process is running as user ID 1000", str(ctx.exception))

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.os.stat")
    def test_check_owner_wrong_permission(self, mock_stat):
        mock_stat.return_value = MagicMock(st_uid=1000, st_mode=0o644)
        with self.assertRaises(PermissionError) as ctx:
            check_owner_and_permission("/dev/shm/sem.test", current_uid=1000)
        self.assertIn("permission is", str(ctx.exception))
        self.assertIn("expected 0o600", str(ctx.exception))

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.os.stat")
    def test_check_owner_ok(self, mock_stat):
        mock_stat.return_value = MagicMock(st_uid=1000, st_mode=0o600)
        check_owner_and_permission("/dev/shm/sem.test", current_uid=1000)  # 不应抛出


def create_request():
    execute_request = ExecuteRequest()
    metadata = SequenceGroupMetadata()
    s64_array = array.array("q", [1, 3, 4])
    metadata.block_tables.append(s64_array.tobytes())
    execute_request.execute_model_request.seq_group_metadata_list.append(metadata)
    return execute_request


def create_sem(name, v):
    # 创建信号量
    sem_p = posix_ipc.Semaphore(
        name=name,  # 信号量名称（必须以/开头）
        flags=posix_ipc.O_CREAT,  # 不存在则创建
        mode=0o600,  # 权限位（用户读写）
        initial_value=0,  # 初始计数值
    )
    if sem_p.value == 0 and v == 1:
        sem_p.release()
    if sem_p.value == 1 and v == 0:
        sem_p.acquire()
    return sem_p


class TestSharedMemoryChannel(unittest.TestCase):
    def setUp(self):
        self.name_prefix = "test_prefix"
        self.local_rank = 0
        self.channel = SharedMemoryChannel(self.name_prefix, self.local_rank)

    def test_init(self):
        self.assertEqual(self.channel.name_prefix, self.name_prefix)
        self.assertEqual(self.channel.local_rank_id, self.local_rank)
        self.assertIsNone(self.channel.shared_memory)
        self.assertIsNone(self.channel.producer_semaphore)
        self.assertIsNone(self.channel.consumer_semaphore)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_open_channel(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_check_owner_and_permission.return_value = None
        self.channel.open_channel("request")
        sem_prod_name = f"{self.name_prefix}_request_produce_{self.local_rank}"
        sem_cons_name = f"{self.name_prefix}_request_consume_{self.local_rank}"
        mock_sem.assert_has_calls(
            [call(sem_prod_name, flags=0), call(sem_cons_name, flags=0)],
            any_order=False,
        )
        shm_name = f"{self.name_prefix}_request"
        mock_shm.assert_called_once_with(shm_name)
        self.assertIsNotNone(self.channel.shared_memory)

        mock_sem.reset_mock()
        mock_shm.reset_mock()
        self.channel.open_channel("response")
        sem_prod_name = f"{self.name_prefix}_response_produce_{self.local_rank}"
        sem_cons_name = f"{self.name_prefix}_response_consume_{self.local_rank}"
        mock_sem.assert_has_calls(
            [call(sem_prod_name, flags=0), call(sem_cons_name, flags=0)]
        )
        shm_name = f"{self.name_prefix}_response"
        mock_shm.assert_called_once_with(shm_name)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.span_end")
    def test_receive_message_empty_buffer(self, mock_span_end):
        mock_sem = MagicMock()
        mock_shm = MagicMock()
        mock_shm.buf = bytearray(0)

        self.channel.consumer_semaphore = mock_sem
        self.channel.shared_memory = mock_shm

        with self.assertRaises(ValueError) as ctx:
            self.channel.receive_message(ExecuteRequest)
        self.assertEqual(
            str(ctx.exception), "Shared memory buffer is empty or invalid."
        )
        mock_sem.acquire.assert_called_once()
        mock_span_end.assert_not_called()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.span_end")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.span_start")
    def test_send_message(self, mock_span_start, mock_span_end):
        mock_msg = MagicMock()
        mock_msg.SerializeToString.return_value = b"test_response"
        mock_sem = MagicMock()
        mock_shm_buf = bytearray(1024)
        mock_shm = MagicMock()
        mock_shm.buf = mock_shm_buf

        self.channel.producer_semaphore = mock_sem
        self.channel.consumer_semaphore = mock_sem
        self.channel.shared_memory = mock_shm

        buffer_offset = 100
        self.channel.send_message(mock_msg, buffer_offset)

        mock_sem.acquire.assert_called_once()
        mock_sem.release.assert_called_once()
        msg_len = len(b"test_response")
        written_len = int.from_bytes(
            mock_shm_buf[buffer_offset: buffer_offset + 4], "little"
        )
        self.assertEqual(written_len, msg_len)
        written_data = mock_shm_buf[buffer_offset + 4: buffer_offset + 4 + msg_len]
        self.assertEqual(written_data, b"test_response")

        mock_span_start.assert_called_once_with(
            "SerializeResponses", domain="Connector"
        )
        mock_span_end.assert_called_once()

    def test_send_message_exceeds_size(self):
        mock_msg = MagicMock()
        mock_msg.SerializeToString.return_value = b"x" * (
            SharedMemoryChannel.DEFAULT_SHARED_MEMORY_SIZE
        )
        mock_sem = MagicMock()
        self.channel.producer_semaphore = mock_sem
        self.channel.shared_memory = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            self.channel.send_message(mock_msg, buffer_offset=0)
        self.assertIn("exceeds shared memory size limit", str(ctx.exception))
        mock_sem.acquire.assert_not_called()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.span_end")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.span_start")
    def test_receive_message_invalid_length(self, mock_span_start, mock_span_end):
        mock_sem = MagicMock()
        mock_shm_buf = bytearray(1024)
        mock_shm_buf[0:4] = (0).to_bytes(4, "little")  # msg_length = 0
        mock_shm = MagicMock()
        mock_shm.buf = mock_shm_buf

        self.channel.consumer_semaphore = mock_sem
        self.channel.shared_memory = mock_shm
        self.channel.producer_semaphore = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            self.channel.receive_message(ExecuteRequest)
        self.assertIn("Invalid message length", str(ctx.exception))

    def test_send_binary_data(self):
        mock_sem = MagicMock()
        mock_shm_buf = bytearray(1024)
        mock_shm = MagicMock()
        mock_shm.buf = mock_shm_buf

        self.channel.producer_semaphore = mock_sem
        self.channel.consumer_semaphore = mock_sem
        self.channel.shared_memory = mock_shm

        byte_message = b"test_binary_data"
        self.channel.send_binary_data(byte_message)

        mock_sem.acquire.assert_called_once()
        mock_sem.release.assert_called_once()
        msg_len = int.from_bytes(mock_shm_buf[0:4], "little")
        self.assertEqual(msg_len, len(byte_message))
        self.assertEqual(bytes(mock_shm_buf[4:4 + msg_len]), byte_message)

    def test_send_binary_data_exceeds_size(self):
        mock_sem = MagicMock()
        self.channel.producer_semaphore = mock_sem
        self.channel.shared_memory = MagicMock()

        byte_message = b"x" * SharedMemoryChannel.DEFAULT_SHARED_MEMORY_SIZE
        with self.assertRaises(ValueError) as ctx:
            self.channel.send_binary_data(byte_message)
        self.assertIn("exceeds shared memory size limit", str(ctx.exception))
        mock_sem.acquire.assert_not_called()

    def test_get_request(self):
        sem_p = create_sem("/test_shm_execute_produce_0", 0)
        sem_c = create_sem("/test_shm_execute_consume_0", 1)
        shm.SharedMemory("/test_shm_execute", create=True, size=4096)

        mem = SharedMemoryChannel("test_shm", 0)
        request = create_request()
        mem.open_channel("execute")
        proto_data = request.SerializeToString()
        length = len(proto_data)
        mem.shared_memory.buf[0:4] = length.to_bytes(4, "little")
        mem.shared_memory.buf[4: length + 4] = proto_data

        request2 = mem.receive_message(ExecuteRequest)

        sem_p.close()
        sem_c.close()
        mem.shared_memory.close()

        metadata2 = request2.execute_model_request.seq_group_metadata_list[0]
        blocks = np.frombuffer(metadata2.block_tables[0], dtype=np.int64).tolist()
        self.assertEqual(blocks[0], 1)
        self.assertEqual(blocks[1], 3)
        self.assertEqual(blocks[2], 4)


class TestSharedMemCommunication(unittest.TestCase):
    def setUp(self):
        SharedMemCommunication._instance = None
        self.mock_config = MagicMock()
        self.mock_config.shm_name_prefix = "test_shm"
        self.mock_config.local_rank = 0
        self.mock_config.npu_num_per_dp = 2
        self.mock_config.global_rank = 0
        self.mock_config.global_world_size = 4
        self.mock_config.local_world_size = 2
        self.mock_config.npu_device_id = "0"

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_init(self, mock_channel_cls):
        mock_request_channel = MagicMock()
        mock_response_channel = MagicMock()
        mock_channel_cls.side_effect = [mock_request_channel, mock_response_channel] * 4

        comm = SharedMemCommunication(self.mock_config)

        self.assertEqual(len(comm._channels), 4)
        for channel_name in SharedMemCommunication.CHANNEL_NAMES:
            self.assertIn(channel_name, comm._channels)
            self.assertEqual(
                comm._channels[channel_name]["request"], mock_request_channel
            )
            self.assertEqual(
                comm._channels[channel_name]["response"], mock_response_channel
            )
            prefix = f"{self.mock_config.shm_name_prefix}_{channel_name}"
            mock_channel_cls.assert_any_call(prefix, 0)
            mock_request_channel.open_channel.assert_any_call("request")
            mock_response_channel.open_channel.assert_any_call("response")

        self.assertIsInstance(comm.request_router, RequestRouter)
        self.assertFalse(comm._is_running)
        self.assertEqual(comm._threads, [])

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.CoreThread")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_start(self, mock_channel_cls, mock_core_thread):
        mock_request_channel = MagicMock()
        mock_response_channel = MagicMock()
        mock_channel_cls.side_effect = [mock_request_channel, mock_response_channel] * 4

        mock_thread_instance = MagicMock()
        mock_core_thread.return_value = mock_thread_instance

        comm = SharedMemCommunication(self.mock_config)
        comm.start()

        self.assertEqual(mock_core_thread.call_count, 3)
        for channel_name in SharedMemCommunication.CHANNELS_WITH_REQUEST_LISTENER:
            mock_core_thread.assert_any_call(
                target=comm._process_incoming_requests,
                args=(channel_name,),
                daemon=True,
                name=channel_name
            )
        self.assertEqual(mock_thread_instance.start.call_count, 3)
        self.assertEqual(mock_thread_instance.join.call_count, 3)
        self.assertTrue(comm._is_running)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_send_transfer_response_cls(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        with patch.object(
            SharedMemCommunication, "send_response", new_callable=MagicMock
        ) as mock_send_response:
            _ = SharedMemCommunication.get_instance(self.mock_config)
            mock_response = MagicMock(spec=ExecuteModelResponse)

            SharedMemCommunication.send_transfer_response_cls(mock_response)

            mock_send_response.assert_called_once_with(mock_response, is_transfer=True)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_transfer(self, mock_channel_cls):
        mock_pd_link_channel = MagicMock()
        mock_transfer_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["shared_sync_link"]["response"] = mock_pd_link_channel
        comm._channels["transfer"]["response"] = mock_transfer_channel

        mock_response = MagicMock(spec=ExecuteResponse)
        mock_response.HasField.side_effect = lambda x: x == "pd_link_response"
        comm.send_response(mock_response, is_transfer=True)
        mock_pd_link_channel.send_message.assert_called_once_with(
            mock_response, buffer_offset=0
        )

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_command(self, mock_channel_cls):
        mock_shared_sync_link_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["shared_sync_link"]["response"] = mock_shared_sync_link_channel

        mock_response = MagicMock(spec=ExecuteResponse)
        mock_response.HasField.side_effect = lambda x: x == "lora_operation_response"
        comm.send_response(mock_response, is_command=True)
        mock_shared_sync_link_channel.send_message.assert_called_once_with(
            mock_response, buffer_offset=0
        )

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_non_zero_rank(self, mock_channel_cls):
        self.mock_config.local_rank = 1
        mock_response_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["execute"]["response"] = mock_response_channel

        mock_response = MagicMock(spec=ExecuteResponse)
        mock_response.HasField.return_value = False

        comm.send_response(mock_response)

        mock_response_channel.producer_semaphore.acquire.assert_called_once()
        mock_response_channel.consumer_semaphore.release.assert_called_once()
        mock_response_channel.send_message.assert_not_called()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_stop(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        comm = SharedMemCommunication(self.mock_config)
        comm._is_running = True
        comm.stop()
        self.assertFalse(comm._is_running)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_is_running(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        comm = SharedMemCommunication(self.mock_config)
        self.assertFalse(comm.is_running())
        comm._is_running = True
        self.assertTrue(comm.is_running())

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.AdaptiveGarbageCollector")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.RequestRouter")
    @patch.object(SharedMemCommunication, "_apply_config_to_request")
    def test_process_incoming_requests(
        self,
        mock_apply_config,
        mock_request_router_cls,
        mock_garbage_collector,
        mock_shm,
        mock_sem,
    ):
        with (
            patch(
                "mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission"
            ) as mock_check_owner_and_permission,
        ):
            mock_sem.return_value = MagicMock()
            mock_shm.return_value = MagicMock()
            mock_check_owner_and_permission.return_value = None
            mock_request_router_instance = MagicMock()
            mock_request_router_instance.accept = MagicMock()
            mock_request_router_cls.return_value = mock_request_router_instance

            mock_channel = MagicMock()
            mock_request = MagicMock(spec=ExecuteRequest)
            mock_request.execute_type = ExecuteType.MODEL_INFER

            def stop_loop_after_first_call(*args, **kwargs):
                comm._is_running = False
                return mock_request

            mock_channel.receive_message.side_effect = stop_loop_after_first_call

            comm = SharedMemCommunication(self.mock_config)
            comm._channels["execute"]["request"] = mock_channel
            comm._is_running = True

            comm._process_incoming_requests("execute")

            mock_channel.receive_message.assert_called_once_with(ExecuteRequest)
            mock_apply_config.assert_called_once_with(mock_request)
            mock_request_router_instance.accept.assert_called_once_with(mock_request)
            mock_garbage_collector.get_instance().request_counter_increase.assert_called_once()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_apply_config_to_request_init(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        comm = SharedMemCommunication(self.mock_config)
        mock_request = MagicMock(spec=ExecuteRequest)
        mock_request.execute_type = MODEL_INIT
        original_config = {"init_key": "init_val"}
        mock_request.config = original_config.copy()

        comm._apply_config_to_request(mock_request)

        expected_config = original_config.copy()
        expected_config.update(
            {
                "rank": "0",
                "global_rank": "0",
                "global_world_size": "4",
                "local_rank": "0",
                "local_world_size": "2",
                "npu_device_id": "0",
                "world_size": "4",
            }
        )
        self.assertEqual(comm._init_request_config, expected_config)
        self.assertIn("rank", mock_request.config)
        self.assertEqual(mock_request.config["global_rank"], "0")
        self.assertEqual(mock_request.config["npu_device_id"], "0")

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_apply_config_to_request_non_init(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None

        comm = SharedMemCommunication(self.mock_config)
        comm._init_request_config = {"saved_key": "saved_val"}
        mock_request = MagicMock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_request.config = {}

        comm._apply_config_to_request(mock_request)

        self.assertEqual(mock_request.config["saved_key"], "saved_val")
        self.assertEqual(mock_request.config["world_size"], "4")
        self.assertEqual(mock_request.config["local_rank"], "0")

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_apply_config_to_request_no_global_rank(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None

        self.mock_config.global_rank = None
        self.mock_config.global_world_size = None
        comm = SharedMemCommunication(self.mock_config)
        comm._init_request_config = {}
        mock_request = MagicMock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_request.config = {}

        comm._apply_config_to_request(mock_request)

        self.assertEqual(mock_request.config["rank"], "0")
        self.assertEqual(mock_request.config["world_size"], "2")

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_error_channel(self, mock_channel_cls):
        mock_error_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["execute_error"]["response"] = mock_error_channel

        response = ExecuteResponse()
        response.execute_model_response.err_msg = "test error"

        comm.send_response(response)
        mock_error_channel.send_message.assert_called_once()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_init_results(self, mock_channel_cls):
        mock_execute_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["execute"]["response"] = mock_execute_channel

        response = ExecuteResponseBuilder.build_from_init_result({"npuBlockNum": "10"})

        comm.send_response(response)
        mock_execute_channel.send_message.assert_called_once()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_recover_command(self, mock_channel_cls):
        mock_sync_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["shared_sync_link"]["response"] = mock_sync_channel

        mock_response = MagicMock(spec=ExecuteResponse)
        mock_response.HasField.side_effect = lambda x: False

        comm.send_response(mock_response, is_recover_command=True)
        mock_sync_channel.send_message.assert_called_once()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_response_transfer_channel(self, mock_channel_cls):
        mock_transfer_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["transfer"]["response"] = mock_transfer_channel

        mock_response = MagicMock(spec=ExecuteResponse)
        mock_response.HasField.side_effect = lambda x: False

        comm.send_response(mock_response, is_transfer=True)
        mock_transfer_channel.send_message.assert_called_once()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_send_model_execute_response_cls(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        with patch.object(
            SharedMemCommunication, "send_response", new_callable=MagicMock
        ) as mock_send_response:
            _ = SharedMemCommunication.get_instance(self.mock_config)
            mock_response = MagicMock(spec=ExecuteModelResponse)
            SharedMemCommunication.send_model_execute_response_cls(mock_response)
            mock_send_response.assert_called_once_with(mock_response)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_send_model_execute_binary_data_cls(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        with patch.object(
            SharedMemCommunication, "send_model_execute_binary_data", new_callable=MagicMock
        ) as mock_send_binary:
            _ = SharedMemCommunication.get_instance(self.mock_config)
            SharedMemCommunication.send_model_execute_binary_data_cls(b"binary_data")
            mock_send_binary.assert_called_once_with(b"binary_data")

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_send_command_response_cls(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        with patch.object(
            SharedMemCommunication, "send_response", new_callable=MagicMock
        ) as mock_send_response:
            _ = SharedMemCommunication.get_instance(self.mock_config)
            mock_response = MagicMock(spec=ExecuteModelResponse)
            SharedMemCommunication.send_command_response_cls(mock_response)
            mock_send_response.assert_called_once_with(mock_response, is_command=True)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission")
    def test_send_recover_command_response_cls(self, mock_check_owner_and_permission, mock_shm, mock_sem):
        mock_sem.return_value = MagicMock()
        mock_shm.return_value = MagicMock()
        mock_check_owner_and_permission.return_value = None
        with patch.object(
            SharedMemCommunication, "send_response", new_callable=MagicMock
        ) as mock_send_response:
            _ = SharedMemCommunication.get_instance(self.mock_config)
            mock_response = MagicMock(spec=ExecuteModelResponse)
            SharedMemCommunication.send_recover_command_response_cls(mock_response)
            mock_send_response.assert_called_once_with(mock_response, is_recover_command=True)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.posix_ipc.Semaphore")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.shm.SharedMemory")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.AdaptiveGarbageCollector")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.RequestRouter")
    @patch.object(SharedMemCommunication, "_apply_config_to_request")
    def test_process_incoming_requests_kv_transfer(
        self, mock_apply_config, mock_request_router_cls, mock_garbage_collector,
        mock_shm, mock_sem,
    ):
        with patch(
            "mindie_llm.connector.request_listener.shared_mem_communication.check_owner_and_permission"
        ) as mock_check:
            mock_sem.return_value = MagicMock()
            mock_shm.return_value = MagicMock()
            mock_check.return_value = None
            mock_request_router_instance = MagicMock()
            mock_request_router_cls.return_value = mock_request_router_instance

            mock_channel = MagicMock()
            mock_request = MagicMock(spec=ExecuteRequest)
            mock_request.execute_type = ExecuteType.KV_TRANSFER

            def stop_loop(*args, **kwargs):
                comm._is_running = False
                return mock_request

            mock_channel.receive_message.side_effect = stop_loop

            comm = SharedMemCommunication(self.mock_config)
            comm._channels["transfer"]["request"] = mock_channel
            comm._is_running = True

            comm._process_incoming_requests("transfer")

            mock_garbage_collector.get_instance().request_counter_increase.assert_called_once()

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.RequestRouterEdge")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.RequestRouter")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_init_layerwise_master(self, mock_channel_cls, mock_router_cls, mock_edge_cls):
        mock_request_channel = MagicMock()
        mock_response_channel = MagicMock()
        mock_channel_cls.side_effect = [mock_request_channel, mock_response_channel] * 4

        self.mock_config.layerwise_disaggregated = "true"
        self.mock_config.layerwise_disaggregated_role_type = "master"
        self.mock_config.parent_pid = 12345

        comm = SharedMemCommunication(self.mock_config)

        mock_edge_cls.assert_called_once_with(12345)
        self.assertIs(comm.request_router, mock_edge_cls.return_value)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.RequestRouterCloud")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.RequestRouter")
    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_init_layerwise_slave(self, mock_channel_cls, mock_router_cls, mock_cloud_cls):
        mock_request_channel = MagicMock()
        mock_response_channel = MagicMock()
        mock_channel_cls.side_effect = [mock_request_channel, mock_response_channel] * 4

        self.mock_config.layerwise_disaggregated = "true"
        self.mock_config.layerwise_disaggregated_role_type = "slave"
        self.mock_config.parent_pid = 12345

        comm = SharedMemCommunication(self.mock_config)

        mock_cloud_cls.assert_called_once_with(12345)
        self.assertIs(comm.request_router, mock_cloud_cls.return_value)

    @patch("mindie_llm.connector.request_listener.shared_mem_communication.SharedMemoryChannel")
    def test_send_model_execute_binary_data_non_zero_rank(self, mock_channel_cls):
        self.mock_config.local_rank = 1
        self.mock_config.npu_num_per_dp = 2
        mock_response_channel = MagicMock()
        comm = SharedMemCommunication(self.mock_config)
        comm._channels["execute"]["response"] = mock_response_channel

        comm.send_model_execute_binary_data(b"data")

        mock_response_channel.producer_semaphore.acquire.assert_called_once()
        mock_response_channel.consumer_semaphore.release.assert_called_once()
        mock_response_channel.send_binary_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()