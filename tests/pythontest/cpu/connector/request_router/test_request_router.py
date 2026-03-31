# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import Mock, patch, MagicMock
import queue

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType
from mindie_llm.connector.request_router.request_router import RequestRouter


def stop_router_threads(router):
    """停止路由器的所有线程"""
    finalize_request = Mock(spec=ExecuteRequest)
    finalize_request.execute_type = ExecuteType.MODEL_FINALIZE
    
    router.inference_queue.put(finalize_request)
    router.pdlink_queue.put(finalize_request)
    router.transfer_queue.put(finalize_request)
    router.command_queue.put(finalize_request)
    router.query_queue.put(finalize_request)
    
    router.inference_related_thread.join(timeout=2)
    router.trans_related_thread.join(timeout=2)
    router.pdlink_related_thread.join(timeout=2)
    router.command_related_thread.join(timeout=2)
    router.query_related_thread.join(timeout=2)


class TestRequestRouterInit(unittest.TestCase):
    """测试 RequestRouter 初始化相关功能"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    def test_init_creates_queues(self):
        self.assertIsInstance(self.router.inference_queue, queue.Queue)
        self.assertIsInstance(self.router.transfer_queue, queue.Queue)
        self.assertIsInstance(self.router.pdlink_queue, queue.Queue)
        self.assertIsInstance(self.router.command_queue, queue.Queue)
        self.assertIsInstance(self.router.query_queue, queue.Queue)

    def test_init_starts_threads(self):
        self.assertTrue(self.router.inference_related_thread.is_alive())
        self.assertTrue(self.router.trans_related_thread.is_alive())
        self.assertTrue(self.router.pdlink_related_thread.is_alive())
        self.assertTrue(self.router.command_related_thread.is_alive())
        self.assertTrue(self.router.query_related_thread.is_alive())

    def test_init_default_values(self):
        self.assertIsNone(self.router.enable_dp_distributed)


class TestRequestRouterConfigMethods(unittest.TestCase):
    """测试配置相关静态方法"""

    def test_get_config_dict(self):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "standard"),
            ("cpu_mem", "2048"),
            ("model_path", "/path/to/model")
        ]

        result = RequestRouter.get_config_dict(mock_config)

        self.assertEqual(result["infer_mode"], "standard")
        self.assertEqual(result["cpu_mem"], "2048")
        self.assertEqual(result["model_path"], "/path/to/model")

    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    def test_get_model_impl_config_standard(self, mock_base_config):
        model_config = {"infer_mode": "standard"}
        mock_config_instance = Mock()
        mock_base_config.return_value = mock_config_instance

        result = RequestRouter.get_model_impl_config(model_config)

        mock_base_config.assert_called_once_with(model_config)
        self.assertEqual(result, mock_config_instance)

    @patch("mindie_llm.connector.request_router.request_router.DmiConfig")
    def test_get_model_impl_config_dmi(self, mock_dmi_config):
        model_config = {"infer_mode": "dmi"}
        mock_config_instance = Mock()
        mock_dmi_config.return_value = mock_config_instance

        result = RequestRouter.get_model_impl_config(model_config)

        mock_dmi_config.assert_called_once_with(model_config)
        self.assertEqual(result, mock_config_instance)

    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_get_model_impl_config_invalid_mode(self, mock_logger):
        model_config = {"infer_mode": "invalid"}

        result = RequestRouter.get_model_impl_config(model_config)

        mock_logger.error.assert_called_once()
        self.assertIsNone(result)

    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    def test_get_model_impl_config_default_mode(self, mock_base_config):
        model_config = {}
        mock_config_instance = Mock()
        mock_base_config.return_value = mock_config_instance

        result = RequestRouter.get_model_impl_config(model_config)

        mock_base_config.assert_called_once_with(model_config)
        self.assertEqual(result, mock_config_instance)


class TestRequestRouterInitialize(unittest.TestCase):
    """测试初始化流程"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    @patch("mindie_llm.connector.request_router.request_router.send_model_execute_response")
    @patch("mindie_llm.connector.request_router.request_router.RouterImpl")
    def test_initialize_standard_mode(self, mock_router_impl_cls, mock_send, mock_base_config):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "standard"),
            ("cpu_mem", "2048"),
            ("model_path", "/path/to/standard/model"),
            ("device", "npu"),
            ("distributed_enable", True)
        ]

        mock_base_config_instance = Mock()
        mock_model_config = dict(mock_config.items.return_value)
        mock_base_config_instance.model_config = mock_model_config
        mock_base_config_instance.distributed_enable = True
        mock_base_config.return_value = mock_base_config_instance

        mock_router_impl_instance = Mock()
        mock_router_impl_instance.initialize.return_value = {"status": "ok"}
        mock_router_impl_cls.return_value = mock_router_impl_instance

        self.router.initialize(mock_config)

        mock_base_config.assert_called_once_with(mock_model_config)
        mock_router_impl_instance.initialize.assert_called_once_with(mock_base_config_instance)
        mock_send.assert_called_once()
        self.assertTrue(self.router.enable_dp_distributed)

    @patch("mindie_llm.connector.request_router.request_router.DmiConfig")
    @patch("mindie_llm.connector.request_router.request_router.send_model_execute_response")
    @patch("mindie_llm.connector.request_router.request_router.RouterImpl")
    def test_initialize_dmi_mode(self, mock_router_impl_cls, mock_send, mock_dmi_config):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "dmi"),
            ("cpu_mem", "1024"),
            ("model_path", "/path/to/model"),
            ("device", "npu"),
            ("distributed_enable", False)
        ]

        mock_dmi_config_instance = Mock()
        mock_model_config = dict(mock_config.items.return_value)
        mock_dmi_config_instance.model_config = mock_model_config
        mock_dmi_config_instance.distributed_enable = False
        mock_dmi_config.return_value = mock_dmi_config_instance

        mock_router_impl_instance = Mock()
        mock_router_impl_instance.initialize.return_value = {"status": "ok"}
        mock_router_impl_cls.return_value = mock_router_impl_instance

        self.router.initialize(mock_config)

        mock_dmi_config.assert_called_once_with(mock_model_config)
        mock_router_impl_cls.assert_called_once()
        mock_router_impl_instance.initialize.assert_called_once_with(mock_dmi_config_instance)
        mock_send.assert_called_once()
        self.assertFalse(self.router.enable_dp_distributed)

    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_initialize_invalid_mode(self, mock_logger):
        mock_config = Mock()
        mock_config.items.return_value = [("infer_mode", "invalid")]

        with self.assertRaises((UnboundLocalError, AttributeError)):
            self.router.initialize(mock_config)
        mock_logger.error.assert_called_once()


class TestRequestRouterAccept(unittest.TestCase):
    """测试请求分发功能"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    def test_accept_model_infer(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER

        self.router.accept(mock_request)

        self.assertEqual(self.router.inference_queue.qsize(), 1)
        self.assertEqual(self.router.transfer_queue.qsize(), 0)
        self.assertEqual(self.router.pdlink_queue.qsize(), 0)
        self.assertEqual(self.router.command_queue.qsize(), 0)
        self.assertEqual(self.router.query_queue.qsize(), 0)

    def test_accept_start_command_exec(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.START_COMMAND_EXEC

        self.router.accept(mock_request)

        self.assertEqual(self.router.inference_queue.qsize(), 1)

    def test_accept_recover_command_exec(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.RECOVER_COMMAND_EXEC

        self.router.accept(mock_request)

        self.assertEqual(self.router.inference_queue.qsize(), 1)

    def test_accept_pd_link(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PD_LINK

        self.router.accept(mock_request)

        self.assertEqual(self.router.pdlink_queue.qsize(), 1)
        self.assertEqual(self.router.inference_queue.qsize(), 0)

    def test_accept_kv_transfer(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.KV_TRANSFER

        self.router.accept(mock_request)

        self.assertEqual(self.router.transfer_queue.qsize(), 1)

    def test_accept_clear_command_exec(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.CLEAR_COMMAND_EXEC

        self.router.accept(mock_request)

        self.assertEqual(self.router.transfer_queue.qsize(), 1)

    def test_accept_lora_operation(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.LORA_OPERATION

        self.router.accept(mock_request)

        self.assertEqual(self.router.command_queue.qsize(), 1)

    def test_accept_pause_command_exec(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PAUSE_COMMAND_EXEC

        self.router.accept(mock_request)

        self.assertEqual(self.router.command_queue.qsize(), 1)

    def test_accept_pause_command_exec_roce(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PAUSE_COMMAND_EXEC_ROCE

        self.router.accept(mock_request)

        self.assertEqual(self.router.command_queue.qsize(), 1)

    def test_accept_pd_link_status_query(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PD_LINK_STATUS_QUERY

        self.router.accept(mock_request)

        self.assertEqual(self.router.query_queue.qsize(), 1)

    def test_accept_model_finalize(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        self.router.accept(mock_request)

        self.assertEqual(self.router.inference_queue.qsize(), 1)
        self.assertEqual(self.router.pdlink_queue.qsize(), 1)
        self.assertEqual(self.router.transfer_queue.qsize(), 1)
        self.assertEqual(self.router.command_queue.qsize(), 1)
        self.assertEqual(self.router.query_queue.qsize(), 1)

    def test_accept_unknown_type(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999

        self.router.accept(mock_request)

        self.assertEqual(self.router.inference_queue.qsize(), 1)


class TestRequestRouterDoInference(unittest.TestCase):
    """测试推理处理线程"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_model_infer(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_inference()

        self.router.router_impl.execute.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_model_infer_second(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER_SECOND
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_inference()

        self.router.router_impl.execute.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_inference_model_finalize(self, mock_logger, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request]):
            self.router.do_inference()

        self.router.router_impl.finalize.assert_called_once()
        mock_logger.info.assert_called_with("[python thread: infer] model finalized.")

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_text_generator_cleanup(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_inference()

        self.router.router_impl.seq_ctrl.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_recover_command_exec(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.RECOVER_COMMAND_EXEC
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_inference()

        self.router.router_impl.recover_command_exec.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_inference_start_command_exec(self, _):
        self.router.router_impl.is_inference_pause = True
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.START_COMMAND_EXEC
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_inference()

        self.router.router_impl.recover_command_exec.assert_called_once_with(mock_request)
        self.assertFalse(self.router.router_impl.is_inference_pause)

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_inference_unknown_type(self, mock_logger, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.inference_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_inference()

        mock_logger.error.assert_called()

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.prof_step")
    def test_do_inference_queue_empty(self, mock_prof_step, _):
        call_count = [0]

        def side_effect_timeout(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise queue.Empty()
            mock_request = Mock(spec=ExecuteRequest)
            mock_request.execute_type = ExecuteType.MODEL_FINALIZE
            return mock_request

        with patch.object(self.router.inference_queue, 'get', side_effect=side_effect_timeout):
            self.router.do_inference()

        mock_prof_step.assert_called()


class TestRequestRouterDoPdlink(unittest.TestCase):
    """测试 PD Link 处理线程"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_pdlink_pd_link(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PD_LINK
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.pdlink_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_pdlink()

        self.router.router_impl.pd_role.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_pdlink_model_finalize(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.pdlink_queue, 'get', side_effect=[mock_request]):
            self.router.do_pdlink()

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_pdlink_unknown_type(self, mock_logger, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.pdlink_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_pdlink()

        mock_logger.error.assert_called()


class TestRequestRouterDoTransfer(unittest.TestCase):
    """测试传输处理线程"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_transfer_kv_transfer(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.KV_TRANSFER
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.transfer_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_transfer()

        self.router.router_impl.transfer_data.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_transfer_clear_command_exec(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.CLEAR_COMMAND_EXEC
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.transfer_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_transfer()

        self.router.router_impl.recover_command_exec.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_transfer_model_finalize(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.transfer_queue, 'get', side_effect=[mock_request]):
            self.router.do_transfer()

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_transfer_unknown_type(self, mock_logger, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.transfer_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_transfer()

        mock_logger.error.assert_called()


class TestRequestRouterDoCommand(unittest.TestCase):
    """测试命令处理线程"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_command_lora_operation(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.LORA_OPERATION
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.command_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_command()

        self.router.router_impl.process_lora_operation.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_command_pause_command_exec(self, _):
        self.router.router_impl.is_inference_pause = False
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PAUSE_COMMAND_EXEC
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.command_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_command()

        self.router.router_impl.recover_command_exec.assert_called_once_with(mock_request)
        self.assertTrue(self.router.router_impl.is_inference_pause)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_command_pause_command_exec_roce(self, _):
        self.router.router_impl.is_inference_pause = False
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PAUSE_COMMAND_EXEC_ROCE
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.command_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_command()

        self.router.router_impl.recover_command_exec.assert_called_once_with(mock_request)
        self.assertTrue(self.router.router_impl.is_inference_pause)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_command_model_finalize(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.command_queue, 'get', side_effect=[mock_request]):
            self.router.do_command()

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_command_unknown_type(self, mock_logger, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.command_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_command()

        mock_logger.error.assert_called()


class TestRequestRouterDoQuery(unittest.TestCase):
    """测试查询处理线程"""

    def setUp(self):
        self.router = RequestRouter()
        self.router.router_impl = Mock()

    def tearDown(self):
        stop_router_threads(self.router)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_query_pd_link_status_query(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.PD_LINK_STATUS_QUERY
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.query_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_query()

        self.router.router_impl.query_link_status.assert_called_once_with(mock_request)

    @patch("mindie_llm.utils.status.CoreThread")
    def test_do_query_model_finalize(self, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.query_queue, 'get', side_effect=[mock_request]):
            self.router.do_query()

    @patch("mindie_llm.utils.status.CoreThread")
    @patch("mindie_llm.connector.request_router.request_router.logger")
    def test_do_query_unknown_type(self, mock_logger, _):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = 9999
        finalize_request = Mock(spec=ExecuteRequest)
        finalize_request.execute_type = ExecuteType.MODEL_FINALIZE

        with patch.object(self.router.query_queue, 'get', side_effect=[mock_request, finalize_request]):
            self.router.do_query()

        mock_logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
