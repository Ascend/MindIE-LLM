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
import threading
import struct
import numpy as np

from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteRequest,
    ForwardType,
    TGCleanupRequest,
    ExecuteResponse,
    ExecuteType,
    LoraOperationType,
    LoraOperationStatus,
    PDErrorCode,
)
from mindie_llm.model_wrapper.utils.config import DmiConfig
from mindie_llm.model_wrapper.utils.error import ModelWrapperErrorCode
from mindie_llm.connector.request_router.router_impl import (
    RouterImpl,
    _print_component_error_log,
    SRC_BLOCK_TABLE_KEY,
    DST_BLOCK_TABLE_KEY,
    REQ_INDEX,
    MindieLlmStatusCode
)
from mindie_llm.connector.common.input_metadata_composite import InputMetadataComposite
from mindie_llm.connector.common import send_model_execute_response, send_transfer_response, send_command_response
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder as RealExecuteResponseBuilder
from mindie_llm.utils.log.error_code import ErrorCode, ErrorCodeException
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager


class TestRouterImplUtils(unittest.TestCase):
    def test_print_component_error_log(self):
        with self.assertRaises(RuntimeError) as ctx:
            _print_component_error_log(Exception("ACL stream synchronize failed"))
        self.assertIn("HCCL execute error", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            _print_component_error_log(
                Exception("The Inner error is reported as above. The process exits for this inner error"))
        self.assertIn("CANN execute error", str(ctx.exception))

    def test_print_component_error_log_other_exception(self):
        """其它异常不应被重新抛出，仅记录日志"""
        _print_component_error_log(ValueError("some other error"))  # 不应抛出


class TestRouterImpl(unittest.TestCase):
    def setUp(self):
        self.router = RouterImpl()
        self.mock_generator = Mock()
        self.router.generator = self.mock_generator
        self.router.config = Mock(spec=DmiConfig)
        self.router.config.distributed_enable = False

    def test_init(self):
        test_router = RouterImpl()
        self.assertIsNone(test_router.config)
        self.assertIsNone(test_router.max_seq_len)
        self.assertIsNone(test_router.rank)
        self.assertEqual(test_router.tp_size, 1)
        self.assertEqual(test_router.dp_size, 1)
        self.assertIsInstance(test_router.lock, type(threading.Lock()))
        self.assertFalse(test_router.has_inited)
        self.assertEqual(test_router.empty_batch_task_id, -10)
        self.assertEqual(test_router.block_id_for_empty_req, -1)

    def test_check_output_valid(self):
        mock_output = Mock()
        mock_output.token_ids = np.array([[1, 2, 3]], dtype=np.uint32)
        mock_output.eos_info = np.array([[0, 0, 1]], dtype=np.uint32)
        mock_output.num_top_tokens = np.array([3], dtype=np.int32)
        RouterImpl.check_output(mock_output)  # 不应抛出异常

    def test_check_output_invalid(self):
        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=None,
                eos_info=np.array([1]),
                num_top_tokens=np.array([1])
            )
            RouterImpl.check_output(mock_output)

        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=np.array([], dtype=np.uint32),
                eos_info=np.array([1], dtype=np.uint32),
                num_top_tokens=np.array([1])
            )
            RouterImpl.check_output(mock_output)

        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=np.array([4294967296], dtype=np.uint64),
                eos_info=np.array([1], dtype=np.uint32),
                num_top_tokens=np.array([1])
            )
            RouterImpl.check_output(mock_output)

    def test_get_id_to_block_table_map(self):
        id_to_block_table_map = {
            "inst1": {SRC_BLOCK_TABLE_KEY: [], DST_BLOCK_TABLE_KEY: [], REQ_INDEX: []},
            "inst2": {SRC_BLOCK_TABLE_KEY: [], DST_BLOCK_TABLE_KEY: [], REQ_INDEX: []}
        }
        params = {
            "id_to_block_table_map": id_to_block_table_map,
            "segment_len": 3,
            "src_block_table": [1, 2, -1, 3, 4, 5],
            "dst_block_table": [10, 20, 30, 40, 50, 60],
            "inst_ids": ["inst1", "inst2"],
            "index": 0
        }
        RouterImpl._get_id_to_block_table_map(**params)

        self.assertEqual(id_to_block_table_map["inst1"][SRC_BLOCK_TABLE_KEY], [1, 2])
        self.assertEqual(id_to_block_table_map["inst1"][DST_BLOCK_TABLE_KEY], [10, 20])
        self.assertEqual(id_to_block_table_map["inst2"][SRC_BLOCK_TABLE_KEY], [3, 4, 5])
        self.assertEqual(id_to_block_table_map["inst2"][DST_BLOCK_TABLE_KEY], [30, 40, 50])

    @patch('mindie_llm.connector.request_router.router_impl.set_npu_compile_mode')
    @patch('mindie_llm.connector.request_router.router_impl.Generator')
    def test_initialize(self, mock_generator_cls, mock_set_compile):
        # 构建与业务代码一致的配置
        mock_config = Mock(spec=DmiConfig)
        mock_config.max_seq_len = 1024
        mock_config.rank = 0
        mock_config.local_rank = 0
        mock_config.npu_device_id = 0
        mock_config.tp_size = 2
        mock_config.dp_size = 2
        mock_config.cache_block_size = 64
        mock_config.model_weight_path = "/mock/path/weights"
        mock_config.model_config = {
            "model_weight_path": "/mock/path/weights",
            "other_param": "test_value"
        }
        mock_config.distributed_enable = False

        mock_generator = mock_generator_cls.return_value
        mock_generator.is_mix_model = False
        mock_generator.max_position_embeddings = 2048

        mock_kvcache_settings = Mock()
        mock_kvcache_settings.num_npu_blocks = 10
        mock_kvcache_settings.num_cpu_blocks = 5
        mock_generator.kvcache_settings = mock_kvcache_settings

        mock_model_wrapper = Mock()
        mock_mapping = Mock()
        mock_mapping.has_dp = Mock(return_value=False)
        mock_model_wrapper.mapping = mock_mapping
        mock_generator.model_wrapper = mock_model_wrapper

        result = self.router.initialize(mock_config)

        self.assertEqual(self.router.max_seq_len, 1024)
        self.assertEqual(self.router.dp_rank_id, 0)
        self.assertEqual(self.router.block_size, 64)
        mock_generator_cls.assert_called_once_with(model_config=mock_config.model_config)
        mock_set_compile.assert_called_once()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cpuBlockNum"], "5")
        self.assertIn("kvCacheDescs", result)
        self.assertEqual(result["kvCacheDescs"][0]["npuBlockNum"], 9)
        self.assertEqual(result["kvCacheDescs"][0]["blockSize"], 64)

        mock_config.max_seq_len = 3000
        with self.assertLogs(logger="llm", level='WARNING') as log:
            self.router.initialize(mock_config)
            self.assertGreater(len(log.output), 0)
            self.assertIn("WARN:llm:[MIE04E13030A]", log.output[0])
            self.assertIn("from model(/mock/path/weights)", log.output[0])

        # Test DP scenario
        mock_mapping.has_dp = Mock(return_value=True)
        result_with_dp = self.router.initialize(mock_config)
        self.assertEqual(result_with_dp["kvCacheDescs"][0]["npuBlockNum"], 8)

    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    def test_seq_ctrl(self, mock_send_response):
        mock_request = Mock(spec=ExecuteRequest)
        mock_cleanup_req = Mock(spec=TGCleanupRequest)
        target_seq_ids = [1, 2, 3]
        mock_cleanup_req.seq_ids = target_seq_ids
        mock_request.text_generator_cleanup_request = mock_cleanup_req

        self.router.seq_ctrl(mock_request)

        call_args = self.mock_generator.clear_cache.call_args
        self.assertIsInstance(call_args[0][0], np.ndarray)
        self.assertEqual(call_args[0][0].dtype, np.int_)
        self.assertTrue(np.array_equal(call_args[0][0], np.array(target_seq_ids, dtype=int)))

    def test_seq_ctrl_empty_seq_ids(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_cleanup_req = Mock(spec=TGCleanupRequest)
        mock_cleanup_req.seq_ids = []
        mock_request.text_generator_cleanup_request = mock_cleanup_req

        with self.assertRaises(ValueError) as ctx:
            self.router.seq_ctrl(mock_request)
        self.assertIn("SEQ_IDS_TO_CLEAR", str(ctx.exception))

    @patch.object(RouterImpl, '_generate')
    @patch.object(RouterImpl, '_mix')
    def test_execute_forward_types(self, mock_mix, mock_generate):
        mock_request = Mock(spec=ExecuteRequest)
        mock_execute_model_req = Mock()
        mock_execute_model_req.seq_group_metadata_list = [Mock()]
        mock_request.execute_model_request = mock_execute_model_req

        mock_execute_model_req.forward_type = ForwardType.DECODE
        self.router.execute(mock_request)
        mock_generate.assert_called_once_with(mock_request, is_prefill=False, is_mix=False)

        mock_execute_model_req.forward_type = ForwardType.PREFILL
        self.router.execute(mock_request)
        mock_generate.assert_called_with(mock_request, is_prefill=True, is_mix=False)

        mock_execute_model_req.forward_type = ForwardType.MIXED
        self.router.execute(mock_request)
        mock_mix.assert_called_once_with(mock_request)

        mock_execute_model_req.forward_type = 999
        self.router.execute(mock_request)

    def test_execute_unknown_forward_type_logs(self):
        """未知 forward_type 时应记录 ERROR 日志"""
        mock_request = Mock(spec=ExecuteRequest)
        mock_execute_model_req = Mock()
        mock_execute_model_req.seq_group_metadata_list = [Mock()]
        mock_execute_model_req.forward_type = 999
        mock_request.execute_model_request = mock_execute_model_req

        with self.assertLogs(logger="llm", level='ERROR') as log_ctx:
            self.router.execute(mock_request)
        self.assertTrue(any("Unknown forward_type" in msg for msg in log_ctx.output))

    @patch('mindie_llm.connector.request_router.router_impl.convert_pull_kv_request_to_input_metadata_composite')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    def test_transfer_data(self, mock_response_builder, mock_send_response, mock_convert):
        mock_request = Mock(spec=ExecuteRequest)

        mock_pull_kv_info1 = Mock()
        mock_pull_kv_info1.dst_block_tables = [np.array([10, 20], dtype=np.int64).tobytes()]
        mock_pull_kv_info1.src_block_tables = [np.array([1, 2], dtype=np.int64).tobytes()]
        mock_pull_kv_info1.seq_group_metadata = Mock(request_id="1001", prompt_lens=struct.pack("<q", 100))
        mock_pull_kv_info1.cluster_id = "10000"

        mock_pull_kv_info2 = Mock()
        mock_pull_kv_info2.dst_block_tables = [np.array([30, 40], dtype=np.int64).tobytes()]
        mock_pull_kv_info2.src_block_tables = [np.array([3, 4], dtype=np.int64).tobytes()]
        mock_pull_kv_info2.seq_group_metadata = Mock(request_id="1002", prompt_lens=struct.pack("<q", 100))
        mock_pull_kv_info2.cluster_id = "20000"

        mock_pull_kv_request = Mock()
        mock_pull_kv_request.pull_kv_infos = [mock_pull_kv_info1, mock_pull_kv_info2]
        mock_request.pull_kv_request = mock_pull_kv_request

        self.router.block_size = 64
        self.router.config.p_inst_enable_sp_cp = False
        self.router.config.remote_sp_size = 1
        self.router.config.dp_inst_id_to_cluster_id = {1:[1001], 2:[1002]}
        self.router.config.enable_mtp = False 
        self.mock_generator.kvcache_settings.num_npu_blocks = 10

        self.mock_generator.pull_kv.return_value = (MindieLlmStatusCode.SUCCESS, None)

        mock_response = Mock(spec=ExecuteResponse)
        mock_response_builder.build_from_transfer_result.return_value = mock_response

        self.router.transfer_data(mock_request)

        mock_convert.assert_called_once_with(
            mock_request.pull_kv_request,
            10,
            64,
            self.router.config
        )

        self.assertGreater(len(self.mock_generator.pull_kv.call_args[0][1]), 0)

        mock_response_builder.build_from_transfer_result.assert_called_with(
            0, {"1001": ModelWrapperErrorCode.SUCCESS, "1002": ModelWrapperErrorCode.SUCCESS}
        )
        mock_send_response.assert_called_once_with(mock_response)


    @patch('mindie_llm.connector.request_router.router_impl.get_attribute_info')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    def test_pd_role(self, mock_attribute_info,mock_send_response, mock_convert):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.pd_link_request = MagicMock()
        self.router.config.remote_unlink_cluster_id = []
        self.router.config.need_switch = False
        self.router.config.remote_link_cluster_id = []
        self.router.config.remote_link_device_physical_id = []
        self.router.config.remote_link_device_ips = []
        self.router.config.remote_link_host_ip = []
        self.router.config.remote_super_device_id = None
        self.router.config.remote_super_pod_id = None
        self.mock_generator.link.return_value = []
        self.router.pd_role(mock_request)

    @patch('mindie_llm.connector.request_router.router_impl.get_attribute_info')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_pd_role_unlink_exception(self, mock_send_response, mock_attribute_info):
        """unlink 抛出 HCCL 异常时，_print_component_error_log 会抛出 RuntimeError"""
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.pd_link_request = MagicMock()
        self.router.config.remote_unlink_cluster_id = {0: [1001]}
        self.router.config.need_switch = False
        self.router.config.remote_link_cluster_id = {}
        self.mock_generator.unlink.side_effect = Exception("ACL stream synchronize failed")

        with self.assertRaises(RuntimeError) as ctx:
            self.router.pd_role(mock_request)
        self.assertIn("HCCL execute error", str(ctx.exception))

    @patch('mindie_llm.connector.request_router.router_impl.get_attribute_info')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_pd_role_unlink_other_exception_returns_error(self, mock_send_response, mock_attribute_info):
        """unlink 抛出非 HCCL/CANN 异常时，应返回 PD_UNLINK_ERROR 而非抛出"""
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.pd_link_request = MagicMock()
        self.router.config.remote_unlink_cluster_id = {0: [1001]}
        self.router.config.need_switch = False
        self.router.config.remote_link_cluster_id = {}
        self.mock_generator.unlink.side_effect = ValueError("some other error")

        self.router.pd_role(mock_request)

        mock_send_response.assert_called_once()
        proto = mock_send_response.call_args[0][0]
        self.assertEqual(proto.status, ModelWrapperErrorCode.PD_UNLINK_ERROR.value)

    @patch('mindie_llm.connector.request_router.router_impl.get_attribute_info')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_pd_role_need_switch(self, mock_send_response, mock_attribute_info):
        """need_switch 为 True 时应调用 switch_role"""
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.pd_link_request = MagicMock()
        self.router.config.remote_unlink_cluster_id = []
        self.router.config.need_switch = True
        self.router.config.role = "encoder"
        self.router.config.remote_link_cluster_id = {}
        self.router.config.remote_link_device_physical_id = {}
        self.router.config.remote_link_device_ips = {}
        self.router.config.remote_link_host_ip = {}
        self.router.config.remote_super_device_id = None
        self.router.config.remote_super_pod_id = None
        self.mock_generator.link.return_value = []

        self.router.pd_role(mock_request)

        self.mock_generator.switch_role.assert_called_once_with("encoder")

    @patch('mindie_llm.connector.request_router.router_impl.get_attribute_info')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_pd_role_link_failed(self, mock_send_response, mock_attribute_info):
        """link 返回失败列表时，应构造 failed_link_info 并返回"""
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.pd_link_request = MagicMock()
        self.router.config.remote_unlink_cluster_id = []
        self.router.config.need_switch = False
        self.router.config.remote_link_cluster_id = {1: [1001]}
        self.router.config.remote_link_device_ips = {1: ["10.0.0.1"]}
        self.router.config.remote_link_device_physical_id = {}
        self.router.config.remote_link_host_ip = {}
        self.router.config.remote_super_device_id = None
        self.router.config.remote_super_pod_id = None
        self.mock_generator.link.return_value = [
            ("10.0.0.1", ModelWrapperErrorCode.PD_LINK_ERROR)
        ]

        self.router.pd_role(mock_request)

        mock_send_response.assert_called_once()
        proto = mock_send_response.call_args[0][0]
        self.assertEqual(proto.status, ModelWrapperErrorCode.PD_LINK_ERROR.value)
        self.assertEqual(len(proto.pd_link_response.failed_link_info), 1)
        self.assertEqual(proto.pd_link_response.failed_link_info[0].cluster_id, "1001")
        self.assertEqual(proto.pd_link_response.failed_link_info[0].pd_error_code, PDErrorCode.PD_LINK_ERROR)

    @patch('mindie_llm.connector.request_router.router_impl.send_command_response')
    def test_load_lora(self, mock_send_response):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.lora_operation_request = MagicMock()
        mock_request.lora_operation_request.lora_op_type = LoraOperationType.LOAD
        mock_request.lora_operation_request.lora_name = "fake_name"
        mock_request.lora_operation_request.lora_path = "fake_path"
        self.mock_generator.load_lora.return_value = LoraOperationStatus.LORA_CMD_SUCCESS
        self.router.process_lora_operation(mock_request)

    @patch('mindie_llm.connector.request_router.router_impl.send_command_response')
    def test_unload_lora(self, mock_send_response):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.lora_operation_request = MagicMock()
        mock_request.lora_operation_request.lora_op_type = LoraOperationType.UNLOAD
        mock_request.lora_operation_request.lora_name = "fake_name"
        mock_request.lora_operation_request.lora_path = "fake_path"
        self.mock_generator.unload_lora.return_value = LoraOperationStatus.LORA_CMD_SUCCESS
        self.router.process_lora_operation(mock_request)

    def test_process_lora_operation_unknown_type(self):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.lora_operation_request = MagicMock()
        mock_request.lora_operation_request.lora_op_type = 999  # 未知类型
        mock_request.lora_operation_request.lora_name = "fake_name"
        mock_request.lora_operation_request.lora_path = "fake_path"

        with self.assertRaises(UnboundLocalError):
            self.router.process_lora_operation(mock_request)
        self.mock_generator.load_lora.assert_not_called()
        self.mock_generator.unload_lora.assert_not_called()

    @patch('mindie_llm.connector.request_router.router_impl.send_command_response')
    def test_recover_command_exec(self, mock_send_command):
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.recover_command_request = Mock(command="CMD_PAUSE_ENGINE")

        ret_dict = {
            "npu_device_id": 0,
            "command_result": 0,
            "error_msg": "",
        }
        self.mock_generator.execute_recover_command.return_value = ret_dict

        self.router.recover_command_exec(mock_request)

        self.mock_generator.execute_recover_command.assert_called_once_with("CMD_PAUSE_ENGINE")
        mock_send_command.assert_called_once()

    @patch.object(RouterImpl, '_execute_empty_batch')
    def test_execute_forward_type_dummy(self, mock_empty_batch):
        mock_request = Mock(spec=ExecuteRequest)
        mock_execute_model_req = Mock()
        mock_execute_model_req.forward_type = ForwardType.DUMMY
        mock_request.execute_model_request = mock_execute_model_req

        self.router.execute(mock_request)

        mock_empty_batch.assert_called_once_with(mock_request)

    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    @patch('mindie_llm.connector.request_router.router_impl.make_dummy_input_metadata')
    def test_execute_empty_batch_err_msg(self, mock_make_dummy, mock_send):
        """ErrorCodeException 时应发送 err_msg 响应"""
        self.router.layerwise_disaggregated = False
        self.router.local_rank = 0
        self.router.config.infer_mode = "normal"
        self.mock_generator.kvcache_settings.num_npu_blocks = 10
        mock_dummy = Mock()
        mock_make_dummy.return_value = mock_dummy
        self.mock_generator.generate_token.side_effect = ErrorCodeException(ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY)

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_request.execute_model_request = Mock(forward_type=ForwardType.DUMMY)

        self.router._execute_empty_batch(mock_request)

        self.assertEqual(mock_send.call_count, 2)  # err_msg + empty proto
        first_call_proto = mock_send.call_args_list[0][0][0]
        self.assertEqual(first_call_proto.msg_type, ExecuteType.EXECUTE_ERROR)
        self.assertIn("MIE05E01000A", first_call_proto.execute_model_response.err_msg)

    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    @patch('mindie_llm.connector.request_router.router_impl.make_dummy_input_metadata')
    @patch('mindie_llm.connector.request_router.router_impl.make_dummy_input_metadata_dmi_decoder')
    def test_execute_empty_batch_dmi_decoder(self, mock_make_dmi_decoder, mock_make_dummy, mock_send):
        """dmi + decoder 时应调用 input_metadata_queue.put 和 make_dummy_input_metadata_dmi_decoder"""
        self.router.layerwise_disaggregated = False
        self.router.config.infer_mode = "dmi"
        self.router.config.role = "decoder"
        self.mock_generator.kvcache_settings.num_npu_blocks = 10
        mock_dummy = Mock()
        mock_dmi_decoder_result = Mock()
        mock_make_dummy.return_value = mock_dummy
        mock_make_dmi_decoder.return_value = mock_dmi_decoder_result
        self.mock_generator.generate_token.return_value = None

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_request.execute_model_request = Mock(forward_type=ForwardType.DUMMY)

        self.router._execute_empty_batch(mock_request)

        self.mock_generator.input_metadata_queue.put.assert_called_once_with(mock_dummy)
        mock_make_dmi_decoder.assert_called_once()
        self.mock_generator.generate_token.assert_called_once_with(mock_dmi_decoder_result)

    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    def test_finalize(self, mock_send):
        self.router.metrics = MagicMock()
        self.router.finalize()
        # 验证mock调用
        self.router.metrics.output.assert_called_once()
        mock_send.assert_called_once()

    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    @patch('mindie_llm.connector.request_router.router_impl.convert_pull_kv_request_to_input_metadata_composite')
    def test_transfer_data_with_sp_enabled(self, mock_convert, _mock_send, _mock_builder):
        mock_metadata = Mock()
        mock_convert.return_value = mock_metadata

        mock_request = Mock(spec=ExecuteRequest)
        mock_kv1 = Mock(
            dst_block_tables=[np.array([1, 2], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([10, 20], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="2001", prompt_lens=struct.pack("<q", 100)),
            cluster_id="20000"
        )
        mock_kv2 = Mock(
            dst_block_tables=[np.array([5, 6, 7, 8], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([50, 60, 70, 80], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="2002", prompt_lens=struct.pack("<q", 100)),
            cluster_id="30000"
        )
        mock_pull = Mock(pull_kv_infos=[mock_kv1, mock_kv2])
        mock_request.pull_kv_request = mock_pull

        self.router.block_size = 64
        self.router.config.p_inst_enable_sp_cp = True
        self.router.config.remote_sp_size = 2
        self.router.config.remote_cp_size = 1
        self.router.config.dp_inst_id_to_cluster_id = {2:[2001], 3:[2002]}

        
        self.mock_generator.kvcache_settings.num_npu_blocks = 20
        self.mock_generator.pull_kv.return_value = (MindieLlmStatusCode.SUCCESS, None)

        self.router.transfer_data(mock_request)

        self.assertTrue(self.mock_generator.pull_kv.called)
        self.assertEqual(len(self.mock_generator.pull_kv.call_args[0][1]), 2)

    @patch('mindie_llm.connector.request_router.router_impl.convert_pull_kv_request_to_input_metadata_composite')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_transfer_data_failure_case(self, mock_send, mock_convert):
        mock_metadata = Mock()
        mock_convert.return_value = mock_metadata

        mock_request = Mock(spec=ExecuteRequest)

        mock_kv = Mock(
            dst_block_tables=[np.array([10], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([1], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="3001", prompt_lens=struct.pack("<q", 100)),
            cluster_id="30000"
        )
        mock_request.pull_kv_request = Mock(pull_kv_infos=[mock_kv])

        self.router.block_size = 64
        self.router.config.p_inst_enable_sp_cp = False
        self.router.config.dp_inst_id_to_cluster_id = {3:[3001]}
        self.mock_generator.kvcache_settings.num_npu_blocks = 10

        # failed_p_id 应为底层实际的 cluster_id（3001），而非 dp_instance_id（30000）
        # 因为 pull_kv_items 中使用的是通过 dp_inst_id_to_cluster_id 映射后的实际 cluster_id
        self.mock_generator.pull_kv.return_value = (
            ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR,
            3001
        )

        self.router.transfer_data(mock_request)

        mock_send.assert_called_once()
        proto_response = mock_send.call_args[0][0]
        self.assertEqual(proto_response.msg_type, ExecuteType.KV_TRANSFER)
        # 顶层 status 恒为 SUCCESS 以便 executor 释放 kv cache；实际成败看各条 pd_error_code
        self.assertEqual(proto_response.status, ModelWrapperErrorCode.SUCCESS.value)
        self.assertEqual(proto_response.pull_kv_response.pull_kv_results[0].pd_error_code, ModelWrapperErrorCode.PD_PULL_KV_ERROR.value)
        self.assertEqual(proto_response.pull_kv_response.pull_kv_results[0].request_id, "3001")

    @patch('mindie_llm.connector.request_router.router_impl.convert_pull_kv_request_to_input_metadata_composite')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_transfer_data_pull_kv_fail_dp1_scale_down_p(self, mock_send, mock_convert):
        """
        模拟 2P1D 静态缩容场景（dp_size=1）：
        - 缩容1个P节点时，pull_kv 失败，generator.pull_kv 返回的 failed_p_id 是实际 cluster_id
        - pull_kv_info.cluster_id 是 dp_instance_id（格式：pInstanceId * 10000 + dpRank）
        - 修复后的代码通过 dp_inst_id_to_cluster_id 映射正确检测到失败
        - 若使用修复前的代码（直接比较 dp_instance_id），则无法检测到失败，D 节点会崩溃
        """
        mock_metadata = Mock()
        mock_convert.return_value = mock_metadata

        mock_request = Mock(spec=ExecuteRequest)

        # dp_instance_id = 10000，表示 P instance 1, dpRank 0
        # 通过映射 10000 // 10000 = 1，得到实际 cluster_id [1000000001]
        mock_kv = Mock(
            dst_block_tables=[np.array([10, 20], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([1, 2], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="req_450", prompt_lens=struct.pack("<q", 100)),
            cluster_id="10000"
        )
        mock_request.pull_kv_request = Mock(pull_kv_infos=[mock_kv])

        self.router.block_size = 64
        self.router.dp_size = 1
        self.router.config.p_inst_enable_sp_cp = False
        # P instance 1 对应的实际 cluster_id 为 1000000001（与日志中 Destroy cluster id:1000000001 对应）
        self.router.config.dp_inst_id_to_cluster_id = {1: [1000000001]}
        self.mock_generator.kvcache_settings.num_npu_blocks = 10

        # 缩容P节点导致 pull_kv 失败，返回的 failed_p_id 是实际 cluster_id
        self.mock_generator.pull_kv.return_value = (
            ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR,
            1000000001  # 实际 cluster_id，非 dp_instance_id
        )

        self.router.transfer_data(mock_request)

        mock_send.assert_called_once()
        proto_response = mock_send.call_args[0][0]
        # 顶层 status 恒为 SUCCESS；修复后能正确检测 pull_kv 失败，各条 pd_error_code 为 PD_PULL_KV_ERROR
        self.assertEqual(proto_response.status, ModelWrapperErrorCode.SUCCESS.value)
        self.assertEqual(proto_response.pull_kv_response.pull_kv_results[0].request_id, "req_450")
        self.assertEqual(
            proto_response.pull_kv_response.pull_kv_results[0].pd_error_code,
            ModelWrapperErrorCode.PD_PULL_KV_ERROR.value
        )

    @patch('mindie_llm.connector.request_router.router_impl.convert_pull_kv_request_to_input_metadata_composite')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_transfer_data_pull_kv_fail_partial_two_p_nodes(self, mock_send, mock_convert):
        """
        模拟 2P1D 缩容其中1个P的场景：
        - 2个请求分别来自 P1 和 P2
        - P1 被缩容，pull_kv 失败（failed_p_id 指向 P1 的 cluster_id）
        - P2 正常
        - 验证：来自 P1 的请求被正确标记为失败，后续请求也被标记为失败
          （因为 result_code 不重置，一旦出现失败后续全部标记失败）
        """
        mock_metadata = Mock()
        mock_convert.return_value = mock_metadata

        mock_request = Mock(spec=ExecuteRequest)

        # 请求1：来自 P1（dp_instance_id=10000，P instance 1）
        mock_kv1 = Mock(
            dst_block_tables=[np.array([10], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([1], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="req_100", prompt_lens=struct.pack("<q", 100)),
            cluster_id="10000"
        )
        # 请求2：来自 P2（dp_instance_id=20000，P instance 2）
        mock_kv2 = Mock(
            dst_block_tables=[np.array([30], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([3], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="req_200", prompt_lens=struct.pack("<q", 100)),
            cluster_id="20000"
        )
        mock_request.pull_kv_request = Mock(pull_kv_infos=[mock_kv1, mock_kv2])

        self.router.block_size = 64
        self.router.dp_size = 1
        self.router.config.p_inst_enable_sp_cp = False
        self.router.config.dp_inst_id_to_cluster_id = {
            1: [1001],  # P1 的 cluster_id
            2: [2001],  # P2 的 cluster_id
        }
        self.mock_generator.kvcache_settings.num_npu_blocks = 10

        # P1（cluster_id=1001）被缩容，pull_kv 失败
        self.mock_generator.pull_kv.return_value = (
            ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR,
            1001  # P1 的实际 cluster_id
        )

        self.router.transfer_data(mock_request)

        mock_send.assert_called_once()
        proto_response = mock_send.call_args[0][0]
        # 顶层 status 恒为 SUCCESS；各条结果通过 pd_error_code 表示失败
        self.assertEqual(proto_response.status, ModelWrapperErrorCode.SUCCESS.value)
        # req_100（来自P1）应标记为失败
        results = {r.request_id: r.pd_error_code for r in proto_response.pull_kv_response.pull_kv_results}
        self.assertEqual(results["req_100"], ModelWrapperErrorCode.PD_PULL_KV_ERROR.value)

    @patch('mindie_llm.connector.request_router.router_impl.convert_pull_kv_request_to_input_metadata_composite')
    @patch('mindie_llm.connector.request_router.router_impl.send_transfer_response')
    def test_transfer_data_pull_kv_fail_dp_gt1(self, mock_send, mock_convert):
        """
        dp_size > 1 场景下的失败检测：
        - dp_instance_id 直接用作映射 key（不做 // 10000 换算）
        - 验证 dp_size > 1 时 cluster_id 映射也能正确检测 pull_kv 失败
        """
        mock_metadata = Mock()
        mock_convert.return_value = mock_metadata

        mock_request = Mock(spec=ExecuteRequest)

        # dp_size > 1 时，cluster_id 传入的直接是 dp_instance_id
        mock_kv = Mock(
            dst_block_tables=[np.array([10], dtype=np.int64).tobytes()],
            src_block_tables=[np.array([1], dtype=np.int64).tobytes()],
            seq_group_metadata=Mock(request_id="req_500", prompt_lens=struct.pack("<q", 100)),
            cluster_id="5"
        )
        mock_request.pull_kv_request = Mock(pull_kv_infos=[mock_kv])

        self.router.block_size = 64
        self.router.dp_size = 2  # dp_size > 1
        self.router.config.p_inst_enable_sp_cp = False
        # dp_instance_id=5 对应的实际 cluster_ids
        self.router.config.dp_inst_id_to_cluster_id = {5: [5001, 5002]}
        self.mock_generator.kvcache_settings.num_npu_blocks = 10

        # pull_kv 失败，返回其中一个实际 cluster_id
        self.mock_generator.pull_kv.return_value = (
            ErrorCode.TEXT_GENERATOR_PD_PULL_KV_ERROR,
            5001
        )

        self.router.transfer_data(mock_request)

        mock_send.assert_called_once()
        proto_response = mock_send.call_args[0][0]
        # 顶层 status 恒为 SUCCESS；dp_size > 1 时通过映射正确检测失败，pd_error_code 为 PD_PULL_KV_ERROR
        self.assertEqual(proto_response.status, ModelWrapperErrorCode.SUCCESS.value)
        self.assertEqual(proto_response.pull_kv_response.pull_kv_results[0].request_id, "req_500")
        self.assertEqual(
            proto_response.pull_kv_response.pull_kv_results[0].pd_error_code,
            ModelWrapperErrorCode.PD_PULL_KV_ERROR.value
        )

    def test_initialize_distributed_mode(self):
        with patch('mindie_llm.connector.request_router.router_impl.set_npu_compile_mode') as _, \
                patch('mindie_llm.connector.request_router.router_impl.Generator') as mock_generator_cls:
            mock_config = Mock(spec=DmiConfig)
            mock_config.distributed_enable = True
            mock_config.rank = 2
            mock_config.tp_size = 4
            mock_config.dp_size = 2
            mock_config.max_seq_len = 1024
            mock_config.cache_block_size = 64
            mock_config.model_config = {"model_weight_path": "/test"}
            mock_config.local_rank = 0
            mock_config.npu_device_id = 0
            mock_generator = mock_generator_cls.return_value
            mock_generator.is_mix_model = False
            mock_generator.max_position_embeddings = 2048
            mock_mapping = Mock(has_dp=lambda: True)
            mock_generator.model_wrapper = Mock(mapping=mock_mapping)
            mock_generator.kvcache_settings = Mock(num_npu_blocks=20, num_cpu_blocks=5)

            result = self.router.initialize(mock_config)

            self.assertEqual(self.router.dp_rank_id, 0)
            self.assertEqual(result["kvCacheDescs"][0]["npuBlockNum"], 18)
            mock_generator_cls.assert_called_once()

    def test_check_output_invalid_eos_info(self):
        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=np.array([1, 2], dtype=np.uint32),
                eos_info=None,
                num_top_tokens=np.array([2], dtype=np.int32)
            )
            RouterImpl.check_output(mock_output)

        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=np.array([1, 2], dtype=np.uint32),
                eos_info=np.array([], dtype=np.uint32),
                num_top_tokens=np.array([2], dtype=np.int32)
            )
            RouterImpl.check_output(mock_output)

        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=np.array([1, 2], dtype=np.uint32),
                eos_info=np.array([-1, 0], dtype=np.int32),
                num_top_tokens=np.array([2], dtype=np.int32)
            )
            RouterImpl.check_output(mock_output)

        with self.assertRaises(ValueError):
            mock_output = Mock(
                token_ids=np.array([1, 2], dtype=np.uint32),
                eos_info=np.array([4294967296, 0], dtype=np.uint64),
                num_top_tokens=np.array([2], dtype=np.int32)
            )
            RouterImpl.check_output(mock_output)

    def test_check_output_shape_none(self):
        with self.assertRaises(ValueError):
            mock_output = Mock()
            mock_ids = Mock()
            mock_ids.shape = None
            mock_output.token_ids = mock_ids
            mock_output.eos_info = np.array([[0, 1]], dtype=np.uint32)
            mock_output.num_top_tokens = np.array([2], dtype=np.int32)
            RouterImpl.check_output(mock_output)

        with self.assertRaises(ValueError):
            mock_output = Mock()
            mock_output.token_ids = np.array([[1, 2]], dtype=np.uint32)
            mock_eos = Mock()
            mock_eos.shape = None
            mock_output.eos_info = mock_eos
            mock_output.num_top_tokens = np.array([2], dtype=np.int32)
            RouterImpl.check_output(mock_output)

    def test_handle_requests_mix_model(self):
        self.router.is_mix_model = True
        self.mock_metadata = Mock()
        self.mock_composite = Mock(spec=InputMetadataComposite)
        self.mock_composite.input_metadata = self.mock_metadata
        mock_num_top_tokens = np.array([3], dtype=np.int32)
        mock_output = Mock(
            token_ids=np.array([[1, 2, 3]], dtype=np.uint32),
            eos_info=np.array([[0, 0, 1]], dtype=np.uint32),
            num_top_tokens=mock_num_top_tokens
        )
        self.mock_generator.generate_token.return_value = mock_output

        result = self.router._handle_requests(self.mock_composite)

        self.mock_generator.generate_token.assert_called_once_with(self.mock_metadata)
        self.assertEqual(result, self.mock_generator.generate_token.return_value)

    def test_handle_requests_returns_none(self):
        self.mock_composite = Mock(spec=InputMetadataComposite)
        self.mock_composite.input_metadata = Mock()
        self.mock_generator.generate_token.return_value = None

        result = self.router._handle_requests(self.mock_composite)

        self.assertIsNone(result)

    @patch.object(RouterImpl, 'check_output')
    def test_handle_requests_inference_pause_skips_check_output(self, mock_check_output):
        """is_inference_pause 为 True 时应跳过 check_output"""
        self.router.is_inference_pause = True
        self.mock_composite = Mock(spec=InputMetadataComposite)
        self.mock_composite.input_metadata = Mock()
        mock_output = Mock(
            token_ids=np.array([[1, 2, 3]], dtype=np.uint32),
            eos_info=np.array([[0, 0, 1]], dtype=np.uint32),
            num_top_tokens=np.array([3], dtype=np.int32),
        )
        self.mock_generator.generate_token.return_value = mock_output

        result = self.router._handle_requests(self.mock_composite)

        mock_check_output.assert_not_called()
        self.assertIsNotNone(result)

    def test_prepare_kv_block_with_block_op(self):
        self.mock_composite = Mock(spec=InputMetadataComposite)
        self.mock_composite.block_op = [("swap", 1, 2)]

        self.router._prepare_kv_block(self.mock_composite)

        self.mock_generator.swap.assert_called_once_with([("swap", 1, 2)])

    def test_prepare_kv_block_without_block_op(self):
        """block_op 为 None 或空时，不应调用 swap"""
        self.mock_composite = Mock(spec=InputMetadataComposite)
        self.mock_composite.block_op = None

        self.router._prepare_kv_block(self.mock_composite)

        self.mock_generator.swap.assert_not_called()

    def test_seq_ctrl_layerwise_cleanup(self):
        self.router.layerwise_disaggregated = True
        self.router.generator.plugin_manager = Mock()
        mock_request = Mock(spec=ExecuteRequest)
        mock_cleanup_req = Mock(spec=TGCleanupRequest)
        mock_cleanup_req.seq_ids = [1, 2]
        mock_request.text_generator_cleanup_request = mock_cleanup_req
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP

        self.router.seq_ctrl(mock_request)

        self.mock_generator.plugin_manager.set_clean_sequence_ids.assert_called_once_with([1, 2])

    @patch('mindie_llm.connector.request_router.router_impl.convert_execute_model_request_to_input_metadata_composite')
    @patch.object(RouterImpl, '_handle_requests')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    def test_generate_other_rank(self, mock_send, mock_builder, mock_handle, mock_convert):
        self.router.local_rank = 3
        self.router.tp_size = 2
        self.router.config.distributed_enable = False
        self.mock_output = Mock(
            token_ids=np.array([[1, 2, 3]], dtype=np.uint32),
            eos_info=np.array([[0, 0, 1]], dtype=np.uint32),
            num_top_tokens=np.array([3], dtype=np.int32)
        )
        self.mock_composite = Mock()
        self.mock_composite.block_copy = None
        self.mock_request = Mock(spec=ExecuteRequest)
        self.mock_request.execute_model_request = Mock()
        self.mock_request.execute_type = ExecuteType.MODEL_INFER

        mock_convert.return_value = self.mock_composite
        mock_handle.return_value = self.mock_output
        mock_proto = Mock()
        mock_builder.build_from_generate_output.return_value = mock_proto

        self.router._generate(self.mock_request, is_prefill=False, is_mix=True)

        mock_builder.build_from_generate_output.assert_called_once_with(None, self.mock_request.execute_type)
        mock_send.assert_called_once_with(mock_proto)

    @patch('mindie_llm.connector.request_router.router_impl.convert_execute_model_request_to_input_metadata_composite')
    @patch.object(RouterImpl, '_handle_requests')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    def test_generate_err_msg_send(self, mock_send, mock_builder, mock_handle, mock_convert):
        """_handle_requests 抛出 ErrorCodeException 时应先发送 err_msg 再发送空响应"""
        mock_builder.build_from_err_msg.side_effect = lambda msg: RealExecuteResponseBuilder.build_from_err_msg(msg)

        self.router.local_rank = 0
        self.router.tp_size = 2
        self.router.config.distributed_enable = True
        self.router.layerwise_disaggregated = False

        mock_composite = Mock()
        mock_composite.block_copy = None
        mock_composite.input_metadata = Mock(all_sequence_ids=np.array([1]), batch_size=1)
        mock_convert.return_value = mock_composite

        mock_handle.side_effect = ErrorCodeException(ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY)

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_model_request = Mock()
        mock_request.execute_type = ExecuteType.MODEL_INFER

        self.router._generate(mock_request, is_prefill=False, is_mix=False)

        self.assertGreaterEqual(mock_send.call_count, 2)
        err_proto = mock_send.call_args_list[0][0][0]
        self.assertEqual(err_proto.msg_type, ExecuteType.EXECUTE_ERROR)

    @patch('mindie_llm.connector.request_router.router_impl.convert_execute_model_request_to_input_metadata_composite')
    @patch.object(RouterImpl, '_handle_requests')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    def test_generate_block_copy(self, mock_send, mock_builder, mock_handle, mock_convert):
        """input_metadata_composite.block_copy 有值时应调用 copy_blocks"""
        self.router.local_rank = 0
        self.router.tp_size = 2
        self.router.config.distributed_enable = True

        mock_output = Mock(
            token_ids=np.array([[1, 2, 3]], dtype=np.uint32),
            eos_info=np.array([[0, 0, 1]], dtype=np.uint32),
            num_top_tokens=np.array([3], dtype=np.int32),
            sequence_ids=np.array([1]),
        )
        mock_output.collate = Mock()

        mock_composite = Mock()
        mock_composite.block_copy = [[1, 2]]  # format: list of [src, dst] pairs
        mock_composite.input_metadata = Mock()
        mock_convert.return_value = mock_composite

        mock_handle.return_value = mock_output
        mock_builder.build_from_generate_output.return_value = Mock()

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_model_request = Mock()
        mock_request.execute_type = ExecuteType.MODEL_INFER

        self.router._generate(mock_request, is_prefill=False, is_mix=False)

        self.mock_generator.copy_blocks.assert_called_once()
        np.testing.assert_array_equal(
            self.mock_generator.copy_blocks.call_args[0][0],
            np.array([[1, 2]])
        )

    @patch('mindie_llm.connector.request_router.router_impl.convert_execute_model_request_to_input_metadata_composite')
    @patch.object(RouterImpl, '_handle_requests')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    def test_generate_rank(self, mock_send, mock_builder, mock_handle, mock_convert):
        self.router.local_rank = 2
        self.router.tp_size = 2
        self.router.config.distributed_enable = True
        self.mock_output = Mock(
            token_ids=np.array([[1, 2, 3]], dtype=np.uint32),
            eos_info=np.array([[0, 0, 1]], dtype=np.uint32),
            num_top_tokens=np.array([3], dtype=np.int32),
            sequence_ids=np.array([1, 2, 3], dtype=np.int32)
        )
        self.mock_composite = Mock()
        self.mock_composite.block_copy = None
        self.mock_request = Mock(spec=ExecuteRequest)
        self.mock_request.execute_model_request = Mock()
        self.mock_request.execute_type = ExecuteType.MODEL_INFER

        mock_convert.return_value = self.mock_composite
        mock_handle.return_value = self.mock_output
        mock_proto = Mock()
        mock_builder.build_from_generate_output.return_value = mock_proto

        self.router._generate(self.mock_request, is_prefill=False, is_mix=True)

        mock_builder.build_from_generate_output.assert_called_once_with(self.mock_output,
                                                                        self.mock_request.execute_type)
        mock_send.assert_called_once_with(mock_proto)

    @patch('mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication')
    def test_send_model_execute_response(self, mock_shared_mem):
        test_proto = ExecuteResponse(
            msg_type=1,
            status=0
        )
        send_model_execute_response(test_proto)
        mock_shared_mem.send_model_execute_response_cls.assert_called_once_with(test_proto)

    @patch('mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication')
    def test_send_transfer_response(self, mock_shared_mem):
        test_proto = ExecuteResponse(
            msg_type=1,
            status=0
        )
        send_transfer_response(test_proto)
        mock_shared_mem.send_transfer_response_cls.assert_called_once_with(test_proto)

    @patch('mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication')
    def test_send_command_response(self, mock_shared_mem):
        test_proto = ExecuteResponse(
            msg_type=1,
            status=0
        )
        send_command_response(test_proto)
        mock_shared_mem.send_command_response_cls.assert_called_once_with(test_proto)

    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.convert_execute_model_request_to_input_metadata_composite')
    def test_generate_layerwise_disaggregated(self, mock_metadata_composite, mock_response_builder, mock_send):
        self.router.layerwise_disaggregated = True
        mock_metadata_composite.side_effect = lambda *args, **kwargs: Mock(spec=InputMetadataComposite)
        execute_request = Mock(spec=ExecuteRequest)
        execute_request.execute_model_request = Mock()
        mock_response = Mock(spec=ExecuteResponse)
        mock_response_builder.build_from_transfer_result.return_value = mock_response
        self.router.generator.generate_token = Mock()
        self.router.generator.generate_token.return_value = None

        metadata = LwdMetadata(0, 0, False, 0, True, False, 0, 0, 1024)
        lwd_metadata_manager.set_metadata(metadata)
        self.router._generate(execute_request, False, False)
        mock_metadata_composite.assert_called_once()

        metadata = LwdMetadata(1, 1, False, 0, True, False, 0, 0, 1024)
        lwd_metadata_manager.set_metadata(metadata)
        self.router._generate(execute_request,False, False)

    @patch('mindie_llm.connector.request_router.router_impl.send_model_execute_response')
    @patch('mindie_llm.connector.request_router.router_impl.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.router_impl.convert_execute_model_request_to_input_metadata_composite')
    def test_mix(self, mock_metadata_composite, mock_response_builder, mock_send):
        execute_request = Mock(spec=ExecuteRequest)
        execute_request.execute_model_request = Mock()
        seq_group_metadata = Mock()
        execute_request.execute_model_request = Mock()
        mock_response = Mock(spec=ExecuteResponse)
        mock_response_builder.build_from_transfer_result.return_value = mock_response
        self.router.generator.generate_token = Mock()
        self.router.generator.generate_token.return_value = None

        seq_group_metadata.is_req_prefill = [True]
        execute_request.execute_model_request.seq_group_metadata_list = [seq_group_metadata]
        self.router._mix(execute_request)

        seq_group_metadata.is_req_prefill = [False]
        execute_request.execute_model_request.seq_group_metadata_list = [seq_group_metadata]
        self.router._mix(execute_request)

        seq_group_metadata.is_req_prefill = [True, False]
        execute_request.execute_model_request.seq_group_metadata_list = [seq_group_metadata]
        self.router._mix(execute_request)

if __name__ == '__main__':
    unittest.main()