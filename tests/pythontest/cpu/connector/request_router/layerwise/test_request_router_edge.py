# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import multiprocessing
import queue
import json
import unittest

from unittest.mock import MagicMock, Mock, patch

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType
from mindie_llm.connector.request_router.layerwise.request_router_edge import RequestRouterEdge, DecisionType
from mindie_llm.utils.layerwise.share_memory import SharedMemoryManager
from mindie_llm.connector.request_router.router_impl import RouterImpl
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager
from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.utils.layerwise.input_metadata import EdgeCloudInputMetadata, pd_exec_matadata_instance
from mindie_llm.connector.common.input_metadata_composite import InputMetadataComposite
from mindie_llm.connector.request_router.layerwise.request_router_lwd import LONG_SEQ_LEN_MIN, DecisionType


class TestRequestRouterEdge(unittest.TestCase):
    def setUp(self):
        self.router = RequestRouterEdge('0')
        self.router.router_impl = Mock()
        self.router.ctrl_comm = self.router.router_impl.ctrl_comm
        self.router.rank = 0
        self.router.mem_manager = self.get_shared_memery()

    @classmethod
    def setUpClass(cls):
        print("TestRequestRouterEdge start")

    @classmethod
    def tearDownClass(cls):
        print("TestRequestRouterEdge end")

    def test_init(self):
        self.assertIsInstance(self.router.inference_queue, queue.Queue)
        self.assertIsInstance(self.router.transfer_queue, queue.Queue)
        self.assertIsInstance(self.router.link_queue, queue.Queue)
        self.assertIsInstance(self.router.prefill_queue, queue.Queue)
        self.assertIsInstance(self.router.decode_queue, queue.Queue)
        self.assertIsInstance(self.router.clean_up_queue, queue.Queue)
        self.assertTrue(self.router.inference_related_thread.is_alive())
        self.assertTrue(self.router.trans_related_thread.is_alive())
        self.assertIsNone(self.router.enable_dp_distributed)

    @patch('mindie_llm.connector.common.send_model_execute_response')
    @patch('mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication.send_model_execute_response_cls')
    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    @patch('mindie_llm.connector.request_router.request_router.RouterImpl')
    @patch.object(LwdCommunicationManager, 'initialize')
    @patch.object(SharedMemoryManager, 'initialize')
    def test_initialize_standard_mode(self, mock_mem, mock_edge_cloud, mock_router_impl, mock_base_config, *args):
        mock_config = Mock()
        mock_config.items.return_value = [
            ("infer_mode", "standard"),
            ("cpu_mem", "2048"),
            ("model_path", "/path/to/standard/model"),
            ("device", "npu"),
            ("distributed_enable", True),
            ("local_rank", "0"),
            ("model_id", "0"),
            ("rank", "0"),
            ("world_size", "2"),
            ("npu_device_id", "0"),
            ("npu_device_ids", "0,1"),
            ("cpu_mem", "0"),
            ("npu_mem", "-1"),
            ("max_seq_len", "133000"),
            ("max_iter_times", "136096"),
            ("max_prefill_tokens", "133000"),
            ("max_input_len", "133000"),
            ("block_size", "128"),
            ("layerwiseDisaggregated", "true"),
            ("layerwiseDisaggregatedRoleType", "master"),
            ("edgeIpAddress", "127.0.0.0"),
            ("cloudIpAddress", "127.0.0.0"),
            ("max_iter_times", "136096"),
            ("max_batch_size", "200"),
            ("trust_remote_code", "false"),
            ("backend_type", "atb"),
            ("models", json.dumps({"startLayerNum": 1}))
        ]

        mock_base_config_instance = Mock()
        mock_base_config_instance.dp_size = 1
        mock_base_config_instance.cp_size = 1
        mock_base_config_instance.max_seq_len = 133000
        mock_base_config_instance.max_prefill_tokens = 133000
        mock_model_config = dict(mock_config.items.return_value)
        mock_base_config_instance.model_config = mock_model_config
        mock_base_config.return_value = mock_base_config_instance

        router_impl = Mock()
        router_impl.initialize.return_value = {"status": "ok"}
        router_impl.generator.model_wrapper = Mock()
        router_impl.generator.model_wrapper.model_runner = Mock()
        router_impl.generator.model_wrapper.model_runner.config = Mock()
        router_impl.generator.model_wrapper.model_runner.config.num_hidden_layers = 64
        mock_router_impl.return_value = router_impl

        # 模拟 router_impl 和 edge_cloud_comm
        self.router.router_impl = Mock(spec=RouterImpl)
        self.router.ctrl_comm = Mock()
        self.router.data_comm = Mock()

        # 模拟 SharedMemCommunication._instance
        mock_shared_mem_comm_instance = Mock()
        mock_shared_mem_comm_instance.send_response = Mock()
        SharedMemCommunication._instance = mock_shared_mem_comm_instance

        self.router.initialize(mock_config)


    def test_calc_decision_type(self):
        test_cases = [
          
            (0.005, None, None, None, ExecuteRequest(), True, None, DecisionType.DO_DECODE_FIRST),
            (None, ExecuteRequest(), True, None, None, None, None, DecisionType.DO_PREFILL_FIRST),
            (None, ExecuteRequest(), False, True, None, None, None, DecisionType.DO_PREFILL_LAST),
            (None, ExecuteRequest(), False, False, None, None, None, DecisionType.WAIT_COMM),     # P尾通讯未完成，循环等待
            (None, None, None, None, ExecuteRequest(), True, None, DecisionType.DO_DECODE_FIRST),
            (None, None, None, None, ExecuteRequest(), False, True, DecisionType.DO_DECODE_LAST),
            (None, None, None, None, ExecuteRequest(), False, False, DecisionType.WAIT_COMM),     # D尾通讯未完成，循环等待
            (None, None, None, None, None, None, None, DecisionType.WAIT_COMM),     # 没有任务，阻塞等待
            
        ]
        for i, (force_wait_d_time, prefill_execute_request, prefill_first, Prefill_comm_finish, decode_execute_request, decode_first, Decode_comm_finish, expected_decision_type) in enumerate(test_cases):
            with self.subTest(f"Test Case {i+1}"):
                self.router.force_wait_d_time = force_wait_d_time
                self.router.prefill_request = prefill_execute_request
                self.router.prefill_first = prefill_first
                self.router.prefill_comm_finish = Prefill_comm_finish
                self.router.prefill_last_request = prefill_execute_request
                self.router.decode_request = decode_execute_request
                self.router.decode_first = decode_first
                self.router.decode_first = decode_first
                self.router.decode_comm_finish = Decode_comm_finish
                
                self.router.calc_decision_type()
                self.assertEqual(self.router.decision_type, expected_decision_type, f"Failed on test case {i+1}")

    def test_do_decode_first(self):
        self.router.decode_request = ExecuteRequest()
        self.router.decode_first = True
        func = self.router.process_func.get(DecisionType.DO_DECODE_FIRST)
        func()

        self.assertFalse(self.router.decode_first)
        self.assertIsNotNone(self.router.decode_request)
    
    def test_do_decode_last(self):
        self.router.decode_request = ExecuteRequest()
        self.router.decode_first = False
        self.router.decode_comm_finish = True
        self.router.ctrl_comm.decode_comm_finish = True
        func = self.router.process_func.get(DecisionType.DO_DECODE_LAST)
        func()

        self.assertIsNone(self.router.decode_request)
        self.assertFalse(self.router.decode_comm_finish)
        self.assertFalse(self.router.ctrl_comm.decode_comm_finish)

    def test_do_prefill_first(self):
        self.router.prefill_request = ExecuteRequest()
        self.router.prefill_first = True
        func = self.router.process_func.get(DecisionType.DO_PREFILL_FIRST)
        func()

        self.assertFalse(self.router.prefill_first)
        self.assertIsNone(self.router.prefill_request)

    def test_do_prefill_last(self):
        self.router.prefill_last_request = ExecuteRequest()
        self.router.prefill_first = False
        self.router.prefill_comm_finish = True
        func = self.router.process_func.get(DecisionType.DO_PREFILL_LAST)
        func()

        self.assertIsNone(self.router.prefill_last_request)
        self.assertFalse(self.router.prefill_comm_finish)


    def mock_prefill_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_execute_model_request = Mock()
        mock_execute_model_request.forward_type = ForwardType.PREFILL
        mock_request.execute_model_request = mock_execute_model_request
        return mock_request
    
    def mock_decode_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INFER
        mock_execute_model_request = Mock()
        mock_execute_model_request.forward_type = ForwardType.DECODE
        mock_request.execute_model_request = mock_execute_model_request
        return mock_request
    
    def mock_init_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_INIT
        mock_request.config = True
        return mock_request
    
    def mock_finalize_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.MODEL_FINALIZE
        return mock_request

    def mock_generator_cleanup_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP
        return mock_request

    def test_do_clean_up(self):
        self.router.initialize = Mock()
        self.router.router_impl.finalize = Mock()
        self.router.router_impl.seq_ctrl = Mock()

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP
        self.router.clean_up_queue.put(mock_request)
        self.router.do_clean_up()
        self.router.router_impl.seq_ctrl.assert_called_once()

    def test_recv_prefill(self):
        self.router.ctrl_comm.recv_prefill = Mock()
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 1
        self.router.ctrl_comm.prefill_recv_msg = None

        self.router.prefill_comm_finish = False
        self.router.recv_prefill()
        self.assertTrue(self.router.prefill_comm_finish)
    
    def test_recv_decode(self):
        self.router.decode_comm_finish = False
        self.router.ctrl_comm = Mock()
        self.router.ctrl_comm.recv_decode = Mock()
        self.router.recv_decode()
        self.assertTrue(self.router.decode_comm_finish)

    @patch('time.time')
    def test_check_10ms_for_next_decode(self, mock_time):
        self.router.force_wait_d_time = None
        self.router.check_10ms_for_next_decode()
        self.assertEqual(mock_time.call_count, 0)

        self.router.force_wait_d_time = 1
        mock_time.return_value = 1.0078125
        self.router.check_10ms_for_next_decode()
        self.assertEqual(self.router.wait_d_time_gap, 0.0078125)

        self.router.force_wait_d_time = 0
        mock_time.return_value = 0.12
        self.router.check_10ms_for_next_decode()
        self.assertIsNone(self.router.force_wait_d_time)

    def test_arrange_exec_stage(self):
        test_cases = [
            (0, 0, False, 0, False, False, 0, 0, DecisionType.DO_DECODE_FIRST),
            (1, 1, True, 0, False, False, 0, 0, DecisionType.DO_DECODE_LAST), 
            (0, 0, False, 0, True, False, 0, 0, DecisionType.DO_PREFILL_FIRST), 
            (1, 1, True, 0, True, False, 0, 0, DecisionType.DO_PREFILL_LAST)
        ]
        self.router.decode_request = ExecuteRequest()
        self.router.prefill_request = ExecuteRequest()
        self.router.prefill_last_request = ExecuteRequest()
        for start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx, decision_type in test_cases:
            self.router.decision_type = decision_type
            self.router.arrange_exec_stage()
            metadata_ = lwd_metadata_manager.get_metadata()
            self.assertEqual(metadata_.start_exec_layer, start_exec_layer)
            self.assertEqual(metadata_.end_exec_layer, end_exec_layer)
            self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token)
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer)
            self.assertEqual(metadata_.is_prefill, is_prefill)
            self.assertEqual(metadata_.is_long_seq, is_long_seq)
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx)
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx)


    def test_recv_decision_type(self):
        self.router.decode_request = ExecuteRequest()
        self.router.prefill_request = ExecuteRequest()
        self.router.prefill_last_request = ExecuteRequest()
        self.router.rank = 0
        self.router.decision_type = DecisionType.DO_DECODE_FIRST
        self.router.broadcast_decision_type()
        self.router.decision_type = DecisionType.WAIT_COMM

        self.router.rank = 1
        self.router.recv_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_DECODE_FIRST)

        self.router.rank = 0
        self.router.decision_type = DecisionType.DO_PREFILL_LAST
        self.router.broadcast_decision_type()
        self.router.decision_type = DecisionType.WAIT_COMM

        self.router.rank = 1
        self.router.recv_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)


    def test_exceute_inference_request(self):
        self.router.decision_type = DecisionType.WAIT_COMM
        self.router.exceute_inference_request()
        self.router.decision_type = DecisionType.DO_PREFILL_FIRST
        self.router.prefill_request = ExecuteRequest()
        self.router.exceute_inference_request()
        self.assertFalse(self.router.prefill_first)
        self.assertIsNone(self.router.prefill_request)


    def test_save_inference_request(self):
        self.router.initialize = Mock()
        self.router.router_impl.seq_ctrl = Mock()
        self.router.save_inference_request(self.mock_prefill_request())
        self.router.save_inference_request(self.mock_decode_request())
        self.router.save_inference_request(self.mock_init_request())
        self.router.save_inference_request(self.mock_finalize_request())
        self.router.save_inference_request(self.mock_generator_cleanup_request())
        self.assertEqual(self.router.prefill_queue.qsize(), 1)
        self.assertEqual(self.router.decode_queue.qsize(), 1)
        self.assertEqual(self.router.clean_up_queue.qsize(), 1)
    
    def test_get_all_request(self):
        self.router.initialize = Mock()
        self.router.router_impl.seq_ctrl = Mock()
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 0
        self.router.parse_all_dp_batches_seq_lens = MagicMock(return_value=[0])
        self.router.calc_max_seq_len = MagicMock(return_value=0)
        self.router.calc_curr_dp_seq_len = MagicMock(return_value=0)
        self.router.calc_batch_size = MagicMock(return_value=1)
        
        self.router.inference_queue.put(self.mock_prefill_request())
        self.router.inference_queue.put(self.mock_decode_request())
        self.router.inference_queue.put(self.mock_init_request())
        self.router.inference_queue.put(self.mock_finalize_request())
        self.router.inference_queue.put(self.mock_generator_cleanup_request())
        
        self.router.get_all_request()
        self.assertTrue(self.router.prefill_first)
        self.assertTrue(self.router.decode_first)

    def test_do_inference(self):
        self.router.get_all_request = Mock()
        self.router.recv_prefill = Mock()
        self.router.recv_decode = Mock()
        self.router.check_10ms_for_next_decode = Mock()
        self.router.calc_decision_type = Mock()
        self.router.arrange_exec_stage = Mock()
        self.router.broadcast_decision_type = Mock()
        self.router.recv_decision_type = Mock()
        self.router.exceute_inference_request = Mock()
        self.router.rank = 0

        timeout_seconds = 0.5

        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

        self.router.comm_initialized = True
        self.router.prefill_request = self.mock_prefill_request()
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

        self.router.rank = 1 
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

    def get_shared_memery(self):
        share_mem_manager = SharedMemoryManager('0')
        is_producer = True if self.router.rank == 0 else False
        share_mem_manager.initialize(is_producer, 1)   # 2 cards 1 master

        return share_mem_manager
    
    def test_prefill_chunk(self):
        self.router.calc_curr_dp_seq_len = Mock(return_value=18000)
        self.router.prefill_chunk_instance = Mock()
        self.router.prefill_chunk_instance.map_prefill_chunk_num = Mock()
        self.router.prefill_chunk_instance.map_prefill_chunk_num.return_value = 6
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 0
        self.router.calc_max_seq_len = MagicMock(return_value=18000)
        self.router.inference_queue.put(self.mock_prefill_request())
        self.router.wait_prefill_last = False
        self.router.calc_batch_size = Mock(return_value=1)
        self.router.get_all_request()
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_FIRST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(0, 0, False, True, False, False, 0, True, 0, 3000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_FIRST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(0, 0, False, True, False, False, 0, True, 3000, 6000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.prefill_last_request = self.mock_prefill_request()
        self.router.prefill_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(1, 1, False, True, False, False, 0, True, 0, 3000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_FIRST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(0, 0, False, True, False, False, 0, True, 6000, 9000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.prefill_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(1, 1, False, True, False, False, 0, True, 3000, 6000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_FIRST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(0, 0, False, True, False, False, 0, True, 9000, 12000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.prefill_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(1, 1, False, True, False, False, 0, True, 6000, 9000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_FIRST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(0, 0, False, True, False, False, 0, True, 12000, 15000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.prefill_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(1, 1, False, True, False, False, 0, True, 9000, 12000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_FIRST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(0, 0, False, True, False, False, 0, True, 15000, 18000, 0, 18000, True))
        self.router.exceute_inference_request()
        
        self.router.prefill_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(1, 1, False, True, False, False, 0, True, 12000, 15000, 0, 18000, False))
        self.router.exceute_inference_request()
        
        self.router.prefill_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL_LAST)
        self.router.arrange_exec_stage()
        self.assertEqual(lwd_metadata_manager.get_metadata(), LwdMetadata(1, 1, True, True, False, False, 0, True, 15000, 18000, 0, 18000, True))
        self.router.exceute_inference_request()
        
    def test_is_request_long_seq(self):
        prefill_request = self.mock_prefill_request()
        seq_lens = MagicMock()
        seq_lens.seq_lens = [LONG_SEQ_LEN_MIN+1]
        prefill_request.execute_model_request.all_dp_batches_seq_lens = [seq_lens]
        prefill_request.execute_model_request.seq_group_metadata_list = [100]
        
        is_long_seq = self.router.is_request_long_seq(prefill_request)
        self.assertTrue(is_long_seq)
        
    @patch("mindie_llm.connector.common.input_metadata_builder.parse_all_dp_batches_seq_lens")
    def test_calc_curr_dp_seq_len(self, mock_parse):
        self.router.router_impl.generator = Mock()
        self.router.router_impl.generator.plugin_manager = Mock()
        self.router.router_impl.generator.plugin_manager.model_wrapper = Mock()
        self.router.router_impl.generator.plugin_manager.model_wrapper.mapping = Mock()
        self.router.router_impl.generator.plugin_manager.model_wrapper.mapping.attn_dp = Mock()
        self.router.router_impl.generator.plugin_manager.model_wrapper.mapping.attn_dp.rank = 0
        
        mock_parse.return_value = [[10, 20], [30, 40]]
        execute_request = Mock()
        execute_request.execute_model_request.seq_group_metadata_list = [
            Mock(dp_rank_id=0),
            Mock(dp_rank_id=1)                                                             
            ]
        seq_lens = MagicMock()
        seq_lens.seq_lens = [[10, 20]]
        execute_request.execute_model_request.all_dp_batches_seq_lens = [seq_lens]

        result = self.router.calc_curr_dp_seq_len(execute_request)
        self.assertEqual(result, 30)
      
  
class TestEdgeCloudInputMetadata(unittest.TestCase):
    def setUp(self):
        self.pd_exec_matadata = pd_exec_matadata_instance

    @classmethod
    def setUpClass(cls):
        print("TestEdgeCloudInputMetadata start")

    @classmethod
    def tearDownClass(cls):
        print("TestEdgeCloudInputMetadata end")

    def test_have_input_metadata(self):
        test_cases = [
            LwdMetadata(1, 1, True, False, False, False, 0, False, 0, 0, 0, 1024),        # D尾
            LwdMetadata(1, 1, True, True, False, False, 0, False, 0, 0, 0, 1024),         # P尾
            LwdMetadata(8, 16, False, True, False, False, 62, False, 0, 0, 0, 1024),      # 云侧P非首块
            LwdMetadata(0, 1, False, True, False, False, 62, True, 2500, 5000, 0, 1024)   # 云侧P非首块chunk
        ]
        for metadata in test_cases:
            lwd_metadata_manager.set_metadata(metadata)
            layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
            self.assertTrue(EdgeCloudInputMetadata.have_input_metadata(layerwise_disaggregated_exe_stage))

        test_cases = [
            LwdMetadata(0, 0, False, False, False, False, 0, False, 0, 0, 0, 1024),         # D首
            LwdMetadata(0, 0, False, True, False, False, 0, False, 0, 0, 0, 1024),        # P首
            LwdMetadata(0, 8, False, True, False, False, 62, False, 0, 0, 0, 1024),       # 云侧P首块
            LwdMetadata(0, 1, False, True, False, False, 62, True, 0, 5000, 0, 1024),      # 云侧P首块chunk
            LwdMetadata(0, 62, True, False, False, False, 62, False, 0, 0, 0, 1024)         # 云侧D
        ]
        for metadata in test_cases:
            lwd_metadata_manager.set_metadata(metadata)
            layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
            self.assertFalse(EdgeCloudInputMetadata.have_input_metadata(layerwise_disaggregated_exe_stage))

    def test_need_storage_input_metadata(self):
        test_cases = [
            LwdMetadata(0, 0, False, False, False, False, 0, False, 0, 0, 0, 1024),         # D首
            LwdMetadata(0, 0, False, 0, False, False, False, True, 0, 0, 0, 1024),        # P首
            LwdMetadata(0, 8, False, True, False, False, 62, False, 0, 0, 0, 1024),       # 云侧P首块
            LwdMetadata(0, 1, False, True, False, False, 62, True, 0, 5000, 0, 1024)      # 云侧P首块chunk
        ]
        for metadata in test_cases:
            lwd_metadata_manager.set_metadata(metadata)
            layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
            self.assertTrue(EdgeCloudInputMetadata.need_storage_input_metadata(layerwise_disaggregated_exe_stage))

        test_cases = [
            LwdMetadata(1, 1, True, False, False, False, 0, False, 0, 0, 0, 1024),          # D尾
            LwdMetadata(1, 1, True, True, False, False, 0, False, 0, 0, 0, 1024),         # P尾
            LwdMetadata(8, 16, False, True, False, False, 62, False, 0, 0, 0, 1024),      # 云侧P非首块
            LwdMetadata(0, 1, False, True, False, False, 62, True, 2500, 5000, 0, 1024),   # 云侧P非首块chunk
            LwdMetadata(0, 62, True, False, False, False, 62, False, 0, 0, 0, 1024)         # 云侧D
        ]
        for metadata in test_cases:
            lwd_metadata_manager.set_metadata(metadata)
            layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
            self.assertFalse(EdgeCloudInputMetadata.need_storage_input_metadata(layerwise_disaggregated_exe_stage))

    def test_set_and_get_metadata(self):
        prefill_metadata = InputMetadataComposite()
        decode_metadata = InputMetadataComposite()
        metadata = LwdMetadata(1, 1, True, True, False, False, 0, False, 0, 0, 0, 1024)
        lwd_metadata_manager.set_metadata(metadata)
        layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
        self.pd_exec_matadata.set_input_metadata(prefill_metadata, True)
        self.pd_exec_matadata.set_input_metadata(decode_metadata, False)
        self.assertEqual(prefill_metadata, self.pd_exec_matadata.get_input_metadata(True, layerwise_disaggregated_exe_stage))
        metadata = LwdMetadata(1, 1, True, False, False, False, 0, False, 0, 0, 0, 1024)
        lwd_metadata_manager.set_metadata(metadata)
        layerwise_disaggregated_exe_stage = lwd_metadata_manager.get_metadata()
        self.assertEqual(decode_metadata, self.pd_exec_matadata.get_input_metadata(False, layerwise_disaggregated_exe_stage))

if __name__ == "__main__":
    unittest.main()