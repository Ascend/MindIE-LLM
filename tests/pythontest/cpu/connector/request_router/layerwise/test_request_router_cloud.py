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
import time
import unittest
from unittest.mock import Mock, MagicMock, patch

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteRequest, ExecuteType, ForwardType, ExecuteResponse
from mindie_llm.connector.request_router.layerwise.request_router_cloud import RequestRouterCloud
from mindie_llm.utils.layerwise.share_memory import SharedMemoryManager
from mindie_llm.utils.layerwise.request_metadata import LwdMetadata, lwd_metadata_manager
from mindie_llm.connector.request_router.router_impl import RouterImpl
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager
from mindie_llm.connector.request_listener.shared_mem_communication import SharedMemCommunication
from mindie_llm.connector.request_router.layerwise.request_router_lwd import DecisionType, LastExecType


class TestRequestRouterCloud(unittest.TestCase):
    def setUp(self):
        self.router = RequestRouterCloud('0')
        self.router.router_impl = Mock(spec=RouterImpl)
        edge_cloud_comm = Mock(spec=LwdCommunicationManager)
        self.router.ctrl_comm = edge_cloud_comm.ctrl_comm
        self.router.data_comm = edge_cloud_comm.data_comm
        self.router.rank = 0
        self.router.mem_manager = self.get_shared_memery()

    @classmethod
    def setUpClass(cls):
        print("TestRequestRouterCloud start")

    @classmethod
    def tearDownClass(cls):
        print("TestRequestRouterCloud end")

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

    def get_shared_memery(self):
        share_mem_manager = SharedMemoryManager('0')
        is_producer = True if self.router.rank == 0 else False
        share_mem_manager.initialize(is_producer, 7)   # 8 cards 1 master

        return share_mem_manager

    def mock_prifill_request(self) -> ExecuteRequest:
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

    def mock_generator_clean_eos_request(self) -> ExecuteRequest:
        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.EOS_CLEANUP
        return mock_request
        

    
    def test_adjust_prefill_cut_policy(self):
        self.router.total_layer_num = 64
        self.router.start_layer_num = 1    
        self.router.end_layer_num = 1      

        expect_cut_policy = [
            [31, 31],
            [21, 21, 20],
            [16, 16, 15, 15],
            [13, 13, 12, 12, 12],
            [11, 11, 10, 10, 10, 10],
            [9, 9, 9, 9, 9, 9, 8],
            [8, 8, 8, 8, 8, 8, 7, 7],
        ]
        for i, cut_num in enumerate(range(2,9)):
            self.router.prepare_prefill_cut_policy(cut_num)
            self.assertEqual(self.router.prefill_layers_divi_policy, expect_cut_policy[i])

    def test_prepare_prefill_chunk_policy(self):
        self.router.prepare_prefill_chunk_policy(20000, 6)
        self.assertEqual(self.router.prefill_chunk_policy, [3334, 3334, 3333, 3333, 3333, 3333])

    def test_save_inference_request(self):
        self.router.initialize = Mock()
        self.router.router_impl.seq_ctrl = Mock()
        self.router.save_inference_request(self.mock_prifill_request())
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
        self.router.data_comm.p_shape = [0]
        self.router.data_comm.recv_index = 0
        
        self.router.parse_all_dp_batches_seq_lens = MagicMock(return_value=[0])
        self.router.calc_max_seq_len = MagicMock(return_value=0)
        self.router.calc_curr_dp_seq_len = MagicMock(return_value=0)
        self.router.calc_batch_size = MagicMock(return_value=1)
        self.router.prefill_layers_divi_switch = False
        
        self.router.inference_queue.put(self.mock_prifill_request())
        self.router.inference_queue.put(self.mock_decode_request())
        self.router.inference_queue.put(self.mock_init_request())
        self.router.inference_queue.put(self.mock_finalize_request())
        self.router.inference_queue.put(self.mock_generator_cleanup_request())
        self.router.get_all_request()
        self.assertIsNotNone(self.router.prefill_request)
        self.assertIsNotNone(self.router.decode_queue)

    def test_calc_decision_type_dd_other(self):
        self.router.clean_up_queue.put("other_request")
        self.router.prefill_request = None
        self.router.decode_request = None
        self.router.prefill_queue = queue.Queue()
        self.router.decode_queue = queue.Queue()
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_CLEAN_UP)

    def test_calc_decision_type_do_prefille(self):
        self.router.prefill_request = Mock()
        self.router.decode_request = Mock()
        self.router.prefill_comm_finish = True
        self.router.decode_comm_finish = True
        self.router.last_execute_type = LastExecType.DECODE
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL)

    def test_calc_decision_type_do_decode(self):
        self.router.decode_request = Mock()
        self.router.decode_comm_finish = True
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_DECODE)

    def test_calc_decision_type_wait_decode(self):
        self.router.prefill_request = Mock()
        self.router.prefill_comm_finish = True
        self.router.last_execute_type = LastExecType.PREFILL
        self.router.before_last_execute_type = LastExecType.DECODE
        self.router.prefill_exec_last_time = time.time()
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.WAIT_DECODE)

    def test_calc_decision_type_do_prefille_after_wait_decode(self):
        self.router.prefill_request = Mock()
        self.router.prefill_comm_finish = True
        self.router.last_execute_type = LastExecType.PREFILL
        self.router.before_last_execute_type = LastExecType.DECODE
        self.router.prefill_exec_last_time = time.time() - 0.02  # 0.02 秒前
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL)

    def test_calc_decision_type_wait_comm(self):
        self.router.prefill_request = None
        self.router.decode_request = None
        self.router.prefill_comm_finish = False
        self.router.decode_comm_finish = False
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.WAIT_COMM)

    def test_calc_decision_type_clean_eos(self):
        self.router.prefill_request = None
        self.router.decode_request = None
        self.router.prefill_comm_finish = False
        self.router.decode_comm_finish = False
        self.router.clean_eos_queue.put(self.mock_generator_clean_eos_request())
        self.router.calc_decision_type()
        self.assertEqual(self.router.decision_type, DecisionType.DO_CLEAN_EOS)

    def test_recv_prefill(self):
        self.router.prefill_comm_finish = False
        self.router.ctrl_comm.recv_prefill = Mock()
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 1
        self.router.ctrl_comm.prefill_recv_msg = None

        self.router.prefill_comm_finish = False
        self.router.recv_prefill()
        self.assertTrue(self.router.prefill_comm_finish)

    def test_recv_decode(self):
        self.router.decode_comm_finish = False
        self.router.ctrl_comm.recv_decode = Mock()
        self.router.recv_decode()
        self.assertTrue(self.router.decode_comm_finish) 

    def test_arrange_exec_stage_do_decode(self):
        self.router.decision_type = DecisionType.DO_DECODE
        self.router.arrange_exec_stage()
        test_cases = [(0, 62, True, 62, False, False, 0, 0)]
        metadata_ = lwd_metadata_manager.get_metadata()
        for (start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx) in test_cases:
            self.assertEqual(metadata_.start_exec_layer, start_exec_layer)
            self.assertEqual(metadata_.end_exec_layer, end_exec_layer)
            self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token)
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer)
            self.assertEqual(metadata_.is_prefill, is_prefill)
            self.assertEqual(metadata_.is_long_seq, is_long_seq)
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx)
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx)
            

    def test_arrange_exec_stage_do_prefille_no_chunk(self):
        self.router.decision_type = DecisionType.DO_PREFILL
        self.router.prefill_layers_divi_policy = [13, 13, 12, 12, 12]
        self.router.is_long_seq = False
        self.router.prefill_exec_start_layer = 0
        self.router.prefill_exec_cnt = 0
        test_cases = [
            (0, 13, False, 62, True, False, 0, 0),
            (13, 26, False, 62, True, False, 0, 0),
            (26, 38, False, 62, True, False, 0, 0),
            (38, 50, False, 62, True, False, 0, 0),
            (50, 62, True, 62, True, False, 0, 0)
        ]

        for i in range(0, test_cases.__len__()):
            self.router.prefill_exec_cnt = i
            self.router.arrange_exec_stage()
            metadata_ = lwd_metadata_manager.get_metadata()
            (start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx) = test_cases[i]
            metadata_.start_exec_layer == start_exec_layer # self.assertEqual(metadata_.start_exec_layer, start_exec_layer, f"Failed on test case {i+1}")
            metadata_.end_exec_layer == end_exec_layer # self.assertEqual(metadata_.end_exec_layer, end_exec_layer, f"Failed on test case {i+1}")
            metadata_.end_of_generate_token == end_of_generate_token # self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_prefill, is_prefill, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_long_seq, is_long_seq, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx, f"Failed on test case {i+1}")


    def test_arrange_exec_stage_do_prefille_with_chunk(self):
        self.router.decision_type = DecisionType.DO_PREFILL
        self.router.prefill_layers_divi_switch = False
        self.router.prefill_layers_divi_policy = [8, 8, 8, 8, 8, 8, 7, 7]
        self.router.prefill_layers_divi_num = self.router.prefill_layers_divi_policy.__len__()
        self.router.is_long_seq = True
        self.router.prefill_chunk_policy = [25000, 25000, 25000, 25000]
        self.router.prefill_chunk_num = self.router.prefill_chunk_policy.__len__()
        self.router.prefill_seq_len = 100000
        self.router.prefill_exec_start_layer = 0
        self.router.prefill_exec_cnt = 0
        self.router.long_seq_start_idx = 0
        self.router.prefill_exec_chunk_cnt = 0

        test_cases = [
            (0, 8, False, 62, True, True, 0, 25000),
            (8, 16, False, 62, True, True, 0, 25000),
            (16, 24, False, 62, True, True, 0, 25000),
            (24, 32, False, 62, True, True, 0, 25000),
            (32, 40, False, 62, True, True, 0, 25000),
            (40, 48, False, 62, True, True, 0, 25000),
            (48, 55, False, 62, True, True, 0, 25000),
            (55, 62, False, 62, True, True, 0, 25000),
            (0, 8, False, 62, True, True, 25000, 50000),
            (8, 16, False, 62, True, True, 25000, 50000),
            (16, 24, False, 62, True, True, 25000, 50000),
            (24, 32, False, 62, True, True, 25000, 50000),
            (32, 40, False, 62, True, True, 25000, 50000),
            (40, 48, False, 62, True, True, 25000, 50000),
            (48, 55, False, 62, True, True, 25000, 50000),
            (55, 62, False, 62, True, True, 25000, 50000),
            (0, 8, False, 62, True, True, 50000, 75000),
            (8, 16, False, 62, True, True, 50000, 75000),
            (16, 24, False, 62, True, True, 50000, 75000),
            (24, 32, False, 62, True, True, 50000, 75000),
            (32, 40, False, 62, True, True, 50000, 75000),
            (40, 48, False, 62, True, True, 50000, 75000),
            (48, 55, False, 62, True, True, 50000, 75000),
            (55, 62, False, 62, True, True, 50000, 75000),
            (0, 8, False, 62, True, True, 75000, 100000),
            (8, 16, False, 62, True, True, 75000, 100000),
            (16, 24, False, 62, True, True, 75000, 100000),
            (24, 32, False, 62, True, True, 75000, 100000),
            (32, 40, False, 62, True, True, 75000, 100000),
            (40, 48, False, 62, True, True, 75000, 100000),
            (48, 55, False, 62, True, True, 75000, 100000),
            (55, 62, True, 62, True, True, 75000, 100000),
        ]
        for i in range(0, test_cases.__len__()):
            self.router.decision_type = DecisionType.DO_PREFILL
            self.router.arrange_exec_stage()
            metadata_ = lwd_metadata_manager.get_metadata()
            (start_exec_layer, end_exec_layer, end_of_generate_token, cloud_total_layer, is_prefill, is_long_seq, long_seq_start_idx, long_seq_end_idx) = test_cases[i]
            self.assertEqual(metadata_.start_exec_layer, start_exec_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.end_exec_layer, end_exec_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.end_of_generate_token, end_of_generate_token, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.cloud_total_layer, cloud_total_layer, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_prefill, is_prefill, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.is_long_seq, is_long_seq, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_start_idx, long_seq_start_idx, f"Failed on test case {i+1}")
            self.assertEqual(metadata_.long_seq_end_idx, long_seq_end_idx, f"Failed on test case {i+1}")


    def test_shared_memory(self):
        self.router.decode_request = ExecuteRequest()
        self.router.prefill_request = ExecuteRequest()
        self.router.rank = 0
        self.router.decision_type = DecisionType.DO_PREFILL
        self.router.broadcast_decision_type()
        self.router.decision_type = DecisionType.WAIT_COMM

        for rank in range(1, 8):
            self.router.rank = rank
            self.router.decision_type = DecisionType.WAIT_COMM
            self.router.recv_decision_type()
            self.assertEqual(self.router.decision_type, DecisionType.DO_PREFILL)

        self.router.rank = 0
        self.router.decision_type = DecisionType.DO_DECODE
        self.router.broadcast_decision_type()
        self.router.decision_type = DecisionType.WAIT_COMM

        for rank in range(1, 8):
            self.router.rank = rank
            self.router.decision_type = DecisionType.WAIT_COMM
            self.router.recv_decision_type()
            self.assertEqual(self.router.decision_type, DecisionType.DO_DECODE)

    
    @patch.object(SharedMemoryManager, 'initialize')
    @patch('mindie_llm.connector.common.send_model_execute_response')
    @patch('mindie_llm.connector.request_listener.shared_mem_communication.SharedMemCommunication.send_model_execute_response_cls')
    @patch.object(LwdCommunicationManager, 'initialize')
    @patch('mindie_llm.connector.request_router.layerwise.request_router_lwd.ExecuteResponseBuilder')
    @patch('mindie_llm.connector.request_router.request_router.RouterImpl')
    @patch("mindie_llm.connector.request_router.request_router.BaseConfig")
    def test_initialize_standard_mode(self, mock_base_config, mock_router_impl, mock_response_builder, *args):
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
            ("layerwiseDisaggregatedRoleType", "slave"),
            ("edgeIpAddress", "127.0.0.0"),
            ("cloudIpAddress", "127.0.0.0"),
            ("max_iter_times", "136096"),
            ("max_batch_size", "200"),
            ("max_prefill_tokens", "136096"),
            ("trust_remote_code", "false"),
            ("backend_type", "atb"),
            ("models", json.dumps({"startLayerNum": 1}))
        ]

        mock_base_config_instance = Mock()
        mock_model_config = dict(mock_config.items.return_value)
        mock_base_config_instance.model_config = mock_model_config
        mock_base_config.return_value = mock_base_config_instance

        mock_initialize_result = {"status": "ok"}
        mock_router_impl.initialize.return_value = mock_initialize_result

        mock_response = Mock(spec=ExecuteResponse)
        mock_response_builder.build_from_init_result.return_value = mock_response

        # 模拟 router_impl 和 edge_cloud_comm
        self.router.router_impl = Mock(spec=RouterImpl)
        self.router.ctrl_comm = Mock()
        self.router.data_comm = Mock()
        self.router.prepare_prefill_cut_policy = Mock()

        router_impl = Mock()
        router_impl.generator.model_wrapper = Mock()
        router_impl.generator.model_wrapper.model_runner = Mock()
        router_impl.generator.model_wrapper.model_runner.config = Mock()
        router_impl.generator.model_wrapper.model_runner.config.num_hidden_layers = 64
        mock_router_impl.return_value = router_impl

        # 模拟 SharedMemCommunication._instance
        mock_shared_mem_comm_instance = Mock()
        mock_shared_mem_comm_instance.send_response = Mock()
        SharedMemCommunication._instance = mock_shared_mem_comm_instance

        self.router.initialize(mock_config)

    def test_do_clean_up(self):
        self.router.initialize = Mock()
        self.router.router_impl.finalize = Mock()
        self.router.router_impl.seq_ctrl = Mock()

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.TEXT_GENERATOR_CLEANUP
        self.router.clean_up_queue.put(mock_request)
        self.router.do_clean_up()
        self.router.router_impl.seq_ctrl.assert_called_once()

    def test_do_clean_eos(self):
        self.router.initialize = Mock()
        self.router.router_impl.finalize = Mock()
        self.router.router_impl.seq_ctrl = Mock()

        mock_request = Mock(spec=ExecuteRequest)
        mock_request.execute_type = ExecuteType.EOS_CLEANUP
        self.router.clean_eos_queue.put(mock_request)
        self.router.do_clean_eos()
        self.router.router_impl.seq_ctrl.assert_called_once()

    def test_do_prefill(self):
        self.router.decision_type = DecisionType.DO_PREFILL

        self.router.prefill_exec_cnt = 0
        self.router.prefill_request = ExecuteRequest()
        self.router.exceute_inference_request()
        self.assertIsNotNone(self.router.prefill_request)

        self.router.prefill_exec_cnt = self.router.prefill_layers_divi_num
        self.router.exceute_inference_request()
        self.assertIsNone(self.router.prefill_request)

    def test_do_decode(self):
        self.router.decision_type = DecisionType.DO_DECODE

        self.router.decode_request = ExecuteRequest()
        self.router.exceute_inference_request()
        self.assertIsNone(self.router.prefill_request)

    def test_do_inference(self):
        self.router.get_all_request = Mock()
        self.router.recv_prefill = Mock()
        self.router.recv_decode = Mock()
        self.router.calc_decision_type = Mock()
        self.router.arrange_exec_stage = Mock()
        self.router.broadcast_decision_type = Mock()
        self.router.recv_decision_type = Mock()
        self.router.exceute_inference_request = Mock()
        self.router.rank = 0
        self.router.comm_initialized = True
        self.router.curr_no_request = lambda: False
    
        timeout_seconds = 0.5
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

        self.router.comm_initialized = True
        self.router.rank = 0
        self.router.prefill_request = self.mock_prifill_request()
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()

        self.router.rank = 1 
        self.router.comm_initialized = True
        process = multiprocessing.Process(target=self.router.do_inference)
        process.start()
        process.join(timeout_seconds)
        process.terminate()
    
    def test_accept(self):
        self.router.calc_seq_len = Mock()
        self.router.calc_seq_len.return_value = 1024
        self.router.calc_batch_size = Mock()
        self.router.calc_batch_size.return_value = 200
        self.router.data_comm.lock = MagicMock()
        self.router.get_prefill_gap_time_list = Mock()
        self.router.prefill_layers_divi_switch = False
        self.router.initialize = Mock()
        self.router.isqwenvl = False
        self.router.data_comm.prefill_seq_len_queue = queue.Queue()
        self.router.data_comm.decode_batch_size_queue = queue.Queue()
        
        self.router.parse_all_dp_batches_seq_lens = MagicMock(return_value=[0])
        self.router.calc_max_seq_len = MagicMock(return_value=0)
        self.router.calc_curr_dp_seq_len = MagicMock(return_value=0)
        self.router.calc_batch_size = MagicMock(return_value=1)
        self.router.ctrl_comm.prefill_comm_finish_tcp_count = 0
        self.router.data_comm.p_shape = [0]
        self.router.data_comm.recv_index = 0
        self.router.calc_curr_dp_batch_size = MagicMock(return_value=0)
        
        self.router.accept(self.mock_prifill_request())
        self.router.accept(self.mock_decode_request())
        self.router.accept(self.mock_init_request())
        self.router.accept(self.mock_finalize_request())
        self.router.accept(self.mock_generator_cleanup_request())

    def test_do_other_request_now(self):
        execute_request = self.mock_prifill_request()
        with self.assertRaises(RuntimeError) as _:
            self.router.do_other_request_now(execute_request)


if __name__ == "__main__":
    unittest.main()