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
from unittest.mock import patch
import numpy as np

from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteResponse,
    ExecuteType
)
from mindie_llm.connector.common.response_builder import ExecuteResponseBuilder, SUCCESS_STATUS
from mindie_llm.model_wrapper.utils.error import ModelWrapperErrorCode
from mindie_llm.text_generator.utils.generation_output import GenerationOutput


class TestExecuteResponseBuilder(unittest.TestCase):

    def test_empty_constant(self):
        self.assertIsInstance(ExecuteResponseBuilder.EMPTY, ExecuteResponse)
        self.assertEqual(ExecuteResponseBuilder.EMPTY.status, 0)

    def test_build_from_init_result_normal(self):
        init_results = {"param1": "value1", "param2": "value2"}
        response = ExecuteResponseBuilder.build_from_init_result(init_results)

        self.assertEqual(response.msg_type, ExecuteType.MODEL_INIT)
        self.assertEqual(response.status, SUCCESS_STATUS)
        self.assertEqual(len(response.init_results.init_result_map), 2)
        self.assertEqual(response.init_results.init_result_map["param1"], "value1")
        self.assertEqual(response.init_results.init_result_map["param2"], "value2")

    def test_build_from_init_result_empty(self):
        response = ExecuteResponseBuilder.build_from_init_result({})

        self.assertEqual(response.msg_type, ExecuteType.MODEL_INIT)
        self.assertEqual(response.status, SUCCESS_STATUS)
        self.assertEqual(len(response.init_results.init_result_map), 0)

    def test_build_from_init_result_with_kv_cache_descs(self):
        init_results = {
            "param1": "value1",
            "kvCacheDescs": [
                {"npuBlockNum": "16", "blockSize": 256, "compressionRatio": 2, "cacheType": 1},
                {"npuBlockNum": 8},  # exercise default values
            ],
        }
        response = ExecuteResponseBuilder.build_from_init_result(init_results)

        self.assertEqual(response.msg_type, ExecuteType.MODEL_INIT)
        self.assertEqual(response.status, SUCCESS_STATUS)
        self.assertEqual(response.init_results.init_result_map["param1"], "value1")
        self.assertNotIn("kvCacheDescs", response.init_results.init_result_map)

        self.assertEqual(len(response.init_results.kv_cache_descs), 2)
        d0 = response.init_results.kv_cache_descs[0]
        self.assertEqual(d0.npu_block_num, 16)
        self.assertEqual(d0.block_size, 256)
        self.assertEqual(d0.compression_ratio, 2)
        self.assertEqual(d0.cache_type, 1)

        d1 = response.init_results.kv_cache_descs[1]
        self.assertEqual(d1.npu_block_num, 8)
        self.assertEqual(d1.block_size, 128)
        self.assertEqual(d1.compression_ratio, 1)
        self.assertEqual(d1.cache_type, 0)

    @patch('mindie_llm.connector.common.response_builder.span_start')
    @patch('mindie_llm.connector.common.response_builder.span_end')
    @patch('mindie_llm.connector.common.response_builder.span_attr')
    def test_build_from_generate_output_none(self, mock_span_attr, mock_span_end, mock_span_start):
        response = ExecuteResponseBuilder.build_from_generate_output(None, ExecuteType.MODEL_INFER)

        self.assertEqual(response, ExecuteResponseBuilder.EMPTY)
        mock_span_start.assert_not_called()
        mock_span_attr.assert_not_called()
        mock_span_end.assert_not_called()

    def test_build_from_generate_output_use_cpp(self):
        generation_output = GenerationOutput(
            sequence_ids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
            parent_sequence_ids=np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
            group_indices=[(0, 2), (2, 4), (4, 6)],
            token_ids=np.array([np.array([100, 200]), np.array([101, 201]), np.array([102, 202]), np.array([103, 203]),
                                np.array([104, 204]), np.array([105, 205])]),
            logprobs=np.array([np.array([-0.1, -0.1], dtype=np.float32), np.array([-0.2, -0.2], dtype=np.float32),
                               np.array([-0.3, -0.3], dtype=np.float32), np.array([-0.4, -0.4], dtype=np.float32),
                               np.array([-0.5, -0.5], dtype=np.float32),
                               np.array([-0.6, -0.6], dtype=np.float32)], dtype=np.float32),
            eos_info=np.array([np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                               np.array([0, 1])]),
            top_token_ids=np.array(
                [[[100, 200]], [[101, 201]], [[102, 202]], [[103, 203]], [[104, 204]], [[105, 205]]]),
            top_logprobs=np.array(
                [[[-0.1, -0.1]], [[-0.2, -0.2]], [[-0.3, -0.3]], [[-0.1, -0.1]], [[-0.2, -0.2]], [[-0.3, -0.3]]],
                dtype=np.float32),
            num_new_tokens=np.array([1, 1, 1, 1, 1, 1], dtype=np.int32),
            num_top_tokens=np.array([2, 2, 2, 2, 2, 2], dtype=np.int32),
            cumulative_logprobs=np.array([-0.1, -0.2, -0.3, -0.1, -0.2, -0.3], dtype=np.float32),
            finish_reason=np.array([1, 1, 1, 1, 1, 1]),
            truncation_indices=np.array([1, 1, 1, 1, 1, 1]),
            current_token_indices=[1, 1, 1, 1, 1, 1],
            trace_ids=np.array([0, 1, 2])
        )
        response_byte_use_cpp = ExecuteResponseBuilder.build_from_generate_output_use_cpp(generation_output)
        self.assertGreater(len(response_byte_use_cpp), 0)
        execute_response = ExecuteResponse()
        execute_response.ParseFromString(response_byte_use_cpp)
        execute_response2 = ExecuteResponseBuilder.build_from_generate_output(generation_output, 1)
        self.assertEqual(execute_response, execute_response2)
        response_byte_use_python = execute_response2.SerializeToString()
        self.assertEqual(response_byte_use_cpp, response_byte_use_python)

    def test_lwd_build_from_generate_output_use_cpp(self):
        generation_output = GenerationOutput(
            sequence_ids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
            parent_sequence_ids=np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
            group_indices=[(0, 2), (2, 4), (4, 6)],
            token_ids=np.array([np.array([100, 200]), np.array([101, 201]), np.array([102, 202]), np.array([103, 203]),
                                np.array([104, 204]), np.array([105, 205])]),
            logprobs=np.array([np.array([-0.1, -0.1], dtype=np.float32), np.array([-0.2, -0.2], dtype=np.float32),
                               np.array([-0.3, -0.3], dtype=np.float32), np.array([-0.4, -0.4], dtype=np.float32),
                               np.array([-0.5, -0.5], dtype=np.float32),
                               np.array([-0.6, -0.6], dtype=np.float32)], dtype=np.float32),
            eos_info=np.array([np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1]),
                               np.array([0, 1])]),
            top_token_ids=np.array(
                [[[100, 200]], [[101, 201]], [[102, 202]], [[103, 203]], [[104, 204]], [[105, 205]]]),
            top_logprobs=np.array(
                [[[-0.1, -0.1]], [[-0.2, -0.2]], [[-0.3, -0.3]], [[-0.1, -0.1]], [[-0.2, -0.2]], [[-0.3, -0.3]]],
                dtype=np.float32),
            num_new_tokens=np.array([1, 1, 1, 1, 1, 1], dtype=np.int32),
            num_top_tokens=np.array([2, 2, 2, 2, 2, 2], dtype=np.int32),
            cumulative_logprobs=np.array([-0.1, -0.2, -0.3, -0.1, -0.2, -0.3], dtype=np.float32),
            finish_reason=np.array([1, 1, 1, 1, 1, 1]),
            truncation_indices=np.array([1, 1, 1, 1, 1, 1]),
            current_token_indices=[1, 1, 1, 1, 1, 1],
            trace_ids=np.array([0, 1, 2])
        )
        is_prefill = True
        response_byte_use_cpp = ExecuteResponseBuilder.lwd_build_from_generate_output_use_cpp(generation_output, is_prefill)
        self.assertGreater(len(response_byte_use_cpp), 0)
        execute_response = ExecuteResponse()
        execute_response.ParseFromString(response_byte_use_cpp)
        execute_response2 = ExecuteResponseBuilder.build_from_generate_output(generation_output, 1)
        execute_response2.execute_model_response.layerwise_is_prefill = is_prefill
        self.assertEqual(execute_response, execute_response2)
        response_byte_use_python = execute_response2.SerializeToString()
        self.assertEqual(response_byte_use_cpp, response_byte_use_python)

    @patch('mindie_llm.connector.common.response_builder.span_start')
    @patch('mindie_llm.connector.common.response_builder.span_end')
    @patch('mindie_llm.connector.common.response_builder.span_attr')
    def test_build_from_generate_output_single_group(self, mock_span_attr, mock_span_end, mock_span_start):
        class MockGenerateOutput:
            def __init__(self):
                self.group_indices = [(0, 2)]
                self.sequence_ids = np.array([[100, 101]])
                self.parent_sequence_ids = np.array([[200, 201]])
                self.token_ids = np.array([[1, 2], [3, 4]])
                self.logprobs = np.array([[0.1, 0.2], [0.3, 0.4]])
                self.eos_info = np.array([(1, 0), (0, 1)],
                                         dtype=[('a', int), ('b', int)])
                self.truncation_indices = np.array([5, 6])
                self.cumulative_logprobs = np.array([1.0, 2.0])
                self.top_token_ids = np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]])
                self.top_logprobs = np.array([[[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])
                self.num_top_tokens = np.array([2, 2])

        generate_output = MockGenerateOutput()
        event_type = ExecuteType.MODEL_INFER
        response = ExecuteResponseBuilder.build_from_generate_output(generate_output, event_type)

        self.assertEqual(response.msg_type, event_type)
        self.assertEqual(response.status, SUCCESS_STATUS)
        self.assertEqual(len(response.execute_model_response.outputs), 1)

        group_output = response.execute_model_response.outputs[0]
        self.assertEqual(len(group_output.samples), 2)

        seq1 = group_output.samples[0]
        self.assertEqual(seq1.seq_id, 100)
        self.assertEqual(seq1.parent_seq_id, 200)
        self.assertEqual(seq1.output_token, [1, 2])
        expected_logprob = [0.1, 0.2]
        self.assertEqual(len(seq1.logprob), len(expected_logprob))
        for actual, expected in zip(seq1.logprob, expected_logprob):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertEqual(seq1.finish_reason, 1)
        self.assertEqual(seq1.num_speculative_tokens, 0)
        self.assertEqual(seq1.truncation_index, 5)
        self.assertEqual(seq1.cumulative_logprobs, 1.0)
        self.assertEqual(seq1.num_parallel_tokens, 2)
        self.assertEqual(seq1.top_token_ids, [5, 6, 7, 8])
        expected_top_logprobs = [0.5, 0.6, 0.7, 0.8]
        self.assertEqual(len(seq1.top_logprobs), len(expected_top_logprobs))
        for actual, expected in zip(seq1.top_logprobs, expected_top_logprobs):
            self.assertAlmostEqual(actual, expected, places=4)

        seq2 = group_output.samples[1]
        self.assertEqual(seq2.seq_id, 101)
        self.assertEqual(seq2.parent_seq_id, 201)
        self.assertEqual(seq2.output_token, [3, 4])
        expected_logprob2 = [0.3, 0.4]
        self.assertEqual(len(seq2.logprob), len(expected_logprob2))
        for actual, expected in zip(seq2.logprob, expected_logprob2):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertEqual(seq2.finish_reason, 0)
        self.assertEqual(seq2.num_speculative_tokens, 1)
        self.assertEqual(seq2.truncation_index, 6)
        self.assertEqual(seq2.cumulative_logprobs, 2.0)
        self.assertEqual(seq2.num_parallel_tokens, 2)
        self.assertEqual(seq2.top_token_ids, [9, 10, 11, 12])
        expected_top_logprobs2 = [0.9, 1.0, 1.1, 1.2]
        self.assertEqual(len(seq2.top_logprobs), len(expected_top_logprobs2))
        for actual, expected in zip(seq2.top_logprobs, expected_top_logprobs2):
            self.assertAlmostEqual(actual, expected, places=4)

        mock_span_start.assert_called_once_with("parse_generate", domain="connector")
        mock_span_attr.assert_called_once_with(mock_span_start.return_value, "size", 1)
        mock_span_end.assert_called_once_with(mock_span_start.return_value)

    def test_build_from_generate_output_multi_groups(self):
        class MockGenerateOutput:
            def __init__(self):
                self.group_indices = [(0, 1), (1, 3)]
                self.sequence_ids = np.array([[500], [501], [502]])
                self.parent_sequence_ids = np.array([[600], [601], [602]])
                self.token_ids = np.array([[10], [11], [12]])
                self.logprobs = np.array([[0.01], [0.02], [0.03]])
                self.eos_info = np.array([(2, 3), (4, 5), (6, 7)], dtype=[('a', int), ('b', int)])
                self.truncation_indices = np.array([100, 101, 102])
                self.cumulative_logprobs = np.array([3.0, 4.0, 5.0])
                self.top_token_ids = np.array([[[1]], [[2]], [[3]]])
                self.top_logprobs = np.array([[[0.1]], [[0.2]], [[0.3]]])
                self.num_top_tokens = np.array([1, 1, 1])

        generate_output = MockGenerateOutput()
        response = ExecuteResponseBuilder.build_from_generate_output(generate_output, ExecuteType.MODEL_INFER)

        self.assertEqual(len(response.execute_model_response.outputs), 2)
        self.assertEqual(len(response.execute_model_response.outputs[0].samples), 1)
        self.assertEqual(len(response.execute_model_response.outputs[1].samples), 2)

    def test_build_from_transfer_result_normal(self):
        pull_kv_response_list = {
            "req1": ModelWrapperErrorCode.SUCCESS,
            "req2": ModelWrapperErrorCode.PD_PULL_KV_ERROR
        }
        status = 200
        response = ExecuteResponseBuilder.build_from_transfer_result(status, pull_kv_response_list)

        self.assertEqual(response.msg_type, ExecuteType.KV_TRANSFER)
        self.assertEqual(response.status, status)
        self.assertEqual(len(response.pull_kv_response.pull_kv_results), 2)

        result1 = response.pull_kv_response.pull_kv_results[0]
        self.assertEqual(result1.request_id, "req1")
        self.assertEqual(result1.pd_error_code, ModelWrapperErrorCode.SUCCESS.value)

        result2 = response.pull_kv_response.pull_kv_results[1]
        self.assertEqual(result2.request_id, "req2")
        self.assertEqual(result2.pd_error_code, ModelWrapperErrorCode.PD_PULL_KV_ERROR.value)

    def test_build_from_transfer_result_empty(self):
        response = ExecuteResponseBuilder.build_from_transfer_result(500, {})

        self.assertEqual(response.msg_type, ExecuteType.KV_TRANSFER)
        self.assertEqual(response.status, 500)
        self.assertEqual(len(response.pull_kv_response.pull_kv_results), 0)

    def test_build_from_err_msg(self):
        err_msg = "Test error message"
        response = ExecuteResponseBuilder.build_from_err_msg(err_msg)

        self.assertEqual(response.msg_type, ExecuteType.EXECUTE_ERROR)
        self.assertTrue(response.HasField("execute_model_response"))
        self.assertEqual(response.execute_model_response.err_msg, err_msg)

    def test_build_from_err_msg_empty(self):
        response = ExecuteResponseBuilder.build_from_err_msg("")

        self.assertEqual(response.msg_type, ExecuteType.EXECUTE_ERROR)
        self.assertEqual(response.execute_model_response.err_msg, "")

    def test_build_from_err_msg_non_str(self):
        """err_msg 非字符串时，应使用空字符串"""
        response = ExecuteResponseBuilder.build_from_err_msg(123)
        self.assertEqual(response.msg_type, ExecuteType.EXECUTE_ERROR)
        self.assertEqual(response.execute_model_response.err_msg, "")

        response = ExecuteResponseBuilder.build_from_err_msg(None)
        self.assertEqual(response.msg_type, ExecuteType.EXECUTE_ERROR)
        self.assertEqual(response.execute_model_response.err_msg, "")

    def test_build_from_recover_command_result_pause(self):
        responses_dict = {
            "npu_device_id": 0,
            "command_result": 0,
            "error_msg": "",
        }
        response = ExecuteResponseBuilder.build_from_recover_command_result(
            responses_dict, "CMD_PAUSE_ENGINE"
        )
        self.assertEqual(response.msg_type, ExecuteType.PAUSE_COMMAND_EXEC)
        self.assertEqual(response.status, SUCCESS_STATUS)
        self.assertEqual(response.recover_command_response.npu_device_id, 0)
        self.assertEqual(response.recover_command_response.command_result, 0)
        self.assertEqual(response.recover_command_response.error_msg, "")

    def test_build_from_recover_command_result_clear(self):
        responses_dict = {
            "npu_device_id": 1,
            "command_result": 0,
            "error_msg": "no error",
        }
        response = ExecuteResponseBuilder.build_from_recover_command_result(
            responses_dict, "CMD_CLEAR_TRANSER"
        )
        self.assertEqual(response.msg_type, ExecuteType.CLEAR_COMMAND_EXEC)
        self.assertEqual(response.recover_command_response.npu_device_id, 1)
        self.assertEqual(response.recover_command_response.command_result, 0)

    def test_build_from_recover_command_result_reinit(self):
        responses_dict = {
            "npu_device_id": 2,
            "command_result": 0,
            "error_msg": "",
        }
        response = ExecuteResponseBuilder.build_from_recover_command_result(
            responses_dict, "CMD_REINIT_NPU"
        )
        self.assertEqual(response.msg_type, ExecuteType.RECOVER_COMMAND_EXEC)
        self.assertEqual(response.recover_command_response.npu_device_id, 2)

    def test_build_from_recover_command_result_start(self):
        responses_dict = {
            "npu_device_id": 3,
            "command_result": 0,
            "error_msg": "",
        }
        response = ExecuteResponseBuilder.build_from_recover_command_result(
            responses_dict, "CMD_START_ENGINE"
        )
        self.assertEqual(response.msg_type, ExecuteType.START_COMMAND_EXEC)
        self.assertEqual(response.recover_command_response.npu_device_id, 3)


if __name__ == '__main__':
    unittest.main()