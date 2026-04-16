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

from mindie_llm.connector.common.model_execute_data_pb2 import ExecuteType
from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteResponse,
    PullKVResponse,
)
from mindie_llm.connector.common.model_execute_data_pb2 import SequenceOutput
from mindie_llm.connector.common.model_execute_data_pb2 import (
    CompletionSequenceGroupOutput,
)
from mindie_llm.utils.prof.profiler import span_start, span_end, span_attr

DEFAULT_BLOCK_SIZE = 128

SUCCESS_STATUS = 0


class ExecuteResponseBuilder:
    EMPTY = ExecuteResponse()

    @staticmethod
    def build_from_init_result(init_results) -> ExecuteResponse:
        proto_response = ExecuteResponse(msg_type=ExecuteType.MODEL_INIT, status=SUCCESS_STATUS)
        kv_descs = init_results.get("kvCacheDescs")
        for key, value in init_results.items():
            if key == "kvCacheDescs":
                continue
            proto_response.init_results.init_result_map[key] = value

        if kv_descs:
            for desc in kv_descs:
                block_size = desc.get("blockSize", DEFAULT_BLOCK_SIZE)
                compression_ratio = desc.get("compressionRatio", 1)
                npu_block_num = desc.get("npuBlockNum")
                cache_type = desc.get("cacheType", 0)
                proto_response.init_results.kv_cache_descs.add(
                    block_size=int(block_size),
                    compression_ratio=int(compression_ratio),
                    npu_block_num=int(npu_block_num),
                    cache_type=int(cache_type),
                )
        return proto_response

    @staticmethod
    def build_from_generate_output_use_cpp(generate_output) -> bytes:
        prof = span_start("parse_generate_cpp", domain="connector")
        from _mindie_llm_connector import convert_generate_output

        proto_response_binary = convert_generate_output(generate_output)
        span_end(prof)
        return proto_response_binary

    @staticmethod
    def lwd_build_from_generate_output_use_cpp(generate_output, is_prefill) -> bytes:
        prof = span_start("parse_generate_cpp", domain="connector")
        from _mindie_llm_connector import lwd_convert_generate_output

        proto_response_binary = lwd_convert_generate_output(generate_output, is_prefill)
        span_end(prof)
        return proto_response_binary

    @staticmethod
    def build_from_err_msg(err_msg: str = "") -> ExecuteResponse:
        proto_response = ExecuteResponse(msg_type=ExecuteType.EXECUTE_ERROR)
        proto_response.execute_model_response.err_msg = err_msg if isinstance(err_msg, str) else ""
        return proto_response

    @staticmethod
    def build_from_generate_output(generate_output, event_type) -> ExecuteResponse:
        if generate_output is None:
            return ExecuteResponseBuilder.EMPTY

        prof = span_start("parse_generate", domain="connector")

        proto_response = ExecuteResponse(msg_type=event_type, status=SUCCESS_STATUS)

        for start_index, end_index in generate_output.group_indices:
            seq_ids = generate_output.sequence_ids[start_index:end_index].ravel()
            seq_count = len(seq_ids)
            parent_ids = generate_output.parent_sequence_ids[start_index:end_index].ravel()
            output_ids = generate_output.token_ids[start_index:end_index]
            log_probs = generate_output.logprobs[start_index:end_index]
            eos_attrs = generate_output.eos_info[start_index:end_index]
            truncations = generate_output.truncation_indices[start_index:end_index].ravel()
            cum_log_probs = generate_output.cumulative_logprobs[start_index:end_index].ravel()
            top_token_ids = generate_output.top_token_ids[start_index:end_index][
                :, :, : generate_output.num_top_tokens[start_index]
            ]
            top_log_probs = generate_output.top_logprobs[start_index:end_index][
                :, :, : generate_output.num_top_tokens[start_index]
            ]

            sequence_outputs = [
                SequenceOutput(
                    seq_id=seq_ids[i],
                    parent_seq_id=parent_ids[i],
                    output_token=output_ids[i].tolist(),
                    logprob=log_probs[i].tolist(),
                    finish_reason=eos_attrs[i][0].item(),
                    num_speculative_tokens=eos_attrs[i][1].item(),
                    truncation_index=truncations[i],
                    cumulative_logprobs=cum_log_probs[i],
                    num_parallel_tokens=len(output_ids[i]),
                    top_token_ids=top_token_ids[i].ravel().tolist(),
                    top_logprobs=top_log_probs[i].ravel().tolist(),
                )
                for i in range(seq_count)
            ]

            output = CompletionSequenceGroupOutput(samples=sequence_outputs)
            proto_response.execute_model_response.outputs.append(output)
        span_attr(prof, "size", len(generate_output.group_indices))
        span_end(prof)
        return proto_response

    @staticmethod
    def build_from_transfer_result(status, pull_kv_response_list: dict):
        # status 在 cpp 侧并没有使用
        proto_response = ExecuteResponse(
            msg_type=ExecuteType.KV_TRANSFER,
            status=status,
            pull_kv_response=PullKVResponse(),
        )
        for key, value in pull_kv_response_list.items():
            proto_response.pull_kv_response.pull_kv_results.append(
                PullKVResponse.PullKVResult(request_id=str(key), pd_error_code=value.value)
            )
        return proto_response

    @staticmethod
    def build_from_recover_command_result(responses_dict: str, command: str):
        msg_type = None
        if command == "CMD_PAUSE_ENGINE":
            msg_type = ExecuteType.PAUSE_COMMAND_EXEC
        elif command == "CMD_PAUSE_ENGINE_ROCE":
            msg_type = ExecuteType.PAUSE_COMMAND_EXEC_ROCE
        elif command == "CMD_CLEAR_TRANSER":
            msg_type = ExecuteType.CLEAR_COMMAND_EXEC
        elif command == "CMD_REINIT_NPU":
            msg_type = ExecuteType.RECOVER_COMMAND_EXEC
        elif command == "CMD_START_ENGINE":
            msg_type = ExecuteType.START_COMMAND_EXEC

        proto_response = ExecuteResponse(
            msg_type=msg_type,
            status=SUCCESS_STATUS,
        )
        proto_response.recover_command_response.npu_device_id = responses_dict["npu_device_id"]
        proto_response.recover_command_response.command_result = responses_dict["command_result"]
        proto_response.recover_command_response.error_msg = responses_dict["error_msg"]
        return proto_response
