# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import struct
import base64
import json
from dataclasses import dataclass
from typing import List

import numpy as np

from mindie_llm.connector.common.model_execute_data_pb2 import (
    ExecuteModelRequest,
    PDLinkRequest,
    ForwardType,
    SequenceGroupMetadata,
    PDRole,
    PullKVRequest
)

from mindie_llm.connector.common.input_metadata_composite import InputMetadataComposite
from mindie_llm.model_wrapper.utils.common_util import split_list_equally, ip_string_to_list

from mindie_llm.text_generator.utils.input_metadata import InputMetadata, SIMULATE_SEQUENCE_ID
from mindie_llm.utils.prof.profiler import span_start, span_end, span_attr
from ...utils.log.logging import logger

REPETITION_PENALTY_INDEX = 0  # item 0 is used for repetition penalty in ndarray
FREQUENCY_PENALTY_INDEX = 1  # item 1 is used for frequency penalty in ndarray
PRESENCE_PENALTY_INDEX = 2  # item 2 is used for presence penalty in ndarray
TEMPERATURE_INDEX = 3  # item 3 is used for temperature in ndarray
TOP_K_INDEX = 4  # item 4 is used for top_k in ndarray
TOP_P_INDEX = 5  # item 5 is used for top_p in ndarray
SAMPLING_INDEX = 6
TOP_LOGPROBS_INDEX = 7  # item 7 is used for top_p in ndarray

SAMPLING_PARAMS_DTYPE = "f4"
SAMPLING_DTYPE = [
    ("repetition_penalty", SAMPLING_PARAMS_DTYPE),
    ("frequency_penalty", SAMPLING_PARAMS_DTYPE),
    ("presence_penalty", SAMPLING_PARAMS_DTYPE),
    ("temperature", SAMPLING_PARAMS_DTYPE),
    ("top_k", SAMPLING_PARAMS_DTYPE),
    ("top_p", SAMPLING_PARAMS_DTYPE),
    ("do_sample", SAMPLING_PARAMS_DTYPE),
    ("top_logprobs", SAMPLING_PARAMS_DTYPE),
]

PLACEHOLDER_TOKEN = -1


@dataclass(slots=True)
class ConvertPara:
    is_prefill: bool = False
    is_mix: bool = False


def convert_bytes_to_list(byte_data):
    if len(byte_data) & 0b111 != 0:
        raise ValueError("The length of 'byte_data' must be a multiple of 8")
    return list(struct.unpack(f"<{len(byte_data) // 8}q", byte_data)) if byte_data else []


def parse_all_dp_batches_seq_lens(all_dp_batches_seq_lens):
    all_dp_seq_lens = []
    for dp_batch_seq_lens in all_dp_batches_seq_lens:
        all_dp_seq_lens.append(list(dp_batch_seq_lens.seq_lens))
    return all_dp_seq_lens


def parse_sampling_parameters(seq_group_metadata):
    sampling = np.array(
        [(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)],
        dtype=SAMPLING_DTYPE,
    )
    if seq_group_metadata.HasField("do_sample"):
        sampling[0][SAMPLING_INDEX] = 1.0 if seq_group_metadata.do_sample else 0.0

    if not seq_group_metadata.HasField("sampling_params"):
        return sampling

    sampling_params = seq_group_metadata.sampling_params

    if sampling_params.HasField("repetition_penalty"):
        sampling[0][REPETITION_PENALTY_INDEX] = sampling_params.repetition_penalty

    if sampling_params.HasField("frequency_penalty"):
        sampling[0][FREQUENCY_PENALTY_INDEX] = sampling_params.frequency_penalty

    if sampling_params.HasField("presence_penalty"):
        sampling[0][PRESENCE_PENALTY_INDEX] = sampling_params.presence_penalty

    if sampling_params.HasField("top_logprobs"):
        sampling[0][TOP_LOGPROBS_INDEX] = sampling_params.top_logprobs

    if sampling_params.HasField("temperature"):
        sampling[0][TEMPERATURE_INDEX] = sampling_params.temperature

    if sampling_params.HasField("top_k"):
        sampling[0][TOP_K_INDEX] = sampling_params.top_k

    if sampling_params.HasField("top_p"):
        sampling[0][TOP_P_INDEX] = sampling_params.top_p

    return sampling


def parse_swap_blocks(blocks_to_swap_in, blocks_to_swap_out):
    if not (blocks_to_swap_in or blocks_to_swap_out):
        return None

    block_op_list = []
    # 解析swapin的block_id，设置decision_type = 0, 满足Generator.swap方法格式要求
    for block_pair_swap_in in blocks_to_swap_in:
        block_op_list.append([0, block_pair_swap_in.num1, block_pair_swap_in.num2])
    # 解析swapout的block_id，设置decision_type = 1, 满足Generator.swap方法格式要求
    for block_pair_swap_out in blocks_to_swap_out:
        block_op_list.append([1, block_pair_swap_out.num1, block_pair_swap_out.num2])
    # Generator.swap方法取输入的第0维形成一个二重list，此处应组一个三维list
    return [block_op_list]


def generate_lora_strings(seq_group_metadata):
    lora_id: str = seq_group_metadata.lora_id
    return None if not lora_id or lora_id == "None" else lora_id


def get_batch_size(request, is_prefill, is_mix):
    if is_mix:
        is_req_prefill = []
        for seq_group_metadata in request.seq_group_metadata_list:
            is_req_prefill.extend(seq_group_metadata.is_req_prefill)
        total_bs = len(request.seq_group_metadata_list)
        prefill_bs = np.sum(is_req_prefill)
        decode_bs = total_bs - prefill_bs
    else:
        total_bs = len(request.seq_group_metadata_list)
        prefill_bs = 0
        decode_bs = 0
        if is_prefill:
            prefill_bs = total_bs
        else:
            decode_bs = total_bs
    return total_bs, prefill_bs, decode_bs


def make_dummy_input_metadata(execute_request, num_npu_blocks, model_config, lwd_exe_stage=None):
    block_padding = model_config.max_seq_len // model_config.cache_block_size
    block_id_for_empty_req = num_npu_blocks - 1
    sp_batch_tokens = None
    if model_config.p_inst_enable_sp_cp:
        sp_rank_tokens = [[1] + [0] * (model_config.sp_size - 1)]
        sp_batch_tokens = np.array(sp_rank_tokens)
        all_sp_block_tables = [
            [
                [block_id_for_empty_req] + [-1] * (block_padding - 1)
            ] + [
                [-1] * block_padding for _ in range(model_config.sp_size - 1)
            ]
        ]
        block_tables = np.array(all_sp_block_tables, dtype=np.int64)
    else:
        block_tables = np.array([[block_id_for_empty_req] + [-1] * (block_padding - 1)], dtype=np.int64)

    dp_rank_id = (model_config.rank // model_config.tp_size) % model_config.dp_size
    metadata = InputMetadata(
        batch_size=1,
        batch_request_ids=np.array([18446744073709551], dtype=np.int64),
        batch_max_output_lens=np.array([1], dtype=np.int64),
        block_tables=block_tables,
        max_block_size=model_config.cache_block_size,
        has_sampling=True,
        is_prefill=True,
        input_ids=np.array([30], dtype=np.int64),
        batch_seq_len=np.array([1], dtype=np.int64),
        total_seq_num=1,
        batch_sampling_params=np.array(
            [(np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan)],
            dtype=SAMPLING_DTYPE
        ),
        batch_stop_strings=[[]],
        batch_stop_token_ids=[None],
        computed_blocks=None,
        remote_computed_blocks=None,
        adapter_ids=[None],
        num_npu_blocks=num_npu_blocks,
        batch_dp_rank_ids=np.array([dp_rank_id], dtype=np.int64),
        batch_ignore_eos=np.array([None]),
        batch_skip_special_tokens=np.array([None]),
        batch_include_stop=np.array([False]),
        trace_ids=[None],
        batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64)],
        batch_best_of=np.array([1]),
        batch_logprobs=np.array([None]),
        batch_seeds=np.array([25]),
        batch_n=np.array([1]),
        batch_use_beam_search=np.array([False]),
        reserved_sequence_ids=[np.array([], dtype=np.int64)],
        is_dummy_batch=True,
        sp_tokens=sp_batch_tokens,
        layerwise_disaggregated_exe_stage=lwd_exe_stage
    )
    metadata.seq_lens = []
    for dp_batch_seq_lens in execute_request.execute_model_request.all_dp_batches_seq_lens:
        metadata.seq_lens.append(list(dp_batch_seq_lens.seq_lens))

    return metadata


def make_dummy_input_metadata_dmi_decoder(source_input_metadata, num_npu_blocks, model_config):
    block_padding = model_config.max_seq_len // model_config.cache_block_size
    block_id_for_empty_req = num_npu_blocks - 1
    metadata = InputMetadata(
        batch_size=1,
        batch_request_ids=np.array([18446744073709551], dtype=np.int64),
        batch_max_output_lens=np.array([1], dtype=np.int64),
        block_tables=np.array([[block_id_for_empty_req] +
                               [-1] * (block_padding - 1)], dtype=np.int64),
        max_block_size=model_config.cache_block_size,
        has_sampling=False,
        is_prefill=False,
        input_ids=np.array([], dtype=np.int64),
        batch_seq_len=np.array([1], dtype=np.int64),
        total_seq_num=1,
        batch_sampling_params=np.array([], dtype=np.float64),
        batch_stop_strings=[[]],
        batch_stop_token_ids=[],
        computed_blocks=None,
        remote_computed_blocks=None,
        adapter_ids=[None],
        num_npu_blocks=num_npu_blocks,
        batch_dp_rank_ids=source_input_metadata.batch_dp_rank_ids,
        batch_ignore_eos=np.array([]),
        batch_skip_special_tokens=np.array([]),
        batch_include_stop=np.array([]),
        trace_ids=None,
        batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64)],
        batch_best_of=np.array([1]),
        batch_logprobs=np.array([]),
        batch_seeds=np.array([]),
        batch_n=np.array([1]),
        batch_use_beam_search=np.array([False]),
        reserved_sequence_ids=[np.array([], dtype=np.int64)],
        is_dummy_batch=True
    )
    return metadata


def build_simulate_block_table(
        seq_group_metadata,
        block_id_for_simulate_req: int,
        is_sp_enable: bool,
        is_cp_enable: bool,
        config
) -> tuple:
    """
    为虚推请求构建 block table。
    
    虚推请求需要特殊处理，使用固定 block id 作为占位符。在 SP/CP 场景下，
    需要根据 sp_rank_block_num 构造正确长度的 simulate block table，
    以确保与其他请求组 batch 时 numpy 维度能够对齐。    
    """
    simulate_block_table = [block_id_for_simulate_req]
    sp_rank_block_num = None
    sp_rank_token_num = None
    
    if is_sp_enable or is_cp_enable:
        sp_rank_block_num = list(seq_group_metadata.sp_rank_block_num)
        sp_rank_token_num = list(seq_group_metadata.sp_rank_token_num)
        # 如果 sp_rank_block_num 或 sp_rank_token_num 为空，需要根据 config 填充默认值以确保维度正确
        if config is not None:
            scp_size = config.sp_size * config.cp_size
            if not sp_rank_block_num:
                sp_rank_block_num = [1] + [0] * (scp_size - 1)
            if not sp_rank_token_num:
                sp_rank_token_num = [1] + [0] * (scp_size - 1)
        total_blocks = sum(sp_rank_block_num)
        if total_blocks > 0:
            seq_blocks = [block_id_for_simulate_req] + [-1] * (total_blocks - 1)
        else:
            seq_blocks = simulate_block_table
    else:
        seq_blocks = simulate_block_table
    
    return seq_blocks, sp_rank_block_num, sp_rank_token_num


def convert_execute_model_request_to_input_metadata_composite(
        request: ExecuteModelRequest,
        num_npu_blocks,
        block_size,
        convert_para=ConvertPara(),
        is_mix_model=False,
        layerwise_disaggregated_exe_stage=None,
        config=None
) -> InputMetadataComposite:
    """
    将ExecuteModelRequest转换为InputMetadataComposite。
    """
    span = span_start("convert_request", domain="connector")
    is_prefill = convert_para.is_prefill or request.forward_type == ForwardType.PREFILL

    batch_input_ids = np.array([], dtype=np.int64)
    batch_sampling = []
    batch_stop_token_ids = []
    batch_stop_strings = []
    batch_ignore_eos = []
    batch_skip_special_tokens = []
    batch_include_stop = []
    adapter_ids = []
    batch_seq_ids = []
    batch_logprobs = []
    batch_best_of = []
    batch_n = []
    batch_reserved_seq_ids = []
    batch_use_beam_search = []
    batch_req_ids = []
    batch_server_ids = []
    batch_seq_len = []

    block_max_len = 0
    ibis_block_tables = []
    ibis_max_output_len = []
    ibis_batch_seed = []
    ibis_batch_dp_rank_ids = []
    ibis_batch_sp_tokens = []
    ibis_batch_sp_rank_id = []
    ibis_batch_is_append_block = []
    ibis_batch_prefill_block_rank_id = []
    ibis_batch_block_rank_id = []
    is_sp_enable = False
    is_cp_enable = False
    is_mtp_enable = False
    is_req_prefill = []
    is_req_last_chunk = []
    split_start_pos = []
    split_end_pos = []
    batch_response_format = None
    
    # prefix cache
    computed_blocks = None
    remote_computed_blocks = None

    block_copy = [[item.num1, item.num2] for item in request.blocks_to_copy] if request.blocks_to_copy else None
    block_op = parse_swap_blocks(request.blocks_to_swap_in, request.blocks_to_swap_out)
    # 虚推请求使用最后一个 block (num_npu_blocks - 1) 作为占位符
    block_id_for_simulate_req = num_npu_blocks - 1

    for seq_group_metadata in request.seq_group_metadata_list:
        batch_req_ids.append(seq_group_metadata.request_id)
        batch_server_ids.append(seq_group_metadata.server_id)
        batch_seq_len.extend(convert_bytes_to_list(seq_group_metadata.prompt_lens))

        seq_ids = convert_bytes_to_list(seq_group_metadata.seqIds)
        seq_num = len(seq_ids)
        sampling_params = seq_group_metadata.sampling_params
        max_output_len = sampling_params.max_output_len
        ibis_max_output_len.extend([max_output_len for _ in range(seq_num)])
        ibis_batch_seed.append(abs(sampling_params.seed))
        ibis_batch_dp_rank_ids.append(seq_group_metadata.dp_rank_id)
        if config is not None:
            is_sp_enable = config.sp_size > 1
            is_cp_enable = config.cp_size > 1
            is_mtp_enable = config.enable_mtp
        lwd_multi_nodes_enable = getattr(config, 'model_config', {}).get('lwd_multi_nodes_enable', 'false') == 'true'
        lwd_is_slave = getattr(config, 'model_config', {}).get('layerwiseDisaggregatedRoleType', 'master') == 'slave'
        is_scp_enbale = is_sp_enable or is_cp_enable

        # 根据请求类型构建 block table
        if seq_ids[0] == SIMULATE_SEQUENCE_ID:
            seq_blocks, sp_rank_block_num, sp_rank_token_num = build_simulate_block_table(
                seq_group_metadata, block_id_for_simulate_req, is_sp_enable, is_cp_enable, config
            )
        else:
            seq_blocks = convert_bytes_to_list(seq_group_metadata.block_tables[0])
            if lwd_multi_nodes_enable and lwd_is_slave and is_scp_enbale:
                seq_blocks = convert_bytes_to_list(seq_group_metadata.lwd_cloud_metadata.lwd_cloud_block_tables)
            sp_rank_block_num = None
            sp_rank_token_num = None

        if is_sp_enable or is_cp_enable:
            block_max_len = max(block_max_len, len(seq_blocks))
            # 虚推请求使用函数返回的 sp_rank_token_num，正常请求从 protobuf 获取
            if seq_ids[0] == SIMULATE_SEQUENCE_ID:
                ibis_batch_sp_tokens.append(sp_rank_token_num)
            else:
                if lwd_multi_nodes_enable and lwd_is_slave:
                    lwd_cloud_metadata = seq_group_metadata.lwd_cloud_metadata
                    ibis_batch_sp_tokens.append(list(lwd_cloud_metadata.lwd_cloud_sp_rank_token_num))
                    sp_rank_block_num = list(lwd_cloud_metadata.lwd_cloud_sp_rank_block_num)
                else:
                    ibis_batch_sp_tokens.append(list(seq_group_metadata.sp_rank_token_num))
                    sp_rank_block_num = list(seq_group_metadata.sp_rank_block_num)
            if lwd_multi_nodes_enable and lwd_is_slave:
                lwd_cloud_metadata = seq_group_metadata.lwd_cloud_metadata
                ibis_batch_sp_rank_id.append(lwd_cloud_metadata.lwd_cloud_sp_rank_id)
                ibis_batch_block_rank_id.append(lwd_cloud_metadata.lwd_cloud_append_block_rank_id)
            else:
                ibis_batch_sp_rank_id.append(seq_group_metadata.sp_rank_id)
                ibis_batch_block_rank_id.append(seq_group_metadata.append_block_rank_id)
            if is_mtp_enable:
                ibis_batch_is_append_block.append(seq_group_metadata.is_append_block)
                if is_prefill:
                    ibis_batch_prefill_block_rank_id.append(list(seq_group_metadata.prefill_block_rank_id))
            start = 0
            sp_block_tables = []
            for block_num in sp_rank_block_num:
                # SP场景下，block table增加一个sp维度
                sp_block_tables.append(seq_blocks[start: start + block_num])
                start += block_num
            ibis_block_tables.append(sp_block_tables)
        else:
            ibis_block_tables.extend(split_list_equally(seq_blocks, seq_num))
            block_max_len = max(block_max_len, len(ibis_block_tables[-1]))

        if is_mix_model:
            is_req_prefill.extend(seq_group_metadata.is_req_prefill)
            is_req_last_chunk.extend(seq_group_metadata.is_req_last_chunk)
            split_start_pos.extend(seq_group_metadata.split_start_pos)
            split_end_pos.extend(seq_group_metadata.split_end_pos)

    seq_lens = parse_all_dp_batches_seq_lens(request.all_dp_batches_seq_lens)

    padded = []
    if is_sp_enable or is_cp_enable:
        for sp_rows in ibis_block_tables:
            padded_sp_rows = [sp_row + [-1] * (block_max_len - len(sp_row)) for sp_row in sp_rows]
            padded.append(padded_sp_rows)
    else:
        padded = [row + [-1] * (block_max_len - len(row)) for row in ibis_block_tables]
    block_tables = np.array(padded)
    max_output_len = np.array(ibis_max_output_len)
    batch_seed = np.array(ibis_batch_seed)
    batch_dp_rank_ids = np.array(ibis_batch_dp_rank_ids)
    total_seq_len = sum(batch_seq_len)
    batch_sp_tokens = None
    batch_sp_rank_id = None
    batch_is_append_block = None
    batch_prefill_block_rank_id = None
    batch_block_rank_id = None
    if any(ibis_batch_sp_tokens):
        batch_sp_tokens = np.array(ibis_batch_sp_tokens)
    if ibis_batch_sp_rank_id:
        batch_sp_rank_id = np.array(ibis_batch_sp_rank_id)
    if ibis_batch_is_append_block:
        batch_is_append_block = np.array(ibis_batch_is_append_block)
    if ibis_batch_prefill_block_rank_id:
        max_len = max(len(x) for x in ibis_batch_prefill_block_rank_id)
        batch_prefill_block_rank_id = np.array([
            np.pad(arr, (0, max_len - len(arr)), constant_values=-1)
            for arr in ibis_batch_prefill_block_rank_id
        ])
    if ibis_batch_block_rank_id:
        batch_block_rank_id = np.array(ibis_batch_block_rank_id)
    if is_prefill:
        prefill_params = parse_para_is_prefill(request.seq_group_metadata_list, block_size, config)
        batch_sampling = prefill_params["batch_sampling"]
        batch_input_ids = prefill_params["batch_input_ids"]
        batch_stop_token_ids = prefill_params["batch_stop_token_ids"]
        batch_stop_strings = prefill_params["batch_stop_strings"]
        batch_ignore_eos = prefill_params["batch_ignore_eos"]
        batch_skip_special_tokens = prefill_params["batch_skip_special_tokens"]
        batch_include_stop = prefill_params["batch_include_stop"]
        batch_seq_ids = prefill_params["batch_seq_ids"]
        batch_best_of = prefill_params["batch_best_of"]
        batch_n = prefill_params["batch_n"]
        batch_logprobs = prefill_params["batch_logprobs"]
        batch_reserved_seq_ids = prefill_params["batch_reserved_seq_ids"]
        batch_use_beam_search = prefill_params["batch_use_beam_search"]
        adapter_ids = prefill_params["adapter_ids"]
        computed_blocks = prefill_params["computed_blocks"]
        remote_computed_blocks = prefill_params["remote_computed_blocks"]
        batch_response_format = prefill_params["batch_response_format"]
    else:
        # decode 阶段：收集必要的批次元数据
        # 注意：在 decode 阶段，每个 seq_group 可能包含多个 sequence
        # response_format_array 需要与 all_sequence_ids 一一对应（sequence 级别）
        batch_response_format = []
        for seq_group_metadata in request.seq_group_metadata_list:
            seq_ids = convert_bytes_to_list(seq_group_metadata.seqIds)
            num_sequences = len(seq_ids)
            
            batch_seq_ids.append(
                np.array(seq_ids, copy=True, dtype=np.int64)
            )
            reserved_seqs_id_tensor = list(seq_group_metadata.reserved_seq_ids)
            if reserved_seqs_id_tensor is None:
                reserved_id = np.array([], dtype=np.int64)
            else:
                reserved_id = np.array(reserved_seqs_id_tensor, copy=True, dtype=np.int64)
            batch_reserved_seq_ids.append(reserved_id)
            
            # 收集 response_format（每个 sequence 都需要相同的 response_format）
            response_format = None
            if seq_group_metadata.HasField("response_format"):
                response_format = seq_group_metadata.response_format
            # Decode 阶段：按 sequence 数量扩展
            batch_response_format.extend([response_format] * num_sequences)
    batch_ignore_eos = np.array(batch_ignore_eos)
    batch_skip_special_tokens = np.array(batch_skip_special_tokens)
    batch_include_stop = np.array(batch_include_stop)
    batch_logprobs = np.array(batch_logprobs)

    metadata = InputMetadata(
        batch_size=len(request.seq_group_metadata_list),
        batch_request_ids=np.array(batch_req_ids),
        batch_max_output_lens=max_output_len,
        block_tables=block_tables,
        max_block_size=block_size,
        has_sampling=True,
        is_prefill=is_prefill,
        is_mix=convert_para.is_mix,
        input_ids=batch_input_ids,
        batch_seq_len=np.array(batch_seq_len),
        total_seq_num=total_seq_len,
        batch_sampling_params=np.array(batch_sampling),
        batch_stop_strings=batch_stop_strings,
        batch_stop_token_ids=batch_stop_token_ids,
        computed_blocks=computed_blocks,
        remote_computed_blocks=remote_computed_blocks,
        adapter_ids=adapter_ids,
        num_npu_blocks=num_npu_blocks,
        batch_dp_rank_ids=batch_dp_rank_ids,
        batch_ignore_eos=batch_ignore_eos,
        batch_skip_special_tokens=batch_skip_special_tokens,
        batch_include_stop=batch_include_stop,
        trace_ids=batch_req_ids,
        simulator_ids=batch_server_ids,
        batch_sequence_ids=batch_seq_ids,
        batch_best_of=batch_best_of,
        batch_logprobs=batch_logprobs,
        batch_seeds=batch_seed,
        batch_n=batch_n,
        batch_use_beam_search=batch_use_beam_search,
        reserved_sequence_ids=batch_reserved_seq_ids,
        sp_tokens=batch_sp_tokens,
        sp_rank_id=batch_sp_rank_id,
        seq_lens=seq_lens,
        is_append_block=batch_is_append_block,
        prefill_block_rank_id=batch_prefill_block_rank_id,
        block_rank_id=batch_block_rank_id,
        layerwise_disaggregated_exe_stage=layerwise_disaggregated_exe_stage,
        batch_response_format=batch_response_format
    )

    input_metadata_composite = InputMetadataComposite()
    input_metadata_composite.input_metadata = metadata
    input_metadata_composite.block_copy = block_copy
    input_metadata_composite.block_op = block_op

    if is_mix_model:
        _, _, decode_bs = get_batch_size(request, is_prefill, convert_para.is_mix)
        mix_params = {
            "is_mix": convert_para.is_mix,
            "decode_bs": decode_bs,
            "is_req_prefill": is_req_prefill,
            "split_start_pos": split_start_pos,
            "split_end_pos": split_end_pos,
            "is_req_last_chunk": is_req_last_chunk
        }
        update_mix_metadata(input_metadata_composite, mix_params)

    span_attr(span, "size", metadata.batch_size)
    span_end(span)

    return input_metadata_composite


def convert_pull_kv_request_to_input_metadata_composite(
        request: PullKVRequest,
        num_npu_blocks,
        block_size,
        config=None
) -> InputMetadataComposite:
    batch_req_ids = []
    batch_server_ids = []
    batch_seq_len = []

    block_max_len = 0
    ibis_block_tables = []
    ibis_max_output_len = []
    ibis_batch_seed = []
    ibis_batch_dp_rank_ids = []
    ibis_batch_sp_tokens = []

    is_sp_enable = False
    is_cp_enable = False

    # prefix cache
    computed_blocks = None
    remote_computed_blocks = None

    for pull_kv_info in request.pull_kv_infos:
        batch_req_ids.append(pull_kv_info.seq_group_metadata.request_id)
        batch_server_ids.append(pull_kv_info.seq_group_metadata.server_id)
        batch_seq_len.extend(convert_bytes_to_list(pull_kv_info.seq_group_metadata.prompt_lens))

        seq_ids = convert_bytes_to_list(pull_kv_info.seq_group_metadata.seqIds)
        seq_num = len(seq_ids)
        sampling_params = pull_kv_info.seq_group_metadata.sampling_params
        max_output_len = sampling_params.max_output_len
        ibis_max_output_len.extend([max_output_len for _ in range(seq_num)])
        ibis_batch_seed.append(abs(sampling_params.seed))
        ibis_batch_dp_rank_ids.append(pull_kv_info.seq_group_metadata.dp_rank_id)
        if config is not None:
            is_sp_enable = config.sp_size > 1
            is_cp_enable = config.cp_size > 1
        seq_blocks = convert_bytes_to_list(pull_kv_info.seq_group_metadata.block_tables[0])

        if is_sp_enable or is_cp_enable:
            block_max_len = max(block_max_len, len(seq_blocks))
            ibis_batch_sp_tokens.append(list(pull_kv_info.seq_group_metadata.sp_rank_token_num))

            sp_rank_block_num = list(pull_kv_info.seq_group_metadata.sp_rank_block_num)
            start = 0
            sp_block_tables = []
            for block_num in sp_rank_block_num:
                # SP场景下，block table增加一个sp维度
                sp_block_tables.append(seq_blocks[start: start + block_num])
                start += block_num
            ibis_block_tables.append(sp_block_tables)
        else:
            ibis_block_tables.extend(split_list_equally(seq_blocks, seq_num))
            block_max_len = max(block_max_len, len(ibis_block_tables[-1]))

    padded = []
    if is_sp_enable or is_cp_enable:
        for sp_rows in ibis_block_tables:
            padded_sp_rows = [sp_row + [-1] * (block_max_len - len(sp_row)) for sp_row in sp_rows]
            padded.append(padded_sp_rows)
    else:
        padded = [row + [-1] * (block_max_len - len(row)) for row in ibis_block_tables]
    block_tables = np.array(padded)
    max_output_len = np.array(ibis_max_output_len)
    batch_seed = np.array(ibis_batch_seed)
    batch_dp_rank_ids = np.array(ibis_batch_dp_rank_ids)
    total_seq_len = sum(batch_seq_len)
    batch_sp_tokens = None
    if any(ibis_batch_sp_tokens):
        batch_sp_tokens = np.array(ibis_batch_sp_tokens)

    seq_group_metadata_list = [pull_kv_info.seq_group_metadata for pull_kv_info in request.pull_kv_infos]

    prefill_params = parse_para_is_prefill(seq_group_metadata_list, block_size, config)
    batch_sampling = prefill_params["batch_sampling"]
    batch_input_ids = prefill_params["batch_input_ids"]
    batch_stop_token_ids = prefill_params["batch_stop_token_ids"]
    batch_stop_strings = prefill_params["batch_stop_strings"]
    batch_ignore_eos = prefill_params["batch_ignore_eos"]
    batch_skip_special_tokens = prefill_params["batch_skip_special_tokens"]
    batch_include_stop = prefill_params["batch_include_stop"]
    batch_seq_ids = prefill_params["batch_seq_ids"]
    batch_best_of = prefill_params["batch_best_of"]
    batch_n = prefill_params["batch_n"]
    batch_logprobs = prefill_params["batch_logprobs"]
    batch_reserved_seq_ids = prefill_params["batch_reserved_seq_ids"]
    batch_use_beam_search = prefill_params["batch_use_beam_search"]
    adapter_ids = prefill_params["adapter_ids"]
    computed_blocks = prefill_params["computed_blocks"]
    remote_computed_blocks = prefill_params["remote_computed_blocks"]

    batch_ignore_eos = np.array(batch_ignore_eos)
    batch_skip_special_tokens = np.array(batch_skip_special_tokens)
    batch_include_stop = np.array(batch_include_stop)
    batch_logprobs = np.array(batch_logprobs)

    metadata = InputMetadata(
        batch_size=len(seq_group_metadata_list),
        batch_request_ids=np.array(batch_req_ids),
        batch_max_output_lens=max_output_len,
        block_tables=block_tables,
        max_block_size=block_size,
        has_sampling=True,
        is_prefill=True,
        input_ids=batch_input_ids,
        batch_seq_len=np.array(batch_seq_len),
        total_seq_num=total_seq_len,
        batch_sampling_params=np.array(batch_sampling),
        batch_stop_strings=batch_stop_strings,
        batch_stop_token_ids=batch_stop_token_ids,
        computed_blocks=computed_blocks,
        remote_computed_blocks=remote_computed_blocks,
        adapter_ids=adapter_ids,
        num_npu_blocks=num_npu_blocks,
        batch_dp_rank_ids=batch_dp_rank_ids,
        batch_ignore_eos=batch_ignore_eos,
        batch_skip_special_tokens=batch_skip_special_tokens,
        batch_include_stop=batch_include_stop,
        trace_ids=batch_req_ids,
        simulator_ids=batch_server_ids,
        batch_sequence_ids=batch_seq_ids,
        batch_best_of=batch_best_of,
        batch_logprobs=batch_logprobs,
        batch_seeds=batch_seed,
        batch_n=batch_n,
        batch_use_beam_search=batch_use_beam_search,
        reserved_sequence_ids=batch_reserved_seq_ids,
        sp_tokens=batch_sp_tokens,
        seq_lens=[[1]],
    )

    input_metadata_composite = InputMetadataComposite()
    input_metadata_composite.input_metadata = metadata
    return input_metadata_composite


def pad_input_ids(input_ids: list, cp_size):
    in_lens = len(input_ids)
    pad_num = cp_size * 2 - (in_lens % (cp_size * 2))
    input_ids += [PLACEHOLDER_TOKEN] * pad_num


def parse_para_is_prefill(seq_group_metadata_list: List[SequenceGroupMetadata], block_size, config=None) -> dict:
    batch_sampling = []
    batch_input_ids = []
    batch_stop_token_ids = []
    batch_stop_strings = []
    batch_ignore_eos = []
    batch_skip_special_tokens = []
    batch_include_stop = []
    batch_seq_ids = []
    batch_best_of = np.array([], dtype=np.int64)
    batch_n = np.array([], dtype=np.int64)
    batch_logprobs = []
    batch_reserved_seq_ids = []
    batch_use_beam_search = np.array([], dtype=bool)
    batch_response_format = []
    
    # prefix cache
    computed = []
    remote_computed = []

    scp_size = 1
    cp_size = 1
    if config is not None:
        cp_size = config.cp_size
        scp_size = config.sp_size * config.cp_size
    for seq_group_metadata in seq_group_metadata_list:
        sampling_params = parse_sampling_parameters(seq_group_metadata)
        batch_sampling.append(sampling_params[0])

        input_ids = convert_bytes_to_list(seq_group_metadata.prompt_token_ids)

        if cp_size > 1 and len(input_ids) % (cp_size * 2) != 0:
            pad_input_ids(input_ids, cp_size)

        # splitfuse时decode请求需要向input_ids中加入占位符，在插件中从cache里获取真正token
        # TODO：splitfuse叠加mtp时需要进一步适配input_ids占位符个数
        if not input_ids:
            batch_input_ids.extend([0] * len(seq_group_metadata.split_start_pos))
        else:
            batch_input_ids.extend(input_ids)

        stop_ids = list(seq_group_metadata.stop_token_ids)
        batch_stop_token_ids.append(stop_ids if stop_ids and len(stop_ids) > 0 else None)
        stop_string_bytes = list(seq_group_metadata.stop)
        stop_strings = None
        if stop_string_bytes and len(stop_string_bytes) > 0:
            stop_str_decode = base64.b64decode(stop_string_bytes[0].strip()).decode("utf-8")
            if stop_str_decode:
                stop_strings = json.loads(stop_str_decode)
        batch_stop_strings.append(stop_strings if stop_strings and len(stop_strings) > 0 else None)

        batch_ignore_eos.append(
            seq_group_metadata.ignore_eos if seq_group_metadata.HasField("ignore_eos") else None
        )
        batch_skip_special_tokens.append(
            seq_group_metadata.skip_special_tokens if seq_group_metadata.HasField("skip_special_tokens") else None
        )
        batch_include_stop.append(seq_group_metadata.include_stop_str_in_output)

        seq_ids = convert_bytes_to_list(seq_group_metadata.seqIds)
        num_sequences = len(seq_ids)
        
        batch_seq_ids.append(
            np.array(seq_ids, copy=True, dtype=np.int64)
        )
        reserved_seqs_id_tensor = list(seq_group_metadata.reserved_seq_ids)
        reserved_id = np.array(reserved_seqs_id_tensor, copy=True,
                               dtype=np.int64) if reserved_seqs_id_tensor else np.array([], dtype=np.int64)
        batch_reserved_seq_ids.append(reserved_id)

        sampling_params_detail = seq_group_metadata.sampling_params
        batch_best_of = np.append(batch_best_of, np.array([sampling_params_detail.best_of]))
        batch_n = np.append(batch_n, np.array([sampling_params_detail.n]))
        logprobs = (
            sampling_params_detail.logprobs
            if sampling_params_detail.HasField("logprobs")
            else None
        )
        batch_logprobs.append(logprobs if logprobs is not None else None)
        batch_use_beam_search = np.append(
            batch_use_beam_search, np.array([sampling_params_detail.use_beam_search], dtype=bool)
        )
        # 解析 response_format（每个 sequence 都需要相同的 response_format）
        response_format = None
        if seq_group_metadata.HasField("response_format"):
            response_format = seq_group_metadata.response_format
        # Prefill 阶段通常每个 seq_group 只有 1 个 sequence，但为了一致性也按 sequence 数量扩展
        batch_response_format.extend([response_format] * num_sequences)

        # 解析每条request已计算的block数量，解析成一维list，如不存在将值置为None
        # 虚推请求不使用 prefix cache，填充 0 值以确保维度对齐
        if seq_ids[0] == SIMULATE_SEQUENCE_ID:
            if scp_size >= 1:
                computed.extend([0] * scp_size)
                remote_computed.extend([0] * scp_size)
            continue
        
        seq_scp_size = len(seq_group_metadata.sp_rank_block_num)
        computed_ = convert_bytes_to_list(seq_group_metadata.computed_block_lens)
        remote_computed_ = convert_bytes_to_list(seq_group_metadata.remote_computed_block_lens)
        
        if seq_scp_size > 1:
            sp_rank_id = seq_group_metadata.sp_rank_id
            if len(input_ids) == sum(computed_) * block_size:
                computed_[sp_rank_id] -= 1
                remote_computed_[sp_rank_id] -= 1
                del_order_id = sum(remote_computed_[:sp_rank_id + 1])
            elif len(input_ids) == sum(remote_computed_) * block_size:
                remote_computed_[sp_rank_id] -= 1
                del_order_id = sum(remote_computed_[:sp_rank_id + 1])

        computed.extend(computed_)
        remote_computed.extend(remote_computed_)

    computed_blocks = None
    remote_computed_blocks = None

    lwd_multi_nodes_enable = getattr(config, 'model_config', {}).get('lwd_multi_nodes_enable', 'false') == 'true'
    if scp_size > 1 and not lwd_multi_nodes_enable:
        if computed:
            computed_blocks = np.array(computed, dtype=np.int64).reshape(-1, scp_size)
            if np.count_nonzero(computed_blocks) == 0:
                computed_blocks = None
        if remote_computed:
            remote_computed_blocks = np.array(remote_computed, dtype=np.int64).reshape(-1, scp_size)
            if np.count_nonzero(remote_computed_blocks) == 0:
                remote_computed_blocks = None
    else:
        if computed:
            computed_blocks = np.array(computed, dtype=np.int64)
            if np.count_nonzero(computed_blocks) == 0:
                computed_blocks = None
        if remote_computed:
            remote_computed_blocks = np.array(remote_computed, dtype=np.int64)
            if np.count_nonzero(remote_computed_blocks) == 0:
                remote_computed_blocks = None

    adapter_ids = list(map(generate_lora_strings, seq_group_metadata_list))

    batch_input_ids = np.array(batch_input_ids, dtype=np.int64)

    return {
        "batch_sampling": batch_sampling,
        "batch_input_ids": batch_input_ids,
        "batch_stop_token_ids": batch_stop_token_ids,
        "batch_stop_strings": batch_stop_strings,
        "batch_ignore_eos": batch_ignore_eos,
        "batch_skip_special_tokens": batch_skip_special_tokens,
        "batch_include_stop": batch_include_stop,
        "batch_seq_ids": batch_seq_ids,
        "batch_best_of": batch_best_of,
        "batch_n": batch_n,
        "batch_logprobs": batch_logprobs,
        "batch_reserved_seq_ids": batch_reserved_seq_ids,
        "batch_use_beam_search": batch_use_beam_search,
        "adapter_ids": adapter_ids,
        "computed_blocks": computed_blocks,
        "remote_computed_blocks": remote_computed_blocks,
        "batch_response_format": batch_response_format,
    }


def update_mix_metadata(input_metadata_composite, mix_params):
    is_mix = mix_params["is_mix"]
    decode_bs = mix_params["decode_bs"]
    split_start_pos = mix_params["split_start_pos"]
    split_end_pos = mix_params["split_end_pos"]
    is_req_prefill = mix_params["is_req_prefill"]
    if (is_req_prefill and
        (not isinstance(is_req_prefill, list) or not all(isinstance(item, bool) for item in is_req_prefill))):
        raise TypeError("'is_req_prefill' must be a boolean list")
    is_req_last_chunk = mix_params["is_req_last_chunk"]
    input_metadata_composite.decode_batch_size = decode_bs

    batch_seq_len = np.array(split_end_pos) - np.array(split_start_pos)
    total_seq_len = sum(batch_seq_len)

    # 相比PD竞争需要刷新的参数
    metadata = input_metadata_composite.input_metadata
    metadata.total_seq_num = total_seq_len
    metadata.batch_seq_len = batch_seq_len

    # 纯decode请求的batch input_ids加入占位符
    if not metadata.is_prefill:
        metadata.input_ids = np.array([0] * (len(is_req_prefill) - sum(is_req_prefill)))

    # 相比PD竞争需要新增的参数
    metadata.is_mix = is_mix
    metadata.split_start_position = np.array(split_start_pos)
    metadata.split_end_position = np.array(split_end_pos)
    metadata.batch_last_prompt = np.array(is_req_last_chunk)
    metadata.batch_is_prefill = np.array(is_req_prefill)

    # 重新计算max_seq_len和block_table
    if len(metadata.split_end_position) > 0:
        metadata.max_seq_len = max(metadata.split_end_position)
    else:
        metadata.max_seq_len = 0
    max_block_num = np.count_nonzero(metadata.block_tables > -1, axis=-1).max()
    metadata.batch_block_tables = metadata.block_tables[:, :max_block_num].astype(np.int32)


def get_attribute_info(link_request: PDLinkRequest):
    pd_link_info = link_request.pd_link_info[0]
    # attribute_info，建链信息，shape=(1,7)
    # 第一位表示设置成什么角色，1代表prefill，2代表decoder，其他为unknown
    # 第二位role是否需要切换，1表示需要切换，0表示不切换
    # 第三位：link_num
    # 第四位：unlink_num
    # 第五位：host_ip_num
    # 第六位：super_id_num
    # 第七位：contains_dp_instance_ids
    pd_role = 0
    if pd_link_info.pd_role == PDRole.PREFILL_ROLE:
        pd_role = 1
    elif pd_link_info.pd_role == PDRole.DECODE_ROLE:
        pd_role = 2
    if_change_role = 1 if pd_link_info.change_role else 0
    attribute_data = [
        pd_role,
        if_change_role,
        pd_link_info.link_num,
        pd_link_info.unlink_num,
        pd_link_info.host_ip_num,
        pd_link_info.super_id_num,
        pd_link_info.contains_dp_instance_ids
    ]
    attribute_info = np.array([attribute_data], dtype=np.int64)

    """
     DEVICES 设备信息. shape=(link_num + unlink_num, device_num+1, device_info_num)

     A2 场景下 device_info_num=9
     第0行: 元素 0~7 表示远端主机的 ip, 支持 IPv4(加 -1 PADDING扩展长度为8) 或 IPv6, 元素 8 表示远端主机的 cluster_id, 例如:
            IPv4: [127, 0, 0, 1, -1, -1, -1, -1, 100]
            IPv6: [0x2001, 0xdb8, 0, 0, 0, 0, 0, 0x1, 100]
     接下来 device_num 行同理: 元素 0~7 表示远端设备的 ip 地址, 元素 8 表示远端设备的 physical_id

     A3 场景下 device_info_num=10
     第0行增加元素 9 表示远端主机的超节点 id (remote_super_pod_id)
     接下来 device_num 行增加元素 9 表示远端设备在超节点上的 id (remote_super_device_id)
     """
    if len(pd_link_info.link_info) > 0:
        device_num = len(pd_link_info.link_info[0].device_info)
    else:
        device_num = len(pd_link_info.unlink_info[0].device_info)

    # 检查是否有超节点信息（A3场景），不仅依赖super_id_num，还要检查实际的超节点字段
    has_super_info = pd_link_info.super_id_num > 0

    def _has_super_info_in_remote_infos(remote_infos):
        """检查RemoteInfo列表中是否包含超节点信息"""
        for remote_info in remote_infos:
            # 检查host_info中的super_pod_id
            for host_info in remote_info.host_info:
                if host_info.HasField("super_pod_id"):
                    return True
            # 检查device_info中的super_device_id
            for device_info in remote_info.device_info:
                if device_info.HasField("super_device_id"):
                    return True
        return False

    if not has_super_info:
        # 检查link_info中的超节点信息，如果没有，检查unlink_info中的超节点信息
        has_super_info = (
            _has_super_info_in_remote_infos(pd_link_info.link_info) or
            _has_super_info_in_remote_infos(pd_link_info.unlink_info)
        )

    if has_super_info:
        device_info_num = 10  # A3 场景
    else:
        device_info_num = 9  # A2 场景

    device_pd_num = pd_link_info.unlink_num + pd_link_info.link_num
    host_ip_num_per_dp = 1
    # In mspd scenario, globalipinfo doesn't have DpInstanceId
    # In large EP scenario, globalipinfo has DpInstanceId
    if pd_link_info.contains_dp_instance_ids == 1:
        if (pd_link_info.link_num != 0):
            host_ip_num_per_dp = pd_link_info.host_ip_num // pd_link_info.link_num
    device_data = np.zeros((device_pd_num, device_num + host_ip_num_per_dp, device_info_num), dtype=np.int64)
    # 处理link_info
    for i, _link_info in enumerate(pd_link_info.link_info):
        for j, host_info in enumerate(_link_info.host_info):
            host_ip_list = ip_string_to_list(host_info.host_ip)
            host_info_list = host_ip_list + [int(host_info.cluster_id)]
            if host_info.HasField("super_pod_id"):
                host_info_list.append(host_info.super_pod_id)
            device_data[i, j, :] = np.pad(
                host_info_list, (0, device_info_num - len(host_info_list)), constant_values=-1
            )
        for j, device_info in enumerate(_link_info.device_info):
            device_ip_list = ip_string_to_list(device_info.device_ip)
            device_info_list = device_ip_list + [device_info.physical_id]
            if device_info.HasField("super_device_id"):
                device_info_list.append(device_info.super_device_id)
            device_data[i, j + host_ip_num_per_dp, :] = np.pad(
                device_info_list, (0, device_info_num - len(device_info_list)), constant_values=-1
            )
    # 处理unlink_info
    for i, _link_info in enumerate(pd_link_info.unlink_info):
        for j, host_info in enumerate(_link_info.host_info):
            host_ip_list = ip_string_to_list(host_info.host_ip)
            host_info_list = host_ip_list + [int(host_info.cluster_id)]
            if host_info.HasField("super_pod_id"):
                host_info_list.append(host_info.super_pod_id)
            device_data[len(pd_link_info.link_info) + i, j, :] = np.pad(
                host_info_list, (0, device_info_num - len(host_info_list)), constant_values=-1
            )
        for j, device_info in enumerate(_link_info.device_info):
            device_ip_list = ip_string_to_list(device_info.device_ip)
            device_info_list = device_ip_list + [device_info.physical_id]
            if device_info.HasField("super_device_id"):
                device_info_list.append(device_info.super_device_id)
            device_data[len(pd_link_info.link_info) + i, j + host_ip_num_per_dp, :] = np.pad(
                device_info_list, (0, device_info_num - len(device_info_list)), constant_values=-1
            )
    # handle policy
    instance_ids = list(pd_link_info.instance2sp.keys())
    sp_sizes = [pd_link_info.instance2sp[instance_id] for instance_id in instance_ids]
    cp_sizes = [pd_link_info.instance2cp[instance_id] for instance_id in instance_ids]
    policy = np.array(list(zip(instance_ids, sp_sizes, cp_sizes)), dtype=np.int64)

    return (attribute_info, device_data, policy)