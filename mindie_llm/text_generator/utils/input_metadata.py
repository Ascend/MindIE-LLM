# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt

from mindie_llm.utils.log import ErrorCode, logger
from .request import Request
from .config import SAMPLING_DTYPE

# Reserved sequence ID for simulate inference requests
# Should match SIMULATE_SEQUENCE_ID in C++ code (src/include/request_response/request_id.h)
SIMULATE_SEQUENCE_ID = 9223372036854774


def get_batch_size(is_prefill_pre_batch, requests):
    is_prefill = True in is_prefill_pre_batch
    is_mix = (True in is_prefill_pre_batch) and (False in is_prefill_pre_batch)

    if is_mix:
        total_bs = len(requests)
        prefill_bs = np.sum(is_prefill_pre_batch)
        decode_bs = total_bs - prefill_bs
    else:
        total_bs = len(requests)
        decode_bs = 0
        if not is_prefill:
            decode_bs = total_bs
    return is_prefill, is_mix, decode_bs


@dataclass
class LwdMetadata():
    start_exec_layer: int = 0
    end_exec_layer: int = 0
    end_of_generate_token: bool = True
    is_prefill: bool = True
    is_dummy_batch: bool = False
    request_dp_empty: bool = False
    cloud_total_layer: int = 62
    is_long_seq: bool = False
    long_seq_start_idx: int = 0
    long_seq_end_idx: int = 0
    long_seq_next_end_idx: int = 0
    prefill_total_seq_len: int = 0
    is_last_chunk: bool = False


@dataclass(slots=True)
class InputMetadata:
    # basic attributes must be passed
    batch_size: int
    batch_request_ids: np.ndarray  # (bs,)
    batch_sequence_ids: List[np.ndarray]
    batch_max_output_lens: np.ndarray  # (bs,)
    block_tables: np.ndarray  # (bs, block_num)
    reserved_sequence_ids: List[np.ndarray]

    # basic attributes with default values
    has_sampling: bool = True
    is_mix: Optional[bool] = None
    is_prefill: bool = False
    max_block_size: int = 128
    num_npu_blocks: int = 0

    # attributes for multi-sequence
    batch_n: Optional[np.ndarray] = None  # (bs,)
    batch_best_of: Optional[np.ndarray] = None  # (bs,)
    batch_use_beam_search: Optional[np.ndarray] = None  # (bs,)

    # normal attributes
    adapter_ids: Optional[List[Optional[str]]] = None
    batch_ignore_eos: Optional[np.ndarray] = None  # (bs,)
    batch_include_stop: Optional[np.ndarray] = None  # (bs,)
    batch_logprobs: Optional[np.ndarray] = None  # (bs,)
    batch_sampling_params: Optional[np.ndarray] = None  # (bs,)
    batch_seeds: Optional[np.ndarray] = None  # (bs,)
    batch_seq_len: Optional[np.ndarray] = None  # (bs,)
    batch_skip_special_tokens: Optional[np.ndarray] = None  # (bs,)
    batch_stop_strings: Optional[List[Optional[List[str]]]] = None
    batch_stop_token_ids: Optional[List[Optional[List[Union[int, List[int]]]]]] = None
    input_ids: Optional[np.ndarray] = None  # prefill: (token_num,), decode: (0,)
    total_seq_num: Optional[int] = None
    trace_ids: Optional[List[Any]] = None
    simulator_ids: Optional[List[Any]] = None
    # JSON 结构化输出约束 (response_format)
    batch_response_format: Optional[List[Optional[str]]] = None

    # attributes for prefixcache
    computed_blocks: Optional[np.ndarray] = None
    remote_computed_blocks: Optional[np.ndarray] = None

    # attributes for features
    split_start_position: Optional[np.ndarray] = None
    split_end_position: Optional[np.ndarray] = None
    batch_last_prompt: Optional[np.ndarray] = None
    batch_is_prefill: Optional[np.ndarray] = None
    batch_dp_rank_ids: Optional[np.ndarray] = None  # (bs,)

    # kvp and cp
    sp_tokens: Optional[np.ndarray] = None
    sp_rank_id: Optional[np.ndarray] = None
    is_append_block: Optional[np.ndarray] = None
    block_rank_id: Optional[np.ndarray] = None
    prefill_block_rank_id: Optional[np.ndarray] = None

    # attributes computed automatically
    all_sequence_ids: Optional[npt.NDArray[np.int64]] = None  # (bs,)
    batch_block_tables: Optional[np.ndarray] = None  # (bs, block_num)

    max_seq_len: Optional[int] = None
    max_batch_size: Optional[int] = None

    # dp dummy batch
    is_dummy_batch: Optional[bool] = False

    # sequence lengths among all DP rank
    seq_lens: Optional[List[List[int]]] = None

    layerwise_disaggregated_exe_stage: LwdMetadata = None

    # 动态切块新增

    def __post_init__(self):
        self.all_sequence_ids = np.concatenate(self.batch_sequence_ids)

        if self.is_prefill:
            if not self.is_mix:
                for sequence_ids in self.batch_sequence_ids:
                    if len(sequence_ids) != 1:
                        raise ValueError("Each request must have only 1 sequence in prefilling stage.")

            self.max_seq_len = max(self.batch_seq_len) if self.batch_seq_len is not None else 0

            if self.max_block_size == 0:
                message = (
                    'It has been detected that `max_block_size` is set to 0, but `max_block_size` must be '
                    'greater than 0 to ensure the program runs correctly. If you are unsure of an appropriate '
                    'value, you can simply omit this parameter.'
                )
                logger.error(message, ErrorCode.TEXT_GENERATOR_MAX_BLOCK_SIZE_INVALID)
                raise ZeroDivisionError(message)
            default_max_block_num = math.ceil(self.max_seq_len / self.max_block_size)
        else:
            default_max_block_num = np.count_nonzero(self.block_tables > -1, axis=-1).max()

        if self.sp_tokens is not None:
            max_block_num = np.count_nonzero(self.block_tables > -1, axis=-1).max()
            self.batch_block_tables = self.block_tables[:, :, :max_block_num].astype(np.int32)
        else:
            self.batch_block_tables = self.block_tables[:, :default_max_block_num].astype(np.int32)
    
    @staticmethod
    def form_block_tables(llm_requests: List[Request]) -> np.ndarray:
        batch_size = len(llm_requests)
        block_tables = [req.block_tables for req in llm_requests]
        first_block_table = block_tables[0]

        if first_block_table.ndim == 1:
            max_block_num = max(len(block_table) for block_table in block_tables)
            result = np.full((batch_size, max_block_num), -1, dtype=np.int32)
            for i, block_table in enumerate(block_tables):
                result[i, :len(block_table)] = block_table

        elif first_block_table.ndim == 2:
            scp_size = first_block_table.shape[0] 
            max_block_num = max(block_table.shape[1] for block_table in block_tables)
            result = np.full((batch_size, scp_size, max_block_num), -1, dtype=np.int32)
            for i, block_table in enumerate(block_tables):
                result[i, :, :block_table.shape[1]] = block_table
        return result

    @classmethod
    def from_requests(
        cls,
        llm_requests: list[Request],
        req_block_tables: np.ndarray | None = None,
        is_prefill: bool | np.ndarray = False,
        max_block_size: int = 128
    ):
        enable_splitfuse = False
        is_mix = None
        split_start_pos = None
        split_end_pos = None
        batch_last_prompt = None
        batch_is_prefill = None
        if isinstance(is_prefill, np.ndarray):
            batch_is_prefill = is_prefill
            is_prefill, is_mix, _ = get_batch_size(is_prefill, llm_requests)
            split_start_pos = np.asarray([request.split_start_position for request in llm_requests])
            split_end_pos = np.asarray([request.split_end_position for request in llm_requests])
            batch_last_prompt = np.asarray([request.last_prompt for request in llm_requests])
            enable_splitfuse = True

        batch_size = len(llm_requests)
        batch_request_ids = np.asarray([req.req_id for req in llm_requests])
        batch_sequence_ids = [np.asarray(list(req.sequences.keys())) for req in llm_requests]
        reserved_sequence_ids = [np.asarray(req.reserved_seq_ids) for req in llm_requests]
        
        adapter_ids = None
        batch_best_of = None
        batch_n = None
        batch_use_beam_search = None
        batch_ignore_eos = None
        batch_include_stop = None
        batch_logprobs = None
        batch_seeds = None
        batch_skip_special_tokens = None
        batch_stop_strings = None
        batch_stop_token_ids = None

        # prefix cache
        batch_computed_blocks = None
        batch_remote_computed_blocks = None

        batch_dp_rank_ids = None
        batch_sampling_params = None
        batch_seq_len = None
        batch_response_format = None
        has_sampling = any(req.has_sampling for req in llm_requests)
        input_ids_list = []
        total_seq_num = 0
        batch_sp_tokens = []
        batch_sp_rank_ids = []
        batch_prefill_block_rank_id = []
        batch_is_append_block = []
        batch_block_rank_id = []

        if is_prefill:
            batch_seq_len = []
            total_seq_num = 0
            batch_best_of = []
            batch_n = []
            batch_use_beam_search = []
            batch_ignore_eos = []
            batch_include_stop = []
            batch_logprobs = []
            batch_max_output_lens = []
            batch_skip_special_tokens = []
            batch_stop_strings = []
            batch_stop_token_ids = []
            batch_seeds = []
            adapter_ids = []
            batch_dp_rank_ids = []
            batch_response_format = []
            batch_computed_blocks = []
            batch_remote_computed_blocks = []

            for llm_request in llm_requests:
                input_ids_list.extend(llm_request.input_ids)
                seq_num = llm_request.input_ids.shape[0]
                batch_seq_len.append(seq_num)
                total_seq_num += seq_num
                request_best_of = llm_request.best_of if llm_request.best_of else 1
                batch_best_of.append(request_best_of)
                batch_n.append(llm_request.n if llm_request.n else request_best_of)
                batch_use_beam_search.append(llm_request.use_beam_search)
                batch_ignore_eos.append(llm_request.ignore_eos)
                batch_include_stop.append(llm_request.include_stop_str_in_output)
                batch_logprobs.append(llm_request.has_logprobs)
                batch_max_output_lens.extend([llm_request.max_new_tokens] * len(llm_request.sequences.keys()))
                batch_seeds.append(llm_request.seed)
                batch_skip_special_tokens.append(llm_request.skip_special_tokens)
                batch_stop_strings.append(llm_request.stop_strings)
                batch_stop_token_ids.append(llm_request.stop_token_ids)
                adapter_ids.append(llm_request.adapter_id)
                batch_dp_rank_ids.append(llm_request.dp_rank_id)
                batch_response_format.append(llm_request.response_format)
                batch_computed_blocks.append(llm_request.computed_blocks)
                batch_remote_computed_blocks.append(llm_request.remote_computed_blocks)
                if llm_request.sp_tokens is not None:
                    batch_sp_tokens.append(llm_request.sp_tokens)
                    batch_sp_rank_ids.append(llm_request.sp_rank_id)
                    batch_prefill_block_rank_id.append(llm_request.prefill_block_rank_id)

            if batch_computed_blocks:
                computed_blocks = np.array(batch_computed_blocks, dtype=np.int64)
                if np.count_nonzero(computed_blocks) == 0:
                    batch_computed_blocks = []
            if batch_remote_computed_blocks:
                remote_computed_blocks = np.array(batch_remote_computed_blocks, dtype=np.int64)
                if np.count_nonzero(remote_computed_blocks) == 0:
                    batch_remote_computed_blocks = []

            if batch_prefill_block_rank_id:
                max_len = max(len(x) for x in batch_prefill_block_rank_id)
                batch_prefill_block_rank_id = [
                    np.pad(arr, (0, max_len - len(arr)), constant_values=-1)
                    for arr in batch_prefill_block_rank_id
                ]

            if all(e is None for e in batch_stop_strings):
                batch_stop_strings = None
            if all(e is None for e in batch_stop_token_ids):
                batch_stop_token_ids = None
            if has_sampling:
                batch_sampling_params = []
                for llm_request in llm_requests:
                    batch_sampling_params.append(llm_request.sampling_params[0])
                batch_sampling_params = np.array(batch_sampling_params, SAMPLING_DTYPE)
            if enable_splitfuse:
                batch_seq_len = split_end_pos - split_start_pos
                total_seq_num = sum(batch_seq_len)
        else:
            batch_max_output_lens = []
            batch_response_format = []
            batch_dp_rank_ids = []
            for llm_request in llm_requests:
                num_sequences = len(llm_request.sequences.keys())
                batch_max_output_lens.extend([llm_request.max_new_tokens] * num_sequences)
                # 收集 response_format（每个 sequence 都需要相同的 response_format）
                batch_response_format.extend([llm_request.response_format] * num_sequences)
                batch_dp_rank_ids.append(llm_request.dp_rank_id)
                if llm_request.sp_tokens is not None:
                    batch_sp_tokens.append(llm_request.sp_tokens)
                    batch_sp_rank_ids.append(llm_request.sp_rank_id)
                    batch_is_append_block.append(llm_request.is_append_block)
                    batch_block_rank_id.append(llm_request.block_rank_id)

        batch_max_output_lens = np.array(batch_max_output_lens, dtype=np.int64)
        if req_block_tables is not None:
            block_tables = req_block_tables
        else:
            block_tables = InputMetadata.form_block_tables(llm_requests)

        return cls(
            batch_size=batch_size,
            batch_request_ids=batch_request_ids,
            batch_sequence_ids=batch_sequence_ids,
            batch_max_output_lens=batch_max_output_lens,
            block_tables=block_tables,
            reserved_sequence_ids=reserved_sequence_ids,
            input_ids=np.asarray(input_ids_list, dtype=np.int64),
            total_seq_num=total_seq_num,
            batch_sampling_params=batch_sampling_params,
            batch_seq_len=np.asarray(batch_seq_len),
            max_block_size=max_block_size,
            is_prefill=is_prefill,
            has_sampling=has_sampling,
            batch_best_of=np.asarray(batch_best_of),
            batch_n=np.asarray(batch_n),
            batch_use_beam_search=np.asarray(batch_use_beam_search),
            batch_ignore_eos=np.asarray(batch_ignore_eos),
            batch_include_stop=np.asarray(batch_include_stop),
            batch_logprobs=np.asarray(batch_logprobs),
            batch_seeds=np.asarray(batch_seeds),
            batch_skip_special_tokens=np.asarray(batch_skip_special_tokens),
            batch_stop_strings=batch_stop_strings,
            batch_stop_token_ids=batch_stop_token_ids,
            adapter_ids=adapter_ids,
            batch_dp_rank_ids=np.asarray(batch_dp_rank_ids) if batch_dp_rank_ids else None,
            sp_tokens=np.asarray(batch_sp_tokens) if batch_sp_tokens else None,
            sp_rank_id=np.asarray(batch_sp_rank_ids) if batch_sp_rank_ids else None,
            prefill_block_rank_id=np.asarray(batch_prefill_block_rank_id) if batch_prefill_block_rank_id else None,
            is_append_block=np.asarray(batch_is_append_block) if batch_is_append_block else None,
            block_rank_id=np.asarray(batch_block_rank_id) if batch_block_rank_id else None,
            is_mix=is_mix,
            split_start_position=split_start_pos,
            split_end_position=split_end_pos,
            batch_last_prompt=batch_last_prompt,
            batch_response_format=batch_response_format,
            batch_is_prefill=batch_is_prefill,
            computed_blocks=np.asarray(batch_computed_blocks) if batch_computed_blocks else None,
            remote_computed_blocks=np.asarray(batch_remote_computed_blocks) if batch_remote_computed_blocks else None
        )
