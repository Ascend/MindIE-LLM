# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from typing import Optional
from itertools import count
import math

import numpy as np

from .generation_metadata import GenerationParams
from .config import SAMPLING_DTYPE


class Request:
    counter = count()
    
    def __init__(
        self,
        req_id: int,
        seq_id: int,
        input_ids: np.ndarray,
        generation_params: GenerationParams,
        has_sampling: bool = False,
        sampling_params: Optional[np.ndarray] = None,
        max_placeholder_num: int = 0
    ):
        self.req_id = req_id
        self.reserved_seq_ids = []
        self.input_ids = input_ids

        self.adapter_id = generation_params.adapter_id
        self.best_of = generation_params.best_of
        self.has_logprobs = generation_params.logprobs
        self.ignore_eos = generation_params.ignore_eos
        self.include_stop_str_in_output = generation_params.include_stop_str_in_output
        self.length_penalty = generation_params.length_penalty
        self.max_new_tokens = generation_params.max_new_tokens
        self.n = generation_params.n
        self.seed = generation_params.seed
        self.skip_special_tokens = generation_params.skip_special_tokens
        self.stop_strings = generation_params.stop_strings
        self.stop_token_ids = generation_params.stop_token_ids
        self.use_beam_search = generation_params.use_beam_search
        self.response_format = generation_params.response_format

        self.block_tables = None
        self.has_sampling = has_sampling
        self.sampling_params = sampling_params
        self.input_length = np.size(self.input_ids)
        self.output_len = 0
        self.dp_rank_id = 0
        self.num_sequence_blocks = 0

        # max_placeholder_num contains num_new_tokens
        self.max_placeholder_num = max_placeholder_num
        
        # kvp or cp with mtp
        self.sp_tokens = None
        self.sp_rank_id = None
        self.prefill_block_rank_id = None
        self.block_rank_id = None
        self.is_append_block = None

        # prefix cache
        self.remote_computed_blocks = 0
        self.computed_blocks = 0

        # mix
        self.split_start_position = 0
        self.split_end_position = self.input_length
        self.last_prompt = True

        self.sequences = {seq_id: Sequence(seq_id)}
        self.completed = []

    def __repr__(self):
        return (f"Request(req_id={self.req_id}, "
                f"input_ids={self.input_ids[:50]}, "
                f"max_new_tokens={self.max_new_tokens}, "
                f"max_placeholder_num={self.max_placeholder_num}), "
                f"input_length={self.input_length}), "
                f"output_len={self.output_len})")

    @classmethod
    def from_warmup(
        cls,
        input_len: int,
        max_output_len: int,
        max_placeholder_num: int = 1,
        warmup_topk_size: int = 1000,
        enable_warmup_sampling: bool = True
    ):
        req_id = next(Request.counter)
        input_ids = np.ones(input_len, dtype=np.int64)
        sampling_params_ins = np.array([(1.1, 0.1, 0.1, 0.9, warmup_topk_size, 0.8, True, False)], SAMPLING_DTYPE)
        if not enable_warmup_sampling:
            sampling_params_ins = np.array([(1.0, 0.0, 0.0, 1.0, warmup_topk_size, 1.0, False, False)], SAMPLING_DTYPE)
        return cls(
            req_id,
            req_id,
            input_ids,
            GenerationParams(max_new_tokens=max_output_len, seed=0, ignore_eos=True),
            has_sampling=True,
            sampling_params=sampling_params_ins,
            max_placeholder_num=max_placeholder_num
        )

    @classmethod
    def request_from_token(cls, input_ids, sampling_params, generation_params=None, req_id=0, seq_id=None):
        if seq_id is None:
            seq_id = req_id
        input_ids = np.array(input_ids, dtype=np.int64)
        has_sampling = True if sampling_params is not None else False
        return cls(req_id, seq_id, input_ids, generation_params, has_sampling, sampling_params)

    def update_with_parallel_strategy(
        self,
        dp_rank_id: int,
        scp_size: int,
        block_size: int,
        num_new_token: int = 0,
        is_prefill: bool = True
    ):
        self.dp_rank_id = dp_rank_id

        if scp_size > 1:
            if is_prefill:
                self.sp_tokens, self.sp_rank_id = self._compute_sp_tokens_and_rank_id(block_size, scp_size)
            else:
                self.is_append_block = False
                num_empty_slot = self._get_num_empty_slots(self.sp_rank_id)
                if (self.max_placeholder_num > num_empty_slot):
                    next_sp_rank = (self.sp_rank_id + 1) % scp_size
                    reserved_slots = self.max_placeholder_num - num_empty_slot
                    next_rank_num_empty_slot = self._get_num_empty_slots(next_sp_rank)
                    if (next_rank_num_empty_slot < reserved_slots):
                        self.is_append_block = True
                        self.block_rank_id = next_sp_rank

                if num_empty_slot >= num_new_token:
                    self.sp_tokens[self.sp_rank_id] += num_new_token
                else:
                    self.sp_tokens[self.sp_rank_id] += num_empty_slot
                    self.sp_rank_id = (self.sp_rank_id + 1) % scp_size
                    self.sp_tokens[self.sp_rank_id] += (num_new_token - num_empty_slot)

    def update_with_features(
        self,
        is_mix_model: bool,
        scp_size: int,
        is_prefill: bool = True
    ):
        if is_prefill:
            if scp_size > 1:
                self.computed_blocks = [0] * scp_size
                self.remote_computed_blocks = [0] * scp_size
            if is_mix_model:
                self.split_start_position = 0
                self.split_end_position = self.input_length
                self.last_prompt = True
        else:
            if is_mix_model:
                self.split_start_position = self.input_length + self.output_len + self.max_placeholder_num - 1
                self.split_end_position = self.input_length + self.output_len + self.max_placeholder_num
                self.last_prompt = True

    def update_block_table(
        self,
        scp_size: int,
        block_size: int,
        is_prefill: bool = True
    ):
        if scp_size > 1:
            if is_prefill:
                required_block_num = self._get_block_counts(block_size, is_prefill=is_prefill)
                block_nums = (self.sp_tokens + block_size - 1) // block_size
                max_blocks = block_nums.max()
                self.block_tables = np.full((scp_size, required_block_num), -1, dtype=np.int32)
                for i in range(scp_size):
                    self.block_tables[i, :block_nums[i]] = 0

                self.prefill_block_rank_id = self._build_prefill_block_rank_id(max_blocks, scp_size)

            else:
                if self.is_append_block and self.block_rank_id is not None:
                    row = self.block_tables[self.block_rank_id]
                    pad_indices = np.where(row == -1)[0]
                    row[pad_indices[0]] = 0
        else:
            required_block_num = self._get_block_counts(block_size, is_prefill=is_prefill)
            self.block_tables = np.zeros(required_block_num, dtype=np.int32)
    
    # this method only called in prefill stage
    def build(
        self,
        dp_rank_id: int,
        scp_size: int,
        block_size: int,
        is_mix_model: bool,
    ):
        self.update_with_parallel_strategy(dp_rank_id, scp_size, block_size)
        self.update_with_features(is_mix_model, scp_size)
        self.update_block_table(scp_size, block_size)

    # this method only called in decode stage
    def step(
        self,
        num_new_token: int,
        scp_size: int,
        block_size: int,
        is_mix_model: bool
    ):
        self.update_with_parallel_strategy(self.dp_rank_id, scp_size, block_size, num_new_token, False)
        self.update_with_features(is_mix_model, scp_size, False)
        self.update_block_table(scp_size, block_size, False)  
        self.output_len += num_new_token

    def _compute_sp_tokens_and_rank_id(self, block_size, scp_size):
        sp_tokens = np.zeros(scp_size, dtype=np.int32)
        rank_idx = 0
        remaining = self.input_length
        block_count = 0

        while remaining > 0:
            tokens_in_block = min(remaining, block_size)

            is_last_block = (remaining <= block_size)
            has_used_all_ranks = (block_count >= scp_size)

            if is_last_block and has_used_all_ranks:
                rank_idx = scp_size - 1

            sp_tokens[rank_idx] += tokens_in_block

            remaining -= tokens_in_block
            block_count += 1

            if remaining > 0:
                if not (is_last_block and has_used_all_ranks):
                    rank_idx = (rank_idx + 1) % scp_size
        sp_rank_id = rank_idx
        return sp_tokens, sp_rank_id

    def _build_prefill_block_rank_id(self, max_blocks, scp_size):
        prefill_block_rank_id = []
        for col in range(max_blocks):
            for rank in range(scp_size):
                if self.block_tables[rank, col] != -1:
                    prefill_block_rank_id.append(rank)
        return np.array(prefill_block_rank_id, dtype=np.int32)

    def _get_block_counts(self, block_size: int, is_prefill: bool):
        if is_prefill:
            total_length = self.input_length
        else:
            total_length = self.input_length + self.output_len + self.max_placeholder_num
        return math.ceil(total_length / block_size)
    
    def _get_num_empty_slots(self, sp_rank_id, block_size: int = 128):
        used_slot_num = self.sp_tokens[sp_rank_id]
        block_count = np.count_nonzero(self.block_tables[sp_rank_id] != -1)
        return block_count * block_size - used_slot_num


class Sequence:
    def __init__(self, seq_id):
        self.seq_id = seq_id

        self.block_tables = None
        self.kv_length = 0

        self.cumulative_logprobs = 0
        self.eos_flag = 0
        self.logprobs = []
        self.out_token_list = []
        self.top_logprobs = []
        self.truncation_indices = []