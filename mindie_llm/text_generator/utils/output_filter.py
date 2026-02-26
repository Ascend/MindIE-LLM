# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Any, List, Union

import numpy as np
from numba import njit

from .tg_infer_context_store import TGInferContextStore
from .config import ResponseConfig, CacheConfig
from .input_metadata import InputMetadata
from .sampling_output import SamplingOutput
from .tg_decode_util import decode_one


@njit
def check_column_equals_numba(array, indices, value):
    result = np.empty(array.shape[0], dtype=np.bool_)
    for i in range(array.shape[0]):
        result[i] = (array[i, indices[i]] == value)
    return result


class OutputFilter:
    def __init__(self, cache_config: CacheConfig, tg_infer_context: TGInferContextStore, tokenizer: Any,
                 async_inference: bool = False):
        self.cache_config = cache_config
        self.tg_infer_context = tg_infer_context
        self.tokenizer = tokenizer
        self.async_inference = async_inference

        self.eos_token_id: List[Union[int, List[int]]] = self.cache_config.eos_token_id
        self.ignore_eos = self.cache_config.ignore_eos
        self.rank = self.cache_config.rank
        self.tokenizer_sliding_window_size = self.cache_config.tokenizer_sliding_window_size
        self.layerwise_disaggregated = tg_infer_context.context_params.layerwise_disaggregated
        self.layerwise_disaggregated_role_type = tg_infer_context.context_params.layerwise_disaggregated_role_type

        # Warm up: Numba JIT compilation is triggered on the first call and can make the first request slow.
        # Trigger compilation here with a minimal input to reduce first-request latency.
        check_column_equals_numba(np.zeros((2, 2), dtype=np.int64), np.zeros(2, dtype=np.int64), 0)

    def decode_one(self, input_tokens, skip_special_tokens):
        return decode_one(self.tokenizer, input_tokens, skip_special_tokens, self.tokenizer_sliding_window_size)

    def filter_by_async(self, cache_ids, filter_ids_arr, end_reason):
        async_eos_flags = self.tg_infer_context.get_once_end_flag(cache_ids)
        async_eos_idx = np.nonzero(async_eos_flags)[0]
        if async_eos_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, async_eos_idx)
        end_reason[async_eos_idx] = ResponseConfig.EOS
        return filter_ids_arr

    def filter_by_eos(self, cache_ids, next_token_ids, num_new_tokens, filter_ids_arr, end_reason):
        if len(self.eos_token_id) == 1 and isinstance(self.eos_token_id[0], int):
            checked = check_column_equals_numba(next_token_ids, num_new_tokens - 1, self.eos_token_id[0])
        else:
            output_len_count = self.tg_infer_context.get_output_len_count(cache_ids)
            num_seqs = len(next_token_ids)
            checked = np.array([False] * num_seqs)
            for eos in self.eos_token_id:
                if isinstance(eos, int):
                    checked |= next_token_ids[np.arange(num_seqs), num_new_tokens - 1] == eos
                else:
                    eos_length = len(eos)
                    res = []
                    for i, output_ids in enumerate(self.tg_infer_context.get_all_output_ids(cache_ids)):
                        input_ids_without_padding = np.concatenate([output_ids[:output_len_count[i]],
                                                                    next_token_ids[i]])
                        if len(input_ids_without_padding) < eos_length:
                            res.append(False)
                        else:
                            res.append(np.all(input_ids_without_padding[-eos_length:] == eos))
                    checked |= np.array(res)

        checked[self.tg_infer_context.get_ignore_eos(cache_ids)] = False
        eos_idx = np.nonzero(checked)[0]
        if eos_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, eos_idx)
        end_reason[eos_idx] = ResponseConfig.EOS
        return filter_ids_arr

    def filter_by_length(self, cache_ids, batch_max_output_lens, num_new_tokens, filter_ids_arr, end_reason):
        output_len_count = self.tg_infer_context.get_output_len_count(cache_ids)
        cached_seq_lens = self.tg_infer_context.get_seq_lens(cache_ids)
        exceed_seq_limit_idx = np.nonzero(
            cached_seq_lens + num_new_tokens >= self.cache_config.max_seq_len)[0]
        exceed_user_output_limit_idx = np.nonzero(
            output_len_count + num_new_tokens >= batch_max_output_lens)[0]
        exceed_global_output_limit_idx = np.nonzero(
            output_len_count + num_new_tokens >= self.cache_config.max_gen_len)[0]

        if exceed_seq_limit_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, exceed_seq_limit_idx)
        if exceed_user_output_limit_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, exceed_user_output_limit_idx)
        if exceed_global_output_limit_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, exceed_global_output_limit_idx)

        end_reason[exceed_seq_limit_idx] = ResponseConfig.REACH_MAX_SEQ_LEN
        end_reason[exceed_user_output_limit_idx] = ResponseConfig.REACH_MAX_OUTPUT_LEN
        end_reason[exceed_global_output_limit_idx] = ResponseConfig.REACH_MAX_OUTPUT_LEN
        return filter_ids_arr

    def filter_by_stop(self, cache_ids, next_token_ids, num_new_tokens, filter_ids_arr, end_reason):
        truncation_indices = np.zeros(len(cache_ids), dtype=np.int_)
        if (self.tg_infer_context.is_empty_string_stopping_criteria() and
                self.tg_infer_context.is_empty_stopping_criteria()):
            return filter_ids_arr, truncation_indices
        checked_strings = np.zeros(len(cache_ids), dtype=np.bool_)
        checked_ids = np.zeros(len(cache_ids), dtype=np.bool_)
        string_stopping_criteria = self.tg_infer_context.get_string_stopping_criteria(cache_ids)
        stopping_criteria = self.tg_infer_context.get_stopping_criteria(cache_ids)
        for i, cache_id in enumerate(cache_ids):
            new_token_len = num_new_tokens[i]
            string_stopping_criterion = string_stopping_criteria[i]
            stopping_criterion = stopping_criteria[i]
            switch_stop_string = string_stopping_criterion is not None
            switch_stop_id = stopping_criterion is not None

            if switch_stop_string or switch_stop_id:
                cur_len = 0
                for len_tmp in range(new_token_len):
                    total_len = self.tg_infer_context.get_output_len_count(cache_id)
                    cur_len = len_tmp + 1
                    new_token_ids = np.concatenate(
                        [self.tg_infer_context.get_all_output_ids(cache_id)[:total_len], next_token_ids[i][:cur_len]])
                    new_token = self.decode_one(new_token_ids,
                        bool(self.tg_infer_context.get_skip_special_tokens(cache_id)))
                    cached_output_text = self.tg_infer_context.append_and_return_output_text(cache_id, new_token)

                    if switch_stop_string:
                        truncation_idx = string_stopping_criterion(cached_output_text, new_token,
                                                    include_stop=self.tg_infer_context.get_include_stop(cache_id))
                        if truncation_idx is not None:
                            checked_strings[i] = True
                            truncation_indices[i] = truncation_idx
                            break

                    if switch_stop_id:
                        checked_ids[i] = stopping_criterion(new_token_ids)
                        if checked_ids[i]:
                            if not self.tg_infer_context.get_include_stop(cache_id):
                                truncation_indices[i] = -len(new_token)
                            break
                if cur_len == 0:
                    raise RuntimeError('Empty `token_ids` generated!')
                num_new_tokens[i] = cur_len
        stop_strings_idx = np.nonzero(checked_strings)[0]
        stop_ids_idx = np.nonzero(checked_ids)[0]
        if stop_ids_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, stop_ids_idx)
        if stop_strings_idx.size != 0:
            filter_ids_arr = np.union1d(filter_ids_arr, stop_strings_idx)
        end_reason[stop_ids_idx] = ResponseConfig.STOP_TOKEN_IDS
        end_reason[stop_strings_idx] = ResponseConfig.STOP_STRINGS
        return filter_ids_arr, truncation_indices

    def filter_finished_sequences(
        self,
        cache_ids: np.ndarray,
        metadata: InputMetadata,
        sampling_output: SamplingOutput
    ):
        filter_ids_arr = np.array([], dtype=np.int32)
        if sampling_output.repeating_indices is not None:
            cache_ids = cache_ids[sampling_output.repeating_indices]
            batch_max_output_lens = metadata.batch_max_output_lens[sampling_output.repeating_indices]
        else:
            batch_max_output_lens = metadata.batch_max_output_lens
        end_reason = np.zeros_like(cache_ids)
        if self.async_inference:
            filter_ids_arr = self.filter_by_async(cache_ids, filter_ids_arr, end_reason)
        is_layerwise_slave = self.layerwise_disaggregated and self.layerwise_disaggregated_role_type == "slave"
        if not self.ignore_eos and not is_layerwise_slave:
            filter_ids_arr = self.filter_by_eos(cache_ids, sampling_output.token_ids,
                                                sampling_output.num_new_tokens, filter_ids_arr, end_reason)
        filter_ids_arr, truncation_indices = self.filter_by_stop(
            cache_ids, sampling_output.token_ids, sampling_output.num_new_tokens, filter_ids_arr, end_reason)
        filter_ids_arr = self.filter_by_length(
            cache_ids, batch_max_output_lens, sampling_output.num_new_tokens, filter_ids_arr, end_reason)

        # 对于没有完成prefill的req，不可以清除后处理参数
        if metadata.is_mix is not None:
            idx = np.where(metadata.batch_last_prompt == 0)[0]
            unfinished_idx = np.isin(filter_ids_arr, idx)
            filter_ids_arr = filter_ids_arr[~unfinished_idx]

        if filter_ids_arr.shape[0] == 0:
            filter_ids_arr = None
        return end_reason, filter_ids_arr, truncation_indices