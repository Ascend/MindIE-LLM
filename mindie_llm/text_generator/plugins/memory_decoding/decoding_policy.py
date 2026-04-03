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
from collections import defaultdict

import numpy as np

from .dynamic_decoding import DynamicBranch
from .tokens_knowledge_base_cache import TokensKnowledgeBaseCache
from ...utils.model_input import ModelInput
from ....modeling.backend_type import BackendType
from ....utils.env import ENV
from ....utils.log.logging import logger

# trie-tree 查询窗口默认长度
DEFAULT_BRANCH_LENGTH = 16

# 召回草稿时需要匹配的前缀长度
PREFIX_WINDOW_SIZE = 2

# 启动动态解码长度的batch size阈值
DYNAMIC_GAMMA_BS = 4
INITIAL_BRANCH_LENGTH = 3


class DecodingPolicy:
    def __init__(self, kwargs, infer_context, decode_length):
        self.infer_context = infer_context
        self.decode_length = decode_length
        self.device = kwargs.get('device')
        self.eos_token_id = kwargs.get('eos_token_id')
        self.last_accepted_token = {}
        self.dynamic_algo = kwargs.get('dynamic_algo', False)
        self.cache_size = kwargs.get('cache_size')
        self.max_gen_len = kwargs.get('max_gen_len')
        self.max_block_size = kwargs.get('max_block_size')

        # tokens_knowledge_base speculative decoding
        self.prefix_ids = [[] for _ in range(self.cache_size)]
        self.tokens_knowledge_base_cache = TokensKnowledgeBaseCache(dynamic_algo=self.dynamic_algo)
        if isinstance(self.eos_token_id, int):
            self.tokens_knowledge_base_cache.eos = (self.eos_token_id,)
        elif isinstance(self.eos_token_id, list) and self.eos_token_id:
            self.tokens_knowledge_base_cache.eos = tuple(self.eos_token_id)
        self.tokens_knowledge_base_cache.stop_words = {}  # stop_words
        self.tokens_knowledge_base_param = defaultdict(int)
        self.tokens_knowledge_base_param['branch_length'] = DEFAULT_BRANCH_LENGTH
        self.free_block = []
        self.next_draft = {}
        self.branch_length = INITIAL_BRANCH_LENGTH
        self.last_decoding_len = {}
        self.dynamic_branch = DynamicBranch(decode_length)

    @staticmethod
    def verify_single_draft_process(offset, draft_ids, next_token_ids, batch_index, next_token_pair):
        (next_tokens_indices, next_tokens_length) = next_token_pair
        all_accepted = True
        for i in range(offset + 1, offset + len(draft_ids)):
            if next_token_ids[i] == draft_ids[i - offset]:
                next_tokens_indices[batch_index].append(i)
                next_tokens_length[batch_index] += 1
            else:
                all_accepted = False
                next_tokens_indices[batch_index].append(i)
                next_tokens_length[batch_index] += 1
                break

        return all_accepted, next_tokens_indices, next_tokens_length

    def verify_single(self, next_token_ids, decoding_ids):
        next_tokens_indices = [[] for _ in range(len(decoding_ids))]
        next_tokens_length = [0 for _ in range(len(decoding_ids))]
        offset = 0
        for batch_index, decoding_id in enumerate(decoding_ids):
            if len(decoding_id) == 1:
                next_tokens_indices[batch_index].append(offset)
                next_tokens_length[batch_index] += 1
                offset += 1
            else:
                draft_ids = decoding_id[1:]
                next_tokens_indices[batch_index].append(offset)
                next_tokens_length[batch_index] += 1
                if next_token_ids[offset] != draft_ids[0]:
                    offset += len(decoding_id)
                    continue
                next_token_pair = (next_tokens_indices, next_tokens_length)
                all_accepted, next_tokens_indices, next_tokens_length = self.verify_single_draft_process(
                    offset, draft_ids, next_token_ids, batch_index, next_token_pair)
                if all_accepted:
                    next_tokens_indices[batch_index].append(offset + len(draft_ids))
                    next_tokens_length[batch_index] += 1
                offset += len(decoding_ids[batch_index])
        return next_tokens_indices, next_tokens_length

    def add_prompt_to_cache(self, model_inputs, cache_ids):
        batch_size = len(cache_ids)
        cumulative_seq_len = 0
        host_context_length = model_inputs.context_length
        host_input_ids = model_inputs.input_ids
        for i in range(batch_size):
            seq_len = host_context_length[i]
            start_idx = cumulative_seq_len
            end_idx = cumulative_seq_len + seq_len
            ids = host_input_ids[start_idx + 1: end_idx]
            if self.dynamic_algo:
                self.tokens_knowledge_base_cache.add(
                    ids, search_size=self.branch_length + 1,
                    pattern='input', use_batch=cache_ids[i])
            else:
                self.tokens_knowledge_base_cache.add(
                    ids, search_size=self.tokens_knowledge_base_param['branch_length'] + 1,
                    pattern='input', use_batch=cache_ids[i])
            self.prefix_ids[cache_ids[i]] = host_input_ids[start_idx: end_idx]
            cumulative_seq_len += seq_len

    def filter_out_eos(self, end_reason, cache_ids: np.ndarray):
        for i, reason in enumerate(end_reason):
            if reason != 0:
                self.tokens_knowledge_base_cache.output_add([], final=True, pattern='output',
                                                            use_batch=cache_ids[i])
        self.tokens_knowledge_base_cache.changed_input_prefix_trees.clear()

    def verify(self, next_token_ids, decoding_ids, cache_ids):
        next_tokens_indices, next_tokens_length = self.verify_single(next_token_ids, decoding_ids)

        for idx, cid in enumerate(cache_ids):
            self.last_accepted_token[cid] = next_tokens_length[idx]
            if self.dynamic_algo:
                self.last_decoding_len[cid] = len(decoding_ids[idx])

        return next_tokens_indices, np.array(next_tokens_length, dtype=np.int32)

    def get_remain_slot_static_process(self, cache_ids, decoding_lengths, input_data):
        half_length = math.floor(self.decode_length / 2)
        if len(cache_ids) > DYNAMIC_GAMMA_BS:
            for idx, cid in enumerate(cache_ids):
                if cid in self.last_accepted_token and self.last_accepted_token[cid] < half_length:
                    decoding_lengths[idx] = half_length

        for idx, cid in enumerate(cache_ids):
            remain_gen_len = self.max_gen_len - self.infer_context.get_output_len_count(cid)
            decoding_lengths[idx] = min(decoding_lengths[idx], remain_gen_len)

        host_block_table = input_data.block_tables
        host_slots = input_data.slots
        cur_block_idx = self.infer_context.get_last_block_idx(cache_ids)
        for idx, cur_slot in enumerate(host_slots):
            if (cur_block_idx[idx] + 1) == len(host_block_table[idx]):
                cur_block = host_block_table[idx][cur_block_idx[idx]]
                remain_slot = ((cur_block + 1) * self.max_block_size) - cur_slot
                remain_slot = max(remain_slot, 1)
                decoding_lengths[idx] = min(remain_slot, decoding_lengths[idx])

        return decoding_lengths

    def get_remain_slot(self, input_data, cache_ids):
        decoding_lengths = [self.decode_length] * len(cache_ids)
        if self.dynamic_algo:
            self.branch_length, decoding_lengths = self.dynamic_branch.dynamic_decoding(cache_ids, self.branch_length,
                                                                                        self.last_accepted_token,
                                                                                        self.last_decoding_len)
        else:
            decoding_lengths = self.get_remain_slot_static_process(cache_ids, decoding_lengths, input_data)
        return decoding_lengths

    def calc_decoding_info(self, input_data: ModelInput, cache_ids: np.ndarray):
        decoding_lengths = self.get_remain_slot(input_data, cache_ids)
        if self.dynamic_algo:
            branch_length = self.get_remain_slot(input_data, cache_ids)
        else:
            branch_length = self.tokens_knowledge_base_param.get('branch_length', DEFAULT_BRANCH_LENGTH)
        batch_indices = cache_ids.tolist()
        decoding_qids = []
        for batch_index in batch_indices:
            decoding_qids.append(self.prefix_ids[batch_index][-PREFIX_WINDOW_SIZE:])

        if self.dynamic_algo:
            decoding_ids, decoding_masks, hit_sizes = (self.tokens_knowledge_base_cache.
                                                       get_all_batch_draft(decoding_qids,
                                                                           batch_decoding_length=decoding_lengths,
                                                                           search_size=branch_length[0],
                                                                           indices=batch_indices))
        else:
            decoding_ids, decoding_masks, hit_sizes = (self.tokens_knowledge_base_cache.
                                                       get_all_batch_draft(decoding_qids,
                                                                           batch_decoding_length=decoding_lengths,
                                                                           search_size=branch_length,
                                                                           indices=batch_indices))
        return decoding_ids, decoding_masks, hit_sizes

    def calc_new_tensor(self, input_data, cache_ids, decoding_ids, decoding_masks, batch_indices):
        input_ids = []
        new_input_lengths = []
        for index, temp_decoding_ids in enumerate(decoding_ids):
            input_ids += temp_decoding_ids
            new_input_lengths.append(len(temp_decoding_ids) +
                                     self.infer_context.get_last_position_ids(batch_indices[index]))

        new_input_ids_tensor = np.asarray(input_ids, dtype=np.int64)
        new_input_lengths_tensor = np.asarray(new_input_lengths, dtype=np.int64)
        new_max_seq_len = np.max(new_input_lengths)

        position_ids = []
        for index, decoding_mask in enumerate(decoding_masks):
            position_ids += ((np.sum(decoding_mask, axis=1) +
                              self.infer_context.get_last_position_ids(batch_indices[index])) - 1).tolist()
        new_position_ids_tensor = np.asarray(position_ids, dtype=np.int64)

        host_block_table = input_data.block_tables

        host_slots = input_data.slots

        cur_block_idx = self.infer_context.get_last_block_idx(cache_ids)
        new_slots_length = np.sum([len(x) for x in decoding_ids])

        start = 0
        new_slots = np.empty(new_slots_length)
        for idx, start_slot in enumerate(host_slots):
            cur_block = host_block_table[idx][cur_block_idx[idx]]
            next_block_slot_count = 0

            if (start_slot + len(decoding_ids[idx])) > ((cur_block + 1) * self.max_block_size):
                next_block_slot_count = ((start_slot + len(decoding_ids[idx])) -
                                         ((cur_block + 1) * self.max_block_size))

            temp = np.arange(start_slot, start_slot + len(decoding_ids[idx]))
            if next_block_slot_count > 0:
                next_block = host_block_table[idx][cur_block_idx[idx] + 1]
                back_temp = np.arange(next_block * self.max_block_size,
                                      next_block * self.max_block_size + next_block_slot_count)
                temp[-next_block_slot_count:] = back_temp

            new_slots[start: start + len(decoding_ids[idx])] = temp
            start += len(decoding_ids[idx])
        res = (new_input_lengths_tensor, new_max_seq_len, new_input_ids_tensor, new_position_ids_tensor, new_slots)
        return res

    def update_infer_input(self, input_data: ModelInput, cache_ids: np.ndarray):
        batch_indices = cache_ids.tolist()
        decoding_ids, decoding_masks, _ = self.calc_decoding_info(input_data, cache_ids)

        new_input_lengths_tensor, new_max_seq_len, new_input_ids_tensor, new_position_ids_tensor, new_slots \
            = self.calc_new_tensor(input_data, cache_ids, decoding_ids, decoding_masks, batch_indices)
        new_model_inputs = ModelInput(input_ids=new_input_ids_tensor,
                                      position_ids=new_position_ids_tensor,
                                      block_tables=input_data.block_tables,
                                      slots=new_slots,
                                      context_length=new_input_lengths_tensor,
                                      cached_context_length=new_input_lengths_tensor,
                                      max_seq_len=new_max_seq_len,
                                      prefill_head_indices=input_data.prefill_head_indices,
                                      is_prefill=input_data.is_prefill)
        logger.debug(f'update_infer_input new_model_inputs.input_ids: {new_model_inputs.input_ids}')
        if ENV.framework_backend == BackendType.MS:
            new_model_inputs.slots = new_model_inputs.slots.astype(np.int32)
        return new_model_inputs, decoding_ids, decoding_masks

    def truncate_long_tokens(self, cache_ids, input_metadata, sampling_output, next_tokens_indices):
        output_token_len = self.infer_context.get_output_len_count(cache_ids)[sampling_output.repeating_indices]
        output_space_left = input_metadata.batch_max_output_lens[sampling_output.repeating_indices] - output_token_len
        for idx, token_indices in enumerate(next_tokens_indices):
            num_new_tokens = len(token_indices)
            if output_space_left[idx] < num_new_tokens:
                num_new_tokens = output_space_left[idx]
            next_tokens_indices[idx] = token_indices[:num_new_tokens]