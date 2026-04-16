# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np
import torch

from ...utils.model_input import ModelInput
from ...utils.input_metadata import SIMULATE_SEQUENCE_ID
from ....utils.tensor import tensor_backend, op


class CacheEngine:
    def __init__(self, infer_context, num_speculative_tokens):
        self.infer_context = infer_context
        self.num_speculative_tokens = num_speculative_tokens

    def cache_update(self, cached_ids, hidden_states, sampling_output, is_prefill, token_alias_len):
        batch_size = len(cached_ids)
        start_idx = 0
        hidden_states_cpu = hidden_states.cpu()
        for i in range(batch_size):
            token_len = sampling_output.num_new_tokens[i]
            self.infer_context.set_mtp_last_token_num(cached_ids[i], token_len)
            self.infer_context.set_mtp_hidden_states_prefix(
                cached_ids[i],
                token_alias_len[i],
                hidden_states_cpu[start_idx : start_idx + token_alias_len[i]],
            )
            start_idx += token_alias_len[i]


class DecodingPolicy:
    def __init__(
        self,
        generator_backend,
        infer_context,
        model_wrapper,
        num_speculative_tokens,
        device_and_type,
        plugin_data_param,
        model_role,
        eos_token_id,
        max_block_size,
    ):
        self.infer_context = infer_context
        self.plugin_data_param = plugin_data_param
        self.generator_backend = generator_backend
        self.num_speculative_tokens = num_speculative_tokens
        (self.kv_device, self.kv_dtype) = device_and_type
        self.mtp_cache = CacheEngine(self.infer_context, self.num_speculative_tokens)
        self.model_wrapper = model_wrapper
        self.eos_token_id = eos_token_id
        self.model_role = model_role
        self.max_block_size = max_block_size

    @staticmethod
    def repeat_sample_param(param_tensor, tokens_num_per_batch):
        if param_tensor is None:
            return None
        result = []
        for tensor_tmp, n in zip(param_tensor, tokens_num_per_batch):
            repeat_tensor = tensor_tmp.repeat(n, 1)
            result.append(repeat_tensor)
        out_tensor = op.cat(result, dim=0)
        return out_tensor

    @staticmethod
    def verify_greedy_one_batch(verify_guess_tokens, next_guess_tokens):
        gg = 0
        for eg, guess_tokens in enumerate(verify_guess_tokens):
            correct = next_guess_tokens[eg]
            guess = guess_tokens
            if guess != correct:
                break
            gg += 1
        return gg

    def handle_input(self, model_inputs, input_metadata, cached_idx, hit_mask=None):
        q_len = None
        attn_mask = None
        self.plugin_data_param.mtp_model_inputs = None
        self.plugin_data_param.hidden_states = None
        if input_metadata.is_prefill:
            start_idx = 0
            slot_list = []
            for i in range(input_metadata.batch_size):
                seq_len = input_metadata.batch_seq_len[i]
                tmp_slot = model_inputs.slots[start_idx : start_idx + seq_len]
                slot_list.append(tmp_slot)
                if self.infer_context.spcp_parallel_info.scp_size > 1:
                    self.infer_context.set_mtp_seq_block_rank_id(
                        cached_idx[i],
                        len(input_metadata.prefill_block_rank_id[i]),
                        np.array(input_metadata.prefill_block_rank_id[i]),
                    )
                start_idx += seq_len
            if self.infer_context.spcp_parallel_info.scp_size > 1:
                self.infer_context.set_mtp_last_rank(cached_idx, input_metadata.sp_rank_id)
        else:
            model_inputs, q_len, attn_mask = self.decode_model_input_update(
                model_inputs, input_metadata, cached_idx, hit_mask=hit_mask
            )
        return model_inputs, q_len, attn_mask

    def handle_sampling(self, sampling_metadata):
        if sampling_metadata is None or sampling_metadata.is_prefill:
            return
        batch_size = len(sampling_metadata.all_sequence_ids)
        logits_num_per_batch = [self.num_speculative_tokens + 1] * batch_size
        top_k_idx = self.repeat_sample_param(sampling_metadata.top_k_idx, logits_num_per_batch)
        sampling_metadata.top_k_idx = top_k_idx
        top_k_disabled_mask = self.repeat_sample_param(sampling_metadata.top_k_disabled_mask, logits_num_per_batch)
        sampling_metadata.top_k_disabled_mask = top_k_disabled_mask
        repetition_penalty = self.repeat_sample_param(sampling_metadata.repetition_penalty, logits_num_per_batch)
        sampling_metadata.repetition_penalty = repetition_penalty
        frequency_penalty = self.repeat_sample_param(sampling_metadata.frequency_penalty, logits_num_per_batch)
        sampling_metadata.frequency_penalty = frequency_penalty
        presence_penalty = self.repeat_sample_param(sampling_metadata.presence_penalty, logits_num_per_batch)
        sampling_metadata.presence_penalty = presence_penalty
        temperature = self.repeat_sample_param(sampling_metadata.temperature, logits_num_per_batch)
        sampling_metadata.temperature = temperature
        if self.generator_backend.backend_type == "torch":
            top_p = self.repeat_sample_param(sampling_metadata.top_p, logits_num_per_batch)
            sampling_metadata.top_p = top_p
            if sampling_metadata.do_sample_tensor is not None:
                do_sample_tensor = self.repeat_sample_param(
                    sampling_metadata.do_sample_tensor, logits_num_per_batch
                ).squeeze(1)
                sampling_metadata.do_sample_tensor = do_sample_tensor
            if sampling_metadata.random_number_generators is not None:
                sampling_metadata.random_number_generators = [
                    gen
                    for gen in sampling_metadata.random_number_generators
                    for _ in range(self.num_speculative_tokens + 1)
                ]

    def decode_model_input_update(self, model_inputs, input_metadata, cached_idx, hit_mask=None):
        # 小模型使用的model_inputs构造
        mtp_model_inputs, hidden_states = self.get_mtp_draft_model_inputs(
            model_inputs, input_metadata, cached_idx, hit_mask=hit_mask
        )
        self.plugin_data_param.mtp_model_inputs = mtp_model_inputs
        self.plugin_data_param.hidden_states = hidden_states
        # 大模型使用的model_inputs构造
        model_inputs_new = self.get_mtp_main_model_inputs(model_inputs, input_metadata, cached_idx)
        # 大小模型复用q_lens和mask
        q_lens = [self.num_speculative_tokens + 1] * input_metadata.batch_size
        attn_mask = None
        return model_inputs_new, q_lens, attn_mask

    def get_mtp_draft_model_inputs(self, model_inputs, input_metadata, cached_idx, hit_mask=None):
        if self.model_role == "standard":
            model_inputs_mtp, hidden_states = self.get_mtp_draft_model_inputs_standard(
                model_inputs, input_metadata, cached_idx, hit_mask=hit_mask
            )
        else:
            model_inputs_mtp, hidden_states = self.get_mtp_draft_model_inputs_pd(
                model_inputs, input_metadata, cached_idx, hit_mask=hit_mask
            )
        return model_inputs_mtp, hidden_states

    def get_mtp_draft_model_inputs_pd(self, model_inputs, input_metadata, cached_idx, hit_mask=None):
        final_block = self.generator_backend.cache_pool.kvcache_settings.num_npu_blocks - 1
        dummy_slot = final_block * input_metadata.max_block_size

        if input_metadata.is_dummy_batch:
            block_tables = np.array([[final_block]], dtype=np.int32)
            mtp_model_inputs = ModelInput(
                input_ids=np.zeros(self.num_speculative_tokens + 1, dtype=np.int64),
                position_ids=np.arange(self.num_speculative_tokens + 1, dtype=np.int64),
                block_tables=block_tables,
                slots=np.array([dummy_slot] * (self.num_speculative_tokens * 2), dtype=np.int64),
                context_length=np.array([self.num_speculative_tokens + 1], dtype=np.int64),
                cached_context_length=np.array([self.num_speculative_tokens + 1], dtype=np.int64),
                max_seq_len=4,
                prefill_head_indices=np.array([0], dtype=np.int64),
                is_prefill=input_metadata.is_prefill,
                adapter_ids=input_metadata.adapter_ids,
                dp_rank_ids=input_metadata.batch_dp_rank_ids,
                sp_tokens=input_metadata.sp_tokens,
            )
            hidden_states = self.get_input_hidden_states([0])
        elif np.any(self.infer_context.get_output_len_count(cached_idx) == 0):
            mtp_model_inputs, hidden_states = self.get_mtp_draft_model_inputs_standard(
                model_inputs, input_metadata, cached_idx, hit_mask
            )
            if hit_mask is None:
                decode_mask = self.infer_context.get_output_len_count(cached_idx) != 0
                decode_mask_for_slots = np.repeat(decode_mask, self.num_speculative_tokens * 2)
                # 修改slots
                mtp_model_inputs.slots = np.where(decode_mask_for_slots, mtp_model_inputs.slots, dummy_slot)
                # 修改blocktable
                mtp_model_inputs.block_tables[~decode_mask, :] = final_block
            else:
                first_decoding_mask = (self.infer_context.get_output_len_count(cached_idx) == 0) & ~hit_mask
                first_decoding_mask_for_slots = np.repeat(first_decoding_mask, self.num_speculative_tokens * 2)
                mtp_model_inputs.slots[first_decoding_mask_for_slots] = dummy_slot
                mtp_model_inputs.block_tables[first_decoding_mask, :] = final_block

            hidden_states = self.get_input_hidden_states(cached_idx)
        else:
            mtp_model_inputs, hidden_states = self.get_mtp_draft_model_inputs_standard(
                model_inputs, input_metadata, cached_idx, hit_mask
            )
        return mtp_model_inputs, hidden_states

    def sp_token_and_slot_calc_by_context_length(
        self, context_length, block_rank_id, block_tables, slots_num_per_batch
    ):
        scp_rank = self.infer_context.spcp_parallel_info.scp_rank
        block_table = [row[row != -1] for row in block_tables]
        pos = np.arange(context_length)
        block_index = pos // self.max_block_size
        block_offset = pos % self.max_block_size
        rank_id = block_rank_id[block_index]
        is_current = rank_id == scp_rank

        block_to_cache = np.zeros(len(block_rank_id), dtype=int)
        draft_sp_token = np.zeros(len(block_table), dtype=int)
        for rank, _ in enumerate(block_table):
            mask = block_rank_id == rank
            block_to_cache[mask] = block_table[rank]
            draft_sp_token[rank] = np.sum((rank_id == rank))

        slots_ids = self.infer_context.block_to_slots(block_to_cache[block_index], block_offset)
        final_slots = np.where(is_current, slots_ids, -1)

        result = final_slots[-slots_num_per_batch:]
        return draft_sp_token, result

    def get_mtp_draft_model_inputs_standard(self, model_inputs, input_metadata, cached_idx, hit_mask=None):
        batch_size = input_metadata.batch_size
        batch_block_tables = input_metadata.batch_block_tables

        cached_block_rank_id = None
        # 首先获取上一轮的生成长度信息
        cached_last_token_num = self.infer_context.get_mtp_last_token_num(cached_idx)
        if self.infer_context.spcp_parallel_info.scp_size > 1:
            cached_block_rank_id = self.infer_context.get_mtp_seq_block_rank_id(cached_idx)

        speculative_len = self.num_speculative_tokens + 1
        decode_token_num_pre_batch = batch_size * speculative_len
        new_input_ids = np.zeros(decode_token_num_pre_batch, dtype=np.int64)
        new_position_ids = np.zeros(decode_token_num_pre_batch, dtype=np.int32)

        new_context_length = np.zeros(batch_size, dtype=np.int32)
        prefill_head_indices_new = np.zeros(batch_size, dtype=np.int32)
        slots_num_per_batch = 2 * self.num_speculative_tokens
        new_slots = np.full(batch_size * slots_num_per_batch, -1, dtype=np.int32)
        start_idx = 0
        draft_sp_tokens = np.zeros((batch_size, self.infer_context.spcp_parallel_info.scp_size), dtype=np.int32)
        is_need_mask = [0] * batch_size
        for i in range(batch_size):
            # mtp场景下，虚推需要特殊处理，构造对应特殊的slots并置为-1不占用正常请求slots
            if (
                input_metadata.all_sequence_ids is not None
                and input_metadata.all_sequence_ids[i] == SIMULATE_SEQUENCE_ID
            ):
                new_slots[i * slots_num_per_batch : (i + 1) * slots_num_per_batch] = -1
                new_context_length[i] = self.num_speculative_tokens
                new_position_ids[start_idx : start_idx + speculative_len] = np.arange(speculative_len)
                prefill_head_indices_new[i] = start_idx + speculative_len - 1
                start_idx += speculative_len
                continue
            last_token_num = cached_last_token_num[i]
            if last_token_num == 0:  # 对于PD分离场景，第一次decode无法获取到prefill输出的token数，因此这里保护成1
                last_token_num = 1
            if hit_mask is None or not hit_mask[i]:
                if i >= len(cached_idx):
                    message = f"Index of cached_idx is out of range (i={i}, len={len(cached_idx)})"
                    raise ValueError(message)
                all_input_id = self.infer_context.get_all_input_ids(cached_idx[i])
                all_input_len = self.infer_context.get_seq_lens(cached_idx[i])
                new_input_ids[start_idx : start_idx + last_token_num] = all_input_id[
                    all_input_len - last_token_num : all_input_len
                ]
                context_length_first_mtp = model_inputs.context_length[i] - last_token_num + self.num_speculative_tokens
                context_length_for_slot = (self.num_speculative_tokens - 1) + context_length_first_mtp
                new_context_length[i] = context_length_first_mtp
                block_tables = batch_block_tables[i]
                new_position_ids[start_idx : start_idx + speculative_len] = np.arange(
                    model_inputs.position_ids[i] - last_token_num + 1,
                    model_inputs.position_ids[i] - last_token_num + 1 + speculative_len,
                )

                if self.infer_context.spcp_parallel_info.scp_size == 1:
                    new_slots[i * slots_num_per_batch : (i + 1) * slots_num_per_batch] = (
                        self.infer_context.block_table_to_slots(block_tables).reshape(-1)[
                            context_length_for_slot - slots_num_per_batch : context_length_for_slot
                        ]
                    )
                else:
                    (
                        draft_sp_tokens[i],
                        new_slots[i * slots_num_per_batch : (i + 1) * slots_num_per_batch],
                    ) = self.sp_token_and_slot_calc_by_context_length(
                        model_inputs.position_ids[i] - last_token_num + 1 + self.num_speculative_tokens,
                        cached_block_rank_id[i],
                        input_metadata.block_tables[i],
                        slots_num_per_batch,
                    )
                    is_need_mask[i] = int(new_slots[(i + 1) * slots_num_per_batch - 1] != -1)
                # 新的lmhead indices是可接受的最后一个长度， 但是每次start_idx 的累加要是 MTP + 1
                prefill_head_indices_new[i] = start_idx + last_token_num - 1
            else:
                cur_pos = model_inputs.position_ids[i]
                new_position_ids[start_idx : start_idx + speculative_len] = np.arange(
                    cur_pos + 1, cur_pos + speculative_len + 1
                )
                block_tables = batch_block_tables[i]
                candidate_slots = self.infer_context.block_table_to_slots(block_tables).reshape(-1)
                if self.infer_context.spcp_parallel_info.scp_size == 1:
                    new_slots[i * slots_num_per_batch : (i + 1) * slots_num_per_batch] = candidate_slots[
                        model_inputs.context_length[i] - 1 : model_inputs.context_length[i] - 1 + slots_num_per_batch
                    ]
                else:
                    (
                        draft_sp_tokens[i],
                        new_slots[i * slots_num_per_batch : (i + 1) * slots_num_per_batch],
                    ) = self.sp_token_and_slot_calc_by_context_length(
                        model_inputs.position_ids[i] + 1 + self.num_speculative_tokens,
                        cached_block_rank_id[i],
                        input_metadata.block_tables[i],
                        slots_num_per_batch,
                    )
                    is_need_mask[i] = int(new_slots[(i + 1) * slots_num_per_batch - 1] != -1)
                new_context_length[i] = model_inputs.context_length[i] + self.num_speculative_tokens
                prefill_head_indices_new[i] = start_idx + speculative_len - 1
            start_idx += speculative_len

        new_max_seq_len = max(new_context_length)

        batch_is_need_mask = None
        if self.infer_context.spcp_parallel_info.scp_size > 0:
            batch_is_need_mask = is_need_mask
        mtp_model_inputs = ModelInput(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            block_tables=model_inputs.block_tables.copy(),
            slots=new_slots,
            context_length=new_context_length,
            cached_context_length=new_context_length,
            max_seq_len=new_max_seq_len,
            prefill_head_indices=prefill_head_indices_new,
            is_prefill=model_inputs.is_prefill,
            adapter_ids=model_inputs.adapter_ids,
            dp_rank_ids=model_inputs.dp_rank_ids,
            sp_tokens=draft_sp_tokens,
            is_need_mask=batch_is_need_mask,
        )

        # 从cache中获取hidden_states进行组batch拼接
        hidden_states = self.get_input_hidden_states(cached_idx)
        return mtp_model_inputs, hidden_states

    def get_mtp_main_model_inputs(self, model_inputs, input_metadata, cached_idx):
        batch_size = input_metadata.batch_size
        new_input_ids = np.zeros(batch_size * (self.num_speculative_tokens + 1), dtype=np.int64)
        new_position_ids = np.zeros(batch_size * (self.num_speculative_tokens + 1), dtype=np.int32)
        new_slots = np.full(batch_size * (self.num_speculative_tokens + 1), -1, dtype=np.int32)
        new_context_length = model_inputs.context_length + self.num_speculative_tokens
        new_max_seq_len = model_inputs.max_seq_len + self.num_speculative_tokens

        cached_block_rank_id = None
        if self.infer_context.spcp_parallel_info.scp_size > 1:
            cached_block_rank_id = self.infer_context.get_mtp_seq_block_rank_id(cached_idx)
        draft_sp_tokens = np.zeros((batch_size, self.infer_context.spcp_parallel_info.scp_size), dtype=np.int32)
        sp_tokens = None
        is_need_mask = [0] * batch_size
        if self.infer_context.spcp_parallel_info.scp_size > 1:
            row_indices = np.arange(model_inputs.sp_tokens.shape[0])
            sp_tokens = model_inputs.sp_tokens.copy()
            sp_tokens[row_indices, input_metadata.sp_rank_id] += self.num_speculative_tokens

        for i in range(batch_size):
            input_len_per_batch = self.num_speculative_tokens + 1
            # mtp场景下，虚推需要特殊处理，构造对应特殊的slots并置为-1不占用正常请求slots
            if (
                input_metadata.all_sequence_ids is not None
                and input_metadata.all_sequence_ids[i] == SIMULATE_SEQUENCE_ID
            ):
                new_slots[i * input_len_per_batch : (i + 1) * input_len_per_batch] = -1
                start_pos_ids = model_inputs.position_ids[i]
                tmp_pos_ids = np.arange(start_pos_ids, start_pos_ids + input_len_per_batch)
                new_position_ids[i * input_len_per_batch : (i + 1) * input_len_per_batch] = tmp_pos_ids
                continue
            new_input_ids[i * input_len_per_batch] = model_inputs.input_ids[i]
            block_tables = input_metadata.batch_block_tables[i]
            start_pos_ids = model_inputs.position_ids[i]
            tmp_pos_ids = np.arange(start_pos_ids, start_pos_ids + input_len_per_batch)
            new_position_ids[i * input_len_per_batch : (i + 1) * input_len_per_batch] = tmp_pos_ids

            if self.infer_context.spcp_parallel_info.scp_size == 1:
                end_idx = new_context_length[i]
                start_idx = model_inputs.context_length[i] - 1
                tmp_slots = self.infer_context.block_table_to_slots(block_tables).reshape(-1)[start_idx:end_idx]

                new_slots[i * input_len_per_batch : (i + 1) * input_len_per_batch] = tmp_slots
            else:
                (
                    draft_sp_tokens[i],
                    new_slots[i * input_len_per_batch : (i + 1) * input_len_per_batch],
                ) = self.sp_token_and_slot_calc_by_context_length(
                    new_context_length[i],
                    cached_block_rank_id[i],
                    input_metadata.block_tables[i],
                    input_len_per_batch,
                )
                is_need_mask[i] = int(new_slots[(i + 1) * input_len_per_batch - 1] != -1)

        batch_is_need_mask = None
        if self.infer_context.spcp_parallel_info.scp_size > 0:
            batch_is_need_mask = is_need_mask
        new_model_inputs = ModelInput(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            block_tables=model_inputs.block_tables,
            slots=new_slots,
            context_length=new_context_length,
            cached_context_length=new_context_length.copy(),
            max_seq_len=new_max_seq_len,
            prefill_head_indices=model_inputs.prefill_head_indices,
            is_prefill=model_inputs.is_prefill,
            adapter_ids=model_inputs.adapter_ids,
            dp_rank_ids=model_inputs.dp_rank_ids,
            sp_tokens=draft_sp_tokens,
            is_need_mask=batch_is_need_mask,
        )

        return new_model_inputs

    def get_input_hidden_states(self, cached_idx):
        # 首先获取上一轮的生成长度信息
        hidden_states_list = []
        for _, cache_id in enumerate(cached_idx):
            select_hidden_states = self.infer_context.get_mtp_hidden_states(cache_id)
            if len(select_hidden_states) < (self.num_speculative_tokens + 1):
                message = (
                    f"The select_hidden_states {len(select_hidden_states)}"
                    f" is less than num_speculative_tokens {self.num_speculative_tokens + 1}"
                )
                raise ValueError(message)
            slice_hidden_states = select_hidden_states[: self.num_speculative_tokens + 1]
            hidden_states_list.append(slice_hidden_states)
        hidden_states_tensor = torch.cat(hidden_states_list, dim=0)
        return hidden_states_tensor

    def stop_criteria(self, sampling_output, output_space_left, next_tokens_indices):
        for idx, token_indices in enumerate(next_tokens_indices):
            num_new_tokens = len(token_indices)
            seq_token_ids = sampling_output.token_ids[token_indices]
            for i, token_id in enumerate(seq_token_ids):
                if isinstance(self.eos_token_id, int):
                    if token_id == self.eos_token_id:
                        num_new_tokens = i + 1
                        break
                elif isinstance(self.eos_token_id, list) and self.eos_token_id:
                    if token_id in self.eos_token_id:
                        num_new_tokens = i + 1
                        break

            if output_space_left[idx] <= 0:
                output_space_left[idx] = 1
            if output_space_left[idx] < num_new_tokens:
                num_new_tokens = output_space_left[idx]
            next_tokens_indices[idx] = token_indices[:num_new_tokens]

    def all_token_ids_padding(
        self,
        sampling_metadata,
        next_guess_logits_num_per_batch,
        batch_size,
        draft_tokens,
    ):
        if sampling_metadata.all_token_ids is None:
            return None
        max_len = max(next_guess_logits_num_per_batch)
        total_logits_num = sum(next_guess_logits_num_per_batch)

        input_ids_len = tensor_backend.shape(sampling_metadata.all_token_ids, -1)
        input_ids_pad = np.zeros((total_logits_num, input_ids_len + max_len), dtype=np.int64)
        index = 0
        for batch in range(batch_size):
            input_ids = np.expand_dims(
                tensor_backend.numpy(tensor_backend.cpu(sampling_metadata.all_token_ids[batch])),
                axis=0,
            )
            input_ids_pad[index : index + next_guess_logits_num_per_batch[batch], :input_ids_len] = np.repeat(
                input_ids, next_guess_logits_num_per_batch[batch], axis=0
            )
            input_ids_pad[index : index + next_guess_logits_num_per_batch[batch], input_ids_len:] = -1
            guess_index = index + 1
            index += next_guess_logits_num_per_batch[batch]
            draft_tokens_per_batch = draft_tokens[
                batch * self.num_speculative_tokens : (batch + 1) * self.num_speculative_tokens
            ]
            if len(draft_tokens_per_batch) >= 1:
                guess_set = draft_tokens_per_batch
                for idx in range(len(guess_set)):
                    input_ids_pad[guess_index, input_ids_len : input_ids_len + idx + 1] = np.array(
                        list(guess_set[0 : idx + 1])
                    )
                    guess_index += 1
        input_ids_pad = tensor_backend.tensor(
            input_ids_pad,
            dtype=sampling_metadata.all_token_ids.dtype,
            device=self.kv_device,
        )
        return input_ids_pad
