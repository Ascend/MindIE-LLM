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

from .decoding_policy import DecodingPolicy
from ..plugin import Plugin
from ....utils.log.logging import logger
from ....utils.tensor import backend
from ....utils.env import ENV


NPU_STR = "npu"


class MtpPlugin(Plugin):
    def __init__(
        self,
        generator_backend,
        kvcache_settings,
        infer_context,
        plugin_data_param,
        **kwargs,
    ):
        super().__init__()
        self.pad_token_id = 0
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.kvcache_settings = kvcache_settings
        self.infer_context = infer_context

        kv_device = self.model_wrapper.device
        kv_dtype = self.kvcache_settings.dtype
        device_and_type = (kv_device, kv_dtype)
        self.plugin_data_param = plugin_data_param
        self.num_speculative_tokens = kwargs.get("num_speculative_tokens")
        self.model_role = kwargs.get("model_role", "standard")
        self.decoding_policy = DecodingPolicy(
            generator_backend,
            self.infer_context,
            self.model_wrapper,
            self.num_speculative_tokens,
            device_and_type,
            plugin_data_param,
            self.model_role,
            kwargs.get("eos_token_id"),
            kwargs.get("max_block_size"),
        )
        self.rank = generator_backend.rank
        self.max_seq_len = kwargs.get("max_seq_len")

        self.token_range = torch.arange(self.num_speculative_tokens + 1, device=NPU_STR)

    def model_inputs_update(
        self,
        model_inputs,
        input_metadata,
        sampling_metadata,
        cache_ids,
        input_len_mask,
        **kwargs,
    ):
        hit_mask = kwargs.get("hit_mask")
        model_inputs, q_len, attention_mask = self.decoding_policy.handle_input(
            model_inputs, input_metadata, cache_ids, hit_mask=hit_mask
        )
        input_len_mask = (q_len, attention_mask)

        self.decoding_policy.handle_sampling(sampling_metadata)

        return model_inputs, input_len_mask

    def prepare_masks_for_filling(
        self,
        model_inputs,
        current_dp_sequence_ids,
        current_all_sequence_ids,
        last_all_sequence_ids,
    ):
        if ENV.model_runner_exp:
            return self.prepare_masks_for_filling_exp(
                model_inputs,
                current_dp_sequence_ids,
                current_all_sequence_ids,
                last_all_sequence_ids,
            )
        masks = {}
        if last_all_sequence_ids is not None:
            speculative_length = self.num_speculative_tokens + 1
            hit_mask = np.isin(current_dp_sequence_ids, last_all_sequence_ids)
            if hit_mask.any():
                hit_sequence_ids = current_dp_sequence_ids[hit_mask]
                hit_indices = np.where(hit_sequence_ids[:, None] == last_all_sequence_ids[None, :])[1]
                hit_mask_per_token = np.repeat(hit_mask, speculative_length)
                hit_size = len(hit_indices)
                hit_increments = (
                    np.repeat(np.arange(speculative_length).reshape(-1, 1), hit_size, axis=1).transpose().reshape(-1)
                )
                hit_indices_per_token = np.repeat(hit_indices * speculative_length, speculative_length) + hit_increments
                masks["hit_mask"] = hit_mask
                masks["hit_mask_tensor"] = self.generator_backend.to_tensor(hit_mask)
                all_hit_mask = np.isin(current_all_sequence_ids, last_all_sequence_ids)
                all_hit_sequence_ids = current_all_sequence_ids[all_hit_mask]
                masks["all_hit_mask_tensor"] = self.generator_backend.to_tensor(all_hit_mask)
                all_hit_indices = np.where(all_hit_sequence_ids[:, None] == last_all_sequence_ids[None, :])[1]
                masks["all_hit_indices"] = all_hit_indices
                masks["hit_indices"] = hit_indices
                masks["hit_indices_per_token_tensor"] = self.generator_backend.to_tensor(hit_indices_per_token)
                masks["hit_mask_per_token_tensor"] = self.generator_backend.to_tensor(hit_mask_per_token)
                masks["hit_speculative_length"] = np.full(len(all_hit_indices), speculative_length)
                masks["hit_dp_speculative_length"] = np.full(len(hit_indices), speculative_length)
                hit_arange = np.arange(hit_size)
                masks["hit_arange_tensor"] = self.generator_backend.to_tensor(hit_arange)
                hit_mask_mod = self.generator_backend.to_tensor(
                    np.arange(len(hit_mask_per_token)) % speculative_length != 0
                )
                masks["hit_mask_mod"] = hit_mask_mod
                hit_block_tables = model_inputs.block_tables_array[hit_mask]
                candidate_slots = self.infer_context.block_table_to_slots(hit_block_tables).reshape(hit_size, -1)
                hit_block_indices = np.repeat(hit_arange, speculative_length)
                masks["candidate_slots"] = self.generator_backend.to_tensor(candidate_slots)
                masks["hit_block_indices"] = self.generator_backend.to_tensor(hit_block_indices)
                masks["hit_increments"] = self.generator_backend.to_tensor(hit_increments)
        return masks

    def fill_in_model_result(
        self,
        input_metadata,
        model_inputs,
        model_kwargs,
        model_output_wrapper,
        filling_masks,
        cache_ids,
    ):
        speculative_length = self.num_speculative_tokens + 1
        mtp_model_inputs = model_kwargs.get("sub_model_inputs")
        hidden_states = model_kwargs.get("hidden_states")
        lm_head_local_dp = model_kwargs.get("lm_head_local_dp", None)
        input_lengths_sp = model_kwargs.get("input_lengths_sp", None)
        hit_mask = filling_masks.get("hit_mask")
        if hit_mask is not None:
            model_output_hidden_states = model_output_wrapper.model_output.hidden_states
            sampling_output = model_output_wrapper.sampling_output

            # Move the hit_token_ids of mtp model to device.
            hit_indices = filling_masks.get("hit_indices")
            hit_token_ids = sampling_output.token_ids[hit_indices]
            hit_token_ids_cols = hit_token_ids.shape[1]
            if hit_token_ids_cols < speculative_length:
                padding_width = ((0, 0), (0, speculative_length - hit_token_ids_cols))
                hit_token_ids = np.pad(hit_token_ids, padding_width, "constant", constant_values=0)
            elif hit_token_ids_cols > speculative_length:
                logger.warning(
                    "Found the number of output tokens exceeds the speculative length, "
                    "which will be truncated forcibly."
                )
                hit_token_ids = hit_token_ids[:, :speculative_length]
            hit_token_ids = self.generator_backend.to_tensor_async(hit_token_ids)

            # Get the device tensor of subtrahend of lmhead indices.
            # The default prefill_head_indices is calculated assuming the number of tokens is speculative_length.
            # The true head indices equals to the default prefill_head_indices minus head_indices_subtrahend.
            all_hit_indices = filling_masks.get("all_hit_indices")
            hit_speculative_length = filling_masks.get("hit_speculative_length")
            all_hit_num_tokens = sampling_output.num_new_tokens[all_hit_indices]
            head_indices_subtrahend = hit_speculative_length - all_hit_num_tokens
            head_indices_subtrahend = self.generator_backend.to_tensor_async(head_indices_subtrahend)

            # Move the hit_num_tokens and hit_num_tokens_per_token to device.
            hit_num_tokens = sampling_output.num_new_tokens[hit_indices]
            hit_num_tokens_tensor = self.generator_backend.to_tensor_async(hit_num_tokens)
            hit_num_tokens_per_token = backend.repeat_interleave(hit_num_tokens_tensor, speculative_length)

            # Get masks.
            hit_mask_tensor = filling_masks.get("hit_mask_tensor")
            hit_mask_per_token = filling_masks.get("hit_mask_per_token_tensor")
            all_hit_mask_tensor = filling_masks.get("all_hit_mask_tensor")
            hit_indices_per_token = filling_masks.get("hit_indices_per_token_tensor")
            hit_arange_tensor = filling_masks.get("hit_arange_tensor")
            hit_mask_mod = filling_masks.get("hit_mask_mod")

            # assignation
            hit_token_ids_tensor = hit_token_ids.reshape(-1)
            mtp_model_inputs.input_ids[hit_mask_per_token] = hit_token_ids_tensor
            mtp_model_inputs.prefill_head_indices[all_hit_mask_tensor] -= head_indices_subtrahend
            hit_hidden_states = model_output_hidden_states[hit_indices_per_token]
            hidden_states[hit_mask_per_token] = hit_hidden_states
            if lm_head_local_dp is not None and not (len(lm_head_local_dp) == 1 and lm_head_local_dp[0] == 0):
                hit_dp_speculative_length = filling_masks.get("hit_dp_speculative_length")
                lm_head_local_dp[hit_mask] -= self.generator_backend.to_tensor(
                    hit_dp_speculative_length - hit_num_tokens
                )

            model_inputs.position_ids[hit_mask_per_token] += hit_num_tokens_per_token
            model_inputs.context_length[hit_mask] += hit_num_tokens
            model_inputs.input_lengths[hit_mask_tensor] += hit_num_tokens_tensor
            model_inputs.max_seq_len = max(model_inputs.context_length)
            if self.infer_context.spcp_parallel_info.scp_size == 1:
                offset_start_indices = model_inputs.input_lengths[hit_mask_tensor] - speculative_length
                hit_increments = filling_masks.get("hit_increments")
                block_offsets = backend.repeat_interleave(offset_start_indices, speculative_length) + hit_increments
                candidate_slots = filling_masks.get("candidate_slots")
                hit_block_indices = filling_masks.get("hit_block_indices")
                model_inputs.slots[hit_mask_per_token] = candidate_slots[hit_block_indices, block_offsets]
            else:
                model_inputs.cached_context_length[hit_mask] += hit_num_tokens
                indices = np.where(hit_mask)[0]
                for _, idx in enumerate(indices):
                    sp_tokens, tmp_slots = self.decoding_policy.sp_token_and_slot_calc_by_context_length(
                        model_inputs.cached_context_length[idx],
                        self.infer_context.get_mtp_seq_block_rank_id(cache_ids[idx]),
                        input_metadata.block_tables[idx],
                        speculative_length,
                    )
                    model_inputs.slots[idx * speculative_length : (idx + 1) * speculative_length] = (
                        self.generator_backend.to_tensor_async(tmp_slots)
                    )
                    model_inputs.is_need_mask[idx] = int(model_inputs.slots[(idx + 1) * speculative_length - 1] != -1)
                    model_inputs.sp_tokens[idx] = sp_tokens
                    input_lengths_sp[idx] = sp_tokens[
                        self.infer_context.spcp_parallel_info.cp_rank
                        * self.infer_context.spcp_parallel_info.sp_size : (
                            self.infer_context.spcp_parallel_info.cp_rank + 1
                        )
                        * self.infer_context.spcp_parallel_info.sp_size
                    ][self.infer_context.spcp_parallel_info.sp_rank]

            hit_mask_per_token[hit_mask_mod] = False
            model_inputs.input_ids[hit_mask_per_token] = hit_token_ids[hit_arange_tensor, hit_num_tokens_tensor - 1]

    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        if ENV.model_runner_exp:
            return self.sample_preprocess_exp(logits, result, sampling_metadata, input_metadata)
        self.sampling_param = sampling_metadata
        self.decoding_policy.sampling_param = self.sampling_param
        self.input_metadata = input_metadata
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        if sampling_metadata is None or sampling_metadata.is_prefill:
            return logits
        draft_tokens = result[2].cpu()
        all_sequence_ids = sampling_metadata.all_sequence_ids
        batch_size = len(all_sequence_ids)

        logits_num_per_batch = [self.num_speculative_tokens + 1] * batch_size
        input_ids_pad = self.decoding_policy.all_token_ids_padding(
            sampling_metadata, logits_num_per_batch, batch_size, draft_tokens
        )
        sampling_metadata.all_token_ids = input_ids_pad

        req_id_new = np.concatenate([[req_id] * n for req_id, n in zip(all_sequence_ids, logits_num_per_batch)])
        sampling_metadata.all_sequence_ids = req_id_new
        sampling_metadata.parent_sequence_ids = req_id_new

        return logits

    def plugin_verify(self, sampling_output, cache_ids, result):
        sampling_output.repeating_indices = np.arange(len(cache_ids))
        if self.input_metadata.is_prefill:
            return
        draft_token = result[2].cpu()
        next_tokens_uncheck = sampling_output.token_ids
        input_metadata = self.input_metadata
        next_tokens_indices = []
        out_seq_len = 1 if input_metadata.is_prefill else (self.num_speculative_tokens + 1)
        start_pos = 0
        draft_token_num_per_batch = self.num_speculative_tokens
        for batch in range(input_metadata.batch_size):
            end = start_pos + out_seq_len
            next_guess_by_batch = next_tokens_uncheck[start_pos:end]

            verify_guess_tokens = draft_token[
                batch * draft_token_num_per_batch : (batch + 1) * draft_token_num_per_batch
            ].view(-1)

            indices = self.decoding_policy.verify_greedy_one_batch(verify_guess_tokens, next_guess_by_batch)
            next_tokens_indices.append(list(range(start_pos, start_pos + indices + 1)))
            start_pos += out_seq_len
        output_token_len = self.infer_context.get_output_len_count(cache_ids)
        output_space_left1 = self.input_metadata.batch_max_output_lens - output_token_len
        output_space_left2 = self.max_seq_len - self.infer_context.get_seq_lens(cache_ids)
        output_space_left = np.minimum(output_space_left1, output_space_left2)
        self.decoding_policy.stop_criteria(sampling_output, output_space_left, next_tokens_indices)
        self.reshape_speculative_outputs(sampling_output, next_tokens_indices)

    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        result, _ = la_cache_input
        hidden_states = None
        if isinstance(result, tuple):
            hidden_states = result[1]
        if is_prefill:
            token_alias_len = [1] * len(cache_ids)
        else:
            token_alias_len = [self.num_speculative_tokens + 1] * len(cache_ids)

        self.decoding_policy.mtp_cache.cache_update(
            cache_ids, hidden_states, sampling_output, is_prefill, token_alias_len
        )

    def plugin_cache_clear(self, cache_ids, finish_reason):
        self.infer_context.last_sampling_metadata.clear()
        pass

    # Functions with the exp suffix are only used when backendType is torch.
    def prepare_masks_for_filling_exp(
        self,
        model_inputs,
        current_dp_sequence_ids,
        current_all_sequence_ids,
        last_all_sequence_ids,
    ):
        masks = {}
        if last_all_sequence_ids is not None:
            speculative_length = self.num_speculative_tokens + 1
            hit_mask = np.isin(current_dp_sequence_ids, last_all_sequence_ids)
            all_hit_mask = np.isin(current_all_sequence_ids, last_all_sequence_ids)
            if hit_mask.any():
                hit_sequence_ids = current_dp_sequence_ids[hit_mask]
                hit_indices = np.where(hit_sequence_ids[:, None] == last_all_sequence_ids[None, :])[1]
                hit_mask_per_token = np.repeat(hit_mask, speculative_length)
                hit_size = len(hit_indices)
                hit_increments = (
                    np.repeat(np.arange(speculative_length).reshape(-1, 1), hit_size, axis=1).transpose().reshape(-1)
                )
                hit_indices_per_token = np.repeat(hit_indices * speculative_length, speculative_length) + hit_increments
                masks["hit_mask"] = hit_mask
                masks["hit_mask_tensor"] = self.generator_backend.to_tensor(hit_mask)
                masks["hit_mask_local_tensor"] = torch.masked_select(
                    torch.arange(hit_mask.shape[0], device=NPU_STR),
                    masks["hit_mask_tensor"],
                )
                masks["hit_indices"] = hit_indices
                masks["hit_indices_tensor"] = self.generator_backend.to_tensor(hit_indices)
                masks["hit_indices_per_token_tensor"] = self.generator_backend.to_tensor(hit_indices_per_token)
                masks["hit_mask_per_token_tensor"] = self.generator_backend.to_tensor(hit_mask_per_token)
                masks["hit_local_indices_per_token_tensor"] = torch.masked_select(
                    torch.arange(hit_mask_per_token.shape[0], device=NPU_STR),
                    masks["hit_mask_per_token_tensor"],
                )
                masks["hit_dp_speculative_length"] = torch.tensor(
                    np.full(len(hit_indices), speculative_length), device=NPU_STR
                )
                hit_arange = np.arange(hit_size)
                masks["hit_arange_tensor"] = self.generator_backend.to_tensor(hit_arange)
                hit_mask_mod = self.generator_backend.to_tensor(
                    np.arange(len(hit_mask_per_token)) % speculative_length != 0
                )
                masks["hit_mask_mod"] = hit_mask_mod
                hit_mask_per_token[hit_mask_mod.cpu().numpy()] = False
                hit_mask_per_token_tensor = self.generator_backend.to_tensor(hit_mask_per_token)
                masks["hit_mask_per_token_mod_tensor"] = torch.masked_select(
                    torch.arange(hit_mask_per_token_tensor.shape[0], device=NPU_STR),
                    hit_mask_per_token_tensor,
                )
                hit_block_tables = model_inputs.block_tables_array[hit_mask]
                candidate_slots = self.infer_context.block_table_to_slots(hit_block_tables).reshape(hit_size, -1)
                hit_block_indices = np.repeat(hit_arange, speculative_length)
                masks["candidate_slots"] = self.generator_backend.to_tensor(candidate_slots)
                masks["hit_block_indices"] = self.generator_backend.to_tensor(hit_block_indices)
                masks["hit_increments"] = self.generator_backend.to_tensor(hit_increments)
            if all_hit_mask.any():
                masks["all_hit_mask"] = all_hit_mask
                all_hit_mask_tensor = self.generator_backend.to_tensor(all_hit_mask)
                masks["all_hit_mask_tensor"] = torch.masked_select(
                    torch.arange(all_hit_mask.shape[0], device=NPU_STR),
                    all_hit_mask_tensor,
                )
                all_hit_sequence_ids = current_all_sequence_ids[all_hit_mask]
                all_hit_indices = np.where(all_hit_sequence_ids[:, None] == last_all_sequence_ids[None, :])[1]
                masks["all_hit_indices"] = torch.tensor(all_hit_indices, device=NPU_STR)
                masks["hit_speculative_length"] = torch.tensor(
                    np.full(len(all_hit_indices), speculative_length), device=NPU_STR
                )
        return masks

    def fill_in_model_result_exp(
        self,
        input_metadata,
        model_inputs,
        model_kwargs,
        model_output_wrapper,
        filling_masks,
        cache_ids,
    ):
        speculative_length = self.num_speculative_tokens + 1
        mtp_model_inputs = model_kwargs.get("sub_model_inputs")
        hidden_states = model_kwargs.get("hidden_states")
        lm_head_local_dp = model_kwargs.get("lm_head_local_dp", None)
        input_lengths_sp = model_kwargs.get("input_lengths_sp", None)
        hit_mask = filling_masks.get("hit_mask")
        all_hit_mask = filling_masks.get("all_hit_mask")
        sampling_output = model_output_wrapper.sampling_output
        if all_hit_mask is not None:
            # Get the device tensor of subtrahend of lmhead indices.
            # The default prefill_head_indices is calculated assuming the number of tokens is speculative_length.
            # The true head indices equals to the default prefill_head_indices minus head_indices_subtrahend.
            all_hit_indices = filling_masks.get("all_hit_indices")
            hit_speculative_length = filling_masks.get("hit_speculative_length")
            all_hit_mask_tensor = filling_masks.get("all_hit_mask_tensor")
            all_hit_num_tokens = torch.index_select(sampling_output.num_new_tokens, dim=0, index=all_hit_indices)
            head_indices_subtrahend = hit_speculative_length - all_hit_num_tokens
            mtp_model_inputs.prefill_head_indices.scatter_add_(
                dim=0,
                index=all_hit_mask_tensor,
                src=-head_indices_subtrahend.to(torch.int32),
            )
        if hit_mask is not None:
            model_output_hidden_states = model_output_wrapper.model_output.hidden_states

            # Move the hit_token_ids of mtp model to device.
            hit_indices = filling_masks.get("hit_indices")
            hit_indices_tensor = filling_masks.get("hit_indices_tensor")
            hit_token_ids = torch.index_select(sampling_output.token_ids, dim=0, index=hit_indices_tensor)
            hit_token_ids_cols = hit_token_ids.shape[1]
            if hit_token_ids_cols < speculative_length:
                hit_token_ids = backend.nn.functional.pad(
                    hit_token_ids,
                    (0, speculative_length - hit_token_ids_cols, 0, 0),
                    mode="constant",
                    value=0,
                )
            elif hit_token_ids_cols > speculative_length:
                logger.warning(
                    "Found the number of output tokens exceeds the speculative length, "
                    "which will be truncated forcibly."
                )
                hit_token_ids = hit_token_ids[:, :speculative_length]

            # Move the hit_num_tokens and hit_num_tokens_per_token to device.
            hit_num_tokens_tensor = torch.index_select(sampling_output.num_new_tokens, dim=0, index=hit_indices_tensor)
            hit_num_tokens_per_token = backend.repeat_interleave(hit_num_tokens_tensor, speculative_length)
            hit_num_tokens_numpy = sampling_output.num_new_tokens_numpy[hit_indices]

            # Get masks.
            hit_indices_per_token = filling_masks.get("hit_indices_per_token_tensor")
            hit_arange_tensor = filling_masks.get("hit_arange_tensor")
            hit_mask_local_tensor = filling_masks.get("hit_mask_local_tensor")
            hit_local_indices_per_token_tensor = filling_masks.get("hit_local_indices_per_token_tensor")

            # assignation
            hit_token_ids_flatten = hit_token_ids.flatten()
            mtp_model_inputs.input_ids.scatter_(
                dim=0,
                index=hit_local_indices_per_token_tensor,
                src=hit_token_ids_flatten,
            )

            if model_output_wrapper.input_metadata.is_prefill:
                model_output_hidden_states = model_output_hidden_states.repeat(self.num_speculative_tokens + 1, 1)
                hidden_states = hidden_states.repeat(self.num_speculative_tokens + 1, 1)
            hidden_dim = hidden_states.shape[-1]
            hit_hidden_states = backend.index_select(model_output_hidden_states, 0, hit_indices_per_token)
            scatter_index = hit_local_indices_per_token_tensor.unsqueeze(-1).expand(-1, hidden_dim)
            hidden_states.scatter_(0, scatter_index, hit_hidden_states)
            model_kwargs["hidden_states"] = hidden_states

            if lm_head_local_dp is not None and not input_metadata.is_dummy_batch:
                hit_dp_speculative_length = filling_masks.get("hit_dp_speculative_length")
                lm_head_local_dp.scatter_add_(
                    dim=0,
                    index=hit_mask_local_tensor,
                    src=hit_num_tokens_tensor - hit_dp_speculative_length,
                )

            model_inputs.position_ids.scatter_add_(0, hit_local_indices_per_token_tensor, hit_num_tokens_per_token)
            model_inputs.context_length[hit_mask] += hit_num_tokens_numpy
            model_inputs.max_seq_len = max(model_inputs.context_length)
            model_inputs.input_lengths.scatter_add_(0, hit_mask_local_tensor, hit_num_tokens_tensor.to(torch.int32))
            model_inputs.forward_context.attn_metadata.max_seq_len = model_inputs.max_seq_len
            if self.infer_context.spcp_parallel_info.scp_size == 1:
                offset_start_indices = (
                    torch.index_select(model_inputs.input_lengths, 0, hit_mask_local_tensor) - speculative_length
                )
                hit_increments = filling_masks.get("hit_increments")
                block_offsets = backend.repeat_interleave(offset_start_indices, speculative_length) + hit_increments
                candidate_slots = filling_masks.get("candidate_slots")
                hit_block_indices = filling_masks.get("hit_block_indices")
                hit_local_indices_per_token_tensor = filling_masks.get("hit_local_indices_per_token_tensor")
                model_inputs.forward_context.attn_metadata.slot_mapping.scatter_(
                    0,
                    hit_local_indices_per_token_tensor,
                    candidate_slots[hit_block_indices, block_offsets],
                )
            else:
                model_inputs.cached_context_length[hit_mask] += hit_num_tokens_tensor
                indices = np.where(hit_mask)[0]
                for _, idx in enumerate(indices):
                    sp_tokens, tmp_slots = self.decoding_policy.sp_token_and_slot_calc_by_context_length(
                        model_inputs.cached_context_length[idx],
                        self.infer_context.get_mtp_seq_block_rank_id(cache_ids[idx]),
                        input_metadata.block_tables[idx],
                        speculative_length,
                    )
                    model_inputs.slots[idx * speculative_length : (idx + 1) * speculative_length] = (
                        self.generator_backend.to_tensor_async(tmp_slots)
                    )
                    model_inputs.sp_tokens[idx] = sp_tokens
                    input_lengths_sp[idx] = sp_tokens[
                        self.infer_context.spcp_parallel_info.cp_rank
                        * self.infer_context.spcp_parallel_info.sp_size : (
                            self.infer_context.spcp_parallel_info.cp_rank + 1
                        )
                        * self.infer_context.spcp_parallel_info.sp_size
                    ][self.infer_context.spcp_parallel_info.sp_rank]

            actual_seq_lengths_query = torch.ones_like(model_inputs.input_lengths, dtype=torch.int32)
            actual_seq_lengths_query = torch.cumsum(actual_seq_lengths_query, dim=0, dtype=torch.int32)
            model_inputs.forward_context.attn_metadata.actual_seq_lengths_query = (
                actual_seq_lengths_query * speculative_length
            )
            model_inputs.forward_context.attn_metadata.actual_seq_lengths_kv = model_inputs.input_lengths
            model_inputs.forward_context.sub_forward_context.attn_metadata.actual_seq_lengths_kv = (
                model_inputs.input_lengths.clone()
            )
            model_inputs.forward_context.sub_forward_context.attn_metadata.actual_seq_lengths_query = (
                model_inputs.forward_context.attn_metadata.actual_seq_lengths_query.clone()
            )
            model_inputs.forward_context.attn_metadata.seq_lens = model_inputs.input_lengths

            hit_mask_per_token_mod_tensor = filling_masks.get("hit_mask_per_token_mod_tensor")
            model_inputs.input_ids.scatter_(
                0,
                hit_mask_per_token_mod_tensor,
                hit_token_ids[hit_arange_tensor, hit_num_tokens_tensor - 1],
            )

    def sample_preprocess_exp(self, logits, result, sampling_metadata, input_metadata):
        self.input_metadata = input_metadata
        self.sampling_param = sampling_metadata
        self.decoding_policy.sampling_param = self.sampling_param
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        if sampling_metadata is None or sampling_metadata.is_prefill:
            return logits
        return logits

    def compose_model_inputs_exp(self, sampling_metadata):
        all_sequence_ids = sampling_metadata.all_sequence_ids
        batch_size = len(all_sequence_ids)
        logits_num_per_batch = [self.num_speculative_tokens + 1] * batch_size
        req_id_new = np.concatenate([[req_id] * n for req_id, n in zip(all_sequence_ids, logits_num_per_batch)])
        sampling_metadata.all_sequence_ids = req_id_new
        sampling_metadata.parent_sequence_ids = req_id_new
        return sampling_metadata

    def plugin_verify_exp(self, sampling_output, cache_ids, result):
        sampling_output.repeating_indices = np.arange(len(cache_ids))
        if self.input_metadata.is_prefill:
            return
        draft_token = result[2]
        next_tokens_uncheck = sampling_output.token_ids
        input_metadata = self.input_metadata
        out_seq_len = 1 if input_metadata.is_prefill else (self.num_speculative_tokens + 1)

        batch_size = next_tokens_uncheck.shape[0] // out_seq_len
        next_guess_all = next_tokens_uncheck.view(batch_size, out_seq_len)
        verify_guess_all = draft_token.view(batch_size, self.num_speculative_tokens)
        matches = verify_guess_all == next_guess_all[:, : self.num_speculative_tokens]
        indices_counts = torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1) + 1

        max_num_tokens = out_seq_len
        token_range = self.token_range[:max_num_tokens].unsqueeze(0).expand(batch_size, -1)
        mask = token_range < indices_counts.unsqueeze(1)
        batch_offsets_cpu = torch.arange(batch_size) * out_seq_len
        raw_token_ids = sampling_output.token_ids.view(batch_size, out_seq_len)
        raw_logprobs = sampling_output.logprobs.view(batch_size, out_seq_len)

        sampling_output.token_ids = torch.where(mask, raw_token_ids, self.pad_token_id)
        sampling_output.logprobs = torch.where(mask, raw_logprobs, -9999.0)
        if sampling_output.top_token_ids.shape[-1] > 0:
            k_dim = sampling_output.top_token_ids.shape[-1]
            raw_top_tokens = sampling_output.top_token_ids.view(batch_size, out_seq_len, k_dim)
            raw_top_probs = sampling_output.top_logprobs.view(batch_size, out_seq_len, k_dim)

            full_mask_k = mask.unsqueeze(-1)
            sampling_output.top_token_ids = torch.where(full_mask_k, raw_top_tokens, self.pad_token_id)
            sampling_output.top_logprobs = torch.where(full_mask_k, raw_top_probs, -9999.0)
        else:
            sampling_output.top_token_ids = torch.zeros(
                (batch_size, max_num_tokens, 0), dtype=torch.int64, device=NPU_STR
            )
            sampling_output.top_logprobs = torch.zeros(
                (batch_size, max_num_tokens, 0), dtype=torch.float32, device=NPU_STR
            )

        sampling_output.num_new_tokens = indices_counts
        if sampling_output.sequence_ids is not None:
            seq_ids_npu = torch.as_tensor(sampling_output.sequence_ids)
            parent_ids_npu = torch.as_tensor(sampling_output.parent_sequence_ids)

            sampling_output.sequence_ids = seq_ids_npu[batch_offsets_cpu].reshape(batch_size).numpy()
            sampling_output.parent_sequence_ids = parent_ids_npu[batch_offsets_cpu].reshape(batch_size).numpy()

        sampling_output.group_indices = [(i, i + 1) for i in range(batch_size)]
