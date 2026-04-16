# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np

from .decoding_policy import DecodingPolicy
from ..plugin import Plugin
from ....utils.tensor import op, tensor_backend
from ....utils.log.logging import logger

MAX_DECODING_LEN = 16
MIN_DECODING_LEN = 1


class MemoryDecodingPlugin(Plugin):
    def __init__(
        self,
        generator_backend,
        kvcache_settings,
        infer_context,
        plugin_data_param,
        **kwargs,
    ):
        super().__init__()
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.kvcache_settings = kvcache_settings
        self.infer_context = infer_context

        self.plugin_data_param = plugin_data_param

        self.device = self.model_wrapper.device
        self.decoding_length = kwargs.get("decoding_length", 16)
        if self.decoding_length > MAX_DECODING_LEN:
            logger.warning(f"decoding_length is larger than max value {MAX_DECODING_LEN}, run with max value!")
            self.decoding_length = MAX_DECODING_LEN
        if self.decoding_length < MIN_DECODING_LEN:
            logger.warning(f"decoding_length is smaller than min value {MIN_DECODING_LEN}, run with min value!")
            self.decoding_length = MIN_DECODING_LEN
        self.decoding_policy = DecodingPolicy(kwargs, self.infer_context, self.decoding_length)
        self.memory_decoding_decoding_ids = None
        self.input_metadata = None

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

    def set_atten_mask_ms(self, model_inputs, kv_dtype):
        kv_device = self.model_wrapper.device
        atten_mask = self.model_wrapper.model_runner.attn_mask.get_attn_mask(
            model_inputs.max_seq_len, kv_dtype, kv_device
        )
        if atten_mask[0][1] > 0:
            atten_mask = atten_mask * -10000.0
        return atten_mask

    def calc_decoding_info(self, input_metadata, model_inputs, cache_ids, q_lens, decoding_masks):
        decoding_ids = None
        if not input_metadata.is_prefill:
            model_inputs, decoding_ids, decoding_masks = self.decoding_policy.update_infer_input(
                model_inputs, cache_ids
            )
            q_lens = [len(x) for x in decoding_ids]
            kv_dtype = self.kvcache_settings.dtype
            atten_mask = self.set_atten_mask_ms(model_inputs, kv_dtype)
            req_mask = None
            start_row = 0
            bs = len(q_lens)
            for i in range(bs):
                start = int(model_inputs.context_length[i] - q_lens[i])
                end = int(model_inputs.context_length[i])

                if req_mask is None:
                    req_mask = atten_mask[start:end, :]
                else:
                    req_mask = op.cat((req_mask, atten_mask[start:end]), 0)
                start_row += q_lens[i]
            decoding_masks = req_mask
        res = (model_inputs, decoding_ids, decoding_masks, q_lens)
        return res

    def all_token_ids_padding(self, sampling_metadata, next_guess_logits_num_per_batch, batch_size):
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
            if len(self.memory_decoding_decoding_ids[batch]) > 1:
                guess_set = self.memory_decoding_decoding_ids[batch]
                for idx in range(len(guess_set) - 1):
                    input_ids_pad[guess_index, input_ids_len : input_ids_len + idx + 1] = np.array(
                        list(guess_set[1 : idx + 2])
                    )
                    guess_index += 1
        input_ids_pad = tensor_backend.tensor(
            input_ids_pad,
            dtype=sampling_metadata.all_token_ids.dtype,
            device=self.device,
        )
        return input_ids_pad

    def sample_preprocess(self, logits, result, sampling_metadata, input_metadata):
        self.sampling_param = sampling_metadata
        self.decoding_policy.sampling_param = self.sampling_param
        self.input_metadata = input_metadata
        if sampling_metadata is None:
            return logits
        all_sequence_ids = sampling_metadata.all_sequence_ids
        batch_size = len(all_sequence_ids)
        # q_lens为None或者为空列表的情况都在prefill阶段
        if self.plugin_data_param.q_len is None or self.plugin_data_param.q_len == []:
            self.plugin_data_param.q_len = [1] * logits.shape[0]
            self.decoding_policy.qlens = self.plugin_data_param.q_len

        if sampling_metadata.is_prefill:
            return logits

        logits_num_per_batch = self.plugin_data_param.q_len

        input_ids_pad = self.all_token_ids_padding(sampling_metadata, logits_num_per_batch, batch_size)
        sampling_metadata.all_token_ids = input_ids_pad

        req_id_new = np.concatenate([[req_id] * n for req_id, n in zip(all_sequence_ids, logits_num_per_batch)])
        sampling_metadata.all_sequence_ids = req_id_new
        sampling_metadata.parent_sequence_ids = req_id_new

        if sampling_metadata is not None:
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

        return logits

    def model_inputs_update(
        self,
        model_inputs,
        input_metadata,
        sampling_metadata,
        cache_ids,
        input_len_mask,
        **kwargs,
    ):
        (q_lens, decoding_masks) = input_len_mask

        if input_metadata.is_prefill:
            self.decoding_policy.add_prompt_to_cache(model_inputs, cache_ids)

        model_inputs, decoding_ids, decoding_masks, q_lens = self.calc_decoding_info(
            input_metadata, model_inputs, cache_ids, q_lens, decoding_masks
        )

        self.memory_decoding_decoding_ids = decoding_ids

        input_len_mask = (q_lens, decoding_masks)
        return model_inputs, input_len_mask

    def plugin_verify(self, sampling_output, cache_ids, result):
        sampling_output.repeating_indices = np.arange(len(cache_ids))
        if not self.input_metadata.is_prefill:
            next_tokens_indices, _ = self.decoding_policy.verify(
                sampling_output.token_ids, self.memory_decoding_decoding_ids, cache_ids
            )
            self.decoding_policy.truncate_long_tokens(
                cache_ids, self.input_metadata, sampling_output, next_tokens_indices
            )
            self.reshape_speculative_outputs(sampling_output, next_tokens_indices)

    def plugin_cache_update(self, cache_ids, sampling_output, la_cache_input, is_prefill=False):
        for i, cache_id in enumerate(cache_ids):
            seq_token_ids = sampling_output.token_ids[i][: sampling_output.num_new_tokens[i]].tolist()
            self.decoding_policy.prefix_ids[cache_id] = list(self.decoding_policy.prefix_ids[cache_id]) + seq_token_ids

            if self.decoding_policy.dynamic_algo:
                self.decoding_policy.tokens_knowledge_base_cache.output_add(
                    seq_token_ids,
                    search_size=self.decoding_policy.branch_length + 1,
                    final=False,
                    pattern="output",
                    use_batch=cache_id,
                )
            else:
                self.decoding_policy.tokens_knowledge_base_cache.output_add(
                    seq_token_ids,
                    search_size=self.decoding_policy.tokens_knowledge_base_param["branch_length"] + 1,
                    final=False,
                    pattern="output",
                    use_batch=cache_id,
                )

    def plugin_cache_clear(self, cache_ids, finish_reason):
        self.infer_context.last_sampling_metadata.clear()
        if len(finish_reason) > 0:
            self.decoding_policy.filter_out_eos(finish_reason, cache_ids)
