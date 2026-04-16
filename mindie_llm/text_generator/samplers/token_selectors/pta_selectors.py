# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
from functools import wraps

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F

from .token_selector import TokenSelector
from ...utils.sampling_output import SamplingOutput
from ...utils.sampling_metadata import SamplingMetadata

PTA_SELECTOR_REGISTRY = {}


def register_class(name):
    def decorator(cls):
        PTA_SELECTOR_REGISTRY[name] = cls

        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        return wrapper

    return decorator


@register_class("greedy_search")
class GreedySearchTokenSelector(TokenSelector):
    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata) -> SamplingOutput:
        batch_size = len(logits)
        if metadata is not None and metadata.max_logprobs > 0:
            logprobs = torch.log_softmax(logits, dim=-1)
            top_logprobs, top_token_ids = torch.topk(logprobs, metadata.max_logprobs, dim=-1)
            top_logprobs = top_logprobs.to(torch.float32).cpu().numpy()
            top_token_ids = top_token_ids.to(torch.int64).cpu().numpy()
            logprobs = top_logprobs[:, 0]
            token_ids = top_token_ids[:, 0]
        else:
            token_ids = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
            logprobs = np.full((batch_size,), -9999.0, dtype=np.float32)
            top_token_ids = np.zeros((batch_size, 0), dtype=np.int64)
            top_logprobs = np.zeros((batch_size, 0), dtype=np.float32)
        sampling_output = SamplingOutput(
            sequence_ids=metadata.all_sequence_ids if metadata is not None else None,
            parent_sequence_ids=metadata.all_sequence_ids if metadata is not None else None,
            group_indices=[(i, i + 1) for i in range(batch_size)],
            repeating_indices=np.arange(batch_size),
            token_ids=token_ids,
            logprobs=logprobs,
            top_token_ids=top_token_ids,
            top_logprobs=top_logprobs,
            cumulative_logprobs=np.zeros(batch_size, dtype=np.float32),
            num_new_tokens=np.ones(batch_size, dtype=np.int64),
            num_top_tokens=metadata.num_top_tokens if metadata is not None else None,
        )
        return sampling_output


@register_class("sampling")
class SamplingTokenSelector(TokenSelector):
    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata):
        do_sample_array = metadata.do_sample_array
        indices_array = np.where(do_sample_array > 0)[0]
        indices_tensor = torch.tensor(indices_array).to(logits.device)
        argmax_indices = torch.nonzero(torch.tensor(do_sample_array == 0)).squeeze(1)
        filtered_logits = logits[indices_tensor]
        sampled_probs = F.softmax(filtered_logits, dim=-1)
        sampled_token_ids = self.multinomial(sampled_probs, 1, metadata.seed_array[indices_array]).squeeze(1)
        token_ids = torch.tensor([-1] * len(do_sample_array), device=logits.device)
        token_ids.scatter_(0, indices_tensor, sampled_token_ids)
        filtered_logits = logits[argmax_indices]
        argmax_token_ids = filtered_logits.argmax(dim=-1)
        token_ids.scatter_(0, argmax_indices, argmax_token_ids)
        return token_ids.cpu().numpy(), np.array([]), np.array([[]]), np.array([[]])

    @staticmethod
    def multinomial(prob_matrix, num_samples, seeds):
        random_values = []
        for s in seeds:
            s = s.astype(np.int32)
            s = s % np.iinfo(np.int32).max
            np.random.seed(s)
            random_values.append(np.random.rand(num_samples))
        sorted_prob, indices = torch.sort(prob_matrix, descending=True)
        cdf_matrix = torch.cumsum(sorted_prob, dim=-1)
        selected_ids = torch.searchsorted(cdf_matrix, torch.tensor(np.array(random_values)).to(prob_matrix.device))
        selected_token_ids = torch.gather(indices, -1, selected_ids)
        return selected_token_ids


@register_class("beam_search")
class BeamSearchTokenSelector(TokenSelector):
    def __init__(self, selector_params):
        super().__init__(selector_params)
        self.eos_token_id = self.params.eos_token_id
        self.num_eos = len(self.eos_token_id)
        self.candidate_expansion_factor = self.params.candidate_expansion_factor
        self.filter_value = self.params.filter_value
        self.splitfuse_enabled = self.params.splitfuse_enabled

    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata) -> SamplingOutput:
        info = torch.finfo(logits.dtype)
        if torch.isinf(logits).any():
            logits[logits == torch.inf] = info.max
            logits[logits == -torch.inf] = info.min
        nan_mask = torch.isnan(logits)
        if nan_mask.any():
            logits[nan_mask] = 0
        if metadata.output_lengths is None or (metadata.output_lengths < 0).any():
            raise ValueError("A certain output length of beam search is invalid.")
        min_num_candidates = (self.num_eos + 1) * min(metadata.beam_width_array)
        max_num_candidates = (self.num_eos + 1) * metadata.max_beam_width
        max_needed_tokens = max(metadata.max_logprobs, max_num_candidates)

        logprobs = torch.log_softmax(logits, dim=1)
        top_logprobs, top_token_ids = torch.topk(logprobs, max_needed_tokens, dim=1)
        top_logprobs = top_logprobs.to(torch.float32).to("cpu", non_blocking=True)
        top_token_ids = top_token_ids.to(torch.int64).to("cpu", non_blocking=True)

        group_width = []
        group_beam_width = []
        for i, sequence_group in enumerate(metadata.batch_sequence_ids):
            group_width.append(len(sequence_group))
            group_beam_width.append(metadata.beam_width_array[metadata.group_indices[i][0]])
        group_width_array = np.array(group_width)
        group_beam_width_array = np.array(group_beam_width)
        batch_size = len(metadata.batch_sequence_ids)
        min_group_width, max_group_width = min(group_width), max(group_width)

        torch_npu.npu.current_stream().synchronize()
        top_logprobs = top_logprobs.numpy()
        top_token_ids = top_token_ids.numpy()
        # 要求metadata.cumulative_logprobs数量与seqs数量一致，然后广播
        average_logprobs = (metadata.cumulative_logprobs[:, None] + top_logprobs) / (
            metadata.output_lengths[:, None] + 1
        )

        num_seqs = len(logits)
        # Deep copy is neccessary only when the maximum number of logprobs is larger than the minimum beam width.
        if metadata.max_logprobs is not None and metadata.max_logprobs > min_num_candidates:
            masked_token_ids = copy.deepcopy(top_token_ids)
        else:
            masked_token_ids = top_token_ids

        if min_num_candidates != max_num_candidates:
            max_candidates_arange = np.arange(max_needed_tokens)
            num_valid_candidates = (self.num_eos + 1) * metadata.beam_width_array
            col_padding_mask = max_candidates_arange[None, :].repeat(num_seqs, axis=0) >= num_valid_candidates[:, None]
            masked_token_ids[col_padding_mask] = -1
            average_logprobs[col_padding_mask] = self.filter_value

        # construct row padding
        if min_group_width != max_group_width:
            row_padding_matrix = np.arange(max_group_width)[None, :].repeat(batch_size, axis=0)
            row_padding_mask = row_padding_matrix < group_width_array[:, None]
            flatten_padding_mask = row_padding_mask.flatten()
            row_padding_indices = np.cumsum(flatten_padding_mask) - 1
            row_padding_indices = np.where(flatten_padding_mask, row_padding_indices, -1)
            masked_token_ids = np.pad(masked_token_ids, ((0, 1), (0, 0)), constant_values=-1)
            average_logprobs = np.pad(average_logprobs, ((0, 1), (0, 0)), constant_values=self.filter_value)
            masked_token_ids = masked_token_ids[row_padding_indices]
            average_logprobs = average_logprobs[row_padding_indices]
        masked_token_ids = masked_token_ids.reshape(batch_size, -1)
        average_logprobs = average_logprobs.reshape(batch_size, -1)

        eos_mask = masked_token_ids == self.eos_token_id[0]
        for other_eos in self.eos_token_id[1:]:
            eos_mask |= masked_token_ids == other_eos
        eos_average_logprobs = average_logprobs[eos_mask]
        average_logprobs[eos_mask] = 1

        # get top-n indices
        n_indices = np.argpartition(-average_logprobs, max_needed_tokens - 1, axis=1)[:, :max_needed_tokens]
        n_order = np.argsort(np.take_along_axis(average_logprobs, n_indices, axis=1), axis=1)[:, ::-1]
        beam_indices = np.take_along_axis(n_indices, n_order, axis=1)
        average_cumulative_logprobs = np.take_along_axis(average_logprobs, beam_indices, axis=1)
        group_width_with_dummy = np.concatenate(([0], group_width_array))
        group_start_indices = np.cumsum(group_width_with_dummy)[:-1][:, None]
        row_ids = group_start_indices + beam_indices // max_needed_tokens
        col_ids = beam_indices % max_needed_tokens

        # output
        num_eos_array = np.sum(eos_mask, axis=1)
        valid_width_array = group_beam_width_array + num_eos_array
        position_matrix = np.arange(max_needed_tokens)[None, :].repeat(batch_size, axis=0)
        valid_mask = position_matrix < valid_width_array[:, None]
        beam_mask = position_matrix < valid_width_array[:, None]
        beam_mask[position_matrix < num_eos_array[:, None]] = False

        sequence_indices = np.arange(num_seqs)
        parent_indices = sequence_indices[row_ids[valid_mask]]
        parent_sequence_ids = metadata.all_sequence_ids[parent_indices]
        token_ids = top_token_ids[row_ids[valid_mask], col_ids[valid_mask]]
        logprobs = top_logprobs[row_ids[valid_mask], col_ids[valid_mask]]
        average_cumulative_logprobs = average_cumulative_logprobs[valid_mask]
        max_top_tokens = max(metadata.num_top_tokens) if metadata.num_top_tokens is not None else 0
        top_token_ids = top_token_ids[row_ids[valid_mask]][:, :max_top_tokens]
        top_logprobs = top_logprobs[row_ids[valid_mask]][:, :max_top_tokens]

        # 匹配sequence id
        is_seq_prefill = metadata.is_seq_prefill
        is_prefill = metadata.is_prefill
        # 混合+纯p请求
        is_mix_has_prefill = self.splitfuse_enabled and True in is_seq_prefill
        is_all_prefill = not self.splitfuse_enabled and is_prefill
        if is_mix_has_prefill or is_all_prefill:
            sequence_ids = []
            first_prefill_idx = -sum(is_seq_prefill) if self.splitfuse_enabled else -len(metadata.all_sequence_ids)
            for i in range(first_prefill_idx, 0):
                original_sequence_ids = metadata.batch_sequence_ids[i]
                sequence_group = [
                    np.full((num_eos_array[i]), -1),
                    original_sequence_ids,
                    metadata.reserved_sequence_ids[i],
                ]
                sequence_ids.extend(sequence_group)
            prefill_seq_ids = np.concatenate(sequence_ids)
        # 混合+纯d请求
        is_mix_has_decode = self.splitfuse_enabled and False in is_seq_prefill
        is_all_decode = not self.splitfuse_enabled and not is_prefill
        if is_mix_has_decode or is_all_decode:
            decode_req_last_idx = (
                -sum(is_seq_prefill)
                if is_seq_prefill is not None and sum(is_seq_prefill) != 0
                else len(parent_sequence_ids)
            )
            decode_valid_mask = valid_mask[:decode_req_last_idx]
            decode_last_idx = sum(decode_valid_mask.flatten())
            new_seq_mask = np.zeros_like(parent_sequence_ids[:decode_last_idx], dtype=np.bool_)
            used_ids = set()
            for i, seq_id in enumerate(parent_sequence_ids[:decode_last_idx]):
                if seq_id in used_ids and average_cumulative_logprobs[i] <= 0:
                    new_seq_mask[i] = True
                elif average_cumulative_logprobs[i] > 0:
                    new_seq_mask[i] = False
                else:
                    new_seq_mask[i] = False
                    used_ids.add(seq_id)
            beam_indices = row_ids[:decode_req_last_idx][beam_mask[:decode_req_last_idx]]
            children_indices = row_ids[:decode_req_last_idx][valid_mask[:decode_req_last_idx]]
            children_indices[new_seq_mask] = np.arange(num_seqs)[:decode_req_last_idx][
                ~np.isin(np.arange(num_seqs)[:decode_req_last_idx], beam_indices)
            ]
            decode_seq_ids = metadata.all_sequence_ids[children_indices]

        # get sequence ids
        is_mix_all_prefill = self.splitfuse_enabled and False not in is_seq_prefill
        is_mix_all_decode = self.splitfuse_enabled and True not in is_seq_prefill
        if is_mix_all_prefill or is_all_prefill:
            sequence_ids = prefill_seq_ids
        elif is_mix_all_decode or is_all_decode:
            sequence_ids = decode_seq_ids
        else:
            sequence_ids = np.concatenate((decode_seq_ids, prefill_seq_ids))

        output_eos_mask = average_cumulative_logprobs == 1
        average_cumulative_logprobs[output_eos_mask] = eos_average_logprobs
        sequence_ids[output_eos_mask] = -1

        # get group indices
        start_idx = 0
        group_indices = []
        for w in valid_width_array:
            group_indices.append((start_idx, start_idx + w))
            start_idx += w

        sampling_output = SamplingOutput(
            sequence_ids=sequence_ids,
            parent_sequence_ids=parent_sequence_ids,
            group_indices=group_indices,
            repeating_indices=parent_indices,
            token_ids=token_ids,
            logprobs=logprobs,
            top_token_ids=top_token_ids,
            top_logprobs=top_logprobs,
            cumulative_logprobs=average_cumulative_logprobs,
            num_new_tokens=np.ones(len(parent_sequence_ids), dtype=np.int64),
            num_top_tokens=metadata.num_top_tokens[parent_indices] if metadata.num_top_tokens is not None else None,
        )
        return sampling_output
