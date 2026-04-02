# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from functools import wraps

import mindspore as ms
import numpy as np
from _cpu_logits_handler import _PostProcessingManager
from mindspore import mint, Tensor

from .token_selector import TokenSelector
from ...utils.sampling_metadata import SamplingMetadata
from ...utils.sampling_output import SamplingOutput
from ....utils.env import ENV
from ....utils.log.logging import logger, print_log

MS_SELECTOR_REGISTRY = {}


def register_class(name):
    def decorator(cls):
        MS_SELECTOR_REGISTRY[name] = cls

        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        return wrapper

    return decorator


@register_class('top_k_top_p_sampling')
class TopKTopPSamplingTokenSelector(TokenSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = _PostProcessingManager.get_instance(self.params.num_threads, self.params.npu_id)
        self.filter_value = self.params.filter_value
        self.speed_mode, self.use_approx = bool(ENV.speed_mode_type & 2), bool(ENV.speed_mode_type & 1)

    def __call__(self, logits: ms.Tensor, metadata: SamplingMetadata):
        if metadata.is_prefill:
            self.configure(metadata)
        # top_k
        max_top_k = int(metadata.max_top_k)
        top_k = metadata.top_k_idx

        kth_logits = None
        if max_top_k > 0:
            k_logits, k_index = mint.topk(logits, max_top_k, 1)
            if metadata.top_k_disabled_mask is not None:
                kth_logits = mint.gather(k_logits, 1, top_k).reshape(-1, 1)
                kth_logits.masked_fill(metadata.top_k_disabled_mask, self.filter_value)
                indices_to_remove = logits < kth_logits
                logits.masked_fill(indices_to_remove, self.filter_value)
                sorted_logits, sorted_indices = mint.sort(logits, descending=True)
            else:
                return self._fusion_top_k_top_p_sampling(k_logits, k_index, metadata)
        else:
            sorted_logits, sorted_indices = mint.sort(logits, descending=True)

        # top_p
        top_p = metadata.top_p

        cutoff_logit = None
        if top_p is not None:
            cumulative_probs = mint.nn.functional.softmax(sorted_logits, dim=-1)
            cumulative_probs = mint.cumsum(cumulative_probs, dim=-1)
            cutoff_index = mint.sum(cumulative_probs < top_p, dim=-1, keepdim=True)

            # Avoid out-of-bounds when top_p is close to 1.
            max_cutoff_index = sorted_logits.shape[-1] - 1
            cutoff_index = mint.where(cutoff_index > max_cutoff_index, max_cutoff_index, cutoff_index)
            cutoff_logit = mint.gather(sorted_logits, -1, cutoff_index)

        if kth_logits is not None and cutoff_logit is not None:
            sorted_logits = sorted_logits.masked_fill(sorted_logits < mint.maximum(cutoff_logit, kth_logits),
                                                      self.filter_value)
        elif kth_logits is not None:
            sorted_logits = sorted_logits.masked_fill(sorted_logits < kth_logits, self.filter_value)
        elif cutoff_logit is not None:
            sorted_logits = sorted_logits.masked_fill(sorted_logits < cutoff_logit, self.filter_value)

            # sampling
        return self._do_sampling(sorted_logits, sorted_indices, metadata)

    def multinomial(self, sorted_prob, sorted_indices, num_samples, seeds):
        random_values = []
        for cur_seed in seeds:
            try:
                np.random.seed(cur_seed)
            except ValueError as e:
                print_log(self.params.rank, logger.warning,
                          f'The seed is out of range: {e}. And will use [cur_seed % 2**32] by default.')
                np.random.seed(cur_seed % 2 ** 32)
            random_values.append(np.random.rand(num_samples))
        cdf_matrix = mint.cumsum(sorted_prob, dim=-1)
        selected_ids = mint.searchsorted(cdf_matrix, ms.Tensor(random_values))
        selected_tokens = mint.gather(sorted_indices, -1, selected_ids)
        return selected_tokens

    def clear(self, request_ids: np.ndarray):
        self.processor.delete_configs(request_ids)

    def configure(self, metadata: SamplingMetadata):
        if metadata.all_sequence_ids is None:
            metadata.all_sequence_ids = np.arange(len(metadata.do_sample_array))
        self.processor.set_batch_configs(metadata.all_sequence_ids,
                                         metadata.top_k_array,
                                         metadata.top_p_array,
                                         metadata.do_sample_array,
                                         metadata.top_logprobs_array,
                                         metadata.seed_array,
                                         self.params.sampling_method)

    def _fusion_top_k_top_p_sampling(self, logits: ms.Tensor, index: ms.Tensor, metadata):
        # In order to support bf16 and avoid poor cast performance on the CPU, logits is casted to fp32 in advance.
        host_dtype = "float32"
        logits = logits.astype(ms.float32).asnumpy()
        index = index.asnumpy()
        logits_addr = logits.ctypes.data
        index_addr = index.ctypes.data

        batch_size = len(metadata.all_sequence_ids)
        token_ids_array, _ = self.processor.next_token_chooser(metadata.all_sequence_ids, logits_addr, index_addr,
                                                               batch_size, logits.shape[1],
                                                               metadata.max_logprobs, host_dtype,
                                                               self.speed_mode, self.use_approx)
        if self.speed_mode:
            token_ids_array = index[np.arange(index.shape[0]), token_ids_array]
        sampling_output = SamplingOutput(
            group_indices=metadata.group_indices,
            sequence_ids=metadata.all_sequence_ids,
            parent_sequence_ids=metadata.parent_sequence_ids,
            token_ids=token_ids_array[:, 0],
            logprobs=np.full((len(logits),), -9999.0, dtype=np.float32),
            top_token_ids=np.array([[]]),
            top_logprobs=np.array([[]]),
            cumulative_logprobs=np.zeros(batch_size, dtype=np.float32),
            repeating_indices=np.arange(batch_size),
            num_new_tokens=np.ones(len(logits), dtype=np.int64),
            num_top_tokens=metadata.num_top_tokens,
            seeds=metadata.seed_array
        )
        return sampling_output

    def _do_sampling(self, sorted_logits: ms.Tensor, sorted_indices: ms.Tensor, metadata: SamplingMetadata):
        do_sample_array = metadata.do_sample_array
        argmax_indices_array = np.squeeze(np.argwhere(np.equal(do_sample_array, 0)), 1)
        argmax_indices = ms.Tensor.from_numpy(argmax_indices_array)
        indices_array = np.where(do_sample_array > 0)[0]
        indices = ms.Tensor.from_numpy(indices_array)
        seed_array = metadata.seed_array[indices_array]

        sampled_indices = sorted_indices[indices]
        filtered_logits = sorted_logits[indices]
        sampled_probs = mint.nn.functional.softmax(filtered_logits, dim=-1)
        sampled_tokens = self.multinomial(sampled_probs, sampled_indices, 1,
                                          seed_array).squeeze(1)
        tokens = Tensor([-1] * len(do_sample_array), ms.int64)
        tokens = mint.scatter(tokens, 0, index=indices, src=sampled_tokens)
        if argmax_indices_array.size != 0:
            filtered_indices = sorted_indices[argmax_indices]
            selected_tokens = filtered_indices[:, 0]
            tokens = mint.scatter(tokens, 0, index=argmax_indices, src=selected_tokens)
        token_ids = tokens.reshape(-1).asnumpy()
        sampling_output = SamplingOutput(
            group_indices=metadata.group_indices,
            sequence_ids=metadata.all_sequence_ids,
            parent_sequence_ids=metadata.parent_sequence_ids,
            token_ids=token_ids,
            logprobs=np.full((len(token_ids),), -9999.0, dtype=np.float32),
            top_token_ids=np.array([[]]),
            top_logprobs=np.array([[]]),
            cumulative_logprobs=np.zeros(len(token_ids), dtype=np.float32),
            repeating_indices=np.arange(len(metadata.all_sequence_ids)),
            num_new_tokens=np.ones(len(token_ids), dtype=np.int64),
            num_top_tokens=metadata.num_top_tokens,
            seeds=metadata.seed_array
        )
        return sampling_output


@register_class('beam_search')
class BeamSearchTokenSelector(TokenSelector):
    def __call__(self, logits: ms.Tensor, metadata: SamplingMetadata):
        tokens = mint.argmax(logits, -1)
        return tokens


@register_class('greedy_search')
class GreedySearchTokenSelector(TokenSelector):
    def __call__(self, logits: ms.Tensor, metadata: SamplingMetadata):
        tokens = mint.argmax(logits, -1)
        sampling_output = SamplingOutput(
            group_indices=metadata.group_indices,
            sequence_ids=metadata.all_sequence_ids,
            parent_sequence_ids=metadata.parent_sequence_ids,
            token_ids=tokens.reshape(-1).asnumpy(),
            logprobs=np.full((len(logits),), -9999.0, dtype=np.float32),
            top_token_ids=np.array([[]]),
            top_logprobs=np.array([[]]),
            cumulative_logprobs=np.zeros(len(logits), dtype=np.float32),
            repeating_indices=np.arange(len(metadata.all_sequence_ids)),
            num_new_tokens=np.ones(len(logits), dtype=np.int64),
            num_top_tokens=metadata.num_top_tokens,
            seeds=metadata.seed_array
        )
        return sampling_output