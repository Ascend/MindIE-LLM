# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import array
from copy import deepcopy
from typing import Union, Tuple, Iterable, List, Callable
from functools import partial
import math
import torch
import numpy as np
import numpy.typing as npt

from .kvcache_settings import KVCacheSettings
from .config import CacheConfig, ContextParams, SpCpParallelInfo, DEFAULT_SAMPLING_PARAMS
from .input_metadata import InputMetadata, SAMPLING_DTYPE, SIMULATE_SEQUENCE_ID
from .sampling_metadata import SamplingMetadata
from .sampling_output import SamplingOutput

from ...utils.log.error_code import ErrorCode
from ...utils.log.logging import logger
from ...utils.tensor import tensor_backend
from ...modeling.backend_type import BackendType
from .stopping_criteria import make_mixed_eos, strings_eos
from .tg_decode_util import decode_one


class DictContext:
    """dictionary context, like lora_ids, stopping_criteria etc"""

    __slots__ = [
        "stopping_criteria",
        "string_stopping_criteria",
        "output_texts",
        "trace_ids",
        "lora_adapter_id",
        "random_number_generators"
    ]

    def __init__(self):
        self.stopping_criteria = {}
        self.string_stopping_criteria = {}
        self.output_texts = {}
        self.trace_ids = {}
        self.lora_adapter_id = {}
        self.random_number_generators = {}

    def add_context(self, context_handles, metadata: InputMetadata, **kwargs):
        batch_adapter_ids = metadata.adapter_ids
        if batch_adapter_ids:
            for i, idx in enumerate(context_handles):
                adapter_id = batch_adapter_ids[i]
                if adapter_id:
                    self.lora_adapter_id[idx] = adapter_id

        trace_ids = metadata.trace_ids
        for i, idx in enumerate(context_handles):
            trace_id = trace_ids[i]
            if not trace_id:
                trace_id = metadata.batch_request_ids[i]
            self.trace_ids[idx] = trace_id
        # others needs to be added here
    
    def fork_context(
        self, children_context_handles: npt.NDArray[np.int32], parents_context_handles: npt.NDArray[np.int32]
    ) -> None:
        """Fork the dictionary-based context fields from parents to children."""
        for i, child_idx in enumerate(children_context_handles):
            parent_idx = parents_context_handles[i]
            self.output_texts[child_idx] = self.output_texts.get(parent_idx)
            self.stopping_criteria[child_idx] = self.stopping_criteria.get(parent_idx)
            self.string_stopping_criteria[child_idx] = self.string_stopping_criteria.get(parent_idx)
            self.lora_adapter_id[child_idx] = self.lora_adapter_id.get(parent_idx)
            self.trace_ids[child_idx] = self.trace_ids.get(parent_idx)

    def clear_context(self, context_handles: Union[Iterable[int], npt.NDArray[np.int32]]):
        for context_handle in context_handles:
            self.clear_single_context(context_handle)

    def clear_single_context(self, context_handle: int):
        """originally clear_cache_in_dict"""
        if context_handle in self.stopping_criteria:
            del self.stopping_criteria[context_handle]
        if context_handle in self.string_stopping_criteria:
            del self.string_stopping_criteria[context_handle]
        if context_handle in self.output_texts:
            del self.output_texts[context_handle]
        if context_handle in self.trace_ids:
            del self.trace_ids[context_handle]
        if context_handle in self.lora_adapter_id:
            del self.lora_adapter_id[context_handle]

    def reset_all_context(self):
        self.output_texts.clear()
        self.stopping_criteria.clear()
        self.string_stopping_criteria.clear()
        self.lora_adapter_id.clear()
        self.trace_ids.clear()

    def append_and_return_output_text(self, context_handle, new_token):
        self.output_texts[context_handle] += new_token
        return self.output_texts[context_handle]

    def is_empty_stopping_criteria(self):
        return len(self.stopping_criteria) == 0

    def get_stopping_criteria(self, context_handles):
        return [self.stopping_criteria.get(context_handle) for context_handle in context_handles]

    def is_empty_string_stopping_criteria(self):
        return len(self.string_stopping_criteria) == 0

    def get_string_stopping_criteria(self, context_handles):
        return [self.string_stopping_criteria.get(context_handle) for context_handle in context_handles]

    def get_trace_ids(self, context_handles):
        return [self.trace_ids.get(context_handle) for context_handle in context_handles]


class NdarrayContext:
    """use slot pool to replace used index numpy flag vectors"""

    __slots__ = [
        "context_params",
        "default_sampling_params",
        "capacity",
        "pool",
        "cache_config",
        "last_input_ids",
        "last_position_ids",
        "seq_lens",
        "cpu_cached_seq_idx",
        "output_len_count",
        "used_block_idx",
        "used_block_offset",
        "cumulative_logprobs",
        "num_top_tokens",
        "all_input_ids",
        "all_output_ids",
        "pending_cleanup_flags",
        "sampling_params",
        "seeds",
        "best_of",
        "n",
        "use_beam_search",
        "ignore_eos",
        "include_stop",
        "skip_special_tokens",
        "base_capacity",
        "mtp_last_slots",
        "mtp_last_token_num",
        "mtp_hidden_states",
        "spcp_parallel_info",
        "mtp_last_rank",
        "mtp_seq_block_rank_id"
    ]

    def __init__(
        self,
        context_params: ContextParams,
        default_sampling_params: np.ndarray,
        cache_config: CacheConfig,
        spcp_parallel_info: SpCpParallelInfo,
        capacity=2048,
    ):
        self.context_params = context_params
        self.cache_config = cache_config
        start_slot_idx = 1 # reserve slot 0 for dummy batch and simulate inference
        self.pool = array.array("i", range(start_slot_idx, capacity))  # free slot index pool
        self.spcp_parallel_info = spcp_parallel_info
        self.capacity = capacity
        self.base_capacity = capacity

        self.default_sampling_params = default_sampling_params

        self.last_input_ids = np.zeros(self.capacity, dtype=np.int64)  # originally cached_input_ids
        self.last_position_ids = np.zeros(self.capacity, dtype=np.int32)  # originally cached_position_id
        self.seq_lens = np.zeros(self.capacity, dtype=np.int32)  # originally cached_seq_lens
        self.cpu_cached_seq_idx = np.zeros((self.capacity, self.spcp_parallel_info.scp_size),
                                           dtype=np.int32)  # related to sp cp
        self.output_len_count = np.zeros(self.capacity, dtype=np.int32)  # ??
        self.used_block_idx = np.zeros(self.capacity, dtype=np.int32)
        self.used_block_offset = np.zeros(self.capacity, dtype=np.int32)  # token offset in block
        self.cumulative_logprobs = np.zeros(self.capacity, dtype=np.float32)
        self.num_top_tokens = np.zeros(self.capacity, dtype=np.int32)
        max_seq_len = self.cache_config.max_seq_len
        max_gen_len = self.cache_config.max_gen_len
        # for async inference
        if self.context_params.async_infer:
            max_seq_len += 1
            max_gen_len += 1

        self.all_input_ids = np.full((self.capacity, max_seq_len), self.cache_config.pad_token_id, dtype=np.int32)
        self.all_output_ids = np.full((self.capacity, max_gen_len), self.cache_config.pad_token_id, dtype=np.int32)
        self.pending_cleanup_flags = np.zeros(self.capacity, dtype=np.bool_)

        sampling_params = [self.default_sampling_params for _ in range(self.capacity)]
        self.sampling_params = np.array(sampling_params)
        self.seeds = np.zeros(self.capacity, dtype=np.uint64)

        # multi-sequence
        self.best_of = np.ones(self.capacity, dtype=np.int32)
        self.n = np.ones(self.capacity, dtype=np.int32)
        self.use_beam_search = np.zeros(self.capacity, dtype=np.bool_)

        # stop
        self.ignore_eos = np.zeros(self.capacity, dtype=np.bool_)
        self.include_stop = np.zeros(self.capacity, dtype=np.bool_)
        self.skip_special_tokens = np.ones(self.capacity, dtype=np.bool_)

        # for mtp, just init, no matter if mtp is enabled or not
        self.mtp_last_slots = np.zeros((self.capacity, context_params.mtp_num_speculative_tokens + 1), dtype=np.int32)
        self.mtp_last_token_num = np.zeros(self.capacity, dtype=np.int32)
        self.mtp_hidden_states = tensor_backend.zeros(
            (self.capacity, context_params.mtp_num_speculative_tokens + 1, context_params.mtp_hidden_size),
            dtype=context_params.mtp_kv_dtype,
        )
        if self.spcp_parallel_info.scp_size > 1:
            self.mtp_last_rank = np.full(self.cache_config.cache_size, self.cache_config.pad_rank_id, dtype=np.int32)
            # column +1 for case where seq_len is almost equal to max_seq_len and is_append_block is true.
            self.mtp_seq_block_rank_id = np.full(
                (
                    self.cache_config.cache_size,
                    math.ceil((self.cache_config.max_seq_len + 1) / self.cache_config.max_block_size) + 1,
                ),
                self.cache_config.pad_rank_id,
                dtype=np.int32,
            )

        # cached_end_flag 结束标志，仅在非CB场景需要, never used, and now removed

    def fork_context(
        self, children_context_handles: npt.NDArray[np.int32], parents_context_handles: npt.NDArray[np.int32]
    ) -> None:
        """Copy data from parent to child in all_ndarray_context."""
        self.last_position_ids[children_context_handles] = self.last_position_ids[parents_context_handles]
        self.seq_lens[children_context_handles] = self.seq_lens[parents_context_handles]
        self.cpu_cached_seq_idx[children_context_handles] = self.cpu_cached_seq_idx[parents_context_handles]
        self.output_len_count[children_context_handles] = self.output_len_count[parents_context_handles]
        self.used_block_idx[children_context_handles] = self.used_block_idx[parents_context_handles]
        self.used_block_offset[children_context_handles] = self.used_block_offset[parents_context_handles]
        self.cumulative_logprobs[children_context_handles] = self.cumulative_logprobs[parents_context_handles]
        self.num_top_tokens[children_context_handles] = self.num_top_tokens[parents_context_handles]
        self.all_input_ids[children_context_handles] = self.all_input_ids[parents_context_handles]
        self.seeds[children_context_handles] = self.seeds[parents_context_handles]
        self.best_of[children_context_handles] = self.best_of[parents_context_handles]
        self.n[children_context_handles] = self.n[parents_context_handles]
        self.use_beam_search[children_context_handles] = self.use_beam_search[parents_context_handles]
        self.ignore_eos[children_context_handles] = self.ignore_eos[parents_context_handles]
        self.include_stop[children_context_handles] = self.include_stop[parents_context_handles]
        self.skip_special_tokens[children_context_handles] = self.skip_special_tokens[parents_context_handles]
        self.sampling_params[children_context_handles] = self.sampling_params[parents_context_handles]
        if self.context_params.mtp_enable:
            self.mtp_last_slots[children_context_handles] = self.mtp_last_slots[parents_context_handles]
            self.mtp_last_token_num[children_context_handles] = self.mtp_last_token_num[parents_context_handles]
            self.mtp_hidden_states[children_context_handles] = self.mtp_hidden_states[parents_context_handles]

    def clear_context(self, context_handles: Union[int, npt.NDArray[np.int32]]):
        """originally recover_default_cache, avoid single context reset!!"""
        if isinstance(context_handles, int):
            context_handles = np.array([context_handles], dtype=np.int32)
        for context_handle in context_handles:
            self._free_slot(context_handle)
        self.last_input_ids[context_handles] = 0  # no cleanup for input in recover_default_cache
        self.output_len_count[context_handles] = 0
        self.last_position_ids[context_handles] = 0
        self.seq_lens[context_handles] = 0
        self.cpu_cached_seq_idx[context_handles, :] = 0
        self.used_block_idx[context_handles] = 0
        self.used_block_offset[context_handles] = 0
        self.cumulative_logprobs[context_handles] = 0
        self.all_input_ids[context_handles, :] = self.cache_config.pad_token_id
        self.all_output_ids[context_handles, :] = self.cache_config.pad_token_id
        self.best_of[context_handles] = 1
        self.n[context_handles] = 1
        self.use_beam_search[context_handles] = False
        self.ignore_eos[context_handles] = False
        self.include_stop[context_handles] = False
        self.num_top_tokens[context_handles] = 0
        self.seeds[context_handles] = 0
        self.skip_special_tokens[context_handles] = True

        if self.context_params.mtp_enable:
            self.mtp_hidden_states[context_handles] = 0
            self.mtp_last_slots[context_handles] = 0
            self.mtp_last_token_num[context_handles] = 0
            if self.spcp_parallel_info.scp_size > 1:
                self.mtp_last_rank[context_handles] = self.cache_config.pad_rank_id
                self.mtp_seq_block_rank_id[context_handles, :] = self.cache_config.pad_rank_id

        self.sampling_params[context_handles] = self.default_sampling_params

    def allocate_slot(self) -> int:
        if self.pool:
            return self.pool.pop()
        self._grow_capacity()
        return self.pool.pop()

    def _free_slot(self, slot_idx: int) -> None:
        # In layerwise_disaggregated, slot 0 is reserved for dummy batches and recomputation and cannot be freed.
        if self.context_params.layerwise_disaggregated and slot_idx == 0:
            return
        self.pool.append(slot_idx)

    def _grow_capacity(self):
        self.pool.extend(range(self.capacity, self.capacity + self.base_capacity))
        self.capacity += self.base_capacity

        pad_args = {"pad_width": (0, self.base_capacity), "mode": "constant", "constant_values": (0, 0)}

        self.last_input_ids = np.pad(self.last_input_ids, **pad_args)
        self.last_position_ids = np.pad(self.last_position_ids, **pad_args)
        self.seq_lens = np.pad(self.seq_lens, **pad_args)
        self.cpu_cached_seq_idx = np.pad(self.cpu_cached_seq_idx, ((0, self.base_capacity), (0, 0)), constant_values=0)
        self.output_len_count = np.pad(self.output_len_count, **pad_args)
        self.used_block_idx = np.pad(self.used_block_idx, **pad_args)
        self.used_block_offset = np.pad(self.used_block_offset, **pad_args)
        self.cumulative_logprobs = np.pad(self.cumulative_logprobs, **pad_args)
        self.num_top_tokens = np.pad(self.num_top_tokens, **pad_args)
        # for async inference
        self.all_input_ids = np.pad(
            self.all_input_ids, ((0, self.base_capacity), (0, 0)), constant_values=self.cache_config.pad_token_id
        )
        self.all_output_ids = np.pad(
            self.all_output_ids, ((0, self.base_capacity), (0, 0)), constant_values=self.cache_config.pad_token_id
        )
        self.pending_cleanup_flags = np.pad(self.pending_cleanup_flags, **pad_args)

        grow_sampling_params = np.array([self.default_sampling_params for _ in range(self.base_capacity)])
        self.sampling_params = np.concatenate((self.sampling_params, grow_sampling_params), axis=0)

        self.seeds = np.pad(self.seeds, **pad_args)
        # multi-sequence
        self.best_of = np.pad(self.best_of, (0, self.base_capacity), constant_values=1)
        self.n = np.pad(self.n, (0, self.base_capacity), constant_values=1)
        self.use_beam_search = np.pad(self.use_beam_search, **pad_args)

        # stop
        self.ignore_eos = np.pad(self.ignore_eos, **pad_args)
        self.include_stop = np.pad(self.include_stop, **pad_args)
        self.skip_special_tokens = np.pad(self.skip_special_tokens, (0, self.base_capacity), constant_values=True)

        # for mtp
        self.mtp_last_slots = np.pad(self.mtp_last_slots, ((0, self.base_capacity), (0, 0)), constant_values=0)
        self.mtp_last_token_num = np.pad(self.mtp_last_token_num, **pad_args)
        self.mtp_hidden_states = tensor_backend.pad(
            self.mtp_hidden_states, (0, 0, 0, 0, 0, self.base_capacity), mode="constant", value=0
        )
        if self.spcp_parallel_info.scp_size > 1:
            self.mtp_last_rank = np.pad(
                self.mtp_last_rank, (0, self.capacity), constant_values=self.cache_config.pad_rank_id
            )
            self.mtp_seq_block_rank_id = np.pad(
                self.mtp_seq_block_rank_id,
                ((0, self.cache_config.cache_size), (0, 0)),
                constant_values=self.pad_rank_id,
            )
        logger.info(f"The capacity of context is expanded due to exhaustion. The current capacity is {self.capacity}")


class BatchContext:
    """BatchContext provides batch joinable context for InputMetadata and SamplingMetadata"""

    __slots__ = [
        "kvcache_settings",
        "kv_slots",
        "sequence_context_slot_map",
        "all_ndarray_context",
        "batch_context_config",
        "spcp_parallel_info",
        "context_params",
        "default_sampling_params",
        "all_dict_context",
        "device",
        "to_tensor",
        "position_ids_gen_func",
        "tokenizer",
        "tokenizer_sliding_window_size"
    ]

    def __init__(
        self,
        kvcache_settings: KVCacheSettings,
        context_params: ContextParams,  # misc params
        batch_context_config: CacheConfig,  # for sampling params
        spcp_parallel_info: SpCpParallelInfo,  # for sp, cp parallel info
        device,
        tokenizer,
        tokenizer_sliding_window_size,
        position_ids_gen_func: Callable = None
    ):
        self.kvcache_settings = kvcache_settings
        # block 0: [slot0, slot1, slot2, slot3]
        # block 1: [slot4, slot5, slot6, slot7]
        # kv_slots contains [[0, 1, 2, 3], [4, 5, 6, 7]]
        # originally CacheManager.slots
        self.kv_slots = np.arange(
            0, kvcache_settings.num_npu_blocks * kvcache_settings.block_size, dtype=np.int32
        ).reshape(kvcache_settings.num_npu_blocks, kvcache_settings.block_size)
        self.context_params = context_params
        self.sequence_context_slot_map = {}  # originally sequence_cache_map
        self.batch_context_config = batch_context_config
        self._init_default_sampling_params()

        self.spcp_parallel_info = spcp_parallel_info

        self.all_ndarray_context = NdarrayContext(
            context_params=context_params,
            default_sampling_params=self.default_sampling_params,
            cache_config=self.batch_context_config,
            spcp_parallel_info=self.spcp_parallel_info,
            capacity=self.batch_context_config.cache_size,
        )
        self.all_dict_context = DictContext()

        self.device = device

        def torch_to_tensor(data):
            return torch.tensor(data, device=self.device)

        self.to_tensor = torch_to_tensor

        self.position_ids_gen_func = position_ids_gen_func
        self.tokenizer = tokenizer
        self.tokenizer_sliding_window_size = tokenizer_sliding_window_size

    @staticmethod
    def replace_nans_with_default(
        default_sampling_params: np.ndarray,
        sampling_params: np.ndarray,
        batch_best_of: np.ndarray,
        batch_use_beam_search: np.ndarray,
        batch_logprobs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """sampling param ndarray has nans, replace them with default, originally preprocess_sampling_params
        process post-processing params according to their special combinations.

        This method accomplishes the following things:
        1. For requests where the do_sample parameter is not provided, if any of temperature, top_k, or top_p are passed
            in, it will set do_sample to `True`.
        2. Any missing parameters will be filled with their default values.
        3. If the `use_beam_search` of certain request is `True`, the corresponding `top_logprobs` must be > 0 to
            calculate cumulative logprobs.
        4. If logprobs is `True`, the corresponding top_logprobs must be > 0 to get at least one logprob when greedy
            searching.
        5. The `do_sample` will be set to True if `best_of` > 1 when `use_beam_search` is False.
        6. When temperature is set to 0, do_sample will be forcibly set to False, and temperature will be set to 1, to
            prevent potential issues in subsequent processing.

        Args:
            sampling_params: A numpy array containing the post-processing params of the current batch with the
                structural dtype of `SAMPLING_DTYPE`.
            batch_best_of: A numpy int array of the `best_of` parameters of the current batch.
            batch_use_beam_search: A numpy bool array of the `use_beam_search` parameters of the current batch.
            batch_logprobs: A numpy array of the (boolean) `logprobs` parameters of the current batch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The first array is processed `sampling_params` with the same dtype as the
                input `sampling_params`, and the second one is the `logprobs` before modified to 1 when best_of > 1.
        """
        temperature_key = "temperature"
        do_sample_key = "do_sample"
        top_logprobs_key = "top_logprobs"

        temperature_nan_indices = np.isnan(sampling_params[temperature_key])
        top_k_nan_indices = np.isnan(sampling_params["top_k"])
        top_p_nan_indices = np.isnan(sampling_params["top_p"])
        do_sample_nan_indices = np.isnan(sampling_params[do_sample_key])
        not_nan_indices = (~temperature_nan_indices | ~top_k_nan_indices | ~top_p_nan_indices) & do_sample_nan_indices
        sampling_params[do_sample_key][not_nan_indices] = True

        for name in sampling_params.dtype.names:
            nan_mask = np.isnan(sampling_params[name])
            sampling_params[name][nan_mask] = default_sampling_params[name]

        if batch_logprobs is None:
            batch_logprobs = np.zeros(len(sampling_params), dtype=np.bool_)
        else:
            batch_logprobs[np.vectorize(lambda x: x is None)(batch_logprobs)] = False
            batch_logprobs = batch_logprobs.astype(np.bool_)
        old_top_tokens_logprobs = deepcopy(sampling_params[top_logprobs_key]).astype(np.int32)
        if np.asarray(~batch_logprobs & (sampling_params[top_logprobs_key] > 0)).any():
            raise ValueError(
                "It is not supported to pass requests to `text_generator` with `top_logprobs` > 0 "
                "when `logprobs` is False."
            )
        need_logprobs_mask = (batch_use_beam_search | batch_logprobs) & (sampling_params[top_logprobs_key] == 0)
        sampling_params[top_logprobs_key][need_logprobs_mask] = 1

        temperature_zero_mask = sampling_params[temperature_key] == 0
        if (~batch_use_beam_search & temperature_zero_mask & (batch_best_of > 1)).any():
            raise ValueError(
                "It is not supported to pass requests to `text_generator` with `best_of` > 1 when "
                "`temperature` is 0."
            )
        sampling_params[do_sample_key][np.asarray(batch_best_of > 1)] = True
        closed_sampling_mask = temperature_zero_mask | batch_use_beam_search
        sampling_params[do_sample_key][closed_sampling_mask] = False
        sampling_params[temperature_key][temperature_zero_mask] = 1
        return sampling_params, old_top_tokens_logprobs

    @staticmethod
    def check_token_ids_within_vocab(token_ids, vocab_size: int):
        if not token_ids.size:
            return
        max_id = token_ids.max().item()
        if max_id > vocab_size:
            raise ValueError(f"The max of token_ids ({max_id}) is > vocab_size ({vocab_size}).")

    def allocate_context_slot(self, sequence_id) -> int:
        """originally append_cache and InputManager.get_cache_id, return context_handle"""
        context_slot = self.all_ndarray_context.allocate_slot()
        self.sequence_context_slot_map[sequence_id] = context_slot
        return context_slot

    def get_context_slot(self, sequence_id, is_prefill: bool) -> int:
        """read from sequence id to context slot map or allocate a new context slot ,
        originally InputManager.get_cache_id, return context_handle
        """
        if sequence_id in self.sequence_context_slot_map:
            return self.sequence_context_slot_map[sequence_id]
        else:
            if not is_prefill:
                message = (
                    f"There is no cached data for sequence {sequence_id}. This could be due to one of the "
                    f"following reasons: 1. It has not gone through the prefilling stage. 2. The sequence "
                    f"was determined to have stopped but was still passed into the decoding process."
                )
                logger.error(message, ErrorCode.TEXT_GENERATOR_MISSING_PREFILL_OR_INVALID_DECODE_REQ)
                raise RuntimeError(f"The sequence {sequence_id} is not prefilled before decoding.")
            return self.allocate_context_slot(sequence_id)

    def pop_context_handles(self, sequence_ids: Iterable[int]) -> List[int]:
        """return context_handles"""
        context_handles = []
        for sequence_id in sequence_ids:
            if sequence_id in self.sequence_context_slot_map:
                context_handles.append(self.sequence_context_slot_map.pop(sequence_id))
        return context_handles
    
    def filter_finished_context(self, context_handles):
        """return finished sequence ids and context handles if context_handles is in sequence_context_slot_map."""
        filtered_sequence_ids = []
        filtered_context_handles = []
        for k, v in self.sequence_context_slot_map.items():
            if v in context_handles:
                filtered_sequence_ids.append(k)
                # It is possible that some cache ids are not in map because of abortion.
                filtered_context_handles.append(v)
        if len(context_handles) != len(filtered_context_handles):
            filtered_context_handles = np.asarray(filtered_context_handles, dtype=np.int_)
        else:
            filtered_context_handles = context_handles
        filtered_sequence_ids = np.asarray(filtered_sequence_ids, dtype=np.int_)
        return filtered_sequence_ids, filtered_context_handles

    def clear_context_by_handles(self, context_handles: npt.NDArray[np.int32]):
        """originally clear_cache_by_ids"""
        if len(context_handles) == 0:
            return
        self.all_ndarray_context.clear_context(context_handles)
        self.all_dict_context.clear_context(context_handles)

    # originally update_cache_before_prefilling, and update_cache
    def update_context(
        self,
        context_handles: np.ndarray,  # internally context slot indices
        updated_ndarrays: Tuple[np.ndarray, ...],
        input_metadata: InputMetadata,
        sampling_args: Tuple[SamplingMetadata, SamplingOutput],  # when decode
        **kwargs,
    ) -> None:
        """originally BatchCache.update_cache returns logprobs, now None"""
        is_first_update = kwargs.get("is_first_update")
        if is_first_update:
            last_position_ids, input_lengths, prefill_new_tokens = updated_ndarrays
            self.all_ndarray_context.last_position_ids[context_handles] = last_position_ids
            self.all_ndarray_context.seq_lens[context_handles] = input_lengths

            if self.spcp_parallel_info.scp_size > 1:
                rr_sp_rank_id = input_metadata.sp_rank_id
                sp_tokens = input_metadata.sp_tokens
                pad_token_count = kwargs.get("pad_token_count")
                sp_tokens_copy = sp_tokens.copy()
                sp_tokens_copy[np.arange(len(context_handles)), rr_sp_rank_id] -= pad_token_count

                self.all_ndarray_context.cpu_cached_seq_idx[context_handles] = sp_tokens_copy - 1
                self.all_ndarray_context.last_position_ids[context_handles] -= pad_token_count
                self.all_ndarray_context.seq_lens[context_handles] -= pad_token_count
            else:
                self.all_ndarray_context.cpu_cached_seq_idx[context_handles, self.spcp_parallel_info.scp_rank] = (
                    input_lengths - 1
                )

            if input_metadata.batch_n is not None:
                input_metadata.batch_n[np.vectorize(lambda x: x is None)(input_metadata.batch_n)] = 1
                self.all_ndarray_context.n[context_handles] = input_metadata.batch_n

            if input_metadata.batch_best_of is not None:
                none_mask = np.vectorize(lambda x: x is None)(input_metadata.batch_best_of)
                input_metadata.batch_best_of[none_mask] = self.all_ndarray_context.n[context_handles][none_mask]
                self.all_ndarray_context.best_of[context_handles] = input_metadata.batch_best_of

            if input_metadata.batch_use_beam_search is not None:
                self.all_ndarray_context.use_beam_search[context_handles] = input_metadata.batch_use_beam_search

            if input_metadata.batch_ignore_eos is not None:
                self.all_ndarray_context.ignore_eos[context_handles] = input_metadata.batch_ignore_eos

            if input_metadata.batch_skip_special_tokens is not None:
                input_metadata.batch_skip_special_tokens[
                    np.vectorize(lambda x: x is None)(input_metadata.batch_skip_special_tokens)
                ] = True
                self.all_ndarray_context.skip_special_tokens[context_handles] = input_metadata.batch_skip_special_tokens

            if input_metadata.batch_include_stop is not None:
                self.all_ndarray_context.include_stop[context_handles] = input_metadata.batch_include_stop

            if input_metadata.batch_stop_strings:
                for i, idx in enumerate(context_handles):
                    stop_strings = input_metadata.batch_stop_strings[i]
                    if stop_strings:
                        if input_metadata.batch_use_beam_search is not None and input_metadata.batch_use_beam_search[i]:
                            logger.warning('The parameter `stop` is ignored for the request '
                                           f'{input_metadata.batch_request_ids[i]}'
                                           f', because it uses beam search which does not support it.')
                            continue
                        self.all_dict_context.output_texts[idx] = ""
                        self.all_dict_context.string_stopping_criteria[idx] = partial(
                            strings_eos, stop_strings=stop_strings
                        )

            if input_metadata.batch_stop_token_ids:
                for i, idx in enumerate(context_handles):
                    stop_token_ids = input_metadata.batch_stop_token_ids[i]
                    if stop_token_ids:
                        if input_metadata.batch_use_beam_search is not None and input_metadata.batch_use_beam_search[i]:
                            logger.warning('The parameter `stop_token_ids` is ignored for the request '
                                           f'{input_metadata.batch_request_ids[i]}'
                                           f', because it uses beam search which does not support it.')
                            continue
                        self.all_dict_context.output_texts[idx] = ""
                        self.all_dict_context.stopping_criteria[idx] = make_mixed_eos(stop_token_ids)

            if input_metadata.adapter_ids:
                for i, idx in enumerate(context_handles):
                    adapter_id = input_metadata.adapter_ids[i]
                    if adapter_id:
                        self.all_dict_context.lora_adapter_id[idx] = adapter_id

            if input_metadata.trace_ids:
                for i, idx in enumerate(context_handles):
                    trace_id = input_metadata.trace_ids[i]
                    if not trace_id:
                        trace_id = input_metadata.batch_request_ids[i]
                    self.all_dict_context.trace_ids[idx] = trace_id

            is_pd_separate = kwargs.get("is_pd_separate")
            if is_pd_separate:
                if any(np.asarray(input_metadata.batch_best_of > 1)):
                    raise RuntimeError("Prefilling-decoding separation deployment does not support best_of > 1.")
                for i, token in enumerate(prefill_new_tokens):
                    prefill_text = decode_one(self.tokenizer, [token], input_metadata.batch_skip_special_tokens[i],
                                              self.tokenizer_sliding_window_size)
                    self.all_dict_context.output_texts[context_handles[i]] = prefill_text

                self.all_ndarray_context.last_input_ids[context_handles] = prefill_new_tokens
                self.all_ndarray_context.all_output_ids[context_handles, 0] = prefill_new_tokens
                self.all_ndarray_context.output_len_count[context_handles] += len(prefill_new_tokens)
                self.all_ndarray_context.used_block_idx[context_handles] = (
                    self.all_ndarray_context.cpu_cached_seq_idx[context_handles, self.spcp_parallel_info.scp_rank]
                    // self.batch_context_config.max_block_size
                )
                self.all_ndarray_context.used_block_offset[context_handles] = (
                    self.all_ndarray_context.cpu_cached_seq_idx[context_handles, self.spcp_parallel_info.scp_rank]
                    % self.batch_context_config.max_block_size
                )
        else:
            sampling_metadata, sampling_output = sampling_args
            valid_indices = np.where((context_handles != -1) & (context_handles != 0))[0]
            valid_context_handles = context_handles[valid_indices]
            next_token_ids = sampling_output.token_ids[valid_indices]
            num_new_tokens = sampling_output.num_new_tokens[valid_indices]

            updating_valid_indices = valid_indices
            updating_valid_context_handles = valid_context_handles
            updating_token_ids = next_token_ids
            updating_num_tokens = num_new_tokens
            if input_metadata.batch_is_prefill is not None:  # splitfuse branch
                batch_is_prefill = input_metadata.batch_is_prefill
                batch_last_prompt = input_metadata.batch_last_prompt
                batch_is_prefill = batch_is_prefill[sampling_output.repeating_indices]
                batch_last_prompt = batch_last_prompt[sampling_output.repeating_indices]
                batch_is_prefill = batch_is_prefill[valid_indices]
                batch_last_prompt = batch_last_prompt[valid_indices]
                batch_is_prefill_bool = batch_is_prefill.astype(bool)
                batch_last_prompt_bool = batch_last_prompt.astype(bool)
                last_prefilling_mask = (~batch_is_prefill_bool) | batch_last_prompt_bool
                updating_valid_indices = updating_valid_indices[last_prefilling_mask]
                updating_valid_context_handles = updating_valid_context_handles[last_prefilling_mask]
                updating_token_ids = updating_token_ids[last_prefilling_mask]
                updating_num_tokens = updating_num_tokens[last_prefilling_mask]

            self.all_ndarray_context.last_input_ids[updating_valid_context_handles] = updating_token_ids[
                np.arange(len(updating_token_ids)), updating_num_tokens - 1
            ]

            output_indices = self.all_ndarray_context.output_len_count[
                updating_valid_context_handles, np.newaxis
            ] + np.arange(updating_token_ids.shape[1])
            all_token_indices = self.all_ndarray_context.seq_lens[
                updating_valid_context_handles, np.newaxis
            ] + np.arange(updating_token_ids.shape[1])

            output_indices_out_of_bound = (
                len(output_indices) > 0
                and output_indices.max().item() >= self.all_ndarray_context.all_output_ids.shape[1]
            )
            all_token_indices_out_of_bound = (
                len(all_token_indices) > 0
                and all_token_indices.max().item() >= self.all_ndarray_context.all_input_ids.shape[1]
            )

            if output_indices_out_of_bound or all_token_indices_out_of_bound:
                updating_valid_context_handles_list = updating_valid_context_handles.tolist()
                for i, cache_id in enumerate(updating_valid_context_handles_list):
                    if sampling_metadata is not None:
                        self.all_ndarray_context.all_input_ids[
                            cache_id,
                            self.all_ndarray_context.seq_lens[cache_id]: self.all_ndarray_context.seq_lens[cache_id]
                            + updating_num_tokens[i],
                        ] = updating_token_ids[i][: updating_num_tokens[i]]
                    self.all_ndarray_context.all_output_ids[
                        cache_id,
                        self.all_ndarray_context.output_len_count[cache_id]: self.all_ndarray_context.output_len_count[
                            cache_id
                        ]
                        + updating_num_tokens[i],
                    ] = updating_token_ids[i][: updating_num_tokens[i]]
            else:
                self.all_ndarray_context.all_output_ids[
                    updating_valid_context_handles.reshape(-1, 1), output_indices
                ] = updating_token_ids
                if sampling_metadata is not None:
                    self.all_ndarray_context.all_input_ids[
                        updating_valid_context_handles.reshape(-1, 1), all_token_indices
                    ] = updating_token_ids

            if sampling_metadata is not None and sampling_metadata.is_prefill and sampling_output.seeds is not None:
                self.all_ndarray_context.seeds[updating_valid_context_handles] = sampling_output.seeds[
                    updating_valid_indices
                ]
            if sampling_output.logprobs is not None:
                next_logprobs = sampling_output.logprobs[updating_valid_indices]
                beam_mask = self.all_ndarray_context.use_beam_search[updating_valid_context_handles]
                self.all_ndarray_context.cumulative_logprobs[updating_valid_context_handles[beam_mask]] += np.sum(
                    next_logprobs[beam_mask], axis=1
                )

            self.all_ndarray_context.output_len_count[updating_valid_context_handles] += updating_num_tokens
            self.all_ndarray_context.seq_lens[updating_valid_context_handles] += updating_num_tokens
            if self.spcp_parallel_info.scp_size > 1:
                # Filter sp_rank_id using updating_valid_indices to match the shape of updating_valid_context_handles
                updating_sp_rank_id = input_metadata.sp_rank_id[updating_valid_indices]
                self.all_ndarray_context.cpu_cached_seq_idx[updating_valid_context_handles, updating_sp_rank_id] \
                    += updating_num_tokens - 1
            else:
                self.all_ndarray_context.cpu_cached_seq_idx[
                    updating_valid_context_handles, self.spcp_parallel_info.scp_rank] += updating_num_tokens
            self.all_ndarray_context.last_position_ids[updating_valid_context_handles] += updating_num_tokens

            self.all_ndarray_context.used_block_idx[valid_context_handles] = (
                self.all_ndarray_context.cpu_cached_seq_idx[valid_context_handles, self.spcp_parallel_info.scp_rank]
                // self.batch_context_config.max_block_size
            )
            self.all_ndarray_context.used_block_offset[valid_context_handles] = (
                self.all_ndarray_context.cpu_cached_seq_idx[valid_context_handles, self.spcp_parallel_info.scp_rank]
                % self.batch_context_config.max_block_size
            )

    def join_context(self, context_handles: np.ndarray, metadata: InputMetadata, hit_mask: np.ndarray = None):
        """Join other batch context to self, originally get_cache_tensor

        Args:
            context_handles: Array of context handles
            metadata: Input metadata
            hit_mask: Hit mask, used in asynchronous inference scenarios, indicates whether the context_handles is
            same between current turn and last turn.

        Returns:
            A tuple containing information such as input IDs, position IDs, block tables, and slots
        """
        if self.spcp_parallel_info.scp_size > 1:
            if not self.context_params.async_infer:
                self.all_ndarray_context.cpu_cached_seq_idx[context_handles, metadata.sp_rank_id] += 1
                self.all_ndarray_context.used_block_idx[context_handles] = \
                    (self.all_ndarray_context.cpu_cached_seq_idx[context_handles, self.spcp_parallel_info.scp_rank]
                    // self.batch_context_config.max_block_size)
                self.all_ndarray_context.used_block_offset[context_handles] = \
                    (self.all_ndarray_context.cpu_cached_seq_idx[context_handles, self.spcp_parallel_info.scp_rank]
                    % self.batch_context_config.max_block_size)
            if self.context_params.mtp_enable:
                append_block_mask = metadata.is_append_block
                self.all_ndarray_context.mtp_seq_block_rank_id[
                    context_handles[append_block_mask],
                    np.argmin(
                        self.all_ndarray_context.mtp_seq_block_rank_id[context_handles[append_block_mask]], axis=1
                    ),
                ] = metadata.block_rank_id[append_block_mask]

        # 初始化块索引和偏移量
        used_block_idx = self.all_ndarray_context.used_block_idx[context_handles]
        used_block_offset = self.all_ndarray_context.used_block_offset[context_handles]

        if self.context_params.async_infer and hit_mask is not None:
            # sequence ids和cache ids是一一对应的
            if self.spcp_parallel_info.scp_size > 1:
                self.all_ndarray_context.cpu_cached_seq_idx[context_handles, metadata.sp_rank_id] += 1
            cpu_cached_seq_token_idx = self.all_ndarray_context.cpu_cached_seq_idx[
                context_handles, self.spcp_parallel_info.scp_rank]
            if self.spcp_parallel_info.scp_size == 1:
                cpu_cached_seq_token_idx[hit_mask] += self.context_params.max_generated_tokens
            used_block_idx = cpu_cached_seq_token_idx // self.batch_context_config.max_block_size
            used_block_offset = cpu_cached_seq_token_idx % self.batch_context_config.max_block_size

        slots = None
        if not self.context_params.mtp_enable:
            block_idx = metadata.batch_block_tables[range(len(metadata.all_sequence_ids)), used_block_idx]
            # make sure the best way
            slots = self.kv_slots[block_idx, used_block_offset]
            if self.spcp_parallel_info.scp_size > 1:
                rank_ids = metadata.sp_rank_id
                mask = (rank_ids % self.spcp_parallel_info.scp_size != self.spcp_parallel_info.scp_rank)
                slots[mask] = -1
            # 虚推请求不写入 KV cache，设置 slots = -1 跳过 ReshapeAndCache
            simulate_infer_mask = (metadata.all_sequence_ids == SIMULATE_SEQUENCE_ID)
            if simulate_infer_mask.any():
                slots[simulate_infer_mask] = -1
        
        input_lengths = self.all_ndarray_context.seq_lens[context_handles]
        adapter_ids = [self.all_dict_context.lora_adapter_id.get(idx) for idx in context_handles]
        # delete unused ret: cu_seqlen_prefill and prefill_head_indices
        ret = (
            self.all_ndarray_context.last_input_ids[context_handles],
            self.all_ndarray_context.last_position_ids[context_handles],
            slots,
            input_lengths,
            input_lengths.max(),
            adapter_ids
        )
        return ret

    def fork_context(
        self, children_context_handles: npt.NDArray[np.int32], parents_context_handles: npt.NDArray[np.int32]
    ) -> None:
        """Fork context from parents to children by copying both ndarray and dict contexts."""
        self.all_ndarray_context.fork_context(children_context_handles, parents_context_handles)
        self.all_dict_context.fork_context(children_context_handles, parents_context_handles)

    # for original batchCache use
    def block_to_slots(self, block_id: npt.NDArray[np.int32], slot_offset_in_block: npt.NDArray[np.int32]):
        return self.kv_slots[block_id, slot_offset_in_block]

    # for original input manager use
    def block_table_to_slots(self, block_table: npt.NDArray[np.int32]):
        return self.kv_slots[block_table]

    # region sampling related methods
    def build_sampling_meta(
        self, context_handles: np.ndarray, metadata: InputMetadata, is_prefill: bool
    ) -> SamplingMetadata:
        """originally build_sampling_metadata_from_prefilling, build_sampling_metadata_from_cache
        their difference is on preprocess_sampling_params, should be unified
        args:
            context_handles, input_metadata
        """
        if is_prefill:
            start = 0
            for i, (length) in enumerate(metadata.batch_seq_len):
                self.all_ndarray_context.all_input_ids[context_handles[i], :length] = metadata.input_ids[
                    start:start + length
                ]
                start += length

            batch_best_of = self.all_ndarray_context.best_of[context_handles]
            batch_use_beam_search = self.all_ndarray_context.use_beam_search[context_handles]

            batch_sampling_params, old_top_tokens_logprobs = BatchContext.replace_nans_with_default(
                self.default_sampling_params,
                metadata.batch_sampling_params,
                batch_best_of,
                batch_use_beam_search,
                metadata.batch_logprobs,
            )

            sampling_param_msg = f"""
            Sampling parameters for trace ids {metadata.trace_ids}:
            {{
                "temperature": {batch_sampling_params['temperature']},
                "top_k": {batch_sampling_params['top_k']},
                "top_p": {batch_sampling_params['top_p']},
                "do_sample": {batch_sampling_params['do_sample']},
                "seed": {metadata.batch_seeds},
                "repetition_penalty": {batch_sampling_params['repetition_penalty']},
                "frequency_penalty": {batch_sampling_params['frequency_penalty']},
                "presence_penalty": {batch_sampling_params['presence_penalty']},
                "ignore_eos": {metadata.batch_ignore_eos}
            }}
            """
            logger.debug(sampling_param_msg)

            self.all_ndarray_context.sampling_params[context_handles] = batch_sampling_params
            self.all_ndarray_context.num_top_tokens[context_handles] = old_top_tokens_logprobs

            metadata.batch_seeds[np.vectorize(lambda x: x is None)(metadata.batch_seeds)] = 0

            if metadata.batch_seeds is not None and self.context_params.generator_backend_type == BackendType.TORCH:
                for i, idx in enumerate(context_handles):
                    seed = metadata.batch_seeds[i]
                    gen = torch.Generator(device=self.device)
                    if seed is not None:
                        gen.manual_seed(int(seed))
                    else:
                        gen.manual_seed(i)
                    self.all_dict_context.random_number_generators[idx] = gen

            sampling_metadata = SamplingMetadata.from_batch(
                input_metadata=metadata,
                batch_sampling_params=batch_sampling_params,
                num_top_tokens=self.all_ndarray_context.num_top_tokens[context_handles],
                to_tensor=self.to_tensor,
                batch_seeds=metadata.batch_seeds,
                batch_best_of=metadata.batch_best_of,
                batch_n=metadata.batch_n,
                batch_use_beam_search=metadata.batch_use_beam_search,
                batch_output_lengths=self.all_ndarray_context.output_len_count[context_handles],
                batch_cumulative_logprobs=self.all_ndarray_context.cumulative_logprobs[context_handles],
                random_number_generators=[
                    self.all_dict_context.random_number_generators.get(idx, None)
                    for idx in context_handles
                ]
            )

            all_token_ids = self.all_ndarray_context.all_input_ids[context_handles, : metadata.max_seq_len]
            sampling_metadata.update_token_ids(all_token_ids, None)

        else:
            sampling_metadata = SamplingMetadata.from_batch(
                input_metadata=metadata,
                batch_sampling_params=self.all_ndarray_context.sampling_params[context_handles],
                num_top_tokens=self.all_ndarray_context.num_top_tokens[context_handles],
                to_tensor=self.to_tensor,
                batch_seeds=self.all_ndarray_context.seeds[context_handles],
                batch_best_of=self.all_ndarray_context.best_of[context_handles],
                batch_n=self.all_ndarray_context.n[context_handles],
                batch_use_beam_search=self.all_ndarray_context.use_beam_search[context_handles],
                batch_output_lengths=self.all_ndarray_context.output_len_count[context_handles],
                batch_cumulative_logprobs=self.all_ndarray_context.cumulative_logprobs[context_handles],
                random_number_generators=[
                    self.all_dict_context.random_number_generators.get(idx, None)
                    for idx in context_handles
                ]
            )

        return sampling_metadata

    def sync_sampling_token_ids(
        self, context_handles: npt.NDArray[np.int32], sampling_metadata, max_seq_len
    ) -> SamplingMetadata:
        """originally update_sampling_metadata, sync token ids in sampling metadata with context"""
        max_out_len = self.all_ndarray_context.output_len_count[context_handles].max()
        all_token_ids = self.all_ndarray_context.all_input_ids[context_handles, :max_seq_len]
        output_token_ids = self.all_ndarray_context.all_output_ids[context_handles, :max_out_len]
        self.check_token_ids_within_vocab(output_token_ids, self.batch_context_config.vocab_size)
        sampling_metadata.update_token_ids(all_token_ids, output_token_ids)
        cumulative_logprobs = self.all_ndarray_context.cumulative_logprobs[context_handles]
        output_lengths = self.all_ndarray_context.output_len_count[context_handles]
        sampling_metadata.update_beam_search(cumulative_logprobs, output_lengths)
        return sampling_metadata

    def get_mix_decode_cache_for_splitfuse(self, context_handles, decode_idx, metadata, hit_mask=None):
        input_ids = self.all_ndarray_context.last_input_ids[context_handles]
        max_seq_len = self.all_ndarray_context.seq_lens[context_handles].max()
        position_ids = self.all_ndarray_context.last_position_ids[context_handles]
        input_lengths = self.all_ndarray_context.seq_lens[context_handles]

        if hit_mask is not None:
            hit_mask = hit_mask[decode_idx]
            cpu_cached_seq_token_idx = self.all_ndarray_context.cpu_cached_seq_idx[
                context_handles, self.spcp_parallel_info.scp_rank]
            cpu_cached_seq_token_idx[hit_mask] += 1
            used_block_idx = cpu_cached_seq_token_idx // self.batch_context_config.max_block_size
            used_block_offset = cpu_cached_seq_token_idx % self.batch_context_config.max_block_size
            block_idx = metadata.batch_block_tables[decode_idx, used_block_idx]
            slots = self.kv_slots[block_idx, used_block_offset]
        else:
            block_idx = metadata.batch_block_tables[
                decode_idx, self.all_ndarray_context.used_block_idx[context_handles]]
            slots = self.block_to_slots(block_idx, self.all_ndarray_context.used_block_offset[context_handles])
        decode_result = (input_ids, max_seq_len, position_ids, input_lengths, slots)
        return decode_result

    def build_sampling_meta_for_splitfuse(self, context_handles: np.ndarray, metadata: InputMetadata, prefill_seq_idx):
        batch_size = metadata.batch_size
        num_top_tokens = np.zeros(batch_size, dtype=np.int32)
        start = sum(not is_prefill_tmp for is_prefill_tmp in metadata.batch_is_prefill)
        for i in prefill_seq_idx:
            length = metadata.batch_seq_len[i]
            start_pos = metadata.split_start_position[i]
            self.all_ndarray_context.all_input_ids[context_handles[i], start_pos: start_pos + length] = \
                metadata.input_ids[start: start + length]
            start += length

        prefill_req_idx = prefill_seq_idx
        if any(seq_ids.shape[0] > 1 for seq_ids in metadata.batch_sequence_ids):
            prefill_req_idx = [idx_ for idx_, _seq in enumerate(metadata.batch_sequence_ids) if _seq.shape[0] == 1]

        batch_best_of = self.all_ndarray_context.best_of[context_handles][prefill_req_idx]
        batch_use_beam_search = self.all_ndarray_context.use_beam_search[context_handles][prefill_req_idx]
        batch_sampling_params = metadata.batch_sampling_params[prefill_req_idx]
        metadata.batch_sampling_params[prefill_req_idx], num_top_tokens[prefill_req_idx] = \
            BatchContext.replace_nans_with_default(
                self.default_sampling_params,
                batch_sampling_params, batch_best_of, batch_use_beam_search, metadata.batch_logprobs[prefill_req_idx])

        sampling_param_msg = f"""
        Sampling parameters for trace ids {metadata.trace_ids}:
        {{
            "temperature": {metadata.batch_sampling_params[prefill_req_idx]['temperature']},
            "top_k": {metadata.batch_sampling_params[prefill_req_idx]['top_k']},
            "top_p": {metadata.batch_sampling_params[prefill_req_idx]['top_p']},
            "do_sample": {metadata.batch_sampling_params[prefill_req_idx]['do_sample']},
            "seed": {metadata.batch_seeds},
            "repetition_penalty": {metadata.batch_sampling_params[prefill_req_idx]['repetition_penalty']},
            "frequency_penalty": {metadata.batch_sampling_params[prefill_req_idx]['frequency_penalty']},
            "presence_penalty": {metadata.batch_sampling_params[prefill_req_idx]['presence_penalty']},
            "ignore_eos": {metadata.batch_ignore_eos}
        }}
        """
        logger.debug(sampling_param_msg)

        # 刷新采样参数和top_tokens的cache
        count = 0
        for idx, cache_id in enumerate(context_handles):
            if idx in prefill_seq_idx:
                self.all_ndarray_context.sampling_params[cache_id] = metadata.batch_sampling_params[
                    prefill_req_idx[count]]
                self.all_ndarray_context.num_top_tokens[cache_id] = num_top_tokens[
                    prefill_req_idx[count]]
                count += 1

        # 如果是beamsearch场景需要计算beamsize偏移量
        all_idx = set(range(len(metadata.batch_sampling_params)))
        decode_idx = all_idx - set(prefill_req_idx)
        offset = 0
        for idx in decode_idx:
            metadata.batch_sampling_params[idx] = self.all_ndarray_context.sampling_params[context_handles[offset]]
            offset += len(metadata.batch_sequence_ids[idx])

        # beamsearch场景下 根据 idx 中每个子数组的长度重复 data 中的元素
        def expend_data(data, lengths):
            return np.concatenate([np.repeat(data[i], lengths[i]) for i in range(len(data))])

        # temperature, topk, topp, typicalp, do_sample, seeds, repetition, frequency, presence, watermark
        lengths = [len(arr) for arr in metadata.batch_sequence_ids]
        sampling_metadata = SamplingMetadata.from_batch(
            input_metadata=metadata,
            batch_sampling_params=expend_data(metadata.batch_sampling_params, lengths),
            batch_n=self.all_ndarray_context.n[context_handles],
            batch_seeds=self.all_ndarray_context.seeds[context_handles],
            batch_best_of=self.all_ndarray_context.best_of[context_handles],
            num_top_tokens=self.all_ndarray_context.num_top_tokens[context_handles],
            batch_output_lengths=self.all_ndarray_context.output_len_count[context_handles],
            batch_use_beam_search=self.all_ndarray_context.use_beam_search[context_handles],
            batch_cumulative_logprobs=self.all_ndarray_context.cumulative_logprobs[context_handles],
            to_tensor=self.to_tensor,
            is_seq_prefill=metadata.batch_is_prefill,
            is_mix=metadata.is_mix
        )
        return sampling_metadata

    def update_context_for_splitfuse(self, metadata, context_handles, input_lengths, last_position_ids):
        # 由于后处理参数的刷新只有最后一个prefill切块才有，而postion和Input需要每个切块刷新，因此分阶段刷新

        # 用全部的 seq_id 展开
        req_indices = np.concatenate(
            [np.repeat(np.array([idx]), len(value)) for idx, value in enumerate(metadata.batch_sequence_ids)])
        prefill_seq_idx = np.where(metadata.batch_is_prefill)[0]
        end_pos_seq_idx = np.where(metadata.batch_last_prompt)[0]

        prefill_seq_last_time_idx = np.intersect1d(prefill_seq_idx, end_pos_seq_idx)
        prefill_req_last_time_idx = req_indices[
            np.where(np.logical_and(metadata.batch_last_prompt, metadata.batch_is_prefill))[0]]
        prefill_req_other_time_idx = req_indices[np.setdiff1d(prefill_seq_idx, prefill_req_last_time_idx)]

        # 刷新最后一次prefill的部分
        if len(prefill_req_last_time_idx) != 0:
            new_metadata = deepcopy(metadata)
            new_metadata.batch_best_of = metadata.batch_best_of[prefill_req_last_time_idx] \
                if metadata.batch_best_of is not None else None
            new_metadata.batch_ignore_eos = metadata.batch_ignore_eos[prefill_req_last_time_idx] \
                if metadata.batch_ignore_eos is not None else None
            new_metadata.batch_include_stop = metadata.batch_include_stop[prefill_req_last_time_idx] \
                if metadata.batch_include_stop is not None else None
            new_metadata.batch_logprobs = metadata.batch_logprobs[prefill_req_last_time_idx] \
                if metadata.batch_logprobs is not None else None
            new_metadata.batch_skip_special_tokens = metadata.batch_skip_special_tokens[prefill_req_last_time_idx] \
                if metadata.batch_skip_special_tokens is not None else None
            new_metadata.batch_stop_strings = [metadata.batch_stop_strings[j] for j in prefill_req_last_time_idx] \
                if metadata.batch_stop_strings is not None else None
            new_metadata.batch_stop_token_ids = [metadata.batch_stop_token_ids[j] for j in prefill_req_last_time_idx] \
                if metadata.batch_stop_token_ids is not None else None
            new_metadata.trace_ids = [metadata.trace_ids[j] for j in prefill_req_last_time_idx] \
                if metadata.trace_ids is not None else None
            new_metadata.batch_request_ids = metadata.batch_request_ids[prefill_req_last_time_idx] \
                if metadata.batch_request_ids is not None else None

            new_metadata.batch_use_beam_search = metadata.batch_use_beam_search[prefill_req_last_time_idx] \
                if metadata.batch_use_beam_search is not None else None

            new_metadata.batch_n = metadata.batch_n[prefill_req_last_time_idx] \
                if metadata.batch_n is not None else None

            self.update_context(
                context_handles[prefill_seq_last_time_idx],
                (
                    last_position_ids[prefill_seq_last_time_idx],
                    input_lengths[prefill_seq_last_time_idx],
                    None
                ),
                new_metadata,
                sampling_args=(None, None),
                is_first_update=True
            )

        # 刷新非最后一次prefill的部分
        if len(prefill_req_other_time_idx) != 0:
            context_handles = context_handles[prefill_req_other_time_idx]
            self.all_ndarray_context.last_position_ids[context_handles] = \
                last_position_ids[prefill_req_other_time_idx]
            self.all_ndarray_context.seq_lens[context_handles] = input_lengths[prefill_req_other_time_idx]
            self.all_ndarray_context.cpu_cached_seq_idx[context_handles, self.spcp_parallel_info.scp_rank] = \
                input_lengths[prefill_req_other_time_idx] - 1

    def reset_all_context(self):
        all_indices_except_0 = np.arange(self.all_ndarray_context.capacity)[1:]
        self.all_ndarray_context.clear_context(all_indices_except_0)
        if self.context_params.async_infer:
            self.all_ndarray_context.pending_cleanup_flags[:] = False
        self.all_dict_context.reset_all_context()
        self.sequence_context_slot_map.clear()

    def _init_default_sampling_params(self):
        default_sampling_params = []
        for key, value in DEFAULT_SAMPLING_PARAMS.items():
            if getattr(self.batch_context_config.model_wrapper_config, key, None) is not None:
                default_sampling_params.append(getattr(self.batch_context_config.model_wrapper_config, key))
            else:
                default_sampling_params.append(value)
        self.default_sampling_params = np.array(tuple(default_sampling_params), dtype=SAMPLING_DTYPE)
