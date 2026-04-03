# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, Tuple, Callable
import numpy as np
import numpy.typing as npt

from mindie_llm.utils.log.logging import logger
from .input_metadata import SIMULATE_SEQUENCE_ID
from .kvcache_settings import KVCacheSettings
from .batch_context import BatchContext
from .input_metadata import InputMetadata
from .model_input import ModelInput
from .request_sampling_cache import RequestsSamplingCache
from .sampling_metadata import SamplingMetadata
from .sampling_output import SamplingOutput
from .config import CacheConfig, ContextParams, SpCpParallelInfo


def prepare_cp_input(metadata: InputMetadata, cp_size: int):
    '''
    generate pad_token_count and cp_tokens
    update metadata for cp
    '''
    pad_token_count = (-metadata.batch_seq_len) % (2 * cp_size)
    metadata.batch_seq_len += pad_token_count
    metadata.sp_tokens[np.arange(metadata.batch_size), metadata.sp_rank_id] += pad_token_count
    metadata.total_seq_num = metadata.batch_seq_len.sum()
    metadata.max_seq_len = metadata.batch_seq_len.max()
    cp_tokens = np.repeat(metadata.batch_seq_len // cp_size, cp_size).reshape(-1, cp_size)
    return pad_token_count, cp_tokens


class TGInferContextStore:
    """
    TGInferContextStore (originally InputManager) is used to manage the infer context of the text generator.
    The context is required for ModelInputs composing to avoid frequent dispatch from scheduler
    Important contexts include batch forward needed context, sampling context.
    """

    __slots__ = [
        "_batch_context",  # forward required input and sampling context
        "kvcache_settings",
        "last_sampling_metadata",  # last time accessed sampling metadata, weird design
        "spcp_parallel_info",
        "context_params",
        "device",
        "aborted_context_handles",
        "cache_config",
        "block_size",
    ]

    def __init__(
        self,
        kvcache_settings: KVCacheSettings,
        batch_context_config: CacheConfig,  # should be merged with context_params
        # from model_wrapper.mapping.attn_cp, model_wrapper.mapping.attn_inner_sp
        spcp_parallel_info,
        device,  # from model_wrapper device
        context_params: ContextParams,
        tokenizer,
        tokenizer_sliding_window_size,
        position_ids_gen_func: Callable = None,
    ):
        self.block_size = batch_context_config.max_block_size
        self.kvcache_settings = kvcache_settings
        self.last_sampling_metadata = RequestsSamplingCache()
        if spcp_parallel_info is not None:
            self.spcp_parallel_info = SpCpParallelInfo(spcp_parallel_info[0], spcp_parallel_info[1])
        else:
            self.spcp_parallel_info = SpCpParallelInfo() # for mindspore backend
        self.context_params = context_params
        self.device = device
        self._batch_context = BatchContext(
            kvcache_settings,
            context_params,
            batch_context_config,
            self.spcp_parallel_info,
            device,
            tokenizer,
            tokenizer_sliding_window_size,
            position_ids_gen_func,
        )
        self.aborted_context_handles = []
        self.cache_config = batch_context_config

    def get_batch_context_handles(self, metadata: InputMetadata) -> np.ndarray:
        """get batch context handles, originally save_input_cache, allocate new context handles or get existing context
        handles move InputManager.get_cache_id here
        """
        context_handles = np.zeros(len(metadata.all_sequence_ids), np.int32)
        if metadata.is_dummy_batch:
            return context_handles
        for i, sequence_id in enumerate(metadata.all_sequence_ids):
            if sequence_id == SIMULATE_SEQUENCE_ID:
                context_handles[i] = 0
            # metadata.is_prefill is used to determine if the batch has prefill sequence
            # might hide bugs for mixed prefill/decode batch
            else:
                context_handles[i] = self._batch_context.get_context_slot(sequence_id, metadata.is_prefill)
        return context_handles

    def compose_model_inputs(
            self, metadata: InputMetadata, context_handles: Optional[np.ndarray], **kwargs
    ) -> Tuple:
        """
        originally concatenate, add context when prefilling or join context when decoding
        kwargs:
        """
        pad_token_count = None
        cp_tokens = None
        sampling_metadata = None
        if metadata.is_prefill:
            # prefill阶段无法使用缓存
            pad_token_count = np.zeros(metadata.batch_size, dtype=np.int32)
            if self.spcp_parallel_info.scp_size > 1:
                metadata.batch_block_tables = metadata.batch_block_tables[:, self.spcp_parallel_info.scp_rank, :]
                if self.spcp_parallel_info.cp_size > 1:
                    pad_token_count, cp_tokens = prepare_cp_input(metadata, self.spcp_parallel_info.cp_size)
                if self.cache_config.bos_token_id == -1:
                    raise RuntimeError(
                        "BOS token ID is invalid (-1). This token is required for proper sequence processing. "
                        "Please check your model configuration and ensure a valid BOS token ID is provided."
                    )
                metadata.input_ids = np.where(metadata.input_ids == -1, self.cache_config.bos_token_id,
                                            metadata.input_ids)
            last_input_ids, position_ids, all_tokens_kv_slots, seq_lengths, prefill_head_indices = (
                self._update_context_before_prefill(
                    context_handles,
                    metadata,
                    self._batch_context.position_ids_gen_func,
                    pad_token_count=pad_token_count,
                    **kwargs,
                )
            )
            max_seq_len = metadata.max_seq_len
            adapter_ids = metadata.adapter_ids
            if metadata.has_sampling:
                sampling_metadata = self._batch_context.build_sampling_meta(context_handles, metadata, is_prefill=True)
                self.last_sampling_metadata.add_to_cache(metadata.all_sequence_ids, sampling_metadata)  # !!!!
        else:
            if self.spcp_parallel_info.scp_size > 1:
                metadata.batch_block_tables = metadata.batch_block_tables[:, self.spcp_parallel_info.scp_rank, :]
            last_input_ids, position_ids, all_tokens_kv_slots, seq_lengths, max_seq_len, adapter_ids = (
                self._batch_context.join_context(context_handles, metadata, kwargs.get("hit_mask", None))
            )
            prefill_head_indices = None

            if metadata.has_sampling:
                sampling_metadata = self.last_sampling_metadata.get_from_cache(
                    sequence_ids=metadata.all_sequence_ids
                )
                if sampling_metadata is None or sampling_metadata.is_prefill:
                    sampling_metadata = self._batch_context.build_sampling_meta(
                        context_handles, metadata, is_prefill=False
                    )
                    self.last_sampling_metadata.add_to_cache(
                        sequence_ids=metadata.all_sequence_ids, sampling_metadata=sampling_metadata
                    )
                self._batch_context.sync_sampling_token_ids(context_handles, sampling_metadata, max_seq_len)
        trace_ids = None
        if context_handles is not None:
            trace_ids = self._batch_context.all_dict_context.get_trace_ids(context_handles)

        model_inputs = ModelInput(
            input_ids=last_input_ids,
            position_ids=position_ids,
            block_tables=metadata.batch_block_tables,
            slots=all_tokens_kv_slots,
            context_length=seq_lengths,
            cached_context_length=seq_lengths,
            max_seq_len=max_seq_len,
            prefill_head_indices=prefill_head_indices,
            is_prefill=metadata.is_prefill,
            adapter_ids=adapter_ids,
            dp_rank_ids=metadata.batch_dp_rank_ids,
            sp_tokens=metadata.sp_tokens,
            cp_tokens=cp_tokens,
            seq_lens=metadata.seq_lens,
            pad_token_count=pad_token_count
        )
        res = (model_inputs, sampling_metadata, trace_ids)
        # compose model inputs !!!!
        return res

    def clear_context_by_seq_ids(self, sequence_ids: npt.NDArray[np.int64]):
        """Clear context by sequence IDs, used when sending aborted requests or exceptions occur."""
        context_handles = self._batch_context.pop_context_handles(sequence_ids)
        # 跳过清理虚拟推理的context_handle 0
        context_handles = [h for h in context_handles if h != 0]
        if self.context_params.async_infer:
            # 异步推理时，将缓存句柄加入 aborted_context_handles 列表，等到后处理完成时调用 clear_aborted_context 来清理
            self.aborted_context_handles.extend(context_handles)
        else:
            context_handles = np.array(context_handles, dtype=np.int32)
            self._batch_context.clear_context_by_handles(context_handles)

    def clear_aborted_context(self):
        """originally check_aborted_sequences"""
        if len(self.aborted_context_handles) > 0:
            aborted_context_handles = np.array(self.aborted_context_handles, dtype=np.int32)
            self.aborted_context_handles.clear()
            self._batch_context.clear_context_by_handles(aborted_context_handles)
            self._batch_context.all_ndarray_context.pending_cleanup_flags[aborted_context_handles] = False

    def clear_finished_context(self, sequence_ids: npt.NDArray[np.int64], context_handles: npt.NDArray[np.int32]):
        """Clear context when inference is finished."""
        non_virtual_mask = context_handles != 0
        context_handles_to_clear = context_handles[non_virtual_mask]
        sequence_ids_to_clear = sequence_ids[non_virtual_mask]
        if self.context_params.async_infer:
            if self.context_params.layerwise_disaggregated:
                last_end_mask = self._batch_context.all_ndarray_context.pending_cleanup_flags[context_handles_to_clear]
                sequence_ids_to_clear = sequence_ids_to_clear[last_end_mask]
                context_handles_to_clear = context_handles_to_clear[last_end_mask]
            else:
                # get finished context from last round
                context_handles_to_clear = np.where(self._batch_context.all_ndarray_context.pending_cleanup_flags)[0]
                sequence_ids_to_clear, context_handles_to_clear = self._batch_context.filter_finished_context(
                    context_handles_to_clear
                )
            # reverse once_end_flags from current round
            if context_handles.size == 0:
                self._batch_context.all_ndarray_context.pending_cleanup_flags[:] = False
            else:
                cur_once_end_flags = self._batch_context.all_ndarray_context.pending_cleanup_flags[context_handles]
                if not self.context_params.layerwise_disaggregated:
                    self._batch_context.all_ndarray_context.pending_cleanup_flags[:] = False
                self._batch_context.all_ndarray_context.pending_cleanup_flags[context_handles] = ~cur_once_end_flags

        self._batch_context.pop_context_handles(sequence_ids_to_clear)
        self._batch_context.clear_context_by_handles(context_handles_to_clear)
        return sequence_ids_to_clear

    def update_context(
        self,
        context_handles: np.ndarray,
        filtered_indices: np.ndarray,
        metadata: Tuple[InputMetadata, SamplingMetadata],
        sampling_output: SamplingOutput,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """originally update_input, might be unncessary! It does: 1. update context with new input sampling meta
        data 2 get finished seq ids and context handles, should be splitted
        """
        input_metadata, sampling_metadata = metadata
        self._batch_context.update_context(
            context_handles,
            updated_ndarrays=None,
            input_metadata=input_metadata,
            sampling_args=(sampling_metadata, sampling_output)
        )

        finished_context_handles = np.array([], dtype=np.int32)
        finished_sequence_ids = np.array([], dtype=np.int64)
        if filtered_indices is not None:
            finished_context_handles = context_handles[filtered_indices]
            cache_mask = finished_context_handles != -1
            finished_context_handles = finished_context_handles[cache_mask]
            if sampling_metadata is None:
                finished_sequence_ids = input_metadata.all_sequence_ids[filtered_indices[cache_mask]]
            else:
                finished_sequence_ids = sampling_output.sequence_ids[filtered_indices[cache_mask]]
        return finished_context_handles, finished_sequence_ids

    # originally assign_new_cache, for beam search and best of n
    def fork_context(self, sampling_output: SamplingOutput) -> np.ndarray:
        """
        Fork context for beam search or best-of-N sampling.
        For each sequence:
        - If it already has a context (e.g., kept beam), reuse it.
        - If it's a new child, allocate a new context slot via get_context_slot(prefill=True).
        - Then, if it has a valid parent, fork the parent's context to it.
        
        Args:
            sampling_output (SamplingOutput): Contains sequence_ids and parent_sequence_ids.

        Returns:
            np.ndarray: context handles corresponding to each sequence in sampling_output.
        """
        sequence_ids = sampling_output.sequence_ids
        parent_sequence_ids = sampling_output.parent_sequence_ids

        result_handles = np.full(len(sequence_ids), -1, dtype=np.int32)
        child_handles = []
        parent_handles = []

        for i, (seq_id, parent_seq_id) in enumerate(zip(sequence_ids, parent_sequence_ids)):
            if seq_id == -1:
                continue  # padding

            try:
                # Try to get existing context; if not exists, allocate one (pretend it's prefill)
                # Note: We use is_prefill=True to allow allocation for new children
                handle = self._batch_context.get_context_slot(sequence_id=seq_id, is_prefill=True)
                result_handles[i] = handle
            except Exception as e:
                raise RuntimeError(f"Failed to get or create context for sequence {seq_id}") from e

            # Only fork if this is a true child (seq_id != parent_seq_id)
            if seq_id != parent_seq_id:
                try:
                    # Parent must already have context (must have been prefilled)
                    parent_handle = self._batch_context.get_context_slot(sequence_id=parent_seq_id, is_prefill=False)
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot fork context for sequence {seq_id}: parent {parent_seq_id} has no context."
                    ) from e

                child_handles.append(handle)
                parent_handles.append(parent_handle)

        # Batch fork
        if child_handles:
            self._batch_context.fork_context(
                children_context_handles=np.array(child_handles, dtype=np.int32),
                parents_context_handles=np.array(parent_handles, dtype=np.int32),
            )
            
        return result_handles

    def get_last_block_idx(self, context_handles):
        return self._batch_context.all_ndarray_context.used_block_idx[context_handles]

    def block_table_to_slots(self, block_table: npt.NDArray[np.int32]):
        return self._batch_context.block_table_to_slots(block_table)

    def block_to_slots(self, block_id: npt.NDArray[np.int32], slot_offset_in_block: npt.NDArray[np.int32]):
        return self._batch_context.block_to_slots(block_id, slot_offset_in_block)

    def append_and_return_output_text(self, context_handles, new_token):
        return self._batch_context.all_dict_context.append_and_return_output_text(context_handles, new_token)

    def is_empty_stopping_criteria(self):
        return self._batch_context.all_dict_context.is_empty_stopping_criteria()

    def get_stopping_criteria(self, context_handles):
        return self._batch_context.all_dict_context.get_stopping_criteria(context_handles)

    def is_empty_string_stopping_criteria(self):
        return self._batch_context.all_dict_context.is_empty_string_stopping_criteria()

    def get_string_stopping_criteria(self, context_handles):
        return self._batch_context.all_dict_context.get_string_stopping_criteria(context_handles)

    def get_once_end_flag(self, context_handles):
        return self._batch_context.all_ndarray_context.pending_cleanup_flags[context_handles]

    def set_once_end_flag(self, context_handles, once_end_flag):
        self._batch_context.all_ndarray_context.pending_cleanup_flags[context_handles] = once_end_flag

    def get_output_len_count(self, context_handles):
        return self._batch_context.all_ndarray_context.output_len_count[context_handles]

    def get_ignore_eos(self, context_handles):
        return self._batch_context.all_ndarray_context.ignore_eos[context_handles]

    def get_seq_lens(self, context_handles):
        return self._batch_context.all_ndarray_context.seq_lens[context_handles]

    def get_skip_special_tokens(self, context_handles):
        return self._batch_context.all_ndarray_context.skip_special_tokens[context_handles]

    def get_include_stop(self, context_handles):
        return self._batch_context.all_ndarray_context.include_stop[context_handles]

    def get_mtp_last_token_num(self, context_handles):
        return self._batch_context.all_ndarray_context.mtp_last_token_num[context_handles]

    def set_mtp_last_token_num(self, context_handles, mtp_last_token_num):
        self._batch_context.all_ndarray_context.mtp_last_token_num[context_handles] = mtp_last_token_num

    def get_mtp_hidden_states(self, context_handles):
        return self._batch_context.all_ndarray_context.mtp_hidden_states[context_handles]

    def set_mtp_hidden_states_prefix(self, context_handles, alias_len, hidden_states_prefix):
        self._batch_context.all_ndarray_context.mtp_hidden_states[context_handles, :alias_len] = hidden_states_prefix

    def get_all_input_ids(self, context_handles):
        return self._batch_context.all_ndarray_context.all_input_ids[context_handles]

    def get_last_position_ids(self, context_handles):
        return self._batch_context.all_ndarray_context.last_position_ids[context_handles]

    def get_all_output_ids(self, context_handles):
        return self._batch_context.all_ndarray_context.all_output_ids[context_handles]

    def get_mtp_last_rank(self, context_handles):
        return self._batch_context.all_ndarray_context.mtp_last_rank[context_handles]

    def set_mtp_last_rank(self, context_handles, sp_rank_id):
        self._batch_context.all_ndarray_context.mtp_last_rank[context_handles] = sp_rank_id

    def get_mtp_seq_block_rank_id(self, context_handles):
        return self._batch_context.all_ndarray_context.mtp_seq_block_rank_id[context_handles]

    def set_mtp_seq_block_rank_id(self, context_handles, prefill_block_rank_id_len, prefill_block_rank_id):
        self._batch_context.all_ndarray_context.mtp_seq_block_rank_id[context_handles, :prefill_block_rank_id_len] = (
            prefill_block_rank_id
        )

    def splitfuse_concatenate(self, metadata: InputMetadata, context_handles, warmup=False, hit_mask=None):
        is_prefill = metadata.is_prefill
        if is_prefill:
            model_inputs, sampling_metadata, q_len, trace_ids = \
                self.concatenate_mix(metadata, context_handles, warmup=warmup, hit_mask=hit_mask)
        else:
            model_inputs, sampling_metadata, trace_ids = \
                self.compose_model_inputs(metadata, context_handles, warmup=warmup, hit_mask=hit_mask)
            q_len = None
            sampling_metadata.is_seq_prefill = metadata.batch_is_prefill
        q_lens = q_len.tolist() if q_len is not None else None
        res = (model_inputs, sampling_metadata, q_lens, trace_ids)
        return res

    def concatenate_mix(self, metadata: InputMetadata, context_handles: np.ndarray, warmup=False, hit_mask=None):
        input_ids = np.array(metadata.input_ids, dtype=np.int64)
        position_ids = np.zeros(metadata.total_seq_num, dtype=np.int32)
        slots = np.zeros(metadata.total_seq_num, dtype=np.int32)
        bs = len(metadata.split_start_position)
        input_lengths = np.zeros(bs, dtype=np.int32)
        last_position_ids = np.zeros(bs, dtype=np.int32)
        prefill_head_indices = np.arange(bs, dtype=np.int64)
        q_lens = np.ones(bs, dtype=np.int32)  # mix input
        max_seq_len_decode = 0

        cumsum_seq_len = np.cumsum(metadata.batch_seq_len)

        decode_seq_idx = np.where(~metadata.batch_is_prefill)[0]
        if len(decode_seq_idx) > 0:
            decode_token_ids = cumsum_seq_len[decode_seq_idx] - 1
            (input_ids_decode, max_seq_len_decode, position_ids_decode, input_lengths_decode, slots_decode) = (
                self._batch_context.get_mix_decode_cache_for_splitfuse(
                    context_handles[decode_seq_idx], decode_seq_idx, metadata, hit_mask
                )
            )
            slots[decode_token_ids] = slots_decode
            input_ids[decode_token_ids] = input_ids_decode
            position_ids[decode_token_ids] = position_ids_decode
            last_position_ids[decode_seq_idx] = position_ids_decode
            input_lengths[decode_seq_idx] = input_lengths_decode
        max_seq_len = max(metadata.max_seq_len, max_seq_len_decode)

        prefill_seq_idx = np.where(metadata.batch_is_prefill)[0]
        if len(prefill_seq_idx) > 0:
            prefill_seq_lens = metadata.batch_seq_len[prefill_seq_idx]
            start_positions = (cumsum_seq_len - metadata.batch_seq_len)[prefill_seq_idx]
            end_positions = cumsum_seq_len[prefill_seq_idx]
            prefill_head_indices = cumsum_seq_len - 1
            last_position_ids[prefill_seq_idx] = metadata.split_end_position[prefill_seq_idx] - 1
            input_lengths[prefill_seq_idx] = metadata.split_end_position[prefill_seq_idx]
            q_lens[prefill_seq_idx] = prefill_seq_lens
            for _, (i, start_idx, end_idx) in enumerate(zip(prefill_seq_idx, start_positions, end_positions)):
                position_ids[start_idx:end_idx] = \
                    range(metadata.split_start_position[i], metadata.split_end_position[i])

                if not warmup:
                    slots[start_idx:end_idx] = \
                        self.block_table_to_slots(metadata.batch_block_tables[i]).reshape(-1)[
                            metadata.split_start_position[i]:metadata.split_end_position[i]
                        ]
                else:
                    slots[start_idx] = 0
                    slots[start_idx + 1:end_idx] = -1

        self._batch_context.update_context_for_splitfuse(metadata, context_handles, input_lengths, last_position_ids)
        trace_ids = None
        if context_handles is not None:
            trace_ids = self._batch_context.all_dict_context.get_trace_ids(context_handles)
        sampling_metadata = None
        if metadata.has_sampling:
            sampling_metadata = self._batch_context.build_sampling_meta_for_splitfuse(
                context_handles, metadata, prefill_seq_idx
            )
            sampling_metadata = self._batch_context.sync_sampling_token_ids(
                context_handles, sampling_metadata, max_seq_len
            )
            self.last_sampling_metadata.add_to_cache(metadata.all_sequence_ids, sampling_metadata)

        model_inputs = ModelInput(input_ids=input_ids,
                                  position_ids=position_ids,
                                  block_tables=metadata.batch_block_tables,
                                  slots=slots,
                                  context_length=input_lengths,
                                  cached_context_length=input_lengths,
                                  max_seq_len=max_seq_len,
                                  prefill_head_indices=prefill_head_indices,
                                  is_prefill=True,
                                  dp_rank_ids=metadata.batch_dp_rank_ids)
        res = (model_inputs, sampling_metadata, q_lens, trace_ids)
        return res

    def reset_all_context(self):
        self._batch_context.reset_all_context()
        self.aborted_context_handles.clear()
        logger.info("Cache reset triggered by recover command.")

    def _update_context_before_prefill(
        self,
        context_handles,
        metadata: InputMetadata,
        position_ids_gen_func: Callable,
        **kwargs,
    ):
        pad_token_count = kwargs.get("pad_token_count")
        position_ids = np.zeros(metadata.total_seq_num, dtype=np.int32)
        total_seq_num = metadata.total_seq_num
        metadata.input_ids = np.where(metadata.input_ids == -1, self.cache_config.pad_token_id, metadata.input_ids)
        all_tokens_kv_slots = -np.ones(total_seq_num, dtype=np.int32)  # all kv slots for this batch
        seq_lengths = np.zeros(metadata.batch_size, dtype=np.int32)
        last_position_ids = np.zeros(metadata.batch_size, dtype=np.int32)
        prefill_head_indices = np.zeros(metadata.batch_size, dtype=np.int64)  # head index is prompt last token index
        cumulative_seq_len = 0
        input_ids = metadata.input_ids  # flattened input_ids for the batch
        # prompt last token id
        prefill_new_tokens = np.zeros(metadata.batch_size, dtype=np.int32) \
            if kwargs.get("is_pd_separate", False) else None
        for i in range(metadata.batch_size):
            seq_len = metadata.batch_seq_len[i]
            start_idx = cumulative_seq_len
            end_idx = cumulative_seq_len + seq_len
            position_ids[start_idx:end_idx] = position_ids_gen_func(input_ids[start_idx:end_idx])
            last_position_ids[i] = position_ids[end_idx - 1]
            self._prepare_seq_kv_slots(
                metadata.batch_block_tables[i],
                (start_idx, end_idx, all_tokens_kv_slots),  # all_tokens_kv_slots is generated after method call
                (
                    metadata.sp_tokens[i] if metadata.sp_tokens is not None else None
                ),  # actually [sp_rank0_chunk_len, sp_rank1_chunk_len]
                **kwargs,
            )
            seq_lengths[i] = seq_len
            cumulative_seq_len += seq_len
            prefill_head_indices[i] = cumulative_seq_len - pad_token_count[i] - 1
            if prefill_new_tokens is not None:
                prefill_new_tokens[i] = input_ids[cumulative_seq_len - 1]

        self._batch_context.update_context(
            context_handles,
            (last_position_ids, seq_lengths, prefill_new_tokens),
            metadata,
            sampling_args=(None, None),
            is_first_update=True,
            pad_token_count=pad_token_count,
            is_pd_separate=kwargs.get("is_pd_separate", False)
        )
        # 虚推请求不写入 KV cache，设置 slots = -1 跳过 ReshapeAndCache
        cumulative_idx = 0
        for i in range(metadata.batch_size):
            seq_len = metadata.batch_seq_len[i]
            if metadata.all_sequence_ids[i] == SIMULATE_SEQUENCE_ID:
                all_tokens_kv_slots[cumulative_idx:cumulative_idx + seq_len] = -1
            cumulative_idx += seq_len
        return input_ids, position_ids, all_tokens_kv_slots, seq_lengths, prefill_head_indices

    def _gen_scp_slots(self, start_idx, seq_sp_tokens, slots: Optional[np.ndarray], block_tables) -> np.ndarray:
        sp_start_idx = start_idx + self.spcp_parallel_info.scp_rank * self.block_size
        sp_len = seq_sp_tokens[self.spcp_parallel_info.scp_rank]
        for block_table in block_tables:
            if block_table == -1 or sp_len <= 0:
                break
            sp_len_i = min(sp_len, self.block_size)
            slots[sp_start_idx: sp_start_idx + sp_len_i] = \
                self._batch_context.block_table_to_slots(block_table).reshape(-1)[:sp_len_i]
            sp_len -= self.block_size
            sp_start_idx += self.spcp_parallel_info.scp_size * self.block_size
        return slots

    def _gen_scp_slots_last_rank(
            self, start_idx, seq_sp_tokens, slots: Optional[np.ndarray], block_tables) -> np.ndarray:
        sp_start_idx = start_idx + self.spcp_parallel_info.scp_rank * self.block_size
        sp_len = seq_sp_tokens[self.spcp_parallel_info.scp_rank]
        seq_len = np.sum(seq_sp_tokens)
        for block_table in block_tables:
            if block_table == -1 or sp_len <= 0:
                break
            sp_len_i = min(sp_len, self.block_size)
            if sp_len <= self.block_size:
                slots[start_idx + seq_len - sp_len_i: start_idx + seq_len] = \
                    self._batch_context.block_table_to_slots(block_table).reshape(-1)[:sp_len_i]
            else:
                slots[sp_start_idx: sp_start_idx + sp_len_i] = \
                    self._batch_context.block_table_to_slots(block_table).reshape(-1)[:sp_len_i]
            sp_len -= self.block_size
            sp_start_idx += self.spcp_parallel_info.scp_size * self.block_size
        return slots

    def _prepare_seq_kv_slots(
        self,
        block_table: np.ndarray,
        start_end_token_slots: Tuple[int, int, np.ndarray],
        seq_sp_chunk_lens: np.ndarray = None,  # [sp_rank0_chunk_len, sp_rank1_chunk_len]
        **kwargs,
    ):
        start_idx, end_idx, all_tokens_kv_slots = start_end_token_slots
        if kwargs.get("warmup", None):
            if start_idx == 0:
                all_tokens_kv_slots[0] = 0
                all_tokens_kv_slots[1:end_idx] = -1
            else:
                all_tokens_kv_slots[start_idx:end_idx] = -1
            return
        if self.spcp_parallel_info.scp_size > 1:
            if self.spcp_parallel_info.scp_rank == self.spcp_parallel_info.scp_size - 1:
                all_tokens_kv_slots = \
                    self._gen_scp_slots_last_rank(start_idx, seq_sp_chunk_lens, all_tokens_kv_slots, block_table)
            else:
                all_tokens_kv_slots = \
                    self._gen_scp_slots(start_idx, seq_sp_chunk_lens, all_tokens_kv_slots, block_table)

        else:
            all_tokens_kv_slots[start_idx:end_idx] = self._batch_context.block_table_to_slots(block_table).reshape(-1)[
                : end_idx - start_idx
            ]
