# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import IntEnum
from typing import Any, List, Optional

import numpy as np

from .logits_handlers import get_handler_registry
from .logits_handlers.logits_handler import LogitsHandlerList
from .sampler_params import HandlerParams, SelectorParams
from .token_selectors import get_selector_registry
from ..utils.config import SamplerConfig, HandlingBackend
from ..utils.sampling_output import SamplingOutput
from ..utils.sampling_metadata import SamplingMetadata
from ...utils.log.logging import logger


class SelectorType(IntEnum):
    GREEDY_SEARCH = 0
    RANDOM_SAMPLING = 1
    BEAM_SEARCH = 2


class Sampler:
    def __init__(self, sampler_config: SamplerConfig):
        self.backend_type = sampler_config.backend_type
        self.npu_id = sampler_config.npu_id
        self.num_threads = sampler_config.num_threads
        self.rank = sampler_config.rank
        self.splitfuse_enabled = sampler_config.splitfuse_enabled

        self.handling_policy = sampler_config.handling_policy
        self.selection_policy = sampler_config.selection_policy

        self.batch_input_ids = None
        self.batch_input_ids_tensor = None
        self.batch_output_ids = None
        self.batch_output_ids_tensor = None
        self.batch_size = 0

        self.handler_mapping = {}
        self.handler_params = HandlerParams(backend_type=self.backend_type, rank=self.rank)
        self.handlers = LogitsHandlerList()
        self.metadata_cache = None
        self.need_configuring = False
        self.selector_params = SelectorParams(npu_id=self.npu_id,
                                              num_threads=self.num_threads,
                                              splitfuse_enabled=self.splitfuse_enabled)

        # There will be greedy_search_selector, sampling_selector and beam_search_selector
        # in selectors after initialisation.
        self.selectors = []
        self.selector = None
        self.fusion_sampling = True
        
        # 结构化输出 handler 配置 (response_format / guided decoding)
        # 注意: bitmask 生成和 FSM 状态管理由 PluginManager 负责
        self.enable_guided_decoding = sampler_config.enable_guided_decoding

    def __call__(
        self,
        batch_logits: Any,
        sampling_metadata: SamplingMetadata
    ) -> SamplingOutput:
        """
        The shape of two key outputs in SamplingOutput:
            token_ids: [num_seqs, num_new_tokens]
            top_token_ids: [num_seqs, num_new_tokens, num_top_tokens]
        """
        from mindie_llm.utils.prof.profiler import span_start, span_end, tensor_attr, span_attr, Level
        prof = span_start(name="Sampler", level=Level.DETAILED)
        prof = span_attr(prof, "batch_logits", lambda: tensor_attr(batch_logits))
        prof = span_attr(prof, "sampling_metadata", lambda: str(sampling_metadata))
        if id(sampling_metadata) != id(self.metadata_cache):
            self.metadata_cache = sampling_metadata
            self.handler_params.batch_size, self.handler_params.vocab_size = batch_logits.shape
            self.switch_handlers(sampling_metadata)
            self.switch_selector(sampling_metadata)

        if self.backend_type == HandlingBackend.MS:
            from mindspore.common.api import _pynative_executor as ms_pyexecutor
            ms_pyexecutor.set_async_for_graph(True)
            batch_logits = self.handlers(batch_logits, sampling_metadata)
            if sampling_metadata is None or sampling_metadata.do_sample_array is None:
                output = self.selectors[0](batch_logits, sampling_metadata)
            else:
                output = self.selectors[1](batch_logits, sampling_metadata)
            ms_pyexecutor.sync()
            ms_pyexecutor.set_async_for_graph(False)
        else:
            batch_logits = self.handlers(batch_logits, sampling_metadata)
            output = self.selector(batch_logits, sampling_metadata)

        self.handler_params.clear_token_counts()
        output.token_ids = output.token_ids.astype(np.int64)
        prof = span_attr(prof, "sampling_output", lambda: str(output))
        span_end(prof)
        return output

    @staticmethod
    def merge_sampling_output(
        sampling_metadata: SamplingMetadata,
        first_output: SamplingOutput,
        second_output: SamplingOutput,
        first_indices: List[int]
    ):
        if len(first_output.group_indices) != len(first_indices):
            raise RuntimeError('The length of sampling indices does not match the length of sampling output.')

        group_indices = []
        sequence_ids = []
        parent_sequence_ids = []
        token_ids = []
        logprobs = []
        top_token_ids = []
        top_logprobs = []
        cumulative_logprobs = []
        num_new_tokens = []
        num_top_tokens = []
        seeds = []
        idx = 0

        if first_output.seeds is not None or second_output.seeds is not None:
            if first_output.seeds is None:
                first_output.seeds = np.zeros(len(first_output.token_ids), dtype=np.int64)
            if second_output.seeds is None:
                second_output.seeds = np.zeros(len(second_output.token_ids), dtype=np.int64)

        def append_outputs(current_sampling_output: SamplingOutput, offset: int, start_idx: int):
            start, end = current_sampling_output.group_indices[offset]
            end_idx = start_idx + end - start
            group_interval = (start_idx, end_idx)
            group_indices.append(group_interval)
            sequence_ids.append(current_sampling_output.sequence_ids[start:end])
            parent_sequence_ids.append(current_sampling_output.parent_sequence_ids[start:end])
            token_ids.append(current_sampling_output.token_ids[start:end])
            logprobs.append(current_sampling_output.logprobs[start:end])
            top_token_ids.append(current_sampling_output.top_token_ids[start:end])
            top_logprobs.append(current_sampling_output.top_logprobs[start:end])
            cumulative_logprobs.append(current_sampling_output.cumulative_logprobs[start:end])
            num_new_tokens.append(current_sampling_output.num_new_tokens[start:end])
            if current_sampling_output.num_top_tokens is not None:
                num_top_tokens.append(current_sampling_output.num_top_tokens[start:end])
            if current_sampling_output.seeds is not None:
                seeds.append(current_sampling_output.seeds[start:end])
            return end_idx

        first_offset = 0
        second_offset = 0
        for i in range(len(first_output.group_indices) + len(second_output.group_indices)):
            if first_offset < len(first_indices) and first_indices[first_offset] == i:
                idx = append_outputs(first_output, first_offset, idx)
                first_offset += 1
            else:
                idx = append_outputs(second_output, second_offset, idx)
                second_offset += 1

        parent_sequence_ids = np.concatenate(parent_sequence_ids)
        repeating_indices = np.where(parent_sequence_ids[:, None] == sampling_metadata.all_sequence_ids)[1]

        max_num_top = max(first_output.top_token_ids.shape[1], second_output.top_token_ids.shape[1])
        for i, top_token_ids_ins in enumerate(top_token_ids):
            padding = np.full((top_token_ids_ins.shape[0], max_num_top - top_token_ids_ins.shape[1]), -1)
            top_token_ids[i] = np.concatenate([top_token_ids_ins, padding], axis=-1)
            padding = np.full((top_logprobs[i].shape[0], max_num_top - top_logprobs[i].shape[1]), -9999.0)
            top_logprobs[i] = np.concatenate([top_logprobs[i], padding], axis=-1)

        sampling_output = SamplingOutput(
            group_indices=group_indices,
            sequence_ids=np.concatenate(sequence_ids),
            parent_sequence_ids=parent_sequence_ids,
            repeating_indices=repeating_indices,
            token_ids=np.concatenate(token_ids),
            logprobs=np.concatenate(logprobs),
            top_token_ids=np.concatenate(top_token_ids),
            top_logprobs=np.concatenate(top_logprobs),
            cumulative_logprobs=np.concatenate(cumulative_logprobs),
            num_new_tokens=np.concatenate(num_new_tokens),
            num_top_tokens=np.concatenate(num_top_tokens) if num_top_tokens else None,
            seeds=np.concatenate(seeds) if seeds else None
        )
        return sampling_output

    @staticmethod
    def split_sampling_metadata(sampling_metadata: SamplingMetadata, split_mask: np.ndarray):
        retained_batch_sequence_ids = []
        discarded_batch_sequence_ids = []
        retained_reserved_sequence_ids = []
        discarded_reserved_sequence_ids = []
        retained_group_indices = []
        discarded_group_indices = []
        retained_idx = 0
        discarded_idx = 0
        retained_indices = []
        for i, (start, end) in enumerate(sampling_metadata.group_indices):
            if split_mask[start]:
                retained_batch_sequence_ids.append(sampling_metadata.batch_sequence_ids[i])
                retained_reserved_sequence_ids.append(sampling_metadata.reserved_sequence_ids[i])
                num_sequences = len(sampling_metadata.batch_sequence_ids[i])
                retained_group_indices.append((retained_idx, retained_idx + num_sequences))
                retained_idx += num_sequences
                retained_indices.append(i)
            else:
                discarded_batch_sequence_ids.append(sampling_metadata.batch_sequence_ids[i])
                discarded_reserved_sequence_ids.append(sampling_metadata.reserved_sequence_ids[i])
                num_sequences = len(sampling_metadata.batch_sequence_ids[i])
                discarded_group_indices.append((discarded_idx, discarded_idx + num_sequences))
                discarded_idx += num_sequences

        retained_sampling_metadata = SamplingMetadata(
            batch_sequence_ids=retained_batch_sequence_ids,
            reserved_sequence_ids=retained_reserved_sequence_ids,
            is_prefill=sampling_metadata.is_prefill,
            is_mix=sampling_metadata.is_mix,
            all_sequence_ids=sampling_metadata.all_sequence_ids[split_mask],
            parent_sequence_ids=sampling_metadata.parent_sequence_ids[split_mask],
            group_indices=retained_group_indices,
            to_tensor=sampling_metadata.to_tensor
        )
        discarded_sampling_metadata = SamplingMetadata(
            batch_sequence_ids=discarded_batch_sequence_ids,
            reserved_sequence_ids=discarded_reserved_sequence_ids,
            is_prefill=sampling_metadata.is_prefill,
            is_mix=sampling_metadata.is_mix,
            all_sequence_ids=sampling_metadata.all_sequence_ids[~split_mask],
            parent_sequence_ids=sampling_metadata.parent_sequence_ids[~split_mask],
            group_indices=discarded_group_indices,
            to_tensor=sampling_metadata.to_tensor
        )

        attribute_keys = ['repetition_penalty', 'frequency_penalty', 'presence_penalty', 'temperature', 'top_k_array',
                          'top_k_idx', 'top_k_disabled_mask', 'top_p_array', 'top_p', 'do_sample_array',
                          'top_logprobs_array', 'seed_array', 'num_top_tokens', 'beam_width_array', 'best_of_array',
                          'use_beam_search_array', 'output_lengths', 'cumulative_logprobs',
                          'all_token_ids', 'output_token_ids', 'is_seq_prefill']
        for attribute_key in attribute_keys:
            attribute = getattr(sampling_metadata, attribute_key)
            if attribute is not None:
                setattr(retained_sampling_metadata, attribute_key, attribute[split_mask])
                setattr(discarded_sampling_metadata, attribute_key, attribute[~split_mask])

        attribute_keys_with_max = [('max_top_k', 'top_k_array'), ('max_logprobs', 'top_logprobs_array'),
                                   ('max_beam_width', 'beam_width_array')]
        for attribute_key, associated_key in attribute_keys_with_max:
            associated_attribute = getattr(retained_sampling_metadata, associated_key)
            if associated_attribute is not None:
                setattr(retained_sampling_metadata, attribute_key, max(associated_attribute))
            associated_attribute = getattr(discarded_sampling_metadata, associated_key)
            if associated_attribute is not None:
                setattr(discarded_sampling_metadata, attribute_key, max(associated_attribute))

        return retained_sampling_metadata, discarded_sampling_metadata, retained_indices

    def configure(self, sampling_metadata: SamplingMetadata):
        if self.need_configuring:
            self.selectors[SelectorType.RANDOM_SAMPLING].configure(sampling_metadata)

    def clear_cache(self, sequence_ids: np.ndarray):
        if self.need_configuring:
            self.selectors[SelectorType.RANDOM_SAMPLING].clear(sequence_ids)

    def initialize(self, device, eos_token_id):
        self.selector_params.device = device
        self.selector_params.eos_token_id = eos_token_id
        self.__initialize_handlers_and_selectors()
    
    def switch_handlers(self, sampling_metadata):
        if sampling_metadata is None:
            self.handlers.clear()
        else:
            self.handlers.clear()
            self.__check_and_append(sampling_metadata.repetition_penalty, 'repetition_penalty')
            self.__check_and_append(sampling_metadata.frequency_penalty, 'frequency_penalty')
            self.__check_and_append(sampling_metadata.presence_penalty, 'presence_penalty')
            if sampling_metadata.do_sample_array is not None:
                self.__check_and_append(sampling_metadata.temperature, 'temperature')
                if not self.fusion_sampling:
                    self.__check_and_append(sampling_metadata.top_k_idx, 'top_k')
                    self.__check_and_append(sampling_metadata.top_p, 'top_p')
            # 结构化输出 handler (bitmask 由 preprocess 阶段生成)
            self.__check_and_append(sampling_metadata.guided_bitmask, 'guided_decoding')

    def switch_selector(self, sampling_metadata):
        if sampling_metadata is not None:
            if sampling_metadata.use_beam_search_array is not None:
                if sampling_metadata.use_beam_search_array.all():
                    self.selector = self.selectors[SelectorType.BEAM_SEARCH]
                else:
                    self.selector = self.__split_and_select
            elif sampling_metadata.do_sample_array is not None:
                self.selector = self.selectors[SelectorType.RANDOM_SAMPLING]
            else:
                self.selector = self.selectors[SelectorType.GREEDY_SEARCH]
        else:
            self.selector = self.selectors[SelectorType.GREEDY_SEARCH]

    def __split_and_select(
        self,
        batch_logits: Any,
        sampling_metadata: SamplingMetadata
    ) -> SamplingOutput:
        # slice batch-logits into two parts: `greedy+sampling` and `beam`
        beam_metadata, greedy_sampling_metadata, beam_search_indices = (
            self.split_sampling_metadata(sampling_metadata, sampling_metadata.use_beam_search_array))

        beam_search_output = self.selectors[SelectorType.BEAM_SEARCH](
            batch_logits[sampling_metadata.use_beam_search_array], beam_metadata)

        if sampling_metadata.do_sample_array is None:
            greedy_sampling_output = self.selectors[SelectorType.GREEDY_SEARCH](
                batch_logits[~sampling_metadata.use_beam_search_array], greedy_sampling_metadata)
        else:
            greedy_sampling_output = self.selectors[SelectorType.RANDOM_SAMPLING](
                batch_logits[~sampling_metadata.use_beam_search_array], greedy_sampling_metadata)

        output = self.merge_sampling_output(sampling_metadata, beam_search_output,
                                            greedy_sampling_output, beam_search_indices)
        return output

    def __check_and_append(self, parameter, parameter_name):
        if parameter is not None:
            self.handlers.append(self.handler_mapping.get(parameter_name))

    def __find_handler(self, parameter_name):
        handling_backend = self.handling_policy.get(parameter_name)
        if not handling_backend:
            raise ValueError(f'The current handling policy config does not support this parameter: {parameter_name}!')
        backend_registry = get_handler_registry(handling_backend)
        if not backend_registry:
            raise ValueError(f'No such handler backend: {handling_backend}!')
        parameter_handler_cls = backend_registry.get(parameter_name)
        if not parameter_handler_cls:
            raise ValueError(f'The backend {handling_backend} does not have such handler: {parameter_name}!')
        return parameter_handler_cls

    def __find_selector(self, parameter_name):
        selection_backend = self.selection_policy.get(parameter_name)
        if not selection_backend:
            raise ValueError(f'The current selection policy config does not support this parameter: {parameter_name}!')
        backend_registry = get_selector_registry(selection_backend)
        if not backend_registry:
            raise ValueError(f'No such selector backend: {selection_backend}!')
        parameter_selector_cls = backend_registry.get(parameter_name)
        if not parameter_selector_cls:
            raise ValueError(f'The backend {selection_backend} does not have such selector: {parameter_name}!')
        return parameter_selector_cls

    def __initialize_handlers_and_selectors(self):
        handling_params = ['repetition_penalty', 'frequency_penalty', 'presence_penalty', 'temperature']
        selection_params = ['greedy_search']
        fusion_sampling_key = 'top_k_top_p_sampling'
        sampling_backend = self.selection_policy.get(fusion_sampling_key)
        self.fusion_sampling = True
        if sampling_backend == HandlingBackend.PTA:
            handling_params.append('top_k')
            handling_params.append('top_p')
            selection_params.append('sampling')
            self.fusion_sampling = False
        elif sampling_backend == HandlingBackend.ATB:
            selection_params.append(fusion_sampling_key)
        else:
            selection_params.append(fusion_sampling_key)
            self.need_configuring = True
        selection_params.append('beam_search')
        
        if self.enable_guided_decoding:
            handling_params.append('guided_decoding')
        
        for param_name in handling_params:
            parameter_handler_cls = self.__find_handler(param_name)
            self.handler_mapping[param_name] = parameter_handler_cls(self.handler_params)
        for param_name in selection_params:
            parameter_selector_cls = self.__find_selector(param_name)
            self.selectors.append(parameter_selector_cls(self.selector_params))
        self.selector = self.selectors[0]