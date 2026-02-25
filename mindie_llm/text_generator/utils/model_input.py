# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from threading import Event
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np

from mindie_llm.runtime.model_runner.forward_context_exp import ForwardContext
from .input_metadata import InputMetadata
from .sampling_metadata import SamplingMetadata
from ...utils.tensor import BackendTensor


@dataclass
class ModelInput:
    input_ids: Union[np.ndarray, BackendTensor]
    position_ids: Optional[Union[np.ndarray, BackendTensor]]
    block_tables: Union[np.ndarray, BackendTensor]
    slots: Union[np.ndarray, BackendTensor]
    context_length: Union[np.ndarray, List[int]]
    max_seq_len: int
    prefill_head_indices: Optional[np.ndarray]
    is_prefill: bool
    query_length: Optional[np.ndarray] = None
    adapter_ids: Optional[List[str]] = None
    dp_rank_ids: Optional[np.ndarray] = None

    # attributes for cp and kvp
    sp_tokens: Optional[np.ndarray] = None
    cp_tokens: Optional[np.ndarray] = None
    pad_token_count: Optional[np.ndarray] = None
    cached_context_length: Optional[Union[np.ndarray, List[int]]] = None
    is_need_mask: Optional[List[int]] = None

    # attributes for prefixcache
    sp_computed_slots_padding_idx: Optional[np.ndarray] = None
    sp_computed_slots_order: Optional[List[List[int]]] = None
    all_rank_prefix_lens: Optional[np.ndarray] = None
    per_rank_prefix_lens: Optional[np.ndarray] = None

    # attributes for async inference
    acl_inputs: Optional[List[BackendTensor]] = None
    acl_param: Optional[str] = None
    block_tables_array: Optional[np.ndarray] = None
    input_lengths: Optional[BackendTensor] = None  # The alias of context_length in model_runner
    lm_head_indices: Optional[BackendTensor] = None  # The alias of prefill_head_indices in model_runner
    seq_lens: Optional[List[List[int]]] = None
    kwargs: Optional[Dict[str, Any]] = None
    layerwise_disaggregated_exe_stage = None

    forward_context: ForwardContext = None

    def __post_init__(self):
        if self.cached_context_length is None:
            self.cached_context_length = self.context_length


@dataclass
class ModelInputWrapper:
    cache_ids: np.ndarray
    input_metadata: InputMetadata
    model_inputs: ModelInput
    model_kwargs: Dict[str, Any]
    sampling_metadata: SamplingMetadata
    trace_ids: List[Any]
    current_dp_sequence_ids: np.ndarray
    postprocess_done: Event
    filling_masks: np.ndarray = None