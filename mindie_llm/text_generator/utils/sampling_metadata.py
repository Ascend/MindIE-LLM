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
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .input_metadata import InputMetadata
from ...utils.decorators.deprecation import deprecated
from ...utils.log.logging import logger
from ...utils.tensor import backend
from ...utils.validation import (UPPER_SAFE_BATCH_SIZE, LOWER_SAFE_BATCH_SIZE, UPPER_SAFE_SEQUENCE_LENGTH,
                                 LOWER_SAFE_REPETITION_PENALTY, LOWER_SAFE_TEMPERATURE, UPPER_SAFE_LOGPROBS,
                                 LOWER_SAFE_LOGPROBS, UPPER_SAFE_SEED, LOWER_SAFE_SEED, SAFE_TOP_K, UPPER_SAFE_TOP_P,
                                 LOWER_SAFE_TOP_P, UPPER_SAFE_BEAM_WIDTH, LOWER_SAFE_BEAM_WIDTH,
                                 UPPER_SAFE_BEST_OF, LOWER_SAFE_BEST_OF,
                                 InconsistencyError, OutOfBoundsError, UnsupportedTypeError)


def validate_1d_batch(param_key: str, param_value: np.ndarray) -> None:
    if not isinstance(param_value, np.ndarray):
        raise UnsupportedTypeError(param_key, 'numpy.ndarray')
    if len(param_value.shape) != 1:
        raise InconsistencyError(param_key, f'dimension of `{param_key}', '1')
    batch_size = len(param_value)
    if batch_size > UPPER_SAFE_BATCH_SIZE or batch_size < LOWER_SAFE_BATCH_SIZE:
        raise OutOfBoundsError(param_key, 'SAFE_BATCH_SIZE', f'[{LOWER_SAFE_BATCH_SIZE}, {UPPER_SAFE_BATCH_SIZE}]')


def validate_2d_batch(param_key: str, param_value: np.ndarray) -> None:
    if not isinstance(param_value, np.ndarray):
        raise UnsupportedTypeError(param_key, 'numpy.ndarray')
    if len(param_value.shape) != 2:
        raise InconsistencyError(param_key, f'dimension of `{param_key}', '2')
    batch_size = len(param_value)
    if batch_size > UPPER_SAFE_BATCH_SIZE or batch_size < LOWER_SAFE_BATCH_SIZE:
        raise OutOfBoundsError(param_key, 'SAFE_BATCH_SIZE', f'[{LOWER_SAFE_BATCH_SIZE}, {UPPER_SAFE_BATCH_SIZE}]')
    if len(param_value[0]) > UPPER_SAFE_SEQUENCE_LENGTH:
        raise OutOfBoundsError(param_key, 'UPPER_SAFE_SEQUENCE_LENGTH', UPPER_SAFE_SEQUENCE_LENGTH)


class PenaltyMetadata:
    def __init__(self,
                 has_penalty: bool,
                 repetition_penalty_tensor,
                 frequency_penalty_tensor,
                 presence_penalty_tensor
                 ):
        self.has_penalty = has_penalty
        self.repetition_penalty = repetition_penalty_tensor
        self.frequency_penalty = frequency_penalty_tensor
        self.presence_penalty = presence_penalty_tensor

    def __repr__(self):
        return (f"PenaltyMetadata:\n"
                f"has_penalty: {self.has_penalty},"
                f"repetition_penalty: {self.repetition_penalty},"
                f"frequency_penalty: {self.frequency_penalty},"
                f"presence_penalty: {self.presence_penalty},"
                )


class TopKMetadata:
    def __init__(self,
                 top_k_tensor,
                 top_k_array: np.ndarray,
                 top_k_max: int,
                 top_k_disabled_mask_tensor):
        self.top_k_tensor = top_k_tensor
        self.top_k_array = top_k_array
        self.max_top_k = top_k_max
        self.top_k_disabled_mask_tensor = top_k_disabled_mask_tensor

    def __repr__(self):
        return (f"TopKMetadata:\n"
                f"top_k_tensor: {self.top_k_tensor},"
                f"max_top_k: {self.max_top_k},"
                f"top_k_disabled_mask_tensor: {self.top_k_disabled_mask_tensor},"
                )


@dataclass
class TopPMetadata:
    def __init__(self,
                 top_p_tensor,
                 top_p_array: np.ndarray):
        self.top_p_tensor = top_p_tensor
        self.top_p_array = top_p_array

    def __repr__(self):
        return (f"TopPMetadata:\n"
                f"top_p_tensor: {self.top_p_tensor},"
                f"top_p_array: {self.top_p_array},"
                )


@dataclass
class SeedMetadata:
    def __init__(self,
                 seed_array: np.ndarray):
        self.seed_array = seed_array

    def __repr__(self):
        return (f"SeedMetadata:\n"
                f"seed_array: {self.seed_array}"
                )


@dataclass
class DoSampleMetadata:
    def __init__(self,
                 do_sample_tensor,
                 do_sample_array: np.ndarray):
        self.do_sample_tensor = do_sample_tensor
        self.do_sample_array = do_sample_array

    def __repr__(self):
        return (f"DoSampleMetadata:\n"
                f"do_sample_tensor: {self.do_sample_tensor},"
                f"do_sample_array: {self.do_sample_array},"
                )


@dataclass
class SamplingData:
    """
    A deprecated data class containing some metadata for sampling. It must be instantiated by `from_numpy` method,
    otherwise the safety validation will be skipped.
    """

    @deprecated('SamplingData is deprecated, please use SamplingMetadata instead.')
    def __init__(
        self,
        all_input_ids: Any,
        output_ids: Any,
        is_prefill: bool = True,
        request_ids: Optional[np.ndarray] = None,
        to_tensor: Optional[callable] = None
    ):
        self.all_input_ids = all_input_ids
        self.output_ids = output_ids
        self.is_prefill = is_prefill
        self.request_ids = request_ids
        self.to_tensor = to_tensor

    def __repr__(self):
        return (f"SamplingMetadata:\n"
                f"all_input_ids: {self.all_input_ids},"
                f"output_ids: {self.output_ids},"
                f"is_prefill: {self.is_prefill},"
                f"request_ids: {self.request_ids},"
                f"to_tensor: {self.to_tensor}"
                )

    @classmethod
    def from_numpy(
        cls,
        all_input_ids: Optional[np.ndarray] = None,
        output_ids: Optional[np.ndarray] = None,
        to_tensor: Optional[callable] = None,
        is_prefill: bool = True,
        request_ids: Optional[np.ndarray] = None
    ):
        if request_ids is not None:
            validate_1d_batch('request_ids', request_ids)
            if all_input_ids is not None and len(all_input_ids) != len(request_ids):
                raise InconsistencyError('all_input_ids', 'length of `all_input_ids`', 'length of `request_ids`')
            if output_ids is not None and len(output_ids) != len(request_ids):
                raise InconsistencyError('output_ids', 'length of `output_ids`', 'length of `request_ids`')
        else:
            validate_2d_batch('all_input_ids', all_input_ids) if all_input_ids is not None else None
            validate_2d_batch('output_ids', output_ids) if output_ids is not None else None
        if not isinstance(is_prefill, bool):
            raise UnsupportedTypeError('is_prefill', 'bool')

        # The user must guarantee the `to_tensor` function returning correct Tensor type for the backend used.
        all_input_ids_tensor = to_tensor(all_input_ids) if all_input_ids is not None else None
        output_ids_tensor = to_tensor(output_ids) if output_ids is not None else None
        return SamplingData(
            all_input_ids=all_input_ids_tensor,
            output_ids=output_ids_tensor,
            is_prefill=is_prefill,
            request_ids=request_ids,
            to_tensor=to_tensor
        )


@dataclass
class SamplingParam:
    """
    A deprecated data class containing some parameters for sampling. It must be instantiated by `from_numpy` method,
    otherwise the safety validation will be skipped.
    """

    @deprecated('SamplingParam is deprecated, please use SamplingMetadata instead.')
    def __init__(
        self,
        penalty_meta: PenaltyMetadata,
        temperature,
        top_k_meta: TopKMetadata,
        top_p_meta: TopPMetadata,
        seed_meta: SeedMetadata,
        do_sample_meta: DoSampleMetadata,
    ):
        self.penalty_meta = penalty_meta
        self.temperature = temperature
        self.top_k_meta = top_k_meta
        self.top_p_meta = top_p_meta
        self.seed_meta = seed_meta
        self.do_sample_meta = do_sample_meta

    def __repr__(self):
        return (f"SamplingMetadata:\n"
                f"penalty_meta: {self.penalty_meta},"
                f"temperature: {self.temperature},"
                f"top_k_meta: {self.top_k_meta},"
                f"top_p: {self.top_p_meta},"
                f"seed: {self.seed_meta},"
                f"do_sample: {self.do_sample_meta},")

    @classmethod
    def from_numpy(
        cls,
        repetition_penalty: Optional[np.ndarray] = None,
        frequency_penalty: Optional[np.ndarray] = None,
        presence_penalty: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        top_k: Optional[np.ndarray] = None,
        top_p: Optional[np.ndarray] = None,
        seed: Optional[np.ndarray] = None,
        do_sample: Optional[np.ndarray] = None,
        to_tensor: Optional[callable] = None
    ) -> "SamplingParam":
        has_penalty = False
        rp_tensor = None
        if repetition_penalty is not None:
            validate_1d_batch('repetition_penalty', repetition_penalty)
            if np.asarray(repetition_penalty != 1.0).any():
                if np.asarray(repetition_penalty <= LOWER_SAFE_REPETITION_PENALTY).any():
                    raise OutOfBoundsError('repetition_penalty', 'LOWER_SAFE_REPETITION_PENALTY',
                                           LOWER_SAFE_REPETITION_PENALTY)
                rp_tensor = to_tensor(repetition_penalty)
                has_penalty = True
        fp_tensor = None
        if frequency_penalty is not None:
            validate_1d_batch('frequency_penalty', frequency_penalty)
            if np.asarray(frequency_penalty != 0.0).any():
                fp_tensor = to_tensor(frequency_penalty)
                has_penalty = True
        pp_tensor = None
        if presence_penalty is not None:
            validate_1d_batch('presence_penalty', presence_penalty)
            if np.asarray(presence_penalty != 0.0).any():
                pp_tensor = to_tensor(presence_penalty)
                has_penalty = True
        penalty_meta = PenaltyMetadata(has_penalty=has_penalty,
                                       repetition_penalty_tensor=rp_tensor,
                                       frequency_penalty_tensor=fp_tensor,
                                       presence_penalty_tensor=pp_tensor)

        do_sample_tensor = None
        tem_tensor = None
        top_k_meta = None
        top_p_meta = None
        seed_meta = None
        if do_sample is not None:
            validate_1d_batch('do_sample', do_sample)
            if any(do_sample):

                if temperature is not None:
                    validate_1d_batch('temperature', temperature)
                    if np.asarray(temperature != 1.0).any():
                        if np.asarray(temperature < LOWER_SAFE_TEMPERATURE).any():
                            raise OutOfBoundsError('temperature', 'LOWER_SAFE_TEMPERATURE', LOWER_SAFE_TEMPERATURE)
                        t0_mask = np.asarray(temperature == 0)
                        do_sample[t0_mask] = False
                        temperature[t0_mask] = 1.0
                        tem_tensor = to_tensor(temperature)

                do_sample_tensor = to_tensor(do_sample)

                max_top_k = 0
                top_k_tensor = None
                top_k_disabled_mask = None
                if top_k is not None:
                    validate_1d_batch('top_k', top_k)
                    if np.asarray(top_k < SAFE_TOP_K).any():
                        raise OutOfBoundsError('top_k', 'SAFE_TOP_K', SAFE_TOP_K)
                    if top_k.dtype != np.int_:
                        top_k = top_k.astype(np.int_)
                    if np.asarray(top_k != 0).any():
                        top_k_tmp = np.maximum(top_k - 1, 0)
                        max_top_k = top_k.max()
                        top_k_tensor = to_tensor(top_k_tmp).unsqueeze(1)
                        disabled = top_k == 0
                        top_k_disabled_mask = None
                        if np.asarray(disabled).any():
                            top_k_disabled_mask = to_tensor(disabled).view(-1, 1)
                top_k_meta = TopKMetadata(top_k_tensor=top_k_tensor,
                                          top_k_array=top_k,
                                          top_k_max=max_top_k,
                                          top_k_disabled_mask_tensor=top_k_disabled_mask)

                top_p_tensor = None
                if top_p is not None:
                    validate_1d_batch('top_p', top_p)
                    if np.asarray(top_p < LOWER_SAFE_TOP_P).any() or np.asarray(top_p > UPPER_SAFE_TOP_P).any():
                        raise OutOfBoundsError('top_p', 'SAFE_TOP_P', f'[{LOWER_SAFE_TOP_P}, {UPPER_SAFE_TOP_P}]')
                    if np.asarray(top_p != 1).any():
                        top_p_tensor = to_tensor(top_p).unsqueeze(1)
                top_p_meta = TopPMetadata(top_p_tensor, top_p)

                if seed is not None:
                    validate_1d_batch('seed', seed)
                    if np.asarray(seed < LOWER_SAFE_SEED).any() or np.asarray(seed > UPPER_SAFE_SEED).any():
                        raise OutOfBoundsError('seed', 'SAFE_SEED', f'[{LOWER_SAFE_SEED}, {UPPER_SAFE_SEED}]')
                    if seed.dtype != np.int64:
                        seed = seed.astype(np.int64)
                seed_meta = SeedMetadata(seed)

        do_sample_meta = DoSampleMetadata(do_sample_tensor, do_sample)
        return SamplingParam(
            penalty_meta=penalty_meta,
            temperature=tem_tensor,
            top_k_meta=top_k_meta,
            top_p_meta=top_p_meta,
            seed_meta=seed_meta,
            do_sample_meta=do_sample_meta,
        )


@dataclass
class SamplingMetadata:
    """
    The data class containing some metadata for sampling. It must be instantiated by `from_numpy` or 'from_batch'
    method, otherwise the safety validation will be skipped.
    """
    batch_sequence_ids: List[np.ndarray]
    reserved_sequence_ids: List[np.ndarray]
    is_prefill: bool
    is_mix: bool
    all_sequence_ids: np.ndarray
    parent_sequence_ids: np.ndarray
    group_indices: List[Tuple[int, int]]

    to_tensor: Callable

    repeating_indices: Optional[backend.Tensor] = None
    repeating_indices_array: Optional[np.ndarray] = None
    min_tokens_array: Optional[np.ndarray] = None
    repetition_penalty: Optional[backend.Tensor] = None
    frequency_penalty: Optional[backend.Tensor] = None
    presence_penalty: Optional[backend.Tensor] = None
    temperature: Optional[backend.Tensor] = None
    top_k_array: Optional[np.ndarray] = None
    top_k_idx: Optional[backend.Tensor] = None
    top_k_disabled_mask: Optional[backend.Tensor] = None
    max_top_k: int = 0
    top_p_array: Optional[np.ndarray] = None
    top_p: Optional[backend.Tensor] = None
    do_sample_array: Optional[np.ndarray] = None
    do_sample_tensor: Optional[backend.Tensor] = None
    top_logprobs_array: Optional[np.ndarray] = None
    max_logprobs: Optional[int] = None
    seed_array: Optional[np.ndarray] = None
    num_top_tokens: Optional[np.ndarray] = None
    beam_width_array: Optional[np.ndarray] = None
    max_beam_width: int = 1
    best_of_array: Optional[np.ndarray] = None
    use_beam_search_array: Optional[np.ndarray] = None
    output_lengths: Optional[np.ndarray] = None
    cumulative_logprobs: Optional[np.ndarray] = None
    random_number_generators: Optional[List[backend.Generator]] = None

    all_token_ids: Optional[backend.Tensor] = None
    output_token_ids: Optional[backend.Tensor] = None
    is_seq_prefill: Optional[np.ndarray] = None
    
    # 预计算的结构化输出 bitmask (由 preprocess 阶段填充)
    # 形状: [batch_size, vocab_size // 32]，int32 数组
    guided_bitmask: Optional[np.ndarray] = None

    # KV / batch context 句柄，与 all_sequence_ids 一一对应（由 batch_context.build_sampling_meta 填入）
    cache_ids: Optional[np.ndarray] = None

    @classmethod
    def from_numpy(
        cls,
        batch_sequence_ids: List[np.ndarray],
        reserved_sequence_ids: List[np.ndarray] = None,
        is_prefill: bool = True,
        repetition_penalty: Optional[np.ndarray] = None,
        frequency_penalty: Optional[np.ndarray] = None,
        presence_penalty: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        top_k: Optional[np.ndarray] = None,
        top_p: Optional[np.ndarray] = None,
        do_sample: Optional[np.ndarray] = None,
        top_logprobs: Optional[np.ndarray] = None,
        seeds: Optional[np.ndarray] = None,
        num_top_tokens: Optional[np.ndarray] = None,
        n: Optional[np.ndarray] = None,
        best_of: Optional[np.ndarray] = None,
        use_beam_search: Optional[np.ndarray] = None,
        output_lengths: Optional[np.ndarray] = None,
        cumulative_logprobs: Optional[np.ndarray] = None,
        to_tensor: Optional[callable] = None,
        all_sequence_ids: Optional[np.ndarray] = None,
        is_seq_prefill: Optional[np.ndarray] = None,
        is_mix: bool = False,
        cache_ids: Optional[np.ndarray] = None,
        random_number_generators: Optional[List[backend.Generator]] = None
    ) -> 'SamplingMetadata':
        batch_size = len(batch_sequence_ids)
        if not isinstance(is_prefill, bool):
            raise UnsupportedTypeError('is_prefill', 'bool')
        group_indices = []
        start = 0
        for sequence_ids in batch_sequence_ids:
            end = start + len(sequence_ids)
            group_indices.append((start, end))
            start = end
        if all_sequence_ids is None:
            all_sequence_ids = np.concatenate(batch_sequence_ids)
        validate_1d_batch('all_sequence_ids', all_sequence_ids)
        num_seqs = len(all_sequence_ids)
        if cache_ids is not None:
            validate_1d_batch('cache_ids', cache_ids)
            if len(cache_ids) != num_seqs:
                raise InconsistencyError(
                    'cache_ids', 'length of `cache_ids`', 'length of `all_sequence_ids`'
                )

        repetition_penalty_key = 'repetition_penalty'
        frequency_penalty_key = 'frequency_penalty'
        presence_penalty_key = 'presence_penalty'
        temperature_key = 'temperature'
        max_top_k_key = 'max_top_k'
        top_p_key = 'top_p'
        do_sample_key = 'do_sample'
        do_sample_tensor_key = 'do_sample_tensor'
        seeds_key = 'seeds'
        top_logprobs_key = 'top_logprobs'
        max_logprobs_key = 'max_logprobs'
        beam_width_key = 'beam_width'
        best_of_key = 'best_of'
        use_beam_search_key = 'use_beam_search'
        sampling_params = {max_top_k_key: 0}

        # validate sampling params
        if repetition_penalty is not None and np.asarray(repetition_penalty != 1.0).any():
            validate_1d_batch(repetition_penalty_key, repetition_penalty)
            if np.asarray(repetition_penalty <= LOWER_SAFE_REPETITION_PENALTY).any():
                raise OutOfBoundsError(repetition_penalty_key, 'LOWER_SAFE_REPETITION_PENALTY',
                                       LOWER_SAFE_REPETITION_PENALTY)
            sampling_params[repetition_penalty_key] = to_tensor(repetition_penalty).unsqueeze(1)
        if frequency_penalty is not None and np.asarray(frequency_penalty != 0.0).any():
            validate_1d_batch(frequency_penalty_key, frequency_penalty)
            sampling_params[frequency_penalty_key] = to_tensor(frequency_penalty).unsqueeze(1)
        if presence_penalty is not None and np.asarray(presence_penalty != 0.0).any():
            validate_1d_batch(presence_penalty_key, presence_penalty)
            sampling_params[presence_penalty_key] = to_tensor(presence_penalty).unsqueeze(1)

        if do_sample is not None and any(do_sample):
            validate_1d_batch(do_sample_key, do_sample)
            sampling_params[do_sample_key] = do_sample
            sampling_params[do_sample_tensor_key] = to_tensor(do_sample)

            if temperature is not None and np.asarray(temperature != 1.0).any():
                validate_1d_batch(temperature_key, temperature)
                if np.asarray(temperature < LOWER_SAFE_TEMPERATURE).any():
                    raise OutOfBoundsError(temperature_key, 'LOWER_SAFE_TEMPERATURE', LOWER_SAFE_TEMPERATURE)
                sampling_params[temperature_key] = to_tensor(temperature).unsqueeze(1)

            if top_k is not None:
                top_k[~do_sample & top_k > 0] = 0
                if np.asarray(top_k != 0).any():
                    validate_1d_batch('top_k', top_k)
                    if np.asarray(top_k < SAFE_TOP_K).any():
                        raise OutOfBoundsError('top_k', 'SAFE_TOP_K', SAFE_TOP_K)
                    sampling_params[max_top_k_key] = top_k.max()
                    sampling_params['top_k_idx'] = to_tensor(np.maximum(top_k - 1, 0)).unsqueeze(1)
                    disabled = top_k == 0
                    top_k_disabled_mask = None
                    if np.asarray(disabled).any():
                        top_k_disabled_mask = to_tensor(disabled).view(-1, 1)
                    sampling_params['top_k_disabled_mask'] = top_k_disabled_mask

            if top_p is not None and np.asarray(top_p != 1.0).any():
                validate_1d_batch(top_p_key, top_p)
                if np.asarray(top_p < LOWER_SAFE_TOP_P).any() or np.asarray(top_p > UPPER_SAFE_TOP_P).any():
                    raise OutOfBoundsError(top_p_key, 'SAFE_TOP_P', f'[{LOWER_SAFE_TOP_P}, {UPPER_SAFE_TOP_P}]')
                sampling_params[top_p_key] = to_tensor(top_p).unsqueeze(1)

            if seeds is not None:
                validate_1d_batch(seeds_key, seeds)
                if np.asarray(seeds < LOWER_SAFE_SEED).any() or np.asarray(seeds > UPPER_SAFE_SEED).any():
                    raise OutOfBoundsError('seed', 'SAFE_SEED', f'[{LOWER_SAFE_SEED}, {UPPER_SAFE_SEED}]')
                if seeds.dtype != np.int64:
                    seeds = seeds.astype(np.int64)
                sampling_params[seeds_key] = seeds
            else:
                sampling_params[seeds_key] = np.random.randint(
                    low=LOWER_SAFE_SEED, high=UPPER_SAFE_SEED + 1, size=batch_size, dtype=np.int64)

        if top_logprobs is not None:
            validate_1d_batch(top_logprobs_key, top_logprobs)
            if np.asarray(top_logprobs < LOWER_SAFE_LOGPROBS).any() or np.asarray(
                top_logprobs > UPPER_SAFE_LOGPROBS).any():
                raise OutOfBoundsError(top_logprobs_key, 'SAFE_LOGPROBS',
                                       f'[{LOWER_SAFE_LOGPROBS}, {UPPER_SAFE_LOGPROBS}]')
            if top_logprobs.dtype != np.int64:
                top_logprobs = top_logprobs.astype(np.int64)
            sampling_params[top_logprobs_key] = top_logprobs
            sampling_params[max_logprobs_key] = int(max(top_logprobs))
        else:
            sampling_params[top_logprobs_key] = np.zeros(batch_size, dtype=np.int64)
            sampling_params[max_logprobs_key] = 0

        if best_of is not None and np.asarray(best_of > 1).any():
            validate_1d_batch(best_of_key, best_of)
            if np.asarray(best_of < LOWER_SAFE_BEST_OF).any() or np.asarray(best_of > UPPER_SAFE_BEST_OF).any():
                raise OutOfBoundsError(best_of_key, 'SAFE_BEST_OF',
                                       f'[{LOWER_SAFE_BEST_OF}, {UPPER_SAFE_BEST_OF}]')
            if best_of.dtype != np.int64:
                best_of = best_of.astype(np.int64)
            sampling_params[best_of_key] = best_of

        if use_beam_search is not None and np.asarray(use_beam_search).any():
            validate_1d_batch(use_beam_search_key, use_beam_search)
            if use_beam_search.dtype != np.bool_:
                use_beam_search = use_beam_search.astype(np.bool_)
            sampling_params[use_beam_search_key] = use_beam_search

            if n is not None:
                validate_1d_batch(beam_width_key, n)
                if np.asarray(n < LOWER_SAFE_BEAM_WIDTH).any() or np.asarray(
                    n > UPPER_SAFE_BEAM_WIDTH).any():
                    raise OutOfBoundsError(beam_width_key, 'SAFE_BEAM_WIDTH',
                                           f'[{LOWER_SAFE_BEAM_WIDTH}, {UPPER_SAFE_BEAM_WIDTH}]')
                if n.dtype != np.int64:
                    n = n.astype(np.int64)
            else:
                logger.warning('Certain sequences use beam search, but no beam width was specified.')
                n = np.ones(num_seqs, dtype=np.int64)
            sampling_params[beam_width_key] = n
            sampling_params['max_beam_width'] = max(n)

            if output_lengths is not None:
                sampling_params['output_lengths'] = output_lengths

            if cumulative_logprobs is not None:
                sampling_params['cumulative_logprobs'] = cumulative_logprobs

        sampling_metadata = SamplingMetadata(
            batch_sequence_ids=batch_sequence_ids,
            reserved_sequence_ids=reserved_sequence_ids,
            is_prefill=is_prefill,
            all_sequence_ids=all_sequence_ids,
            parent_sequence_ids=all_sequence_ids,
            group_indices=group_indices,
            to_tensor=to_tensor,
            repetition_penalty=sampling_params.get(repetition_penalty_key),
            frequency_penalty=sampling_params.get(frequency_penalty_key),
            presence_penalty=sampling_params.get(presence_penalty_key),
            temperature=sampling_params.get(temperature_key),
            top_k_array=top_k,
            top_k_idx=sampling_params.get('top_k_idx'),
            top_k_disabled_mask=sampling_params.get('top_k_disabled_mask'),
            max_top_k=sampling_params.get(max_top_k_key),
            top_p_array=top_p,
            top_p=sampling_params.get(top_p_key),
            do_sample_array=sampling_params.get(do_sample_key),
            do_sample_tensor=sampling_params.get(do_sample_tensor_key),
            seed_array=sampling_params.get(seeds_key),
            top_logprobs_array=sampling_params.get(top_logprobs_key),
            max_logprobs=sampling_params.get(max_logprobs_key),
            num_top_tokens=num_top_tokens,
            beam_width_array=sampling_params.get(beam_width_key),
            max_beam_width=sampling_params.get('max_beam_width'),
            best_of_array=sampling_params.get(best_of_key),
            use_beam_search_array=sampling_params.get(use_beam_search_key),
            output_lengths=sampling_params.get('output_lengths'),
            cumulative_logprobs=sampling_params.get('cumulative_logprobs'),
            is_seq_prefill=is_seq_prefill,
            is_mix=is_mix,
            cache_ids=cache_ids,
            random_number_generators=random_number_generators
        )
        return sampling_metadata

    @classmethod
    def from_batch(
        cls,
        input_metadata: InputMetadata,
        batch_sampling_params: np.ndarray,
        num_top_tokens: np.ndarray,
        to_tensor: Callable,
        **kwargs
    ) -> 'SamplingMetadata':
        sampling_metadata = cls.from_numpy(
            batch_sequence_ids=input_metadata.batch_sequence_ids,
            reserved_sequence_ids=input_metadata.reserved_sequence_ids,
            is_prefill=input_metadata.is_prefill,
            all_sequence_ids=input_metadata.all_sequence_ids,
            repetition_penalty=batch_sampling_params['repetition_penalty'].astype(np.float32),
            frequency_penalty=batch_sampling_params['frequency_penalty'].astype(np.float32),
            presence_penalty=batch_sampling_params['presence_penalty'].astype(np.float32),
            temperature=batch_sampling_params['temperature'].astype(np.float32),
            top_k=batch_sampling_params['top_k'].astype(np.int_),
            top_p=batch_sampling_params['top_p'].astype(np.float32),
            do_sample=batch_sampling_params['do_sample'].astype(np.bool_),
            top_logprobs=batch_sampling_params['top_logprobs'].astype(np.int_),
            seeds=kwargs.get('batch_seeds'),
            num_top_tokens=num_top_tokens,
            best_of=kwargs.get('batch_best_of'),
            n=kwargs.get('batch_n'),
            use_beam_search=kwargs.get('batch_use_beam_search'),
            output_lengths=kwargs.get('batch_output_lengths'),
            cumulative_logprobs=kwargs.get('batch_cumulative_logprobs'),
            to_tensor=to_tensor,
            is_seq_prefill=kwargs.get('is_seq_prefill'),
            is_mix=kwargs.get('is_mix'),
            cache_ids=kwargs.get('cache_ids'),
            random_number_generators=kwargs.get('random_number_generators')
        )
        return sampling_metadata

    @classmethod
    def from_deprecated(cls, sampling_data: SamplingData, sampling_param: SamplingParam) -> 'SamplingMetadata':
        if sampling_data.request_ids is None:
            batch_sequence_ids = [np.array([i]) for i in range(len(sampling_data.all_input_ids))]
            logger.warning('The `request_ids` is not provided, so it will be automatically generated. The instance in'
                           'this mode can be used only once, or it will lead to using incorrect sampling parameters.')
        else:
            batch_sequence_ids = [np.array([i]) for i in sampling_data.request_ids]

        repetition_penalty = None
        frequency_penalty = None
        presence_penalty = None
        if getattr(sampling_param, 'penalty_meta', None) is not None:
            repetition_penalty = sampling_param.penalty_meta.repetition_penalty.cpu().numpy()
            frequency_penalty = sampling_param.penalty_meta.frequency_penalty.cpu().numpy()
            presence_penalty = sampling_param.penalty_meta.presence_penalty.cpu().numpy()

        top_k = None
        if getattr(sampling_param, 'top_k_meta', None) is not None:
            top_k = sampling_param.top_k_meta.top_k_array

        top_p = None
        if getattr(sampling_param, 'top_p_meta', None) is not None:
            top_p = sampling_param.top_p_meta.top_p_array

        seeds = None
        if getattr(sampling_param, 'seed_meta', None) is not None:
            seeds = sampling_param.seed_meta.seed_array

        sampling_metadata = cls.from_numpy(
            batch_sequence_ids=batch_sequence_ids,
            is_prefill=sampling_data.is_prefill,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=sampling_param.temperature.cpu().numpy() if sampling_param.temperature is not None else None,
            top_k=top_k,
            top_p=top_p,
            do_sample=sampling_param.do_sample_meta.do_sample_array,
            seeds=seeds,
            to_tensor=sampling_data.to_tensor
        )
        sampling_metadata.all_token_ids = sampling_data.all_input_ids
        sampling_metadata.output_token_ids = sampling_data.output_ids
        return sampling_metadata

    def update_token_ids(self, all_token_ids: Optional[np.ndarray], output_token_ids: Optional[np.ndarray]) -> None:
        if all_token_ids is not None and self.repetition_penalty is not None:
            all_token_ids_key = 'all_token_ids'
            if len(all_token_ids) != len(self.all_sequence_ids):
                raise InconsistencyError(all_token_ids_key, 'length of `all_token_ids`',
                                            'length of `all_sequence_ids`')
            if len(all_token_ids[0]) > UPPER_SAFE_SEQUENCE_LENGTH:
                raise OutOfBoundsError(all_token_ids_key, 'UPPER_SAFE_SEQUENCE_LENGTH', UPPER_SAFE_SEQUENCE_LENGTH)
            self.all_token_ids = self.to_tensor(all_token_ids)
        if output_token_ids is not None:
            if self.frequency_penalty is not None or self.presence_penalty is not None:
                output_token_ids_key = 'output_token_ids'
                if len(output_token_ids) != len(self.all_sequence_ids):
                    raise InconsistencyError(output_token_ids_key, 'length of `output_token_ids`',
                                                'length of `all_sequence_ids`')
                if len(output_token_ids[0]) > UPPER_SAFE_SEQUENCE_LENGTH:
                    raise OutOfBoundsError(output_token_ids_key, 'UPPER_SAFE_SEQUENCE_LENGTH',
                                           UPPER_SAFE_SEQUENCE_LENGTH)
                self.output_token_ids = self.to_tensor(output_token_ids)

    def update_beam_search(self, cumulative_logprobs, output_lengths):
        if self.cumulative_logprobs is not None:
            self.cumulative_logprobs = cumulative_logprobs
            self.output_lengths = output_lengths