# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import random
import unittest

import numpy as np

from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.text_generator.samplers.sampler import Sampler
from mindie_llm.text_generator.utils.config import HandlingBackend, SamplerConfig
from mindie_llm.text_generator.utils.sampling_metadata import SamplingData, SamplingParam, SamplingMetadata

BACKEND_TYPE = BackendType.ATB


class NumpySampler:
    def __init__(self, dtype):
        self.dtype = dtype

    @staticmethod
    def count_token_ids(logits, token_ids):
        token_ids_counts = np.zeros((len(logits), len(logits[0]) + 1))
        token_ids_row = np.repeat(np.arange(len(token_ids)), len(token_ids[0]))
        token_ids_col = token_ids.flatten()
        token_ids_indices = (token_ids_row, token_ids_col)
        np.add.at(token_ids_counts, token_ids_indices, np.ones_like(token_ids_col))
        return token_ids_counts[:, :-1]

    @staticmethod
    def compute_logprobs(logits):
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, None]
        logprobs = np.log(probs)
        return probs, logprobs

    @staticmethod
    def apply_top_k(logits, top_k):
        sorted_indices = np.argsort(-logits)
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=1)
        top_k_results = []
        for i, indices in enumerate(sorted_indices):
            probs = np.exp(sorted_logits[i][:top_k[i]]) / np.sum(np.exp(sorted_logits[i][:top_k[i]]))
            top_k_results.append((indices[:top_k[i]], probs))
        return zip(*top_k_results)

    @staticmethod
    def apply_top_p(batch_indices, batch_probs, top_p):
        top_p_results = []
        for i, probs in enumerate(batch_probs):
            cumsum_probs = np.cumsum(probs)
            last_id = 0
            if top_p[i] == 1:
                top_p_results.append((batch_indices[i], probs))
            else:
                for j, c_p in enumerate(cumsum_probs):
                    if c_p >= top_p[i]:
                        last_id = j
                        break
                probs = probs[:last_id + 1] / cumsum_probs[last_id]
                top_p_results.append((batch_indices[i][:last_id + 1], probs))
        return zip(*top_p_results)

    @staticmethod
    def apply_sampling(batch_indices, batch_probs, do_sample, seed):
        token_ids = []
        logprobs = []
        for i, do in enumerate(do_sample):
            if do:
                rng = np.random.default_rng(seed[i])
                noise = rng.exponential(size=len(batch_probs[i]))
                index = np.argmax(batch_probs[i] / noise)
                token_ids.append(batch_indices[i][index])
                logprobs.append(np.log(batch_probs[i][index]))
            else:
                token_ids.append(batch_indices[i][0])
                logprobs.append(np.log(batch_probs[i][0]))
        return np.array(token_ids), np.array(logprobs)

    def apply_repetition_penalty(self, logits, all_token_ids, repetition_penalty):
        logits = np.append(logits, np.zeros((len(logits), 1)), axis=1)
        scores = np.take_along_axis(logits, all_token_ids, axis=1)
        scores = np.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
        np.put_along_axis(logits, all_token_ids, scores, axis=1)
        return logits[:, :-1].astype(self.dtype)

    def apply_frequency_penalty(self, logits, token_ids_counts, frequency_penalty):
        logits -= token_ids_counts * frequency_penalty
        return logits.astype(self.dtype)

    def apply_presence_penalty(self, logits, token_ids_counts, presence_penalty):
        token_ids_mask = token_ids_counts > 0
        logits -= token_ids_mask * presence_penalty
        return logits.astype(self.dtype)

    def apply_temperature(self, logits, temperature):
        logits /= temperature
        return logits.astype(self.dtype)


class TestSampler(unittest.TestCase):
    @staticmethod
    def golden(data, params):
        logits = data.get('logits')
        sampler = NumpySampler(logits.dtype)
        all_token_ids = data.get('all_token_ids')
        output_token_ids = data.get('output_token_ids')
        vocab_size = params.get('vocab_size')

        repetition_penalty = params.get('repetition_penalty')[:, None]
        frequency_penalty = params.get('frequency_penalty')[:, None]
        presence_penalty = params.get('presence_penalty')[:, None]
        temperature = params.get('temperature')
        top_k = params.get('top_k')
        top_p = params.get('top_p')
        seed = params.get('seed')
        do_sample = params.get('do_sample')

        do_sample[np.asarray(temperature == 0)] = False
        temperature[np.asarray(temperature == 0)] = 1
        top_k[~do_sample] = 0
        top_p[~do_sample] = 1.0
        top_k[np.asarray(top_k == 0)] = vocab_size
        temperature = temperature[:, None]

        logits = sampler.apply_repetition_penalty(logits, all_token_ids, repetition_penalty)
        token_ids_counts = sampler.count_token_ids(logits, output_token_ids)
        logits = sampler.apply_frequency_penalty(logits, token_ids_counts, frequency_penalty)
        logits = sampler.apply_presence_penalty(logits, token_ids_counts, presence_penalty)
        logits = sampler.apply_temperature(logits, temperature)
        token_ids, probs = sampler.apply_top_k(logits, top_k)
        token_ids, probs = sampler.apply_top_p(token_ids, probs, top_p)
        chosen_token_ids, chosen_logprobs = sampler.apply_sampling(token_ids, probs, do_sample, seed)
        golden_samples = {
            'chosen_token_ids': chosen_token_ids,
            'chosen_logprobs': chosen_logprobs,
            'token_ids': token_ids,
            'probs': probs
        }
        return golden_samples

    def setUp(self):
        self.device = 'npu'
        self.backend_type = BACKEND_TYPE

        if self.backend_type == BackendType.ATB:
            import torch

            def to_tensor_torch(data_):
                return torch.tensor(data_, device=self.device)

            self.to_tensor = to_tensor_torch
        elif self.backend_type == BackendType.MS:
            import mindspore as ms

            def to_tensor_ms(data_):
                return ms.Tensor(data_) if data_.size > 0 else None

            self.to_tensor = to_tensor_ms
        else:
            raise ValueError('No such backend type.')

    def test_sampler_precision(self):
        # set basic params
        batch_size = 3
        dtype = np.float16
        max_input_length = 16
        max_output_length = 16
        vocab_size = 16
        pad_token_id = vocab_size

        # set sampling params
        params = {
            'handling_policy': {
                'repetition_penalty': HandlingBackend.PTA,
                'frequency_penalty': HandlingBackend.PTA,
                'presence_penalty': HandlingBackend.PTA,
                'temperature': HandlingBackend.PTA,
                'top_k': HandlingBackend.PTA,
                'top_p': HandlingBackend.PTA
            },
            'selection_policy': {
                'greedy_search': HandlingBackend.PTA,
                'sampling': HandlingBackend.PTA,
                'top_k_top_p_sampling': HandlingBackend.CPU,
                'beam_search': HandlingBackend.PTA
            },
            'num_threads': 16,
            'test_seed': 0,
            'vocab_size': vocab_size,
            'repetition_penalty': np.array([1, 0.9, 1.1]).astype(dtype),
            'frequency_penalty': np.array([0.9, 1, 1.1]).astype(dtype),
            'presence_penalty': np.array([0.9, 1.1, 1]).astype(dtype),
            'temperature': np.array([0.8, 1, 1.2]).astype(dtype),
            'top_k': np.array([0, 10, 15]),
            'top_p': np.array([1, 0.8, 0.5]).astype(dtype),
            'seed': np.array([1, 2, 3]),
            'do_sample': np.array([False, True, True])
        }

        # make some random inputs
        batch_logits = np.random.randn(batch_size, vocab_size).astype(dtype)
        batch_input_ids = []
        for _ in range(batch_size):
            input_length = random.randint(1, max_input_length)
            input_ids = [random.randint(0, vocab_size - 1) for _ in range(input_length)]
            batch_input_ids.append(input_ids)
        batch_output_ids = []
        for _ in range(batch_size):
            output_length = random.randint(1, max_output_length)
            output_ids = [random.randint(0, vocab_size - 1) for _ in range(output_length)]
            batch_output_ids.append(output_ids)

        batch_sequence_ids = [in_ids + out_ids for in_ids, out_ids in zip(batch_input_ids, batch_output_ids)]
        max_length = max(len(i) for i in batch_output_ids)
        batch_output_ids_array = np.array(
            [i + [pad_token_id] * (max_length - len(i)) for i in batch_output_ids], dtype=np.int32)
        max_length = max(len(i) for i in batch_sequence_ids)
        batch_sequence_ids_array = np.array(
            [i + [pad_token_id] * (max_length - len(i)) for i in batch_sequence_ids], dtype=np.int32)
        data = {
            'logits': batch_logits,
            'all_token_ids': batch_sequence_ids_array,
            'output_token_ids': batch_output_ids_array,
            'is_prefill': True
        }

        golden_samples = self.golden(data, params)
        token_ids, logprobs = self.__sample(data, params)
        for i, token_id in enumerate(token_ids):
            if not params.get('do_sample')[i]:
                self.assertEqual(token_id, golden_samples.get('chosen_token_ids')[i])
            else:
                self.assertIn(token_id, golden_samples.get('token_ids')[i])
                idx = np.where(golden_samples.get('token_ids')[i] == token_id)
                self.assertAlmostEqual(logprobs[i], np.log(golden_samples.get('probs')[i][idx]), delta=0.01)

    def test_async_sampling(self):
        handling_policy = {
            'repetition_penalty': HandlingBackend.PTA,
            'frequency_penalty': HandlingBackend.PTA,
            'presence_penalty': HandlingBackend.PTA,
            'temperature': HandlingBackend.PTA,
            'top_k': HandlingBackend.PTA,
            'top_p': HandlingBackend.PTA
        }
        selection_policy = {
            'greedy_search': HandlingBackend.PTA,
            'sampling': HandlingBackend.PTA,
            'top_k_top_p_sampling': HandlingBackend.CPU,
            'beam_search': HandlingBackend.PTA
        }
        num_threads = 8
        sampler_config = SamplerConfig(
            backend_type=self.backend_type,
            handling_policy=handling_policy,
            selection_policy=selection_policy,
            num_threads=num_threads
        )
        sampler = Sampler(sampler_config)
        sampler.initialize('npu', [1])
        batch_size = 3
        vocab_size = 128
        dtype = np.float16
        logits = np.random.randn(batch_size, vocab_size).astype(dtype)
        sampling_metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            reserved_sequence_ids=[np.array([]), np.array([3]), np.array([4])],
            is_prefill=True,
            top_k=np.array([3, 50, 1000]),
            top_p=np.array([0.9, 0.9, 0.9]),
            do_sample=np.array([False, True, False]),
            top_logprobs=np.array([5, 5, 5]),
            seeds=np.array([0, 0, 0]),
            num_top_tokens=np.array([5, 5, 5]),
            n=np.array([1, 1, 2]),
            best_of=np.array([1, 2, 2]),
            use_beam_search=np.array([False, False, True]),
            cumulative_logprobs=np.array([0, 0, -0.1]),
            output_lengths=np.array([1, 1, 1]),
            to_tensor=self.to_tensor,
            all_sequence_ids=np.array([0, 1, 2])
        )
        sampler(self.to_tensor(logits), sampling_metadata)

    def __sample(self, data, params):
        handling_policy = params.get('handling_policy')
        selection_policy = params.get('selection_policy')
        num_threads = params.get('num_threads')
        test_seed = params.get('test_seed')
        random.seed(test_seed)

        sampler_config = SamplerConfig(
            backend_type=self.backend_type,
            handling_policy=handling_policy,
            selection_policy=selection_policy,
            num_threads=num_threads
        )
        sampler = Sampler(sampler_config)
        sampler.initialize('npu', [1])

        sampling_data = SamplingData.from_numpy(
            data.get('all_token_ids'),
            data.get('output_token_ids'),
            self.to_tensor,
            data.get('is_prefill')
        )

        sampling_params = SamplingParam.from_numpy(
            repetition_penalty=params.get('repetition_penalty'),
            frequency_penalty=params.get('frequency_penalty'),
            presence_penalty=params.get('presence_penalty'),
            temperature=params.get('temperature'),
            top_k=params.get('top_k'),
            top_p=params.get('top_p'),
            seed=params.get('seed'),
            do_sample=params.get('do_sample'),
            to_tensor=self.to_tensor
        )

        sampling_metadata = SamplingMetadata.from_deprecated(sampling_data, sampling_params)
        output = sampler(self.to_tensor(data.get('logits')), sampling_metadata)
        return output.token_ids, output.logprobs


if __name__ == '__main__':
    unittest.main()