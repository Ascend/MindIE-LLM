# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from mindie_llm.text_generator.plugins.mtp.decoding_policy import DecodingPolicy
from mindie_llm.text_generator.plugins.mtp.mtp_plugin import MtpPlugin
from mindie_llm.text_generator.utils.model_input import ModelInput
from mindie_llm.utils.log.logging import logger


class TestAsyncMTP(unittest.TestCase):
    @patch('mindie_llm.text_generator.plugins.mtp.decoding_policy.DecodingPolicy',
           return_value=MagicMock(DecodingPolicy))
    def setUp(self, mock):
        self.device = 'npu'
        generator_backend = MagicMock()
        generator_backend.model_wrapper = MagicMock()
        generator_backend.model_wrapper.device = self.device
        generator_backend.rank = 0
        kvcache_settings = MagicMock()
        kvcache_settings.dtype = torch.float16
        infer_context = MagicMock()
        infer_context.spcp_parallel_info.scp_size = 1
        block_size = 128
        num_npu_blocks = 3

        def block_table_to_slots_side_effect(block_tables):
            return infer_context._batch_context.kv_slots[block_tables]
        
        infer_context.block_table_to_slots.side_effect = block_table_to_slots_side_effect
        infer_context._batch_context.kv_slots = np.arange(num_npu_blocks * block_size).reshape(num_npu_blocks, -1)
        self.mtp_plugin = MtpPlugin(
            generator_backend=generator_backend,
            kvcache_settings=kvcache_settings,
            infer_context=infer_context,
            output_filter=MagicMock(),
            plugin_data_param=MagicMock(),
            num_speculative_tokens=1
        )

        def to_tensor(data):
            return torch.tensor(data, device=self.device)

        def to_tensor_async(array):
            host_tensor = torch.from_numpy(array).pin_memory()
            device_tensor = host_tensor.to(self.device, non_blocking=True)
            return device_tensor

        generator_backend.to_tensor = to_tensor
        generator_backend.to_tensor_async = to_tensor_async

    def test_fill_in_model_result(self):
        # inputs for mask preparing
        model_inputs = ModelInput(
            input_ids=torch.tensor([1000, 1001]).to(self.device),
            position_ids=torch.tensor([2, 3]).to(self.device),
            block_tables=torch.tensor([[0]]).to(self.device),
            slots=torch.tensor([2, 3]).to(self.device),
            context_length=np.array([4]),
            cached_context_length=np.array([4]),
            max_seq_len=4,
            prefill_head_indices=None,
            is_prefill=False,
            block_tables_array=np.array([[0]]),
            input_lengths=torch.tensor([4]).to(self.device)
        )
        current_dp_sequence_ids = np.array([100])
        current_all_sequence_ids = np.array([100, 102])
        last_all_sequence_ids = np.array([100, 101, 102])

        # inputs for filling
        sub_model_inputs = ModelInput(
            input_ids=torch.tensor([1000, 1001]).to(self.device),
            position_ids=torch.tensor([2, 3]).to(self.device),
            block_tables=torch.tensor([[0]]).to(self.device),
            slots=torch.tensor([2, 3]).to(self.device),
            context_length=np.array([4]),
            cached_context_length=np.array([4]),
            max_seq_len=3,
            prefill_head_indices=torch.tensor([1, 3]).to(self.device),
            is_prefill=False,
            block_tables_array=np.array([[0]]),
            input_lengths=torch.tensor([4]).to(self.device),
        )
        model_kwargs = {
            'sub_model_inputs': sub_model_inputs,
            'hidden_states': torch.randn((2, 7168)).to(self.device)
        }

        model_output_wrapper = MagicMock()
        model_output_wrapper.model_output = MagicMock()
        model_output_wrapper.model_output.hidden_states = torch.randn((6, 7168)).to(self.device)
        model_output_wrapper.sampling_output = MagicMock()
        model_output_wrapper.sampling_output.token_ids = np.array([[1110, 1111], [2221, 2222], [3332, 0]])
        model_output_wrapper.sampling_output.num_new_tokens = np.array([2, 2, 1])

        preparing_kwargs = {
            'model_inputs': model_inputs,
            'current_dp_sequence_ids': current_dp_sequence_ids,
            'current_all_sequence_ids': current_all_sequence_ids,
            'last_all_sequence_ids': last_all_sequence_ids
        }
        filling_kwargs = {
            'model_inputs': model_inputs,
            'model_kwargs': model_kwargs,
            'model_output_wrapper': model_output_wrapper,
            'cache_ids': MagicMock(),
            'input_metadata': MagicMock()
        }
        expected_results = {
            'sub_input_ids': torch.tensor([1110, 1111]).to(self.device),
            'prefill_head_indices': torch.tensor([1, 2]).to(self.device),
            'input_hidden_states': model_output_wrapper.model_output.hidden_states[:2],
            'main_input_ids': torch.tensor([1111, 1001]).to(self.device),
            'position_ids': torch.tensor([4, 5]).to(self.device),
            'input_lengths': torch.tensor([6]).to(self.device),
            'slots': torch.tensor([4, 5]).to(self.device)
        }
        self._run_mtp_filling(preparing_kwargs, filling_kwargs, expected_results)

    def _run_mtp_filling(self, preparing_kwargs, filling_kwargs, expected_results):
        # test for precision
        sub_model_inputs = filling_kwargs.get('model_kwargs', {}).get('sub_model_inputs')
        input_hidden_states = filling_kwargs.get('model_kwargs', {}).get('hidden_states')
        model_inputs = filling_kwargs.get('model_inputs')
        filling_masks = self.mtp_plugin.prepare_masks_for_filling(**preparing_kwargs)
        self.mtp_plugin.fill_in_model_result(**filling_kwargs, filling_masks=filling_masks)
        self.assertTrue(torch.equal(sub_model_inputs.input_ids, expected_results.get('sub_input_ids')))
        self.assertTrue(torch.equal(sub_model_inputs.prefill_head_indices,
                                    expected_results.get('prefill_head_indices')))
        self.assertTrue(torch.allclose(input_hidden_states, expected_results.get('input_hidden_states')))
        self.assertTrue(torch.equal(model_inputs.input_ids, expected_results.get('main_input_ids')))
        self.assertTrue(torch.equal(model_inputs.position_ids, expected_results.get('position_ids')))
        self.assertTrue(torch.equal(model_inputs.input_lengths, expected_results.get('input_lengths')))
        self.assertTrue(torch.equal(model_inputs.slots, expected_results.get('slots')))

        # The performance of first running is not stable, so we need another test for performance
        filling_masks = self.mtp_plugin.prepare_masks_for_filling(**preparing_kwargs)
        start_time = time.time()
        self.mtp_plugin.fill_in_model_result(**filling_kwargs, filling_masks=filling_masks)
        end_time = time.time()
        filling_time = (end_time - start_time) * 1000
        self.assertLess(filling_time, 8)  # severe degradation
        if filling_time > 3.5:
            logger.warning(f'The filling performance decreases: {filling_time} ms')
        else:
            logger.info(f'The filling time: {filling_time} ms')


if __name__ == "__main__":
    unittest.main()