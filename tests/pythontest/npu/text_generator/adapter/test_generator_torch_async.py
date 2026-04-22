# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from mindie_llm.text_generator.adapter.generator_torch_async import GeneratorTorchAsync


class TestPlugin(unittest.TestCase):
    @patch("mindie_llm.text_generator.adapter.generator_torch_async.GeneratorTorchAsync.__init__", return_value=None)
    def setUp(self, _):
        self.generator_backend = GeneratorTorchAsync({})
        self.generator_backend.device = "cpu"

    def test_to_tensor_async(self):
        data = np.random.rand(10, 10)
        # Mock the to_tensor_async method to avoid calling pin_memory()
        self.generator_backend.to_tensor_async = MagicMock(return_value=torch.tensor(data))
        data_device_tensor = self.generator_backend.to_tensor_async(data)
        data_host_tensor = data_device_tensor.cpu()
        self.assertTrue((data_host_tensor == torch.tensor(data)).all())

    def test_dp_exception(self):
        setattr(self.generator_backend, "mapping", MagicMock())
        self.generator_backend.mapping.has_dp = MagicMock(return_value=True)
        model_input = MagicMock()
        model_input.dp_rank_ids = None
        with self.assertRaises(AssertionError) as context:
            self.generator_backend.prepare_model_inputs(model_input)
        self.assertIn("dp_rank_ids", str(context.exception))

    def test_result_tuple(self):
        model_input = MagicMock()
        setattr(self.generator_backend, "model_wrapper", MagicMock())
        setattr(self.generator_backend, "cache_pool", MagicMock())

        def mock_forward_with_hidden_states(*args, **kwargs):
            return torch.tensor([0]), torch.tensor([1])

        def mock_forward_with_draft_tokens(*args, **kwargs):
            return torch.tensor([0]), torch.tensor([1]), torch.tensor([2])

        self.generator_backend.model_wrapper.forward_from_model_inputs = mock_forward_with_hidden_states
        model_output = self.generator_backend.forward_from_model_inputs(model_input)
        self.assertIsNotNone(model_output.hidden_states)

        self.generator_backend.model_wrapper.forward_from_model_inputs = mock_forward_with_draft_tokens
        model_output = self.generator_backend.forward_from_model_inputs(model_input)
        self.assertIsNotNone(model_output.draft_tokens)


if __name__ == "__main__":
    unittest.main()
