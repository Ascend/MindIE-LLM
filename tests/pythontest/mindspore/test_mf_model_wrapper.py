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
from unittest.mock import patch

import numpy as np

from mindie_llm.modeling.model_wrapper.ms.mf_model_wrapper import MFModelWrapper, SwapDirection


class TestMFModelWrapper(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    @patch("mindie_llm.modeling.model_wrapper.ms.mf_model_wrapper.ModelRunner")
    @patch("mindie_llm.modeling.model_wrapper.ms.mf_model_wrapper.ModelInfo")
    @patch("mindspore.Tensor")
    def setUp(self, model_runner, model_info, tensor):
        class ModelConfig:
            def __init__(self):
                self.max_position_embedding = 1

        class ModelRunner:
            def __init__(self, *args, **kwargs):
                self.model_config = ModelConfig()
                self.tokenizer = ModelConfig()
                self.dtype = 1
                self.num_layers = 1
                self.head_size = 1
                self.num_kv_heads = 1
                self.item_size = 1
                self.itemsize = 1
                self.use_legacy = False

            @staticmethod
            def forward(**kwargs):
                return kwargs

        model_runner.return_value = ModelRunner()
        model_info.return_value = ModelRunner()
        tensor.return_value = ModelRunner()

        self.model_wrapper = MFModelWrapper(1, 0, "1", 1, 1, 1, 1)

    @patch("mindie_llm.modeling.model_wrapper.ms.mf_model_wrapper.MFModelWrapper.get_model_input_ids",
           return_value=(1, 2))
    @patch("mindie_llm.modeling.model_wrapper.ms.mf_model_wrapper.MFModelWrapper.generate_position_ids",
           return_value=range(3))
    def test_init(self, _1, _2):
        class ModelInputs:
            def __init__(self, *args, **kwargs):
                self.context_length = 1
                self.is_prefill = False
                self.block_tables = []
                self.position_ids = []
                self.adapter_ids = []
                self.prefill_head_indices = 1

        model_inputs = ModelInputs()
        self.model_wrapper.forward(model_inputs)
        self.model_wrapper.generate_position_ids(model_inputs)
        self.assertEqual(SwapDirection.RECOMPUTE, 2)
        self.assertEqual(SwapDirection.NPU_TO_CPU, 1)
        self.assertEqual(SwapDirection.CPU_TO_NPU, 0)

    def test_get_model_input_ids_case_not_use_legacy(self):
        class ModelInputs:
            def __init__(self, *args, **kwargs):
                self.input_ids = np.array([[1, 2, 3], [4, 5, 6]])
                self.slots = np.array([[7, 8], [9, 10]])

        model_inputs = ModelInputs()
        golden_input_ids = np.array([1, 2, 3, 4, 5, 6])
        golden_slots = np.array([7, 8, 9, 10])
        input_ids, slots = self.model_wrapper.get_model_input_ids(model_inputs)

        self.assertTrue((input_ids == golden_input_ids).all())
        self.assertTrue((slots == golden_slots).all())

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()