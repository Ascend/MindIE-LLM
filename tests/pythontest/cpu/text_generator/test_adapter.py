#!/usr/bin/env python
# coding=utf-8
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
from unittest.mock import MagicMock, patch

from mindie_llm.text_generator.adapter import get_generator_backend  # noqa: E402
from mindie_llm.text_generator.adapter.generator_ms import GeneratorMS  # noqa: E402
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch  # noqa: E402


class TestAdapter(unittest.TestCase):
    @patch('mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch',
           return_value=MagicMock(GeneratorTorch))
    def test_get_generator_torch(self, mock_generator_torch):
        backend_type = 'atb'
        generator_backend = get_generator_backend({'backend_type': backend_type})
        self.assertIsInstance(generator_backend, GeneratorTorch)

    @patch('mindie_llm.text_generator.adapter.generator_ms.GeneratorMS',
           return_value=MagicMock(GeneratorMS))
    def test_get_generator_ms(self, mock_generator_ms):
        backend_type = 'ms'
        generator_backend = get_generator_backend({'backend_type': backend_type})
        self.assertIsInstance(generator_backend, GeneratorMS)

    def test_get_generator_backend_exception(self):
        backend_type = 'xxx'
        with self.assertRaises(NotImplementedError):
            get_generator_backend({'backend_type': backend_type})


if __name__ == '__main__':
    unittest.main()
