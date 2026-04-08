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

import torch_npu

from mindie_llm.model_wrapper.utils.env import ENV
from mindie_llm.model_wrapper.utils.npu_compile import set_npu_compile_mode


class TestSetNpuCompileMode(unittest.TestCase):
    def setUp(self):
        pass

    def test_torch(self):
        set_npu_compile_mode()
        self.assertTrue(torch_npu.npu.is_jit_compile_false())
    
    def test_ms(self):
        ENV.framework_backend = "ms"
        set_npu_compile_mode()
        ENV.framework_backend = "atb"
