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
from unittest.mock import patch
from mindie_llm.model_wrapper.utils.npu_compile import set_npu_compile_mode


class TestSetNpuCompileMode(unittest.TestCase):
    @patch('mindie_llm.model_wrapper.utils.npu_compile.logger')
    @patch('mindie_llm.model_wrapper.utils.npu_compile.torch_npu')
    @patch('mindie_llm.model_wrapper.utils.npu_compile.torch')
    def test_soc_version_in_list(self, mock_torch, mock_torch_npu, mock_logger):
        mock_torch_npu._C._npu_get_soc_version.return_value = 104
        set_npu_compile_mode()
        mock_torch.npu.set_compile_mode.assert_called_once_with(jit_compile=False)
        mock_torch.npu.set_option.assert_not_called()
        mock_logger.info.assert_not_called()

    @patch('mindie_llm.model_wrapper.utils.npu_compile.logger')
    @patch('mindie_llm.model_wrapper.utils.npu_compile.torch_npu')
    @patch('mindie_llm.model_wrapper.utils.npu_compile.torch')
    def test_soc_version_not_in_list(self, mock_torch, mock_torch_npu, mock_logger):
        mock_torch_npu._C._npu_get_soc_version.return_value = 100
        set_npu_compile_mode()
        mock_torch.npu.set_compile_mode.assert_called_once_with(jit_compile=False)
        mock_torch.npu.set_option.assert_called_once_with(
            {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceNansum"}
        )
        mock_logger.info.assert_called_once_with("310P,some op does not support")

if __name__ == '__main__':
    unittest.main()
