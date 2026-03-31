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
from unittest.mock import patch, MagicMock, Mock
import torch
import numpy as np

from mindie_llm.text_generator.utils.batch_context import DictContext, NdarrayContext, BatchContext
from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
from mindie_llm.text_generator.utils.config import ContextParams, CacheConfig, SpCpParallelInfo, DEFAULT_SAMPLING_PARAMS
from mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess import SplitFusePreprocess
from mindie_llm.text_generator.utils.tg_infer_context_store import TGInferContextStore


class TestSplitFusePreprocess(unittest.TestCase):

    @patch("mindie_llm.text_generator.plugins.splitfuse.splitfuse_preprocess.ENV.framework_backend", new='atb')
    def test_make_attention_mask_is300i(self):
        infer_context = MagicMock()
        model_wrapper = MagicMock()
        model_wrapper.device = 'npu:0'
        model_wrapper.model_runner = MagicMock()
        model_wrapper.model_runner.attn_mask = MagicMock()
        model_wrapper.model_runner.attn_mask.get_attn_mask.return_value = torch.ones(5, 5)
        kvcache_settings = MagicMock()
        kvcache_settings.dtype = torch.float16
        splitfuse_preprocess = SplitFusePreprocess(infer_context, model_wrapper, kvcache_settings)
        splitfuse_preprocess.is_300i = True
        splitfuse_preprocess.async_inference = False
        model_inputs = MagicMock()
        model_inputs.max_seq_len = 1
        model_inputs.context_length = [3, 3]
        input_metadata = MagicMock()
        input_metadata.is_prefill = True
        q_lens = [2, 2]
        hit_mask = None

        req_mask = splitfuse_preprocess.make_attention_mask(model_inputs, input_metadata, q_lens, hit_mask)

        golden_mask = torch.ones(4, 5)
        self.assertTrue(torch.allclose(req_mask, golden_mask))


if __name__ == '__main__':
    unittest.main()
