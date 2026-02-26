# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import time
import unittest
from unittest.mock import MagicMock, Mock

import numpy as np

from mindie_llm.utils.env import ENV
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.text_generator.utils.config import ContextParams, SpCpParallelInfo


class TestError(unittest.TestCase):
    def setUp(self):
        # Set environment variables
        ENV.framework_backend = 'atb'
        ENV.log_file_level = 'ERROR'
        ENV.log_to_file = 'true'
        
        # Correct way to get logger
        from mindie_llm.utils.log.logging_base import get_logger, Component
        logger = get_logger(Component.LLM)
        
        # Get actual log file path
        self.log_path = None
        for handler in logger.logger.handlers:
            if hasattr(handler, 'baseFilename'):  # Ensure it's a file handler
                self.log_path = handler.baseFilename
                break
                
        if not self.log_path:
            raise RuntimeError("No file handler found in logger")

        backend_type_key = 'backend_type'

        self.test_generator_backend_params = {
            backend_type_key: 'xxx'
        }
        self.test_logits_shape_params = {
            backend_type_key: ENV.framework_backend,
            'batch_sequence_ids': [np.array([0]), np.array([1]), np.array([2])],
            'top_k': np.array([0, 1, 10]),
            'top_p': np.array([1.0, 0.9, 0.6]),
            'seed': np.array([0, 1, 2 ** 63 - 1]),
            'do_sample': np.array([True, True, True]),
            'logits': np.random.normal(size=(5, 65536))
        }
        self.test_invalid_decoding_params = {
            backend_type_key: ENV.framework_backend
        }
        self.test_block_size_params = {
            'max_block_size': 0
        }
        self.test_eos_token_id_params = {
            'eos_token_id': 'abc'
        }

    def test_generator_backend(self):
        """The test for `TEXT_GENERATOR_GENERATOR_BACKEND_INVALID`"""
        from mindie_llm.text_generator.adapter import get_generator_backend

        log_bak = self._clear_log()
        with self.assertRaises(NotImplementedError):
            get_generator_backend(self.test_generator_backend_params)
        log = self._read_log()
        self.assertIn(str(ErrorCode.TEXT_GENERATOR_GENERATOR_BACKEND_INVALID), log)
        self._recover_log(log_bak)

    def test_logits_shape(self):
        """The test for `TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH`"""
        from mindie_llm.text_generator.samplers.sampler_params import SelectorParams
        from mindie_llm.text_generator.samplers.token_selectors.cpu_selectors import TopKTopPSamplingTokenSelector
        from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata
        from mindie_llm.utils.tensor import tensor_backend

        log_bak = self._clear_log()
        with self.assertRaises(ValueError):
            selector_params = SelectorParams()
            top_k_top_p_sampling_token_selector = TopKTopPSamplingTokenSelector(selector_params)
            sampling_metadata = SamplingMetadata.from_numpy(
                batch_sequence_ids=self.test_logits_shape_params.get('batch_sequence_ids'),
                top_k=self.test_logits_shape_params.get('top_k'),
                top_p=self.test_logits_shape_params.get('top_p'),
                seeds=self.test_logits_shape_params.get('seed'),
                do_sample=self.test_logits_shape_params.get('do_sample'),
                top_logprobs=np.array([1, 1, 1]),
                to_tensor=tensor_backend.tensor
            )
            top_k_top_p_sampling_token_selector.configure(sampling_metadata)
            top_k_top_p_sampling_token_selector(
                logits=tensor_backend.tensor(self.test_logits_shape_params.get('logits')),
                metadata=sampling_metadata
            )
        top_k_top_p_sampling_token_selector.clear(sampling_metadata.all_sequence_ids)
        log = self._read_log()
        self.assertIn(str(ErrorCode.TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH), log)
        self._recover_log(log_bak)

    def test_invalid_decoding(self):
        """The test for `TEXT_GENERATOR_MISSING_PREFILL_OR_INVALID_DECODE_REQ`"""
        from mindie_llm.modeling.model_wrapper.model_info import ModelInfo
        from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
        from mindie_llm.text_generator.utils.config import CacheConfig
        from mindie_llm.text_generator.utils.tg_infer_context_store import TGInferContextStore
        from mindie_llm.text_generator.utils.input_metadata import InputMetadata
        from mindie_llm.utils.tensor import tensor_backend

        log_bak = self._clear_log()
        with self.assertRaises(RuntimeError):
            model_info = ModelInfo(
                device='cpu',
                dtype=tensor_backend.get_backend().float16,
                data_byte_size=tensor_backend.tensor([], dtype=tensor_backend.get_backend().float16).element_size(),
                num_layers=1,
                num_kv_heads=8,
                head_size=128
            )
            model_wrapper = MagicMock(model_info=model_info)
            model_wrapper.mapping.attn_cp.group_size = 1
            model_wrapper.mapping.attn_inner_sp.group_size = 1
            cache_config = CacheConfig()
            kvcache_settings = KVCacheSettings(
                0, model_info, 5, 5, 128, self.test_invalid_decoding_params.get("backend_type"), False
            )
            spcp_info = (Mock(group_size=1, rank=0), Mock(group_size=1, rank=0))
            context_params = ContextParams(distributed=False)
            tokenizer = MagicMock()
            infer_context = TGInferContextStore(
                kvcache_settings=kvcache_settings,
                batch_context_config=cache_config,
                spcp_parallel_info=spcp_info,
                device='cpu',
                context_params=context_params,
                tokenizer=tokenizer,
                tokenizer_sliding_window_size=3
            )
            input_metadata = InputMetadata(
                batch_size=1,
                batch_request_ids=np.array([0]),
                batch_sequence_ids=[np.array([0])],
                batch_max_output_lens=np.array([1024]),
                block_tables=np.array([[0]]),
                is_prefill=False,
                reserved_sequence_ids=[np.array([0])],
            )
            infer_context.get_batch_context_handles(input_metadata)
        log = self._read_log()
        self.assertIn(str(ErrorCode.TEXT_GENERATOR_MISSING_PREFILL_OR_INVALID_DECODE_REQ), log)
        self._recover_log(log_bak)

    def test_block_size(self):
        """The test for `TEXT_GENERATOR_MAX_BLOCK_SIZE_INVALID`"""
        from mindie_llm.text_generator.utils.input_metadata import InputMetadata

        log_bak = self._clear_log()
        with self.assertRaises(ZeroDivisionError):
            InputMetadata(
                batch_size=1,
                batch_request_ids=np.array([0]),
                batch_sequence_ids=[np.array([0])],
                batch_max_output_lens=np.array([1024]),
                block_tables=np.array([[0]]),
                is_prefill=True,
                max_block_size=self.test_block_size_params.get('max_block_size'),
                reserved_sequence_ids=[np.array([0])],
            )
        log = self._read_log()
        self.assertIn(str(ErrorCode.TEXT_GENERATOR_MAX_BLOCK_SIZE_INVALID), log)
        self._recover_log(log_bak)

    def test_eos_token_id(self):
        """The test for 'TEXT_GENERATOR_EOS_TOKEN_ID_TYPE_INVALID'"""
        from mindie_llm.text_generator.utils.config import CacheConfig

        log_bak = self._clear_log()
        eos_token_id = self.test_eos_token_id_params.get('eos_token_id')
        with self.assertRaises(ValueError):
            cache_config = CacheConfig()
            cache_config.set_eos_token_id(eos_token_id)
        log = self._read_log()
        self.assertIn(str(ErrorCode.TEXT_GENERATOR_EOS_TOKEN_ID_TYPE_INVALID), log)
        self._recover_log(log_bak)

    def _clear_log(self):
        log_bak = ''
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as file:
                log_bak = file.read()
            with open(self.log_path, 'w') as file:
                file.write('')
        return log_bak

    def _read_log(self):
        time.sleep(0.1)
        with open(self.log_path, 'r') as file:
            log = file.read()
        return log

    def _recover_log(self, content):
        with open(self.log_path, 'w') as file:
            file.write(content)


if __name__ == '__main__':
    # If you need to check the error information, please use this command:
    # MINDIE_LOG_TO_STDOUT=1 python tests/pythontest/npu/test_error.py
    # Otherwise, use this command directly:
    # python tests/pythontest/npu/test_error.py
    unittest.main()