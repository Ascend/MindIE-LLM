# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import unittest
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 检测 xgrammar 是否可用
try:
    import xgrammar as xgr
    XGRAMMAR_AVAILABLE = True
except ImportError:
    XGRAMMAR_AVAILABLE = False

from mindie_llm.text_generator.plugins.structured_output import apply_token_bitmask_inplace
from mindie_llm.text_generator.plugins.structured_output.structured_output_manager import (
    StructuredOutputConfig,
    GuidedDecodingBackendType,
    GrammarBackend,
    StructuredOutputManager,
    CompiledGrammar,
)
from mindie_llm.text_generator.plugins.structured_output.structured_output_grammar import (
    StructuredOutputRequest,
    StructuredOutputType,
    XgrammarGrammar,
)


class FakeTokenizer:
    """模拟 Tokenizer"""
    def __init__(self):
        self.vocab = {f"token_{i}": i for i in range(1000)}
    
    def get_vocab(self):
        return self.vocab


class RealTokenizerForXgrammar:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.vocab = {}
        chars = ['{', '}', '[', ']', '"', ',', ':', ' ', '\n', '\t']
        for i, char in enumerate(chars):
            self.vocab[char] = i
        for i in range(10):
            self.vocab[str(i)] = len(self.vocab)
        for i in range(26):
            self.vocab[chr(ord('a') + i)] = len(self.vocab)
            self.vocab[chr(ord('A') + i)] = len(self.vocab)
        current_size = len(self.vocab)
        for i in range(current_size, vocab_size):
            self.vocab[f"<token_{i}>"] = i
    
    def get_vocab(self):
        return self.vocab
    
    def encode(self, text):
        result = []
        for char in text:
            if char in self.vocab:
                result.append(self.vocab[char])
            else:
                result.append(0)
        return result
    
    def decode(self, token_ids):
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(reverse_vocab.get(tid, '') for tid in token_ids)


class TestStructuredOutputConfig(unittest.TestCase):
    """StructuredOutputConfig 测试"""

    def test_default_config(self):
        config = StructuredOutputConfig()
        self.assertEqual(config.backend, GuidedDecodingBackendType.XGRAMMAR)
        self.assertTrue(config.xgrammar_any_whitespace)
        self.assertEqual(config.grammar_cache_size, 100)
        self.assertEqual(config.bitmask_prealloc_batch, 64)

    def test_custom_config(self):
        config = StructuredOutputConfig(
            backend=GuidedDecodingBackendType.XGRAMMAR,
            xgrammar_any_whitespace=False,
            grammar_cache_size=50,
            bitmask_prealloc_batch=128
        )
        self.assertFalse(config.xgrammar_any_whitespace)
        self.assertEqual(config.grammar_cache_size, 50)
        self.assertEqual(config.bitmask_prealloc_batch, 128)


class TestGrammarBackend(unittest.TestCase):
    """GrammarBackend 测试"""

    def setUp(self):
        """设置测试环境"""
        self.tokenizer = FakeTokenizer()
        self.vocab_size = 32000
        self.config = StructuredOutputConfig()

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_init_xgrammar_raises_when_tokenizer_info_fails(self, mock_logger):
        mock_xgr = MagicMock()
        mock_xgr.TokenizerInfo.from_huggingface.side_effect = Exception("Failed")
        
        with patch.dict('sys.modules', {'xgrammar': mock_xgr}):
            with self.assertRaises(RuntimeError) as context:
                GrammarBackend(
                    backend_type=GuidedDecodingBackendType.XGRAMMAR,
                    tokenizer=self.tokenizer,
                    vocab_size=self.vocab_size,
                    config=self.config
                )
            self.assertIn("TokenizerInfo.from_huggingface failed", str(context.exception))
            mock_xgr.TokenizerInfo.assert_not_called()


class TestStructuredOutputManager(unittest.TestCase):
    """StructuredOutputManager 测试"""

    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.vocab_size = 32000
        self.config = StructuredOutputConfig(
            grammar_cache_size=10,
            bitmask_prealloc_batch=4
        )

    def test_init(self):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertEqual(manager.tokenizer, self.tokenizer)
        self.assertEqual(manager.vocab_size, self.vocab_size)
        self.assertEqual(manager.config, self.config)
        self.assertIsNone(manager._backend)  # 延迟初始化
        self.assertEqual(len(manager._grammar_cache), 0)
        self.assertEqual(len(manager._request_grammars), 0)

    def test_init_default_config(self):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size
        )
        self.assertIsNotNone(manager.config)
        self.assertEqual(manager.config.backend, GuidedDecodingBackendType.XGRAMMAR)

    def test_init_bitmask_buffer(self):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertIsNotNone(manager._bitmask_buffer)
        expected_width = (self.vocab_size + 31) // 32
        self.assertEqual(manager._bitmask_buffer.shape, (self.config.bitmask_prealloc_batch, expected_width))
        self.assertEqual(manager._bitmask_buffer.dtype, np.int32)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_ensure_backend_lazy_init(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        backend = manager._ensure_backend()
        self.assertIsNotNone(backend)
        mock_backend_class.assert_called_once()
        self.assertEqual(manager._backend, mock_backend)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_ensure_backend_singleton(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        backend1 = manager._ensure_backend()
        backend2 = manager._ensure_backend()
        self.assertEqual(backend1, backend2)
        mock_backend_class.assert_called_once()

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_compile_grammar_with_cache(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        compiled1 = manager._compile_grammar(
            StructuredOutputType.JSON_OBJECT,
            '{"type": "object"}'
        )
        compiled2 = manager._compile_grammar(
            StructuredOutputType.JSON_OBJECT,
            '{"type": "object"}'
        )
        self.assertEqual(compiled1, compiled2)
        self.assertEqual(mock_backend.compile_grammar.call_count, 1)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_compile_grammar_cache_eviction(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend.compile_grammar.return_value = MagicMock()
        mock_backend_class.return_value = mock_backend
        config = StructuredOutputConfig(grammar_cache_size=2)
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=config
        )
        manager._compile_grammar(StructuredOutputType.JSON_OBJECT, '{"type": "object"}')
        manager._compile_grammar(StructuredOutputType.JSON_SCHEMA, '{"type": "string"}')
        manager._compile_grammar(StructuredOutputType.JSON_SCHEMA, '{"type": "integer"}')
        self.assertLessEqual(len(manager._grammar_cache), config.grammar_cache_size)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_init_success(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        result = manager.grammar_init("req_001", request)
        self.assertIsNotNone(result)
        self.assertIsNotNone(request.grammar)
        self.assertIn("req_001", manager._request_grammars)

    def test_grammar_init_none_request(self):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        result = manager.grammar_init("req_001", None)
        self.assertIsNone(result)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_grammar_init_failure(self, mock_logger, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend.compile_grammar.side_effect = Exception("Compilation failed")
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        result = manager.grammar_init("req_001", request)
        self.assertIsNone(result)
        mock_logger.error.assert_called_once()

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_bitmask_no_requests(self, mock_backend_class):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        result = manager.grammar_bitmask([])
        self.assertIsNone(result)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_bitmask_no_constraints(self, mock_backend_class):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        result = manager.grammar_bitmask(["req_001", "req_002"])
        self.assertIsNone(result)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_bitmask_with_constraints(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.fill_bitmask = MagicMock()
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        bitmask = manager.grammar_bitmask(["req_001", "req_002"])
        self.assertIsNotNone(bitmask)
        self.assertEqual(bitmask.shape[0], 2)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_bitmask_terminated(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = True
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        
        bitmask = manager.grammar_bitmask(["req_001"])
        
        # 已终止的 grammar 应该设置 full mask
        self.assertIsNotNone(bitmask)
        self.assertTrue(np.all(bitmask[0, :] == -1))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_bitmask_apply_flags(self, mock_backend_class):
        """测试 apply_bitmask_flags"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        bitmask = manager.grammar_bitmask(
            ["req_001", "req_002"],
            apply_bitmask_flags=[True, False]
        )
        self.assertIsNotNone(bitmask)


    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_accept_tokens_success(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.accept_tokens.return_value = True
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        result = manager.accept_tokens("req_001", [100, 200, 300])
        self.assertTrue(result)

    def test_accept_tokens_no_grammar(self):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        result = manager.accept_tokens("req_001", [100, 200])
        self.assertTrue(result)  # 没有约束，返回成功

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_should_advance(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        manager.should_advance("req_001")
        manager.should_advance("req_002")

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_is_terminated(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = True
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        
        self.assertTrue(manager.is_terminated("req_001"))
        self.assertTrue(manager.is_terminated("req_002"))  # 没有 grammar，视为已完成

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_clear_requests(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request1 = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        request2 = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request1)
        manager.grammar_init("req_002", request2)
        manager.clear_requests(["req_001", "req_002"])
        self.assertNotIn("req_001", manager._request_grammars)
        self.assertNotIn("req_002", manager._request_grammars)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_get_request_grammar(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        result = manager.get_request_grammar("req_001")
        self.assertIsNotNone(result)
        result = manager.get_request_grammar("req_002")
        self.assertIsNone(result)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_has_structured_output(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init("req_001", request)
        self.assertTrue(manager.has_structured_output("req_001"))
        self.assertFalse(manager.has_structured_output("req_002"))

    def test_shutdown(self):
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        manager.shutdown()
        self.assertEqual(len(manager._request_grammars), 0)

    def test_process_batch_for_generation_empty(self):
        """process_batch_for_generation：空 sequence_ids 或空 response_format_array 返回 None。"""
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertIsNone(manager.process_batch_for_generation([], ["{}"]))
        self.assertIsNone(manager.process_batch_for_generation([1], []))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_process_batch_for_generation_no_constraint(self, mock_backend_class):
        """process_batch_for_generation：无任何约束时返回 None。"""
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertIsNone(manager.process_batch_for_generation([1, 2], [None, None]))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_process_batch_for_generation_init_failure_warning(self, mock_logger, mock_backend_class):
        """process_batch_for_generation：某请求 init grammar 失败时记录 warning 仍返回 bitmask。"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.fill_bitmask = MagicMock()
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        mock_backend_class.create_grammar.return_value = mock_grammar
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        # 第一个请求用有效 schema 初始化成功，第二个用无效 response_format 会 init 失败
        result = manager.process_batch_for_generation(
            [1, 2],
            ['{"type": "json_object"}', "invalid not json"]
        )
        self.assertIsNotNone(result)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_process_batch_for_generation_with_constraint(self, mock_backend_class):
        """process_batch_for_generation：有约束时初始化并返回 bitmask。"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.fill_bitmask = MagicMock()
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        mock_backend_class.create_grammar.return_value = mock_grammar
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        result = manager.process_batch_for_generation([1], ['{"type": "json_object"}'])
        self.assertIsNotNone(result)

    def test_update_states_after_sampling_empty(self):
        """update_states_after_sampling：空 sequence_ids 或 None token_ids 直接返回。"""
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        manager.update_states_after_sampling([], np.array([1]))
        manager.update_states_after_sampling([1], None)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_update_states_after_sampling_token_rejected(self, mock_logger, mock_backend_class):
        """update_states_after_sampling：accept_tokens 返回 False 时记录 warning。"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.accept_tokens.return_value = False
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init(1, request)
        manager.update_states_after_sampling([1], np.array([100]))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_update_states_after_sampling_exception(self, mock_logger, mock_backend_class):
        """update_states_after_sampling：accept_tokens 抛异常时记录 warning。"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.accept_tokens.side_effect = ValueError("bad token")
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init(1, request)
        manager.update_states_after_sampling([1], np.array([100]))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_clear_finished_requests(self, mock_backend_class):
        """clear_finished_requests 调用 clear_requests。"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init(1, request)
        manager.clear_finished_requests(np.array([1, 2]))
        self.assertNotIn(1, manager._request_grammars)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_init_grammar_from_response_format_invalid_returns_false(self, mock_backend_class):
        """_init_grammar_from_response_format：无效 response_format 返回 False。"""
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertFalse(manager._init_grammar_from_response_format(1, None))
        self.assertFalse(manager._init_grammar_from_response_format(1, "not json {{{"))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_init_grammar_from_response_format_grammar_init_none_returns_false(self, mock_logger, mock_backend_class):
        """_init_grammar_from_response_format：grammar_init 返回 None 时返回 False。"""
        mock_backend = MagicMock()
        mock_backend.compile_grammar.side_effect = Exception("compile fail")
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertFalse(manager._init_grammar_from_response_format(1, '{"type": "json_object"}'))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_init_grammar_from_response_format_exception_returns_false(self, mock_logger, mock_backend_class):
        """_init_grammar_from_response_format：解析或编译异常时返回 False。"""
        mock_backend = MagicMock()
        mock_backend.compile_grammar.side_effect = RuntimeError("backend error")
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertFalse(manager._init_grammar_from_response_format(1, '{"type": "json_object"}'))
        self.assertTrue(mock_logger.error.called)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_grammar_bitmask_buffer_resize(self, mock_backend_class):
        """grammar_bitmask：batch_size > 预分配 buffer 时扩容。"""
        config = StructuredOutputConfig(bitmask_prealloc_batch=2)
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.fill_bitmask = MagicMock()
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init(1, request)
        bitmask = manager.grammar_bitmask([1, 2, 3, 4])
        self.assertIsNotNone(bitmask)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_grammar_bitmask_fill_bitmask_exception(self, mock_logger, mock_backend_class):
        """grammar_bitmask：fill_bitmask 抛异常时记录 warning 并继续。"""
        mock_backend = MagicMock()
        mock_compiled = MagicMock()
        mock_grammar = MagicMock(spec=XgrammarGrammar)
        mock_grammar.is_terminated.return_value = False
        mock_grammar.fill_bitmask.side_effect = RuntimeError("fill error")
        mock_backend.compile_grammar.return_value = mock_compiled
        mock_backend.create_grammar.return_value = mock_grammar
        mock_backend_class.return_value = mock_backend
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        manager.grammar_init(1, request)
        bitmask = manager.grammar_bitmask([1])
        self.assertIsNotNone(bitmask)

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager._get_xgrammar_module')
    def test_grammar_backend_init_raises_when_xgrammar_none(self, mock_get_xgr):
        """GrammarBackend 初始化时 _get_xgrammar_module 返回 None 则抛出 ImportError。"""
        mock_get_xgr.return_value = None
        with self.assertRaises(ImportError):
            GrammarBackend(
                backend_type=GuidedDecodingBackendType.XGRAMMAR,
                tokenizer=self.tokenizer,
                vocab_size=self.vocab_size,
                config=self.config
            )

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager._get_xgrammar_module')
    def test_compile_xgrammar_unsupported_output_type_raises(self, mock_get_xgr):
        """_compile_xgrammar 收到不支持的 output_type 时抛出 ValueError。"""
        mock_xgr = MagicMock()
        mock_get_xgr.return_value = mock_xgr
        mock_xgr.TokenizerInfo.from_huggingface.return_value = MagicMock()
        mock_xgr.GrammarCompiler.return_value = MagicMock()
        backend = GrammarBackend(
            backend_type=GuidedDecodingBackendType.XGRAMMAR,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        fake_output_type = MagicMock()
        fake_output_type.value = "unsupported_type"
        with self.assertRaises(ValueError):
            backend.compile_grammar(fake_output_type, '{"type": "object"}')

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.logger')
    def test_should_advance_grammar_none_returns_false(self, mock_logger):
        """should_advance 在 grammar 为 None 时记录 warning 并返回 False。"""
        manager = StructuredOutputManager(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            config=self.config
        )
        self.assertFalse(manager.should_advance(999))

    @patch('mindie_llm.text_generator.plugins.structured_output.structured_output_manager.GrammarBackend')
    def test_get_cache_key(self, mock_backend_class):
        """_get_cache_key 生成稳定键。"""
        key1 = StructuredOutputManager._get_cache_key(
            StructuredOutputType.JSON_OBJECT,
            '{"type": "object"}'
        )
        key2 = StructuredOutputManager._get_cache_key(
            StructuredOutputType.JSON_OBJECT,
            '{"type": "object"}'
        )
        self.assertEqual(key1, key2)
        self.assertIn(StructuredOutputType.JSON_OBJECT.value, key1)
        key3 = StructuredOutputManager._get_cache_key(
            StructuredOutputType.JSON_SCHEMA,
            '{"type": "string"}'
        )
        self.assertNotEqual(key1, key3)
        self.assertIn(StructuredOutputType.JSON_SCHEMA.value, key3)


if __name__ == '__main__':
    unittest.main()
