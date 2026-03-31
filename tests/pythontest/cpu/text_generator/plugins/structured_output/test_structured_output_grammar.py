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
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 检测 xgrammar 是否可用
try:
    import xgrammar as xgr
    XGRAMMAR_AVAILABLE = True
except ImportError:
    XGRAMMAR_AVAILABLE = False

from mindie_llm.text_generator.plugins.structured_output.structured_output_grammar import (
    StructuredOutputRequest,
    StructuredOutputType,
    XgrammarGrammar,
)


class TestStructuredOutputType(unittest.TestCase):
    """StructuredOutputType 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        self.assertEqual(StructuredOutputType.JSON_OBJECT.value, "json_object")
        self.assertEqual(StructuredOutputType.JSON_SCHEMA.value, "json_schema")
        self.assertEqual(len(StructuredOutputType), 2)

    def test_enum_from_string(self):
        """测试从字符串创建枚举"""
        self.assertEqual(StructuredOutputType("json_object"), StructuredOutputType.JSON_OBJECT)
        self.assertEqual(StructuredOutputType("json_schema"), StructuredOutputType.JSON_SCHEMA)

    def test_enum_invalid_string(self):
        """测试无效字符串抛出异常"""
        with self.assertRaises(ValueError):
            StructuredOutputType("invalid_type")


class TestStructuredOutputRequest(unittest.TestCase):
    """StructuredOutputRequest 测试"""

    def test_from_response_format_none(self):
        result = StructuredOutputRequest.from_response_format(None)
        self.assertIsNone(result)

    def test_from_response_format_json_object(self):
        response_format = '{"type": "json_object"}'
        result = StructuredOutputRequest.from_response_format(response_format)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, StructuredOutputType.JSON_OBJECT)
        self.assertEqual(result.grammar_spec, '{"type": "object"}')

    def test_from_response_format_json_schema_with_schema_field(self):
        response_format = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        })
        result = StructuredOutputRequest.from_response_format(response_format)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, StructuredOutputType.JSON_SCHEMA)
        self.assertIn("type", json.loads(result.grammar_spec))

    def test_from_response_format_json_schema_direct(self):
        response_format = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "age_schema",
                "type": "object",
                "properties": {"age": {"type": "integer"}}
            }
        })
        result = StructuredOutputRequest.from_response_format(response_format)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, StructuredOutputType.JSON_SCHEMA)

    def test_from_response_format_json_string(self):
        response_format = '{"type": "json_object"}'
        result = StructuredOutputRequest.from_response_format(response_format)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.output_type, StructuredOutputType.JSON_OBJECT)

    def test_from_response_format_invalid_json_string(self):
        response_format = '{type: json_object}'  # 缺少引号
        result = StructuredOutputRequest.from_response_format(response_format)
        self.assertIsNone(result)

    def test_from_response_format_missing_type(self):
        response_format = '{"json_schema": {}}'
        result = StructuredOutputRequest.from_response_format(response_format)
        self.assertIsNone(result)

    def test_from_response_format_invalid_type(self):
        response_format = '{"type": "invalid"}'
        result = StructuredOutputRequest.from_response_format(response_format)
        self.assertIsNone(result)
    
    def test_from_response_format_json_schema_missing_name(self):
        response_format = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        })
        result = StructuredOutputRequest.from_response_format(response_format)
        self.assertIsNone(result)
    
    def test_from_response_format_json_schema_empty_name(self):
        response_format = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        })
        result = StructuredOutputRequest.from_response_format(response_format)
        self.assertIsNone(result)
    
    def test_from_response_format_json_schema_name_not_string(self):
        response_format = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": 123,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        })
        result = StructuredOutputRequest.from_response_format(response_format)
        self.assertIsNone(result)

    def test_request_grammar_field(self):
        request = StructuredOutputRequest(
            output_type=StructuredOutputType.JSON_OBJECT,
            grammar_spec='{"type": "object"}'
        )
        self.assertIsNone(request.grammar)


class TestXgrammarGrammar(unittest.TestCase):
    """XgrammarGrammar 类测试"""

    def setUp(self):
        self.mock_matcher = MagicMock()
        self.vocab_size = 32000
        self.mock_ctx = MagicMock()
        
        self.grammar = XgrammarGrammar(
            matcher=self.mock_matcher,
            vocab_size=self.vocab_size,
            ctx=self.mock_ctx
        )

    def test_init(self):
        self.assertEqual(self.grammar.vocab_size, self.vocab_size)
        self.assertEqual(self.grammar.ctx, self.mock_ctx)
        self.assertEqual(self.grammar.num_processed_tokens, 0)
        self.assertFalse(self.grammar.is_terminated())

    def test_accept_tokens_success(self):
        self.mock_matcher.accept_token.return_value = True
        self.mock_matcher.is_terminated.return_value = False
        
        result = self.grammar.accept_tokens("req_001", [100, 200, 300])
        
        self.assertTrue(result)
        self.assertEqual(self.mock_matcher.accept_token.call_count, 3)
        self.assertEqual(self.grammar.num_processed_tokens, 3)

    def test_accept_tokens_rejected(self):
        self.mock_matcher.accept_token.side_effect = [True, False]  # 第二个 token 被拒绝
        self.mock_matcher.is_terminated.return_value = False
        
        result = self.grammar.accept_tokens("req_001", [100, 200])
        
        self.assertFalse(result)
        self.assertEqual(self.grammar.num_processed_tokens, 1)

    def test_accept_tokens_terminated(self):
        self.mock_matcher.accept_token.return_value = True
        self.mock_matcher.is_terminated.side_effect = [False, True]  # 第二个 token 后终止
        
        result = self.grammar.accept_tokens("req_001", [100, 200])
        
        self.assertTrue(result)
        self.assertTrue(self.grammar.is_terminated())
        self.assertEqual(self.mock_matcher.accept_token.call_count, 2)

    def test_accept_tokens_already_terminated(self):
        self.grammar._is_terminated = True
        
        result = self.grammar.accept_tokens("req_001", [100, 200])
        
        self.assertTrue(result)
        self.mock_matcher.accept_token.assert_not_called()

    def test_fill_bitmask_normal(self):
        bitmask = np.zeros((2, 1000), dtype=np.int32)
        
        self.grammar.fill_bitmask(bitmask, idx=0)
        
        self.mock_matcher.fill_next_token_bitmask.assert_called_once_with(bitmask, 0)

    def test_fill_bitmask_terminated(self):
        self.grammar._is_terminated = True
        bitmask = np.zeros((2, 1000), dtype=np.int32)
        
        self.grammar.fill_bitmask(bitmask, idx=0)
        
        self.assertTrue(np.all(bitmask[0, :] == -1))
        self.mock_matcher.fill_next_token_bitmask.assert_not_called()

    def test_is_terminated(self):
        self.assertFalse(self.grammar.is_terminated())
        
        self.grammar._is_terminated = True
        self.assertTrue(self.grammar.is_terminated())

    def test_num_processed_tokens_property(self):
        self.assertEqual(self.grammar.num_processed_tokens, 0)
        
        self.mock_matcher.accept_token.return_value = True
        self.mock_matcher.is_terminated.return_value = False
        self.grammar.accept_tokens("req_001", [100, 200, 300])
        
        self.assertEqual(self.grammar.num_processed_tokens, 3)


@unittest.skipIf(not XGRAMMAR_AVAILABLE, "xgrammar not installed, skipping real backend tests")
class TestXgrammarGrammarReal(unittest.TestCase):
    
    def setUp(self):
        self.vocab_size = 32000
        
        class SimpleTokenizer:
            def __init__(self):
                # 使用列表来存储，这样索引自然就是 ID
                self.vocab_list = ['{', '}', '[', ']', '"', ',', ':', ' ', '\n']
                for i in range(10):
                    self.vocab_list.append(str(i))
                for i in range(26):
                    self.vocab_list.append(chr(ord('a') + i))

            def get_vocab(self):
                return self.vocab_list

        self.tokenizer = SimpleTokenizer()
        
        self.tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab=self.tokenizer.get_vocab(),
            vocab_type=xgr.VocabType.RAW,
            vocab_size=self.vocab_size,
        )
        
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
    
    def test_real_compile_json_object_grammar(self):
        schema = '{"type": "object"}'
        ctx = self.compiler.compile_json_schema(schema, any_whitespace=True)
        
        self.assertIsNotNone(ctx)
    
    def test_real_create_grammar_matcher(self):
        schema = '{"type": "object"}'
        ctx = self.compiler.compile_json_schema(schema, any_whitespace=True)
        
        matcher = xgr.GrammarMatcher(ctx)
        
        self.assertIsNotNone(matcher)
        
        grammar = XgrammarGrammar(
            matcher=matcher,
            vocab_size=self.vocab_size,
            ctx=ctx
        )
        
        self.assertIsNotNone(grammar)
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(grammar.num_processed_tokens, 0)
    
    def test_real_fill_bitmask(self):
        schema = '{"type": "object"}'
        ctx = self.compiler.compile_json_schema(schema, any_whitespace=True)
        matcher = xgr.GrammarMatcher(ctx)
        
        grammar = XgrammarGrammar(
            matcher=matcher,
            vocab_size=self.vocab_size,
            ctx=ctx
        )
        
        bitmask_width = (self.vocab_size + 31) // 32
        bitmask = np.zeros((1, bitmask_width), dtype=np.int32)
        
        grammar.fill_bitmask(bitmask, idx=0)
        
        self.assertFalse(np.all(bitmask == 0))
    

if __name__ == '__main__':
    unittest.main()