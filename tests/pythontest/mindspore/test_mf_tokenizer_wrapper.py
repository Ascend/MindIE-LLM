# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np


class TestMFTokenizerWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mindformers_mock = MagicMock()
        sys.modules['mindformers'] = cls.mindformers_mock
        cls.mindformers_mock.get_model.return_value = (MagicMock(), MagicMock())
        
        from mindie_llm.modeling.model_wrapper.ms.mf_tokenizer_wrapper import MFTokenizerWrapper, CONTENT_KEY
        cls.MFTokenizerWrapper = MFTokenizerWrapper
        cls.CONTENT_KEY = CONTENT_KEY

    def setUp(self):
        self.mindformers_mock.get_model.reset_mock()
        self.tokenizer_mock = MagicMock()
        self.input_builder_mock = MagicMock()
        self.mindformers_mock.get_model.return_value = (self.tokenizer_mock, self.input_builder_mock)
        
        self.wrapper = self.MFTokenizerWrapper("test_model")

    def test_initialization(self):
        self.mindformers_mock.get_model.assert_called_once_with("test_model")
        self.assertEqual(self.wrapper.tokenizer, self.tokenizer_mock)
        self.assertEqual(self.wrapper.input_builder, self.input_builder_mock)
        self.assertIsNone(self.wrapper.toolscallprocessor)

    def test_tokenize(self):
        self.tokenizer_mock.tokenize.return_value = [1, 2, 3]
        result = self.wrapper.tokenize("test input", some_arg="value")
        self.tokenizer_mock.tokenize.assert_called_once_with("test input", some_arg="value")
        self.assertEqual(result, [1, 2, 3])

    def test_decode_non_stream(self):
        self.tokenizer_mock.decode.return_value = "decoded text"
        result = self.wrapper.decode([1, 2, 3], False, False, False, False)

        called_with_array = self.tokenizer_mock.decode.call_args[0][0]
        self.assertTrue(np.array_equal(called_with_array, np.array([1, 2, 3])))
        self.assertEqual(result, {self.CONTENT_KEY: "decoded text"})

    def test_decode_stream(self):
        self.tokenizer_mock.decode.side_effect = ["full text", "pre"]
        result = self.wrapper.decode(
            [1, 2, 3, 4, 5], False, False, False, True,
            curr_decode_index=3, prev_decode_index=1
        )
        self.assertEqual(result, {self.CONTENT_KEY: "full text"[len("pre"):]})

    def test_decode_edge_cases(self):
        self.tokenizer_mock.decode.side_effect = ["error�", "pre"]
        result = self.wrapper.decode(
            [1, 2, 3], False, False, False, True,
            curr_decode_index=2, prev_decode_index=1
        )
        self.assertEqual(result, {self.CONTENT_KEY: ""})

    def test_encode_with_chatting(self):
        """测试 encode 方法的 is_chatting=True 分支"""
        self.input_builder_mock.make_context.return_value = [1, 2, 3, 4]
        result = self.wrapper.encode("test input", is_chatting=True, some_arg="value")
        
        self.input_builder_mock.make_context.assert_called_once_with(0, "test input", some_arg="value")
        self.assertEqual(result, [1, 2, 3, 4])

    def test_encode_without_chatting(self):
        """测试 encode 方法的 is_chatting=False 分支"""
        mock_tensor = MagicMock()
        mock_tensor.tolist.return_value = [1, 2, 3]
        self.tokenizer_mock.return_value = {"input_ids": [mock_tensor]}
        
        result = self.wrapper.encode("test input", is_chatting=False, some_arg="value")
        
        self.tokenizer_mock.assert_called_once_with("test input", some_arg="value")
        self.assertEqual(result, [1, 2, 3])

    def test_encode_default_chatting(self):
        """测试 encode 方法的默认 is_chatting 行为"""
        mock_tensor = MagicMock()
        mock_tensor.tolist.return_value = [5, 6, 7]
        self.tokenizer_mock.return_value = {"input_ids": [mock_tensor]}
        
        result = self.wrapper.encode("test input")
        
        self.tokenizer_mock.assert_called_once_with("test input")
        self.assertEqual(result, [5, 6, 7])

    def test_tokenize_exception(self):
        """测试 tokenize 方法的异常处理"""
        test_exception = Exception("Tokenize failed")
        self.tokenizer_mock.tokenize.side_effect = test_exception
        
        with self.assertRaises(Exception) as context:
            self.wrapper.tokenize("test input")
        
        self.assertEqual(str(context.exception), "Tokenize failed")

    def test_detokenize_exception(self):
        """测试 _detokenize 方法的异常处理"""
        test_exception = Exception("Decode failed")
        self.tokenizer_mock.decode.side_effect = test_exception
        
        with self.assertRaises(Exception) as context:
            self.wrapper.decode([1, 2, 3], False, False, False, False)
        
        self.assertEqual(str(context.exception), "Decode failed")

    def test_detokenize_stream_exception(self):
        """测试 _detokenize_stream 方法的异常处理"""
        test_exception = Exception("Stream decode failed")
        self.tokenizer_mock.decode.side_effect = test_exception
        
        with self.assertRaises(Exception) as context:
            self.wrapper.decode(
                [1, 2, 3], False, False, False, True,
                curr_decode_index=2, prev_decode_index=1
            )
        
        self.assertEqual(str(context.exception), "Stream decode failed")

    def test_decode_stream_with_default_indices(self):
        """测试流式解码使用默认索引参数"""
        self.tokenizer_mock.decode.side_effect = ["full text", ""]
        result = self.wrapper.decode([1, 2, 3], False, False, False, True)
        
        # 验证使用默认的 curr_decode_index=-1, prev_decode_index=-1
        self.assertEqual(result, {self.CONTENT_KEY: "full text"})

    def test_decode_stream_equal_length_texts(self):
        """测试流式解码中 full_text 和 pre_text 长度相等的情况"""
        self.tokenizer_mock.decode.side_effect = ["same", "same"]
        result = self.wrapper.decode(
            [1, 2, 3], False, False, False, True,
            curr_decode_index=2, prev_decode_index=1
        )
        self.assertEqual(result, {self.CONTENT_KEY: ""})

    def test_decode_stream_shorter_full_text(self):
        """测试流式解码中 full_text 比 pre_text 短的情况"""
        self.tokenizer_mock.decode.side_effect = ["short", "longer text"]
        result = self.wrapper.decode(
            [1, 2, 3], False, False, False, True,
            curr_decode_index=2, prev_decode_index=1
        )
        self.assertEqual(result, {self.CONTENT_KEY: ""})

    def test_decode_with_skip_special_tokens(self):
        """测试带有 skip_special_tokens=True 的解码"""
        self.tokenizer_mock.decode.return_value = "decoded without special tokens"
        result = self.wrapper.decode([1, 2, 3], True, False, False, False)
        
        called_args = self.tokenizer_mock.decode.call_args
        self.assertTrue(called_args[1]['skip_special_tokens'])
        self.assertEqual(result, {self.CONTENT_KEY: "decoded without special tokens"})

    def test_decode_with_all_parameters(self):
        """测试解码方法的所有参数组合"""
        self.tokenizer_mock.decode.return_value = "full decode"
        result = self.wrapper.decode(
            [1, 2, 3], True, True, True, False,
            extra_param="test"
        )
        self.assertEqual(result, {self.CONTENT_KEY: "full decode"})

    def test_numpy_array_handling(self):
        """测试 numpy 数组处理的正确性"""
        token_ids = [1, 2, 3, 4, 5]
        self.tokenizer_mock.decode.return_value = "array test"
        
        result = self.wrapper.decode(token_ids, False, False, False, False)
        
        # 验证传递给 tokenizer.decode 的是 numpy 数组
        called_array = self.tokenizer_mock.decode.call_args[0][0]
        self.assertIsInstance(called_array, np.ndarray)
        self.assertTrue(np.array_equal(called_array, np.array(token_ids)))
        self.assertEqual(result, {self.CONTENT_KEY: "array test"})

    def test_stream_decode_numpy_slicing(self):
        """测试流式解码中 numpy 数组切片的正确性"""
        token_ids = [1, 2, 3, 4, 5]
        self.tokenizer_mock.decode.side_effect = ["slice test", "slice"]
        
        result = self.wrapper.decode(
            token_ids, False, False, False, True,
            curr_decode_index=4, prev_decode_index=2
        )
        
        # 验证切片操作
        calls = self.tokenizer_mock.decode.call_args_list
        self.assertEqual(len(calls), 2)
        
        # 第一次调用：input_tensor[2:]
        first_call_array = calls[0][0][0]
        self.assertTrue(np.array_equal(first_call_array, np.array([3, 4, 5])))
        
        # 第二次调用：input_tensor[2:4]
        second_call_array = calls[1][0][0]
        self.assertTrue(np.array_equal(second_call_array, np.array([3, 4])))
        
        self.assertEqual(result, {self.CONTENT_KEY: " test"})


if __name__ == '__main__':
    unittest.main()