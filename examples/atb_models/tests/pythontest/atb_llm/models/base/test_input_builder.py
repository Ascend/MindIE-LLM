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
from unittest.mock import Mock, MagicMock, patch, call
from copy import deepcopy
import json

from atb_llm.models.base.input_builder import InputBuilder


class FakeTokenizer:
    def __init__(self, chat_template=""):
        self.chat_template = chat_template if chat_template else "<fake_chat_template>"

    def encode(self, text):
        return [ord(c) for c in text]  # Limit to avoid huge numbers
        
    def decode(self, token_ids):
        try:
            return ''.join([chr(id_) for id_ in token_ids if 32 <= id_ <= 126])
        except (ValueError, TypeError):
            return "decoded_text"


class FakeTokenizerWithChatTemplate(FakeTokenizer):
    def __init__(self, chat_template=""):
        super().__init__(chat_template)

    def apply_chat_template(self, conversation, add_generation_prompt=True, **kwargs):
        result = self.chat_template
        for turn in conversation:
            content = turn.get('content', '')
            result += content
            
        if 'tools_msg' in kwargs:
            tools_content = kwargs['tools_msg'].get('tools', '')
            result += json.dumps(tools_content)

        if add_generation_prompt:
            result += "<generation_prompt>"

        if kwargs.get("tokenize", True):
            result = self.encode(result)

        return result
        

class TestInputBuilder(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = FakeTokenizerWithChatTemplate()
        self.max_length = 200
        self.input_builder = InputBuilder(
            tokenizer=self.mock_tokenizer,
            max_length=self.max_length
        )
    
    def test_init(self):
        """init with default parameters."""
        tokenizer = FakeTokenizer()
        builder = InputBuilder(tokenizer)
        
        self.assertEqual(builder.tokenizer, tokenizer)
        self.assertEqual(builder.system_role_name, "system")
        self.assertEqual(builder.user_role_name, "user")
        self.assertEqual(builder.tool_call_name, "tools_call")
        self.assertEqual(builder.role_key, "role")
        self.assertEqual(builder.max_length, 2048)
        self.assertEqual(builder.rank, 0)
    
        """init with custom parameters."""
        tokenizer = FakeTokenizer()
        custom_template = "custom template"
        builder = InputBuilder(
            tokenizer=tokenizer,
            chat_template=custom_template,
            system_role_name="sys",
            user_role_name="usr",
            max_length=512
        )
        
        self.assertEqual(builder.tokenizer, tokenizer)
        self.assertEqual(builder.system_role_name, "sys")
        self.assertEqual(builder.user_role_name, "usr")
        self.assertEqual(builder.max_length, 512)
        self.assertEqual(builder.tokenizer.chat_template, custom_template)
        
    def test_generate_position_ids(self):
        input_ids = [1, 2, 3, 4, 5]  # Use list instead of numpy array
        position_ids = InputBuilder.generate_position_ids(input_ids)
        
        expected = list(range(len(input_ids)))
        self.assertEqual(list(position_ids), expected)
        
    @patch('atb_llm.models.base.input_builder.print_log')
    def test_make_context(self, mock_print_log):
        """Test make_context with simple conversation."""
        conversation = [
            {"role": "user", "content": "Hello"}
        ]
        desired_result_str = f"<fake_chat_template>Hello<generation_prompt>"
        desired_result = self.mock_tokenizer.encode(desired_result_str)
        
        result = self.input_builder.make_context(
            rank=0, 
            conversation=conversation,
            add_generation_prompt=True
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(result, desired_result)
    
        """Test make_context with tools_call message."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "tools_call", "tools": {"name": "fake_func"}}
        ]
        desired_result_str = f"<fake_chat_template>Hello{json.dumps(conversation[1].get('tools'))}"
        desired_result = self.mock_tokenizer.encode(desired_result_str)
        
        result = self.input_builder.make_context(
            rank=0,
            conversation=conversation,
            add_generation_prompt=False,
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(result, desired_result)
    
        """Test make_context with empty conversation."""
        conversation = []
        
        with self.assertRaises(ValueError) as cm:
            self.input_builder.make_context(
                rank=0,
                conversation=conversation,
                adapt_to_max_length=True
            )
        
        self.assertIn("The conversation is empty!", str(cm.exception))
    
        """Test make_context with only system message."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"}
        ]
        
        with self.assertRaises(ValueError) as cm:
            self.input_builder.make_context(
                rank=0,
                conversation=conversation,
                adapt_to_max_length=True
            )
        
        self.assertIn("There is not any queries in the conversation", str(cm.exception))
    
        """Test make_context with adapt_to_max_length."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        result = self.input_builder.make_context(
            rank=0,
            conversation=conversation,
            adapt_to_max_length=True
        )
        
        self.assertIsInstance(result, list)
        mock_print_log.assert_called()
    
        """Test make_context with adapt_to_max_length and last query is not from user."""
        conversation = [
            {"role": "assistant", "content": "Hello there"}
        ]
        
        result = self.input_builder.make_context(
            rank=0,
            conversation=conversation,
            adapt_to_max_length=True
        )
        
        self.assertIsInstance(result, list)
        # Check that warning was logged
        warning_calls = [call for call in mock_print_log.call_args_list 
                        if len(call[0]) > 2 and "not offered by user" in str(call[0][2])]
        self.assertGreater(len(warning_calls), 0)
    
        """Test make_context with content exceeding max length."""
        long_content = "a" * 200
        conversation = [
            {"role": "user", "content": long_content}
        ]
        
        ori_max_length = self.input_builder.max_length
        self.input_builder.max_length = 5
        
        result = self.input_builder.make_context(
            rank=0,
            conversation=conversation,
            adapt_to_max_length=True
        )
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), self.input_builder.max_length)
        
        warning_calls = [call for call in mock_print_log.call_args_list 
                        if len(call[0]) > 2 and "has been truncated" in str(call[0][2])]
        self.assertGreater(len(warning_calls), 0)
        self.input_builder.max_length = ori_max_length
    
        """Test make_context with multi-turn conversation."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi, howw are you?"},
            {"role": "assistant", "content": "I'm fine. Thank you. And you?"},
            {"role": "user", "content": "It's good to hear that. I'm doing well"},
            {"role": "assistant", "content": "Good. Have a nice day!"},
            {"role": "user", "content": "Thanks."},
        ]
        
        result = self.input_builder.make_context(
            rank=0,
            conversation=conversation,
            adapt_to_max_length=True
        )

        self.assertIsInstance(result, list)
        mock_print_log.assert_called()
    
    def test_apply_chat_template(self):
        """Test _apply_chat_template when tokenizer lacks apply_chat_template."""
        tokenizer = FakeTokenizer()
        builder = InputBuilder(tokenizer)
        conversation = [{"role": "user", "content": "Hello"}]
        
        with self.assertRaises(RuntimeError) as cm:
            builder._apply_chat_template(conversation)
        
        self.assertIn("transformers version is detected to be <4.34", str(cm.exception))
        """Test _apply_chat_template when tokenizer has no chat_template."""
        tokenizer = FakeTokenizerWithChatTemplate()
        tokenizer.chat_template = ""
        builder = InputBuilder(tokenizer)
        
        with self.assertRaises(RuntimeError) as cm:
            builder._apply_chat_template(conversation)
        
        self.assertIn("it is not configured with a `chat_template`", str(cm.exception))

        kwargs = {"truncation": 0}
        with self.assertRaises(RuntimeError) as cm:
            builder._apply_chat_template(conversation,** kwargs)
        
        self.assertIn("it is not configured with a `chat_template`", str(cm.exception))

        """Test _apply_chat_template with TruncationSide.RIGHT (default)"""
        tokenizer = FakeTokenizerWithChatTemplate(chat_template="<test_template>")
        builder = InputBuilder(tokenizer)
        conversation = [{"role": "user", "content": "test right truncation"}]
        kwargs = {"add_generation_prompt": False}
        
        input_ids = builder._apply_chat_template(conversation, **kwargs)
        
        expected_ids = tokenizer.encode("<test_template>test right truncation")
        self.assertEqual(input_ids, expected_ids)

        """Test _apply_chat_template with TruncationSide.LEFT + exceed max_length"""
        long_content = "a" * 10
        conversation = [{"role": "user", "content": long_content}]
        max_length = 5
        kwargs = {
            "truncation": 1,
            "max_length": max_length,
            "add_generation_prompt": False
        }
        
        input_ids = builder._apply_chat_template(conversation, **kwargs)
        tokenizer.encode("<test_template>" + long_content)
    

if __name__ == '__main__':
    unittest.main()