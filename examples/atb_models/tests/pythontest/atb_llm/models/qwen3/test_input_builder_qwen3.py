# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from atb_llm.models.qwen3.input_builder_qwen3 import Qwen3InputBuilder
from atb_llm.models import TruncationSide

ROLE = "role"
CONTENT = "content"
INPUT_STR = "who are you"


class MockTokenizer:
    def __init__(self, use_fast=False, enable_thinking=None):
        self.use_fast = use_fast
        self.init_kwargs = {"enable_thinking": enable_thinking}
        self.chat_template = "test"

    def apply_chat_template(self, conversation, enable_thinking=False, tools=None, **kwargs):
        if tools:
            return [i for i in range(500)]
        max_length = kwargs.get("max_length", 200)
        # 修复：处理max_length为None的场景
        if max_length is None:
            max_length = 200
        if kwargs.get("truncation"):
            return [i for i in range(min(200, max_length))]
        return [i for i in range(200)]

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


class MockTokenizerNoChatTemplate:
    def __init__(self, use_fast=False):
        self.use_fast = use_fast
        self.init_kwargs = {}
        self.chat_template = "test"

    @classmethod
    def encode(cls, prompt, add_special_tokens):
        return prompt


class TestQwen3InputBuilder(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.input_builder = Qwen3InputBuilder(self.tokenizer)
        self.input_builder.chat_template_kwargs = "chat_template_kwargs"
        self.input_builder.truncation = "truncation"

    def test_apply_chat_template_default(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        chat_template_kwargs_think = {"enable_thinking": "true", "truncation": "1"}
        chat_template_kwargs_nothink = {"enable_thinking": "false"}
        kwargs_think = {self.input_builder.chat_template_kwargs: chat_template_kwargs_think}
        kwargs_nothink = {self.input_builder.chat_template_kwargs: chat_template_kwargs_nothink}
        
        user_prompt_think = self.input_builder._apply_chat_template(user_conversation,** kwargs_think)
        user_prompt_nothink = self.input_builder._apply_chat_template(user_conversation, **kwargs_nothink)
        
        self.assertIsNotNone(user_prompt_think)
        self.assertIsInstance(user_prompt_think, list)
        self.assertIsNotNone(user_prompt_nothink)
        self.assertIsInstance(user_prompt_nothink, list)

    def test_apply_chat_template_raise(self):
        tokenizer = MockTokenizerNoChatTemplate()
        input_builder = Qwen3InputBuilder(tokenizer)
        input_builder.chat_template_kwargs = "chat_template_kwargs"
        input_builder.truncation = "truncation"
        
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        with self.assertRaises(RuntimeError) as cm:
            input_builder._apply_chat_template(user_conversation)
        self.assertIn("transformers version is detected to be <4.34", str(cm.exception))

        tokenizer = MockTokenizer()
        tokenizer.chat_template = ""
        input_builder = Qwen3InputBuilder(tokenizer)
        input_builder.chat_template_kwargs = "chat_template_kwargs"
        input_builder.truncation = "truncation"
        with self.assertRaises(RuntimeError) as cm:
            input_builder._apply_chat_template(user_conversation)
        self.assertIn("it is not configured with a `chat_template`", str(cm.exception))

    def test_apply_chat_template_truncation_right(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        chat_template_kwargs = {
            "enable_thinking": True,
            "truncation": TruncationSide.RIGHT,
            "max_length": 50
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation,** kwargs)
        self.assertEqual(len(input_ids), 50)

    def test_apply_chat_template_truncation_left(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        chat_template_kwargs = {
            "enable_thinking": False,
            "truncation": TruncationSide.LEFT,
            "max_length": 50
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation, **kwargs)
        self.assertEqual(len(input_ids), 50)
        self.assertEqual(input_ids, [i for i in range(150, 200)])

    def test_apply_chat_template_with_tools(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        tools_msg = {
            "tools": [{"name": "code_interpreter", "description": "Execute code"}],
            "tool_choice": "code_interpreter"
        }
        chat_template_kwargs = {
            "enable_thinking": True,
            "truncation": TruncationSide.RIGHT,
            "max_length": 100
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(
            user_conversation, tools_msg=tools_msg,** kwargs
        )

    def test_apply_chat_template_tools_truncation_left(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        tools_msg = {
            "tools": [{"name": "code_interpreter", "description": "Execute code"}],
            "tool_choice": "code_interpreter"
        }
        chat_template_kwargs = {
            "enable_thinking": False,
            "truncation": TruncationSide.LEFT,
            "max_length": 100
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(
            user_conversation, tools_msg=tools_msg, **kwargs
        )
        self.assertEqual(len(input_ids), 100)

    def test_apply_chat_template_config_enable_thinking(self):
        tokenizer = MockTokenizer(enable_thinking=False)
        input_builder = Qwen3InputBuilder(tokenizer)
        input_builder.chat_template_kwargs = "chat_template_kwargs"
        input_builder.truncation = "truncation"
        
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        kwargs = {input_builder.chat_template_kwargs: {"truncation": TruncationSide.RIGHT}}
        
        input_ids = input_builder._apply_chat_template(user_conversation,** kwargs)
        self.assertEqual(len(input_ids), 200)

    def test_apply_chat_template_no_max_length(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR * 10}]
        chat_template_kwargs = {
            "enable_thinking": True,
            "truncation": TruncationSide.RIGHT
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation, **kwargs)
        self.assertEqual(len(input_ids), 200)

    def test_apply_chat_template_empty_content(self):
        user_conversation = [{ROLE: "user", CONTENT: None}]
        chat_template_kwargs = {"enable_thinking": True}
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation,** kwargs)
        self.assertIsInstance(input_ids, list)

    def test_apply_chat_template_truncation_left_no_truncate(self):
        user_conversation = [{ROLE: "user", CONTENT: INPUT_STR}]
        chat_template_kwargs = {
            "enable_thinking": False,
            "truncation": TruncationSide.LEFT,
            "max_length": 300
        }
        kwargs = {self.input_builder.chat_template_kwargs: chat_template_kwargs}
        
        input_ids = self.input_builder._apply_chat_template(user_conversation, **kwargs)
        self.assertEqual(len(input_ids), 200)


if __name__ == '__main__':
    unittest.main()
