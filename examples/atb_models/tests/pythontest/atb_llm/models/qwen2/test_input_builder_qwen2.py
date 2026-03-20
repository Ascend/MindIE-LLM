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
import copy
from typing import List, Dict, Any
from enum import Enum
import json

from atb_llm.models.qwen2.input_builder_qwen2 import (
    Qwen2InputBuilder,
    Message,
    ContentItem,
    Function,
    ToolCall,
    QwenFnCallPrompt,
    get_function_description,
    extract_text_from_message,
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL,
    ZH,
    EN,
    AUTO,
    FN_NAME,
    FN_EXIT
)
from atb_llm.models.base.input_builder import InputBuilder


class TruncationSide(int, Enum):
    DISABLE = 0
    LEFT = 1
    RIGHT = -1


class MockTokenizer:
    def __init__(self):
        self.chat_template = "mock_chat_template"
    
    def apply_chat_template(self, conversation: List[Dict],** kwargs) -> List[int]:
        tools = kwargs.get("tools", None)
        if tools:
            return [i for i in range(300)]
        return [i for i in range(200)]


class TestQwen2InputBuilder(unittest.TestCase):
    def setUp(self):
        # 每次测试都重新初始化全新的MockTokenizer，彻底避免属性污染
        self.mock_tokenizer = MockTokenizer()
        self.qwen2_builder = Qwen2InputBuilder(
            tokenizer=self.mock_tokenizer,
            is_qwen1_5_or_2=True
        )
        self.qwen2_builder.truncation = "truncation"
        self.qwen2_builder.chat_template_kwargs = "chat_template_kwargs"
        
        self.qwen25_builder = Qwen2InputBuilder(
            tokenizer=MockTokenizer(),
            is_qwen1_5_or_2=False
        )
        self.qwen25_builder.truncation = "truncation"
        self.qwen25_builder.chat_template_kwargs = "chat_template_kwargs"

        self.tools_msg = {
            "tools": [
                {
                    "function": {
                        "name": "code_interpreter",
                        "name_for_human": "Code Interpreter",
                        "name_for_model": "code_interpreter",
                        "description": "Execute Python code",
                        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}
                    }
                }
            ],
            "tool_choice": "code_interpreter"
        }

    def test_apply_chat_template_basic(self):
        conversation = [{"role": USER, "content": "Test"}]
        kwargs = {
            self.qwen2_builder.chat_template_kwargs: {
                self.qwen2_builder.truncation: TruncationSide.RIGHT,
                "max_length": 100
            }
        }
        input_ids = self.qwen2_builder._apply_chat_template(conversation, **kwargs)
        self.assertIsInstance(input_ids, list)
        self.assertEqual(len(input_ids), 200)

    def test_truncation_right(self):
        conversation = [{"role": USER, "content": "Test right truncation" * 100}]
        max_length = 50
        kwargs = {
            self.qwen2_builder.chat_template_kwargs: {
                self.qwen2_builder.truncation: TruncationSide.RIGHT,
                "max_length": max_length
            }
        }
        input_ids = self.qwen2_builder._apply_chat_template(conversation,** kwargs)
        self.assertEqual(len(input_ids), 200, "Length remains unchanged after right truncation (handled by tokenizer)")

    def test_truncation_left(self):
        conversation = [{"role": USER, "content": "Test left truncation" * 100}]
        max_length = 50
        kwargs = {
            self.qwen2_builder.chat_template_kwargs: {
                self.qwen2_builder.truncation: TruncationSide.LEFT,
                "max_length": max_length
            }
        }
        input_ids = self.qwen2_builder._apply_chat_template(conversation, **kwargs)
        self.assertEqual(len(input_ids), max_length, "Length equals max_length after left truncation")

    def test_truncation_disable(self):
        conversation = [{"role": USER, "content": "Test disable truncation" * 100}]
        kwargs = {
            self.qwen2_builder.chat_template_kwargs: {
                self.qwen2_builder.truncation: TruncationSide.DISABLE,
                "max_length": 50
            }
        }
        input_ids = self.qwen2_builder._apply_chat_template(conversation,** kwargs)
        self.assertEqual(len(input_ids), 200, "Length remains unchanged when truncation is disabled")

    def test_apply_chat_template_qwen25(self):
        conversation = [{"role": USER, "content": "Qwen2.5 test"}]
        kwargs = {
            self.qwen25_builder.chat_template_kwargs: {
                self.qwen25_builder.truncation: TruncationSide.LEFT,
                "max_length": 100
            }
        }
        input_ids = self.qwen25_builder._apply_chat_template(conversation, **kwargs)
        self.assertEqual(len(input_ids), 100)
        
        input_ids_with_tools = self.qwen25_builder._apply_chat_template(
            conversation, tools_msg=self.tools_msg,** kwargs
        )
        self.assertEqual(len(input_ids_with_tools), 100)

    def test_apply_chat_template_with_tools(self):
        conversation = [
            {"role": SYSTEM, "content": "You are an assistant"},
            {"role": USER, "content": "Execute 1+1"},
            {"role": ASSISTANT, "content": "", "tool_calls": [ToolCall(function=Function(name="code_interpreter", arguments="1+1"))]},
            {"role": TOOL, "content": "2"}
        ]
        kwargs = {
            self.qwen2_builder.chat_template_kwargs: {
                self.qwen2_builder.truncation: TruncationSide.LEFT,
                "max_length": 150
            }
        }
        input_ids = self.qwen2_builder._apply_chat_template(
            conversation, tools_msg=self.tools_msg, **kwargs
        )
        self.assertEqual(len(input_ids), 150)

    def test_simulate_response_completion_with_chat(self):
        messages = [
            Message(role=USER, content="Hello"),
            Message(role=ASSISTANT, content="Hi!")
        ]
        result = self.qwen2_builder.simulate_response_completion_with_chat(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], USER)
        self.assertIn("Hello\n\nHi!", result[0]["content"])

        messages_list = [
            Message(role=USER, content=[ContentItem(text="Hello")]),
            Message(role=ASSISTANT, content=[ContentItem(text="Hi!")])
        ]
        result_list = self.qwen2_builder.simulate_response_completion_with_chat(messages_list)
        self.assertEqual(len(result_list), 1)
        self.assertEqual(len(result_list[0]["content"]), 3)

        messages_no_assistant = [
            Message(role=USER, content="Test"),
            Message(role=USER, content="Test2")
        ]
        result_no_assistant = self.qwen2_builder.simulate_response_completion_with_chat(messages_no_assistant)
        self.assertEqual(len(result_no_assistant), 2)

    def test_chat_template_not_exist(self):
        self.mock_tokenizer.chat_template = None
        with self.assertRaises(RuntimeError) as exc:
            self.qwen2_builder._apply_chat_template([{"role": USER, "content": "Test"}])
        self.assertIn("chat_template", str(exc.exception))

    def test_content_item_all_types(self):
        text_item = ContentItem(text="Test text")
        self.assertEqual(text_item.type, "text")
        self.assertEqual(text_item.value, "Test text")

        image_item = ContentItem(image="https://test.png")
        self.assertEqual(image_item.type, "image")
        self.assertEqual(image_item.value, "https://test.png")

        file_item = ContentItem(file="test.txt")
        self.assertEqual(file_item.type, "file")
        self.assertEqual(file_item.value, "test.txt")

        with self.assertRaises(ValueError) as exc:
            ContentItem(text="Test", image="https://test.png")
        self.assertIn("Exactly one of 'text', 'image', or 'file' must be provided", str(exc.exception))

        with self.assertRaises(ValueError) as exc:
            ContentItem()
        self.assertIn("Exactly one of 'text', 'image', or 'file' must be provided", str(exc.exception))

    def test_message_model_all_fields(self):
        msg_basic = Message(role=USER, content="Test")
        self.assertEqual(msg_basic.role, USER)
        self.assertEqual(msg_basic.get("non_exist", "default"), "default")

        func = Function(name="test_func", arguments='{"a":1}')
        msg_func = Message(role=ASSISTANT, content="", function_call=func)
        self.assertEqual(msg_func.function_call.name, "test_func")
        self.assertEqual(msg_func.function_call.arguments, '{"a":1}')

        tool_call = ToolCall(function=func)
        msg_tool = Message(role=ASSISTANT, content="", tool_calls=[tool_call])
        self.assertEqual(len(msg_tool.tool_calls), 1)
        self.assertEqual(msg_tool.tool_calls[0].function.name, "test_func")

        msg_extra = Message(role=USER, content="Test", extra={"key": "value"})
        self.assertEqual(msg_extra.extra["key"], "value")

        msg_empty = Message(role=USER)
        self.assertEqual(msg_empty.content, "")

    def test_function_toolcall_model(self):
        func = Function(name="code_interpreter", arguments="print(1+1)")
        self.assertEqual(func.name, "code_interpreter")
        self.assertEqual(func.arguments, "print(1+1)")
        self.assertIn("code_interpreter", repr(func))

        tool_call = ToolCall(function=func)
        self.assertEqual(tool_call.function.name, "code_interpreter")
        self.assertIn("code_interpreter", repr(tool_call))

    def test_get_function_description(self):
        func_zh = {
            "function": {
                "name": "code_interpreter",
                "name_for_human": "Code Interpreter",
                "name_for_model": "code_interpreter",
                "description": "Execute Python code",
                "parameters": {"type": "object"}
            }
        }
        desc_zh = get_function_description(func_zh, ZH)
        self.assertIn("### Code Interpreter", desc_zh)
        self.assertIn("此工具的输入应为Markdown代码块。", desc_zh)

        func_en = {
            "function": {
                "name": "code_interpreter",
                "name_for_human": "Code Interpreter",
                "name_for_model": "code_interpreter",
                "description": "Execute Python code",
                "parameters": {"type": "object"}
            }
        }
        desc_en = get_function_description(func_en, EN)
        self.assertIn("### Code Interpreter", desc_en)
        self.assertIn("Enclose the code within triple backticks (`) at the beginning and end of the code.", desc_en)

    def test_extract_text_from_message(self):
        msg_str = Message(role=USER, content="Test text")
        self.assertEqual(extract_text_from_message(msg_str), "Test text")

        msg_list = Message(role=USER, content=[ContentItem(text="Test")])
        with self.assertRaises(TypeError):
            extract_text_from_message(msg_list)

    def test_qwen_fncall_prompt_preprocess(self):
        messages = [
            Message(role=USER, content="Execute 1+1"),
            Message(role=ASSISTANT, content="", tool_calls=[ToolCall(function=Function(name="code_interpreter", arguments="1+1"))]),
            Message(role=TOOL, content="2")
        ]
        functions = self.tools_msg["tools"]
        
        processed_zh = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=messages,
            functions=functions,
            lang=ZH,
            parallel_function_calls=False,
            function_choice=AUTO
        )
        self.assertEqual(len(processed_zh), 3)
        self.assertIn("## 你拥有如下工具", processed_zh[0].content[0].text)

        processed_en_parallel = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=messages,
            functions=functions,
            lang=EN,
            parallel_function_calls=True,
            function_choice="code_interpreter"
        )
        self.assertEqual(len(processed_en_parallel), 3)
        self.assertIn("## You have access to the following tools", processed_en_parallel[0].content[0].text)
        self.assertEqual(processed_en_parallel[-1].content, f"{FN_NAME}: code_interpreter")

        processed_none = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=messages,
            functions=functions,
            lang=ZH,
            parallel_function_calls=False,
            function_choice="none"
        )
        self.assertEqual(len(processed_none), 3)


class TestBaseModelCompatibleDict(unittest.TestCase):
    def test_base_model_compatible_dict(self):
        msg = Message(role=USER, content="Test")
        
        self.assertEqual(msg["role"], USER)
        msg["content"] = "New content"
        self.assertEqual(msg.content, "New content")

        dump = msg.model_dump()
        self.assertEqual(dump["role"], USER)
        dump_json = msg.model_dump_json()
        self.assertEqual(json.loads(dump_json)["content"], "New content")

        self.assertIn("role", str(msg))
        self.assertIn("Message", repr(msg))

        self.assertEqual(msg.get("role"), USER)
        self.assertEqual(msg.get("non_exist", "default"), "default")


if __name__ == '__main__':
    unittest.main(verbosity=2)