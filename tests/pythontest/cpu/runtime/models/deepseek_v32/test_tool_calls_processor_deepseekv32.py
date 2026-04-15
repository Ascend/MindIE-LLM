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
import json

from mindie_llm.runtime.models.deepseek_v32.tool_calls_processor_deepseekv32 import (
    INIT_RETURN_NONE,
    TOOL_CALLS,
    NAME,
    ARGUMENTS,
    ToolCallsProcessorDeepseekv32,
)


class MockTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(token_ids)


class TestToolCallsProcessorDeepseekV32(unittest.TestCase):

    def setUp(self):
        """Initializes the processor and injects all required schemas for stream & non-stream tests."""
        self.processor = ToolCallsProcessorDeepseekv32(tokenizer=MockTokenizer())
        self.processor.tools = [
            # --- Stream Test Schemas ---
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {"properties": {"city": {"type": "string"}}}
                }
            },
            {
                "function": {
                    "name": "update_user",
                    "parameters": {"properties": {"user_data": {"type": "object"}}}
                }
            },
            {
                "function": {
                    "name": "execute_script",
                    "parameters": {"properties": {"script": {"type": "string"}}}
                }
            },
            {
                "function": {
                    "name": "calculator",
                    "parameters": {"properties": {"formula": {"type": "string"}}}
                }
            },
            # --- Non-Stream Type Conversion Schemas ---
            {
                "function": {
                    "name": "calculate_area",
                    "parameters": {
                        "properties": {
                            "base": {"type": "integer"},
                            "height": {"type": "integer"},
                            "is_triangle": {"type": "boolean"},
                            "unit": {"type": "string"}
                        }
                    }
                }
            },
            {
                "function": {
                    "name": "complex_task",
                    "parameters": {
                        "properties": {
                            "config": {"type": "object"}
                        }
                    }
                }
            }
        ]

    # =========================================================================
    # Helper Methods
    # =========================================================================
    def _simulate_stream(self, chunks: list) -> str:
        """Helper method to simulate stream arrival and aggregate JSON deltas."""
        accumulated_xml = ""
        emitted_deltas = []
        
        self.processor.current_tool_id = 0
        self.processor.current_tool_name_sent = False
        self.processor.current_tool_arguments_sent = False
        
        for chunk in chunks:
            accumulated_xml += chunk
            res = self.processor._parse_dsml_stream_xml(accumulated_xml, chunk)
            
            if res and res != INIT_RETURN_NONE and TOOL_CALLS in res:
                for tc in res[TOOL_CALLS]:
                    if "arguments" in tc.get("function", {}):
                        emitted_deltas.append(tc["function"]["arguments"])
                        
        return "".join(emitted_deltas)

    # =========================================================================
    # Streaming Tests (Snapshot-Diffing & Corner Cases)
    # =========================================================================
    def test_stream_empty_parameters_new_format(self):
        chunks = [
            '<invoke name="get_current_time">',
            '</invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{}')

    def test_stream_nested_dict_injection_legacy_format(self):
        chunks = [
            '<｜DSML｜invoke name="update_user">\n',
            '<｜DSML｜parameter name="user_data">',
            '{"name": "Alice",',
            ' "age": 18}</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{"user_data": {"name": "Alice", "age": 18}}')

    def test_stream_unescaped_quotes_and_newlines_new_format(self):
        chunks = [
            '<invoke name="execute_script">\n',
            '<parameter name="script">',
            '```python\n',
            'print("Hello")\n',
            '```\n',
            '</parameter>\n',
            '</invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        expected_content = '```python\\nprint(\\"Hello\\")\\n```\\n'
        self.assertIn(expected_content, final_json)

    def test_stream_attribute_reordering(self):
        chunks = [
            '<invoke name="get_weather">\n',
            '<parameter \n  string="true"   name="city"  >',
            'Tokyo',
            '</parameter>\n',
            '</invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{"city": "Tokyo"}')

    def test_stream_single_character_drip_mixed(self):
        full_xml = (
            '<｜DSML｜invoke name="calculator">\n'
            '<parameter name="formula">1+1</parameter>\n'
            '</｜DSML｜invoke>'
        )
        chunks = list(full_xml)
        
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{"formula": "1+1"}')

    def test_stream_multiple_invocations_isolation_dual_mode(self):
        chunks = [
            '<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter name="city">London</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>\n',
            '<invoke name="calculator">\n',
            '<parameter name="formula">2+2</parameter>\n',
            '</invoke>'
        ]
        
        accumulated_xml = ""
        deltas_tool_0 = []
        deltas_tool_1 = []
        
        self.processor.current_tool_id = -1
        
        for chunk in chunks:
            accumulated_xml += chunk
            res = self.processor._parse_dsml_stream_xml(accumulated_xml, chunk)
            
            if res and res != INIT_RETURN_NONE and TOOL_CALLS in res:
                tc = res[TOOL_CALLS][0]
                if "arguments" in tc.get("function", {}):
                    if tc["index"] == 0:
                        deltas_tool_0.append(tc["function"]["arguments"])
                    elif tc["index"] == 1:
                        deltas_tool_1.append(tc["function"]["arguments"])
                        
        json_0 = "".join(deltas_tool_0)
        json_1 = "".join(deltas_tool_1)
        
        self.assertEqual(json_0, '{"city": "London"}')
        self.assertEqual(json_1, '{"formula": "2+2"}')

    # =========================================================================
    # Non-Streaming Tests (Schema-Aware Type Coercion)
    # =========================================================================
    def test_non_streaming_type_conversion(self):
        """Tests if decode correctly converts int/bool based on injected schema."""
        content = (
            '<function_calls>\n'
            '<invoke name="calculate_area">\n'
            '<parameter name="base">10</parameter>\n'
            '<parameter name="height">5</parameter>\n'
            '<parameter name="is_triangle">true</parameter>\n'
            '<parameter name="unit">cm</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        
        result = self.processor.decode(content)
        
        self.assertIn(TOOL_CALLS, result)
        tool_call = result[TOOL_CALLS][0]["function"]
        self.assertEqual(tool_call[NAME], "calculate_area")
        
        args_dict = json.loads(tool_call[ARGUMENTS])
        self.assertIsInstance(args_dict["base"], int)
        self.assertEqual(args_dict["base"], 10)
        self.assertIsInstance(args_dict["is_triangle"], bool)
        self.assertTrue(args_dict["is_triangle"])
        self.assertIsInstance(args_dict["unit"], str)
        self.assertEqual(args_dict["unit"], "cm")

    def test_non_streaming_object_type(self):
        """Tests handling of raw JSON objects in non-streaming decoding."""
        content = (
            '<function_calls>\n'
            '<invoke name="complex_task">\n'
            '<parameter name="config">{"timeout": 30, "retry": true}</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        
        result = self.processor.decode(content)
        tool_call = result[TOOL_CALLS][0]["function"]
        args_dict = json.loads(tool_call[ARGUMENTS])
        
        self.assertEqual(args_dict["config"]["timeout"], 30)
        self.assertTrue(args_dict["config"]["retry"])

    def test_non_streaming_fallback_to_string(self):
        """Tests fallback mechanism when schema is completely missing."""
        self.processor.tools = None
        content = (
            '<function_calls>\n'
            '<invoke name="unknown_tool">\n'
            '<parameter name="val">123</parameter>\n'
            '</invoke>\n'
            '</function_calls>'
        )
        
        result = self.processor.decode(content)
        args_dict = json.loads(result[TOOL_CALLS][0]["function"][ARGUMENTS])
        
        self.assertIsInstance(args_dict["val"], str)
        self.assertEqual(args_dict["val"], "123")


if __name__ == '__main__':
    unittest.main()