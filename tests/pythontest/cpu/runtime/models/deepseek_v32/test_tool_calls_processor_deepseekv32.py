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

from mindie_llm.runtime.models.deepseek_v32.tool_calls_processor_deepseekv32 import (
    INIT_RETURN_NONE,
    TOOL_CALLS,
    ToolCallsProcessorDeepseekv32,
)


class MockTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(token_ids)


class TestToolCallsProcessorStreamCornerCases(unittest.TestCase):

    def setUp(self):
        """Initializes the processor and injects testing schemas."""
        self.processor = ToolCallsProcessorDeepseekv32(tokenizer=MockTokenizer())
        self.processor.tools = [
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
            }
        ]

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

    def test_stream_empty_parameters(self):
        """Tests tool invocation with no parameters."""
        chunks = [
            '<｜DSML｜invoke name="get_current_time">',
            '</｜DSML｜invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{}')

    def test_stream_nested_dict_injection(self):
        """Tests parameter extraction when value is a raw JSON object."""
        chunks = [
            '<｜DSML｜invoke name="update_user">\n',
            '<｜DSML｜parameter name="user_data">',
            '{"name": "Alice",',
            ' "age": 18}</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{"user_data": {"name": "Alice", "age": 18}}')

    def test_stream_unescaped_quotes_and_newlines(self):
        """Tests automatic escaping of newlines and quotes within string parameters."""
        chunks = [
            '<｜DSML｜invoke name="execute_script">\n',
            '<｜DSML｜parameter name="script">',
            '```python\n',
            'print("Hello")\n',
            '```\n',
            '</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        expected_content = '```python\\nprint(\\"Hello\\")\\n```\\n'
        self.assertIn(expected_content, final_json)

    def test_stream_attribute_reordering(self):
        """Tests parsing tolerance when XML attributes are unordered or padded."""
        chunks = [
            '<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter \n  string="true"   name="city"  >',
            'Tokyo',
            '</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>'
        ]
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{"city": "Tokyo"}')

    def test_stream_single_character_drip(self):
        """Tests snapshot-diffing stability under extreme token fragmentation."""
        full_xml = (
            '<｜DSML｜invoke name="calculator">\n'
            '<｜DSML｜parameter name="formula">1+1</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>'
        )
        chunks = list(full_xml)
        
        final_json = self._simulate_stream(chunks)
        self.assertEqual(final_json, '{"formula": "1+1"}')

    def test_stream_multiple_invocations_isolation(self):
        """Tests state reset and data isolation during consecutive tool calls."""
        chunks = [
            '<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter name="city">London</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>\n',
            '<｜DSML｜invoke name="calculator">\n',
            '<｜DSML｜parameter name="formula">2+2</｜DSML｜parameter>\n',
            '</｜DSML｜invoke>'
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

if __name__ == '__main__':
    unittest.main()