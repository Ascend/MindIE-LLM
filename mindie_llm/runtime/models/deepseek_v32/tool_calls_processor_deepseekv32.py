# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Implement part of this file based on vllm-project/vllm
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import re
from typing import Dict, Pattern, Any
from ..base.tool_calls_processor import ToolCallsProcessorWithXml, \
DeltaFunctionCall, DeltaToolCall, ToolCallsProcessorManager


INIT_RETURN_NONE = {}

CONTENT = "content"
TOOL_CALLS = "tool_calls"
NAME = "name"
ARGUMENTS = "arguments"
STRING_TYPE = "string"


class ToolCallsProcessorDeepseekv3Base(ToolCallsProcessorWithXml):
    @property
    def tool_calls_start_token(self) -> str:
        return "<｜DSML｜function_calls>"

    @property
    def tool_calls_end_token(self) -> str:
        return "<｜tool▁calls▁end｜>"

    @property
    def tool_calls_start_token_id(self) -> int:
        return 128806

    @property
    def tool_calls_end_token_id(self) -> int:
        return 128807

    @property
    def tool_call_start_token(self) -> str:
        return "<｜tool▁call▁begin｜>"

    @property
    def tool_call_end_token(self) -> str:
        return "<｜tool▁call▁end｜>"

    @property
    def tool_call_start_token_id(self) -> int:
        return 128808

    @property
    def tool_call_end_token_id(self) -> int:
        return 128809

    @property
    def decode_spilt_token(self) -> int:
        return self.tool_calls_start_token

    @property
    def tool_call_regex(self) -> Pattern:
        raise NotImplementedError("Subclasses must implement the 'tool_call_regex' property.")

    @property
    def stream_tool_call_portion_regex(self) -> Pattern:
        raise NotImplementedError("Subclasses must implement the 'stream_tool_call_protion_regex' property.")

    @property
    def stream_tool_call_name_regex(self) -> Pattern:
        raise NotImplementedError("Subclasses must implement the 'stream_tool_call_name_regex' property.")

    @staticmethod
    def _get_tool_calls_json(matches):
        tool_calls = []
        try:
            for match in matches:
                name, arguments = match.values()
                tool_calls.append({"name": name, "arguments": arguments})
        except Exception:
            tool_calls = []
        return tool_calls

    def _preprocess_delta_text(self, delta_text):
        if self.tool_calls_start_token is not None:
            delta_text = delta_text.replace(self.tool_calls_start_token,
                                            "").replace(self.tool_call_end_token,
                                                        "")
        return delta_text

    def _decode_stream_tool_calls(self, tool_call_portion_dict):
        try:
            tool_call_portion = tool_call_portion_dict["tool_call_portion"]
            delta_text = tool_call_portion_dict["delta_text"]
            current_tool_call = self._get_current_tool_call_with_regex(tool_call_portion) if tool_call_portion else None
        except Exception:
            # Invalid JSON fragment newline characters.
            return INIT_RETURN_NONE

        # case1：send function name
        if not self.current_tool_name_sent:
            if current_tool_call is None or not current_tool_call.get(NAME):
                return INIT_RETURN_NONE
            self.current_tool_name_sent = True
            return {TOOL_CALLS: [
                DeltaToolCall(index=self.current_tool_id, type="function", id=self._random_tool_calls_id(),
                                function=DeltaFunctionCall(name=current_tool_call.get(NAME))).model_dump(
                    exclude_none=True)
            ]}

        delta = {}
        # case2：send param
        cur_arguments = current_tool_call.get(ARGUMENTS)
        if cur_arguments and not self.current_tool_arguments_sent:
            # case2-1:send arguments contains structure.example {"arguments":"{\"order_id\": \""}
            delta = {TOOL_CALLS: [
                DeltaToolCall(index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=cur_arguments).model_dump(exclude_none=True))
                .model_dump(exclude_none=True)
            ]}
            self.current_tool_arguments_sent = True
        elif cur_arguments and self.current_tool_arguments_sent:
            # case2-2:arguments delta content
            delta_arguments_text = _find_overlapping(cur_arguments, delta_text)
            delta = {TOOL_CALLS: [
                DeltaToolCall(index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=delta_arguments_text)).\
                                model_dump(exclude_none=True)
            ]}
        return delta

    def _get_current_tool_call_with_regex(self, tool_call_portion):
        current_tool_call = {}
        current_tool_call_matches = (
            self.stream_tool_call_portion_regex.match(
                tool_call_portion))
        if current_tool_call_matches:
            _, tool_name, tool_args = (
                current_tool_call_matches.groups())
            current_tool_call[NAME] = tool_name
            current_tool_call[ARGUMENTS] = tool_args
        else:
            current_tool_call_name_matches = (
                self.stream_tool_call_name_regex.match(
                    tool_call_portion))
            if current_tool_call_name_matches:
                _, tool_name = (
                    current_tool_call_name_matches.groups())
                current_tool_call[NAME] = tool_name
                current_tool_call[ARGUMENTS] = ""
        return current_tool_call


def _find_overlapping(str_a, str_b):
    max_possible = min(len(str_a), len(str_b))
    a_suffix_b_prefix = ""
    
    for length in range(max_possible, 0, -1):
        if str_a.endswith(str_b[:length]):
            a_suffix_b_prefix = str_b[:length]
            break

    return a_suffix_b_prefix


@ToolCallsProcessorManager.register_module(["deepseek_v2", "deepseek_v3", "deepseekv2", "deepseekv3"])
class ToolsCallProcessorDeepseekv3(ToolCallsProcessorDeepseekv3Base):
    @property
    def tool_call_regex(self) -> Pattern:
        return re.compile(
            r"<｜tool▁call▁begin｜>(?P<type>.*)<｜tool▁sep｜>(?P<function_name>.*)\n"
            r"```json\n(?P<function_arguments>.*)\n```<｜tool▁call▁end｜>"
        )

    @property
    def stream_tool_call_portion_regex(self) -> Pattern:
        return re.compile(
            r"(?P<type>.*)<｜tool▁sep｜>(?P<function_name>.*)\n```json\n(?P<function_arguments>.*[^\n`])"
        )

    @property
    def stream_tool_call_name_regex(self) -> Pattern:
        return re.compile(
            r"(?P<type>.*)<｜tool▁sep｜>(?P<function_name>.*)\n"
        )


@ToolCallsProcessorManager.register_module(["deepseek_v32", "deepseekv32"])
class ToolCallsProcessorDeepseekv32(ToolCallsProcessorDeepseekv3Base):
    """
    Parser for DeepSeek V3/3.2 XML-formatted tool calls.
    
    Features:
    - Snapshot-Diffing for seamless token-level streaming.
    - Schema-aware type coercion for nested/complex JSON parameters.
    - Hard cut-off mechanism to mitigate post-tool-call hallucinations.
    
    Note: DSML refers to the proprietary XML-like syntax (e.g., `<｜DSML｜function_calls>`) 
    used natively by DeepSeek V3/3.2 models for tool calling outputs.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tool_call_complete_regex = re.compile(
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>", re.DOTALL
        )
        self.invoke_complete_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter[^>]*?name="([^"]+)"[^>]*>(.*?)</｜DSML｜parameter>', re.DOTALL
        )

    def parse_tool_calls_v32(self, text: str) -> list[Dict[str, Any]]:
        """Non-streaming parameter extraction."""
        tool_call_match = self.tool_call_complete_regex.search(text)
        if not tool_call_match:
            return []
        
        tool_call_content = tool_call_match.group(1)
        tool_calls = []
        
        invoke_matches = self.invoke_complete_regex.findall(tool_call_content)
        for tool_name, parameter_content in invoke_matches:
            tool_info = {
                "name": tool_name,
                "parameters": {}
            }
            
            param_matches = self.parameter_complete_regex.findall(parameter_content)
            for param_name, param_value in param_matches:
                tool_info["parameters"][param_name.strip()] = param_value.strip()
            
            tool_calls.append(tool_info)
        
        return tool_calls

    def decode(self, content: str) -> Dict[str, Any]:
        """Non-streaming decode entry."""
        lines = content.strip()
        matches = self.parse_tool_calls_v32(lines)
    
        tool_calls = self._get_tool_calls_json(matches) if matches else None
        if not tool_calls:
            return {CONTENT: lines}
            
        call_res = []
        for item in tool_calls:
            tool_call = {
                NAME: item[NAME],
                ARGUMENTS: json.dumps(item[ARGUMENTS], ensure_ascii=False) \
                    if isinstance(item[ARGUMENTS], dict) else item[ARGUMENTS]
            }
            res = {
                "type": "function",
                "id": self._random_tool_calls_id(),
                "function": tool_call
            }
            call_res.append(res)
        spilt_token = self.decode_spilt_token
        return {CONTENT: content.split(spilt_token)[0], TOOL_CALLS: call_res}

    def decode_stream(self, all_token_ids: list[int], 
                      prev_decode_index: int, 
                      curr_decode_index: int,
                      skip_special_tokens: bool, 
                      delta_text: str) -> dict[str, Any]:
        """
        Stream decoding entry. 
        Intercepts XML tags and calculates tool call deltas.
        """
        try:
            full_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=skip_special_tokens)
            start_tag = "<｜DSML｜function_calls>"
            end_tag = "</｜DSML｜function_calls>"

            # Phase 1: Prefix interception (Drop partial start tags)
            if start_tag not in full_text:
                for i in range(1, len(start_tag)):
                    if full_text.endswith(start_tag[:i]):
                        return INIT_RETURN_NONE
                return {CONTENT: delta_text}

            tool_call_portion = full_text.split(start_tag)[-1]

            # Phase 2: Suffix interception (Hard Cut-off / Anti-Hallucination)
            # Permanently mutes the stream once the end tag appears to prevent 
            # the model from generating fake responses or reasoning post-call.
            if end_tag in tool_call_portion:
                return INIT_RETURN_NONE

            return self._parse_dsml_stream_xml(tool_call_portion, delta_text)

        except Exception:
            return {CONTENT: ""}

    def _get_param_type_from_schema(self, tool_name: str, param_name: str) -> str:
        """
        Extracts parameter type from injected self.tools schema.
        Defaults to 'string' if schema is unavailable or undefined.
        """
        if not getattr(self, "tools", None):
            return STRING_TYPE
            
        for tool in self.tools:
            # Handle both dict and object schema formats
            func = tool.get("function", {}) if isinstance(tool, dict) else getattr(tool, "function", None)
            if not func:
                continue
                
            t_name = func.get("name") if isinstance(func, dict) else getattr(func, "name", "")
            if t_name == tool_name:
                params = func.get("parameters", {}) if isinstance(func, dict) else getattr(func, "parameters", {})
                props = params.get("properties", {}) if isinstance(params, dict) else getattr(params, "properties", {})
                
                # Safely extract parameter definition
                if isinstance(props, dict):
                    param_info = props.get(param_name, {})
                else:
                    param_info = getattr(props, "get", lambda x, y: {})(param_name, {})
                    
                if isinstance(param_info, dict):
                    return param_info.get("type", STRING_TYPE)
                return STRING_TYPE
                
        return STRING_TYPE

    def _parse_dsml_stream_xml(self, xml_text: str, delta_text: str) -> dict[str, Any]:
        """State machine for XML stream."""
        invokes = xml_text.split("<｜DSML｜invoke")
        if len(invokes) < 2:
            return INIT_RETURN_NONE 

        current_invoke_xml = "<｜DSML｜invoke" + invokes[-1]
        tool_index = len(invokes) - 2  

        if getattr(self, "current_tool_id", -1) != tool_index:
            self.current_tool_id = tool_index
            self.current_tool_name_sent = False
            self.current_tool_arguments_sent = False

        if not self.current_tool_name_sent:
            name_match = re.search(r'name="([^"]+)"', current_invoke_xml)
            if not name_match:
                return INIT_RETURN_NONE
            self.current_tool_name_sent = True
            return {TOOL_CALLS: [
                DeltaToolCall(
                    index=self.current_tool_id,
                    type="function",
                    id=self._random_tool_calls_id(),
                    function=DeltaFunctionCall(name=name_match.group(1))
                ).model_dump(exclude_none=True)
            ]}

        if delta_text and current_invoke_xml.endswith(delta_text):
            prev_xml = current_invoke_xml[:-len(delta_text)]
        else:
            prev_xml = current_invoke_xml[:max(0, len(current_invoke_xml) - len(delta_text))]

        prev_json = self._convert_xml_to_json_string(prev_xml)
        curr_json = self._convert_xml_to_json_string(current_invoke_xml)

        delta_args = curr_json[len(prev_json):]

        if delta_args:
            if not self.current_tool_arguments_sent:
                delta_args = "{" + delta_args
                
            self.current_tool_arguments_sent = True
            return {TOOL_CALLS: [
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=delta_args)
                ).model_dump(exclude_none=True)
            ]}

        return INIT_RETURN_NONE

    def _convert_xml_to_json_string(self, xml_text: str) -> str:
        """
        Converts incomplete XML snapshots into valid JSON string fragments 
        utilizing injected schema for type precision.
        """
        if "<｜DSML｜invoke" not in xml_text:
            return ""

        tool_name_match = re.search(r'<｜DSML｜invoke\s+name="([^"]+)"', xml_text)
        tool_name = tool_name_match.group(1) if tool_name_match else ""

        stream_param_pattern = re.compile(
            r'<｜DSML｜parameter[^>]*?name="([^"]+)"[^>]*>(.*?)(?:</｜DSML｜parameter>|$)',
            re.DOTALL
        )
        matches = list(stream_param_pattern.finditer(xml_text))

        json_parts = []
        for match in matches:
            p_name = match.group(1)
            p_value = match.group(2)
            
            tag_str = match.group(0).split('>')[0]
            close_tag = "</｜DSML｜parameter>"
            is_closed = match.group(0).endswith(close_tag)

            if not is_closed:
                for i in range(1, len(close_tag)):
                    if p_value.endswith(close_tag[:i]):
                        p_value = p_value[:-i]
                        break

            schema_type = self._get_param_type_from_schema(tool_name, p_name)
            is_string_type = schema_type in [STRING_TYPE, "str"]

            if not getattr(self, "tools", None):
                if 'string="false"' in tag_str:
                    is_string_type = False
                elif 'string="true"' in tag_str:
                    is_string_type = True

            if is_string_type:
                escaped_val = p_value.replace('"', '\\"').replace('\n', '\\n')
                part = f'"{p_name}": "{escaped_val}'
                if is_closed:
                    part += '"'
            else:
                clean_val = p_value.strip()
                if not clean_val:
                    part = f'"{p_name}": '
                else:
                    part = f'"{p_name}": {clean_val}'
                
            json_parts.append(part)

        json_str = "{" + ", ".join(json_parts)
        if "</｜DSML｜invoke>" in xml_text:
            json_str += "}"
            
        return json_str
