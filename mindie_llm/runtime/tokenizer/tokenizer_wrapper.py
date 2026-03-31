# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from enum import Enum
import json

from ..models import get_router_ins
from ..config.load_config import LoadConfig
from ...utils.log.logging import logger


REASON_CONTENT_KEY = "reasoning_content"
CONTENT_KEY = "content"
METADATA_KEY = "metadata"


class TruncationSide(int, Enum):
    DISABLE = 0
    LEFT = 1
    RIGHT = -1


class TokenizerWrapper:
    """A class for the upper layer to call the model's customized tokenizer.

    This class provides objects such as the model's configuration, tokenizer, and input builder, etc. The
    `input_builder` can assemble the prompt according to the chat template, and is a core function of the chat service
    interface.

    Args:
        model_name_or_path: The model weight path or model identifier.
    """

    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        model_dict_key = "models_dict"
        if model_dict_key in kwargs.keys():
            kwargs[model_dict_key] = None if kwargs[model_dict_key] == '' else json.loads(kwargs[model_dict_key])
        load_config = LoadConfig(
            model_name_or_path=model_name_or_path,
            models_dict=kwargs[model_dict_key]
        )
        router_ins = get_router_ins(load_config)
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.tool_calls_processor = router_ins.tool_calls_processor
        self.reasoning_parser = router_ins.reasoning_parser
        self.llm_config = router_ins.llm_config
        self.enable_thinking = self.tokenizer.init_kwargs.get("enable_thinking", True)
        
    def encode(self, inputs, **kwargs) -> list[int]:
        """Encodes input text or conversation into token IDs.

        Args:
            inputs: Either a raw string or a list of chat messages (when is_chatting=True).
            **kwargs: Additional arguments passed to the tokenizer or input builder.

        Returns:
            List[int]: List of token IDs representing the encoded input.
        """
        is_chatting = kwargs.pop("is_chatting", False)
        if is_chatting:
            token_ids = self.input_builder.make_context(0, inputs, **kwargs)
        else:
            truncation_method = kwargs.pop("truncation", TruncationSide.RIGHT)
            kwargs["truncation"] = True
            if truncation_method == TruncationSide.RIGHT:
                self.tokenizer.truncation_side = "right"
            else:
                self.tokenizer.truncation_side = "left"
            token_ids = self.tokenizer(inputs, **kwargs)["input_ids"][0].tolist()
        return token_ids

    def decode(
            self, all_token_ids: list[int], 
            skip_special_tokens: bool, 
            use_tool_calls: bool, 
            is_chat_req: bool,
            stream: bool, 
            **kwargs
        ) -> dict:
        """Decodes token IDs into structured output, supporting reasoning and tool calls.

        Args:
            all_token_ids: Complete list of generated token IDs.
            skip_special_tokens: Whether to skip special tokens during decoding.
            use_tool_calls: Whether tool call parsing is enabled.
            is_chat_req: Whether the request is a chat-style interaction.
            stream: Whether this is a streaming (incremental) decode.
            **kwargs: Additional arguments, including metadata and decoding indices.

        Returns:
            dict: Decoded result containing content, reasoning content, tool calls, and metadata.
        """
        metadata = kwargs.get(METADATA_KEY, {})
        self.tool_calls_processor.tools = metadata.get('tools', None)
        use_reasoning_parser = self._is_use_reasoning_parser(metadata)
        if not stream:

            """Case 1: separate the reasoning content ; tool call parsing for final content; 
            combine: reasoning_content + content + tool_calls"""
            if use_reasoning_parser and use_tool_calls and is_chat_req:
                reasoning_result = self._extract_reasoning_content(all_token_ids, skip_special_tokens)
                tool_calls_result = self.tool_calls_processor.decode(reasoning_result.get(CONTENT_KEY, ""))
                tool_calls_result.update({REASON_CONTENT_KEY: reasoning_result.get(REASON_CONTENT_KEY, "")})
                result = tool_calls_result

            # Case 2: return the reasoning content and final content.
            elif use_reasoning_parser and is_chat_req:
                result = self._extract_reasoning_content(all_token_ids, skip_special_tokens)

            # Case 3: directly parse the complete text using tool calls.
            elif use_tool_calls and is_chat_req:
                result = self.tool_calls_processor.decode(self._tokenizer_decode(
                    all_token_ids, skip_special_tokens=skip_special_tokens))

            # Case 4: Simple decoding to plain text
            else:
                result = {
                    CONTENT_KEY: self._tokenizer_decode(all_token_ids, skip_special_tokens=skip_special_tokens)
                }

            if self.reasoning_parser is None:
                return result
            reasoning_tokens = self.reasoning_parser.count_reasoning_tokens(all_token_ids)
            result.setdefault(METADATA_KEY, {})["reasoning_tokens"] = reasoning_tokens
            return result

        else:
            # Extract Streaming Index Parameters
            curr_decode_index = kwargs.get("curr_decode_index", -1)
            prev_decode_index = kwargs.get("prev_decode_index", -1)

            # Calculate the actual delta text
            curr_and_prev_content = self._tokenizer_decode(
                all_token_ids[prev_decode_index:], skip_special_tokens=skip_special_tokens)
            pre_text = self._tokenizer_decode(
                all_token_ids[prev_decode_index:curr_decode_index], skip_special_tokens=skip_special_tokens)
            # No new content is added or the characters are incomplete.
            if len(curr_and_prev_content) <= len(pre_text) or curr_and_prev_content.endswith("�"):
                if not metadata.get("req_end_flag", False):
                    if use_tool_calls:
                        return {"update_index": False, METADATA_KEY: metadata}
                    return {"update_index": False}
            delta_text = curr_and_prev_content[len(pre_text):]

            # Case 1: call get_combined_stream_result
            if use_reasoning_parser and use_tool_calls and is_chat_req:
                return self._get_combined_stream_result(all_token_ids, prev_decode_index, curr_decode_index,
                                                       skip_special_tokens, delta_text, metadata)
                                                       
            # Case 2: call streaming method of the reasoning parser
            elif use_reasoning_parser and is_chat_req:
                return self._extract_reasoning_content_streaming(all_token_ids, curr_decode_index, skip_special_tokens)
            
            # Case 3: call streaming method of tool call method
            elif use_tool_calls and is_chat_req:
                return self._extract_tool_calls_streaming(all_token_ids, prev_decode_index, curr_decode_index,
                                                         skip_special_tokens, delta_text, metadata)
            # Case 4: return incremental text directly
            else:
                return {CONTENT_KEY: delta_text}

    def _extract_reasoning_content(self, all_token_ids: list[int], skip_special_tokens: bool) -> dict:
        """Extracts reasoning and final content from a complete token sequence.

        Args:
            all_token_ids: Full list of generated token IDs.
            skip_special_tokens: Whether to skip special tokens during decoding.

        Returns:
            Dict[str, str]: Dictionary with 'reasoning_content' and 'content' keys.
        """
        reasoning_content_token_ids, content_token_ids = self.reasoning_parser.single_process_reasoning(all_token_ids)
        return {
            REASON_CONTENT_KEY: self._tokenizer_decode(
                reasoning_content_token_ids, skip_special_tokens=skip_special_tokens),
            CONTENT_KEY: self._tokenizer_decode(content_token_ids, skip_special_tokens=skip_special_tokens)
        }

    def _extract_reasoning_content_streaming(
            self, 
            all_token_ids: list[int], 
            curr_decode_index: int,
            skip_special_tokens: bool
        ) -> dict:
        reasoning_content_token_ids, content_token_ids = self.reasoning_parser.stream_process_reasoning(
            all_token_ids, curr_decode_index)
        # decode delta tokens
        reasoning_result = {
            REASON_CONTENT_KEY: self._tokenizer_decode(
                reasoning_content_token_ids, skip_special_tokens=skip_special_tokens),
            CONTENT_KEY: self._tokenizer_decode(content_token_ids, skip_special_tokens=skip_special_tokens)
        }
        return {k: v for k, v in reasoning_result.items() if v}

    def _extract_tool_calls_streaming(
            self, 
            all_token_ids: list[int], 
            prev_decode_index: int, 
            curr_decode_index: int,
            skip_special_tokens: bool, 
            delta_text: str, 
            metadata: dict
        ) -> dict:
        """Parses incremental tool call updates in streaming mode.
           Multiple tokenizer processes, compatible with the same 
           request toolcallsprocessor with different objects

        Args:
            all_token_ids: Full list of token IDs.
            prev_decode_index: Previous decoding index.
            curr_decode_index: Current decoding index.
            skip_special_tokens: Whether to skip special tokens.
            delta_text: Newly generated text segment.
            metadata: Current parser state (tool name/arg sent flags, tool ID).

        Returns:
            Dict[str, Any]: Tool call delta or fallback content, plus updated metadata.
        """
        if hasattr(self.tool_calls_processor, 'decode_stream'):
            # fetch the internal states of tool call parser: function name, arguments structure, called tool id
            self.tool_calls_processor.current_tool_name_sent = metadata.get("current_tool_name_sent")
            self.tool_calls_processor.current_tool_arguments_sent = metadata.get("current_tool_arguments_sent")
            self.tool_calls_processor.current_tool_id = metadata.get("current_tool_id")
            # call stream decode 
            result = self.tool_calls_processor.decode_stream(all_token_ids, prev_decode_index, curr_decode_index,
                                                        skip_special_tokens, delta_text)
            # persist states
            result.update({"metadata": {"current_tool_name_sent": self.tool_calls_processor.current_tool_name_sent,
                "current_tool_arguments_sent": self.tool_calls_processor.current_tool_arguments_sent,
                "current_tool_id": self.tool_calls_processor.current_tool_id}})
        else:
            logger.warning("Streaming function call parsing is not supported by the current model.")
            result = {CONTENT_KEY: delta_text}
        return result

    def _get_combined_stream_result(
            self, 
            all_token_ids: list[int], 
            prev_decode_index: int, 
            curr_decode_index: int,
            skip_special_tokens: bool, 
            delta_text: str, 
            metadata: dict
        ) -> dict:
        """
        Preferentially parse the thought chain.
        When the end tag of the thought chain is encountered, parse the function call.

        get_combined_stream_result()
        │
        ├── State 1: Reasoning in progress (full_has_over_think = False)
        │ └── Return delta reasoning content
        │
        ├── State 2: Reasoning end tag encountered (delta_has_over_think = True)
        │ ├── Extract reasoning content
        │ ├── Recalculate indices for tool call portion
        │ ├── Parse delta tool call updates
        │ └── Merge and return results
        │
        └── State 3: Tool call in progress (full_has_over_think = True, delta_has_over_think = False)
        └── Return delta tool call content
        """
        delta_has_over_think = self.reasoning_parser.is_reasoning_end(all_token_ids[curr_decode_index:])
        full_has_over_think = self.reasoning_parser.is_reasoning_end(all_token_ids)
        if not full_has_over_think:
            return self._extract_reasoning_content_streaming(all_token_ids, curr_decode_index, skip_special_tokens)
        elif full_has_over_think and delta_has_over_think:  # New content includes thought chain end tag
            reasoning_content_token_ids, content_token_ids = self.reasoning_parser.stream_process_reasoning(
                all_token_ids, curr_decode_index)
            if not content_token_ids:
                return {REASON_CONTENT_KEY: self._tokenizer_decode(reasoning_content_token_ids, \
                        skip_special_tokens=skip_special_tokens)} if reasoning_content_token_ids else {}
            # case : New content = reasoning content + </think> + tool_call content
            prev_decode_index = curr_decode_index
            curr_decode_index = len(all_token_ids) - len(content_token_ids)
            # update delta_text
            curr_and_prev_content = self._tokenizer_decode(
                all_token_ids[prev_decode_index:], skip_special_tokens=skip_special_tokens)
            pre_text = self._tokenizer_decode(
                all_token_ids[prev_decode_index:curr_decode_index], skip_special_tokens=skip_special_tokens)
            # No new content is added.
            if len(curr_and_prev_content) <= len(pre_text):
                return {"update_index": False, "metadata": metadata}
            delta_text = curr_and_prev_content[len(pre_text):]
            tool_calls_result = self._extract_tool_calls_streaming(all_token_ids, prev_decode_index, curr_decode_index,
                                                                  skip_special_tokens, delta_text, metadata)
            tool_calls_result.update(
                {REASON_CONTENT_KEY:
                    self._tokenizer_decode(reasoning_content_token_ids, skip_special_tokens=skip_special_tokens)})
            return {k: v for k, v in tool_calls_result.items() if v}
        else:
            return self._extract_tool_calls_streaming(all_token_ids, prev_decode_index, curr_decode_index, \
                                                    skip_special_tokens, delta_text, metadata)

    def _tokenizer_decode(self, outputs: list[int], **kwargs) -> str:
        """Decodes token IDs to a string using the internal tokenizer.

        Args:
            outputs: List of token IDs.
            **kwargs: Additional arguments passed to tokenizer.decode.

        Returns:
            str: Decoded string.
        """
        return self.tokenizer.decode(outputs, **kwargs)

    def _is_use_reasoning_parser(self, metadata: dict) -> bool:
        """
        judge need to post-thinking analysis.
        True, the model must support thinking and llm_config.llm.enable_reasoning is opened
        """
        if not self.llm_config.llm.enable_reasoning:
            return False
        if self.reasoning_parser is None:
            return False
        # Manually disable the model reasoning capability. Request_Switch > Weight_Config_Switch
        if metadata and "req_enable_thinking" in metadata:
            return metadata.get("req_enable_thinking")
        return self.enable_thinking
