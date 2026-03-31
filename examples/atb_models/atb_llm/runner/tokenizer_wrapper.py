# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import json
from enum import Enum
from typing import Dict
import numpy as np

from ..models import get_model
from ..utils.file_utils import standardize_path, check_file_safety
from ..utils.log.logging import logger

REASON_CONTENT_KEY = "reasoning_content"
CONTENT_KEY = "content"
METADATA_KEY = "metadata"


class TruncationSide(int, Enum):
    Disable = 0
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
        router_ins = get_model(model_name_or_path, **kwargs)
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.postprocessor = router_ins.postprocessor
        self.tokenize = router_ins.tokenize # This is only used by multi-modal models
        self.tool_calls_parser = router_ins.toolscallprocessor
        self.reasoning_parser = router_ins.reasoning_parser
        self.llm_config = router_ins.llm_config
        self.enable_thinking = self.tokenizer.init_kwargs.get("enable_thinking", True)
        self.obfuscation_func = None
        self.enable_data_obfuscation = False
        if self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir is not None:
            from ai_asset_obfuscate import data_asset_obfuscation
            self.obfuscation_func = data_asset_obfuscation.DataAssetObfuscation(self.config.vocab_size)
            self.enable_data_obfuscation = True
            file_name_list = ["kms_ca.pem", "kms_cfs.pem", "kms_cfs.key", "kms_client_key_enc.txt", \
                "aiguard_psk", "aiguard_psk_enc.txt"]
            file_list = []
            for name in file_name_list:
                file_path = standardize_path(
                    os.path.join(self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir, name))
                check_file_safety(file_path)
                file_list.append(file_path)

            tls_info = (
                file_list[0], file_list[1], file_list[2],
                self.llm_config.llm.pmcc_obfuscation_options.kms_agent_port,
                self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir,
                file_list[3],
            )
            psk_info = (
                file_list[4],
                self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir,
                file_list[5],
            )
            res, msg = self.obfuscation_func.set_seed_safer(tls_info, psk_info)
            if res != 0:
                raise RuntimeError(f"Data obfuscation init failed, please refer to MindIE official document, " \
                                f"error msg: {msg}")
        
    def encode(self, inputs, **kwargs):
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
        if self.enable_data_obfuscation:
            if len(token_ids) > 0 and isinstance(token_ids[0], list):
                token_ids = self.obfuscation_func.data_2d_obf(token_ids)
            elif len(token_ids) > 0 and isinstance(token_ids[0], int):
                token_ids = self.obfuscation_func.data_1d_obf(token_ids)
        return token_ids

    def decode(self, all_token_ids: list[int], skip_special_tokens: bool, use_tool_calls: bool, is_chat_req: bool,
                   stream: bool, **kwargs):
        metadata = kwargs.get(METADATA_KEY, {})
        self.tool_calls_parser.tools = metadata.get('tools', None)
        use_reasoning_parser = self._is_use_reasoning_parser(metadata)
        if not stream:
            if use_reasoning_parser and use_tool_calls and is_chat_req:
                reasoning_result = self.extract_reasoning_content(all_token_ids, skip_special_tokens)
                tool_calls_result = self.tool_calls_parser.decode(reasoning_result.get(CONTENT_KEY, ""))
                tool_calls_result.update({REASON_CONTENT_KEY: reasoning_result.get(REASON_CONTENT_KEY, "")})
                result = tool_calls_result
            elif use_reasoning_parser and is_chat_req:
                result = self.extract_reasoning_content(all_token_ids, skip_special_tokens)
            elif use_tool_calls and is_chat_req:
                result = self.tool_calls_parser.decode(self._tokenizer_decode(
                    all_token_ids, skip_special_tokens=skip_special_tokens))
            else:
                result = {
                    CONTENT_KEY: self._tokenizer_decode(all_token_ids, skip_special_tokens=skip_special_tokens)
                }

            if not self.config.is_reasoning_model:
                return result
            reasoning_tokens = self.reasoning_parser.count_reasoning_tokens(all_token_ids)
            result.setdefault(METADATA_KEY, {})["reasoning_tokens"] = reasoning_tokens
            return result

        else:
            curr_decode_index = kwargs.get("curr_decode_index", -1)
            prev_decode_index = kwargs.get("prev_decode_index", -1)
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
            if use_reasoning_parser and use_tool_calls and is_chat_req:
                return self.get_combined_stream_result(all_token_ids, prev_decode_index, curr_decode_index,
                                                       skip_special_tokens, delta_text, metadata)

            elif use_reasoning_parser and is_chat_req:
                return self.extract_reasoning_content_streaming(all_token_ids, curr_decode_index, skip_special_tokens)
            elif use_tool_calls and is_chat_req:
                return self.extract_tool_calls_streaming(all_token_ids, prev_decode_index, curr_decode_index,
                                                         skip_special_tokens, delta_text, metadata)
            else:
                return {CONTENT_KEY: delta_text}

    def extract_reasoning_content(self, all_token_ids: list[int], skip_special_tokens: bool) -> Dict:
        reasoning_content_token_ids, content_token_ids = self.reasoning_parser.single_process_reasoning(all_token_ids)
        return {
            REASON_CONTENT_KEY: self._tokenizer_decode(
                reasoning_content_token_ids, skip_special_tokens=skip_special_tokens),
            CONTENT_KEY: self._tokenizer_decode(content_token_ids, skip_special_tokens=skip_special_tokens)
        }

    def extract_reasoning_content_streaming(self, all_token_ids: list[int], curr_decode_index: int,
                                            skip_special_tokens: bool):
        reasoning_content_token_ids, content_token_ids = self.reasoning_parser.stream_process_reasoning(
            all_token_ids, curr_decode_index)
        reasoning_result = {
            REASON_CONTENT_KEY: self._tokenizer_decode(
                reasoning_content_token_ids, skip_special_tokens=skip_special_tokens),
            CONTENT_KEY: self._tokenizer_decode(content_token_ids, skip_special_tokens=skip_special_tokens)
        }
        return {k: v for k, v in reasoning_result.items() if v}

    def extract_tool_calls_streaming(self, all_token_ids: list[int], prev_decode_index: int, curr_decode_index: int,
                                     skip_special_tokens: bool, delta_text: str, metadata: Dict):
        # Multiple tokenizer processes, compatible with the same request toolscallprocessor with different objects
        if hasattr(self.tool_calls_parser, 'decode_stream'):
            self.tool_calls_parser.current_tool_name_sent = metadata.get("current_tool_name_sent")
            self.tool_calls_parser.current_tool_arguments_sent = metadata.get("current_tool_arguments_sent")
            self.tool_calls_parser.current_tool_id = metadata.get("current_tool_id")
            result = self.tool_calls_parser.decode_stream(all_token_ids, prev_decode_index, curr_decode_index,
                                                        skip_special_tokens, delta_text)
            result.update({"metadata": {"current_tool_name_sent": self.tool_calls_parser.current_tool_name_sent,
                "current_tool_arguments_sent": self.tool_calls_parser.current_tool_arguments_sent,
                "current_tool_id": self.tool_calls_parser.current_tool_id}})
        else:
            logger.warning("Streaming function call parsing is not supported by the current model.")
            result = {CONTENT_KEY: delta_text}
        return result

    def get_combined_stream_result(self, all_token_ids: list[int], prev_decode_index: int, curr_decode_index: int,
                                   skip_special_tokens: bool, delta_text: str, metadata: Dict):
        """
        Preferentially parse the thought chain.
        When the end tag of the thought chain is encountered, parse the function call.
        """
        delta_has_over_think = self.reasoning_parser.is_reasoning_end(all_token_ids[curr_decode_index:])
        full_has_over_think = self.reasoning_parser.is_reasoning_end(all_token_ids)
        if not full_has_over_think:
            return self.extract_reasoning_content_streaming(all_token_ids, curr_decode_index, skip_special_tokens)
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
            tool_calls_result = self.extract_tool_calls_streaming(all_token_ids, prev_decode_index, curr_decode_index,
                                                                  skip_special_tokens, delta_text, metadata)
            tool_calls_result.update(
                {REASON_CONTENT_KEY:
                    self._tokenizer_decode(reasoning_content_token_ids, skip_special_tokens=skip_special_tokens)})
            return {k: v for k, v in tool_calls_result.items() if v}
        else:
            return self.extract_tool_calls_streaming(all_token_ids, prev_decode_index, curr_decode_index, \
                                                    skip_special_tokens, delta_text, metadata)

    def _tokenizer_decode(self, outputs, **kwargs):
        if self.enable_data_obfuscation:
            outputs_list = outputs if isinstance(outputs, list) else [outputs]
            if len(outputs_list) > 0 and isinstance(outputs_list[0], list):
                outputs_list = np.array(self.obfuscation_func.data_2d_deobf(outputs_list))
            elif len(outputs_list) > 0 and isinstance(outputs_list[0], int):
                outputs_list = np.array(self.obfuscation_func.data_1d_deobf(outputs_list))
            outputs = outputs_list
        return self.tokenizer.decode(outputs, **kwargs)

    def _is_use_reasoning_parser(self, metadata: Dict) -> bool:
        """
        judge need to post-thinking analysis.
        True, the model must support thinking and llm_config.llm.enable_reasoning is opened
        """
        if not self.llm_config.llm.enable_reasoning:
            return False
        if not self.config.is_reasoning_model:
            return False
        # Manually disable the model reasoning capability. Request_Switch > Weight_Config_Switch
        if metadata and "req_enable_thinking" in metadata:
            return metadata.get("req_enable_thinking")
        return self.enable_thinking