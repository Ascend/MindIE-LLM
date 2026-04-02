#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
from mindformers import get_model
from mindie_llm.modeling.model_wrapper.wrapper import BaseTokenizerWrapper

CONTENT_KEY = "content"


class MFTokenizerWrapper(BaseTokenizerWrapper):
    def __init__(self, model_id: str, **kwargs):
        tokenizer, input_builder = get_model(model_id)
        self.tokenizer = tokenizer
        self.input_builder = input_builder
        self.toolscallprocessor = None

    def tokenize(self, inputs, **kwargs):
        return self.tokenizer.tokenize(inputs, **kwargs)
    
    def encode(self, inputs, **kwargs):
        is_chatting = kwargs.pop("is_chatting", False)
        if is_chatting:
            return self.input_builder.make_context(0, inputs, **kwargs)
        else:
            return self.tokenizer(inputs, **kwargs)["input_ids"][0].tolist()

    def decode(self, all_token_ids: list[int], skip_special_tokens: bool, use_tool_calls: bool, is_chat_req: bool,
                   stream: bool, **kwargs):
        if not stream:
            return self._detokenize(all_token_ids, skip_special_tokens)
        else:
            curr_decode_index = kwargs.get("curr_decode_index", -1)
            prev_decode_index = kwargs.get("prev_decode_index", -1)
            return self._detokenize_stream(all_token_ids, prev_decode_index, curr_decode_index, skip_special_tokens)
    
    def _detokenize(self, all_token_ids: list[int], skip_special_tokens: bool):
        in_array = np.array(all_token_ids, copy=False)
        new_text = self.tokenizer.decode(in_array, skip_special_tokens=skip_special_tokens)

        return {CONTENT_KEY: new_text}
    
    def _detokenize_stream(self, all_token_ids: list[int], prev_decode_index: int, curr_decode_index: int,
                           skip_special_tokens: bool):
        input_tensor = np.array(all_token_ids, copy=False)
        start_index = prev_decode_index
        full_text = self.tokenizer.decode(input_tensor[start_index:],
                                          skip_special_tokens=skip_special_tokens)
        pre_text = self.tokenizer.decode(input_tensor[start_index:curr_decode_index],
                                         skip_special_tokens=skip_special_tokens)
        if len(full_text) > len(pre_text) and not full_text.endswith("�"):
            return {CONTENT_KEY: full_text[len(pre_text):]}
        else:
            return {CONTENT_KEY: ""}
