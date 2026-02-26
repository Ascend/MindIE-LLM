# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

def decode_one(tokenizer, input_tokens, skip_special_tokens, tokenizer_sliding_window_size):
    pre_index = len(input_tokens) - 1
    start_index = max(pre_index - tokenizer_sliding_window_size, 0)
    window_text = tokenizer.decode(input_tokens[start_index:], skip_special_tokens=skip_special_tokens)
    if pre_index == 0:
        pre_text = ''
    else:
        pre_text = tokenizer.decode(input_tokens[start_index:pre_index],
                                            skip_special_tokens=skip_special_tokens)
        pre_text = pre_text.rstrip('�')

    if len(window_text) > len(pre_text) and not window_text.endswith('�'):
        return window_text[len(pre_text):]
    else:
        return ''