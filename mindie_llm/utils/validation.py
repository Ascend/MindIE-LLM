# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Any, List, Union

SAFE_STRING_LENGTH = 10 ** 6
SAFE_LIST_LENGTH = 10 ** 3

UPPER_SAFE_BLOCK_SIZE = 2 ** 10
LOWER_SAFE_BLOCK_SIZE = 1
UPPER_SAFE_NPU_MEM = 64
LOWER_SAFE_NPU_MEM = 1

UPPER_SAFE_BATCH_SIZE = 819200
LOWER_SAFE_BATCH_SIZE = 1
UPPER_SAFE_SEQUENCE_LENGTH = 10 ** 6
LOWER_SAFE_SEQUENCE_LENGTH = 1

LOWER_SAFE_REPETITION_PENALTY = 0
LOWER_SAFE_TEMPERATURE = 0
SAFE_TOP_K = 0
UPPER_SAFE_TOP_P = 1
LOWER_SAFE_TOP_P = 0
UPPER_SAFE_LOGPROBS = 20
LOWER_SAFE_LOGPROBS = 0
UPPER_SAFE_SEED = 2 ** 63 - 1
LOWER_SAFE_SEED = 0
UPPER_SAFE_BEAM_WIDTH = 8192
LOWER_SAFE_BEAM_WIDTH = 1
UPPER_SAFE_BEST_OF = 8192
LOWER_SAFE_BEST_OF = 1


class ValidationError(Exception):
    def __init__(self, param_name: str, detail: str) -> None:
        message = f'The parameter `{param_name}` is invalid: ' + detail
        super().__init__(message)


class InconsistencyError(ValidationError):
    def __init__(self, param_name: str, param_attr: str, other_attr: str) -> None:
        detail = f'The {param_attr} is not equal to {other_attr}.'
        super().__init__(param_name, detail)


class OutOfBoundsError(ValidationError):
    def __init__(self, param_name: str, limit_key: str, limit_value: Union[int, float, str]) -> None:
        detail = (f'It exceeds the safety limit `{limit_value}` as `{limit_key}`. If you believe this boundary value is'
                  f' unreasonable, please modify the boundary value under path `mindie_llm.utils.validation`.')
        super().__init__(param_name, detail)


class UnsupportedTypeError(ValidationError):
    def __init__(self, param_name: str, valid_type: str) -> None:
        detail = f'Its type is not supported, which should be `{valid_type}`.'
        super().__init__(param_name, detail)


def validate_list(param_name: str, param_list: List[Any]) -> None:
    if not isinstance(param_list, list):
        raise UnsupportedTypeError(param_name, 'List[Any]')
    if len(param_list) > SAFE_LIST_LENGTH:
        raise OutOfBoundsError(param_name, 'SAFE_LIST_LENGTH', SAFE_LIST_LENGTH)


def validate_string(param_name: str, param_str: str) -> None:
    if len(param_str) > SAFE_STRING_LENGTH:
        raise OutOfBoundsError(param_name, 'SAFE_STRING_LENGTH', SAFE_STRING_LENGTH)