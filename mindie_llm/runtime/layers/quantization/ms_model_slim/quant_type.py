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


class InferenceMode(str, Enum):
    """
    Enumeration for different inference stages.
    """
    PREFILL = "PREFILL"
    DECODE = "DECODE"


class QuantType(str, Enum):
    """
    Enumeration for supported quantization types.
    """
    FLOAT = "FLOAT"
    W8A8 = "W8A8"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W8A8_MIX = "W8A8_MIX"
    W8A8_MXFP8 = "W8A8_MXFP8"
    W4A8_DYNAMIC = "W4A8_DYNAMIC"
