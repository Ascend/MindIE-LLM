# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import Enum


class FeatureType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    LAYERWISE_PREFILL = "layerwise-prefill"
    LAYERWISE_DECODE = "layerwise-decode"
    SINGLE_LORA = "singielora"
    MULTI_LORA = "multilora"
    DAP = "dap"
    FLASHCOMM = "flashcomm"
    SPECULATE = "speculate"
    SPLITFUSE = "splitfuse"
    MOONCAKE = "mempool"

COMPATIBLE_MATRIX: dict[FeatureType, list[FeatureType]] = {
    FeatureType.PREFILL: [FeatureType.SINGLE_LORA, FeatureType.MULTI_LORA, FeatureType.FLASHCOMM, \
        FeatureType.DAP, FeatureType.SPLITFUSE, FeatureType.MOONCAKE],
    FeatureType.DECODE: [FeatureType.SINGLE_LORA, FeatureType.MULTI_LORA, FeatureType.SPECULATE],
    FeatureType.SINGLE_LORA: [FeatureType.PREFILL, FeatureType.DECODE, FeatureType.SPLITFUSE, FeatureType.FLASHCOMM],
    FeatureType.MULTI_LORA: [FeatureType.PREFILL, FeatureType.DECODE, FeatureType.SPLITFUSE, FeatureType.FLASHCOMM],
    FeatureType.DAP: [FeatureType.PREFILL],
    FeatureType.FLASHCOMM: [FeatureType.PREFILL, FeatureType.SPLITFUSE, FeatureType.SINGLE_LORA, \
        FeatureType.MULTI_LORA],
    FeatureType.SPECULATE: [FeatureType.DECODE],
    FeatureType.SPLITFUSE: [FeatureType.PREFILL, FeatureType.FLASHCOMM, FeatureType.DAP, \
        FeatureType.LAYERWISE_PREFILL, FeatureType.SINGLE_LORA, FeatureType.MULTI_LORA],
    FeatureType.LAYERWISE_PREFILL: [FeatureType.SPLITFUSE],
    FeatureType.LAYERWISE_DECODE: [],
    FeatureType.MOONCAKE: [FeatureType.PREFILL],
}