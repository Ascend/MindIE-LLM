# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# Implement part of this file based on vllm-project/vllm
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any
from abc import ABC, abstractmethod

import torch

from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase


class QuantizationConfigBase(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]:
        """List of quantization config filenames in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfigBase":
        """Create quantization config instance from model's quantization config file."""
        raise NotImplementedError

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizationMethodBase | None:
        """Retrieve the appropriate quantization method for a given layer."""
        raise NotImplementedError

    @abstractmethod
    def get_quant_type_by_weight_name(
        self, prefix: str | list[str], suffix: str) -> str:
        """Retrieve the quantization type for a specific weight parameter."""
        raise NotImplementedError
