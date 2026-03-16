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

from dataclasses import dataclass, field
from typing import List

import torch
import torch_npu


def _maybe_npu_prefetch(inputs: torch.Tensor,
                       dependency: torch.Tensor,
                       max_size: int = 0,
                       offset: int = 0) -> None:
    input_size = inputs.element_size() * inputs.numel()
    if max_size <= 0 or max_size > input_size:
        max_size = input_size
    torch_npu.npu_prefetch(inputs, dependency, max_size, offset)


def _prefetch_preprocess(weight: torch.Tensor, start_flag: torch.Tensor,
                        max_weight_size: int, weight_prefetch_stream: torch.npu.Stream) -> None:
    calculation_stream = torch.npu.current_stream()
    weight_prefetch_stream.wait_stream(calculation_stream)
    with torch.npu.stream(weight_prefetch_stream):
        _maybe_npu_prefetch(inputs=weight,
                        dependency=start_flag,
                        max_size=max_weight_size)


class WeightPrefetchMethod:
    """
    Unified weight prefetch method.
    """
    def __init__(self) -> None:
        self.prefetch_stream = None
        self.prefetch_weights = {}
        self.enable_prefetch = False

    def is_prefetch_enabled(self) -> bool:
        return self.enable_prefetch

    def enable_weight_prefetch(self):
        if self.enable_prefetch is False:
            self.prefetch_stream = torch.npu.Stream(device=torch.npu.current_device())
            self.enable_prefetch = True

    def disable_weight_prefetch(self):
        self.enable_prefetch = False
        self.prefetch_stream = None

    def add_prefetch_weight(self, weight_name: str, weight: torch.Tensor) -> None:
        self.prefetch_weights[weight_name] = weight

    def prefetch_weight_preprocess(
            self, weight: torch.Tensor,
            start_flag: torch.Tensor,
            ratio: float = 1) -> None:

        weight_size = weight.data.element_size() * weight.data.numel() * ratio
        _prefetch_preprocess(weight=weight,
                            start_flag=start_flag,
                            max_weight_size=int(weight_size),
                            weight_prefetch_stream=self.prefetch_stream)
    
    def prefetch_weight_postprocess(self) -> None:
        calculation_stream = torch.npu.current_stream()
        calculation_stream.wait_stream(self.prefetch_stream)


weight_prefetcher = WeightPrefetchMethod()
