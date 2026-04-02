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

"""使用样例
from mindie_llm.utils.tensor import backend, op

backend.Tensor

operations: 
    op.ones
    op.full

type:
    backend.int64
    backend.float32

npu/hal 
    mindspore 中的 torch.npu 等效于 mindspore.hal
    npu.synchronize()
    npu.max_memory_allocated()
    npu.get_device_properties(self.rank).total_memory 
"""

from typing import TypeAlias

from mindie_llm.utils.tensor.ms_tensor import MindSporeBackend
from mindie_llm.utils.tensor.torch_tensor import TorchBackend
from mindie_llm.utils.tensor.llm_tensor import LLMBackend
from mindie_llm.utils.env import ENV
from mindie_llm.modeling.backend_type import BackendType


def _set_tensor_backend() -> LLMBackend:
    _backend = MindSporeBackend() if ENV.framework_backend == BackendType.MS else TorchBackend()
    return _backend


tensor_backend = _set_tensor_backend()
backend = tensor_backend.get_backend()
BackendTensor: TypeAlias = backend.Tensor
op = tensor_backend.get_op()

npu = tensor_backend.get_npu()
hal = tensor_backend.get_hal()
