# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import sys
from unittest.mock import MagicMock
import torch


def pytest_configure(config):
    """
    pytest 启动时自动执行一次，全局生效
    所有测试文件运行前都会先 mock
    """
    # 全局 Mock 模块
    MOCK_MODULES = [
        "torch_npu",
        "torch_npu._C",
        "torch_npu._C._distributed_c10d",
        "acl",
        "numba",
        "llm_datadist",
        "numba.core",
        "mindspore",
        "mindspore.nn",
        "mindspore.ops",
        "mindspore.common",
        "mindspore.common.dtype",
        "hccl",
        "ascend",
    ]

    for mod in MOCK_MODULES:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # 全局 mock torch.npu
    torch.npu = MagicMock()
    torch.npu.Stream = MagicMock()
