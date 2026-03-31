#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch_npu

from mindie_llm.utils.log.logging import logger


def set_npu_compile_mode():
    torch.npu.set_compile_mode(jit_compile=False)
    # pylint: disable=protected-access：
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version not in [104, 220, 221, 222, 223, 224]:
        logger.info("310P,some op does not support")
        option = {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceNansum"}
        torch.npu.set_option(option)
