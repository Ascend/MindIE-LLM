# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import sys
from enum import Enum
from dataclasses import dataclass

import torch
import torch_npu

from .cpu_binding import execute_command
from .env import ENV


class Topo(str, Enum):
    pcie = "pcie"
    hccs = "hccs"
    xlink = "xlink"


class CommunicationLibrary(str, Enum):
    hccl = "hccl"
    lccl = "lccl"


@dataclass
class NPUSocInfo:
    soc_name: str = ""
    soc_version: int = -1
    need_nz: bool = False
    matmul_nd_nz: bool = False
    support_bf16 = True

    def __post_init__(self):
        self.soc_version = torch_npu._C._npu_get_soc_version()
        if self.soc_version in (100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205):
            self.need_nz = True
            self.support_bf16 = False

    @property
    def communication_backend(self):
        return CommunicationLibrary.lccl \
        if self.is_support_lccl() and not ENV.hccl_enable else CommunicationLibrary.hccl

    @staticmethod
    def is_support_hccs():
        npu_smi_info = execute_command(["npu-smi", "info", "-t", "topo"])
        legend_index = npu_smi_info.find("Legend")
        if Topo.hccs in npu_smi_info[:legend_index].lower() or \
            Topo.xlink in npu_smi_info[:legend_index].lower():
            return True
        return False

    @staticmethod
    def is_rc_device():
        lspci_output = execute_command(["lspci"])
        pci_info = [line for line in lspci_output.split("\n") if "accelerators" in line.strip()]
        return not pci_info

    def is_support_lccl(self):
        return not self.need_nz and self.is_support_hccs()

    def is_300i(self):
        if self.soc_version in (200, 201, 202, 203, 204, 205):
            return True
        return False


def load_atb_speed():
    lib_path = os.path.join(ENV.atb_speed_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)
    sys.path.append(os.path.join(ENV.atb_speed_home_path, 'lib'))


def is_lcoc_enable(need_nz):
    lcoc_enable = ENV.lcoc_enable and (not need_nz)
    return lcoc_enable