# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from ...utils.config import HandlingBackend


def get_selector_registry(handling_backend):
    if handling_backend == HandlingBackend.ATB:
        from .atb_selectors import ATB_SELECTOR_REGISTRY
        return ATB_SELECTOR_REGISTRY
    elif handling_backend == HandlingBackend.CPU:
        from .cpu_selectors import CPU_SELECTOR_REGISTRY
        return CPU_SELECTOR_REGISTRY
    elif handling_backend == HandlingBackend.PTA:
        from .pta_selectors import PTA_SELECTOR_REGISTRY
        return PTA_SELECTOR_REGISTRY
    elif handling_backend == HandlingBackend.MS:
        from .ms_selectors import MS_SELECTOR_REGISTRY
        return MS_SELECTOR_REGISTRY
    else:
        raise NotImplementedError(
            f'{handling_backend} not implemented, supported backends `atb/cpu/pta/ms`')