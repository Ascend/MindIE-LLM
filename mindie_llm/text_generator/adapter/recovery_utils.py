# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch_npu

from ...utils.log.logging import logger


def is_uce_error_addr_overlap_tensor_addr(uce_addr_start, uce_addr_end, tensor_addr_start, tensor_addr_end):
    return (uce_addr_start >= tensor_addr_start) and (uce_addr_end <= tensor_addr_end)


def get_tensor_address_range(input_tensor):
    addr_start = input_tensor.data_ptr()
    addr_end = addr_start + input_tensor.numel() * input_tensor.element_size()
    return addr_start, addr_end


def check_and_recover_uce_in_cache(uce_addr_start, uce_addr_end, cache_tensor, layer_idx, cache_type):
    cache_addr_start, cache_addr_end = get_tensor_address_range(cache_tensor)
    if is_uce_error_addr_overlap_tensor_addr(uce_addr_start, uce_addr_end, cache_addr_start, cache_addr_end):
        torch_npu.npu._recovery.update_npu_tensor_to_safe(cache_tensor)
        logger.info(f"HBM UCE address in {cache_type} of layer {layer_idx}, update {cache_type} to safe")
        return True
    else:
        logger.info(
            f"HBM UCE address not in {cache_type} address ({cache_addr_start}, {cache_addr_end}) of layer {layer_idx}"
        )
        return False
