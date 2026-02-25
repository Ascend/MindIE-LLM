# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import math
import acl
from ...utils.log.logging import logger
from ...utils.tensor import npu

INT8_BYTES_SIZE = 1
TOTAL_MEMORY = 60 * 1024 * 1024 * 1024
MM_LONG_SEQ_MEMORY = 26 * 1024 * 1024 * 1024
MM_NORMAL_SEQ_MEMORY = 18 * 1024 * 1024 * 1024
MM_LONG_SEQ_TOKENLEN = 4096

# 显存预警阈值梯度
THRESHOLD_GRAD = 0.05


def calc_block_mem(model_info, block_size, num_speculative_tokens=None):
    if num_speculative_tokens is None:
        num_speculative_tokens = 0
    total_head_size = model_info.num_kv_heads * model_info.head_size
    # k_total_head_size和v_total_head_size需成对定义
    if model_info.k_head_size > 0 or model_info.v_head_size > 0:
        k_total_head_size = model_info.num_kv_heads * model_info.k_head_size
        v_total_head_size = model_info.num_kv_heads * model_info.v_head_size
    else:
        k_total_head_size = total_head_size
        v_total_head_size = total_head_size
    num_layers = model_info.num_layers + (num_speculative_tokens >= 1)
    per_layer_k_cache_bytes_size = [model_info.data_byte_size for layer_id in range(num_layers)]
    model_mem_size = num_layers * v_total_head_size * model_info.data_byte_size
    if model_info.kvcache_quant_layers:
        for i, kvcache_quant in enumerate(model_info.kvcache_quant_layers):
            if kvcache_quant:
                per_layer_k_cache_bytes_size[i] = INT8_BYTES_SIZE
    for bytes_size in per_layer_k_cache_bytes_size:
        model_mem_size += k_total_head_size * bytes_size
    if model_info.index_head_dim is not None:
        index_total_head_size = model_info.index_head_dim * model_info.num_index_heads
        model_mem_size += num_layers * index_total_head_size * model_info.data_byte_size
    block_mem_size = model_mem_size * block_size
    return block_mem_size


def calc_npu_mem(block_nums, model_info, block_size, num_speculative_tokens=None):
    block_mem_size = calc_block_mem(model_info, block_size, num_speculative_tokens)
    npu_mem_size = block_nums * block_mem_size
    return npu_mem_size


def gb(mem_size):
    return float(mem_size / (1024 ** 3))


class NpuMemoryWatcher:
    def __init__(self):
        self.warmup_mem = 0
        self.threshold = 0
        self.warmup = True

    def watch_npu_mem(self, rank_id, tag, is_multimodal=False, max_input_len=2048, trigger_count=-1):
        if self.warmup:
            npu.synchronize()
            free_mem, total_mem, _ = acl.rt.get_mem_info(1)
            peak_mem = total_mem - free_mem

            if is_multimodal and total_mem >= TOTAL_MEMORY:
                memory_threshold = (
                    MM_LONG_SEQ_MEMORY
                    if max_input_len > MM_LONG_SEQ_TOKENLEN
                    else MM_NORMAL_SEQ_MEMORY
                )
                if free_mem < memory_threshold:
                    error_message = (
                        f"Warmup failed, because of multimodal model inference out of memory "
                        f"when `maxInputTokenLen` set to {max_input_len}. Please try to "
                        f"decrease `maxPrefillTokens` in config.json of mindie-service."
                    )
                    print_log(rank_id, logger.error, error_message)
                    raise RuntimeError("NPU out of memory. " + error_message)

            logger.info(f"{tag}, peak mem: {gb(peak_mem):.2f}G, total_mem: {gb(total_mem):.2f}G")
            return total_mem, peak_mem

        else:
            if trigger_count == 0:
                free_mem, total_mem, _ = acl.rt.get_mem_info(1)
                peak_mem = total_mem - free_mem
                remaining_mem_warmup = total_mem - self.warmup_mem

                if remaining_mem_warmup == 0:
                    raise RuntimeError("NPU out of memory.")

                remaining_mem_reduction = (peak_mem - self.warmup_mem) / remaining_mem_warmup

            # 检查是否超过阈值
                if remaining_mem_reduction > self.threshold:
                    logger.warning(
                        f"After warmup, mem is {gb(self.warmup_mem):.2f}G, left mem is {gb(remaining_mem_warmup):.2f}G."
                        f"{tag}, peak mem is: {gb(peak_mem):.2f}G, available mem is: {gb(free_mem):.2f}G. "
                        f"Remaining memory decreased {100 * remaining_mem_reduction:.2f}% compared to the warmup phase."
                    )
                    self.threshold = math.ceil(remaining_mem_reduction / THRESHOLD_GRAD) * THRESHOLD_GRAD
            else:
                total_mem = 0
                peak_mem = 0
            return total_mem, peak_mem

    def _set_warmup_mem_(self, warmup_mem):
        self.warmup_mem = warmup_mem
        self.warmup = False