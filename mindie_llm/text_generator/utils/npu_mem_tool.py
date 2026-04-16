# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Part of this file implemented based on vllm project.
#
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
import gc
from dataclasses import dataclass
import contextlib
import acl
import torch

from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.utils.tensor import npu


INT8_BYTES_SIZE = 1
TOTAL_MEMORY = 60 * 1024 * 1024 * 1024
MM_LONG_SEQ_MEMORY = 26 * 1024 * 1024 * 1024
MM_NORMAL_SEQ_MEMORY = 18 * 1024 * 1024 * 1024
MM_LONG_SEQ_TOKENLEN = 4096

# 显存预警阈值梯度
THRESHOLD_GRAD = 0.05


class WeightMemoryProfiler:
    def __init__(self):
        self.before_load_weight = 0
        self.after_load_weight = 0
        self.model_weight = 0

    def __enter__(self):
        self.before_load_weight = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        npu.empty_cache()
        npu.reset_peak_memory_stats()
        self.after_load_weight = npu.max_memory_allocated()
        self.model_weight = self.after_load_weight - self.before_load_weight


# MemorySnapshot, MemoryProfilingResult and memory_profiling are implemented with reference to vLLM.
# https://github.com/vllm-project/vllm/blob/main/vllm/utils/mem_utils.py
@dataclass
class MemorySnapshot:
    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    npu_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0

    auto_measure: bool = True

    def __post_init__(self) -> None:
        if self.auto_measure:
            self.measure()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            npu_memory=self.npu_memory - other.npu_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            auto_measure=False,
        )

    def __repr__(self) -> str:
        return (
            f"torch_peak={gb(self.torch_peak):.2f}GiB, "
            f"free_memory={gb(self.free_memory):.2f}GiB, "
            f"total_memory={gb(self.total_memory):.2f}GiB, "
            f"npu_memory={gb(self.npu_memory):.2f}GiB, "
            f"torch_memory={gb(self.torch_memory):.2f}GiB, "
            f"non_torch_memory={gb(self.non_torch_memory):.2f}GiB, "
            f"auto_measure={self.auto_measure}."
        )

    def measure(self) -> None:
        self.torch_peak = npu.max_memory_allocated()

        npu.current_stream().synchronize()

        self.free_memory, self.total_memory, _ = acl.rt.get_mem_info(1)

        self.npu_memory = self.total_memory - self.free_memory

        self.torch_memory = npu.memory_reserved()

        self.non_torch_memory = self.npu_memory - self.torch_memory


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    total_memory: int = 0

    def __post_init__(self) -> None:
        self.before_profile = MemorySnapshot(auto_measure=False)
        self.after_profile = MemorySnapshot(auto_measure=False)

    def __repr__(self) -> str:
        return (
            f"Total npu memory: "
            f"{gb(self.total_memory):.2f} GiB; "
            f"Total non KV cache memory: "
            f"{gb(self.non_kv_cache_memory):.2f} GiB; "
            f"torch peak memory increase: "
            f"{gb(self.torch_peak_increase):.2f} GiB; "
            f"non-torch forward increase memory: "
            f"{gb(self.non_torch_increase):.2f} GiB."
        )


@contextlib.contextmanager
def memory_profiling(baseline_non_torch: int = 0, weights_memory: int = 0, backend_type=None):
    gc.collect()
    npu.empty_cache()
    npu.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_profile.measure()

    yield result

    gc.collect()
    if backend_type != "torch":
        npu.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = result.after_profile.non_torch_memory - baseline_non_torch
    result.total_memory = result.before_profile.total_memory

    non_torch_memory = result.non_torch_increase
    peak_activation_memory = result.torch_peak_increase
    result.non_kv_cache_memory = non_torch_memory + peak_activation_memory + weights_memory


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
    """Convert memory size from Bytes to GiB.

    Args:
        mem_size (int): Memory size in Bytes.

    Returns:
        float: Memory size in GiB.
    """
    return float(mem_size / (1024**3))


class NpuMemoryWatcher:
    def __init__(self):
        self.warmup_mem = 0
        self.threshold = 0
        self.warmup = True

    def watch_npu_mem(self, rank_id, tag, is_multimodal=False, max_input_len=2048, trigger_count=-1):
        if self.warmup:
            # Ensure that the warmup is completed before checking the memory,
            # otherwise the memory statistics of npu-smi may be incomplete.
            torch.npu.current_stream().synchronize()
            free_mem, total_mem, _ = acl.rt.get_mem_info(1)
            peak_mem = total_mem - free_mem

            if is_multimodal and "310P" not in acl.get_soc_name().upper() and total_mem >= TOTAL_MEMORY:
                memory_threshold = MM_LONG_SEQ_MEMORY if max_input_len > MM_LONG_SEQ_TOKENLEN else MM_NORMAL_SEQ_MEMORY
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

    def _set_warmup_mem(self, warmup_mem):
        self.warmup_mem = warmup_mem
        self.warmup = False
