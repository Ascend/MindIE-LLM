# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from dataclasses import dataclass, field
import pathlib

MB = 1024 * 1024
MAX_LOG_FILE_SIZE = 500 * MB


def get_benchmark_filepath():
    home_path = os.getenv("MINDIE_LLM_HOME_PATH") if os.getenv("MINDIE_LLM_HOME_PATH") is not None else ""
    return os.getenv("MINDIE_LLM_BENCHMARK_FILEPATH", os.path.join(home_path, "logs/benchmark.jsonl"))


def get_benchmark_reserving_ratio():
    return float(os.getenv("MINDIE_LLM_BENCHMARK_RESERVING_RATIO", "0.1"))


def get_use_mb_swapper():
    value = os.getenv("MINDIE_LLM_USE_MB_SWAPPER", os.getenv("MIES_USE_MB_SWAPPER", "0"))
    if value not in ["0", "1"]:
        raise ValueError("MINDIE_LLM_USE_MB_SWAPPER and MIES_USE_MB_SWAPPER should be 0 or 1")
    if value == "1":
        return True
    else:
        return False


def get_performance_prefix_tree():
    return os.getenv("PERFORMANCE_PREFIX_TREE_ENABLE", "0") == "1"


def get_visible_devices():
    value = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    if value is None or value.strip() == "":
        return None
    try:
        return list(map(int, value.split(",")))
    except ValueError as e:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES should be in format {device_id},{device_id},...,{device_id}") from e


@dataclass
class EnvLLMVar:
    """
    Environment Variables
    """

    # Size of dynamically allocated memory pool during model runtime (unit: GB)
    reserved_memory_gb: int = field(default_factory=lambda: int(os.getenv("RESERVED_MEMORY_GB", "0")))

    # Which devices to use
    visible_devices: list[int] | None = field(default_factory=get_visible_devices)
    # Whether to bind CPU cores
    bind_cpu: bool = field(default_factory=lambda: os.getenv("BIND_CPU", "1") == "1")

    memory_fraction: float = field(default_factory=lambda: float(os.getenv("NPU_MEMORY_FRACTION", "0.8")))

    # Whether to record performance data required for service benchmarking
    benchmark_enable: bool = field(default_factory=lambda: os.getenv("MINDIE_LLM_BENCHMARK_ENABLE", "0") == "1")
    benchmark_enable_async: bool = field(default_factory=lambda: os.getenv("MINDIE_LLM_BENCHMARK_ENABLE", "0") == "2")
    benchmark_filepath: str = field(default_factory=get_benchmark_filepath)
    benchmark_reserving_ratio: float = field(default_factory=get_benchmark_reserving_ratio)

    # Whether to enable memory bridge based swapper optimization
    use_mb_swapper: bool = field(default_factory=get_use_mb_swapper)

    # Select post-processing acceleration mode
    speed_mode_type: int = field(default_factory=lambda: int(os.getenv("POST_PROCESSING_SPEED_MODE_TYPE", "0")))

    rank: int = field(default_factory=lambda: int(os.getenv("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.getenv("WORLD_SIZE", "1")))

    performance_prefix_tree: bool = field(default_factory=get_performance_prefix_tree)

    def __post_init__(self):
        # Validation
        if self.reserved_memory_gb >= 64 or self.reserved_memory_gb < 0:
            raise ValueError("RESERVED_MEMORY_GB should be in the range of 0 to 64, 64 is not inclusive.")

        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError("NPU_MEMORY_FRACTION should be in the range of 0 to 1.0, 0.0 is not inclusive.")

        if self.world_size < 0:
            raise ValueError("WORLD_SIZE should not be a number less than 0.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError("LOCAL_RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")

        if len(self.benchmark_filepath) > 1024:
            raise ValueError("The path length of MINDIE_LLM_BENCHMARK_FILEPATH exceeds the limit 1024 characters.")

        if not pathlib.Path(self.benchmark_filepath).is_absolute():
            raise ValueError("The path of MINDIE_LLM_BENCHMARK_FILEPATH must be absolute.")

        if pathlib.Path(self.benchmark_filepath).is_dir():
            raise ValueError("The path of MINDIE_LLM_BENCHMARK_FILEPATH is a directory and not a file.")

        if pathlib.Path(self.benchmark_filepath).exists():
            if not os.access(pathlib.Path(self.benchmark_filepath), os.R_OK):
                raise PermissionError("The path of MINDIE_LLM_BENCHMARK_FILEPATH is not permitted to be read.")


LLMENV = EnvLLMVar()
