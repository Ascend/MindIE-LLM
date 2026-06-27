# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import subprocess
from dataclasses import dataclass
from itertools import accumulate
from typing import List, Dict, Tuple, Union

import psutil
import torch_npu

from .env import ENV
from .log import logger


def execute_command(cmd_list):
    with subprocess.Popen(cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        out, err = p.communicate(timeout=1000)
    res = out.decode()
    return res


def _parse_mem_line_free_b(out: str) -> Tuple[int, int]:
    """Parse first `Mem:` line from `free -b` output: (total_bytes, available_bytes)."""
    for line in out.split("\n"):
        line = line.strip()
        if not line.startswith("Mem:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            break
        total = int(parts[1])
        if total == 0:
            break
        if len(parts) >= 7:
            available = int(parts[6])
        else:
            available = int(parts[3])
        return total, available
    raise ValueError("free fallback failed")


def _host_mem_total_bytes_from_free():
    """Host DRAM total from `free -b` Mem line (bytes)."""
    out = execute_command(["free", "-b"])
    total, _ = _parse_mem_line_free_b(out)
    return total


def _host_mem_remain_ratio_from_free():
    """Host DRAM usage ratio from `free -b` (byte counts, stable to parse)."""
    out = execute_command(["free", "-b"])
    total, available = _parse_mem_line_free_b(out)
    return 1 - (available / total)


@dataclass
class DeviceInfo:
    _info_line: str = ""
    npu_id: int = 0
    chip_id: int = 0
    chip_logic_id: Union[int, str] = 0
    chip_name: str = ""

    def __post_init__(self):
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = self._info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


class NpuHbmInfo:
    visible_npu_ids: List = None
    hbm_capacity: int = None
    hbm_usage: int = None

    @classmethod
    def set_visible_devices(cls, world_size):
        if cls.visible_npu_ids:
            return
        if ENV.visible_devices is None:
            # Retrieve all chip logic IDs via the 'npu-smi info -m' command.
            devices = sorted(list(_get_device_map_info().keys()))
        else:
            devices = ENV.visible_devices
        device_map_info = _get_device_map_info()
        npu_ids = []
        for device in devices:
            device_info = device_map_info.get(device)
            npu_ids.append(device_info.npu_id)
        cls.visible_npu_ids = npu_ids

    @classmethod
    def get_hbm_capacity(cls, rank, world_size, need_nz):
        soc_version = torch_npu._C._npu_get_soc_version()
        if cls.hbm_capacity:
            return cls.hbm_capacity
        if not cls.visible_npu_ids:
            cls.set_visible_devices(world_size)

        from .initial import NPUSocInfo

        # If it is an RC device, skip npu-smi and use free -b directly
        if NPUSocInfo.is_rc_device():
            logger.info("NPU device is working in Root Complex (RC) mode.")
            try:
                cls.hbm_capacity = _host_mem_total_bytes_from_free()
                return cls.hbm_capacity
            except ValueError as e:
                raise ValueError(
                    "Failed to parse host memory capacity via 'free -b' in RC mode. "
                    "Please ensure the 'procps' package is installed (e.g., 'apt-get install -y procps') "
                    "and the container is started with sufficient privileges to execute 'free -b'."
                ) from e

        npu_id = cls.visible_npu_ids[rank]
        memory_info = execute_command(["npu-smi", "info", "-i", f"{npu_id}", "-t", "memory"]).split("\n")[1:]
        if soc_version == 240:
            hbm_capacity_key = "Capacity(MB)"
        elif not need_nz:
            hbm_capacity_key = "HBM Capacity(MB)"
        else:
            hbm_capacity_key = "DDR Capacity(MB)"
        for line in memory_info:
            try:
                key, value = line.strip().split(":", 2)
                if key.strip() == hbm_capacity_key:
                    cls.hbm_capacity = int(value.strip()) * 1024 * 1024
                    return cls.hbm_capacity
            except ValueError:
                pass  # Skip the invalid lines silently while searching for HBM capacity
        raise ValueError("not found valid hbm capactiy")

    @classmethod
    def get_hbm_usage(cls, rank, world_size, need_nz):
        if cls.hbm_usage:
            return cls.hbm_usage
        if not cls.visible_npu_ids:
            cls.set_visible_devices(world_size)

        from .initial import NPUSocInfo

        # If it is an RC device, skip npu-smi and use free -b directly
        if NPUSocInfo.is_rc_device():
            logger.info("NPU device is working in Root Complex (RC) mode.")
            try:
                return _host_mem_remain_ratio_from_free()
            except ValueError as e:
                raise ValueError(
                    "Failed to parse host memory capacity via 'free -b' in RC mode. "
                    "Please ensure the 'procps' package is installed (e.g., 'apt-get install -y procps') "
                    "and the container is started with sufficient privileges to execute 'free -b'."
                ) from e

        npu_id = cls.visible_npu_ids[rank]
        usage_info = execute_command(["npu-smi", "info", "-i", f"{npu_id}", "-t", "usages"]).split("\n")[1:]
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version == 240:
            hbm_capacity_key = "Memory Usage Rate(%)"
        elif not need_nz:
            hbm_capacity_key = "HBM Usage Rate(%)"
        else:
            hbm_capacity_key = "DDR Usage Rate(%)"
        for line in usage_info:
            try:
                key, value = line.strip().split(":", 2)
                if key.strip() == hbm_capacity_key:
                    hbm_usage = (float(value.strip()) + 1) / 100
                    return hbm_usage
            except ValueError:
                pass  # Skip the invalid lines silently while searching for HBM usage data
        raise ValueError("not found valid hbm usage")


def _get_device_map_info() -> Dict[int, DeviceInfo]:
    device_map_info = {}
    device_map = execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
    for line in device_map:
        line = line.strip()
        if not line:  # Skip empty lines.
            continue
        try:
            device_info = DeviceInfo(line)
            if isinstance(device_info.chip_logic_id, int):
                device_map_info[device_info.chip_logic_id] = device_info
        except (ValueError, IndexError):
            continue
    return device_map_info


def _get_pcie_info(devices: List[int], keyword="PCIeBusInfo"):
    device_map_info = _get_device_map_info()
    device_pcie_tbl = {}
    for device in devices:
        device_info = device_map_info.get(device)
        if not device_info:
            warn_msg = "Can not get device info for device, skipping PCIe binding for this device."
            logger.warning(warn_msg)
            raise RuntimeError(warn_msg)
        pcie_info = (
            execute_command(
                ["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}", "-c", f"{device_info.chip_id}"]
            )
            .strip()
            .split("\n")
        )
        for _ in pcie_info:
            # Normalize the output string by removing spaces to handle hardware variations
            # (e.g., 'PCIe Bus Info' vs. 'PCIeBusInfo').
            line = "".join(_.split())
            if line.startswith(keyword):
                device_pcie_tbl[device] = line[len(keyword) + 1 :]
                break

    return device_pcie_tbl


def _get_numa_info(pcie_tbl, keyword="NUMAnode"):
    device_numa_tbl = dict()  # key is device id, value is numa id
    numa_devices_tbl = dict()  # key is numa id, value is device id list

    for device, pcie_no in pcie_tbl.items():
        numa_info = execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).split("\n")
        for _ in numa_info:
            line = "".join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1 :])
                device_numa_tbl[device] = numa_id

                devices = numa_devices_tbl.get(numa_id, None)
                if devices is None:
                    numa_devices_tbl[numa_id] = list()

                numa_devices_tbl[numa_id].append(device)
                break

    return device_numa_tbl, numa_devices_tbl


def _get_numa_info_v2(devices: List[int], keyword="NUMAnode(s)") -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    In scenarios where NPU NUMA affinity is not supported, distribute devices evenly across each NUMA node.
    :param devices:
    :param keyword:
    :return:
    """
    # Optimize NUMA node detection.
    numa_nodes = 1
    numa_info = execute_command(["lscpu"]).split("\n")
    for _ in numa_info:
        line = "".join(_.split())
        if keyword not in line:
            continue
        numa_nodes = int(line[-1])
        break

    # Distribute devices evenly to each NUMA node.
    device_per_numa, tail_device = divmod(len(devices), numa_nodes)
    device_count_per_numa_list = [device_per_numa + (i < tail_device) for i in range(numa_nodes)]

    # Assign devices in ascending, continuous order to each NUMA node.
    ends = list(accumulate(device_count_per_numa_list))
    starts = [0] + ends[:-1]

    numa_devices_tbl = {ind: devices[start:end] for ind, (start, end) in enumerate(zip(starts, ends))}

    device_numa_tbl = {device: numa for numa, _devices in numa_devices_tbl.items() for device in _devices}

    return device_numa_tbl, numa_devices_tbl


def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    cpu_info = execute_command(["lscpu"]).split("\n")
    for _ in cpu_info:
        line = "".join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    warn_msg = "Cannot obtain CPU range for NUMA while executing `lscpu`."
                    logger.warning(warn_msg)
                    raise RuntimeError(warn_msg)

                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

            numa_id = int(split_info[0].replace(keyword1, "").replace(keyword2, ""))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


# 'CPU_BINDING_NUM' sets cores per process. If unset, it's calculated via NUMA utilization ratio.
# E.g., ratio 0.5 on 64 cores uses 32 cores, distributed evenly among NPUs on the NUMA node.
def bind_cpus(rank_id, ratio=0.5):
    visible_devices = ENV.visible_devices

    if visible_devices is None:
        # Retrieve all chip logic IDs via the 'npu-smi info -m' command.
        devices = sorted(list(_get_device_map_info().keys()))
    else:
        devices = ENV.visible_devices

    # Retrieve the mapping between NPU and PCIe.
    device_pcie_tbl = _get_pcie_info(devices)
    # Retrieve the mapping between NPU and NUMA based on PCIe information.
    device_numa_tbl, numa_devices_tbl = _get_numa_info(device_pcie_tbl)
    if not device_numa_tbl or not numa_devices_tbl:
        device_numa_tbl, numa_devices_tbl = _get_numa_info_v2(devices)
    # Retrieve CPU core allocation information corresponding to the used NUMA node.
    cpu_idx_tbl = _get_cpu_info(list(numa_devices_tbl.keys()))

    # NPU ID for the current rank.
    cur_device = devices[rank_id]
    # Retrieve the NUMA ID corresponding to the NPU.
    numa_id = device_numa_tbl.get(cur_device)

    # Retrieve information about NPUs sharing this NUMA node.
    shard_devices = numa_devices_tbl.get(numa_id)
    # Sort by NPU ID.
    shard_devices.sort()

    # Retrieve all CPU ID information on this NUMA node.
    all_cpus = cpu_idx_tbl.get(numa_id)
    logger.info(
        f"rank_id: {rank_id}, device_id: {cur_device}, "
        f"numa_id: {numa_id}, shard_devices: {shard_devices}, cpus: {all_cpus}"
    )

    cpu_nums = len(all_cpus)
    # Calculate the number of cores allocated to the NPU sharing this NUMA node.
    if ENV.cpu_binding_num is None:
        cpu_num_per_device = int(cpu_nums * ratio // len(shard_devices))
    else:
        cpu_num_per_device = int(ENV.cpu_binding_num)
        if len(shard_devices) * cpu_num_per_device > cpu_nums:
            err_msg = (
                f"CPU num in numa {numa_id} to assign {cpu_num_per_device} for every device is not enough, "
                f"please decrease the value of CPU_BINDING_NUM!"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if cpu_num_per_device < 0:
            err_msg = "CPU_BINDING_NUM should not be less than 0."
            logger.error(err_msg)
            raise ValueError(err_msg)

    # Retrieve the index information for this NPU.
    idx = shard_devices.index(cur_device)
    # Allocate the CPU IDs to be bound to this NPU.
    binding_cpus = [all_cpus[_] for _ in range(idx * cpu_num_per_device, (idx + 1) * cpu_num_per_device)]

    # cpu bind
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    logger.info(f"process {p.pid}, new_affinity is {new_affinity}, cpu count {cpu_num_per_device}")
