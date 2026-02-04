# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from enum import Enum
import threading
from functools import cached_property

import torch
import acl

from mindie_llm.runtime.utils.helpers.command_executor import execute_command
from mindie_llm.runtime.utils.helpers.env import ENV
from mindie_llm.utils.log.logging import logger


class Topo(str, Enum):
    """Enumerates supported inter-device communication topologies on Ascend NPUs."""
    pcie = "pcie"
    hccs = "hccs"
    xlink = "xlink"


class DeviceType(str, Enum):
    """Enumerates supported Ascend NPU device types based on SoC name."""
    ASCEND_910B = "ASCEND_910B"
    ASCEND_910_93 = "ASCEND_910_93"
    ASCEND_310P = "ASCEND_310P"
    ASCEND_910_95 = "ASCEND_910_95"


@dataclass
class _DeviceInfo:
    """
    Dataclass representing detailed information about a single Ascend NPU device.

    This class parses a raw line of output from the `npu-smi info -m` command
    and extracts structured fields such as NPU ID, chip ID, logical chip ID,
    and chip name. The logical chip ID may be numeric or alphanumeric
    (e.g., 'N/A' or '0'), so it is stored as an `int | str`.

    Attributes:
        _info_line (str): Raw input line from `npu-smi info -m`; used only during initialization.
        npu_id (int): Physical NPU index assigned by the system.
        chip_id (int): Physical chip index on the NPU module.
        chip_logic_id (int | str): Logical chip ID; converted to int if numeric, otherwise kept as str.
        chip_name (str): Human-readable name of the chip (e.g., "Ascend910B").

    Note:
        This class is not a singleton and is intended to be instantiated once per device line.
        Parsing occurs in `__post_init__`, and malformed lines may cause exceptions—
        callers should handle this (e.g., via try-except in parsing loops).
    """
    _info_line: str = ""
    npu_id: int = 0
    chip_id: int = 0
    chip_logic_id: int | str = 0
    chip_name: str = ""

    def __post_init__(self):
        """
        Parse the raw `_info_line` into structured device attributes.

        Splits the input line into four fields using whitespace as delimiter (maxsplit=3).
        Converts `npu_id` and `chip_id` to integers.
        Converts `chip_logic_id` to int only if it is numeric; otherwise, leaves it as string.

        Raises:
            ValueError: If the line does not contain at least four whitespace-separated fields.
        """
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = self._info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


class _NPUNodeInfo:
    """Singleton manager for global NPU device topology and PCIe bus information.

    This class provides lazy-loaded, cached access to:
      - Full device metadata (via `npu-smi info -m`)
      - Per-device PCIe bus identifiers (via `npu-smi info -t board`)

    It ensures `npu-smi` is invoked at most once per information type,
    improving efficiency in multi-call scenarios (e.g., during CPU binding setup).

    The singleton pattern guarantees a single source of truth for NPU topology
    across the process lifetime.

    Attributes:
        _device_info_map (dict[int, _DeviceInfo]): Cache of logical chip ID → device info.
        _pcie_info_cache (dict[int, str]): Cache of logical chip ID → PCIe bus ID (e.g., "0000:1a:00.0").
        _initialized (bool): Flag to prevent re-initialization in __init__.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implements thread-safe singleton creation using double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize internal caches; safe to call multiple times due to `_initialized` guard."""
        if self._initialized:
            return
        self._device_info_map: dict[int, _DeviceInfo] = {}
        self._pcie_info_cache: dict[int, str] = {}
        self.soc_name: str = ""
        self.only_supports_nz: bool = False
        self.need_nz: bool = False  # Legacy compatibility flag for ATB models; runtime does not use this

        self.soc_name = torch.npu.get_device_properties().name

        # SoC names known to support only NZ format and not BF16
        nz_only_names = {
            "Ascend910PremiumA",
            "Ascend910ProA",
            "Ascend910A",
            "Ascend910ProB",
            "Ascend910B",
            "Ascend310P1",
            "Ascend310P2",
            "Ascend310P3",
            "Ascend310P4",
            "Ascend310P5",
            "Ascend310P7"
        }
        if self.soc_name in nz_only_names:
            self.only_supports_nz = True
            self.need_nz = True
        self._visible_device_ids = None
        self._initialized = True

    @staticmethod
    def is_support_hccs() -> bool:
        """
        Determines whether the system supports HCCS or XLink topology via npu-smi command.

        Executes `npu-smi info -t topo` and checks if "hccs" or "xlink" appears in the topology legend section.

        Returns:
            bool: True if HCCS or XLink is detected; False otherwise.
        """
        npu_smi_info = execute_command(["npu-smi", "info", "-t", "topo"])
        # Find the position of "Legend" to avoid false positives in device table
        legend_index = npu_smi_info.find("Legend")
        # Check if either HCCS or XLink appears before the Legend section
        if Topo.hccs in npu_smi_info[:legend_index].lower() or \
            Topo.xlink in npu_smi_info[:legend_index].lower():
            return True
        return False

    @cached_property
    def visible_device_ids(self):
        """
        Determine and cache the list of visible physical NPU IDs for the current process group.

        If `visible_device_ids` is already set, this method returns immediately (idempotent).
        Otherwise, it resolves visible devices either from:
          - `ENV.visible_devices` (if set, assumed to be logical chip IDs), or
          - auto-detected logical chip IDs from `npu-smi info -m`.

        It then maps each logical device to its corresponding physical `npu_id` using
        `get_npu_node_info().get_device_info_map()` and stores the result in `visible_device_ids`.

        Note:
            This method assumes that logical device indices (e.g., from `ENV.visible_devices`)
            correspond to keys in the device info map returned by `get_npu_node_info().get_device_info_map()`.
        """
        if self._visible_device_ids is None:
            if ENV.visible_devices is None:
                # Retrieve all chip logic IDs via the 'npu-smi info -m' command.
                devices = sorted(list(get_npu_node_info().get_device_info_map().keys()))
            else:
                devices = ENV.visible_devices
            self._visible_device_ids = devices
        return self._visible_device_ids

    def get_device_info_map(self) -> dict[int, _DeviceInfo]:
        """Retrieve a mapping of logical NPU chip IDs to their device information.

        Executes the system command `npu-smi info -m`, skips the header line,
        and parses each subsequent non-empty line into an `_DeviceInfo` object.
        Only devices with a **numeric** `chip_logic_id` are included in the returned map,
        as they are assumed to represent active, addressable compute units.

        Returns:
            dict[int, _DeviceInfo]: A dictionary where keys are integer logical chip IDs
            (e.g., 0, 1, 2...) and values are corresponding device metadata objects.

        Note:
            - Lines that fail to parse (e.g., due to unexpected format) are silently skipped.
            - Devices with non-numeric `chip_logic_id` (e.g., "N/A") are excluded from the map.
            - Relies on the `npu-smi` CLI tool being available in the system PATH.
            - Result is cached in `self._device_info_map` after first call.
        """
        if self._device_info_map:
            return self._device_info_map

        device_info_map = {}
        device_map = execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
        for line in device_map:
            line = line.strip()
            if not line:  # Skip empty lines.
                continue
            try:
                device_info = _DeviceInfo(line)
                if isinstance(device_info.chip_logic_id, int):
                    device_info_map[device_info.chip_logic_id] = device_info
            except (ValueError, IndexError):
                pass
        self._device_info_map = device_info_map
        return self._device_info_map

    def get_pcie_info(self, devices: list[int], keyword: str = "PCIeBusInfo") -> dict[int, str]:
        """Retrieve PCIe bus information for a list of logical NPU devices.

        Queries `npu-smi info -t board` for each device to extract its PCIe bus ID
        (e.g., "0000:XX:XX.X") using the specified keyword. The result maps logical
        device IDs (chip logic IDs) to their PCIe bus strings.

        Args:
            devices (list[int]): List of logical NPU device IDs (e.g., [0, 1, 2]).
            keyword (str): The keyword prefix to look for in `npu-smi` board output.
                Default is "PCIeBusInfo".

        Returns:
            dict[int, str]: Mapping from logical device ID to PCIe bus ID string.

            Example:
                {
                    0: "0000:1a:00.0",
                    1: "0000:1b:00.0",
                    2: "0000:3a:00.0"
                }

        Raises:
            RuntimeError: If device info cannot be retrieved for any device.

        Note:
            Whitespace in `npu-smi` output is normalized (removed) to handle formatting
            variations across hardware or driver versions.
        """
        device_info_map = self.get_device_info_map()
        device_pcie = {}
        for device in devices:
            device_info = device_info_map.get(device)
            if not device_info:
                warn_msg = f"Can not get device info for device {device}, skipping PCIe binding for this device."
                logger.warning(warn_msg)
                raise RuntimeError(warn_msg)
            pcie_info = execute_command(["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}",
                                         "-c", f"{device_info.chip_id}"]).strip().split("\n")
            for _ in pcie_info:
                # Normalize the output string by removing spaces to handle hardware variations
                line = ''.join(_.split())
                if line.startswith(keyword):
                    pcie_id = line[len(keyword) + 1:]
                    device_pcie[device] = pcie_id
                    self._pcie_info_cache[device] = pcie_id  # cache for future calls
                    break

        return device_pcie

    def get_device_type(self) -> DeviceType:
        """
        Determines the Ascend device type based on the detected SoC name.
        Maps numeric SoC name lists to known Ascend device types.

        Returns:
            DeviceType: The corresponding device type.

        Raises:
            RuntimeError: If the SoC name is unrecognized or unsupported.
        """
        if self.soc_name in {
            "Ascend910B1",
            "Ascend910B2",
            "Ascend910B2C",
            "Ascend910B3",
            "Ascend910B4",
            "Ascend910B4-1",
        }:
            return DeviceType.ASCEND_910B
        elif self.soc_name in {
            "Ascend910_9391",
            "Ascend910_9392",
            "Ascend910_9381",
            "Ascend910_9382",
            "Ascend910_9372",
            "Ascend910_9362"
        }:
            return DeviceType.ASCEND_910_93
        elif self.soc_name in {
            "Ascend310P1",
            "Ascend310P2",
            "Ascend310P3",
            "Ascend310P4",
            "Ascend310P5",
            "Ascend310P7"
        }:
            return DeviceType.ASCEND_310P
        else:
            raise RuntimeError(f"Can not support soc_name: {self.soc_name}.")


def get_npu_node_info() -> _NPUNodeInfo:
    """Returns the singleton instance of _NPUNodeInfo.

    This is the recommended way to access global NPU device topology information.
    The instance is lazily initialized on first call.

    Returns:
        _NPUNodeInfo: The singleton device manager instance.
    """
    return _NPUNodeInfo()


class _NPUHbmInfo:
    """
    Utility class to manage and query HBM (High Bandwidth Memory) information for Ascend NPU devices.

    This class caches global HBM capacity and usage metrics across distributed processes.
    It supports device visibility control via environment or auto-detection, and handles
    chip-specific memory naming conventions (e.g., DDR vs HBM) based on SoC type and data format.

    Attributes:
        hbm_capacity (int | None): Total HBM capacity in bytes, cached globally after first query.
        hbm_usage (int | None): Not used in current implementation; may be reserved for future use.
            (Note: `get_hbm_usage` returns a float ratio, not stored in this attribute.)

    Class Behavior:
        - Uses class-level caching for efficiency in multi-process or multi-call scenarios.
        - Relies on `npu-smi` CLI for low-level hardware queries.
        - Adapts parsing logic based on SoC name (e.g., Ascend310B1) and data layout (`need_nz` flag).
    """
    _hbm_capacity: int = None

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> '_NPUHbmInfo':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_hbm_capacity_usage() -> tuple[int, float]:
        """
        Retrieve total HBM (or DDR) capacity and current HBM (or DDR) usage ratio for the NPU.

        Returns:
            tuple[int, float]: Total memory capacity in bytes, 
                Memory usage ratio, e.g., 0.42 for 42% usage (adjusted to (value + 1) / 100).
        """
        free_mem, total_mem, _ = acl.rt.get_mem_info(1)
        peak_mem = total_mem - free_mem
        return total_mem, float(peak_mem) / total_mem

    @classmethod
    def get_hbm_capacity(cls) -> int:
        """
        Retrieve total HBM (or DDR) capacity for the NPU assigned to the given rank.

        Returns:
            int: Total memory capacity in bytes.

        Note:
            The capacity is cached at the class level and assumed uniform across all ranks.
        """
        if cls._hbm_capacity is None:
            _capacity = cls.get_hbm_capacity_usage()[0]
            cls._hbm_capacity = _capacity
        return cls._hbm_capacity
         
    @classmethod
    def get_hbm_usage(cls) -> float:
        """
        Retrieve current HBM (or DDR) usage ratio for the NPU assigned to the given rank.

        Returns:
            float: Memory usage ratio, e.g., 0.42 for 42% usage (adjusted to (value + 1) / 100).
        """
        return cls.get_hbm_capacity_usage()[1]


def get_npu_hbm_info() -> _NPUHbmInfo:
    """Returns the singleton instance of _NpuHbmInfo."""
    return _NPUHbmInfo()
