# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import os
from dataclasses import dataclass
import ipaddress
from typing import Optional

from mindie_llm.runtime.utils.helpers.safety.file import safe_open

DEVICE = "device"
SERVER_LIST = "server_list"


@dataclass
class EnvVar:
    """
    Environment variables configuration for Mindie LLM runtime.

    This class parses and validates environment variables required for distributed training
    on Huawei Ascend NPU devices. It includes checks for valid ranges, IP addresses, and
    rank table file structure.
    """
    visible_devices: Optional[str] = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    bind_cpu: bool = os.getenv("BIND_CPU", "1") == "1"
    cpu_binding_num: Optional[str] = os.getenv("CPU_BINDING_NUM", None)
    rank_table_file: str = os.getenv("RANK_TABLE_FILE", "")
    master_ip: Optional[str] = os.getenv("MASTER_IP", "127.0.0.1")
    master_port: Optional[str] = os.getenv("MASTER_PORT", None)

    def __post_init__(self) -> None:
        """
        Post-initialization validation for all environment variables.

        This method performs range checks, IP validation, rank table validation, and other
        sanity checks on the environment variables. It raises appropriate ValueError exceptions
        if any validation fails.
        """

        if self.cpu_binding_num is not None:
            if not isinstance(self.cpu_binding_num, int):
                raise ValueError("CPU_BINDING_NUM must be an integer or None")

        if self.visible_devices is not None:
            try:
                self.visible_devices = [int(x) for x in self.visible_devices.split(',')]
            except ValueError as e:
                raise ValueError("ASCEND_RT_VISIBLE_DEVICES should be in format "
                                 "{device_id},{device_id},...,{device_id}") from e

        # Check rank table file structure
        self.check_rank_table(self.rank_table_file)
        
        if self.rank_table_file:
            rank_table_file = json.load(safe_open(self.rank_table_file, 'r', encoding='utf-8'))
            self.master_ip = rank_table_file[SERVER_LIST][0]['container_ip']
        
        if self.master_ip is not None and not self.is_valid_ip(self.master_ip):
            raise ValueError(f"MASTER_IP '{self.master_ip}' is invalid")

        if self.master_port is not None:
            self.master_port = int(self.master_port)
        if self.master_port is not None and (self.master_port < 0 or self.master_port > 65535):
            raise ValueError(f"MASTER_PORT {self.master_port} must be in range [0, 65535]")


    @staticmethod
    def is_valid_ip(ip: str) -> bool:
        """
        Validate if the given string is a valid IPv4 or IPv6 address.

        Args:
            ip (str): IP address string to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            addr = ipaddress.ip_address(ip)
            if addr.is_unspecified:
                return False
            return True
        except ValueError:
            return False

    def check_rank_table(self, rank_table_file: str) -> None:
        """
        Validate the structure of the rank table file.

        Args:
            rank_table_file (str): Path to the rank table JSON file.

        Raises:
            ValueError: If the rank table file has invalid structure or values.
        """
        if not rank_table_file:
            return

        with safe_open(rank_table_file, 'r', encoding='utf-8') as device_file:
            ranktable = json.load(device_file)

        # Calculate total world_size from rank table
        world_size = 0
        for server in ranktable[SERVER_LIST]:
            world_size += len(server[DEVICE])

        # Validate rank_id values
        for server in ranktable[SERVER_LIST]:
            for device in server[DEVICE]:
                rank_id = int(device["rank_id"])
                if rank_id >= world_size:
                    raise ValueError(f"rank_id {rank_id} must be less than world_size {world_size}")

        # Validate device_ip and server_id as valid IPs
        for server in ranktable[SERVER_LIST]:
            if not self.is_valid_ip(server["server_id"]):
                raise ValueError(f"Invalid server_id: {server['server_id']}")

            for device in server[DEVICE]:
                if not self.is_valid_ip(device["device_ip"]):
                    raise ValueError(f"Invalid device_ip: {device['device_ip']}")


ENV = EnvVar()
