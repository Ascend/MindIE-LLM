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

import ipaddress
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from mindie_llm.model_wrapper.utils.config import DmiConfig
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.file_utils import safe_open

ROLE_MASTER = 'master'
ROLE_SLAVE = 'slave'
SERVER_LIST = 'server_list'
SERVER_COUNT = 'server_count'
HOST = 'host'
DEVICE = 'device'


@dataclass
class LwdCommConfig:
    rank: Optional[int] = None
    npu_device_id: Optional[int] = None
    role_type: Optional[str] = None
    edge_ip_address: Optional[str] = None
    cloud_ip_addresses: Optional[List[str]] = None
    hccl_comm_edge_ip_port: Optional[str] = None
    tcp_comm_cloud_ip_port: Optional[List[str]] = None
    npu_edge_num: Optional[int] = None
    npu_cloud_num: Optional[int] = None
    max_input_len: int = 1
    multi_nodes_infer_enabled: bool = False
    multi_nodes_ctrl_port: Optional[str] = None
    multi_nodes_is_master: bool = False
    comm_group_size: int = 1    # 通信域划分数量, 包括按dp和cp进行划分, 两者不兼容开启
    npu_net_host: bool = False


class LwdCommunicationManager:
    __slots__ = (
        "config",
        "ctrl_comm",
        "data_comm",
        "generator",
    )

    def __init__(self):
        self.config = LwdCommConfig()
        self.ctrl_comm = None
        self.data_comm = None
        self.generator = None

    @staticmethod
    def is_valid_ip(ip):
        if ip is None:
            return False
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_port(port):
        return port >= 1024 and port <= 65535

    def communication_config_check(self, model_config: DmiConfig) -> bool:
        if self.config.role_type is None or self.config.role_type not in [ROLE_MASTER, ROLE_SLAVE]:
            logger.error("[layerwiseDisaggregated] The configuration is invalid" \
                        "because the role is %s.", self.config.role_type)
            return False

        if len(self.config.cloud_ip_addresses) == 0 or len(self.config.cloud_ip_addresses) > 2:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "layerwiseDisaggregatedSlaveIpAddress " \
                        "only support 1 to 2 ip address.")
            return False

        edge_ip_valid = LwdCommunicationManager.is_valid_ip(self.config.edge_ip_address)
        cloud_ip_valid = True
        for cloud_ip_address in self.config.cloud_ip_addresses:
            cloud_ip_valid = LwdCommunicationManager.is_valid_ip(cloud_ip_address)
            if not cloud_ip_valid:
                break

        if not edge_ip_valid or not cloud_ip_valid:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "layerwiseDisaggregatedMasterIpAddress is %s and " \
                        "layerwiseDisaggregatedSlaveIpAddress is %s.", self.config.edge_ip_address,
                        self.config.cloud_ip_addresses)
            return False

        if 'models' not in model_config.model_config:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "there is no models configuration."
                        )
            return False
        models = json.loads(model_config.model_config['models'])

        if 'layerwiseDisaggregatedMasterDeviceNum' not in models or \
            'layerwiseDisaggregatedSlaveDeviceNum' not in models:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because" \
                        "models are missing configuration items.")
            return False

        npu_edge_num = models['layerwiseDisaggregatedMasterDeviceNum']
        npu_cloud_num = models['layerwiseDisaggregatedSlaveDeviceNum']

        if npu_edge_num not in [2] or npu_cloud_num not in [8, 16]:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "the npuEdgeNum or npuCloudNum is illegal."
                        )
            return False

        if self.config.tcp_comm_cloud_ip_port is None or self.config.hccl_comm_edge_ip_port is None:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "there is no layerwiseDisaggregatedDataPort " \
                        "or layerwiseDisaggregatedCrtlPort configuration."
                        )
            return False

        if len(self.config.tcp_comm_cloud_ip_port) != 2:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "layerwiseDisaggregatedCrtlPort need two port."
                        )
            return False

        tcp_comm_cloud_ip_port1 = self.config.tcp_comm_cloud_ip_port[0]
        tcp_comm_cloud_ip_port2 = self.config.tcp_comm_cloud_ip_port[1]

        if not tcp_comm_cloud_ip_port1.isdigit() or not tcp_comm_cloud_ip_port2.isdigit():
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "the layerwiseDisaggregatedCrtlPort is not a number."
                        )
            return False

        if not LwdCommunicationManager.is_valid_port(int(tcp_comm_cloud_ip_port1)) or \
            not LwdCommunicationManager.is_valid_port(int(tcp_comm_cloud_ip_port2)):
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "the TCP port is not within the range of 1024 to 65535."
                        )
            return False

        if not LwdCommunicationManager.is_valid_port(int(self.config.hccl_comm_edge_ip_port)):
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "the HCCL port is not within the range of 1024 to 65535."
                        )
            return False

        self.config.tcp_comm_cloud_ip_port = "[" + tcp_comm_cloud_ip_port1 + ", " + tcp_comm_cloud_ip_port2 + "]"

        self.config.npu_edge_num = npu_edge_num
        self.config.npu_cloud_num = npu_cloud_num

        if self.config.multi_nodes_infer_enabled:
            if len(self.config.cloud_ip_addresses) < 2:
                logger.error("[layerwiseDisaggregated] The configuration is invalid because " \
                            "the layerwiseDisaggregatedSlaveIpAddress is a single ip address."
                            )
                return False
            if self.config.role_type == ROLE_SLAVE and self.config.multi_nodes_is_master is None:
                logger.error("[layerwiseDisaggregated] The configuration is invalid because " \
                            "the layerwiseDisaggregatedMultiNodesIsMaster is not a number."
                            )
                return False
            if self.config.role_type == ROLE_SLAVE and self.config.multi_nodes_ctrl_port is None:
                logger.error("[layerwiseDisaggregated] The configuration is invalid because " \
                            "the layerwiseDisaggregatedMultiNodesCtrlPort is not a number."
                            )
                return False

            if not LwdCommunicationManager.is_valid_port(int(self.config.multi_nodes_ctrl_port)):
                logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "the TCP port is not within the range of 1024 to 65535."
                        )
                return False
        return True

    def communication_init(self) -> bool:
        self.ctrl_comm = self.generator.model_wrapper.model_runner.ctrl_comm
        self.data_comm = self.generator.model_wrapper.model_runner.data_comm
        if self.ctrl_comm is None or self.data_comm is None:
            return False

        multi_nodes_infer_args = None
        server_ip = self.config.cloud_ip_addresses[0]
        if self.config.multi_nodes_infer_enabled:
            multi_nodes_infer_args = {
                'is_master': self.config.multi_nodes_is_master,
                'cloud_ctrl_port': self.config.multi_nodes_ctrl_port,
                'cloud_ctrl_address': self.config.cloud_ip_addresses[0],
                'comm_group_size': self.config.comm_group_size,
                'max_input_len': self.config.max_input_len,
            }
            if self.config.role_type == ROLE_MASTER:
                if self.config.rank < self.config.comm_group_size:
                    server_ip = self.config.cloud_ip_addresses[self.config.rank]
            else:
                if not self.config.multi_nodes_is_master:
                    server_ip = self.config.cloud_ip_addresses[1]
        self.ctrl_comm.init_tcp_link(rank=self.config.rank, role=self.config.role_type,
                                     server_ip=server_ip, server_port=self.config.tcp_comm_cloud_ip_port,
                                     multi_nodes_infer_args=multi_nodes_infer_args)
        if not self.ctrl_comm.is_edge_cloud_ctrl_comm_success():
            logger.error("[layerwiseDisaggregated] The tcp connection of LayerwiseDisaggregated is fail, " \
                        f"ip={self.config.edge_ip_address} port={self.config.tcp_comm_cloud_ip_port}.")
            return False

        if not self.data_comm.init_finish:
            if self.data_comm.get_lwd_rank_file() is None:
                self.data_comm.init_hccl()
                if self.data_comm.init_finish:
                    self.data_comm.hccl_comm_warmup(self.generator.hidden_size)
            else:
                return False

        self.data_comm.initialize(self.config.npu_device_id)
        self.data_comm.hidden_malloc()

        return True

    def parse(self, model_config: DmiConfig):
        self.config.role_type = model_config.parse("layerwiseDisaggregatedRoleType", required=False, default_value=None)
        self.config.edge_ip_address = model_config.parse("layerwiseDisaggregatedMasterIpAddress", \
                                                required=False)
        self.config.cloud_ip_addresses = model_config.parse_list("layerwiseDisaggregatedSlaveIpAddress", \
                                                required=False)
        self.config.hccl_comm_edge_ip_port = model_config.parse("layerwiseDisaggregatedDataPort", \
                                                required=False)
        self.config.tcp_comm_cloud_ip_port = model_config.parse_list("layerwiseDisaggregatedCrtlPort", \
                                                required=False)
        self.config.multi_nodes_infer_enabled = model_config.parse("layerwiseDisaggregatedMultiNodesInferEnabled", \
                                                required=False, default_value='false') == 'true'
        self.config.multi_nodes_ctrl_port = model_config.parse("layerwiseDisaggregatedMultiNodesCtrlPort", \
                                                required=False)
        self.config.multi_nodes_is_master = model_config.parse("lwd_multi_nodes_is_master", required=False, \
                                                default_value='false') == 'true'
        self.config.rank = model_config.local_rank
        self.config.npu_device_id = model_config.npu_device_id
        if model_config.dp_size > 1 and model_config.cp_size > 1:
            raise ValueError("[layerwiseDisaggregated] dp and cp can't be enabled simultaneously.")
        self.config.comm_group_size = max(model_config.dp_size, model_config.cp_size)
        self.config.max_input_len = max(model_config.max_seq_len, model_config.max_prefill_tokens)

    def lwd_ranktable_parse_check_proc(self, lwd_rank_table) -> bool:
        if SERVER_LIST not in lwd_rank_table or SERVER_COUNT not in lwd_rank_table:
            logger.error("[layerwiseDisaggregated] The lwd_rank_table_file is missing configuration item " \
                         "(%s or %s).", SERVER_LIST, SERVER_COUNT)
            return False

        server_list = lwd_rank_table[SERVER_LIST]
        server_count = int(lwd_rank_table[SERVER_COUNT])
        if int(lwd_rank_table[SERVER_COUNT]) != len(server_list):
            logger.error("[layerwiseDisaggregated] The lwd_rank_table_file contains incorrect configuration item, " \
                         "server_list contains %d servers, server_count is %d.", len(server_list), server_count)
            return False

        device_num_list = []
        for server in server_list:
            if DEVICE not in server:
                logger.error("[layerwiseDisaggregated] The lwd_rank_table_file is missing configuration item (device).")
                return False
            device_num_list.append(len(server[DEVICE]))
            para_net_pos = server.get("para_net_position")
            if para_net_pos == HOST:
                self.config.npu_net_host = True if self.config.role_type == ROLE_MASTER else False

        if device_num_list[-1] != self.config.npu_edge_num:
            logger.error("[layerwiseDisaggregated] The master configuration in the lwd_rank_table_file " \
                         "should be placed after the slave configuration.")
            return False

        return True

    def lwd_ranktable_parse_check(self) -> bool:
        lwd_rank_table_file = os.environ.get('LAYERWISE_DISAGGREGATED_RANK_TABLE_FILE')
        if lwd_rank_table_file is None:
            return True

        try:
            with safe_open(lwd_rank_table_file, 'r', encoding='utf-8') as f:
                lwd_rank_table = json.load(f)
        except FileNotFoundError as e:
            logger.error("[layerwiseDisaggregated] %s", lwd_rank_table_file, e)
            return False
        except json.JSONDecodeError as e:
            logger.error("[layerwiseDisaggregated] The file %s is invalid json file, %s", lwd_rank_table_file, e)
            return False

        return self.lwd_ranktable_parse_check_proc(lwd_rank_table)

    def communication_config_verify(self, model_config: DmiConfig) -> bool:
        self.parse(model_config)
        ret1 = self.communication_config_check(model_config)
        ret2 = self.lwd_ranktable_parse_check()
        if ret1 and ret2:
            comm_args = {
                'edge_ip_address': self.config.edge_ip_address,
                'cloud_ip_address': self.config.cloud_ip_addresses,
                'edge_npu_num': self.config.npu_edge_num,
                'cloud_npu_num': self.config.npu_cloud_num,
                'data_port': self.config.hccl_comm_edge_ip_port,
                'multi_nodes_ctrl_port': self.config.multi_nodes_ctrl_port,
                'multi_nodes_infer_enabled': self.config.multi_nodes_infer_enabled,
                'multi_nodes_is_master': self.config.multi_nodes_is_master,
                'comm_group_size': self.config.comm_group_size,
                'max_input_len': self.config.max_input_len,
                'npu_net_host': self.config.npu_net_host}
            model_config.model_config['lwd_comm_args'] = comm_args
            return True
        else:
            return False

    def initialize(self, model_config: DmiConfig, initialize_result, generator):
        self.generator = generator

        if not self.communication_init():
            initialize_result["status"] = "error"

        logger.info(
                    "[layerwiseDisaggregated] global rank:%s: return initialize success " \
                    "result: %s", self.config.rank, initialize_result
                    )
