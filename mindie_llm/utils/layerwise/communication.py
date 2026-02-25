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
from mindie_llm.model_wrapper.utils.config import DmiConfig
from mindie_llm.utils.log.logging import logger

ROLE_MASTER = 'master'
ROLE_SLAVE = 'slave'


class LwdCommunicationManager:
    __slots__ = (
        "rank",
        "role_type",
        "edge_ip_address",
        "cloud_ip_addresses",
        "ctrl_comm",
        "data_comm",
        "hccl_comm_edge_ip_port",
        "tcp_comm_cloud_ip_port",
        "npu_edge_num",
        "npu_cloud_num",
        "generator",
        "multi_nodes_infer_enabled",
        "multi_nodes_ctrl_port",
        "multi_nodes_is_master",
        "multi_nodes_dp_size",
    )

    def __init__(self):
        self.rank = None
        self.role_type = None
        self.edge_ip_address = None
        self.cloud_ip_addresses = None
        self.ctrl_comm = None
        self.data_comm = None
        self.hccl_comm_edge_ip_port = None
        self.tcp_comm_cloud_ip_port = None
        self.npu_edge_num = None
        self.npu_cloud_num = None
        self.generator = None
        self.multi_nodes_infer_enabled = None
        self.multi_nodes_ctrl_port = None
        self.multi_nodes_is_master = None
        self.multi_nodes_dp_size = 1

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
        if self.role_type is None or self.role_type not in [ROLE_MASTER, ROLE_SLAVE]:
            logger.error("[layerwiseDisaggregated] The configuration is invalid" \
                        "because the role is %s.", self.role_type)
            return False

        if len(self.cloud_ip_addresses) == 0 or len(self.cloud_ip_addresses) > 2:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "layerwiseDisaggregatedSlaveIpAddress " \
                        "only support 1 to 2 ip address.")
            return False

        edge_ip_valid = LwdCommunicationManager.is_valid_ip(self.edge_ip_address)
        cloud_ip_valid = True
        for cloud_ip_address in self.cloud_ip_addresses:
            cloud_ip_valid = LwdCommunicationManager.is_valid_ip(cloud_ip_address)
            if not cloud_ip_valid:
                break

        if not edge_ip_valid or not cloud_ip_valid:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "layerwiseDisaggregatedMasterIpAddress is %s and " \
                        "layerwiseDisaggregatedSlaveIpAddress is %s.", self.edge_ip_address, self.cloud_ip_addresses)
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

        if self.tcp_comm_cloud_ip_port is None or self.hccl_comm_edge_ip_port is None:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "there is no layerwiseDisaggregatedDataPort " \
                        "or layerwiseDisaggregatedCrtlPort configuration."
                        )
            return False

        if len(self.tcp_comm_cloud_ip_port) != 2:
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "layerwiseDisaggregatedCrtlPort need two port."
                        )
            return False

        tcp_comm_cloud_ip_port1 = self.tcp_comm_cloud_ip_port[0]
        tcp_comm_cloud_ip_port2 = self.tcp_comm_cloud_ip_port[1]

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

        if not LwdCommunicationManager.is_valid_port(int(self.hccl_comm_edge_ip_port)):
            logger.error(
                        "[layerwiseDisaggregated] The configuration is invalid because " \
                        "the HCCL port is not within the range of 1024 to 65535."
                        )
            return False

        self.tcp_comm_cloud_ip_port = "[" + tcp_comm_cloud_ip_port1 + ", " + tcp_comm_cloud_ip_port2 + "]"

        self.npu_edge_num = npu_edge_num
        self.npu_cloud_num = npu_cloud_num

        if self.multi_nodes_infer_enabled:
            if len(self.cloud_ip_addresses) < 2:
                logger.error("[layerwiseDisaggregated] The configuration is invalid because " \
                            "the layerwiseDisaggregatedSlaveIpAddress is a single ip address."
                            )
                return False
            if self.role_type == ROLE_SLAVE and self.multi_nodes_is_master is None:
                logger.error("[layerwiseDisaggregated] The configuration is invalid because " \
                            "the layerwiseDisaggregatedMultiNodesIsMaster is not a number."
                            )
                return False
            if self.role_type == ROLE_SLAVE and self.multi_nodes_ctrl_port is None:
                logger.error("[layerwiseDisaggregated] The configuration is invalid because " \
                            "the layerwiseDisaggregatedMultiNodesCtrlPort is not a number."
                            )
                return False

            if not LwdCommunicationManager.is_valid_port(int(self.multi_nodes_ctrl_port)):
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
        server_ip = self.cloud_ip_addresses[0]
        if self.multi_nodes_infer_enabled:
            multi_nodes_infer_args = {
                'is_master': self.multi_nodes_is_master,
                'cloud_ctrl_port': self.multi_nodes_ctrl_port,
                'cloud_ctrl_address': self.cloud_ip_addresses[0],
                'dp_size': self.multi_nodes_dp_size,
            }
            if self.role_type == ROLE_MASTER:
                if self.rank < self.multi_nodes_dp_size:
                    server_ip = self.cloud_ip_addresses[self.rank]
            else:
                if not self.multi_nodes_is_master:
                    server_ip = self.cloud_ip_addresses[1]
        self.ctrl_comm.init_tcp_link(rank=self.rank, role=self.role_type,
                                     server_ip=server_ip, server_port=self.tcp_comm_cloud_ip_port,
                                     multi_nodes_infer_args=multi_nodes_infer_args)
        if not self.ctrl_comm.is_edge_cloud_ctrl_comm_success():
            logger.error("[layerwiseDisaggregated] The tcp connection of LayerwiseDisaggregated is fail, " \
                        f"ip={self.edge_ip_address} port={self.tcp_comm_cloud_ip_port}.")
            return False

        data_comm_args = {'edge_ip': self.edge_ip_address, 'edge_port': self.hccl_comm_edge_ip_port,
                          'npuEdgeNum': self.npu_edge_num, 'npuCloudNum': self.npu_cloud_num}
        self.data_comm.init_hccl(rank=self.rank, role=self.role_type, data_comm_args=data_comm_args,
                                 multi_nodes_infer_args=multi_nodes_infer_args)
        if self.data_comm.init_finish:
            self.data_comm.hccl_comm_warmup(self.generator.hidden_size)
            return True
        else:
            return False

    def parse(self, model_config: DmiConfig):
        self.role_type = model_config.parse("layerwiseDisaggregatedRoleType", required=False, default_value=None)
        self.edge_ip_address = model_config.parse("layerwiseDisaggregatedMasterIpAddress", \
                                                required=False)
        self.cloud_ip_addresses = model_config.parse_list("layerwiseDisaggregatedSlaveIpAddress", \
                                                required=False)
        self.hccl_comm_edge_ip_port = model_config.parse("layerwiseDisaggregatedDataPort", \
                                                required=False)
        self.tcp_comm_cloud_ip_port = model_config.parse_list("layerwiseDisaggregatedCrtlPort", \
                                                required=False)
        self.multi_nodes_infer_enabled = model_config.parse("layerwiseDisaggregatedMultiNodesInferEnabled", \
                                                required=False, default_value='false') == 'true'
        self.multi_nodes_ctrl_port = model_config.parse("layerwiseDisaggregatedMultiNodesCtrlPort", \
                                                required=False)
        self.multi_nodes_is_master = model_config.parse("lwd_multi_nodes_is_master", required=False, \
                                                default_value='false') == 'true'

    def initialize(self, model_config: DmiConfig, initialize_result, generator):
        self.rank = model_config.local_rank
        self.generator = generator
        self.multi_nodes_dp_size = model_config.dp_size

        self.parse(model_config)

        if self.communication_config_check(model_config):
            if not self.communication_init():
                initialize_result["status"] = "error"
        else:
            initialize_result["status"] = "error"

        logger.info(
                    "[layerwiseDisaggregated] global rank:%s: return initialize success " \
                    "result: %s", self.rank, initialize_result
                    )
