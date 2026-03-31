# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import Mock, MagicMock, patch
from mindie_llm.model_wrapper.utils.config import BaseConfig
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager


class TestEdgeCloudComm(unittest.TestCase):
    def setUp(self):
        self.router = LwdCommunicationManager()
        self.mock_generator = Mock()
        self.router.generator = self.mock_generator

    def test_init(self):
        test_router = LwdCommunicationManager()
        self.assertIsNone(test_router.config.rank)

    def test_is_valid_port(self):
        result = LwdCommunicationManager.is_valid_port(100)
        self.assertFalse(result)
        result = LwdCommunicationManager.is_valid_port(65537)
        self.assertFalse(result)
        result = LwdCommunicationManager.is_valid_port(10000)
        self.assertTrue(result)

    def test_communication_config_check(self):
        model_config = {'local_rank':1, 'model_id':'', 'rank':1, 'world_size':2, 'npu_device_id':1,
                        'npu_device_ids':'0,1', 'cpu_mem':0, 'npu_mem':0, 'max_seq_len':65536, 'max_iter_times':999,
                        'max_prefill_tokens':65536, "block_size":1, "distributed_enable":True}
        config = BaseConfig(model_config)

        config.model_config.update({"layerwiseDisaggregatedSlaveIpAddress":"1,2"})
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001"})

        # 未配置roleType
        router = LwdCommunicationManager()
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # 配置roleType为edgecloud
        config.model_config.update({"layerwiseDisaggregatedRoleType":"edgecloud"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # 配置roleType为edge
        config.model_config.update({"layerwiseDisaggregatedRoleType":"master"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # 云IP错误
        config.model_config.update({"layerwiseDisaggregatedSlaveIpAddress":"1"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # 边IP错误
        config.model_config.update({"layerwiseDisaggregatedSlaveIpAddress":"1.1.1.1"})
        config.model_config.update({"layerwiseDisaggregatedMasterIpAddress":"1.1.1.300"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # 未配置models
        config.model_config.update({"layerwiseDisaggregatedMasterIpAddress":"1.1.1.3"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # models配置项缺失错误
        config.model_config.update({"models":'{"xx":1,"yy":2}'})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # models的npuEdgeNum配置错误
        config.model_config.update({"models":'{"layerwiseDisaggregatedMasterDeviceNum": 3, \
                                    "layerwiseDisaggregatedSlaveDeviceNum": 8}'})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # 未配置hccl端口错误
        config.model_config.update({"models":'{"layerwiseDisaggregatedMasterDeviceNum": 2, \
                                    "layerwiseDisaggregatedSlaveDeviceNum": 8}'})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # hccl端口值错误
        config.model_config.update({"layerwiseDisaggregatedDataPort":100})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # tcp端口错误
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001,a"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # tcp端口值错误
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001,100"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        # hccl端口值错误
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001,10002"})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertFalse(result)

        config.model_config.update({"layerwiseDisaggregatedDataPort":10000})
        router.parse(config)
        result = router.communication_config_check(config)
        self.assertTrue(result)

    def test_lwd_ranktable_parse_check_proc(self):
        router = LwdCommunicationManager()
        router.config.npu_edge_num = 2

        lwd_rank_table = {
            "server_count": "2"
        }
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertFalse(result)

        lwd_rank_table = {
            "server_count": "2",
            "server_list": [
                {
                    "server_id": "172.16.0.1"
                },
                {
                    "server_id": "172.16.0.2"
                }
            ]
        }
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertFalse(result)

        lwd_rank_table = {
            "server_count": "2",
            "server_list": [
                {
                    "server_id": "172.16.0.1",
                    "container_ip": "172.16.0.1",
                    "host_ip": "172.16.0.1",
                    "para_net_position": "device",
                    "para_net_protocol": "rdam",
                    "device": [
                        {
                            "rank_id": "0",
                            "device_id": "0",
                            "device_ip": "192.168.1.2",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "1",
                            "device_id": "1",
                            "device_ip": "192.168.1.3",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "2",
                            "device_id": "2",
                            "device_ip": "192.168.1.4",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "3",
                            "device_id": "3",
                            "device_ip": "192.168.1.5",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "4",
                            "device_id": "4",
                            "device_ip": "192.168.1.6",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "5",
                            "device_id": "5",
                            "device_ip": "192.168.1.7",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "6",
                            "device_id": "6",
                            "device_ip": "192.168.1.8",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "7",
                            "device_id": "7",
                            "device_ip": "192.168.1.9",
                            "device_port": "16666",
                            "host_port": "60002"
                        }
                    ]
                },
                {
                    "server_id": "172.16.0.2",
                    "container_ip": "172.16.0.2",
                    "host_ip": "172.16.0.2",
                    "para_net_position": "host",
                    "para_net_protocol": "rdam",
                    "device": [
                        {
                            "rank_id": "8",
                            "device_id": "0",
                            "device_ip": "192.168.2.2",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "9",
                            "device_id": "1",
                            "device_ip": "192.168.2.2",
                            "device_port": "16666",
                            "host_port": "60002"
                        }
                    ]
                }
            ]
        }

        # server_count与server_list匹配检查
        del lwd_rank_table["server_count"]
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertFalse(result)

        lwd_rank_table.update({"server_count": "3"})
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertFalse(result)
        lwd_rank_table["server_count"] = "2"

        # NPU走host网卡配置检查
        router.config.role_type = 'slave'
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertTrue(result)
        self.assertFalse(router.config.npu_net_host)

        router.config.role_type = 'master'
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertTrue(result)
        self.assertTrue(router.config.npu_net_host)

        # ranktable中云的配置在前，边的配置在后，检查配置顺序
        lwd_rank_table["server_list"][0], lwd_rank_table["server_list"][1] = \
            lwd_rank_table["server_list"][1], lwd_rank_table["server_list"][0]
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertFalse(result)

        lwd_rank_table["server_list"][0], lwd_rank_table["server_list"][1] = \
            lwd_rank_table["server_list"][1], lwd_rank_table["server_list"][0]
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertTrue(result)

    def test_lwd_ranktable_parse_check_proc_for_device(self):
        router = LwdCommunicationManager()
        router.config.npu_edge_num = 2

        lwd_rank_table = {
            "server_count": "2",
            "server_list": [
                {
                    "server_id": "172.16.0.1",
                    "container_ip": "172.16.0.1",
                    "host_ip": "172.16.0.1",
                    "device": [
                        {
                            "rank_id": "0",
                            "device_id": "0",
                            "device_ip": "192.168.1.2",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "1",
                            "device_id": "1",
                            "device_ip": "192.168.1.3",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "2",
                            "device_id": "2",
                            "device_ip": "192.168.1.4",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "3",
                            "device_id": "3",
                            "device_ip": "192.168.1.5",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "4",
                            "device_id": "4",
                            "device_ip": "192.168.1.6",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "5",
                            "device_id": "5",
                            "device_ip": "192.168.1.7",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "6",
                            "device_id": "6",
                            "device_ip": "192.168.1.8",
                            "device_port": "16666",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "7",
                            "device_id": "7",
                            "device_ip": "192.168.1.9",
                            "device_port": "16666",
                            "host_port": "60002"
                        }
                    ]
                },
                {
                    "server_id": "172.16.0.2",
                    "container_ip": "172.16.0.2",
                    "host_ip": "172.16.0.2",
                    "device": [
                        {
                            "rank_id": "8",
                            "device_id": "0",
                            "device_ip": "192.168.2.2",
                            "device_port": "16666",
                            "net_position": "host",
                            "net_protocol": "rdam",
                            "host_port": "60002"
                        },
                        {
                            "rank_id": "9",
                            "device_id": "1",
                            "device_ip": "192.168.2.2",
                            "device_port": "16666",
                            "net_position": "host",
                            "net_protocol": "rdam",
                            "host_port": "60002"
                        }
                    ]
                }
            ]
        }

        router.config.role_type = 'master'
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertTrue(result)
        self.assertTrue(router.config.npu_net_host)

        del lwd_rank_table["server_list"][1]["device"][0]["net_position"]
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertFalse(result)

        router.config.npu_net_host = False
        del lwd_rank_table["server_list"][1]["device"][1]["net_position"]
        result = router.lwd_ranktable_parse_check_proc(lwd_rank_table)
        self.assertTrue(result)
        self.assertFalse(router.config.npu_net_host)

    def test_communication_config_verify(self):
        router = LwdCommunicationManager()

        model_config = {'local_rank':1, 'model_id':'', 'rank':1, 'world_size':2, 'npu_device_id':1,
                        'npu_device_ids':'0,1', 'cpu_mem':0, 'npu_mem':0, 'max_seq_len':65536, 'max_iter_times':999,
                        'max_prefill_tokens':65536, "block_size":1, "distributed_enable":True}
        config = BaseConfig(model_config)

        config.model_config.update({"layerwiseDisaggregatedRoleType":"master"})
        config.model_config.update({"layerwiseDisaggregatedSlaveIpAddress":"1.1.1.1"})
        config.model_config.update({"layerwiseDisaggregatedMasterIpAddress":"1.1.1.3"})
        config.model_config.update({"models":'{"layerwiseDisaggregatedMasterDeviceNum": 2, \
                                    "layerwiseDisaggregatedSlaveDeviceNum": 8}'})
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001,10002"})
        result = router.communication_config_verify(config)
        self.assertFalse(result)

        config.model_config.update({"layerwiseDisaggregatedDataPort":10000})
        result = router.communication_config_verify(config)
        self.assertTrue(result)

    def test_initialize(self):
        router = LwdCommunicationManager()
        generator = MagicMock()
        generator.model_wrapper.model_runner.ctrl_comm = None
        generator.model_wrapper.model_runner.data_comm = None

        model_config = {'local_rank':1, 'model_id':'', 'rank':1, 'world_size':2, 'npu_device_id':1,
                        'npu_device_ids':'0,1', 'cpu_mem':0, 'npu_mem':0, 'max_seq_len':65536, 'max_iter_times':999,
                        'max_prefill_tokens':65536, "block_size":1, "distributed_enable":True}
        config = BaseConfig(model_config)

        initialize_result = {"status":"ok"}
        config.model_config.update({"layerwiseDisaggregatedRoleType":"master"})
        config.model_config.update({"layerwiseDisaggregatedSlaveIpAddress":"1.1.1.1"})
        config.model_config.update({"layerwiseDisaggregatedMasterIpAddress":"1.1.1.3"})
        config.model_config.update({"models":'{"layerwiseDisaggregatedMasterDeviceNum": 2, \
                                    "layerwiseDisaggregatedSlaveDeviceNum": 8}'})
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001,10002"})
        router.parse(config)
        router.initialize(config, initialize_result, generator)
        self.assertEqual(initialize_result["status"], "error")

    def test_communication_init(self):
        router = LwdCommunicationManager()
        router.generator = MagicMock()

        model_config = {'local_rank':1, 'model_id':'', 'rank':1, 'world_size':2, 'npu_device_id':1,
                        'npu_device_ids':'0,1', 'cpu_mem':0, 'npu_mem':0, 'max_seq_len':65536, 'max_iter_times':999,
                        'max_prefill_tokens':65536, "block_size":1, "distributed_enable":True}
        config = BaseConfig(model_config)
        config.model_config.update({"layerwiseDisaggregatedRoleType":"master"})
        config.model_config.update({"layerwiseDisaggregatedSlaveIpAddress":"1.1.1.1"})
        config.model_config.update({"layerwiseDisaggregatedMasterIpAddress":"1.1.1.3"})
        config.model_config.update({"models":'{"layerwiseDisaggregatedMasterDeviceNum": 2, \
                                    "layerwiseDisaggregatedSlaveDeviceNum": 8}'})
        config.model_config.update({"layerwiseDisaggregatedCrtlPort":"10001,10002"})
        config.model_config.update({"layerwiseDisaggregatedDataPort":10000})
        router.parse(config)

        # data_comm ctrl_comm初始化失败
        router.generator.model_wrapper.model_runner.ctrl_comm = None
        router.generator.model_wrapper.model_runner.data_comm = None
        result = router.communication_init()
        self.assertFalse(result)

        # ctrl_comm not success
        router.generator.model_wrapper.model_runner.ctrl_comm = MagicMock()
        router.generator.model_wrapper.model_runner.data_comm = MagicMock()
        router.generator.model_wrapper.model_runner.ctrl_comm.is_edge_cloud_ctrl_comm_success = \
            MagicMock(return_value=False)
        result = router.communication_init()
        self.assertFalse(result)

        # data_comm未完成init
        router.generator.model_wrapper.model_runner.ctrl_comm = MagicMock()
        router.generator.model_wrapper.model_runner.data_comm = MagicMock()
        router.generator.model_wrapper.model_runner.data_comm.init_finish = False
        result = router.communication_init()
        self.assertFalse(result)

        # initialize success
        router.generator.model_wrapper.model_runner.ctrl_comm = MagicMock()
        router.generator.model_wrapper.model_runner.data_comm = MagicMock()
        result = router.communication_init()
        self.assertTrue(result)

    @patch("mindie_llm.runtime.utils.npu.device_utils.acl.rt.get_mem_info")
    def test_ctrl_comm_sync_npu_smi_info(self, mock_get_mem_info):
        router = LwdCommunicationManager()
        mock_get_mem_info.return_value = (5000, 20000, 0)

        router.config.npu_smi_info_sync = False
        result = router.ctrl_comm_sync_npu_smi_info()
        self.assertTrue(result)

        router.config.npu_smi_info_sync = True
        router.ctrl_comm = MagicMock()
        router.ctrl_comm.npu_smi_info_sync_is_done.return_value = True
        result = router.ctrl_comm_sync_npu_smi_info()
        self.assertTrue(result)

        router.ctrl_comm.npu_smi_info_sync_is_done.return_value = False
        result = router.ctrl_comm_sync_npu_smi_info()
        self.assertFalse(result)

    def test_peer_npu_smi_info(self):
        router = LwdCommunicationManager()
        router.ctrl_comm = MagicMock()
        npu_smi_info = {'hbm_capacity': 65452113920, 'soc_name': 'Ascend910B3'}

        router.config.role_type = 'master'
        router.set_peer_npu_smi_info(npu_smi_info)
        npu_smi_info_read = router.get_peer_npu_smi_info()
        self.assertEqual(npu_smi_info_read, router.ctrl_comm.cloud_npu_smi_info)

        router.config.role_type = 'slave'
        router.set_peer_npu_smi_info(npu_smi_info)
        npu_smi_info_read = router.get_peer_npu_smi_info()
        self.assertEqual(npu_smi_info_read, router.ctrl_comm.edge_npu_smi_info)


if __name__ == '__main__':
    unittest.main()