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
from unittest.mock import Mock, MagicMock
from mindie_llm.model_wrapper.utils.config import BaseConfig
from mindie_llm.utils.layerwise.communication import LwdCommunicationManager


class TestEdgeCloudComm(unittest.TestCase):
    def setUp(self):
        self.router = LwdCommunicationManager()
        self.mock_generator = Mock()
        self.router.generator = self.mock_generator

    def test_init(self):
        test_router = LwdCommunicationManager()
        self.assertIsNone(test_router.rank)

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

    def test_initialize(self):
        router = LwdCommunicationManager()
        generator = MagicMock()

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

        initialize_result = {"status":"ok"}
        config.model_config.update({"layerwiseDisaggregatedDataPort":10000})
        router.parse(config)
        router.initialize(config, initialize_result, generator)
        self.assertEqual(initialize_result["status"], "ok")

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

if __name__ == '__main__':
    unittest.main()