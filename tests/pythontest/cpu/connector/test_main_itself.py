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
from unittest.mock import patch, MagicMock
import argparse

import mindie_llm.connector.main as main_module


class TestMainFunctions(unittest.TestCase):
    def setUp(self):
        self.valid_args = argparse.Namespace(
            local_rank=0,
            local_world_size=1,
            global_rank=0,
            global_world_size=1,
            npu_num_per_dp=1,
            npu_device_id=0,
            parent_pid=1,
            shm_name_prefix="/integrated_testing",
            communication_type="shared_meme",
            use_mock_model=False
        )

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_from_cmd(self, mock_parse_args):
        # 模拟命令行参数
        mock_parse_args.return_value = self.valid_args
        args = main_module.parse_from_cmd()
        self.assertEqual(args.local_rank, 0)
        self.assertEqual(args.local_world_size, 1)
        self.assertEqual(args.global_rank, 0)
        self.assertEqual(args.global_world_size, 1)
        self.assertEqual(args.npu_num_per_dp, 1)
        self.assertEqual(args.npu_device_id, 0)
        self.assertEqual(args.parent_pid, 1)
        self.assertEqual(args.shm_name_prefix, "/integrated_testing")
        self.assertEqual(args.communication_type, "shared_meme")
        self.assertFalse(args.use_mock_model)

    def test_check_config_valid(self):
        self.assertTrue(main_module.check_config(self.valid_args))
    
    def test_check_config_invalid_local_rank(self):
        invalid_args = self.valid_args
        invalid_args.local_rank = -1
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_local_world_size(self):
        invalid_args = self.valid_args
        invalid_args.local_world_size = 0
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_global_rank(self):
        invalid_args = self.valid_args
        invalid_args.global_rank = -1
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_global_world_size(self):
        invalid_args = self.valid_args
        invalid_args.global_world_size = 0
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_npu_num_per_dp(self):
        invalid_args = self.valid_args
        invalid_args.npu_num_per_dp = 0
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_npu_device_id(self):
        invalid_args = self.valid_args
        invalid_args.npu_device_id = -1
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_parent_pid(self):
        invalid_args = self.valid_args
        invalid_args.parent_pid = -1
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    def test_check_config_invalid_shm_name_prefix(self):
        invalid_args = self.valid_args
        invalid_args.shm_name_prefix = "whatever"
        result = main_module.check_config(invalid_args)
        self.assertTrue(result)

    def test_check_config_invalid_communication_type(self):
        invalid_args = self.valid_args
        invalid_args.communication_type = "invalid"
        result = main_module.check_config(invalid_args)
        self.assertFalse(result)

    @patch('signal.signal')
    def test_register_signal(self, mock_signal):
        agent = MagicMock()
        main_module.register_signal(agent)
        self.assertEqual(mock_signal.call_count, 2)

    @patch('mindie_llm.connector.main.check_config')
    @patch('mindie_llm.connector.main.parse_from_cmd')
    def test_main_config_failed(self, mock_parse, mock_check):
        """测试配置检查失败的情况"""
        mock_check.return_value = False
        result = main_module.main()
        self.assertEqual(result, -1)
        mock_parse.assert_called_once()

    @patch('mindie_llm.connector.main.AdaptiveGarbageCollector')
    @patch('mindie_llm.connector.main.RequestListener')
    @patch('mindie_llm.connector.main.check_config')
    @patch('mindie_llm.connector.main.parse_from_cmd')
    def test_main_listener_start_failed(self, mock_parse, mock_check,
                                        mock_request_listener_cls, mock_adaptive_gc):
        mock_config = MagicMock()
        mock_parse.return_value = mock_config
        mock_check.return_value = True

        mock_request_listener = MagicMock()
        mock_request_listener_cls.get_instance.return_value = mock_request_listener
        mock_request_listener.start.return_value = False

        with patch('mindie_llm.connector.main.logger') as mock_logger:
            result = main_module.main()
            mock_logger.error.assert_called_once_with("request listener cannot be launched.")
            self.assertEqual(result, -1)


if __name__ == "__main__":
    unittest.main()