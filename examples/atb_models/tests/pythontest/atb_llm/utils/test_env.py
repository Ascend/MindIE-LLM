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
import os
import json
from unittest.mock import patch, mock_open


class TestOmniAttentionPatternFile(unittest.TestCase):
    """Test omni_attention_pattern_file processing logic"""

    def setUp(self):
        """Setup before each test"""
        # Save original environment variable
        self.original_env = os.environ.get("ATB_LLM_OMNI_ATTENTION_PATTERN_FILE")

    def tearDown(self):
        """Cleanup after each test"""
        # Restore original environment variable
        if self.original_env is not None:
            os.environ["ATB_LLM_OMNI_ATTENTION_PATTERN_FILE"] = self.original_env
        elif "ATB_LLM_OMNI_ATTENTION_PATTERN_FILE" in os.environ:
            del os.environ["ATB_LLM_OMNI_ATTENTION_PATTERN_FILE"]

    @patch("atb_llm.utils.file_utils.check_file_safety")
    @patch("atb_llm.utils.file_utils.standardize_path")
    def test_omni_attention_pattern_file_standardize_exception_case_propagation(
        self, mock_standardize, mock_check_safety
    ):
        """Test handling when path standardization throws exception"""
        test_path = "/invalid/path/pattern.txt"

        # Configure Mock - throw exception for specific path, return original value for other paths
        def mock_standardize_side_effect(path):
            if path == test_path:
                raise ValueError("Path standardization failed")
            return path

        mock_standardize.side_effect = mock_standardize_side_effect
        mock_check_safety.return_value = None

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()
        env_var.omni_attention_pattern_file = test_path

        with self.assertRaises(ValueError) as context:
            env_var.__post_init__()

        self.assertIn("Path standardization failed", str(context.exception))

    @patch("atb_llm.utils.file_utils.check_file_safety")
    @patch("atb_llm.utils.file_utils.standardize_path")
    def test_omni_attention_pattern_file_check_safety_exception_case_propagation(
        self, mock_standardize, mock_check_safety
    ):
        """Test handling when file safety check throws exception"""
        test_path = "/path/to/unsafe_pattern.txt"
        standardized_path = "/standardized/path/to/unsafe_pattern.txt"

        # Configure Mock - return specific value for specific path, return original value for other paths
        def mock_standardize_side_effect(path):
            if path == test_path:
                return standardized_path
            return path

        def mock_check_safety_side_effect(path, *args):
            if path == standardized_path:
                raise ValueError("File safety check failed")
            return None

        mock_standardize.side_effect = mock_standardize_side_effect
        mock_check_safety.side_effect = mock_check_safety_side_effect

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()
        env_var.omni_attention_pattern_file = test_path

        with self.assertRaises(ValueError) as context:
            env_var.__post_init__()

        self.assertIn("File safety check failed", str(context.exception))


class TestEnvVarUpdate(unittest.TestCase):
    """Test EnvVar update method"""

    def setUp(self):
        """Setup before each test"""
        # Save original environment variables
        self.original_env = {}
        env_vars = [
            "ATB_LLM_LOGITS_SAVE_ENABLE",
            "ATB_LLM_LOGITS_SAVE_FOLDER",
            "ATB_LLM_TOKEN_IDS_SAVE_ENABLE",
            "ATB_LLM_TOKEN_IDS_SAVE_FOLDER",
            "MODELTEST_DATASET_SPECIFIED",
        ]
        for var in env_vars:
            self.original_env[var] = os.environ.get(var)

    def tearDown(self):
        """Cleanup after each test"""
        # Restore original environment variables
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_update_with_default_values(self):
        """Test update method with default values"""
        # Clear all related environment variables
        env_vars = [
            "ATB_LLM_LOGITS_SAVE_ENABLE",
            "ATB_LLM_LOGITS_SAVE_FOLDER",
            "ATB_LLM_TOKEN_IDS_SAVE_ENABLE",
            "ATB_LLM_TOKEN_IDS_SAVE_FOLDER",
            "MODELTEST_DATASET_SPECIFIED",
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()
        env_var.update()

        # Verify default values
        self.assertFalse(env_var.logits_save_enable)
        self.assertEqual(env_var.logits_save_folder, "./")
        self.assertFalse(env_var.token_ids_save_enable)
        self.assertEqual(env_var.token_ids_save_folder, "./")
        self.assertIsNone(env_var.modeltest_dataset_specified)

    def test_update_with_custom_values(self):
        """Test update method with custom values"""
        # Set custom environment variables
        os.environ["ATB_LLM_LOGITS_SAVE_ENABLE"] = "1"
        os.environ["ATB_LLM_LOGITS_SAVE_FOLDER"] = "/custom/logits"
        os.environ["ATB_LLM_TOKEN_IDS_SAVE_ENABLE"] = "1"
        os.environ["ATB_LLM_TOKEN_IDS_SAVE_FOLDER"] = "/custom/tokens"
        os.environ["MODELTEST_DATASET_SPECIFIED"] = "custom_dataset"

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()
        env_var.update()

        # Verify custom values
        self.assertTrue(env_var.logits_save_enable)
        self.assertEqual(env_var.logits_save_folder, "/custom/logits")
        self.assertTrue(env_var.token_ids_save_enable)
        self.assertEqual(env_var.token_ids_save_folder, "/custom/tokens")
        self.assertEqual(env_var.modeltest_dataset_specified, "custom_dataset")

    def test_update_with_false_values(self):
        """Test update method with false values"""
        # Set false environment variables
        os.environ["ATB_LLM_LOGITS_SAVE_ENABLE"] = "0"
        os.environ["ATB_LLM_TOKEN_IDS_SAVE_ENABLE"] = "false"

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()
        env_var.update()

        # Verify false values
        self.assertFalse(env_var.logits_save_enable)
        self.assertFalse(env_var.token_ids_save_enable)


class TestEnvVarCheckRanktable(unittest.TestCase):
    """Test EnvVar check_ranktable method"""

    def setUp(self):
        """Setup before each test"""
        # Save original environment variables
        self.original_env = {}

    def tearDown(self):
        """Cleanup after each test"""
        # Restore original environment variables
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_check_ranktable_with_none_file(self):
        """Test check_ranktable with None file"""
        # Set valid framework backend to avoid initialization errors

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should not raise any exception when rank_table_file is None
        try:
            env_var.check_ranktable(None)
        except Exception as e:
            self.fail(f"check_ranktable raised {e} unexpectedly!")

    def test_check_ranktable_with_empty_string(self):
        """Test check_ranktable with empty string"""
        # Set valid framework backend to avoid initialization errors

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should not raise any exception when rank_table_file is empty string
        try:
            env_var.check_ranktable("")
        except Exception as e:
            self.fail(f"check_ranktable raised {e} unexpectedly!")

    @patch("atb_llm.utils.file_utils.safe_open")
    def test_check_ranktable_with_valid_ranktable(self, mock_safe_open):
        """Test check_ranktable with valid ranktable"""
        # Set valid framework backend to avoid initialization errors

        # Mock valid ranktable JSON
        valid_ranktable = {
            "server_list": [
                {
                    "server_id": "192.168.1.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "1", "device_ip": "192.168.1.11"},
                    ],
                },
                {
                    "server_id": "192.168.1.2",
                    "device": [
                        {"rank_id": "2", "device_ip": "192.168.1.20"},
                        {"rank_id": "3", "device_ip": "192.168.1.21"},
                    ],
                },
            ]
        }

        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(valid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should not raise any exception with valid ranktable
        try:
            env_var.check_ranktable("/path/to/valid_ranktable.json")
        except Exception as e:
            self.fail(f"check_ranktable raised {e} unexpectedly!")

    @patch("atb_llm.utils.file_utils.safe_open")
    def test_check_ranktable_with_invalid_rank_id(self, mock_safe_open):
        """Test check_ranktable with invalid rank_id (>= world_size)"""

        # Mock ranktable with invalid rank_id
        invalid_ranktable = {
            "server_list": [
                {
                    "server_id": "192.168.1.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {
                            "rank_id": "5",
                            "device_ip": "192.168.1.11",
                        },  # rank_id 5 >= world_size 2
                    ],
                }
            ]
        }

        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(invalid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should raise ValueError for invalid rank_id
        with self.assertRaises(ValueError) as context:
            env_var.check_ranktable("/path/to/invalid_ranktable.json")

        self.assertIn("rank_id should be a number less than world size", str(context.exception))

    @patch("atb_llm.utils.file_utils.safe_open")
    def test_check_ranktable_with_invalid_device_ip(self, mock_safe_open):
        """Test check_ranktable with invalid device IP"""

        # Mock ranktable with invalid device IP
        invalid_ranktable = {
            "server_list": [
                {
                    "server_id": "192.168.1.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "1", "device_ip": "invalid_ip"},  # Invalid IP
                    ],
                }
            ]
        }

        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(invalid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should raise ValueError for invalid device IP
        with self.assertRaises(ValueError) as context:
            env_var.check_ranktable("/path/to/invalid_ranktable.json")

        self.assertIn("device_ip is invalid", str(context.exception))

    @patch("atb_llm.utils.file_utils.safe_open")
    def test_check_ranktable_with_invalid_server_id(self, mock_safe_open):
        """Test check_ranktable with invalid server_id IP"""

        # Mock ranktable with invalid server_id
        invalid_ranktable = {
            "server_list": [
                {
                    "server_id": "invalid_server_ip",  # Invalid server IP
                    "device": [
                        {"rank_id": "0", "device_ip": "192.168.1.10"},
                        {"rank_id": "1", "device_ip": "192.168.1.11"},
                    ],
                }
            ]
        }

        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(invalid_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should raise ValueError for invalid server_id
        with self.assertRaises(ValueError) as context:
            env_var.check_ranktable("/path/to/invalid_ranktable.json")

        self.assertIn("server_id is invalid", str(context.exception))

    @patch("atb_llm.utils.file_utils.safe_open")
    def test_check_ranktable_with_complex_valid_ranktable(self, mock_safe_open):
        """Test check_ranktable with complex valid ranktable (multiple servers)"""

        # Mock complex valid ranktable JSON
        complex_ranktable = {
            "server_list": [
                {
                    "server_id": "10.0.0.1",
                    "device": [
                        {"rank_id": "0", "device_ip": "10.0.0.10"},
                        {"rank_id": "1", "device_ip": "10.0.0.11"},
                        {"rank_id": "2", "device_ip": "10.0.0.12"},
                    ],
                },
                {
                    "server_id": "10.0.0.2",
                    "device": [
                        {"rank_id": "3", "device_ip": "10.0.0.20"},
                        {"rank_id": "4", "device_ip": "10.0.0.21"},
                    ],
                },
                {
                    "server_id": "10.0.0.3",
                    "device": [
                        {"rank_id": "5", "device_ip": "10.0.0.30"},
                        {"rank_id": "6", "device_ip": "10.0.0.31"},
                        {"rank_id": "7", "device_ip": "10.0.0.32"},
                    ],
                },
            ]
        }

        # Mock file operations
        mock_file = mock_open(read_data=json.dumps(complex_ranktable))
        mock_safe_open.return_value.__enter__.return_value = mock_file.return_value

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        env_var = EnvVar()

        # Should not raise any exception with complex valid ranktable
        try:
            env_var.check_ranktable("/path/to/complex_ranktable.json")
        except Exception as e:
            self.fail(f"check_ranktable raised {e} unexpectedly!")


class TestEnvVarIsValidIp(unittest.TestCase):
    """Test EnvVar is_valid_ip static method"""

    def test_is_valid_ip_with_valid_ipv4(self):
        """Test is_valid_ip with valid IPv4 addresses"""

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        # Test valid IPv4 addresses
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "127.0.0.1",
            "0.0.0.0",
            "255.255.255.255",
        ]

        for ip in valid_ips:
            with self.subTest(ip=ip):
                self.assertTrue(EnvVar.is_valid_ip(ip))

    def test_is_valid_ip_with_valid_ipv6(self):
        """Test is_valid_ip with valid IPv6 addresses"""

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        # Test valid IPv6 addresses
        valid_ipv6s = ["::1", "2001:db8::1", "fe80::1", "::ffff:192.168.1.1"]

        for ip in valid_ipv6s:
            with self.subTest(ip=ip):
                self.assertTrue(EnvVar.is_valid_ip(ip))

    def test_is_valid_ip_with_invalid_ips(self):
        """Test is_valid_ip with invalid IP addresses"""

        # Clear module cache to ensure reload
        import sys

        if "atb_llm.utils.env" in sys.modules:
            del sys.modules["atb_llm.utils.env"]

        from atb_llm.utils.env import EnvVar

        # Test invalid IP addresses
        invalid_ips = [
            "invalid_ip",
            "256.256.256.256",
            "192.168.1",
            "192.168.1.1.1",
            "192.168.-1.1",
            "",
            "localhost",
            "192.168.1.a",
        ]

        for ip in invalid_ips:
            with self.subTest(ip=ip):
                self.assertFalse(EnvVar.is_valid_ip(ip))


if __name__ == "__main__":
    unittest.main()
