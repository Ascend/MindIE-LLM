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
import unittest

# Set required env var before any mindie import triggers EnvVar() validation
os.environ.setdefault("MINDIE_LLM_BENCHMARK_FILEPATH", "/tmp/benchmark.jsonl")

from mindie_llm.utils.log.error_code import (
    ErrorCode,
    ErrorCodeException,
    convert_exception_to_error_code,
    is_force_stop_exception,
)


class TestErrorCodeEnum(unittest.TestCase):
    """Test ErrorCode enum values and structure."""

    def test_enum_inherits_str(self):
        self.assertTrue(issubclass(ErrorCode, str))

    def test_str_returns_value(self):
        self.assertEqual(str(ErrorCode.ACL_GRAPH_OUT_OF_MEMORY), "MIE05E000006")

    def test_runtime_codes_have_correct_prefix(self):
        for code in ErrorCode:
            if "RUNTIME" in code.name or code.name.startswith("ATB_") or code.name.startswith("ACL_GRAPH_"):
                self.assertTrue(code.value.startswith("MIE05E000"), f"{code.name} has wrong prefix")

    def test_text_generator_codes_have_correct_prefix(self):
        for code in ErrorCode:
            if code.name.startswith("TEXT_GENERATOR_"):
                self.assertTrue(code.value.startswith("MIE05E01"), f"{code.name} has wrong prefix")

    def test_backend_codes_have_correct_prefix(self):
        for code in ErrorCode:
            if code.name.startswith("BACKEND_"):
                self.assertTrue(code.value.startswith("MIE05E02"), f"{code.name} has wrong prefix")

    def test_llm_manager_codes_have_correct_prefix(self):
        for code in ErrorCode:
            if code.name.startswith("LLM_MANAGER_"):
                self.assertTrue(code.value.startswith("MIE05E03"), f"{code.name} has wrong prefix")

    def test_acl_graph_oom_value(self):
        self.assertEqual(ErrorCode.ACL_GRAPH_OUT_OF_MEMORY.value, "MIE05E000006")

    def test_atb_models_oom_value(self):
        self.assertEqual(ErrorCode.ATB_MODELS_OUT_OF_MEMORY.value, "MIE05E000005")

    def test_text_generator_oom_value(self):
        self.assertEqual(ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY.value, "MIE05E01000A")

    def test_no_duplicate_values(self):
        values = [code.value for code in ErrorCode]
        self.assertEqual(len(values), len(set(values)), "Duplicate error code values found")


class TestErrorCodeException(unittest.TestCase):
    """Test ErrorCodeException behavior."""

    def test_holds_error_code(self):
        exc = ErrorCodeException(ErrorCode.ACL_GRAPH_OUT_OF_MEMORY)
        self.assertIs(exc.error_code, ErrorCode.ACL_GRAPH_OUT_OF_MEMORY)

    def test_message_includes_name_and_value(self):
        exc = ErrorCodeException(ErrorCode.ACL_GRAPH_OUT_OF_MEMORY)
        self.assertIn(ErrorCode.ACL_GRAPH_OUT_OF_MEMORY.name, str(exc))
        self.assertIn(ErrorCode.ACL_GRAPH_OUT_OF_MEMORY.value, str(exc))

    def test_is_runtime_error_subclass(self):
        self.assertTrue(issubclass(ErrorCodeException, RuntimeError))


class TestConvertExceptionToErrorCode(unittest.TestCase):
    """Test convert_exception_to_error_code function."""

    def test_matches_oom_error_code(self):
        result = convert_exception_to_error_code("something MIE05E000005 in the message")
        self.assertEqual(result, ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY)

    def test_matches_hbm_ecc_error_code(self):
        result = convert_exception_to_error_code("HBM MULTI BIT ECC ERROR occurred")
        self.assertEqual(result, ErrorCode.TEXT_GENERATOR_HBM_MULTI_BIT_ECC_ERROR)

    def test_returns_none_for_unrecognized(self):
        result = convert_exception_to_error_code("some random error")
        self.assertIsNone(result)

    def test_returns_none_for_empty_string(self):
        result = convert_exception_to_error_code("")
        self.assertIsNone(result)


class TestIsForceStopException(unittest.TestCase):
    """Test is_force_stop_exception function."""

    def test_detects_force_stop(self):
        self.assertTrue(is_force_stop_exception(RuntimeError("FORCE STOP")))

    def test_returns_false_for_normal_error(self):
        self.assertFalse(is_force_stop_exception(RuntimeError("normal error")))

    def test_returns_false_for_non_runtime_error(self):
        self.assertFalse(is_force_stop_exception(ValueError("FORCE STOP")))

    def test_case_insensitive(self):
        self.assertTrue(is_force_stop_exception(RuntimeError("force stop")))


if __name__ == "__main__":
    unittest.main()
