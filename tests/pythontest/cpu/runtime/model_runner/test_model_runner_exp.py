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
import sys
import unittest
from unittest.mock import MagicMock, patch

# Set required env var before any mindie import triggers EnvVar() validation
os.environ.setdefault("MINDIE_LLM_BENCHMARK_FILEPATH", "/tmp/benchmark.jsonl")

# Mock torch.npu before importing any mindie modules
import torch  # noqa: E402

if not hasattr(torch, "npu"):
    torch.npu = MagicMock()
torch.npu.config.allow_internal_format = True
torch.npu.current_stream.return_value.synchronize.return_value = None
torch.npu.FloatTensor = MagicMock
torch.npu.IntTensor = MagicMock

# Import DeviceType safely (enum only, no hardware access)
from mindie_llm.runtime.utils.npu.device_utils import DeviceType  # noqa: E402

# Mock get_npu_node_info before any chain import triggers hardware detection
mock_node_info = MagicMock()
mock_node_info.get_device_type.return_value = DeviceType.ASCEND_910_93
mock_node_info.get_hbm_capacity.return_value = 0
mock_node_info.get_hbm_usage.return_value = 0

# Patch device_utils before the model_runner_exp import chain
import mindie_llm.runtime.utils.npu.device_utils as device_utils_mod  # noqa: E402

device_utils_mod.get_npu_node_info = MagicMock(return_value=mock_node_info)
device_utils_mod.get_npu_hbm_info = MagicMock()

# Replace mie_ops with a mock to skip hardware-specific imports
mock_mie_ops = MagicMock()
sys.modules["mindie_llm.runtime.ops.mie_ops"] = mock_mie_ops

# Now import model_runner_exp with all mocks in place
if "mindie_llm.runtime.model_runner.model_runner_exp" in sys.modules:
    del sys.modules["mindie_llm.runtime.model_runner.model_runner_exp"]

from mindie_llm.runtime.model_runner import model_runner_exp  # noqa: E402
from mindie_llm.runtime.model_runner.model_runner_exp import ModelRunnerExp  # noqa: E402

# ModelRunnerExp is wrapped by @auto_speculative_method_router which replaces
# the class with a factory function.  The original class is accessible via
# __wrapped__ (set by functools.wraps).
_ModelRunnerExpClass = getattr(ModelRunnerExp, "__wrapped__", ModelRunnerExp)


class TestModelRunnerExpSourceStructure(unittest.TestCase):
    """Structural tests that verify the source has the expected decorators."""

    def test_decorator_import_exists(self):
        """The file should import the exception_handler."""
        import ast
        import inspect

        source = inspect.getsource(model_runner_exp)
        tree = ast.parse(source)

        found_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                if "exception_handler" in names:
                    found_import = True
                    break
        self.assertTrue(found_import, "@exception_handler import not found in model_runner_exp.py")

    def test_exception_handler_decorator_before_class(self):
        """The @exception_handler decorator should appear before class ModelRunnerExp."""
        import ast
        import inspect

        source = inspect.getsource(model_runner_exp)
        tree = ast.parse(source)

        found_decorator = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerExp":
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "exception_handler":
                        found_decorator = True
                        break
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "exception_handler":
                        found_decorator = True
                        break
                break

        self.assertTrue(found_decorator, "@exception_handler decorator not found on ModelRunnerExp")

    def test_auto_speculative_method_router_present(self):
        """@auto_speculative_method_router should still be present as outer decorator."""
        import ast
        import inspect

        source = inspect.getsource(model_runner_exp)
        tree = ast.parse(source)

        found_router = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerExp":
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if isinstance(func, ast.Attribute) and "auto_speculative_method_router" in func.attr:
                            found_router = True
                            break
                        elif isinstance(func, ast.Name) and func.id == "auto_speculative_method_router":
                            found_router = True
                            break
                break

        self.assertTrue(found_router, "@auto_speculative_method_router decorator not found on ModelRunnerExp")


class TestModelRunnerExpOomContract(unittest.TestCase):
    """Verify the OOM contract: forward/compile/load_weights are wrapped."""

    def test_forward_is_wrapped(self):
        """forward method should be wrapped by _torch_oom_handler (has __wrapped__)."""
        forward = _ModelRunnerExpClass.__dict__.get("forward")
        self.assertIsNotNone(forward)
        self.assertTrue(hasattr(forward, "__wrapped__"), "forward should be wrapped by exception_handler")

    def test_compile_is_wrapped(self):
        """compile method should be wrapped by _torch_oom_handler."""
        compile_method = _ModelRunnerExpClass.__dict__.get("compile")
        self.assertIsNotNone(compile_method)
        self.assertTrue(hasattr(compile_method, "__wrapped__"), "compile should be wrapped by exception_handler")

    def test_load_weights_is_wrapped(self):
        """load_weights method should be wrapped by _torch_oom_handler."""
        lw = _ModelRunnerExpClass.__dict__.get("load_weights")
        self.assertIsNotNone(lw)
        self.assertTrue(hasattr(lw, "__wrapped__"), "load_weights should be wrapped by exception_handler")


class TestKVCacheInfo(unittest.TestCase):
    """Test KVCacheInfo utility class."""

    def setUp(self):
        self.tensor_a = torch.tensor([1, 2, 3])
        self.tensor_b = torch.tensor([4, 5, 6])

    def test_check_diff_initial_none(self):
        """When kcache_id/vcache_id are None, check_diff should return True."""
        info = model_runner_exp.KVCacheInfo()
        self.assertTrue(info.check_diff([(self.tensor_a, self.tensor_b)]))

    def test_check_diff_all_match(self):
        """When ids and shapes all match, check_diff should return False."""
        info = model_runner_exp.KVCacheInfo()
        info.kcache_id = id(self.tensor_a)
        info.vcache_id = id(self.tensor_b)
        info.kcache_shape = self.tensor_a.shape
        info.vcache_shape = self.tensor_b.shape
        self.assertFalse(info.check_diff([(self.tensor_a, self.tensor_b)]))

    def test_check_diff_kcache_id_mismatch(self):
        """When kcache_id changes, check_diff should return True."""
        info = model_runner_exp.KVCacheInfo()
        old_tensor = torch.tensor([0])
        info.kcache_id = id(old_tensor)
        info.kcache_shape = old_tensor.shape
        info.vcache_id = id(self.tensor_b)
        info.vcache_shape = self.tensor_b.shape
        self.assertTrue(info.check_diff([(self.tensor_a, self.tensor_b)]))

    def test_check_diff_kcache_shape_mismatch(self):
        """When kcache_shape changes, check_diff should return True."""
        info = model_runner_exp.KVCacheInfo()
        reshaped = self.tensor_a.reshape(-1, 1)
        info.kcache_id = id(self.tensor_a)
        info.kcache_shape = reshaped.shape  # different from self.tensor_a.shape
        info.vcache_id = id(self.tensor_b)
        info.vcache_shape = self.tensor_b.shape
        self.assertTrue(info.check_diff([(self.tensor_a, self.tensor_b)]))

    def test_check_diff_vcache_id_mismatch(self):
        """When vcache_id changes, check_diff should return True."""
        info = model_runner_exp.KVCacheInfo()
        info.kcache_id = id(self.tensor_a)
        info.kcache_shape = self.tensor_a.shape
        old_tensor = torch.tensor([0])
        info.vcache_id = id(old_tensor)
        info.vcache_shape = old_tensor.shape
        self.assertTrue(info.check_diff([(self.tensor_a, self.tensor_b)]))

    def test_check_diff_first_call_sets_ids(self):
        """On first call with None ids, kcache_id starts unset."""
        info = model_runner_exp.KVCacheInfo()
        info.kcache_id = id(self.tensor_a)
        info.kcache_shape = self.tensor_a.shape
        # vcache_id is None -> vcache_diff should trigger
        self.assertTrue(info.check_diff([(self.tensor_a, self.tensor_b)]))


class TestDeepseekV32FeatureValidation(unittest.TestCase):
    """Test the DeepSeek-V3.2 feature combination validation (_validate_deepseekv32_feature_combinations)."""

    def setUp(self):
        self.base_kwargs = {"cp": 1, "dp": 1, "role": "standard", "plugin_params": ""}

    # --- Individual incompatible combinations ---

    def test_cp_dp_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=16, dp=2, plugin_params="")
        self.assertIn("CP(cp=16) + DP(dp=2)", str(ctx.exception))
        self.assertIn("DeepSeek-V3.2", str(ctx.exception))

    def test_cp_async_scheduling_raises(self):
        with patch.object(model_runner_exp.ENV_utils, "async_inference", True, create=True):
            with self.assertRaises(ValueError) as ctx:
                _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=16, dp=1, plugin_params="")
        self.assertIn("CP(cp=16) + Async Scheduling", str(ctx.exception))

    def test_cp_prefix_cache_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=16, dp=1, plugin_params='{"plugin_type":"prefix_cache"}'
            )
        self.assertIn("CP(cp=16) + Prefix Cache", str(ctx.exception))

    def test_cp_splitfuse_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=16, dp=1, plugin_params='{"plugin_type":"splitfuse"}'
            )
        self.assertIn("CP(cp=16) + Chunked Prefill", str(ctx.exception))

    def test_mtp_splitfuse_pd_mixed_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, role="standard", plugin_params='{"plugin_type":"mtp, splitfuse"}'
            )
        self.assertIn("MTP + Chunked Prefill(PD-mixed)", str(ctx.exception))

    # --- Combinations that should pass ---

    def test_single_feature_no_raise(self):
        """Using CP, MTP, Prefix Cache, or splitfuse alone should not raise."""
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=1, dp=8, plugin_params="")
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=16, dp=1, plugin_params="")
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, plugin_params='{"plugin_type":"mtp"}'
            )
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, plugin_params='{"plugin_type":"prefix_cache"}'
            )
        except ValueError:
            self.fail("Single feature alone should not raise ValueError")

    def test_mtp_splitfuse_pd_separated_passes(self):
        """MTP+splitfuse in PD-separated mode (role=prefill/decoder) should pass."""
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, role="prefill", plugin_params='{"plugin_type":"mtp, splitfuse"}'
            )
        except ValueError:
            self.fail("PD-separated mode should not raise for MTP+splitfuse")

    def test_invalid_plugin_params_no_raise(self):
        """Invalid JSON in plugin_params should be handled gracefully."""
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=1, dp=1, plugin_params="not-json")
        except ValueError:
            self.fail("Invalid plugin_params should not raise ValueError from parsing")

    def test_empty_plugin_params_no_raise(self):
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=1, dp=1, plugin_params="")
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=1, dp=1, plugin_params=None)
        except ValueError:
            self.fail("Empty plugin_params should not raise")

    # --- Multiple violations ---

    def test_multiple_violations_aggregated(self):
        with patch.object(model_runner_exp.ENV_utils, "async_inference", True, create=True):
            with self.assertRaises(ValueError) as ctx:
                _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                    cp=16, dp=2, plugin_params='{"plugin_type":"mtp, prefix_cache, splitfuse"}'
                )
        msg = str(ctx.exception)
        self.assertIn("CP(cp=16) + DP(dp=2)", msg)
        self.assertIn("CP(cp=16) + Async Scheduling", msg)
        self.assertIn("CP(cp=16) + Prefix Cache", msg)
        self.assertIn("CP(cp=16) + Chunked Prefill", msg)
        self.assertIn("MTP + Chunked Prefill(PD-mixed)", msg)


class TestDeepseekV32FeatureValidationParametrized(unittest.TestCase):
    """Parametrized tests for validation covering all combination permutations."""

    # (cp, dp, role, plugin_type, async_inference, expected_checks)
    # Each tuple: (cp, dp, role, plugin_type_str, async_flag, list_of_expected_check_substrings)
    _VALIDATION_CASES = [
        # --- Should raise errors ---
        (16, 1, "standard", "", True, ["CP.*16.*Async Scheduling"]),
        (16, 1, "standard", "prefix_cache", False, ["CP.*16.*Prefix Cache"]),
        (16, 1, "standard", "splitfuse", False, ["CP.*16.*Chunked Prefill"]),
        (1, 1, "standard", "mtp, splitfuse", False, ["MTP.*Chunked Prefill.*PD-mixed"]),
        (
            16,
            1,
            "standard",
            "mtp, splitfuse, prefix_cache",
            False,
            ["CP.*16.*Chunked Prefill", "CP.*16.*Prefix Cache", "MTP.*Chunked Prefill.*PD-mixed"],
        ),
        (16, 2, "standard", "", False, ["CP.*16.*DP.*2"]),
        (32, 4, "standard", "", False, ["CP.*32.*DP.*4"]),
        # --- Should pass (no error) ---
        (1, 1, "standard", "", False, []),
        (1, 8, "standard", "", False, []),
        (16, 1, "standard", "", False, []),
        (1, 1, "prefill", "", False, []),
        (1, 1, "decoder", "", False, []),
        (1, 1, "standard", "mtp", False, []),
        (1, 1, "standard", "prefix_cache", False, []),
        (1, 1, "standard", "splitfuse", False, []),
        (1, 1, "prefill", "mtp, splitfuse", False, []),
        (1, 1, "decoder", "mtp, splitfuse", False, []),
        (1, 1, "standard", "prefix_cache, mtp", False, []),
        (16, 1, "prefill", "mtp", False, []),
        (16, 1, "decoder", "mtp", False, []),
    ]

    def test_validate_cases(self):
        for cp, dp, role, plugin_type, async_val, expected in self._VALIDATION_CASES:
            with self.subTest(cp=cp, dp=dp, role=role, plugin_type=plugin_type):
                plugin_params = f'{{"plugin_type":"{plugin_type}"}}' if plugin_type else ""
                with patch.object(model_runner_exp.ENV_utils, "async_inference", async_val, create=True):
                    if expected:
                        with self.assertRaises(ValueError) as ctx:
                            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                                cp=cp, dp=dp, role=role, plugin_params=plugin_params
                            )
                        msg = str(ctx.exception)
                        for pattern in expected:
                            self.assertRegex(msg, pattern)
                    else:
                        try:
                            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                                cp=cp, dp=dp, role=role, plugin_params=plugin_params
                            )
                        except ValueError as e:
                            self.fail(f"Unexpected ValueError: {e}")

    def test_validate_plugin_params_corner_cases(self):
        """Test plugin_params parsing edge cases."""
        cases = [
            (16, '{"plugin_type":" prefix_cache "}', "Prefix Cache", "leading/trail spaces"),
            (16, '{"plugin_type":"prefix_cache,splitfuse"}', "Prefix Cache", "comma no space"),
            (16, '{"plugin_type":"MTP"}', None, "case insensitive should not match"),
            (16, "", None, "empty string"),
            (16, None, None, "None"),
            (
                1,
                '{"plugin_type":"mtp, prefix_cache, splitfuse"}',
                "MTP + Chunked Prefill",
                "all plugins no CP (MTP+splitfuse triggers PD-mixed)",
            ),
        ]
        for cp, plugin_params, expected_substr, desc in cases:
            with self.subTest(case=desc):
                kwargs = {"cp": cp, "dp": 1, "plugin_params": plugin_params}
                if expected_substr:
                    with self.assertRaises(ValueError) as ctx:
                        _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(**kwargs)
                    self.assertIn(expected_substr, str(ctx.exception))
                else:
                    try:
                        _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(**kwargs)
                    except ValueError:
                        self.fail(f"Unexpected ValueError for case: {desc}")

    def test_validate_role_detection(self):
        """Test role-based validation for PD-mixed vs PD-separated."""
        base_params = {"cp": 1, "dp": 1, "plugin_params": '{"plugin_type":"mtp, splitfuse"}'}

        # PD-mixed (role=standard): should raise
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(role="standard", **base_params)
        self.assertIn("MTP", str(ctx.exception))

        # PD-separated roles: should NOT raise
        for role in ("prefill", "decoder", "flex"):
            with self.subTest(role=role):
                try:
                    _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(role=role, **base_params)
                except ValueError:
                    self.fail(f"Role '{role}' should pass for MTP+splitfuse")


class TestKVCacheInfoEdgeCases(unittest.TestCase):
    """Edge cases for KVCacheInfo that tests additional scenarios."""

    def setUp(self):
        self.info = model_runner_exp.KVCacheInfo()
        self.t_a = torch.tensor([1, 2, 3])
        self.t_b = torch.tensor([4, 5, 6])

    def test_check_diff_none_vcache_shape_initial(self):
        """KVCacheInfo with only kcache set: vcache None triggers diff."""
        self.info.kcache_id = id(self.t_a)
        self.info.kcache_shape = self.t_a.shape
        self.assertTrue(self.info.check_diff([(self.t_a, self.t_b)]))

    def test_check_diff_none_kcache_id_initial(self):
        """KVCacheInfo with only vcache set: kcache None triggers diff."""
        self.info.vcache_id = id(self.t_b)
        self.info.vcache_shape = self.t_b.shape
        self.assertTrue(self.info.check_diff([(self.t_a, self.t_b)]))

    def test_check_diff_after_full_match_then_change(self):
        """After a full match, changing kcache should trigger diff."""
        a2 = torch.tensor([10, 20])
        b2 = torch.tensor([30, 40])
        self.info.kcache_id = id(a2)
        self.info.vcache_id = id(b2)
        self.info.kcache_shape = a2.shape
        self.info.vcache_shape = b2.shape
        self.assertFalse(self.info.check_diff([(a2, b2)]))

        # Now change to different tensors
        self.assertTrue(self.info.check_diff([(self.t_a, self.t_b)]))

    def test_check_diff_same_id_different_shape(self):
        """kcache same id but different shape: shape_diff triggers."""
        reshaped = self.t_a.reshape(-1, 1)
        # id() is the same since it's the same object
        self.info.kcache_id = id(self.t_a)
        self.info.kcache_shape = reshaped.shape
        self.info.vcache_id = id(self.t_b)
        self.info.vcache_shape = self.t_b.shape
        # Since self.t_a.shape != reshaped.shape
        self.assertTrue(self.info.check_diff([(self.t_a, self.t_b)]))

    def test_check_diff_multiple_tensors_in_kv_caches(self):
        """kv_caches list with multiple entries (tests [0] indexing)."""
        extra = torch.tensor([99])
        self.info.kcache_id = id(self.t_a)
        self.info.kcache_shape = self.t_a.shape
        self.info.vcache_id = id(self.t_b)
        self.info.vcache_shape = self.t_b.shape
        self.assertFalse(self.info.check_diff([(self.t_a, self.t_b), (extra, extra)]))

    def test_check_diff_empty_after_initialized(self):
        """check_diff after all fields set and matching."""
        self.info.kcache_id = id(self.t_a)
        self.info.kcache_shape = self.t_a.shape
        self.info.vcache_id = id(self.t_b)
        self.info.vcache_shape = self.t_b.shape
        self.assertFalse(self.info.check_diff([(self.t_a, self.t_b)]))

    def test_check_diff_reset_to_none(self):
        """After setting to None, should trigger diff again."""
        self.info.kcache_id = id(self.t_a)
        self.info.kcache_shape = self.t_a.shape
        self.info.vcache_id = id(self.t_b)
        self.info.vcache_shape = self.t_b.shape
        self.assertFalse(self.info.check_diff([(self.t_a, self.t_b)]))

        info2 = model_runner_exp.KVCacheInfo()
        self.assertTrue(info2.check_diff([(self.t_a, self.t_b)]))


class TestValidationMethodDirect(unittest.TestCase):
    """Direct tests for the static validation method."""

    def test_validation_with_default_role(self):
        """When role is not provided, it defaults to 'standard'."""
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, plugin_params='{"plugin_type":"mtp, splitfuse"}'
            )
        self.assertIn("MTP + Chunked Prefill", str(ctx.exception))

    def test_validation_with_cp_equals_1(self):
        """CP=1 should never trigger CP-related checks."""
        test_params = [
            (1, 2, "standard", "", False),
            (1, 1, "standard", "", True),
            (1, 1, "standard", "splitfuse", False),
            (1, 1, "standard", "prefix_cache", False),
        ]
        for cp, dp, role, plugin_type, async_val in test_params:
            with self.subTest(cp=cp, dp=dp, plugin_type=plugin_type):
                pp = f'{{"plugin_type":"{plugin_type}"}}' if plugin_type else ""
                with patch.object(model_runner_exp.ENV_utils, "async_inference", async_val, create=True):
                    try:
                        _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                            cp=cp, dp=dp, role=role, plugin_params=pp
                        )
                    except ValueError as e:
                        if "CP" in str(e):
                            self.fail(f"CP=1 should not trigger CP checks: {e}")

    def test_validation_message_format(self):
        """Error message contains all expected informational parts."""
        with patch.object(model_runner_exp.ENV_utils, "async_inference", True, create=True):
            with self.assertRaises(ValueError) as ctx:
                _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                    cp=16, dp=2, plugin_params='{"plugin_type":"mtp, prefix_cache, splitfuse"}'
                )
        msg = str(ctx.exception)
        self.assertIn("DeepSeek-V3.2", msg)
        self.assertIn("Incompatible", msg)
        self.assertIn("deployment guide", msg)
        self.assertIn("CP", msg)

    def test_validation_multiple_cp_violations(self):
        """Multiple CP violations are all listed in one error."""
        with self.assertRaises(ValueError) as ctx:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=16, dp=1, plugin_params='{"plugin_type":"prefix_cache, splitfuse"}'
            )
        msg = str(ctx.exception)
        self.assertIn("Prefix Cache", msg)
        self.assertIn("Chunked Prefill", msg)

    def test_validation_no_plugin_params_key(self):
        """When plugin_params is missing from kwargs entirely."""
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=1, dp=1)
        except ValueError:
            self.fail("Missing plugin_params should not raise")


class TestKVCacheInfoAdditional(unittest.TestCase):
    """Additional KVCacheInfo tests for completeness."""

    def test_kvcache_info_initial_state(self):
        """New KVCacheInfo has all None fields."""
        info = model_runner_exp.KVCacheInfo()
        self.assertIsNone(info.kcache_id)
        self.assertIsNone(info.vcache_id)
        self.assertIsNone(info.kcache_shape)
        self.assertIsNone(info.vcache_shape)

    def test_kvcache_info_repr(self):
        """KVCacheInfo is a plain class without custom __repr__."""
        info = model_runner_exp.KVCacheInfo()
        info.kcache_id = 123
        info.kcache_shape = torch.Size([4, 128])
        self.assertEqual(info.kcache_id, 123)
        self.assertEqual(info.kcache_shape, torch.Size([4, 128]))

    def test_kvcache_info_partial_initialization(self):
        """Partially initialized KVCacheInfo should still work."""
        info = model_runner_exp.KVCacheInfo()
        info.kcache_id = 42
        info.kcache_shape = torch.Size([2, 64])
        self.assertEqual(info.kcache_id, 42)
        self.assertIsNone(info.vcache_id)
        self.assertIsNone(info.vcache_shape)

    def test_check_diff_with_3d_tensors(self):
        """check_diff with 3D tensor shapes."""
        info = model_runner_exp.KVCacheInfo()
        k = torch.zeros(2, 3, 4)
        v = torch.zeros(2, 3, 4)
        info.kcache_id = id(k)
        info.kcache_shape = k.shape
        info.vcache_id = id(v)
        info.vcache_shape = v.shape
        self.assertFalse(info.check_diff([(k, v)]))

    def test_check_diff_shape_mismatch_3d(self):
        """Shape mismatch with 3D tensors."""
        info = model_runner_exp.KVCacheInfo()
        k = torch.zeros(2, 3, 4)
        k2 = torch.zeros(2, 4, 4)
        v = torch.zeros(2, 3, 4)
        info.kcache_id = id(k)
        info.kcache_shape = k2.shape
        info.vcache_id = id(v)
        info.vcache_shape = v.shape
        self.assertTrue(info.check_diff([(k, v)]))


class TestValidationEdgeCases(unittest.TestCase):
    """Edge cases for validation method."""

    def test_validation_with_empty_plugin_type(self):
        """plugin_params with empty plugin_type should not raise."""
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, plugin_params='{"plugin_type":""}'
            )
        except ValueError:
            self.fail("Empty plugin_type should not raise")

    def test_validation_with_extra_plugin_fields(self):
        """plugin_params with extra fields beyond plugin_type."""
        try:
            _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(
                cp=1, dp=1, plugin_params='{"plugin_type":"mtp","num_speculative_tokens":2,"extra_field":"val"}'
            )
        except ValueError:
            self.fail("Extra fields in plugin_params should not raise")

    def test_validation_with_invalid_json_multiple(self):
        """Multiple invalid plugin_params values."""
        for bad_val in ["{invalid}", "", "   ", "null", '{"bad"}']:
            with self.subTest(bad_val=bad_val):
                try:
                    _ModelRunnerExpClass._validate_deepseekv32_feature_combinations(cp=1, dp=1, plugin_params=bad_val)
                except ValueError:
                    self.fail(f"Invalid JSON '{bad_val}' should not raise")


if __name__ == "__main__":
    unittest.main()
