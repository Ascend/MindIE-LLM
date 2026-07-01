# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Lightweight Generator unit tests that avoid real model/NPU initialization."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np


class _LoraOperationStatus:
    UNSUPPORT_CMD = 0
    LORA_CMD_SUCCESS = 1
    INVALID_LORA_PATH = 2
    SLOTS_FULL = 3
    DUPLICATED_LORA_ID = 4
    INVALID_LORA_ID = 5
    INVALID_LORA_RANK = 6
    INVALID_lora_name = 7


mock_model_execute_data_pb2 = MagicMock()
mock_model_execute_data_pb2.LoraOperationStatus = _LoraOperationStatus
sys.modules["mindie_llm.connector.common.model_execute_data_pb2"] = mock_model_execute_data_pb2

from mindie_llm.text_generator.generator import (  # noqa: E402
    DmiModeNodeRole,
    Generator,
    PDModelConfig,
    STANDARD_TAG,
    WarmupParams,
)
from mindie_llm.utils.log.error_code import ErrorCode, ErrorCodeException  # noqa: E402
from mindie_llm.text_generator.utils.config import ResponseConfig  # noqa: E402
from mindie_llm.text_generator.utils.generation_output import GenerationOutput  # noqa: E402


def _make_generator():
    with patch.object(Generator, "__init__", return_value=None):
        generator = Generator({})
    generator.rank = 0
    generator.block_size = 4
    generator.scp_size = 1
    generator.is_mix_model = False
    generator.num_speculative_tokens = 0
    generator.separate_deployment_worker = None
    generator.async_inference = False
    generator.generator_backend = Mock()
    generator.generator_backend.enable_dap = True
    generator.generator_backend.kv_pool_backend = "pool"
    generator.generator_backend.backend_type = "atb"
    generator.generator_backend.update_cache_policy = Mock()
    generator.model_wrapper = Mock()
    generator.model_wrapper.adapter_manager = None
    generator.model_info = Mock()
    generator.cpu_mem = 2
    generator.backend_type = "atb"
    generator.enable_dap = False
    generator.enable_prefix_cache = False
    generator.enable_mtp = False
    generator.max_generated_tokens = 1
    generator.warmup_topk_size = 16
    generator.enable_warmup_with_sampling = False
    generator.vocab_size = 100
    generator.is_multimodal = False
    generator.distributed_enable = False
    generator.dp_size = 1
    generator.cp_size = 1
    generator.sp_size = 1
    generator.max_batch_size = 2
    generator.max_prefill_batch_size = 2
    generator.max_prefill_tokens = 8
    generator.layerwise_disaggregated = False
    generator.pd_config = SimpleNamespace(model_role=STANDARD_TAG)
    return generator


def _generation_output(finish_reason, num_new_tokens=None):
    finish_reason = np.asarray(finish_reason, dtype=np.int32)
    if num_new_tokens is None:
        num_new_tokens = np.ones_like(finish_reason)
    return GenerationOutput(
        sequence_ids=np.arange(len(finish_reason)),
        parent_sequence_ids=np.arange(len(finish_reason)),
        group_indices=[(0, len(finish_reason))],
        token_ids=np.ones((len(finish_reason), 1), dtype=np.int64),
        logprobs=np.zeros((len(finish_reason), 1), dtype=np.float32),
        top_token_ids=np.zeros((len(finish_reason), 1, 1), dtype=np.int64),
        top_logprobs=np.zeros((len(finish_reason), 1, 1), dtype=np.float32),
        num_new_tokens=np.asarray(num_new_tokens, dtype=np.int64),
        num_top_tokens=np.zeros(len(finish_reason), dtype=np.int64),
        cumulative_logprobs=np.zeros(len(finish_reason), dtype=np.float32),
        finish_reason=finish_reason,
        truncation_indices=np.zeros(len(finish_reason), dtype=np.int64),
        current_token_indices=np.zeros(len(finish_reason), dtype=np.int64),
    )


def _profiler_patches():
    return (
        patch("mindie_llm.utils.prof.profiler.Level", SimpleNamespace(DETAILED="detailed")),
        patch("mindie_llm.utils.prof.profiler.span_start", return_value="prof"),
        patch("mindie_llm.utils.prof.profiler.span_attr", side_effect=lambda prof, *args, **kwargs: prof),
        patch("mindie_llm.utils.prof.profiler.span_end"),
    )


class TestGeneratorConfigAndWarmupHelpers(unittest.TestCase):
    def test_warmup_params_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "max_seq_len must be a positive integer"):
            WarmupParams(max_seq_len=0)

    def test_pd_model_config_parses_super_device_fields(self):
        config = PDModelConfig(
            {
                "role": "prefill",
                "local_instance_id": "3",
                "remote_device_ips": "1.1.1.1,2.2.2.2",
                "local_super_pod_id": "7",
                "local_super_device_id": "9",
                "kv_rdma_sl": "1",
                "kv_rdma_tc": "2",
            }
        )

        self.assertEqual(config.model_role, "prefill")
        self.assertEqual(config.local_cluster_id, 3)
        self.assertEqual(config.remote_device_ips, ["1.1.1.1", "2.2.2.2"])
        self.assertEqual(config.local_super_pod_id, 7)
        self.assertEqual(config.local_super_device_id, 9)
        self.assertEqual(config.kv_rdma_sl, 1)
        self.assertEqual(config.kv_rdma_tc, 2)

    def test_temporarily_disable_restores_generator_backend_flags(self):
        generator = _make_generator()
        generator.async_inference = True
        generator.backend_type = "atb"

        with generator._temporarily_disable(dap=True, async_inference=True, mem_pool="pool"):
            self.assertFalse(generator.generator_backend.enable_dap)
            self.assertFalse(generator.async_inference)
            self.assertEqual(generator.generator_backend.kv_pool_backend, "")

        self.assertTrue(generator.generator_backend.enable_dap)
        self.assertTrue(generator.async_inference)
        self.assertEqual(generator.generator_backend.kv_pool_backend, "pool")

    @patch("mindie_llm.text_generator.generator.calc_block_mem", return_value=100)
    def test_validate_warmup_memory_success_and_failure(self, _):
        generator = _make_generator()
        generator.model_info = Mock()
        generator.block_size = 4
        generator.scp_size = 1

        self.assertGreater(generator._validate_warmup_memory(WarmupParams(max_seq_len=8), 1000), 0)
        with self.assertRaisesRegex(RuntimeError, "Required block number"):
            generator._validate_warmup_memory(WarmupParams(max_seq_len=8), 100)

    def test_filter_end_reqs_returns_only_continue_requests(self):
        generator = _make_generator()
        requests = [Mock(name="req0"), Mock(name="req1"), Mock(name="req2")]
        output = _generation_output([ResponseConfig.CONTINUE, 2, ResponseConfig.CONTINUE], [3, 4, 5])

        remaining, token_counts = generator._filter_end_reqs(requests, output)

        self.assertEqual(remaining, [requests[0], requests[2]])
        self.assertEqual(token_counts, [3, 5])

    def test_get_request_lengths_by_dp_covers_distribution_and_prefix_adjustment(self):
        generator = _make_generator()
        generator.pd_config.model_role = STANDARD_TAG
        generator.max_batch_size = 3
        generator.max_prefill_batch_size = 2
        generator.max_prefill_tokens = 10
        generator.cp_size = 2
        generator.layerwise_disaggregated = False

        result = generator._get_request_lengths_by_dp(max_len=6, do_prefix_cache_warmup=True)

        self.assertLessEqual(len(result), 3)
        self.assertLessEqual(result[0], 6)

    def test_update_request_for_prefix_cache_scalar_and_parallel_blocks(self):
        generator = _make_generator()
        generator.dp_size = 2
        generator.scp_size = 1
        reqs = [SimpleNamespace(input_ids=np.ones(8)), SimpleNamespace(input_ids=np.ones(8))]
        generator._update_request_for_prefix_cache(reqs)
        self.assertEqual(reqs[0].computed_blocks, 0)
        self.assertEqual(reqs[1].remote_computed_blocks, 0)

        generator.scp_size = 2
        generator.cp_size = 1
        generator.block_size = 2
        reqs = [SimpleNamespace(input_ids=np.ones(8)), SimpleNamespace(input_ids=np.ones(8))]
        generator._update_request_for_prefix_cache(reqs)
        self.assertEqual(sum(reqs[0].computed_blocks), 4)
        self.assertEqual(len(reqs[0].remote_computed_blocks), 2)

        with self.assertRaisesRegex(ValueError, "must be divisible"):
            generator._update_request_for_prefix_cache([SimpleNamespace(input_ids=np.ones(8))])

    def test_warmup_dispatch_methods_choose_expected_paths(self):
        generator = _make_generator()
        params = WarmupParams()
        generator.enable_prefix_cache = True
        generator.backend_type = "atb"
        generator._auto_warmup_prefill = Mock()
        generator._warmup_prefill(params)
        generator._auto_warmup_prefill.assert_called_once_with(params, do_prefix_cache_warmup=True)

        generator.backend_type = "torch"
        generator._auto_warmup_decode = Mock()
        generator._warmup_decode(params)
        self.assertEqual(generator._auto_warmup_decode.call_count, 2)

        generator.enable_dap = True
        generator._auto_warmup = Mock()
        generator._temporarily_disable = Mock(wraps=generator._temporarily_disable)
        generator.backend_type = "atb"
        generator._warmup_standard(params)
        self.assertEqual(generator._auto_warmup.call_count, 2)

    def test_warmup_specified_dispatches_by_role_and_prints_warning(self):
        generator = _make_generator()
        params = WarmupParams()
        generator._warmup_standard = Mock()
        generator._warmup_prefill = Mock()
        generator._warmup_decode = Mock()

        generator.pd_config.model_role = STANDARD_TAG
        generator._warmup_specified(params, 1024)
        generator._warmup_standard.assert_called_once_with(params)

        generator.pd_config.model_role = "prefill"
        generator._warmup_specified(params, 1024)
        generator._warmup_prefill.assert_called_once_with(params)

        generator.pd_config.model_role = "decoder"
        generator._warmup_specified(params, 1024)
        generator._warmup_decode.assert_called_once_with(params)

    def test_auto_warmup_decode_pushes_prefill_metadata_and_filters_remaining(self):
        generator = _make_generator()
        generator.input_metadata_queue = __import__("queue").Queue()
        generator.block_size = 4
        generator.is_mix_model = False
        generator.scp_size = 1
        req = Mock()
        req.step = Mock()
        generator._generate_warmup_requests = Mock(return_value=[req])
        generator._execute_warm_up = Mock(return_value="decode-output")
        generator._filter_end_reqs = Mock(return_value=([], []))

        with patch("mindie_llm.text_generator.generator.InputMetadata.from_requests", return_value="prefill-metadata"):
            generator._auto_warmup_decode(WarmupParams())

        self.assertEqual(generator.input_metadata_queue.get_nowait(), "prefill-metadata")
        req.step.assert_called_once_with(num_new_token=1, scp_size=1, block_size=4, is_mix_model=False)
        generator._execute_warm_up.assert_called_once_with(requests=[req], is_prefill=False)

    def test_warmup_decode_iteration_raises_for_unfinished_non_torch_backend(self):
        generator = _make_generator()
        generator.backend_type = "atb"
        generator.scp_size = 1
        generator.block_size = 4
        generator.is_mix_model = False
        req = Mock()
        generator._execute_warm_up = Mock(return_value="out")
        generator._filter_end_reqs = Mock(return_value=([req], [1]))

        with self.assertRaisesRegex(RuntimeError, "Decode warmup did not finish"):
            generator._warmup_decode_iteration([req], [1])

        req.step.assert_called_once_with(1, 1, 4, False)

    def test_execute_warm_up_uses_mix_prefill_flags_and_reraises_oom(self):
        generator = _make_generator()
        generator.is_mix_model = True
        generator.npu_mem = 9
        generator.generate_token = Mock(return_value="warm")
        requests = [Mock(), Mock()]

        with patch("mindie_llm.text_generator.generator.InputMetadata.from_requests", return_value="metadata") as frm:
            self.assertEqual(generator._execute_warm_up(requests, is_prefill=False), "warm")

        np.testing.assert_array_equal(frm.call_args.args[2], np.array([False, False]))
        generator.generate_token.assert_called_once_with("metadata", warmup=True)

        generator.generate_token.side_effect = RuntimeError("NPU out of memory: simulated")
        with patch("mindie_llm.text_generator.generator.InputMetadata.from_requests", return_value="metadata"):
            with self.assertRaisesRegex(RuntimeError, "NPU out of memory"):
                generator._execute_warm_up(requests, is_prefill=True)

    def test_get_warm_up_reqs_builds_one_dp_group_when_distributed(self):
        generator = _make_generator()
        generator.distributed_enable = True
        generator.dp_size = 4
        generator.max_generated_tokens = 3
        generator._get_request_lengths_by_dp = Mock(return_value=[2, 3])

        with patch("mindie_llm.text_generator.generator.Request.from_warmup") as from_warmup:
            req0 = Mock(input_length=2)
            req1 = Mock(input_length=3)
            from_warmup.side_effect = [req0, req1]
            reqs = generator._get_warm_up_reqs(WarmupParams(max_seq_len=4), max_output_len=2)

        self.assertEqual(reqs, [req0, req1])
        self.assertEqual(from_warmup.call_count, 2)
        from_warmup.assert_any_call(
            2,
            max_output_len=2,
            max_placeholder_num=3,
            warmup_topk_size=16,
            enable_warmup_sampling=False,
            vocab_size=100,
            is_multimodal=False,
        )
        req0.build.assert_called_once_with(dp_rank_id=0, scp_size=1, block_size=4, is_mix_model=False)

    def test_generate_warmup_requests_uses_mtp_output_len_and_logs(self):
        generator = _make_generator()
        generator.enable_mtp = True
        reqs = [SimpleNamespace(input_length=4), SimpleNamespace(input_length=5)]
        generator._get_warm_up_reqs = Mock(return_value=reqs)

        result = generator._generate_warmup_requests(WarmupParams(), do_prefix_cache_warmup=True)

        self.assertEqual(result, reqs)
        generator._get_warm_up_reqs.assert_called_once()
        self.assertEqual(generator._get_warm_up_reqs.call_args.kwargs["max_output_len"], 2)
        self.assertTrue(generator._get_warm_up_reqs.call_args.kwargs["do_prefix_cache_warmup"])

    def test_auto_warmup_runs_decode_when_prefill_has_remaining_requests(self):
        generator = _make_generator()
        prefill_reqs = [Mock()]
        prefill_output = _generation_output([ResponseConfig.CONTINUE], [2])
        generator._generate_warmup_requests = Mock(return_value=prefill_reqs)
        generator._execute_warm_up = Mock(return_value=prefill_output)
        generator._filter_end_reqs = Mock(return_value=(prefill_reqs, [2]))
        generator._warmup_decode_iteration = Mock()

        generator._auto_warmup(WarmupParams(), do_dap_warmup=True)

        generator._generate_warmup_requests.assert_called_once_with(
            unittest.mock.ANY, do_prefix_cache_warmup=False, do_dap_warmup=True
        )
        generator._execute_warm_up.assert_called_once_with(requests=prefill_reqs, is_prefill=True)
        generator._warmup_decode_iteration.assert_called_once_with(prefill_reqs, [2])

    def test_auto_warmup_prefill_raises_when_decode_reqs_remain(self):
        generator = _make_generator()
        generator._generate_warmup_requests = Mock(return_value=["req"])
        generator._execute_warm_up = Mock(return_value="prefill-output")
        generator._filter_end_reqs = Mock(return_value=(["remaining"], [1]))

        with self.assertRaisesRegex(ValueError, "Expected 0 decode requests"):
            generator._auto_warmup_prefill(WarmupParams(), do_prefix_cache_warmup=True)

        generator._generate_warmup_requests.assert_called_once_with(WarmupParams(), True)

    def test_update_kvcache_settings_builds_separate_deployment_layouts(self):
        generator = _make_generator()
        worker = Mock()
        generator.separate_deployment_worker = worker
        fake_settings = SimpleNamespace(
            k_head_size=8,
            v_head_size=4,
            kvcache_quant_layers=[True, False, True],
            num_layers=4,
            num_npu_blocks=3,
            k_block_shape=(1, 2),
            k_block_quant_shape=(1, 1),
            v_block_shape=(1, 3),
            index_head_dim=1,
            index_block_shape=(1, 4),
            dtype_str="float16",
            backend_type="atb",
        )

        with patch("mindie_llm.text_generator.generator.KVCacheSettings", return_value=fake_settings):
            result = generator._update_kvcache_settings(5)

        self.assertIs(result, fake_settings)
        self.assertEqual(worker.build.call_count, 4)
        generator.generator_backend.update_cache_policy.assert_called_once_with(fake_settings, worker)

    def test_update_kvcache_settings_builds_equal_kv_layout_and_reraises_negative_dim(self):
        generator = _make_generator()
        worker = Mock()
        generator.separate_deployment_worker = worker
        fake_settings = SimpleNamespace(
            k_head_size=4,
            v_head_size=4,
            num_layers=3,
            num_npu_blocks=2,
            k_block_shape=(2, 2),
            dtype_str="float16",
        )

        with patch("mindie_llm.text_generator.generator.KVCacheSettings", return_value=fake_settings):
            result = generator._update_kvcache_settings(5)
        self.assertIs(result, fake_settings)
        worker.build.assert_called_once_with(
            model_id=0, num_tensors=6, num_blocks=2, blockshape=(2, 2), dtype=unittest.mock.ANY
        )

        generator.generator_backend.update_cache_policy.side_effect = RuntimeError(
            "Trying to create tensor with negative dimension -1"
        )
        with patch("mindie_llm.text_generator.generator.KVCacheSettings", return_value=fake_settings):
            with self.assertRaisesRegex(RuntimeError, "negative dimension"):
                generator._update_kvcache_settings(5)


class TestGeneratorPublicWrappers(unittest.TestCase):
    def test_build_inputs_clear_cache_copy_blocks_and_swap_delegate_to_helpers(self):
        generator = _make_generator()
        generator.model_wrapper.make_context = Mock(side_effect=[[1, 2], [3, 4]])
        conversations = [[{"role": "user", "content": "hi"}], [{"role": "assistant", "content": "ok"}]]

        self.assertEqual(generator.build_inputs(conversations, add_generation_prompt=True), [[1, 2], [3, 4]])
        generator.model_wrapper.make_context.assert_any_call(conversations[0], add_generation_prompt=True)

        generator.infer_context = Mock()
        sequence_ids = np.array([1, 2], dtype=np.int64)
        generator.clear_cache(sequence_ids)
        generator.infer_context.clear_context_by_seq_ids.assert_called_once_with(sequence_ids)
        generator.generator_backend.clear_cache.assert_called_once_with(sequence_ids)

        generator.copy_blocks_ops = None
        generator.generator_backend.cache_pool.npu_cache = ["cache"]
        generator.to_tensor = Mock()
        with patch("mindie_llm.text_generator.generator.BlockCopy") as block_copy_cls:
            block_copy = block_copy_cls.return_value
            generator.copy_blocks({1: 2})
        block_copy_cls.assert_called_once_with("atb", ["cache"], generator.to_tensor)
        block_copy.copy_blocks.assert_called_once_with({1: 2})

        generator.swap([[[0, 1, 2]]])
        np.testing.assert_array_equal(generator.generator_backend.swap_cache.call_args.args[0], np.array([[0, 1, 2]]))

    def test_check_batch_size_limit_prefill_and_decode_paths(self):
        generator = _make_generator()
        generator.max_batch_size = 2
        generator.max_prefill_batch_size = 1

        generator.check_batch_size_limit(is_prefill=True, is_mix=False, batch_size=2)
        generator.check_batch_size_limit(is_prefill=False, is_mix=False, batch_size=3)
        generator.check_batch_size_limit(is_prefill=True, is_mix=True, batch_size=3)

    def test_generate_pads_sequence_block_tables_before_building_metadata(self):
        generator = _make_generator()
        generator.generate_token = Mock(return_value="generated")
        seq0 = SimpleNamespace(block_tables=np.array([1, 2], dtype=np.int32))
        seq1 = SimpleNamespace(block_tables=np.array([3], dtype=np.int32))
        requests = [
            SimpleNamespace(sequences={10: seq0}),
            SimpleNamespace(sequences={20: seq1}),
        ]

        with patch("mindie_llm.text_generator.generator.InputMetadata.from_requests", return_value="metadata") as frm:
            self.assertEqual(generator.generate(requests, is_prefill=True), "generated")

        np.testing.assert_array_equal(seq1.block_tables, np.array([3, -1], dtype=np.int32))
        frm.assert_called_once()
        generator.generate_token.assert_called_once_with("metadata")

    def test_prefill_decode_and_generate_mix_delegate_to_generate_token(self):
        generator = _make_generator()
        generator.generate = Mock(side_effect=["prefill-out", "decode-out"])

        self.assertEqual(generator.prefill(["req"]), "prefill-out")
        self.assertEqual(generator.decode(["req"]), "decode-out")
        generator.generate.assert_any_call(["req"], is_prefill=True)
        generator.generate.assert_any_call(["req"], is_prefill=False)

        generator.generate_token = Mock(return_value="mix-out")
        mix_requests = [SimpleNamespace(block_tables=np.array([1])), SimpleNamespace(block_tables=np.array([2]))]
        with patch("mindie_llm.text_generator.generator.InputMetadata.from_requests", return_value="mix-metadata"):
            self.assertEqual(generator.generate_mix(mix_requests, np.array([True, False])), "mix-out")
        generator.generate_token.assert_called_once_with("mix-metadata")


class TestGeneratorStatusMappings(unittest.TestCase):
    def test_load_lora_maps_adapter_manager_errors(self):
        generator = _make_generator()
        manager = Mock()
        generator.model_wrapper.adapter_manager = manager

        manager.load_adapter.return_value = None
        self.assertEqual(generator.load_lora("ok", "/tmp/a"), _LoraOperationStatus.LORA_CMD_SUCCESS)

        cases = [
            (FileNotFoundError(), _LoraOperationStatus.INVALID_LORA_PATH),
            (RuntimeError("LORA MEMORY ERROR"), _LoraOperationStatus.SLOTS_FULL),
            (RuntimeError("DUPLICATED LORA ID"), _LoraOperationStatus.DUPLICATED_LORA_ID),
            (RuntimeError("INVALID LORA ID"), _LoraOperationStatus.INVALID_LORA_ID),
            (RuntimeError("INVALID LORA RANK"), _LoraOperationStatus.INVALID_LORA_RANK),
            (RuntimeError("other"), _LoraOperationStatus.UNSUPPORT_CMD),
        ]
        for exc, expected in cases:
            manager.load_adapter.side_effect = exc
            self.assertEqual(generator.load_lora("bad", "/tmp/b"), expected)

    def test_unload_lora_maps_success_invalid_id_and_unsupported(self):
        generator = _make_generator()
        self.assertEqual(generator.unload_lora("missing-manager"), _LoraOperationStatus.UNSUPPORT_CMD)

        manager = Mock()
        generator.model_wrapper.adapter_manager = manager
        manager.unload_adapter.return_value = None
        self.assertEqual(generator.unload_lora("ok"), _LoraOperationStatus.LORA_CMD_SUCCESS)

        manager.unload_adapter.side_effect = RuntimeError("INVALID LORA ID")
        self.assertEqual(generator.unload_lora("bad"), _LoraOperationStatus.INVALID_lora_name)

        manager.unload_adapter.side_effect = RuntimeError("other")
        self.assertEqual(generator.unload_lora("bad"), _LoraOperationStatus.UNSUPPORT_CMD)

    def test_execute_recover_command_start_pause_roce_clear_and_reinit_failure(self):
        generator = _make_generator()
        generator.npu_device_id = 2
        generator.infer_context = Mock()
        generator.plugin_manager = Mock()
        generator.plugin_manager.reset_async_pipeline = Mock()
        generator.generator_backend.execute_recover_command.return_value = {"command_result": 0, "error_msg": ""}

        start = generator.execute_recover_command("CMD_START_ENGINE")
        self.assertEqual(start["command_result"], 0)
        self.assertFalse(generator.is_inference_pause)
        generator.plugin_manager.reset_async_pipeline.assert_called_once()

        roce = generator.execute_recover_command("CMD_PAUSE_ENGINE_ROCE")
        self.assertEqual(roce["command_result"], 0)
        self.assertTrue(generator.is_inference_pause)
        self.assertTrue(generator.plugin_manager.is_inference_pause)

        clear = generator.execute_recover_command("CMD_CLEAR_TRANSER")
        self.assertEqual(clear["command_result"], 0)

        generator.generator_backend.execute_recover_command.side_effect = RuntimeError("boom")
        reinit = generator.execute_recover_command("CMD_REINIT_NPU")
        self.assertEqual(reinit["command_result"], 1)
        self.assertIn("CMD_REINIT_NPU", reinit["error_msg"])


class TestGeneratorGenerateTokenAndWarmup(unittest.TestCase):
    def _metadata(self):
        metadata = Mock()
        metadata.batch_seq_len = np.array([6, 2], dtype=np.int64)
        metadata.batch_dp_rank_ids = np.array([0, 1], dtype=np.int64)
        metadata.computed_blocks = np.array([1, 0], dtype=np.int64)
        metadata.is_prefill = True
        metadata.is_mix = False
        metadata.all_sequence_ids = np.array([1, 2], dtype=np.int64)
        return metadata

    def test_generate_token_non_async_collates_and_configures_pd_sampler(self):
        generator = _make_generator()
        generator.rank = 0
        generator.max_prefill_tokens = 100
        generator.model_wrapper.mapping.attn_dp.rank = 0
        generator.generator_backend.block_size = 4
        generator.pd_config.model_role = DmiModeNodeRole.DECODER
        generator.input_metadata_queue = __import__("queue").Queue()
        pass_metadata = Mock()
        pass_metadata.is_dummy_batch = False
        generator.input_metadata_queue.put(pass_metadata)
        sampling_metadata = Mock()
        sampling_metadata.do_sample_array = np.array([True])
        generator.infer_context = Mock()
        generator.infer_context.get_batch_context_handles.return_value = [9]
        generator.infer_context.compose_model_inputs.return_value = ("inputs", sampling_metadata, ["trace"])
        output = _generation_output([0, 0])
        output.trace_ids = ["trace"]
        output.simulator_ids = ["sim"]
        generator.plugin_manager = Mock()
        generator.plugin_manager.generate_token.return_value = output
        generator.async_inference = False
        generator.is_inference_pause = False

        metadata = self._metadata()
        metadata.is_prefill = False
        with _profiler_patches()[0], _profiler_patches()[1], _profiler_patches()[2], _profiler_patches()[3]:
            result = generator.generate_token(metadata, warmup=False)

        self.assertIs(result, output)
        generator.generator_backend.configure_sampler.assert_called_once_with(sampling_metadata)
        self.assertIsNotNone(result.eos_info)

    def test_generate_token_async_layerwise_pd_raises_runtime_error(self):
        generator = _make_generator()
        generator.model_wrapper.mapping.attn_dp.rank = 0
        generator.generator_backend.block_size = 4
        generator.pd_config.model_role = "decoder"
        generator.input_metadata_queue = __import__("queue").Queue()
        generator.plugin_manager = Mock()
        generator.async_inference = True
        generator.layerwise_disaggregated = True
        generator.is_inference_pause = False

        with _profiler_patches()[0], _profiler_patches()[1], _profiler_patches()[2], _profiler_patches()[3]:
            with self.assertRaisesRegex(RuntimeError, "Disaggregated-pd"):
                generator.generate_token(self._metadata(), warmup=False)

    def test_generate_token_not_implemented_clears_cache(self):
        generator = _make_generator()
        generator.model_wrapper.mapping.attn_dp.rank = 0
        generator.input_metadata_queue = __import__("queue").Queue()
        generator.plugin_manager = None
        generator.async_inference = False
        generator.clear_cache = Mock()
        metadata = self._metadata()
        metadata.batch_seq_len = np.array([0], dtype=np.int64)

        with _profiler_patches()[0], _profiler_patches()[1], _profiler_patches()[2], _profiler_patches()[3]:
            with self.assertRaises(NotImplementedError):
                generator.generate_token(metadata)

        generator.clear_cache.assert_called_once_with(metadata.all_sequence_ids)

    def test_generate_token_marks_fault_for_error_code_exception(self):
        generator = _make_generator()
        generator.model_wrapper.mapping.attn_dp.rank = 0
        generator.input_metadata_queue = __import__("queue").Queue()
        generator.plugin_manager = Mock()
        err = ErrorCodeException(ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY)
        generator.plugin_manager.generate_token.side_effect = err
        generator.async_inference = False
        metadata = self._metadata()
        metadata.batch_seq_len = np.array([0], dtype=np.int64)

        with _profiler_patches()[0], _profiler_patches()[1], _profiler_patches()[2], _profiler_patches()[3]:
            with self.assertRaises(ErrorCodeException):
                generator.generate_token(metadata)

        self.assertTrue(generator.generator_backend.is_fault_device)

    def test_generate_token_paused_force_stop_returns_empty(self):
        generator = _make_generator()
        generator.model_wrapper.mapping.attn_dp.rank = 0
        generator.input_metadata_queue = __import__("queue").Queue()
        generator.plugin_manager = Mock()
        generator.plugin_manager.generate_token.side_effect = RuntimeError("FORCE STOP now")
        generator.async_inference = False
        generator.is_inference_pause = True
        metadata = self._metadata()
        metadata.batch_seq_len = np.array([0], dtype=np.int64)

        with _profiler_patches()[0], _profiler_patches()[1], _profiler_patches()[2], _profiler_patches()[3]:
            result = generator.generate_token(metadata)

        self.assertEqual(result.sequence_ids.size, 0)
        generator.generator_backend.notify_force_stop_exception.assert_called_once()

    def test_warm_up_specified_soc_multimodal_and_profile_paths(self):
        generator = _make_generator()
        params = WarmupParams(max_prefill_tokens=4, max_seq_len=4, max_input_len=1, max_iter_times=2)
        generator.npu_mem = 9
        generator.soc_version = 240
        generator._warmup_specified = Mock()
        self.assertEqual(generator.warm_up(params), 5)
        generator._warmup_specified.assert_called_once_with(params, 5 * (1024**3))

        generator.npu_mem = -1
        generator.soc_version = None
        generator.is_multimodal = True
        generator.world_size = 2
        generator.model_wrapper.config.num_hidden_layers = 2
        generator.model_wrapper.config.num_key_value_heads = 2
        generator.model_wrapper.config.hidden_size = 8
        generator.model_wrapper.config.num_attention_heads = 4
        self.assertGreaterEqual(generator.warm_up(params), 1)

        generator.is_multimodal = False
        generator.pd_config.model_role = STANDARD_TAG
        generator._warmup_standard = Mock()
        generator._validate_warmup_memory = Mock(return_value=3)
        generator.world_size = 1
        generator.backend_type = "atb"
        with (
            patch("mindie_llm.text_generator.generator.acl") as acl_mod,
            patch("mindie_llm.text_generator.generator.ENV") as env,
        ):
            acl_mod.rt.get_mem_info.return_value = (80, 100, 0)
            env.memory_fraction = 0.9
            self.assertEqual(generator.warm_up(params), 3)
        generator._warmup_standard.assert_called_once_with(params)

    def test_init_plugin_manager_wires_context_output_filter_and_plugin(self):
        generator = _make_generator()
        generator.is_separated_pd = False
        generator.hidden_size = 16
        generator.model_wrapper.model_info.dtype = "float16"
        generator.model_wrapper.model_info.device = "cpu"
        generator.model_wrapper.mapping.attn_inner_sp = "sp"
        generator.model_wrapper.mapping.attn_cp = "cp"
        generator.model_wrapper.generate_position_ids = Mock()
        generator.distributed_enable = False
        generator.layerwise_disaggregated_role_type = ""
        generator.cache_config = Mock()
        generator.tokenizer = Mock()
        generator.tokenizer_sliding_window_size = 3
        generator.watcher = Mock()
        plugin_manager = Mock()

        with (
            patch("mindie_llm.text_generator.generator.TGInferContextStore") as context_cls,
            patch("mindie_llm.text_generator.generator.OutputFilter") as output_filter_cls,
            patch("mindie_llm.text_generator.generator.get_plugin", return_value=plugin_manager) as get_plugin,
        ):
            generator._init_plugin_manager("kv-settings", ["mtp"], {"num_speculative_tokens": 1})

        context_cls.assert_called_once()
        output_filter_cls.assert_called_once_with(
            generator.cache_config, context_cls.return_value, generator.tokenizer, generator.async_inference
        )
        get_plugin.assert_called_once()
        plugin_manager.initialize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
