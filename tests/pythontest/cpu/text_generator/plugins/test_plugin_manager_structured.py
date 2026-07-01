# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""PluginManager 结构化输出相关路径（Mock，CPU 可运行，不依赖 NPU）"""

import unittest
import sys
import queue
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType, PluginManager
from mindie_llm.text_generator.utils.input_metadata import SIMULATE_SEQUENCE_ID
from mindie_llm.utils.log.error_code import ErrorCodeException


@dataclass
class _TensorPayload:
    ids: torch.Tensor
    scores: torch.Tensor
    name: str


def _make_plugin_manager():
    mock_gb = Mock()
    mock_gb.model_wrapper = Mock()
    mock_gb.sampler = Mock()
    mock_gb.rank = 0
    infer = Mock()
    infer.context_params.async_infer = False
    infer.context_params.max_generated_tokens = 100
    return PluginManager(
        generator_backend=mock_gb,
        kvcache_settings=Mock(),
        infer_context=infer,
        output_filter=Mock(),
        is_mix_model=False,
        plugin_list=[],
        model_role="master",
        watcher=Mock(),
    )


def _sampling_metadata_and_output(*, batch: int, is_prefill: bool):
    """构造 postprocess 所需的 sampling_metadata / sampling_output 最小字段。"""
    sm = Mock()
    sm.best_of_array = None
    sm.is_prefill = is_prefill
    sm.use_beam_search_array = None
    sm.all_sequence_ids = np.arange(1, batch + 1)
    sm.group_indices = None

    so = Mock()
    so.token_ids = np.arange(1, batch + 1).reshape(-1, 1)
    so.logprobs = (np.arange(1, batch + 1) * 0.1).reshape(-1, 1)
    if batch == 2:
        so.top_token_ids = np.array([[[1]], [[2]]])
        so.top_logprobs = np.array([[[0.0]], [[0.0]]])
        so.num_top_tokens = np.array([1, 1])
    else:
        so.top_token_ids = np.array([[[1, 2]], [[3, 4]], [[5, 6]]][:batch])
        so.top_logprobs = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]][:batch])
        so.num_top_tokens = np.full(batch, 2, dtype=int)
    so.num_new_tokens = np.ones(batch, dtype=int)
    so.cumulative_logprobs = np.arange(1, batch + 1) * 0.1
    so.finish_reason = np.zeros(batch, dtype=int)
    so.sequence_ids = np.arange(1, batch + 1)
    so.parent_sequence_ids = np.arange(1, batch + 1)
    so.group_indices = None
    return sm, so


# 三元 cache_ids 场景下 postprocess 用例共用的 filter / update 返回值
_POSTPROCESS_FILTER_STD = (
    np.array([0, 1, 0]),
    np.array([1], dtype=np.int64),
    np.array([], dtype=np.int64),
)
_POSTPROCESS_UPDATE_STD = (
    np.array([2], dtype=np.int64),
    np.array([2], dtype=np.int64),
)


def _attach_postprocess_mocks(
    pm,
    *,
    filter_return,
    update_return,
    clear_finished_return,
    output_len_count,
):
    pm.output_filter.filter_finished_sequences = Mock(return_value=filter_return)
    pm.infer_context.update_context = Mock(return_value=update_return)
    pm.infer_context.clear_finished_context = Mock(return_value=clear_finished_return)
    pm.infer_context.clear_aborted_context = Mock()
    pm.infer_context.get_output_len_count = Mock(return_value=output_len_count)
    pm.plugin_cache_update_manager = Mock()
    pm.plugin_cache_clear_manager = Mock()
    pm.filter_splitfuse_token_ids = Mock()


class TestPluginManagerStructuredHelpers(unittest.TestCase):
    """_fill_in_model_result_exp 等辅助方法"""

    def test_fill_in_model_result_exp_scatter_branch(self):
        pm = _make_plugin_manager()
        miw = Mock()
        miw.filling_masks = {
            "hit_sequence_ids_mask": np.array([True, False]),
            "hit_indices_tensor": torch.tensor([0], dtype=torch.long),
            "update_indices": torch.tensor([0], dtype=torch.long),
            "ones_int32": torch.tensor([1], dtype=torch.int32),
            "ones_int64": torch.tensor([1], dtype=torch.int64),
        }
        miw.model_inputs = Mock()
        miw.model_inputs.input_ids = torch.zeros(2, dtype=torch.long)
        miw.model_inputs.position_ids = torch.zeros(2, dtype=torch.long)
        miw.model_inputs.input_lengths = torch.zeros(2, dtype=torch.int32)
        miw.model_inputs.context_length = np.array([5, 6], dtype=np.int32)
        miw.model_inputs.max_seq_len = 6
        miw.model_inputs.q_lens = None
        miw.model_inputs.forward_context = Mock()
        miw.model_inputs.forward_context.attn_metadata = Mock()
        miw.model_inputs.forward_context.attn_metadata.max_seq_len = 0

        mow = Mock()
        flat = torch.tensor([99])
        sel = Mock(return_value=Mock(flatten=Mock(return_value=flat)))
        mow.sampling_output = Mock()
        mow.sampling_output.token_ids = Mock(index_select=sel)

        pm._fill_in_model_result_exp(miw, mow)
        self.assertEqual(int(miw.model_inputs.input_ids[0]), 99)
        self.assertEqual(int(miw.model_inputs.context_length[0]), 6)

    def test_to_host_converts_tensor_fields_and_keeps_plain_fields(self):
        data = _TensorPayload(
            ids=torch.tensor([1, 2], dtype=torch.int64),
            scores=torch.tensor([1.5], dtype=torch.bfloat16),
            name="batch",
        )

        result = PluginManager._to_host(data)

        self.assertIsInstance(result.ids, np.ndarray)
        self.assertIsInstance(result.scores, np.ndarray)
        self.assertEqual(result.name, "batch")

    def test_initialize_loads_plugin_and_sets_prefix_mempool(self):
        pm = _make_plugin_manager()
        pm.plugin_list = ["prefix_cache"]
        prefix_cache = Mock()
        prefix_cache.mempool_type = MemPoolType.SYNC_WRITE
        module = SimpleNamespace(PrefixCachePlugin=Mock(return_value=prefix_cache))

        with (
            patch("mindie_llm.text_generator.plugins.plugin_manager.importlib.import_module", return_value=module),
            patch.object(pm, "_init_structured_output_manager") as init_structured,
        ):
            pm.initialize()

        self.assertIs(pm.prefix_cache, prefix_cache)
        self.assertEqual(pm.mempool_type, MemPoolType.SYNC_WRITE)
        init_structured.assert_called_once()

    def test_wait_put_finish_handles_success_and_timeout(self):
        pm = _make_plugin_manager()
        pm.plugin_list = ["prefix_cache"]
        pm.prefix_cache = Mock()
        pm.prefix_cache.save_timeout = 0.01
        input_metadata = Mock(is_prefill=True)

        pm.prefix_cache.save_event.wait.return_value = True
        pm.wait_put_finish(input_metadata)

        pm.prefix_cache.save_event.wait.return_value = False
        pm.wait_put_finish(input_metadata)

        self.assertEqual(pm.prefix_cache.save_event.wait.call_count, 2)

    def test_reset_async_pipeline_drains_queues_and_primes_output(self):
        pm = _make_plugin_manager()
        pm.async_inference = True
        pm.input_queue = __import__("queue").Queue()
        pm.output_queue = __import__("queue").Queue()
        pm.input_queue.put("old-input")
        pm.output_queue.put("old-output")
        pm.error_code_collected_in_async = "fault"
        pm.previous_batch_is_prefill = True
        pm.mem_det_trigger_counter = 7

        pm.reset_async_pipeline()

        self.assertTrue(pm.input_queue.empty())
        self.assertEqual(pm.output_queue.qsize(), 1)
        self.assertIsNone(pm.error_code_collected_in_async)
        self.assertFalse(pm.previous_batch_is_prefill)
        self.assertEqual(pm.mem_det_trigger_counter, 0)

    def test_init_structured_output_manager_disables_when_tokenizer_missing(self):
        pm = _make_plugin_manager()
        pm.generator_backend.tokenizer = None

        pm._init_structured_output_manager()

        self.assertFalse(pm._structured_output_enabled)
        self.assertIsNone(pm._structured_output_manager)

    def test_mem_det_counter_resets_after_interval(self):
        pm = _make_plugin_manager()
        pm.mem_det_trigger_counter = 1000

        pm.mem_det_trigger_counter_acc()

        self.assertEqual(pm.mem_det_trigger_counter, 0)

    def test_model_inputs_update_manager_adjusts_simulate_context_length(self):
        pm = _make_plugin_manager()
        model_inputs = Mock()
        model_inputs.context_length = [0]
        input_metadata = Mock()
        input_metadata.all_sequence_ids = [SIMULATE_SEQUENCE_ID]
        input_metadata.is_prefill = False

        result, q_len, mask = pm.model_inputs_update_manager(model_inputs, input_metadata, None, [1])

        self.assertIs(result, model_inputs)
        self.assertIsNone(q_len)
        self.assertIsNone(mask)
        self.assertEqual(model_inputs.context_length[0], 1)

    def test_plugin_managers_call_plugin_methods_and_exp_verify(self):
        pm = _make_plugin_manager()
        plugin = Mock()
        plugin.model_inputs_update.return_value = ("updated-inputs", ("q", "mask"))
        plugin.sample_preprocess.return_value = "filtered-logits"
        pm.plugin_list = ["mtp"]
        pm.mtp = plugin
        input_metadata = Mock()
        input_metadata.all_sequence_ids = None
        input_metadata.is_prefill = True

        model_inputs, q_len, mask = pm.model_inputs_update_manager("inputs", input_metadata, "sampling", [1])
        logits = pm.sample_preprocess_manager("logits", "result", "sampling", input_metadata)
        pm.plugin_cache_update_manager([1], "sampling-output", ("result", "sampling"), True)
        pm.plugin_cache_clear_manager([1], "finish")

        self.assertEqual(model_inputs, "updated-inputs")
        self.assertEqual((q_len, mask), ("q", "mask"))
        self.assertEqual(logits, "filtered-logits")
        plugin.plugin_cache_update.assert_called_once()
        plugin.plugin_cache_clear.assert_called_once()

        sampling_output = Mock()
        sampling_output.token_ids = torch.tensor([1, 2])
        sampling_output.logprobs = torch.tensor([0.0, 0.0])
        sampling_output.top_token_ids = torch.tensor([1, 2])
        sampling_output.top_logprobs = torch.tensor([0.0, 0.0])
        with patch("mindie_llm.text_generator.plugins.plugin_manager.ENV") as env:
            env.model_runner_exp = True
            env.async_inference = True
            pm.plugin_verify_manager(sampling_output, [1], "result")
        plugin.plugin_verify_exp.assert_called_once_with(sampling_output, [1], "result")
        self.assertEqual(sampling_output.token_ids.shape, (2, 1))

    def test_fill_in_model_result_exp_delegates_to_plugin_method(self):
        pm = _make_plugin_manager()
        plugin = Mock()
        pm.plugin_list = ["mtp"]
        pm.mtp = plugin
        miw = Mock()
        miw.filling_masks = {"hit_sequence_ids_mask": np.array([True])}
        miw.model_inputs = Mock()
        miw.input_metadata = "metadata"
        miw.model_kwargs = {"a": 1}
        miw.cache_ids = [1]
        mow = Mock()

        pm._fill_in_model_result_exp(miw, mow)

        plugin.fill_in_model_result_exp.assert_called_once_with(
            "metadata", miw.model_inputs, {"a": 1}, mow, miw.filling_masks, [1]
        )


class TestPluginManagerPreprocessWithStructuredOutput(unittest.TestCase):
    """preprocess：compose_model_inputs 后进入结构化 bitmask"""

    def setUp(self):
        self.plugin_manager = _make_plugin_manager()
        self.mock_infer_context = self.plugin_manager.infer_context

    def _wire_compose(self, sampling_metadata):
        mock_model_inputs = Mock()
        self.mock_infer_context.get_batch_context_handles = Mock(return_value=[1, 2])
        self.mock_infer_context.compose_model_inputs = Mock(
            return_value=(mock_model_inputs, sampling_metadata, [100, 200])
        )

    def test_preprocess_with_structured_output_manager(self):
        sm = Mock()
        sm.all_sequence_ids = np.array([1, 2])
        self._wire_compose(sm)

        input_metadata = Mock()
        input_metadata.is_prefill = True
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.batch_predicted_token_ids = None
        input_metadata.batch_response_format = [{"type": "json_object"}, None]

        mgr = Mock()
        self.plugin_manager._structured_output_manager = mgr

        self.assertEqual(len(self.plugin_manager.preprocess(input_metadata)), 4)

    def test_preprocess_mix_model_uses_splitfuse_preprocess(self):
        pm = self.plugin_manager
        pm.is_mix_model = True
        pm.mix_preprocess = Mock()
        pm.infer_context.get_batch_context_handles = Mock(return_value=[9])
        pm.mix_preprocess.splitfuse_preprocess.splitfuse_preprocess.return_value = (
            "model-inputs",
            [10],
            "sampling",
            "q-len",
            "mask",
            ["trace"],
        )
        input_metadata = Mock()
        input_metadata.is_prefill = True
        input_metadata.batch_response_format = None

        result = pm.preprocess(input_metadata, warmup=True, hit_mask=np.array([True]))

        self.assertEqual(result, ([10], "model-inputs", "sampling", ["trace"]))
        self.assertEqual(pm.plugin_data_param.q_len, "q-len")
        self.assertEqual(pm.plugin_data_param.mask, "mask")

    def test_preprocess_decode_structured_uses_context_response_format_and_exp_plugin(self):
        pm = self.plugin_manager
        plugin = Mock()
        plugin.compose_model_inputs_exp.return_value = "sampling-exp"
        pm.plugin_list = ["mtp"]
        pm.mtp = plugin
        sm = Mock()
        sm.is_prefill = False
        self._wire_compose(sm)
        pm._structured_output_manager = Mock()
        pm.infer_context.get_response_format = Mock(return_value=[{"type": "json_object"}])
        input_metadata = Mock()
        input_metadata.is_prefill = False
        input_metadata.batch_response_format = None

        with patch("mindie_llm.text_generator.plugins.plugin_manager.ENV") as env:
            env.model_runner_exp = True
            result = pm.preprocess(input_metadata)

        self.assertEqual(result[2], "sampling-exp")
        pm.infer_context.get_response_format.assert_called_once_with([1, 2])
        pm._structured_output_manager.build_and_assign_structured_guided_bitmask.assert_called_once()

    def test_preprocess_without_structured_response_format_skips_manager(self):
        sm = Mock()
        sm.all_sequence_ids = np.array([1, 2])
        self._wire_compose(sm)

        input_metadata = Mock()
        input_metadata.is_prefill = True
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.batch_response_format = None

        mgr = Mock()
        self.plugin_manager._structured_output_manager = mgr

        self.assertEqual(len(self.plugin_manager.preprocess(input_metadata)), 4)


class TestPluginManagerGenerateFlows(unittest.TestCase):
    def setUp(self):
        self.plugin_manager = _make_plugin_manager()
        self.plugin_manager.model_wrapper.model_runner.clear_internal_tensors = Mock()
        self.plugin_manager.model_wrapper.mapping.attn_dp.rank = 0
        self.plugin_manager.watcher.watch_npu_mem = Mock()
        self.input_metadata = Mock()
        self.input_metadata.block_tables = np.array([[0, -1]])
        self.input_metadata.simulator_ids = ["sim"]
        self.input_metadata.is_prefill = False
        self.input_metadata.is_dummy_batch = False

    def test_generate_token_sync_old_graph_tuple_result_async_mempool(self):
        pm = self.plugin_manager
        pm.plugin_list = []
        pm.mempool_type = MemPoolType.ASYNC_WRITE
        pm.preprocess = Mock(return_value=([1], "model-inputs", "sampling", ["trace"]))
        pm.model_inputs_update_manager = Mock(return_value=("model-inputs", "q", "mask"))
        pm.generator_backend.forward = Mock(return_value=("logits", "hidden"))
        pm.sample_preprocess_manager = Mock(return_value="filtered")
        sampling_output = Mock()
        pm.generator_backend.sample = Mock(return_value=sampling_output)
        expected = Mock()
        pm.postprocess = Mock(return_value=expected)
        pm.wait_put_finish = Mock()

        result = pm.generate_token(self.input_metadata, warmup=False)

        self.assertIs(result, expected)
        self.assertEqual(result.trace_ids, ["trace"])
        self.assertEqual(result.simulator_ids, ["sim"])
        pm.generator_backend.forward.assert_called_once()
        self.assertIn("spec_mask", pm.generator_backend.forward.call_args.kwargs)
        pm.wait_put_finish.assert_called_once_with(self.input_metadata)
        pm.mem_det_trigger_counter_acc()

    def test_generate_token_sync_prefix_async_put_and_inference_pause_force_stop(self):
        pm = self.plugin_manager
        pm.plugin_list = ["prefix_cache"]
        pm.prefix_cache = Mock()
        pm.prefix_cache.mempool_type = MemPoolType.ASYNC_WRITE
        pm.preprocess = Mock(return_value=([1], "model-inputs", "sampling", ["trace"]))
        pm.model_inputs_update_manager = Mock(return_value=("model-inputs", None, None))
        pm.generator_backend.forward = Mock(side_effect=RuntimeError("FORCE STOP requested"))
        pm.is_inference_pause = True

        result = pm.generate_token(self.input_metadata, warmup=False)

        self.assertEqual(result.sequence_ids.size, 0)
        pm.prefix_cache.async_put_prefix_kvcache_to_mempool.assert_called_once_with(self.input_metadata, [1])
        pm.generator_backend.notify_force_stop_exception.assert_called_once()

    def test_generate_token_async_mock_output_and_collected_error_code(self):
        pm = self.plugin_manager
        pm.async_inference = True
        pm.input_queue = queue.Queue()
        pm.output_queue = queue.Queue()
        pm.generator_backend.get_new_stream.return_value.__enter__ = Mock(return_value=None)
        pm.generator_backend.get_new_stream.return_value.__exit__ = Mock(return_value=False)
        pm.generator_backend.dp = 1
        pm.generator_backend.prepare_model_inputs = Mock(return_value=(Mock(), None))
        pm.generator_backend.synchronize = Mock()
        pm.infer_context.last_sampling_metadata.clear = Mock()
        pm.preprocess = Mock(return_value=([1], Mock(), "sampling", ["trace"]))
        pm.model_inputs_update_manager = Mock(
            side_effect=lambda model_input, *args, **kwargs: (model_input, None, None)
        )
        pm._prepare_masks_for_filling = Mock(return_value={})
        pm.postprocess = Mock()
        self.input_metadata.all_sequence_ids = np.array([1, 2])
        self.input_metadata.batch_is_prefill = None
        self.input_metadata.batch_sequence_ids = [np.array([1]), np.array([2])]
        self.input_metadata.is_prefill = False

        model_output_wrapper = Mock()
        model_output_wrapper.is_mock = True
        model_output_wrapper.model_output = None
        model_output_wrapper.cache_ids = None
        model_output_wrapper.input_metadata = self.input_metadata
        model_output_wrapper.sampling_output = Mock()
        model_output_wrapper.execution_done = None
        model_output_wrapper.launch_done = None
        pm.output_queue.put(model_output_wrapper)

        output = pm.generate_token_async(self.input_metadata, warmup=True)

        self.assertEqual(output.sequence_ids.size, 2)
        self.assertFalse(pm.warmup_is_end)
        self.assertTrue(pm.input_queue.qsize() >= 1)

        pm.error_code_collected_in_async = SimpleNamespace(name="FAULT", value="E")
        pm.output_queue.put(model_output_wrapper)
        with self.assertRaises(ErrorCodeException):
            pm.generate_token_async(self.input_metadata)


class TestPluginManagerPostprocess(unittest.TestCase):
    def setUp(self):
        self.plugin_manager = _make_plugin_manager()
        self.plugin_manager.sampler = Mock()
        initial_mod = SimpleNamespace(NPUSocInfo=Mock(return_value=SimpleNamespace(need_nz=False)))
        env_mod = SimpleNamespace(ENV=SimpleNamespace(enable_greedy_search_opt=False))
        sys.modules.setdefault("atb_llm", SimpleNamespace())
        sys.modules.setdefault("atb_llm.utils", SimpleNamespace())
        sys.modules["atb_llm.utils.initial"] = initial_mod
        sys.modules["atb_llm.utils.env"] = env_mod

    def test_postprocess_forks_context_and_clears_finished_sampling_cache(self):
        pm = self.plugin_manager
        cache_ids = [10, 11, 12]
        sampling_metadata, sampling_output = _sampling_metadata_and_output(batch=3, is_prefill=True)
        sampling_metadata.best_of_array = np.array([2, 1, 1])
        sampling_output.is_structured_accepted = None
        input_metadata = Mock()
        input_metadata.is_prefill = True
        input_metadata.is_dummy_batch = False
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        forked_cache_ids = [20, 21, 22]
        pm.infer_context.fork_context = Mock(return_value=forked_cache_ids)
        _attach_postprocess_mocks(
            pm,
            filter_return=_POSTPROCESS_FILTER_STD,
            update_return=_POSTPROCESS_UPDATE_STD,
            clear_finished_return=np.array([2], dtype=np.int64),
            output_len_count=np.array([3, 4, 5], dtype=np.int64),
        )

        output = pm.postprocess(cache_ids, input_metadata, np.zeros((3, 1)), sampling_metadata, sampling_output)

        pm.infer_context.fork_context.assert_called_once_with(sampling_output)
        pm.sampler.clear_cache.assert_called_once()
        pm.plugin_cache_update_manager.assert_called_once()
        pm.plugin_cache_clear_manager.assert_called_once_with(forked_cache_ids, _POSTPROCESS_FILTER_STD[0])
        np.testing.assert_array_equal(output.current_token_indices, np.array([3, 4, 5]))

    def test_postprocess_without_sampling_uses_input_sequence_ids_and_skips_dummy_clear(self):
        pm = self.plugin_manager
        pm.async_inference = True
        sampling_output = Mock()
        sampling_output.is_structured_accepted = np.array([True, True])
        sampling_output.token_ids = np.array([[1], [2]])
        sampling_output.logprobs = np.zeros((2, 1), dtype=np.float32)
        sampling_output.top_token_ids = np.zeros((2, 1, 1), dtype=np.int64)
        sampling_output.top_logprobs = np.zeros((2, 1, 1), dtype=np.float32)
        sampling_output.num_new_tokens = np.ones(2, dtype=np.int64)
        sampling_output.num_top_tokens = np.zeros(2, dtype=np.int64)
        sampling_output.cumulative_logprobs = np.zeros(2, dtype=np.float32)
        sampling_output.group_indices = [(0, 2)]

        input_metadata = Mock()
        input_metadata.is_prefill = False
        input_metadata.is_dummy_batch = True
        input_metadata.all_sequence_ids = np.array([7, 8], dtype=np.int64)
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        _attach_postprocess_mocks(
            pm,
            filter_return=(np.array([0, 0]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            update_return=(np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            clear_finished_return=np.array([7], dtype=np.int64),
            output_len_count=np.array([1, 1], dtype=np.int64),
        )

        output = pm.postprocess([1, 2], input_metadata, np.zeros((2, 1)), None, sampling_output)

        np.testing.assert_array_equal(output.sequence_ids, np.array([7, 8], dtype=np.int64))
        pm.infer_context.clear_finished_context.assert_not_called()
        pm.sampler.clear_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
