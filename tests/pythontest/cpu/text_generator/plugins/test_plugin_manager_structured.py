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
from unittest.mock import Mock

import numpy as np
import torch

from mindie_llm.text_generator.plugins.plugin_manager import PluginManager


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


if __name__ == "__main__":
    unittest.main()
