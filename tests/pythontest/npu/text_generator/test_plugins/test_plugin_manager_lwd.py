# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import queue
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from mindie_llm.text_generator.utils.input_metadata import InputMetadata, LwdMetadata
from mindie_llm.text_generator.plugins.plugin_manager_lwd import PluginManagerLwd
from mindie_llm.text_generator.utils.model_input import ModelInputWrapper


class TestPluginLwd(unittest.TestCase):
    def setUp(self):
        generator_backend = Mock()
        kvcache_settings = Mock()
        infer_context = Mock()
        infer_context.spcp_parallel_info = Mock(cp_size=1)
        output_filter = Mock()
        is_mix_model = Mock()
        plugin_list = Mock()
        model_role = Mock()
        self.plugin_lwd = PluginManagerLwd(generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
            plugin_list, model_role)
        self.plugin_lwd.return_queue = queue.Queue()
        self.plugin_lwd.async_inference = True
        self.plugin_lwd.forward_loop = Mock()
        self.metadata_ = Mock()

        self.plugin = self.plugin_lwd

        # ====== model_inputs mock ======
        self.model_inputs = Mock()
        self.model_inputs.input_ids = MagicMock()
        self.model_inputs.position_ids = MagicMock()
        self.model_inputs.input_lengths = MagicMock()

        self.model_inputs.context_length = np.array([1, 2, 3], dtype=np.int32)
        self.model_inputs.max_seq_len = 3

        self.model_inputs.forward_context = Mock()
        self.model_inputs.forward_context.attn_metadata = Mock()
        self.model_inputs.forward_context.attn_metadata.max_seq_len = 3

        # ====== sampling_output mock ======
        self.sampling_output = Mock()
        token_ids_tensor = MagicMock()
        token_ids_tensor.index_select.return_value = MagicMock()
        token_ids_tensor.index_select.return_value.flatten.return_value = np.array([10, 11])
        self.sampling_output.token_ids = token_ids_tensor

        # ====== model_output_wrapper ======
        self.model_output_wrapper = Mock()
        self.model_output_wrapper.sampling_output = self.sampling_output

        # ====== model_input_wrapper ======
        self.model_input_wrapper = Mock()
        self.model_input_wrapper.model_inputs = self.model_inputs

    @classmethod
    def setUpClass(cls):
        print("TestPluginLwd start")

    @classmethod
    def tearDownClass(cls):
        print("TestPluginLwd end")

    def test_lwd_sampling_output(self):
        self.metadata_.batch_size = 1
        self.metadata_.all_sequence_ids = Mock()
        self.plugin_lwd.lwd_sampling_output(self.metadata_)

    def test_should_skip_return_tokens(self):
        self.metadata_.is_prefill = True
        self.metadata_.layerwise_disaggregated_exe_stage = Mock()
        self.metadata_.layerwise_disaggregated_exe_stage.start_exec_layer = 0
        self.metadata_.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        self.plugin_lwd.should_skip_return_tokens(self.metadata_)

        self.metadata_.is_prefill = False
        self.metadata_.layerwise_disaggregated_exe_stage = Mock()
        self.metadata_.layerwise_disaggregated_exe_stage.start_exec_layer = 1
        self.metadata_.layerwise_disaggregated_exe_stage.end_exec_layer = 1
        self.plugin_lwd.should_skip_return_tokens(self.metadata_)

    def test_set_clean_sequence_ids(self):
        generator_backend = Mock()
        kvcache_settings = Mock()
        infer_context = Mock()
        infer_context.spcp_parallel_info = Mock(cp_size=1)
        output_filter = Mock()
        is_mix_model = Mock()
        plugin_list = Mock()
        model_role = Mock()
        self.plugin_lwd = PluginManagerLwd(generator_backend, kvcache_settings, infer_context, output_filter, is_mix_model,
            plugin_list, model_role)
        infer_context.clear_cache = Mock()
        self.plugin_lwd.set_clean_sequence_ids([100, 101])

    def test_generate_token(self):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()

        tmp_metadata = InputMetadata(batch_size=1, is_prefill=True, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata())
        self.plugin_lwd.generate_token(tmp_metadata)

    def test_generate_token_async_edge(self):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.plugin_verify_manager = Mock()

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}

        def to_tensor_mock(data):
            tensor_mock = MagicMock()
            tensor_mock.nonzero.return_value = (np.array([0]),)
            return tensor_mock

        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.forward_from_model_inputs = Mock()
        generator_backend_.to_tensor = Mock(side_effect=to_tensor_mock)
        self.plugin_lwd.generator_backend = generator_backend_

        tmp_metadata = InputMetadata(batch_size=1, is_prefill=True, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata())
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

        self.plugin_lwd.is_mix_model = None
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

    @patch("mindie_llm.text_generator.utils.generation_output.GenerationOutput.fill_dummy")
    def test_generate_token_async_edge1(self, dummy_mock):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.plugin_verify_manager = Mock()

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}

        def to_tensor_mock(data):
            tensor_mock = MagicMock()
            tensor_mock.nonzero.return_value = (np.array([0]),)
            return tensor_mock

        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.forward_from_model_inputs = Mock()
        generator_backend_.to_tensor = Mock(side_effect=to_tensor_mock)
        self.plugin_lwd.generator_backend = generator_backend_
        tmp_metadata = InputMetadata(batch_size=1, is_prefill=False, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata(end_exec_layer=0))
        output_res = Mock()
        output_res.input_metadata = Mock()
        output_res.input_metadata.is_prefill = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        output_res.input_metadata.all_sequence_ids = [0]
        output_res.cache_ids = [0]
        self.plugin_lwd.output_queue.put(output_res)

        self.plugin_lwd.postprocess.return_value = MagicMock()
        self.plugin_lwd.postprocess.return_value.sequence_ids = [0, 1]
        self.plugin_lwd._fill_in_model_result = Mock()
        self.plugin_lwd.clean_sequence_ids = [0]
        res_out = MagicMock()
        res_out.sequence_ids = [0]
        res_out.finish_reason = [0]
        self.plugin_lwd.return_queue.put(res_out)
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)
        self.plugin_lwd.plugin_list.append("mtp")
        self.plugin_lwd.is_mix_model = False
        
        def prepare_model_inputs_(model_input, q_lens, spec_mask, sub_model_inputs, hidden_states):
            return [0, 1], {"a": 1}
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

    @patch("mindie_llm.text_generator.utils.generation_output.GenerationOutput.fill_dummy")
    def test_generate_token_async_cloud(self, dummy_mock):
        self.plugin_lwd.preprocess = Mock()
        self.plugin_lwd.preprocess.return_value = [Mock(), Mock(), Mock(), Mock()]
        self.plugin_lwd.is_mix_model = True
        self.plugin_lwd.model_inputs_update_manager = Mock()
        self.plugin_lwd.model_inputs_update_manager.return_value = [Mock(), Mock(), Mock()]
        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.sample_preprocess_manager = Mock()
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.plugin_verify_manager = Mock()

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}

        def to_tensor_mock(data):
            tensor_mock = MagicMock()
            tensor_mock.nonzero.return_value = (np.array([0]),)
            return tensor_mock

        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.forward_from_model_inputs = Mock()
        generator_backend_.to_tensor = Mock(side_effect=to_tensor_mock)
        self.plugin_lwd.generator_backend = generator_backend_
        tmp_metadata = InputMetadata(batch_size=1, is_prefill=True, batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])], batch_max_output_lens=[100, 101], block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101], all_sequence_ids=[100, 101], layerwise_disaggregated_exe_stage=LwdMetadata())
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

        self.plugin_lwd.is_mix_model = None
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

        output_res = Mock()
        output_res.input_metadata = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        output_res.cache_ids = [0]
        self.plugin_lwd.output_queue.put(output_res)
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

        self.plugin_lwd.role_type = "slave"
        tmp_val1 = MagicMock()
        tmp_val1.cpu = MagicMock()
        tmp_val = MagicMock()
        tmp_val.logits = [[tmp_val1]]
        self.plugin_lwd.generator_backend.forward_from_model_inputs = MagicMock(return_value=tmp_val)
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)
        
        self.plugin_lwd.plugin_list.append("mtp")
        
        def prepare_model_inputs_(model_input, q_lens, spec_mask, sub_model_inputs, hidden_states):
            return [0, 1], {"a": 1}
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        self.plugin_lwd.output_queue.put(output_res)
        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)
        
    def test_prepare_inputs_for_longseq_chunk(self):
        layerwise_disaggregated_exe_stage = LwdMetadata()
        input_metadata = Mock()
        input_metadata.layerwise_disaggregated_exe_stage = layerwise_disaggregated_exe_stage
        model_input_wrapper = Mock(spec=ModelInputWrapper)
        model_input_wrapper.model_kwargs = {}
        model_input_wrapper.input_metadata = input_metadata
        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_lengths = [None] * 128
        model_input_wrapper.model_inputs.lm_head_indices = [None] * 128
        model_input_wrapper.model_inputs.input_ids = [None] * 128
        model_input_wrapper.model_inputs.position_ids = [None] * 128
        model_input_wrapper.model_inputs.block_tables = np.zeros((1,128), dtype=np.int32)
        model_input_wrapper.model_inputs.slots = [None] * 128
        self.plugin_lwd.generator_backend.block_size = 2
        self.plugin_lwd.prepare_inputs_for_longseq_chunk(model_input_wrapper)
        model_input_wrapper.input_metadata.layerwise_disaggregated_exe_stage.long_seq_start_idx = 1
        self.plugin_lwd.prepare_inputs_for_longseq_chunk(model_input_wrapper)

    def test_fill_in_model_result_with_updates(self):
        """命中 mask 且 update_indices 非空，完整 scatter 路径"""

        filling_masks = {
            "hit_sequence_ids_mask": np.array([True, False, True]),
            "hit_indices_tensor": MagicMock(),
            "update_indices": np.array([0, 2]),
            "ones_int32": np.array([1, 1], dtype=np.int32),
            "ones_int64": np.array([1, 1], dtype=np.int64),
        }

        self.model_input_wrapper.filling_masks = filling_masks

        self.plugin._fill_in_model_result_exp(
            self.model_input_wrapper,
            self.model_output_wrapper
        )

        # ====== token index_select 被调用 ======
        self.sampling_output.token_ids.index_select.assert_called_once()

        # ====== scatter 被调用 ======
        self.model_inputs.input_ids.scatter_.assert_called_once()
        self.model_inputs.position_ids.scatter_add_.assert_called_once()
        self.model_inputs.input_lengths.scatter_add_.assert_called_once()

        # ====== context_length 更新 ======
        self.assertEqual(
            self.model_inputs.context_length.tolist(),
            [2, 2, 4]
        )

        # ====== max_seq_len 更新 ======
        self.assertEqual(self.model_inputs.max_seq_len, 4)
        self.assertEqual(
            self.model_inputs.forward_context.attn_metadata.max_seq_len,
            4
        )

    def test_fill_in_model_result_no_update_indices(self):
        """命中 mask 但 update_indices 为空，不走 scatter"""

        filling_masks = {
            "hit_sequence_ids_mask": np.array([True, True, False]),
            "hit_indices_tensor": MagicMock(),
            "update_indices": [],
            "ones_int32": np.array([], dtype=np.int32),
            "ones_int64": np.array([], dtype=np.int64),
        }

        self.model_input_wrapper.filling_masks = filling_masks

        self.plugin._fill_in_model_result_exp(
            self.model_input_wrapper,
            self.model_output_wrapper
        )

        # ====== scatter 不应被调用 ======
        self.model_inputs.input_ids.scatter_.assert_not_called()
        self.model_inputs.position_ids.scatter_add_.assert_not_called()
        self.model_inputs.input_lengths.scatter_add_.assert_not_called()

        # ====== context_length 仍然增加 ======
        self.assertEqual(
            self.model_inputs.context_length.tolist(),
            [2, 3, 3]
        )

        # ====== max_seq_len 更新 ======
        self.assertEqual(self.model_inputs.max_seq_len, 3)

    def test_fill_in_model_result_no_hit_mask(self):
        """没有 hit_sequence_ids_mask，整个函数应安全返回"""

        filling_masks = {}
        self.model_input_wrapper.filling_masks = filling_masks

        self.plugin._fill_in_model_result_exp(
            self.model_input_wrapper,
            self.model_output_wrapper
        )

        # ====== 不应调用任何 tensor 操作 ======
        self.sampling_output.token_ids.index_select.assert_not_called()
        self.model_inputs.input_ids.scatter_.assert_not_called()
        self.model_inputs.position_ids.scatter_add_.assert_not_called()
        self.model_inputs.input_lengths.scatter_add_.assert_not_called()

    def test_generate_token_async_edge_longseq_cp_sp(self):
        spcp_parallel_info = Mock()
        spcp_parallel_info.cp_size = 2
        spcp_parallel_info.sp_size = 8
        spcp_parallel_info.scp_size = 16
        self.plugin_lwd.infer_context.spcp_parallel_info = spcp_parallel_info
        self.plugin_lwd.role_type = 0

        self.plugin_lwd.preprocess = Mock()
        mock_model_input = Mock()
        mock_sampling_metadata = Mock()
        self.plugin_lwd.preprocess.return_value = [[0, 1], mock_model_input, mock_sampling_metadata, [0]]

        self.plugin_lwd.model_inputs_update_manager_longseq_chunk_cp = Mock()
        self.plugin_lwd.model_inputs_update_manager_longseq_chunk_cp.return_value = [mock_model_input, None, None]

        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.is_mix_model = False

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}

        def to_tensor_mock(data):
            tensor_mock = MagicMock()
            tensor_mock.nonzero.return_value = (np.array([0]),)
            return tensor_mock

        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.to_tensor = Mock(side_effect=to_tensor_mock)
        generator_backend_.block_size = 128
        tmp_val1 = MagicMock()
        tmp_val1.cpu = MagicMock()
        tmp_val = MagicMock()
        tmp_val.logits = [[tmp_val1]]
        generator_backend_.forward_from_model_inputs = MagicMock(return_value=tmp_val)
        self.plugin_lwd.generator_backend = generator_backend_

        self.plugin_lwd._prepare_masks_for_filling = Mock()
        self.plugin_lwd._prepare_masks_for_filling.return_value = {}

        self.plugin_lwd.last_sequence_ids = []

        lwd_metadata = LwdMetadata(
            start_exec_layer=0,
            end_exec_layer=1,
            end_of_generate_token=False,
            is_prefill=True,
            is_long_seq=True,
            long_seq_start_idx=16384,
            long_seq_end_idx=32768,
            request_dp_empty=False
        )

        input_ids_mock = MagicMock()
        input_ids_mock.shape = [32768]

        tmp_metadata = InputMetadata(
            batch_size=1,
            is_prefill=True,
            batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])],
            batch_max_output_lens=np.array([100, 101]),
            block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101],
            all_sequence_ids=[100, 101],
            layerwise_disaggregated_exe_stage=lwd_metadata
        )

        tmp_metadata.input_ids = input_ids_mock
        tmp_metadata.sp_tokens = np.array([0], dtype=np.int64)
        tmp_metadata.total_seq_num = 32768
        tmp_metadata.batch_seq_len = np.array([32768], dtype=np.int64)
        tmp_metadata.batch_block_tables = np.zeros((1, 2, 256), dtype=np.int32)

        output_res = Mock()
        output_res.input_metadata = Mock()
        output_res.input_metadata.is_prefill = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        output_res.input_metadata.all_sequence_ids = [100, 101]
        output_res.input_metadata.is_dummy_batch = False
        output_res.cache_ids = [0, 1]
        output_res.model_output = Mock()
        output_res.model_output.hidden_states = None
        output_res.model_output.logits = Mock()
        output_res.sampling_metadata = mock_sampling_metadata
        output_res.sampling_output = Mock()
        output_res.launch_done = None
        self.plugin_lwd.output_queue.put(output_res)

        generation_output_mock = MagicMock()
        generation_output_mock.sequence_ids = [100, 101]
        generation_output_mock.finish_reason = [0, 0]
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.postprocess.return_value = generation_output_mock

        self.plugin_lwd.put_prefix_kvcache_to_mempool = Mock()
        self.plugin_lwd.should_skip_return_tokens = Mock()
        self.plugin_lwd.should_skip_return_tokens.return_value = False
        self.plugin_lwd.clean_sequence_ids = None
        self.plugin_lwd.max_generated_tokens = 100

        self.plugin_lwd.generate_token_async_edge(tmp_metadata)

    def test_generate_token_async_cloud_longseq_cp_sp(self):
        spcp_parallel_info = Mock()
        spcp_parallel_info.cp_size = 2
        spcp_parallel_info.sp_size = 8
        spcp_parallel_info.scp_size = 16
        self.plugin_lwd.infer_context.spcp_parallel_info = spcp_parallel_info
        self.plugin_lwd.role_type = 1

        self.plugin_lwd.preprocess = Mock()
        mock_model_input = Mock()
        mock_sampling_metadata = Mock()
        self.plugin_lwd.preprocess.return_value = [[0, 1], mock_model_input, mock_sampling_metadata, [0]]

        self.plugin_lwd.model_inputs_update_manager_longseq_chunk_cp = Mock()
        self.plugin_lwd.model_inputs_update_manager_longseq_chunk_cp.return_value = [mock_model_input, None, None]

        self.plugin_lwd.plugin_list = ["atb", "acc"]
        self.plugin_lwd.is_mix_model = False

        def read_lock_():
            import threading
            lock_ = threading.Lock()
            return lock_

        def prepare_model_inputs_(model_input, q_lens, attn_mask):
            return [0, 1], {"a": 1}

        def to_tensor_mock(data):
            tensor_mock = MagicMock()
            tensor_mock.nonzero.return_value = (np.array([0]),)
            return tensor_mock

        generator_backend_ = Mock()
        generator_backend_.get_new_stream = read_lock_
        generator_backend_.prepare_model_inputs = prepare_model_inputs_
        generator_backend_.dp = 0
        generator_backend_.to_tensor = Mock(side_effect=to_tensor_mock)
        generator_backend_.block_size = 128
        tmp_val1 = MagicMock()
        tmp_val1.cpu = MagicMock()
        tmp_val = MagicMock()
        tmp_val.logits = [[tmp_val1]]
        generator_backend_.forward_from_model_inputs = MagicMock(return_value=tmp_val)
        self.plugin_lwd.generator_backend = generator_backend_

        self.plugin_lwd._prepare_masks_for_filling = Mock()
        self.plugin_lwd._prepare_masks_for_filling.return_value = {}

        self.plugin_lwd.last_sequence_ids = []

        lwd_metadata = LwdMetadata(
            start_exec_layer=0,
            end_exec_layer=1,
            end_of_generate_token=False,
            is_prefill=True,
            is_long_seq=True,
            long_seq_start_idx=2048,
            long_seq_end_idx=4096,
            request_dp_empty=False
        )

        input_ids_mock = MagicMock()
        input_ids_mock.shape = [32768]

        tmp_metadata = InputMetadata(
            batch_size=1,
            is_prefill=True,
            batch_request_ids=np.array([0]),
            batch_sequence_ids=[np.array([0])],
            batch_max_output_lens=np.array([100, 101]),
            block_tables=np.array([[0, 1]]),
            reserved_sequence_ids=[100, 101],
            all_sequence_ids=[100, 101],
            layerwise_disaggregated_exe_stage=lwd_metadata
        )

        tmp_metadata.input_ids = input_ids_mock
        tmp_metadata.sp_tokens = np.array([0], dtype=np.int64)
        tmp_metadata.total_seq_num = 32768
        tmp_metadata.batch_seq_len = np.array([32768], dtype=np.int64)
        tmp_metadata.batch_block_tables = np.zeros((1, 2, 256), dtype=np.int32)

        output_res = Mock()
        output_res.input_metadata = Mock()
        output_res.input_metadata.is_prefill = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage = Mock()
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token = True
        output_res.input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer = 0
        output_res.input_metadata.all_sequence_ids = [100, 101]
        output_res.input_metadata.is_dummy_batch = False
        output_res.cache_ids = [0, 1]
        output_res.model_output = Mock()
        output_res.model_output.hidden_states = None
        output_res.model_output.logits = Mock()
        output_res.sampling_metadata = mock_sampling_metadata
        output_res.sampling_output = Mock()
        output_res.launch_done = None
        self.plugin_lwd.output_queue.put(output_res)

        generation_output_mock = MagicMock()
        generation_output_mock.sequence_ids = [100, 101]
        generation_output_mock.finish_reason = [0, 0]
        self.plugin_lwd.postprocess = Mock()
        self.plugin_lwd.postprocess.return_value = generation_output_mock

        self.plugin_lwd.put_prefix_kvcache_to_mempool = Mock()
        self.plugin_lwd.should_skip_return_tokens = Mock()
        self.plugin_lwd.should_skip_return_tokens.return_value = False
        self.plugin_lwd.clean_sequence_ids = None
        self.plugin_lwd.max_generated_tokens = 100

        self.plugin_lwd.generate_token_async_cloud(tmp_metadata)

if __name__ == "__main__":
    unittest.main()