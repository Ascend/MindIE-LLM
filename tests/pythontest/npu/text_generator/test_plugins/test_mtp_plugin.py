# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Mock missing modules
sys.modules["mindie_llm.connector.common.model_execute_data_pb2"] = MagicMock()
sys.modules["mindie_llm.connector.common.model_execute_data_pb2"].LoraOperationStatus = MagicMock()
sys.modules["mindie_llm.connector.common.model_execute_data_pb2"].LoraOperationStatus.Value = MagicMock(return_value=0)

from mindie_llm.text_generator.generator import Generator  # noqa: E402
from mindie_llm.text_generator.utils.request import Request  # noqa: E402
from mindie_llm.text_generator.utils.input_metadata import InputMetadata  # noqa: E402
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch  # noqa: E402
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams  # noqa: E402
from mindie_llm.text_generator.plugins.mtp.decoding_policy import DecodingPolicy  # noqa: E402
from mindie_llm.text_generator.plugins.mtp.mtp_plugin import MtpPlugin  # noqa: E402
from mindie_llm.text_generator.utils.model_input import ModelInput  # noqa: E402
from tests.pythontest.npu import FakeModelRunner, FakeParallelInfo  # noqa: E402

PLUGIN_PARAMS = '{"plugin_type": "mtp","num_speculative_tokens":1}'
SPECULATION_GAMMA = 1


class TestMTPPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules["_libatb_torch"] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        del sys.modules["_libatb_torch"]

    def setUp(self):
        self.model_config = {
            "backend_bin_path": "/usr/local/Ascend/mindie/2.0.RC1/mindie-llm/bin/",
            "backend_modelInstance_id": "0",
            "backend_type": "atb",
            "block_size": "128",
            "cpu_mem": "0",
            "deploy_type": "INTER_PROCESS",
            "dp": "1",
            "executor_type": "LLM_EXECUTOR_PYTHON",
            "globalRankIds": "",
            "globalWorldSize": "0",
            "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
            "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
            "interNodeTLSEnabled": "1",
            "interNodeTlsCaFiles": "ca.pem,",
            "interNodeTlsCaPath": "security/grpc/ca/",
            "interNodeTlsCert": "security/grpc/certs/server.pem",
            "interNodeTlsCrlFiles": "server_crl.pem,",
            "interNodeTlsCrlPath": "security/grpc/certs/",
            "interNodeTlsPk": "security/grpc/keys/server.key.pem",
            "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
            "isMaster": "0",
            "localIP": "",
            "local_rank": "0",
            "masterIP": "",
            "max_input_len": "2048",
            "max_iter_times": "512",
            "max_prefill_tokens": "8192",
            "max_seq_len": "2560",
            "model_id": "/home/data/llama3",
            "model_instance_number": "1",
            "model_instance_type": "Standard",
            "model_name": "deepseekv3",
            "moe_tp": "1",
            "multiNodesInferEnabled": "0",
            "multiNodesInferPort": "1120",
            "npu_device_id": "0",
            "npu_device_ids": "0,1,2,3",
            "npu_mem": "-1",
            "rank": "0",
            "slaveIPs": "",
            "tp": "4",
            "trust_remote_code": "0",
            "world_size": "4",
            "num_speculative_tokens": "0",
            "max_batch_size": "5",
            "max_prefill_batch_size": "5",
            "distributed_enable": "false",
            "vocab_size": 100000,
            "enable_warmup_with_sampling": "false",
            "cp": "1",
            "sp": "1",
            "moe_ep": "1",
        }

        plugin_dict = {"plugin_params": PLUGIN_PARAMS, "speculation_gamma": SPECULATION_GAMMA}
        self.model_config.update(plugin_dict)

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config["dp"]),
            tp=int(self.model_config["tp"]),
            sp=int(self.model_config["sp"]),
            cp=int(self.model_config["cp"]),
        )

        self.fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)

    @patch("mindie_llm.text_generator.plugins.mtp.mtp_plugin.NPU_STR", "cpu")
    @patch("torch.npu.synchronize", return_value=None)
    @patch("atb_llm.runner.model_runner.ModelRunner")
    @patch.object(Generator, "warm_up")
    @patch.object(Generator, "_update_kvcache_settings")
    @patch.object(Generator, "_init_plugin_manager")
    @patch.object(GeneratorTorch, "_get_obfuscation_func")
    @patch.object(GeneratorTorch, "forward")
    def test_generate_token_greedy(
        self,
        mock_forward,
        mock_obfuscation_func,
        mock_init_plugin_manager,
        mock_update_kvcache_settings,
        mock_warm_up,
        mock_model_runner,
        mock_npu_sync,
    ):
        # Mock _update_kvcache_settings to return an object with num_npu_blocks attribute
        mock_kvcache_settings = MagicMock()
        mock_kvcache_settings.num_npu_blocks = 10
        mock_update_kvcache_settings.return_value = mock_kvcache_settings

        # Mock _init_plugin_manager to avoid plugin manager initialization
        mock_init_plugin_manager.return_value = None

        def side_effect_forward(model_inputs, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            logits_dim0 = 0
            draft_token = torch.zeros(1, dtype=torch.int64)
            if not model_inputs.is_prefill:
                for i in kwargs.get("q_lens"):
                    logits_dim0 += i
            else:
                logits_dim0 = len(model_inputs.context_length)
            if hidden_states is not None:
                logits_dim0 = SPECULATION_GAMMA
            logits = torch.zeros(logits_dim0, 10)  # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4

            if hidden_states is None:
                hidden_states = torch.rand(logits_dim0, 4096)

            return logits, hidden_states, draft_token

        mock_model_runner.return_value = self.fake_model_runner
        mock_forward.side_effect = side_effect_forward
        mock_obfuscation_func.return_value = None
        mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        # Manually set infer_context to avoid initialization errors
        mock_infer_context = MagicMock()
        mock_batch_context = MagicMock()
        mock_sampling_params = MagicMock()
        mock_sampling_params.dtype = np.float32
        mock_batch_context.default_sampling_params = mock_sampling_params
        mock_infer_context._batch_context = mock_batch_context
        generator.infer_context = mock_infer_context

        # Manually set plugin_manager to avoid AttributeError
        mock_plugin_manager = MagicMock()
        # Create a mock generation output that will change finish_reason after a few calls
        call_count = 0

        def mock_generate_token(*args, **kwargs):
            nonlocal call_count
            mock_generation_output = MagicMock()
            # After 5 calls, set finish_reason to 1 to break the loop
            if call_count < 5:
                mock_generation_output.finish_reason = [0]
            else:
                mock_generation_output.finish_reason = [1]
            mock_generation_output.token_ids = [[8]]
            call_count += 1
            return mock_generation_output

        mock_plugin_manager.generate_token = MagicMock(side_effect=mock_generate_token)
        generator.plugin_manager = mock_plugin_manager

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0.0, 0.0, 0, 1, 1, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

        gen_len = 20
        req = Request.request_from_token(
            input1, sampling_params=greedy_param, generation_params=GenerationParams(max_new_tokens=gen_len)
        )
        meta_data = InputMetadata.from_requests([req], block_tables, True)
        meta_data.batch_block_tables = block_tables
        meta_data.block_rank_id = np.array([4])
        meta_data.is_append_block = np.array([False])
        generation_output = generator.generate_token(meta_data)

        # 自回归推理
        meta_data.is_prefill = False
        tokens_list = []
        while generation_output.finish_reason[0] == 0:
            generation_output = generator.generate_token(meta_data)
            tokens_list.extend(generation_output.token_ids[0])

        # 验证greedy是否每轮都选择logits最大的token
        is_greedy = True
        for token in tokens_list:
            if token != 8:
                is_greedy = False
                break
        self.assertTrue(is_greedy)

    @patch("mindie_llm.text_generator.plugins.mtp.mtp_plugin.NPU_STR", "cpu")
    @patch("torch.npu.synchronize", return_value=None)
    @patch("atb_llm.runner.model_runner.ModelRunner")
    @patch.object(Generator, "warm_up")
    @patch.object(Generator, "_update_kvcache_settings")
    @patch.object(Generator, "_init_plugin_manager")
    @patch.object(GeneratorTorch, "_get_obfuscation_func")
    @patch.object(GeneratorTorch, "forward")
    def test_generate_token_sampling(
        self,
        mock_forward,
        mock_obfuscation_func,
        mock_init_plugin_manager,
        mock_update_kvcache_settings,
        mock_warm_up,
        mock_model_runner,
        mock_npu_sync,
    ):
        def side_effect_forward(model_inputs, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            logits_dim0 = 0
            draft_token = torch.zeros(1, dtype=torch.int64)
            if not model_inputs.is_prefill:
                for i in kwargs.get("q_lens"):
                    logits_dim0 += i
            else:
                logits_dim0 = len(model_inputs.context_length)
            if hidden_states is not None:
                logits_dim0 = SPECULATION_GAMMA
            logits = torch.zeros(logits_dim0, 10)  # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4

            if hidden_states is None:
                hidden_states = torch.rand(logits_dim0, 4096)

            return logits, hidden_states, draft_token

        # Mock _update_kvcache_settings to return an object with num_npu_blocks attribute
        mock_kvcache_settings = MagicMock()
        mock_kvcache_settings.num_npu_blocks = 10
        mock_update_kvcache_settings.return_value = mock_kvcache_settings

        # Mock _init_plugin_manager to avoid plugin manager initialization
        mock_init_plugin_manager.return_value = None

        mock_model_runner.return_value = self.fake_model_runner
        mock_obfuscation_func.return_value = None
        mock_forward.side_effect = side_effect_forward
        mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        # Manually set infer_context to avoid initialization errors
        mock_infer_context = MagicMock()
        mock_batch_context = MagicMock()
        mock_sampling_params = MagicMock()
        mock_sampling_params.dtype = np.float32
        mock_batch_context.default_sampling_params = mock_sampling_params
        mock_infer_context._batch_context = mock_batch_context
        generator.infer_context = mock_infer_context

        # Manually set plugin_manager to avoid AttributeError
        mock_plugin_manager = MagicMock()
        # Create a mock generation output that will change finish_reason after a few calls
        call_count = 0

        def mock_generate_token(*args, **kwargs):
            nonlocal call_count
            mock_generation_output = MagicMock()
            # After 5 calls, set finish_reason to 1 to break the loop
            if call_count < 5:
                mock_generation_output.finish_reason = [0]
            else:
                mock_generation_output.finish_reason = [1]
            mock_generation_output.token_ids = [[8]]
            call_count += 1
            return mock_generation_output

        mock_plugin_manager.generate_token = MagicMock(side_effect=mock_generate_token)
        generator.plugin_manager = mock_plugin_manager

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0.0, 0.0, 0.7, 3.0, 0.92, True, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

        gen_len = 20
        req = Request.request_from_token(
            input1, sampling_params=greedy_param, generation_params=GenerationParams(max_new_tokens=gen_len)
        )
        meta_data = InputMetadata.from_requests([req], block_tables, True)
        meta_data.batch_block_tables = block_tables
        meta_data.block_rank_id = np.array([4])
        meta_data.is_append_block = np.array([False])
        generation_output = generator.generate_token(meta_data)

        # 自回归推理
        meta_data.is_prefill = False
        tokens_list = []
        while generation_output.finish_reason[0] == 0:
            generation_output = generator.generate_token(meta_data)
            tokens_list.extend(generation_output.token_ids[0])

        # 验证sample是否每轮都选择topk内的token
        is_sampling_valid = True
        valid_token_set = [2, 5, 8]
        for token in tokens_list:
            if token in valid_token_set:
                continue
            else:
                is_sampling_valid = False
                break
        self.assertTrue(is_sampling_valid)


class TestMTP(unittest.TestCase):
    @patch(
        "mindie_llm.text_generator.plugins.mtp.decoding_policy.DecodingPolicy", return_value=MagicMock(DecodingPolicy)
    )
    def setUp(self, mock):
        # Manually mock NPU_STR to 'cpu'
        import mindie_llm.text_generator.plugins.mtp.mtp_plugin

        original_npu_str = mindie_llm.text_generator.plugins.mtp.mtp_plugin.NPU_STR
        mindie_llm.text_generator.plugins.mtp.mtp_plugin.NPU_STR = "cpu"
        self.device = "cpu"
        generator_backend = MagicMock()
        generator_backend.model_wrapper = MagicMock()
        generator_backend.model_wrapper.device = self.device
        generator_backend.rank = 0
        generator_backend.cache_pool = MagicMock()
        kvcache_settings = MagicMock()
        kvcache_settings.dtype = torch.float16
        infer_context = MagicMock()
        infer_context._batch_context = MagicMock()
        infer_context._batch_context.all_ndarray_context = MagicMock()
        block_size = 128
        num_npu_blocks = 3
        self.num_npu_blocks = num_npu_blocks
        self.block_size = block_size

        def get_seq_lens_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.seq_lens[context_handles]

        def get_mtp_last_token_num_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.mtp_last_token_num[context_handles]

        def get_output_len_count_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.output_len_count[context_handles]

        def get_all_input_ids_side_effect(context_handles):
            return infer_context._batch_context.all_ndarray_context.all_input_ids[context_handles]

        def block_table_to_slots_side_effect(block_tables):
            return infer_context._batch_context.kv_slots[block_tables]

        infer_context.get_seq_lens.side_effect = get_seq_lens_side_effect
        infer_context.get_mtp_last_token_num.side_effect = get_mtp_last_token_num_side_effect
        infer_context.get_output_len_count.side_effect = get_output_len_count_side_effect
        infer_context.get_all_input_ids.side_effect = get_all_input_ids_side_effect
        infer_context.block_table_to_slots.side_effect = block_table_to_slots_side_effect
        infer_context._batch_context.kv_slots = np.arange(num_npu_blocks * block_size).reshape(num_npu_blocks, -1)
        infer_context.spcp_parallel_info.scp_size = 1
        infer_context._batch_context.all_ndarray_context.mtp_last_token_num = np.array([1, 1, 0, 0], dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.seq_lens = np.array([1, 2, 1, 1], dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.all_input_ids = np.full((10, 200), 3, dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.output_len_count = np.array([1, 2, 0, 0], dtype=np.int32)

        generator_backend.cache_pool.kvcache_settings.num_npu_blocks = num_npu_blocks
        model_role = "decode"
        self.mtp_plugin = MtpPlugin(
            generator_backend=generator_backend,
            kvcache_settings=kvcache_settings,
            infer_context=infer_context,
            output_filter=MagicMock(),
            plugin_data_param=MagicMock(),
            num_speculative_tokens=1,
            model_role=model_role,
        )

        # Store original NPU_STR to restore later
        self.original_npu_str = original_npu_str

        self.mtp_plugin.decoding_policy.num_speculative_tokens = 1
        self.mtp_plugin.decoding_policy.infer_context = infer_context

        def to_tensor(data):
            return torch.tensor(data, device=self.device)

        def to_tensor_async(array):
            if self.device == "cpu":
                return torch.from_numpy(array)
            host_tensor = torch.from_numpy(array).pin_memory()
            device_tensor = host_tensor.to(self.device, non_blocking=True)
            return device_tensor

        generator_backend.to_tensor = to_tensor
        generator_backend.to_tensor_async = to_tensor_async

    def tearDown(self):
        # Restore original NPU_STR
        import mindie_llm.text_generator.plugins.mtp.mtp_plugin

        mindie_llm.text_generator.plugins.mtp.mtp_plugin.NPU_STR = self.original_npu_str

    def test_get_mtp_draft_model_inputs_pd_dummy(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(self.mtp_plugin.decoding_policy, "get_input_hidden_states") as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=1,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1], dtype=np.int64),
                block_tables=np.array([[self.num_npu_blocks - 1]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64)],
                batch_best_of=np.array([1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([])],
                is_dummy_batch=True,
            )
            model_inputs = ModelInput(
                input_ids=torch.tensor([1000, 1001]).to(self.device),
                position_ids=torch.tensor([2, 3]).to(self.device),
                block_tables=torch.tensor([[0]]).to(self.device),
                slots=torch.tensor([2, 3]).to(self.device),
                context_length=np.array([4]),
                cached_context_length=np.array([4]),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0]]),
                input_lengths=torch.tensor([4]).to(self.device),
            )

            model_inputs_mtp, hidden_states = self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(
                model_inputs, metadata, 0, None
            )
            self.assertEqual(
                hidden_states.shape, (2, 768), f"Expected hidden_states shape (2, 768), but got {hidden_states.shape}"
            )
            expected = np.array([0, 0])
            self.assertTrue(
                np.array_equal(model_inputs_mtp.input_ids, expected),
                f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}",
            )

    def test_get_mtp_draft_model_inputs_decode(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(self.mtp_plugin.decoding_policy, "get_input_hidden_states") as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=2,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1, 1], dtype=np.int64),
                block_tables=np.array([[0], [0]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1, 1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[
                    np.array([9223372036854775], dtype=np.int64),
                    np.array([9223372036854773], dtype=np.int64),
                ],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([]), np.array([])],
                is_dummy_batch=False,
            )
            model_inputs = ModelInput(
                input_ids=np.array([1000, 1001], dtype=np.int64),
                position_ids=np.array([2, 3], dtype=np.int32),
                block_tables=np.array([[0], [0]], dtype=np.int32),
                slots=np.array([2, 3], dtype=np.int32),
                context_length=np.array([2, 2], dtype=np.int32),
                cached_context_length=np.array([2, 2], dtype=np.int32),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0], [0]]),
                input_lengths=torch.tensor([2, 2]).to(self.device),
            )

            model_inputs_mtp, hidden_states = self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(
                model_inputs, metadata, np.array([1, 1], dtype=np.int32), None
            )
            expected = np.array([3, 0, 3, 0])
            self.assertTrue(
                np.array_equal(model_inputs_mtp.input_ids, expected),
                f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}",
            )

    def test_get_mtp_draft_model_inputs_first_decode(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(self.mtp_plugin.decoding_policy, "get_input_hidden_states") as mock_method:
            mock_method.return_value = mock_hidden_states
            metadata = InputMetadata(
                batch_size=2,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1, 1], dtype=np.int64),
                block_tables=np.array([[0], [0]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1, 1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[
                    np.array([9223372036854775], dtype=np.int64),
                    np.array([9223372036854773], dtype=np.int64),
                ],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([]), np.array([])],
                is_dummy_batch=False,
            )
            model_inputs = ModelInput(
                input_ids=np.array([1000, 1001], dtype=np.int64),
                position_ids=np.array([2, 3], dtype=np.int32),
                block_tables=np.array([[0], [0]], dtype=np.int32),
                slots=np.array([2, 3], dtype=np.int32),
                context_length=np.array([2, 2], dtype=np.int32),
                cached_context_length=np.array([2, 2], dtype=np.int32),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0], [0]]),
                input_lengths=torch.tensor([2, 2]).to(self.device),
            )

            model_inputs_mtp, hidden_states = self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(
                model_inputs, metadata, np.array([2, 3], dtype=np.int32), None
            )
            expected = np.array([3, 0, 3, 0])
            self.assertTrue(
                np.array_equal(model_inputs_mtp.input_ids, expected),
                f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}",
            )

    def test_get_mtp_draft_model_inputs_with_sp(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(self.mtp_plugin.decoding_policy, "get_input_hidden_states") as mock_method:
            mock_method.return_value = mock_hidden_states
            InputMetadata(
                batch_size=2,
                batch_request_ids=np.array([18446744073709551], dtype=np.int64),
                batch_max_output_lens=np.array([1, 1], dtype=np.int64),
                block_tables=np.array([[0], [0]], dtype=np.int64),
                max_block_size=self.block_size,
                has_sampling=False,
                is_prefill=False,
                input_ids=np.array([]),
                batch_seq_len=np.array([1, 1], dtype=np.int64),
                total_seq_num=1,
                batch_sampling_params=np.array([], dtype=np.float64),
                batch_stop_strings=[],
                batch_stop_token_ids=[],
                computed_blocks=None,
                adapter_ids=[],
                num_npu_blocks=self.num_npu_blocks,
                batch_dp_rank_ids=np.array([0], dtype=np.int64),
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64)],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([])],
                is_dummy_batch=False,
            )
            ModelInput(
                input_ids=np.array([1000, 1001], dtype=np.int64),
                position_ids=np.array([2, 3], dtype=np.int32),
                block_tables=np.array([[0], [0]], dtype=np.int32),
                slots=np.array([2, 3], dtype=np.int32),
                context_length=np.array([2, 2], dtype=np.int32),
                cached_context_length=np.array([2, 2], dtype=np.int32),
                max_seq_len=4,
                prefill_head_indices=None,
                is_prefill=False,
                block_tables_array=np.array([[0], [0]]),
                input_lengths=torch.tensor([2, 2]).to(self.device),
            )
            self.mtp_plugin.infer_context.spcp_parallel_info.scp_size = 2
            self.mtp_plugin.decoding_policy.max_block_size = 128
            self.mtp_plugin.decoding_policy.sp_token_and_slot_calc_by_context_length(
                1, np.array([0, 0]), np.array([[0]]), 2
            )

            self.mtp_plugin.infer_context.spcp_parallel_info.scp_size = 1

    def test_all_token_ids_padding(self):
        sampling_metadata = MagicMock()
        sampling_metadata.all_token_ids = torch.tensor(np.full((1, 10), 3, dtype=np.int64), device=self.device)
        draft_tokens = np.full((1, 1), 5, dtype=np.int64)
        next_guess_logits_num_per_batch = np.array([2], dtype=np.int32)
        batch_size = 1
        input_ids_pad = self.mtp_plugin.decoding_policy.all_token_ids_padding(
            sampling_metadata, next_guess_logits_num_per_batch, batch_size, draft_tokens
        )
        expected = torch.tensor(
            [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, -1]], device=self.device
        )

        self.assertTrue(
            torch.equal(input_ids_pad, expected),
            f"Output does not match expected.\nOutput:\n{input_ids_pad.cpu()}\nExpected:\n{expected.cpu()}",
        )

    def test_prepare_masks_for_filling(self):
        # 1. 准备输入数据
        model_inputs = MagicMock()
        # 模拟 block_tables_array [batch_size, num_blocks]
        model_inputs.block_tables_array = np.array([[0], [1], [2]])

        current_dp_sequence_ids = np.array([10, 11, 12])
        current_all_sequence_ids = np.array([10, 11, 12])
        # 模拟只有 10 和 11 在上一轮中存在（即 hit）
        last_all_sequence_ids = np.array([11, 10, 99])

        masks = self.mtp_plugin.prepare_masks_for_filling(
            model_inputs, current_dp_sequence_ids, current_all_sequence_ids, last_all_sequence_ids
        )

        # 3. 断言验证
        self.assertIn("hit_mask", masks)
        # current_dp_ids[0]=10 在 last 中, [1]=11 在 last 中, [2]=12 不在
        np.testing.assert_array_equal(masks["hit_mask"], [True, True, False])
        self.assertIn("hit_indices", masks)
        # 10 在 last 的索引 1, 11 在 last 的索引 0
        np.testing.assert_array_equal(masks["hit_indices"], [1, 0])
        self.assertIn("candidate_slots", masks)
        # 验证返回的是否为 tensor
        self.assertTrue(torch.is_tensor(masks["candidate_slots"]))

    def test_fill_in_model_result_with_hit(self):
        # 1. 构造 Mock 对象和数据
        spec_len = self.mtp_plugin.num_speculative_tokens + 1  # 1 + 1 = 2
        batch_size = 2

        input_metadata = MagicMock()
        cache_ids = [101, 102]

        # 模拟 model_inputs
        model_inputs = MagicMock()
        model_inputs.position_ids = torch.zeros(batch_size * spec_len, device=self.device)
        model_inputs.context_length = np.array([10, 20])
        model_inputs.input_lengths = torch.tensor([10, 20], device=self.device)
        model_inputs.input_ids = torch.zeros(batch_size * spec_len, dtype=torch.long, device=self.device)
        model_inputs.slots = torch.full((batch_size * spec_len,), -1, dtype=torch.long, device=self.device)

        # 模拟 sub_model (MTP) 的输入
        mtp_inputs = MagicMock()
        mtp_inputs.input_ids = torch.zeros(batch_size * spec_len, dtype=torch.long, device=self.device)
        mtp_inputs.prefill_head_indices = torch.tensor([5, 5], device=self.device)

        model_kwargs = {
            "sub_model_inputs": mtp_inputs,
            "hidden_states": torch.zeros((batch_size * spec_len, 128), device=self.device),
            "lm_head_local_dp": None,
        }

        # 模拟 model_output
        model_output_wrapper = MagicMock()
        model_output_wrapper.model_output.hidden_states = torch.randn((batch_size * spec_len, 128), device=self.device)

        sampling_output = MagicMock()
        # 假设 hit 了两个请求，每个请求生成了 2 个 token
        sampling_output.token_ids = np.array([[100, 101], [200, 201]])
        sampling_output.num_new_tokens = np.array([2, 2])
        model_output_wrapper.sampling_output = sampling_output

        # 2. 构造 filling_masks (手动模拟 prepare_masks_for_filling 的输出)
        filling_masks = {
            "hit_mask": np.array([True, True]),
            "hit_mask_tensor": torch.tensor([True, True], device=self.device),
            "hit_indices": np.array([0, 1]),
            "all_hit_indices": np.array([0, 1]),
            "all_hit_mask_tensor": torch.tensor([True, True], device=self.device),
            "hit_mask_per_token_tensor": torch.tensor([True] * (batch_size * spec_len), device=self.device),
            "hit_indices_per_token_tensor": torch.arange(batch_size * spec_len, device=self.device),
            "hit_speculative_length": np.array([spec_len, spec_len]),
            "hit_arange_tensor": torch.arange(batch_size, device=self.device),
            "hit_mask_mod": torch.tensor([False, True, False, True], device=self.device),  # 取模不为0的位
            "hit_increments": torch.tensor([0, 1, 0, 1], device=self.device),
            "candidate_slots": torch.arange(100, 100 + batch_size * 128, device=self.device).reshape(batch_size, 128),
            "hit_block_indices": torch.tensor([0, 0, 1, 1], device=self.device),
        }

        # 3. 执行测试
        self.mtp_plugin.fill_in_model_result(
            input_metadata, model_inputs, model_kwargs, model_output_wrapper, filling_masks, cache_ids
        )

        # 4. 验证核心逻辑
        # 验证 context_length 是否增加了 num_new_tokens (10+2, 20+2)
        np.testing.assert_array_equal(model_inputs.context_length, [12, 22])

        # 验证 MTP 的输入 ids 是否被填入 (sampling_output 的 token_ids)
        expected_ids = torch.tensor([100, 101, 200, 201], device=self.device)
        self.assertTrue(torch.equal(mtp_inputs.input_ids, expected_ids))

        # 验证 prefill_head_indices 修正 (5 - (2-2)) = 5
        self.assertEqual(mtp_inputs.prefill_head_indices[0].item(), 5)

    def test_fill_in_model_result_no_hit(self):
        # 测试 hit_mask 为 None 时不执行任何逻辑
        model_inputs = MagicMock()
        model_inputs.context_length = np.array([10])
        filling_masks = {"hit_mask": None}

        self.mtp_plugin.fill_in_model_result(None, model_inputs, {}, None, filling_masks, None)

        # 验证数据未被修改
        self.assertEqual(model_inputs.context_length[0], 10)

    def test_prepare_masks_for_filling_exp_hit(self):
        """测试 prepare_masks_for_filling_exp 在有命中（hit）情况下的逻辑"""
        # 1. 准备 Mock 数据
        speculative_tokens = self.mtp_plugin.num_speculative_tokens
        speculative_tokens + 1  # 1 + 1 = 2

        model_inputs = MagicMock()
        # 假设 batch_size 为 3, 只有前两个命中
        model_inputs.block_tables_array = np.array([[0], [1], [2]])

        current_dp_ids = np.array([100, 101, 102])
        current_all_ids = np.array([100, 101, 102])
        # 100 对应 last 索引 1, 101 对应 last 索引 0
        last_all_ids = np.array([101, 100, 999])

        # 2. 执行测试
        masks = self.mtp_plugin.prepare_masks_for_filling_exp(
            model_inputs, current_dp_ids, current_all_ids, last_all_ids
        )

        # 3. 验证掩码字典内容
        self.assertIn("hit_mask", masks)
        self.assertIn("hit_mask_local_tensor", masks)
        self.assertIn("all_hit_mask_tensor", masks)  # 这是 masked_select 的结果

        # 验证 hit_mask 内容: [True, True, False]
        np.testing.assert_array_equal(masks["hit_mask"], [True, True, False])

        # 验证 hit_indices: 100 在 last 的 1, 101 在 last 的 0
        np.testing.assert_array_equal(masks["hit_indices"], [1, 0])

        # 验证 hit_mask_local_tensor (torch.masked_select 应该返回命中的原始下标 [0, 1])
        expected_local_indices = torch.tensor([0, 1], device=self.device)
        self.assertTrue(torch.equal(masks["hit_mask_local_tensor"], expected_local_indices))

        # 验证 hit_mask_per_token_mod_tensor
        # 当 spec_len = 2 时，per_token 长度为 6 (3*2)
        # 只有命中请求的起始 token 会被保留在 mod_tensor 中 (即 index 0 和 2)
        expected_mod_indices = torch.tensor([0, 2], device=self.device)
        self.assertTrue(torch.equal(masks["hit_mask_per_token_mod_tensor"], expected_mod_indices))

        # 验证 Tensor 设备
        self.assertEqual(masks["hit_dp_speculative_length"].device.type, self.device)
        self.assertEqual(masks["all_hit_indices"].device.type, self.device)

    def test_prepare_masks_for_filling_exp_no_hit(self):
        """测试 prepare_masks_for_filling_exp 在完全没有命中的情况"""
        model_inputs = MagicMock()
        current_dp_ids = np.array([1, 2])
        current_all_ids = np.array([1, 2])
        last_all_ids = np.array([3, 4])  # 无交集

        masks = self.mtp_plugin.prepare_masks_for_filling_exp(
            model_inputs, current_dp_ids, current_all_ids, last_all_ids
        )

        # 应该返回空字典，因为没有元素满足 hit_mask.any() 或 all_hit_mask.any()
        self.assertEqual(masks, {})

    def test_prepare_masks_for_filling_exp_none_last(self):
        """测试 last_all_sequence_ids 为 None 的情况"""
        masks = self.mtp_plugin.prepare_masks_for_filling_exp(MagicMock(), np.array([1]), np.array([1]), None)
        self.assertEqual(masks, {})

    def test_fill_in_model_result_exp_hit(self):
        """测试 fill_in_model_result_exp 在有命中情况下的全流程填充逻辑"""
        # 1. 基础参数设置
        spec_len = self.mtp_plugin.num_speculative_tokens + 1  # 2
        batch_size = 2
        hidden_dim = 128

        # 2. 构造 model_inputs (模拟模型运行时的输入张量)
        model_inputs = MagicMock()
        model_inputs.position_ids = torch.zeros(batch_size * spec_len, dtype=torch.long, device=self.device)
        model_inputs.context_length = np.array([10, 20])
        model_inputs.input_lengths = torch.zeros(batch_size * spec_len, dtype=torch.int32, device=self.device)
        model_inputs.input_ids = torch.zeros(batch_size * spec_len, dtype=torch.long, device=self.device)
        model_inputs.max_seq_len = 0

        # 模拟 forward_context
        model_inputs.forward_context.attn_metadata.slot_mapping = torch.zeros(batch_size * spec_len, device=self.device)
        model_inputs.forward_context.attn_metadata.max_seq_len = 0

        # 3. 构造 model_kwargs
        mtp_inputs = MagicMock()
        mtp_inputs.input_ids = torch.zeros(batch_size * spec_len, dtype=torch.long, device=self.device)
        mtp_inputs.prefill_head_indices = torch.tensor([10, 10], dtype=torch.int32, device=self.device)

        model_kwargs = {
            "sub_model_inputs": mtp_inputs,
            "hidden_states": torch.zeros((batch_size * spec_len, hidden_dim), device=self.device),
            "input_lengths_sp": None,
        }

        # 4. 模拟模型输出 (Sampling Output)
        model_output_wrapper = MagicMock()
        model_output_wrapper.input_metadata.is_prefill = False
        model_output_wrapper.model_output.hidden_states = torch.ones(
            (batch_size * spec_len, hidden_dim), device=self.device
        )

        sampling_output = MagicMock()
        # token_ids: batch=2, tokens=2
        sampling_output.token_ids = torch.tensor([[101, 102], [201, 202]], device=self.device)
        sampling_output.num_new_tokens = torch.tensor([2, 2], device=self.device)
        sampling_output.num_new_tokens_numpy = np.array([2, 2])
        model_output_wrapper.sampling_output = sampling_output

        # 5. 构造 filling_masks (对应 prepare_masks_for_filling_exp 的输出)
        filling_masks = {
            "hit_mask": np.array([True, True]),
            "all_hit_mask": np.array([True, True]),
            "all_hit_indices": torch.tensor([0, 1], device=self.device),
            "hit_speculative_length": torch.tensor([spec_len, spec_len], device=self.device),
            "all_hit_mask_tensor": torch.tensor([0, 1], device=self.device),
            "hit_indices": np.array([0, 1]),
            "hit_indices_tensor": torch.tensor([0, 1], device=self.device),
            "hit_indices_per_token_tensor": torch.arange(batch_size * spec_len, device=self.device),
            "hit_local_indices_per_token_tensor": torch.arange(batch_size * spec_len, device=self.device),
            "hit_arange_tensor": torch.tensor([0, 1], device=self.device),
            "hit_mask_local_tensor": torch.tensor([0, 2], device=self.device),  # 假设对应 batch 索引
            "hit_dp_speculative_length": torch.tensor([spec_len, spec_len], device=self.device),
            "hit_increments": torch.tensor([0, 1, 0, 1], device=self.device),
            "candidate_slots": torch.ones((batch_size, 256), device=self.device),  # 模拟 slot 池
            "hit_block_indices": torch.tensor([0, 0, 1, 1], device=self.device),
            "hit_mask_per_token_mod_tensor": torch.tensor([0, 2], device=self.device),
        }

        # 6. 执行调用
        input_metadata = MagicMock()
        input_metadata.is_dummy_batch = False

        self.mtp_plugin.fill_in_model_result_exp(
            input_metadata, model_inputs, model_kwargs, model_output_wrapper, filling_masks, cache_ids=[0, 1]
        )

        # 7. 断言验证
        # 验证 MTP 输入 ID 填充 (通过 scatter_)
        expected_mtp_ids = torch.tensor([101, 102, 201, 202], device=self.device)
        self.assertTrue(torch.equal(mtp_inputs.input_ids, expected_mtp_ids))

        # 验证 context_length 更新 (10+2, 20+2)
        np.testing.assert_array_equal(model_inputs.context_length, [12, 22])

        # 验证 scatter_add_ 操作 (prefill_head_indices)
        # subtrahend = 2 - 2 = 0 -> 10 + 0 = 10
        self.assertEqual(mtp_inputs.prefill_head_indices[0], 10)

        # 验证 hidden_states 是否被 scatter 填充 (全 1)
        self.assertEqual(model_kwargs["hidden_states"].sum().item(), batch_size * spec_len * hidden_dim)

        # 验证 max_seq_len 是否更新
        self.assertEqual(model_inputs.max_seq_len, 22)

    def test_fill_in_model_result_exp_empty_mask(self):
        """验证当 mask 为空时不进行任何操作"""
        model_inputs = MagicMock()
        model_inputs.context_length = np.array([10])

        # 没有任何 hit 的 mask
        filling_masks = {"hit_mask": None, "all_hit_mask": None}

        self.mtp_plugin.fill_in_model_result_exp(MagicMock(), model_inputs, {}, MagicMock(), filling_masks, [])

        self.assertEqual(model_inputs.context_length[0], 10)

    def test_sample_preprocess_exp(self):
        """测试采样预处理，验证 logits 提取和参数更新"""
        # 1. 模拟数据
        logits_in = torch.randn(2, 100)
        result_tuple = (logits_in, None, torch.tensor([1, 2]))
        sampling_metadata = MagicMock()
        sampling_metadata.is_prefill = False
        input_metadata = MagicMock()

        # 2. 执行调用
        out_logits = self.mtp_plugin.sample_preprocess_exp(None, result_tuple, sampling_metadata, input_metadata)

        # 3. 断言验证
        self.assertTrue(torch.equal(out_logits, logits_in))
        self.assertEqual(self.mtp_plugin.input_metadata, input_metadata)
        self.assertEqual(self.mtp_plugin.decoding_policy.sampling_param, sampling_metadata)

    def test_compose_model_inputs_exp(self):
        """测试输入组装，验证序列 ID 的重复展开逻辑"""
        # 1. 模拟数据 (batch_size=2, num_speculative_tokens=1 -> spec_len=2)
        sampling_metadata = MagicMock()
        sampling_metadata.all_sequence_ids = [10, 20]
        self.mtp_plugin.num_speculative_tokens = 1

        # 2. 执行调用
        updated_metadata = self.mtp_plugin.compose_model_inputs_exp(sampling_metadata)

        # 3. 断言验证: 每个 ID 应该重复 spec_len 次
        expected_ids = [10, 10, 20, 20]
        np.testing.assert_array_equal(updated_metadata.all_sequence_ids, expected_ids)
        np.testing.assert_array_equal(updated_metadata.parent_sequence_ids, expected_ids)

    def test_plugin_verify_exp_hit_logic(self):
        """测试插件验证逻辑：模拟投机 Token 命中和不命中的过滤"""
        # 1. 设置参数 (num_speculative_tokens=2, spec_len=3)
        self.mtp_plugin.num_speculative_tokens = 2
        self.mtp_plugin.pad_token_id = 0
        self.mtp_plugin.token_range = torch.arange(5, device=self.device)

        # 模拟 decoding 阶段
        self.mtp_plugin.input_metadata = MagicMock()
        self.mtp_plugin.input_metadata.is_prefill = False

        # 2. 构造采样输出
        sampling_output = MagicMock()
        # 假设 batch_size=2，每个请求 3 个 token (总共 6 个)
        # 请求0: [10, 20, 30]
        # 请求1: [40, 50, 60]
        sampling_output.token_ids = torch.tensor([10, 20, 30, 40, 50, 60], device=self.device)
        sampling_output.logprobs = torch.zeros(6, device=self.device)
        sampling_output.top_token_ids = torch.zeros((6, 0), device=self.device)  # 模拟 k=0 情况

        # 模拟 ID 属性
        sampling_output.sequence_ids = [100, 100, 100, 200, 200, 200]
        sampling_output.parent_sequence_ids = [0, 0, 0, 1, 1, 1]

        # 3. 构造 result 包含的 draft_tokens (verify_guess_all)
        # 每个请求有 2 个投机 token
        # 请求0: 投机 [10, 20] -> 全中
        # 请求1: 投机 [40, 99] -> 第一个中，第二个错
        draft_tokens = torch.tensor([[10, 20], [40, 99]], device=self.device)
        result = (None, None, draft_tokens)

        # 4. 执行调用
        self.mtp_plugin.plugin_verify_exp(sampling_output, cache_ids=[0, 1], result=result)

        # 5. 断言验证
        # 请求0 匹配 [10, 20], indices_counts = 2(匹配数) + 1 = 3
        # 请求1 匹配 [40], 99!=50, indices_counts = 1(匹配数) + 1 = 2
        expected_counts = torch.tensor([3, 2], device=self.device)
        self.assertTrue(torch.equal(sampling_output.num_new_tokens, expected_counts))

        # 验证 Token ID 过滤后的结果
        # 重新 view 为 (batch, spec_len) 后：
        # 请求0: [10, 20, 30] (全保留)
        # 请求1: [40, 50, 0] (因为 indices_counts=2，第3个token即 index 2 被 mask 掉)
        res_tokens = sampling_output.token_ids.view(2, 3)
        self.assertEqual(res_tokens[0, 2].item(), 30)
        self.assertEqual(res_tokens[1, 2].item(), 0)  # 被 padding

        # 验证 Logprobs 过滤
        res_logprobs = sampling_output.logprobs.view(2, 3)
        self.assertEqual(res_logprobs[1, 2].item(), -9999.0)

        # 验证 sequence_ids 重新映射 (取每个 batch 的第一个)
        np.testing.assert_array_equal(sampling_output.sequence_ids, [100, 200])

    def test_plugin_verify_exp_prefill(self):
        """测试 prefill 阶段直接返回，不进行校验逻辑"""
        self.mtp_plugin.input_metadata = MagicMock()
        self.mtp_plugin.input_metadata.is_prefill = True

        sampling_output = MagicMock()
        sampling_output.token_ids = torch.tensor([1, 2, 3])

        # 如果 is_prefill 为 True，函数应在处理 draft_token 之前 return
        self.mtp_plugin.plugin_verify_exp(sampling_output, [0], result=None)

        # 验证逻辑：由于 prefill 直接 return，repeating_indices 应该被赋值但 token_ids 不变
        self.assertTrue(hasattr(sampling_output, "repeating_indices"))
        self.assertTrue(torch.equal(sampling_output.token_ids, torch.tensor([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
