# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import json
import sys
from pathlib import Path

import unittest
from unittest.mock import patch, MagicMock
import torch

import numpy as np

from atb_llm.utils.dist import FakeGroup
from atb_llm.utils.layers import AttentionMask
from atb_llm.utils.file_utils import safe_open
from atb_llm.models.llama.config_llama import LlamaConfig
from mindie_llm.utils.env import ENV
from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.text_generator.generator import Generator
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.config import ModelConfig
from mindie_llm.text_generator.utils.input_metadata import InputMetadata
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from mindie_llm.text_generator.plugins.mtp.decoding_policy import DecodingPolicy
from mindie_llm.text_generator.plugins.mtp.mtp_plugin import MtpPlugin
from mindie_llm.text_generator.utils.model_input import ModelInput


current_file_path = Path(__file__).resolve()
target_dir = current_file_path.parent.parent.parent.parent

MODEL_PATH = target_dir.joinpath("test_weights/llama3")
PLUGIN_PARAMS = '{\"plugin_type\": \"mtp\",\"num_speculative_tokens\":1}'
SPECULATION_GAMMA = 1


class FakeModel:
    max_position_embeddings = 12345


class FakeModelRunner:
    def __init__(self):
        with safe_open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
            config_dict = json.loads(f.read())
        
        config = LlamaConfig.from_dict(config_dict)
        self.config = config
        self.config_dict = config_dict
        self.llm_config = MagicMock()
        self.tokenizer = None
        from atb_llm.utils.mapping import Mapping
        self.mapping = Mapping(world_size=ENV.world_size, rank=ENV.local_rank)
        self.process_group = FakeGroup(rank=ENV.local_rank, size=ENV.world_size)
        self.device = torch.device('cpu')
        self.dtype = torch.bfloat16

        self.kv_cache_dtype = torch.float16
        self.num_layers = config_dict['num_hidden_layers']
        self.num_kv_heads = config_dict['num_key_value_heads']
        self.head_size = config_dict['hidden_size'] // config_dict['num_key_value_heads']
        self.k_head_size = self.head_size
        self.v_head_size = self.head_size
        self.kvcache_quant_layers = []

        self.max_position_embeddings = config_dict['max_position_embeddings']
        self.soc_info = None
        self.adapter_manager = None
        self.lora_adapter = None
        self.attn_mask = AttentionMask.static(1024, dtype=torch.float16)
        self.model = None
        self.enable_nz = False

    @staticmethod
    def decode():
        return "A test string"

    @staticmethod
    def generate_position_ids(input_ids):
        return range(len(input_ids))
    
    def load_weights(self, **kwargs):
        self.model = FakeModel()
        self.model.max_position_embeddings = self.max_position_embeddings
        return None

    def clear_internal_tensors(self):
        pass


class TestPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules['_libatb_torch'] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        del sys.modules['_libatb_torch']

    def setUp(self):
        args = MagicMock()
        args.model_path = MODEL_PATH
        args.plugin_params = PLUGIN_PARAMS
        args.speculation_gamma = SPECULATION_GAMMA
        args.load_tokenizer = False
        args.trust_remote_code = True

        env_rank = 'rank'
        env_world_size = 'world_size'
        env_local_rank = 'local_rank'
        input_dict = {
            env_rank: 0,
            env_world_size: 1,
            env_local_rank: 0,
            **vars(args)
        }

        backend_type = input_dict.get('backend_type', 'atb')
        self.backend_type = BackendType.MS if backend_type and backend_type.lower() == BackendType.MS \
            else BackendType.ATB
        
        self.ignore_eos = input_dict.get('ignore_eos', False)
        self.load_tokenizer = input_dict.get('load_tokenizer', True)
        self.rank = input_dict.get(env_rank, '0')
        self.local_rank = input_dict.get(env_local_rank, self.rank)

        self.max_input_length = input_dict.get('max_input_length', 1024)
        self.max_output_length = input_dict.get('max_output_length', 20)
        self.max_batch_size = input_dict.get('max_batch_size', 200)
        self.max_prefill_batch_size = input_dict.get('max_prefill_batch_size', 200)
        self.max_prefill_tokens = input_dict.get('max_prefill_tokens', 4096)
        self.max_position_embeddings = input_dict.get('max_position_embeddings', 2048)
        self.max_seq_len = self.max_position_embeddings if self.max_position_embeddings else \
            self.max_input_length + self.max_output_length
        self.model_path = input_dict.get('model_path')
        npu_id = input_dict.get('npu_id')
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.npu_mem = input_dict.get('npu_mem', 4)
        self.plugin_params = input_dict.get('plugin_params')
        self.speculation_gamma = input_dict.get('speculation_gamma')
        self.world_size = input_dict.get(env_world_size, '1')
        self.eos_token_id = None

        self.model_role = input_dict.get('model_role', 'standard')
        self.local_model_instance_id = input_dict.get('local_model_instance_id', None)
        self.local_device_ip = input_dict.get('local_device_ip', None)
        self.remote_model_instance_ids = input_dict.get('remote_model_instance_ids', None)
        self.remote_device_ips = input_dict.get('remote_device_ips', '')
        self.model_name = "llama"

        self.model_config = {
            'model_name': self.model_name,
            'backend_type': self.backend_type,
            'block_size': 128,
            'cpu_mem': 20,
            'ignore_eos': self.ignore_eos,
            'load_tokenizer': self.load_tokenizer,
            'local_rank': self.local_rank,
            'max_input_len': self.max_input_length,
            'max_iter_times': self.max_output_length,
            'max_batch_size': self.max_batch_size,
            'max_prefill_batch_size': self.max_prefill_batch_size,
            'max_prefill_tokens': self.max_prefill_tokens,
            'max_seq_len': self.max_seq_len,
            'model_id': self.model_path,
            'npu_device_id': self.npu_id,
            'npu_mem': self.npu_mem,
            'num_threads': 8,
            'plugin_params': self.plugin_params,
            'speculation_gamma': self.speculation_gamma,
            'rank': self.rank,
            'world_size': self.world_size,
            'model_role': self.model_role,
            'local_model_instance_id': self.local_model_instance_id,
            'local_device_ip': self.local_device_ip,
            'remote_model_instance_ids': self.remote_model_instance_ids,
            'remote_device_ips': self.remote_device_ips,
            'trust_remote_code': True,
            'vocab_size': 100000,
            'enable_warmup_with_sampling': False
        }

        if isinstance(self.model_config, ModelConfig):
            self.model_config = vars(self.model_config)
        if self.eos_token_id is not None:
            self.model_config['eos_token_id'] = self.eos_token_id

    def test_generate_token_greedy(self):
        
        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

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
            logits = torch.zeros(logits_dim0, 10) # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            
            if hidden_states is None:
                hidden_states = torch.rand(logits_dim0, 4096)
            
            return logits, hidden_states, draft_token
        
        with patch.object(GeneratorTorch, 'forward') as mock_forward, \
             patch('atb_llm.utils.dist.initialize_distributed') as mock_initialize_distributed, \
             patch('atb_llm.runner.model_runner.ModelRunner', return_value=FakeModelRunner()) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.NPUSocInfo.support_nz', return_value=True) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.KVCacheSettings') as mock_kvcache_settings_class, \
             patch('torch.npu.synchronize', return_value=None) as _, \
             patch('mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch._get_obfuscation_func', \
                    return_value=None) as _:
        
            mock_initialize_distributed.side_effect = side_effect_initialize_distributed
            mock_forward.side_effect = side_effect_forward
            mock_kvcache_settings = MagicMock(dtype=torch.float16)
            mock_kvcache_settings_class.return_value = mock_kvcache_settings
            generator = Generator(self.model_config)

            la_plugin = generator.plugin
            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
            
            gen_len = 20
            req = Request.request_from_token(input1, sampling_params=greedy_param,
                                             generation_params=GenerationParams(max_new_tokens=gen_len))
            req.has_sampling = True
            meta_data = InputMetadata.from_requests([req], block_tables, True)
            meta_data.batch_block_tables = block_tables
            meta_data.block_rank_id = np.array([4])
            meta_data.is_append_block = np.array([False])
            generation_output = la_plugin.generate_token(meta_data)

            # 自回归推理
            meta_data.is_prefill = False
            tokens_list = []
            while generation_output.finish_reason[0] == 0:
                generation_output = la_plugin.generate_token(meta_data)
                tokens_list.extend(generation_output.token_ids[0])

            # 验证greedy是否每轮都选择logits最大的token
            is_greedy = True
            for token in tokens_list:
                if token != 8:
                    is_greedy = False
                    break
            self.assertTrue(is_greedy)

    def test_generate_token_sampling(self):
        
        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

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
            logits = torch.zeros(logits_dim0, 10) # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            
            if hidden_states is None:
                hidden_states = torch.rand(logits_dim0, 4096)
            
            return logits, hidden_states, draft_token
        
        with patch.object(GeneratorTorch, 'forward') as mock_forward, \
             patch('atb_llm.utils.dist.initialize_distributed') as mock_initialize_distributed, \
             patch('atb_llm.runner.model_runner.ModelRunner', return_value=FakeModelRunner()) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.NPUSocInfo.support_nz', return_value=True) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.KVCacheSettings') as mock_kvcache_settings_class, \
             patch('torch.npu.synchronize', return_value=None) as _, \
             patch('mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch._get_obfuscation_func', \
                    return_value=None) as _:
        
            mock_initialize_distributed.side_effect = side_effect_initialize_distributed
            mock_forward.side_effect = side_effect_forward
            mock_kvcache_settings = MagicMock(dtype=torch.float16)
            mock_kvcache_settings_class.return_value = mock_kvcache_settings
            generator = Generator(self.model_config)

            la_plugin = generator.plugin
            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, True, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
            
            gen_len = 20
            req = Request.request_from_token(input1, sampling_params=greedy_param,
                                             generation_params=GenerationParams(max_new_tokens=gen_len))
            req.has_sampling = True
            meta_data = InputMetadata.from_requests([req], block_tables, True)
            meta_data.batch_block_tables = block_tables
            meta_data.block_rank_id = np.array([4])
            meta_data.is_append_block = np.array([False])
            generation_output = la_plugin.generate_token(meta_data)

            # 自回归推理
            meta_data.is_prefill = False
            tokens_list = []
            while generation_output.finish_reason[0] == 0:
                generation_output = la_plugin.generate_token(meta_data)
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
    @patch('mindie_llm.text_generator.plugins.mtp.decoding_policy.DecodingPolicy',
           return_value=MagicMock(DecodingPolicy))
    def setUp(self, mock):
        self.device = 'npu'
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
        infer_context._batch_context.all_ndarray_context.mtp_last_token_num = np.array([1,1,0,0], dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.seq_lens = np.array([1,2,1,1], dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.all_input_ids = np.full((10,200), 3, dtype=np.int32)
        infer_context._batch_context.all_ndarray_context.output_len_count = np.array([1,2,0,0], dtype=np.int32)
        
        generator_backend.cache_pool.kvcache_settings.num_npu_blocks = num_npu_blocks
        model_role = 'decode'
        self.mtp_plugin = MtpPlugin(
            generator_backend=generator_backend,
            kvcache_settings=kvcache_settings,
            infer_context=infer_context,
            output_filter=MagicMock(),
            plugin_data_param=MagicMock(),
            num_speculative_tokens=1,
            model_role=model_role
        )

        self.mtp_plugin.decoding_policy.num_speculative_tokens = 1
        self.mtp_plugin.decoding_policy.infer_context = infer_context

        def to_tensor(data):
            return torch.tensor(data, device=self.device)

        def to_tensor_async(array):
            host_tensor = torch.from_numpy(array).pin_memory()
            device_tensor = host_tensor.to(self.device, non_blocking=True)
            return device_tensor

        generator_backend.to_tensor = to_tensor
        generator_backend.to_tensor_async = to_tensor_async

    def test_get_mtp_draft_model_inputs_pd_dummy(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
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
                batch_tools=[],
                batch_tool_choice=[],
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
                is_dummy_batch=True
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
                input_lengths=torch.tensor([4]).to(self.device)
            )

            model_inputs_mtp, hidden_states = \
                self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(model_inputs, metadata, 0, None)
            self.assertEqual(hidden_states.shape, (2, 768), 
                           f"Expected hidden_states shape (2, 768), but got {hidden_states.shape}")
            expected = np.array([0, 0])
            self.assertTrue(np.array_equal(model_inputs_mtp.input_ids, expected), 
                          f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}")

    def test_get_mtp_draft_model_inputs_decode(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
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
                batch_tools=[],
                batch_tool_choice=[],
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64), np.array([9223372036854773], dtype=np.int64)],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([]), np.array([])],
                is_dummy_batch=False
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
                input_lengths=torch.tensor([2, 2]).to(self.device)
            )

            model_inputs_mtp, hidden_states = \
                self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(model_inputs, metadata,
                                                                           np.array([1, 1], dtype=np.int32), None)
            expected = np.array([3, 0, 3, 0])
            self.assertTrue(np.array_equal(model_inputs_mtp.input_ids, expected),
                          f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}")

    def test_get_mtp_draft_model_inputs_first_decode(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
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
                batch_tools=[],
                batch_tool_choice=[],
                batch_ignore_eos=np.array([]),
                batch_skip_special_tokens=np.array([]),
                batch_include_stop=np.array([]),
                trace_ids=None,
                batch_sequence_ids=[np.array([9223372036854775], dtype=np.int64), np.array([9223372036854773], dtype=np.int64)],
                batch_best_of=np.array([1, 1]),
                batch_logprobs=np.array([]),
                batch_seeds=np.array([]),
                batch_n=np.array([1, 1]),
                batch_use_beam_search=np.array([False]),
                reserved_sequence_ids=[np.array([]), np.array([])],
                is_dummy_batch=False
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
                input_lengths=torch.tensor([2, 2]).to(self.device)
            )

            model_inputs_mtp, hidden_states = \
                self.mtp_plugin.decoding_policy.get_mtp_draft_model_inputs(model_inputs, metadata,
                                                                           np.array([2, 3], dtype=np.int32), None)
            expected = np.array([3, 0, 3, 0])
            self.assertTrue(np.array_equal(model_inputs_mtp.input_ids, expected),
                          f"Expected input_ids {expected}, but got {model_inputs_mtp.input_ids}")

    def test_get_mtp_draft_model_inputs_with_sp(self):
        mock_hidden_states = torch.randn(2, 768)
        with patch.object(
            self.mtp_plugin.decoding_policy,
            'get_input_hidden_states'
        ) as mock_method:
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
                batch_tools=[],
                batch_tool_choice=[],
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
                is_dummy_batch=False
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
                input_lengths=torch.tensor([2, 2]).to(self.device)
            )
            self.mtp_plugin.infer_context.spcp_parallel_info.scp_size = 2
            self.mtp_plugin.decoding_policy.max_block_size = 128
            self.mtp_plugin.decoding_policy.sp_token_and_slot_calc_by_context_length(1, np.array([0, 0]), np.array([[0]]), 2)

            self.mtp_plugin.infer_context.spcp_parallel_info.scp_size = 1

    def test_all_token_ids_padding(self):
        sampling_metadata = MagicMock()
        sampling_metadata.all_token_ids = torch.tensor(np.full((1, 10), 3, dtype=np.int64), device=self.device)
        draft_tokens = np.full((1, 1), 5, dtype=np.int64)
        next_guess_logits_num_per_batch = np.array([2], dtype=np.int32)
        batch_size = 1
        input_ids_pad = self.mtp_plugin.decoding_policy.all_token_ids_padding(sampling_metadata, 
                                                                              next_guess_logits_num_per_batch,
                                                                              batch_size, draft_tokens)
        expected = torch.tensor([
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -1, -1],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, -1]
        ], device=self.device)

        self.assertTrue(
            torch.equal(input_ids_pad, expected),
            f"Output does not match expected.\nOutput:\n{input_ids_pad.cpu()}\nExpected:\n{expected.cpu()}"
        )

if __name__ == "__main__":
    unittest.main()