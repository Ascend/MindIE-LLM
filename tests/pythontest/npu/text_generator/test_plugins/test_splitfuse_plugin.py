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


current_file_path = Path(__file__).resolve()
target_dir = current_file_path.parent.parent.parent.parent

MODEL_PATH = target_dir.joinpath("test_weights/llama3")
PLUGIN_PARAMS = '{\"plugin_type\": \"splitfuse\"}'
SPECULATION_GAMMA = 30


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
        self.soc_info = MagicMock()
        self.soc_info.is_300i.return_value = False
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
            env_rank: ENV.rank,
            env_world_size: ENV.world_size,
            env_local_rank: ENV.local_rank,
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
        self.max_prefill_batch_size = input_dict.get('max_prefil_batch_size', 200)
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
            'enable_warmup_with_sampling': False
        }

        if isinstance(self.model_config, ModelConfig):
            self.model_config = vars(self.model_config)
        if self.eos_token_id is not None:
            self.model_config['eos_token_id'] = self.eos_token_id

    def test_generate_token_gready(self):
        
        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            if model_inputs.is_prefill:
                token_num = model_inputs.prefill_head_indices.shape[0]
            else:
                token_num = len(model_inputs.input_ids)
            logits = torch.zeros(token_num, 10) # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            return logits
        
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
            mock_kvcache_settings = MagicMock(dtype=None)
            mock_kvcache_settings_class.return_value = mock_kvcache_settings
            generator = Generator(self.model_config)

            splitfuse_plugin = generator.plugin
            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
            
            gen_len = 2
            req = Request.request_from_token(input1, sampling_params=greedy_param,
                                             generation_params=GenerationParams(max_new_tokens=gen_len))
            req.split_start_position = 0
            req.split_end_position = len(input1)
            meta_data = InputMetadata.from_requests([req], block_tables, np.array([1]))
            meta_data.batch_block_tables = block_tables
            meta_data.batch_is_prefill = np.array([True])

            generation_output = splitfuse_plugin.generate_token(meta_data)

            # 自回归推理
            meta_data.is_prefill = False
            tokens_list = []
            while generation_output.finish_reason[0] == 0:
                generation_output = splitfuse_plugin.generate_token(meta_data)
                tokens_list.extend(generation_output.token_ids[0])

            # 验证greedy是否每轮都选择logits最大的token
            is_greedy = True
            for token in tokens_list:
                if token != 8:
                    is_greedy = False
                    break
            self.assertTrue(is_greedy)

if __name__ == "__main__":
    unittest.main()