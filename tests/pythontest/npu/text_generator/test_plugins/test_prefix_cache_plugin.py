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
from typing import List
from types import SimpleNamespace

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
PLUGIN_PARAMS = '{\"plugin_type\": \"prefix_cache\"}'
CP = 2


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
        self.mapping = Mapping(world_size=2, rank=ENV.local_rank, tp=1, cp=CP, moe_ep=2)
        self.process_group = FakeGroup(rank=ENV.local_rank, size=2)
        self.device = torch.device('cpu')

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


class FakeMemPool:
    def __init__(self, backend, config_path, **kwargs):
        pass
    
    @classmethod
    def create_pool(cls, backend: str, config_path: str, role: str = "scheduler", **kwargs):
        return cls(backend, config_path, **kwargs)

    def put(self, keys, tensors, **kwargs) -> List[bool]:
        return [True] * len(keys)
    
    def get(self, keys, tensors, **kwargs) -> List[bool]:
        return [True] * len(keys)


mock_mempool_module = SimpleNamespace(MemPool=FakeMemPool)
sys.modules['mindie_llm.text_generator.mempool'] = mock_mempool_module


class TestPlugin(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.tensor_names = {}

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
        self.world_size = 2
        self.eos_token_id = '[12]'

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
        return super().setUp()

    def test_generate_token_gready(self):
        
        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            if model_inputs.is_prefill:
                token_num = model_inputs.prefill_head_indices.shape[0]
            else:
                token_num = model_inputs.input_ids.shape[0]
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
            self.model_config['kv_pool_backend'] = "mooncake"
            self.model_config['kv_pool_config_path'] = "a.json"
            generator = Generator(self.model_config)

            prefix_cache_plugin = generator.plugin
            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = np.array([1] * 200)
            block_tables = np.array([[0, 1]]).reshape(1, 1, -1)
            mock_kv_block_num = [1] * 100
            prefix_cache_plugin.generator_backend.cache_pool.npu_cache = [(mock_kv_block_num, mock_kv_block_num)] * 100
            gen_len = 2
            req = Request.request_from_token(input1, 
                                             sampling_params=greedy_param, 
                                             generation_params=GenerationParams(max_new_tokens=gen_len))
            meta_data = InputMetadata.from_requests([req], block_tables, True)
            meta_data.block_tables = block_tables
            meta_data.batch_block_tables = block_tables
            meta_data.sp_rank_id = np.array([1])
            meta_data.sp_tokens = np.array([128, 72]).reshape(1, -1)
            # 无复用，使用fa算子做prefill
            generation_output = prefix_cache_plugin.generate_token(meta_data)
            
            # 有复用，使用qlen > 1 的 pa算子做prefill
            remote_computed_blocks = np.ones(CP, dtype=np.int64).reshape(1, -1)
            remote_computed_blocks[0, 1] = 0
            
            # 都在本地命中
            meta_data.computed_blocks = np.zeros(CP, dtype=np.int64).reshape(1, -1)
            meta_data.computed_blocks[0, 1] = 1
            meta_data.remote_computed_blocks = remote_computed_blocks
            prefix_cache_plugin.generator_backend.backend_type = 'ms'
            meta_data.batch_block_tables = block_tables
            generation_output = prefix_cache_plugin.generate_token(meta_data)

            # 都在远端存储池命中
            meta_data.computed_blocks = np.zeros(CP, dtype=np.int64).reshape(1, -1)
            meta_data.remote_computed_blocks = remote_computed_blocks
            prefix_cache_plugin.generator_backend.backend_type = 'atb'
            meta_data.batch_block_tables = block_tables
            generation_output = prefix_cache_plugin.generate_token(meta_data)

            # 自回归推理
            meta_data.is_prefill = False
            tokens_list = []
            while generation_output.finish_reason[0] == 0:
                meta_data.batch_block_tables = block_tables
                generation_output = prefix_cache_plugin.generate_token(meta_data)
                tokens_list.extend(generation_output.token_ids[0])

            # 验证greedy是否每轮都选择logits最大的token
            check_greedy = 1
            for token in tokens_list:
                if token != 8:
                    check_greedy = 0
                    break
            self.assertEqual(check_greedy, 1)
    
    def _get_tensor_slid_effect(self, tensor_name):
        tensor_shape = self.tensor_names.get(tensor_name).get("shape")
        tensor_dtype = self.tensor_names.get(tensor_name).get("dtype")
        return torch.rand(tensor_shape, dtype=tensor_dtype)

    def _get_slice_slid_effect(self, tensor_name):
        tensor = self._get_tensor_slid_effect(tensor_name)
        tensor.get_shape = tensor.size
        return tensor


if __name__ == "__main__":
    unittest.main()