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
from types import SimpleNamespace

import unittest
from unittest.mock import patch, MagicMock
import torch

import numpy as np

from mindie_llm.text_generator.generator import Generator
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.input_metadata import InputMetadata
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from tests.pythontest.npu import FakeMemPool, FakeModelRunner, FakeParallelInfo


PLUGIN_PARAMS = '{\"plugin_type\": \"prefix_cache\"}'
CP = 2

mock_mempool_module = SimpleNamespace(MemPool=FakeMemPool)
sys.modules['mindie_llm.text_generator.mempool'] = mock_mempool_module


class TestPrefixCahcePlugin(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.tensor_names = {}

    @classmethod
    def setUpClass(cls):
        sys.modules['_libatb_torch'] = MagicMock()
        mock_cpu_handler = SimpleNamespace()
        mock_cpu_handler._PostProcessingManager = MagicMock()
        sys.modules['_cpu_logits_handler'] = mock_cpu_handler

    @classmethod
    def tearDownClass(cls):
        del sys.modules['_libatb_torch']
        del sys.modules['_cpu_logits_handler']

    def setUp(self):
        self.model_config = {
            'backend_bin_path': '/usr/local/Ascend/mindie/2.0.RC1/mindie-llm/bin/',
            'backend_modelInstance_id': '0', 'backend_type': 'atb', 'block_size': '128',
            'cpu_mem': '0', 'deploy_type': 'INTER_PROCESS', 'dp': '1', 'executor_type': 'LLM_EXECUTOR_PYTHON',
            'globalRankIds': '', 'globalWorldSize': '0', 'interNodeKmcKsfMaster': 'tools/pmt/master/ksfa',
            'interNodeKmcKsfStandby': 'tools/pmt/standby/ksfb', 'interNodeTLSEnabled': '1',
            'interNodeTlsCaFiles': 'ca.pem,', 'interNodeTlsCaPath': 'security/grpc/ca/',
            'interNodeTlsCert': 'security/grpc/certs/server.pem', 'interNodeTlsCrlFiles': 'server_crl.pem,',
            'interNodeTlsCrlPath': 'security/grpc/certs/', 'interNodeTlsPk': 'security/grpc/keys/server.key.pem',
            'interNodeTlsPkPwd': 'security/grpc/pass/mindie_server_key_pwd.txt', 'isMaster': '0', 'localIP': '',
            'local_rank': '0', 'masterIP': '', 'max_input_len': '2048',
            'max_iter_times': '512', 'max_prefill_tokens': '8192', 'max_seq_len': '2560',
            'model_id': '/home/data/llama3', 'model_instance_number': '1',
            'model_instance_type': 'Standard', 'model_name': 'llama3', 'moe_tp': '1',
            'multiNodesInferEnabled': '0', 'multiNodesInferPort': '1120', 'npu_device_id': '0',
            'npu_device_ids': '0,1,2,3', 'npu_mem': '-1', 'rank': '0', 'slaveIPs': '',
            'tp': '4', 'trust_remote_code': '0', 'world_size': '4',
            'num_speculative_tokens': '0', 'max_batch_size': '5', 'max_prefill_batch_size': '5',
            'distributed_enable': 'false', 'vocab_size': 100000, 'enable_warmup_with_sampling': 'false',
            'cp': '2', 'sp': '1', 'moe_ep': '1'
        }

        plugin_dict = {'plugin_params': PLUGIN_PARAMS}
        self.model_config.update(plugin_dict)

        if hasattr(self, 'model_config') and self.model_config:
            # 创建一个模拟的 generator_backend
            self.generator_backend = MagicMock()
            self.generator_backend.mapping = MagicMock()
            self.generator_backend.mapping.attn_dp = MagicMock()
            self.generator_backend.mapping.attn_dp.rank = np.array([1])
            self.generator_backend.rank = int(self.model_config['rank'])
            self.generator_backend.local_rank = int(self.model_config['local_rank'])

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config['dp']),
            tp=int(self.model_config['tp']),
            sp=int(self.model_config['sp']),
            cp=int(self.model_config['cp'])
        )
        self.fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)
        return super().setUp()

    @patch('torch.npu.synchronize', return_value=None)
    @patch('atb_llm.runner.model_runner.ModelRunner')
    @patch.object(Generator, 'warm_up')
    @patch.object(GeneratorTorch, '_get_obfuscation_func')
    @patch.object(GeneratorTorch, 'forward')
    def test_generate_token_greedy(
        self,
        mock_forward,
        mock_obfuscation_func,
        mock_warm_up,
        mock_model_runner,
        mock_npu_sync,
    ):

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
        
        mock_model_runner.return_value = self.fake_model_runner
        mock_forward.side_effect = side_effect_forward
        mock_obfuscation_func.return_value = None
        mock_warm_up.return_value = 10

        self.model_config['kv_pool_backend'] = "mooncake"
        self.model_config['kv_pool_config_path'] = "a.json"

        generator = Generator(self.model_config)

        prefix_cache_plugin = generator.plugin_manager
        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0., 0., 0, 1., 1., False, 0)], dtype=sample_dtype)
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
        prefix_cache_plugin.generator_backend.backend_type = 'atb'
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

    def test_async_put_prefix_kvcache_to_mempool_function(self):
        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            if model_inputs.is_prefill:
                token_num = model_inputs.prefill_head_indices.shape[0]
            else:
                token_num = model_inputs.input_ids.shape[0]
            logits = torch.zeros(token_num, 10)
            return logits

        with patch.object(GeneratorTorch, 'forward') as mock_forward, \
             patch('atb_llm.utils.dist.initialize_distributed') as mock_initialize_distributed, \
             patch('atb_llm.runner.model_runner.ModelRunner', return_value=self.fake_model_runner) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.NPUSocInfo.support_nz', return_value=True) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.KVCacheSettings') as mock_kvcache_settings_class, \
             patch('torch.npu.synchronize', return_value=None) as _, \
             patch('torch.npu.set_device', return_value=None) as _, \
             patch('torch.npu.set_stream', return_value=None) as _, \
             patch('torch.npu.Stream', return_value=MagicMock()) as mock_stream, \
             patch('mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch._get_obfuscation_func', \
                    return_value=None) as _:

            mock_initialize_distributed.side_effect = side_effect_initialize_distributed
            mock_forward.side_effect = side_effect_forward
            mock_kvcache_settings = MagicMock(dtype=None)
            mock_kvcache_settings_class.return_value = mock_kvcache_settings
            self.model_config['kv_pool_backend'] = "mooncake"
            self.model_config['kv_pool_config_path'] = "a.json"
            self.model_config['kv_pool_async_write'] = "true"
            generator = Generator(self.model_config)

            prefix_cache_plugin = generator.plugin_manager

            # 创建测试输入元数据
            mock_metadata = MagicMock()
            mock_metadata.is_prefill = True
            mock_metadata.computed_blocks = np.array([[1, 0]])
            mock_metadata.remote_computed_blocks = np.array([[1, 0]])

            # 创建测试 cache_ids
            cache_ids = [0]

            # 测试 async_put_prefix_kvcache_to_mempool
            prefix_cache_plugin.prefix_cache.async_put_prefix_kvcache_to_mempool(mock_metadata, cache_ids)

            # 测试 put_prefix_kvcache_put_task_queue
            mock_metadata.batch_dp_rank_ids = [0]
            mock_metadata.remote_computed_blocks = np.array([1])
            prefix_cache_plugin.prefix_cache.put_prefix_kvcache_put_task_queue(mock_metadata, cache_ids)

            # 参数构建
            mock_metadata.batch_dp_rank_ids = np.array([0, 1])
            mock_metadata.remote_computed_blocks = np.array([[1, 1]])
            mock_metadata.batch_size = 1
            mock_metadata.batch_seq_len = [128]
            mock_metadata.max_block_size = 128
            mock_metadata.batch_block_tables = np.array([[[0, 1]]])
            prefix_cache_plugin.infer_context = MagicMock()
            prefix_cache_plugin.infer_context.get_all_input_ids.return_value = np.array([[1, 2, 3, 4, 5] * 256])
            prefix_cache_plugin.infer_context.get_seq_lens.return_value = [128]
            mock_kv_block_num = [1] * 100
            prefix_cache_plugin.generator_backend.cache_pool.npu_cache = [(mock_kv_block_num, mock_kv_block_num)] * 100

            # 测试 put_prefix_kvcache_to_mempool
            prefix_cache_plugin.put_prefix_kvcache_to_mempool(mock_metadata, cache_ids)

    def test_get_prefix_kvcache_from_mempool(self):
        from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType
        from mindie_llm.text_generator.plugins.prefix_cache.prefix_cache_plugin import PrefixCachePlugin

        plugin = PrefixCachePlugin.__new__(PrefixCachePlugin)
        plugin.mempool_type = MemPoolType.SYNC_WRITE
        plugin.scp_size = 2
        plugin.scp_rank = 0
        plugin.num_put_layers = 61
        plugin.generator_backend = self.generator_backend
        plugin.model_name = "llama"

        metadata = SimpleNamespace(
            is_prefill=True,
            computed_blocks=None,
            remote_computed_blocks=np.array([[0]]),
            batch_size=1,
            batch_dp_rank_ids=[0],
            batch_seq_len=[10],
            input_ids=np.array([1, 2, 3, 4, 5]),
            max_block_size=128,
            batch_block_tables=np.array([[0, 1]]).reshape(1, 1, -1),
            block_tables=np.array([[0, 1]]).reshape(1, 1, -1)
        )

        plugin.get_prefix_kvcache_from_mempool(metadata)

    def test_hash_and_prefix_key(self):
        from mindie_llm.text_generator.plugins.prefix_cache.prefix_cache_plugin import (
            cpp_style_hash, hash_combine, PrefixCachePlugin
        )

        self.assertEqual(cpp_style_hash(123), 123)
        self.assertNotEqual(cpp_style_hash("abc"), 0)

        seed = hash_combine(0, 10)
        self.assertNotEqual(seed, 0)

        plugin = PrefixCachePlugin.__new__(PrefixCachePlugin)
        plugin.scp_size = 1
        plugin.tp_rank = 0
        plugin.tp_size = 1
        plugin.model_name = "llama"

        hash_val = plugin.hash_block(0, [1, 2, 3, 4])
        self.assertNotEqual(hash_val, 0)

        key = plugin.get_prefix_keys(hash_val)
        self.assertIn("llama", key)

    def test_enable_prefixcache_flags(self):
        from mindie_llm.text_generator.plugins.prefix_cache.prefix_cache_plugin import PrefixCachePlugin

        meta = SimpleNamespace(
            is_prefill=True,
            computed_blocks=np.array([1]),
            remote_computed_blocks=np.array([1])
        )

        self.assertTrue(PrefixCachePlugin.enable_local_prefixcache(meta))
        self.assertTrue(PrefixCachePlugin.enable_remmote_prefixcache(meta))

        meta.computed_blocks = None
        self.assertFalse(PrefixCachePlugin.enable_local_prefixcache(meta))

    def test_attention_mask(self):
        from mindie_llm.text_generator.plugins.prefix_cache.prefix_cache_plugin import AttentionMask as PluginAttentionMask

        am = PluginAttentionMask(None)

        am.atten_mask_cache = am.get_attn_mask(4)
        self.assertEqual(am.atten_mask_cache.shape, (4, 4))

        am._seq_len_cache = 4
        am.update_attn_cache(5)
        self.assertEqual(am.atten_mask_cache.shape, (5, 5))

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