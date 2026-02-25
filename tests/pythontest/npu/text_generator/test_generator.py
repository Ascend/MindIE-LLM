#!/usr/bin/env python
# coding=utf-8
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
import os
import unittest
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from ddt import ddt, data, unpack

from atb_llm.utils.dist import FakeGroup
from mindie_llm.utils.env import ENV
from mindie_llm.utils.status import MindieLlmStatusCode
from mindie_llm.text_generator.generator import Generator, PDInterface, PDModelConfig
from mindie_llm.text_generator.samplers.sampler import Sampler
from mindie_llm.text_generator.utils.generation_output import GenerationOutput
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.input_metadata import InputMetadata, SAMPLING_DTYPE
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from mindie_llm.connector.common.model_execute_data_pb2 import LoraOperationStatus
from tests.pythontest.npu.text_generator.test_plugins.test_plugin_manager import FakeModelRunner
from mindie_llm.text_generator.utils.tg_infer_context_store import TGInferContextStore
from mindie_llm.text_generator.utils.kvcache_settings import KVCacheSettings
@ddt
class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules['_libatb_torch'] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        del sys.modules['_libatb_torch']

    def setUp(self):
        self.model_config = {
            'backend_bin_path': '/usr/local/Ascend/mindie/2.0.RC1/mindie-llm/bin/',
            'backend_log_file': '/usr/local/Ascend/mindie/2.0.RC1/mindie-service/logs/mindie-server.log',
            'backend_modelInstance_id': '0', 'backend_type': 'atb', 'block_size': '128',
            'cpu_mem': '5', 'deploy_type': 'INTER_PROCESS', 'dp': '8', 'executor_type': 'LLM_EXECUTOR_PYTHON',
            'globalRankIds': '', 'globalWorldSize': '0', 'interNodeTLSEnabled': '1',
            'interNodeTlsCaFiles': 'ca.pem,', 'interNodeTlsCaPath': 'security/grpc/ca/',
            'interNodeTlsCert': 'security/grpc/certs/server.pem', 'interNodeTlsCrlFiles': 'server_crl.pem,',
            'interNodeTlsCrlPath': 'security/grpc/certs/', 'interNodeTlsPk': 'security/grpc/keys/server.key.pem',
            'isMaster': '0', 'localIP': '',
            'local_rank': '0', 'log_error': '1', 'log_file_num': '20', 'log_file_size': '20', 'log_info': '1',
            'log_verbose': '0', 'log_warning': '1', 'masterIP': '', 'max_input_len': '2048',
            'max_iter_times': '512', 'max_prefill_tokens': '8192', 'max_seq_len': '2560',
            'model_id': '/home/data/DeepSeek-V2-Lite-Chat/', 'model_instance_number': '1',
            'model_instance_type': 'Standard', 'model_name': 'deepseekv2', 'moe_tp': '8',
            'multiNodesInferEnabled': '0', 'multiNodesInferPort': '1120', 'npu_device_id': '0',
            'npu_device_ids': '0,1,2,3,4,5,6,7', 'npu_mem': '-1', 'rank': '0', 'slaveIPs': '',
            'speculation_gamma': '0', 'tp': '1', 'trust_remote_code': '0', 'world_size': '8',
            'num_speculative_tokens': '1', 'max_batch_size': '200', 'max_prefill_batch_size': '200', 
            'distributed_enable': 'false', 'vocab_size': 100000,'enable_warmup_with_sampling': 'false'
        }

    @patch('mindie_llm.text_generator.generator.calc_npu_mem')
    @patch('mindie_llm.text_generator.generator.calc_block_mem')
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.adapter.generator_backend.Sampler",
           return_value=MagicMock(spec=Sampler))
    @patch("mindie_llm.text_generator.generator.KVCacheSettings",
           return_value=MagicMock(spec=KVCacheSettings))
    @patch("mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch")
    @patch("mindie_llm.text_generator.generator.Generator._Generator__get_warm_up_reqs")
    @patch("mindie_llm.text_generator.generator.Generator._Generator__execute_warm_up")
    @patch("mindie_llm.text_generator.generator.TGInferContextStore",
           return_value=MagicMock(TGInferContextStore))
    def test_init_npu_mem_dynamic(
        self, mock_infer_context, mock_generate_inputs_warm_up_backend,
        mock_get_warm_up_reqs, mock_generator_torch, mock_kvcache_settings,
        mock_sampler, mock_model_runner, mock_calc_block_mem, mock_calc_npu_mem
    ):
        mock_infer_context_ins = mock_infer_context.return_value
        mock_infer_context_ins.context_params = MagicMock()
        mock_infer_context_ins.context_params.async_infer = False

        mock_calc_npu_mem.return_value = 10
        mock_calc_block_mem.return_value = 1024
        mock_ins = mock_model_runner.return_value
        mock_ins.kv_cache_dtype = torch.float16
        mock_ins.k_head_size = 128
        mock_ins.v_head_size = 128

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.model_wrapper.mapping = MagicMock()
        mock_generator_torch_ins.model_wrapper.config.eos_token_id = 0
        mock_generator_torch_ins.obfuscation_func = None
        mock_generator_torch_ins.model_wrapper.is_multimodal = False

        mock_get_warm_up_reqs.return_value = \
            ([Request.from_warmup(128, 128)], np.array([[0, 0]]), [2])
        mock_kvcache_settings.return_value.slots = None
        mock_kvcache_settings.return_value.dtype = None
        generator = Generator(self.model_config)
        self.assertIsNotNone(generator)

    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.adapter.generator_backend.Sampler",
           return_value=MagicMock(spec=Sampler))
    @patch("mindie_llm.text_generator.generator.KVCacheSettings",
           return_value=MagicMock(spec=KVCacheSettings))
    @patch("mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch")
    @patch("mindie_llm.text_generator.generator.Generator._Generator__get_warm_up_reqs")
    @patch("mindie_llm.text_generator.generator.Generator._Generator__execute_warm_up")
    @patch("mindie_llm.text_generator.generator.TGInferContextStore",
           return_value=MagicMock(TGInferContextStore))
    def test_init_npu_mem_static(
        self, mock_infer_context, mock_generate_inputs_warm_up_backend,
        mock_get_warm_up_reqs, mock_generator_torch, mock_kvcache_settings,
        mock_sampler, mock_model_runner
    ):
        mock_infer_context_ins = mock_infer_context.return_value
        mock_infer_context_ins.context_params = MagicMock()
        mock_infer_context_ins.context_params.async_infer = False

        mock_ins = mock_model_runner.return_value
        mock_ins.kv_cache_dtype = torch.float16
        mock_ins.k_head_size = 128
        mock_ins.v_head_size = 128

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.model_wrapper.mapping = MagicMock()
        mock_generator_torch_ins.model_wrapper.config.eos_token_id = 0
        mock_generator_torch_ins.obfuscation_func = None

        mock_kvcache_settings.return_value.num_npu_blocks = 10
        mock_get_warm_up_reqs.return_value = \
            ([Request.from_warmup(128, 128)], np.array([[0, 0]]), [2])
        self.model_config["npu_mem"] = 10
        mock_kvcache_settings.return_value.slots = None
        mock_kvcache_settings.return_value.dtype = None
        generator = Generator(self.model_config)
        self.assertIsNotNone(generator)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_get_warm_up_reqs(self, mock_init):
        generator = Generator(self.model_config)
        generator.rank = 0
        generator.num_speculative_tokens = 1
        generator.max_prefill_tokens = 256
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.mapping.attn_dp.group_size = 8
        generator.model_wrapper.mapping.attn_inner_sp.group_size = 1
        generator.block_size = 128
        generator.vocab_size = 100
        generator.warmup_topk_size = 100
        generator.enable_warmup_with_sampling = False
        generator.dp_size = 8
        generator.sp_size = 1
        generator.cp_size = 1
        generator.scp_size = 1
        generator.distributed_enable = False
        generator.separate_deployment_worker = None
        generator.lwd_multi_nodes_enable = False

        warm_up_params = (256, 256, 128, 128)
        prefill_reqs, block_tables, prefill_blocks = \
            generator._Generator__get_warm_up_reqs(1024, warm_up_params)
        self.assertEqual(len(prefill_reqs), 8)
        self.assertEqual(prefill_reqs[0].input_ids.shape[0], 256)
        self.assertEqual(prefill_reqs[0].max_new_tokens, 0)
        self.assertTrue((block_tables == np.array([[0, 0]])).all())
        self.assertListEqual(prefill_blocks, [2] * 8)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_get_warm_up_reqs_raise(self, mock_init):
        generator = Generator(self.model_config)
        generator.rank = 0
        generator.num_speculative_tokens = 1
        generator.max_prefill_tokens = 256
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.mapping.attn_dp.group_size = 8
        generator.block_size = 128
        generator.model_wrapper.mapping.attn_inner_sp.group_size = 0
        generator.dp_size = 8
        generator.sp_size = 1
        generator.cp_size = 1
        generator.scp_size = 1
        generator.distributed_enable = False
        generator.separate_deployment_worker = None
        generator.lwd_multi_nodes_enable = False

        with self.assertRaises(RuntimeError) as context:
            warm_up_params = (256, 256, 128, 128)
            _, _, _ = generator._Generator__get_warm_up_reqs(1, warm_up_params)
        self.assertIn("Warmup failed.", str(context.exception))

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_generate_token_plugin_none_raise_error(self, _):
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        input_metadata = MagicMock(spec=InputMetadata)

        with self.assertRaises(AttributeError):
            _ = generator.generate_token(input_metadata)

    def test_generate_token(self):

        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10) # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
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
            generator.vocab_size = 100
            generator.warmup_topk_size = 100

            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

            gen_len = 2
            req = Request.request_from_token(input1, 
                                             sampling_params=greedy_param, 
                                             generation_params=GenerationParams(max_new_tokens=gen_len))

            meta_data = InputMetadata.from_requests([req], block_tables, True)
            meta_data.batch_block_tables = block_tables

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

    @patch('mindie_llm.utils.decorators.time_decorator')
    @patch('mindie_llm.utils.env.ENV.benchmark_enable_async')
    @patch('mindie_llm.utils.env.ENV.benchmark_enable')
    @patch('mindie_llm.text_generator.utils.npu_mem_tool.calc_npu_mem')
    @patch('mindie_llm.text_generator.utils.npu_mem_tool.calc_block_mem')
    @patch("mindie_llm.text_generator.adapter.generator_backend.Sampler",
           return_value=MagicMock(spec=Sampler))
    @patch("mindie_llm.text_generator.generator.KVCacheSettings",
           return_value=MagicMock(spec=KVCacheSettings))
    @patch("mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch")
    @patch("mindie_llm.text_generator.generator.Generator._Generator__get_warm_up_reqs")
    @patch("mindie_llm.text_generator.generator.Generator._Generator__execute_warm_up")
    @patch("mindie_llm.text_generator.generator.TGInferContextStore",
           return_value=MagicMock(TGInferContextStore))
    def test_async_decode(self, mock_infer_context, mock_generate_inputs_warm_up_backend, mock_get_warm_up_reqs,
                          mock_generator_torch, mock_kvcache_settings, mock_sampler,
                          mock_calc_block_mem, mock_calc_npu_mem, mock_benchmark_enable, mock_benchmark_enable_async,
                          mock_timer):
        with patch('atb_llm.runner.model_runner.ModelRunner', return_value=FakeModelRunner()) as mock_model_runner:
            mock_infer_context_ins = mock_infer_context.return_value
            mock_infer_context_ins.context_params = MagicMock()
            mock_infer_context_ins.context_params.async_infer = False

            mock_calc_npu_mem.return_value = 10
            mock_calc_block_mem.return_value = 1024
            mock_ins = mock_model_runner.return_value
            mock_ins.kv_cache_dtype = torch.float16
            mock_ins.k_head_size = 128
            mock_ins.v_head_size = 128

            mock_generator_torch_ins = mock_generator_torch.return_value
            mock_generator_torch_ins.model_info.k_head_size = 128
            mock_generator_torch_ins.model_info.v_head_size = 128
            mock_generator_torch_ins.model_wrapper.model_runner = mock_ins
            mock_generator_torch_ins.model_wrapper.mapping = MagicMock()
            mock_generator_torch_ins.model_wrapper.config.eos_token_id = 0
            mock_generator_torch_ins.obfuscation_func = None
            mock_generator_torch_ins.model_wrapper.is_multimodal = False

            mock_get_warm_up_reqs.return_value = \
                ([Request.from_warmup(128, 128)], np.array([[0, 0]]), [2])
            mock_kvcache_settings.return_value.dtype = None
            mock_kvcache_settings.return_value.dtype_str = "float16"
            mock_kvcache_settings.return_value.k_head_size = 128
            mock_kvcache_settings.return_value.v_head_size = 128
            model_config = {**self.model_config}
            model_config['role'] = 'decoder'
            generator = Generator(model_config)
            self.assertEqual(generator.separate_deployment_worker.role, 'decoder')

            generator.input_metadata_queue = queue.Queue()
            generator.input_metadata_queue.put(MagicMock())
            generator.infer_context.get_batch_context_handles = MagicMock(return_value=None)
            mock_sampling_metadata = MagicMock()
            mock_sampling_metadata.is_dummy_batch = False
            mock_sampling_metadata.do_sample_array = np.array([True])
            generator.infer_context.compose_model_inputs = MagicMock(return_value=(None, mock_sampling_metadata, None))
            generator.generator_backend.configure_sampler = MagicMock(return_value=None)
            generator.async_inference = True
            mock_generation_output = MagicMock()
            mock_generation_output.trace_ids = np.array([0])
            generator.plugin.generate_token_async = MagicMock(return_value=mock_generation_output)

            mock_timer.log_time = MagicMock(return_value=None)
            mock_timer.log_time_async = MagicMock(return_value=None)
            input_metadata = MagicMock()
            input_metadata.is_prefill = False
            input_metadata.batch_seq_len = np.array([128,128])
            generation_output = generator.generate_token(input_metadata)
            self.assertEqual(generation_output, mock_generation_output)

            generator.plugin = None
            with self.assertRaises((NotImplementedError, Exception)):
                generator.generate_token(input_metadata=MagicMock())

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_generate(self, _):
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        input1 = [5159, 636, 374, 31346, 323, 358]
        greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=SAMPLING_DTYPE)
        gen_len = 2
        req1 = Request.request_from_token(input1, 
                                          sampling_params=greedy_param, 
                                          generation_params=GenerationParams(max_new_tokens=gen_len))
        req1.sequences[0].block_tables = np.array([0])
        req2 = Request.request_from_token(input1, 
                                          sampling_params=greedy_param, 
                                          generation_params=GenerationParams(max_new_tokens=gen_len+128))
        req2.sequences[0].block_tables = np.array([1, 2])
        requests = [req1, req2]

        def mock_generate_token(_):
            return GenerationOutput(sequence_ids=np.array([0, 1]),
                                    parent_sequence_ids=np.array([0, 1]),
                                    group_indices=[(0, 1), (1, 2)])

        generator.generate_token = mock_generate_token
        generation_output = generator.prefill(requests)
        self.assertIsInstance(generation_output, GenerationOutput)
        generation_output = generator.decode(requests)
        self.assertIsInstance(generation_output, GenerationOutput)
        req1.block_tables = np.array([0, -1])
        req2.block_tables = np.array([1, 2])
        generation_output = generator.generate_mix(requests, is_prefill_batch=np.array([False, True]))
        self.assertIsInstance(generation_output, GenerationOutput)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_load_lora_not_active(self, _):
        """测试lora特性未使能时的返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = None
        ret = generator.load_lora("fake_id", "fake_path")
        self.assertEqual(ret, LoraOperationStatus.UNSUPPORT_CMD)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_load_lora_success(self, _):
        """测试lora特性使能时的加载成功返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = MagicMock()
        generator.model_wrapper.adapter_manager.load_adapter = MagicMock()
        ret = generator.load_lora("fake_id", "fake_path")
        self.assertEqual(ret, LoraOperationStatus.LORA_CMD_SUCCESS)

    @data(("LORA MEMORY ERROR", LoraOperationStatus.SLOTS_FULL),
          ("DUPLICATED LORA ID", LoraOperationStatus.DUPLICATED_LORA_ID),
          ("INVALID LORA ID", LoraOperationStatus.INVALID_LORA_ID),
          ("INVALID LORA RANK", LoraOperationStatus.INVALID_LORA_RANK))
    @unpack
    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_load_lora_fail(self, exception, expected_ret, _):
        """测试lora特性使能时的加载失败返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = MagicMock()
        generator.model_wrapper.adapter_manager.load_adapter = MagicMock()
        generator.model_wrapper.adapter_manager.load_adapter.side_effect = Exception(exception)
        ret = generator.load_lora("fake_id", "fake_path")
        self.assertEqual(ret, expected_ret)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_unload_lora_not_active(self, _):
        """测试lora特性未使能时的返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = None
        ret = generator.unload_lora("fake_id")
        self.assertEqual(ret, LoraOperationStatus.UNSUPPORT_CMD)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_unload_lora_success(self, _):
        """测试lora特性使能时的卸载成功返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = MagicMock()
        generator.model_wrapper.adapter_manager.unload_adapter = MagicMock()
        ret = generator.unload_lora("fake_id")
        self.assertEqual(ret, LoraOperationStatus.LORA_CMD_SUCCESS)

    def test_generate_token_flex(self):

        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10) # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
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
            config_dict = {
                'role': 'flex',
                'local_instance_id': 0,
                'local_device_ip': '127.0.0.1',
                'npu_device_id': 0,
                'local_physical_device_id': 0,
                'local_host_ip': '127.0.0.1',
                'remote_device_ips': '127.0.0.2'
            }
            self.model_config.update(config_dict)
            generator = Generator(self.model_config)
            generator.vocab_size = 100
            generator.warmup_topk_size = 100

            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

            gen_len = 2
            req = Request.request_from_token(input1, 
                                            sampling_params=greedy_param, 
                                            generation_params=GenerationParams(max_new_tokens=gen_len))

            meta_data = InputMetadata.from_requests([req], block_tables, True)
            meta_data.batch_block_tables = block_tables

            generation_output = generator.generate_token(meta_data)

            # 自回归推理
            meta_data.is_prefill = False
            tokens_list = []
            # flex decode
            generator.input_metadata_queue.put(meta_data)
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

    def test_init_prefill(self):

        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10) # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
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

            config_dict = {
                'role': 'prefill',
                'local_instance_id': 0,
                'local_device_ip': '127.0.0.1',
                'npu_device_id': 0,
                'local_physical_device_id': 0,
                'local_host_ip': '127.0.0.1',
                'remote_device_ips': '127.0.0.2'
            }
            self.model_config.update(config_dict)
            generator = Generator(self.model_config)
            generator.vocab_size = 100
            generator.warmup_topk_size = 100

    def test_generate_token_no_plugin(self):

        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10) # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
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
            generator.plugin = None
            generator.vocab_size = 100
            generator.warmup_topk_size = 100

            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

            gen_len = 2
            req = Request.request_from_token(input1, 
                                            sampling_params=greedy_param, 
                                            generation_params=GenerationParams(max_new_tokens=gen_len))

            meta_data = InputMetadata.from_requests([req], block_tables, True)
            meta_data.batch_block_tables = block_tables

            with self.assertRaises(NotImplementedError) as context:
                _ = generator.generate_token(meta_data)
            self.assertIn("not implemented", str(context.exception))

    def test_generate_prefill(self):

        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10) # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
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
            generator.vocab_size = 100
            generator.warmup_topk_size = 100

            sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
            greedy_param = np.array([(1.0, 0., 0., 0.7, 3., 0.92, False, 0)], dtype=sample_dtype)
            input1 = [5159, 636, 374, 31346, 323, 358]
            block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
            gen_len = 2
            req = Request.request_from_token(input1, 
                                             sampling_params=greedy_param, 
                                             generation_params=GenerationParams(max_new_tokens=gen_len))
            for _, v in req.sequences.items():
                v.block_tables = block_tables
            _ = generator.generate([req], True)

    def test_init_benchmark_file_exists(self):

        def side_effect_initialize_distributed(rank, npu_id, world_size):
            return FakeGroup(rank, world_size), torch.device("cpu")

        with patch.object(GeneratorTorch, 'forward') as _, \
             patch('atb_llm.utils.dist.initialize_distributed') as mock_initialize_distributed, \
             patch('atb_llm.runner.model_runner.ModelRunner', return_value=FakeModelRunner()) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.NPUSocInfo.support_nz', return_value=True) as _, \
             patch('mindie_llm.text_generator.utils.kvcache_settings.KVCacheSettings') as mock_kvcache_settings_class, \
             patch('torch.npu.synchronize', return_value=None) as _, \
             patch('mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch._get_obfuscation_func', \
                    return_value=None) as _:

            mock_initialize_distributed.side_effect = side_effect_initialize_distributed
            mock_kvcache_settings = MagicMock(dtype=None)
            mock_kvcache_settings_class.return_value = mock_kvcache_settings
            ENV.benchmark_filepath = "./tmp.txt"
            if not os.path.exists(ENV.benchmark_filepath):
                # 使用 'w' 模式创建文件
                with open(ENV.benchmark_filepath, 'w') as file:
                    file.write('Hello, world!')
                os.chmod(ENV.benchmark_filepath, 0o600)
            _ = Generator(self.model_config)

    def test_init_eos_token(self):
        with (
            patch("mindie_llm.text_generator.generator.calc_npu_mem") as mock_calc_npu_mem,
            patch("mindie_llm.text_generator.generator.calc_block_mem") as mock_calc_block_mem,
            patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner") as mock_model_runner,
            patch(
                "mindie_llm.text_generator.adapter.generator_backend.Sampler", return_value=MagicMock(spec=Sampler)
            ) as _,
            patch(
                "mindie_llm.text_generator.generator.KVCacheSettings", return_value=MagicMock(spec=KVCacheSettings)
            ) as mock_kvcache_settings,
            patch("mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch") as mock_generator_torch,
            patch("mindie_llm.text_generator.generator.Generator._Generator__get_warm_up_reqs") as mock_warmup_reqs,
            patch("mindie_llm.text_generator.generator.Generator._Generator__execute_warm_up") as _,
            patch(
                "mindie_llm.text_generator.generator.TGInferContextStore", return_value=MagicMock(TGInferContextStore)
            ) as mock_infer_context,
        ):
            mock_infer_context_ins = mock_infer_context.return_value
            mock_infer_context_ins.context_params = MagicMock()
            mock_infer_context_ins.context_params.async_infer = False

            mock_calc_npu_mem.return_value = 10
            mock_calc_block_mem.return_value = 1024
            mock_ins = mock_model_runner.return_value
            mock_ins.kv_cache_dtype = torch.float16
            mock_ins.k_head_size = 128
            mock_ins.v_head_size = 128

            mock_generator_torch_ins = mock_generator_torch.return_value
            mock_generator_torch_ins.model_wrapper.mapping = MagicMock()
            mock_generator_torch_ins.model_wrapper.config.eos_token_id = None
            mock_generator_torch_ins.tokenizer.eos_token_id = 0
            mock_generator_torch_ins.model_wrapper.config.pad_token_id = None
            mock_generator_torch_ins.model_wrapper.config.vocab_size = 100
            mock_generator_torch_ins.tokenizer.pad_token_id = 0
            mock_generator_torch_ins.obfuscation_func = None
            mock_generator_torch_ins.model_wrapper.is_multimodal = False

            mock_warmup_reqs.return_value = \
                ([Request.from_warmup(128, 128)], np.array([[0, 0]]), [2])
            mock_kvcache_settings.return_value.slots = None
            mock_kvcache_settings.return_value.dtype = None
            generator = Generator(self.model_config)
            self.assertIsNotNone(generator)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_non_atb_backend(self, _):
        """测试非atb后端不支持恢复命令"""
        generator = Generator(self.model_config)
        generator.backend_type = 'ms'
        generator.npu_device_id = 0
        
        result = generator.execute_recover_command("CMD_REINIT_NPU")
        
        self.assertEqual(result["command_result"], 1)
        self.assertIn("Recovery commands are only supported by 'atb' backend", result["error_msg"])
        self.assertEqual(result["npu_device_id"], 0)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.acl")
    def test_execute_recover_command_reinit_npu_success(self, mock_acl, _):
        """测试CMD_REINIT_NPU命令成功执行"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.infer_context = MagicMock()
        generator.infer_context.reset_all_context = MagicMock()
        generator.generator_backend = MagicMock()
        generator.generator_backend.execute_recover_command = MagicMock(return_value={
            "command_result": 0,
            "error_msg": "",
            "npu_device_id": 0
        })
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.resume_hccl_comm = MagicMock()
        mock_acl.rt.set_device = MagicMock()
        
        result = generator.execute_recover_command("CMD_REINIT_NPU")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        generator.infer_context.reset_all_context.assert_called_once()
        generator.generator_backend.execute_recover_command.assert_called_once_with("CMD_REINIT_NPU")
        mock_acl.rt.set_device.assert_called_once_with(0)
        generator.model_wrapper.resume_hccl_comm.assert_called_once()

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_reinit_npu_backend_failure(self, _):
        """测试CMD_REINIT_NPU命令后端执行失败"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.infer_context = MagicMock()
        generator.infer_context.reset_all_context = MagicMock()
        generator.generator_backend = MagicMock()
        generator.generator_backend.execute_recover_command = MagicMock(return_value={
            "command_result": 1,
            "error_msg": "Backend error",
            "npu_device_id": 0
        })
        generator.model_wrapper = MagicMock()
        
        result = generator.execute_recover_command("CMD_REINIT_NPU")
        
        self.assertEqual(result["command_result"], 1)
        self.assertEqual(result["error_msg"], "Backend error")
        generator.infer_context.reset_all_context.assert_called_once()
        generator.generator_backend.execute_recover_command.assert_called_once_with("CMD_REINIT_NPU")

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.acl")
    def test_execute_recover_command_reinit_npu_exception(self, mock_acl, _):
        """测试CMD_REINIT_NPU命令执行时抛出异常"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.infer_context = MagicMock()
        generator.infer_context.reset_all_context = MagicMock()
        generator.generator_backend = MagicMock()
        generator.generator_backend.execute_recover_command = MagicMock(side_effect=Exception("Test exception"))
        generator.model_wrapper = MagicMock()
        mock_acl.rt.set_device = MagicMock()
        
        result = generator.execute_recover_command("CMD_REINIT_NPU")
        
        self.assertEqual(result["command_result"], 1)
        self.assertIn("Failed to execute recovery command", result["error_msg"])
        self.assertEqual(result["npu_device_id"], 0)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.time")
    def test_execute_recover_command_start_engine(self, mock_time, _):
        """测试CMD_START_ENGINE命令"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = True
        generator.plugin = MagicMock()
        generator.plugin.last_sequence_ids = [1, 2, 3]
        generator.plugin.is_inference_pause = True
        generator.plugin.output_queue = None  # 没有 output_queue
        
        result = generator.execute_recover_command("CMD_START_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertIsNone(generator.plugin.last_sequence_ids)
        self.assertFalse(generator.plugin.is_inference_pause)
        self.assertFalse(generator.is_inference_pause)
        mock_time.sleep.assert_called_once_with(1)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.time")
    @patch("mindie_llm.text_generator.utils.model_output.ModelOutputWrapper")
    def test_execute_recover_command_start_engine_with_output_queue(self, mock_model_output_wrapper_class, mock_time, _):
        """测试CMD_START_ENGINE命令，包含output_queue的情况"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = True
        generator.plugin = MagicMock()
        generator.plugin.last_sequence_ids = [1, 2, 3]
        generator.plugin.is_inference_pause = True
        
        # 创建空的 output_queue
        mock_queue = MagicMock()
        mock_queue.empty = MagicMock(return_value=True)
        mock_queue.put = MagicMock()
        generator.plugin.output_queue = mock_queue
        
        mock_empty_output = MagicMock()
        mock_model_output_wrapper_class.make_empty = MagicMock(return_value=mock_empty_output)
        
        result = generator.execute_recover_command("CMD_START_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertIsNone(generator.plugin.last_sequence_ids)
        self.assertFalse(generator.plugin.is_inference_pause)
        self.assertFalse(generator.is_inference_pause)
        mock_time.sleep.assert_called_once_with(1)
        mock_queue.put.assert_called_once_with(mock_empty_output)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.time")
    def test_execute_recover_command_start_engine_output_queue_not_empty(self, mock_time, _):
        """测试CMD_START_ENGINE命令，output_queue不为空的情况"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = True
        generator.plugin = MagicMock()
        generator.plugin.last_sequence_ids = [1, 2, 3]
        generator.plugin.is_inference_pause = True
        
        # 创建非空的 output_queue
        mock_queue = MagicMock()
        mock_queue.empty = MagicMock(return_value=False)
        mock_queue.put = MagicMock()
        generator.plugin.output_queue = mock_queue
        
        result = generator.execute_recover_command("CMD_START_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertIsNone(generator.plugin.last_sequence_ids)
        self.assertFalse(generator.plugin.is_inference_pause)
        self.assertFalse(generator.is_inference_pause)
        mock_time.sleep.assert_called_once_with(1)
        # output_queue 不为空时不应该调用 put
        mock_queue.put.assert_not_called()

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_pause_engine(self, _):
        """测试CMD_PAUSE_ENGINE命令"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = False
        generator.plugin = MagicMock()
        generator.plugin.is_inference_pause = False
        generator.generator_backend = MagicMock()
        generator.generator_backend.execute_recover_command = MagicMock(return_value={
            "command_result": 0,
            "error_msg": "",
            "npu_device_id": 0
        })
        
        result = generator.execute_recover_command("CMD_PAUSE_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertTrue(generator.is_inference_pause)
        self.assertTrue(generator.plugin.is_inference_pause)
        generator.generator_backend.execute_recover_command.assert_called_once_with("CMD_PAUSE_ENGINE")

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_clear_transer(self, _):
        """测试CMD_CLEAR_TRANSER命令"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        
        result = generator.execute_recover_command("CMD_CLEAR_TRANSER")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_unknown_command(self, _):
        """测试未知命令"""
        generator = Generator(self.model_config)
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        
        result = generator.execute_recover_command("CMD_UNKNOWN")
        
        self.assertEqual(result["command_result"], 1)
        self.assertIn("Unknown recovery command", result["error_msg"])
        self.assertEqual(result["npu_device_id"], 0)


class TestPDInterface(unittest.TestCase):

    def setUp(self):
        self.config_dict = {
            'role': 'standard',
            'local_instance_id': 0,
            'local_device_ip': '127.0.0.1',
            'npu_device_id': 0,
            'local_physical_device_id': 0,
            'local_host_ip': '127.0.0.1',
            'remote_device_ips': '127.0.0.2'
        }
        self.pd_config = PDModelConfig(self.config_dict)
        self.pd_interface = PDInterface(self.pd_config)
        self.original_npu = globals().get('npu', None)
        npu_mock = MagicMock()
        npu_mock.set_device = MagicMock()
        npu_mock.max_memory_allocated = MagicMock(return_value=1024)
        globals()['npu'] = npu_mock

    def tearDown(self):
        # 恢复全局 npu 与 MindieLlmStatusCode
        if self.original_npu is not None:
            globals()['npu'] = self.original_npu

    def test_link(self):
        """测试 link 方法"""
        worker_mock = MagicMock()
        worker_mock.link.return_value = "dummy_link"
        self.pd_interface.separate_deployment_worker = worker_mock

        # 使用新的多链接参数结构进行测试
        remote_cluster_ids = {1: [10, 11]}
        remote_physical_device_ids = {1: [20, 21]}
        remote_device_ips = {1: ["192.168.1.2", "192.168.1.3"]}
        host_ips = {1: ["192.168.1.100", "192.168.1.101"]}
        remote_super_device_ids = {1: [8650754, 8650755]}
        remote_super_pod_ids = {1: [0, 0]}
        
        result = self.pd_interface.link(
            remote_cluster_ids=remote_cluster_ids,
            remote_physical_device_ids=remote_physical_device_ids,
            remote_device_ips=remote_device_ips,
            host_ips=host_ips,
            remote_super_device_ids=remote_super_device_ids,
            remote_super_pod_ids=remote_super_pod_ids
        )
        
        self.assertEqual(result, "dummy_link")
        worker_mock.link.assert_called_once_with(
            remote_cluster_ids=remote_cluster_ids,
            remote_physical_device_ids=remote_physical_device_ids,
            remote_device_ips=remote_device_ips,
            host_ips=host_ips,
            remote_super_device_ids=remote_super_device_ids,
            remote_super_pod_ids=remote_super_pod_ids
        )

    def test_unlink(self):
        """测试 unlink 方法"""
        worker_mock = MagicMock()
        worker_mock.unlink.return_value = "dummy_unlink"
        self.pd_interface.separate_deployment_worker = worker_mock

        result = self.pd_interface.unlink(1)
        self.assertEqual(result, "dummy_unlink")
        worker_mock.unlink.assert_called_once_with(1)

    def test_switch_role(self):
        """测试 switch_role 方法"""
        new_role = 'new_role'
        self.pd_interface.switch_role(new_role)
        self.assertEqual(self.pd_interface.pd_config.model_role, new_role)

    def test_pull_kv_success(self):
        """测试 pull_kv 成功分支"""
        self.pd_interface.device_inited = False
        # 构造一个 mock worker，模拟 pull_blocks 返回 SUCCESS
        worker_mock = MagicMock()
        worker_mock.pull_blocks.return_value = MindieLlmStatusCode.SUCCESS
        self.pd_interface.separate_deployment_worker = worker_mock

        # 构造一个带有 is_prefill 属性的输入 metadata 对象
        dummy_input_metadata = MagicMock()
        dummy_input_metadata.is_prefill = False
        # pd_info 列表中只有一项，model_instance_id 为 10
        pd_infos = [(10, [1, 2], [3, 4])]
        ret, model_instance_id = self.pd_interface.pull_kv(dummy_input_metadata, pd_infos)
        self.assertEqual(ret, MindieLlmStatusCode.SUCCESS)
        self.assertEqual(model_instance_id, 0)
        worker_mock.pull_blocks.assert_called_once_with(remote_model_instance_id=10,
                                                        src_block_table=[1, 2],
                                                        dst_block_table=[3, 4])

    def test_pull_kv_failure(self):
        """测试 pull_kv 当 pull_blocks 返回错误时直接返回错误"""
        self.pd_interface.device_inited = False
        worker_mock = MagicMock()
        worker_mock.pull_blocks.return_value = "FAIL"
        self.pd_interface.separate_deployment_worker = worker_mock

        dummy_input_metadata = MagicMock()
        dummy_input_metadata.is_prefill = False
        # pd_info 中的 model_instance_id 为 99
        pd_infos = [(99, [1], [2])]
        ret, model_instance_id = self.pd_interface.pull_kv(dummy_input_metadata, pd_infos)
        self.assertEqual(ret, "FAIL")
        self.assertEqual(model_instance_id, 99)
        # 由于提前返回，队列中不应有 input_metadata
        self.assertTrue(self.pd_interface.input_metadata_queue.empty())
        worker_mock.pull_blocks.assert_called_once_with(remote_model_instance_id=99,
                                                        src_block_table=[1],
                                                        dst_block_table=[2])

    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDist')
    @patch('mindie_llm.text_generator.utils.separate_deployment_engine.LLMDataDistConfig')
    def test_init_sepd_engine(self, mock_llm_data_dist_config, mock_llm_data_dist):
        """测试 _init_sepd_engine"""
        self.config_dict = {
            'role': 'flex',
            'local_instance_id': 0,
            'local_device_ip': '127.0.0.1',
            'npu_device_id': 0,
            'local_physical_device_id': 0,
            'local_host_ip': '127.0.0.1',
            'remote_device_ips': '127.0.0.2',
            'local_super_device_id': 0,
            'local_super_pod_id': 0
        }
        self.pd_config = PDModelConfig(self.config_dict)
        self.pd_interface = PDInterface(self.pd_config)
        self.pd_interface._init_sepd_engine()

if __name__ == "__main__":
    unittest.main()
