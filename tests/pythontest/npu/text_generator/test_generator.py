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
import os
import queue
import unittest
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from ddt import ddt, data, unpack

from mindie_llm.utils.env import ENV
from mindie_llm.utils.status import MindieLlmStatusCode
from mindie_llm.text_generator.generator import Generator, PDInterface, PDModelConfig, STANDARD_TAG
from mindie_llm.text_generator.utils.generation_output import GenerationOutput
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.input_metadata import InputMetadata, SAMPLING_DTYPE
from mindie_llm.text_generator.adapter import generator_torch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from mindie_llm.connector.common.model_execute_data_pb2 import LoraOperationStatus
from mindie_llm.modeling.model_wrapper.model_info import ModelInfo
from mindie_llm.utils.log.error_code import ErrorCode, ErrorCodeException

from tests.pythontest.npu import FakeModelRunner, FakeModelWrapper, FakeParallelInfo


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
            'cpu_mem': '5', 'deploy_type': 'INTER_PROCESS', 'dp': '1', 'executor_type': 'LLM_EXECUTOR_PYTHON',
            'globalRankIds': '', 'globalWorldSize': '0', 'interNodeKmcKsfMaster': 'tools/pmt/master/ksfa',
            'interNodeKmcKsfStandby': 'tools/pmt/standby/ksfb', 'interNodeTLSEnabled': '1',
            'interNodeTlsCaFiles': 'ca.pem,', 'interNodeTlsCaPath': 'security/grpc/ca/',
            'interNodeTlsCert': 'security/grpc/certs/server.pem', 'interNodeTlsCrlFiles': 'server_crl.pem,',
            'interNodeTlsCrlPath': 'security/grpc/certs/', 'interNodeTlsPk': 'security/grpc/keys/server.key.pem',
            'interNodeTlsPkPwd': 'security/grpc/pass/mindie_server_key_pwd.txt', 'isMaster': '0', 'localIP': '',
            'local_rank': '0', 'masterIP': '', 'max_input_len': '2048',
            'max_iter_times': '512', 'max_prefill_tokens': '8192', 'max_seq_len': '2560',
            'model_id': '/home/data/llama3', 'model_instance_number': '1',
            'model_instance_type': 'Standard', 'model_name': 'deepseekv2', 'moe_tp': '1',
            'multiNodesInferEnabled': '0', 'multiNodesInferPort': '1120', 'npu_device_id': '0',
            'npu_device_ids': '0,1,2,3,4,5,6,7', 'npu_mem': '-1', 'rank': '0', 'slaveIPs': '',
            'speculation_gamma': '0', 'tp': '4', 'trust_remote_code': '0', 'world_size': '4',
            'num_speculative_tokens': '0', 'max_batch_size': '5', 'max_prefill_batch_size': '5',
            'distributed_enable': 'false', 'vocab_size': 100000, 'enable_warmup_with_sampling': 'false',
            'cp': '1', 'sp': '1', 'moe_ep': '1'
        }

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config['dp']),
            tp=int(self.model_config['tp']),
            sp=int(self.model_config['sp']),
            cp=int(self.model_config['cp'])
        )
        self.fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)

        self.generation_output_prefill = GenerationOutput(
            sequence_ids=np.array([0, 1, 2, 3, 4]),
            parent_sequence_ids=np.array([0, 1, 2, 3, 4]),
            group_indices=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            token_ids=np.array([[273], [273], [273], [273], [273]]),
            logprobs=np.array([[0.], [0.], [0.], [0.], [0.]], dtype=np.float32),
            top_token_ids=np.array([], dtype=np.int32),
            top_logprobs=np.array([], dtype=np.float32),
            num_new_tokens=np.array([1, 1, 1, 1, 1]),
            num_top_tokens=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            cumulative_logprobs=np.array([0., 0., 0., 0., 0.], dtype=np.float32),
            finish_reason=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            truncation_indices=np.array([0, 0, 0, 0, 0]),
            current_token_indices=np.array([0, 0, 0, 0, 0], dtype=np.int32), 
            eos_info=None, trace_ids=[None, None, None, None, None],
            simulator_ids=None
        )
        
        self.generation_output_decode = GenerationOutput(
            sequence_ids=np.array([0, 1, 2, 3, 4]),
            parent_sequence_ids=np.array([0, 1, 2, 3, 4]),
            group_indices=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            token_ids=np.array([[273], [273], [273], [273], [273]]),
            logprobs=np.array([[0.], [0.], [0.], [0.], [0.]], dtype=np.float32),
            top_token_ids=np.array([], dtype=np.int32),
            top_logprobs=np.array([], dtype=np.float32),
            num_new_tokens=np.array([1, 1, 1, 1, 1]),
            num_top_tokens=np.array([0, 0, 0, 0, 0], dtype=np.int32),
            cumulative_logprobs=np.array([0., 0., 0., 0., 0.], dtype=np.float32),
            finish_reason=np.array([6, 6, 6, 6, 6], dtype=np.int32),
            truncation_indices=np.array([0, 0, 0, 0, 0]),
            current_token_indices=np.array([0, 0, 0, 0, 0], dtype=np.int32), 
            eos_info=None, trace_ids=[None, None, None, None, None],
            simulator_ids=None
        )

    @patch.object(generator_torch, 'GeneratorTorch')
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.plugins.plugin_manager.PluginManager.generate_token")
    def test_init(
        self,
        mock_plugin_manager_generate_token,
        mock_model_runner,
        mock_generator_torch
    ):
        mock_model_runner.return_value = self.fake_model_runner
        fake_runner = mock_model_runner.return_value
        mock_model_info = ModelInfo(fake_runner.device,
                                    fake_runner.kv_cache_dtype,
                                    2,
                                    fake_runner.num_layers,
                                    fake_runner.num_kv_heads,
                                    fake_runner.head_size,
                                    k_head_size=fake_runner.k_head_size,
                                    v_head_size=fake_runner.v_head_size,
                                    enable_nz=False,
                                    kvcache_quant_layers=fake_runner.kvcache_quant_layers)
        
        mock_plugin_manager_generate_token.return_value = self.generation_output_decode
        fake_model_wrapper = FakeModelWrapper(mock_model_info, fake_runner)

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.model_wrapper = fake_model_wrapper
        mock_generator_torch_ins.obfuscation_func = None
        mock_generator_torch_ins.model_info = mock_model_info

        # test init with specific npu_mem
        self.model_config["npu_mem"] = 10

        # assume benchmark file already exists
        ENV.benchmark_filepath = "./tmp.txt"
        if not os.path.exists(ENV.benchmark_filepath):
            # 使用 'w' 模式创建文件
            with open(ENV.benchmark_filepath, 'w') as file:
                file.write('Hello, world!')
            os.chmod(ENV.benchmark_filepath, 0o600)

        generator = Generator(self.model_config)
        self.assertIsNotNone(generator)
        self.assertFalse(os.path.exists(ENV.benchmark_filepath))

    @data('prefill', 'decoder')
    @patch.object(generator_torch, 'GeneratorTorch')
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.plugins.plugin_manager.PluginManager.generate_token")
    def test_init_with_pd_role(
        self,
        role,
        mock_plugin_manager_generate_token,
        mock_model_runner,
        mock_generator_torch
    ):
        mock_model_runner.return_value = self.fake_model_runner

        fake_runner = mock_model_runner.return_value
        mock_model_info = ModelInfo(fake_runner.device,
                                    fake_runner.kv_cache_dtype,
                                    2,
                                    fake_runner.num_layers,
                                    fake_runner.num_kv_heads,
                                    fake_runner.head_size,
                                    k_head_size=fake_runner.k_head_size,
                                    v_head_size=fake_runner.v_head_size,
                                    enable_nz=False,
                                    kvcache_quant_layers=fake_runner.kvcache_quant_layers)

        fake_model_wrapper = FakeModelWrapper(mock_model_info, fake_runner)

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.enable_dap = False
        mock_generator_torch_ins.model_wrapper = fake_model_wrapper
        mock_generator_torch_ins.model_info = mock_model_info
        mock_generator_torch_ins.obfuscation_func = None

        mock_plugin_manager_generate_token.return_value = self.generation_output_decode

        config_dict = {
            'role': role,
            'local_instance_id': 0,
            'local_device_ip': '127.0.0.1',
            'npu_device_id': 0,
            'local_physical_device_id': 0,
            'local_host_ip': '127.0.0.1',
            'remote_device_ips': '127.0.0.2'
        }
        self.model_config.update(config_dict)

        try:
            generator = Generator(self.model_config)
            self.assertIsNotNone(generator)
        except RuntimeError as e:
            if "Warmup failed" in str(e):
                pass
            else:
                raise

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_generate_token_plugin_none_raise_error(self, _):
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        input_metadata = MagicMock(spec=InputMetadata)

        with self.assertRaises(AttributeError):
            _ = generator.generate_token(input_metadata)

    
    @data('standard', 'flex')
    @patch.object(generator_torch, 'GeneratorTorch')
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.plugins.plugin_manager.PluginManager.generate_token")
    def test_init_with_standard_flex(
        self,
        role,
        mock_plugin_manager_generate_token,
        mock_model_runner,
        mock_generator_torch
    ):
        mock_model_runner.return_value = self.fake_model_runner

        fake_runner = mock_model_runner.return_value
        mock_model_info = ModelInfo(fake_runner.device,
                                    fake_runner.kv_cache_dtype,
                                    2,
                                    fake_runner.num_layers,
                                    fake_runner.num_kv_heads,
                                    fake_runner.head_size,
                                    k_head_size=fake_runner.k_head_size,
                                    v_head_size=fake_runner.v_head_size,
                                    enable_nz=False,
                                    kvcache_quant_layers=fake_runner.kvcache_quant_layers)

        fake_model_wrapper = FakeModelWrapper(mock_model_info, fake_runner)

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.enable_dap = False
        mock_generator_torch_ins.model_wrapper = fake_model_wrapper
        mock_generator_torch_ins.model_info = mock_model_info
        mock_generator_torch_ins.obfuscation_func = None
        mock_generator_torch_ins.backend_type = 'atb'
        
        mock_plugin_manager_generate_token.side_effect = [
            self.generation_output_prefill, self.generation_output_decode
        ]
        
        config_dict = {
            'role': role,
            'local_instance_id': 0,
            'local_device_ip': '127.0.0.1',
            'npu_device_id': 0,
            'local_physical_device_id': 0,
            'local_host_ip': '127.0.0.1',
            'remote_device_ips': '127.0.0.2'
        }
        self.model_config.update(config_dict)
        
        try:
            generator = Generator(self.model_config)
            self.assertIsNotNone(generator)
        except RuntimeError as e:
            if "Warmup failed" in str(e):
                pass
            else:
                raise

    @patch.object(generator_torch, 'GeneratorTorch')
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.plugins.plugin_manager.PluginManager.generate_token")
    def test_init_with_prefixcache(
        self,
        mock_plugin_manager_generate_token,
        mock_model_runner,
        mock_generator_torch
    ):
        config_dict = {'plugin_params': '{\"plugin_type\": \"prefix_cache\"}', 'cp': '2'}
        self.model_config.update(config_dict)

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config['dp']),
            tp=int(self.model_config['tp']),
            sp=int(self.model_config['sp']),
            cp=int(self.model_config['cp'])
        )
        fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)

        mock_model_runner.return_value = fake_model_runner

        fake_runner = mock_model_runner.return_value
        mock_model_info = ModelInfo(fake_runner.device,
                                    fake_runner.kv_cache_dtype,
                                    2,
                                    fake_runner.num_layers,
                                    fake_runner.num_kv_heads,
                                    fake_runner.head_size,
                                    k_head_size=fake_runner.k_head_size,
                                    v_head_size=fake_runner.v_head_size,
                                    enable_nz=False,
                                    kvcache_quant_layers=fake_runner.kvcache_quant_layers)

        fake_model_wrapper = FakeModelWrapper(mock_model_info, fake_runner)

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.enable_dap = False
        mock_generator_torch_ins.model_wrapper = fake_model_wrapper
        mock_generator_torch_ins.model_info = mock_model_info
        mock_generator_torch_ins.obfuscation_func = None
        mock_generator_torch_ins.backend_type = 'atb'

        mock_plugin_manager_generate_token.side_effect = [
            self.generation_output_prefill, self.generation_output_decode,
            self.generation_output_prefill, self.generation_output_decode
        ]
        
        try:
            generator = Generator(self.model_config)
            self.assertIsNotNone(generator)
        except RuntimeError as e:
            if "Warmup failed" in str(e):
                pass
            else:
                raise
    
    @patch.object(generator_torch, 'GeneratorTorch')
    @patch("mindie_llm.modeling.model_wrapper.atb.atb_model_wrapper.ModelRunner")
    @patch("mindie_llm.text_generator.plugins.plugin_manager.PluginManager.generate_token")
    def test_init_with_structured_output(
        self,
        mock_plugin_manager_generate_token,
        mock_model_runner,
        mock_generator_torch
    ):
        config_dict = {'enable_structured_output': True}
        self.model_config.update(config_dict)

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config['dp']),
            tp=int(self.model_config['tp']),
            sp=int(self.model_config['sp']),
            cp=int(self.model_config['cp'])
        )
        fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)

        mock_model_runner.return_value = fake_model_runner

        fake_runner = mock_model_runner.return_value
        mock_model_info = ModelInfo(fake_runner.device,
                                    fake_runner.kv_cache_dtype,
                                    2,
                                    fake_runner.num_layers,
                                    fake_runner.num_kv_heads,
                                    fake_runner.head_size,
                                    k_head_size=fake_runner.k_head_size,
                                    v_head_size=fake_runner.v_head_size,
                                    enable_nz=False,
                                    kvcache_quant_layers=fake_runner.kvcache_quant_layers)

        fake_model_wrapper = FakeModelWrapper(mock_model_info, fake_runner)

        mock_generator_torch_ins = mock_generator_torch.return_value
        mock_generator_torch_ins.enable_dap = False
        mock_generator_torch_ins.model_wrapper = fake_model_wrapper
        mock_generator_torch_ins.model_info = mock_model_info
        mock_generator_torch_ins.obfuscation_func = None
        mock_generator_torch_ins.backend_type = 'atb'

        mock_tokenizer = MagicMock()
        mock_tokenizer.__len__ = MagicMock(return_value=1000)
        mock_generator_torch_ins.tokenizer = mock_tokenizer

        mock_plugin_manager_generate_token.side_effect = [
            self.generation_output_prefill, self.generation_output_decode,
            self.generation_output_prefill, self.generation_output_decode
        ]
        
        try:
            generator = Generator(self.model_config)
            self.assertTrue(generator.plugin_manager._structured_output_enabled)
            self.assertIsNotNone(generator.plugin_manager._structured_output_manager)
        except RuntimeError as e:
            if "Warmup failed" in str(e):
                pass
            else:
                raise

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

        def mock_generate_token_side_effect(*args, **kwargs):
            return GenerationOutput(
                sequence_ids=np.array([0, 1]),
                parent_sequence_ids=np.array([0, 1]),
                group_indices=[(0, 1), (1, 2)]
            )

        generator.generate_token = MagicMock(side_effect=mock_generate_token_side_effect)
        generation_output = generator.prefill(requests)
        self.assertIsInstance(generation_output, GenerationOutput)
        generation_output = generator.decode(requests)
        self.assertIsInstance(generation_output, GenerationOutput)
        req1.block_tables = np.array([0, -1])
        req2.block_tables = np.array([1, 2])
        generation_output = generator.generate_mix(requests, is_prefill_batch=np.array([False, True]))
        self.assertIsInstance(generation_output, GenerationOutput)
        self.assertEqual(generator.generate_token.call_count, 3)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_load_lora_not_active(self, _):
        """测试lora特性未使能时的返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = None
        generator.separate_deployment_worker = None
        ret = generator.load_lora("fake_id", "fake_path")
        self.assertEqual(ret, LoraOperationStatus.UNSUPPORT_CMD)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_load_lora_success(self, _):
        """测试lora特性使能时的加载成功返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = MagicMock()
        generator.model_wrapper.adapter_manager.load_adapter = MagicMock()
        generator.separate_deployment_worker = None
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
        generator.separate_deployment_worker = None
        ret = generator.load_lora("fake_id", "fake_path")
        self.assertEqual(ret, expected_ret)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_unload_lora_not_active(self, _):
        """测试lora特性未使能时的返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = None
        generator.separate_deployment_worker = None
        ret = generator.unload_lora("fake_id")
        self.assertEqual(ret, LoraOperationStatus.UNSUPPORT_CMD)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_unload_lora_success(self, _):
        """测试lora特性使能时的卸载成功返回值"""
        generator = Generator(self.model_config)
        generator.model_wrapper = MagicMock()
        generator.model_wrapper.adapter_manager = MagicMock()
        generator.model_wrapper.adapter_manager.unload_adapter = MagicMock()
        generator.separate_deployment_worker = None
        ret = generator.unload_lora("fake_id")
        self.assertEqual(ret, LoraOperationStatus.LORA_CMD_SUCCESS)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.acl")
    def test_execute_recover_command_reinit_npu_success(self, mock_acl, _):
        """测试CMD_REINIT_NPU命令成功执行"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.plugin = MagicMock()
        generator.plugin_manager = MagicMock()
        generator.infer_context = MagicMock()
        generator.infer_context.reset_all_context = MagicMock()
        generator.generator_backend = MagicMock()
        generator.generator_backend.execute_recover_command = MagicMock(return_value={
            "command_result": 0,
            "error_msg": "",
            "npu_device_id": 0
        })
        generator.model_wrapper = MagicMock()
        
        result = generator.execute_recover_command("CMD_REINIT_NPU")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        generator.infer_context.reset_all_context.assert_called_once()
        generator.generator_backend.execute_recover_command.assert_called_once_with("CMD_REINIT_NPU")

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_reinit_npu_backend_failure(self, _):
        """测试CMD_REINIT_NPU命令后端执行失败"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.plugin = MagicMock()
        generator.plugin_manager = MagicMock()
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
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.plugin = MagicMock()
        generator.plugin_manager = MagicMock()
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
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = True
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.last_sequence_ids = [1, 2, 3]
        generator.plugin_manager.is_inference_pause = True
        generator.plugin_manager.output_queue = None  # 没有 output_queue
        
        result = generator.execute_recover_command("CMD_START_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertIsNone(generator.plugin_manager.last_sequence_ids)
        self.assertFalse(generator.plugin_manager.is_inference_pause)
        self.assertFalse(generator.is_inference_pause)
        mock_time.sleep.assert_called_once_with(1)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.time")
    @patch("mindie_llm.text_generator.utils.model_output.ModelOutputWrapper")
    def test_execute_recover_command_start_engine_with_output_queue(self, mock_model_output_wrapper_class, mock_time, _):
        """测试CMD_START_ENGINE命令，包含output_queue的情况"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = True
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.last_sequence_ids = [1, 2, 3]
        generator.plugin_manager.is_inference_pause = True
        
        # 创建空的 output_queue
        mock_queue = MagicMock()
        mock_queue.empty = MagicMock(return_value=True)
        mock_queue.put = MagicMock()
        generator.plugin_manager.output_queue = mock_queue
        
        mock_empty_output = MagicMock()
        mock_model_output_wrapper_class.make_empty = MagicMock(return_value=mock_empty_output)
        
        result = generator.execute_recover_command("CMD_START_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertIsNone(generator.plugin_manager.last_sequence_ids)
        self.assertFalse(generator.plugin_manager.is_inference_pause)
        self.assertFalse(generator.is_inference_pause)
        mock_time.sleep.assert_called_once_with(1)
        mock_queue.put.assert_called_once_with(mock_empty_output)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    @patch("mindie_llm.text_generator.generator.time")
    def test_execute_recover_command_start_engine_output_queue_not_empty(self, mock_time, _):
        """测试CMD_START_ENGINE命令，output_queue不为空的情况"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = True
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.last_sequence_ids = [1, 2, 3]
        generator.plugin_manager.is_inference_pause = True
        
        # 创建非空的 output_queue
        mock_queue = MagicMock()
        mock_queue.empty = MagicMock(return_value=False)
        mock_queue.put = MagicMock()
        generator.plugin_manager.output_queue = mock_queue
        
        result = generator.execute_recover_command("CMD_START_ENGINE")
        
        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertIsNone(generator.plugin_manager.last_sequence_ids)
        self.assertFalse(generator.plugin_manager.is_inference_pause)
        self.assertFalse(generator.is_inference_pause)
        mock_time.sleep.assert_called_once_with(1)
        # output_queue 不为空时不应该调用 put
        mock_queue.put.assert_not_called()

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_pause_engine(self, _):
        """测试CMD_PAUSE_ENGINE命令"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = False
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.is_inference_pause = False
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
        self.assertTrue(generator.plugin_manager.is_inference_pause)
        generator.generator_backend.execute_recover_command.assert_called_once_with("CMD_PAUSE_ENGINE")

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_pause_engine_backend_failure(self, _):
        """测试CMD_PAUSE_ENGINE命令后端执行失败时仍正确设置pause状态"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = False
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.is_inference_pause = False
        generator.generator_backend = MagicMock()
        generator.generator_backend.execute_recover_command = MagicMock(return_value={
            "command_result": 1,
            "error_msg": "Stop device failed",
            "npu_device_id": 0
        })

        result = generator.execute_recover_command("CMD_PAUSE_ENGINE")

        self.assertEqual(result["command_result"], 1)
        self.assertEqual(result["error_msg"], "Stop device failed")
        self.assertEqual(result["npu_device_id"], 0)
        # 即使后端失败，pause 状态应在调用 backend 前已设置
        self.assertTrue(generator.is_inference_pause)
        self.assertTrue(generator.plugin_manager.is_inference_pause)
        generator.generator_backend.execute_recover_command.assert_called_once_with("CMD_PAUSE_ENGINE")

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_pause_engine_roce(self, _):
        """测试CMD_PAUSE_ENGINE_ROCE命令"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        generator.is_inference_pause = False
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.is_inference_pause = False

        result = generator.execute_recover_command("CMD_PAUSE_ENGINE_ROCE")

        self.assertEqual(result["command_result"], 0)
        self.assertEqual(result["error_msg"], "")
        self.assertEqual(result["npu_device_id"], 0)
        self.assertTrue(generator.is_inference_pause)
        self.assertTrue(generator.plugin_manager.is_inference_pause)

    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_execute_recover_command_clear_transer(self, _):
        """测试CMD_CLEAR_TRANSER命令"""
        generator = Generator(self.model_config)
        generator.separate_deployment_worker = None
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
        generator.separate_deployment_worker = None
        generator.backend_type = 'atb'
        generator.npu_device_id = 0
        
        result = generator.execute_recover_command("CMD_UNKNOWN")
        
        self.assertEqual(result["command_result"], 1)
        self.assertIn("Unknown recovery command", result["error_msg"])
        self.assertEqual(result["npu_device_id"], 0)

    @patch('mindie_llm.utils.prof.profiler.span_end')
    @patch('mindie_llm.utils.prof.profiler.span_attr')
    @patch('mindie_llm.utils.prof.profiler.span_start')
    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_generate_token_sets_fault_device_when_exception_maps_to_error_code(
        self, _, mock_span_start, mock_span_attr, mock_span_end
    ):
        """异常信息命中 convert_exception_to_error_code 时设置 is_fault_device 并抛出 ErrorCodeException。"""
        mock_span_start.return_value = None
        generator = Generator(self.model_config)
        generator.pd_config = MagicMock()
        generator.pd_config.model_role = STANDARD_TAG
        generator.input_metadata_queue = queue.Queue()
        generator.rank = 0
        generator.async_inference = False
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.generate_token.side_effect = RuntimeError(
            "backend reported MIE05E0000005 in stack"
        )
        generator.generator_backend = MagicMock()
        generator.generator_backend.is_fault_device = False
        im = MagicMock(spec=InputMetadata)
        im.batch_seq_len = np.array([0])
        im.is_prefill = False
        with self.assertRaises(ErrorCodeException) as cm:
            generator.generate_token(im, warmup=False)
        self.assertEqual(cm.exception.error_code, ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY)
        self.assertTrue(generator.generator_backend.is_fault_device)

    @patch('mindie_llm.utils.prof.profiler.span_end')
    @patch('mindie_llm.utils.prof.profiler.span_attr')
    @patch('mindie_llm.utils.prof.profiler.span_start')
    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_generate_token_notify_force_stop_when_inference_paused(
        self, _, mock_span_start, mock_span_attr, mock_span_end
    ):
        """推理暂停时 FORCE STOP 异常应调用 notify_force_stop_exception 并返回空 GenerationOutput。"""
        mock_span_start.return_value = None
        generator = Generator(self.model_config)
        generator.pd_config = MagicMock()
        generator.pd_config.model_role = STANDARD_TAG
        generator.input_metadata_queue = queue.Queue()
        generator.rank = 0
        generator.async_inference = False
        generator.is_inference_pause = True
        generator.plugin_manager = MagicMock()
        generator.plugin_manager.generate_token.side_effect = RuntimeError("User FORCE STOP request")
        generator.generator_backend = MagicMock()
        generator.generator_backend.notify_force_stop_exception = MagicMock()
        im = MagicMock(spec=InputMetadata)
        im.batch_seq_len = np.array([0])
        im.is_prefill = False
        out = generator.generate_token(im, warmup=False)
        self.assertIsInstance(out, GenerationOutput)
        self.assertEqual(out.sequence_ids.size, 0)
        generator.generator_backend.notify_force_stop_exception.assert_called_once()

    @patch('mindie_llm.utils.prof.profiler.span_end')
    @patch('mindie_llm.utils.prof.profiler.span_attr')
    @patch('mindie_llm.utils.prof.profiler.span_start')
    @patch("mindie_llm.text_generator.generator.Generator.__init__", return_value=None)
    def test_generate_token_force_stop_reraises_when_not_paused(
        self, _, mock_span_start, mock_span_attr, mock_span_end
    ):
        """非暂停状态下 FORCE STOP 仍按未知异常向上抛出。"""
        mock_span_start.return_value = None
        generator = Generator(self.model_config)
        generator.pd_config = MagicMock()
        generator.pd_config.model_role = STANDARD_TAG
        generator.input_metadata_queue = queue.Queue()
        generator.rank = 0
        generator.async_inference = False
        generator.is_inference_pause = False
        generator.plugin_manager = MagicMock()
        err = RuntimeError("FORCE STOP abort")
        generator.plugin_manager.generate_token.side_effect = err
        generator.generator_backend = MagicMock()
        im = MagicMock(spec=InputMetadata)
        im.batch_seq_len = np.array([0])
        im.is_prefill = False
        with self.assertRaises(RuntimeError) as cm:
            generator.generate_token(im, warmup=False)
        self.assertIs(cm.exception, err)
        generator.generator_backend.notify_force_stop_exception.assert_not_called()


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
        remote_dp_instance_ids = {1: [5, 6]}
        local_dp_instance_id = 99        
        
        self.pd_interface.link(
            remote_cluster_ids=remote_cluster_ids,
            remote_physical_device_ids=remote_physical_device_ids,
            remote_device_ips=remote_device_ips,
            host_ips=host_ips,
            remote_super_device_ids=remote_super_device_ids,
            remote_super_pod_ids=remote_super_pod_ids,
            remote_dp_instance_ids=remote_dp_instance_ids,
            local_dp_instance_id=local_dp_instance_id
        )
        
        worker_mock.link.assert_called_once_with(
            remote_cluster_ids=remote_cluster_ids,
            remote_physical_device_ids=remote_physical_device_ids,
            remote_device_ips=remote_device_ips,
            host_ips=host_ips,
            remote_super_device_ids=remote_super_device_ids,
            remote_super_pod_ids=remote_super_pod_ids,
            remote_dp_instance_ids=remote_dp_instance_ids,
            local_dp_instance_id=local_dp_instance_id
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