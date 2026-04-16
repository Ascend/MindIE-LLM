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
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass
import torch

import numpy as np

from mindie_llm.text_generator.generator import Generator
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.text_generator.utils.sampling_output import SamplingOutput
from mindie_llm.text_generator.plugins.plugin_manager import PluginManager

from mindie_llm.text_generator.utils.input_metadata import InputMetadata
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams

from tests.pythontest.npu import FakeModelRunner, FakeParallelInfo


class TestPluginManager(unittest.TestCase):
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
            "model_name": "deepseekv2",
            "moe_tp": "1",
            "multiNodesInferEnabled": "0",
            "multiNodesInferPort": "1120",
            "npu_device_id": "0",
            "npu_device_ids": "0,1,2,3",
            "npu_mem": "-1",
            "rank": "0",
            "slaveIPs": "",
            "speculation_gamma": "0",
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

        self.mock_npu_sync = patch("torch.npu.synchronize", return_value=None).start()
        self.mock_model_runner = patch("atb_llm.runner.model_runner.ModelRunner").start()
        self.mock_warm_up = patch.object(Generator, "warm_up").start()
        self.mock_obfuscation_func = patch.object(GeneratorTorch, "_get_obfuscation_func").start()
        self.mock_forward = patch.object(GeneratorTorch, "forward").start()

        # 确保测试结束后停止所有 patch
        self.addCleanup(patch.stopall)

    def test_generate_token_greedy(self):
        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10)  # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
            return logits

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config["dp"]),
            tp=int(self.model_config["tp"]),
            sp=int(self.model_config["sp"]),
            cp=int(self.model_config["cp"]),
        )
        self.mock_model_runner.return_value = FakeModelRunner(parallel_info=fake_parallel_info)
        self.mock_obfuscation_func.return_value = None
        self.mock_forward.side_effect = side_effect_forward
        self.mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0.0, 0.0, 0, 1.0, 1.0, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )

        gen_len = 2
        req = Request.request_from_token(
            input1,
            sampling_params=greedy_param,
            generation_params=GenerationParams(max_new_tokens=gen_len),
        )

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

    @patch("mindie_llm.utils.env.ENV.async_inference", return_value=True)
    def test_generate_token_async(self, mock_env_asycn_on):
        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10)  # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
            return logits

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config["dp"]),
            tp=int(self.model_config["tp"]),
            sp=int(self.model_config["sp"]),
            cp=int(self.model_config["cp"]),
        )
        self.mock_model_runner.return_value = FakeModelRunner(parallel_info=fake_parallel_info, device="npu")

        self.mock_npu_sync.return_value = None
        self.mock_obfuscation_func.return_value = None
        self.mock_forward.side_effect = side_effect_forward
        self.mock_warm_up.return_value = 10

        generator = Generator(self.model_config)

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0.0, 0.0, 0.7, 3.0, 0.92, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )

        gen_len = 2
        req = Request.request_from_token(
            input1,
            sampling_params=greedy_param,
            generation_params=GenerationParams(max_new_tokens=gen_len),
        )

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

    @patch("mindie_llm.utils.env.ENV.async_inference", return_value=True)
    def test_forward_loop_exit_on_sample_exception(self, mock_env_asycn_on):
        def side_effect_forward(model_inputs, **kwargs):
            logits = torch.zeros(1, 10)  # 假定词表长度为10
            logits[0][2] = 2
            logits[0][5] = 3
            logits[0][8] = 4
            return logits

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config["dp"]),
            tp=int(self.model_config["tp"]),
            sp=int(self.model_config["sp"]),
            cp=int(self.model_config["cp"]),
        )
        self.mock_model_runner.return_value = FakeModelRunner(parallel_info=fake_parallel_info, device="npu")

        self.mock_npu_sync.return_value = None
        self.mock_obfuscation_func.return_value = None
        self.mock_forward.side_effect = side_effect_forward
        self.mock_warm_up.return_value = 10

        generator = Generator(self.model_config)
        generator.plugin_manager.generator_backend.sample = MagicMock(side_effect=RuntimeError("mock sample error"))
        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0.0, 0.0, 0.7, 3.0, 0.92, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )

        gen_len = 2
        req = Request.request_from_token(
            input1,
            sampling_params=greedy_param,
            generation_params=GenerationParams(max_new_tokens=gen_len),
        )

        meta_data = InputMetadata.from_requests([req], block_tables, True)
        meta_data.batch_block_tables = block_tables

        with patch("os._exit") as mock_exit:
            generator.generate_token(meta_data)
            generator.plugin_manager.forward_thread.join(timeout=5)
            self.assertTrue(mock_exit.called)
            mock_exit.assert_called_with(1)


class TestPluginManagerStaticMethods(unittest.TestCase):
    """测试PluginManager的静态方法"""

    def test_unsqueeze_sampling_output(self):
        """测试unsqueeze_sampling_output方法"""
        sampling_output = SamplingOutput(
            sequence_ids=np.array([1, 2, 3]),
            parent_sequence_ids=np.array([1, 2, 3]),
            group_indices=[(0, 1), (1, 2), (2, 3)],
            repeating_indices=np.array([0, 1, 2]),
            token_ids=np.array([1, 2, 3]),
            logprobs=np.array([0.1, 0.2, 0.3]),
            top_token_ids=np.array([[1, 2], [3, 4], [5, 6]]),
            top_logprobs=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            cumulative_logprobs=np.array([0.1, 0.2, 0.3]),
            num_new_tokens=np.array([1, 1, 1]),
            num_top_tokens=np.array([2, 2, 2]),
        )

        PluginManager.unsqueeze_sampling_output(sampling_output)

        # 验证维度扩展
        self.assertEqual(sampling_output.token_ids.shape, (3, 1))
        self.assertEqual(sampling_output.logprobs.shape, (3, 1))
        self.assertEqual(sampling_output.top_token_ids.shape, (3, 1, 2))
        self.assertEqual(sampling_output.top_logprobs.shape, (3, 1, 2))

    def test_filter_splitfuse_token_ids_with_repeating_indices(self):
        """测试filter_splitfuse_token_ids方法 - 有repeating_indices的情况"""
        input_metadata = MagicMock()
        input_metadata.batch_is_prefill = np.array([True, False, True])
        input_metadata.batch_last_prompt = np.array([False, True, True])

        sampling_output = MagicMock()
        # repeating_indices的长度应该与batch_is_prefill在索引后的长度一致
        # 原始batch_is_prefill长度为3，经过repeating_indices=[0,2]索引后长度为2
        sampling_output.repeating_indices = np.array([0, 2])
        sampling_output.token_ids = np.array([1, 2])  # 长度应该与repeating_indices一致

        PluginManager.filter_splitfuse_token_ids(input_metadata, sampling_output)

        # 验证token_ids被正确设置为-1
        # batch_is_prefill[repeating_indices] = [True, True], batch_last_prompt[repeating_indices] = [False, True]
        # 所以第一个位置(True & ~False=True)应该被设为-1
        expected = np.array([-1, 2])
        np.testing.assert_array_equal(sampling_output.token_ids, expected)

    def test_filter_splitfuse_token_ids_without_repeating_indices(self):
        """测试filter_splitfuse_token_ids方法 - 无repeating_indices的情况"""
        input_metadata = MagicMock()
        input_metadata.batch_is_prefill = np.array([True, False, True])
        input_metadata.batch_last_prompt = np.array([False, True, True])

        sampling_output = MagicMock()
        sampling_output.repeating_indices = None
        sampling_output.token_ids = np.array([1, 2, 3])

        PluginManager.filter_splitfuse_token_ids(input_metadata, sampling_output)

        # 验证token_ids被正确设置为-1
        expected = np.array([-1, 2, 3])
        np.testing.assert_array_equal(sampling_output.token_ids, expected)

    def test_filter_splitfuse_token_ids_no_prefill(self):
        """测试filter_splitfuse_token_ids方法 - batch_is_prefill为None的情况"""
        input_metadata = MagicMock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        sampling_output = MagicMock()
        sampling_output.repeating_indices = None
        sampling_output.token_ids = np.array([1, 2, 3])

        # 应该不抛出异常且不修改token_ids
        original_token_ids = sampling_output.token_ids.copy()
        PluginManager.filter_splitfuse_token_ids(input_metadata, sampling_output)
        np.testing.assert_array_equal(sampling_output.token_ids, original_token_ids)

    def test_to_host_with_tensor(self):
        """测试_to_host方法 - 包含tensor字段"""

        @dataclass
        class TestData:
            tensor_field: torch.Tensor
            int_field: int
            str_field: str

        tensor = torch.tensor([1.0, 2.0, 3.0])
        data = TestData(tensor_field=tensor, int_field=42, str_field="test")

        result = PluginManager._to_host(data)

        # 验证tensor被转换为numpy数组
        self.assertIsInstance(result.tensor_field, np.ndarray)
        np.testing.assert_array_equal(result.tensor_field, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result.int_field, 42)
        self.assertEqual(result.str_field, "test")

    def test_to_host_none(self):
        """测试_to_host方法 - 输入为None"""
        result = PluginManager._to_host(None)
        self.assertIsNone(result)


class TestPluginManagerMethods(unittest.TestCase):
    """测试PluginManager的实例方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_clear_cache_without_cache_ids(self):
        """测试clear_cache方法 - 无cache_ids"""
        sequence_ids = [1, 2, 3]
        self.plugin_manager.clear_cache(sequence_ids)

        self.mock_infer_context.clear_context_by_seq_ids.assert_called_once_with(sequence_ids)
        self.mock_generator_backend.sampler.clear_cache.assert_called_once_with(sequence_ids)

    def test_clear_cache_with_cache_ids(self):
        """测试clear_cache方法 - 有cache_ids"""
        sequence_ids = [1, 2, 3]
        cache_ids = [10, 20, 30]
        self.plugin_manager.clear_cache(sequence_ids, cache_ids)

        self.mock_infer_context.clear_finished_context.assert_called_once_with(sequence_ids, cache_ids)

    def test_mem_det_trigger_counter_acc(self):
        """测试mem_det_trigger_counter_acc方法"""
        # 初始值为0
        self.plugin_manager.mem_det_trigger_counter = 0

        # 累加到MEM_DETECT_INTERVAL
        for _ in range(1001):
            self.plugin_manager.mem_det_trigger_counter_acc()

        # 验证计数器被重置为0
        self.assertEqual(self.plugin_manager.mem_det_trigger_counter, 0)

    def test_model_inputs_update_manager_no_plugins(self):
        """测试model_inputs_update_manager - 无插件"""
        model_inputs = Mock()
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.all_sequence_ids = [1, 2, 3]
        input_metadata.is_prefill = True
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        sampling_metadata = Mock()
        cache_ids = [1, 2, 3]

        result = self.plugin_manager.model_inputs_update_manager(
            model_inputs, input_metadata, sampling_metadata, cache_ids
        )

        self.assertEqual(len(result), 3)

    def test_sample_preprocess_manager_no_plugins(self):
        """测试sample_preprocess_manager - 无插件"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = Mock()
        sampling_metadata = Mock()
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        output = self.plugin_manager.sample_preprocess_manager(logits, result, sampling_metadata, input_metadata)

        # 无插件时应该直接返回logits
        self.assertTrue(torch.equal(output, logits))

    def test_plugin_verify_manager_no_plugins(self):
        """测试plugin_verify_manager - 无插件"""
        sampling_output = MagicMock()
        sampling_output.token_ids = np.array([1, 2, 3])
        cache_ids = [1, 2, 3]
        result = Mock()

        # 应该正常执行不抛出异常
        self.plugin_manager.plugin_verify_manager(sampling_output, cache_ids, result)

    def test_plugin_cache_update_manager_no_plugins(self):
        """测试plugin_cache_update_manager - 无插件"""
        cache_ids = [1, 2, 3]
        sampling_output = Mock()
        la_cache_input = (Mock(), Mock())
        is_prefill = True

        # 应该正常执行不抛出异常
        self.plugin_manager.plugin_cache_update_manager(cache_ids, sampling_output, la_cache_input, is_prefill)

    def test_plugin_cache_clear_manager_no_plugins(self):
        """测试plugin_cache_clear_manager - 无插件"""
        cache_ids = [1, 2, 3]
        finish_reason = [0, 0, 0]

        # 应该正常执行不抛出异常
        self.plugin_manager.plugin_cache_clear_manager(cache_ids, finish_reason)

    def test_put_prefix_kvcache_to_mempool_no_plugin(self):
        """测试put_prefix_kvcache_to_mempool - 无prefix_cache插件"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        cache_ids = [1, 2, 3]

        # 应该正常执行不抛出异常
        self.plugin_manager.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)

    def test_put_prefix_kvcache_to_mempool_with_plugin(self):
        """测试put_prefix_kvcache_to_mempool - 有prefix_cache插件"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        cache_ids = [1, 2, 3]

        # 添加prefix_cache插件mock
        mock_prefix_cache = Mock()
        mock_prefix_cache.put_prefix_kvcache_to_mempool = Mock()
        self.plugin_manager.prefix_cache = mock_prefix_cache
        self.plugin_manager.plugin_list = ["prefix_cache"]

        self.plugin_manager.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)

        mock_prefix_cache.put_prefix_kvcache_to_mempool.assert_called_once_with(input_metadata, cache_ids)


class TestPluginManagerPlugins(unittest.TestCase):
    """测试PluginManager的插件相关功能"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=["test_plugin"],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_model_inputs_update_manager_with_plugin(self):
        """测试model_inputs_update_manager - 有插件"""
        mock_plugin = Mock()
        mock_plugin.model_inputs_update = Mock(return_value=(Mock(), (Mock(), Mock())))
        self.plugin_manager.test_plugin = mock_plugin

        model_inputs = Mock()
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.all_sequence_ids = [1, 2, 3]
        input_metadata.is_prefill = True
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        sampling_metadata = Mock()
        cache_ids = [1, 2, 3]

        self.plugin_manager.model_inputs_update_manager(model_inputs, input_metadata, sampling_metadata, cache_ids)

        mock_plugin.model_inputs_update.assert_called_once()

    def test_sample_preprocess_manager_with_plugin(self):
        """测试sample_preprocess_manager - 有插件"""
        mock_plugin = Mock()
        mock_plugin.sample_preprocess = Mock(return_value=torch.tensor([[1.0, 2.0]]))
        self.plugin_manager.test_plugin = mock_plugin

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = Mock()
        sampling_metadata = Mock()
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        self.plugin_manager.sample_preprocess_manager(logits, result, sampling_metadata, input_metadata)

        mock_plugin.sample_preprocess.assert_called_once()

    def test_plugin_verify_manager_with_plugin(self):
        """测试plugin_verify_manager - 有插件"""
        mock_plugin = Mock()
        mock_plugin.plugin_verify = Mock()
        self.plugin_manager.test_plugin = mock_plugin

        sampling_output = MagicMock()
        sampling_output.token_ids = np.array([[1], [2], [3]])
        cache_ids = [1, 2, 3]
        result = Mock()

        self.plugin_manager.plugin_verify_manager(sampling_output, cache_ids, result)

        mock_plugin.plugin_verify.assert_called_once()

    def test_plugin_cache_update_manager_with_plugin(self):
        """测试plugin_cache_update_manager - 有插件"""
        mock_plugin = Mock()
        mock_plugin.plugin_cache_update = Mock()
        self.plugin_manager.test_plugin = mock_plugin

        cache_ids = [1, 2, 3]
        sampling_output = Mock()
        la_cache_input = (Mock(), Mock())
        is_prefill = True

        self.plugin_manager.plugin_cache_update_manager(cache_ids, sampling_output, la_cache_input, is_prefill)

        mock_plugin.plugin_cache_update.assert_called_once()

    def test_plugin_cache_clear_manager_with_plugin(self):
        """测试plugin_cache_clear_manager - 有插件"""
        mock_plugin = Mock()
        mock_plugin.plugin_cache_clear = Mock()
        self.plugin_manager.test_plugin = mock_plugin

        cache_ids = [1, 2, 3]
        finish_reason = [0, 0, 0]

        self.plugin_manager.plugin_cache_clear_manager(cache_ids, finish_reason)

        mock_plugin.plugin_cache_clear.assert_called_once()


class TestPluginManagerFillInModelResult(unittest.TestCase):
    """测试_fill_in_model_result方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0
        self.mock_generator_backend.to_tensor = Mock(side_effect=lambda x: torch.tensor(x))
        self.mock_generator_backend.mapping = Mock()
        self.mock_generator_backend.mapping.has_attn_cp = Mock(return_value=False)

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_fill_in_model_result_no_hit_mask(self):
        """测试_fill_in_model_result - 无hit_sequence_ids_mask"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        model_input_wrapper = Mock()
        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.position_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.input_lengths = Mock()
        model_input_wrapper.model_inputs.context_length = np.array([1, 2, 3])
        model_input_wrapper.model_inputs.max_seq_len = 3
        model_input_wrapper.model_kwargs = {}

        model_output_wrapper = Mock()
        model_output_wrapper.sampling_output = Mock()
        model_output_wrapper.sampling_output.token_ids = np.array([10, 20, 30])

        filling_masks = {}
        cache_ids = [1, 2, 3]

        # 应该正常执行不抛出异常
        self.plugin_manager._fill_in_model_result(
            input_metadata,
            model_input_wrapper,
            model_output_wrapper,
            filling_masks,
            cache_ids,
        )

    def test_fill_in_model_result_with_hit_mask(self):
        """测试_fill_in_model_result - 有hit_sequence_ids_mask"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        model_input_wrapper = Mock()
        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.position_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.input_lengths = torch.tensor([1, 1, 1])
        model_input_wrapper.model_inputs.context_length = np.array([1, 2, 3])
        model_input_wrapper.model_inputs.max_seq_len = 3
        model_input_wrapper.model_kwargs = {}

        model_output_wrapper = Mock()
        model_output_wrapper.sampling_output = Mock()
        # token_ids需要足够长以支持hit_indices索引
        model_output_wrapper.sampling_output.token_ids = np.array([10, 20, 30])

        filling_masks = {
            "hit_sequence_ids_mask": np.array([True, False, True]),
            "hit_indices": np.array([0, 2]),  # 索引0和2，需要token_ids长度至少为3
            "hit_sequence_ids_mask_tensor": torch.tensor([True, False, True]),
        }
        cache_ids = [1, 2, 3]

        # 应该正常执行不抛出异常
        self.plugin_manager._fill_in_model_result(
            input_metadata,
            model_input_wrapper,
            model_output_wrapper,
            filling_masks,
            cache_ids,
        )

    def test_fill_in_model_result_with_hit_mask_per_token(self):
        """测试_fill_in_model_result - 有hit_mask_per_token"""
        self.plugin_manager.is_mix_model = True

        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        model_input_wrapper = Mock()
        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.position_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.input_lengths = torch.tensor([1, 1, 1])
        model_input_wrapper.model_inputs.context_length = np.array([1, 2, 3])
        model_input_wrapper.model_inputs.max_seq_len = 3
        model_input_wrapper.model_kwargs = {}

        model_output_wrapper = Mock()
        model_output_wrapper.sampling_output = Mock()
        model_output_wrapper.sampling_output.token_ids = np.array([10, 20, 30])

        filling_masks = {
            "hit_sequence_ids_mask": np.array([True, False, True]),
            "hit_indices": np.array([0, 2]),
            "hit_sequence_ids_mask_tensor": torch.tensor([True, False, True]),
            "hit_mask_per_token": torch.tensor([True, False, True]),
        }
        cache_ids = [1, 2, 3]

        # 应该正常执行不抛出异常
        self.plugin_manager._fill_in_model_result(
            input_metadata,
            model_input_wrapper,
            model_output_wrapper,
            filling_masks,
            cache_ids,
        )


class TestPluginManagerPrepareMasks(unittest.TestCase):
    """测试_prepare_masks_for_filling方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0
        self.mock_generator_backend.to_tensor = Mock(side_effect=lambda x: torch.tensor(x))

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )
        self.plugin_manager.last_sequence_ids = np.array([1, 2, 3])

    def test_prepare_masks_for_filling_no_hit(self):
        """测试_prepare_masks_for_filling - 无命中"""
        model_inputs = Mock()
        current_dp_sequence_ids = np.array([4, 5, 6])  # 与last_sequence_ids无交集
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.all_sequence_ids = np.array([4, 5, 6])
        input_metadata.batch_is_prefill = None
        input_metadata.is_prefill = True

        result = self.plugin_manager._prepare_masks_for_filling(model_inputs, current_dp_sequence_ids, input_metadata)

        # 无命中时应该返回空字典或只有基本mask
        self.assertIsInstance(result, dict)

    def test_prepare_masks_for_filling_with_hit(self):
        """测试_prepare_masks_for_filling - 有命中"""
        model_inputs = Mock()
        current_dp_sequence_ids = np.array([1, 4, 5])  # 1在last_sequence_ids中
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.all_sequence_ids = np.array([1, 4, 5])
        input_metadata.batch_is_prefill = None
        input_metadata.is_prefill = True

        result = self.plugin_manager._prepare_masks_for_filling(model_inputs, current_dp_sequence_ids, input_metadata)

        self.assertIsInstance(result, dict)
        if "hit_sequence_ids_mask" in result:
            self.assertTrue(result["hit_sequence_ids_mask"][0])  # 第一个元素应该命中

    def test_prepare_masks_for_filling_dp_filtered_prefill_mask(self):
        """测试_prepare_masks_for_filling - DP过滤后的prefill mask不会越界"""
        model_inputs = Mock()
        current_dp_sequence_ids = np.array([], dtype=np.int64)
        input_metadata = Mock()
        input_metadata.batch_is_prefill = np.array([True])
        input_metadata.batch_last_prompt = None
        input_metadata.all_sequence_ids = np.array([1], dtype=np.int64)
        input_metadata.is_prefill = True

        result = self.plugin_manager._prepare_masks_for_filling(
            model_inputs,
            current_dp_sequence_ids,
            input_metadata,
            np.array([], dtype=bool),
        )

        self.assertEqual(result, {})

    def test_prepare_masks_for_filling_mix_model_dp_filtered_token_num(self):
        """测试_prepare_masks_for_filling - mix模型使用DP过滤后的token_num_per_seq"""
        self.plugin_manager.is_mix_model = True
        model_inputs = Mock()
        current_dp_sequence_ids = np.array([2, 3], dtype=np.int64)
        input_metadata = Mock()
        input_metadata.batch_is_prefill = np.array([False, False, False, False])
        input_metadata.batch_last_prompt = None
        input_metadata.all_sequence_ids = np.array([1, 2, 3, 4], dtype=np.int64)
        input_metadata.is_prefill = False

        result = self.plugin_manager._prepare_masks_for_filling(
            model_inputs,
            current_dp_sequence_ids,
            input_metadata,
            np.array([False, False]),
            np.array([1, 1], dtype=np.int64),
        )

        np.testing.assert_array_equal(result["hit_mask_per_token"].cpu().numpy(), np.array([True, True]))


class TestPluginManagerGetTokenNumPerSeq(unittest.TestCase):
    """测试_get_token_num_per_seq方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0
        self.mock_generator_backend.block_size = 128

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_get_token_num_per_seq_no_computed_blocks(self):
        """测试_get_token_num_per_seq - 无computed_blocks"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.split_end_position = np.array([100, 200, 300])
        input_metadata.split_start_position = np.array([0, 0, 0])
        input_metadata.computed_blocks = None

        result = self.plugin_manager._get_token_num_per_seq(input_metadata)

        expected = np.array([100, 200, 300])
        np.testing.assert_array_equal(result, expected)

    def test_get_token_num_per_seq_with_computed_blocks(self):
        """测试_get_token_num_per_seq - 有computed_blocks"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.split_end_position = np.array([300, 400, 500])
        input_metadata.split_start_position = np.array([0, 0, 100])
        input_metadata.computed_blocks = np.array([1, 2, 0])
        input_metadata.batch_is_prefill = np.array([True, True, False])

        result = self.plugin_manager._get_token_num_per_seq(input_metadata)

        # 验证计算逻辑
        self.assertEqual(len(result), 3)


class TestPluginManagerStructuredOutput(unittest.TestCase):
    """测试结构化输出相关方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0
        self.mock_generator_backend.tokenizer = Mock()
        self.mock_generator_backend.tokenizer.vocab_size = 10000

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
            enable_structured_output=False,  # 禁用以避免导入问题
        )

    def test_init_structured_output_manager_disabled(self):
        """测试_init_structured_output_manager - 禁用状态"""
        # enable_structured_output=False时应该不初始化
        self.assertIsNone(self.plugin_manager._structured_output_manager)
        self.assertFalse(self.plugin_manager._structured_output_enabled)


class TestPluginManagerGenerateToken(unittest.TestCase):
    """测试generate_token方法的各种路径"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    @patch("mindie_llm.text_generator.plugins.plugin_manager.ENV")
    def test_generate_token_with_tuple_result(self, mock_env):
        """测试generate_token - 返回tuple的情况"""
        mock_env.model_runner_exp = False

        input_metadata = Mock()
        input_metadata.is_dummy_batch = False
        input_metadata.all_sequence_ids = np.array([1, 2, 3])
        input_metadata.block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )
        input_metadata.is_prefill = True
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        # Mock preprocess
        mock_model_inputs_from_preprocess = Mock()
        mock_model_inputs_from_preprocess.block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )
        self.plugin_manager.preprocess = Mock(
            return_value=(
                [1, 2, 3],
                mock_model_inputs_from_preprocess,
                Mock(),
                [100, 200, 300],
            )
        )

        # Mock model_inputs_update_manager
        mock_model_inputs = Mock()
        mock_model_inputs.context_length = np.array([1, 1, 1])
        mock_model_inputs.input_ids = torch.zeros(3, 10, dtype=torch.long)
        mock_model_inputs.position_ids = torch.zeros(3, 10, dtype=torch.long)
        mock_model_inputs.input_lengths = torch.tensor([1, 1, 1])
        mock_model_inputs.max_seq_len = 3
        mock_model_inputs.block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )

        self.plugin_manager.model_inputs_update_manager = Mock(return_value=(mock_model_inputs, None, None))

        # Mock forward返回tuple
        self.mock_generator_backend.forward = Mock(return_value=(torch.tensor([[0.1, 0.2, 0.3]]), None))

        # Mock sample
        mock_sampling_output = Mock()
        mock_sampling_output.token_ids = np.array([[1], [2], [3]])
        mock_sampling_output.logprobs = np.array([[0.1], [0.2], [0.3]])
        mock_sampling_output.top_token_ids = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
        mock_sampling_output.top_logprobs = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]])
        mock_sampling_output.num_new_tokens = np.array([1, 1, 1])
        mock_sampling_output.num_top_tokens = np.array([2, 2, 2])
        mock_sampling_output.cumulative_logprobs = np.array([0.1, 0.2, 0.3])
        mock_sampling_output.finish_reason = np.array([0, 0, 0])
        mock_sampling_output.sequence_ids = np.array([1, 2, 3])
        mock_sampling_output.parent_sequence_ids = np.array([1, 2, 3])
        mock_sampling_output.group_indices = None
        self.mock_generator_backend.sample = Mock(return_value=mock_sampling_output)

        # Mock postprocess相关
        self.plugin_manager.sample_preprocess_manager = Mock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))
        self.plugin_manager.put_prefix_kvcache_to_mempool = Mock()

        # Mock output_filter
        self.mock_output_filter.filter_finished_sequences = Mock(
            return_value=(
                np.array([0, 0, 0]),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        )

        # Mock infer_context
        self.mock_infer_context.update_context = Mock(
            return_value=(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        )
        self.mock_infer_context.clear_finished_context = Mock(return_value=np.array([], dtype=np.int64))
        self.mock_infer_context.clear_aborted_context = Mock()
        self.mock_infer_context.get_output_len_count = Mock(return_value=np.array([1, 1, 1]))

        # 执行测试
        result = self.plugin_manager.generate_token(input_metadata)

        # 验证结果
        self.assertIsNotNone(result)
        self.mock_generator_backend.forward.assert_called_once()
        self.mock_generator_backend.sample.assert_called_once()

    @patch("mindie_llm.text_generator.plugins.plugin_manager.ENV")
    def test_generate_token_with_non_tuple_result(self, mock_env):
        """测试generate_token - 返回非tuple的情况"""
        mock_env.model_runner_exp = False

        input_metadata = Mock()
        input_metadata.is_dummy_batch = False
        input_metadata.all_sequence_ids = np.array([1, 2, 3])
        input_metadata.block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )
        input_metadata.is_prefill = True
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        # Mock preprocess
        mock_model_inputs_from_preprocess = Mock()
        mock_model_inputs_from_preprocess.block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )
        self.plugin_manager.preprocess = Mock(
            return_value=(
                [1, 2, 3],
                mock_model_inputs_from_preprocess,
                Mock(),
                [100, 200, 300],
            )
        )

        # Mock model_inputs_update_manager
        mock_model_inputs = Mock()
        mock_model_inputs.context_length = np.array([1, 1, 1])
        mock_model_inputs.input_ids = torch.zeros(3, 10, dtype=torch.long)
        mock_model_inputs.position_ids = torch.zeros(3, 10, dtype=torch.long)
        mock_model_inputs.input_lengths = torch.tensor([1, 1, 1])
        mock_model_inputs.max_seq_len = 3
        mock_model_inputs.block_tables = np.array(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        )
        self.plugin_manager.model_inputs_update_manager = Mock(return_value=(mock_model_inputs, None, None))

        # Mock forward返回非tuple
        self.mock_generator_backend.forward = Mock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))

        # Mock sample
        mock_sampling_output = Mock()
        mock_sampling_output.token_ids = np.array([[1], [2], [3]])
        mock_sampling_output.logprobs = np.array([[0.1], [0.2], [0.3]])
        mock_sampling_output.top_token_ids = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
        mock_sampling_output.top_logprobs = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]])
        mock_sampling_output.num_new_tokens = np.array([1, 1, 1])
        mock_sampling_output.num_top_tokens = np.array([2, 2, 2])
        mock_sampling_output.cumulative_logprobs = np.array([0.1, 0.2, 0.3])
        mock_sampling_output.finish_reason = np.array([0, 0, 0])
        mock_sampling_output.sequence_ids = np.array([1, 2, 3])
        mock_sampling_output.parent_sequence_ids = np.array([1, 2, 3])
        mock_sampling_output.group_indices = None
        self.mock_generator_backend.sample = Mock(return_value=mock_sampling_output)

        # Mock postprocess相关
        self.plugin_manager.sample_preprocess_manager = Mock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))
        self.plugin_manager.put_prefix_kvcache_to_mempool = Mock()

        # Mock output_filter
        self.mock_output_filter.filter_finished_sequences = Mock(
            return_value=(
                np.array([0, 0, 0]),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        )

        # Mock infer_context
        self.mock_infer_context.update_context = Mock(
            return_value=(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        )
        self.mock_infer_context.clear_finished_context = Mock(return_value=np.array([], dtype=np.int64))
        self.mock_infer_context.clear_aborted_context = Mock()
        self.mock_infer_context.get_output_len_count = Mock(return_value=np.array([1, 1, 1]))

        # 执行测试
        result = self.plugin_manager.generate_token(input_metadata)

        # 验证结果
        self.assertIsNotNone(result)


class TestPluginManagerPreprocess(unittest.TestCase):
    """测试preprocess方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_preprocess_non_mix_model(self):
        """测试preprocess - 非mix_model情况"""
        input_metadata = Mock()
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None
        input_metadata.batch_response_format = None

        # Mock infer_context
        mock_model_inputs = Mock()
        mock_sampling_metadata = Mock()
        self.mock_infer_context.get_batch_context_handles = Mock(return_value=[1, 2, 3])
        self.mock_infer_context.compose_model_inputs = Mock(
            return_value=(mock_model_inputs, mock_sampling_metadata, [100, 200, 300])
        )

        result = self.plugin_manager.preprocess(input_metadata)

        self.assertEqual(len(result), 4)
        self.mock_infer_context.compose_model_inputs.assert_called_once()


class TestPluginManagerPostprocess(unittest.TestCase):
    """测试postprocess方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_postprocess_with_sampling_metadata(self):
        """测试postprocess - 有sampling_metadata的情况"""
        cache_ids = [1, 2, 3]
        input_metadata = Mock()
        input_metadata.is_dummy_batch = False
        input_metadata.all_sequence_ids = np.array([1, 2, 3])
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        # Mock result
        result = torch.tensor([[0.1, 0.2, 0.3]])

        # Mock sampling_metadata
        sampling_metadata = Mock()
        sampling_metadata.best_of_array = None
        sampling_metadata.is_prefill = True
        sampling_metadata.use_beam_search_array = None
        sampling_metadata.all_sequence_ids = np.array([1, 2, 3])

        # Mock sampling_output
        sampling_output = Mock()
        sampling_output.token_ids = np.array([[1], [2], [3]])
        sampling_output.logprobs = np.array([[0.1], [0.2], [0.3]])
        sampling_output.top_token_ids = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
        sampling_output.top_logprobs = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]])
        sampling_output.num_new_tokens = np.array([1, 1, 1])
        sampling_output.num_top_tokens = np.array([2, 2, 2])
        sampling_output.cumulative_logprobs = np.array([0.1, 0.2, 0.3])
        sampling_output.finish_reason = np.array([0, 0, 0])
        sampling_output.sequence_ids = np.array([1, 2, 3])
        sampling_output.parent_sequence_ids = np.array([1, 2, 3])
        sampling_output.group_indices = None

        # Mock output_filter
        self.mock_output_filter.filter_finished_sequences = Mock(
            return_value=(
                np.array([0, 0, 0]),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        )

        # Mock infer_context
        self.mock_infer_context.update_context = Mock(
            return_value=(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        )
        self.mock_infer_context.clear_finished_context = Mock(return_value=np.array([], dtype=np.int64))
        self.mock_infer_context.clear_aborted_context = Mock()
        self.mock_infer_context.get_output_len_count = Mock(return_value=np.array([1, 1, 1]))

        # Mock plugin方法
        self.plugin_manager.plugin_cache_update_manager = Mock()
        self.plugin_manager.plugin_cache_clear_manager = Mock()
        self.plugin_manager.filter_splitfuse_token_ids = Mock()

        result_output = self.plugin_manager.postprocess(
            cache_ids, input_metadata, result, sampling_metadata, sampling_output
        )

        self.assertIsNotNone(result_output)

    def test_postprocess_without_sampling_metadata(self):
        """测试postprocess - 无sampling_metadata的情况"""
        cache_ids = [1, 2, 3]
        input_metadata = Mock()
        input_metadata.is_dummy_batch = False
        input_metadata.all_sequence_ids = np.array([1, 2, 3])
        input_metadata.batch_is_prefill = None
        input_metadata.batch_last_prompt = None

        # Mock result
        result = torch.tensor([[0.1, 0.2, 0.3]])

        # sampling_metadata为None
        sampling_metadata = None

        # Mock sampling_output
        sampling_output = Mock()
        sampling_output.token_ids = np.array([[1], [2], [3]])
        sampling_output.logprobs = np.array([[0.1], [0.2], [0.3]])
        sampling_output.top_token_ids = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
        sampling_output.top_logprobs = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]])
        sampling_output.num_new_tokens = np.array([1, 1, 1])
        sampling_output.num_top_tokens = np.array([2, 2, 2])
        sampling_output.cumulative_logprobs = np.array([0.1, 0.2, 0.3])
        sampling_output.finish_reason = np.array([0, 0, 0])
        sampling_output.sequence_ids = np.array([1, 2, 3])
        sampling_output.parent_sequence_ids = np.array([1, 2, 3])
        sampling_output.group_indices = None

        # Mock output_filter
        self.mock_output_filter.filter_finished_sequences = Mock(
            return_value=(
                np.array([0, 0, 0]),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        )

        # Mock infer_context
        self.mock_infer_context.update_context = Mock(
            return_value=(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        )
        self.mock_infer_context.clear_finished_context = Mock(return_value=np.array([], dtype=np.int64))
        self.mock_infer_context.clear_aborted_context = Mock()
        self.mock_infer_context.get_output_len_count = Mock(return_value=np.array([1, 1, 1]))

        # Mock plugin方法
        self.plugin_manager.plugin_cache_update_manager = Mock()
        self.plugin_manager.plugin_cache_clear_manager = Mock()
        self.plugin_manager.filter_splitfuse_token_ids = Mock()

        result_output = self.plugin_manager.postprocess(
            cache_ids, input_metadata, result, sampling_metadata, sampling_output
        )

        self.assertIsNotNone(result_output)


class TestPluginManagerAsyncInference(unittest.TestCase):
    """测试异步推理相关功能"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0
        self.mock_generator_backend.to_tensor = Mock(side_effect=lambda x: torch.tensor(x))
        self.mock_generator_backend.mapping = Mock()
        self.mock_generator_backend.mapping.has_attn_cp = Mock(return_value=False)

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = True  # 启用异步推理
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_init_with_async_inference(self):
        """测试异步推理初始化"""
        # 验证队列和线程已创建
        self.assertIsNotNone(self.plugin_manager.input_queue)
        self.assertIsNotNone(self.plugin_manager.output_queue)
        self.assertIsNotNone(self.plugin_manager.forward_thread)


class TestPluginManagerFillInModelResultExp(unittest.TestCase):
    """测试_fill_in_model_result_exp方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_fill_in_model_result_exp_no_hit_mask(self):
        """测试_fill_in_model_result_exp - 无hit_sequence_ids_mask"""
        model_input_wrapper = Mock()
        model_input_wrapper.filling_masks = {}
        model_input_wrapper.model_inputs = Mock()

        model_output_wrapper = Mock()
        model_output_wrapper.sampling_output = Mock()

        # 应该正常执行不抛出异常
        self.plugin_manager._fill_in_model_result_exp(model_input_wrapper, model_output_wrapper)

    def test_fill_in_model_result_exp_with_empty_update_indices(self):
        """测试_fill_in_model_result_exp - update_indices为空"""
        model_input_wrapper = Mock()
        model_input_wrapper.filling_masks = {
            "hit_sequence_ids_mask": np.array([True, False, True]),
            "hit_indices_tensor": torch.tensor([0, 2]),
            "update_indices": torch.tensor([], dtype=torch.long),
            "ones_int32": torch.tensor([], dtype=torch.int32),
            "ones_int64": torch.tensor([], dtype=torch.int64),
        }

        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.position_ids = torch.zeros(3, dtype=torch.long)
        model_input_wrapper.model_inputs.input_lengths = torch.tensor([1, 1, 1])
        model_input_wrapper.model_inputs.context_length = np.array([1, 2, 3], dtype=np.int32)
        model_input_wrapper.model_inputs.max_seq_len = 3
        model_input_wrapper.model_inputs.forward_context = Mock()
        model_input_wrapper.model_inputs.forward_context.attn_metadata = Mock()
        model_input_wrapper.model_inputs.forward_context.attn_metadata.max_seq_len = 3

        model_output_wrapper = Mock()
        model_output_wrapper.sampling_output = Mock()
        model_output_wrapper.sampling_output.token_ids = Mock()
        model_output_wrapper.sampling_output.token_ids.index_select = Mock(return_value=Mock())
        model_output_wrapper.sampling_output.token_ids.index_select.return_value.flatten = Mock(
            return_value=torch.tensor([10, 30])
        )

        # 应该正常执行不抛出异常
        self.plugin_manager._fill_in_model_result_exp(model_input_wrapper, model_output_wrapper)

    def test_fill_in_model_result_exp_with_hit_mask_per_token(self):
        """测试_fill_in_model_result_exp - splitfuse/mix 按 token 回填（与 _fill_in_model_result 一致）"""
        self.mock_generator_backend.mapping = Mock()
        self.mock_generator_backend.mapping.has_attn_cp = Mock(return_value=False)

        model_input_wrapper = Mock()
        model_input_wrapper.filling_masks = {
            "hit_sequence_ids_mask": np.array([True, False]),
            "hit_indices_tensor": torch.tensor([0, 1]),
            "hit_mask_per_token": torch.tensor([True, False, True, False], dtype=torch.bool),
            "hit_sequence_ids_mask_tensor": torch.tensor([True, False], dtype=torch.bool),
        }

        model_input_wrapper.model_inputs = Mock()
        model_input_wrapper.model_inputs.input_ids = torch.zeros(4, dtype=torch.long)
        model_input_wrapper.model_inputs.position_ids = torch.zeros(4, dtype=torch.long)
        model_input_wrapper.model_inputs.input_lengths = torch.tensor([1, 1], dtype=torch.long)
        model_input_wrapper.model_inputs.context_length = np.array([1, 1], dtype=np.int32)
        model_input_wrapper.model_inputs.max_seq_len = 1
        model_input_wrapper.model_inputs.forward_context = Mock()
        model_input_wrapper.model_inputs.forward_context.attn_metadata = Mock()
        model_input_wrapper.model_inputs.forward_context.attn_metadata.max_seq_len = 1

        model_output_wrapper = Mock()
        model_output_wrapper.sampling_output = Mock()
        model_output_wrapper.sampling_output.token_ids = torch.tensor([[7], [8]], dtype=torch.long)

        self.plugin_manager._fill_in_model_result_exp(model_input_wrapper, model_output_wrapper)

        torch.testing.assert_close(
            model_input_wrapper.model_inputs.input_ids,
            torch.tensor([7, 0, 8, 0], dtype=torch.long),
        )
        torch.testing.assert_close(
            model_input_wrapper.model_inputs.position_ids,
            torch.tensor([1, 0, 1, 0], dtype=torch.long),
        )
        torch.testing.assert_close(
            model_input_wrapper.model_inputs.input_lengths,
            torch.tensor([2, 1], dtype=torch.long),
        )
        np.testing.assert_array_equal(
            model_input_wrapper.model_inputs.context_length,
            np.array([2, 1], dtype=np.int32),
        )


class TestPluginManagerClearCache(unittest.TestCase):
    """测试clear_cache方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    def test_clear_cache_with_has_sampling_false(self):
        """测试clear_cache - has_sampling为False"""
        sequence_ids = [1, 2, 3]
        cache_ids = [10, 20, 30]

        self.plugin_manager.clear_cache(sequence_ids, cache_ids, has_sampling=False)

        # 验证只清理infer_context，不清理sampler
        self.mock_infer_context.clear_finished_context.assert_called_once_with(sequence_ids, cache_ids)
        self.mock_generator_backend.sampler.clear_cache.assert_not_called()


class TestPluginManagerInitialize(unittest.TestCase):
    """测试initialize方法"""

    def setUp(self):
        self.mock_generator_backend = Mock()
        self.mock_generator_backend.model_wrapper = Mock()
        self.mock_generator_backend.sampler = Mock()
        self.mock_generator_backend.rank = 0

        self.mock_kvcache_settings = Mock()
        self.mock_infer_context = Mock()
        self.mock_infer_context.context_params.async_infer = False
        self.mock_infer_context.context_params.max_generated_tokens = 100
        self.mock_output_filter = Mock()
        self.mock_watcher = Mock()

        self.plugin_manager = PluginManager(
            generator_backend=self.mock_generator_backend,
            kvcache_settings=self.mock_kvcache_settings,
            infer_context=self.mock_infer_context,
            output_filter=self.mock_output_filter,
            is_mix_model=False,
            plugin_list=[],
            model_role="master",
            watcher=self.mock_watcher,
        )

    @patch("mindie_llm.text_generator.plugins.plugin_manager.importlib")
    def test_initialize_with_plugins(self, mock_importlib):
        """测试initialize - 有插件的情况"""
        self.plugin_manager.plugin_list = ["test_plugin"]
        self.plugin_manager.is_mix_model = False

        # Mock importlib
        mock_module = Mock()
        mock_plugin_class = Mock()
        mock_plugin_instance = Mock()
        mock_plugin_class.return_value = mock_plugin_instance
        mock_module.TestPluginPlugin = mock_plugin_class
        mock_importlib.import_module.return_value = mock_module

        # Mock _init_structured_output_manager
        self.plugin_manager._init_structured_output_manager = Mock()

        self.plugin_manager.initialize()

        # 验证插件被正确加载
        mock_importlib.import_module.assert_called_once_with(
            "mindie_llm.text_generator.plugins.test_plugin.test_plugin_plugin"
        )
        self.assertEqual(self.plugin_manager.test_plugin, mock_plugin_instance)


if __name__ == "__main__":
    unittest.main()
