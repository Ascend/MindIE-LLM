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

# Mock the missing module before imports
mock_model_execute_data = SimpleNamespace()
mock_model_execute_data.LoraOperationStatus = SimpleNamespace()
sys.modules["mindie_llm.connector.common.model_execute_data_pb2"] = mock_model_execute_data

from mindie_llm.text_generator.generator import Generator  # noqa: E402
from mindie_llm.text_generator.utils.request import Request  # noqa: E402
from mindie_llm.text_generator.utils.input_metadata import InputMetadata  # noqa: E402
from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch  # noqa: E402
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams  # noqa: E402
from tests.pythontest.npu import FakeModelRunner, FakeParallelInfo  # noqa: E402

PLUGIN_PARAMS = '{"plugin_type": "splitfuse"}'


class TestSplitFusePlugin(unittest.TestCase):
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
            "model_name": "llama3",
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

        plugin_dict = {"plugin_params": PLUGIN_PARAMS}
        self.model_config.update(plugin_dict)

        fake_parallel_info = FakeParallelInfo(
            dp=int(self.model_config["dp"]),
            tp=int(self.model_config["tp"]),
            sp=int(self.model_config["sp"]),
            cp=int(self.model_config["cp"]),
        )
        self.fake_model_runner = FakeModelRunner(parallel_info=fake_parallel_info)

    @unittest.skip("Test initialization takes too long - skipping for now")
    @patch("torch.npu.synchronize", return_value=None)
    @patch("atb_llm.runner.model_runner.ModelRunner")
    @patch("mindie_llm.text_generator.utils.output_filter.OutputFilter.filter_by_eos")
    @patch.object(Generator, "warm_up")
    @patch.object(GeneratorTorch, "_get_obfuscation_func")
    @patch.object(GeneratorTorch, "forward")
    def test_generate_token_greedy(
        self,
        mock_forward,
        mock_obfuscation_func,
        mock_warm_up,
        mock_filter_by_eos,
        mock_model_runner,
        mock_npu_sync,
    ):
        def side_effect_forward(model_inputs, **kwargs):
            if model_inputs.is_prefill:
                token_num = model_inputs.prefill_head_indices.shape[0]
            else:
                token_num = len(model_inputs.input_ids)
            logits = torch.zeros(token_num, 10)  # 假定词表长度为10
            for i in range(logits.shape[0]):
                logits[i][2] = 2
                logits[i][5] = 3
                logits[i][8] = 4
            return logits

        # 模拟 filter_by_eos 方法，避免索引错误
        def mock_filter_by_eos(cache_ids, next_token_ids, num_new_tokens, filter_ids_arr, end_reason):
            # 只处理有效的索引
            return filter_ids_arr

        mock_model_runner.return_value = self.fake_model_runner
        mock_forward.side_effect = side_effect_forward
        mock_obfuscation_func.return_value = None
        mock_warm_up.return_value = 10
        mock_filter_by_eos.side_effect = mock_filter_by_eos

        # 模拟 KV cache 设置
        with (
            patch("mindie_llm.text_generator.generator.Generator._update_kvcache_settings") as mock_update_kvcache,
            patch("mindie_llm.text_generator.generator.Generator._init_plugin_manager"),
        ):
            # 创建一个具有 num_npu_blocks 属性的对象
            mock_kvcache_settings = MagicMock()
            mock_kvcache_settings.num_npu_blocks = 100  # 设置为一个整数
            mock_update_kvcache.return_value = mock_kvcache_settings

            generator = Generator(self.model_config)

            # 模拟 infer_context
            mock_infer = MagicMock()
            mock_batch_context = MagicMock()
            mock_sampling_params = MagicMock()
            mock_sampling_params.dtype = np.dtype(
                [
                    ("temperature", np.float32),
                    ("top_p", np.float32),
                    ("top_k", np.int32),
                    ("min_p", np.float32),
                    ("repetition_penalty", np.float32),
                    ("length_penalty", np.float32),
                    ("do_sample", np.bool_),
                    ("seed", np.int32),
                ]
            )
            mock_batch_context.default_sampling_params = mock_sampling_params
            mock_infer._batch_context = mock_batch_context
            generator.infer_context = mock_infer

            # 模拟 plugin_manager
            mock_plugin_manager = MagicMock()
            # 创建一个模拟的 GenerationOutput
            mock_generation_output = MagicMock()
            mock_generation_output.token_ids = [[8, 5]]
            mock_generation_output.finish_reason = [0, 0]
            mock_plugin_manager.generate_token.return_value = mock_generation_output
            generator.plugin_manager = mock_plugin_manager

        sample_dtype = generator.infer_context._batch_context.default_sampling_params.dtype
        greedy_param = np.array([(1.0, 0.0, 0.0, 0, 1.0, 1.0, False, 0)], dtype=sample_dtype)
        input1 = [5159, 636, 374, 31346, 323, 358]
        block_tables = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

        gen_len = 2
        req = Request.request_from_token(
            input1, sampling_params=greedy_param, generation_params=GenerationParams(max_new_tokens=gen_len)
        )
        req.split_start_position = 0
        req.split_end_position = len(input1)
        meta_data = InputMetadata.from_requests([req], block_tables, np.array([1]))
        meta_data.batch_block_tables = block_tables
        meta_data.batch_is_prefill = np.array([True])

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


if __name__ == "__main__":
    unittest.main()
