# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# 注意：atb_llm 相关的导入语句位于文件中间（第194-197行），
# 因为需要先在下方创建并设置模拟模块，然后才能导入
import os
import sys
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
from unittest.mock import MagicMock
import torch

# Mock Level.DETAILED
from mindie_llm.utils.prof.profiler import Level

if not hasattr(Level, "DETAILED"):
    setattr(Level, "DETAILED", Level.INFO)


# Mock atb_llm module to avoid import errors
class MockAttentionMask:
    @staticmethod
    def static(max_seq_len, dtype=torch.float16):
        return MagicMock()


def mock_safe_open(file_path, mode="r", **kwargs):
    return open(file_path, mode, **kwargs)


class MockLlamaConfig:
    @classmethod
    def from_dict(cls, config_dict):
        config = MagicMock()
        config.num_hidden_layers = config_dict.get("num_hidden_layers", 0)
        config.num_key_value_heads = config_dict.get("num_key_value_heads", 0)
        config.hidden_size = config_dict.get("hidden_size", 0)
        config.max_position_embeddings = config_dict.get("max_position_embeddings", 0)
        config.eos_token_id = config_dict.get("eos_token_id", 2)
        config.top_k = config_dict.get("top_k", 1000)
        config.vocab_size = config_dict.get("vocab_size", 100000)
        return config


# Mock ModelRunner class
class MockModelRunner:
    def __init__(self):
        # Add model attribute with eplb_level
        self.model = MagicMock()
        self.model.eplb_level = 0


def mock_generate_mem_pool_event_key():
    return "mock_event_key"


# Mock ENV class
class MockENV:
    speed_mode_type = 0
    enable_expert_hotpot_gather = False
    enable_greedy_search_opt = False


# Mock EplbExpertDataCollect class
class MockEplbExpertDataCollect:
    def get_topk(self):
        return []

    def get_decode_token_num_per_expert(self):
        return []

    decode_forward_count = 0
    prefill_forward_count = 0


# Mock moe_utils module
class MockEPLBType:
    DYNAMIC_EPLB = 0


def mock_save_eplb_data(*args, **kwargs):
    pass


# Create mock atb_llm module
mock_atb_llm = MagicMock()
mock_atb_llm.utils.layers.attention.attention_mask.AttentionMask = MockAttentionMask
mock_atb_llm.utils.file_utils.safe_open = mock_safe_open
mock_atb_llm.models.llama.config_llama.LlamaConfig = MockLlamaConfig
mock_atb_llm.utils.eplb_expert_data_collect.EplbExpertDataCollect = MockEplbExpertDataCollect
mock_atb_llm.utils.moe_utils.EPLBType = MockEPLBType
mock_atb_llm.utils.moe_utils.save_eplb_data = mock_save_eplb_data
mock_atb_llm.utils.initial = MagicMock()
mock_npu_soc_info = MagicMock()
mock_npu_soc_info.need_nz = False
mock_atb_llm.utils.initial.NPUSocInfo = MagicMock(return_value=mock_npu_soc_info)
mock_atb_llm.utils.initial.NPUSocInfo = MagicMock()
mock_atb_llm.utils.env.ENV = MockENV
mock_atb_llm.models.InferenceMode = MagicMock()


# Mock atb_llm.runner module
mock_atb_llm.runner.tokenizer_wrapper.TokenizerWrapper = MagicMock()
mock_atb_llm.runner.model_runner.ModelRunner = MockModelRunner
mock_atb_llm.runner.model_runner.generate_mem_pool_event_key = mock_generate_mem_pool_event_key

# Mock atb_llm.models.deepseekv2 module
mock_atb_llm.models.deepseekv2 = MagicMock()
mock_atb_llm.models.deepseekv2.eplb = MagicMock()
mock_atb_llm.models.deepseekv2.eplb.eplb_planner = MagicMock()
mock_atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker = MagicMock()

# Add mock module to sys.modules
sys.modules["atb_llm"] = mock_atb_llm
sys.modules["atb_llm.utils"] = mock_atb_llm.utils
sys.modules["atb_llm.utils.layers"] = mock_atb_llm.utils.layers
sys.modules["atb_llm.utils.layers.attention"] = mock_atb_llm.utils.layers.attention
sys.modules["atb_llm.utils.layers.attention.attention_mask"] = mock_atb_llm.utils.layers.attention.attention_mask
sys.modules["atb_llm.utils.file_utils"] = mock_atb_llm.utils.file_utils
sys.modules["atb_llm.models"] = mock_atb_llm.models
sys.modules["atb_llm.models.deepseekv2"] = mock_atb_llm.models.deepseekv2
sys.modules["atb_llm.models.deepseekv2.eplb"] = mock_atb_llm.models.deepseekv2.eplb
sys.modules["atb_llm.models.deepseekv2.eplb.eplb_planner"] = mock_atb_llm.models.deepseekv2.eplb.eplb_planner
sys.modules["atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker"] = (
    mock_atb_llm.models.deepseekv2.eplb.eplb_planner.eplb_worker
)
sys.modules["atb_llm.utils.env"] = mock_atb_llm.utils.env
sys.modules["atb_llm.utils.eplb_expert_data_collect"] = mock_atb_llm.utils.eplb_expert_data_collect
sys.modules["atb_llm.utils.moe_utils"] = mock_atb_llm.utils.moe_utils
sys.modules["atb_llm.utils.initial"] = mock_atb_llm.utils.initial
sys.modules["atb_llm.models"] = mock_atb_llm.models
sys.modules["atb_llm.models.llama"] = mock_atb_llm.models.llama
sys.modules["atb_llm.models.llama.config_llama"] = mock_atb_llm.models.llama.config_llama
sys.modules["atb_llm.runner"] = mock_atb_llm.runner
sys.modules["atb_llm.runner.tokenizer_wrapper"] = mock_atb_llm.runner.tokenizer_wrapper
sys.modules["atb_llm.runner.model_runner"] = mock_atb_llm.runner.model_runner

# Mock _cpu_logits_handler module
mock_cpu_logits_handler = MagicMock()
mock_cpu_logits_handler._PostProcessingManager = MagicMock()

# Create a mock processor with next_token_chooser method that returns two values
mock_processor = MagicMock()
# Return a tuple of (token_ids_array, logprobs)
mock_processor.next_token_chooser = MagicMock(
    return_value=(
        np.array([[0, 1, 2, 3, 4, 5]]),  # token_ids_array
        np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]),  # logprobs
    )
)
# Add other necessary methods
mock_processor.set_batch_configs = MagicMock()
mock_processor.delete_configs = MagicMock()

mock_cpu_logits_handler._PostProcessingManager.get_instance = MagicMock(return_value=mock_processor)

# Mock _cpu_logits_handler module
mock_cpu_logits_handler = MagicMock()
mock_cpu_logits_handler._PostProcessingManager = MagicMock()

# Create a mock processor with next_token_chooser method that returns two values
mock_processor = MagicMock()
# Return a tuple of (token_ids_array, logprobs)
mock_processor.next_token_chooser = MagicMock(
    return_value=(
        np.array([[0, 1, 2, 3, 4, 5]]),  # token_ids_array
        np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]),  # logprobs
    )
)
# Add other necessary methods
mock_processor.set_batch_configs = MagicMock()
mock_processor.delete_configs = MagicMock()

mock_cpu_logits_handler._PostProcessingManager.get_instance = MagicMock(return_value=mock_processor)


# Add mock module to sys.modules
sys.modules["_cpu_logits_handler"] = mock_cpu_logits_handler

# Mock acl module
mock_acl = MagicMock()
mock_acl.rt.get_mem_info = MagicMock(return_value=(1024, 2048, 0))
sys.modules["acl"] = mock_acl


from atb_llm.utils.layers.attention.attention_mask import AttentionMask  # noqa: E402
from atb_llm.utils.file_utils import safe_open  # noqa: E402
from atb_llm.models.llama.config_llama import LlamaConfig  # noqa: E402
from mindie_llm.modeling.model_wrapper.model_info import ModelInfo  # noqa: E402

current_file_path = Path(__file__).resolve()
target_dir = current_file_path.parent.parent
MODEL_PATH = target_dir.joinpath("test_weights/llama3")


@dataclass
class FakeParallelInfo:
    dp: int = 1
    tp: int = 1
    cp: int = 1
    sp: int = 1


class FakeModel:
    max_position_embeddings = 12345


class FakeModelRunner:
    def __init__(self, parallel_info: FakeParallelInfo, device: str = "cpu"):
        with safe_open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
            config_dict = json.loads(f.read())

        config = LlamaConfig.from_dict(config_dict)
        self.config = config
        self.config_dict = config_dict
        self.llm_config = MagicMock()
        self.tokenizer = None

        self.mapping = MagicMock()
        self.mapping.attn_dp.group_size = parallel_info.dp
        self.mapping.attn_dp.rank = 0
        self.mapping.attn_tp.group_size = parallel_info.tp
        self.mapping.attn_tp.rank = 0
        self.mapping.attn_inner_sp.group_size = parallel_info.sp
        self.mapping.attn_inner_sp.rank = 0
        self.mapping.attn_cp.group_size = parallel_info.cp
        self.mapping.attn_cp.rank = 0

        self.mapping.has_dp = MagicMock(return_value=True) if parallel_info.dp > 1 else MagicMock(return_value=False)
        self.mapping.has_attn_cp = (
            MagicMock(return_value=True) if parallel_info.cp > 1 else MagicMock(return_value=False)
        )
        self.mapping.has_attn_inner_sp = (
            MagicMock(return_value=True) if parallel_info.sp > 1 else MagicMock(return_value=False)
        )

        self.process_group = MagicMock()
        self.device = torch.device(device=device)
        self.dtype = torch.bfloat16

        self.kv_cache_dtype = torch.float16
        self.num_layers = config_dict["num_hidden_layers"]
        self.num_kv_heads = config_dict["num_key_value_heads"]
        self.head_size = config_dict["hidden_size"] // config_dict["num_key_value_heads"]
        self.k_head_size = self.head_size
        self.v_head_size = self.head_size
        self.kvcache_quant_layers = []

        self.max_position_embeddings = config_dict["max_position_embeddings"]

        # Add model attribute with eplb_level
        self.model = MagicMock()
        self.model.eplb_level = 0
        self.model.is_multimodal = False
        self.soc_info = MagicMock()
        self.soc_info.is_300i = MagicMock(return_value=False)
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

    def forward(self, *args, **kwargs):
        logits = torch.zeros(1, 10)  # 假定词表长度为10
        logits[0][2] = 2
        logits[0][5] = 3
        logits[0][8] = 4
        return logits

    def clear_internal_tensors(self):
        pass


class FakeModelWrapper:
    def __init__(self, model_info: ModelInfo, model_runner: FakeModelRunner):
        # 使用 MagicMock 自动支持任意属性链
        self.config = MagicMock()
        self.config.eos_token_id = 0
        self.config.bos_token_id = 1
        self.config.top_k = 1000
        self.config.vocab_size = 130000

        self.mapping = MagicMock()
        self.mapping.attn_inner_sp.group_size = model_runner.mapping.attn_inner_sp.group_size
        self.mapping.attn_inner_sp.rank = 0
        self.mapping.attn_cp.group_size = model_runner.mapping.attn_cp.group_size
        self.mapping.attn_cp.rank = 0
        self.mapping.attn_tp.group_size = model_runner.mapping.attn_tp.group_size
        self.mapping.attn_tp.rank = 0
        self.mapping.attn_dp.group_size = model_runner.mapping.attn_dp.group_size
        self.mapping.attn_dp.rank = 0
        self.dp_size = model_runner.mapping.attn_dp.group_size
        self.sp_size = model_runner.mapping.attn_inner_sp.group_size
        self.cp_size = model_runner.mapping.attn_cp.group_size

        self.is_multimodal = False
        self.model_info = model_info
        self.model_runner = model_runner

        self.generate_position_ids = self.model_runner.generate_position_ids


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
