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
from pathlib import Path
from typing import List
from dataclasses import dataclass

from unittest.mock import MagicMock
import torch

from atb_llm.utils.layers import AttentionMask
from atb_llm.utils.file_utils import safe_open
from atb_llm.models.llama.config_llama import LlamaConfig
from mindie_llm.modeling.model_wrapper.model_info import ModelInfo

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
    def __init__(self, parallel_info: FakeParallelInfo, device: str = 'cpu'):
        with safe_open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
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

        self.mapping.has_dp = (
            MagicMock(return_value=True) 
            if parallel_info.dp > 1
            else MagicMock(return_value=False)
        )
        self.mapping.has_attn_cp = (
            MagicMock(return_value=True)
            if parallel_info.cp > 1
            else MagicMock(return_value=False)
        )
        self.mapping.has_attn_inner_sp = (
            MagicMock(return_value=True)
            if parallel_info.sp > 1
            else MagicMock(return_value=False)
        )
        
        self.process_group = MagicMock()
        self.device = torch.device(device=device)
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
        logits = torch.zeros(1, 10) # 假定词表长度为10
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
