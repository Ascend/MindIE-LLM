# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from mindie_llm.runtime.models.qwen3.config_qwen3 import Qwen3Config


def test_qwen3_config_minimal():
    config = Qwen3Config.from_dict({
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6, 
    })
    assert config.use_qk_norm is True
    assert config.is_reasoning_model is True