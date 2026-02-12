# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass

from mindie_llm.runtime.models.qwen2.config_qwen2 import Qwen2Config


@dataclass
class Qwen3Config(Qwen2Config):
    """Configuration class for Qwen3 model.

    Extends HuggingFaceConfig with Qwen3-specific attributes.
    """
    use_qk_norm: bool = True
    is_reasoning_model: bool = True
    attention_bias = False
    
    def __init__(self, **kwargs):
        """Initializes Qwen3 configuration with optional keyword arguments.
        """
        super().__init__(**kwargs)
        
        
