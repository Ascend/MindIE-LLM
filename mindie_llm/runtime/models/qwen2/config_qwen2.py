# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass, field, fields
from typing import Any

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig, BaseRopeScaling
from mindie_llm.runtime.utils.helpers.parameter_validators import (
    FloatParameterValidator, RangeParamaterValidator, IntParameterValidator, Field
)
from mindie_llm.utils.log.logging import logger


@dataclass
class Qwen2RopeScaling(BaseRopeScaling):
    """Qwen2-specific RoPE scaling configuration.
    
    Qwen2 uses a simple RoPE configuration with support for partial rotary embedding
    and different rope types (default, linear, dynamic, yarn, longrope, llama3).
    """
    rope_type: str = field(default='default', metadata={
        'validator': RangeParamaterValidator([
            'default', 'yarn',
        ])
    })
    
    factor: float = field(default=1.0, metadata={
        'validator': FloatParameterValidator(Field(ge=-65504, le=65504))
    })
    
    original_max_position_embeddings: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True)
    })
    
    partial_rotary_factor: float = field(default=1.0, metadata={
        'validator': FloatParameterValidator(Field(gt=0), allow_none=True)
    })
    
    def __post_init__(self):
        if self.factor > 1.0:
            if self.original_max_position_embeddings is not None:
                self.max_position_embeddings = int(self.factor * \
                    self.original_max_position_embeddings)
            else:
                _msg = ("Rope Scaling Failed. The `rope_scaling.factor` > 1.0, "
                    "but `rope_scaling.original_max_position_embeddings` is None. "
                    "Please check your `rope_scaling` in model's config.json.")
                logger.warning(_msg)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], 
                  rope_theta: float = 1000000.0,
                  max_position_embeddings: int | None = None) -> 'Qwen2RopeScaling':
        if config_dict is None:
            config_dict = {}
        
        rope_type = config_dict.get('rope_type', config_dict.get('type', 'default'))
        factor = config_dict.get('factor', 1.0)
        original_max = config_dict.get('original_max_position_embeddings', None)
        partial_rotary_factor = config_dict.get('partial_rotary_factor', 1.0)
        
        if factor > 1.0 and original_max is not None:
            max_position_embeddings = int(factor * original_max)
        
        return cls(
            rope_type=rope_type,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            factor=factor,
            original_max_position_embeddings=original_max,
            partial_rotary_factor=partial_rotary_factor
        )


@dataclass
class Qwen2Config(HuggingFaceConfig):
    """Configuration class for Qwen2 model.

    Extends HuggingFaceConfig with Qwen2-specific attributes.
    """
    attention_bias = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads
    
    def _create_rope_scaling(self, rope_scaling_dict: dict[str, str] | None,
                             rope_theta: float,
                             max_position_embeddings: int | None) -> Qwen2RopeScaling:
        """Create Qwen2-specific RoPE scaling configuration.
        
        Args:
            rope_scaling_dict: The rope_scaling dictionary from config.json
            rope_theta: The model's rope_theta value
            max_position_embeddings: The model's max_position_embeddings value
            
        Returns:
            Configured Qwen2RopeScaling instance
        """
        return Qwen2RopeScaling.from_dict(
            rope_scaling_dict,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )