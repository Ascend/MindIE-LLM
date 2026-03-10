# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional, Any
from dataclasses import dataclass, field

from mindie_llm.runtime.config.huggingface_config import HuggingFaceConfig, BaseRopeScaling
from mindie_llm.runtime.utils.helpers.parameter_validators import IntParameterValidator, FloatParameterValidator, Field
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.log.logging import logger


@dataclass
class DeepSeekV3RopeScaling(BaseRopeScaling):
    """DeepSeek-V3-specific RoPE scaling configuration.
    
    DeepSeek-V3 uses YaRN scaling with DeepSeek-specific parameters like mscale_all_dim.
    """
    rope_type: str = field(default='deepseek_yarn')
    
    factor: float = field(default=1.0, metadata={
        'validator': FloatParameterValidator(Field(ge=-65504, le=65504))
    })
    
    original_max_position_embeddings: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True)
    })
    
    beta_fast: int = field(default=32, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647))
    })
    beta_slow: int = field(default=1, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647))
    })
    
    mscale: float = field(default=1.0, metadata={
        'validator': FloatParameterValidator(Field(gt=0), allow_none=True)
    })
    
    mscale_all_dim: float | None = field(default=None, metadata={
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
                  rope_theta: float = 10000.0,
                  max_position_embeddings: int | None = None) -> 'DeepSeekV3RopeScaling':
        if config_dict is None:
            config_dict = {}
        
        rope_type = config_dict.get('rope_type', config_dict.get('type', 'deepseek_yarn'))
        factor = config_dict.get('factor', 1.0)
        original_max = config_dict.get('original_max_position_embeddings', None)
        beta_fast = config_dict.get('beta_fast', 32)
        beta_slow = config_dict.get('beta_slow', 1)
        mscale = config_dict.get('mscale', 1.0)
        mscale_all_dim = config_dict.get('mscale_all_dim', None)
        
        if factor > 1.0 and original_max is not None:
            max_position_embeddings = int(factor * original_max)
        
        return cls(
            rope_type=rope_type,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            factor=factor,
            original_max_position_embeddings=original_max,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            mscale=mscale,
            mscale_all_dim=mscale_all_dim
        )


@dataclass
class DeepseekV3Config(HuggingFaceConfig):
    model_type: str = "deepseekv3"
    vocab_size: int = 102400
    hidden_size: int = 5120
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 60
    num_attention_heads: int = 128
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    rope_theta: float = 10000.0
    start_reasoning_token_id: int = 128798
    end_reasoning_token_id: int = 128799
    is_nzcasted: bool = False

    def __init__(self, **kwargs):
        # (NOTE): delete kwargs in the future
        super().__init__(**kwargs)
        self.attribute_map = {
            'head_dim': 'qk_nope_head_dim',
        }
        # (NOTE): add default value for compatibility
        if "ep_level" not in kwargs:
            self.ep_level = 1
        if self.model_type in ["deepseek_v3"]:
            self.is_reasoning_model = True
        # (NOTE): get from config
        self.index_n_heads = 64
        self.index_head_dim = 128
        self.index_topk = 2048

    def validate(self):
        super().validate()
        
        validators = {
            'num_experts_per_tok': IntParameterValidator(Field(ge=1, le=256), allow_none=True),
            'n_shared_experts': IntParameterValidator(Field(ge=0, le=256), allow_none=True),
            'first_k_dense_replace': IntParameterValidator(Field(ge=0, le=61), allow_none=False),
            'n_routed_experts': IntParameterValidator(Field(ge=2, le=256), allow_none=True),
            'q_lora_rank': IntParameterValidator(Field(ge=1, le=1536), allow_none=True),
            'qk_nope_head_dim': IntParameterValidator(Field(ge=1, le=128), allow_none=True),
            'qk_rope_head_dim': IntParameterValidator(Field(ge=1, le=64), allow_none=True)
        }

        for key, validator in validators.items():
            value = getattr(self, key)
            validator.validate(value, key)         

        if getattr(self, "num_experts_per_tok") > getattr(self, "n_routed_experts"):
            msg = f"self.num_experts_per_tok should be less than self.n_routed_experts, " \
                  f"but {self.num_experts_per_tok=}, {self.n_routed_experts=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if getattr(self, "first_k_dense_replace") > getattr(self, "num_hidden_layers"):
            msg = f"self.first_k_dense_replace should be less than self.num_hidden_layers, " \
                  f"but {self.first_k_dense_replace=}, {self.num_hidden_layers=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if self.topk_method not in ["greedy", "group_limited_greedy", "noaux_tc"]:
            msg = "`topk_method`'s type field must be one of ['greedy', 'group_limited_greedy', 'noaux_tc'], " \
                  f"got {self.topk_method}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        # Validate top-k parameter consistency
        if self.topk_method == "greedy" and self.topk_group != self.n_group and self.n_group != 1:
            msg = f"`topk_method is `greedy`, please set `topk_group=1` and `n_group=1`, " \
                  f"got topk_group={self.topk_group}, n_group={self.n_group}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)
    
    def _create_rope_scaling(self, rope_scaling_dict: dict[str, str] | None,
                             rope_theta: float,
                             max_position_embeddings: int | None) -> DeepSeekV3RopeScaling:
        """Create DeepSeek-V3-specific RoPE scaling configuration.
        
        Args:
            rope_scaling_dict: The rope_scaling dictionary from config.json
            rope_theta: The model's rope_theta value
            max_position_embeddings: The model's max_position_embeddings value
            
        Returns:
            Configured DeepSeekV3RopeScaling instance
        """
        return DeepSeekV3RopeScaling.from_dict(
            rope_scaling_dict,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )
