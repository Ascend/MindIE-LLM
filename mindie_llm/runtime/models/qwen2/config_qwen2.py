# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from typing import Any, List, Optional, Tuple
from dataclasses import dataclass, field


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
        
        if factor > 1.0 and original_max is not None:
            max_position_embeddings = int(factor * original_max)
        
        return cls(
            rope_type=rope_type,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            factor=factor,
            original_max_position_embeddings=original_max,
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


    @classmethod
    def map_weight_to_model(cls, weight_name: str) -> List[str]:
        """
        Map W8A8SC weight name (GPT-2 style transformer format) to HuggingFace model format.

        Converts weight names like "transformer.h.0.attn.c_attn.weight" to
        "model.layers.0.self_attn.qkv_proj.weight" for W8A8SC quantization.

        Args:
            weight_name: Weight name in transformer format.

        Returns:
            List of mapped keys. The first element is the primary mapped key.
            For packed weights (qkv_proj, gate_up_proj), additional separated keys
            are appended (e.g., q_proj, k_proj, v_proj for qkv_proj).
        """
        replace_rules = [
            ("transformer.", "model."),
            (".wte", ".embed_tokens"),
            (".h.", ".layers."),
            (".attn.", ".self_attn."),
            (".c_attn.", ".qkv_proj."),
            (".mlp.c_proj.", ".mlp.down_proj."),  # ".mlp.c_proj." must before ".c_proj."
            (".c_proj.", ".o_proj."),
            (".w2_w1.", ".gate_up_proj."),
        ]

        mapped_name = weight_name
        for old, new in replace_rules:
            mapped_name = mapped_name.replace(old, new)

        keys = [mapped_name]

        def _extract_suffix(name: str, suffixes: Tuple[str, ...]) -> Optional[str]:
            for suffix in suffixes:
                if name.endswith(suffix):
                    return suffix
            return None

        if ".qkv_proj." in mapped_name:
            qkv_suffixes = (".weight", ".index", ".info", ".input_scale", ".input_offset", ".deq_scale", ".quant_bias")
            suffix = _extract_suffix(mapped_name, qkv_suffixes)
            if suffix:
                base = mapped_name.replace(".qkv_proj.", ".")
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    keys.append(base.replace(suffix, f".{proj}{suffix}"))

        elif ".gate_up_proj." in mapped_name:
            gate_suffixes = (".weight", ".scale")
            suffix = _extract_suffix(mapped_name, gate_suffixes)
            if suffix:
                base = mapped_name.replace(".gate_up_proj.", ".")
                for proj in ["gate_proj", "up_proj"]:
                    keys.append(base.replace(suffix, f".{proj}{suffix}"))

        return keys

    @classmethod
    def map_model_to_weight(cls, module_prefix: str) -> str:
        """
        Map HuggingFace model format to W8A8SC weight format (GPT-2 style).

        Inverse operation of map_weight_to_model. Used for looking up weight file
        paths in W8A8SC format models.

        Args:
            module_prefix: Module prefix in model format.

        Returns:
            Weight file prefix in transformer format.
        """
        reverse_replace_rules = [
            ("model.layers", "transformer.h"),
            (".self_attn.", ".attn."),
            (".qkv_proj", ".c_attn"),
            (".attn.o_proj", ".attn.c_proj"),
            (".gate_up_proj", ".w2_w1"),
            (".mlp.down_proj", ".mlp.c_proj"),
            (".input_layernorm", ".ln_1"),
            (".post_attention_layernorm", ".ln_2"),
            ("model.norm", "transformer.ln_f"),
            ("model.embed_tokens", "transformer.wte"),
            ("model.lm_head", "transformer.wte"),
        ]

        weight_prefix = module_prefix
        for old, new in reverse_replace_rules:
            weight_prefix = weight_prefix.replace(old, new)

        return weight_prefix
    
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
