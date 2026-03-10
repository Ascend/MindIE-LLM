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

from transformers.configuration_utils import PretrainedConfig

from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.helpers.parameter_validators import (
    IntParameterValidator, FloatParameterValidator, RangeParamaterValidator, Field,
    CompositeParameterValidator, ListParameterValidator
)
from mindie_llm.utils.log.logging import logger


@dataclass
class GenerationConfig:
    """Model generation configuration parameters."""
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int | list[int]] = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        field_names = {field.name for field in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)

    def validate(self, vocab_size: int):
        """Validate token IDs against vocabulary size."""
        if not isinstance(vocab_size, int):
            logger.error("`vocab_size` in the config.json from model weight path must be an integer.")
            raise ValueError("`vocab_size` in the config.json from model weight path must be an integer.")
        validators = {
            'pad_token_id': IntParameterValidator(Field(ge=-1, le=vocab_size), allow_none=True),
            'bos_token_id': IntParameterValidator(Field(ge=0, le=vocab_size - 1), allow_none=True),
            'eos_token_id': CompositeParameterValidator({
                int: IntParameterValidator(Field(ge=0, le=vocab_size - 1)),
                list: ListParameterValidator(CompositeParameterValidator({
                    int: IntParameterValidator(Field(ge=0, le=vocab_size - 1)),
                    list: ListParameterValidator(IntParameterValidator(Field(ge=0, le=vocab_size - 1)),
                    Field(min_length=0, max_length=2147483647))
                }), Field(min_length=0, max_length=2147483647))
            }, allow_none=True)
        }

        for key, validator in validators.items():
            value = getattr(self, key)
            validator.validate(value, key)


@dataclass
class BaseRopeScaling:
    """Base class for RoPE scaling configuration.
    
    Provides the basic fields rope_theta and max_position_embeddings.
    Each model should implement its own RopeScaling class that inherits from this base class.
    """
    rope_theta: float = field(default=1000000.0, metadata={
        'validator': FloatParameterValidator(Field(ge=-1e9, le=1e9))
    })
    
    max_position_embeddings: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True)
    })
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], 
                  rope_theta: float = 1000000.0,
                  max_position_embeddings: int | None = None) -> 'BaseRopeScaling':
        """Create BaseRopeScaling instance from config dictionary.
        
        Args:
            config_dict: The rope_scaling dictionary from config.json
            rope_theta: The model's rope_theta value
            max_position_embeddings: The model's max_position_embeddings value
            
        Returns:
            Configured BaseRopeScaling instance
        """
        if config_dict is None:
            config_dict = {}
        
        return cls(
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )


@dataclass
class HuggingFaceConfig(PretrainedConfig):
    """Base configuration class for models with validation."""
    vocab_size: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(gt=0), allow_none=True)})
    hidden_size: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True)})
    intermediate_size: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=2147483647), allow_none=True)})
    num_attention_heads: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=10000), allow_none=True)})
    num_key_value_heads: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=10000), allow_none=True)})
    head_dim: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=1000), allow_none=True)})
    rms_norm_eps: float | None = field(default=None, metadata={
        'validator': FloatParameterValidator(Field(ge=0, le=1), allow_none=True)})
    num_hidden_layers: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(ge=1, le=1000), allow_none=True)})
    max_position_embeddings: int | None = field(default=None, metadata={
        'validator': IntParameterValidator(Field(gt=0), allow_none=True)})
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int | list[int]] = None
    
    rope_scaling: BaseRopeScaling | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rope_scaling_dict = kwargs.get('rope_scaling', None)
        rope_theta = kwargs.get('rope_theta', 1000000.0)
        max_position_embeddings = kwargs.get('max_position_embeddings', None)
        
        if hasattr(self, '_create_rope_scaling'):
            self.rope_scaling = self._create_rope_scaling(
                rope_scaling_dict or {},
                rope_theta,
                max_position_embeddings
            )
        else:
            self.rope_scaling = BaseRopeScaling.from_dict(
                rope_scaling_dict,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'HuggingFaceConfig':
        """Create validated config from dictionary."""
        config = cls(**config_dict)
        config.validate()
        return config

    def validate(self):
        """Validate configuration parameters."""
        self._validate_config()
        self._validate_rope_scaling()

    def get_num_attention_heads_per_rank(self) -> int:
        """Returns the number of attention heads per rank."""
        parallel_info_manager = get_parallel_info_manager()
        num_attention_heads = self.num_attention_heads
        group_size = parallel_info_manager.get(ParallelType.ATTN_TP).group_size
        return (num_attention_heads + group_size - 1) // group_size

    def get_num_kv_heads_per_rank(self) -> int:
        """Returns the number of kv heads per rank."""
        parallel_info_manager = get_parallel_info_manager()
        tp_size = parallel_info_manager.get(ParallelType.ATTN_TP).group_size
        num_key_value_heads = self.num_key_value_heads
        if num_key_value_heads < tp_size:
            repeat_times = tp_size // num_key_value_heads
        else:
            repeat_times = 1
        return (num_key_value_heads * repeat_times + tp_size - 1) // tp_size

    def _validate_config(self):
        """Validate common configuration parameters."""
        for field_name, field_value in self.__dataclass_fields__.items():
            validator = field_value.metadata.get('validator')
            if validator:
                validator.validate(getattr(self, field_name), field_name)

        validators = {
            'pad_token_id': IntParameterValidator(Field(ge=-1, le=self.vocab_size), allow_none=True),
            'bos_token_id': IntParameterValidator(Field(ge=0, le=self.vocab_size - 1), allow_none=True),
            'eos_token_id': CompositeParameterValidator({
                int: IntParameterValidator(Field(ge=0, le=self.vocab_size - 1)),
                list: ListParameterValidator(CompositeParameterValidator({
                    int: IntParameterValidator(Field(ge=0, le=self.vocab_size - 1)),
                    list: ListParameterValidator(IntParameterValidator(Field(ge=0, le=self.vocab_size - 1)),
                    Field(min_length=0, max_length=2147483647))
                }), Field(min_length=0, max_length=2147483647))
            }, allow_none=True),
        }
        for key, validator in validators.items():
            value = getattr(self, key)
            validator.validate(value, key)

    def _validate_rope_scaling(self):
        """Validate rope scaling configuration."""
        if self.rope_scaling is None:
            return

        for local_field in fields(self.rope_scaling):
            validator = local_field.metadata.get('validator')
            if validator:
                validator.validate(getattr(self.rope_scaling, local_field.name), local_field.name)
