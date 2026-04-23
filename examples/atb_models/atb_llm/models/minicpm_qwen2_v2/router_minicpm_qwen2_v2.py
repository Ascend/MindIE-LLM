# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass

import numpy as np
from transformers import AutoProcessor

from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from .config_minicpm_qwen2_v2 import Minicpmqwen2v2Config
from .input_builder_minicpm_qwen2_v2 import inner_tokenize, Minicpmqwen2v2InputBuilder
from ..base.router import BaseRouter
from ..base.config import QuantizationConfig
from ..base.model_utils import safe_from_pretrained
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class Minicpmqwen2v2Router(BaseRouter):
    is_multimodal: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.processor = safe_from_pretrained(
            AutoProcessor, self.config.model_name_or_path, trust_remote_code=self.trust_remote_code
        )

    def tokenize(self, inputs, **kwargs):
        return inner_tokenize(inputs=inputs, config=self.config, processor=self.processor, **kwargs)

    def check_config_minicpmqwen2v2(self, config):
        super().check_config(config)
        attribute_ranges = {"mm_hidden_size": (1, 2147483647), "num_key_value_heads": (1, 2147483647)}
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                logger.error(
                    f"`self._config.{attr}` must be between {min_val} and {max_val}.",
                    ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE,
                )
                raise ValueError(f"`self._config.{attr}` must be between {min_val} and {max_val}.")

    def get_config(self):
        config = Minicpmqwen2v2Config.from_dict(self.config_dict)
        config.model_name_or_path = self.model_name_or_path
        setattr(config, "quantization_config", QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        self.check_config_minicpmqwen2v2(config)
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast,
        )

    def get_generation_config(self):
        generation_config = super().get_generation_config()
        return generation_config

    def get_input_builder(self, **kwargs):
        return Minicpmqwen2v2InputBuilder(config=self.config, processor=self.processor, **kwargs)

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        return position_ids
