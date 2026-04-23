# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
import os
from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoProcessor
from atb_llm.models.base.config import QuantizationConfig
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.qwen2_vl.config_qwen2_vl import Qwen2vlConfig
from atb_llm.models.qwen2_vl.data_preprocess_qwen2_vl import fetch_image, fetch_video, process_shared_memory
from atb_llm.models.qwen2_vl.input_builder_qwen2_vl import Qwen2vlInputBuilder
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger

VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_FACTOR = 28
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
PYTORCH_TENSOR = "pt"
IMAGE = "image"
VIDEO = "video"
TEXT = "text"

messages_template = [
    {
        "role": "user",
        "content": [],
    }
]


@dataclass
class Qwen2vlRouter(BaseRouter):
    is_multimodal: bool = True
    _processor: Any = None

    @property
    def image_processor(self):
        if not hasattr(self, "_image_processor"):
            self._image_processor = self.get_image_processor()
        elif self._image_processor is None:
            self._image_processor = self.get_image_processor()
        return self._image_processor

    @property
    def processor(self):
        if not hasattr(self, "_processor"):
            self._processor = self.get_processor()
        elif self._processor is None:
            self._processor = self.get_processor()
        return self._processor

    def tokenize(self, inputs, **kwargs):
        image_token_id = getattr(self.config, "image_token_id", IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", VIDEO_TOKEN_ID)

        vision_info_list = []
        message_list = []
        shm_name_save_path = kwargs.get("shm_name_save_path", None)

        for single_input in inputs:
            if single_input.get(IMAGE, None):
                message_list.append(single_input)

                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[IMAGE]))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")

                images_inputs, feature_lens = fetch_image(self.image_processor, single_input)

                shared_memory_result = process_shared_memory(
                    images_inputs.pixel_values, shm_name_save_path, images_inputs.image_grid_thw
                )
                vision_info_list.append(
                    [
                        shared_memory_result["pixel_values_shm_name"],
                        shared_memory_result["pixel_values_shape_value"],
                        shared_memory_result["thw_value"],
                        feature_lens,
                        image_token_id,
                    ]
                )

            elif single_input.get(VIDEO, None):
                message_list.append(single_input)

                if shm_name_save_path is None:
                    shm_name_save_dir = os.path.dirname(os.path.dirname(single_input[VIDEO]))
                    shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
                # 默认 fps 为 2
                video_single_message = {"type": "video", "video": single_input[VIDEO], "fps": 2.0}
                video_inputs, feature_lens, second_per_grid_t = fetch_video(
                    self.image_processor, video_single_message, IMAGE_FACTOR
                )
                shared_memory_result = process_shared_memory(
                    video_inputs.pixel_values_videos, shm_name_save_path, video_inputs.video_grid_thw, second_per_grid_t
                )
                vision_info_list.append(
                    [
                        shared_memory_result["pixel_values_shm_name"],
                        shared_memory_result["pixel_values_shape_value"],
                        shared_memory_result["thw_value"],
                        shared_memory_result["second_per_grid_t_shm_name"],
                        shared_memory_result["second_per_grid_t_shape_value"],
                        feature_lens,
                        video_token_id,
                    ]
                )

            elif single_input.get(TEXT, None):
                message_list.append(single_input)
            else:
                logger.error(
                    "The input field currently only supports 'image', 'video' or 'text'.",
                    ErrorCode.ATB_MODELS_PARAM_INVALID,
                )
                raise TypeError("The input field currently only supports 'image', 'video' or 'text'.")
        return self.process_token(vision_info_list, message_list)

    def get_input_builder(self):
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            return Qwen2vlInputBuilder(
                self.tokenizer,
                self.image_processor,
                self.processor,
                self.config,
                max_length=self.config.max_position_embeddings,
            )
        return Qwen2vlInputBuilder(self.tokenizer, self.image_processor, self.processor, self.config)

    def get_config(self):
        config = Qwen2vlConfig.from_dict(self.config_dict)
        setattr(config, "quantization_config", QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        config.model_name_or_path = self.model_name_or_path
        self.check_config_qwen2_vl(config)
        return config

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_image_processor(self):
        return self.processor.image_processor

    def get_processor(self):
        return safe_from_pretrained(AutoProcessor, self.model_name_or_path, use_fast=True)

    def check_config_qwen2_vl(self, config):
        super().check_config(config)
        attribute_ranges = {
            "mm_hidden_size": (1, 2147483647),
            "num_key_value_heads": (1, 2147483647),
        }
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

    def process_token(self, vision_info_list, message_list):
        messages = copy.deepcopy(messages_template)
        messages[0]["content"] = message_list
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not prompt:
            prompt = self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].flatten()
        new_input_ids = input_ids
        vision_start_token_id = getattr(self.config, "vision_start_token_id", VISION_START_TOKEN_ID)
        vision_end_token_id = getattr(self.config, "vision_end_token_id", VISION_END_TOKEN_ID)
        image_token_id = getattr(self.config, "image_token_id", IMAGE_TOKEN_ID)
        video_token_id = getattr(self.config, "video_token_id", VIDEO_TOKEN_ID)
        bos_pos = torch.where(torch.eq(input_ids, vision_start_token_id))[0]
        eos_pos = torch.where(torch.eq(input_ids, vision_end_token_id))[0]

        vision_num = bos_pos.size(0)
        expand_token_ids = []
        pre = 0
        for i in range(0, vision_num, 1):
            feature_lens = vision_info_list[i][-2]
            text_token = input_ids[pre : bos_pos[i]]
            pre = eos_pos[i] + 1
            if vision_info_list[i][-1] == image_token_id:
                vision_pad_token = torch.cat(
                    [
                        torch.tensor([vision_start_token_id], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][0]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][1]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][2]], dtype=input_ids.dtype),
                        torch.full((feature_lens - 3,), vision_info_list[i][-1], dtype=input_ids.dtype),
                        torch.tensor([vision_end_token_id], dtype=input_ids.dtype),
                    ]
                )
            elif vision_info_list[i][-1] == video_token_id:
                vision_pad_token = torch.cat(
                    [
                        torch.tensor([vision_start_token_id], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][0]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][1]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][2]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][3]], dtype=input_ids.dtype),
                        torch.tensor([vision_info_list[i][4]], dtype=input_ids.dtype),
                        torch.full((feature_lens - 5,), vision_info_list[i][-1], dtype=input_ids.dtype),
                        torch.tensor([vision_end_token_id], dtype=input_ids.dtype),
                    ]
                )
            if text_token.size(0) != 0:
                expand_token_ids.append(text_token)

            if vision_pad_token.size(0) != 0:
                expand_token_ids.append(vision_pad_token)

        text_token = input_ids[pre:]
        if text_token.size(0) != 0:
            expand_token_ids.append(text_token)

        if expand_token_ids:
            new_input_ids = torch.cat(expand_token_ids)
        return new_input_ids
