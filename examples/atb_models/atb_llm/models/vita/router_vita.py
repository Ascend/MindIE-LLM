# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from atb_llm.utils import multimodal_utils
from atb_llm.utils.shm_utils import encode_shm_name_to_int64, encode_shape_to_int64, create_shm
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter
from ..base.config import QuantizationConfig
from ..base.model_utils import safe_from_pretrained
from .modeling_vita_audio import AudioEncoderProcessor
from .tool import tokenizer_image_audio_token, get_rawvideo_dec, VideoConfig, dynamic_preprocess
from .conversation import conv_templates
from .input_builder_vita import VitaInputBuilder

IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
IMAGE_TAG = -1
VIDEO_TAG = -2
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"
MAX_IMAGE_LENGTH = 16


@dataclass
class InputAttrs:
    text: Optional[str] | None
    image_path: Optional[str] | None
    video_path: Optional[str] | None
    audio_path: Optional[str] | None


def from_list_format(list_format: List[Dict]):
    text = None
    image = None
    video = None
    audio = None
    for ele in list_format:
        if "text" in ele:
            text = ele["text"]
        elif "image_url" in ele or "image" in ele:
            image = ele.get("image") or ele.get("image_url")
        elif "video_url" in ele or "video" in ele:
            video = ele.get("video") or ele.get("video_url")
        elif "audio_url" in ele or "audio" in ele:
            audio = ele.get("audio") or ele.get("audio_url")
        else:
            logger.error("Unsupported element: " + str(ele) + ".", ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise KeyError("Unsupported element: " + str(ele) + ".")
    return InputAttrs(text, image, video, audio)


@dataclass
class VitaRouter(BaseRouter):
    is_multimodal: bool = True

    def process_images(self, images, model_cfg, image_processor):
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)

        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def get_audio_processor(self):
        audio_path = os.path.join(self.config.model_name_or_path, self.config.mm_audio_encoder)
        with open(os.path.join(audio_path, "train.yaml"), "r") as fin:
            audio_configs = yaml.safe_load(fin)
        audio_processor = AudioEncoderProcessor(dataset_conf=audio_configs["dataset_conf"])
        return audio_processor

    def process_media(self, input_meida, conv, image_processor):
        media_tensor = []
        qs = ""
        modality = "lang"
        img = "image"
        vid = "video"
        max_audio_id = 0
        for ele in input_meida:
            if "text" in ele:
                text = ele.get("text")
                new_q = qs + text
                qs = ""
                conv.append_message(conv.roles[0], new_q)
            elif "image_url" in ele or img in ele:
                image_path = ele.get(img) or ele.get("image_url")
                image = multimodal_utils.safe_load_multimodal_source(Image.open, image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image, p_num = dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True)
                image_tensor = self.process_images(image, self.config, image_processor).half()
                qs += DEFAULT_IMAGE_TOKEN * p_num[0] + "\n"
                modality = img
                maxlen_image_id = p_num[0] * 255
                media_tensor.append((image_path, image_tensor, maxlen_image_id))
            elif "video_url" in ele or vid in ele:
                video = ele.get(vid) or ele.get("video_url")
                video_config = VideoConfig(
                    video_path=video,
                    image_processor=image_processor,
                    max_frames=MAX_IMAGE_LENGTH,
                    video_framerate=1,
                    image_aspect_ratio=getattr(self.config, "image_aspect_ratio", None),
                )
                video_frames, slice_len = get_rawvideo_dec(video_config)
                image_tensor = video_frames.half()
                qs += DEFAULT_IMAGE_TOKEN * slice_len + "\n"
                modality = vid
                maxlen_image_id = slice_len * 255
                media_tensor.append((video, image_tensor, maxlen_image_id))
            elif "audio_url" in ele or "audio" in ele:
                audio = ele.get("audio") or ele.get("audio_url")
                new_q = qs + DEFAULT_AUDIO_TOKEN
                qs = ""
                conv.append_message(conv.roles[0], new_q)
                audio_process = self.get_audio_processor()
                audio_input, _ = audio_process.process(os.path.join(audio))
                audio_length = audio_input.shape[0]
                audio_input = torch.unsqueeze(audio_input, dim=0).half()
                audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
                max_audio_id = math.ceil((((audio_length - 3) // 2 - 2) // 2 + 1) / 2) - 1
                media_tensor.append((audio, audio_input, max_audio_id))
            else:
                logger.error("Unsupported element: " + str(ele) + ".", ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise KeyError("Unsupported element: " + str(ele) + ".")
        return conv, media_tensor, modality

    def build_input_ids(self, media_tensor, input_ids, media_pos):
        new_input_ids = []
        start = 0
        for i, (path, tensor, length) in enumerate(media_tensor):
            shm_name_save_dir = os.path.dirname(os.path.dirname(path))
            shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
            tensor = tensor.numpy()
            shm = create_shm(tensor.nbytes, shm_name_save_path)
            shared_array = np.ndarray(tensor.shape, dtype=tensor.dtype, buffer=shm.buf)
            shared_array[:] = tensor
            shm_name = encode_shm_name_to_int64(shm.name)
            shape_value = encode_shape_to_int64(tensor.shape)
            intermediate_token = torch.tensor([shm_name, shape_value])
            if i == 0:
                new_input_ids = torch.cat(
                    [
                        input_ids[start : media_pos[i] + 1],
                        intermediate_token,
                        torch.full((length - 2,), IGNORE_INDEX, dtype=input_ids.dtype),
                    ]
                )
            else:
                new_input_ids = torch.cat(
                    [
                        new_input_ids,
                        input_ids[start : media_pos[i] + 1],
                        intermediate_token,
                        torch.full((length - 2,), IGNORE_INDEX, dtype=input_ids.dtype),
                    ]
                )
            start = media_pos[i] + 1
        return start, new_input_ids

    def tokenize(self, inputs, **kwargs):
        tokenizer = safe_from_pretrained(AutoTokenizer, self.config.model_name_or_path, use_fast=True)
        image_processor = safe_from_pretrained(
            CLIPImageProcessor, os.path.join(self.config.model_name_or_path, self.config.mm_vision_tower)
        )
        if self.config.model_type.lower() == "vita-qwen2":
            self.conv_mode = "qwen2p5_instruct"
        else:
            self.conv_mode = "mixtral_two"
        conv = conv_templates[self.conv_mode].copy()
        conv, media_tensor, modality = self.process_media(inputs, conv, image_processor)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)

        input_ids = (
            tokenizer_image_audio_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        ).flatten()
        image_pos = torch.where(torch.eq(input_ids, IMAGE_TOKEN_INDEX))[0]
        audio_pos = torch.where(torch.eq(input_ids, AUDIO_TOKEN_INDEX))[0]
        image_final_pos = []
        audio_final_pos = []
        for idx in image_pos:
            if input_ids[idx + 1] != IMAGE_TOKEN_INDEX:
                image_final_pos.append(idx)
        for idx in audio_pos:
            if input_ids[idx + 1] != AUDIO_TOKEN_INDEX:
                audio_final_pos.append(idx)
        image_final_pos.extend(audio_final_pos)
        image_final_pos.sort()
        media_pos = image_final_pos
        if len(media_tensor) != 0:
            start, new_input_ids = self.build_input_ids(media_tensor, input_ids, media_pos)
        else:
            start = 0
            new_input_ids = input_ids
        if start == 0:
            new_input_ids = input_ids
        else:
            new_input_ids = torch.cat(
                [
                    new_input_ids,
                    input_ids[start:],
                ]
            )
        return new_input_ids

    def check_config_vita(self, config):
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

    def get_config(self):
        model_type = self.config_dict.get("model_type").lower()
        config_cls = self.get_config_cls()
        if "mixtral" in model_type:
            config_cls.model_type = "vita-mixtral"
        elif "qwen" in model_type:
            config_cls.model_type = "vita-Qwen2"
        config = config_cls.from_dict(self.config_dict)
        setattr(config, "quantization_config", QuantizationConfig(**{}))
        if self.max_position_embeddings:
            config.max_position_embeddings = self.max_position_embeddings
        config.torch_dtype = torch.float16
        config.model_name_or_path = self.model_name_or_path
        self.check_config_vita(config)
        return config

    def get_tokenizer(self):
        use_fast = True
        return safe_from_pretrained(
            AutoTokenizer,
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=use_fast,
        )

    def get_input_builder(self):
        return VitaInputBuilder(self.tokenizer, self.config)
