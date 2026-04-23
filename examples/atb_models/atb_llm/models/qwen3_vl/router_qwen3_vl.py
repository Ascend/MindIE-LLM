# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import copy
from typing import Dict, List, Any
from dataclasses import dataclass
import torch
from transformers import AutoProcessor
import PIL

from atb_llm.models.base.router import BaseRouter
from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils.shm_utils import process_shared_memory
from atb_llm.utils.multimodal_utils import safe_open_image, check_video_path

from .input_builder_qwen3_vl import Qwen3vlInputBuilder


IMAGE = "image"
VIDEO = "video"
TEXT = "text"
TYPE = "type"

messages_template = [
    {
        "role": "user",
        "content": [],
    }
]


@dataclass
class Qwen3vlRouter(BaseRouter):
    is_multimodal: bool = True
    _processor: Any = None

    def __post_init__(self):
        super().__post_init__()
        self.model_type = "qwen3_vl"
        self.model_type_cap = "Qwen3vl"

    @property
    def processor(self):
        if not hasattr(self, "_processor") or self._processor is None:
            self._processor = self.get_processor()
        return self._processor

    def tokenize(self, inputs: List[Dict], **kwargs):
        text = ""
        message_list = []
        shm_name_save_path = kwargs.get("shm_name_save_path", None)
        for item in inputs:
            if item.get(IMAGE, None):
                img_fname = item[IMAGE]
                img = safe_open_image(PIL.Image, img_fname)
                message_list.append({TYPE: IMAGE, IMAGE: img})
                if shm_name_save_path is None:
                    shm_name_save_path = self.input_builder.get_shm_name_save_path(img_fname)
            if item.get(VIDEO, None):
                video_path = check_video_path(item[VIDEO])
                message_list.append({TYPE: VIDEO, VIDEO: video_path})
                if shm_name_save_path is None:
                    shm_name_save_path = self.input_builder.get_shm_name_save_path(video_path)
            if item.get(TEXT, None):
                text = item[TEXT]
                message_list.append({TYPE: TEXT, TEXT: text})
        messages = copy.deepcopy(messages_template)
        messages[0]["content"] = message_list
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        shm_info = process_shared_memory(inputs, shm_name_save_path)
        input_ids = self.input_builder.update_token_id(inputs, shm_info)
        return input_ids

    def get_input_builder(self):
        return Qwen3vlInputBuilder(self.tokenizer, self.config, self.processor)

    def get_config(self):
        config_cls = self.get_config_cls()
        config = config_cls.from_dict(self.config_dict)
        config.model_name_or_path = self.model_name_or_path
        if config.text_config.dtype == torch.bfloat16:
            config.torch_dtype = torch.bfloat16
        elif config.text_config.dtype == torch.float16:
            config.torch_dtype = torch.float16
        else:
            err_msg = "`torch_dtype` is only supported for type `float16` and `bfloat16`"
            raise NotImplementedError(err_msg)
        super().check_config(config)
        return config

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_processor(self):
        return safe_from_pretrained(AutoProcessor, self.model_name_or_path)
