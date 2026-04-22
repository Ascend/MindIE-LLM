# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""PyTorch MiniCPM_Qwen2_V2 model."""

from copy import deepcopy
from typing import List
import json
from PIL import Image
import numpy as np
import torch

from atb_llm.models.base.flash_causal_multimodal import get_llm_model, MultiModalLLm
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from .resampler import Resampler
from .modeling_navit_siglip import SiglipVisionTransformer
from .input_builder_minicpm_qwen2_v2 import encode_video
from ...utils.shm_utils import get_data_from_shm
from ...utils.multimodal_utils import safe_open_image

MSG_CONTENT = "content"
INPUT_IDS = "input_ids"
VISION_HIDDEN_STATES = "image_pixel_array"
TGT_SIZES = "tgt_sizes"
PIXEL_VALUES = "pixel_values"
SCALE_EMB = "scale_emb"
IMAGE_BOUND = "image_bound"
SEP_PATTERN = "/n"
IMAGE_PATTERN = "(<image>./</image>)"
IMAGE_PAD = 128244  # <unk> token
VISION_START_TOKEN_ID = 151660
VISION_END_TOKEN_ID = 151661
MAX_VISION_BS_NZ = 128
MAX_VISION_BS = 64000


def process_qs(text, image):
    if isinstance(text, str):
        text = json.loads(text)
    copy_text = deepcopy(text)

    if len(text) == 0:
        raise RuntimeError("text is empty")

    if image is not None and isinstance(copy_text[0][MSG_CONTENT], str):
        copy_text[0][MSG_CONTENT] = [image, copy_text[0][MSG_CONTENT]]
    images = []
    for i, msg in enumerate(copy_text):
        role = msg["role"]
        content = msg[MSG_CONTENT]
        if role not in ["user", "assistant"]:
            logger.error("`role` must be user or assistant.", ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise RuntimeError("`role` must be user or assistant.")
        if i == 0 and role != "user":
            raise RuntimeError("The role of first msg should be user")
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append(IMAGE_PATTERN)
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg[MSG_CONTENT] = "\n".join(cur_msgs)
    return copy_text, images


class FlashMinicpmqwen2v2ForCausalLM(MultiModalLLm):
    def __init__(self, config, weights, **kwargs):
        self.config = config
        if not config.quantize:
            setattr(config, "quantize", None)
        else:
            setattr(config, "quantize", config.quantize)
        self.image_token_id = getattr(self.config, "image_token_id", IMAGE_PAD)
        self.vision_start_token_id = getattr(self.config, "vision_start_token_id", VISION_START_TOKEN_ID)
        self.vision_end_token_id = getattr(self.config, "vision_end_token_id", VISION_END_TOKEN_ID)
        super().__init__(config, weights, **kwargs)
        if self.soc_info.need_nz:
            self.max_vision_bs = MAX_VISION_BS_NZ
        else:
            self.max_vision_bs = MAX_VISION_BS
        self.weights = weights
        self.vision_tower = None
        self.language_model = None
        self.model_type = None
        self.init_multimodal()
        self.vocab_size = config.vocab_size
        self.vision_dim = self.vision_tower.embed_dim
        self.embed_dim = self.config.hidden_size
        self.init_resampler(self.embed_dim, self.vision_dim)
        self.multi_modal_projector = None

    @staticmethod
    def init_resamplerweight(module, weights):
        resampler_weights = [resampler_weight for resampler_weight in module.state_dict().keys()]
        for resampler_weight in resampler_weights:
            saved_weight = torch.nn.Parameter(weights.get_tensor(f"resampler.{resampler_weight}"), requires_grad=False)
            resampler_weight_list = resampler_weight.split(".")
            target_module = module
            for nxt_module in resampler_weight_list[:-1]:
                target_module = getattr(target_module, nxt_module)
            setattr(target_module, resampler_weight_list[-1], saved_weight)

    def init_vit(self):
        self.vision_tower = SiglipVisionTransformer(self.config.vision_config)
        self.init_tower_weight(self.vision_tower, self.weights, "vpm")
        setattr(self.vision_tower, "embed_dim", self.vision_tower.embeddings.embed_dim)
        setattr(self.vision_tower, "patch_size", self.vision_tower.embeddings.patch_size)

    def init_llm(self):
        self.model_type = self.config.model_type
        model_cls = get_llm_model("qwen2")
        self.language_model = model_cls(
            config=self.config.text_config,
            weights=self.weights,
            lmhead_prefix="llm.lm_head",
            model_prefix="llm.model",
            transformer_wte_parallel=False,
        )
        self.language_model.skip_word_embedding = True

    def init_resampler(self, embed_dim, vision_dim):
        self.resampler = Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )
        self.init_resamplerweight(self.resampler, self.weights)

    def init_multimodal(self):
        self.init_vit()
        self.init_llm()

    def get_input_embeddings(self):
        return self.language_model.transformer.wte

    def prepare_vision_embeds(self, input_ids, pixel_values, tgt_sizes):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # exist image
        if pixel_values:
            image_pixel_array = []
            image_pixel_array.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])
            image_pixel_array = torch.nn.utils.rnn.pad_sequence(image_pixel_array, batch_first=True, padding_value=0.0)
            batch_size, length, _ = image_pixel_array.shape
            image_pixel_array = image_pixel_array.permute(0, 2, 1).reshape(batch_size, 3, -1, length)
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])
            patch_attn_mask = torch.zeros((batch_size, 1, max_patches), dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True
            if batch_size > self.max_vision_bs:
                hs = []
                for i in range(0, batch_size, self.max_vision_bs):
                    start_idx = i
                    end_idx = i + self.max_vision_bs
                    tmp_hs = self.vision_tower(
                        image_pixel_array[start_idx:end_idx].type(self.dtype),
                        patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                        tgt_sizes=tgt_sizes[start_idx:end_idx],
                    ).last_hidden_state
                    hs.append(tmp_hs)
                image_features = torch.cat(hs, dim=0)
            else:
                image_features = self.vision_tower(
                    image_pixel_array.type(self.dtype), patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                ).last_hidden_state
            image_features = self.resampler(image_features, tgt_sizes)
            image_mask = input_ids == self.image_token_id
            inputs_embeds[image_mask] = image_features.view(-1, image_features.shape[-1])

        return inputs_embeds

    def prepare_prefill_token_service(self, input_ids):
        bos_pos = torch.where(torch.eq(input_ids, self.vision_start_token_id))[0]
        eos_pos = torch.where(torch.eq(input_ids, self.vision_end_token_id))[0]
        vision_num = bos_pos.shape[0]
        pixel_values = []
        tgt_sizes = []
        input_ids = input_ids.clone()
        for n in range(vision_num):
            single_input_ids = input_ids[bos_pos[n] : eos_pos[n] + 1]
            shm_index = torch.where(single_input_ids.lt(-1))[0]
            shm_tensor = single_input_ids[shm_index]
            shm_length = shm_index.size(0) // 2
            shm_name, shape_value = shm_tensor[0], shm_tensor[1]
            tgt_sizes.append(get_data_from_shm(shm_name, shape_value, np.int64, device=self.device))
            for i in range(shm_length - 1):
                shm_name = shm_tensor[2 + i * 2]
                shape_value = shm_tensor[2 + i * 2 + 1]
                image_pixel = get_data_from_shm(shm_name, shape_value, np.float32, device=self.device)
                pixel_values.append(image_pixel)
        bos_mask = torch.where(torch.eq(input_ids, self.vision_start_token_id))[0]
        eos_mask = torch.where(torch.eq(input_ids, self.vision_end_token_id))[0]
        pad_mask = torch.where(input_ids.lt(-1))[0]
        input_ids[bos_mask] = IMAGE_PAD
        input_ids[eos_mask] = IMAGE_PAD
        input_ids[pad_mask] = IMAGE_PAD
        return self.prepare_vision_embeds(input_ids, pixel_values, tgt_sizes)

    def prepare_prefill_token(self, multimodalinputs_list, processor):
        input_ids_list = []
        pixel_values = []
        tgt_sizes = []
        input_ids_shape = []
        if not isinstance(multimodalinputs_list, List):
            multimodalinputs_list = [multimodalinputs_list]
        for multimodalinputs in multimodalinputs_list:
            text = multimodalinputs.text
            image = multimodalinputs.image
            video = multimodalinputs.video
            if image is not None:
                image = safe_open_image(Image, image)
                text = [text]
                msgs, images = process_qs(text, image)
            elif video is not None:
                images = encode_video(video)
                msgs = [{"role": "user", "content": (IMAGE_PATTERN + SEP_PATTERN) * len(images) + text[MSG_CONTENT]}]
                images = [images]
            prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, images, return_tensors="pt").to(self.device)
            input_ids_list.append(inputs[INPUT_IDS])
            input_ids_shape.append(inputs[INPUT_IDS].shape[-1])
            pixel_values.extend(inputs[PIXEL_VALUES][0])
            tgt_sizes.extend(inputs[TGT_SIZES])
        input_ids = torch.cat(input_ids_list, dim=1)
        all_input_embeds = self.prepare_vision_embeds(input_ids, pixel_values, tgt_sizes).squeeze(0)
        all_input_embeds = all_input_embeds.split(input_ids_shape, dim=0)

        return all_input_embeds
