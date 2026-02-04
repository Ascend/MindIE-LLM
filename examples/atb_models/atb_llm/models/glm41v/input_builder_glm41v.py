# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#          http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import copy
import itertools
from typing import Dict, List
import numpy as np
import torch
import PIL

from atb_llm.utils.shm_utils import process_shared_memory, get_data_from_shm
from atb_llm.utils.multimodal_utils import safe_open_image, check_video_path
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger
from atb_llm.models.base.input_builder import InputBuilder

CONTENT = "content"
IMAGE = "image"
VIDEO = "video"
TEXT = "text"
TYPE = "type"
URL = "url"
_SHM_TOKEN_LEN = 8

template = {
        "role": "",
        "content": [],
    }


class Glm41vInputBuilder(InputBuilder):
    def __init__(self, tokenizer, config, processor, **kwargs):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        super().__init__(tokenizer, **kwargs)
        self.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.image_token_id = self.config.image_token_id
        self.image_start_token_id = self.config.image_start_token_id
        self.image_end_token_id = self.config.image_end_token_id
        self.video_start_token_id = self.config.video_start_token_id
        self.video_end_token_id = self.config.video_end_token_id

    @staticmethod
    def get_shm_name_save_path(file_path: str):
        """
        Generates a file path for storing shared memory names by navigating up two directory levels.

        Args:
            file_path: The original file path from which to derive the shared memory save path
        Returns:
            str: A full file path pointing to 'shm_name.txt' located two directory levels
                above the input file_path 
        Example:
            If file_path is '/project/data/inputs/current/file.jpg'
            Returns: '/project/data/shm_name.txt'
        """
        shm_name_save_dir = os.path.dirname(os.path.dirname(file_path))
        shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
        return shm_name_save_path

    @staticmethod
    def check_image_and_video_concurrency(content: List[Dict]):
        """
        Check for concurrent presence of image and video content in the same prompt.

        Validates that the content list does not contain both image and video entries.

        Args:
            content: List of dictionaries representing content items. Each dictionary may contain
                    the image or video keys to indicate content type.

        Raises:
            ValueError: When both image and video content types are detected in the same prompt.
        """
        if any(item.get(IMAGE) for item in content) \
            and any(item.get(VIDEO) for item in content):
            msg = "Image and video cannot exist in single prompt at the same time."
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(msg)

    def generate_position_ids(self, input_ids):
        position_ids = np.arange(len(input_ids), dtype=np.int64)
        if np.any(np.equal(input_ids, self.image_start_token_id)):
            boi_pos = np.where(np.equal(input_ids, self.image_start_token_id))[0][0].item()
            image_grid_thw_shm_name = input_ids[boi_pos + 3]
            image_grid_thw_shape_value = input_ids[boi_pos + 4]
            video_grid_thw_shm_name = input_ids[boi_pos + 7]
            video_grid_thw_shape_value = input_ids[boi_pos + 8]
            image_grid_thw = get_data_from_shm(image_grid_thw_shm_name, image_grid_thw_shape_value, np.int32) \
                if image_grid_thw_shm_name else None
            video_grid_thw = get_data_from_shm(video_grid_thw_shm_name, video_grid_thw_shape_value, np.int32) \
                if video_grid_thw_shm_name else None
            image_index, video_index = 0, 0
            video_group_index = 0
            input_tokens = input_ids.copy()
            for i in range(_SHM_TOKEN_LEN):
                input_tokens[boi_pos + i + 1] = self.image_token_id
            input_type_group = self._get_token_type(input_tokens)
            video_frame_num = 1
            st_idx = 0
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == IMAGE:
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        image_grid_thw[image_index][0].item(),
                        image_grid_thw[image_index][1].item() // self.spatial_merge_size,
                        image_grid_thw[image_index][2].item() // self.spatial_merge_size,
                    )
                    st_idx += max(llm_grid_t, llm_grid_h, llm_grid_w)
                    image_index += 1
                    video_frame_num = 1
                elif modality_type == VIDEO:
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        video_frame_num,
                        video_grid_thw[video_index][1].item() // self.spatial_merge_size,
                        video_grid_thw[video_index][2].item() // self.spatial_merge_size,
                    )
                    st_idx += max(llm_grid_t, llm_grid_h, llm_grid_w)
                    video_group_index += 1
                    if video_group_index >= video_grid_thw[video_index][0]:
                        video_index += 1
                        video_group_index = 0
                    video_frame_num += 1
                else:
                    text_len = end_idx - start_idx
                    st_idx += text_len
                    video_frame_num = 1
            position_ids[-1] = st_idx - 1
        return position_ids

    def make_context(
        self,
        rank: int,
        conversation: List[Dict[str, List[Dict]]],
        **kwargs
    ):
        if isinstance(conversation[0][CONTENT], str):
            for item in conversation:
                item[CONTENT] = [{TEXT: item[CONTENT]}]
        elif not isinstance(conversation[0][CONTENT], list):
            err_msg = "The `conversation` \"content\" should be a List[Dict] or str."
            logger.error(err_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise TypeError(err_msg)
        shm_name_save_path = kwargs.get('shm_name_save_path', None)
        conversation_list = [] 
        for single_conversation in conversation:
            content = single_conversation[CONTENT]
            self.check_image_and_video_concurrency(content)
            message_list = []
            for item in content:
                if item.get(IMAGE, None):
                    img_fname = item[IMAGE]
                    img = safe_open_image(PIL.Image, img_fname)
                    message_list.append({
                        TYPE: IMAGE,
                        IMAGE: img
                    })
                    if shm_name_save_path is None:
                        shm_name_save_path = self.get_shm_name_save_path(img_fname)
                if item.get(VIDEO, None):
                    video_path = check_video_path(item[VIDEO])
                    message_list.append({
                        TYPE: VIDEO,
                        URL: video_path
                    })
                    if shm_name_save_path is None:
                        shm_name_save_path = self.get_shm_name_save_path(video_path)
                if item.get(TEXT, None):
                    text = item[TEXT]
                    message_list.append({
                        TYPE: TEXT,
                        TEXT: text
                    })
            conversation_list.append({"role": single_conversation["role"], CONTENT: copy.deepcopy(message_list)})
        inputs = self.processor.apply_chat_template(
            conversation_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt'
        )
        shm_info = process_shared_memory(inputs, shm_name_save_path)
        input_ids = self.update_token_id(inputs, shm_info)
        return input_ids
    
    def update_token_id(self, inputs: Dict[str, torch.Tensor], shm_info: Dict):
        """
        Replaces special image tokens in the input sequence with shared memory metadata.

        This method identifies the position between image start/end tokens and replaces
        the subsequent tokens with encoded shared memory information (names and shapes)
        for both image and video data. This allows the model to reference external
        memory segments containing large tensor data instead of carrying the full data
        in the token sequence.

        Args:
            inputs: Dictionary containing the input token sequence with key 'input_ids'
            shm_info: Dictionary containing encoded shared memory metadata from 
                    process_shared_memory(), including shm names and shape values
                    for pixel values and grid dimensions
        Returns:
            Modified input_ids tensor with shared memory references inserted between
            the image start and end token positions.
        Raises:
            RuntimeError: If insufficient space exists between BOI and EOI tokens for
                    all required metadata entries (8 positions needed)
        """
        boi_token_id = self.image_start_token_id
        eoi_token_id = self.image_end_token_id
        input_ids = inputs['input_ids'].flatten()
        if not torch.any(torch.eq(input_ids, self.image_token_id)):
            return input_ids
        boi_pos = torch.where(torch.eq(input_ids, boi_token_id))[0][0].item()
        eoi_pos = torch.where(torch.eq(input_ids, eoi_token_id))[0][0].item()
        if eoi_pos - boi_pos - 1 < len(shm_info):
            err_msg = "Load share memory info to input ids failed."
            logger.error(err_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise RuntimeError(err_msg)
        input_ids[boi_pos + 1] = shm_info['pixel_values_shm_name']
        input_ids[boi_pos + 2] = shm_info['pixel_values_shape_value']
        input_ids[boi_pos + 3] = shm_info['image_grid_thw_shm_name']
        input_ids[boi_pos + 4] = shm_info['image_grid_thw_shape_value']
        input_ids[boi_pos + 5] = shm_info['pixel_values_videos_shm_name']
        input_ids[boi_pos + 6] = shm_info['pixel_values_videos_shape_value']
        input_ids[boi_pos + 7] = shm_info['video_grid_thw_shm_name']
        input_ids[boi_pos + 8] = shm_info['video_grid_thw_shape_value']
        return input_ids
    
    def _get_token_type(self, input_tokens):
        """
        Analyzes and groups input tokens by their semantic type (TEXT, IMAGE, VIDEO).

        This method categorizes tokens in the input sequence and groups consecutive tokens
        of the same type together, creating segments that represent different modalities
        in a multimodal input sequence.

        Args:
            input_tokens: Array or list of token IDs representing the input sequence
        Returns:
            List of tuples where each tuple contains:
            - token_type: The type of tokens in the segment (TEXT/IMAGE/VIDEO)
            - start_index: Starting index of the segment in the input sequence
            - end_index: Ending index (exclusive) of the segment
        Example Output:
            [(TEXT, 0, 5), (IMAGE, 5, 7), (TEXT, 7, 10), (VIDEO, 10, 15), (TEXT, 15, 20)]

        """
        input_token_type = np.full(len(input_tokens), TEXT, dtype=object)
        img_mask = np.equal(input_tokens, self.image_token_id)
        input_token_type[img_mask] = IMAGE
        if np.any(np.equal(input_tokens, self.video_start_token_id)):
            video_starts = np.where(input_tokens == self.video_start_token_id)[0]
            video_stops = np.where(input_tokens == self.video_end_token_id)[0]
            for start, stop in zip(video_starts, video_stops):
                video_mask = np.equal(input_tokens[start: stop + 1], self.image_token_id)
                input_token_type[start: stop + 1][video_mask] = VIDEO
        input_token_type = input_token_type.tolist()
        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))
        return input_type_group