# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional
from enum import Enum
import importlib
import os
import json

from ..models.base.model_utils import safe_get_config_dict
from ..utils import file_utils
from ..utils.configuration_utils import LLMConfig
from ..utils.log import logger

from .base import tool_call_processors_registry

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conf', 'config.json')


class InferenceMode(int, Enum):
    REGRESSION = 0
    SPECULATE = 1
    SPLITFUSE = 2
    PREFIXCACHE = 3


class TruncationSide(int, Enum):
    DISABLE = 0
    LEFT = 1
    RIGHT = -1


def get_model(model_name_or_path: str,
              is_flash_causal_lm: bool = True,
              load_tokenizer: bool = True,
              max_position_embeddings: Optional[int] = None,
              revision: Optional[str] = None,
              tokenizer_path: Optional[str] = None,
              trust_remote_code: bool = False,
              enable_atb_torch: bool = False,
              enable_edge: bool = False,
              llm_config_path: str = None,
              models_dict: dict = None,
              sub_model_path: str = ""):
    model_name_or_path = file_utils.standardize_path(model_name_or_path, check_link=False)
    file_utils.check_path_permission(model_name_or_path)
    model_type_key = 'model_type'
    config_dict = safe_get_config_dict(model_name_or_path)
    config_dict[model_type_key] = config_dict[model_type_key].lower()
    model_type = config_dict[model_type_key]

    llm_config_path = llm_config_path if llm_config_path is not None else DEFAULT_CONFIG_PATH
    llm_config = LLMConfig(llm_config_path)

    if isinstance(models_dict, str):
        try:
            models_dict = json.loads(models_dict)
        except json.JSONDecodeError as e:
            message = "The 'models' field does not conform to JSON format. Please check."
            logger.warning(f'{message}, exception info: {e}')
            models_dict = None

    # models_dict model type convert
    if model_type in ["deepseek_v3", "deepseek_v2", "deepseekv3"] and isinstance(models_dict, dict):
        try:
            models_dict.update({'deepseekv2': models_dict.pop(model_type)})
        except KeyError as e:
            message = "The 'models' field does not contain {model_type}. Please check."
            logger.warning(f'{message}, exception info: {e}')

    llm_config.update(models_dict, allow_new_keys=True, current_path='models')

    model_type = router_model_type(model_type, config_dict, model_type_key, model_name_or_path)

    llm_config.merge_models_config(model_type)

    # 安全校验
    current_path = os.path.dirname(os.path.abspath(__file__))
    supported_models = []
    for foldername in file_utils.safe_listdir(current_path):
        is_folder = os.path.isdir(os.path.join(current_path, foldername))
        skip_base_folder = foldername != "base"
        skip_invalid_folder = not foldername.startswith("_")
        if is_folder and skip_base_folder and skip_invalid_folder:
            supported_models.append(foldername)

    if model_type not in supported_models:
        raise NotImplementedError(
            f"unsupported model type: {model_type}；"
            f"请确认atb_llm.models路径下是否存在名为{model_type}的文件夹。"
        )

    router_path = f"atb_llm.models.{model_type}.router_{model_type}"
    if model_type == "qwen2_moe" or model_type == "qwen3_moe":
        model_type = model_type.replace('_', '')
    if model_type == "qwen2_audio":
        model_type = model_type.replace('_', '')
    if model_type == "qwen2_vl":
        model_type = model_type.replace('_', '')
    if model_type == "qwen3_vl":
        model_type = model_type.replace('_', '')
    if model_type == "qwen3_vl_moe":
        model_type = model_type.replace('_', '')
    if model_type == "minicpm_qwen2_v2":
        model_type = model_type.replace('_', '')
    if model_type == "kimi_k2":
        model_type = model_type.replace('_', '')
    if model_type == "ernie_moe":
        model_type = model_type.replace('_', '')
    if model_type == "glm4_moe":
        model_type = model_type.replace('_', '')
    router = importlib.import_module(router_path)
    router_cls = getattr(router, f"{model_type.capitalize()}Router")
    router_ins = router_cls(
        model_name_or_path,
        config_dict,
        is_flash_causal_lm,
        load_tokenizer,
        max_position_embeddings,
        revision,
        tokenizer_path,
        trust_remote_code,
        enable_atb_torch,
        enable_edge,
        llm_config,
        sub_model_path)
    return router_ins


def router_model_type(model_type, config_dict, model_type_key, model_name_or_path):
    model_type_map = {
        "kclgpt": "codeshell",
        "internvl_chat": "internvl",
        "llava_next_video": "llava_next",
        "bunny-qwen2": "bunny",
        "bunny-minicpm": "bunny",
        'deepseek_v2': 'deepseekv2',
        'deepseek_v3': 'deepseekv2',
        "vita-qwen2": "vita",
        "qwen2_5_vl": "qwen2_vl",
        "ernie4_5_moe": "ernie_moe"
    }
    if model_type in model_type_map:
        model_type = model_type_map[model_type]
    elif model_type == "llava" and "_name_or_path" in config_dict.keys():
        if "yi-vl" in config_dict["_name_or_path"].lower():
            model_type = config_dict[model_type_key] = "yivl"
    elif model_type == "chatglm" and "vision_config" in config_dict:
        model_type = "glm4v"
    elif model_type == "glm4v":
        model_type = "glm41v"
    elif model_type == "minicpmv" and "MiniCPM-V-2_6" in model_name_or_path:
        model_type = "minicpm_qwen2_v2"
    elif model_type == 'multi_modality' and 'aligner_config' in config_dict:
        model_type = 'janus'
    return model_type