# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import importlib
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationConfig

from .input_builder import InputBuilder
from .model_utils import safe_get_tokenizer_from_pretrained, safe_get_config_dict
from .postprocessor import Postprocessor
from .reasoning_parser import ReasoningParser
from .tool_call_parser import ToolsCallProcessor, ToolParserManager
from ...utils.env import ENV
from ...utils.log import logger, message_filter
from ...utils.log.error_code import ErrorCode
from ...utils.configuration_utils import LLMConfig, GraphType
from ...utils import file_utils
from ...utils.parameter_validators import (
    IntParameterValidator, DictionaryParameterValidator, BooleanParameterValidator,
    RangeParamaterValidator, FileParameterValidator, StringParameterValidator, Field
)


GLM4_1V = "glm41v"
QWEN2 = "qwen2"
QWEN3 = "qwen3"
QWEN2_MOE = "qwen2_moe"
QWEN3_MOE = "qwen3_moe"
DEEPSEEKV2 = "deepseekv2"
LLAMA = "llama"
EOS_TOKEN_ID = 'eos_token_id'


def remove_part_of_generation_config(generation_config: dict) -> dict:
    """Using the transformers' GenerationConfig class, update the generation configuration with the default value."""
    ori_gen = GenerationConfig()
    for key in generation_config:
        if key.endswith("_id"):
            continue
        ori_value = getattr(ori_gen, key, None)
        if ori_value is not None:
            generation_config[key] = ori_value
    return generation_config


def check_value(attr_ins, value_ins, min_val, max_val):
    if value_ins < min_val or value_ins > max_val:
        raise ValueError(f"self._config.{attr_ins} must be between {min_val} and {max_val}")


def check_eos(eos_value, min_val, max_val):
    if isinstance(eos_value, int):
        check_value(EOS_TOKEN_ID, eos_value, min_val, max_val)
    elif isinstance(eos_value, list):
        for eos_v in eos_value:
            if isinstance(eos_v, int):
                check_value(EOS_TOKEN_ID, eos_v, min_val, max_val)
            elif isinstance(eos_v, list):
                for v in eos_v:
                    check_value(EOS_TOKEN_ID, v, min_val, max_val)
            else:
                raise ValueError("eos_token_id must be Union[int, List[Union[int, List[int]]]].")
    else:
        raise ValueError("eos_token_id must be Union[int, List[Union[int, List[int]]]].")


@dataclass
class BaseRouter:
    """The base class of router.

    This class should be inherited by the corresponding router subclasses of specified models. A specified model can use
    a subclass router to find its custom properties.
    """
    model_name_or_path: str = ""

    config_dict: Any = None
    is_flash_causal_lm: bool = True
    load_tokenizer: bool = True
    max_position_embeddings: Optional[int] = None
    revision: Optional[str] = None
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    enable_atb_torch: bool = False
    enable_edge: bool = False
    llm_config: LLMConfig = None

    sub_model_path: str = ""
    prealloc_weight_mem_on_npu: bool = False

    # 初始化默认读取的autoconfig，各个模型可能会自定义，self.config会返回后续使用的config，注意不要循环依赖
    _config: Any = None
    _generation_config: Any = None
    _input_builder: Any = None
    _model_cls: Any = None
    _postprocessor: Any = None
    _tokenizer: Any = None
    _custom_chat_template: Any = None
    is_inited: bool = False
    _tool_call_processor: Any = None
    _tool_call_parser: Any = None
    _reasoning_parser: Any = None

    def __post_init__(self):
        self.model_type = self.config_dict['model_type']
        if self.model_type == "glm4v":
            self.model_type = "glm41v"
        if self.model_type == "chatglm" and "vision_config" in self.config_dict:
            self.model_type = "glm4v"
        if self.model_type == "internvl_chat":
            self.model_type = "internvl"
        if self.model_type == "llava_next_video":
            self.model_type = "llava_next"
        if self.model_type == "bunny-qwen2" or self.model_type == "bunny-minicpm":
            self.model_type = "bunny"
        if "vita" in self.model_type:
            self.model_type = "vita"
        if self.model_type == "deepseek_v2" or self.model_type == "deepseek_v3":
            self.model_type = "deepseekv2"
        if self.model_type == "minicpmv" and "MiniCPM-V-2_6" in self.model_name_or_path:
            self.model_type = "minicpm_qwen2_v2"
        if self.model_type == "multi_modality" and 'aligner_config' in self.config_dict:
            self.model_type = "janus"
        if self.model_type == "qwen2_5_vl":
            self.model_type = "qwen2_vl"
        if self.model_type == "ernie4_5_moe":
            self.model_type = "ernie_moe"
        self.model_type_cap = self.model_type.capitalize()
        if self.model_type_cap == "Qwen2_moe":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Qwen2_audio":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Qwen2_vl":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Minicpm_qwen2_v2":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Ernie_moe":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if self.model_type_cap == "Glm4_moe":
            self.model_type_cap = self.model_type_cap.replace('_', '')
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_name_or_path

        if self.llm_config is not None:
            # Python graph construction is deprecated. Force reset to C++ graph if detected.
            if self.llm_config.llm.engine.graph == GraphType.PYTHON:
                logger.warning(
                    "Python Graph is deprecated, "
                    "config field llm.engine.graph will be removed on 2027/3/31."
                    "Automatically falling back to C++ graph."
                )
                self.llm_config.llm.engine.graph = GraphType.CPP
            llm_config_validators = self.get_llm_config_validators()
            self.llm_config.check_config(llm_config_validators)
            self._check_llm_config_model_type(self.model_type)
            self._check_llm_config_orthogonality()
            atb_llm_lcoc_enable_env = os.getenv("ATB_LLM_LCOC_ENABLE")
            if atb_llm_lcoc_enable_env is not None:
                atb_llm_lcoc_enable = message_filter(atb_llm_lcoc_enable_env) == "1"
                if atb_llm_lcoc_enable != self.llm_config.llm.ccl.enable_mc2:
                    logger.warning(
                        "User has defined $ATB_LLM_LCOC_ENABLE environment variable to %s. "
                        "And the $llm_config.llm.ccl.enable_mc2 is %s in LLM's config. "
                        "If one of them is False, the LCOC will not be enabled. "
                        "NOTE: The $ATB_LLM_LCOC_ENABLE environment variable will be deprecated after 2026/07/31.",
                        atb_llm_lcoc_enable, self.llm_config.llm.ccl.enable_mc2)

            self.process_tool_call_options()
        
    @property
    def config(self):
        """The config property, which should not be overridden.

        It uses generation config to update config dictionary at first, and then uses the `get_config` method to get a
        config object. Note that the `get_config` method should use `config_dict` to initialize the config object.
        """
        quantize = "quantize"
        is_nzcasted = "is_nzcasted"
        AtlasGMMPermute = "AtlasGMMPermute"
        if self._config is None:
            self._generation_config = self.generation_config
            if ENV.remove_generation_config_dict:
                self._generation_config = remove_part_of_generation_config(self._generation_config)
            self.config_dict.update(self._generation_config)
            self.config_dict["generation_config"] = self._generation_config
            filename = os.path.join(self.model_name_or_path, 'quant_model_description.json')
            if not file_utils.is_path_exists(filename):
                filename = os.path.join(self.model_name_or_path, self.sub_model_path, 'quant_model_description.json')
            if not file_utils.is_path_exists(filename):
                quantize_type = self.config_dict.get(quantize, None)
                if quantize_type:
                    old_file = f'quant_model_description_{quantize_type.lower()}.json'
                    old_path = os.path.join(self.model_name_or_path, old_file)
                    if file_utils.is_path_exists(old_path):
                        filename = old_path
                    else:
                        old_path_sub = os.path.join(self.model_name_or_path, self.sub_model_path, old_file)
                        if file_utils.is_path_exists(old_path_sub):
                            filename = old_path_sub
            if not file_utils.is_path_exists(filename):
                filename = None
            if filename is not None:
                with file_utils.safe_open(filename, 'r', check_link=False) as f:
                    quant_descs = json.load(f)
                if quant_descs.get("model_quant_type") is not None and self.config_dict.get("quantize") is None:
                    self.config_dict[quantize] = quant_descs.get("model_quant_type").lower()
                if quant_descs.get(is_nzcasted) is not None:
                    self.config_dict[is_nzcasted] = quant_descs.get(is_nzcasted)
                if quant_descs.get(AtlasGMMPermute) is not None:
                    self.config_dict[AtlasGMMPermute] = quant_descs.get(AtlasGMMPermute)
                # To make the least amount of modification,
                # convert the new version `w8a8_mix` to the old version `w8a8_pdmix`.
                if self.config_dict[quantize] == "w8a8_mix":
                    self.config_dict[quantize] = "w8a8_pdmix"
                self.config_dict.setdefault("quantization_config", {}).update(quant_descs)

            self._config = self.get_config()
            if not hasattr(self._config, quantize):
                setattr(self._config, quantize, None)
            if self.max_position_embeddings is not None:
                setattr(self._config, 'max_position_embeddings', self.max_position_embeddings)
        return self._config

    @property
    def generation_config(self):
        """The generation config property, which should not be overridden."""
        if self._generation_config is None:
            self._generation_config = self.get_generation_config()
        return self._generation_config

    @property
    def input_builder(self):
        """The input builder property, which should not be overridden."""
        if self._input_builder is None:
            self._input_builder = self.get_input_builder()
        return self._input_builder

    @property
    def model_cls(self):
        """The model class property, which should not be overridden."""
        if self._model_cls is None:
            self._model_cls = self.get_model_cls()
        return self._model_cls

    @property
    def model_version(self):
        """The model version property, which should not be overridden."""
        return ""

    @property
    def embedding_model_name(self):
        """The model name property, which should not be overridden."""
        return ""

    @property
    def postprocessor(self):
        """The postprocessor property, which should not be overridden."""
        if self._postprocessor is None:
            self._postprocessor = self.get_postprocessor()
        return self._postprocessor

    @property
    def tokenizer(self):
        """The tokenizer property, which should not be overridden."""
        if self._tokenizer is None and self.load_tokenizer:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    @property
    def tool_call_parser(self):
        """The tool call parser property, which should not be overridden."""
        if self._tool_call_parser is None:
            self._tool_call_parser = self.get_tool_call_parser()
        return self._tool_call_parser

    @property
    def custom_chat_template(self):
        """The custom chat template property, which should not be overridden."""
        if self._custom_chat_template is None \
                and self.llm_config is not None \
                and getattr(self.llm_config.llm, "chat_template", None):
            self._custom_chat_template = self.get_custom_chat_template()
        return self._custom_chat_template

    @property
    def toolscallprocessor(self):
        """The tools call processor property, which should not be overridden."""
        if self._tool_call_processor is None:
            self._tool_call_processor = self.get_toolscallprocessor()
        return self._tool_call_processor

    @property
    def reasoning_parser(self):
        """The reasoning processor property, which should not be overridden."""
        if self._reasoning_parser is None:
            self._reasoning_parser = self.get_reasoning_parser()
        return self._reasoning_parser

    @staticmethod
    def check_config(config):
        """The validation of values in config."""

        vocab_size = 0
        vocab_size_field = 'vocab_size'
        if hasattr(config, vocab_size_field):
            vocab_size = getattr(config, vocab_size_field)
        attribute_ranges = {
            vocab_size_field: (1, 2147483647),
            'max_position_embeddings': (1, 2147483647),
            'hidden_size': (1, 2147483647),
            'intermediate_size': (1, 2147483647),
            'num_hidden_layers': (1, 1000),
            'num_attention_heads': (1, 10000),
            'initializer_range': (0, 2147483647),
            'rms_norm_eps': (0, 1),
            'pad_token_id': (-1, vocab_size),
            'bos_token_id': (0, vocab_size - 1),
            EOS_TOKEN_ID: (0, vocab_size - 1),
            'temperature': (0, 2),
            'top_k': (-1, vocab_size),
            'top_p': (0, 1),
            'repetition_penalty': (0, 2),
            'frequency_penalty': (-2, 2),
            'presence_penalty': (-2, 2),
            'max_new_tokens': (0, 10000000),
            'max_length': (0, 2147483647),         
            'user_token_id': (0, vocab_size-1),   
            'assistant_token_id': (0, vocab_size-1),    
            'system_token_id': (0, vocab_size-1),   
            'bot_token_id': (0, vocab_size-1)
        }
        if config.is_reasoning_model:
            attribute_ranges["start_reasoning_token_id"] = (0, vocab_size - 1)
            attribute_ranges["end_reasoning_token_id"] = (0, vocab_size - 1)
        if hasattr(config, "head_dim"):
            attribute_ranges['head_dim'] = (1, 1000)
        if hasattr(config, "num_key_value_heads"):
            attribute_ranges['num_key_value_heads'] = (1, 10000)
  
        gen_config = config.generation_config
        gen_config_dict = vars(gen_config)
        
        for attr, value in gen_config_dict.items():
            if value is not None:
                min_val, max_val = attribute_ranges.get(attr, (None, None))
                if attr == EOS_TOKEN_ID:
                    if min_val is not None and max_val is not None:
                        check_eos(value, min_val, max_val)
                    continue
                if attr in attribute_ranges:
                    if min_val is not None and max_val is not None:
                        check_value(attr, value, min_val, max_val)

        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if attr == EOS_TOKEN_ID:
                check_eos(value, min_val, max_val)
                continue
            check_value(attr, value, min_val, max_val)

        if getattr(config, 'repetition_penalty', None) == 0:
            raise ValueError("repetition_penalty should not be 0.")
        if not isinstance(getattr(config, 'do_sample', None), bool):
            raise ValueError("do_sample must be bool.")

    @classmethod
    def get_llm_config_validators(cls):
        llm_config_validators = {"models": {}}
        llm_validators = DictionaryParameterValidator({
            "ccl": DictionaryParameterValidator({
                "enable_mc2": BooleanParameterValidator()
            }),
            "stream_options": DictionaryParameterValidator({
                "micro_batch": BooleanParameterValidator()
            }),
            "engine": DictionaryParameterValidator({
                "graph": RangeParamaterValidator(range_list=[GraphType.CPP, GraphType.PYTHON])
            }),
            "parallel_options": DictionaryParameterValidator({
                "o_proj_local_tp": IntParameterValidator(Field(ge=1, le=16), special_values=[-1]),
                "lm_head_local_tp": IntParameterValidator(Field(ge=1, le=16), special_values=[-1]),
                "hccl_buffer": IntParameterValidator(Field(ge=1)),
                "hccl_moe_ep_buffer": IntParameterValidator(Field(ge=512)),
                "hccl_moe_tp_buffer": IntParameterValidator(Field(ge=64)),
            }),
            "pmcc_obfuscation_options": DictionaryParameterValidator({
                "enable_model_obfuscation": BooleanParameterValidator(),
                "data_obfuscation_ca_dir": FileParameterValidator(allow_none=True),
                "kms_agent_port": IntParameterValidator(Field(ge=1, le=65535)),
            }),
            "kv_cache_options": DictionaryParameterValidator({
                "enable_nz": BooleanParameterValidator()
            }),
            "weights_options": DictionaryParameterValidator(
                element_validator_mapping={
                    "low_cpu_memory_mode": BooleanParameterValidator()
                },
                allow_addition_key=False
            ),
            "enable_reasoning": BooleanParameterValidator(),
            "tool_call_options": DictionaryParameterValidator({
                "tool_call_parser": StringParameterValidator(Field(max_length=1024), allow_none=True),
            }),
            "chat_template": FileParameterValidator(allow_none=True)
        })
        llm_config_validators["llm"] = llm_validators
        return llm_config_validators

    def process_tool_call_options(self):
        valid_tool_call_processors = ToolParserManager.get_tool_call_processors()
        debug_msg = f"The valid registered ToolsCallPRocessors are: {valid_tool_call_processors}"
        logger.debug(message_filter(debug_msg))

        tool_call_options = getattr(self.llm_config.llm, "tool_call_options", None)
        if not tool_call_options:
            return

        tool_call_parser_name = getattr(tool_call_options, "tool_call_parser", None)
        if tool_call_parser_name:
            if tool_call_parser_name not in valid_tool_call_processors:
                warn_msg = \
                    f"Invalid tool_call_parser: {tool_call_parser_name}. " \
                    f"Please chose from {valid_tool_call_processors.keys()}"
                logger.warning(message_filter(warn_msg))
                logger.warning("Will use default tool_call_parser of model.")
                return
            # If the given tool_call_parser is in the registered ToolsCallProcessors, use it, ohterwise left it null.
            debug_msg = f"Using select tool_call_parser {tool_call_parser_name}"
            logger.debug(message_filter(debug_msg))
            self._tool_call_parser = tool_call_parser_name

    def tokenize(self, inputs: List[Union[str, Dict[str, str]]], **kwargs) -> np.ndarray:
        """Transfer text input or multimodal input to token ids.

        Args:
            inputs: List | List[Dict], when it's List, it means the input for LLM.
                When it's List[Dict], it means the multimodal inputs in interleaved style,
                for example:
                    [
                        {'text': 'Let me show you two pictures'},
                        {'image': 'image_url_or_path'},
                        {'image': 'image_url_or_path'},
                        {'text': 'can you show the differences?'}
                    ]

        Returns:
            numpy.ndarray: The expanded input_ids whose dimension is 1.
        """
        return self.tokenizer([inputs[0]["text"]], return_tensors="np")["input_ids"][0]

    def get_config(self):
        """The default method to get config.

        A subclass router can override it to define a custom method getting config. Note that the `get_config` method
        should use `self.config_dict` instead of the model weight path to construct a config object.
        """
        try:
            config_cls = self.get_config_cls()
            config = config_cls.from_dict(self.config_dict)
        except Exception as e:
            logger.warning(str(e))
            config = PretrainedConfig.from_dict(self.config_dict)
        self.check_config(config)
        return config

    def get_generation_config(self):
        """The default method to get generation config."""
        generation_config_path = os.path.join(self.model_name_or_path, "generation_config.json")
        generation_config = {}
        if os.path.exists(generation_config_path):
            generation_config = safe_get_config_dict(generation_config_path)
        return generation_config

    def get_config_cls(self):
        """The default method to get config class."""
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        if self.model_version:
            model_file_dir_name = model_file_dir_name + \
                                  f"{self.model_version}."
        config_file_name = f'config_{self.model_type}'
        module_path = f"{model_file_dir_name}{config_file_name}"
        module = importlib.import_module(module_path)
        config_cls_name = f"{self.model_type_cap}Config"
        return getattr(module, config_cls_name)

    def get_custom_chat_template(self):
        """The default method to get chat template."""
        custom_chat_template_file = getattr(self.llm_config.llm, "chat_template")
        chat_template = ""
        if custom_chat_template_file and file_utils.is_path_exists(custom_chat_template_file):
            try:
                with file_utils.safe_open(custom_chat_template_file, "r", encoding="utf-8") as f:
                    chat_template = f.read()
            except Exception as e:
                warn_msg = f"Failed to load given chat template '{custom_chat_template_file}': {e}"
                logger.warning(message_filter(warn_msg))
                logger.warning("Will still use default chat template of tokenizer.")
        return chat_template

    def get_input_builder(self):
        """The default method to get input builder."""
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return InputBuilder(self.tokenizer, **kwargs)

    def get_model_cls(self):
        """The default method to get model class.

        This is a basic router method to find model class, which is usually not necessary to be overridden.
        """
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        enable_v3 = False
        if self.llm_config is not None and self.llm_config.llm.engine.graph == GraphType.PYTHON:
            enable_v3 = True
        if self.model_version:
            model_file_dir_name = model_file_dir_name + \
                                  f"{self.model_version}."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        if self.embedding_model_name:  # for embedding model, example: gte-qwen2
            module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}_{self.embedding_model_name}"
        else:
            module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}"

        if self.enable_atb_torch:
            module_path += "_atb"
        if self.enable_edge:
            module_path += "_edge"
        if enable_v3:
            module_path += "_v3"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self.model_type_cap}ForCausalLM"

        if self.enable_atb_torch:
            model_cls_name += "ATB"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        if enable_v3:
            model_cls_name += "V3"
        return getattr(module, model_cls_name)

    def get_postprocessor(self):
        """The default method to get postprocessor."""
        valid_postprocessor = Postprocessor(self.tokenizer, self.config.generation_config)
        check_eos(valid_postprocessor.eos_token_id, 0, self.config.vocab_size - 1)
        return valid_postprocessor

    def get_tokenizer(self):
        """The default method to get tokenizer."""
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            revision=self.revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.trust_remote_code,
            use_fast=True
        )

    def get_reasoning_parser(self):
        """The default method to get reasoning parser."""
        return ReasoningParser(self.config)

    def get_tool_call_parser(self):
        """The default method to get tool call parser."""
        tool_call_parser = self.model_type
        if self.model_version:
            tool_call_parser += f"_{self.model_version}"
        return tool_call_parser

    def get_toolscallprocessor(self):
        """The default method to get tools call processor."""
        try:
            tools_call_processor = ToolParserManager.get_tool_call_processor(self.tool_call_parser)(self.tokenizer)
        except KeyError as e:
            warn_msg = \
                f"Cannot use {self.tool_call_parser} ToolsCallProcessor: {e}. " \
                f"Will use default ToolsCallProcessor."
            logger.warning(message_filter(warn_msg))
            tools_call_processor = ToolsCallProcessor(self.model_version)
        return tools_call_processor
    
    def _check_llm_config_model_type(self, model_type):
        """check the compatibility between llm_config and model_type"""
        if self.llm_config.llm.stream_options.micro_batch and model_type not in [QWEN2, QWEN3, DEEPSEEKV2, GLM4_1V]:
            error_msg = f"stream_options.micro_batch only supports [qwen2, qwen3, deepseekv2, glm4.1v], " \
                        f"but model_type is {model_type}, please refer to the MindIE official document" \
                        f"and modify config.json."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)
        
        if self.llm_config.llm.engine.graph == GraphType.PYTHON and model_type not in [LLAMA, QWEN2, QWEN3]:
            error_msg = f"engine.graph only supports [llama, qwen2, qwen3] when set to \"python\", " \
                        f"but model_type is {model_type}, please refer to the MindIE official document" \
                        f"and modify config.json."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)
        
        if self.llm_config.llm.kv_cache_options.enable_nz and model_type not in [DEEPSEEKV2]:
            error_msg = f"kv_cache_options.enable_nz only supports [deepseekv2] when set to true, " \
                        f"but model_type is {model_type}, please refer to the MindIE official document" \
                        f"and modify config.json."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(error_msg)

        # The following features are configured but are not applicable to some models.
        if self.llm_config.llm.parallel_options.o_proj_local_tp != -1 and model_type != DEEPSEEKV2:
            logger.warning(f"parallel_options.o_proj_local_tp only supports deepseek, {model_type} won't be affected.")
        
        if self.llm_config.llm.parallel_options.lm_head_local_tp != -1 and model_type != DEEPSEEKV2:
            logger.warning(f"parallel_options.lm_head_local_tp only supports deepseek, " \
                        f"{model_type} won't be affected.")

        weights_options = self.llm_config.get("llm").get("weights_options")
        if weights_options is not None and weights_options.low_cpu_memory_mode:
            engine_graph = self.llm_config.get("llm").get("engine").get("graph")
            if engine_graph != GraphType.PYTHON or model_type not in [QWEN2, QWEN3]:
                logger.warning(f"low_cpu_memory_mode is only effective for [qwen2, qwen3] " \
                            f"when engine graph is set to \"python\". " \
                            f"For {model_type} with engine graph {engine_graph}, this mode will not work.")

        if self.llm_config.llm.enable_reasoning and model_type not in [QWEN3, QWEN3_MOE, DEEPSEEKV2]:
            logger.warning(f"enable_reasoning only supports [qwen3, qwen3_moe, deepseek] when set to true, " \
                        f"{model_type} won't be affected.")

    def _check_llm_config_orthogonality(self):
        """check the orthogonality within llm_config"""
        if self.llm_config.llm.stream_options.micro_batch and self.llm_config.llm.ccl.enable_mc2:
            raise ValueError("ccl.enable_mc2 and stream_options.micro_batch can't be both true.")