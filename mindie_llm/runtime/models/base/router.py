# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
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
from dataclasses import dataclass
from typing import Any
from functools import cached_property
import json

from mindie_llm.runtime.utils.helpers.safety.file import safe_open
from mindie_llm.runtime.utils.helpers.safety.hf import safe_get_tokenizer_from_pretrained, safe_get_config_dict
from mindie_llm.runtime.utils.helpers.parameter_validators import (
    IntParameterValidator, DictionaryParameterValidator, BooleanParameterValidator,
    RangeParamaterValidator, FileParameterValidator, StringParameterValidator, Field
)
from mindie_llm.runtime.models.base.input_builder import InputBuilder
from mindie_llm.runtime.models.base.reasoning_parser import CommonReasoningParser
from mindie_llm.runtime.models.base.tool_calls_processor import ToolCallsProcessor, ToolCallsProcessorManager 
from mindie_llm.runtime.config.configuration_utils import LLMConfig
from mindie_llm.runtime.config.huggingface_config import GenerationConfig
from mindie_llm.runtime.config.load_config import LoadConfig
from mindie_llm.utils.log.logging import logger, message_filter



DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'conf', 'config.json'
)


@dataclass
class BaseRouter:
    """The base class of router.

    This class should be inherited by the corresponding router subclasses of specified models. A specified model can use
    a subclass router to find its custom properties.

    Key features:
    - Lazy loading pattern for configuration and resources (via properties)
    - Comprehensive configuration validation
    - Support for tool calling and reasoning capabilities
    - Integration with safety mechanisms for file operations
    """
    config_dict: dict | None = None
    load_config: LoadConfig | None = None

    def __post_init__(self):
        """Post-initialization processing.

        - Sets model type from config_dict
        - Capitalizes model type for class naming
        - Processes tool call options
        """
        self._model_type = self.config_dict['model_type']
        self._model_type_cap = ''.join(part.capitalize() for part in self._model_type.split('_'))

    @classmethod
    def get_llm_config_validators(cls) -> DictionaryParameterValidator:
        """Get validators for LLM runtime configuration.

        Returns:
            Dictionary of validators for LLMConfig parameters
        """
        llm_config_validators = {"models": {}}
        llm_validators = DictionaryParameterValidator({
            "ccl": DictionaryParameterValidator({
                "enable_mc2": BooleanParameterValidator()
            }),
            "stream_options": DictionaryParameterValidator({
                "micro_batch": BooleanParameterValidator()
            }),
            "engine": DictionaryParameterValidator({
                "graph": RangeParamaterValidator(range_list=["cpp", "python"])
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

    @cached_property
    def llm_config(self):
        return self._get_llm_config()

    @cached_property
    def config(self):
        """Cached property for the model configuration object.

        This property returns a validated configuration object of the model.
        It uses the configuration class obtained from `_get_config_cls` to create
        the configuration object from `config_dict`. The configuration is computed
        once and cached for subsequent access.

        Note: This property should not be overridden in subclasses.

        Returns:
            HuggingFaceConfig: A validated configuration object for the model
        """
        self.config_dict.update(self._get_generation_config_dict())
        config_cls = self._get_config_cls()
        return config_cls.from_dict(self.config_dict)
         
    @cached_property
    def generation_config(self):
        """The generation config property, which should not be overridden.

        Returns:
            GenerationConfig object containing generation parameters
        """
        generation_config_dict = self._get_generation_config_dict()
        generation_config = GenerationConfig.from_dict(generation_config_dict)
        generation_config.validate(self.config.vocab_size)
        return generation_config

    @cached_property
    def input_builder(self):
        """The input builder property, which should not be overridden.

        Returns:
            InputBuilder instance for constructing model inputs
        """
        return self._get_input_builder()

    @cached_property
    def model_cls(self):
        """The model class property, which should not be overridden.

        Returns:
            Reference to the model class (e.g., QwenForCausalLM)
        """
        return self._get_model_cls()

    @cached_property
    def draft_cls(self):
        """The draft class property, which should not be overridden."""
        return self._get_draft_cls()

    @cached_property
    def tokenizer(self):
        """The tokenizer property, which should not be overridden.

        Returns:
            Tokenizer instance for text processing
        """
        tokenizer = None
        if self.load_config.load_tokenizer:
            tokenizer = self._get_tokenizer()
        return tokenizer


    @cached_property
    def tool_call_parser(self):
        """The tool call parser property, which should not be overridden.

        Returns:
            Name of the tool call parser to use
        """
        return self._get_tool_calls_parser()

    @cached_property
    def custom_chat_template(self):
        """The custom chat template property, which should not be overridden.

        Returns:
            Custom chat template string if configured, otherwise None
        """
        custom_chat_template = ""
        if getattr(self.llm_config.llm, "chat_template", ""):
            custom_chat_template = self._get_custom_chat_template()
        return custom_chat_template

    @cached_property
    def tool_calls_processor(self):
        """The tools call processor property, which should not be overridden.

        Returns:
            Tools call processor instance for handling tool calls
        """
        self._get_tool_calls_options()
        return self._get_tool_calls_processor()

    @cached_property
    def reasoning_parser(self):
        """The reasoning processor property, which should not be overridden.

        Returns:
            Reasoning parser instance for processing reasoning tokens
        """
        return self._get_reasoning_parser()

    def _get_tool_calls_options(self) -> None:
        """Process tool call options from llm_config.

        Validates the specified tool_call_parser against registered processors and sets up
        the appropriate tool call handling mechanism.
        """
        valid_tool_call_processors = ToolCallsProcessorManager.get_tool_calls_processors()
        debug_msg = f"The valid registered ToolCallsProcessors are: {valid_tool_call_processors}"
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
            # If the given tool_call_parser is in the registered ToolCallsProcessors, use it, otherwise left it null.
            debug_msg = f"Using select tool_call_parser {tool_call_parser_name}"
            logger.debug(message_filter(debug_msg))
            self.tool_call_parser = tool_call_parser_name

    def _get_config_cls(self) -> "HuggingFaceConfig":
        """The default method to get config class.

        Dynamically imports the configuration class based on model type.

        Returns:
            Configuration class for the specific model type
        """
        model_file_dir_name = f"mindie_llm.runtime.models.{self._model_type}."
        config_file_name = f'config_{self._model_type}'
        module_path = f"{model_file_dir_name}{config_file_name}"
        module = importlib.import_module(module_path)
        config_cls_name = f"{self._model_type_cap}Config"
        return getattr(module, config_cls_name)

    def _get_custom_chat_template(self) -> str:
        """The default method to get chat template.

        Attempts to load custom chat template from file specified in llm_config.

        Returns:
            Chat template string if successfully loaded, otherwise empty string
        """
        custom_chat_template_file = getattr(self.llm_config.llm, "chat_template")
        chat_template = ""
        if custom_chat_template_file and os.path.exists(custom_chat_template_file):
            try:
                with safe_open(custom_chat_template_file, "r", encoding="utf-8") as f:
                    chat_template = f.read()
            except Exception as e:
                warn_msg = f"Failed to load given chat template '{custom_chat_template_file}': {e}"
                logger.warning(message_filter(warn_msg))
                logger.warning("Will still use default chat template of tokenizer.")
        return chat_template

    def _get_input_builder(self) -> InputBuilder:
        """The default method to get input builder.

        Creates an InputBuilder instance with appropriate parameters.

        Returns:
            InputBuilder instance configured for the current model
        """
        kwargs = {}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return InputBuilder(self.tokenizer, **kwargs)

    def _get_model_cls(self) -> "BaseModelForCausalLM":
        """The default method to get model class.

        This is a basic router method to find model class, which is usually not necessary to be overridden.

        Returns:
            Model class for the specific model type
        """
        model_file_dir_name = f"mindie_llm.runtime.models.{self._model_type}."
        module_path = f"{model_file_dir_name}{self._model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self._model_type_cap}ForCausalLM"
        return getattr(module, model_cls_name)

    def _get_draft_cls(self) -> "BaseModelForCausalLM":
        """
        Dynamically imports and returns the draft model class for the specified model type.

        This method constructs the module path based on the model type, imports the module, 
        and retrieves the corresponding draft model class.

        Returns:
            type: The draft model class for the specified model type.
        """
        model_file_dir_name = f"mindie_llm.runtime.models.{self._model_type}."
        module_path = f"{model_file_dir_name}{self._model_type}_mtp"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self._model_type_cap}MTP"
        return getattr(module, model_cls_name)

    def _get_tokenizer(self) -> "AutoTokenizer":
        """The default method to get tokenizer.

        Uses safe_get_tokenizer_from_pretrained with security checks.

        Returns:
            Tokenizer instance with appropriate configuration
        """
        return safe_get_tokenizer_from_pretrained(
            self.load_config.tokenizer_path,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.load_config.trust_remote_code,
            use_fast=True
        )

    def _get_reasoning_parser(self) -> CommonReasoningParser | None:
        """The default method to get reasoning parser.

        Checks for reasoning tokens in tokenizer and creates appropriate parser.

        Returns:
            CommonReasoningParser instance or None
        """
        def check_token_exists(token) -> int | None:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id == self.tokenizer.unk_token_id:
                    return None
                return token_id
            except Exception:
                return None

        start_reasoning_token_id = check_token_exists("<think>")
        end_reasoning_token_id = check_token_exists("</think>")
        if start_reasoning_token_id is None or end_reasoning_token_id is None: 
            return None
        return CommonReasoningParser(start_reasoning_token_id, end_reasoning_token_id)

    def _get_tool_calls_parser(self) -> str:
        """The default method to get tool call parser.

        Returns:
            Default tool call parser name based on model type
        """
        return self._model_type

    def _get_tool_calls_processor(self) -> ToolCallsProcessor:
        """The default method to get tools call processor.

        Attempts to create a specific tool call processor based on tool_call_parser.

        Returns:
            ToolCallsProcessor instance or default processor
        """
        try:
            tool_calls_processor = \
                ToolCallsProcessorManager.get_tool_calls_processor(self.tool_call_parser)(self.tokenizer)
        except KeyError as e:
            warn_msg = \
                f"Cannot use {self.tool_call_parser} ToolCallsProcessor: {e}. " \
                f"Will use default ToolCallsProcessor."
            logger.warning(message_filter(warn_msg))
            tool_calls_processor = ToolCallsProcessor("")
        return tool_calls_processor

    def _get_generation_config_dict(self) -> dict:
        """The default method to get generation config.

        Attempts to load generation configuration from generation_config.json in model directory.

        Returns:
            Dictionary containing generation configuration parameters
        """
        generation_config_path = os.path.join(self.load_config.model_name_or_path, "generation_config.json")
        generation_config = {}
        if os.path.exists(generation_config_path):
            generation_config = safe_get_config_dict(generation_config_path)
        return generation_config

    def _get_llm_config(self) -> LLMConfig:
        """Load LLM configuration with validation.
        
        Creates LLMConfig from specified or default path, parses models_dict
        if provided as JSON string, merges model-specific configuration,
        and validates with model-specific validators.
        """
        llm_config_path = (
            self.load_config.llm_config_path 
            if self.load_config.llm_config_path is not None 
            else DEFAULT_CONFIG_PATH
        )
        llm_config = LLMConfig(llm_config_path)

        if isinstance(self.load_config.models_dict, str):
            try:
                self.load_config.models_dict = json.loads(self.load_config.models_dict)
            except json.JSONDecodeError as e:
                message = "The 'models' field does not conform to JSON format. Please check."
                logger.warning(f'{message}, exception info: {e}')
                self.load_config.models_dict = None
        llm_config.merge_models_config(self._model_type)

        llm_config_validators = self.get_llm_config_validators()
        llm_config.check_config(llm_config_validators)

        return llm_config
