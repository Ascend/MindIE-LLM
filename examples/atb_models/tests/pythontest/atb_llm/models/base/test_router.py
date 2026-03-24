# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import json
import tempfile
import copy
import unittest
from unittest.mock import patch, Mock

import pytest
from ddt import ddt, data, unpack

from atb_llm.models.base.router import BaseRouter, ToolsCallProcessor, check_eos, check_value
from atb_llm.models.base.config import BaseConfig
from atb_llm.models.base.input_builder import InputBuilder
from atb_llm.models.base.postprocessor import Postprocessor
from atb_llm.utils.configuration_utils import LLMConfig
from atb_llm.utils.parameter_validators import DictionaryParameterValidator


FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    "architectures": [
        "LlamaForCausalLM"
    ],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1198,
    "eos_token_id": 1199,
    "hidden_act": "silu",
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 500,
    "max_position_embeddings": 8192,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 1,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.40.0.dev0",
    "use_cache": True,
    "vocab_size": 1200
}

MODEL_TYPE_CONVERT_MAP = {
    "chatglm": "glm4v",
    "vision_config": "glm4v"
}


def test_check_value_with_valid_range():
    """Test check_value function with valid range."""
    # Should not raise exception
    check_value("test_attr", 5, 0, 10)


def test_check_value_with_invalid_range():
    """Test check_value function with invalid range."""
    with pytest.raises(ValueError):
        check_value("test_attr", 15, 0, 10)


def test_check_eos_with_int():
    """Test check_eos function with integer eos."""
    # Should not raise exception
    check_eos(5, 0, 10)


def test_check_eos_with_list():
    """Test check_eos function with list eos."""
    # Should not raise exception
    check_eos([5, 6], 0, 10)


def test_check_eos_with_nested_list():
    """Test check_eos function with nested list eos."""
    # Should not raise exception
    check_eos([[5, 6], [7, 8]], 0, 10)


def test_check_eos_with_invalid_type():
    """Test check_eos function with invalid type."""
    with pytest.raises(ValueError):
        check_eos("invalid", 0, 10)


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 6
        self.bos_token_id = 7
        self.eos_token_id = 8

    def decode(self, token_id, skip_special_tokens=False):
        """Mock decode method that returns a test string."""
        return "A test string"

    def add_special_tokens(self, tokens):
        """Mock add_special_tokens method."""
        pass

    def __call__(self, text, return_tensors=None):
        """Mock __call__ method for tokenization."""
        return {"input_ids": [[1, 2, 3]]}


@ddt
class TestBaseRouter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.json')
        self.test_json_config1 = {
            "llm": {
                "ccl": {
                    "backend": "lccl",
                    "enable_mc2": "true"
                },
                "stream_options": {
                    "micro_batch": "false"
                },
                "engine": {
                    "graph": "cpp"
                },
                "plugins": {
                    "plugin_type": "",
                    "num_speculative_tokens": 1
                },
                "parallel_options": {
                    "o_proj_local_tp": -1,
                    "lm_head_local_tp": 1,
                    "hccl_buffer": 128,
                    "hccl_moe_ep_buffer": 512,
                    "hccl_moe_tp_buffer": 64
                },
                "kv_cache_options": {
                    "enable_nz": "false"
                },
                "weights_options": {
                    "low_cpu_memory_mode": "false"
                },
                "enable_reasoning": "false",
                "chat_template": "",
                "tool_call_options": {
                    "tool_call_parser": ""
                },
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(self.test_json_config1, f)
        
        self.fake_chat_template_path = os.path.join(self.temp_dir.name, 'fake_caht_template.jinja')
        self.test_chat_template = "FAKE_CHAT_TEMPLATE"
        with open(self.fake_chat_template_path, 'w') as f:
            json.dump(self.test_chat_template, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    @data('chatglm', 'internvl_chat', 'llava_next_video', 'bunny-qwen2', 'bunny-minicpm', 'minicpm_qwen2_v2',
          'vita-mixtral', 'qwen2_moe', 'qwen2_audio', 'qwen2_vl', 'qwen2_5_vl', 'ernie4_5_moe')
    def test_post_init(self, mock_model_type_value):
        config_dict = copy.deepcopy(FAKE_CONFIG_DICT)
        config_dict["model_type"] = mock_model_type_value
        fake_model_name_or_path = FAKE_MODEL_NAME_OR_PATH
        if mock_model_type_value == 'chatglm':
            config_dict['vision_config'] = 'fake_vision_config'
        base_router = BaseRouter(fake_model_name_or_path, config_dict)
        expect_map = {
            "chatglm": "Glm4v",
            "internvl_chat": "Internvl",
            "llava_next_video": "Llava_next",
            "bunny-qwen2": "Bunny",
            "bunny-minicpm": "Bunny",
            "vita-mixtral": "Vita",
            "qwen2_moe": "Qwen2moe",
            "qwen2_audio": "Qwen2audio",
            "qwen2_vl": "Qwen2vl",
            "minicpm_qwen2_v2": "Minicpmqwen2v2",
            "qwen2_5_vl": "Qwen2vl",
            "ernie4_5_moe": "Erniemoe"
        }
        expected_model_type_cap = expect_map[mock_model_type_value] if mock_model_type_value in expect_map \
            else mock_model_type_value
        self.assertEqual(
            base_router.model_type_cap,
            expected_model_type_cap
        )
        self.assertEqual(base_router.tokenizer_path, fake_model_name_or_path)

    @patch('atb_llm.utils.file_utils.standardize_path', return_value="")
    @patch('atb_llm.utils.file_utils.check_path_permission', return_value=None)
    @patch('transformers.configuration_utils.PretrainedConfig.get_config_dict', return_value={})
    @patch("os.path.exists", return_value=False)
    @patch("atb_llm.models.base.router.BaseRouter.get_config_cls")
    def test_config(self, mock_get_config_cls, _1, _2, _3, _4):
        mock_get_config_cls.return_value = BaseConfig
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, max_position_embeddings=12345)
        expected_config = BaseConfig.from_dict(FAKE_CONFIG_DICT)
        expected_config.max_position_embeddings = 12345
        self.assertEqual(base_router.config, expected_config)

    @patch('atb_llm.utils.file_utils.standardize_path', return_value="")
    @patch('atb_llm.utils.file_utils.check_path_permission', return_value=None)
    @patch('transformers.configuration_utils.PretrainedConfig.get_config_dict', return_value={})
    @patch("os.path.exists", return_value=False)
    @patch("atb_llm.models.base.router.BaseRouter.get_config_cls")
    def test_config_with_quantization_config(self, mock_get_config_cls, _1, _2, _3, _4):
        mock_get_config_cls.return_value = BaseConfig
        config_dict_with_quant = copy.deepcopy(FAKE_CONFIG_DICT)
        config_dict_with_quant["quantization_config"] = {"test_key": "test_value"}
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, config_dict_with_quant, max_position_embeddings=12345)
        expected_config = BaseConfig.from_dict(config_dict_with_quant)
        expected_config.max_position_embeddings = 12345
        self.assertEqual(base_router.config, expected_config)

    @patch('atb_llm.utils.file_utils.standardize_path', return_value="")
    @patch('atb_llm.utils.file_utils.check_path_permission', return_value=None)
    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_input_builder(self, _1, _2, _3):
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, max_position_embeddings=12345)
        fake_tokenzier = FakeTokenizer()
        expected_input_builder = InputBuilder(fake_tokenzier, max_length=12345)
        actual_input_builder = base_router.input_builder
        actual_input_builder.tokenizer.decode(None, 1)
        fake_tokenzier.decode(None, 1)
        self.assertEqual(actual_input_builder.max_length, expected_input_builder.max_length) 
    
    def test_model_cls(self):
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, enable_atb_torch=False)
        self.assertEqual(base_router.model_cls.__name__, 'FlashLlamaForCausalLM')

    @data(
        {"eos_token_id": 1199, "repetition_penalty": 1.2, "do_sample": True},
        {"eos_token_id": [1199, 1000], "repetition_penalty": 1.0, "do_sample": False},
        {"head_dim": 64, "repetition_penalty": 1.5, "do_sample": True},
        {"generation_config": {"max_new_tokens": 100, "pad_token_id": 0, "eos_token_id": 2}}
    )
    def test_check_config_valid(self, valid_params):
        config = BaseConfig.from_dict(FAKE_CONFIG_DICT)
        # Add generation_config to config
        config.generation_config = type('MockGenerationConfig', (), {
            'max_new_tokens': 100,
            'pad_token_id': 0,
            'eos_token_id': 2
        })()
        
        for k, v in valid_params.items():
            if k == "generation_config":
                setattr(config.generation_config, k, v)
            else:
                setattr(config, k, v)
        BaseRouter.check_config(config)


    @data(
        ({"eos_token_id": "str"}, ValueError),  
        ({"repetition_penalty": 0}, ValueError),  
        ({"do_sample": "True"}, ValueError),  
        ({"vocab_size": 0}, ValueError),
    )
    @unpack
    def test_check_config_invalid(self, invalid_params, expect_exc):
        config = BaseConfig.from_dict(FAKE_CONFIG_DICT)
        # Add generation_config to config
        config.generation_config = type('MockGenerationConfig', (), {
            'max_new_tokens': 100,
            'pad_token_id': 0,
            'eos_token_id': 2
        })()
        
        for k, v in invalid_params.items():
            if k == "generation_config":
                setattr(config.generation_config, k, v)
            else:
                setattr(config, k, v)
        with self.assertRaises(expect_exc):
            BaseRouter.check_config(config)

    
    @patch('atb_llm.models.base.router.safe_get_tokenizer_from_pretrained')
    def test_tokenizer(self, mock_tokenizer):
        mock_tokenizer.return_value = FakeTokenizer()
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        self.assertIsNotNone(base_router.tokenizer)
        mock_tokenizer.assert_called_once()
        
        no_token_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT, load_tokenizer=False)
        self.assertIsNone(no_token_router.tokenizer)
    
    def test_get_generation_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            gen_path = os.path.join(tmp_dir, "generation_config.json")
            with open(gen_path, "w") as f:
                json.dump({"max_new_tokens": 100, "temperature": 0.7}, f)
            
            router = BaseRouter(tmp_dir, FAKE_CONFIG_DICT)
            self.assertEqual(router.generation_config["max_new_tokens"], 100)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            router = BaseRouter(tmp_dir, FAKE_CONFIG_DICT)
            self.assertEqual(router.generation_config, {})

    @patch('atb_llm.utils.file_utils.standardize_path', return_value="")
    @patch('atb_llm.utils.file_utils.check_path_permission', return_value=None)
    @patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer())
    def test_postprocessor(self, _1, _2, _3):
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        
        # Mock the config and its attributes
        mock_config = Mock()
        mock_config.vocab_size = 1200
        mock_config.generation_config = Mock()
        mock_config.generation_config.max_new_tokens = 100
        mock_config.generation_config.pad_token_id = 0
        mock_config.generation_config.eos_token_id = 2
        base_router._config = mock_config
        
        # Test that postprocessor is created and validated
        postprocessor = base_router.postprocessor
        self.assertIsInstance(postprocessor, Postprocessor)
    
    @patch('atb_llm.models.base.router.safe_get_tokenizer_from_pretrained', return_value=FakeTokenizer())
    def test_toolscallprocesser(self, _1):
        base_router = BaseRouter(FAKE_MODEL_NAME_OR_PATH, FAKE_CONFIG_DICT)
        self.assertEqual(base_router.toolscallprocessor.decode(""), ToolsCallProcessor("").decode(""))
        
    def test_get_llm_config_validators(self):
        base_router = BaseRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )

        llm_config_validators = base_router.get_llm_config_validators()

        self.assertIn("llm", llm_config_validators)
        llm_validator = llm_config_validators["llm"]
        self.assertIsInstance(llm_validator, DictionaryParameterValidator)

    def test_llm_config_validate(self):
        base_router = BaseRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )
        base_router.llm_config = LLMConfig(self.config_path)

        base_router.llm_config.llm.stream_options.micro_batch = True
        with self.assertRaises(ValueError):
            base_router._check_llm_config_model_type("llama")
        
        base_router.llm_config.llm.stream_options.micro_batch = False
        base_router.llm_config.llm.engine.graph = "python"
        with self.assertRaises(ValueError):
            base_router._check_llm_config_model_type("qwen")

        base_router.llm_config.llm.weights_options = type('', (), {})()
        with self.assertRaises(AttributeError):
            base_router._check_llm_config_model_type("qwen2")
        
        base_router.llm_config.llm.engine.graph = "cpp"
        base_router.llm_config.llm.kv_cache_options.enable_nz = True
        with self.assertRaises(ValueError):
            base_router._check_llm_config_model_type("glm")

        base_router.llm_config.llm.stream_options.micro_batch = True
        base_router.llm_config.llm.ccl.enable_mc2 = True
        with self.assertRaises(ValueError):
            base_router._check_llm_config_orthogonality()

    @patch('atb_llm.models.base.router.safe_get_tokenizer_from_pretrained', return_value=FakeTokenizer())
    def test_process_tool_call_options(self, _1):
        base_router = BaseRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )
        base_router.llm_config = LLMConfig(self.config_path)

        with patch('atb_llm.models.base.router.logger') as mock_logger:
            base_router.llm_config.llm.tool_call_options.tool_call_parser = "fake_parser"
            base_router.process_tool_call_options()
            mock_logger.warning.assert_called_with(
                f"Will use default tool_call_parser of model."
            )

        base_router.llm_config.llm.tool_call_options.tool_call_parser = "hermes"
        base_router.process_tool_call_options()
        self.assertEqual(base_router._tool_call_parser, "hermes")

    def test_custom_chat_template(self):
        base_router = BaseRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )
        base_router.llm_config = LLMConfig(self.config_path)

        base_router.llm_config.llm.chat_template = self.fake_chat_template_path
        self.assertEqual(base_router.custom_chat_template, f'"{self.test_chat_template}"')


if __name__ == '__main__':
    unittest.main()