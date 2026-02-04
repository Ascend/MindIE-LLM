# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import torch
import PIL.Image

from atb_llm.models.glm41v.router_glm41v import Glm41vRouter


# Test constants
FAKE_MODEL_NAME_OR_PATH = "fake_model_name_or_path"
FAKE_CONFIG_DICT = {
    'model_type': 'glm4v',
    'text_config': {
        'dtype': 'float16',
        'vocab_size': 151552,
        'max_position_embeddings': 65536,
        'num_hidden_layers': 40,
    },
    'vision_config': {
        'image_size': 336,
        'patch_size': 14,
    }
}

# Default shared memory info structure
DEFAULT_SHM_INFO = {
    'pixel_values_shm_name': 0,
    'pixel_values_shape_value': 0,
    'image_grid_thw_shm_name': 0,
    'image_grid_thw_shape_value': 0,
    'pixel_values_videos_shm_name': 0,
    'pixel_values_videos_shape_value': 0,
    'video_grid_thw_shm_name': 0,
    'video_grid_thw_shape_value': 0,
}


@dataclass
class MockTextConfig:
    dtype: str = "float16"
    vocab_size: int = 151552
    max_position_embeddings: int = 65536


@dataclass
class MockVisionConfig:
    image_size: int = 336
    patch_size: int = 14


@dataclass
class MockConfig:
    def __init__(self):
        self.text_config = MockTextConfig()
        self.vision_config = MockVisionConfig()
        self.model_name_or_path = FAKE_MODEL_NAME_OR_PATH
        self.torch_dtype = torch.float16
        self.is_reasoning_model = False
        self.generation_config = Mock()
        self.do_sample = False
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls()


class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
    
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, 
                           return_dict=True, return_tensors='pt'):
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]])
        }


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151336


class MockInputBuilder:
    def __init__(self, tokenizer, config, processor):
        self.tokenizer = tokenizer
        self.config = config
        self.processor = processor
    
    @staticmethod
    def check_image_and_video_concurrency(content):
        if any(item.get("image") for item in content) \
            and any(item.get("video") for item in content):
            msg = "Image and video cannot exist in single prompt at the same time."
            raise ValueError(msg)
    
    def get_shm_name_save_path(self, file_path):
        return f"/tmp/shm_{file_path.replace('/', '_')}.txt"
    
    def update_token_id(self, inputs, shm_info):
        return inputs['input_ids'].flatten()


class TestGlm41vRouter(unittest.TestCase):
    
    @patch('atb_llm.models.glm41v.router_glm41v.safe_from_pretrained')
    def setUp(self, mock_safe_from_pretrained):
        self.mock_processor = MockProcessor()
        mock_safe_from_pretrained.return_value = self.mock_processor
        self.router = Glm41vRouter(
            model_name_or_path=FAKE_MODEL_NAME_OR_PATH,
            config_dict=FAKE_CONFIG_DICT
        )
    
    @patch('atb_llm.models.glm41v.router_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.router_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.router_glm41v.safe_open_image')
    @patch('atb_llm.models.glm41v.router_glm41v.Glm41vInputBuilder')
    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_text_only(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image, 
                                     mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with text-only input
        Given: A Glm41vRouter instance with mocked dependencies
        When: Tokenize is called with text-only input
        Then: 
        - Tokenization completes successfully (non-null result)
        - Image/video processing functions are not called
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'text': 'Hello, world!'}]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_not_called()
        mock_check_video_path.assert_not_called()
    
    @patch('atb_llm.models.glm41v.router_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.router_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.router_glm41v.safe_open_image')
    @patch('atb_llm.models.glm41v.router_glm41v.Glm41vInputBuilder')
    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_image(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image,
                                  mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with image input
        Given: A Glm41vRouter instance with mocked dependencies
        When: Tokenize is called with image input
        Then: 
        - Tokenization completes successfully (non-null result)
        - Image processing function is called once
        - Video processing function is not called
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_image = MagicMock(spec=PIL.Image.Image)
        mock_safe_open_image.return_value = mock_image
        
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'image': '/path/to/image.jpg'}]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_called_once()
        mock_check_video_path.assert_not_called()
    
    @patch('atb_llm.models.glm41v.router_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.router_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.router_glm41v.safe_open_image')
    @patch('atb_llm.models.glm41v.router_glm41v.Glm41vInputBuilder')
    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_video(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image,
                                  mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with video input
        Given: A Glm41vRouter instance with mocked dependencies
        When: Tokenize is called with video input
        Then: 
        - Tokenization completes successfully (non-null result)
        - Video check function is called once
        - Image processing function is not called
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_check_video_path.return_value = '/path/to/video.mp4'
        
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'video': '/path/to/video.mp4'}]
        result = self.router.tokenize(inputs)
        
        self.assertIsNotNone(result)
        mock_check_video_path.assert_called_once()
        mock_safe_open_image.assert_not_called()
    
    @patch('atb_llm.models.glm41v.router_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.router_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.router_glm41v.safe_open_image')
    @patch('atb_llm.models.glm41v.router_glm41v.Glm41vInputBuilder')
    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_mixed_inputs(self, mock_processor_prop, mock_input_builder_cls, mock_safe_open_image,
                                         mock_check_video_path, mock_process_shm):
        """
        Test tokenize method with mixed text/image/video input
        Given: A Glm41vRouter instance with mocked dependencies
        When: Tokenize is called with mixed text, image and video inputs
        Then: Raise ValueError
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_image = MagicMock(spec=PIL.Image.Image)
        mock_safe_open_image.return_value = mock_image
        mock_check_video_path.return_value = '/path/to/video.mp4'
        
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [
            {'text': 'Describe this image and video'},
            {'image': '/path/to/image.jpg'},
            {'video': '/path/to/video.mp4'}
        ]
        with self.assertRaises(ValueError):
            result = self.router.tokenize(inputs)
    
    @patch('atb_llm.models.glm41v.router_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.router_glm41v.Glm41vInputBuilder')
    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    def test_tokenize_with_shm_name_save_path(self, mock_processor_prop, mock_input_builder_cls, mock_process_shm):
        """
        Test tokenize method with custom shm_name_save_path parameter
        Given: A Glm41vRouter instance with mocked dependencies
        When: Tokenize is called with text input and custom shm path
        Then: 
        - Tokenization completes successfully (non-null result)
        - process_shared_memory is called with the custom shm path parameter
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        self.router._input_builder = mock_input_builder
        
        mock_process_shm.return_value = DEFAULT_SHM_INFO.copy()
        
        inputs = [{'text': 'Hello'}]
        shm_path = '/custom/shm/path.txt'
        result = self.router.tokenize(inputs, shm_name_save_path=shm_path)
        
        self.assertIsNotNone(result)
        mock_process_shm.assert_called_once()
        call_args = mock_process_shm.call_args
        self.assertEqual(call_args[0][1], shm_path)
    
    @patch('atb_llm.models.glm41v.router_glm41v.Glm41vInputBuilder')
    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    @patch.object(Glm41vRouter, 'tokenizer', new_callable=PropertyMock)
    @patch.object(Glm41vRouter, 'config', new_callable=PropertyMock)
    def test_get_input_builder(self, mock_config_prop, mock_tokenizer_prop, mock_processor_prop, mock_input_builder_cls):
        """
        Test get_input_builder method invocation
        Given: A Glm41vRouter instance with mocked dependencies
        When: get_input_builder is called
        Then: 
        - Glm41vInputBuilder is initialized with correct parameters
        - Non-null input builder instance is returned
        """
        mock_processor_prop.return_value = self.mock_processor
        mock_tokenizer_prop.return_value = self.mock_processor.tokenizer
        mock_config_prop.return_value = MockConfig()
        mock_input_builder = MockInputBuilder(self.mock_processor.tokenizer, Mock(), self.mock_processor)
        mock_input_builder_cls.return_value = mock_input_builder
        
        result = self.router.get_input_builder()
        
        mock_input_builder_cls.assert_called_once_with(
            self.mock_processor.tokenizer,
            mock_config_prop.return_value,
            self.mock_processor
        )
        self.assertIsNotNone(result)

    @patch.object(Glm41vRouter, 'processor', new_callable=PropertyMock)
    def test_get_tokenizer(self, mock_processor_prop):
        """
        Test get_tokenizer method invocation
        Given: A Glm41vRouter instance with mocked processor
        When: get_tokenizer is called
        Then: Non-null tokenizer instance is returned
        """
        mock_processor_prop.return_value = self.mock_processor
        
        result = self.router.get_tokenizer()
        
        self.assertIsNotNone(result)
    
    @patch('atb_llm.models.glm41v.router_glm41v.safe_from_pretrained')
    @patch('atb_llm.models.glm41v.router_glm41v.AutoProcessor')
    def test_get_processor(self, mock_auto_processor_cls, mock_safe_from_pretrained):
        """
        Test get_processor method invocation
        Given: A Glm41vRouter instance with mocked dependencies
        When: get_processor is called
        Then: 
        - safe_from_pretrained is called with correct parameters
        - Non-null processor instance is returned
        """
        mock_safe_from_pretrained.return_value = self.mock_processor
        
        result = self.router.get_processor()
        
        mock_safe_from_pretrained.assert_called_once_with(
            mock_auto_processor_cls,
            FAKE_MODEL_NAME_OR_PATH
        )
        self.assertIsNotNone(result)

    @patch('atb_llm.models.base.router.BaseRouter.get_config_cls', return_value=MockConfig)
    def test_get_config(self, mock_config):
        config = self.router.get_config()
        self.assertEqual(config.torch_dtype, torch.float16)


if __name__ == '__main__':
    unittest.main()
