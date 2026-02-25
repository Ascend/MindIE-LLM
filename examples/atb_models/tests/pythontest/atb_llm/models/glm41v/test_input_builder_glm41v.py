# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import unittest
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

import torch
import PIL.Image
import numpy as np

from atb_llm.models.glm41v.input_builder_glm41v import Glm41vInputBuilder


@dataclass
class MockVisionConfig:
    """Mock vision configuration for Glm41vInputBuilder testing"""
    spatial_merge_size: int = 2


@dataclass
class MockConfig:
    """Mock model configuration for Glm41vInputBuilder testing"""
    def __init__(self):
        self.vision_config: MockVisionConfig = None
        self.image_token_id: int = 151343
        self.video_token_id: int = 151344
        self.image_start_token_id = 151339
        self.image_end_token_id = 151340
        self.video_start_token_id: int = 151341
        self.video_end_token_id: int = 151342
        self.vision_config = MockVisionConfig()


class MockTokenizer:
    """Mock tokenizer for Glm41vInputBuilder testing"""
    def __init__(self):
        self.eos_token_id = 151336


class MockProcessor:
    """Mock processor for Glm41vInputBuilder testing"""
    def __init__(self):
        self.tokenizer = MockTokenizer()
    
    def apply_chat_template(self, conversation_list, tokenize=True, add_generation_prompt=True,
                           return_dict=True, return_tensors='pt'):
        """Mock chat template application with dummy input ids"""
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        }


class TestGlm41vInputBuilder(unittest.TestCase):
    """Unit tests for Glm41vInputBuilder class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.tokenizer = MockTokenizer()
        self.processor = MockProcessor()
        self.config = MockConfig()
    
    def test_init(self):
        """Test Glm41vInputBuilder initialization
        
        Given: Mock tokenizer, config and processor
        When: Initialize Glm41vInputBuilder instance
        Then: Instance attributes should match the input fixtures
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        self.assertEqual(input_builder.tokenizer, self.tokenizer)
        self.assertEqual(input_builder.processor, self.processor)
        self.assertEqual(input_builder.config, self.config)
        self.assertEqual(input_builder.spatial_merge_size, self.config.vision_config.spatial_merge_size)
        self.assertEqual(input_builder.image_token_id, self.config.image_token_id)
        self.assertEqual(input_builder.image_start_token_id, self.config.image_start_token_id)
        self.assertEqual(input_builder.image_end_token_id, self.config.image_end_token_id)
        self.assertEqual(input_builder.video_start_token_id, self.config.video_start_token_id)
        self.assertEqual(input_builder.video_end_token_id, self.config.video_end_token_id)
    
    def test_get_shm_name_save_path(self):
        """Test static method get_shm_name_save_path
        
        Given: Absolute and relative file paths
        When: Call get_shm_name_save_path with the file paths
        Then: Should return correct shm file path in the parent's parent directory
        """
        file_path = '/project/data/inputs/current/file.jpg'
        expected_result = '/project/data/inputs/shm_name.txt'
        actual_result = Glm41vInputBuilder.get_shm_name_save_path(file_path)
        self.assertEqual(actual_result, expected_result)
    
    def test_generate_position_ids_without_vision(self):
        """Test generate_position_ids without vision tokens
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        expected_pos_ids = np.arange(len(input_ids), dtype=np.int64)
        result_pos_ids = input_builder.generate_position_ids(input_ids)
        np.testing.assert_array_equal(result_pos_ids, expected_pos_ids)

    @patch('atb_llm.models.glm41v.input_builder_glm41v.get_data_from_shm', return_value=np.array([[2, 3, 4]]))
    def test_generate_position_ids_with_image(self, mock_process_shm):
        """Test generate_position_ids with image tokens
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids = np.arange(16)
        input_ids[3] = self.config.video_start_token_id
        input_ids[4] = self.config.image_start_token_id
        with patch.object(input_builder, '_get_token_type', return_value=[
            ("image", 3, 11),
            ("text", 11, 12)
        ]):
            result_pos_ids = input_builder.generate_position_ids(input_ids)
        self.assertEqual(result_pos_ids[-1], 2)

    @patch('atb_llm.models.glm41v.input_builder_glm41v.get_data_from_shm', return_value=np.array([[2, 3, 4]]))
    def test_generate_position_ids_with_video(self, mock_process_shm):
        """Test generate_position_ids with video tokens
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids = np.arange(13)
        input_ids[3] = self.config.image_start_token_id
        with patch.object(input_builder, '_get_token_type', return_value=[
            ("text", 0, 4),
            ("video", 4, 9),
            ("text", 9, 13)
        ]):
            result_pos_ids = input_builder.generate_position_ids(input_ids)
        self.assertEqual(result_pos_ids[-1], 9)
    
    @patch('atb_llm.models.glm41v.input_builder_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.input_builder_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.input_builder_glm41v.safe_open_image')
    def test_make_context_with_text_only(self, mock_safe_open_image, mock_check_video_path, mock_process_shm):
        """Test make_context with text-only content
        
        Given: Conversation with text-only user content, mocked shared memory processing
        When: Call make_context
        Then: Should return valid result without calling image or video helper methods
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": [{"text": "Hello, world!"}]}]
        mock_process_shm.return_value = {}
        result = input_builder.make_context(rank=0, conversation=conversation)
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_not_called()
        mock_check_video_path.assert_not_called()
        mock_process_shm.assert_called_once()
    
    @patch('atb_llm.models.glm41v.input_builder_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.input_builder_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.input_builder_glm41v.safe_open_image')
    def test_make_context_with_image(self, mock_safe_open_image, mock_check_video_path, mock_process_shm):
        """Test make_context with image content
        
        Given: Conversation with image content, mocked image and shared memory processing
        When: Call make_context
        Then: Should return valid result and call image helper method once
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        mock_image = MagicMock(spec=PIL.Image.Image)
        mock_safe_open_image.return_value = mock_image
        conversation = [
            {"role": "user", "content": [{"text": "Describe this image"}, {"image": "/path/to/image.jpg"}]}
        ]
        mock_process_shm.return_value = {}
        result = input_builder.make_context(rank=0, conversation=conversation)
        self.assertIsNotNone(result)
        mock_safe_open_image.assert_called_once_with(PIL.Image, "/path/to/image.jpg")
        mock_check_video_path.assert_not_called()
    
    @patch('atb_llm.models.glm41v.input_builder_glm41v.process_shared_memory')
    @patch('atb_llm.models.glm41v.input_builder_glm41v.check_video_path')
    @patch('atb_llm.models.glm41v.input_builder_glm41v.safe_open_image')
    def test_make_context_with_video(self, mock_safe_open_image, mock_check_video_path, mock_process_shm):
        """Test make_context with video content
        
        Given: Conversation with video content, mocked video path check and shared memory processing
        When: Call make_context
        Then: Should return valid result and call video helper method once
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        mock_check_video_path.return_value = "/path/to/video.mp4"
        conversation = [
            {"role": "user", "content": [{"text": "Describe this video"}, {"video": "/path/to/video.mp4"}]}
        ]
        mock_process_shm.return_value = {}
        result = input_builder.make_context(rank=0, conversation=conversation)
        self.assertIsNotNone(result)
        mock_check_video_path.assert_called_once_with("/path/to/video.mp4")
        mock_safe_open_image.assert_not_called()
    
    @patch('atb_llm.models.glm41v.input_builder_glm41v.process_shared_memory')
    def test_make_context_with_string_content(self, mock_process_shm):
        """Test make_context with string content (non-list format)
        
        Given: Conversation with string content, mocked shared memory processing
        When: Call make_context
        Then: Should return valid result and call shared memory processing once
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": "Hello, world!"}]
        mock_process_shm.return_value = {}
        result = input_builder.make_context(rank=0, conversation=conversation)
        self.assertIsNotNone(result)
        mock_process_shm.assert_called_once()
    
    def test_make_context_with_invalid_content_type(self):
        """Test make_context with invalid content type
        
        Given: Conversation with integer content (invalid type)
        When: Call make_context
        Then: Should raise TypeError with relevant error message
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": 123}]
        with self.assertRaises(TypeError) as context:
            input_builder.make_context(rank=0, conversation=conversation)
        self.assertIn("content", str(context.exception))
        self.assertIn("List[Dict]", str(context.exception))
    
    @patch('atb_llm.models.glm41v.input_builder_glm41v.process_shared_memory')
    def test_make_context_with_shm_name_save_path(self, mock_process_shm):
        """Test make_context with custom shm_name_save_path
        
        Given: Conversation with text content, custom shm path and mocked shared memory processing
        When: Call make_context with custom shm path
        Then: Should call shared memory processing with the custom path
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        conversation = [{"role": "user", "content": [{"text": "Hello"}]}]
        mock_process_shm.return_value = {}
        custom_shm_path = "/custom/shm/path.txt"
        result = input_builder.make_context(rank=0, conversation=conversation, shm_name_save_path=custom_shm_path)
        self.assertIsNotNone(result)
        mock_process_shm.assert_called_once()
        call_args = mock_process_shm.call_args
        self.assertEqual(call_args[0][1], custom_shm_path)
    
    def test_update_token_id_without_vision(self):
        """Test update_token_id without vision tokens
        
        Given: Input ids without vision tokens, empty shm info
        When: Call update_token_id
        Then: Should return flattened input ids matching expected tensor
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}
        shm_info = {}
        expected_tensor = inputs['input_ids'].flatten()
        result_tensor = input_builder.update_token_id(inputs, shm_info)
        torch.testing.assert_close(result_tensor, expected_tensor)
    
    def test_update_token_id_with_image(self):
        """Test update_token_id with image tokens
        
        Given: Input ids with vision tokens, sufficient padding, valid shm info
        When: Call update_token_id
        Then: Should return valid tensor with shm info in correct positions
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids_list = [1, 2, self.config.image_start_token_id, self.config.image_token_id]
        input_ids_list.extend([0] * 20)  # Sufficient padding
        input_ids_list.append(self.config.image_end_token_id)
        input_ids_list.extend([5, 6, 7])
        inputs = {'input_ids': torch.tensor([input_ids_list])}
        shm_info = {
            'pixel_values_shm_name': 1001,
            'pixel_values_shape_value': 1002,
            'image_grid_thw_shm_name': 1003,
            'image_grid_thw_shape_value': 1004,
            'pixel_values_videos_shm_name': 0,
            'pixel_values_videos_shape_value': 0,
            'video_grid_thw_shm_name': 0,
            'video_grid_thw_shape_value': 0,
        }
        result_tensor = input_builder.update_token_id(inputs, shm_info)
        self.assertIsNotNone(result_tensor)
        self.assertIsInstance(result_tensor, torch.Tensor)
        boi_pos = torch.where(result_tensor == self.config.image_start_token_id)[0][0].item()
        self.assertEqual(result_tensor[boi_pos + 1].item(), shm_info['pixel_values_shm_name'])
    
    def test_update_token_id_insufficient_space(self):
        """Test update_token_id with insufficient space for shm info
        
        Given: Input ids with vision tokens but no padding (insufficient space), valid shm info
        When: Call update_token_id
        Then: Should raise RuntimeError with relevant error message
        """
        input_builder = Glm41vInputBuilder(self.tokenizer, self.config, self.processor)
        input_ids_list = [1, 2, self.config.image_start_token_id, self.config.image_token_id]
        input_ids_list.append(self.config.image_end_token_id)
        inputs = {'input_ids': torch.tensor([input_ids_list])}
        shm_info = {
            'pixel_values_shm_name': 1001,
            'pixel_values_shape_value': 1002,
            'image_grid_thw_shm_name': 1003,
            'image_grid_thw_shape_value': 1004,
            'pixel_values_videos_shm_name': 0,
            'pixel_values_videos_shape_value': 0,
            'video_grid_thw_shm_name': 0,
            'video_grid_thw_shape_value': 0,
        }
        with self.assertRaises(RuntimeError) as context:
            input_builder.update_token_id(inputs, shm_info)
        
        self.assertIn("Load share memory info to input ids failed", str(context.exception))

if __name__ == '__main__':
    unittest.main()
