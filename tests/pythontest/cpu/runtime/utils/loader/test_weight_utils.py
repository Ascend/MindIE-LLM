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
from unittest.mock import MagicMock, patch, Mock
import torch

from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler


class TestWeightsFileHandler(unittest.TestCase):
    """Test cases for WeightsFileHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "/fake/model/path"
        self.extension = ".safetensors"

    def tearDown(self):
        """Clean up after tests."""
        pass

    @patch('mindie_llm.runtime.utils.loader.weight_utils.WeightsFileHandler._load_weight_file_routing')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.WeightsFileHandler._get_weight_filenames')
    def test_init(self, mock_get_weight_filenames, mock_load_routing):
        """Test initialization."""
        mock_filenames = ["file1.safetensors", "file2.safetensors"]
        mock_routing = {"tensor1": "file1.safetensors", "tensor2": "file2.safetensors"}
        
        mock_get_weight_filenames.return_value = mock_filenames
        mock_load_routing.return_value = mock_routing
        
        handler = WeightsFileHandler(self.model_path, self.extension)
        
        self.assertEqual(handler._handlers, {})
        self.assertEqual(handler._filenames, mock_filenames)
        self.assertEqual(handler._routing, mock_routing)
        
        mock_get_weight_filenames.assert_called_once_with(self.model_path, self.extension)
        mock_load_routing.assert_called_once()

    def test_extension_property(self):
        """Test extension property."""
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=[]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value={}):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                self.assertEqual(handler.extension, ".safetensors")

    def test_release_file_handler_with_handlers(self):
        """Test release_file_handler when handlers exist."""
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=[]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value={}):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Add some handlers
                mock_handler1 = MagicMock()
                mock_handler2 = MagicMock()
                handler._handlers = {
                    "file1.safetensors": mock_handler1,
                    "file2.safetensors": mock_handler2
                }
                
                # Release handlers
                handler.release_file_handler()
                
                # Verify handlers were cleared
                self.assertEqual(handler._handlers, {})

    def test_release_file_handler_without_handlers(self):
        """Test release_file_handler when no handlers exist."""
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=[]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value={}):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Verify handlers is empty
                self.assertEqual(handler._handlers, {})
                
                # Release handlers (should not raise error)
                handler.release_file_handler()
                
                # Verify handlers is still empty
                self.assertEqual(handler._handlers, {})

    def test_release_file_handler_multiple_calls(self):
        """Test release_file_handler can be called multiple times."""
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=[]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value={}):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Add handlers
                handler._handlers = {"file1.safetensors": MagicMock()}
                
                # Call release multiple times
                handler.release_file_handler()
                handler.release_file_handler()
                handler.release_file_handler()
                
                # Verify handlers is still empty
                self.assertEqual(handler._handlers, {})

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    def test_get_tensor_single_file(self, mock_safe_open):
        """Test get_tensor with single file."""
        mock_routing = {"tensor1": "file1.safetensors"}
        
        mock_file_handler = MagicMock()
        mock_tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_file_handler.get_tensor.return_value = mock_tensor
        mock_safe_open.return_value = mock_file_handler
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                result = handler.get_tensor("tensor1")
                
                # Verify safe_open was called
                mock_safe_open.assert_called_once_with("file1.safetensors", framework="pytorch")
                
                # Verify get_tensor was called on file handler
                mock_file_handler.get_tensor.assert_called_once_with("tensor1")
                
                # Verify result
                self.assertTrue(torch.allclose(result, mock_tensor))
                
                # Verify handler was cached
                self.assertEqual(len(handler._handlers), 1)
                self.assertIn("file1.safetensors", handler._handlers)

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    def test_get_tensor_multiple_files(self, mock_safe_open):
        """Test get_tensor with multiple files."""
        mock_routing = {
            "tensor1": "file1.safetensors",
            "tensor2": "file2.safetensors"
        }
        
        mock_file_handler1 = MagicMock()
        mock_file_handler2 = MagicMock()
        mock_tensor1 = torch.tensor([1.0, 2.0])
        mock_tensor2 = torch.tensor([3.0, 4.0])
        mock_file_handler1.get_tensor.return_value = mock_tensor1
        mock_file_handler2.get_tensor.return_value = mock_tensor2
        
        mock_safe_open.side_effect = [mock_file_handler1, mock_file_handler2]
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors", "file2.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Get tensor from first file
                result1 = handler.get_tensor("tensor1")
                
                # Get tensor from second file
                result2 = handler.get_tensor("tensor2")
                
                # Verify both handlers were created
                self.assertEqual(mock_safe_open.call_count, 2)
                self.assertEqual(len(handler._handlers), 2)
                
                # Verify results
                self.assertTrue(torch.allclose(result1, mock_tensor1))
                self.assertTrue(torch.allclose(result2, mock_tensor2))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    def test_get_tensor_handler_caching(self, mock_safe_open):
        """Test that file handlers are cached and reused."""
        mock_routing = {"tensor1": "file1.safetensors", "tensor2": "file1.safetensors"}
        
        mock_file_handler = MagicMock()
        mock_tensor1 = torch.tensor([1.0, 2.0])
        mock_tensor2 = torch.tensor([3.0, 4.0])
        mock_file_handler.get_tensor.side_effect = [mock_tensor1, mock_tensor2]
        mock_safe_open.return_value = mock_file_handler
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Get first tensor
                result1 = handler.get_tensor("tensor1")
                
                # Get second tensor from same file
                result2 = handler.get_tensor("tensor2")
                
                # Verify safe_open was called only once (handler was cached)
                mock_safe_open.assert_called_once_with("file1.safetensors", framework="pytorch")
                
                # Verify get_tensor was called twice on the same handler
                self.assertEqual(mock_file_handler.get_tensor.call_count, 2)
                
                # Verify results
                self.assertTrue(torch.allclose(result1, mock_tensor1))
                self.assertTrue(torch.allclose(result2, mock_tensor2))
                
                # Verify handler was cached
                self.assertEqual(len(handler._handlers), 1)

    def test_get_tensor_tensor_not_found(self):
        """Test get_tensor raises ValueError when tensor is not found."""
        mock_routing = {"tensor1": "file1.safetensors"}
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Try to get non-existent tensor
                with self.assertRaises(ValueError) as context:
                    handler.get_tensor("nonexistent_tensor")
                
                self.assertIn("Weight file was not found", str(context.exception))
                self.assertIn("nonexistent_tensor", str(context.exception))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    def test_get_tensor_after_release_file_handler(self, mock_safe_open):
        """Test get_tensor after release_file_handler creates new handlers."""
        mock_routing = {"tensor1": "file1.safetensors"}
        
        mock_file_handler1 = MagicMock()
        mock_file_handler2 = MagicMock()
        mock_tensor1 = torch.tensor([1.0, 2.0])
        mock_tensor2 = torch.tensor([3.0, 4.0])
        mock_file_handler1.get_tensor.return_value = mock_tensor1
        mock_file_handler2.get_tensor.return_value = mock_tensor2
        
        mock_safe_open.side_effect = [mock_file_handler1, mock_file_handler2]
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Get tensor (creates handler)
                result1 = handler.get_tensor("tensor1")
                
                # Verify handler was created
                self.assertEqual(len(handler._handlers), 1)
                mock_safe_open.assert_called_once()
                
                # Release handlers
                handler.release_file_handler()
                self.assertEqual(len(handler._handlers), 0)
                
                # Get tensor again (should create new handler)
                result2 = handler.get_tensor("tensor1")
                
                # Verify new handler was created
                self.assertEqual(mock_safe_open.call_count, 2)
                self.assertEqual(len(handler._handlers), 1)
                
                # Verify results
                self.assertTrue(torch.allclose(result1, mock_tensor1))
                self.assertTrue(torch.allclose(result2, mock_tensor2))

    def test_get_filename_valid_tensor(self):
        """Test _get_filename with valid tensor name."""
        mock_routing = {"tensor1": "file1.safetensors"}
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                filename, tensor_name = handler._get_filename("tensor1")
                
                self.assertEqual(filename, "file1.safetensors")
                self.assertEqual(tensor_name, "tensor1")

    def test_get_filename_invalid_tensor(self):
        """Test _get_filename with invalid tensor name."""
        mock_routing = {"tensor1": "file1.safetensors"}
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value=mock_routing):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                with self.assertRaises(ValueError) as context:
                    handler._get_filename("nonexistent_tensor")
                
                self.assertIn("Weight file was not found", str(context.exception))
                self.assertIn("nonexistent_tensor", str(context.exception))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    def test_get_handler_new_file(self, mock_safe_open):
        """Test _get_handler creates new handler for new file."""
        mock_file_handler = MagicMock()
        mock_safe_open.return_value = mock_file_handler
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value={}):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                result = handler._get_handler("file1.safetensors")
                
                # Verify safe_open was called
                mock_safe_open.assert_called_once_with("file1.safetensors", framework="pytorch")
                
                # Verify handler was cached
                self.assertEqual(result, mock_file_handler)
                self.assertEqual(handler._handlers["file1.safetensors"], mock_file_handler)

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    def test_get_handler_existing_file(self, mock_safe_open):
        """Test _get_handler returns cached handler for existing file."""
        mock_file_handler = MagicMock()
        mock_safe_open.return_value = mock_file_handler
        
        with patch.object(WeightsFileHandler, '_get_weight_filenames', return_value=["file1.safetensors"]):
            with patch.object(WeightsFileHandler, '_load_weight_file_routing', return_value={}):
                handler = WeightsFileHandler(self.model_path, self.extension)
                
                # Get handler first time
                result1 = handler._get_handler("file1.safetensors")
                
                # Get handler second time
                result2 = handler._get_handler("file1.safetensors")
                
                # Verify safe_open was called only once
                mock_safe_open.assert_called_once()
                
                # Verify same handler was returned
                self.assertEqual(result1, result2)
                self.assertEqual(result1, mock_file_handler)

    @patch('mindie_llm.runtime.utils.loader.weight_utils.Path')
    def test_get_weight_filenames_directory_exists(self, mock_path_class):
        """Test _get_weight_filenames when directory exists."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        
        mock_file1 = MagicMock()
        mock_file1.name = "model-00001-of-00002.safetensors"
        mock_file2 = MagicMock()
        mock_file2.name = "model-00002-of-00002.safetensors"
        
        mock_path.glob.return_value = [mock_file1, mock_file2]
        mock_path_class.return_value = mock_path
        
        result = WeightsFileHandler._get_weight_filenames(self.model_path, self.extension)
        
        # Should return list of file paths as strings
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)

    @patch('mindie_llm.runtime.utils.loader.weight_utils.Path')
    def test_get_weight_filenames_no_files(self, mock_path_class):
        """Test _get_weight_filenames raises error when no files found."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.glob.return_value = []
        mock_path_class.return_value = mock_path
        
        with self.assertRaises(FileNotFoundError) as context:
            WeightsFileHandler._get_weight_filenames(self.model_path, self.extension)
        
        self.assertIn("No local weights found", str(context.exception))
        self.assertIn(self.extension, str(context.exception))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.os.path.isfile')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.Path')
    def test_get_weight_filenames_without_index_file(self, mock_path_class, mock_isfile):
        """Test _get_weight_filenames without index file."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        
        mock_file1 = MagicMock()
        mock_file1.name = "model-00001-of-00002.safetensors"
        mock_file2 = MagicMock()
        mock_file2.name = "model-00002-of-00002.safetensors"
        
        mock_path.glob.return_value = [mock_file1, mock_file2]
        mock_path_class.return_value = mock_path
        mock_isfile.return_value = False  # Index file doesn't exist
        
        result = WeightsFileHandler._get_weight_filenames(self.model_path, self.extension)
        
        # Should return all files when index file doesn't exist
        self.assertEqual(len(result), 2)

    @patch('mindie_llm.runtime.utils.loader.weight_utils.logger')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.json.load')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.safe_open')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.os.path.isfile')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.Path')
    def test_get_weight_filenames_with_index_file(self, mock_path_class, mock_isfile, mock_safe_open, mock_json_load, mock_logger):
        """Test _get_weight_filenames with index file filtering."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        
        mock_file1 = MagicMock()
        mock_file1.name = "model-00001-of-00002.safetensors"
        mock_file2 = MagicMock()
        mock_file2.name = "model-00002-of-00002.safetensors"
        mock_file3 = MagicMock()
        mock_file3.name = "model-00003-of-00002.safetensors"  # Not in index
        
        mock_path.glob.return_value = [mock_file1, mock_file2, mock_file3]
        mock_path_class.return_value = mock_path
        mock_isfile.return_value = True  # Index file exists
        
        # Mock index file content
        mock_index_content = {
            "weight_map": {
                "tensor1": "model-00001-of-00002.safetensors",
                "tensor2": "model-00002-of-00002.safetensors"
            }
        }
        mock_file_context = MagicMock()
        mock_file_context.__enter__ = Mock(return_value=mock_file_context)
        mock_file_context.__exit__ = Mock(return_value=False)
        mock_safe_open.return_value = mock_file_context
        mock_json_load.return_value = mock_index_content
        
        result = WeightsFileHandler._get_weight_filenames(self.model_path, self.extension)
        
        # Should return only files in index
        self.assertEqual(len(result), 2)
        # Should log filtered file
        mock_logger.info.assert_called()

    @patch('mindie_llm.runtime.utils.loader.weight_utils.Path')
    def test_get_weight_filenames_path_not_exists(self, mock_path_class):
        """Test _get_weight_filenames raises error when path doesn't exist."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path
        
        with self.assertRaises(FileNotFoundError) as context:
            WeightsFileHandler._get_weight_filenames(self.model_path, self.extension)
        
        self.assertIn("not exists or not a directory", str(context.exception))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.Path')
    def test_get_weight_filenames_not_directory(self, mock_path_class):
        """Test _get_weight_filenames raises error when path is not a directory."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = False  # Not a directory
        mock_path_class.return_value = mock_path
        
        with self.assertRaises(FileNotFoundError) as context:
            WeightsFileHandler._get_weight_filenames(self.model_path, self.extension)
        
        self.assertIn("not exists or not a directory", str(context.exception))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.check_path_permission')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.standardize_path')
    def test_load_weight_file_routing(self, mock_standardize_path, mock_check_permission, mock_safe_open):
        """Test _load_weight_file_routing."""
        filenames = ["file1.safetensors", "file2.safetensors"]
        
        mock_standardize_path.side_effect = lambda x, **kwargs: x  # Return as-is
        
        # Mock file handlers
        mock_file_handler1 = MagicMock()
        mock_file_handler1.__enter__ = Mock(return_value=mock_file_handler1)
        mock_file_handler1.__exit__ = Mock(return_value=False)
        mock_file_handler1.keys.return_value = ["tensor1", "tensor2"]
        
        mock_file_handler2 = MagicMock()
        mock_file_handler2.__enter__ = Mock(return_value=mock_file_handler2)
        mock_file_handler2.__exit__ = Mock(return_value=False)
        mock_file_handler2.keys.return_value = ["tensor3", "tensor4"]
        
        mock_safe_open.side_effect = [mock_file_handler1, mock_file_handler2]
        
        handler = WeightsFileHandler.__new__(WeightsFileHandler)
        handler._filenames = filenames
        
        routing = handler._load_weight_file_routing()
        
        # Verify standardize_path was called for each file
        self.assertEqual(mock_standardize_path.call_count, 2)
        
        # Verify check_path_permission was called for each file
        self.assertEqual(mock_check_permission.call_count, 2)
        
        # Verify routing was created correctly
        self.assertEqual(len(routing), 4)
        self.assertEqual(routing["tensor1"], "file1.safetensors")
        self.assertEqual(routing["tensor2"], "file1.safetensors")
        self.assertEqual(routing["tensor3"], "file2.safetensors")
        self.assertEqual(routing["tensor4"], "file2.safetensors")

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.check_path_permission')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.standardize_path')
    def test_load_weight_file_routing_duplicate_tensor(self, mock_standardize_path, mock_check_permission, mock_safe_open):
        """Test _load_weight_file_routing raises error for duplicate tensor."""
        filenames = ["file1.safetensors", "file2.safetensors"]
        
        mock_standardize_path.side_effect = lambda x, **kwargs: x
        
        # Mock file handlers with duplicate tensor name
        mock_file_handler1 = MagicMock()
        mock_file_handler1.__enter__ = Mock(return_value=mock_file_handler1)
        mock_file_handler1.__exit__ = Mock(return_value=False)
        mock_file_handler1.keys.return_value = ["tensor1", "tensor2"]
        
        mock_file_handler2 = MagicMock()
        mock_file_handler2.__enter__ = Mock(return_value=mock_file_handler2)
        mock_file_handler2.__exit__ = Mock(return_value=False)
        mock_file_handler2.keys.return_value = ["tensor2", "tensor3"]  # tensor2 is duplicate
        
        mock_safe_open.side_effect = [mock_file_handler1, mock_file_handler2]
        
        handler = WeightsFileHandler.__new__(WeightsFileHandler)
        handler._filenames = filenames
        
        with self.assertRaises(ValueError) as context:
            handler._load_weight_file_routing()
        
        self.assertIn("Weight was found in multiple files", str(context.exception))

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.check_path_permission')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.standardize_path')
    def test_load_weight_file_routing_empty_files(self, mock_standardize_path, mock_check_permission, mock_safe_open):
        """Test _load_weight_file_routing with empty files."""
        filenames = ["file1.safetensors"]
        
        mock_standardize_path.side_effect = lambda x, **kwargs: x
        
        mock_file_handler = MagicMock()
        mock_file_handler.__enter__ = Mock(return_value=mock_file_handler)
        mock_file_handler.__exit__ = Mock(return_value=False)
        mock_file_handler.keys.return_value = []  # No tensors
        
        mock_safe_open.return_value = mock_file_handler
        
        handler = WeightsFileHandler.__new__(WeightsFileHandler)
        handler._filenames = filenames
        
        routing = handler._load_weight_file_routing()
        
        # Should return empty routing
        self.assertEqual(len(routing), 0)

    @patch('mindie_llm.runtime.utils.loader.weight_utils.safetensors.safe_open')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.check_path_permission')
    @patch('mindie_llm.runtime.utils.loader.weight_utils.standardize_path')
    def test_load_weight_file_routing_single_file(self, mock_standardize_path, mock_check_permission, mock_safe_open):
        """Test _load_weight_file_routing with single file."""
        filenames = ["file1.safetensors"]
        
        mock_standardize_path.side_effect = lambda x, **kwargs: x
        
        mock_file_handler = MagicMock()
        mock_file_handler.__enter__ = Mock(return_value=mock_file_handler)
        mock_file_handler.__exit__ = Mock(return_value=False)
        mock_file_handler.keys.return_value = ["tensor1", "tensor2", "tensor3"]
        
        mock_safe_open.return_value = mock_file_handler
        
        handler = WeightsFileHandler.__new__(WeightsFileHandler)
        handler._filenames = filenames
        
        routing = handler._load_weight_file_routing()
        
        # Verify routing was created correctly
        self.assertEqual(len(routing), 3)
        self.assertEqual(routing["tensor1"], "file1.safetensors")
        self.assertEqual(routing["tensor2"], "file1.safetensors")
        self.assertEqual(routing["tensor3"], "file1.safetensors")


if __name__ == '__main__':
    unittest.main()
