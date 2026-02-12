#!/usr/bin/env python
# coding=utf-8
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
from unittest.mock import MagicMock, patch
import torch

from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding, ParallelLMHead
from mindie_llm.runtime.layers.quantization.unquantized import (
    UnquantizedEmbeddingMethod,
    UnquantizedLinearMethod,
)
from mindie_llm.runtime.layers.parameter import BaseParameter, ColumnParameter
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager


class TestVocabParallelEmbedding(unittest.TestCase):
    """Test cases for VocabParallelEmbedding with UnquantizedEmbeddingMethod."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock parallel info
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 1
        self.mock_parallel_info.process_group = None

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.word_embed_tp = self.mock_parallel_info

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init_without_quant_config(self):
        """Test initialization without quantization config."""
        num_embeddings = 1000
        embedding_dim = 512

        layer = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self.assertEqual(layer.num_embeddings, num_embeddings)
        self.assertEqual(layer.embedding_dim, embedding_dim)
        self.assertIsInstance(layer.quant_method, UnquantizedEmbeddingMethod)
        self.assertEqual(layer.tp_rank, 0)
        self.assertEqual(layer.tp_size, 1)
        self.assertFalse(layer.is_parallel)
        self.assertEqual(layer.output_partition_size, embedding_dim)
        self.assertIsNotNone(layer.weight)

    def test_init_with_custom_dtype(self):
        """Test initialization with custom dtype."""
        num_embeddings = 1000
        embedding_dim = 512
        params_dtype = torch.float16

        layer = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            params_dtype=params_dtype,
        )

        self.assertEqual(layer.params_dtype, params_dtype)
        self.assertEqual(layer.weight.data.dtype, params_dtype)

    def test_init_with_prefix(self):
        """Test initialization with prefix."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            prefix="embed_tokens",
        )

        self.assertEqual(layer.prefix, "embed_tokens")

    def test_init_with_partition_weights_false(self):
        """Test initialization with partition_weights=False."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=False,
        )

        self.assertFalse(layer.is_parallel)
        self.assertEqual(layer.output_partition_size, 512)

    def test_init_with_partition_weights_true_single_rank(self):
        """Test initialization with partition_weights=True but single rank."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=True,
        )

        # Even with partition_weights=True, if tp_size=1, is_parallel should be False
        self.assertFalse(layer.is_parallel)
        self.assertEqual(layer.output_partition_size, 512)

    def test_init_with_partition_weights_true_multi_rank(self):
        """Test initialization with partition_weights=True and multiple ranks."""
        # Set up multi-rank parallel info
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.rank = 0

        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=True,
        )

        self.assertTrue(layer.is_parallel)
        self.assertEqual(layer.output_partition_size, 256)  # 512 / 2

    def test_weight_shape(self):
        """Test that weight shape is correct."""
        num_embeddings = 1000
        embedding_dim = 512

        layer = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        expected_shape = (num_embeddings, embedding_dim)
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader_non_parallel(self):
        """Test weight loading in non-parallel mode."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=False,
        )

        # Weight shape is [num_embeddings, embedding_dim] = [1000, 512]
        loaded_weight = torch.randn(1000, 512)
        param = layer.weight

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight)

    def test_weight_loader_parallel(self):
        """Test weight loading in parallel mode."""
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.rank = 0

        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=True,
        )

        # Full weight shape is [num_embeddings, embedding_dim] = [1000, 512]
        # Will be partitioned along embedding_dim dimension
        loaded_weight = torch.randn(1000, 512)
        param = layer.weight

        # Mock the load_row_parallel_weight method
        with patch.object(param, 'load_row_parallel_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight, tp_rank=0)

    @patch('torch.nn.functional.embedding')
    @patch('torch.distributed.all_gather_into_tensor')
    def test_forward_non_parallel(self, mock_all_gather, mock_embedding):
        """Test forward pass in non-parallel mode."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=False,
        )

        # Initialize weight with known values
        # Weight shape is [num_embeddings, embedding_dim] = [1000, 512]
        layer.weight.data = torch.randn(1000, 512)

        # Create input token indices (1D tensor)
        x = torch.tensor([0, 1, 2])

        # Mock embedding to return expected output shape
        mock_embedding.return_value = torch.randn(3, 512)

        output = layer.forward(x)

        # Verify embedding was called with correct arguments
        mock_embedding.assert_called_once()
        call_args = mock_embedding.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.equal(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # In non-parallel mode, should directly return embedding output
        self.assertEqual(output.shape, (3, 512))
        mock_all_gather.assert_not_called()

    @patch('torch.nn.functional.embedding')
    @patch('torch.distributed.all_gather_into_tensor')
    def test_forward_parallel(self, mock_all_gather, mock_embedding):
        """Test forward pass in parallel mode."""
        self.mock_parallel_info.group_size = 2
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.process_group = MagicMock()

        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            partition_weights=True,
        )

        # Initialize weight
        # Weight shape is [num_embeddings, output_partition_size] = [1000, 256]
        layer.weight.data = torch.randn(1000, 256)  # Partitioned size

        # Create input token indices (1D tensor)
        x = torch.tensor([0, 1, 2])

        # Mock embedding to return partitioned output
        mock_embedding.return_value = torch.randn(3, 256)

        # Mock all_gather to simulate gathering from multiple ranks
        def mock_all_gather_side_effect(output_tensor, input_tensor, group=None):
            # Simulate gathering: copy input to all positions
            output_tensor.copy_(input_tensor.repeat(2, 1, 1))

        mock_all_gather.side_effect = mock_all_gather_side_effect

        output = layer.forward(x)

        # Verify embedding was called
        mock_embedding.assert_called_once()
        call_args = mock_embedding.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.equal(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Should call all_gather in parallel mode
        mock_all_gather.assert_called_once()
        self.assertEqual(output.shape, (3, 512))  # After gathering and reshaping

    def test_extra_repr(self):
        """Test extra_repr method."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            params_dtype=torch.float32,
        )

        repr_str = layer.extra_repr()
        self.assertIn("num_embeddings=1000", repr_str)
        self.assertIn("embedding_dim=512", repr_str)
        self.assertIn("UnquantizedEmbeddingMethod", repr_str)
        self.assertIn("tp_size=1", repr_str)

    def test_unquantized_embedding_method_apply(self):
        """Test that UnquantizedEmbeddingMethod.apply is called correctly."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Initialize weight
        # Weight shape is [num_embeddings, embedding_dim] = [1000, 512]
        layer.weight.data = torch.randn(1000, 512)

        # Create input token indices (1D tensor)
        x = torch.tensor([0, 1, 2])

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = torch.randn(3, 512)
            output = layer.forward(x)

            mock_apply.assert_called_once_with(layer, x)
            self.assertEqual(output.shape, (3, 512))

    @patch('torch.nn.functional.embedding')
    def test_unquantized_embedding_method_apply_real_computation(self, mock_embedding):
        """Test UnquantizedEmbeddingMethod.apply with mocked embedding function."""
        layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Initialize weight with known values for verification
        # Weight shape is [num_embeddings, embedding_dim] = [1000, 512]
        torch.manual_seed(42)
        layer.weight.data = torch.randn(1000, 512)

        # Create input token indices (1D tensor)
        x = torch.tensor([0, 1, 2])

        # Mock embedding to return expected output
        mock_embedding.return_value = torch.randn(3, 512)

        # Call forward with mocked embedding
        output = layer.forward(x)

        # Verify embedding was called with correct arguments
        mock_embedding.assert_called_once()
        call_args = mock_embedding.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.equal(call_args[0][0], x))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Verify output shape
        self.assertEqual(output.shape, (3, 512))


class TestParallelLMHead(unittest.TestCase):
    """Test cases for ParallelLMHead with UnquantizedLinearMethod."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock parallel info for word_embed_tp (used in parent init)
        self.mock_word_embed_tp = MagicMock()
        self.mock_word_embed_tp.rank = 0
        self.mock_word_embed_tp.group_size = 1

        # Create mock parallel info for lm_head_tp (used in _post_init)
        self.mock_lm_head_tp = MagicMock()
        self.mock_lm_head_tp.rank = 0
        self.mock_lm_head_tp.group_size = 1
        self.mock_lm_head_tp.process_group = None

        # Create mock parallel info manager
        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.word_embed_tp = self.mock_word_embed_tp
        self.mock_parallel_info_manager.lm_head_tp = self.mock_lm_head_tp

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

    def tearDown(self):
        """Clean up after tests."""
        set_parallel_info_manager(None)

    def test_init_without_bias(self):
        """Test initialization without bias."""
        num_embeddings = 1000
        embedding_dim = 512

        layer = ParallelLMHead(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            bias=False,
        )

        self.assertEqual(layer.num_embeddings, num_embeddings)
        self.assertEqual(layer.embedding_dim, embedding_dim)
        self.assertFalse(layer.has_bias)
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        self.assertIsNone(layer.bias)

    def test_init_with_bias(self):
        """Test initialization with bias."""
        num_embeddings = 1000
        embedding_dim = 512

        layer = ParallelLMHead(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            bias=True,
        )

        self.assertTrue(layer.has_bias)
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.data.shape, (1000,))  # Full vocab size in non-parallel

    def test_init_with_custom_dtypes(self):
        """Test initialization with custom weight and bias dtypes."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            bias=True,
            weight_dtype=torch.float16,
            bias_dtype=torch.float32,
        )

        self.assertEqual(layer.weight_dtype, torch.float16)
        self.assertEqual(layer.bias_dtype, torch.float32)
        self.assertEqual(layer.weight.data.dtype, torch.float16)
        self.assertEqual(layer.bias.data.dtype, torch.float32)

    def test_init_uses_lm_head_tp(self):
        """Test that ParallelLMHead uses lm_head_tp instead of word_embed_tp."""
        self.mock_lm_head_tp.group_size = 2
        self.mock_lm_head_tp.rank = 1

        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Should use lm_head_tp, not word_embed_tp
        self.assertEqual(layer.tp_rank, 1)
        self.assertEqual(layer.tp_size, 2)
        self.assertEqual(layer.parallel_info, self.mock_lm_head_tp)

    def test_weight_shape_linear(self):
        """Test that weight shape is correct for linear layer."""
        num_embeddings = 1000
        embedding_dim = 512

        layer = ParallelLMHead(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        # For linear layer, weight shape should be [output_size, input_size]
        expected_shape = (num_embeddings, embedding_dim)
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_shape_parallel(self):
        """Test weight shape in parallel mode."""
        self.mock_lm_head_tp.group_size = 2
        self.mock_lm_head_tp.rank = 0

        num_embeddings = 1000
        embedding_dim = 512

        layer = ParallelLMHead(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        expected_shape = (500, embedding_dim)  # 1000 / 2
        self.assertEqual(layer.weight.data.shape, expected_shape)

    def test_weight_loader_column_parameter(self):
        """Test weight loading with ColumnParameter."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Create a ColumnParameter
        param = ColumnParameter(torch.randn(1000, 512))
        loaded_weight = torch.randn(1000, 512)

        # Mock the load_column_parallel_weight method
        with patch.object(param, 'load_column_parallel_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight, tp_rank=0)

    def test_weight_loader_base_parameter(self):
        """Test weight loading with BaseParameter."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Create a BaseParameter
        param = BaseParameter(torch.randn(1000))
        loaded_weight = torch.randn(1000)

        # Mock the load_weight method
        with patch.object(param, 'load_weight') as mock_load:
            layer.weight_loader(param, loaded_weight)
            mock_load.assert_called_once_with(loaded_weight=loaded_weight)

    def test_tie_weights(self):
        """Test tie_weights method."""
        embed_layer = VocabParallelEmbedding(
            num_embeddings=1000,
            embedding_dim=512,
            prefix="embed_tokens",
        )

        lm_head = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            prefix="lm_head",
        )

        result = lm_head.tie_weights(embed_layer)

        self.assertEqual(lm_head.prefix, "embed_tokens")
        self.assertEqual(result, lm_head)

    @patch('torch.nn.functional.linear')
    @patch('torch.distributed.all_gather_into_tensor')
    def test_forward_without_indices(self, mock_all_gather, mock_linear):
        """Test forward pass without lm_head_indices."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Initialize weight
        layer.weight.data = torch.randn(1000, 512)

        # Create input hidden states
        hidden_states = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 1000)

        output = layer.forward(hidden_states)

        # Verify linear was called
        mock_linear.assert_called_once()
        call_args = mock_linear.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.allclose(call_args[0][0], hidden_states))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Should call quant_method.apply
        self.assertEqual(output.shape, (2, 3, 1000))
        mock_all_gather.assert_not_called()  # Single rank, no gather

    @patch('torch.nn.functional.linear')
    @patch('torch.distributed.all_gather_into_tensor')
    def test_forward_with_indices(self, mock_all_gather, mock_linear):
        """Test forward pass with lm_head_indices."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Initialize weight
        layer.weight.data = torch.randn(1000, 512)

        # Create input hidden states
        hidden_states = torch.randn(1024, 512)
        lm_head_indices = torch.tensor([0, 2])  # Select first and third sequences

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 1000)

        output = layer.forward(hidden_states, lm_head_indices)

        # Verify linear was called (after gathering selected sequences)
        mock_linear.assert_called_once()
        # Should gather only selected sequences
        self.assertEqual(output.shape, (2, 1000))
        mock_all_gather.assert_not_called()  # Single rank, no gather

    @patch('torch.nn.functional.linear')
    @patch('torch.distributed.all_gather_into_tensor')
    def test_forward_parallel(self, mock_all_gather, mock_linear):
        """Test forward pass in parallel mode."""
        self.mock_lm_head_tp.group_size = 2
        self.mock_lm_head_tp.rank = 0
        self.mock_lm_head_tp.process_group = MagicMock()

        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
        )

        # Initialize weight (partitioned)
        layer.weight.data = torch.randn(500, 512)

        # Create input hidden states
        hidden_states = torch.randn(3, 512)

        # Mock linear to return partitioned output
        mock_linear.return_value = torch.randn(3, 500)

        # Mock all_gather
        def mock_all_gather_side_effect(output_tensor, input_tensor, group=None):
            output_tensor.copy_(input_tensor.repeat(2, 1, 1))

        mock_all_gather.side_effect = mock_all_gather_side_effect

        output = layer.forward(hidden_states)

        # Verify linear was called
        mock_linear.assert_called_once()
        call_args = mock_linear.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.allclose(call_args[0][0], hidden_states))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Should call all_gather in parallel mode
        mock_all_gather.assert_called_once()
        self.assertEqual(output.shape, (3, 1000))

    def test_unquantized_linear_method_apply(self):
        """Test that UnquantizedLinearMethod.apply is called correctly."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            bias=True,
        )

        # Initialize weight and bias
        layer.weight.data = torch.randn(1000, 512)
        layer.bias.data = torch.randn(1000)

        # Create input
        hidden_states = torch.randn(2, 3, 512)

        # Mock the quant_method.apply
        with patch.object(layer.quant_method, 'apply') as mock_apply:
            mock_apply.return_value = torch.randn(2, 3, 1000)
            output = layer.forward(hidden_states)

            mock_apply.assert_called_once_with(layer, hidden_states)
            self.assertEqual(output.shape, (2, 3, 1000))

    @patch('torch.nn.functional.linear')
    def test_unquantized_linear_method_with_bias(self, mock_linear):
        """Test UnquantizedLinearMethod with bias."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            bias=True,
        )

        # Initialize weight and bias
        layer.weight.data = torch.randn(1000, 512)
        layer.bias.data = torch.randn(1000)

        # Create input
        hidden_states = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 1000)

        # Forward should use bias
        output = layer.forward(hidden_states)

        # Verify linear was called with weight
        mock_linear.assert_called_once()
        call_args = mock_linear.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.allclose(call_args[0][0], hidden_states))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 1000))

    @patch('torch.nn.functional.linear')
    def test_unquantized_linear_method_without_bias(self, mock_linear):
        """Test UnquantizedLinearMethod without bias."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            bias=False,
        )

        # Initialize weight
        layer.weight.data = torch.randn(1000, 512)

        # Create input
        hidden_states = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 1000)

        # Forward should not use bias
        output = layer.forward(hidden_states)

        # Verify linear was called with weight only (no bias)
        mock_linear.assert_called_once()
        call_args = mock_linear.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.allclose(call_args[0][0], hidden_states))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 1000))
        self.assertIsNone(layer.bias)

    @patch('torch.nn.functional.linear')
    def test_unquantized_linear_method_skip_bias_add(self, mock_linear):
        """Test UnquantizedLinearMethod with skip_bias_add attribute."""
        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            bias=True,
        )

        # Initialize weight and bias
        layer.weight.data = torch.randn(1000, 512)
        layer.bias.data = torch.randn(1000)
        
        # Set skip_bias_add to True
        layer.skip_bias_add = True

        # Create input
        hidden_states = torch.randn(2, 3, 512)

        # Mock linear to return expected output
        mock_linear.return_value = torch.randn(2, 3, 1000)

        # Forward should skip bias add when skip_bias_add=True
        output = layer.forward(hidden_states)

        # Verify linear was called (bias handling is done in apply method)
        mock_linear.assert_called_once()
        call_args = mock_linear.call_args
        self.assertEqual(len(call_args[0]), 2)
        self.assertTrue(torch.allclose(call_args[0][0], hidden_states))
        self.assertTrue(torch.allclose(call_args[0][1], layer.weight.data))
        # Verify output shape
        self.assertEqual(output.shape, (2, 3, 1000))

    def test_bias_shape_parallel(self):
        """Test bias shape in parallel mode."""
        self.mock_lm_head_tp.group_size = 2
        self.mock_lm_head_tp.rank = 0

        layer = ParallelLMHead(
            num_embeddings=1000,
            embedding_dim=512,
            bias=True,
        )

        # In parallel mode, bias should be partitioned
        expected_shape = (500,)  # 1000 / 2
        self.assertEqual(layer.bias.data.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
