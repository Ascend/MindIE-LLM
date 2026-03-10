# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import sys
from unittest.mock import patch, MagicMock
import torch
from mindie_llm.runtime.layers.embedding.rotary_embedding.base import RotaryEmbedding


# Mock torch_npu at import time to avoid ModuleNotFoundError
if 'torch_npu' not in sys.modules:
    sys.modules['torch_npu'] = MagicMock()
    sys.modules['torch_npu.npu'] = MagicMock()


class TestRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        # Mock NPU kernel to return tensors with correct shape (no actual computation)
        self.npu_patch = patch('torch_npu.npu_apply_rotary_pos_emb', side_effect=self._mock_npu_kernel)
        self.npu_patch.start()

    def tearDown(self):
        self.npu_patch.stop()

    def _mock_npu_kernel(self, query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Mock NPU kernel that preserves input shapes without actual computation."""
        # Return clones with same shape to simulate successful operation
        return query.clone(), key.clone()

    def test_init_buffers_registered(self):
        rope = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float16
        ).to(self.device)

        # Check that buffers are registered (cos_sin_cache removed in new implementation)
        self.assertTrue(hasattr(rope, 'inv_freq'))
        self.assertTrue(hasattr(rope, 'cos_cache'))
        self.assertTrue(hasattr(rope, 'sin_cache'))

        # Check shapes only
        self.assertEqual(rope.inv_freq.shape, (32,))  # rotary_dim // 2
        self.assertEqual(rope.cos_cache.shape, (128, 32))
        self.assertEqual(rope.sin_cache.shape, (128, 32))

        # Check device
        self.assertEqual(rope.cos_cache.device.type, self.device)

    def test_external_cos_sin_path_shape(self):
        # This path requires: neox=True, head_size=128, rotary_dim=128
        rope = RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=256,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float16
        ).to(self.device)

        positions = torch.arange(4, device=self.device)
        cos, sin = rope.get_cos_sin_for_positions(positions)  # [4, 64]

        # Shape adaptation: [4, 64] -> [4, 1, 1, 128]
        rope.cos_indexed_cache = cos.unsqueeze(1).unsqueeze(1).repeat(1, 1, 1, 2)  # 64 * 2 = 128
        rope.sin_indexed_cache = sin.unsqueeze(1).unsqueeze(1).repeat(1, 1, 1, 2)

        query = torch.randn(1, 4, 2, 128, dtype=torch.float16, device=self.device)
        key = torch.randn(1, 4, 2, 128, dtype=torch.float16, device=self.device)

        q_out, k_out = rope(positions, query, key)

    def test_get_cos_sin_for_positions_shape(self):
        rope = RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=128,
            base=10000.0,
            dtype=torch.float32
        ).to(self.device)

        positions = torch.tensor([10, 20, 30], device=self.device)
        cos, sin = rope.get_cos_sin_for_positions(positions)
        self.assertEqual(cos.shape, (3, 32))
        self.assertEqual(sin.shape, (3, 32))

    def test_buffers_are_non_persistent(self):
        rope = RotaryEmbedding(
            head_size=32,
            rotary_dim=32,
            max_position_embeddings=64,
            base=10000.0
        )
        state_dict = rope.state_dict()
        # Non-persistent buffers must NOT appear in state_dict
        for buf_name in ['inv_freq', 'cos_cache', 'sin_cache']:
            self.assertNotIn(buf_name, state_dict)


if __name__ == '__main__':
    unittest.main()