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
import torch
from mindie_llm.runtime.layers.embedding.rotary_embedding.yarn_scaling_rope import (
    YarnScalingRotaryEmbedding
)
from mindie_llm.runtime.layers.embedding.rotary_embedding.deepseek_v3_yarn_scaling_rope import (
    DeepseekV3YarnRotaryEmbedding
)


class TestYarnScalingRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.dtype = torch.float32

    def test_yarn_scaling_rope_full_rotation(self):
        rope = YarnScalingRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            original_max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=self.dtype,
            factor=2.0,
            beta_fast=32,
            beta_slow=1,
            extrapolation_factor=1.0,
            attention_factor=1.0,
            apply_yarn_scaling=True,
            truncate=True,
            mscale=1.0
        ).to(self.device)

        # Check buffers exist
        for name in ["inv_freq", "cos_cache", "sin_cache", "cos_sin_cache"]:
            self.assertTrue(hasattr(rope, name))

        # Check shapes
        self.assertEqual(rope.inv_freq.shape, (32,))  # rotary_dim // 2
        extended_len = int(2.0 * 2048)
        self.assertEqual(rope.cos_cache.shape, (extended_len, 32))
        self.assertEqual(rope.cos_sin_cache.shape, (extended_len, 64))

        # Check device
        self.assertEqual(rope.cos_cache.device.type, self.device)

    def test_yarn_scaling_rope_partial_rotation(self):
        rope = YarnScalingRotaryEmbedding(
            head_size=128,
            rotary_dim=64,  # partial
            original_max_position_embeddings=1024,
            base=10000.0,
            is_neox_style=False,
            dtype=torch.bfloat16,
            factor=4.0,
            beta_fast=16,
            beta_slow=2,
            mscale=1.2
        ).to(self.device)

        extended_len = int(4.0 * 1024)
        self.assertEqual(rope.cos_cache.shape, (extended_len, 32))  # 64 // 2
        self.assertEqual(rope.cos_sin_cache.shape, (extended_len, 64))
        self.assertEqual(rope.cos_cache.dtype, torch.bfloat16)

    def test_deepseek_v3_yarn_rope(self):
        rope = DeepseekV3YarnRotaryEmbedding(
            dim=128,
            original_max_position_embeddings=4096,
            base=10000.0,
            factor=8.0,
            beta_fast=64,
            beta_slow=1,
            is_neox_style=True,
            dtype=torch.float16,
            mscale=1.0,
            mscale_all_dim=0.0
        ).to(self.device)

        # Deepseek uses rotary_dim = dim
        extended_len = int(8.0 * 4096)
        self.assertEqual(rope.cos_cache.shape, (extended_len, 64))  # 128 // 2
        self.assertEqual(rope.cos_sin_cache.shape, (extended_len, 128))
        self.assertEqual(rope.cos_cache.device.type, self.device)
        self.assertEqual(rope.cos_cache.dtype, torch.float16)

        # Buffers registered
        for name in ["inv_freq", "cos_cache", "sin_cache", "cos_sin_cache"]:
            self.assertTrue(hasattr(rope, name))

    def test_buffers_are_non_persistent(self):
        rope = YarnScalingRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            original_max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            factor=2.0
        )
        state_dict = rope.state_dict()
        for buf in ["inv_freq", "cos_cache", "sin_cache", "cos_sin_cache"]:
            self.assertNotIn(buf, state_dict)

    def test_deepseek_buffers_non_persistent(self):
        rope = DeepseekV3YarnRotaryEmbedding(
            dim=64,
            original_max_position_embeddings=2048,
            factor=2.0
        )
        state_dict = rope.state_dict()
        for buf in ["inv_freq", "cos_cache", "sin_cache", "cos_sin_cache"]:
            self.assertNotIn(buf, state_dict)


if __name__ == '__main__':
    unittest.main()