# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""单元测试：structured_output_bitmask 模块（apply_token_bitmask_inplace 等）。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mindie_llm.text_generator.plugins.structured_output.structured_output_bitmask import (
    apply_token_bitmask_inplace,
    apply_token_bitmask_inplace_npu,
)


class TestApplyTokenBitmaskInplaceNpu(unittest.TestCase):
    """apply_token_bitmask_inplace_npu 的测试（需 torch）。"""

    @unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
    def test_vocab_size_leq_mask_coverage(self):
        """vocab_size <= mask_coverage_len：只覆盖 effective_len，不写尾部 -inf。"""
        vocab_size = 64
        # bitmask shape [batch, vocab_size//32] = [1, 2]；每格全位 1（0xFFFFFFFF）表示全部允许
        bitmask = np.full((1, 2), -1, dtype=np.int32)
        logits = torch.zeros(1, vocab_size)
        apply_token_bitmask_inplace_npu(
            logits,
            torch.from_numpy(bitmask),
            vocab_size,
        )
        # 全位 1 的 bitmask 不屏蔽任何位置，logits 应保持为 0（未置 -inf）
        self.assertFalse(torch.isinf(logits).any())

    @unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
    def test_vocab_size_gt_mask_coverage_tail_filled_with_inf(self):
        """vocab_size > mask_coverage_len：尾部 [effective_len:vocab_size] 置为 -inf。"""
        # mask 覆盖 32*2=64 个 token，vocab_size=100 > 64
        vocab_size = 100
        bitmask = np.ones((1, 2), dtype=np.int32)
        logits = torch.zeros(1, vocab_size)
        apply_token_bitmask_inplace_npu(
            logits,
            torch.from_numpy(bitmask),
            vocab_size,
        )
        effective_len = 64
        self.assertTrue(torch.isinf(logits[..., effective_len:]).all())

    @unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
    def test_mask_zero_positions_become_inf(self):
        """bitmask 为 0 的位置对应 logit 被置为 -inf。"""
        # 单个 int32 低 32 位：仅第 0 位为 1，其余为 0
        bitmask = np.array([[1]], dtype=np.int32)
        vocab_size = 32
        logits = torch.zeros(1, vocab_size)
        apply_token_bitmask_inplace_npu(
            logits,
            torch.from_numpy(bitmask),
            vocab_size,
        )
        self.assertTrue(torch.isinf(logits[0, 1:]).all())


class TestApplyTokenBitmaskInplace(unittest.TestCase):
    """apply_token_bitmask_inplace（numpy bitmask -> 调 NPU 版）的测试。"""

    @unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
    def test_success_applies_bitmask(self):
        """成功路径：numpy bitmask 转 tensor 并应用。"""
        vocab_size = 32
        # 全位 1（0xFFFFFFFF）表示全部允许，不屏蔽任何位置
        bitmask = np.full((1, 1), -1, dtype=np.int32)
        logits = torch.zeros(1, vocab_size)
        apply_token_bitmask_inplace(logits, bitmask, vocab_size)
        self.assertFalse(torch.isinf(logits).any())

    @unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
    def test_exception_logs_and_reraise(self):
        """NPU 应用失败时重新抛出异常。"""
        bitmask = np.ones((1, 1), dtype=np.int32)
        logits = torch.zeros(1, 32)
        with patch(
            "mindie_llm.text_generator.plugins.structured_output.structured_output_bitmask.apply_token_bitmask_inplace_npu",
            side_effect=RuntimeError("NPU error"),
        ):
            with self.assertRaises(RuntimeError):
                apply_token_bitmask_inplace(logits, bitmask, 32)


if __name__ == "__main__":
    unittest.main()
