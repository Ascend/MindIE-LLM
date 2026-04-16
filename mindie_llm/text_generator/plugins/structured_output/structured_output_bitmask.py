# coding=utf-8
# Copyright (c) 2023-2024 Jeremy Rifkin
# Copyright (c) 2024 by XGrammar Contributors
# Copyright (c) 2008 Google Inc.
# Copyright (c) 2017 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file implements token bitmask application for guided decoding,
# based on xgrammar (https://github.com/mlc-ai/xgrammar).
#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
结构化输出（xgrammar）约束解码时，将 token bitmask 应用到 logits 的工具函数。
仅被 PTA/MS 的 GuidedDecodingLogitsHandler 调用。
"""

from typing import TYPE_CHECKING

import numpy as np

from mindie_llm.utils.log.logging import logger

if TYPE_CHECKING:
    import torch


def apply_token_bitmask_inplace_npu(
    logits: "torch.Tensor",
    bitmask: "torch.Tensor",
    vocab_size: int,
) -> None:
    """在 NPU 上原地将 bitmask 应用到 logits（将不允许的 token 置为 -inf）。"""
    import torch

    mask_expanded = torch.repeat_interleave(bitmask, 32, dim=-1)
    bit_indices = torch.arange(32, device=logits.device, dtype=torch.int32).repeat(bitmask.shape[-1])
    bit_masks = (mask_expanded >> bit_indices) & 1

    mask_coverage_len = bit_masks.shape[-1]
    effective_len = min(vocab_size, mask_coverage_len)
    logits[..., :effective_len] = logits[..., :effective_len].masked_fill_(bit_masks == 0, float("-inf"))

    if vocab_size > effective_len:
        logits[..., effective_len:vocab_size] = float("-inf")


def apply_token_bitmask_inplace(
    logits: "torch.Tensor",
    bitmask: np.ndarray,
    vocab_size: int,
) -> None:
    """
    原地应用 token bitmask 到 logits（结构化输出 / xgrammar 约束解码用）。

    将 bitmask 中为 0 的位置对应的 logit 设为 -inf，
    使这些 token 在 softmax 后概率为 0。

    Args:
        logits: [batch_size, vocab_size] 的 logits tensor
        bitmask: [batch_size, vocab_size // 32] 的 int32 数组
        vocab_size: 词表大小
    """
    try:
        import torch

        bitmask_tensor = torch.from_numpy(np.asarray(bitmask, dtype=np.int32, order="C")).to(logits.device)
        apply_token_bitmask_inplace_npu(logits, bitmask_tensor, vocab_size)

    except Exception as e:
        logger.error(f"[BitmaskApply] apply_token_bitmask failed: {e}")
        raise
