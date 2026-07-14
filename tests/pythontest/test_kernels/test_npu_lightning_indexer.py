# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests

device_id = 0
torch_npu.npu.set_device(int(device_id))


def _lightning_indexer(
    query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, layout_key, selected_count, sparse_mode
):
    layout_key = layout_key
    selected_count = selected_count
    sparse_mode = sparse_mode

    # B S1 N1 D -> B N1 S1 D
    s1 = query.shape[1]
    query = np.transpose(query, axes=(0, 2, 1, 3)).astype(np.float32)
    # B B N2 D -> S2 D -> D S2
    # (B*(S2//block_size), block_size, N2, D)))
    key = np.transpose(key.reshape(query.shape[0], -1, key.shape[-1]), axes=(0, 2, 1)).astype(np.float32)
    # B S1 N1 1 -> B N1 S1 1
    weights = np.transpose(weights, axes=(0, 2, 1, 3)).astype(np.float32)
    # relu_out is B N1 S1 S2
    relu_out = np.maximum(0, query.reshape(query.shape[0], -1, query.shape[-1]) @ key)
    relu_out = relu_out.reshape(query.shape[0], -1, s1, key.shape[-1])
    weight_out = relu_out * weights
    # reduce_out is B 1 S1 S2
    reduce_out = np.sum(weight_out, axis=1, keepdims=True)
    # sparse场景下三角置为-inf
    s1 = reduce_out.shape[2]
    s2 = reduce_out.shape[3]
    if sparse_mode == 3:
        for i in range(s1):
            reduce_out[:, :, -1 - i, s2 - i :] = float("-inf")
    sorted_indices = np.argsort(-reduce_out, kind="stable", axis=-1)
    # sparse场景下索引输出下三角置为-1
    if sparse_mode == 3:
        for i in range(s1):
            sorted_indices[:, :, -1 - i, s2 - i :] = -1

    sorted_res = sorted_indices[..., :selected_count]
    pad_width = [(0, 0)] * sorted_res.ndim
    pad_width[-1] = (0, selected_count - sorted_res.shape[-1])
    sorted_res = np.pad(sorted_res, pad_width, mode="constant", constant_values=-1)

    return sorted_res.astype(np.int32)


class TestCustomLightningIndexer(TestCase):
    def test_lightning_indexer_eager(self):
        B = 1
        S1 = 1
        S2 = 8192
        N1 = 64
        N2 = 1
        D = 128
        block_size = 256
        T = 8192
        layout_query = "BSND"  # 'TND'

        np.random.seed(0)
        if layout_query == "BSND":
            query = torch.tensor(np.random.uniform(-10, 10, (B, S1, N1, D))).to(torch.bfloat16)
        else:
            query = torch.tensor(np.random.uniform(-10, 10, (T, N1, D))).to(torch.bfloat16)

        key = torch.tensor(np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D))).to(torch.bfloat16)
        weights = torch.tensor(np.random.uniform(-1, 1, (B, S1, N1, 1))).to(torch.bfloat16)
        actual_seq_lengths_query = torch.tensor(np.random.uniform(S1, S1, (B))).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(np.random.uniform(S2, S2, (B))).to(torch.int32)
        block_table = torch.tensor([range(B * S2 // block_size)], dtype=torch.int32).reshape(B, -1)
        layout_key = "PA_BSND"
        selected_count = 2048
        sparse_mode = 3
        cpuout = _lightning_indexer(
            query.to(torch.float).numpy(),
            key.to(torch.float).numpy(),
            weights.to(torch.float).numpy(),
            actual_seq_lengths_query.numpy(),
            actual_seq_lengths_key.numpy(),
            layout_key,
            selected_count,
            sparse_mode,
        )

        torch_npu.npu.set_device(int(device_id))
        query = query.to("npu:%s" % device_id)
        key = key.to("npu:%s" % device_id)
        weights = weights.to("npu:%s" % device_id)
        actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % device_id)
        actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % device_id)
        block_table = block_table.to("npu:%s" % device_id)

        # start run custom ops
        print("======================== PTA eager BEGIN ========================")
        npu_out = torch.ops.mie_ops.npu_lightning_indexer(
            query,
            key,
            weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query=layout_query,
            layout_key=layout_key,
            selected_count=selected_count,
            sparse_mode=sparse_mode,
        )

        # compare result
        npu_out = npu_out.reshape(B, S1, selected_count).cpu()
        cpuout = cpuout.reshape(B, S1, selected_count)
        for i in range(B):
            for j in range(S1):
                for k in range(selected_count):
                    res = npu_out[i][j][k] == cpuout[i][j][k]
                    if not res.all():
                        print("B S K npu cpu = ", i, j, k, npu_out[i][j][k], cpuout[i][j][k])
        print("======================== PTA eager FINISH ========================")


if __name__ == "__main__":
    run_tests()
