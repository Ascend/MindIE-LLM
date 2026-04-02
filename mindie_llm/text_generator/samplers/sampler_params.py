# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from typing import Any, List, Literal
import math

from ...modeling.backend_type import BackendType


@dataclass
class HandlerParams:
    output_token_counts = None

    backend_type: str = None
    batch_size: int = 0
    min_tokens_to_keep = 1
    rank: int = 0
    vocab_size: int = 0

    def count_output_tokens(self, output_token_ids):
        if self.output_token_counts is None:
            if self.backend_type == BackendType.ATB or self.backend_type == BackendType.TORCH:
                import torch
                output_token_counts = torch.zeros((self.batch_size, self.vocab_size + 1),
                                                  dtype=output_token_ids.dtype,
                                                  device=output_token_ids.device)
                output_token_counts.scatter_add_(1, output_token_ids, torch.ones_like(output_token_ids))
                self.output_token_counts = output_token_counts[:, :self.vocab_size]
            elif self.backend_type == BackendType.MS:
                import mindspore as ms
                from mindspore import ops
                output_token_counts = ops.zeros((self.batch_size, self.vocab_size + 1), dtype=ms.int32)
                output_token_counts = ops.tensor_scatter_elements(output_token_counts, output_token_ids,
                                                                  ops.ones_like(output_token_ids), 1)
                self.output_token_counts = output_token_counts[:, :self.vocab_size]

    def clear_token_counts(self):
        self.output_token_counts = None


@dataclass
class SelectorParams:
    candidate_expansion_factor: int = 1
    device: Any = None
    eos_token_id: List[int] = None
    filter_value: float = -math.inf
    npu_id: int = 0
    num_threads: int = 8
    sampling_method: Literal['exponential', 'multinomial'] = 'multinomial'
    splitfuse_enabled: bool = False
    layerwise_disaggregated: bool = False