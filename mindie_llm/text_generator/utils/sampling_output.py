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
from typing import List, Tuple

import numpy as np


@dataclass
class SamplingOutput:
    sequence_ids: np.ndarray
    parent_sequence_ids: np.ndarray
    group_indices: List[Tuple[int, int]]
    repeating_indices: np.ndarray
    token_ids: np.ndarray
    logprobs: np.ndarray
    top_token_ids: np.ndarray
    top_logprobs: np.ndarray
    cumulative_logprobs: np.ndarray
    num_new_tokens: np.ndarray
    num_new_tokens_numpy: np.ndarray = None
    num_top_tokens: np.ndarray = None
    seeds: np.ndarray = None

    def to_deprecated(self):
        return self.token_ids, self.logprobs
    
    def truncate_after_eos(self, eos_token_id: int):
        """
        Truncates the token_ids after the first occurrence of the EOS token in each row,
        and updates num_new_tokens accordingly.
        
        :param eos_token_id: The ID of the EOS (End Of Sequence) token.
        """
        for i in range(self.token_ids.shape[0]):
            # Find the index of the first occurrence of EOS token in the row
            eos_index = np.where(self.token_ids[i] == eos_token_id)[0]
            if eos_index.size > 0:
                eos_index = eos_index[0]
                # Truncate the token_ids after the first EOS
                self.token_ids[i, eos_index + 1:] = 0  # Optionally set to 0 or any padding value
                self.num_new_tokens[i] = eos_index + 1  # Update the num_new_tokens
            else:
                # If no EOS token is found, num_new_tokens remains unchanged
                pass