# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as torch_dist

from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.model_runner.forward_metadata.module_metadata import ModuleMetadata


@dataclass
class DPMetadata(ModuleMetadata):
    """Metadata for data parallel operations.
    
    Attributes:
        num_tokens_across_dp_cpu: Number of tokens across data parallel groups (CPU tensor).
        num_actual_tokens_across_dp_cpu: Actual number of tokens across data parallel groups (CPU tensor).
    """
    num_tokens_across_dp_cpu: torch.Tensor
    num_actual_tokens_across_dp_cpu: torch.Tensor
    max_tokens_across_dp_cpu: int

    def __post_init__(self) -> None:
        """Get and update number of tokens across data parallel groups.
        This method performs an all-reduce operation to gather token counts
        from all data parallel ranks and updates the internal state.
        """
        num_token_cur_dp = self.num_tokens_across_dp_cpu
        dp_para_info = get_parallel_info_manager().get(ParallelType.ATTN_DP)
        num_token_tensor = torch.tensor([
                num_token_cur_dp if i == dp_para_info.rank else 0 for i in range(dp_para_info.group_size)
            ], dtype=torch.int32, device="cpu")
        if dp_para_info.is_enabled():
            torch_dist.all_reduce(num_token_tensor, group=dp_para_info.cpu_process_group)
        self.num_tokens_across_dp_cpu = num_token_tensor
        self.max_tokens_across_dp_cpu = max(self.num_tokens_across_dp_cpu).item()
        self.num_actual_tokens_across_dp_cpu = self.num_tokens_across_dp_cpu

    @staticmethod
    def from_model_input(model_input: Any) -> 'DPMetadata':
        """Create DP metadata from model input.
        
        Args:
            model_input: Model input data.
            
        Returns:
            Created DP metadata instance.
        """
        num_actual_tokens = model_input.input_ids.shape[0]
        return DPMetadata(
            num_tokens_across_dp_cpu=num_actual_tokens,
            num_actual_tokens_across_dp_cpu=num_actual_tokens,
            max_tokens_across_dp_cpu=0
        )

    @staticmethod
    def is_enabled() -> bool:
        """Check if data parallel is enabled.
        
        Returns:
            True if data parallel is enabled, False otherwise.
        """
        return get_parallel_info_manager().get(ParallelType.ATTN_DP).is_enabled()

    @staticmethod
    def register_buffer(max_num_token: int, device: torch.device) -> None:
        """Register buffer for DP metadata.
        
        Args:
            max_num_token: Maximum number of tokens.
            device: Target device.
        """
        pass

    def to_device(self, device: torch.device) -> None:
        """Move DP metadata tensors to the specified device.
        
        Args:
            device: Target device.
        """
        pass

    def copy(self, num_actual_tokens: int, num_tokens: int) -> None:
        """Copy DP metadata with new token counts.
        
        Args:
            num_actual_tokens: Actual number of tokens.
            num_tokens: Total number of tokens (including padding).
        """
        self.num_tokens_across_dp_cpu = torch.tensor(
            [num_tokens] * get_parallel_info_manager().get(ParallelType.ATTN_DP).group_size)
