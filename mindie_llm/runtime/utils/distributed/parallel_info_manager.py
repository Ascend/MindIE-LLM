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
from typing import List
from enum import Enum, unique

import torch
import torch_npu

from torch.distributed import ProcessGroup
import torch.distributed as dist
import torch_npu._C._distributed_c10d as dist_c
from mindie_llm.runtime.utils.distributed.utils import even_divide


DEFAULT_BUFFER_SIZE = 128


@unique
class ParallelType(str, Enum):
    """Enumeration of supported parallelism types."""
    WORLD = "world"
    ATTN_TP = "attn_tp"
    ATTN_DP = "attn_dp"
    ATTN_CP = "attn_cp"
    ATTN_INNER_SP = "attn_inner_sp"
    ATTN_O_PROJ_TP = "attn_o_proj_tp"
    MLP_TP = "mlp_tp"
    WORLD_EMBED_TP = "world_embed_tp"
    LM_HEAD_TP = "lm_head_tp"
    MOE_TP = "moe_tp"
    MOE_EP = "moe_ep"
    MOE_EP_MC2 = "moe_ep_mc2"


@dataclass
class ParallelInfo:
    """Encapsulates metadata for a specific parallel group.

    Attributes:
        buffer_size (int): HCCL communication buffer size used for this parallel group.
            Defaults to `DEFAULT_BUFFER_SIZE`.
        group_size (int): Number of ranks in each parallel group. If 1, parallelism is disabled.
        num_group (Optional[int]): Total number of such parallel groups in the full world.
            For example, with world_size=8 and group_size=2, num_group=4.
        rank_per_group (Optional[list[list[int]]]): A list of groups, where each group is a list
            of global rank IDs. Example: [[0,1], [2,3], [4,5], [6,7]] for TP with group_size=2.
        current_group_id (Optional[int]): Index of the parallel group that the current rank belongs to.
            Corresponds to the index in `rank_per_group`.
        rank (Optional[int]): Local rank within the current parallel group (0-indexed).
            For example, if current global rank is 3 and group is [2,3,4,5], then rank=1.
        process_group (Optional[ProcessGroup]): HCCL-based NPU communication group for collective
            operations (e.g., all-reduce) on device.
        cpu_process_group (Optional[ProcessGroup]): Gloo-based CPU communication group, typically
            used for data parallelism when CPU-side synchronization is required.
    """
    buffer_size: int = DEFAULT_BUFFER_SIZE
    group_size: int = 1
    num_group: int | None = None
    rank_per_group: list[list[int]] | None = None
    current_group_id: int | None = None
    rank: int | None = None
    process_group: ProcessGroup | None = None
    cpu_process_group: ProcessGroup | None = None
    preprocess_group: ProcessGroup | None = None

    def is_enabled(self) -> bool:
        """Checks if this parallel group is active (group_size > 1)."""
        return self.group_size > 1


class ParallelInfoManager:
    """Manages distributed parallel group configurations for LLM inference.

    Supports tensor parallelism (TP), data parallelism (DP), context parallelism (CP),
    sequence parallelism (SP), and MoE-specific parallelism (MoE-TP/EP).

    Attributes:
        world_size (int): Total number of processes.
        rank (int): Global rank of current process.
        local_rank (int): Local device rank (e.g., NPU index).
        parallel_config (Optional[Any]): Parallelism settings from LLM config.
        server_config (Dict[str, Any]): Runtime server configuration.
        hccl_buffer (int): Default HCCL buffer size.
    """
    def __init__(self, local_rank: int, llm_config=None, server_config=None):
        if llm_config is not None:
            self.parallel_config = llm_config.llm.parallel_options
        else:
            self.parallel_config = None
        if self.parallel_config is None:
            self.hccl_buffer = DEFAULT_BUFFER_SIZE
        else:
            self.hccl_buffer = self.parallel_config.hccl_buffer
        self.server_config = {} if server_config is None else server_config

        self.world_size: int = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.is_distribution_enabled = server_config.get("distributed_enable", False)

        # NOTE: --- Legacy convenience attributes (deprecated) begin---
        # Resolve parallel config
        self.world = self._init_tp_parallel_info(self.world_size)
        self.attn_tp = self._init_tp_parallel_info(server_config.get("tp", self.world_size))
        self.attn_dp = self._init_dp_parallel_info(server_config.get("dp", -1))
        self.attn_cp = self._init_dp_parallel_info(server_config.get("cp", -1))
        # NOTE: buffer_size needs to be calculated by a formula
        moe_tp_buffer_size = 64 if self.parallel_config is None else self.parallel_config.hccl_moe_tp_buffer
        self.moe_tp = self._init_tp_parallel_info(server_config.get("moe_tp", -1), moe_tp_buffer_size)
        moe_ep_buffer_size = 512 if self.parallel_config is None else self.parallel_config.hccl_moe_ep_buffer
        self.moe_ep = self._init_dp_parallel_info(server_config.get("moe_ep", -1), moe_ep_buffer_size)
        if self.moe_tp.group_size * self.moe_ep.group_size != self.world_size:
            error_msg = (
                f"MoE parallel strategy process number mismatch: "
                f"global world_size({self.world_size}) != "
                f"moe_tp.group_size({self.moe_tp.group_size}) × moe_ep.group_size({self.moe_ep.group_size})"
            )
            raise ValueError(error_msg)
        self.moe_ep_mc2 = self._init_dp_parallel_info(server_config.get("moe_ep", -1), moe_ep_buffer_size)
        # NOTE: --- Legacy convenience attributes (deprecated) end---

        group_size = server_config.get("sp", -1)
        group_size = 1 if group_size == -1 else group_size
        self.attn_inner_sp = self._init_tp_parallel_info(group_size)
        self.mlp_tp = self._init_tp_parallel_info(server_config.get("tp", self.world_size))
        self.word_embed_tp = self.attn_tp
        self.lm_head_tp = self.mlp_tp

        # Map parallel types to info objects
        self._parallel_type_map = {
            ParallelType.ATTN_TP: self.attn_tp,
            ParallelType.WORLD_EMBED_TP: self.attn_tp,
            ParallelType.ATTN_O_PROJ_TP: self.attn_tp,
            ParallelType.ATTN_DP: self.attn_dp,
            ParallelType.MLP_TP: self.mlp_tp,
            ParallelType.LM_HEAD_TP: self.mlp_tp,
            ParallelType.ATTN_CP: self.attn_cp,
            ParallelType.ATTN_INNER_SP: self.attn_inner_sp,
            ParallelType.MOE_TP: self.moe_tp,
            ParallelType.MOE_EP: self.moe_ep,
            ParallelType.WORLD: self.world,
            ParallelType.MOE_EP_MC2: self.moe_ep_mc2,
        }

    @staticmethod
    def pp_layers(num_layers: int) -> list[int]: 
        # NOTE: Legacy convenience methods (deprecated)
        """Returns layer-to-stage mapping for pipeline parallelism.

        Note:
            Pipeline parallelism is not currently implemented.
        """
        return list(range(num_layers))

    @staticmethod
    def has_pp() -> bool:
        """Checks if pipeline parallelism is enabled (always False)."""
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.PP).is_enabled()
        return False

    @staticmethod
    def _get_current_group_id(rank_per_group: list[list[int]], target_rank_id: int) -> int: 
        """Finds the group index containing the target rank."""
        for idx, group in enumerate(rank_per_group):
            if target_rank_id in group:
                return idx
        return None

    @staticmethod
    def _create_npu_process_group(parallel_info: ParallelInfo, is_all_reduce: bool = False) -> ProcessGroup: 
        """Creates an HCCL process group with custom buffer size.
           Need to traverse all groups to establish communication domains, 
        even if your rank is not present within them, otherwise it will 
        affect the global communication domain count.
        """
        cur_process_group = None
        for group_idx, group_ranks in enumerate(parallel_info.rank_per_group):
            options = dist_c.ProcessGroupHCCL.Options()
            if is_all_reduce:
                options.hccl_config = {"hccl_buffer_size": parallel_info.buffer_size, 
                                    "hccl_op_expansion_mode": 1}
            else:
                options.hccl_config = {"hccl_buffer_size": parallel_info.buffer_size}
            process_group = dist.new_group(ranks=group_ranks, pg_options=options)
            if group_idx == parallel_info.current_group_id:
                cur_process_group = process_group
        return cur_process_group

    @staticmethod
    def _create_cpu_process_group(parallel_info: ParallelInfo) -> ProcessGroup: 
        """Creates a Gloo-based CPU process group for communication (e.g., all-reduce on CPU)."""
        cur_process_group = None
        for group_idx, group_ranks in enumerate(parallel_info.rank_per_group):
            process_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if group_idx == parallel_info.current_group_id:
                cur_process_group = process_group
        return cur_process_group

    def has_attn_cp(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.ATTN_CP).is_enabled()
        return self.get(ParallelType.ATTN_CP).is_enabled()

    def has_attn_inner_sp(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.ATTN_INNER_SP).is_enabled()
        return self.get(ParallelType.ATTN_INNER_SP).is_enabled()

    def has_attn_o_proj_tp(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.ATTN_O_PROJ_TP).is_enabled()
        return self.get(ParallelType.ATTN_O_PROJ_TP).is_enabled()

    def has_dp(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.ATTN_DP).is_enabled()
        return self.get(ParallelType.ATTN_DP).is_enabled()

    def has_attn_tp(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.ATTN_TP).is_enabled()
        return self.get(ParallelType.ATTN_TP).is_enabled()

    def has_moe_tp(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.MOE_TP).is_enabled()
        return self.get(ParallelType.MOE_TP).is_enabled()

    def has_moe_ep(self) -> bool:
        # NOTE: deprecated, will change to parallelInfoManager.get(ParallelType.MOE_EP).is_enabled()
        return self.get(ParallelType.MOE_EP).is_enabled()

    def has_lm_head_local_tp(self) -> bool:
        return self.get(ParallelType.LM_HEAD_TP).group_size < self.world_size

    def get(self, parallel_type: ParallelType) -> ParallelInfo:
        if parallel_type not in self._parallel_type_map:
            raise KeyError(f"Unsupported ParallelType: {parallel_type}")
        return self._parallel_type_map[parallel_type]


    def _init_tp_parallel_info(
        self,
        group_size: int = None,
        hccl_buffersize: int = DEFAULT_BUFFER_SIZE
    ) -> ParallelInfo:
        """Initializes tensor-parallel-style groups (contiguous ranks)."""
        parallel_info = ParallelInfo(buffer_size=hccl_buffersize)        
        if group_size is None or group_size == -1:
            group_size = self.world_size

        parallel_info.group_size = group_size

        parallel_info.num_group = even_divide(self.world_size, parallel_info.group_size)
        parallel_info.rank_per_group = []
        # Create contiguous rank groups: each group contains consecutive global ranks
        for group_idx in range(parallel_info.num_group):
            ranks = range(group_idx * parallel_info.group_size, (group_idx + 1) * parallel_info.group_size)
            parallel_info.rank_per_group.append(list(ranks))
        parallel_info.current_group_id = self._get_current_group_id(parallel_info.rank_per_group, self.rank)
        parallel_info.rank = parallel_info.rank_per_group[parallel_info.current_group_id].index(self.rank)
        parallel_info.process_group = self._create_npu_process_group(parallel_info)
        return parallel_info


    def _init_dp_parallel_info(
        self,
        group_size: int = None,
        hccl_buffersize: int = DEFAULT_BUFFER_SIZE
    ) -> ParallelInfo:
        """Initializes data-parallel-style groups (strided ranks)."""
        parallel_info = ParallelInfo(buffer_size=hccl_buffersize)        
        if group_size is not None and group_size != -1:
            parallel_info.group_size = group_size

        parallel_info.num_group = even_divide(self.world_size, parallel_info.group_size)
        parallel_info.rank_per_group = []
        for group_idx in range(parallel_info.num_group):
            ranks = range(group_idx, self.world_size, parallel_info.num_group)
            parallel_info.rank_per_group.append(list(ranks))
        parallel_info.current_group_id = self._get_current_group_id(parallel_info.rank_per_group, self.rank)
        parallel_info.rank = parallel_info.rank_per_group[parallel_info.current_group_id].index(self.rank)

        parallel_info.process_group = self._create_npu_process_group(parallel_info)
        parallel_info.preprocess_group = self._create_cpu_process_group(parallel_info)
        return parallel_info
