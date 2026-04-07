# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
from dataclasses import dataclass, field
from typing import Callable, List

from enum import Enum, unique
from threading import Lock

import torch
import torch_npu

from torch.distributed import ProcessGroup
import torch.distributed as dist
import torch_npu._C._distributed_c10d as dist_c
from mindie_llm.runtime.utils.distributed.utils import even_divide


DEFAULT_BUFFER_SIZE = 128
HCCL_BACKEND = "hccl"
GLOO_BACKEND = "gloo"


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

    This class holds configuration and runtime information for a logical parallelism group
    (e.g., tensor parallel, pipeline parallel, data parallel) in a distributed training setup.
    It supports lazy initialization of communication process groups on both NPU (via HCCL)
    and CPU (via Gloo) backends.

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
        is_reusable(bool): To determine whether the process group should be reused or not.
        _process_group (Optional[ProcessGroup]): HCCL-based NPU communication group for collective
            operations (e.g., all-reduce) on device.
        _cpu_process_group (Optional[ProcessGroup]): Gloo-based CPU communication group, typically
            used for data parallelism when CPU-side synchronization is required.
        _pg_factory (Optional[Callable[[], ProcessGroup]]): Factory function to lazily create
            the  process group when first accessed.
    """
    buffer_size: int = DEFAULT_BUFFER_SIZE
    group_size: int = 1
    num_group: int | None = None
    rank_per_group: list[list[int]] | None = None
    current_group_id: int | None = None
    rank: int | None = None
    is_reusable: bool = True

    _process_group: ProcessGroup = field(default=None, init=False)
    _cpu_process_group: ProcessGroup = field(default=None, init=False)
    _pg_factory: Callable[[], ProcessGroup] = field(default=None, init=False)

    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        group_size: int = 1,
        num_group: int = None,
        rank_per_group: list[list[int]] = None,
        current_group_id: int = None,
        rank: int = None,
        is_reusable: bool = True
    ):
        # initialize
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.num_group = num_group
        self.rank_per_group = rank_per_group
        self.current_group_id = current_group_id
        self.rank = rank
        self.is_reusable = is_reusable

        self._pg_factory = None
        self._process_group = None
        self._cpu_process_group = None
    
    @property
    def process_group(self) -> ProcessGroup | None:
        """Gets the HCCL-based NPU communication process group.

        Initializes the process group lazily using `_pg_factory` upon first access.
        Returns `None` if no factory is set or if parallelism is disabled (`group_size <= 1`).

        Returns:
            Optional[ProcessGroup]: The initialized NPU process group, or `None`.
        """
        if self._process_group is None and self._pg_factory is not None:
            self._process_group = self._pg_factory(HCCL_BACKEND)
        return self._process_group
    
    @property
    def cpu_process_group(self) -> ProcessGroup | None:
        """Gets the Gloo-based CPU communication process group.

        Initializes the process group lazily using `_cpu_pg_factory` upon first access.
        Returns `None` if no factory is set or if parallelism is disabled.

        Returns:
            Optional[ProcessGroup]: The initialized CPU process group, or `None`.
        """
        if self._cpu_process_group is None and self._pg_factory is not None:
            self._cpu_process_group = self._pg_factory(GLOO_BACKEND)
        return self._cpu_process_group
    
    def set_pg_factory(self, factory: Callable[[], ProcessGroup]) -> None:
        self._pg_factory = factory

    def is_enabled(self) -> bool:
        """Checks if this parallel group is active (group_size > 1).

        Returns:
            bool: `True` if the parallel group is enabled (i.e., contains more than one rank),
                  `False` otherwise.
        """
        return self.group_size > 1
    
    
class ParallelInfoManager:
    """Manages distributed parallel group configurations for LLM inference.

    This manager orchestrates the creation and caching of communication groups for various
    parallelism strategies used in large language model (LLM) inference, including:
    
    - Tensor Parallelism (TP)
    - Data Parallelism (DP)
    - Context Parallelism (CP)
    - Sequence Parallelism (SP)
    - Mixture-of-Experts (MoE) specific parallelism (MoE-TP and MoE-EP)

    It supports both runtime server configuration overrides and static LLM configuration,
    and provides a unified interface to access parallel group metadata via `ParallelType` keys.

    Attributes:
        world_size (int): Total number of processes in the global distributed job.
        rank (int): Global rank of the current process within the world.
        local_rank (int): Local device rank (e.g., NPU or GPU index on the current node).
        parallel_config (Optional[Any]): Parsed parallelism settings from the LLM configuration.
            Expected to contain fields like `hccl_buffer`, `hccl_moe_tp_buffer`, etc.
        server_config (Dict[str, Any]): Runtime server-side configuration that may override
            default parallelism settings (e.g., `"tp"`, `"dp"`, `"moe_tp"`).
        hccl_buffer (int): Default HCCL communication buffer size used for general parallel groups.
        _process_group_cache (dict): Static cache for created `ProcessGroup` instances.
            Key: `(tuple(sorted_ranks), backend, hccl_buffer_size)`
            Value: `torch.distributed.ProcessGroup`
        _process_group_cache_lock (threading.Lock): Thread-safe lock for `_process_group_cache`.
    """

    _process_group_cache: dict[tuple, ProcessGroup] = {}
    _process_group_cache_lock = Lock()

    def __init__(self, local_rank: int, llm_config=None, server_config=None):
        self.hccl_buffer = DEFAULT_BUFFER_SIZE
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
        moe_tp_buffer_size = 64 # moe_tp uses 64M buffer, no matter how long the seq is.
        self.moe_tp = self._init_tp_parallel_info(server_config.get("moe_tp", -1), moe_tp_buffer_size)
        moe_ep_buffer_size = 256 # moe_ep uses 256M buffer, no matter how long the seq is.
        self.moe_ep = self._init_dp_parallel_info(server_config.get("moe_ep", -1), moe_ep_buffer_size)
        if self.moe_tp.group_size * self.moe_ep.group_size != self.world_size:
            error_msg = (
                f"MoE parallel strategy process number mismatch: "
                f"global world_size({self.world_size}) != "
                f"moe_tp.group_size({self.moe_tp.group_size}) × moe_ep.group_size({self.moe_ep.group_size})"
            )
            raise ValueError(error_msg)
        moe_ep_mc2_buffer_size = int(os.getenv("HCCL_BUFFSIZE"))
        self.moe_ep_mc2 = self._init_dp_parallel_info(server_config.get("moe_ep", -1), \
                                                      moe_ep_mc2_buffer_size, is_reusable=False)
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

    @classmethod
    def _get_or_create_process_group(
        cls,
        ranks: list[int],
        backend: str = HCCL_BACKEND,
        hccl_buffer_size: int | None = None,
        is_reusable: bool = True,
        stream_id: int | None = None
    ):
        '''create or return cached Process Group
        '''
        if not is_reusable:
            if backend == HCCL_BACKEND:
                options = dist_c.ProcessGroupHCCL.Options()
                if hccl_buffer_size is not None:
                    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
                return dist.new_group(ranks=ranks, pg_options=options)
            else:
                return dist.new_group(ranks=ranks, backend=backend)

        if stream_id is None:
            # cpu backend stream_id is set -1 as default 
            stream_id = torch_npu.npu.current_stream().stream_id if backend == HCCL_BACKEND else -1
        
        sorted_ranks = tuple(sorted(ranks))
        key = (sorted_ranks, backend, hccl_buffer_size, stream_id)

        with cls._process_group_cache_lock:
            if key in cls._process_group_cache:
                return cls._process_group_cache[key]

            if backend == HCCL_BACKEND:
                options = dist_c.ProcessGroupHCCL.Options()
                if hccl_buffer_size is not None:
                    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
                pg = dist.new_group(ranks=ranks, pg_options=options)
            else:
                pg = dist.new_group(ranks=ranks, backend=backend)
            cls._process_group_cache[key] = pg
            return pg

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
    
    def _make_process_group_factory(
        self,
        parallel_info: ParallelInfo
    ) -> Callable[[], ProcessGroup]:
        """
        Returns a closure that lazily creates the process group for the current rank's subgroup.
        
        The closure captures `rank_per_group`, `backend`, etc., and returns the PG corresponding
        to the group containing `self.rank`.
        """
        def factory(backend: str) -> ProcessGroup:
            current_pg = None
            hccl_buffer_size = parallel_info.buffer_size if backend == HCCL_BACKEND else None
            is_reusable = parallel_info.is_reusable
            for group_ranks in parallel_info.rank_per_group:
                pg = self._get_or_create_process_group(
                    ranks=group_ranks,
                    backend=backend,
                    hccl_buffer_size=hccl_buffer_size,
                    is_reusable=is_reusable
                )
                if self.rank in group_ranks:
                    current_pg = pg
            return current_pg
        return factory


    def _init_tp_parallel_info(
            self, 
            group_size: int = None, 
            hccl_buffersize: int = DEFAULT_BUFFER_SIZE,
            is_reusable: bool = True
        ) -> ParallelInfo:
        """Initializes tensor-parallel-style groups (contiguous ranks)."""        
        if group_size is None or group_size == -1:
            group_size = self.world_size
        
        num_group = even_divide(self.world_size, group_size)

        rank_per_group = [
        list(range(group_idx * group_size, (group_idx + 1) * group_size))
        for group_idx in range(num_group)]

        current_group_id = self._get_current_group_id(rank_per_group, self.rank)
        local_rank = rank_per_group[current_group_id].index(self.rank)

        parallel_info = ParallelInfo(
            buffer_size=hccl_buffersize,
            group_size=group_size,
            num_group=num_group,
            rank_per_group=rank_per_group,
            current_group_id=current_group_id,
            rank=local_rank,
            is_reusable=is_reusable
        )
        pg_factory = self._make_process_group_factory(parallel_info)
        parallel_info.set_pg_factory(pg_factory)
        return parallel_info


    def _init_dp_parallel_info(
            self, 
            group_size: int = None, 
            hccl_buffersize: int = DEFAULT_BUFFER_SIZE,
            is_reusable: bool = True
        ) -> ParallelInfo:
        """Initializes data-parallel-style groups (strided ranks)."""      
        if group_size is not None and group_size == -1:
            group_size = 1
        num_group = even_divide(self.world_size, group_size)
        rank_per_group = [
        list(range(group_idx, self.world_size, num_group))
        for group_idx in range(num_group)
        ]

        current_group_id = self._get_current_group_id(rank_per_group, self.rank)
        local_rank = rank_per_group[current_group_id].index(self.rank)

        parallel_info = ParallelInfo(
            buffer_size=hccl_buffersize,
            group_size=group_size,
            num_group=num_group,
            rank_per_group=rank_per_group,
            current_group_id=current_group_id,
            rank=local_rank,
            is_reusable=is_reusable
        )
        pg_factory = self._make_process_group_factory(parallel_info)
        parallel_info.set_pg_factory(pg_factory)
        return parallel_info
        

