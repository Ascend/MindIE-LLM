# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from enum import Enum
from mindie_llm.runtime.utils.npu.device_utils import DeviceType
from mindie_llm.runtime.model_runner.forward_context_exp import get_mc2_token_capacity
from mindie_llm.utils.log.logging import logger

from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.npu.device_utils import get_npu_node_info

MAX_FUSED_MC2_OPERATOR_CAPACITY = 4096  # limit of operator of fused_mc2


def cal_num_tokens_per_device(parallel_mgr, forward_ctx):
    """Calculation of the number of tokens obtained by a single device before MOE communication"""
    # CP
    if parallel_mgr.get(ParallelType.ATTN_CP).is_enabled():
        num_tokens_per_device = forward_ctx.batch_descriptor.num_tokens
    # DP
    else:
        num_tokens_per_device = forward_ctx.dp_metadata.max_tokens_across_dp_cpu
    # TP + flash_comm
    if (
        parallel_mgr.get(ParallelType.ATTN_TP).is_enabled()
        and forward_ctx.batch_descriptor.is_flash_comm_enabled
    ):
        tp_size = parallel_mgr.get(ParallelType.ATTN_TP).group_size
        num_tokens_per_device = (num_tokens_per_device + tp_size - 1) // tp_size
    return num_tokens_per_device


class MoECommType(Enum):
    """MoE communication operator types."""

    ALLGATHER = "allgather"
    MC2 = "mc2"
    ALLTOALL = "all2all"
    FUSED_ALLTOALL = "fused_all2all"
    FUSED_MC2 = "fused_mc2"


class MoECommStrategyBase:
    """Base class for MoE communication strategy selection."""

    @staticmethod
    def is_applicable(num_experts_per_ep_rank: int) -> bool:
        """Check if the strategy applies to the given context."""
        raise NotImplementedError

    @staticmethod
    def get_comm_type() -> MoECommType:
        """Return the communication type for this strategy."""
        raise NotImplementedError


class FusedMC2Strategy(MoECommStrategyBase):
    """Fused MC2 strategy: highest priority for Ascend 910C.

    Constraints: EP group size <= 32, no MoE TP, tokens within capacity.
    """

    @staticmethod
    def is_applicable(num_experts_per_ep_rank: int) -> bool:
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()
        forward_ctx = get_forward_context()

        # EP must be enabled
        if not parallel_mgr.get(ParallelType.MOE_EP).is_enabled():
            return False
        # Incompatible with MoE TP
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            return False
        # Only Ascend 910C supports fused MC2
        if device_type != DeviceType.ASCEND_910_93:
            return False
        # Check EP group size limit
        if parallel_mgr.get(ParallelType.MOE_EP_MC2).group_size > 32:
            return False
        # Check num tokens for fused_mc2 operator limitation
        num_tokens_per_device = cal_num_tokens_per_device(parallel_mgr, forward_ctx)
        if num_tokens_per_device > MAX_FUSED_MC2_OPERATOR_CAPACITY:
            err_msg = (
                f"num_tokens_per_device({num_tokens_per_device}) is larger than "
                f"MAX_FUSED_MC2_OPERATOR_CAPACITY({MAX_FUSED_MC2_OPERATOR_CAPACITY}),"
                f" please check the config."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)
        return True

    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.FUSED_MC2


class MC2Strategy(MoECommStrategyBase):
    """MC2 strategy: for Ascend 910B/910C decode phase.

    910B: requires world_size >= 16. 910C: strict token capacity check.
    """

    @staticmethod
    def is_applicable(num_experts_per_ep_rank: int) -> bool:
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()
        forward_ctx = get_forward_context()

        if not parallel_mgr.get(ParallelType.MOE_EP).is_enabled():
            return False
        # MoE TP + MoE EP not supported
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            return False
        num_tokens_per_device = cal_num_tokens_per_device(parallel_mgr, forward_ctx)
        if device_type == DeviceType.ASCEND_910B:
            # 910B: capacity and cluster constraints
            if (
                num_tokens_per_device > get_mc2_token_capacity()
                or parallel_mgr.world_size not in {16, 32, 64}
                or num_experts_per_ep_rank > 24
            ):
                return False

        elif device_type == DeviceType.ASCEND_910_93:
            # 910C: strict capacity check
            if (
                num_tokens_per_device > get_mc2_token_capacity()
                or num_experts_per_ep_rank > 24
            ):
                return False
        else:
            return False

        return True

    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.MC2


class All2AllStrategy(MoECommStrategyBase):
    """All2All strategy: fallback for specific scenarios on Ascend 910C.

    Used in prefill phase or when fused MC2 is unavailable.
    """

    @staticmethod
    def is_applicable(num_experts_per_ep_rank: int) -> bool:
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()
        forward_ctx = get_forward_context()
        if not parallel_mgr.get(ParallelType.MOE_EP).is_enabled():
            return False
        # MoE TP + MoE EP not supported
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            return False

        if device_type not in {DeviceType.ASCEND_910B, DeviceType.ASCEND_910_93}:
            return False

        # all2allv is not supported for graph mode(decode stage)
        if not forward_ctx.is_prefill:
            return False
        return True

    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.ALLTOALL


class AllGatherStrategy(MoECommStrategyBase):
    """AllGather strategy: universal fallback with lowest priority."""

    @staticmethod
    def is_applicable(num_experts_per_ep_rank: int) -> bool:
        # Fallback: applicable when no other strategy matches
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()

        # MoE TP + MoE EP not supported
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            return False

        # Attn DP not supported
        if parallel_mgr.get(ParallelType.ATTN_DP).is_enabled():
            return False

        if device_type != DeviceType.ASCEND_910B:
            return False

        return True

    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.ALLGATHER


# Strategy selection order: first applicable strategy is used
MOE_COMM_STRATEGIES = [
    FusedMC2Strategy,  # P0: Optimal (910C Fused MC2)
    MC2Strategy,  # P1: High perf (large cluster/decode)
    All2AllStrategy,  # P2: Fallback for prefill/specific cases
    AllGatherStrategy,  # P3: Universal fallback
]
