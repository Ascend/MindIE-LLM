# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Optional

from enum import Enum
from mindie_llm.runtime.utils.npu.device_utils import DeviceType
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType
from mindie_llm.runtime.model_runner.forward_context_exp import get_mc2_token_capacity
from mindie_llm.utils.log.logging import logger

from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.npu.device_utils import get_npu_node_info


class MoECommType(Enum):
    """Supported MoE communication operator types."""
    ALLGATHER = "allgather"
    MC2 = "mc2"
    ALLTOALL = "all2all"
    FUSED_ALLTOALL = "fused_all2all"
    FUSED_MC2 = "fused_mc2"


class MoECommStrategyBase:
    """Base class for MoE communication strategy selection."""
    
    @staticmethod
    def is_applicable(quant_type: Optional[str], max_num_tokens_per_device: Optional[int]) -> bool:
        """Check if strategy applies to given context."""
        raise NotImplementedError
    
    @staticmethod
    def get_comm_type() -> MoECommType:
        """Return communication type for this strategy."""
        raise NotImplementedError


class FusedMC2Strategy(MoECommStrategyBase):
    """Fused MC2 strategy: highest priority for Ascend 910C.
    
    Requirements: EP group size <= 32, no MoE TP, tokens within capacity.
    """
    
    @staticmethod
    def is_applicable(quant_type: Optional[str], max_num_tokens_per_device: Optional[int]) -> bool:
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()
        forward_ctx = get_forward_context()
        
        # EP must be enabled for any MoE communication strategy
        if not parallel_mgr.get(ParallelType.MOE_EP).is_enabled():
            return False
        # Hardware constraint: only Ascend 910C (910_93) supports fused MC2
        if device_type != DeviceType.ASCEND_910_93:
            return False
        
        # EP group size limit for fused operator efficiency
        is_ep_valid = parallel_mgr.get(ParallelType.MOE_EP_MC2).group_size <= 32
        # Token count must fit within per-device capacity
        tp_size = parallel_mgr.get(ParallelType.ATTN_TP).group_size
        num_tokens_per_device = (forward_ctx.batch_descriptor.num_tokens + tp_size - 1) // tp_size
        is_num_tokens_valid = num_tokens_per_device <= max_num_tokens_per_device
        if not (is_ep_valid and is_num_tokens_valid):
            return False
        # Incompatible with MoE TP parallelism
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            return False
        return True
    
    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.FUSED_MC2


class MC2Strategy(MoECommStrategyBase):
    """MC2 strategy: for Ascend 910B/910C decode phase.
    
    910B: requires world size >= 16. 910C: strict token capacity check.
    """
    
    @staticmethod
    def is_applicable(quant_type: Optional[str], max_num_tokens_per_device: Optional[int]) -> bool:
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()
        forward_ctx = get_forward_context()
        if not parallel_mgr.get(ParallelType.MOE_EP).is_enabled():
            return False
        
        if device_type == DeviceType.ASCEND_910B:
            # 910B: decode phase only, large cluster required
            if forward_ctx.is_prefill or parallel_mgr.world_size < 16:
                return False
        elif device_type == DeviceType.ASCEND_910_93:
            # 910C: decode phase only
            if forward_ctx.is_prefill:
                return False
            # Strict capacity check: fail fast if exceeded
            if forward_ctx.batch_descriptor.num_tokens > get_mc2_token_capacity():
                error_msg = (
                    f"MC2 operator limitation: tokens per card ({forward_ctx.batch_descriptor.num_tokens})"
                    f" exceed the limit of mc2_token_capacity = {get_mc2_token_capacity()}. "
                    f"Please reduce the batch_size during the decode stage."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            logger.error("Unsupported device type: {device_type}")
            raise RuntimeError("Unsupported device type: {device_type}")
        
        # Incompatible with MoE TP parallelism
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            return False
        return True
    
    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.MC2


class All2AllStrategy(MoECommStrategyBase):
    """All2All strategy: fallback for specific scenarios.
    
    910B: W4A8_DYNAMIC quant or Attn DP enabled.
    910C: prefill phase when fused MC2 unavailable.
    """
    
    @staticmethod
    def is_applicable(quant_type: Optional[str], max_num_tokens_per_device: Optional[int]) -> bool:
        device_type = get_npu_node_info().get_device_type()
        parallel_mgr = get_parallel_info_manager()
        if not parallel_mgr.get(ParallelType.MOE_EP).is_enabled():
            return False
        
        if device_type == DeviceType.ASCEND_910B:
            # 910B: prefer W4A8_DYNAMIC; allow fallback if Attn DP enabled
            if quant_type != QuantType.W4A8_DYNAMIC:
                return parallel_mgr.get(ParallelType.ATTN_DP).is_enabled() or \
                       parallel_mgr.get(ParallelType.ATTN_CP).is_enabled()
        
        # Compatibility rule: MoE TP + All2All requires special handling
        if parallel_mgr.get(ParallelType.MOE_TP).is_enabled():
            # Attn DP + MoE TP > 1 is not supported for MC2/All2All
            if parallel_mgr.get(ParallelType.ATTN_DP).is_enabled():
                err_msg = 'MoECommType.MC2 and MoECommType.ALLTOALL: Do not support moe_tp > 1'
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            return False
        return True
    
    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.ALLTOALL


class AllGatherStrategy(MoECommStrategyBase):
    """AllGather strategy: universal fallback with lowest priority."""
    
    @staticmethod
    def is_applicable(quant_type: Optional[str], max_num_tokens_per_device: Optional[int]) -> bool:
        # Fallback: applicable when no other strategy matches
        return True
    
    @staticmethod
    def get_comm_type() -> MoECommType:
        return MoECommType.ALLGATHER


# Strategy priority order: first applicable strategy is selected
MOE_COMM_STRATEGIES = [
    FusedMC2Strategy,      # P0: Optimal (910C Fused)
    MC2Strategy,           # P1: High perf (large cluster/decode)
    All2AllStrategy,       # P2: Specific quant/prefill
    AllGatherStrategy,     # P3: Universal fallback
]
