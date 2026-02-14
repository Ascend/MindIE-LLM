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
from typing import Optional

from mindie_llm.runtime.utils.npu.device_utils import DeviceType, get_npu_node_info
from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
    TokenDispatcherWithAll2AllV
)
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.model_runner.forward_context import get_forward_context
from mindie_llm.utils.log.logging import logger


class MoECommType(Enum):
    ALLGATHER = "allgather"
    MC2 = "mc2"
    ALLTOALL = "all2all"
    FUSED_ALLTOALL = "fused_all2all"


_COMM_TYPE_TO_DISPATCHER = {
    MoECommType.ALLGATHER: TokenDispatcherWithAllGather,
    MoECommType.MC2: TokenDispatcherWithMC2,
    MoECommType.ALLTOALL: TokenDispatcherWithAll2AllV,
}


def select_moe_comm_method(quant_type: Optional[str] = None
                          ) -> Optional[MoECommType]:
    """1. If expert parallel is not enabled, we use all-gather since MC2 and all-to-all
    are designed for expert parallelism.
    2. If expert parallel is enabled, we need to consider the soc version and the
    number of tokens. This is based on the observation that all-gather is more
    efficient than all-to-all when running on _910B.

        a. For _910B, we choose from MC2 and all-gather.

        b. For _910_93, we choose from MC2 and all-to-all.

        In both cases, we use MC2 when in prefill phase.

    Args:
        quant_type (Optional[str], optional): The quantization type. Defaults to None.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType: The selected MoE communication method.
    """
    ascend_device_type = get_npu_node_info().get_device_type()
    parallel_info_manager = get_parallel_info_manager()
    is_prefill = get_forward_context().is_prefill
    enable_expert_parallel = parallel_info_manager.get(ParallelType.MOE_EP).is_enabled()

    # NOTE: Due to unsolved error for mc2 in prefill phase, we disable mc2 in prefill phase.
    # NOTE: Temporarily not use parameter num_tokens to select MoECommType.
    if not enable_expert_parallel:
        moe_comm_type = MoECommType.ALLGATHER
    elif ascend_device_type in {DeviceType.ASCEND_910B}:
        if not is_prefill and parallel_info_manager.world_size >= 16:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLTOALL
    elif ascend_device_type in {DeviceType.ASCEND_910_93}:
        moe_comm_type = MoECommType.MC2 if not is_prefill else MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {ascend_device_type}")

    # NOTE: MoECommType.ALLGATHER: Currently does not support enabling DP.
    if moe_comm_type == MoECommType.ALLGATHER and parallel_info_manager.get(ParallelType.ATTN_DP).is_enabled():
        moe_comm_type = MoECommType.ALLTOALL
        logger.debug('Currently do not support MoECommType.ALLGATHER with dp.')

    # NOTE: MoECommType.MC2 and MoECommType.ALLTOALL: Do not support moe_tp > 1,
    # and require flash_comm to be enabled.
    if moe_comm_type in (MoECommType.MC2, MoECommType.ALLTOALL):
        if parallel_info_manager.get(ParallelType.MOE_TP).is_enabled():
            err_msg = 'MoECommType.MC2 and MoECommType.ALLTOALL: Do not support moe_tp > 1,'
            # try to change to MoECommType.ALLGATHER
            if parallel_info_manager.get(ParallelType.ATTN_DP).is_enabled():
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            else:
                logger.warning(err_msg)
                moe_comm_type = MoECommType.ALLGATHER

    return moe_comm_type


def get_cached_dispatcher(
        moe_comm_type: Optional[MoECommType]) -> Optional[MoETokenDispatcher]:
    if moe_comm_type is None or moe_comm_type not in _COMM_TYPE_TO_DISPATCHER:
        return None
    dispatcher_cls = _COMM_TYPE_TO_DISPATCHER[moe_comm_type]
    return dispatcher_cls()