# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
from typing import List

import torch
import torch.distributed as dist

from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
    select_moe_comm_method,
    get_cached_dispatcher,
    MoECommType,
)
from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
    MoeAllGatherArgs,
    MoeMC2Args,
    MoeAll2AllVArgs,
)
from mindie_llm.runtime.layers.parameter import RowParameter, ColumnParameter
from mindie_llm.runtime.layers.quantization.ms_model_slim.w4a8 import (
    W4A8PerTokenFusedMoEMethod,
)
from mindie_llm.runtime.layers.quantization.quantization_config_base import (
    QuantizationConfigBase,
)
from mindie_llm.runtime.layers.quantization.unquantized import UnquantizedFusedMoEMethod
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.distributed.utils import even_divide
from mindie_llm.runtime.model_runner.forward_context_exp import get_forward_context

# Recommended value for fused operator output buffer size (in elements).
# Determined empirically based on current npu_dispatch_ffn_combine operator constraints.
MAX_OUTPUT_SIZE: int = 65536 * 2


class FusedMoE(CustomLayer):
    def __init__(
        self,
        num_experts: int,
        topk_num: int,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        weight_dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfigBase | None = None,
        prefix: str = "",
        suffix: List[str] = None,
        activation: str = "silu",
    ):
        super().__init__()

        self.parallel_info = get_parallel_info_manager()
        self.moe_tp_rank = get_parallel_info_manager().get(ParallelType.MOE_TP).rank
        self.moe_tp_size = get_parallel_info_manager().get(ParallelType.MOE_TP).group_size
        self.moe_ep_rank = get_parallel_info_manager().get(ParallelType.MOE_EP).rank
        self.moe_ep_size = get_parallel_info_manager().get(ParallelType.MOE_EP).group_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_per_partition = even_divide(self.intermediate_size, self.moe_tp_size)
        self.bias = bias
        self.weight_dtype = weight_dtype or torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix
        self.suffix = suffix
        self.activation = activation
        self.num_experts = num_experts
        self._num_experts_per_ep_rank = (self.num_experts + self.moe_ep_size - 1) // self.moe_ep_size
        self.topk = topk_num
        self.expert_list = assign_experts(self.num_experts, self.moe_ep_size)[self.moe_ep_rank]
        self.num_local_experts = len(self.expert_list)
        self.expert_map = torch.full(size=(self.num_experts,), fill_value=-1, device="npu")
        self.expert_map[self.expert_list] = 1
        if self.quant_config is None:
            self.quant_method = UnquantizedFusedMoEMethod()
        else:
            # Get moe quant method through gate proj weights of expert 0
            self.quant_method = self.quant_config.get_quant_method(self, prefix=f"{self.prefix}.0.{self.suffix[0]}")
        # Only created when MC2 used fused op
        self._moe_ep_group = None

        self._create_weights()
        self._post_init()

    def weight_loader(
        self,
        loaded_weight: torch.Tensor,
        expert_id: int,
        module_suffix: str,
        weight_name: str,
    ):
        param = getattr(self, self.weight_map.get(module_suffix) + weight_name)
        shard_size = self.intermediate_size_per_partition
        loaded_expert_id = self.expert_list.index(expert_id)
        if isinstance(self.quant_method, W4A8PerTokenFusedMoEMethod) and weight_name == "weight":
            # w4a8 quantization: pack two int4 values into one int8 unit; shard size halved
            shard_size = shard_size // 2
        if isinstance(param, RowParameter):
            param.load_expert_row_parallel_weight(loaded_weight, loaded_expert_id, self.moe_tp_rank)
        elif isinstance(param, ColumnParameter):
            if module_suffix == self.suffix[0]:
                param.load_expert_column_parallel_weight(
                    loaded_weight, loaded_expert_id, self.moe_tp_rank, 0, shard_size
                )
            elif module_suffix == self.suffix[2]:
                param.load_expert_column_parallel_weight(
                    loaded_weight,
                    loaded_expert_id,
                    self.moe_tp_rank,
                    shard_size,
                    shard_size,
                )
        else:
            param.load_expert_weight(loaded_weight, loaded_expert_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        moe_comm_type = select_moe_comm_method(
            num_experts_per_ep_rank=self._num_experts_per_ep_rank,
        )
        if moe_comm_type == MoECommType.FUSED_MC2:
            from mindie_llm.runtime.ops import mie_ops  # noqa

            # FUSED_MC2 mode: Directly invoke the fused dispatch + FFN + combine operator
            self._create_moe_ep_group()
            final_hidden_states = torch.empty_like(hidden_states)
            torch.ops.mie_ops.npu_dispatch_ffn_combine(
                x=hidden_states,
                weight1=[self.gate_up_weight],
                weight2=[self.down_weight],
                expert_idx=topk_ids,
                scale1=[self.fused_gate_up_weight_scale],
                scale2=[self.fused_down_weight_scale],
                probs=topk_weights.to(self.weight_dtype).to(torch.float32),
                group=self._moe_ep_group,
                max_output_size=MAX_OUTPUT_SIZE,
                out=final_hidden_states,
            )
            return final_hidden_states

        dispatcher = get_cached_dispatcher(moe_comm_type=moe_comm_type)

        moe_comm_args = self._build_moe_comm_args(moe_comm_type, hidden_states, topk_weights, topk_ids)

        moe_dispatch_output, dispatch_context = dispatcher.token_dispatch(moe_comm_args)

        moe_mlp_out = self.quant_method.apply(
            self,
            x=moe_dispatch_output["hidden_states"],
            group_list=moe_dispatch_output["group_list"],
            group_list_type=moe_dispatch_output["group_list_type"],
            dynamic_scale=moe_dispatch_output["dynamic_scale"],
        )

        final_hidden_states = dispatcher.token_combine(hidden_states=moe_mlp_out, ctx=dispatch_context)

        if moe_comm_type == MoECommType.ALLGATHER:
            # In ALLGATHER-based MoE communication, expert outputs are gathered
            # back to all ranks, but the hidden states are still sharded across
            # Tensor Parallel (TP) ranks. Therefore, an all-reduce over the MLP TP
            # group is needed to merge TP-partial hidden states into a full result.
            dist.all_reduce(final_hidden_states, group=self.parallel_info.world.process_group)
        return final_hidden_states

    def _post_init(self):
        weight_name = set()
        param_prefixes = ["gate_up_", "down_"]
        for name, _ in self.named_parameters():
            for prefix in param_prefixes:
                if prefix in name:
                    weight_name.add(name.replace(prefix, ""))
                    break
        self.weight_list = list(weight_name)
        self.weight_map = {
            self.suffix[0]: param_prefixes[0],
            self.suffix[1]: param_prefixes[1],
            self.suffix[2]: param_prefixes[0],
        }

    def _build_moe_comm_args(
        self,
        moe_comm_type: MoECommType,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        common_kwargs = {
            "hidden_states": hidden_states,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "num_experts": self.num_experts,
        }
        if moe_comm_type == MoECommType.ALLGATHER:
            return MoeAllGatherArgs(
                **common_kwargs,
                top_k=self.topk,
                expert_list=self.expert_list,
                expert_map=self.expert_map,
                with_quant=True,
            )
        elif moe_comm_type == MoECommType.MC2:
            forward_context = get_forward_context()
            return MoeMC2Args(
                **common_kwargs,
                mc2_mask=forward_context.mc2_mask,
                with_quant=True,
                shared_experts=None,
                quantized_x_for_share=None,
                dynamic_scale_for_share=None,
            )
        elif moe_comm_type == MoECommType.ALLTOALL:
            return MoeAll2AllVArgs(**common_kwargs)
        else:
            raise RuntimeError(f"FusedMoE: Unsupported moe_comm_type {moe_comm_type}")

    def _create_weights(self):
        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            weight_dtype=self.weight_dtype,
            bias_dtype=self.weight_dtype,
            weight_loader=None,
        )

    def _create_moe_ep_group(self):
        if self._moe_ep_group is not None:
            return
        parallel_info_manager = get_parallel_info_manager()
        device_group = parallel_info_manager.get(ParallelType.MOE_EP_MC2).process_group
        local_rank = dist.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self._moe_ep_group = backend.get_hccl_comm_name(local_rank)


def assign_experts(expert_count, world_size):
    per_device = math.ceil(expert_count / world_size)
    assignment = []
    if expert_count % world_size == 0:
        for i in range(world_size):
            assignment.append([i * per_device + j for j in range(per_device)])
    else:
        for i in range(world_size - 1):
            assignment.append([i * per_device + j for j in range(per_device)])
        assignment.append([])
        for i in range(expert_count % world_size):
            assignment[-1].append(per_device * (world_size - 1) + i)
    return assignment
