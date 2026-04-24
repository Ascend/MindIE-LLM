# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Implement part of this file based on vllm-project/vllm
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu

from mindie_llm.runtime.model_runner.forward_context_exp import get_forward_context
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.singleton import Singleton
from mindie_llm.runtime.utils.npu.device_utils import DeviceType, get_npu_node_info


@dataclass
class MoeDispatchArgsBase:
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    num_experts: int


@dataclass
class MoeAllGatherArgs(MoeDispatchArgsBase):
    top_k: int
    expert_list: list
    expert_map: torch.Tensor
    with_quant: bool


@dataclass
class MoeMC2Args(MoeDispatchArgsBase):
    mc2_mask: torch.Tensor
    with_quant: bool
    shared_experts: Any
    quantized_x_for_share: Any
    dynamic_scale_for_share: Any


@dataclass
class MoeAll2AllVArgs(MoeDispatchArgsBase):
    pass


@dataclass
class DispatchContextBase:
    pass


@dataclass
class AllGatherDispatchContext(DispatchContextBase):
    expanded_row_idx: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class MC2DispatchContext(DispatchContextBase):
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_experts: int
    with_quant: bool
    mc2_mask: torch.Tensor
    shared_experts: Any
    global_bs: int
    assist_info_for_combine: torch.Tensor
    tp_recv_counts: torch.Tensor
    ep_recv_counts: torch.Tensor
    shared_act: Any
    swiglu_out_scale: Any


@dataclass
class All2AllVDispatchContext(DispatchContextBase):
    topk_weights: torch.Tensor
    num_experts: int
    num_local_experts: int
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor
    input_splits: list
    output_splits: list
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


class MoETokenDispatcher(ABC):
    @abstractmethod
    def token_dispatch(self, args: MoeDispatchArgsBase):
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self, hidden_states: torch.Tensor, ctx: DispatchContextBase):
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithAllGather(Singleton, MoETokenDispatcher):
    def __init__(self):
        super().__init__()
        self.parallel_info = get_parallel_info_manager()
        self.ep_size = self.parallel_info.get(ParallelType.MOE_EP).group_size

    def token_dispatch(self, args: MoeAllGatherArgs):
        num_tokens = args.hidden_states.shape[0]
        num_local_experts = args.num_experts // self.ep_size

        if args.expert_list is not None:
            first_expert_idx = args.expert_list[0]
            last_expert_idx = args.expert_list[-1] + 1
            mask = args.expert_map[args.topk_ids] != -1
            args.topk_weights = args.topk_weights * mask
        else:
            first_expert_idx = 0
            last_expert_idx = num_local_experts

        sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            args.hidden_states,
            args.topk_ids,
            active_num=num_tokens * args.top_k,
            expert_num=args.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=1 if args.with_quant else -1,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        # group_list_type代表group_list的表达形式:
        # 0: group_list中数值为分组轴大小的cumsum结果（累积和）。
        # 1: group_list中数值为分组轴上每组大小
        group_list_type = 1
        dispatch_output = {
            "group_list_type": group_list_type,
            "hidden_states": sorted_hidden_states,
            "group_list": expert_tokens,
            "dynamic_scale": pertoken_scale if args.with_quant else None,
        }
        context = AllGatherDispatchContext(
            expanded_row_idx=expanded_row_idx,
            topk_weights=args.topk_weights,
        )
        return dispatch_output, context

    def token_combine(self, hidden_states: torch.Tensor, ctx: AllGatherDispatchContext):
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states, sorted_indices=ctx.expanded_row_idx, probs=ctx.topk_weights
        )
        return final_hidden_states


class TokenDispatcherWithMC2(Singleton, MoETokenDispatcher):
    def __init__(self):
        self.parallel_info = get_parallel_info_manager()
        device_group = self.parallel_info.get(ParallelType.MOE_EP_MC2).process_group
        local_rank = dist.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = self.parallel_info.moe_ep.rank
        self.ep_world_size = self.parallel_info.moe_ep.group_size
        self.enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        self.ascend_device_type = get_npu_node_info().get_device_type()

    def select_dispatch_mc2_kwargs(self, args: MoeMC2Args):
        if args.with_quant:
            quant_mode = 2
        else:
            quant_mode = 0
        tp_size = get_parallel_info_manager().get(ParallelType.ATTN_TP).group_size
        max_bs = (get_forward_context().dp_metadata.max_tokens_across_dp_cpu + tp_size - 1) // tp_size
        global_bs = max_bs * self.ep_world_size  # must be identical across all ranks.
        kwargs_mc2_dispatch = {
            "x": args.hidden_states,
            "expert_ids": args.topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": args.num_experts,
            "global_bs": global_bs,
            "expert_token_nums_type": 0,  # 0: prefix sum of tokens per expert; 1: token count per expert
        }
        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.ascend_device_type in {DeviceType.ASCEND_910_93}:
            stage1_kwargs.update(
                {
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
            if self.enable_dispatch_v2:
                stage1_kwargs.update(
                    {
                        "x_active_mask": args.mc2_mask,
                    }
                )
        kwargs_mc2_dispatch.update(stage1_kwargs)
        return kwargs_mc2_dispatch

    def token_dispatch(self, args: MoeMC2Args):
        kwargs_mc2_dispatch = self.select_dispatch_mc2_kwargs(args)

        operator_output = (
            torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2_dispatch)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2_dispatch)
        )
        expand_x, dynamic_scale, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts = (
            operator_output[0:6]
        )

        shared_act, swiglu_out_scale = None, None
        if args.with_quant:
            if args.shared_experts is not None:
                share_up_out, _ = args.shared_experts.gate_up_proj(
                    (args.quantized_x_for_share, args.dynamic_scale_for_share)
                )
                shared_gate_up, shared_dequant_scale = share_up_out[0], share_up_out[1]

                shared_act_out = args.shared_experts.act_fn((shared_gate_up, shared_dequant_scale))
                shared_act, swiglu_out_scale = shared_act_out[0], shared_act_out[1]

        else:
            if args.shared_experts is not None:
                shared_gate_up, _ = args.shared_experts.gate_up_proj(args.hidden_states)
                shared_act = args.shared_experts.act_fn(shared_gate_up)

        group_list_type = 0
        dispatch_output = {
            "group_list_type": group_list_type,
            "hidden_states": expand_x,
            "group_list": expert_token_nums,
            "dynamic_scale": dynamic_scale,
        }
        context = MC2DispatchContext(
            topk_ids=args.topk_ids,
            topk_weights=args.topk_weights,
            num_experts=args.num_experts,
            with_quant=args.with_quant,
            mc2_mask=args.mc2_mask,
            shared_experts=args.shared_experts,
            global_bs=kwargs_mc2_dispatch["global_bs"],
            assist_info_for_combine=assist_info_for_combine,
            ep_recv_counts=ep_recv_counts,
            tp_recv_counts=tp_recv_counts,
            shared_act=shared_act,
            swiglu_out_scale=swiglu_out_scale,
        )

        return dispatch_output, context

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor, ctx: MC2DispatchContext):
        # moeCombine
        kwargs_mc2_combine = {
            "expand_x": hidden_states,
            "expert_ids": ctx.topk_ids,
            "expert_scales": ctx.topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": ctx.num_experts,
            "global_bs": ctx.global_bs,
        }
        if ctx.with_quant:
            tp_recv_counts = torch.empty(1, dtype=torch.int32, device=hidden_states.device)
        else:
            tp_recv_counts = ctx.tp_recv_counts
        stage3_kwargs = {
            "ep_send_counts": ctx.ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.enable_dispatch_v2:
            stage3_kwargs.update(
                {
                    "assist_info_for_combine": ctx.assist_info_for_combine,
                }
            )
        else:
            stage3_kwargs.update(
                {
                    "expand_idx": ctx.assist_info_for_combine,
                }
            )
        if self.ascend_device_type in {DeviceType.ASCEND_910_93}:
            stage3_kwargs.update(
                {
                    "tp_send_counts": tp_recv_counts,
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
            if self.enable_dispatch_v2:
                stage3_kwargs.update(
                    {
                        "x_active_mask": ctx.mc2_mask,
                    }
                )
        kwargs_mc2_combine.update(stage3_kwargs)
        return kwargs_mc2_combine

    def token_combine(self, hidden_states: torch.Tensor, ctx: MC2DispatchContext):
        kwargs_mc2_combine = self.get_combine_mc_kwargs(hidden_states, ctx)
        hidden_states = (
            torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2_combine)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_combine(**kwargs_mc2_combine)
        )
        if ctx.shared_experts is None:
            return hidden_states
        else:
            if ctx.with_quant:
                shared_hidden_states, _ = ctx.shared_experts.down_proj((ctx.shared_act, ctx.swiglu_out_scale))
            else:
                shared_hidden_states, _ = ctx.shared_experts.down_proj(ctx.shared_act)
            return hidden_states, shared_hidden_states


def async_all_to_all(
    input_,
    output_split_sizes,
    input_split_sizes,
    group,
):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=input_.device,
        )

    handle = dist.all_to_all_single(
        a2a_out,
        input_.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = dist.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        dist.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        dist.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


class TokenDispatcherWithAll2AllV(Singleton, MoETokenDispatcher):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.parallel_info = get_parallel_info_manager()
        self.ep_rank = self.parallel_info.moe_ep.rank
        self.ep_group = self.parallel_info.moe_ep.process_group
        self.ep_size = self.parallel_info.moe_ep.group_size
        self.with_quant = False

    def token_dispatch(self, args: MoeAll2AllVArgs):
        num_local_experts = args.num_experts // self.ep_size
        preprocess_result = self._dispatch_preprocess(
            args.hidden_states, args.topk_ids, args.num_experts, num_local_experts
        )

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            preprocess_result.permutated_local_input_tokens,
            preprocess_result.output_splits,
            preprocess_result.input_splits,
            self.ep_group,
        )
        dynamic_scale_after_all2all = None
        permute1_ep_all_to_all_handle.wait()
        preprocess_result.permutated_local_input_tokens.untyped_storage().resize_(0)

        (global_input_tokens, dynamic_scale, reversed_global_input_permutation_mapping) = self._dispatch_postprocess(
            global_input_tokens,
            dynamic_scale_after_all2all,
            num_local_experts,
            preprocess_result.global_input_tokens_local_experts_indices,
        )

        dispatch_output = {
            "hidden_states": global_input_tokens,
            "group_list": preprocess_result.tokens_per_expert,
            "dynamic_scale": dynamic_scale,
            "group_list_type": 1,
        }
        context = All2AllVDispatchContext(
            topk_weights=args.topk_weights,
            num_experts=args.num_experts,
            num_local_experts=num_local_experts,
            reversed_local_input_permutation_mapping=preprocess_result.reversed_local_input_permutation_mapping,
            reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
            input_splits=preprocess_result.input_splits,
            output_splits=preprocess_result.output_splits,
            hidden_shape=preprocess_result.hidden_shape,
            hidden_shape_before_permute=preprocess_result.hidden_shape_before_permute,
        )
        return dispatch_output, context

    def token_combine(self, hidden_states: torch.Tensor, ctx: All2AllVDispatchContext):
        hidden_states = self._combine_preprocess(
            hidden_states, ctx.num_local_experts, ctx.reversed_global_input_permutation_mapping
        )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, ctx.input_splits, ctx.output_splits, self.ep_group
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        output = self._combine_postprocess(
            permutated_local_input_tokens,
            ctx.reversed_local_input_permutation_mapping,
            ctx.topk_weights,
            ctx.hidden_shape_before_permute,
            ctx.hidden_shape,
        )

        return output

    @dataclass
    class PreprocessDispatchResult:
        permutated_local_input_tokens: torch.Tensor
        reversed_local_input_permutation_mapping: torch.Tensor
        tokens_per_expert: torch.Tensor
        input_splits: list
        output_splits: list
        hidden_shape: torch.Size
        hidden_shape_before_permute: torch.Size
        global_input_tokens_local_experts_indices: torch.Tensor

    def _dispatch_preprocess(self, hidden_states, topk_ids, num_experts, num_local_experts):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_shape[-1])
        tokens_per_expert, input_splits, output_splits, global_input_tokens_local_experts_indices = self._preprocess(
            topk_ids, num_experts, num_local_experts
        )

        hidden_shape_before_permute = hidden_shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=topk_ids.numel(),
        )
        return self.PreprocessDispatchResult(
            permutated_local_input_tokens=permutated_local_input_tokens,
            reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
            tokens_per_expert=tokens_per_expert,
            input_splits=input_splits,
            output_splits=output_splits,
            hidden_shape=hidden_shape,
            hidden_shape_before_permute=hidden_shape_before_permute,
            global_input_tokens_local_experts_indices=global_input_tokens_local_experts_indices,
        )

    def _preprocess(self, topk_ids: torch.Tensor, num_experts, num_local_experts) -> torch.Tensor:
        num_local_tokens_per_expert = torch.histc(topk_ids, bins=num_experts, min=0, max=num_experts)

        ep_size = self.ep_size

        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(ep_size, num_experts)

        local_expert_indices_offset = self.ep_rank * num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, local_expert_indices[0] : local_expert_indices[-1] + 1
        ]
        if num_global_tokens_per_local_expert is None:
            raise ValueError("num_global_tokens_per_local_expert must be set before sum.")
        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
        # ===================================================
        # num_global_tokens_per_expert shape: [ep_size, num_experts]
        # num_global_tokens_per_local_expert shape: [ep_size, num_local_experts]
        # num_tokens_per_local_expert shape: [num_local_experts]
        # ===================================================
        global_input_tokens_local_experts_indices = None
        if num_local_experts > 1:
            expert_ids_per_ep_rank = torch.arange(num_experts, dtype=torch.int32, device="npu") % num_local_experts
            if num_global_tokens_per_local_expert is None:
                raise ValueError("num_global_tokens_per_local_expert must be set before operations.")
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert, input_splits, output_splits, global_input_tokens_local_experts_indices

    def _dispatch_postprocess(
        self, global_input_tokens, dynamic_scale, num_local_experts, global_input_tokens_local_experts_indices
    ):
        # Early return if no local experts or no tokens
        if num_local_experts <= 1:
            return global_input_tokens, None

        # Handle quantized case
        if self.with_quant:
            expert_idx_2d = global_input_tokens_local_experts_indices.unsqueeze(-1)
            active_num = global_input_tokens_local_experts_indices.numel()

            # Handle case with no active tokens
            if active_num <= 0:
                reversed_global_input_permutation_mapping = global_input_tokens_local_experts_indices
                return global_input_tokens, dynamic_scale, reversed_global_input_permutation_mapping

            # Process with active tokens
            (global_input_tokens, reversed_global_input_permutation_mapping, _, expanded_scale) = (
                torch_npu.npu_moe_init_routing_v2(
                    global_input_tokens,
                    expert_idx_2d,
                    scale=dynamic_scale,
                    active_num=active_num,
                    expert_capacity=0,
                    expert_num=num_local_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, num_local_experts],
                    quant_mode=-1,
                    row_idx_type=0,
                )
            )
            return global_input_tokens, expanded_scale, reversed_global_input_permutation_mapping

        # Handle non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_input_tokens_local_experts_indices
        )
        return global_input_tokens, None, reversed_global_input_permutation_mapping

    def _combine_preprocess(self, hidden_states, num_local_experts, reversed_global_input_permutation_mapping):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, reversed_global_input_permutation_mapping)

        return hidden_states

    def _combine_postprocess(
        self,
        permutated_local_input_tokens,
        reversed_local_input_permutation_mapping,
        topk_weights,
        hidden_shape_before_permute,
        hidden_shape,
    ):
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=reversed_local_input_permutation_mapping.to(torch.int32),
            probs=topk_weights,
            restore_shape=hidden_shape_before_permute,
        )

        # Reshape the output tensor
        output = output.view(hidden_shape)
        return output
