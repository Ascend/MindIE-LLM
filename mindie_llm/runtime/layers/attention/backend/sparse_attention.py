#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Implement part of this file based on vllm-ascend.
#
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
from typing import Optional, NamedTuple, Tuple, Type, Dict
from dataclasses import dataclass
from functools import wraps
import torch
import torch_npu
import torch.nn.functional as F
import torch.distributed as torch_dist
import numpy as np

from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.runtime.layers.linear.linear_op import maybe_all_gather_and_maybe_unpad
from mindie_llm.runtime.utils.distributed.communication_op import all_gather, gather_tensor, allgather_and_reorder
from mindie_llm.runtime.utils.weight_prefetcher import weight_prefetcher

from mindie_llm.runtime.model_runner.forward_context_exp import ForwardContext, get_forward_context
from mindie_llm.runtime.model_runner.input_buffer import input_buffer
from mindie_llm.runtime.model_runner.forward_context import AttentionMetadata
from .abstract import AttentionBackend, AttentionLayer, SelectAttentionImpl


torch.npu.config.allow_internal_format = True


def run_once(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        setattr(self, f.__name__, lambda *a, **k: None)
        return result
    return wrapper


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), [2, 0, 1, 3]
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def trans_rope_weight(weight, rope_dim):
    weight_1 = weight[..., -rope_dim:: 2, :].contiguous()
    weight_2 = weight[..., -rope_dim + 1:: 2, :].contiguous()
    weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)

    return weight.contiguous()


def prepare_cp_prefill_inputs(cp_size, input_ids, position_ids, input_lengths_cumsum, input_lengths):
    cp_input_dict = {}
    # While enable load balancing, the total sequence of each request is divided into cp*2 chunks,
    # and the input sequence on each cp_rank contains two chunks.
    chunk_lengths = (input_lengths // 2).tolist()

    # While enable load balancing, it is used to take the first half
    # and the second half of each request sequence.
    # Example：[q0_0,q0_3,q1_0,q1_3] --gather--> [q0_0,q1_0] [q0_3,q1_3]
    # [0~4,5~9][0~3,4~7] -> [0~4,10~13][5~9,14~17] -> [0~4,10~13,5~9,14~17]
    cp_load_balance_idx_first, cp_load_balance_idx_last = [], []
    base = 0
    for length in input_lengths.tolist():
        length_range = list(range(base, base + length))
        divider = length // 2
        cp_load_balance_idx_first.extend(length_range[:divider])
        cp_load_balance_idx_last.extend(length_range[divider:])
        base += length
    cp_load_balance_idx = cp_load_balance_idx_first + cp_load_balance_idx_last
    cp_input_dict["cp_load_balance_idx"] = torch.tensor(cp_load_balance_idx, dtype=torch.int32).npu()

    # After the load balancing calculation is completed, it is used to restore O by aggregating the
    # outputs of each request together.
    # Example：[o0_0,o1_0] [o0_3,o1_3] --concat--> [o0_0,o1_0,o0_3,o1_3] --gather--> [o0_0,o0_3,o1_0,o1_3]
    cp_o_recover_idx = []
    base = 0
    chunk_lengths_sum = sum(chunk_lengths)
    for chunk_len in chunk_lengths:
        length_range = list(range(base, base + chunk_len))
        cp_o_recover_idx.extend(length_range)
        cp_o_recover_idx.extend([idx + chunk_lengths_sum for idx in length_range])
        base += chunk_len
    cp_input_dict["cp_o_recover_idx"] = torch.tensor(cp_o_recover_idx, dtype=torch.int32).npu()

    # When load balancing, it is used to restore the KVs after AllGather to the normal order.
    #     [k0_0,k0_3,k1_0,k1_3, k0_1,k0_2,k1_1,k1_2] --gather--> [k0_0,k0_1,k0_2,k0_3, k1_0,k1_1,k1_2,k1_3]
    cp_kv_recover_idx = []
    req_offset = 0
    for req_chunk_len in chunk_lengths:  # Traverse all requests.
        gather_idx_per_chunk = [[] for _ in range(cp_size * 2)]
        for cp_rank_id in range(cp_size):  # Traverse the chunks of the current request on each cp_rank.
            rank_offset = cp_rank_id * input_ids.size(0)
            gather_idx_per_chunk[cp_rank_id] = \
                [rank_offset + req_offset + idx for idx in range(req_chunk_len)]
            gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id] = \
                [rank_offset + req_offset + idx for idx in range(req_chunk_len, req_chunk_len * 2)]
        cp_kv_recover_idx.extend(np.array(gather_idx_per_chunk).flatten().tolist())
        req_offset += req_chunk_len * 2
    cp_input_dict["cp_kv_recover_idx"] = torch.tensor(cp_kv_recover_idx, dtype=torch.int32).npu()
    
    input_lengths_cumsum_cp_prev = torch.zeros((len(input_lengths_cumsum)), dtype=torch.int32).npu()
    input_lengths_cumsum_cp_next = torch.zeros((len(input_lengths_cumsum)), dtype=torch.int32).npu()
    
    offset = 0
    for i, _ in enumerate(input_lengths_cumsum):
        input_lengths_cumsum_cp_prev[i] = offset + (input_lengths_cumsum[i] - offset) // 2
        input_lengths_cumsum_cp_next[i] = input_lengths_cumsum[i]
        offset = input_lengths_cumsum[i]

    actual_seq_lengths_kv_cp_prev = position_ids[input_lengths_cumsum_cp_prev - 1] + 1
    actual_seq_lengths_kv_cp_next = position_ids[input_lengths_cumsum_cp_next - 1] + 1

    k_gather_index_prev = []
    k_gather_index_next = []
    k_offset = 0
    for i, _ in enumerate(input_lengths):
        k_gather_index_prev.extend(list(range(k_offset, actual_seq_lengths_kv_cp_prev[i] + k_offset)))
        k_gather_index_next.extend(list(range(k_offset, actual_seq_lengths_kv_cp_next[i] + k_offset)))
        k_offset += (input_lengths[i] * cp_size)
    cp_input_dict['k_gather_index'] = (torch.tensor(k_gather_index_prev, dtype=torch.int32).npu(), \
                                        torch.tensor(k_gather_index_next, dtype=torch.int32).npu())

    actual_seq_lengths_kv_cp_prev = torch.cumsum(actual_seq_lengths_kv_cp_prev, dim=0, dtype=torch.int32)
    actual_seq_lengths_kv_cp_next = torch.cumsum(actual_seq_lengths_kv_cp_next, dim=0, dtype=torch.int32)
    cp_input_dict['actual_seq_lengths_key'] = (actual_seq_lengths_kv_cp_prev, actual_seq_lengths_kv_cp_next)
    cp_input_dict['actual_seq_lengths_query'] = (input_lengths_cumsum // 2, input_lengths_cumsum // 2)

    return cp_input_dict


def get_speculative_reqs_padding_length(num_tokens, num_actual_tokens):
    reqs_padding_length = num_tokens // num_actual_tokens
    last_req_tokens = num_tokens % num_actual_tokens
    if last_req_tokens > 0:
        reqs_padding_length += 1
    return reqs_padding_length, last_req_tokens


class SfaBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "SPARSE_FUSION_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["SfaBackendImpl"]:
        return SfaBackendImpl
    
    @staticmethod
    def get_builder_cls():
        return SfaMetadataBuilder
    

@dataclass
class SfaMetadata(AttentionMetadata):
    # NOTE: This class is used for building attention metadata in the future.
    actual_seq_lengths_kv: torch.Tensor | None = None
    actual_seq_lengths_query: torch.Tensor | None = None
    seq_lens_list: np.ndarray = None
    cp_input_dict: Dict | None = None
    num_speculative_tokens: int = 0
    num_actual_tokens: int = 0
    mc2_mask: torch.Tensor | None = None

    @staticmethod
    def from_model_input(model_inputs, mask, num_speculative_tokens=0):
        if model_inputs.is_prefill:
            q_lens = getattr(model_inputs, "q_lens", None)
            actual_seq_lengths_query = torch.tensor(model_inputs.context_length, dtype=torch.int32) \
                if q_lens is None else torch.tensor(q_lens, dtype=torch.int32)
        else:
            actual_seq_lengths_query = (
                torch.tensor(model_inputs.q_lens, dtype=torch.int32)
                if num_speculative_tokens > 0
                else torch.tensor([1] * model_inputs.block_tables.shape[0], dtype=torch.int32)
            )
        actual_seq_lengths_query = torch.cumsum(actual_seq_lengths_query, dim=0, dtype=torch.int32)
        actual_seq_lengths_kv = torch.tensor(model_inputs.context_length, dtype=torch.int32)
        
        cp_size = get_parallel_info_manager().attn_cp.group_size
        cp_input_dict = None
        if cp_size > 1 and model_inputs.is_prefill:
            cp_input_dict = prepare_cp_prefill_inputs(
                cp_size,
                model_inputs.input_ids,
                model_inputs.position_ids,
                actual_seq_lengths_query,
                actual_seq_lengths_kv
            )

        return SfaMetadata(
            seq_lens=model_inputs.context_length,
            seq_lens_list=model_inputs.context_length,
            slot_mapping=model_inputs.slots,
            block_tables=model_inputs.block_tables,
            attn_mask=mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_query=actual_seq_lengths_query,
            num_speculative_tokens=num_speculative_tokens,
            cp_input_dict=cp_input_dict,
            num_actual_tokens=len(model_inputs.input_ids)
        )

    @staticmethod
    def register_buffer(max_num_token, device):
        input_buffer.register("seq_lens", torch.zeros(max_num_token, dtype=torch.int32, device=device))
        input_buffer.register("block_tables", torch.zeros((max_num_token, 64), dtype=torch.int32, device=device))
        input_buffer.register("slot_mapping", -torch.ones(max_num_token, dtype=torch.int32, device=device))

        input_buffer.register("actual_seq_lengths_kv", torch.zeros(max_num_token, dtype=torch.int32, device=device))
        input_buffer.register("actual_seq_lengths_query", torch.zeros(max_num_token, dtype=torch.int32, device=device))
        input_buffer.register("mc2_mask", torch.zeros(max_num_token, dtype=torch.bool, device=device))

    def to_device(self, device):
        self.block_tables = torch.tensor(self.block_tables, dtype=torch.int32).to(device)
        self.slot_mapping = torch.tensor(self.slot_mapping, dtype=torch.int32).to(device)
        self.seq_lens = torch.tensor(self.seq_lens, dtype=torch.int32).to(device)

        self.actual_seq_lengths_kv = torch.tensor(self.actual_seq_lengths_kv, dtype=torch.int32).to(device)
        self.actual_seq_lengths_query = torch.tensor(self.actual_seq_lengths_query, dtype=torch.int32).to(device)

        mapping = get_parallel_info_manager()
        tp_size = mapping.attn_tp.group_size
        tp_rank = mapping.attn_tp.rank
        num_padded_tokens = self.num_tokens + (tp_size - self.num_tokens % tp_size) % tp_size
        unit_size = num_padded_tokens // tp_size
        all_mask = [1] * self.num_actual_tokens + [0] * (num_padded_tokens - self.num_actual_tokens)
        self.mc2_mask = torch.tensor(all_mask[unit_size * tp_rank:unit_size * (tp_rank + 1)], dtype=torch.bool).npu()

    def copy(self, num_actual_tokens, num_tokens):
        # NOTE: only D2D operation is allowed, should be refactored later
        max_len = self.seq_lens_list.max()
        max_seq_pages = (max_len + 128 - 1) // 128
        num_reqs = num_actual_tokens // (self.num_speculative_tokens + 1)

        input_buffer_slot_mapping = input_buffer.get("slot_mapping")
        input_buffer_slot_mapping[:num_actual_tokens].copy_(self.slot_mapping[:num_actual_tokens])
        self.slot_mapping = input_buffer_slot_mapping[:num_tokens]

        actual_len, last_req_tokens = get_speculative_reqs_padding_length(num_tokens=num_tokens,
                                                                    num_actual_tokens=self.num_speculative_tokens + 1)
        reqs_padding_length = actual_len - self.actual_seq_lengths_kv.shape[0]
        
        input_buffer_seq_lens = input_buffer.get("seq_lens")
        input_buffer_seq_lens.fill_(0)
        input_buffer_seq_lens[:self.seq_lens.shape[0]].copy_(self.seq_lens)
        self.seq_lens = input_buffer_seq_lens[:actual_len]

        if actual_len > num_actual_tokens:
            self.seq_lens_list = self.seq_lens_list.tolist() + [0] * (actual_len - num_actual_tokens)

        input_buffer_actual_seq_lengths_kv = input_buffer.get("actual_seq_lengths_kv")
        input_buffer_actual_seq_lengths_kv.fill_(self.num_speculative_tokens + 1)
        input_buffer_actual_seq_lengths_kv[:self.actual_seq_lengths_kv.shape[-1]].copy_(
            self.actual_seq_lengths_kv[:self.actual_seq_lengths_kv.shape[-1]])
        self.actual_seq_lengths_kv = input_buffer_actual_seq_lengths_kv[:actual_len]
        if last_req_tokens > 0:
            self.actual_seq_lengths_kv[-1] = last_req_tokens

        input_buffer_actual_seq_lengths_query = input_buffer.get("actual_seq_lengths_query")
        input_buffer_actual_seq_lengths_query[:self.actual_seq_lengths_query.shape[-1]].copy_(
            self.actual_seq_lengths_query[:self.actual_seq_lengths_query.shape[-1]])
        self.actual_seq_lengths_query = input_buffer_actual_seq_lengths_query[:actual_len]

        input_buffer_block_tables = input_buffer.get("block_tables")
        input_buffer_block_tables.fill_(0)
        input_buffer_block_tables[:num_reqs, :self.block_tables.shape[-1]].copy_(self.block_tables)
        self.block_tables = input_buffer_block_tables[:actual_len, :]

        unit_size = len(self.mc2_mask)
        input_buffer_mc2_mask = input_buffer.get("mc2_mask")
        input_buffer_mc2_mask[:unit_size].copy_(self.mc2_mask)
        self.mc2_mask = input_buffer_mc2_mask[:unit_size]

    def prepare_dummy_input(self, num_tokens):
        num_actual_tokens = self.num_speculative_tokens + 1
        reqs_padding_length, _ = get_speculative_reqs_padding_length(num_tokens=num_tokens,
                                                                    num_actual_tokens=self.num_speculative_tokens + 1)
        self.seq_lens = self.seq_lens[:reqs_padding_length]
        self.block_tables = self.block_tables[:reqs_padding_length, :]
        self.actual_seq_lengths_kv = input_buffer.get("actual_seq_lengths_kv")[:reqs_padding_length]
        actual_seq_lengths_query = torch.arange(
            num_actual_tokens,
            reqs_padding_length * num_actual_tokens + 1,
            num_actual_tokens,
            dtype=torch.int32,
            device=self.seq_lens.device
        )
        input_buffer.get("actual_seq_lengths_query")[:reqs_padding_length].copy_(actual_seq_lengths_query)
        self.actual_seq_lengths_query = input_buffer.get("actual_seq_lengths_query")[:reqs_padding_length]


class SfaMetadataBuilder:
    # NOTE: This class is used for building attention metadata in the future.
    @staticmethod
    def build(
        common_attn_metadata: AttentionMetadata,
        input_metadata: dict,
    ):
        cp_size = get_parallel_info_manager().attn_cp.group_size
        cp_input_dict = None
        if cp_size > 1 and input_metadata["is_prefill"]:
            input_ids = input_metadata["input_ids"]
            position_ids = input_metadata["position_ids"]
            actual_seq_lengths_query = input_metadata["actual_seq_lengths_query"]
            actual_seq_lengths_kv = input_metadata["actual_seq_lengths_kv"]
            cp_input_dict = prepare_cp_prefill_inputs(cp_size, input_ids, position_ids, 
                                                    actual_seq_lengths_query, actual_seq_lengths_kv)

        attn_metadata = SfaMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_list=common_attn_metadata.seq_lens_list,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_tables=common_attn_metadata.block_tables,
            attn_mask=common_attn_metadata.attn_mask,
            actual_seq_lengths_kv=input_metadata["actual_seq_lengths_kv"],
            actual_seq_lengths_query=input_metadata["actual_seq_lengths_query"],
            cp_input_dict=cp_input_dict
        )
        return attn_metadata

    @staticmethod
    def get_metadata_cls():
        return SfaMetadata


class PrefillSFAPreprocessResult(NamedTuple):
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None
    key_states: Optional[torch.Tensor] = None


class DecodeSFAPreprocessResult(NamedTuple):
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None


@dataclass
class MlapoWeightPack:
    wd_qkv: Optional[torch.Tensor] = None
    deq_scale_qkv: Optional[torch.Tensor] = None
    quant_bias_qkv: Optional[torch.Tensor] = None
    wu_q: Optional[torch.Tensor] = None
    qb_deq_scl: Optional[torch.Tensor] = None
    qb_qt_bias: Optional[torch.Tensor] = None
    gamma0: Optional[torch.Tensor] = None
    beta0: Optional[torch.Tensor] = None
    gamma1: Optional[torch.Tensor] = None
    beta1: Optional[torch.Tensor] = None
    gamma2: Optional[torch.Tensor] = None
    quant_scale0: Optional[torch.Tensor] = None
    quant_offset0: Optional[torch.Tensor] = None
    quant_scale1: Optional[torch.Tensor] = None
    quant_offset1: Optional[torch.Tensor] = None
    ctkv_scale: Optional[torch.Tensor] = None
    q_nope_scale: Optional[torch.Tensor] = None


class SfaBackendImpl(SelectAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        prefix: str,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.q_size = num_heads * head_size

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("In MHA/GQA/MQA, num_kv_heads should be divisible by num_heads.")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.num_key_value_heads_per_rank = self.num_kv_heads
        self.kv_cache = None
        self.pe_cache = None
        self.index_cache = None
        self.block_size = 128   # NOTE: currently hard-coding.
        self.prefix = prefix
        self.parallel_info = get_parallel_info_manager()
        self.cp_size = self.parallel_info.attn_cp.group_size
        self.check_parallel_info()

        # MLA Args
        self.q_lora_rank = kwargs['q_lora_rank']
        self.kv_lora_rank = kwargs['kv_lora_rank']
        self.qk_nope_head_dim = kwargs['qk_nope_head_dim']
        self.qk_rope_head_dim = kwargs['qk_rope_head_dim']
        self.qk_head_dim = kwargs['qk_head_dim']
        self.v_head_dim = kwargs['v_head_dim']
        self.num_heads_per_rank = kwargs['num_heads_per_rank']
        self.softmax_scale = kwargs['softmax_scale']
        self.q_proj = kwargs['q_proj']
        self.q_a_proj, self.q_a_layernorm, self.q_b_proj = self.q_proj
        self.kv_a_proj_with_mqa = kwargs['kv_a_proj_with_mqa']
        self.kv_a_layernorm = kwargs['kv_a_layernorm']
        self.kv_b_proj = kwargs['kv_b_proj']
        self.o_proj = kwargs['o_proj']
        self.indexer = kwargs['indexer']

        self.enable_mlapo = kwargs['enable_mlapo']
        if self.enable_mlapo:
            import mie_ops
        self.input_layernorm = kwargs.get('input_layernorm', None)
        self.mlapo_weight_pack = MlapoWeightPack()

        self.kv_b_proj_w_k = torch.empty(
            self.num_heads_per_rank,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            dtype=self.kv_b_proj.weight.dtype
        )
        self.kv_b_proj_w_v = torch.empty(
            self.num_heads_per_rank,
            self.kv_lora_rank,
            self.v_head_dim,
            dtype=self.kv_b_proj.weight.dtype
        )

    @run_once
    def process_weights_after_loading(self):
        kv_b_proj_weight = self.kv_b_proj.weight.data.T
        expected_shape = (
                self.kv_lora_rank,
                self.num_heads_per_rank * (self.qk_nope_head_dim + self.v_head_dim)
            )
        if kv_b_proj_weight.shape != expected_shape:
            raise RuntimeError(f"{kv_b_proj_weight.shape} != {expected_shape}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads_per_rank,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        kv_b_proj_w_k, kv_b_proj_w_v = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.kv_b_proj_w_k.copy_(kv_b_proj_w_k.permute(1, 2, 0).contiguous())
        self.kv_b_proj_w_v.copy_(kv_b_proj_w_v.transpose(0, 1).contiguous())
        self._process_weights_for_fused_mlapo()

    @run_once
    def _process_weights_for_fused_mlapo(self):
        if not self.enable_mlapo:
            return

        kv_a_proj_wt = self.kv_a_proj_with_mqa.weight.clone()
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, self.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, self.q_a_proj.weight.clone()), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.mlapo_weight_pack.wd_qkv = torch_npu.npu_format_cast(wd_qkv, 29)

        kv_a_proj_deq_scl = self.kv_a_proj_with_mqa.deq_scale.clone()
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, self.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.mlapo_weight_pack.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl,
                                                          self.q_a_proj.deq_scale.clone()), dim=-1).contiguous()

        kv_a_proj_qt_bias = self.kv_a_proj_with_mqa.quant_bias.clone()
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, self.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.mlapo_weight_pack.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias,
                                                           self.q_a_proj.quant_bias.clone()), dim=-1).contiguous()

        wu_q = self.q_b_proj.weight.clone()
        wu_q = wu_q.t().reshape(self.num_heads_per_rank, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        wu_q = trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(self.num_heads_per_rank * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1).contiguous()
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.mlapo_weight_pack.wu_q = torch_npu.npu_format_cast(wu_q, 29)

        qb_deq_scl = self.q_b_proj.deq_scale.clone()
        qb_deq_scl = qb_deq_scl.reshape(self.num_heads_per_rank, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, self.qk_rope_head_dim)
        self.mlapo_weight_pack.qb_deq_scl = qb_deq_scl.reshape(self.num_heads_per_rank * \
                                                               (self.qk_nope_head_dim + self.qk_rope_head_dim))

        qb_qt_bias = self.q_b_proj.quant_bias.data.clone()
        qb_qt_bias = qb_qt_bias.reshape(self.num_heads_per_rank, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, self.qk_rope_head_dim)
        self.mlapo_weight_pack.qb_qt_bias = qb_qt_bias.reshape(self.num_heads_per_rank * \
                                                               (self.qk_nope_head_dim + self.qk_rope_head_dim))

        self.mlapo_weight_pack.gamma0 = self.input_layernorm.weight
        self.mlapo_weight_pack.beta0 = torch.zeros_like(self.mlapo_weight_pack.gamma0)
        self.mlapo_weight_pack.gamma1 = self.q_a_layernorm.weight
        self.mlapo_weight_pack.beta1 = torch.zeros_like(self.mlapo_weight_pack.gamma1)
        self.mlapo_weight_pack.gamma2 = self.kv_a_layernorm.weight
        self.mlapo_weight_pack.quant_scale0 = 1 / self.q_a_proj.input_scale
        self.mlapo_weight_pack.quant_offset0 = self.q_a_proj.input_offset.to(torch.int8)
        self.mlapo_weight_pack.quant_scale1 = 1 / self.q_b_proj.input_scale
        self.mlapo_weight_pack.quant_offset1 = self.q_b_proj.input_offset.to(torch.int8)
        self.mlapo_weight_pack.ctkv_scale = torch.tensor([1],
                                                         dtype=self.kv_b_proj.weight.dtype,
                                                         device=self.kv_b_proj.weight.device)
        self.mlapo_weight_pack.q_nope_scale = torch.tensor([1],
                                                           dtype=self.kv_b_proj.weight.dtype,
                                                           device=self.kv_b_proj.weight.device)

    def check_parallel_info(self):
        if self.parallel_info.get("attn_inner_sp").is_enabled():
            raise RuntimeError("Not support sp!")
        if self.parallel_info.get("attn_cp").is_enabled() and self.parallel_info.get("attn_dp").is_enabled():
            raise RuntimeError(f"Not support cp and dp: cp size is {self.parallel_info.attn_cp.group_size};"
                                f"dp size is {self.parallel_info.attn_dp.group_size}")

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        cp_input_dict,
    ):
        layout_query = "TND"
        layout_key = "TND"
        actual_seq_lengths_q = cp_input_dict['actual_seq_lengths_query']
        actual_seq_lengths_kv = cp_input_dict['actual_seq_lengths_key']
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        actual_seq_lengths_q_prev = actual_seq_lengths_q_prev.to(device=q.device, dtype=torch.int32)
        actual_seq_lengths_kv_prev = actual_seq_lengths_kv_prev.to(device=q.device, dtype=torch.int32)
        k_gather_index_prev, k_gather_index_next = cp_input_dict['k_gather_index']
        k_gather_prev = gather_tensor(past_key_states, k_gather_index_prev)
        k_gather_next = gather_tensor(past_key_states, k_gather_index_next)

        topk_indices_prev, _ = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=k_gather_prev,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev,
            actual_seq_lengths_key=actual_seq_lengths_kv_prev,
            block_table=None,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=2048,
            sparse_mode=3,
        )
        topk_indices_next, _ = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=k_gather_next,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=None,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=2048,
            sparse_mode=3,
        )
        index_out = torch.cat([topk_indices_prev, topk_indices_next], dim=0)
        index_out = gather_tensor(index_out, cp_input_dict["cp_o_recover_idx"])
        return index_out


    def indexer_select(
            self,
            hidden_state: torch.Tensor,
            q_c: torch.Tensor,
            forward_context: ForwardContext, 
            attn_metadata: SfaMetadata,
            cos, sin
        ):
        q = self.indexer.wq_b(q_c) 
        q = q.view(-1, self.indexer.n_heads, self.indexer.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.indexer.head_dim - self.qk_rope_head_dim], dim=-1)

        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.indexer.wk(hidden_state)
        k = self.indexer.k_norm(k_proj).unsqueeze(1)
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.indexer.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2)
        k_pe = torch_npu.npu_interleave_rope(k_pe,
                                             cos.view(-1, 1, 1, self.qk_rope_head_dim),
                                             sin.view(-1, 1, 1, self.qk_rope_head_dim))
        k_pe = k_pe.squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        # cp
        cp_input_dict = attn_metadata.cp_input_dict
        if forward_context.is_prefill and self.cp_size > 1:
            k = allgather_and_reorder(k, self.parallel_info.attn_cp.process_group, cp_input_dict["cp_kv_recover_idx"])

        torch_npu.npu_scatter_nd_update_(self.index_cache.view(-1, k.shape[-1]),
                                         attn_metadata.slot_mapping.view(-1, 1), 
                                         k.view(-1, k.shape[-1]))

        weights = self.indexer.weights_proj(hidden_state)
        actual_seq_lengths_key = attn_metadata.actual_seq_lengths_kv
        if forward_context.is_prefill and self.cp_size > 1:
            q = gather_tensor(q, cp_input_dict["cp_load_balance_idx"])
            weights = gather_tensor(weights, cp_input_dict["cp_load_balance_idx"])
            
            topk_indices = self.do_npu_cp_balance_indexer(
                q=q,
                past_key_states=k,
                indexer_weights=weights,
                cp_input_dict=cp_input_dict,
            )
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q,
                key=self.index_cache,
                weights=weights,
                actual_seq_lengths_query=attn_metadata.actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_tables,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3
            )

        return topk_indices

    def sfa_prefill_preprocess(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        kv_no_split: torch.Tensor,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata,
        cos,
        sin
    ):
        decode_q = self.q_b_proj(q_c)
        bsz, _ = decode_q.shape
        decode_q = decode_q.view(bsz, self.num_heads_per_rank, 1, self.qk_head_dim)
        decode_q_nope, decode_q_pe = torch.split(
            decode_q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        decode_q_nope = decode_q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim).transpose(0, 1)
        decode_q_nope = (
            torch.matmul(decode_q_nope, self.kv_b_proj_w_k)
            .transpose(1, 0)
            .view(bsz, 1, self.num_heads_per_rank, self.kv_lora_rank)
        )

        kv_no_split = kv_no_split.unsqueeze(1).unsqueeze(1)
        cos_cache, sin_cache = cos, sin
        is_output_kv = False
        cp_kv_recover_idx_key = "cp_kv_recover_idx"
        if forward_context.is_prefill and self.cp_size > 1:
            cp_input_dict = attn_metadata.cp_input_dict
            kv_no_split = allgather_and_reorder(kv_no_split, self.parallel_info.attn_cp.process_group, \
                cp_input_dict[cp_kv_recover_idx_key])
            cos_cache = allgather_and_reorder(cos, self.parallel_info.attn_cp.process_group, \
                cp_input_dict[cp_kv_recover_idx_key])
            sin_cache = allgather_and_reorder(sin, self.parallel_info.attn_cp.process_group, \
                cp_input_dict[cp_kv_recover_idx_key])
            is_output_kv = True

        decode_k_rope, decode_k_nope, k_rope_a, k_nope_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos_cache,
            sin_cache,
            attn_metadata.slot_mapping.to(torch.int64),
            self.pe_cache,
            self.kv_cache,
            c_kv_scale=None,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode='PA',
            is_output_kv=is_output_kv)
        
        decode_q_pe = torch_npu.npu_interleave_rope(decode_q_pe,
                                                    cos,
                                                    sin)

        decode_q_nope = decode_q_nope.view(bsz, self.num_heads_per_rank, self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads_per_rank, -1)
        topk_indices = self.indexer_select(hidden_states, q_c, forward_context, attn_metadata, cos, sin)
        key_states = None
        if forward_context.is_prefill and self.cp_size > 1:
            key_states = (k_nope_a, k_rope_a)
        decode_preprocess_res = PrefillSFAPreprocessResult(
            q_nope=decode_q_nope,
            q_pe=decode_q_pe,
            k_nope=decode_k_nope,
            k_pe=decode_k_rope,
            value=decode_k_nope,
            topk_indices=topk_indices,
            key_states=key_states
        )
        return decode_preprocess_res
        
    def sfa_decode_preprocess(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        kv_no_split: torch.Tensor,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata,
        cos,
        sin
    ):
        decode_q = self.q_b_proj(q_c)
        bsz, _ = decode_q.shape
        decode_q = decode_q.view(bsz, self.num_heads_per_rank, 1, self.qk_head_dim)
        decode_q_nope, decode_q_pe = torch.split(
            decode_q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        decode_q_nope = decode_q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim).transpose(0, 1)
        decode_q_nope = (
            torch.matmul(decode_q_nope, self.kv_b_proj_w_k)
            .transpose(1, 0)
            .view(bsz, 1, self.num_heads_per_rank, self.kv_lora_rank)
        )

        kv_no_split = kv_no_split.unsqueeze(1).unsqueeze(1)
        decode_k_rope, decode_k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            attn_metadata.slot_mapping.to(torch.int64),
            self.pe_cache,
            self.kv_cache,
            c_kv_scale=None,
            epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode='PA')
        
        decode_q_pe = torch_npu.npu_interleave_rope(decode_q_pe,
                                                    cos,
                                                    sin)

        decode_q_nope = decode_q_nope.view(bsz, self.num_heads_per_rank, self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads_per_rank, -1)
        topk_indices = self.indexer_select(hidden_states, q_c, forward_context, attn_metadata, cos, sin)
        decode_preprocess_res = DecodeSFAPreprocessResult(
            q_nope=decode_q_nope,
            q_pe=decode_q_pe,
            k_nope=decode_k_nope,
            k_pe=decode_k_rope,
            topk_indices=topk_indices
        )
        return decode_preprocess_res
    
    def sfa_decode_mlapo_preprocess(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata,
        cos, sin
    ):
        bsz, _ = hidden_states.shape

        decode_q_nope, cache1, decode_q_pe, cache2 = torch.ops.mie_ops.npu_mla_process(
            input=hidden_states,
            gamma0=self.mlapo_weight_pack.gamma0,
            beta0=self.mlapo_weight_pack.beta0,
            wdqkv=self.mlapo_weight_pack.wd_qkv,
            descale0=self.mlapo_weight_pack.deq_scale_qkv,
            gamma1=self.mlapo_weight_pack.gamma1,
            beta1=self.mlapo_weight_pack.beta1,
            wuq=self.mlapo_weight_pack.wu_q,
            descale1=self.mlapo_weight_pack.qb_deq_scl,
            gamma2=self.mlapo_weight_pack.gamma2,
            cos=cos,
            sin=sin,
            wuk=self.kv_b_proj_w_k,
            kv_cache=self.kv_cache,
            kv_cache_rope=self.pe_cache,
            slotmapping=attn_metadata.slot_mapping.flatten().to(torch.int32),
            quant_scale0=self.mlapo_weight_pack.quant_scale0,
            quant_offset0=self.mlapo_weight_pack.quant_offset0,
            bias0=self.mlapo_weight_pack.quant_bias_qkv,
            quant_scale1=self.mlapo_weight_pack.quant_scale1,
            quant_offset1=self.mlapo_weight_pack.quant_offset1,
            bias1=self.mlapo_weight_pack.qb_qt_bias,
            ctkv_scale=self.mlapo_weight_pack.ctkv_scale,
            q_nope_scale=self.mlapo_weight_pack.q_nope_scale,
            cache_mode_opt="krope_ctkv",
            quant_mode_opt="per_tensor_quant_asymm",
        )
        decode_k_nope = self.kv_cache
        decode_k_pe = self.pe_cache
        decode_q_nope = decode_q_nope.view(bsz, self.num_heads_per_rank, self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads_per_rank, -1)

        topk_indices = self.indexer_select(hidden_states, q_c, forward_context, attn_metadata, cos, sin)
        decode_preprocess_res = DecodeSFAPreprocessResult(
            q_nope=decode_q_nope,
            q_pe=decode_q_pe,
            k_nope=decode_k_nope,
            k_pe=decode_k_pe,
            topk_indices=topk_indices
        )
        
        return decode_preprocess_res

    def sfa_preprocess(
        self, 
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata,
        cos, sin
    ):
        if self.kv_cache is None or id(self.kv_cache) != id(kv_cache[0]):
            self.kv_cache, self.pe_cache, self.index_cache = kv_cache[0], kv_cache[1], kv_cache[2]

        decode_preprocess_res = None
        prefill_preprocess_res = None

        ckq = self.q_a_proj(hidden_states)
        q_c = self.q_a_layernorm(ckq)
        if not forward_context.is_prefill and self.enable_mlapo:
            decode_preprocess_res = self.sfa_decode_mlapo_preprocess(
                hidden_states,
                q_c,
                forward_context,
                attn_metadata, cos, sin
            )
            return decode_preprocess_res, prefill_preprocess_res

        kv_no_split = self.kv_a_proj_with_mqa(hidden_states)
        if forward_context.is_prefill:
            prefill_preprocess_res = self.sfa_prefill_preprocess(
                hidden_states,
                q_c,
                kv_no_split,
                forward_context,
                attn_metadata, cos, sin
            )
        else:
            decode_preprocess_res = self.sfa_decode_preprocess(
                hidden_states,
                q_c,
                kv_no_split,
                forward_context,
                attn_metadata, cos, sin
            )

        return decode_preprocess_res, prefill_preprocess_res

    def do_cp_balance_attn(
        self,
        prefill_preprocess_res: PrefillSFAPreprocessResult,
        attn_metadata: SfaMetadata
    ):
        k_nope, k_pe = prefill_preprocess_res.key_states
        q_nope = prefill_preprocess_res.q_nope
        q_pe = prefill_preprocess_res.q_pe
        topk_indices = prefill_preprocess_res.topk_indices
        actual_seq_qlen = attn_metadata.actual_seq_lengths_query
        cp_input_dict = attn_metadata.cp_input_dict
        actual_seq_lengths_kv = cp_input_dict['actual_seq_lengths_key']
        layout_query = "TND"
        layout_kv = "TND"
        cp_load_balance_idx_key = "cp_load_balance_idx"
        k_nope = k_nope.squeeze(2)
        k_pe = k_pe.squeeze(2)
        seq_len = q_nope.shape[0]
        split_len = seq_len // 2
        q_nope = gather_tensor(q_nope, cp_input_dict[cp_load_balance_idx_key])
        q_pe = gather_tensor(q_pe, cp_input_dict[cp_load_balance_idx_key])
        topk_indices = gather_tensor(topk_indices, cp_input_dict[cp_load_balance_idx_key])

        q_nope_prev, q_nope_next = torch.split(q_nope, split_len, dim=0)
        q_rope_prev, q_rope_next = torch.split(q_pe, split_len, dim=0)
        topk_indices_prev, topk_indices_next = torch.split(topk_indices, split_len, dim=0)

        actual_seq_qlen_prev, actual_seq_qlen_next = cp_input_dict["actual_seq_lengths_query"]
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv
        k_gather_index_prev, k_gather_index_next = cp_input_dict['k_gather_index']
        k_nope_prev = gather_tensor(k_nope, k_gather_index_prev)
        k_nope_next = gather_tensor(k_nope, k_gather_index_next)
        k_pe_prev = gather_tensor(k_pe, k_gather_index_prev)
        k_pe_next = gather_tensor(k_pe, k_gather_index_next)

        attn_out_prev, _, __ = torch_npu.npu_sparse_flash_attention(
            query=q_nope_prev,
            key=k_nope_prev,
            value=k_nope_prev,
            query_rope=q_rope_prev,
            key_rope=k_pe_prev,
            sparse_indices=topk_indices_prev,
            scale_value=self.softmax_scale,
            sparse_block_size=1,
            block_table=None,
            actual_seq_lengths_query=actual_seq_qlen_prev.to(device=q_nope.device, dtype=torch.int32),
            actual_seq_lengths_kv=actual_seq_lengths_kv_prev.to(device=q_nope.device, dtype=torch.int32),
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=3,
            attention_mode=2,
        )
        attn_out_next, _, __ = torch_npu.npu_sparse_flash_attention(
            query=q_nope_next,
            key=k_nope_next,
            value=k_nope_next,
            query_rope=q_rope_next,
            key_rope=k_pe_next,
            sparse_indices=topk_indices_next,
            scale_value=self.softmax_scale,
            sparse_block_size=1,
            block_table=None,
            actual_seq_lengths_query=actual_seq_qlen_next.to(device=q_nope.device, dtype=torch.int32),
            actual_seq_lengths_kv=actual_seq_lengths_kv_next.to(device=q_nope.device, dtype=torch.int32),
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=3,
            attention_mode=2,
        )
        attn_out = torch.cat([attn_out_prev, attn_out_next], dim=0)
        attn_out = gather_tensor(attn_out, cp_input_dict["cp_o_recover_idx"])
        return attn_out

    def apply_prefill_sfa(
        self,
        prefill_preprocess_res: PrefillSFAPreprocessResult,
        attn_metadata: SfaMetadata
    ):
        if self.cp_size > 1:
            
            output = self.do_cp_balance_attn(
                prefill_preprocess_res,
                attn_metadata
            )
        else:
            output, _, __ = torch_npu.npu_sparse_flash_attention(
                query=prefill_preprocess_res.q_nope,
                key=prefill_preprocess_res.k_nope,
                value=prefill_preprocess_res.k_nope,
                query_rope=prefill_preprocess_res.q_pe,
                key_rope=prefill_preprocess_res.k_pe,
                sparse_indices=prefill_preprocess_res.topk_indices,
                scale_value=self.softmax_scale,
                sparse_block_size=1,
                block_table=attn_metadata.block_tables,
                actual_seq_lengths_query=attn_metadata.actual_seq_lengths_query,
                actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_kv,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
            )
        return self.sfa_postprocess(output)
    
    def apply_decode_sfa(
        self,
        prefill_preprocess_res: DecodeSFAPreprocessResult,
        attn_metadata: SfaMetadata
    ):
        output, _, _ = torch_npu.npu_sparse_flash_attention(
            query=prefill_preprocess_res.q_nope,
            key=prefill_preprocess_res.k_nope,
            value=prefill_preprocess_res.k_nope,
            query_rope=prefill_preprocess_res.q_pe,
            key_rope=prefill_preprocess_res.k_pe,
            sparse_indices=prefill_preprocess_res.topk_indices,
            scale_value=self.softmax_scale,
            sparse_block_size=1,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_query=attn_metadata.actual_seq_lengths_query,
            actual_seq_lengths_kv=attn_metadata.seq_lens,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
        output = output.squeeze(1)
        return self.sfa_postprocess(output)
    
    def sfa_postprocess(self, slc_fa_fusion):
        if self.kv_b_proj_w_v.shape[0] * self.kv_b_proj_w_v.shape[1] < 65536:
            slc_fa_fusion = slc_fa_fusion.view(-1, self.num_heads_per_rank, self.kv_lora_rank)
            attn_output = torch_npu.npu_transpose_batchmatmul(
                slc_fa_fusion,
                self.kv_b_proj_w_v,
                perm_x1=[1, 0, 2],
                perm_x2=[0, 1, 2],
                perm_y=[1, 0, 2]
                )
            attn_output = attn_output.reshape(-1, self.num_heads_per_rank * self.v_head_dim)
        else:
            slc_fa_fusion = slc_fa_fusion.view(-1, self.num_heads_per_rank, self.kv_lora_rank).transpose(0, 1)
            attn_output = torch.bmm(slc_fa_fusion, self.kv_b_proj_w_v)
            attn_output = attn_output.transpose(0, 1).reshape(-1, self.num_heads_per_rank * self.v_head_dim)
        return attn_output

    def mla_epilog(self, o_proj_input):
        attn_output = self.o_proj(o_proj_input)
        return attn_output

    def forward_impl(
        self,
        prefill_preprocess_res,
        decode_preprocess_res,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata
    ):
        if forward_context.is_prefill:
            return self.apply_prefill_sfa(prefill_preprocess_res, attn_metadata)
        else:
            return self.apply_decode_sfa(decode_preprocess_res, attn_metadata)

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor] = None,
        cos=None,
        sin=None
    ) -> torch.Tensor:

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata_dict
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer.prefix]

        if forward_context.batch_descriptor.is_flash_comm_enabled:
            hidden_states = maybe_all_gather_and_maybe_unpad(
                hidden_states,
                get_parallel_info_manager().attn_tp
            )
        decode_preprocess_res, prefill_preprocess_res = self.sfa_preprocess(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            forward_context=forward_context,
            attn_metadata=attn_metadata,
            cos=cos, sin=sin)
        if not forward_context.is_prefill and weight_prefetcher.is_prefetch_enabled():
            weight_prefetcher.prefetch_weight_preprocess(self.o_proj.weight, hidden_states)

        o_proj_input = self.forward_impl(
                prefill_preprocess_res=prefill_preprocess_res,
                decode_preprocess_res=decode_preprocess_res,
                forward_context=forward_context,
                attn_metadata=attn_metadata,
            )
        output = self.mla_epilog(o_proj_input)
        return output
