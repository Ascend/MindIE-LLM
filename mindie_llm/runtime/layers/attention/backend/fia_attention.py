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
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from typing import Tuple, Type
from dataclasses import dataclass
import numpy as np
import torch
import torch_npu

from mindie_llm.runtime.layers.quantization.ms_model_slim.c8 import AttnQuantMethodBase
from mindie_llm.runtime.model_runner.forward_context import AttentionMetadata
from mindie_llm.runtime.model_runner.forward_context_exp import ForwardContext, get_forward_context
from mindie_llm.runtime.model_runner.input_buffer import input_buffer
from .abstract import AttentionBackend, AttentionImpl, AttentionLayer


class FiaAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ATTENTION_V1"

    @staticmethod
    def get_impl_cls() -> Type["FiaAttentionBackendImpl"]:
        return FiaAttentionBackendImpl
    
    @staticmethod
    def get_builder_cls():
        return FiaAttentionMetadataBuilder
    

@dataclass
class FiaAttentionMetadata(AttentionMetadata):
    # NOTE: This class is used for building attention metadata in the future.
    # The following attributes are only filled by `from_model_input`.

    @staticmethod
    def from_model_input(model_inputs, mask, num_speculative_tokens=0):
        return FiaAttentionMetadata(
            seq_lens=model_inputs.context_length,
            slot_mapping=model_inputs.slots,
            block_tables=model_inputs.block_tables,
            attn_mask=mask,
            max_seq_len=model_inputs.max_seq_len,
        )

    @staticmethod
    def register_buffer(max_num_token, device):
        input_buffer.register("seq_lens", torch.zeros(max_num_token, dtype=torch.int32, device=device))
        input_buffer.register("block_tables", torch.zeros((max_num_token, 64), dtype=torch.int32, device=device))
        input_buffer.register("slot_mapping", -torch.ones(max_num_token, dtype=torch.int32, device=device))

    def to_device(self, device):
        self.block_tables = torch.tensor(self.block_tables, dtype=torch.int32).to(device)
        self.slot_mapping = torch.tensor(self.slot_mapping, dtype=torch.int32).to(device)
        self.seq_lens_list = self.seq_lens
        self.seq_lens = torch.tensor(self.seq_lens).to(device)

    def copy(self, num_actual_tokens, num_tokens):
        # D2D operation

        input_buffer_seq_lens = input_buffer.get("seq_lens")
        input_buffer_seq_lens[:num_actual_tokens].copy_(self.seq_lens[:num_actual_tokens])
        input_buffer_seq_lens[num_actual_tokens:num_tokens].fill_(0)
        self.seq_lens = input_buffer_seq_lens[:num_tokens]
        
        self.seq_lens_list = self.seq_lens_list.tolist() + [0] * (num_tokens - num_actual_tokens)

        max_seq_pages = (self.max_seq_len + 128 - 1) // 128

        input_buffer_block_tables = input_buffer.get("block_tables")
        input_buffer_block_tables[:num_actual_tokens, :self.block_tables.shape[-1]].copy_(self.block_tables)
        input_buffer_block_tables[:num_tokens, max_seq_pages:].fill_(0)
        input_buffer_block_tables[num_tokens:, :].fill_(0)
        self.block_tables = input_buffer_block_tables

        input_buffer_slot_mapping = input_buffer.get("slot_mapping")
        input_buffer_slot_mapping[:num_actual_tokens].copy_(self.slot_mapping[:num_actual_tokens])
        self.slot_mapping = input_buffer_slot_mapping[:num_tokens]


class FiaAttentionMetadataBuilder: 
    # NOTE: This class is used for building attention metadata in the future.

    @staticmethod
    def build(
        common_attn_metadata: AttentionMetadata,
        input_metadata: dict,
    ):
        attn_metadata = FiaAttentionMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_list=common_attn_metadata.seq_lens_list,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_tables=common_attn_metadata.block_tables,
            attn_mask=common_attn_metadata.attn_mask,
        )
        return attn_metadata

    @staticmethod
    def get_metadata_cls():
        return FiaAttentionMetadata


class FiaAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        quant_method: AttnQuantMethodBase,
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
        self.key_cache = None
        self.value_cache = None
        self.block_size = 128   # NOTE: currently hard-coding.
        self.quant_method = quant_method

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        if self.key_cache is None or id(self.key_cache) != id(kv_cache[0]):
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        
        if self.quant_method:
            shape = key.shape
            key_int8, value_int8 = self.quant_method.apply(key.view(-1, self.head_size * self.num_kv_heads),
                value.view(-1, self.head_size * self.num_kv_heads))
            key_int8 = key_int8.view(shape)
            value_int8 = value_int8.view(shape)
        torch_npu._npu_reshape_and_cache(
            key=key_int8 if self.quant_method else key,
            value=value_int8 if self.quant_method else value,
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_indices=attn_metadata.slot_mapping)
        return key, value

    def apply_prefill_attention(self, query, key, value, attn_metadata):
        attn_mask = attn_metadata.attn_mask
        seq_lens = torch.cumsum(attn_metadata.seq_lens, dim=0)
        num_tokens = query.shape[0]

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=self.block_size,
            actual_seq_lengths=seq_lens.to(torch.int64),
            actual_seq_lengths_kv=seq_lens.to(torch.int64),
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, -1)
        return attn_output
    
    def apply_decode_attention(self, query, key, value, attn_metadata):
        block_tables = attn_metadata.block_tables
        seq_lens = attn_metadata.seq_lens
        batch_size = seq_lens.shape[0]
        query = query.view(batch_size, 1, self.q_size)
        key = self.key_cache
        value = self.value_cache
        block_size = self.block_size
        if key is not None and value is not None:
            block_size = key.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            scale=self.scale,
            block_table=block_tables,
            actual_seq_lengths=[1] * len(seq_lens),
            actual_seq_lengths_kv=seq_lens,
            antiquant_scale=self.quant_method.kv_dequant_scale if self.quant_method else None,
            antiquant_offset=self.quant_method.kv_dequant_offset if self.quant_method else None,
        )
        attn_output = attn_output.view(batch_size, -1)
        return attn_output
    
    def apply_fullgraph_attention(self, query, attn_metadata):
        block_size = self.block_size
        k_cache = self.key_cache.view(-1, block_size, self.num_kv_heads * self.head_size)
        v_cache = self.value_cache.view(-1, block_size, self.num_kv_heads * self.head_size)
        query = query.reshape(-1, 1, self.q_size)
        seq_lens_list = attn_metadata.seq_lens_list
        batch_size = query.shape[0]
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query,
            k_cache,
            v_cache,
            block_table=attn_metadata.block_tables,
            block_size=block_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            scale=self.scale,
            actual_seq_lengths_kv=seq_lens_list,
            antiquant_scale=self.quant_method.kv_dequant_scale if self.quant_method else None,
            antiquant_offset=self.quant_method.kv_dequant_offset if self.quant_method else None,
        )
        output = torch.empty(
            (batch_size, 1, self.num_heads * self.head_size),
            dtype=query.dtype,
            device=query.device,
        )
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)

        torch_npu.npu_fused_infer_attention_score.out(
            query,
            k_cache,
            v_cache,
            block_table=attn_metadata.block_tables,
            block_size=block_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            scale=self.scale,
            actual_seq_lengths_kv=seq_lens_list,
            antiquant_scale=self.quant_method.kv_dequant_scale if self.quant_method else None,
            antiquant_offset=self.quant_method.kv_dequant_offset if self.quant_method else None,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        output = output.view(batch_size, -1)
        return output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forward_context: ForwardContext,
        attn_metadata: AttentionMetadata
    ):
        if forward_context.is_prefill:
            return self.apply_prefill_attention(query, key, value, attn_metadata)
        elif forward_context.capturing:
            return self.apply_fullgraph_attention(query, attn_metadata)
        else:
            return self.apply_decode_attention(query, key, value, attn_metadata)

    def forward(
        self,
        layer: AttentionLayer,  # NOTE: C8 will use this param in the future.
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer.prefix]
        key, value = self.reshape_and_cache(key, value, kv_cache, attn_metadata)
        return self.forward_impl(query, key, value, forward_context, attn_metadata)