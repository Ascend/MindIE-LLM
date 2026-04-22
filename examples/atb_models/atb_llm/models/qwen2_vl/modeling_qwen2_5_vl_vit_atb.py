# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# Implement part of this file based on from transformers
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.buque
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""PyTorch Qwen25-VL-VIT model."""

import json
import math
from collections import OrderedDict, defaultdict

import _libatb_torch as atb
import torch
import torch.nn as nn
import torch.utils.checkpoint
from atb_llm.common_op_builders.common_op_builder_manager import CommonOpBuilderManager
from atb_llm.common_op_builders.data_type import (
    CommonOpBuilderType,
)
from atb_llm.common_op_builders.linear_parallel.base_linear_parallel_common_op_builder import (
    ParallelType,
    TensorParallelInfo,
    CommunicationBackend,
)
from atb_llm.models.base.flash_causal_lm_atb import AtbGraph
from atb_llm.models.base.model_utils import MlpLinearInfo, AttnLinearInfo, LinearInfo
from atb_llm.models.base.modeling_atb import BaseRMSNorm, BaseMLP
from atb_llm.models.qwen2_vl.modeling_qwen2_vl_vit_atb import FastPatchEmbed, VisionRotaryEmbedding
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorReplicatedLinear,
    load_column_multi,
)
from atb_llm.utils.quantize.quant_type import is_same_type, QuantType, LinearTypeV2
from atb_llm.utils.weights import Weights
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_utils import PretrainedConfig

PADDING_HEAD_DIM = 80
PATCH_MERGER_FACTOR = 4
OUT_DATA_TYPE = {torch.bfloat16: "ACL_BF16", torch.float16: "ACL_FLOAT16"}


class PatchMerger(nn.Module):
    def __init__(self, config, weights, prefix, backend):
        super().__init__()
        self.dtype = weights.dtype
        self.backend = backend
        self.quantize = config.quantize

        self.prefix = prefix
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.patch_merger_mlp_0 = TensorReplicatedLinear.load(
            config, prefix=f"{prefix}.mlp.0", weights=weights, bias=True
        )

        self.patch_merger_mlp_2 = TensorReplicatedLinear.load(
            config, prefix=f"{prefix}.mlp.2", weights=weights, bias=True
        )
        self.linear_info = MlpLinearInfo()
        self.linear_info.up_weight_only = True
        self.linear_info.up_linear = self.patch_merger_mlp_0.linear
        self.linear_info.down_linear = self.patch_merger_mlp_2.linear
        self.linear_info.is_pack = False

        self.patch_merger_ln_q = VisionRMSNorm(f"{prefix}.ln_q", config, weights, self.linear_info)

    def get_weights(self):
        weights_dict = OrderedDict()
        weights_dict.update(self.patch_merger_ln_q.get_weights(f"{self.prefix}.ln_q"))
        weights_dict.update(self.patch_merger_mlp_0.linear.get_weights(f"{self.prefix}.mlp.0"))
        weights_dict.update(self.patch_merger_mlp_2.linear.get_weights(f"{self.prefix}.mlp.2"))
        return weights_dict

    def build_mlp0_graph(self, graph, input_tensor_name, output_tensor_name):
        linear_param = {
            "op_name": "patch_merger_mlp_0_linear",
            "category": CommonOpBuilderType.LINEAR,
            "enable_quant_input": True if self.quantize == QuantType.W8A8 else False,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0,
        }
        linear_param.update({"linear_module": self.linear_info.up_linear})
        linear_tensor_map = {"input": input_tensor_name, "linear_out": output_tensor_name}
        builder = CommonOpBuilderManager.get_builder(linear_param)
        graph = builder.build(graph, linear_tensor_map)

    def build_activation_graph(self, graph, input_tensor_name, output_tensor_name):
        act = atb.BaseOperation(
            op_type="Activation",
            op_param=json.dumps({"activationType": "ACTIVATION_GELU"}),
            op_name="patch_merger_activation",
        )
        graph.operations.append(act)
        graph.add_operation(
            act,
            [input_tensor_name],
            [output_tensor_name],
        )

    def build_mlp2_graph(self, graph, input_tensor_name, output_tensor_name):
        down_linear_param = {
            "op_name": "down_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.linear_info.down_linear,
            "enable_quant_input": True if self.quantize == QuantType.W8A8 else False,
            "default_dtype": self.dtype,
            "group_size": 128 if self.quantize == QuantType.W4A16 else 0,
        }
        down_linear_tensor_map = {
            "input": input_tensor_name,
            "linear_out": output_tensor_name,
        }

        linear_parallel_builder = CommonOpBuilderManager.get_builder(down_linear_param)
        graph = linear_parallel_builder.build(graph, down_linear_tensor_map)

    def reshape_out(self, org_shape):
        return [1, org_shape[1] // PATCH_MERGER_FACTOR, org_shape[2] * PATCH_MERGER_FACTOR]

    def build_graph(self, graph, input_tensor_name, output_tensor_name):
        self.patch_merger_ln_q.build_graph(graph, is_prefill=True)
        graph.add_reshape(f"{self.prefix}.ln_q_out", f"{self.prefix}.ln_q_out_reshape", self.reshape_out)
        self.build_mlp0_graph(graph, f"{self.prefix}.ln_q_out_reshape", f"{self.prefix}.mlp_0_out")
        self.build_activation_graph(graph, f"{self.prefix}.mlp_0_out", f"{self.prefix}.activation_out")
        self.build_mlp2_graph(graph, f"{self.prefix}.activation_out", output_tensor_name)


class VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, weights, prefix, norm_prefix, layer_idx, backend):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.norm_prefix = norm_prefix
        self.backend = backend
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim_ori = self.hidden_size // self.num_heads
        self.head_dim = PADDING_HEAD_DIM
        self.quantize = config.quantize
        if config.enable_atb_vit_tp:
            self.num_heads_pre_rank = (self.num_heads + self.tp_world_size - 1) // self.tp_world_size
        else:
            self.num_heads_pre_rank = self.num_heads
        if config.enable_atb_vit_tp:
            self.qkv = TensorParallelColumnLinear.load_qkv(
                config,
                prefix=f"{self.prefix}.qkv",
                weights=weights,
                bias=True,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
            )

            self.proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{self.prefix}.proj",
                weights=weights,
                bias=True,
                gqa_size=self.head_dim_ori,
            )
        else:
            self.qkv = TensorReplicatedLinear.load(
                config,
                prefix=f"{self.prefix}.qkv",
                weights=weights,
                bias=True,
            )

            self.proj = TensorReplicatedLinear.load(
                config,
                prefix=f"{self.prefix}.proj",
                weights=weights,
                bias=True,
            )

    def pad_qkv_bias(self, bias):
        first_half = bias.reshape(self.num_heads_pre_rank, 3, 80)[:, :, :40]
        second_half = bias.reshape(self.num_heads_pre_rank, 3, 80)[:, :, 40:]
        first_half_padded = torch.nn.functional.pad(first_half, (0, 24))
        second_half_padded = torch.nn.functional.pad(second_half, (0, 24))
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(self.num_heads_pre_rank * 128 * 3)
        return bias_final

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        qkv_weights_dict = self.qkv.linear.get_weights(f"{prefix}.qkv")
        out_proj_weights_dict = self.proj.linear.get_weights(f"{prefix}.proj")
        weights_dict.update(qkv_weights_dict)
        weights_dict.update(out_proj_weights_dict)
        return weights_dict

    def get_windows_attention_in_tensor_names(self):
        return ["q_split_reshape", "k_split_reshape", "v_split_reshape", "cu_window_seqlens"]

    def get_attention_in_tensor_names(self):
        return ["q_split_reshape", "k_split_reshape", "v_split_reshape", "seq_len"]

    def build_qkv_graph(self, graph):
        if self.quantize == QuantType.W8A8:
            input_key_list = [
                f"{self.norm_prefix}_out",
                f"{self.prefix}.qkv.weight",
                f"{self.prefix}.qkv.quant_bias",
                f"{self.prefix}.qkv.deq_scale",
            ]
            out_data_type = OUT_DATA_TYPE.get(self.dtype, "ACL_DT_UNDEFINED")
        else:
            input_key_list = [f"{self.norm_prefix}_out", f"{self.prefix}.qkv.weight", f"{self.prefix}.qkv.bias"]
            out_data_type = "ACL_DT_UNDEFINED"

        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({"transposeB": True, "hasBias": True, "outDataType": out_data_type}),
            op_name="layer_qkv_linear",
        )
        graph.operations.append(linear_op)
        graph.add_operation(linear_op, input_key_list, ["qkv_linear_out"])
        split_op = atb.BaseOperation(
            op_type="Split", op_param=json.dumps({"splitDim": -1, "splitNum": 3}), op_name="layer_qkv_split"
        )
        graph.operations.append(split_op)
        graph.add_operation(
            split_op,
            ["qkv_linear_out"],
            ["q_split", "k_split", "v_split"],
        )

    def build_rope_graph(self, graph):
        rope_param = {
            "op_name": "layer_rope",
            "head_num": self.num_heads_pre_rank,
            "kv_head_num": self.num_heads_pre_rank,
            "category": CommonOpBuilderType.ROPE,
            "is_fa": True,
            "atb_rope_param": {"rotaryCoeff": 2},
        }
        rope_tensor_map = {
            "q": "q_split",
            "k": "k_split",
            "cos_embedding": "cos_embedding",
            "sin_embedding": "sin_embedding",
            "seq_len": "seq_len",
            "q_out": "q_split",
            "k_out": "k_split",
        }
        if self.layer_idx not in self.config.fullatt_block_indexes:
            rope_tensor_map["seq_len"] = "cu_window_seqlens"
        rope_builder = CommonOpBuilderManager.get_builder(rope_param)
        graph = rope_builder.build(graph, rope_tensor_map)

    def reshape_qkv(self, org_shape):
        return [org_shape[0] * org_shape[1], self.num_heads_pre_rank, self.head_dim]

    def reshape_out(self, org_shape):
        return [1, org_shape[0], self.num_heads_pre_rank * org_shape[2]]

    def build_attention_graph(self, graph):
        attention_op = atb.BaseOperation(
            op_type="SelfAttention",
            op_param=json.dumps(
                {
                    "headNum": self.num_heads_pre_rank,
                    "kvHeadNum": self.num_heads_pre_rank,
                    "qkScale": 1.0 / math.sqrt(self.head_dim_ori),
                    "calcType": "PA_ENCODER",
                }
            ),
            op_name="layer_selfattention",
        )
        graph.add_reshape("q_split", "q_split_reshape", self.reshape_qkv)
        graph.add_reshape("k_split", "k_split_reshape", self.reshape_qkv)
        graph.add_reshape("v_split", "v_split_reshape", self.reshape_qkv)
        if self.layer_idx in self.config.fullatt_block_indexes:
            input_key_list = self.get_attention_in_tensor_names()
        else:
            input_key_list = self.get_windows_attention_in_tensor_names()

        output_key_list = ["attn_out"]
        graph.operations.append(attention_op)
        graph.add_operation(attention_op, input_key_list, output_key_list)
        graph.add_reshape("attn_out", "attn_out_reshape", self.reshape_out)

    def build_output_proj_graph(self, graph):
        output_proj_linear_param = {
            "op_name": "layer_output_proj_linear",
            "category": CommonOpBuilderType.LINEAR,
            "linear_module": self.proj.linear,
            "enable_quant_input": True if self.quantize == QuantType.W8A8 else False,
            "default_dtype": self.dtype,
            "group_size": 0,
        }
        output_proj_linear_tensor_map = {"input": "attn_out_reshape", "linear_out": "output_proj_out"}
        output_proj_linear_parallel_param = {
            "op_name": "layer_output_proj_linear_parallel",
            "category": CommonOpBuilderType.LINEAR_PARALLEL,
            "parallel_type": ParallelType.ALL_REDUCE,
            "parallel_info": TensorParallelInfo(rank=self.tp_rank, world_size=self.tp_world_size, backend=self.backend),
            "linear_param": output_proj_linear_param,
            "enable_lcoc": False,
        }
        linear_parallel_builder = CommonOpBuilderManager.get_builder(output_proj_linear_parallel_param)
        graph = linear_parallel_builder.build(graph, output_proj_linear_tensor_map)

    def build_output_proj_graph_without_tp(self, graph):
        if self.quantize == QuantType.W8A8:
            input_quant_op = atb.BaseOperation(
                op_type="Elewise",
                op_param=json.dumps({"elewiseType": "ELEWISE_QUANT_PER_CHANNEL"}),
                op_name="attn_proj_elewise_quant",
            )
            graph.operations.append(input_quant_op)
            graph.add_operation(
                input_quant_op,
                ["attn_out_reshape", f"{self.prefix}.proj.input_scale", f"{self.prefix}.proj.input_offset"],
                ["attn_out_reshape_quant_out"],
            )

        input_key_list = ["attn_out_reshape", f"{self.prefix}.proj.weight", f"{self.prefix}.proj.bias"]
        out_data_type = "ACL_DT_UNDEFINED"
        transpose_b = True
        if self.quantize == QuantType.W8A8:
            input_key_list = [
                "attn_out_reshape_quant_out",
                f"{self.prefix}.proj.weight",
                f"{self.prefix}.proj.bias",
                f"{self.prefix}.proj.deq_scale",
            ]
            out_data_type = OUT_DATA_TYPE.get(self.dtype, "ACL_DT_UNDEFINED")
            if self.tp_world_size > 1:
                transpose_b = False

        linear_op = atb.BaseOperation(
            op_type="Linear",
            op_param=json.dumps({"transposeB": transpose_b, "hasBias": True, "outDataType": out_data_type}),
            op_name="layer_output_proj_linear",
        )
        graph.operations.append(linear_op)
        graph.add_operation(linear_op, input_key_list, ["output_proj_out"])

    def build_graph(self, graph):
        attn_res_add = atb.BaseOperation(
            op_type="Elewise",
            op_param=json.dumps({"elewiseType": "ELEWISE_ADD"}),
            op_name="layer_attn_res_add",
        )

        self.build_qkv_graph(graph)
        self.build_rope_graph(graph)
        self.build_attention_graph(graph)
        if self.config.enable_atb_vit_tp:
            self.build_output_proj_graph(graph)
        else:
            self.build_output_proj_graph_without_tp(graph)
        graph.operations.append(attn_res_add)
        graph.add_operation(attn_res_add, ["hidden_states", "output_proj_out"], ["hidden_states"])


class VisionMlp(BaseMLP):
    def __init__(
        self,
        prefix: str,
        config: PretrainedConfig,
        weights: Weights,
        norm_prefix: str,
        backend: CommunicationBackend = CommunicationBackend.LCCL,
    ):
        super().__init__(prefix, config, weights, norm_prefix, backend)

        # 模型结构
        # Gate Up
        if config.quantize == QuantType.W8A8SC:
            self.gate_up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_up_proj",
                weights=weights,
                bias=False,
            )
            self.linear_info.is_pack = True
            self.linear_info.pack_linear = self.gate_up_proj.linear
        else:
            gate_linear_desc = weights.get_linear_quant_type(f"{prefix}.gate_proj.weight")
            up_linear_desc = weights.get_linear_quant_type(f"{prefix}.up_proj.weight")

            if is_same_type([gate_linear_desc, up_linear_desc]):
                self.gate_up_proj = load_column_multi(
                    config,
                    prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                    weights=weights,
                    head_size=1,
                    bias=True,
                )
                self.linear_info.is_pack = True
                self.linear_info.pack_linear = self.gate_up_proj.linear
                if gate_linear_desc in [LinearTypeV2.FLOAT16, LinearTypeV2.BFLOAT16]:
                    self.linear_info.is_all_float = True
            else:
                self.gate_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.gate_proj",
                    weights=weights,
                    bias=True,
                )
                self.up_proj = TensorParallelColumnLinear.load(
                    config,
                    prefix=f"{prefix}.up_proj",
                    weights=weights,
                    bias=True,
                )
                self.linear_info.is_pack = False
                self.linear_info.split_num = 2
                self.linear_info.gate_linear = self.gate_proj.linear
                self.linear_info.up_linear = self.up_proj.linear
        # Down
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=True,
        )
        self.linear_info.down_linear = self.down_proj.linear

    def pad_tensor(self, tensor, target_shape):
        pad_size = target_shape - tensor.shape[0]
        if len(tensor.shape) == 1:
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))  # 1D padding
        elif len(tensor.shape) == 2:
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size))  # 2D padding
        else:
            raise ValueError(f"Unsupported tensor shape in pad_tensor: {tensor.shape}")
        return padded_tensor

    def get_weights(self, prefix: str) -> OrderedDict:
        """Get gate/up/down linear weights."""
        weights_dict = OrderedDict()
        if self.linear_info.is_pack:
            gate_up_proj_weights_dict = self.gate_up_proj.linear.get_weights(f"{prefix}.gate_up_proj")
            weights_dict.update(gate_up_proj_weights_dict)
        else:
            weights_dict.update(self.gate_proj.linear.get_weights(f"{prefix}.gate_proj"))
            weights_dict.update(self.up_proj.linear.get_weights(f"{prefix}.up_proj"))
        down_weight_dict = self.down_proj.linear.get_weights(f"{prefix}.down_proj")
        weights_dict.update(down_weight_dict)
        return weights_dict


class VisionRMSNorm(BaseRMSNorm):
    def __init__(self, prefix: str, config: PretrainedConfig, weights: Weights, linear_info: LinearInfo):
        super().__init__(prefix, config, weights, linear_info)
        if prefix == "visual.merger.ln_q":
            self.has_bias = False
            self.bias = None


class Qwen25VLVisionLayerATB(nn.Module):
    def __init__(self, config, layer_idx, weights, layer_prefix, backend) -> None:
        super().__init__()
        prefix = f"{layer_prefix}.{layer_idx}"
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.config = config
        self.weight_names = None
        self.layer_graph = None

        self.attn = VisionAttention(
            config=config,
            weights=weights,
            prefix=f"{prefix}.attn",
            norm_prefix=f"{prefix}.norm1",
            layer_idx=self.layer_idx,
            backend=backend,
        )

        self.mlp = VisionMlp(
            config=config,
            weights=weights,
            prefix=f"{prefix}.mlp",
            norm_prefix=f"{prefix}.norm2",
            backend=backend,
        )

        attn_linear_info = AttnLinearInfo()
        attn_linear_info.pack_linear = self.attn.qkv.linear
        attn_linear_info.dense_linear = self.attn.proj.linear
        self.norm1 = VisionRMSNorm(f"{prefix}.norm1", config, weights, attn_linear_info)

        self.norm2 = VisionRMSNorm(f"{prefix}.norm2", config, weights, self.mlp.linear_info)

    def get_weights(self, prefix):
        weights_dict = OrderedDict()
        for name, module in self.named_children():
            weights_dict.update(module.get_weights(f"{prefix}.{name}"))
        self.weight_names = list(weights_dict.keys())
        return weights_dict

    def get_windows_in_tensor_names(self):
        return ["hidden_states", "cu_window_seqlens", "cos_embedding", "sin_embedding"]

    def get_in_tensor_names(self):
        return ["hidden_states", "seq_len", "cos_embedding", "sin_embedding"]

    def get_out_tensor_names(self):
        return ["hidden_states"]

    def build_graph(self, graph):
        self.layer_graph = AtbGraph(f"Qwen25VL_VIT_layer_{self.layer_idx}_graph")
        if self.layer_idx in self.config.fullatt_block_indexes:
            self.layer_graph.add_input_output(
                input=self.weight_names + self.get_in_tensor_names(),
                output=["layer_out"],
            )
        else:
            self.layer_graph.add_input_output(
                input=self.weight_names + self.get_windows_in_tensor_names(),
                output=["layer_out"],
            )
        self.norm1.build_graph(self.layer_graph, is_prefill=True)
        self.attn.build_graph(self.layer_graph)
        self.norm2.build_graph(self.layer_graph, is_prefill=True)
        self.mlp.build_graph(self.layer_graph, is_prefill=False)
        self.layer_graph.build()

        graph.operations.append(self.layer_graph)
        if self.layer_idx in self.config.fullatt_block_indexes:
            graph.add_operation(
                self.layer_graph,
                self.weight_names + self.get_in_tensor_names(),
                self.get_out_tensor_names(),
            )
        else:
            graph.add_operation(
                self.layer_graph,
                self.weight_names + self.get_windows_in_tensor_names(),
                self.get_out_tensor_names(),
            )


class Qwen25VLVisionEncoderATB(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.weights = weights
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.dtype = weights.dtype
        self.soc_info = NPUSocInfo()
        self.backend = self.soc_info.communication_backend
        self.layer_prefix = "visual.blocks"
        self.patch_embed_prefix = "visual.patch_embed"
        self.patch_merger_prefix = "visual.merger"
        layers = []
        for layer_idx in range(config.depth):
            layers.append(Qwen25VLVisionLayerATB(config, layer_idx, weights, self.layer_prefix, self.backend))
        self.layers = nn.ModuleList(layers)
        self.patch_embed = FastPatchEmbed(config, weights, self.patch_embed_prefix)
        self.patch_merger = PatchMerger(config, weights, self.patch_merger_prefix, self.backend)

        self.graph = None
        self.graph_inputs = defaultdict(dict)
        self.graph_outputs = defaultdict(dict)
        self.graph_param = defaultdict(dict)
        self.weight = OrderedDict()

    def prepare_inputs(self, hidden_states, cu_window_seqlens, seqlens, cos_vit_mrope, sin_vit_mrope):
        self.graph_inputs.update({"patch_embed_in_tensor": hidden_states})
        self.graph_inputs.update({"cu_window_seqlens": cu_window_seqlens})
        self.graph_inputs.update({"seq_len": seqlens})
        self.graph_inputs.update({"cos_embedding": cos_vit_mrope})
        self.graph_inputs.update({"sin_embedding": sin_vit_mrope})
        self.graph_param["cu_window_seqlens"] = cu_window_seqlens.cpu().to(torch.int32)
        self.graph_param["seq_len"] = seqlens.cpu().to(torch.int32)
        self.graph_outputs["patch_merger_out_tensor"] = torch.ones(
            1,
            hidden_states.shape[1] // PATCH_MERGER_FACTOR,
            self.config.out_hidden_size,
            dtype=hidden_states.dtype,
            device="npu",
        )

    def forward(self, hidden_states, cu_window_seqlens, seqlens, cos_vit_mrope, sin_vit_mrope):
        self.prepare_inputs(hidden_states, cu_window_seqlens, seqlens, cos_vit_mrope, sin_vit_mrope)
        hidden_states = self.graph.forward(self.graph_inputs, self.graph_outputs, self.graph_param)
        return hidden_states

    def get_in_tensor_names(self):
        return ["patch_embed_in_tensor", "cu_window_seqlens", "seq_len", "cos_embedding", "sin_embedding"]

    def get_out_tensor_names(self):
        return ["patch_merger_out_tensor"]

    def get_weights(self):
        weights_dict = OrderedDict()
        layer_idx = 0
        weights_dict.update(self.patch_embed.get_weights())
        for layer in self.layers:
            weights = layer.get_weights(f"{self.layer_prefix}.{layer_idx}")
            weights_dict.update(weights)
            layer_idx += 1
        weights_dict.update(self.patch_merger.get_weights())
        return weights_dict

    def build_graph(self):
        self.graph.add_input_output(
            input=list(self.weight.keys()) + self.get_in_tensor_names(), output=self.get_out_tensor_names()
        )
        self.patch_embed.build_graph(self.graph, self.get_in_tensor_names()[0], "hidden_states")
        for layer in self.layers:
            layer.build_graph(self.graph)
        self.patch_merger.build_graph(self.graph, "hidden_states", self.get_out_tensor_names()[0])

        self.graph.execute_as_single = False
        self.graph.build()
        self.graph.set_weights(self.weight)

    def init_graph(self):
        self.weight = self.get_weights()
        self.graph = AtbGraph("Qwen25VL_VIT_graph")
        self.build_graph()


class Qwen25VisionTransformerPretrainedModelATB(PreTrainedModel):
    def __init__(self, config, weights, max_seq_len) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.weights = weights
        self.head_dim = config.hidden_size // config.num_heads
        self.window_size = config.window_size
        self.encoder = Qwen25VLVisionEncoderATB(config, self.weights)
        self.rotary_pos_emb = VisionRotaryEmbedding(
            weights.device, weights.dtype, self.head_dim // 2, seqlen=max_seq_len
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        cos = self.rotary_pos_emb.cos_cache[pos_ids].flatten(1)
        sin = self.rotary_pos_emb.sin_cache[pos_ids].flatten(1)
        cos_vit_mrope = cos.repeat(1, 2)
        sin_vit_mrope = sin.repeat(1, 2)
        return cos_vit_mrope, sin_vit_mrope

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        cos_vit_mrope, sin_vit_mrope = self.rot_pos_emb(grid_thw.to(torch.int32))
        cos_vit_mrope = cos_vit_mrope.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        cos_vit_mrope = cos_vit_mrope[window_index, :, :]
        cos_vit_mrope = cos_vit_mrope.reshape(seq_len, -1).unsqueeze(0)
        sin_vit_mrope = sin_vit_mrope.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        sin_vit_mrope = sin_vit_mrope[window_index, :, :]
        sin_vit_mrope = sin_vit_mrope.reshape(seq_len, -1).unsqueeze(0)

        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = torch.diff(cu_window_seqlens)
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        vision_features = self.encoder(
            hidden_states.unsqueeze(0),
            cu_window_seqlens.to(torch.int32),
            seqlens.to(torch.int32),
            cos_vit_mrope,
            sin_vit_mrope,
        )

        hidden_states = vision_features[self.encoder.get_out_tensor_names()[0]].squeeze()
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states
