# Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
from typing import Optional, List, Tuple
import copy
import math
import torch
import torch_npu

from atb_llm.utils.data.layer_adapter import ParallelLMHead
from atb_llm.utils.data.quant_method_adapter import LinearMethodSupportAtbGraph
from mindie_llm.runtime.layers.custom_layer import CustomLayer
from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType
from .modeling_qwen2 import FlashQwenModel
from .modeling_qwen2_refactor import Qwen2Model
from ..base.flash_causal_lm import FlashForCausalLM, DistributedType, LwdLayerStatus
from ..base.graph_manager import ATBGraphManager, DapGraphWrapper, SpeculateGraphWrapper, \
    SplitFuseGraphWrapper, SingleLoraGraphWrapper, MultiLoraGraphWrapper, FlashCommGraphWrapper, \
    MemPoolGraphWrapper, get_layerwise_decode_graph, get_layerwise_prefill_graph
from ..base.graph_manager.layerwise_combined_graph_wrapper import LayerwiseCombinedATBGraphWrapper
from ..base.inputs_modifier.flash_comm_modifier import FlashCommModifier
from ..base.inputs_modifier.long_seq_modifier import LongSeqModifier
from ..base.inputs_modifier.lora_modifier import LoraModifier
from ..base.inputs_modifier.qlen_modifier import QLenModifier
from ..base.inputs_modifier.layerwise_modifier import LayerwiseModifier
from ...utils.env import ENV
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers import load_column_multi, TensorHead, TensorEmbedding
from ...utils.log import logger, print_log
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.linear.linear_utils import LinearUtils
from ...utils.quantize.quant_type import QuantType, LinearTypeV2
from ...utils.op_backend import OpBackend


CPP_QWEN_MODEL_CLASS_NAME = "qwen_QwenDecoderModel"
LINEAR_HAS_BIAS = "linearHasBias"
LWD_START_ID = "startLayerId"
LWD_END_ID = "endLayerId"
LWD_LAYERS = "layers"
LWD_HEAD = "head"
LWD_TAIL = "tail"
LWD_DECODE = "layerwise-decode"

_800_9000_SOCS = (100, 101, 102, 103, 104) # Special Types
DUO_SOCS = (200, 201, 202, 203, 204, 205)
A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 253, 254, 255)


class FlashQwen2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.acl_decoder_regression_operation = None
        super().__init__(config, weights, **kwargs)
        
        self.enable_rope_quant_kvcache = self.config.quantization_config.kv_quant_type is not None \
                                        and not self.config.use_qk_norm

        model_prefix = kwargs.get("model_prefix", "model")
        lmhead_prefix = kwargs.get("lmhead_prefix", "lm_head")
        transformer_wte_parallel = kwargs.get("transformer_wte_parallel", True)
        self.skip_word_embedding = kwargs.get("skip_word_embedding", False)
        self.aclnn_matmul_backend = False
        self._update_matmul_params(self.quantize)
        LinearUtils.soc_info = self.soc_info
        self.prealloc_weight_mem_on_npu = kwargs.get("prealloc_weight_mem_on_npu", False)
        if self.layerwise_disaggregated:
            self.inference_mode.enable_prefill_pa = True
            self.layerwise.load_list = []
            if self.layerwise.split_type == DistributedType.CLOUD:
                start_layer = self.layerwise.edge_start_layer_count
                end_layer = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
                self.layerwise.load_list = list(range(start_layer, end_layer))
            else:
                self.layerwise.load_list = [i for i in range(0, self.layerwise.edge_start_layer_count)]
                edge_start_layer_count = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
                self.layerwise.load_list.extend([i 
                                                 for i in range(edge_start_layer_count, self.config.num_hidden_layers)])

            self.transformer = FlashQwenModel(
                config, weights, model_prefix=model_prefix, lmhead_prefix=lmhead_prefix,
                attn_decode_backend=self.attn_decode_backend, load_list=self.layerwise.load_list,
                layerwise_disaggregated=self.layerwise_disaggregated
            )
        elif self.prealloc_weight_mem_on_npu:
            LinearMethodSupportAtbGraph.set_soc_info(self.soc_info)
            self.model = Qwen2Model(config, model_prefix, quant_config=kwargs.get("quant_config"))
        else:
            self.transformer = FlashQwenModel(
                config, weights, model_prefix=model_prefix, lmhead_prefix=lmhead_prefix,
                attn_decode_backend=self.attn_decode_backend
            )

        if not self.prealloc_weight_mem_on_npu and not transformer_wte_parallel:
            self.transformer.wte = TensorEmbedding(
                prefix=f"{model_prefix}.embed_tokens", weights=weights
            )
            for p in self.transformer.wte.parameters():
                p.requires_grad = False

        if self.prealloc_weight_mem_on_npu:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                bias=False,
                quant_config=kwargs.get("quant_config"),
                prefix=f"lm_head",
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        elif self.quantize == "w8a8sc" or self.quantize == "w16a16sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=False,
            )
        else:
            if config.tie_word_embeddings:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[f"{model_prefix}.embed_tokens"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
            else:
                self.lm_head = load_column_multi(
                    config,
                    prefixes=[f"{lmhead_prefix}"],
                    weights=weights,
                    head_size=1,
                    lm_head=True,
                )
        self.config = config  # for quantize
        self.attn_mask_fake = self.attn_mask.get_attn_mask(1, dtype=self.dtype, device="npu")
        self.place_holder = torch.tensor([1], dtype=self.dtype, device='npu')

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.acl_param = None
        self.ascend_weight = None
        self.decode_weight = None
        self.weight_initialized = False
        if self.layerwise_disaggregated:
            self.layerwise.ascend_weight_head = None
            self.layerwise.ascend_weight_tail = None
            self.layerwise.ascend_weight_internal = []
            self.layerwise.acl_inputs_prefill = None
            self.layerwise.acl_inputs_decode = None
            self.layerwise.acl_param_prefill = None
            self.layerwise.acl_param_decode = None
            self.layerwise.p_out_hidden = None
            self.layerwise.weight_wrappers = []
            self.layerwise.num_hidden_layers = self.config.num_hidden_layers
        self.long_seq_enable = False
        qwen2_5_has_yarn = hasattr(self.config, 'rope_scaling') and self.config.rope_scaling.type == 'yarn'
        qwen3_has_yarn = hasattr(self.config, 'rope_scaling') and self.config.rope_scaling.rope_type == 'yarn'
        if qwen2_5_has_yarn or qwen3_has_yarn:
            self.long_seq_enable = True
            if self.config.rope_scaling.attention_factor is None:
                self.attention_factor = 1.0
            else:
                self.attention_factor = float(self.config.rope_scaling.attention_factor)
            if self.config.rope_scaling.factor is None:
                raise ValueError('config.rope_scaling.factor must be set in config.json')
            if self.config.rope_scaling.factor <= 1:
                self.mscale = self.attention_factor
            else:
                self.mscale = float((0.1 * math.log(self.config.rope_scaling.factor) + 1.0) * self.attention_factor)
        self.acl_operation_inputs = []
        # 若开启,则冒烟测试卡50ms数据需重新调整(layer多一个输出,内存占用变大)
        self.enable_intra_layer_add_norm = False
        self.enable_inter_layer_add_norm = False
        self.enable_swiglu_quant = not (self.soc_info.need_nz or self.mempool_type == MemPoolType.ASYNC_WRITE)
        # Multi engines management
        if self.layerwise_disaggregated:
            prefill_graph = get_layerwise_prefill_graph(self.config, self.layerwise)
            decode_graph = get_layerwise_decode_graph(self.layerwise)
            self.graph_manager = ATBGraphManager(prefill_graph, decode_graph, LayerwiseCombinedATBGraphWrapper)
        else:
            self.graph_manager = ATBGraphManager()
        self.lora_modifier = LoraModifier(weights, self, lora_adapter=kwargs.get("lora_adapter"), \
            lora_model_config=kwargs.get("lora_model_config"))
        self.flash_comm_modifier = FlashCommModifier(weights, self.hidden_size, self.flash_comm_gate(weights))
        self.qlen_modifier = QLenModifier()
        self.long_seq_modifier = LongSeqModifier(self.config)
        self.layerwise_modifier = LayerwiseModifier(self.layerwise)

    def get_weights(self, quantize_type: QuantType = None):
        quantize_type = self.quantize if quantize_type is None else quantize_type
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=['w2', 'w1'],
            down_name='c_proj'
        )
        kwargs = {
            "enable_rope_quant_kvcache": self.enable_rope_quant_kvcache,
            "enable_swiglu_quant": self.enable_swiglu_quant
        }
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper, **kwargs)
        
        weight_wrapper.register_embedding(self.transformer.wte)
        for i in range(self.num_layers):
            layer = self.transformer.h[i]
            
            weight_wrapper.register_layer(layer, quantize_type)
            if self.config.use_qk_norm:
                weight_wrapper.register_model_norm(layer.attn.q_norm)  # q_norm
                weight_wrapper.register_model_norm(layer.attn.k_norm)  # k_norm
            # not support mempool asyncWrite + add_norm
            if self.mempool_type != MemPoolType.ASYNC_WRITE and \
                (self.enable_intra_layer_add_norm or self.enable_inter_layer_add_norm):
                weight_wrapper.register_layer_addrmsnormquant(layer, attn_wrapper, mlp_wrapper, self.quantize)
            if self.soc_info.need_nz and self.adapter_manager is None:
                del layer.attn
                del layer.ln_2
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
        weight_wrapper.register_model_norm(self.transformer.ln_f)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def get_model_weights(self, quant_type: QuantType = None):
        weights = []
        linear_descs = []
        weight_transpose_type = []
        weights.extend(self.model.embed_tokens.get_weights_for_atb_graph())  # length: 1

        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weights_per_layer, linear_descs_per_layer, weight_transpose_type_per_layer = \
                self.get_layer_weights(layer, quant_type=quant_type)

            weights.extend(weights_per_layer)
            linear_descs.append(linear_descs_per_layer.copy())
            linear_descs_per_layer.clear()
            weight_transpose_type.append(weight_transpose_type_per_layer.copy())
            weight_transpose_type_per_layer.clear()

        weights.extend(self.model.norm.get_weights_for_atb_graph(padding=False))  # length: 1
        weights.extend(self.lm_head.get_weights_for_atb_graph(padding=False))  # length: 1

        return weights, linear_descs, weight_transpose_type

    def get_layer_weights(self, layer: CustomLayer, quant_type: QuantType = None):
        weights_per_layer = []
        linear_descs_per_layer = []
        weight_transpose_type_per_layer = []

        weights_per_layer.extend(layer.input_layernorm.get_weights_for_atb_graph())  # length: 4

        qkv_proj_adapter = layer.self_attn.qkv_proj
        weights_per_layer.extend(qkv_proj_adapter.get_weights_for_atb_graph(quant_type=quant_type))  # length: 18
        linear_descs_per_layer.extend(qkv_proj_adapter.get_linear_descs())  # length: 7
        weight_transpose_type_per_layer.extend(qkv_proj_adapter.get_weight_transpose_type())  # length: 7

        o_proj_adapter = layer.self_attn.o_proj
        weights_per_layer.extend(o_proj_adapter.get_weights_for_atb_graph(quant_type=quant_type))  # length: 6
        linear_descs_per_layer.extend(o_proj_adapter.get_linear_descs())  # length: 7
        weight_transpose_type_per_layer.extend(o_proj_adapter.get_weight_transpose_type())  # length: 7

        weights_per_layer.extend(layer.post_attention_layernorm.get_weights_for_atb_graph())  # length: 4

        gate_up_proj_adapter = layer.mlp.gate_up_proj
        weights_per_layer.extend(gate_up_proj_adapter.get_weights_for_atb_graph(quant_type=quant_type))  # length: 12
        linear_descs_per_layer.extend(gate_up_proj_adapter.get_linear_descs())  # length: 7
        weight_transpose_type_per_layer.extend(gate_up_proj_adapter.get_weight_transpose_type())  # length: 7

        down_proj_adapter = layer.mlp.down_proj
        weights_per_layer.extend(
            down_proj_adapter.get_weights_for_atb_graph(
                is_swiglu_quant_enabled=self.enable_swiglu_quant, quant_type=quant_type))  # length: 6
        linear_descs_per_layer.extend(down_proj_adapter.get_linear_descs())  # length: 7
        weight_transpose_type_per_layer.extend(down_proj_adapter.get_weight_transpose_type())  # length: 7

        if self.config.use_qk_norm:
            weights_per_layer.extend(layer.self_attn.q_norm.get_weights_for_atb_graph(padding=False))  # length: 1
            weights_per_layer.extend(layer.self_attn.k_norm.get_weights_for_atb_graph(padding=False))  # length: 1

        return weights_per_layer, linear_descs_per_layer, weight_transpose_type_per_layer

    def get_layerwsie_ascend_param(self, ascend_params, mode, linear_has_bias, wrapper: WeightWrapper):
        modify_ascend_params = copy.deepcopy(ascend_params)
        modify_ascend_params["linearTransposeType"] = wrapper.linear_transpose_types
        modify_ascend_params["linearQuantType"] = wrapper.linear_type
        modify_ascend_params["packQuantType"] = wrapper.pack_quant_type
        modify_ascend_params["layerwiseMode"] = mode
        modify_ascend_params["linearDescs"] = wrapper.linear_descs
        if mode == LwdLayerStatus.EDGE_START_LAYER:
            modify_ascend_params[LINEAR_HAS_BIAS] = linear_has_bias * self.layerwise.edge_start_layer_count
            modify_ascend_params[LWD_START_ID] = 0
            modify_ascend_params[LWD_END_ID] = self.layerwise.edge_start_layer_count
        elif mode == LwdLayerStatus.CLOUD_MIDDLE_LAYER:
            modify_ascend_params[LINEAR_HAS_BIAS] = linear_has_bias * (
                self.config.num_hidden_layers - self.layerwise.edge_start_layer_count -
                self.layerwise.edge_end_layer_count)
            modify_ascend_params[LWD_START_ID] = self.layerwise.edge_start_layer_count
            modify_ascend_params[LWD_END_ID] = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
        elif mode == LwdLayerStatus.EDGE_END_LAYER:
            modify_ascend_params[LINEAR_HAS_BIAS] = linear_has_bias * self.layerwise.edge_end_layer_count
            modify_ascend_params[LWD_START_ID] = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
            modify_ascend_params[LWD_END_ID] = self.config.num_hidden_layers
        modify_ascend_params["numHiddenLayers"] = modify_ascend_params[LWD_END_ID] - modify_ascend_params[LWD_START_ID]
        
        return modify_ascend_params
        
    
    def get_layerwise_weights(self, mode, quantize_type: QuantType = None, layer_no=None, is_prefill=False):
        quantize_type = self.quantize if quantize_type is None else quantize_type
        attn_wrapper = AttnWrapper(
            norm_name='ln_1',
            wrapper_name='attn',
            pack_name='c_attn',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='c_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='ln_2',
            wrapper_name='mlp',
            pack_name='w2_w1',
            sep_names=['w2', 'w1'],
            down_name='c_proj'
        )
        kwargs = {
            "enable_rope_quant_kvcache": self.enable_rope_quant_kvcache,
            "enable_swiglu_quant": self.enable_swiglu_quant
        }
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper, **kwargs)

        if mode == LwdLayerStatus.EDGE_START_LAYER:
            weight_wrapper.register_embedding(self.transformer.wte)
        start_layer = 0
        end_layer = 0
        if mode == LwdLayerStatus.EDGE_START_LAYER:
            start_layer = 0
            end_layer = self.layerwise.edge_start_layer_count
        elif mode == LwdLayerStatus.CLOUD_MIDDLE_LAYER:
            if is_prefill:
                start_layer = self.layerwise.load_list.index(layer_no)
                end_layer = self.layerwise.load_list.index(layer_no) + 1
            else:
                start_layer = self.layerwise.load_list.index(self.layerwise.edge_start_layer_count)
                end_layer = self.layerwise.load_list.index(
                    self.config.num_hidden_layers - self.layerwise.edge_end_layer_count - 1) + 1
        else:
            start_layer = self.layerwise.load_list.index(self.config.num_hidden_layers -
                                                         self.layerwise.edge_end_layer_count)
            end_layer = self.layerwise.load_list.index(self.config.num_hidden_layers - 1) + 1
        for i in range(start_layer, end_layer):
            layer = self.transformer.h[i]
            weight_wrapper.register_layer(layer, quantize_type)
            if self.config.use_qk_norm:
                weight_wrapper.register_model_norm(layer.attn.q_norm)  # q_norm
                weight_wrapper.register_model_norm(layer.attn.k_norm)  # k_norm
            # not support mempool asyncWrite + add_norm
            if self.mempool_type != MemPoolType.ASYNC_WRITE and \
                (self.enable_intra_layer_add_norm or self.enable_inter_layer_add_norm):
                weight_wrapper.register_layer_addrmsnormquant(layer, attn_wrapper, mlp_wrapper, self.quantize)
            if self.soc_info.need_nz and self.adapter_manager is None:
                del layer.attn
                del layer.ln_2
                del layer.mlp
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
        if mode == LwdLayerStatus.EDGE_END_LAYER:
            weight_wrapper.register_model_norm(self.transformer.ln_f)
            weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        linear_types = None
        pack_quant_configs = None
        linear_descs_configs = None
        linear_transpose_types = None

        if not self.layerwise_disaggregated:
            if self.prealloc_weight_mem_on_npu:
                if self.quantize == QuantType.W8A8_PDMIX:
                    self.ascend_weight, linear_descs_configs, linear_transpose_types = self.get_model_weights(
                        quant_type=QuantType.W8A8_DYNAMIC)
                    self.decode_weight, _, _ = self.get_model_weights(quant_type=QuantType.W8A8)
                else:
                    self.ascend_weight, linear_descs_configs, linear_transpose_types = self.get_model_weights()
            else:
                if self.quantize == QuantType.W8A8_PDMIX:
                    weight_wrapper = self.get_weights(quantize_type=QuantType.W8A8_DYNAMIC)
                    decode_weight_wrapper = self.get_weights(quantize_type=QuantType.W8A8)
                    self.decode_weight = decode_weight_wrapper.weights
                else:
                    weight_wrapper = self.get_weights()
        else:
            if self.layerwise.split_type == DistributedType.CLOUD:
                self.layerwise.weight_wrappers = []
                for i in range(self.layerwise.edge_start_layer_count, 
                                self.config.num_hidden_layers - self.layerwise.edge_end_layer_count):
                    self.layerwise.weight_wrappers.append(self.get_layerwise_weights(
                        mode=LwdLayerStatus.CLOUD_MIDDLE_LAYER, layer_no=i, is_prefill=True))
                decode_weight_wapper = self.get_layerwise_weights(mode=LwdLayerStatus.CLOUD_MIDDLE_LAYER,
                                                                is_prefill=False)
            else:
                weight_wrapper_head = self.get_layerwise_weights(mode=LwdLayerStatus.EDGE_START_LAYER)
                weight_wrapper_tail = self.get_layerwise_weights(mode=LwdLayerStatus.EDGE_END_LAYER)

        if not self.prealloc_weight_mem_on_npu and not self.layerwise_disaggregated:    
            self.ascend_weight = weight_wrapper.weights
            linear_types = weight_wrapper.linear_type
            pack_quant_configs = weight_wrapper.pack_quant_type
            linear_descs_configs = weight_wrapper.linear_descs
            linear_transpose_types = weight_wrapper.linear_transpose_types  
        elif not self.prealloc_weight_mem_on_npu and self.layerwise.split_type == DistributedType.EDGE:
            self.layerwise.ascend_weight_head = weight_wrapper_head.weights            
            self.layerwise.ascend_weight_tail = weight_wrapper_tail.weights


        if self.quantize == QuantType.W8A8_PDMIX:
            linear_descs_configs = [[linear_desc if linear_desc != LinearTypeV2.W8A8_PDMIX else
                LinearTypeV2.W8A8_DYNAMIC for linear_desc in linear_descs] for linear_descs in linear_descs_configs]
            decode_linear_descs_configs = [[linear_desc if linear_desc != LinearTypeV2.W8A8_DYNAMIC else
                LinearTypeV2.W8A8 for linear_desc in linear_descs] for linear_descs in linear_descs_configs]
        else:
            if not self.layerwise_disaggregated:
                decode_linear_descs_configs = linear_descs_configs
        
        if self.config.model_type == "qwen3":
            linear_has_bias = [[self.config.attention_bias, False, False, False]] 
        else:
            linear_has_bias = [[True, False, False, False]]
        if self.prealloc_weight_mem_on_npu:
            lm_head_transpose_type = self.lm_head.get_weight_transpose_type()[0]
        else:
            lm_head_transpose_type = self.lm_head.linear.trans_flag
        acl_param_dict = {
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "skipWordEmbedding": self.skip_word_embedding,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "linearTransposeType": linear_transpose_types,
            "lmHeadTransposeType": lm_head_transpose_type,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableSwigluQuant": self.enable_swiglu_quant,
            "enablePreFetchWeight": False,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "isUnpadInputs": True,
            "hiddenSize": self.hidden_size,
            "enableFA3": self.config.quantization_config.fa_quant_type is not None,
            "worldSize": self.tp_world_size,
            "backend": self.soc_info.communication_backend,
            "attnBackend": self.attn_decode_backend,
            "quantGroupSize": self.config.quantization_config.group_size,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "isLongSeq": self.long_seq_enable,
            "enableAddNorm": False,
            "isYarn": self.long_seq_enable,
            "mscale": self.mscale if self.long_seq_enable else 1.0,
            "rankTableFile": ENV.rank_table_file,
            "useQKNorm": self.config.use_qk_norm,
            LINEAR_HAS_BIAS: linear_has_bias * self.config.num_hidden_layers 
                                    if not self.layerwise_disaggregated else None,
            "matmulBackend": OpBackend.ACLNN if self.aclnn_matmul_backend else OpBackend.ATB,
            "enableIntraLayerAddNorm": self.enable_intra_layer_add_norm and \
                                        self.mempool_type != MemPoolType.ASYNC_WRITE,
            "enableInterLayerAddNorm": self.enable_inter_layer_add_norm and \
                                        self.mempool_type != MemPoolType.ASYNC_WRITE,
            "enableGreedySearchOpt": self.enable_greedy_search_opt,
            "enableOmniAttention": self.omni_attention_enable,
            "enableQScale": (self.config.transformers_version == "4.43.1" or
                            self.config.transformers_version == "4.44.0") and \
                            self.config.num_hidden_layers == 28 and \
                            self.soc_info.need_nz,
                            # QwenCode2.5-7B-Instruct/Qwen2.5-7B/1.5B-Instruct模型时为True, 其他模型为False
            "enableModelConfuscation": self.enable_model_obfuscation,
            "modelConfuscationFd": self.obfuscation_fd,
            "mapping": self.mapping.to_dict_v2(),
        }
        if linear_types is not None:
            acl_param_dict["linearQuantType"] = linear_types
        if pack_quant_configs is not None:
            acl_param_dict["packQuantType"] = pack_quant_configs
        encoder_param = {
            **acl_param_dict,
            "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "enableMC2": ENV.enable_mc2,
            "linearDescs": linear_descs_configs,
            "enableRopeQuantKvcache": self.enable_rope_quant_kvcache and not self.omni_attention_enable,
        }
        decoder_param = {
            **acl_param_dict,
            "isPrefill": False,
            "enableLcoc": False,
            "linearDescs": decode_linear_descs_configs if not self.layerwise_disaggregated else None,
            "enableRopeQuantKvcache": self.enable_rope_quant_kvcache,
            "preFetchWeightSize": 0, # MB
        }

        if not self.layerwise_disaggregated:
            #Mooncake池化与lora、dap、flashcomm不适配
            if self.adapter_manager is not None and self.mempool_type == MemPoolType.ASYNC_WRITE:
                raise ValueError("Feature composition not supported: If lora is activated, "
                                 "mempool_type must be DISABLED or SYNC_WRITE.")
            if self.enable_dap and self.mempool_type == MemPoolType.ASYNC_WRITE:
                raise ValueError("Feature composition not supported: If dap is activated, "
                                 "mempool_type must be DISABLED or SYNC_WRITE.")
            if self.mempool_type == MemPoolType.ASYNC_WRITE:
                self.flash_comm_modifier.enable_flash_comm = False

            if self.adapter_manager is not None:
                self.graph_manager.register_graph(MultiLoraGraphWrapper())
                self.graph_manager.register_graph(SingleLoraGraphWrapper())

            if self.enable_dap:
                self.graph_manager.register_graph(DapGraphWrapper())

            if self.prefix_cache_enable:
                self.graph_manager.register_graph(SplitFuseGraphWrapper())

            if self.speculate_enable:
                self.graph_manager.register_graph(SpeculateGraphWrapper())

            if self.flash_comm_modifier.enable_flash_comm:
                self.graph_manager.register_graph(FlashCommGraphWrapper())

            #Mooncake池化
            if self.mempool_type == MemPoolType.ASYNC_WRITE:
                self.graph_manager.register_graph(MemPoolGraphWrapper())

            specified_params = {"decode": decoder_param}
            specified_weight = {"decode": self.decode_weight}
            self.graph_manager.set_param(CPP_QWEN_MODEL_CLASS_NAME, encoder_param, specified_params)
            self.graph_manager.set_weight(self.ascend_weight, specified_weight)
        else:
            if self.layerwise.split_type == DistributedType.EDGE:
                encoder_head_param = self.get_layerwsie_ascend_param(
                    encoder_param, 0, linear_has_bias, weight_wrapper_head
                )
                encoder_tail_param = self.get_layerwsie_ascend_param(
                    encoder_param, 2, linear_has_bias, weight_wrapper_tail
                )
                decoder_head_param = self.get_layerwsie_ascend_param(
                    decoder_param, 0, linear_has_bias, weight_wrapper_head
                )
                decoder_tail_param = self.get_layerwsie_ascend_param(
                    decoder_param, 2, linear_has_bias, weight_wrapper_tail
                )
                layerwise_encoder_param = {
                    LWD_HEAD: encoder_head_param,
                    LWD_TAIL: encoder_tail_param,
                }
                layerwise_weights = {
                    LWD_HEAD: weight_wrapper_head.weights,
                    LWD_TAIL: weight_wrapper_tail.weights,
                }
                specified_params = {
                    LWD_DECODE: {
                        LWD_HEAD: decoder_head_param,
                        LWD_TAIL: decoder_tail_param,
                    },
                }
                
                self.graph_manager.register_graph(SplitFuseGraphWrapper())
                self.graph_manager.set_param(CPP_QWEN_MODEL_CLASS_NAME, layerwise_encoder_param, specified_params)
                self.graph_manager.set_weight(layerwise_weights)
            else:
                params_list = []
                weights_list = []
                for layer in range(0, self.config.num_hidden_layers - \
                                   self.layerwise.edge_end_layer_count - self.layerwise.edge_start_layer_count):
                    encoder_internal_param = self.get_layerwsie_ascend_param(encoder_param, 1,
                                        linear_has_bias, self.layerwise.weight_wrappers[layer])
                    encoder_internal_param[LWD_START_ID] = self.layerwise.edge_start_layer_count + layer
                    encoder_internal_param[LWD_END_ID] = self.layerwise.edge_start_layer_count + layer + 1
                    encoder_internal_param["numHiddenLayers"] = 1
                    encoder_internal_param[LINEAR_HAS_BIAS] = linear_has_bias
                    encoder_internal_param["reuseEmbedTable"] = self.long_seq_enable and layer != 0
                    encoder_internal_param["outputEmbedTable"] = self.long_seq_enable and layer == 0
                    params_list.append(encoder_internal_param)
                    weights_list.append(self.layerwise.weight_wrappers[layer].weights)
                layerwise_encoder_params = {
                    LWD_LAYERS: params_list,
                }
                layerwise_encode_weights = {
                    LWD_LAYERS: weights_list
                }
                specified_params = {
                    LWD_DECODE: self.get_layerwsie_ascend_param(
                        decoder_param, 1, linear_has_bias, decode_weight_wapper
                    ),
                }
                specified_weights = {
                    LWD_DECODE: decode_weight_wapper.weights
                }
                self.graph_manager.register_graph(SplitFuseGraphWrapper())
                self.graph_manager.set_param(CPP_QWEN_MODEL_CLASS_NAME, layerwise_encoder_params, specified_params)
                self.graph_manager.set_weight(layerwise_encode_weights, specified_weights)
        self.weight_initialized = True

    # Static condition check for FlashComm enablement, called during model initialization
    def flash_comm_gate(self, weights) -> bool:
        if self.enable_dap:
            return False
        if self.tp_world_size == 1:
            return False
        if self.soc_info.soc_version in _800_9000_SOCS:
            return False
        # DUO case currently don't support TP>4 scenarios
        if self.soc_info.soc_version in DUO_SOCS and self.tp_world_size > 4:
            return False
        # FlashComm is temporarily not supported for 910 standard card scenarios
        if not self.soc_info.is_support_hccs() and self.soc_info.soc_version in A2_SOCS + A3_SOCS:
            return False
        if weights.quant_desc is None:
            fallback_exceeds_limit = True
        else:
            fallback_count = sum(
                1 for key, value in weights.quant_desc.items()
                if key.endswith(".weight") and value == "FLOAT" and ("mlp.c_proj" in key or "mlp.down_proj" in key)
            )
            # Allow at most ~1/7 of layers with unquantized MLP projections
            fallback_exceeds_limit = fallback_count * 7 > self.num_layers
        if all([
            self.lcoc_enable,
            self.soc_info.communication_backend == "lccl",
            fallback_exceeds_limit,
        ]):
            return False
        return True

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        cos_table, sin_table = self.place_holder, self.place_holder
        if self.long_seq_enable:
            self.rotary_embedding.yarn_scaling_rotary_embedding(self.config, self.device, max_seq_len)
        else:
            self.rotary_embedding.update_cos_sin_cache_total(
                self.dtype,
                self.device,
                self.max_position_embeddings
            )
            cos_table = self.rotary_embedding.get_cos_cached_total()
            sin_table = self.rotary_embedding.get_sin_cached_total()

        attention_mask = kwargs.get('attn_mask', None)
        if attention_mask is None:
            if is_prefill:
                attention_mask = self.attn_mask.get_rope_prefill_mask(self.max_base_len, self.dtype, self.device)
            else:
                attention_mask = self.attn_mask.get_rope_decode_mask(self.dtype, self.device)
        if self.soc_info.need_nz:
            attention_mask = self.transdata_operation.execute([attention_mask])[0]

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

        # qwen-vl will enable skip_word_embedding
        if not is_prefill and self.skip_word_embedding:
            input_ids = self.transformer.wte(input_ids)

        acl_operation_inputs_ = [
            input_ids,  # IN_TENSOR_INPUTIDS
            position_ids, # IN_TENSOR_POSITIONIDS
            cos_table,  # IN_TENSOR_COSEMBED
            sin_table,  # IN_TENSOR_SINEMBED
            attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            self.place_holder,  # IN_TENSOR_KV_CACHE_IDX
            self.place_holder,  # IN_TENSOR_TOKEN_OFFSET
            self.place_holder,
            input_lengths.to(torch.int32),  # IN_TENSOR_SEQ_LENGTHS
            lm_head_indices.to(torch.int64) if is_prefill else self.placeholder,  # IN_TENSOR_LOGTIS_INDICES
        ]  # 0-11

        self.acl_param = {
            "seqLen": input_lengths.tolist()
        }

        self.long_seq_modifier.modify_inputs(
            acl_operation_inputs_,
            self.rotary_embedding,
            position_ids)
        self.qlen_modifier.modify_inputs(
            acl_operation_inputs_,
            self.acl_param,
            input_ids.device,
            is_prefill=is_prefill,
            enable_prefill_pa=False if self.inference_mode is None else self.inference_mode.enable_prefill_pa,
            enable_splitfuse_pa=not self.soc_info.is_300i(),
            **kwargs)
        self.lora_modifier.modify_inputs(
            acl_operation_inputs_,
            kwargs.get("adapter_ids"),
            input_lengths, is_prefill)
        self.flash_comm_modifier.modify_inputs(
            acl_operation_inputs_,
            is_prefill,
            self.acl_param
        )

        self.layerwise_modifier.modify_inputs(
                acl_operation_inputs_,
                is_prefill,
                self.acl_param,
                position_ids,
                input_lengths,
                **kwargs
        )
        self.acl_operation_inputs = acl_operation_inputs_
        self.acl_param = json.dumps(self.acl_param)

        return self.acl_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill, **kwargs):
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage", None)
        runtime_mempool_type = self.mempool_type if self.warmup_is_end else MemPoolType.DISABLED
        acl_model_out = self.graph_manager.select_and_execute(
            self, acl_inputs, acl_param, is_prefill=is_prefill, layerwise_disaggregated_exe_stage=exe_stage,
            mempool_type=runtime_mempool_type)
        try:
            acl_model_out = self.layerwise_modifier.process_out(acl_model_out, is_prefill=is_prefill, **kwargs)
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        return acl_hidden_state

    def execute_dap_ascend_operator(self,
                                acl_inputs: list,
                                acl_param: str,
                                is_prefill: bool) -> torch.Tensor:
        """Execute the Ascend acl operator."""
        if not is_prefill:
            raise NotImplementedError("Dap is not supported for decoder.")
        acl_model_out = self.graph_manager.select_and_execute(self, acl_inputs, acl_param, \
            is_prefill=is_prefill, enable_dap=True)
        if len(acl_model_out) != 2:
            raise RuntimeError("Number of output tensors is not equal to the expected value.")
        return acl_model_out

    def init_kvcache(self, kv_cache):
        kcache_id_diff = self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id_diff = self.ascend_vcache_id != id(kv_cache[0][1])
        kcache_shape_diff = self.ascend_kcache_shape != kv_cache[0][0].shape
        vcache_shape_diff = self.ascend_vcache_shape != kv_cache[0][1].shape
        kcache_diff = not self.ascend_kcache_id or kcache_id_diff or kcache_shape_diff
        vcache_diff = not self.ascend_vcache_id or vcache_id_diff or vcache_shape_diff
        if kcache_diff or vcache_diff:
            k_caches, v_caches = map(lambda x: list(x), zip(*kv_cache))
            print_log(self.tp_rank, logger.info, f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            
            if self.layerwise_disaggregated:
                if self.layerwise.split_type == DistributedType.EDGE:
                    start_cache_num = self.layerwise.load_list.index(
                        self.config.num_hidden_layers - self.layerwise.edge_end_layer_count)
                    end_cache_num = self.layerwise.load_list.index(self.config.num_hidden_layers - 1) + 1
                    k_caches_sp1, k_caches_sp3 = [k_caches[i] for i in range(self.layerwise.edge_start_layer_count)], \
                        [k_caches[i] for i in range(start_cache_num, end_cache_num)]
                    v_caches_sp1, v_caches_sp3 = [v_caches[i] for i in range(self.layerwise.edge_start_layer_count)], \
                        [v_caches[i] for i in range(start_cache_num, end_cache_num)]
                    layerwise_k_caches = {
                        LWD_HEAD: k_caches_sp1,
                        LWD_TAIL: k_caches_sp3,
                    }
                    layerwise_v_caches = {
                        LWD_HEAD: v_caches_sp1,
                        LWD_TAIL: v_caches_sp3,
                    }
                    self.graph_manager.set_kv_cache(layerwise_k_caches, layerwise_v_caches)
                else:
                    start_cache_num = self.layerwise.load_list.index(self.layerwise.edge_start_layer_count)
                    end_cache_num = self.layerwise.load_list.index(
                        self.config.num_hidden_layers - self.layerwise.edge_end_layer_count - 1) + 1
                    k_caches_sp2 = [k_caches[i] 
                                    for i in range(start_cache_num, end_cache_num)]
                    v_caches_sp2 = [v_caches[i] 
                                    for i in range(start_cache_num, end_cache_num)]
                    encoder_kv_caches = {
                        "k": {
                            LWD_LAYERS: [],
                        },
                        "v": {
                            LWD_LAYERS: [],
                        },
                    }
                    for layer in range(self.config.num_hidden_layers - self.layerwise.edge_start_layer_count - \
                                       self.layerwise.edge_end_layer_count):
                        encoder_kv_caches['k']['layers'].append([k_caches[layer]])
                        encoder_kv_caches['v']['layers'].append([v_caches[layer]])
                    specified_kv_caches = {
                        "layerwise-prefill": encoder_kv_caches,
                    }
                    self.graph_manager.set_kv_cache(k_caches_sp2, v_caches_sp2, specified_kv_caches)
            else:
                self.graph_manager.set_kv_cache(k_caches, v_caches)

            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the model.'

        Args:
            input_ids (torch.Tensor): The input ids tensor.
            position_ids (torch.Tensor): The position ids tensor.
            is_prefill (bool): Whether the inference mode is prefill.
            kv_cache (List[Tuple[torch.Tensor, torch.Tensor]]): Key-value cache.
            block_tables (torch.Tensor): Input block tables.
            slots (torch.Tensor): Input slots.
            input_lengths (torch.Tensor): Input lengths.
            max_seq_len (torch): Maximum sequence length.
            lm_head_indices (torch.Tensor, optional): LM head indices. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            torch.Tensor: Output logits.
        """
        self.warmup_is_end = kwargs.get("warmup_is_end", True)
        if not self.weight_initialized:
            self.get_adapter_ids(**kwargs)
            from mindie_llm.runtime.utils.torch_utils import set_default_torch_dtype
            with set_default_torch_dtype(self.config.torch_dtype):
                self.init_ascend_weight()
        self.init_kvcache(kv_cache)

        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                                block_tables, slots, input_lengths, max_seq_len,
                                                                lm_head_indices, **kwargs)
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill, **kwargs)
        return logits
    
    def _update_matmul_params(self, quantize: QuantType):
        a2_socs = (223, 225)
        a3_socs = (253, 255)
        is_float = quantize is None or quantize == QuantType.FLOAT
        is_aclnn_qmm = quantize in (QuantType.W8A8, QuantType.W8A8_DYNAMIC, QuantType.W8A8_PDMIX)
        
        if self.soc_info.soc_version in a2_socs + a3_socs and (is_float or is_aclnn_qmm):
            self.soc_info.matmul_nd_nz = True

            # in (A2 + fp16 + float) or aclnn_qmm cases, using aclnn backend
            if (self.soc_info.soc_version in a2_socs and is_float and self.dtype == torch.float16):
                self.aclnn_matmul_backend = True
            if is_aclnn_qmm:
                self.aclnn_matmul_backend = True
