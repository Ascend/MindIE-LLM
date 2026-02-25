# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Optional, List, Tuple
import json
import math

import torch
import torch_npu

from atb_llm.models.base.graph_manager import ATBGraphManager, SpeculateGraphWrapper, \
    SplitFuseGraphWrapper, SingleLoraGraphWrapper, MultiLoraGraphWrapper, FlashCommGraphWrapper
from atb_llm.models.base.inputs_modifier import FlashCommModifier, QLenModifier, LoraModifier, LongSeqModifier
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.quantize.pack_type import QuantType
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from .config_llama import LlamaConfig
from .modeling_llama import FlashLlamaModel
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.env import ENV
from ...utils.layers import load_column_multi, TensorHead, TensorParallelHead
from ...utils.layers.embedding.cos_sin_table import CosSinTable
from ...utils.layers.linear.linear_utils import LinearUtils
from ...utils.log import logger, print_log
from ...utils.log.error_code import ErrorCode
from ...utils.adapter_manager import AdapterIdsType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.layers.norm.fast_layer_norm import NormType


CPP_LLAMA_MODEL_CLASS_NAME = "llama_LlamaDecoderModel"

_800_9000_SOCS = (100, 101, 102, 103, 104) # specical SOCS
DUO_SOCS = (200, 201, 202, 203, 204, 205)
A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 254, 255)


class FlashLlamaForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, lmhead_prefix="lm_head", model_prefix="model", **kwargs):
        self.long_seq_enable = config.rope_scaling and getattr(config.rope_scaling, 'rope_type', None) == "llama3"
        original_max_position_embeddings = getattr(config.rope_scaling, "original_max_position_embeddings", None)
        if original_max_position_embeddings is not None and \
            config.max_position_embeddings <= original_max_position_embeddings:
            self.long_seq_enable = False
        super().__init__(config, weights, **kwargs)
        self.mc2_enable = False
        self.soc_info.matmul_nd_nz = (self.soc_info.soc_version == 225 or self.soc_info.soc_version == 223) \
            and not self.mc2_enable and ((config.quantize is None) or (config.quantize == QuantType.FLOAT))
        LinearUtils.soc_info = self.soc_info

        self.model = FlashLlamaModel(config, weights, model_prefix, attn_decode_backend=self.attn_decode_backend)
        if self.quantize == "w8a8sc":
            self.lm_head = TensorHead.load_weight(
                config,
                prefix=lmhead_prefix,
                weights=weights,
                is_norm=False,
            )
        elif config.tie_word_embeddings:
            self.lm_head = TensorParallelHead.load(
                config,
                prefix="model.embed_tokens",
                weights=weights,
                is_norm=True,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=[lmhead_prefix],
                weights=weights,
                head_size=1,
                lm_head=True,
            )

        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.position_embedding_type = config.pe_type
        self.skip_word_embedding = False
        if self.position_embedding_type != "ROPE":
            error_msg = "`pe_type` is only support for type: `ROPE`, loaded from config.json -> pe_type."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise AssertionError(error_msg)
        self.cos_embed = None
        self.sin_embed = None
        self.decode_weight = None

        self.cos_sin_table_params = CosSinTable()
        self.cos_sin_table_params.rope_keep_local_base_windows = config.rope_keep_local_base_windows
        self.cos_sin_table_params.rope_vanilla_theta = config.rope_vanilla_theta
        self.cos_sin_table_params.rope_mscale = config.rope_mscale
        self.cos_sin_table_params.rope_given_inv_feq_str = config.rope_given_inv_feq_str
        self.cos_sin_table_params.rope_theta = self.rope_theta

        self.warmup = True

        if self.mapping.has_pp():
            self.num_hidden_layers = len(self.mapping.pp_layers(self.config.num_hidden_layers))
        else:
            self.num_hidden_layers = self.config.num_hidden_layers
        
        self.graph_manager = ATBGraphManager()
        self.flash_comm_modifier = FlashCommModifier(weights, self.hidden_size, self.flash_comm_gate(weights))
        self.qlen_modifier = QLenModifier()
        self.lora_modifier = LoraModifier(weights, self, lora_adapter=kwargs.get("lora_adapter"), \
            lora_model_config=kwargs.get("lora_model_config"))
        if self.long_seq_enable:
            self.long_seq_modifier = LongSeqModifier(self.config)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: LlamaConfig):
        pass

    def get_weights(self, quantize_type: QuantType = None):
        quantize_type = self.quantize if quantize_type is None else quantize_type
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, attn_wrapper, mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_hidden_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, quantize_type)
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.fa_quant_type is not None:
                weight_wrapper.register_layer_qkvquant(layer)
            if self.config.quantization_config.reduce_quant_type is not None:
                weight_wrapper.register_layer_reducequant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        if self.quantize == QuantType.W8A8_PDMIX:
            weight_wrapper = self.get_weights(quantize_type=QuantType.W8A8_DYNAMIC)
            decode_weight_wrapper = self.get_weights(quantize_type=QuantType.W8A8)
            self.decode_weight = decode_weight_wrapper.weights
        else:
            weight_wrapper = self.get_weights()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_descs_configs = weight_wrapper.linear_descs
        linear_transpose_types = weight_wrapper.linear_transpose_types

        if self.quantize == QuantType.W8A8_PDMIX:
            linear_descs_configs = [[linear_desc if linear_desc != LinearTypeV2.W8A8_PDMIX else
                LinearTypeV2.W8A8_DYNAMIC for linear_desc in linear_descs] for linear_descs in linear_descs_configs]
            decode_linear_descs_configs = [[linear_desc if linear_desc != LinearTypeV2.W8A8_DYNAMIC else
                LinearTypeV2.W8A8 for linear_desc in linear_descs] for linear_descs in linear_descs_configs]
        else:
            decode_linear_descs_configs = linear_descs_configs
        
        # 设置模型参数
        rank_table_file = ENV.rank_table_file

        if self.position_embedding_type == "ROPE":
            position_embedding_type = PositionEmbeddingType.ROPE
        else:
            logger.error("error: `pe_type` is only support for type: `ROPE`, \
                             loaded from config.json -> pe_type.", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "skipWordEmbedding": False,
            "isUnpadInputs": True,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": self.model.parallel_embedding,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableKvQuant": self.config.quantization_config.kv_quant_type is not None,
            "enableFA3": self.config.quantization_config.fa_quant_type is not None,
            "enableReduceQuant": self.config.quantization_config.reduce_quant_type is not None,
            "attnBackend": self.attn_decode_backend,
            "rank": self.mapping.rank,
            "worldSize": self.mapping.world_size,
            "backend": self.soc_info.communication_backend,
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": position_embedding_type,
            "enableAddNorm": False,
            "enableCompressHead": self.compress_head_enable,
            "enableOmniAttention": self.omni_attention_enable,
            "quantGroupSize": self.config.quantization_config.group_size,
            "isLongSeq": self.long_seq_enable,
            "hasPp": self.mapping.has_pp(),
            "ppGroupSize": self.mapping.pp.group_size,
            "firstPpRank": self.mapping.is_first_pp_rank(),
            "lastPpRank": self.mapping.is_last_pp_rank(),
            "prevPpRank": self.mapping.prev_pp_rank(),
            "nextPpRank": self.mapping.next_pp_rank(),
            "tpRank": self.tp_rank,
            "tpWorldSize": self.tp_world_size,
            "tpRankRoot": self.mapping.pp.tp.rank_per_group[self.mapping.pp.rank][0],
            "tpDomain": 'tp_' + '_'.join([str(_) for _ in self.mapping.pp.tp.rank_per_group[self.mapping.pp.rank]]),
        }
        if self.config.model_type == "zhinao":
            coder_param.update({"linearHasBias": [[True, False, False, False]] * self.num_hidden_layers})
            coder_param.update({"splitWithStride": True})
        encoder_param = {
            **coder_param, "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "skipWordEmbedding": self.skip_word_embedding,
            "enableMC2": ENV.enable_mc2,
            "linearDescs": linear_descs_configs,
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "enableLcoc": False,
            "linearDescs": decode_linear_descs_configs
        }
        if self.adapter_manager is not None:
            self.graph_manager.register_graph(MultiLoraGraphWrapper())
            self.graph_manager.register_graph(SingleLoraGraphWrapper())

        if self.prefix_cache_enable:
            self.graph_manager.register_graph(SplitFuseGraphWrapper())
        
        if self.speculate_enable:
            self.graph_manager.register_graph(SpeculateGraphWrapper())
        
        if self.flash_comm_modifier.enable_flash_comm:
            self.graph_manager.register_graph(FlashCommGraphWrapper())

        specified_params = {"decode": decoder_param}
        specified_weight = {"decode": self.decode_weight}
        self.graph_manager.set_param(CPP_LLAMA_MODEL_CLASS_NAME, encoder_param, specified_params)
        self.graph_manager.set_weight(self.ascend_weight, specified_weight)

    def init_cos_sin_table(self, max_seq_len, dim, dtype, device):
        if self.cos_sin_table_params.rope_given_inv_feq_str is None \
            and self.cos_sin_table_params.rope_vanilla_theta is None:
            self._init_rope_cos_sin(max_seq_len, dtype, device)
        else:
            self.cos_sin_table_params.dim = dim
            self.cos_sin_table_params.offset = 0
            self.cos_embed, self.sin_embed = self._get_cos_sin_table(
                max_seq_len, dtype, device, self.cos_sin_table_params
                )

    def prepare_inputs_for_ascend(
            self, input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
            **kwargs
    ):
        n_head = self.num_attention_heads
        q_lens = kwargs.get('q_lens', [])
        attention_mask = kwargs.get('attn_mask', None)
        if attention_mask is None:
            if is_prefill:
                attention_mask = self.attn_mask.get_rope_prefill_mask(self.max_base_len, self.dtype, self.device)
            else:
                attention_mask = self.attn_mask.get_rope_decode_mask(self.dtype, self.device)
        if self.soc_info.need_nz:
            attention_mask = self.transdata_operation.execute([attention_mask])[0]

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                            dtype=torch.int64, device=input_ids.device)

        if self.long_seq_enable:
            if max_seq_len is not None and max_seq_len > self.config.max_position_embeddings:
                seq_len = max_seq_len
            else:
                seq_len = self.config.max_position_embeddings
            self.rotary_embedding.dynamic_ntk_rotary_embedding(self.config, seq_len, self.device)

        if self.cos_embed is None and self.sin_embed is None:
            if self.position_embedding_type == "ROPE":
                self.init_cos_sin_table(self.max_position_embeddings, self.head_dim, self.dtype, self.device)
            else:
                logger.error("error: `pe_type` is only support for type: `ROPE`, \
                             loaded from config.json -> pe_type.", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                
        self.acl_param = {
            "seqLen": input_lengths.tolist()
        }

        # self.acl_operation_inputs ordered by void LlamaDecoderModel::ConstructInTensorMap()
        # in examples/atb_models/atb_framework/models/llama/model/decoder_model.cpp
        acl_operation_inputs = [
            self.placeholder if self.skip_word_embedding else input_ids, # default.input_ids
            input_ids if self.skip_word_embedding else self.placeholder, # default.input_embedding
            position_ids.to(torch.int32), # default.positional_ids
            self.cos_embed, # default.cosine_table
            self.sin_embed, # default.sine_table
            attention_mask, # default.attention_mask
            block_tables.to(torch.int32), # default.block_tables
            slots.to(torch.int32), # default.slots
            self.placeholder, # default.kv_cache_idx
            self.placeholder, # default.token_offset
            self.placeholder, # default.place_holder
            input_lengths.to(torch.int32), # default.seq_len
            lm_head_indices.to(torch.int64) if is_prefill else self.placeholder, # default.logits_indices
        ] # 0~12 default inputs
        
        self.qlen_modifier.modify_inputs(
            acl_operation_inputs,
            self.acl_param,
            input_ids.device,
            is_prefill=is_prefill,
            enable_prefill_pa=False if self.inference_mode is None else self.inference_mode.enable_prefill_pa,
            enable_splitfuse_pa=not self.soc_info.is_300i(),
            **kwargs
        ) # 13 q_len

        self.lora_modifier.modify_inputs(
            acl_operation_inputs,
            kwargs.get("adapter_ids"),
            input_lengths,
            is_prefill
        ) # 14~x lora
        
        if self.long_seq_enable:
            self.long_seq_modifier.modify_inputs(
                acl_operation_inputs,
                pos_embed=self.rotary_embedding,
                placeholder=self.placeholder
            ) # x~-1 long_seq

        self.acl_operation_inputs = acl_operation_inputs
        self.acl_param = json.dumps(self.acl_param)
        return self.acl_operation_inputs, self.acl_param
    
    # Static condition check for FlashComm enablement, called during model initialization
    def flash_comm_gate(self, weights):
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

    def execute_ascend_operator(self,
                                acl_inputs,
                                acl_param,
                                is_prefill):
        acl_model_out = self.graph_manager.select_and_execute(self, acl_inputs, acl_param, \
            is_prefill=is_prefill, enable_dap=True)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            logger.error("运行时报错，请开启日志进一步定位问题", 
                         ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        if self.warmup and self.long_seq_enable:
            self.rotary_embedding.set_ntk_cache(self.config.max_position_embeddings, 
                                                self.rotary_embedding.inv_freq, self.device)
            self.warmup = False
        return acl_hidden_state

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
            self.graph_manager.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def _init_rope_cos_sin(self, max_seq_len, dtype, device):
        if self.config.rope_scaling is None:
            self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                             device,
                                                             max_seq_len)

        else:
            scaling_type = self.config.rope_scaling.rope_type
            if scaling_type is None:
                scaling_type = self.config.rope_scaling.type
            if scaling_type == "linear":
                self.rotary_embedding.update_cos_sin_cache_total(dtype,
                                                                 device,
                                                                 max_seq_len)
            elif scaling_type == "llama3":
                self.rotary_embedding.update_llama3_cos_sin_cache_total(self.config,
                                                                        dtype,
                                                                        device,
                                                                        max_seq_len)
            elif scaling_type == "dynamic":
                if self.warmup:
                    print_log(self.tp_rank, logger.info,
                              f"LLaMA no longer enables long sequence support via"
                              " the ENV.long_seq_enable environment variable. "
                              " Long sequence mode is now controlled by rope_scaling.rope_type"
                              " and is active only when rope_type == 'llama3'.")
            else:
                logger.error("Error: only support scaling type: linear, dynamic, check your config.json: scaling type", 
                             ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
                raise ValueError("Unknown RoPE scaling type, check your config.json: rope_scaling type")

        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()