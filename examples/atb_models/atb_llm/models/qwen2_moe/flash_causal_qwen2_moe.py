# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import os
import math
from typing import Optional, List, Tuple

import torch
import torch_npu

from atb_llm.utils.log import logger, print_log
from atb_llm.utils.env import ENV
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper, MoeWeightWrapper
from atb_llm.models.qwen2_moe.modeling_qwen2_moe import FlashQwenModel
from atb_llm.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from atb_llm.utils.moe_utils import ExpertParallelDegree
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.layers import load_column_multi
from atb_llm.utils.layers.embedding.position_yarn_embedding import PositionYarnEmbedding, _ROPE_SCALING_KEYS
from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.op_backend import OpBackend
from atb_llm.utils.moe_utils import assign, EPLBType, calculate_eplb_param
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils import file_utils
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from ..base.graph_manager import ATBGraphManager, SpeculateGraphWrapper, SplitFuseGraphWrapper
from ..base.inputs_modifier.qlen_modifier import QLenModifier

PREFILL = "prefill"
DECODE = "decode"
CPP_QWEN_MOE_MODEL_CLASS_NAME = "qwen_MoeDecoderModel"
A2_SOCS = (220, 221, 222, 223, 224, 225)
A3_SOCS = (250, 251, 252, 253, 254, 255)


class FlashQwen2moeForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.distributed_enable = kwargs.get('distributed_enable', False)

        self.max_batch_size = kwargs.get('max_batch_size', 0)
        if not self.distributed_enable:
            self.max_batch_size = 0

        super().__init__(config, weights, **kwargs)
        self.mix_shared_routing = False

        self.ep = self.mapping.has_moe_ep()
        self.qwen_moe_config = getattr(self.llm_config.models, "qwen_moe", None)
        self._init_ep_level()

        self.llm_config = self.init_eplb_config(self.llm_config, config, self.ep_level)
        config.ep_level = self.ep_level

        self.model = FlashQwenModel(config, weights, llm_config=self.llm_config,
                                    init_expert_table=self.init_expert_table)
        weights.switch_process_group(ProcessGroupType.LM_HEAD)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.config = config  # for quantize
        self.place_holder = torch.tensor([1], dtype=self.dtype, device='npu')

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)

        self.max_decode_dp_token_size = self.max_batch_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.final_hidden_states = []
        self.expert_array = None
        self.expert_group = torch.tensor([1], dtype=torch.int32).npu()
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.acl_param = None
        self.acl_operation_inputs = []
        self.cos_table = None
        self.sin_table = None
        self.acl_param_encoder = None
        self.acl_param_decoder = None
        self.ascend_weight = None

        # Multi engines management
        self.graph_manager = ATBGraphManager()
        self.qlen_modifier = QLenModifier()

        if hasattr(config, "attn_quantize"):
            self.attn_quantize = config.attn_quantize
        elif self.quantize == "w8a8_dynamic":
            self.attn_quantize = "w8a8"
        else:
            self.attn_quantize = self.quantize

        if self.eplb_level == EPLBType.STATIC_EPLB:
            num_moe_layers = self.num_layers
            self.expert_routing_map = [None] * num_moe_layers
            for i in weights.expert_routing_map.keys():
                self.expert_routing_map[i] = \
                    torch.tensor(weights.expert_routing_map[i], dtype=torch.int32).unsqueeze(0)
            self.expert_routing_map = torch.cat(self.expert_routing_map, dim=0).npu()
            self.device_expert = assign(self.config.num_experts + self.num_redundant_experts,
                                            self.mapping.moe_ep.group_size)[self.mapping.moe_ep.rank]
        elif self.eplb_level == EPLBType.FORCE_EPLB:
            fake_topk = torch.arange(self.num_experts)
            fake_topk = torch.cat([fake_topk[::2], fake_topk[1::2]])
            fake_topk = torch.cat([
                fake_topk[self.mapping.moe_ep.rank * self.num_experts // self.mapping.moe_ep.group_size::1],
                fake_topk[:self.mapping.moe_ep.rank * self.num_experts // self.mapping.moe_ep.group_size:1]
                ])
            self.fake_topk = fake_topk.repeat(512).view(-1, config.num_experts_per_tok).to(torch.int32).npu()
        else:
            self.device_expert = assign(self.config.num_experts, self.mapping.moe_ep.group_size)[
                self.mapping.moe_ep.rank]
        self.start_device_expert_id = torch.tensor(self.device_expert[0], dtype=torch.int64).npu().view(-1)
        self.max_device_expert_id = torch.tensor([len(self.device_expert) - 1], dtype=torch.int64).npu().view(-1)

        self.enable_intra_layer_addnorm = True
        self._update_matmul_params(self.quantize)
        self._init_rope()
        self._init_yarn()

    def init_eplb_config(self, llm_config, config, ep_level=ExpertParallelDegree.NO_EP):
        level = 0
        map_file_path = ""
        num_redundant_experts = 0
        mix_shared_routing = False
        self.init_expert_table = None
        if llm_config is not None:
            level = llm_config.models.qwen_moe.eplb.level
            map_file_path = llm_config.models.qwen_moe.eplb.expert_map_file
            flag_level_unvalid = level not in [e.value for e in EPLBType]
            if flag_level_unvalid:
                msg = "Invalid EPLB configuration. " \
                    "Valid values are NO_EPLB(0), STATIC_EPLB(1), DYNAMIC_EPLB(1), or FORCE_EPLB(3)."
                logger.error(msg)
                raise ValueError(msg)
            if ep_level != ExpertParallelDegree.DYNAMIC_EP and level != EPLBType.NO_EPLB:
                msg = "EPLB only supports EP level 2 (ExpertParallelDegree.DYNAMIC_EP). "
                logger.error(msg)
                raise ValueError(msg)
            if level == EPLBType.STATIC_EPLB and not os.path.isfile(map_file_path):
                msg = "Invalid EPLB file path."
                logger.error(msg)
                raise ValueError(msg)
            if level == EPLBType.STATIC_EPLB:
                mix_shared_routing, _, num_redundant_experts = \
                    calculate_eplb_param(map_file_path, config.num_experts)
            if num_redundant_experts < 0:
                msg = f"Invalid number of redundant experts: {num_redundant_experts}"
                logger.error(msg)
                raise ValueError(msg)

        self.eplb_level = level
        self.eplb_expert_map_file = map_file_path
        self.num_redundant_experts = num_redundant_experts
        self.mix_shared_routing = mix_shared_routing

        if ENV.enable_expert_hotpot_gather:
            EplbExpertDataCollect().set_model_ref(self)
            if (ENV.expert_hotpot_dump_path is not None) and self.mapping.rank == 0:
                num_moe_layers = config.num_hidden_layers

                model_gen_config = {
                    "num_moe_layers": num_moe_layers,
                    "collection_Interval": 8, # No parameter needs to be transferred currently.The default value is 8.
                    "num_of_experts": config.num_experts,
                    "num_of_selected_experts": [config.num_experts_per_tok]
                }
                if level == EPLBType.STATIC_EPLB:
                    model_gen_config["eplb_expert_map_file"] = self.eplb_expert_map_file
                for stage in [PREFILL, DECODE]:
                    hotpot_path = os.path.join(ENV.expert_hotpot_dump_path, stage)
                    os.makedirs(hotpot_path, exist_ok=True)
                    hotpot_config_path = os.path.join(hotpot_path, "model_gen_config.json")
                    with file_utils.safe_open(hotpot_config_path, "w", encoding='utf-8') as json_file:
                        json.dump(model_gen_config, json_file, indent=4)

        logger.info(f"EPLB level is : {self.eplb_level}.")
        logger.info(f"EPLB expert map path is : {self.eplb_expert_map_file}.")
        logger.info(f"EPLB redundant experts is : {self.num_redundant_experts}.")
        logger.info(f"EPLB mix shared routing is : {self.mix_shared_routing}")

        return llm_config

    def register_layer_weights(self, weight_wrapper, layer, is_dense_layer=False):
        layer_dict = layer.state_dict()
        # add input layernorm and self_attn weight
        weight_wrapper.soc_info.matmul_nd_nz = self.matmul_nd_nz
        weight_wrapper.register_moe_layer(layer, self.quantize, dense_layer=is_dense_layer,
                                            attn_quantize_type=self.attn_quantize,
                                            qk_norm=self.config.use_qk_norm)

        if self.config.has_shared_expert:
            shared_experts_layer_names = [
                "mlp.shared_expert.gate_up_proj.linear",
                "mlp.shared_expert.down_proj.linear",
            ]
            for layer_name in shared_experts_layer_names:
                weight_wrapper.weights.append(layer_dict[f"{layer_name}.weight"])
            # add shared experts gate weights
            weight_wrapper.weights.append(layer_dict["mlp.shared_expert_gate.weight"])
        else:
            weight_wrapper.weights.extend([self.place_holder] * 3)

    def get_weights(self):
        attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='c_attn',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='c_proj'
        )
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=False,
        )
        weight_wrapper = MoeWeightWrapper(
            self.soc_info,
            self.tp_rank,
            attn_wrapper,
            moe_mlp_wrapper,
            self.config.num_experts
        )
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            self.register_layer_weights(weight_wrapper, layer)

        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):        
        is_w8a8_dynamic = self.quantize == "w8a8_dynamic"
        ascend_weight_wrapper = self.get_weights()
        self.ascend_weight = ascend_weight_wrapper.weights

        pack_quant_types = ascend_weight_wrapper.pack_quant_type
        attn_linear_types = ascend_weight_wrapper.attn_linear_types
        mlp_linear_types = ascend_weight_wrapper.mlp_linear_types
        moe_linear_types = ascend_weight_wrapper.moe_linear_types
        attn_linear_transpose_types = ascend_weight_wrapper.attn_linear_transpose_types
        mlp_linear_transpose_types = ascend_weight_wrapper.mlp_linear_transpose_types
        moe_linear_transpose_types = ascend_weight_wrapper.moe_linear_transpose_types

        acl_param_dict = {
            "isUnpadInputs": True,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "isEmbeddingParallel": True,
            "isLmHeadParallel": True,
            "attnLinearQuantType": attn_linear_types,
            "mlpLinearQuantType": mlp_linear_types,
            "moeLinearQuantType": moe_linear_types,
            "attnLinearTransposeType": attn_linear_transpose_types,
            "mlpLinearTransposeType": mlp_linear_transpose_types,
            "moeLinearTransposeType": moe_linear_transpose_types,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": True,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "rank": self.tp_rank,
            "worldSize": self.mapping.world_size,
            "backend": self.soc_info.communication_backend,
            "packQuantType": pack_quant_types,
            "expertParallelDegree": self.ep_level,
            "rankTableFile": ENV.rank_table_file,
            "numOfExperts": self.num_experts,
            "numOfSelectedExperts": self.config.num_experts_per_tok,
            "routingMethod": 'softMaxTopK' if self.soc_info.need_nz else 'integratedSoftmaxTopK',
            "enableFusedRouting": True,
            "isDenseLayer": self.config.is_dense_layer,
            "hasSharedExpert": self.config.has_shared_expert,
            "useQKNorm": self.config.use_qk_norm,
            "linearHasBias": [[self.config.attention_bias, False, False, False]] * self.config.num_hidden_layers,
            "processLogits": 'normalization' if self.config.norm_topk_prob else 'none',
            "numOfDeviceExperts": len(self.device_expert),
            "maxDecodeDpTokenSize": self.max_decode_dp_token_size,
            "enableInitQuant": True if (is_w8a8_dynamic and (not self.soc_info.need_nz)) else False,
            "enableIntraLayerAddNorm": self.enable_intra_layer_addnorm,
            "enableAllToAllMC2": self.ep_level == ExpertParallelDegree.DYNAMIC_EP,
            "enableDpOut": ENV.enable_dp_partition_up,
            "enableDispatchCombineV2": True,
            "enableEPWB": self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB],
            "numOfRedundantExpert": self.num_redundant_experts,
            "enableExpertCumSumOutput": ENV.enable_expert_hotpot_gather,
            "enableLoadBalance": self.eplb_level == EPLBType.FORCE_EPLB,
            "ropeBackend": self.rope_backend,
        }

        if self.mapping is not None:
            acl_param_dict.update({"mapping": self.mapping.to_dict_v2()})
        encoder_param = {
            **acl_param_dict,
            "isPrefill": True,
            "enableLcoc": self.lcoc_enable,
            "enableGMMSwigluQuant": False
        }
        decoder_param = {
            **acl_param_dict,
            "isPrefill": False,
            "enableLcoc": False,
            "enableGMMSwigluQuant": True if (is_w8a8_dynamic and (not self.soc_info.need_nz)) else False
        }
        if self.prefix_cache_enable:
            self.graph_manager.register_graph(SplitFuseGraphWrapper())
        if self.speculate_enable:
            self.graph_manager.register_graph(SpeculateGraphWrapper())

        specified_params = {"decode": decoder_param}
        self.graph_manager.set_param(CPP_QWEN_MOE_MODEL_CLASS_NAME, encoder_param, specified_params)
        self.graph_manager.set_weight(self.ascend_weight)
    
    def build_dep_inputs(
            self,
            input_ids: torch.Tensor,
            is_prefill: bool,
            lm_head_indices: Optional[torch.Tensor],
            **kwargs
    ):
        """Build inputs for ep."""
        attn_padding_idx = self.placeholder
        attn_unpadding_idx = self.placeholder
        ffn_padding_idx = self.placeholder
        ffn_unpadding_idx = self.placeholder
        lm_head_skip_padding_token_indices = self.placeholder
        gather_prenorm_idx = self.placeholder
        dynamic_ep_idx = self.placeholder
        moe_idx = self.placeholder
        
        dep_inputs = [attn_padding_idx, attn_unpadding_idx, ffn_padding_idx,
            ffn_unpadding_idx, lm_head_skip_padding_token_indices, gather_prenorm_idx,
            self.start_device_expert_id, self.max_device_expert_id, dynamic_ep_idx, moe_idx, self.placeholder]

        if self.mapping.has_dp():
            if ENV.enable_dp_move_up:
                has_tp = self.mapping.has_attn_tp() or self.mapping.lm_head_tp.group_size > 1
                if not self.distributed_enable or (has_tp and self.distributed_enable):
                    dep_inputs = kwargs.get("dep_inputs", None)
                    dep_inputs = dep_inputs[:6] + \
                                    [self.start_device_expert_id, self.max_device_expert_id] + dep_inputs[6:]
                moe_idx = dep_inputs[-2] # note that moe_idx is the second last tensor in dep_inputs
                self.expert_array = torch.ones(moe_idx.shape[0], dtype=torch.float16).npu().view(-1, 1)
            else:
                message = "Please export DP_MOVE_UP_ENABLE=1 when set attn_dp > 1."
                logger.error(message)
                raise RuntimeError(message)
        else:
            if self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
                message = "Currently do not support ep_level=2 when attn_dp=1."
                logger.error(message)
                raise NotImplementedError(message)
            input_length = len(input_ids)
            self.expert_array = torch.tensor(
                            [j for j in range(input_length * self.config.num_experts_per_tok)],
                            dtype=torch.int32
                            ).npu().view(-1)
            
        if self.eplb_level == EPLBType.STATIC_EPLB:
            dep_inputs.append(self.expert_routing_map)
        
        return dep_inputs
    
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
        dep_inputs = self.build_dep_inputs(
            input_ids=input_ids,
            is_prefill=is_prefill,
            lm_head_indices=lm_head_indices,
            **kwargs
            )
        
        # rope
        self.rotary_embedding.update_cos_sin_cache_total(
            self.dtype,
            self.device,
            self.max_position_embeddings
        )
        self.cos_table = self.rotary_embedding.get_cos_cached_total()
        self.sin_table = self.rotary_embedding.get_sin_cached_total()

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

        acl_operation_inputs_ = [
            input_ids,  # IN_TENSOR_INPUTIDS
            position_ids,  # IN_TENSOR_POSITIONIDS
            self.cos_table,  # IN_TENSOR_COSEMBED
            self.sin_table,  # IN_TENSOR_SINEMBED
            attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            self.place_holder,  # IN_TENSOR_KV_CACHE_IDX
            self.place_holder,  # IN_TENSOR_TOKEN_OFFSET
            self.place_holder,
            input_lengths.to(torch.int32),  # IN_TENSOR_SEQ_LENGTHS
            lm_head_indices if (is_prefill or (self.mapping.has_dp() and not ENV.enable_dp_partition_up)) 
                            else self.lm_head_indices_fake,  # IN_TENSOR_LOGTIS_INDICES 11
            self.expert_array,
            self.expert_group,
            self.one_hot,
            self.zero_hot
        ]
        # inputs for ep
        acl_operation_inputs_.extend(dep_inputs)

        if self.eplb_level == EPLBType.FORCE_EPLB:
            fake_topk = self.fake_topk[:dep_inputs[1].shape[0] if self.mapping.has_attn_tp() else len(input_ids)]
            acl_operation_inputs_.append(fake_topk)
        
        self.acl_param = {
            "seqLen": input_lengths.tolist()
        }
        self.qlen_modifier.modify_inputs(
            acl_operation_inputs_,
            self.acl_param,
            input_ids.device,
            is_prefill=is_prefill,
            enable_prefill_pa=False if self.inference_mode is None else self.inference_mode.enable_prefill_pa,
            enable_splitfuse_pa=not self.soc_info.is_300i(),
            **kwargs)
        
        self.acl_operation_inputs = acl_operation_inputs_
        self.acl_param = json.dumps(self.acl_param)

        return self.acl_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs: list,
                                acl_param: str,
                                is_prefill: bool) -> torch.Tensor:
        """Execute the Ascend acl operator."""
        acl_model_out = self.graph_manager.select_and_execute(self, acl_inputs, acl_param, is_prefill=is_prefill)
        try:
            acl_hidden_state = acl_model_out[0]
        except IndexError as e:
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        return acl_hidden_state
    
    def init_kvcache(self, kv_cache: List[Tuple[torch.Tensor]]):
        """Initialzie key-value cache."""
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
            
    def select_logits(self, logits, **kwargs):
        dp_logits_num = kwargs.get("dp_logits_num", None)
        if dp_logits_num is None:
            return logits
        dp_rank_id = self.mapping.attn_dp.rank
        if dp_rank_id == 0:
            logits = logits[:dp_logits_num[dp_rank_id]]
        else:
            logits = logits[dp_logits_num[dp_rank_id - 1]: dp_logits_num[dp_rank_id]]
        return logits

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
        if not self.ascend_weight:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)    
        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        if kwargs.get("is_prefill") and self.distributed_enable:
            logits = self.select_logits(logits, **kwargs)
        return logits

    def _init_yarn(self):
        if self.config.rope_scaling_dict is not None:
            scaling_type = self.config.rope_scaling_dict["type"]

            if scaling_type == "yarn":
                scaling_factor = self.config.rope_scaling_dict["factor"]
                logger.info("Qwen-moe: enable yarn position embedding.")
                kwargs = {
                    key: self.config.rope_scaling_dict[key]
                    for key in _ROPE_SCALING_KEYS
                    if key in self.config.rope_scaling_dict
                }
                yarn_kwargs = PositionYarnEmbedding.StaticInputArgs(
                                            max_position_embeddings=self.max_position_embeddings,
                                            scaling_factor=scaling_factor,
                                            **kwargs,)
                self.rotary_embedding = PositionYarnEmbedding.static_yarn(dim=self.head_size,
                                                                             base=self.rope_theta,
                                                                             device="cpu",
                                                                             yarn_kwargs=yarn_kwargs).to(self.device)
            else:
                msg = f"Unknown RoPE scaling type {scaling_type}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)

    def _update_matmul_params(self, quantize: QuantType):
        if self.soc_info.soc_version in (A2_SOCS + A3_SOCS):
            self.matmul_nd_nz = True
            is_float = quantize is None or quantize == QuantType.FLOAT
            if self.matmul_nd_nz and is_float:
                self.matmul_nd_nz = False
                logger.info("Qwen-moe: Turn off matmul_nd_nz when quantize is float.")
        else:
            self.matmul_nd_nz = False
        logger.info(f"Qwen-moe: matmul_nd_nz is: {self.matmul_nd_nz}")

    def _init_ep_level(self):
        if self.ep:
            self.ep_level = getattr(self.qwen_moe_config, "ep_level", ExpertParallelDegree.DYNAMIC_EP)
        else:
            self.ep_level = ExpertParallelDegree.NO_EP

        if self.ep_level == ExpertParallelDegree.DYNAMIC_EP and self.mapping.rank_table_file == "":
            msg = "Qwen-moe: Please set RANK_TABLE_FILE when ep_level is equal to 2."
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise RuntimeError(msg)

        if self.ep_level == ExpertParallelDegree.DYNAMIC_EP and self.mapping.moe_tp.group_size != 1:
            msg = f"When ep_level is 2, moe_tp only supports a value of 1. \
                    Current moe_tp: {self.mapping.moe_tp.group_size}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(msg)
        
        logger.info(f"Qwen-moe: ExpertParallelDegree is: {self.ep_level}")

    def _init_rope(self):
        enable_aclnn_rope = getattr(self.qwen_moe_config, 'enable_aclnn_rope', False)
        self.rope_backend = OpBackend.ACLNN if enable_aclnn_rope else OpBackend.ATB
        logger.info(f"Qwen-moe: enable_aclnn_rope is: {enable_aclnn_rope}")