# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
from functools import partial
import os
import json
import math
from enum import Enum
import queue
from typing import List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import torch
import torch_npu

from atb_llm.models.base.flash_causal_lm import FlashForCausalLM, DistributedType, LwdLayerStatus
from atb_llm.models.base.model_utils import get_leaf_modules_recursive, get_module_quant_type
from atb_llm.models.deepseekv2.modeling_deepseekv2 import FlashDeepseekV2Model
from atb_llm.models.deepseekv2.config_deepseekv2 import DeepseekV2Config
from atb_llm.utils.layers.attention.process_mla_linear import transdata_3d
from atb_llm.models.deepseekv2.weight_wrapper_deepseekv2 import Deepseekv2WeightWrapper
from atb_llm.utils.data.moe_weight_wrapper import MoeMlpWrapper
from atb_llm.utils.data.weight_wrapper import AttnWrapper
from atb_llm.utils.env import ENV
from atb_llm.utils.layers import PositionRotaryEmbedding
from atb_llm.utils.layers import load_column_multi, TensorHead
from atb_llm.utils.layers.embedding.position_yarn_embedding import PositionYarnEmbedding, _ROPE_SCALING_KEYS

from atb_llm.utils.layers.norm.fast_layer_norm import NormType
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.moe_utils import assign, EPLBType, ExpertParallelDegree, parse_ep_file, random_generation, calculate_eplb_param
from atb_llm.utils.weights import ProcessGroupType
from atb_llm.utils.log import print_log
from atb_llm.utils import file_utils
from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from ...models import InferenceMode

try:
    from atb_llm.utils.prof.profiler import span_start, span_end, tensor_attr, span_attr, Level
except ImportError:
    # Define dummy functions if the module is not found
    def span_start(*args, **kwargs):
        # Maybe add a log warning here if needed
        return None # Or some dummy context manager

    def span_end(*args, **kwargs):
        pass

    def span_attr(*args, **kwargs):
        return None


CPP_DEEPSEEKV2_MODEL_CLASS_NAME = "deepseekV2_DecoderModel"
CPP_DEEPSEEKV2_MTP_MODEL_CLASS_NAME = "deepseekV2_MtpDecoderModel"
SUPPORT_LCOC = "supportLcoc"
BACKEND = "backend"
NUM_HIDDEN_LAYERS = 61
SEQUENCE_LENGTH = "seqLen"
SEQUENCE_LENGTH_SP = "seqLenSp"
SEQUENCE_LENGTH_CP = "seqLenCp"
Q_LEN = "qLen"
DECODER = "decoder"
Q_LENS = "q_lens"
PREFILL = "prefill"
DECODE = "decode"
START_ID = "startLayerId"
END_ID = "endLayerId"
KVCACHE_QUANT_LAYERS = "kvcacheQuantLayers"
MOE_PACK_QUANT_TYPE = "moePackQuantType"
ALLTOALL_LONG_SEQLEN_THRESHOLD = 65536
MAX_ALLTOALL_BUFF_SCALE = 3


class MaskType(int, Enum):
    NO_MASK = 0
    SPEC_MASK = 1
    FREE_MASK = 2


class FlashDeepseekv2ForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights, **kwargs):
        self.num_speculative_tokens = kwargs.get('num_speculative_tokens', int(ENV.deepseek_mtp))
        self.inference_mode = kwargs.get('inference_mode', None)
        if self.num_speculative_tokens:
            if not self.inference_mode and not hasattr(self.inference_mode, 'enable_decode_pa'):
                kwargs['inference_mode'] = InferenceMode.SPECULATE 
            if 'maskType' not in kwargs:
                kwargs['maskType'] = MaskType.FREE_MASK
        self.distributed_enable = kwargs.get('distributed_enable', False)
        self.max_batch_size = kwargs.get('max_batch_size', 0)
        self.total_batch_size = self.max_batch_size
        if not self.distributed_enable:
            self.max_batch_size = 0
        self.acl_encoder_operation_mtp = None
        self.acl_decoder_operation_mtp = None
        self.acl_dap_operation = None
        self.acl_dap_operation_mtp = None
        self.has_prefixcache = False
        kwargs['obfCoefficient'] = 0.5
        super().__init__(config, weights, **kwargs)
        self.maskfree = kwargs.get('maskType', MaskType.NO_MASK) == MaskType.FREE_MASK
        self.model_role = kwargs.get("model_role", "standard")

        if self.llm_config is not None:
            self.ds_config = self.llm_config.models.deepseekv2
            self.parallel_config = self.llm_config.llm.parallel_options
            self.enable_o_proj_local_tp = self.mapping.enable_o_proj_local_tp
            self.enable_lm_head_local_tp = self.mapping.enable_lm_head_local_tp
            self.enable_nz = self.llm_config.llm.kv_cache_options.enable_nz
        else:
            self.ds_config = None
            self.parallel_config = None
            self.enable_o_proj_local_tp = False
            self.enable_lm_head_local_tp = False
            self.enable_nz = False
        if self.prefix_cache_enable and not self.enable_nz:
            msg = "The prefix cache is enabled, but the nz cache is not enabled. Please check the configuration."
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(msg)
        self.ep = self.mapping.has_moe_ep()
        if self.ep:
            if not hasattr(self.ds_config, "ep_level"):
                logger.info("For expert parallel, "
                    "the ep_level variable needs to be defined in the model configuration file."
                    "The available options are 1, 2, or 3.")
                self.ep_level = ExpertParallelDegree.STATIC_EP
            else:
                self.ep_level = self.ds_config.ep_level
        else:
            self.ep_level = ExpertParallelDegree.NO_EP
        self.enable_dispatch_combine_v2 = False if self.ds_config is None else \
                                          self.ds_config.enable_dispatch_combine_v2
        self.enable_oproj_prefetch = False if self.ds_config is None else self.ds_config.enable_oproj_prefetch
        self.enable_mlapo_prefetch = False if self.ds_config is None else self.ds_config.enable_mlapo_prefetch
        self.enable_einsum_nz = (self.model_role == DECODER)

        config.tp_num_attention_heads = self.num_attention_heads
        config.tp_num_key_value_heads = self.num_key_value_heads
        config.num_speculative_tokens = self.num_speculative_tokens

        self.aggregate_threshold = None
        self.buffer_expert_layer_num = None
        self.num_expert_update_ready_countdown = None
        self.acl_all_gather_operation = None
        self.ascend_buffer_weight = []
        self.llm_config = self.init_eplb_config(self.llm_config, config)
        if self.mix_shared_routing and self.eplb_level == EPLBType.NO_EPLB:
            num_moe_layers = self.num_layers - config.first_k_dense_replace
            num_moe_layers += (1 if self.num_speculative_tokens > 0 else 0)
            self.init_expert_table = random_generation(
                num_moe_layers,
                config.n_routed_experts, 
                self.mapping.moe_ep.group_size,
                self.num_redundant_experts,
                mix_shared_routing=self.mix_shared_routing
                )
            self.eplb_level = EPLBType.STATIC_EPLB
            self.llm_config.models.deepseekv2.eplb.level = EPLBType.STATIC_EPLB
            self.num_redundant_experts += self.mapping.moe_ep.group_size
        self.llm_config = self.init_h3p_config(self.llm_config, self.ep_level)
        self.enable_gmmswigluquant = (self.quantize == "w8a8_dynamic" and self.mapping.world_size > 16) or \
                                            getattr(self.ds_config, "enable_gmmswigluquant", False)
        self.is_nzcasted = self.config.is_nzcasted
        enable_atlas_gmm_worldsize = 16 if self.is_nzcasted else 32
        enable_gmm_fused = self.model_role == "decoder" and self.mapping.world_size > enable_atlas_gmm_worldsize
        self.enable_atlas_gmm_fused = True if self.enable_gmmswigluquant and enable_gmm_fused \
                                            else False
        if self.eplb_level == EPLBType.DYNAMIC_EPLB:
            self.enable_gmmswigluquant = True
            self.enable_atlas_gmm_fused = True
            self.enable_lcoc_all2all = False
        if self.llm_config is not None:
            self.llm_config.enable_atlas_gmm_fused = self.enable_atlas_gmm_fused
        config.ep_level = self.ep_level
        config.eplb_level = self.eplb_level
        if hasattr(self.ds_config, 'alltoall_ep_buffer_scale_factors'):
            config.alltoall_ep_buffer_scale_factors = self.ds_config.alltoall_ep_buffer_scale_factors
        
        if self.layerwise_disaggregated:
            self.prefix_cache_enable = True
            self.layerwise.load_list = []
            if self.layerwise.split_type == DistributedType.CLOUD:
                start_layer = self.layerwise.edge_start_layer_count
                end_layer = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
                self.layerwise.load_list = list(range(start_layer, end_layer))
            else:
                self.layerwise.load_list = [i for i in range(0, self.layerwise.edge_start_layer_count)]
                start_layers = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
                other_load_list = [i for i in range(start_layers, self.config.num_hidden_layers)]
                self.layerwise.load_list.extend(other_load_list)
            self.model = FlashDeepseekV2Model(
                config, weights, llm_config=self.llm_config, init_expert_table=self.init_expert_table,
                mix_shared_routing=self.mix_shared_routing, layerwise_disaggregated=self.layerwise_disaggregated,
                load_list=self.layerwise.load_list
            )
            self.layerwise.weight_wrappers = []
            self.layerwise.ascend_weight_head = None
            self.layerwise.ascend_weight_tail = None
            self.layerwise.ascend_weight_internal = []
            self.layerwise.acl_inputs_prefill = None
            self.layerwise.acl_inputs_decode = None
            self.layerwise.acl_inputs_prefill_queue = queue.Queue()
            self.layerwise.acl_param_prefill = None
            self.layerwise.acl_param_decode = None
            self.layerwise.acl_param_prefill_queue = queue.Queue()
            self.layerwise.p_out_hidden = None
            self.layerwise.acl_inputs_prefill_pre = None
            self.layerwise.acl_param_prefill_pre = None
        else:
            self.model = FlashDeepseekV2Model(
                config, weights, llm_config=self.llm_config, init_expert_table=self.init_expert_table,
                mix_shared_routing=self.mix_shared_routing
            )
        if not self.enable_atlas_gmm_fused:
            self.is_nzcasted = False
        self.kvcache_quant_layers = self.model.kvcache_quant_layers
        if ENV.enable_dp_partition_up:
            weights.switch_process_group(ProcessGroupType.LM_HEAD)
        else:
            weights.switch_process_group(ProcessGroupType.MLP)
        if weights.sharded:
            self.lm_head = TensorHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=False,
            )
        else:
            self.lm_head = load_column_multi(
                config,
                prefixes=["lm_head"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        
        self.config = config
        self.weights = weights
        self.acl_encoder_operation_inputs = []
        self.acl_decoder_operation_inputs = []
        self.max_decode_dp_token_size = self.max_batch_size * (self.num_speculative_tokens + 1)

        self.placeholder = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device=self.device)

        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)
        self.padding_idx = config.pad_token_id

        if hasattr(config, "mla_quantize"):
            self.mla_quantize = config.mla_quantize
        else:
            self.mla_quantize = self.quantize
        self.moe_quantize = getattr(config, "moe_quantize", self.quantize)
        self.hidden_dim = config.hidden_size
        self.final_hidden_states = []
        self.expert_array = []

        self.expert_group = torch.arange(1024, dtype=torch.int32).npu() # 1024: const for groupedTopK
        self.routed_scaling_factor = config.routed_scaling_factor
        self.one_hot = torch.tensor([1], dtype=torch.int32).npu()
        self.zero_hot = torch.tensor([0], dtype=torch.int32).npu()
        self.final_bias = torch.zeros([self.config.n_routed_experts, self.config.hidden_size], dtype=self.dtype).npu()

        self.num_of_experts = config.n_routed_experts
        self.num_of_selected_experts = [config.num_experts_per_tok]
        self.enable_init_routing_cutoff = (
            self.ep_level == ExpertParallelDegree.STATIC_EP and
            self.ds_config.enable_init_routing_cutoff if self.ds_config else False
        )
        self.topk_scaling_factor = self.ds_config.topk_scaling_factor if self.ds_config else 1.0
        if self.enable_init_routing_cutoff:
            if not (0.25 <= self.topk_scaling_factor <= 1.0):
                msg = f"Enable the token truncation function for STATIC_EP, \
                topk_scaling_factor must be in range [0.25, 1.0], but got {self.topk_scaling_factor}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)

        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.topk_method = config.topk_method
        self.tp = config.tp if config.tp else True # Defaulting the model to tensor parallel
        self.first_k_dense_replace = config.first_k_dense_replace if config.first_k_dense_replace else 0
        self.n_shared_experts = config.n_shared_experts if config.n_shared_experts else 0
        self.norm_topk_prob = config.norm_topk_prob if config.norm_topk_prob else False
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        if self.maskfree:
            self.atten_mask_free = self.gen_mask(self.num_speculative_tokens + 1, self.dtype).npu()

        self.softmax_scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** (-0.5)
        factor_name = "factor"
        if self.config.rope_scaling_dict is not None:
            mscale_all_dim = self.config.rope_scaling_dict.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling_dict[factor_name]
            if mscale_all_dim:
                mscale = PositionYarnEmbedding.yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        if self.config.rope_scaling_dict is None:
            if hasattr(config, 'rope_scaling') and self.config.rope_scaling_dict is not None:
                self.scaling_factor = self.config.rope_scaling_dict.get(factor_name, 1.0)
            else:
                self.scaling_factor = 1.0
            self.rotary_embedding = PositionRotaryEmbedding.static(
                dim=self.qk_rope_head_dim,
                base=self.rope_theta,
                device="cpu",
                scaling_factor=self.scaling_factor
            ).to(self.device)
        else:
            self.scaling_type = config.rope_scaling_dict["type"]
            self.scaling_factor = config.rope_scaling_dict["factor"]
            if self.scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling_dict[key]
                    for key in _ROPE_SCALING_KEYS
                    if key in self.config.rope_scaling_dict
                }
                yarn_kwargs = PositionYarnEmbedding.StaticInputArgs(
                                            max_position_embeddings=self.max_position_embeddings,
                                            scaling_factor=scaling_factor,
                                            **kwargs,)
                self.rotary_embedding = PositionYarnEmbedding.static_yarn(dim=self.qk_rope_head_dim,
                                                                             base=self.rope_theta,
                                                                             device="cpu",
                                                                             yarn_kwargs=yarn_kwargs).to(self.device)
            else:
                msg = f"Unknown RoPE scaling type {self.scaling_type}"
                logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
                raise ValueError(msg)
        self.mask_start_idx = 0

        self.communication_backend = self.soc_info.communication_backend
        self.p_to_d_weight = False
        self.hotpot_save_count = 0
        if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
            num_moe_layers = self.num_layers - self.first_k_dense_replace
            num_moe_layers += (1 if self.num_speculative_tokens > 0 else 0)
            self.expert_routing_map = [None] * num_moe_layers
            for i in weights.expert_routing_map.keys():
                self.expert_routing_map[i - self.first_k_dense_replace] = \
                    torch.tensor(weights.expert_routing_map[i], dtype=torch.int32).unsqueeze(0)
            self.expert_routing_map = torch.cat(self.expert_routing_map, dim=0).npu()

        self.device_expert = ([[0]] * self.num_dangling_shared_experts + assign(
            self.config.n_routed_experts + self.num_redundant_experts,
            self.mapping.moe_ep.group_size - self.num_dangling_shared_experts
        ))[self.mapping.moe_ep.rank]
        
        if self.ep:
            if self.ep_level in [ExpertParallelDegree.DYNAMIC_EP, ExpertParallelDegree.MIX_EP]:
                self.dep_communication_backend = {'prefill': 'hccl', 'decode': 'hccl'} if self.ds_config is None else {
                    'prefill': self.ds_config.communication_backend.prefill, 
                    'decode': self.ds_config.communication_backend.decode
                }
                self.p_to_d_weight = self.ep_level == ExpertParallelDegree.MIX_EP
            logger.info(f"Expert parallel level is {self.ep_level}.")
            logger.info(f"Experts of rank {self.mapping.moe_ep.rank} are: {self.device_expert}")

        self.num_of_device_expert = len(self.device_expert)
        self.start_device_expert_id = torch.tensor(self.device_expert[0], dtype=torch.int64).npu().view(-1)
        self.max_device_expert_id = torch.tensor([len(self.device_expert) - 1], dtype=torch.int64).npu().view(-1)

        if self.eplb_level == EPLBType.FORCE_EPLB:
            fake_topk = torch.arange(self.num_of_experts)
            fake_topk = torch.cat([fake_topk[::2], fake_topk[1::2]])
            fake_topk = torch.cat([
                fake_topk[self.mapping.moe_ep.rank * self.num_of_experts // self.mapping.moe_ep.group_size::1],
                fake_topk[:self.mapping.moe_ep.rank * self.num_of_experts // self.mapping.moe_ep.group_size:1]
                ])
            self.fake_topk = fake_topk.repeat(512).view(-1, config.num_experts_per_tok).to(torch.int32).npu()

        self.init_padding_idx()

    def init_eplb_config(self, llm_config, config):
        level = 0
        map_file_path = ""
        num_redundant_experts = 0
        num_dangling_shared_experts = 0
        self.init_expert_table = None
        self.mix_shared_routing = getattr(self.ds_config, "mix_shared_routing", False)
        if llm_config is not None:
            level = llm_config.models.deepseekv2.eplb.level
            map_file_path = llm_config.models.deepseekv2.eplb.expert_map_file
            num_dangling_shared_experts = \
                max(getattr(llm_config.models.deepseekv2, "num_dangling_shared_experts", 0), 0)
            flag_level_unvalid = level not in [e.value for e in EPLBType]
            if flag_level_unvalid:
                msg = "Invalid EPLB configuration. " \
                    "Valid values are NO_EPLB(0), STATIC_EPLB(1), DYNAMIC_EPLB(1), or FORCE_EPLB(3)."
                logger.error(msg)
                raise ValueError(msg)
            if level == EPLBType.STATIC_EPLB:
                self.mix_shared_routing, num_dangling_shared_experts, num_redundant_experts = \
                    calculate_eplb_param(map_file_path, config.n_routed_experts)
            if level == EPLBType.DYNAMIC_EPLB:
                self.mix_shared_routing = True
                EplbExpertDataCollect().set_model_ref(self)
                num_redundant_experts = self.init_dynamic_eplb_config(llm_config, config)
            if num_redundant_experts < 0:
                msg = f"Invalid number of redundant experts: {num_redundant_experts}"
                logger.error(msg)
                raise ValueError(msg)
            self.llm_config.models.deepseekv2.num_dangling_shared_experts = num_dangling_shared_experts

        self.eplb_level = level
        self.eplb_expert_map_file = map_file_path
        self.num_redundant_experts = num_redundant_experts
        self.num_dangling_shared_experts = num_dangling_shared_experts
        self.topk_output = ENV.enable_expert_hotpot_gather and (ENV.expert_hotpot_dump_path is not None)

        if ENV.enable_expert_hotpot_gather:
            EplbExpertDataCollect().set_model_ref(self)
            if (ENV.expert_hotpot_dump_path is not None) and self.mapping.rank == 0:
                num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
                if self.num_speculative_tokens:
                    num_moe_layers += 1 # 1: num of mtp layer
                model_gen_config = {
                    "num_moe_layers": num_moe_layers,
                    "collection_Interval": 8, # No parameter needs to be transferred currently.The default value is 8.
                    "num_of_experts": config.n_routed_experts,
                    "num_of_selected_experts": [config.num_experts_per_tok],
                    "num_dangling_shared_experts": num_dangling_shared_experts
                }
                if level == EPLBType.STATIC_EPLB:
                    model_gen_config["eplb_expert_map_file"] = self.eplb_expert_map_file
                for stage in [PREFILL, DECODE]:
                    hotpot_path = os.path.join(ENV.expert_hotpot_dump_path, stage)
                    os.makedirs(hotpot_path, exist_ok=True)
                    for item in os.listdir(hotpot_path):
                        if item.endswith(".csv"):
                            raise RuntimeError("There should be no CSV files in the hotpot dump path.")
                    hotpot_config_path = os.path.join(hotpot_path, "model_gen_config.json")
                    with file_utils.safe_open(hotpot_config_path, "w", encoding='utf-8') as json_file:
                        json.dump(model_gen_config, json_file, indent=4)

        logger.info(f"EPLB level is : {self.eplb_level}.")
        logger.info(f"EPLB expert map path is : {self.eplb_expert_map_file}.")
        logger.info(f"EPLB redundant experts is : {self.num_redundant_experts}.")
        logger.info(f"EPLB mix shared routing is : {self.mix_shared_routing}")
        logger.info(f"Number of external sharing expert is : {self.num_dangling_shared_experts}")

        return llm_config

    def init_dynamic_eplb_config(self, llm_config, config):
        num_moe_layers = self.num_layers - config.first_k_dense_replace
        num_moe_layers += (1 if self.num_speculative_tokens > 0 else 0)
        num_redundant_experts = getattr(llm_config.models.deepseekv2.eplb, "num_redundant_experts", 0)
        buffer_expert_layer_num = getattr(llm_config.models.deepseekv2.eplb, "buffer_expert_layer_num", 1)
        if llm_config.models.deepseekv2.eplb.aggregate_threshold < 1:
            msg = "eplb aggregate threshold is invalid, value must > 0."
            logger.error(msg)
            raise ValueError(msg)
        if buffer_expert_layer_num < 1 or buffer_expert_layer_num > num_moe_layers:
            msg = f"eplb buffer expert layer num is invalid, value must >= 1 and <= {num_moe_layers}."
            logger.error(msg)
            raise ValueError(msg)
        if llm_config.models.deepseekv2.eplb.num_expert_update_ready_countdown < 1:
            msg = "eplb number of expert update countdown is invalid, value must > 0."
            logger.error(msg)
            raise ValueError(msg)
        if (llm_config.models.deepseekv2.eplb.expert_map_file is not None and
                os.path.exists(llm_config.models.deepseekv2.eplb.expert_map_file)):
            if self.mix_shared_routing:
                num_redundant_experts += self.mapping.world_size
            expert_map = parse_ep_file(llm_config.models.deepseekv2.eplb.expert_map_file)
            layer_map = torch.tensor(expert_map[0])
            if layer_map.reshape(-1).shape[0] < num_redundant_experts + config.n_routed_experts:
                if not torch.any(expert_map, config.n_routed_experts):
                    msg = "Missing shared expert ID: expected EPLB table to include " \
                        "shared expert ID ({config.n_routed_experts})."
                    logger.error(msg)
                    raise ValueError(msg)
    
                for i, _ in enumerate(expert_map):
                    for j, _ in enumerate(expert_map[i]):
                        expert_map[i][j].append(config.n_routed_experts)
            
        else:
            expert_map = random_generation(
                num_moe_layers,
                config.n_routed_experts, 
                self.mapping.world_size,
                num_redundant_experts,
                mix_shared_routing=self.mix_shared_routing
                )
            if self.mix_shared_routing:
                num_redundant_experts += self.mapping.world_size
        logger.debug("init_dynamic_eplb_config, "
                     f"shape: {torch.tensor(expert_map).shape}, "
                     f"max: {torch.tensor(expert_map).max()}")
        self.init_expert_table = torch.tensor(expert_map)
        self.aggregate_threshold = llm_config.models.deepseekv2.eplb.aggregate_threshold
        self.buffer_expert_layer_num = buffer_expert_layer_num
        self.num_expert_update_ready_countdown = llm_config.models.deepseekv2.eplb.num_expert_update_ready_countdown
    
        self.acl_all_gather_operation = torch.classes.ModelTorch.ModelTorch("deepseekV2_AllGatherDecoderModel")
        logger.info(f"EPLB aggregate_threshold is : {self.aggregate_threshold}.")
        logger.info(f"EPLB buffer_expert_layer_num is : {self.buffer_expert_layer_num}.")
        logger.info(f"EPLB num_expert_update_ready_countdown is : {self.num_expert_update_ready_countdown}.")
        return num_redundant_experts

    def init_h3p_config(self, config, ep_level=ExpertParallelDegree.NO_EP):
        # Prefill H3P, Hierarchical & Heterogeneous & Hybrid Parallel
        if config is None:
            self.enable_qkvdown_dp = False
            self.enable_gating_dp = False
            self.enable_shared_expert_dp = False
            self.enable_shared_expert_overlap = False
            self.enable_lcoc_tp = False
            self.enable_lcoc_all2all = False
            self.enable_fused_mla = False
        else:
            if ep_level != ExpertParallelDegree.STATIC_EP:
                msg = "H3P moe dp optimization only takes effect in static EP."
                logger.warning(msg)
                config.models.deepseekv2.h3p.enable_gating_dp = False
                config.models.deepseekv2.h3p.enable_shared_expert_dp = False
                config.models.deepseekv2.h3p.enable_shared_expert_overlap = False
            
            if not self.mapping.has_attn_tp() or ep_level == ExpertParallelDegree.NO_EP or \
                ep_level == ExpertParallelDegree.MIX_EP:
                msg = "H3P qkvdown dp optimization only takes effect when there is " \
                "a tp strategy in the attn part and the moe tail uses allgather."
                logger.warning(msg)
                config.models.deepseekv2.h3p.enable_qkvdown_dp = False

            if self.mapping.attn_dp.group_size == 1 and self.mapping.attn_cp.group_size == 1:
                msg = "H3P moe dp optimization only takes effect when there is a dp or cp " \
                "strategy in the attn part and an allgather at the tail of the attn."
                logger.warning(msg)
                config.models.deepseekv2.h3p.enable_gating_dp = False
                config.models.deepseekv2.h3p.enable_shared_expert_dp = False
                config.models.deepseekv2.h3p.enable_shared_expert_overlap = False

            if not config.models.deepseekv2.h3p.enable_shared_expert_dp and \
                config.models.deepseekv2.h3p.enable_shared_expert_overlap:
                msg = "H3P shared expert overlap depend on shared expert dp."
                logger.warning(msg)
                config.models.deepseekv2.h3p.enable_shared_expert_overlap = False

            if self.layerwise_disaggregated:
                if self.layerwise.split_type != DistributedType.CLOUD:
                    msg = "H3P moe dp optimization only takes effect on cloud side."
                    logger.warning(msg)
                    config.models.deepseekv2.h3p.enable_qkvdown_dp = False

            self.enable_qkvdown_dp = config.models.deepseekv2.h3p.enable_qkvdown_dp
            self.enable_gating_dp = config.models.deepseekv2.h3p.enable_gating_dp
            self.enable_shared_expert_dp = config.models.deepseekv2.h3p.enable_shared_expert_dp
            self.enable_shared_expert_overlap = config.models.deepseekv2.h3p.enable_shared_expert_overlap
            self.enable_lcoc_tp = config.llm.ccl.enable_mc2 and self.enable_qkvdown_dp and \
                self.model_role == 'prefill' and \
                self.soc_info.soc_version >= 250 and self.soc_info.soc_version < 260  # 250-260: AtlasA3
            self.enable_lcoc_all2all = False if (ep_level == ExpertParallelDegree.STATIC_EP or \
                ENV.enable_expert_hotpot_gather) \
                else (config.llm.ccl.enable_mc2 and self.model_role == 'prefill' and \
                self.mapping.mlp_tp.group_size == 16 and \
                self.soc_info.soc_version >= 250 and self.soc_info.soc_version < 260)  # 250-260: AtlasA3
            self.enable_fused_mla = not self.mapping.has_attn_cp() and \
                self.soc_info.soc_version >= 250 and self.soc_info.soc_version < 260  # 250-260: AtlasA3
        
        print_log(self.tp_rank, logger.info, f"Enable qkvdown dp is : {self.enable_qkvdown_dp}.")
        print_log(self.tp_rank, logger.info, f"Enable gating dp is : {self.enable_gating_dp}.")
        print_log(self.tp_rank, logger.info, f"Enable shared expert dp is : {self.enable_shared_expert_dp}.")
        print_log(self.tp_rank, logger.info, f"Enable shared expert overlap is : {self.enable_shared_expert_overlap}.")
        print_log(self.tp_rank, logger.info, f"Enable lcoc tp is : {self.enable_lcoc_tp}.")
        print_log(self.tp_rank, logger.info, f"Enable lcoc all2all is : {self.enable_lcoc_all2all}.")
        print_log(self.tp_rank, logger.info, f"Enable fused MLA is : {self.enable_fused_mla}.")
        return config

    def gen_mask(self, q_len, dtype):
        pre_mask_factor = -10000.0
        mask_free = np.full((125 + 2 * q_len, 128), pre_mask_factor) # 125 and 2 is match formula, 128 is block_size
        mask_free = np.triu(mask_free, 2 - q_len)
        return torch.from_numpy(mask_free).to(dtype)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: DeepseekV2Config):
        if not self.layerwise_disaggregated:
            self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
            logger.info(f"when init_ascend_operations, self.prefix_cache_enable is : {self.prefix_cache_enable}")
            if self.prefix_cache_enable:
                self.acl_encoder_operation_prefixcache = \
                    torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
            if self.num_speculative_tokens:
                self.acl_encoder_operation_mtp = torch.classes.ModelTorch.ModelTorch(
                    CPP_DEEPSEEKV2_MTP_MODEL_CLASS_NAME)
                if self.prefix_cache_enable:
                    self.acl_encoder_operation_prefixcache_mtp = \
                        torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MTP_MODEL_CLASS_NAME)
            self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
            if self.num_speculative_tokens:
                self.acl_decoder_operation_mtp = torch.classes.ModelTorch.ModelTorch(
                    CPP_DEEPSEEKV2_MTP_MODEL_CLASS_NAME)
            # Create dap operation by default
            if self.enable_dap:
                self.acl_dap_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
            if self.enable_dap and self.num_speculative_tokens:
                self.acl_dap_operation_mtp = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MTP_MODEL_CLASS_NAME)
        else:
            if self.layerwise.split_type == DistributedType.EDGE:
                self.acl_head_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                self.acl_tail_encoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                self.acl_head_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                self.acl_tail_decoder_operation = torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                self.acl_head_encoder_operation_prefixcache = \
                    torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                self.acl_tail_encoder_operation_prefixcache = \
                    torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
            else:
                self.acl_internal_decoder_operation = torch.classes.ModelTorch.ModelTorch(
                    CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                layers_num = self.config.num_hidden_layers - \
                    self.layerwise.edge_start_layer_count - self.layerwise.edge_end_layer_count
                self.encode_op_list = [torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME) 
                    for _ in range(layers_num)]
                self.encode_op_prefix_cache_list = [torch.classes.ModelTorch.ModelTorch(CPP_DEEPSEEKV2_MODEL_CLASS_NAME)
                    for _ in range(layers_num)]

    def init_padding_idx(self):
        self.attn_padding_idx = self.placeholder
        self.attn_unpadding_idx = self.placeholder
        self.ffn_padding_idx = self.placeholder
        self.ffn_unpadding_idx = self.placeholder
        self.lm_head_skip_padding_token_indices = self.placeholder
        self.gather_prenorm_idx = self.placeholder
        self.padding_idx = self.placeholder
        self.dynamic_ep_idx = self.placeholder
        self.moe_idx = self.placeholder
        self.kv_cache_padding_idx = self.placeholder
        self.kv_cache_unpadding_idx = self.placeholder
        self.dense_tp_padding_idx = self.placeholder
        self.dense_gather_mlpout_idx = self.placeholder
        self.dense_gather_attnaddout_idx = self.placeholder
        self.dense_gather_prenorm_idx = self.placeholder
        self.dense_allgather_unpad_idx = self.placeholder

    def init_weight_wrapper(self, mode=None):
        attn_wrapper = AttnWrapper(norm_name='input_layernorm', wrapper_name='self_attn')
        moe_mlp_wrapper = MoeMlpWrapper(
            norm_name='post_attention_layernorm',
            router_name='gate',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj',
            shared_experts=(self.n_shared_experts > 0 and (not self.num_dangling_shared_experts > 0))
        )
        weight_wrapper = Deepseekv2WeightWrapper(
            self.soc_info, self.tp_rank,
            attn_wrapper, moe_mlp_wrapper,
            self.num_of_experts,
            enable_lcoc_all2all=self.enable_lcoc_all2all,
            moe_is_nzcasted=self.is_nzcasted
        )
        if not self.layerwise_disaggregated:
            weight_wrapper.register_embedding(self.model.embed_tokens)
        else:
            if mode == 0:
                weight_wrapper.register_embedding(self.model.embed_tokens)
        return weight_wrapper

    def reshape_fusion_gmm_weight(self, weight, dim):
        original_shape = weight.shape
        if original_shape[dim] != 4096:
            msg = f"Tensor shape is {original_shape}, the dimension {dim} should be 4096, but got {original_shape[dim]}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
            raise ValueError(msg)

        if dim < 0:
            dim += len(original_shape) 

        weight = weight.view(*original_shape[:dim], 2, 16, 128, *original_shape[dim + 1:])
        weight = weight.transpose(dim, dim + 1).contiguous()
        weight = weight.view(*original_shape[:dim], -1, *original_shape[dim + 1:])

        return weight.contiguous()

    def get_layer_weights(self, weight_wrapper, layer, i, quantize=None, mla_quantize=None, moe_quantize=None):
        enable_lcoc_tp = True if self.enable_lcoc_tp and \
            i > self.first_k_dense_replace and i < NUM_HIDDEN_LAYERS else False
        if i < self.first_k_dense_replace:
            weight_wrapper.register_moe_layer(layer, quantize, dense_layer=True,
                                            attn_quantize_type=mla_quantize,
                                            enable_lcoc_tp=enable_lcoc_tp)
        elif self.ep:
            weight_wrapper.register_moe_layer(layer, quantize,
                            expert_roster=[i for i, _ in enumerate(self.device_expert)],
                            attn_quantize_type=mla_quantize, moe_quantize_type=moe_quantize,
                            enable_lcoc_tp=enable_lcoc_tp, ep_rank=self.mapping.moe_ep.rank,
                            num_dangling_shared_experts=self.num_dangling_shared_experts,
                            mix_shared_routing=self.mix_shared_routing,
                            enable_atlas_gmm_fused=self.enable_atlas_gmm_fused)
            if self.p_to_d_weight:
                weight_wrapper.register_shared_expert_dp2tp(layer, quantize, "shared_experts_tp")
                weight_wrapper.register_router(layer, quantize, "shuffled_gate")
            del layer.mlp
            torch.npu.empty_cache()
        else:
            weight_wrapper.register_moe_layer(layer, quantize, dense_layer=False,
                                            attn_quantize_type=mla_quantize, moe_quantize_type=moe_quantize,
                                            enable_lcoc_tp=enable_lcoc_tp)
            del layer.mlp
            torch.npu.empty_cache()
        if self.config.quantization_config.fa_quant_type is not None:
            weight_wrapper.register_layer_qkvquant(layer)

        if self.soc_info.need_nz:
            del layer.self_attn
            del layer.post_attention_layernorm
            torch.npu.empty_cache()

    def get_weights(self):
        weight_wrapper = self.init_weight_wrapper()
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            self.get_layer_weights(weight_wrapper, layer, i, self.quantize, self.mla_quantize, self.moe_quantize)

        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def get_weights_mtp(self, weight_wrapper):
        # mtp_word_embedding
        weight_wrapper.register_embedding(self.model.embed_tokens)
        # mtp_enorm
        weight_wrapper.weights.append(self.model.mtp_enorm.weight.data)
        # mtp_hnorm
        weight_wrapper.weights.append(self.model.mtp_hnorm.weight.data)
        # mtp_eh_proj
        weight_wrapper.weights.append(self.model.eh_proj.weight.data)

        # mtp_decode_layer
        mtp_layer_id = NUM_HIDDEN_LAYERS
        self.mtp_layer_pos = len(weight_wrapper.weights)
        layer = self.model.mtp_layer
        if not hasattr(self.config, "mtp_quantize"):
            self.get_layer_weights(weight_wrapper, layer, mtp_layer_id)
        else:
            self.get_layer_weights(weight_wrapper, layer, mtp_layer_id, self.quantize, \
                self.mla_quantize, self.quantize)

        # mtp_head
        weight_wrapper.weights.append(self.model.shared_head_norm.weight.data)
        weight_wrapper.weights.append(self.lm_head.linear.weight.data)
        torch.npu.empty_cache()

        return weight_wrapper

    def get_layerwsie_ascend_param(self, ascend_params, mode, wrapper: Deepseekv2WeightWrapper = None,
                                wrapper_list=None):
        modify_ascend_params = copy.deepcopy(ascend_params)
        modify_ascend_params["packQuantType"] = wrapper.pack_quant_type[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.pack_quant_type]
        modify_ascend_params["attnLinearQuantType"] = wrapper.attn_linear_types[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.attn_linear_types]
        modify_ascend_params["mlpLinearQuantType"] = wrapper.mlp_linear_types[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.mlp_linear_types]
        modify_ascend_params["moeLinearQuantType"] = wrapper.moe_linear_types[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.moe_linear_types]
        modify_ascend_params["attnLinearTransposeType"] = wrapper.attn_linear_transpose_types[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.attn_linear_transpose_types]
        modify_ascend_params["mlpLinearTransposeType"] = wrapper.mlp_linear_transpose_types[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.mlp_linear_transpose_types]
        modify_ascend_params["moeLinearTransposeType"] = wrapper.moe_linear_transpose_types[:] if wrapper \
            else [item for wrapper_son in wrapper_list for item in wrapper_son.moe_linear_transpose_types]
        modify_ascend_params["layerwiseMode"] = mode
        if mode == 0:
            modify_ascend_params[START_ID] = 0
            modify_ascend_params[END_ID] = self.layerwise.edge_start_layer_count
            modify_ascend_params[KVCACHE_QUANT_LAYERS] = \
                    [self.kvcache_quant_layers[i] for i in range(self.layerwise.edge_start_layer_count)]
            modify_ascend_params[MOE_PACK_QUANT_TYPE] = wrapper.moe_pack_type if wrapper.moe_pack_type else 0
        elif mode == 1:
            modify_ascend_params[START_ID] = self.layerwise.edge_start_layer_count
            modify_ascend_params[END_ID] = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
            start_layer = self.layerwise.load_list.index(self.layerwise.edge_start_layer_count)
            end_layer = self.layerwise.load_list.index(self.config.num_hidden_layers - \
                                                       self.layerwise.edge_end_layer_count - 1) + 1
            modify_ascend_params[KVCACHE_QUANT_LAYERS] = [self.kvcache_quant_layers[i] \
                for i in range(start_layer, end_layer)]
            if wrapper:
                modify_ascend_params[MOE_PACK_QUANT_TYPE] = wrapper.moe_pack_type if wrapper.moe_pack_type else 0
            else:
                modify_ascend_params[MOE_PACK_QUANT_TYPE] = wrapper_list[-1].moe_pack_type
        elif mode == 2:
            modify_ascend_params[START_ID] = self.config.num_hidden_layers - self.layerwise.edge_end_layer_count
            modify_ascend_params[END_ID] = self.config.num_hidden_layers
            start_layer = self.layerwise.load_list.index(self.config.num_hidden_layers -
                                                         self.layerwise.edge_end_layer_count)
            end_layer = self.layerwise.load_list.index(self.config.num_hidden_layers - 1) + 1
            modify_ascend_params[KVCACHE_QUANT_LAYERS] = [self.kvcache_quant_layers[i] \
                for i in range(start_layer, end_layer)]
            modify_ascend_params[MOE_PACK_QUANT_TYPE] = wrapper.moe_pack_type
        modify_ascend_params["layerwiseDisaggregated"] = True
        modify_ascend_params["numHiddenLayers"] = modify_ascend_params[END_ID] - modify_ascend_params[START_ID]
        
        return modify_ascend_params

    def get_layerwise_weights(self, mode, layer_no=None, is_prefill=False):
        weight_wrapper = self.init_weight_wrapper(mode=mode)
        start_layer = 0
        end_layer = 0
        if mode == 0:
            start_layer = 0
            end_layer = self.layerwise.edge_start_layer_count
        elif mode == 1:
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
            layer = self.model.layers[i]
            self.get_layer_weights(weight_wrapper, layer, self.layerwise.load_list[i],
                                   self.quantize, self.mla_quantize, self.moe_quantize)
            
        if mode == 2:
            weight_wrapper.register_model_norm(self.model.norm)
            weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def init_ascend_weight(self):
        if not self.layerwise_disaggregated:
            weight_wrapper = self.get_weights()
        else:
            if self.layerwise.split_type == DistributedType.CLOUD:
                self.layerwise.weight_wrappers = []
                for i in range(self.layerwise.edge_start_layer_count, self.config.num_hidden_layers -
                               self.layerwise.edge_end_layer_count):
                    self.layerwise.weight_wrappers.append(
                        self.get_layerwise_weights(mode=1, layer_no=i, is_prefill=True))
            else:
                weight_wrapper_head = self.get_layerwise_weights(mode=0)
                weight_wrapper_tail = self.get_layerwise_weights(mode=2)
        if not self.layerwise_disaggregated:
            self.ascend_weight = weight_wrapper.weights[:]
            pack_quant_configs = weight_wrapper.pack_quant_type[:]
            moe_pack_type = weight_wrapper.moe_pack_type
            if self.is_nzcasted:
                self.buffer_replace_weights_ids = [pair for pair in weight_wrapper.buffer_replace_weights_ids if pair]
            if self.eplb_level == EPLBType.DYNAMIC_EPLB:
                self.placeholder_dataptr = weight_wrapper.placeholder.data_ptr()
            attn_linear_types = weight_wrapper.attn_linear_types[:]
            mlp_linear_types = weight_wrapper.mlp_linear_types[:]
            moe_linear_types = weight_wrapper.moe_linear_types[:]

            attn_linear_transpose_types = weight_wrapper.attn_linear_transpose_types[:]
            mlp_linear_transpose_types = weight_wrapper.mlp_linear_transpose_types[:]
            moe_linear_transpose_types = weight_wrapper.moe_linear_transpose_types[:]

            ein_weight_idx = weight_wrapper.ein_weight_idx[:]
        else:
            if self.layerwise.split_type == DistributedType.EDGE:
                self.layerwise.ascend_weight_head = weight_wrapper_head.weights[:]         
                self.layerwise.ascend_weight_tail = weight_wrapper_tail.weights[:]
                moe_pack_type = None
            else:
                moe_pack_type = None

        if self.num_speculative_tokens:
            layer_weight_num = len(self.ascend_weight)
            weight_wrapper = self.get_weights_mtp(weight_wrapper)
            self.ascend_weight_mtp = weight_wrapper.weights[layer_weight_num:]
            if self.is_nzcasted:
                self.buffer_replace_weights_ids_mtp = \
                    [pair for pair in weight_wrapper.buffer_replace_weights_ids[-1:] if pair]
                for i, _ in enumerate(self.buffer_replace_weights_ids_mtp):
                    self.buffer_replace_weights_ids_mtp[i] = \
                        [j - layer_weight_num for j in self.buffer_replace_weights_ids_mtp[i]]

            pack_quant_configs_mtp = weight_wrapper.pack_quant_type[self.config.num_hidden_layers:]
            attn_linear_types_mtp = weight_wrapper.attn_linear_types[self.config.num_hidden_layers:]
            moe_linear_types_mtp = weight_wrapper.moe_linear_types[self.config.num_hidden_layers:]
            mlp_linear_types_mtp = weight_wrapper.mlp_linear_types[self.config.num_hidden_layers:]
            attn_linear_transpose_types_mtp = weight_wrapper.attn_linear_transpose_types[self.config.num_hidden_layers:]
            mlp_linear_transpose_types_mtp = weight_wrapper.mlp_linear_transpose_types[self.config.num_hidden_layers:]
            moe_linear_transpose_types_mtp = weight_wrapper.moe_linear_transpose_types[self.config.num_hidden_layers:]

        is_w8a8_dynamic = self.quantize == "w8a8_dynamic"
        coder_param = {
            "isUnpadInputs": True,
            "normEps": self.config.rms_norm_eps,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": 1, # for MLA
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs if not self.layerwise_disaggregated else None,
            "isEmbeddingParallel": self.config.parallel_embedding if self.config.parallel_embedding else True,
            "isLmHeadParallel": True,
            "attnLinearQuantType": attn_linear_types if not self.layerwise_disaggregated else None,
            "mlpLinearQuantType": mlp_linear_types if not self.layerwise_disaggregated else None,
            "moeLinearQuantType": moe_linear_types if not self.layerwise_disaggregated else None,
            "attnLinearTransposeType": attn_linear_transpose_types if not self.layerwise_disaggregated else None,
            "mlpLinearTransposeType": mlp_linear_transpose_types if not self.layerwise_disaggregated else None,
            "moeLinearTransposeType": moe_linear_transpose_types if not self.layerwise_disaggregated else None,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "enableSwiGLU": False if self.soc_info.need_nz else True,
            "enableSwiGLUQuantForSharedExperts": False,
            "hiddenSize": self.hidden_size,
            'hasSharedExpert': True if self.n_shared_experts > 0 and \
                not self.num_dangling_shared_experts > 0 and \
                not self.mix_shared_routing else False,
            'hasSharedExpertGate': False,
            "rank": self.mapping.rank,
            "qLoraRank": self.config.q_lora_rank if self.config.q_lora_rank is not None else 0,
            "kvLoraRank": self.config.kv_lora_rank,
            "qkNopeHeadDim": self.config.qk_nope_head_dim,
            "qkRopeHeadDim": self.config.qk_rope_head_dim,
            "softmaxScale": self.softmax_scale,
            "maskStartIdx": self.mask_start_idx,
            "numOfExperts": self.num_of_experts,
            "numOfDeviceExperts": self.num_of_device_expert,
            "deviceExpert": self.device_expert,
            "firstKDenseReplace": self.first_k_dense_replace,
            "nSharedExperts": self.n_shared_experts,
            "processLogits": self.get_process_logits_type(),
            "routedScalingFactor": self.routed_scaling_factor,
            "numOfSelectedExperts": self.num_of_selected_experts,
            "numOfGroups": self.n_group,
            "topkGroups": self.topk_group,
            "quantGroupSize": self.config.quantization_config.group_size,
            "routingMethod": self.get_routing_method_type(),
            "worldSize": self.mapping.world_size,
            "rankTableFile": ENV.rank_table_file,
            "qkvHasBias": False,
            "hasP2DWeight": self.p_to_d_weight,
            "enableSwigluQuant": True if is_w8a8_dynamic and (not self.enable_gmmswigluquant) else False,
            "enableInitQuant": True if is_w8a8_dynamic else False,
            "enableCVOverlap": False,
            "finalStateOut": True if self.num_speculative_tokens else False,
            "enableFusedTopk": True if self.topk_method == "noaux_tc" and self.n_group * 32 >= self.num_of_experts \
                                    else False,
            "enableGMMSwigluQuant": True if self.enable_gmmswigluquant else False,
            "enableMlaPreprocess": True if self.q_lora_rank is not None and self.config.hidden_size == 7168 \
                                        and self.mla_quantize == "w8a8" and self.dtype == torch.float16 \
                                        and not self.enable_lcoc_tp else False,
            "enableAllToAllMC2": self.ep_level == ExpertParallelDegree.DYNAMIC_EP,
            "enableGatherPreNorm": True,
            "enableATBGateMatmul": True,
            "lmHeadLocalTp": self.enable_lm_head_local_tp,
            "enableExtraOprojTp": self.enable_o_proj_local_tp,
            "enableLoadBalance": self.eplb_level == EPLBType.FORCE_EPLB,
            "attnOprojPrefetch": self.enable_oproj_prefetch,
            "enableMlaPrefetch": self.enable_mlapo_prefetch,
            "enableEPWB": self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB],
            "numOfRedundantExpert": self.num_redundant_experts,
            "enableExpertCumSumOutput": ENV.enable_expert_hotpot_gather or self.eplb_level == EPLBType.DYNAMIC_EPLB,
            "enableTopkOutput": self.topk_output,
            "numDanglingSharedExperts": self.num_dangling_shared_experts,
            "enableDenseTp": self.mapping.enable_dense_tp,
            "enableGatingDp": self.enable_gating_dp,
            "enableSharedExpertDp": self.enable_shared_expert_dp,
            "enableSharedExpertOverlap": self.enable_shared_expert_overlap,
            "enableInfNan": ENV.enable_inf_nan_mode,
            MOE_PACK_QUANT_TYPE: moe_pack_type if moe_pack_type else 0,
            "isNzCache": self.enable_nz,
            "maxDecodeDpTokenSize": self.max_decode_dp_token_size,
            "enableFA3": self.config.quantization_config.fa_quant_type is not None,
            KVCACHE_QUANT_LAYERS: self.kvcache_quant_layers if not self.layerwise_disaggregated else None,
            "enableAtlasGMMFused": self.enable_atlas_gmm_fused,
            "enableDistributed": self.distributed_enable,
            "enableDispatchCombineV2": self.enable_dispatch_combine_v2,
            "enableLcocAll2All": self.enable_lcoc_all2all,
            "mixSharedRouting": self.mix_shared_routing,
            "enableModelConfuscation": self.enable_model_obfuscation,
            "modelConfuscationFd": self.obfuscation_fd
        }

        if self.enable_atlas_gmm_fused and not self.is_nzcasted:
            for i, _ in enumerate(self.ascend_weight):
                if self.ascend_weight[i].shape[-2:] == (7168, 4096):
                    need_nz = (torch_npu.get_npu_format(self.ascend_weight[i]) == 29)
                    torch_npu.npu_format_cast_(self.ascend_weight[i], 2)
                    cpu_weight = self.ascend_weight[i].cpu()
                    self.ascend_weight[i] = self.reshape_fusion_gmm_weight(cpu_weight, -1).npu()
                    if need_nz:
                        torch_npu.npu_format_cast_(self.ascend_weight[i], 29) 
                    
                if self.ascend_weight[i].shape[-2:] == (4096, 1):
                    self.ascend_weight[i] = self.reshape_fusion_gmm_weight(self.ascend_weight[i], -2)
                    torch_npu.npu_format_cast_(self.ascend_weight[i], 2)
            if self.num_speculative_tokens:
                for i, _ in enumerate(self.ascend_weight_mtp):
                    if self.ascend_weight_mtp[i].shape[-2:] == (7168, 4096):
                        need_nz = (torch_npu.get_npu_format(self.ascend_weight_mtp[i]) == 29)
                        torch_npu.npu_format_cast_(self.ascend_weight_mtp[i], 2)
                        cpu_weight = self.ascend_weight_mtp[i].cpu()
                        self.ascend_weight_mtp[i] = self.reshape_fusion_gmm_weight(cpu_weight, -1).npu()
                        if need_nz:
                            torch_npu.npu_format_cast_(self.ascend_weight_mtp[i], 29) 
                        
                    if self.ascend_weight_mtp[i].shape[-2:] == (4096, 1):
                        self.ascend_weight_mtp[i] = self.reshape_fusion_gmm_weight(self.ascend_weight_mtp[i], -2)
                        torch_npu.npu_format_cast_(self.ascend_weight_mtp[i], 2)
            torch.npu.empty_cache()


        if self.mapping is not None:
            if self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
                min_moe_ep_buffer_size, min_moe_tp_buffer_size = self.calc_moe_buffer_size()
                if self.mapping.moe_ep.buffer_size < min_moe_ep_buffer_size:
                    msg = f"`hccl_moe_ep_buffer` = {self.mapping.moe_ep.buffer_size} is not enough for " \
                        f"batch size = {self.total_batch_size} and MoE EP size = {self.mapping.moe_ep.group_size}, " \
                        f"so the buff size will be set to {min_moe_ep_buffer_size}."
                    logger.warning(msg)
                    self.mapping.moe_ep.buffer_size = min_moe_ep_buffer_size
                if self.mapping.moe_tp.buffer_size < min_moe_tp_buffer_size:
                    msg = f"`hccl_moe_tp_buffer`={self.mapping.moe_tp.buffer_size} is not enough for " \
                        f"batch size = {self.total_batch_size} and MoE TP size = {self.mapping.moe_tp.group_size}, " \
                        f"so the buff size will be set to {min_moe_tp_buffer_size}."
                    logger.warning(msg)
                    self.mapping.moe_tp.buffer_size = min_moe_tp_buffer_size
            coder_param.update({"mapping": self.mapping.to_dict_v2()})

        if coder_param["routingMethod"] not in ['softMaxTopK', 'integratedSoftmaxTopK', 'deviceLimited', 'noAuxTc']:
            msg = "The routingMethod chosen is not valid, please choose among the following:\n \
                  'softMaxTopK': regular routing method with softmax and topk-sort operators\n \
                  'integratedSoftmaxTopK': routing method with the integration of softmax and topk-sort operators\n \
                  'deviceLimited': device-limited routing method (e.g. deepseekv2)\n \
                  'noAuxTc': routing method with sigmoid and gate bias"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        if coder_param["lmHeadLocalTp"] and ENV.deepseek_mtp:
            msg = "LmHeadLocalTp and DeepseekMtp should not be enabled at the same time. \n \
                   If LM_HEAD_LOCAL_TP=1, \
                   please export DP_MOVE_UP_ENABLE=1 and DP_PARTITION_UP_ENABLE=1"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        if coder_param["lmHeadLocalTp"] and not ENV.enable_dp_partition_up:
            msg = "If LmHeadLocalTp is enabled, DpPartitionUp should also be enabled. \n \
                   If LM_HEAD_LOCAL_TP=1 and DEEPSEEK_MTP=1, \
                   please unset one of them."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        if self.ep_level == ExpertParallelDegree.DYNAMIC_EP and self.mapping.moe_tp.group_size > 1:
            msg = f"Dynamic EP only supports moe_tp = 1, but gets moe_tp = {self.mapping.moe_tp.group_size}."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)
        
        if self.mix_shared_routing and self.num_dangling_shared_experts > 0:
            msg = "mix_shared_routing and num_dangling_shared_experts can not be turned on simultaneously."
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

        logger.debug(f"Whether to dump hot expert information: {ENV.enable_expert_hotpot_gather}")
        logger.debug(f"Path for storing tokens processed by experts: {ENV.expert_hotpot_dump_path}")

        encoder_param = {**coder_param, "isPrefill": True, "supportLcoc": False,
                        "numOfExperts": self.num_of_experts + self.num_dangling_shared_experts,
                        "expertParallelDegree": self.ep_level \
                        if self.ep_level != ExpertParallelDegree.MIX_EP \
                            else ExpertParallelDegree.DYNAMIC_EP,
                        "backend": self.communication_backend if self.ep_level < 2 \
                            else self.dep_communication_backend['prefill'],
                        "enableQkvdownDp": self.enable_qkvdown_dp,
                        "enableDpOut": ENV.enable_dp_partition_up,
                        "enableInitRoutingCutoff": self.enable_init_routing_cutoff,
                        "scaledTopk": int(self.num_of_selected_experts[0] * self.topk_scaling_factor) if \
                                        self.num_of_selected_experts[0] else 1,
                        "enableLcocTp": self.enable_lcoc_tp,
                        "enableFusedMLA": self.enable_fused_mla,
                        }
        encoder_param_prefixcache = {**encoder_param,
                         "enablePrefixCache": self.prefix_cache_enable,                         
                         }
        decoder_param = {**coder_param, "isPrefill": False, "supportLcoc": False,
                        "expertParallelDegree": self.ep_level \
                            if self.ep_level != ExpertParallelDegree.MIX_EP \
                            else ExpertParallelDegree.STATIC_EP,
                        "backend": self.communication_backend if self.ep_level < 2 \
                            else self.dep_communication_backend['decode'],
                        "enableSpeculate": self.speculate_enable,
                        "maskfree": self.maskfree,
                        "enableDpOut": ENV.enable_dp_partition_up or bool(self.distributed_enable),
                        "enableQkvdownDp": self.enable_lcoc_tp,
                        "enableLcocTp": self.enable_lcoc_tp,
                        }
        if not self.layerwise_disaggregated:
            if self.acl_encoder_operation is not None:
                self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
                self.acl_encoder_operation.set_weight(self.ascend_weight)
                if self.is_nzcasted:
                    for buffer_list in self.buffer_replace_weights_ids:
                        weight_id = buffer_list[0]
                        self.acl_encoder_operation.set_format_to_nz(weight_id)
                        if self.enable_atlas_gmm_fused:
                            self.acl_encoder_operation.set_format_to_nz(weight_id + 6)
            if self.prefix_cache_enable:
                self.acl_encoder_operation_prefixcache.set_param(json.dumps({**encoder_param_prefixcache}))
                self.acl_encoder_operation_prefixcache.set_weight(self.ascend_weight)

            if self.acl_decoder_operation is not None:
                self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
                if self.enable_einsum_nz:
                    # cast k_b_proj && v_b_proj
                    for pos in ein_weight_idx:
                        self.ascend_weight[pos] = transdata_3d(self.ascend_weight[pos])
                        torch_npu.npu_format_cast_(self.ascend_weight[pos], 29)
                        torch.npu.empty_cache()
                self.acl_decoder_operation.set_weight(self.ascend_weight)
                if self.is_nzcasted:
                    for buffer_list in self.buffer_replace_weights_ids:
                        weight_id = buffer_list[0]
                        self.acl_decoder_operation.set_format_to_nz(weight_id)
                        if self.enable_atlas_gmm_fused:
                            self.acl_decoder_operation.set_format_to_nz(weight_id + 6)
                if self.eplb_level == EPLBType.DYNAMIC_EPLB:
                    if not self.ascend_buffer_weight and self.buffer_replace_weights_ids:
                        start_id = self.buffer_replace_weights_ids[0][0]
                        end_id = self.buffer_replace_weights_ids[0][1]
                        for _ in range(self.buffer_expert_layer_num):
                            for i in range(start_id, end_id):
                                if self.ascend_weight[i].data_ptr() != self.placeholder_dataptr:
                                    self.ascend_buffer_weight.append(torch.empty_like(self.ascend_weight[i]))
        else:
            if self.layerwise.split_type == DistributedType.EDGE:
                encoder_head_param = self.get_layerwsie_ascend_param(encoder_param, 0, weight_wrapper_head)
                encoder_tail_param = self.get_layerwsie_ascend_param(encoder_param, 2, weight_wrapper_tail)
                decoder_head_param = self.get_layerwsie_ascend_param(decoder_param, 0, weight_wrapper_head)
                decoder_tail_param = self.get_layerwsie_ascend_param(decoder_param, 2, weight_wrapper_tail)
                
                encoder_head_prefix_cache_param = {**encoder_head_param, "enablePrefixCache": True}
                encoder_tail_prefix_cache_param = {**encoder_tail_param, "enablePrefixCache": True}
                
                self.acl_head_encoder_operation.set_param(json.dumps({**encoder_head_param}))
                self.weight_wrapper_head = weight_wrapper_head
                self.acl_head_encoder_operation.set_weight(weight_wrapper_head.weights[:])
                self.acl_tail_encoder_operation.set_param(json.dumps({**encoder_tail_param}))
                self.acl_tail_encoder_operation.set_weight(weight_wrapper_tail.weights[:])
                self.acl_head_decoder_operation.set_param(json.dumps({**decoder_head_param}))
                self.acl_head_decoder_operation.set_weight(weight_wrapper_head.weights[:])
                self.acl_tail_decoder_operation.set_param(json.dumps({**decoder_tail_param}))
                self.acl_tail_decoder_operation.set_weight(weight_wrapper_tail.weights[:])
                
                self.acl_head_encoder_operation_prefixcache.set_param(json.dumps({**encoder_head_prefix_cache_param}))
                self.acl_tail_encoder_operation_prefixcache.set_param(json.dumps({**encoder_tail_prefix_cache_param}))
                self.acl_head_encoder_operation_prefixcache.set_weight(weight_wrapper_head.weights[:])
                self.acl_tail_encoder_operation_prefixcache.set_weight(weight_wrapper_tail.weights[:])
            else:
                for layer in range(0, self.config.num_hidden_layers - self.layerwise.edge_end_layer_count - \
                        self.layerwise.edge_start_layer_count):
                    encoder_internal_param = self.get_layerwsie_ascend_param(
                        encoder_param, 1, self.layerwise.weight_wrappers[layer]
                    )
                    encoder_internal_param[START_ID] = self.layerwise.edge_start_layer_count + layer
                    encoder_internal_param[END_ID] = self.layerwise.edge_start_layer_count + layer + 1
                    encoder_internal_param["cloudLastLayerId"] = \
                        self.config.num_hidden_layers - self.layerwise.edge_end_layer_count - 1
                    encoder_internal_param["numHiddenLayers"] = 1
                    encoder_internal_param[KVCACHE_QUANT_LAYERS] = [
                        self.kvcache_quant_layers[self.layerwise.load_list.index(
                            self.layerwise.edge_start_layer_count + layer)]]
                    self.encode_op_list[layer].set_param(json.dumps({**encoder_internal_param}))
                    self.encode_op_list[layer].set_weight(self.layerwise.weight_wrappers[layer].weights[:])
                    
                    encoder_internal_param_prefix_cache = {**encoder_internal_param, "enablePrefixCache": True}
                    self.encode_op_prefix_cache_list[layer].set_param(
                        json.dumps({**encoder_internal_param_prefix_cache})
                    )
                    self.encode_op_prefix_cache_list[layer].set_weight(self.layerwise.weight_wrappers[layer].weights[:])
                decoder_internal_param = self.get_layerwsie_ascend_param(
                    decoder_param, 1, wrapper_list=self.layerwise.weight_wrappers
                )
                decoder_internal_param["cloudLastLayerId"] = \
                    self.config.num_hidden_layers - self.layerwise.edge_end_layer_count - 1
                self.acl_internal_decoder_operation.set_param(json.dumps({**decoder_internal_param}))
                self.acl_internal_decoder_operation.set_weight([weight_tensor \
                    for wrapper in self.layerwise.weight_wrappers \
                    for weight_tensor in wrapper.weights[:]])

        if self.eplb_level == EPLBType.DYNAMIC_EPLB:
            self.acl_all_gather_operation.set_param(json.dumps({**decoder_param}))
        if self.num_speculative_tokens:
            encoder_param_mtp = copy.deepcopy(encoder_param)
            decoder_param_mtp = copy.deepcopy(decoder_param)
            encoder_param_mtp["enableLcocTp"] = False
            decoder_param_mtp["enableLcocTp"] = False
            decoder_param_mtp["enableQkvdownDp"] = False
            if not hasattr(self.config, "mtp_quantize"):
                encoder_param_mtp["enableLcocAll2All"] = False
                decoder_param_mtp["enableLcocAll2All"] = False

            update_param_mtp = dict()
            update_param_mtp["numHiddenLayers"] = 1
            update_param_mtp["firstKDenseReplace"] = 0
            update_param_mtp["packQuantType"] = pack_quant_configs_mtp
            update_param_mtp["attnLinearQuantType"] = attn_linear_types_mtp
            update_param_mtp["mlpLinearQuantType"] = mlp_linear_types_mtp
            update_param_mtp["moeLinearQuantType"] = moe_linear_types_mtp
            update_param_mtp["attnLinearTransposeType"] = attn_linear_transpose_types_mtp
            update_param_mtp["mlpLinearTransposeType"] = mlp_linear_transpose_types_mtp
            update_param_mtp["moeLinearTransposeType"] = moe_linear_transpose_types_mtp
            if not hasattr(self.config, "mtp_quantize"):
                update_param_mtp["enableSwigluQuant"] = False
                update_param_mtp["enableInitQuant"] = False
                update_param_mtp["enableMlaPreprocess"] = False
                update_param_mtp["enableFusedTopk"] = False
                update_param_mtp["enableGMMSwigluQuant"] = False

            update_param_mtp["enableAtlasGMMFused"] = self.enable_atlas_gmm_fused
            update_param_mtp["enableLoadBalance"] = False         
            update_param_mtp[KVCACHE_QUANT_LAYERS] = [False]

            encoder_param_mtp.update(update_param_mtp)
            decoder_param_mtp.update(update_param_mtp)

            if self.acl_encoder_operation_mtp is not None:
                self.acl_encoder_operation_mtp.set_param(json.dumps({**encoder_param_mtp}))
                self.acl_encoder_operation_mtp.set_weight(self.ascend_weight_mtp)
                if self.is_nzcasted:
                    for buffer_list in self.buffer_replace_weights_ids_mtp:
                        weight_id = buffer_list[0]
                        self.acl_encoder_operation_mtp.set_format_to_nz(weight_id)
                        if self.enable_atlas_gmm_fused:
                            self.acl_encoder_operation_mtp.set_format_to_nz(weight_id + 6)
            if self.acl_decoder_operation_mtp is not None:
                self.acl_decoder_operation_mtp.set_param(json.dumps({**decoder_param_mtp}))
                self.acl_decoder_operation_mtp.set_weight(self.ascend_weight_mtp)
                if self.is_nzcasted:
                    for buffer_list in self.buffer_replace_weights_ids_mtp:
                        weight_id = buffer_list[0]
                        self.acl_decoder_operation_mtp.set_format_to_nz(weight_id)
                        if self.enable_atlas_gmm_fused:
                            self.acl_decoder_operation_mtp.set_format_to_nz(weight_id + 6)
                    
            if self.prefix_cache_enable:
                encoder_param_mtp = {**encoder_param_mtp, "enablePrefixCache": self.prefix_cache_enable}
                self.acl_encoder_operation_prefixcache_mtp.set_param(json.dumps({**encoder_param_mtp}))
                self.acl_encoder_operation_prefixcache_mtp.set_weight(self.ascend_weight_mtp)

        if self.enable_dap and self.acl_dap_operation is not None:
            self.acl_dap_operation.set_param(
                json.dumps({"enableDap": True, **encoder_param, BACKEND: "hccl", SUPPORT_LCOC: False}))
            self.acl_dap_operation.set_weight(self.ascend_weight)

        if self.enable_dap and self.num_speculative_tokens and self.acl_dap_operation_mtp is not None:
            self.acl_dap_operation_mtp.set_param(
                json.dumps({"enableDap": True, **encoder_param_mtp, BACKEND: "hccl", SUPPORT_LCOC: False}))
            self.acl_dap_operation_mtp.set_weight(self.ascend_weight_mtp)

    def get_all2all_buffer_factor(self, length, is_prefill=False):
        if hasattr(self.ds_config, "alltoall_ep_buffer_scale_factors"):
            alltoall_ep_buffer_scale_factors = self.ds_config.alltoall_ep_buffer_scale_factors
            for threshold in alltoall_ep_buffer_scale_factors:
                if length >= threshold[0]:
                    return threshold[1]
            return self.mapping.moe_ep.group_size
        if is_prefill and length <= 32:
            return self.mapping.moe_ep.group_size * 2
        elif is_prefill and length >= ALLTOALL_LONG_SEQLEN_THRESHOLD // self.mapping.moe_ep.group_size:
            return MAX_ALLTOALL_BUFF_SCALE
        else:
            max_scale = math.sqrt(self.mapping.moe_ep.group_size)
            min_scale = max(1, max_scale / 2)
            scale = min_scale + (max_scale - min_scale) / \
                (1 + math.exp(math.log2(length) - math.log2(self.mapping.moe_ep.group_size)))
            return scale

    def get_process_logits_type(self) -> str:
        if self.routed_scaling_factor > 1 and self.norm_topk_prob is True:
            return "normScaling"
        elif self.routed_scaling_factor > 1:
            return "scaling"
        return "none"

    def get_routing_method_type(self) -> str:
        if self.topk_method == "noaux_tc":
            return "noAuxTc"
        elif self.topk_method == "group_limited_greedy":
            return "deviceLimited"
        return "softMaxTopK"

    def calc_moe_buffer_size(self):
        moe_ep_buffer_size = math.ceil(
            math.ceil(self.total_batch_size / self.mapping.world_size) * self.hidden_size
            * (self.num_of_experts + self.num_redundant_experts) * 4 / (1024 ** 2)
        ) + 1
        moe_tp_buffer_size = math.ceil(
            math.ceil(self.total_batch_size / self.mapping.world_size) * self.hidden_size
            * self.mapping.moe_tp.group_size * 4 / (1024 ** 2)
        ) + 1
        return moe_ep_buffer_size, moe_tp_buffer_size

    def free_operation_inputs(self, is_prefill):
        if is_prefill:
            self.acl_encoder_operation_inputs = []
        else:
            self.acl_decoder_operation_inputs = []

    def prepare_cp_prefill_inputs(self, input_ids, input_lengths, q_lens):
        cp_size = self.mapping.attn_cp.group_size
        # While enable load balancing, the total sequence of each request is divided into cp*2 chunks,
        # and the input sequence on each cp_rank contains two chunks.
        if self.has_prefixcache:
            input_lengths = torch.tensor(q_lens).npu()

        chunk_lengths = (input_lengths // 2).tolist()

        # While enable load balancing, it is used to take the first half
        # and the second half of each request sequence.
        # Example：[q0_0,q0_3,q1_0,q1_3] --gather--> [q0_0,q1_0] [q0_3,q1_3]
        cp_load_balance_idx_first, cp_load_balance_idx_last = [], []
        base = 0
        for length in input_lengths.tolist():
            length_range = list(range(base, base + length))
            divider = length // 2
            cp_load_balance_idx_first.extend(length_range[:divider])
            cp_load_balance_idx_last.extend(length_range[divider:])
            base += length

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

        self.acl_param = json.dumps({
            SEQUENCE_LENGTH: input_lengths.tolist(),
            Q_LEN: q_lens,
            SEQUENCE_LENGTH_CP: chunk_lengths,
        })
        self.acl_encoder_operation_inputs.extend([
            torch.tensor(chunk_lengths, dtype=torch.int32).npu(),
            torch.tensor(cp_load_balance_idx_first, dtype=torch.int64).npu(),
            torch.tensor(cp_load_balance_idx_last, dtype=torch.int64).npu(),
            torch.tensor(cp_o_recover_idx, dtype=torch.int64).npu(),
            torch.tensor(cp_kv_recover_idx, dtype=torch.int64).npu()])

    def prepare_prefixcache_input(self, input_lens, kv_cache, **kwargs):
        q_lens = kwargs.get('q_lens', None)
        self.has_prefixcache = False
        if self.prefix_cache_enable and q_lens:
            if len(input_lens) != len(q_lens):
                logger.info(f'input_lens:{len(input_lens)} size not equal to q_lens:{len(q_lens)}')
            prefix_lens = [i - j for i, j in zip(input_lens, q_lens)]
            self.prefix_lens = prefix_lens
            
            if self.mapping.has_attn_cp() or self.mapping.has_attn_inner_sp():
                prefix_lens = kwargs.get("per_rank_prefix_lens", None)
                self.prefix_lens = self.prefix_lens if prefix_lens is None else prefix_lens.tolist()
            
            sum_prefix_lens = sum(self.prefix_lens)
            if sum_prefix_lens > 0:
                self.has_prefixcache = True  
                self.in_history_compressed_kv = torch.empty([sum_prefix_lens, self.kv_lora_rank],
                                                                dtype=kv_cache[0][1].dtype).pin_memory()
                self.in_history_compressed_kv = self.in_history_compressed_kv.to(self.device, non_blocking=True)
                self.in_history_k_rope = torch.empty([sum_prefix_lens, self.qk_rope_head_dim], 
                                                        dtype=kv_cache[0][1].dtype).pin_memory()
                self.in_history_k_rope = self.in_history_k_rope.to(self.device, non_blocking=True)
                q_seqlen = torch.from_numpy(np.array(q_lens)).to(torch.int32)  # new tokens
                prefix_seqlen = torch.from_numpy(np.array(self.prefix_lens)).to(torch.int32)  # cache tokens
                if self.config.quantization_config.fa_quant_type is not None:
                    self.in_history_compressed_kv_int = torch.empty([sum_prefix_lens, self.kv_lora_rank],
                                                                    dtype=torch.int8).pin_memory()
                    self.in_history_compressed_kv_int = self.in_history_compressed_kv_int.to(self.device, non_blocking=True)

                self.ring_cur_seqlen = torch.stack([q_seqlen, q_seqlen]).pin_memory()
                self.ring_cur_seqlen = self.ring_cur_seqlen.to(self.device, non_blocking=True)
                self.ring_cache_seqlen = torch.stack([q_seqlen, prefix_seqlen]).pin_memory()
                self.ring_cache_seqlen = self.ring_cache_seqlen.to(self.device, non_blocking=True)
                self.acl_param = json.dumps({
                    SEQUENCE_LENGTH: input_lens,
                    Q_LEN: q_lens if q_lens is not None else [],
                    "ringCurSeqlen": q_lens + q_lens,
                    "ringCacheSeqlen": q_lens + self.prefix_lens,
                })

    def prepare_paddingidx_for_prefixcache_contextparallel(self, **kwargs):
        sp_computed_slots_padding_idx = kwargs.get("sp_computed_slots_padding_idx", None)
        computed_slots_order = kwargs.get("sp_computed_slots_order", None)
        self.kv_cache_padding_idx = sp_computed_slots_padding_idx
        self.kv_cache_unpadding_idx = computed_slots_order
        acl_param = json.loads(self.acl_param)
        q_lens = kwargs.get("q_lens", None)
        chunk_lengths = [x // 2 for x in q_lens]
        q_seqlen = torch.from_numpy(np.array(chunk_lengths)).to(torch.int32)  # new tokens
        all_rank_prefix_lens = kwargs.get("all_rank_prefix_lens", None)
        all_rank_prefix_seqlen = torch.from_numpy(np.array(all_rank_prefix_lens)).to(torch.int32)  # cache tokens
        self.kv_cache_len = torch.stack([q_seqlen, all_rank_prefix_seqlen]).pin_memory()
        self.kv_cache_len = self.kv_cache_len.to(self.device, non_blocking=True)
        acl_param["kvCachelen"] = chunk_lengths + all_rank_prefix_lens
        self.acl_param = json.dumps(acl_param)

    def prepare_paddingidx_for_contextparallel(self, input_ids):
        input_length = len(input_ids)  # The length of each sp_rank input sequence (batch*seq) is the same.

        # The total number of tokens
        self.token_size = input_length * self.mapping.attn_cp.group_size

        # The length that each rank needs to be padded
        padding_length = 0 if input_length % self.mapping.attn_tp.group_size == 0 \
            else self.mapping.attn_tp.group_size - input_length % self.mapping.attn_tp.group_size
        # The token length on each CP group after padding
        input_len_padding_per_group = input_length + padding_length
        # The token length on each rank after Attn ReduceScatter
        input_len_padding_per_rank = input_len_padding_per_group // self.mapping.attn_tp.group_size

        # Before Attn ReduceScatter, the sequence length needs to be padded to be divisible by attn_tp_size.
        self.attn_padding_idx = torch.concat([
            torch.arange(input_length, dtype=torch.int32),
            torch.zeros(padding_length, dtype=torch.int32)
        ]).view(-1).npu()

        # After Attn ReduceScatter, GatherNorm is performed.
        tp_rank = self.mapping.attn_tp.rank
        self.gather_prenorm_idx = self.attn_padding_idx[
                                tp_rank * input_len_padding_per_rank: (tp_rank + 1) * input_len_padding_per_rank]

        # Used to remove padding after global AllGather.
        all_gather_skip_padding_token_indices = torch.concat([
            torch.arange(input_length, dtype=torch.int32) + input_len_padding_per_group * cp_rank
            for cp_rank in range(self.mapping.attn_cp.group_size)], dim=0).npu()

        # Adapt for static EP and dynamic EP.
        if self.ep_level == ExpertParallelDegree.STATIC_EP:
            # UnPadding after Attn AllGather
            self.attn_unpadding_idx = all_gather_skip_padding_token_indices
            # Padding before FFN ReduceScatter
            self.ffn_padding_idx = torch.concat([
                torch.concat([torch.arange(input_length * cp_rank, input_length * (cp_rank + 1), dtype=torch.int32),
                                torch.zeros(input_len_padding_per_group - input_length, dtype=torch.int32)])
                for cp_rank in range(self.mapping.attn_cp.group_size)], dim=0).npu()
        elif self.ep_level == ExpertParallelDegree.DYNAMIC_EP:
            # Attn UnPadding 和 FFN Padding 维持token长度不变
            self.attn_unpadding_idx = torch.arange(input_len_padding_per_rank).view(-1).npu()
            self.ffn_padding_idx = self.attn_unpadding_idx

        # Remove padding after FFN AllGather
        self.ffn_unpadding_idx = torch.arange(len(input_ids), dtype=torch.int32).npu()
        # The FFN AllGather of the last layer is global AllGather
        self.lm_head_skip_padding_token_indices = all_gather_skip_padding_token_indices

    def prepare_paddingidx_for_dataparallel(self, input_ids, is_prefill, **kwargs):
        local_token_size = len(input_ids)
        token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
        self.token_size = token_size_per_dp_group.sum().tolist()
        if token_size_per_dp_group.max().item() % self.mapping.attn_tp.group_size == 0:
            max_token_size_per_dp_group = token_size_per_dp_group.max().item()
        else:
            padding_tmp = \
                self.mapping.attn_tp.group_size - \
                token_size_per_dp_group.max().item() % self.mapping.attn_tp.group_size
            max_token_size_per_dp_group = token_size_per_dp_group.max().item() + padding_tmp

        token_size_per_dp_group_startid = torch.cumsum(token_size_per_dp_group, dim=0)
        token_size_per_dp_group_startid[-1] = 0

        self.lm_head_skip_padding_token_indices = torch.concat([
            torch.concat([torch.arange(j) + max_token_size_per_dp_group * rank_id]) \
                for rank_id, j in enumerate(token_size_per_dp_group)], dim=0).npu()

        atom_dp_size = max_token_size_per_dp_group // self.mapping.attn_tp.group_size
        self.atom_dp_size = atom_dp_size
        input_length_padding = max_token_size_per_dp_group - local_token_size
        self.attn_padding_idx = torch.concat([
                torch.arange(local_token_size, dtype=torch.int32),
                torch.zeros(input_length_padding, dtype=torch.int32)
            ]).view(-1).npu()

        if self.mapping.attn_o_proj_tp.group_size > 1:
            self.gather_prenorm_idx = torch.arange(local_token_size, dtype=torch.int32).npu()
        else:
            self.gather_prenorm_idx = \
                self.attn_padding_idx[self.mapping.attn_tp.rank * atom_dp_size: \
                (self.mapping.attn_tp.rank + 1) * atom_dp_size]

        if self.ep_level == ExpertParallelDegree.DYNAMIC_EP or \
        (self.ep_level == ExpertParallelDegree.MIX_EP and is_prefill):
            self.attn_unpadding_idx = torch.arange(atom_dp_size).view(-1).npu()
            self.ffn_padding_idx = self.attn_unpadding_idx
        else:
            self.attn_unpadding_idx = torch.concat(
                [torch.arange(s) + max_token_size_per_dp_group * i
                    for i, s in enumerate(token_size_per_dp_group)]).view(-1).npu()
            self.ffn_padding_idx = torch.concat([
                torch.concat([torch.arange(j) + token_size_per_dp_group_startid[rank_id - 1],
                torch.zeros(max_token_size_per_dp_group - j, dtype=torch.int32)]) \
                    for rank_id, j in enumerate(token_size_per_dp_group)], dim=0).npu()

        self.ffn_unpadding_idx = torch.arange(local_token_size, dtype=torch.int32).npu()

    def prepare_paddingidx_for_densetpparallel(self, input_lengths, is_prefill, **kwargs):
        token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
        if token_size_per_dp_group is None and not is_prefill:
            cur_size = len(input_lengths.tolist())
            token_size_per_dp_group = [self.max_decode_dp_token_size] * self.mapping.attn_dp.group_size
            token_size_per_dp_group[self.mapping.attn_dp.rank] = cur_size
        tp_rank = self.mapping.attn_tp.rank
        tmp_dp_padding_idx = []
        tmp_dense_gather_mlpout_idx = []
        tmp_dense_gather_attnaddout_idx = []
        tmp_dense_gather_prenorm_idx = []
        tmp_dense_allgather_unpad_idx = []
        if max(token_size_per_dp_group) % self.mapping.attn_tp.group_size == 0:
            max_token_size_per_dp_group = max(token_size_per_dp_group)
        else:
            padding_tmp = \
                self.mapping.attn_tp.group_size - \
                max(token_size_per_dp_group) % self.mapping.attn_tp.group_size
            max_token_size_per_dp_group = max(token_size_per_dp_group) + padding_tmp
        for i in range(self.mapping.dense_dp.group_size):
            slice_size = self.mapping.attn_dp.group_size // self.mapping.dense_dp.group_size
            token_size_per_dense_group = token_size_per_dp_group[i * slice_size: (i + 1) * slice_size]
            start_idx = 0
            prenorm_slice_size = max_token_size_per_dp_group // self.mapping.attn_tp.group_size
            for token_size_current in token_size_per_dense_group:
                reduce_scatter_padding_size = max_token_size_per_dp_group % self.mapping.attn_tp.group_size
                padding_size_per_dense_group = max_token_size_per_dp_group - token_size_current
                padding_size_per_dense_group += (self.mapping.attn_tp.group_size - \
                    reduce_scatter_padding_size) if reduce_scatter_padding_size != 0 else 0
                padding_idx = torch.concat([torch.arange(token_size_current, dtype=torch.int32), \
                    torch.zeros(padding_size_per_dense_group, dtype=torch.int32)]).view(-1)
                tmp_dp_padding_idx.append(padding_idx)
                unpadding_idx = torch.arange(token_size_current, dtype=torch.int32) + start_idx
                tmp_dense_gather_mlpout_idx.append(unpadding_idx)
                tmp_dense_gather_attnaddout_idx.append(torch.arange(token_size_current, dtype=torch.int32))
                start_idx += max_token_size_per_dp_group
                tmp_dense_gather_prenorm_idx.append(padding_idx[tp_rank * prenorm_slice_size: \
                    (tp_rank + 1) * prenorm_slice_size])
                tmp_dense_allgather_unpad_idx.append(unpadding_idx)
        self.dense_tp_padding_idx = tmp_dp_padding_idx[self.mapping.attn_dp.rank].view(-1).npu()
        self.dense_gather_mlpout_idx = tmp_dense_gather_mlpout_idx[self.mapping.attn_dp.rank].view(-1).npu()
        self.dense_gather_attnaddout_idx = tmp_dense_gather_attnaddout_idx[self.mapping.attn_dp.rank].view(-1).npu()
        self.dense_gather_prenorm_idx = tmp_dense_gather_prenorm_idx[self.mapping.attn_dp.rank].view(-1).npu()
        self.dense_allgather_unpad_idx = tmp_dense_allgather_unpad_idx[self.mapping.attn_dp.rank].view(-1).npu()

    def prepare_nodp_paddingidx_for_expertparallel(self, local_token_size):
        token_num_need_to_pad = local_token_size % self.mapping.attn_tp.group_size
        if token_num_need_to_pad > 0:
            padding_num = self.mapping.attn_tp.group_size - local_token_size % self.mapping.attn_tp.group_size
        else:
            padding_num = 0
        token_num_per_rank = (local_token_size + padding_num) // self.mapping.attn_tp.group_size
        self.attn_padding_idx = torch.concat([
            torch.arange(local_token_size, dtype=torch.int32),
            torch.zeros(padding_num, dtype=torch.int32)
        ]).view(-1).npu()
        self.attn_unpadding_idx = torch.arange(token_num_per_rank, dtype=torch.int32).npu()
        self.ffn_padding_idx = self.attn_unpadding_idx
        self.ffn_unpadding_idx = torch.arange(local_token_size, dtype=torch.int32).npu()
        self.lm_head_skip_padding_token_indices = self.ffn_unpadding_idx
        self.gather_prenorm_idx = \
            self.attn_padding_idx[self.mapping.attn_tp.rank * token_num_per_rank: \
            (self.mapping.attn_tp.rank + 1) * token_num_per_rank]

    def prepare_paddingidx_for_expertparallel(self, input_ids, is_prefill=False):
        num_experts_per_tok = self.config.num_experts_per_tok
        if self.mix_shared_routing:
            num_experts_per_tok += 1
        if self.mapping.attn_tp.group_size == 1:
            self.dynamic_ep_idx = torch.tensor(
                [i for i in range(len(input_ids) * num_experts_per_tok)],
                dtype=torch.int32).npu().view(-1)
            dynamic_ep_idx_padding = torch.tensor(
                    [i for i in range(self.attn_unpadding_idx.shape[0] * num_experts_per_tok)],
                    dtype=torch.int32).npu().view(-1)
        else:
            self.dynamic_ep_idx = torch.tensor(
                    [i for i in range(self.attn_unpadding_idx.shape[0] * num_experts_per_tok)],
                    dtype=torch.int32).npu().view(-1)
            dynamic_ep_idx_padding = self.dynamic_ep_idx
        ep_input_length = int(
            dynamic_ep_idx_padding.shape[0] *
            self.get_all2all_buffer_factor(dynamic_ep_idx_padding.shape[0] / num_experts_per_tok, is_prefill))
        all2all_padding = ep_input_length % self.mapping.moe_ep.group_size
        ep_input_length_padding = (
                self.mapping.moe_ep.group_size - all2all_padding) if all2all_padding != 0 else 0

        ep_input_length_padding += ep_input_length
        self.moe_idx = torch.tensor([i + 1 for i in range(ep_input_length_padding)], dtype=torch.int32).npu().view(-1)

        self.expert_array = torch.ones(self.moe_idx.shape[0], dtype=torch.float16).npu().view(-1, 1)

    def prepare_paddingidx_for_mixparallel(self, input_lengths, input_ids, is_prefill, **kwargs):
        local_token_size = len(input_ids)
        self.token_size = local_token_size

        if self.mapping.has_attn_cp():
            self.prepare_paddingidx_for_contextparallel(input_ids)

        if self.mapping.has_dp() and not ENV.enable_dp_move_up:
            self.prepare_paddingidx_for_dataparallel(input_ids, is_prefill, **kwargs)
        elif self.mapping.has_dp() and ENV.enable_dp_move_up:
            token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
            if ENV.enable_dp_partition_up:
                self.token_size = local_token_size
            else:
                self.token_size = token_size_per_dp_group.sum().item()

        if self.mapping.enable_dense_tp:
            self.prepare_paddingidx_for_densetpparallel(input_lengths, is_prefill, **kwargs)

        perf_time_start = 0
        if ENV.benchmark_enable:
            import time
            torch.npu.synchronize()
            perf_time_start = time.time()
        self.expert_array = self.placeholder

        final_hidden_states_token_size = self.token_size
        if self.layerwise_disaggregated and self.layerwise.split_type == DistributedType.CLOUD:
            final_hidden_states_token_size = local_token_size

        final_hidden_states = torch.empty([final_hidden_states_token_size, self.config.hidden_size],
                                            dtype=self.dtype,
                                            device=input_ids.device)

        is_ep = (self.ep_level == ExpertParallelDegree.DYNAMIC_EP or \
            (self.ep_level == ExpertParallelDegree.MIX_EP and is_prefill))

        if is_ep and not (self.mapping.has_dp() or self.mapping.has_attn_cp()):
            self.prepare_nodp_paddingidx_for_expertparallel(local_token_size)

        if is_ep and (not ENV.enable_dp_move_up or self.mapping.has_attn_cp()):
            self.prepare_paddingidx_for_expertparallel(input_ids, is_prefill)

        dep_inputs = [self.attn_padding_idx, self.attn_unpadding_idx, self.ffn_padding_idx,
            self.ffn_unpadding_idx, self.lm_head_skip_padding_token_indices, self.gather_prenorm_idx,
            self.start_device_expert_id, self.max_device_expert_id,
            self.dynamic_ep_idx, self.moe_idx, self.placeholder]

        return dep_inputs, final_hidden_states, perf_time_start
    
    def prepare_csp_input_filter_mask(self,
                                      input_seq_lens,
                                      q_lens):
        q_len = q_lens[0] if q_lens else 1
        input_filter_mask = input_seq_lens.eq(0).repeat_interleave(q_len)
        for seq_id in range(input_seq_lens.shape[0]):
            if input_seq_lens[seq_id] < q_len:
                input_filter_mask[seq_id * q_len: seq_id * q_len + q_len - input_seq_lens[seq_id]] = True
        return input_filter_mask.view(-1, 1, 1)

    # called by super().forward()
    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs):
        perf_time_start = 0
        q_lens = kwargs.get(Q_LENS, [])
        spec_mask = kwargs.get('spec_mask', None)
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                            self.device,
                                                            self.max_position_embeddings)
        self.cos_embed = self.rotary_embedding.get_cos_cached_total()
        self.sin_embed = self.rotary_embedding.get_sin_cached_total()

        dep_inputs, final_hidden_states, perf_time_start = \
            self.prepare_paddingidx_for_mixparallel(input_lengths, input_ids, is_prefill, **kwargs)

        if self.mapping.has_dp() and ENV.enable_dp_move_up:
            has_tp = self.mapping.has_attn_tp() or self.mapping.has_attn_o_proj_tp() or \
                     self.mapping.lm_head_tp.group_size > 1
            if not self.distributed_enable or (has_tp and self.distributed_enable):
                dep_inputs = kwargs.get("dep_inputs", None)
                dep_inputs = dep_inputs[:6] + \
                                [self.start_device_expert_id, self.max_device_expert_id] + dep_inputs[6:]
                if self.enable_lm_head_local_tp:
                    dep_inputs[-1] = dep_inputs[-1] - dep_inputs[-1][0]
            moe_idx = dep_inputs[-2] # note that moe_idx is the second last tensor in dep_inputs
            self.expert_array = torch.ones(moe_idx.shape[0], dtype=torch.float16).npu().view(-1, 1)

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                            dtype=torch.int64, device=input_ids.device)

        if self.eplb_level == EPLBType.FORCE_EPLB:
            fake_topk = self.fake_topk[:dep_inputs[1].shape[0] if self.mapping.has_attn_tp() else len(input_ids)]

        if self.mix_shared_routing:
            if self.ep_level == ExpertParallelDegree.STATIC_EP:
                token_number = self.token_size

                if is_prefill and self.enable_gating_dp:
                    if not self.mapping.has_dp():
                        atom_dp_size = math.ceil(token_number / self.mapping.attn_tp.group_size / self.mapping.attn_cp.group_size)
                    else:
                        token_size_per_dp_group = kwargs.get("token_size_per_dp_group", None)
                        if token_size_per_dp_group.max().item() % self.mapping.attn_tp.group_size == 0:
                            max_token_size_per_dp_group = token_size_per_dp_group.max().item()
                        else:
                            padding_tmp = \
                                self.mapping.attn_tp.group_size - \
                                token_size_per_dp_group.max().item() % self.mapping.attn_tp.group_size
                            max_token_size_per_dp_group = token_size_per_dp_group.max().item() + padding_tmp
                        atom_dp_size = max_token_size_per_dp_group // self.mapping.attn_tp.group_size

                    self.mix_shared_routing_weight = torch.tensor([1] * (atom_dp_size), \
                                                                dtype=torch.float).view(atom_dp_size, 1).npu()
                else:
                    if is_prefill or not self.mapping.has_dp() or not self.distributed_enable:
                        token_number = self.token_size
                    else:
                        token_number = self.max_batch_size * (self.num_speculative_tokens + 1) * \
                                                            self.mapping.attn_dp.group_size

                    self.mix_shared_routing_weight = torch.tensor([1] * (token_number), \
                                                                dtype=torch.float).view(token_number, 1).npu()
                if self.mapping.moe_ep.rank == 0:
                    other_expert = len(self.device_expert)
                else:
                    other_expert = 0
                self.mix_shared_routing_expert = torch.tensor([other_expert] * (token_number), \
                                                                dtype=torch.int32).view(token_number, 1).npu()
                size = math.ceil(token_number / self.mapping.moe_ep.group_size)
                self.mix_shared_routing_expert[size*self.mapping.moe_ep.rank:size*(self.mapping.moe_ep.rank+1)] = 256
            else:
                token_number = dep_inputs[1].shape[0] if self.mapping.has_attn_tp() else len(input_ids)
                self.mix_shared_routing_weight = torch.tensor([1] * (token_number), \
                                                                dtype=torch.float).view(token_number, 1).npu()
                self.mix_shared_routing_expert = torch.tensor([256] * (token_number), \
                                                        dtype=torch.int32).view(token_number, 1).npu()
        if is_prefill:
            input_lens = input_lengths.tolist()
            self.prepare_prefixcache_input(input_lens, kv_cache, **kwargs)

            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, self.dtype,
                                                                    kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            elif self.mapping.has_attn_cp() or self.has_prefixcache or self.enable_fused_mla:
                atten_mask = self.attn_mask.get_attn_mask(512, self.dtype, kv_cache[0][0].device)
            else:
                atten_mask = self.attn_mask.get_attn_mask(128, self.dtype,
                                                                kv_cache[0][0].device)
            if not self.has_prefixcache:
                self.acl_param = json.dumps({
                    SEQUENCE_LENGTH: input_lens,
                    Q_LEN: q_lens if q_lens is not None else [],
                })
    
            self.acl_encoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                atten_mask,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                final_hidden_states,
                torch.tensor(self.prefix_lens, dtype=torch.int32, device=self.device) \
                    if self.has_prefixcache else self.placeholder,  # 复用token_offset位置, pagedloadcache用
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64),
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot,
                self.placeholder
            ]
            if self.eplb_level == EPLBType.FORCE_EPLB:
                self.acl_encoder_operation_inputs.append(fake_topk)
            self.acl_encoder_operation_inputs.extend(dep_inputs)
            if self.mapping.has_attn_cp():
                self.prepare_cp_prefill_inputs(input_ids, input_lengths, q_lens)

            if self.has_prefixcache and self.mapping.has_attn_cp():
                self.prepare_paddingidx_for_prefixcache_contextparallel(**kwargs)
                for b_i in range(block_tables.shape[0]):  # 存在部分命中， 没有block的 rank 要添加一个用于只读的无用块，否则pagegloadcache算子会报错
                    if block_tables[b_i][0] == -1:
                        block_tables[b_i][0] = 0
            if self.has_prefixcache:
                self.acl_encoder_operation_inputs.extend([self.in_history_compressed_kv, \
                    self.in_history_k_rope, self.ring_cur_seqlen, self.ring_cache_seqlen]
                )
                if self.config.quantization_config.fa_quant_type is not None:
                    self.acl_encoder_operation_inputs.append(self.in_history_compressed_kv_int)
            if self.has_prefixcache and self.mapping.has_attn_cp():
                self.acl_encoder_operation_inputs.extend([self.kv_cache_padding_idx, \
                    self.kv_cache_unpadding_idx, self.kv_cache_len])
            if self.mapping.enable_dense_tp:  # new padding idx please add here before
                self.acl_encoder_operation_inputs.append(self.dense_tp_padding_idx)
                self.acl_encoder_operation_inputs.append(self.dense_gather_mlpout_idx)
                self.acl_encoder_operation_inputs.append(self.dense_gather_attnaddout_idx)
                self.acl_encoder_operation_inputs.append(self.dense_gather_prenorm_idx)
                self.acl_encoder_operation_inputs.append(self.dense_allgather_unpad_idx)
            if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:    # new inputs please add here before
                self.acl_encoder_operation_inputs.append(self.expert_routing_map)
                EplbExpertDataCollect().collect_routing_map(self.expert_routing_map, self.mapping.rank)
            if self.mix_shared_routing:
                self.acl_encoder_operation_inputs.append(self.mix_shared_routing_weight)
                self.acl_encoder_operation_inputs.append(self.mix_shared_routing_expert)
            return self.acl_encoder_operation_inputs, self.acl_param, perf_time_start
        else:
            if self.mapping.has_attn_cp():
                # During the decode stage, each CP domain receives the same token as input,
                # and the output token is also the same.
                lm_head_indices = lm_head_indices.repeat(self.mapping.attn_cp.group_size)

            self.acl_param = json.dumps({
                SEQUENCE_LENGTH: input_lengths.tolist(),
                Q_LEN: [int(i) for i in q_lens if i] if q_lens else None,
                SEQUENCE_LENGTH_SP: kwargs.get("input_lengths_sp").tolist()
                                    if self.mapping.has_attn_inner_sp() else [],
            })
            self.acl_decoder_operation_inputs = [
                input_ids,
                position_ids.to(torch.int64),
                self.cos_embed,
                self.sin_embed,
                # mask
                (self.atten_mask_free if self.maskfree else spec_mask)
                    if self.speculate_enable else self.attn_mask_fake,
                block_tables.to(torch.int32),
                slots.to(torch.int32),
                self.placeholder,
                final_hidden_states,
                self.placeholder,
                self.placeholder,
                input_lengths.to(torch.int32),
                lm_head_indices.to(torch.int64) if lm_head_indices is not None \
                                                or (self.mapping.has_dp() and self.mapping.has_mlp_tp()) \
                                                else self.lm_head_indices_fake,
                self.expert_array,
                self.expert_group,
                self.one_hot,
                self.zero_hot,
                torch.tensor(q_lens, dtype=torch.int32, device=self.device) if self.speculate_enable \
                                                                            else self.placeholder
            ]
            if self.eplb_level == EPLBType.FORCE_EPLB:
                self.acl_decoder_operation_inputs.append(fake_topk)
            self.acl_decoder_operation_inputs.extend(dep_inputs)
            if self.mapping.has_attn_inner_sp():
                input_lengths_sp = kwargs.get("input_lengths_sp", None)
                input_filter_mask = self.prepare_csp_input_filter_mask(input_lengths_sp, q_lens)
                self.acl_decoder_operation_inputs.extend([input_lengths_sp, input_filter_mask])
            elif self.mapping.has_attn_cp():
                input_filter_mask = self.prepare_csp_input_filter_mask(input_lengths, q_lens)
                self.acl_decoder_operation_inputs.extend([input_filter_mask])
            if self.mapping.enable_dense_tp:  # new padding idx please add here before
                self.acl_decoder_operation_inputs.append(self.dense_tp_padding_idx)
                self.acl_decoder_operation_inputs.append(self.dense_gather_mlpout_idx)
                self.acl_decoder_operation_inputs.append(self.dense_gather_attnaddout_idx)
                self.acl_decoder_operation_inputs.append(self.dense_gather_prenorm_idx)
                self.acl_decoder_operation_inputs.append(self.dense_allgather_unpad_idx)
            if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:    # new inputs please add here before
                self.acl_decoder_operation_inputs.append(self.expert_routing_map)
                EplbExpertDataCollect().collect_routing_map(self.expert_routing_map, self.mapping.rank)
            if self.mix_shared_routing:
                self.acl_decoder_operation_inputs.append(self.mix_shared_routing_weight)
                self.acl_decoder_operation_inputs.append(self.mix_shared_routing_expert)
            return self.acl_decoder_operation_inputs, self.acl_param, perf_time_start

    def execute_expert_all_gather(self, acl_inputs: torch.Tensor):
        if self.eplb_level == EPLBType.DYNAMIC_EPLB:
            if acl_inputs is None:
                raise ValueError(f"execute_all_gather input can not be None. acl_inputs={acl_inputs}")

            acl_param = json.dumps({})
            acl_out = self.acl_all_gather_operation.execute([acl_inputs], acl_param)
            return acl_out
        else:
            raise EnvironmentError(f"Unsupported eplb level: {self.eplb_level}. Expected: 2.")

    def prepare_inputs_for_decode_ascend(self,
            out_embeddings: torch.Tensor):
        self.acl_decoder_operation_inputs[0] = out_embeddings
        return self.acl_decoder_operation_inputs

    def execute_ascend_operator(self,
                                acl_inputs: list,
                                acl_param: str,
                                is_prefill: bool,
                                is_mtp: bool = False, **kwargs) -> torch.Tensor:
        """Execute the Ascend acl operator."""
        split_part = kwargs.get("split_part", None)
        layer_index = kwargs.get("layer_index", None)
        if not self.num_speculative_tokens:
            if is_prefill and self.prefix_cache_enable and self.has_prefixcache:
                if not self.layerwise_disaggregated:
                    acl_model_out = self.acl_encoder_operation_prefixcache.execute(acl_inputs, acl_param)
                else:
                    if split_part == LwdLayerStatus.EDGE_START_LAYER:
                        acl_model_out = self.acl_head_encoder_operation_prefixcache.execute(acl_inputs, acl_param)
                    elif split_part == LwdLayerStatus.CLOUD_MIDDLE_LAYER:
                        acl_model_out = self.encode_op_prefix_cache_list[layer_index].execute(acl_inputs, acl_param)
                    else:
                        acl_model_out = self.acl_tail_encoder_operation_prefixcache.execute(acl_inputs, acl_param)
                    self.has_prefixcache = False
                
            elif is_prefill:
                if self.layerwise_disaggregated:
                    if split_part == LwdLayerStatus.EDGE_START_LAYER:
                        acl_model_out = self.acl_head_encoder_operation.execute(acl_inputs, acl_param)
                    elif split_part == LwdLayerStatus.CLOUD_MIDDLE_LAYER:
                        acl_model_out = self.encode_op_list[layer_index].execute(acl_inputs, acl_param)
                    else:
                        acl_model_out = self.acl_tail_encoder_operation.execute(acl_inputs, acl_param)
                else:
                    acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
            else:
                if self.layerwise_disaggregated:
                    if split_part == LwdLayerStatus.EDGE_START_LAYER:
                        acl_model_out = self.acl_head_decoder_operation.execute(acl_inputs, acl_param)
                    elif split_part == LwdLayerStatus.CLOUD_MIDDLE_LAYER:
                        acl_model_out = self.acl_internal_decoder_operation.execute(acl_inputs, acl_param)
                    else:
                        acl_model_out = self.acl_tail_decoder_operation.execute(acl_inputs, acl_param)
                else:
                    acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)

            if self.topk_output:
                acl_model_out = EplbExpertDataCollect().split_eplb_expert_data(acl_model_out, is_mtp, True)
            elif ENV.enable_expert_hotpot_gather or self.eplb_level == EPLBType.DYNAMIC_EPLB:
                acl_model_out = EplbExpertDataCollect().split_eplb_expert_data(acl_model_out, is_mtp)

            acl_hidden_state = acl_model_out[0]
            return acl_hidden_state

        if is_mtp:
            encoder_operation = self.acl_encoder_operation_mtp
            decoder_operation = self.acl_decoder_operation_mtp
            if self.has_prefixcache:
                encoder_operation = self.acl_encoder_operation_prefixcache_mtp
        else:
            encoder_operation = self.acl_encoder_operation
            decoder_operation = self.acl_decoder_operation
            if self.has_prefixcache:
                encoder_operation = self.acl_encoder_operation_prefixcache

        if is_prefill:
            acl_model_out = encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = decoder_operation.execute(acl_inputs, acl_param)

        if self.topk_output:
            EplbExpertDataCollect().set_model_ref(self)
            acl_model_out = EplbExpertDataCollect().split_eplb_expert_data(acl_model_out, is_mtp, True)
        elif ENV.enable_expert_hotpot_gather or self.eplb_level == EPLBType.DYNAMIC_EPLB:
            EplbExpertDataCollect().set_model_ref(self)
            acl_model_out = EplbExpertDataCollect().split_eplb_expert_data(acl_model_out, is_mtp)

        try:
            logits = acl_model_out[0]
            last_hidden_states = acl_model_out[1]
        except IndexError as e:
            raise RuntimeError("运行时报错，请开启日志进一步定位问题") from e
        return logits, last_hidden_states


    def prepare_mtp_roll_inputs_for_ascend(self, acl_inputs_mtp, acl_param_mtp, q_lens, 
                                           logits_mtp, hidden_states_mtp, first_op=False, lm_head_local_dp=None):

        next_token = logits_mtp.argmax(dim=-1)
        # all roll

        if lm_head_local_dp is not None:
            lm_head_indices_dp = lm_head_local_dp
        else:
            cumsum_indices = torch.cumsum(q_lens, dim=0)
            lm_head_indices_dp = cumsum_indices - 1
        
        acl_inputs_mtp[0] = torch.roll(acl_inputs_mtp[0], shifts=-1, dims=0)
        acl_inputs_mtp[0][lm_head_indices_dp] = next_token # input_ids
        acl_inputs_mtp[1] += 1 # position_ids

        if first_op:
            acl_inputs_mtp.insert(18, hidden_states_mtp)
        else:
            acl_inputs_mtp[18] = hidden_states_mtp

        return acl_inputs_mtp, acl_param_mtp

    def prepare_repeated_batch(self, acl_inputs, acl_param, q_lens):

        # blocktables
        acl_inputs[5] = acl_inputs[5].repeat_interleave(q_lens, dim=0)
        # input_lengths
        input_lengths = acl_inputs[11]
        repeat_input_lengths = input_lengths.repeat_interleave(q_lens, dim=0)

        offset = torch.cat([torch.arange(-i + 1, 1) for i in q_lens]).npu()
        repeat_input_lengths = repeat_input_lengths + offset

        acl_inputs[11] = repeat_input_lengths.to(torch.int32)

        acl_param = json.dumps({
            SEQUENCE_LENGTH: repeat_input_lengths.tolist(),
        })

        return acl_inputs, acl_param

    def execute_dap_ascend_operator(self,
                                acl_inputs: list,
                                acl_param: str,
                                is_prefill: bool,
                                is_mtp: bool = False) -> torch.Tensor:
        """Execute the Ascend acl operator."""
        if not is_prefill:
            raise NotImplementedError("Dap is not supported for decoder.")
        if is_mtp:
            acl_model_out = self.acl_dap_operation_mtp.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_dap_operation.execute(acl_inputs, acl_param)

        if len(acl_model_out) != 2 * (self.num_speculative_tokens + 1):
            raise RuntimeError("Number of output tensors is not equal to the expected value.")
        return acl_model_out

    def update_mtp_inputs(self, logits_mtp, hidden_states_mtp, q_lens, \
            acl_inputs_mtp, acl_param_mtp, mtp_idx, kwargs):
        fake_data = False
        if self.mapping.has_dp() and not ENV.enable_dp_partition_up:
            lm_head_indices_dp_rank_ids = kwargs.get('lm_head_indices_dp_rank_ids')
            logits_gather_indices = \
                    torch.arange(0, lm_head_indices_dp_rank_ids.shape[0])
            logits_gather_indices = \
                    logits_gather_indices[lm_head_indices_dp_rank_ids == self.mapping.attn_dp.rank]
            if logits_gather_indices.numel() > 0:
                logits_mtp = logits_mtp[logits_gather_indices]
            else:
                fake_data = True

            shard_effective_token_indices = kwargs.get('shard_effective_token_indices')
            hidden_states_mtp = hidden_states_mtp[shard_effective_token_indices]

        if self.mapping.has_attn_cp():
            hidden_states_mtp = hidden_states_mtp[acl_inputs_mtp[0].size(0) * self.mapping.attn_cp.rank: \
                                                  acl_inputs_mtp[0].size(0) * (self.mapping.attn_cp.rank + 1)]

        if fake_data:
            if mtp_idx == 0:
                acl_inputs_mtp.insert(18, hidden_states_mtp)
        else:
            lm_head_local_dp = kwargs.get('lm_head_local_dp', None)
            acl_inputs_mtp, acl_param_mtp = \
            self.prepare_mtp_roll_inputs_for_ascend(
                acl_inputs_mtp, acl_param_mtp, q_lens, logits_mtp, hidden_states_mtp, mtp_idx == 0, lm_head_local_dp)
        if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
            map_id = -3 if self.mix_shared_routing else -1 # -3: eplb routing map id
            acl_inputs_mtp[map_id] = acl_inputs_mtp[map_id][-1:]
        if self.eplb_level == EPLBType.FORCE_EPLB:
            del acl_inputs_mtp[19]
        return acl_inputs_mtp, acl_param_mtp

    def dap_forward(
        self, *args, **kwargs
    ) -> torch.Tensor:
        if self.num_speculative_tokens:
            return self.dap_forward_mtp(*args, **kwargs)
        else:
            return super().dap_forward(*args, **kwargs)

    def dap_forward_mtp(
            self,
            input_ids: List[torch.Tensor],
            position_ids: List[torch.Tensor],
            is_prefill: List[bool],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: List[torch.Tensor],
            slots: List[torch.Tensor],
            input_lengths: List[torch.Tensor],
            max_seq_len: List[int],
            lm_head_indices: List[torch.Tensor | None],
            dap_kwargs: List[dict],
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)
        all_inputs = []
        acl_inputs, acl_param, _ = self.prepare_inputs_for_ascend(
            input_ids[0], position_ids[0], is_prefill[0], kv_cache,
            block_tables[0], slots[0], input_lengths[0], max_seq_len[0],
            lm_head_indices[0], **dap_kwargs[0])
        all_inputs.extend(acl_inputs)
        acl_inputs_successor, acl_param_successor, _ = self.prepare_inputs_for_ascend(
            input_ids[1], position_ids[1], is_prefill[1], kv_cache,
            block_tables[1], slots[1], input_lengths[1], max_seq_len[1],
            lm_head_indices[1], **dap_kwargs[1])
        all_inputs.extend(acl_inputs_successor)

        acl_param_dict = json.loads(acl_param)
        for k, v in json.loads(acl_param_successor).items():
            acl_param_dict[f"{k}_successor"] = v
        # 执行主模型
        outputs = self.execute_dap_ascend_operator(
            all_inputs, json.dumps(acl_param_dict), is_prefill[0])

        logits, hidden_states = (outputs[0], outputs[2]), (outputs[1], outputs[3])
        llm_hidden_states = (hidden_states[0][lm_head_indices[0]], hidden_states[1][lm_head_indices[1]])

        # logits 是一个tuple，里面是两个logits，logits是一个两维NPU tensor [ntokens, vocabSize]
        torch.npu.synchronize()
        logits_mtp, hidden_states_mtp = logits, hidden_states
        for mtp_idx in range(self.num_speculative_tokens):
            self.acl_dap_operation_mtp.set_kv_cache(self.mtp_k_caches[mtp_idx: mtp_idx + 1],
                                                            self.mtp_v_caches[mtp_idx: mtp_idx + 1])
            # 更新变量
            all_mtp_inputs = []
            acl_inputs, acl_param = \
                self.update_mtp_inputs(logits_mtp[0], hidden_states_mtp[0], input_lengths[0],
                    acl_inputs, acl_param, mtp_idx, dap_kwargs[0])
            all_mtp_inputs.extend(acl_inputs)
            acl_inputs_successor, acl_param_successor = \
                self.update_mtp_inputs(logits_mtp[1], hidden_states_mtp[1], input_lengths[1],
                    acl_inputs_successor, acl_param_successor, mtp_idx, dap_kwargs[1])
            all_mtp_inputs.extend(acl_inputs_successor)

            acl_param_dict = json.loads(acl_param)
            for k, v in json.loads(acl_param_successor).items():
                acl_param_dict[f"{k}_successor"] = v
            # 执行mtp
            outputs = self.execute_dap_ascend_operator(
                all_mtp_inputs, json.dumps(acl_param_dict), is_prefill[0], True)
            logits_mtp, hidden_states_mtp = (outputs[0], outputs[2]), (outputs[1], outputs[3])

        return logits, llm_hidden_states

    def select_logits(self, logits, **kwargs):
        dp_logits_num = kwargs.get("dp_logits_num")
        if dp_logits_num is None:
            return logits
        dp_rank_id = self.mapping.attn_dp.rank

        if dp_rank_id == 0:
            logits = logits[:dp_logits_num[dp_rank_id]]
        else:
            logits = logits[dp_logits_num[dp_rank_id - 1]: dp_logits_num[dp_rank_id]]
        
        return logits

    def delete_local_tp_mtp_inputs(self, acl_inputs_mtp):
        if self.mapping.enable_dense_tp:
            acl_inputs_mtp = acl_inputs_mtp[:-5]
        return acl_inputs_mtp

    def add_local_tp_mtp_inputs(self, acl_inputs_mtp):
        if self.mapping.enable_dense_tp:
            acl_inputs_mtp.append(self.dense_tp_padding_idx)
            acl_inputs_mtp.append(self.dense_gather_mlpout_idx)
            acl_inputs_mtp.append(self.dense_gather_attnaddout_idx)
            acl_inputs_mtp.append(self.dense_gather_prenorm_idx)
            acl_inputs_mtp.append(self.dense_allgather_unpad_idx)
        return acl_inputs_mtp

    def layerwise_get_input_param(self, is_prefill, is_end_layer):
        if is_prefill:
            if self.layerwise.acl_inputs_prefill is None:
                self.layerwise.acl_inputs_prefill = self.layerwise.acl_inputs_prefill_queue.get(timeout=900)
                self.layerwise.acl_param_prefill = self.layerwise.acl_param_prefill_queue.get(timeout=900)
            prefill_input = self.layerwise.acl_inputs_prefill
            prefill_param = self.layerwise.acl_param_prefill
            if is_end_layer:
                self.layerwise.acl_inputs_prefill = None
                self.layerwise.acl_param_prefill = None
            return prefill_input, prefill_param
        else:
            decode_input = self.layerwise.acl_inputs_decode
            decode_param = self.layerwise.acl_param_decode
            if is_end_layer:
                self.layerwise.acl_inputs_decode = None
                self.layerwise.acl_param_decode = None
            return decode_input, decode_param

    def layerwise_save_input_param(self, inputs, runtime_param, is_prefill):
        # input[0] is hidden and needs to be replaced each time; no caching is required.
        inputs_copy = [None] + inputs[1:]
        if is_prefill:
            self.layerwise.acl_inputs_prefill_queue.put(inputs_copy)
            self.layerwise.acl_param_prefill_queue.put(runtime_param)
        else:
            self.layerwise.acl_inputs_decode = inputs_copy
            self.layerwise.acl_param_decode = runtime_param
               
    def forward_layerwise_disaggregated_edge(
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
        if not self.layerwise.ascend_weight_head and not self.layerwise.ascend_weight_tail:
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()
        self.init_kvcache(kv_cache)

        layerwise_disaggregated_exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        out_hidden = kwargs.get("out_hidden")
        if out_hidden is not None:
            out_hidden = out_hidden.to(self.dtype)
        if layerwise_disaggregated_exe_stage.start_exec_layer == 0:
            acl_inputs, acl_param, _ = self.prepare_inputs_for_ascend(
                input_ids, position_ids, is_prefill, kv_cache,
                block_tables, slots, input_lengths, max_seq_len,
                lm_head_indices, **kwargs)
            if not is_prefill:
                out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                            split_part=LwdLayerStatus.EDGE_START_LAYER)
                self.layerwise_save_input_param(acl_inputs, acl_param, is_prefill)
                acl_inputs = []
            else:
                out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                          split_part=LwdLayerStatus.EDGE_START_LAYER)
                self.layerwise_save_input_param(acl_inputs, acl_param, is_prefill)
                acl_inputs = []
            return out_hidden
        if layerwise_disaggregated_exe_stage.end_exec_layer == 1:
            if not is_prefill:
                last_input, last_param = self.layerwise_get_input_param(is_prefill, True)
                last_input[0] = out_hidden
                logits = self.execute_ascend_operator(last_input, last_param, is_prefill,
                                                      split_part=LwdLayerStatus.EDGE_END_LAYER)
            else:
                if layerwise_disaggregated_exe_stage.is_long_seq and \
                    layerwise_disaggregated_exe_stage.long_seq_start_idx != 0 and \
                        not layerwise_disaggregated_exe_stage.request_dp_empty:
                    self.has_prefixcache = True
                last_input, last_param = self.layerwise_get_input_param(is_prefill, True)  
                last_input[0] = out_hidden
                logits = self.execute_ascend_operator(last_input, last_param, is_prefill,
                                                      split_part=LwdLayerStatus.EDGE_END_LAYER)
            return logits
            
        return out_hidden
    
    def forward_layerwise_disaggregated_cloud(
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
        if (len(self.layerwise.weight_wrappers) == 0):
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)

        layerwise_disaggregated_exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        out_hidden = kwargs.get("out_hidden")
        if out_hidden is not None:
            out_hidden = out_hidden.to(self.dtype)
        
        if layerwise_disaggregated_exe_stage.start_exec_layer == 0:
            acl_inputs, acl_param, _ = self.prepare_inputs_for_ascend(
                input_ids, position_ids, is_prefill, kv_cache,
                block_tables, slots, input_lengths, max_seq_len,
                lm_head_indices, **kwargs)
        else:
            if is_prefill:
                acl_inputs = self.layerwise.acl_inputs_prefill
                acl_param = self.layerwise.acl_param_prefill
            else:
                acl_inputs = self.layerwise.acl_inputs_decode
                acl_param = self.layerwise.acl_param_decode
        if not is_prefill:
            acl_inputs[0] = out_hidden
            out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                      split_part=LwdLayerStatus.CLOUD_MIDDLE_LAYER)
            acl_inputs = []
        else:
            if layerwise_disaggregated_exe_stage.start_exec_layer == 0:
                acl_inputs[0] = out_hidden
            else:
                acl_inputs[0] = self.layerwise.p_out_hidden
            
            for i in range(layerwise_disaggregated_exe_stage.start_exec_layer,
                            layerwise_disaggregated_exe_stage.end_exec_layer):
                if i > layerwise_disaggregated_exe_stage.start_exec_layer:
                    acl_inputs[0] = out_hidden
                if layerwise_disaggregated_exe_stage.is_long_seq and \
                    layerwise_disaggregated_exe_stage.long_seq_start_idx != 0 and \
                        not layerwise_disaggregated_exe_stage.request_dp_empty:
                    self.has_prefixcache = True
                out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                    split_part=LwdLayerStatus.CLOUD_MIDDLE_LAYER, layer_index=i)
            self.layerwise.p_out_hidden = out_hidden
            # acl_inputs[0] is hidden and needs to be replaced each time; no caching is required.
            self.layerwise.acl_inputs_prefill = [None] + acl_inputs[1:]
            self.layerwise.acl_param_prefill = acl_param
        
        return out_hidden 
    
    def forward_layerwise_disaggregated_warmup(
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
        if (not self.layerwise.ascend_weight_head and not self.layerwise.ascend_weight_tail) and \
                (len(self.layerwise.weight_wrappers) == 0):
            self.get_adapter_ids(**kwargs)
            self.init_ascend_weight()

        self.init_kvcache(kv_cache)

        out_hidden = kwargs.get("out_hidden")
        if out_hidden is not None:
            out_hidden = out_hidden.to(self.dtype)
        acl_inputs, acl_param, perf_start_time = self.prepare_inputs_for_ascend(
            input_ids, position_ids, is_prefill, kv_cache,
            block_tables, slots, input_lengths, max_seq_len,
            lm_head_indices, **kwargs)
        if not is_prefill:
            if self.layerwise.split_type == DistributedType.CLOUD:
                acl_inputs = self.prepare_inputs_for_decode_ascend(out_hidden)
                out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                            split_part=LwdLayerStatus.CLOUD_MIDDLE_LAYER)
                return (out_hidden, perf_start_time)
            out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                        split_part=LwdLayerStatus.EDGE_START_LAYER)
            acl_inputs = self.prepare_inputs_for_decode_ascend(out_hidden)
            logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                        split_part=LwdLayerStatus.EDGE_END_LAYER)
            self.free_operation_inputs(is_prefill)
        else:
            if self.layerwise.split_type == DistributedType.CLOUD:
                for i in range(self.config.num_hidden_layers - self.layerwise.edge_start_layer_count -
                               self.layerwise.edge_end_layer_count):
                    acl_inputs[0] = out_hidden
                    out_hidden = self.execute_ascend_operator(
                                        acl_inputs, acl_param, is_prefill,
                                        split_part=LwdLayerStatus.CLOUD_MIDDLE_LAYER, layer_index=i)
                return (out_hidden, perf_start_time)
            out_hidden = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                        split_part=LwdLayerStatus.EDGE_START_LAYER)
            acl_inputs[0] = out_hidden
            logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill,
                                                        split_part=LwdLayerStatus.EDGE_END_LAYER)
            self.free_operation_inputs(is_prefill)
        return (logits, perf_start_time)

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
        if not self.layerwise_disaggregated:
            if not self.ascend_weight:
                self.get_adapter_ids(**kwargs)
                self.init_ascend_weight()

            if self.num_speculative_tokens:
                return self.forward_mtp(input_ids, position_ids, \
                    is_prefill, kv_cache, block_tables, slots, input_lengths, \
                    max_seq_len, lm_head_indices, **kwargs)

            prof = span_start("prepareInputs", True)
            prof = span_attr(prof, "slots", lambda: tensor_attr(slots, False))
            prof = span_attr(prof, "block_tables", lambda: tensor_attr(block_tables, False))
            prof = span_attr(prof, "input_ids", lambda: tensor_attr(input_ids, False))
            self.init_kvcache(kv_cache)
            acl_inputs, acl_param, perf_start_time = self.prepare_inputs_for_ascend(
                input_ids, position_ids, is_prefill, kv_cache,
                block_tables, slots, input_lengths, max_seq_len,
                lm_head_indices, **kwargs)
            span_end(prof, True)

            prof = span_start("operatorExecute", True)
            logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
            if not is_prefill and self.mapping.has_attn_cp():
                # During the CP decode stage, each CP domain receives the same token as input,
                # resulting in the same output token. As a result, duplicates exist in the aggregated next_token.
                logits = logits[:logits.size(0) // self.mapping.attn_cp.group_size]

            if is_prefill and self.distributed_enable:
                logits = self.select_logits(logits, **kwargs)
            self.free_operation_inputs(is_prefill)
            span_end(prof, True)
            return (logits, perf_start_time)
        else:
            layerwise_disaggregated_exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
            if layerwise_disaggregated_exe_stage is None:
                return self.forward_layerwise_disaggregated_warmup(input_ids, position_ids, is_prefill, kv_cache,
                                                                        block_tables, slots, input_lengths, max_seq_len,
                                                                        lm_head_indices, **kwargs)
            if self.layerwise.split_type == DistributedType.CLOUD:
                return self.forward_layerwise_disaggregated_cloud(input_ids, position_ids, is_prefill, kv_cache,
                                                                        block_tables, slots, input_lengths, max_seq_len,
                                                                        lm_head_indices, **kwargs)
            else:
                return self.forward_layerwise_disaggregated_edge(input_ids, position_ids, is_prefill, kv_cache,
                                                                        block_tables, slots, input_lengths, max_seq_len,
                                                                        lm_head_indices, **kwargs)

    def forward_mtp(
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
        # prefill
        if is_prefill:
            # llm prefill
            self.init_kvcache(kv_cache)
            acl_inputs, acl_param, _ = self.prepare_inputs_for_ascend(
                input_ids, position_ids, is_prefill, kv_cache,
                block_tables, slots, input_lengths, max_seq_len,
                lm_head_indices, **kwargs)
            logits, hidden_states = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
            llm_logits, llm_hidden_states = logits, hidden_states[lm_head_indices]
            q_lens = kwargs.get(Q_LENS, None)
            logits_mtp, hidden_states_mtp = logits, hidden_states
            acl_inputs_mtp, acl_param_mtp = acl_inputs, acl_param
            self.acl_encoder_operation_mtp.set_kv_cache(self.mtp_k_caches[0: 1],
                                                        self.mtp_v_caches[0: 1])
            if self.prefix_cache_enable:
                self.acl_encoder_operation_prefixcache_mtp.set_kv_cache(self.mtp_k_caches[0: 1], 
                                                                        self.mtp_v_caches[0: 1])

            acl_inputs_mtp = self.delete_local_tp_mtp_inputs(acl_inputs_mtp)
            if q_lens is not None:
                q_lens = torch.tensor(q_lens).npu()
                acl_inputs_mtp, acl_param_mtp = \
                    self.update_mtp_inputs(logits_mtp, hidden_states_mtp, q_lens,
                                           acl_inputs_mtp, acl_param_mtp, 0, kwargs)
            else:
                acl_inputs_mtp, acl_param_mtp = \
                    self.update_mtp_inputs(logits_mtp, hidden_states_mtp, input_lengths,
                                           acl_inputs_mtp, acl_param_mtp, 0, kwargs)
            torch.npu.synchronize()
            
            acl_inputs_mtp = self.add_local_tp_mtp_inputs(acl_inputs_mtp)
            logits_mtp, hidden_states_mtp = \
                self.execute_ascend_operator(acl_inputs_mtp, acl_param_mtp, is_prefill, is_mtp=True)

            if self.distributed_enable:
                llm_logits = self.select_logits(llm_logits, **kwargs)
                llm_hidden_states = self.select_logits(llm_hidden_states, **kwargs)
            return (llm_logits, llm_hidden_states)
        else:
            sub_model_inputs = kwargs.get('sub_model_inputs', None)
            if sub_model_inputs is not None:
                return self.forward_mtp_decoding_v2(input_ids, position_ids, \
                    is_prefill, kv_cache, block_tables, slots, input_lengths, \
                    max_seq_len, lm_head_indices, **kwargs)

            hidden_states = kwargs.get('hidden_states', None)
            is_mtp = hidden_states is not None
            if is_mtp:
                q_lens = kwargs.get(Q_LENS)
                q_lens = torch.tensor(q_lens).npu()
                hidden_states = kwargs.get('hidden_states', None)
                self.init_kvcache(kv_cache)
                acl_inputs, acl_param, _ = self.prepare_inputs_for_ascend(
                    input_ids, position_ids, is_prefill, kv_cache,
                    block_tables, slots, input_lengths, max_seq_len,
                    lm_head_indices, **kwargs)

                # mtp decode
                all_logits_mtp = []
                acl_inputs_mtp, acl_param_mtp = acl_inputs, acl_param
                acl_inputs_mtp = self.delete_local_tp_mtp_inputs(acl_inputs_mtp)
                shard_effective_token_indices = kwargs.get('shard_effective_token_indices')
                if self.eplb_level == EPLBType.FORCE_EPLB:
                    acl_inputs[18] = hidden_states
                else:
                    acl_inputs.insert(18, hidden_states.npu())
                if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
                    map_id = -3 if self.mix_shared_routing else -1 # -3: eplb routing map id
                    acl_inputs[map_id] = acl_inputs[map_id][-1:]

                acl_inputs_mtp = self.add_local_tp_mtp_inputs(acl_inputs_mtp)
                for mtp_idx in range(self.num_speculative_tokens):
                    self.acl_decoder_operation_mtp.set_kv_cache(self.mtp_k_caches[mtp_idx: mtp_idx + 1],
                                                                self.mtp_v_caches[mtp_idx: mtp_idx + 1])
 
                    logits_mtp, hidden_states_mtp = \
                        self.execute_ascend_operator(acl_inputs_mtp, acl_param_mtp, is_prefill, is_mtp=True)
                    if self.mapping.has_attn_cp():
                        logits_mtp = logits_mtp[:logits_mtp.size(0) // self.mapping.attn_cp.group_size]
                        hidden_states_mtp = hidden_states_mtp[:hidden_states_mtp.size(0) // 
                                                              self.mapping.attn_cp.group_size]
                    all_logits_mtp.append(logits_mtp)

                    if mtp_idx < self.num_speculative_tokens - 1:
                        fake_data = False
                        if self.mapping.has_dp() and not ENV.enable_dp_partition_up:
                            lm_head_indices_dp_rank_ids = kwargs.get('lm_head_indices_dp_rank_ids')
                            logits_gather_indices = torch.arange(0, lm_head_indices_dp_rank_ids.shape[0])
                            logits_gather_indices = \
                                    logits_gather_indices[lm_head_indices_dp_rank_ids == self.mapping.attn_dp.rank]
                            if logits_gather_indices.numel() > 0:
                                logits_mtp = logits_mtp[logits_gather_indices]
                                hidden_states_mtp = hidden_states_mtp[shard_effective_token_indices]
                            else:
                                fake_data = True

                        if not fake_data:
                            lm_head_local_dp = kwargs.get('lm_head_local_dp', None)
                            acl_inputs_mtp, acl_param_mtp = \
                            self.prepare_mtp_roll_inputs_for_ascend(
                                acl_inputs_mtp, acl_param_mtp, q_lens,
                                logits_mtp, hidden_states_mtp, False, lm_head_local_dp)

                if self.num_speculative_tokens > 1:
                    all_logits_mtp = torch.stack(all_logits_mtp, dim=1)
                    all_logits_mtp = all_logits_mtp.view(-1, all_logits_mtp.shape[-1])
                else:
                    all_logits_mtp = all_logits_mtp[0]
                return all_logits_mtp
            else:
                q_lens = kwargs.get(Q_LENS)
                self.init_kvcache(kv_cache)
                acl_inputs, acl_param, _ = self.prepare_inputs_for_ascend(
                    input_ids, position_ids, is_prefill, kv_cache,
                    block_tables, slots, input_lengths, max_seq_len,
                    lm_head_indices, **kwargs)
                if lm_head_indices is None:
                    lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                   dtype=torch.int64, device=input_ids.device)
                acl_param = json.dumps({
                    "seqLen": input_lengths.tolist(),
                    SEQUENCE_LENGTH_SP: kwargs.get("input_lengths_sp").tolist()
                                        if self.mapping.has_attn_inner_sp() else [],
                    Q_LEN: [int(i) for i in q_lens if i] if q_lens else None
                })

                if not self.speculate_enable:
                    q_lens = kwargs.get(Q_LENS)
                    q_lens = torch.tensor(q_lens).npu()
                    repeat_input_lengths = input_lengths.to(torch.int32).repeat_interleave(q_lens, dim=0)
                    offset = torch.cat([torch.arange(-i + 1, 1) for i in q_lens]).npu()
                    repeat_input_lengths = repeat_input_lengths + offset
                    acl_inputs[11] = repeat_input_lengths.to(torch.int32)

                    acl_param = json.dumps({
                        "seqLen": acl_inputs[11].tolist(),
                    })


                logits, hidden_states = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
                if self.mapping.has_attn_cp():
                    logits = logits[:logits.size(0) // self.mapping.attn_cp.group_size]
                    hidden_states = hidden_states[:hidden_states.size(0) // self.mapping.attn_cp.group_size]
                if lm_head_indices is not None:
                    if not ((ENV.enable_dp_partition_up or bool(self.distributed_enable)) and 
                             self.enable_lm_head_local_tp):
                        hidden_states = hidden_states[lm_head_indices]
                return (logits, hidden_states)

    def forward_mtp_decoding_v2(
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
        prof = span_start(name="forward_mtp_decoding_v2", level=Level.DETAILED)
        prof = span_attr(prof, "slots", lambda: tensor_attr(slots, False))
        prof = span_attr(prof, "block_tables", lambda: tensor_attr(block_tables, False))
        prof = span_attr(prof, "input_id", lambda: tensor_attr(input_ids, False))

        sub_model_inputs = kwargs.get('sub_model_inputs', None)

        hidden_states = kwargs.get('hidden_states', None)
        q_lens_list = kwargs.get(Q_LENS)
        q_lens = torch.tensor(q_lens_list).npu()

        if lm_head_indices is None:
            lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                           dtype=torch.int64, device=input_ids.device)
        if self.mapping.has_attn_cp():
            lm_head_indices = lm_head_indices.repeat(self.mapping.attn_cp.group_size)
        acl_param = json.dumps({
            SEQUENCE_LENGTH: input_lengths.tolist(),
            SEQUENCE_LENGTH_SP: kwargs.get("input_lengths_sp").tolist()
                                if self.mapping.has_attn_inner_sp() else [],
            "qLen": q_lens_list
        })

        self.init_kvcache(kv_cache)
        acl_inputs_mtp, acl_param_mtp, _ = self.prepare_inputs_for_ascend(
            sub_model_inputs.input_ids, sub_model_inputs.position_ids, is_prefill, kv_cache,
            sub_model_inputs.block_tables, sub_model_inputs.slots, sub_model_inputs.context_length, max_seq_len,
            sub_model_inputs.prefill_head_indices, **kwargs)
        # mtp decode
        all_logits_mtp = []
        acl_inputs_mtp = self.delete_local_tp_mtp_inputs(acl_inputs_mtp)

        shard_effective_token_indices = kwargs.get('shard_effective_token_indices')
        if self.eplb_level == EPLBType.FORCE_EPLB:
            acl_inputs_mtp[18] = hidden_states
        else:
            acl_inputs_mtp.insert(18, hidden_states)

        if self.mapping.has_attn_inner_sp():
            acl_inputs_mtp[30] = kwargs.get("sub_input_lengths_sp")
            acl_inputs_mtp[31] = self.prepare_csp_input_filter_mask(acl_inputs_mtp[30], q_lens_list)

        if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
            map_id = -3 if self.mix_shared_routing else -1 # -3: eplb routing map id
            acl_inputs_mtp[map_id] = acl_inputs_mtp[map_id][-1:]

        self.acl_decoder_operation_mtp.set_kv_cache(self.mtp_k_caches[0: 1], 
                                                    self.mtp_v_caches[0: 1])
        # 提前计算好slot
        slot_list = self.mtp_iter_slot_calc(acl_inputs_mtp[6])

        if self.mapping.has_dp() and not ENV.enable_dp_partition_up:
            lm_head_indices_dp_rank_ids = kwargs.get('lm_head_indices_dp_rank_ids')
            logits_gather_indices = torch.arange(0, lm_head_indices_dp_rank_ids.shape[0])
            logits_gather_indices = \
                logits_gather_indices[lm_head_indices_dp_rank_ids == self.mapping.attn_dp.rank]

        acl_inputs_mtp = self.add_local_tp_mtp_inputs(acl_inputs_mtp)
        for mtp_idx in range(self.num_speculative_tokens):
            acl_inputs_mtp[6] = slot_list[mtp_idx]
            logits_mtp, hidden_states_mtp = \
                self.execute_ascend_operator(acl_inputs_mtp, acl_param_mtp, is_prefill, is_mtp=True)
            if self.mapping.has_attn_cp():
                logits_mtp = logits_mtp[:logits_mtp.size(0) // self.mapping.attn_cp.group_size]
                hidden_states_mtp = hidden_states_mtp[:hidden_states_mtp.size(0) // self.mapping.attn_cp.group_size]
            all_logits_mtp.append(logits_mtp)

            def prof_collect_mtp_attr(input_ids, hidden_states_mtp):
                return {
                    "input_ids": tensor_attr(input_ids),
                    "hidden_states_mtp": [tensor_attr(x) for x in hidden_states_mtp]
                }
            prof = span_attr(prof, "mtp_" + str(mtp_idx),
                partial(prof_collect_mtp_attr, acl_inputs_mtp[0], hidden_states_mtp))

            if mtp_idx < self.num_speculative_tokens - 1:
                torch.npu.current_stream().synchronize()
                fake_data = False
                if self.mapping.has_dp() and not ENV.enable_dp_partition_up:
                    if logits_gather_indices.numel() > 0:
                        logits_mtp = logits_mtp[logits_gather_indices]
                        hidden_states_mtp = hidden_states_mtp[shard_effective_token_indices]
                    else:
                        fake_data = True

                if not fake_data:
                    if self.distributed_enable or not self.mapping.has_dp():
                        lm_head_local_dp = sub_model_inputs.prefill_head_indices
                    else:
                        lm_head_local_dp = kwargs.get('lm_head_local_dp', None)
                    acl_inputs_mtp, acl_param_mtp = \
                    self.prepare_mtp_roll_inputs_for_ascend(
                        acl_inputs_mtp, acl_param_mtp, q_lens,
                        logits_mtp, hidden_states_mtp, False, lm_head_local_dp)

        if self.num_speculative_tokens > 1:
            all_logits_mtp = torch.stack(all_logits_mtp, dim=1)
            all_logits_mtp = all_logits_mtp.view(-1, all_logits_mtp.shape[-1])
        else:
            all_logits_mtp = all_logits_mtp[0]

        # all_logits_mtp的顺序： batch0 (draft_logits0~num_speculative_tokens-1) batch1 batch2...
        # input_ids2的顺序 batch0 (draft_token0~num_speculative_tokens-1) batch1 batch2...
        input_ids2 = torch.argmax(all_logits_mtp, dim=-1)
        if self.distributed_enable or not self.mapping.has_dp():
            input_ids_reshaped = input_ids.view(-1, self.num_speculative_tokens + 1)
            input_ids2_reshaped = input_ids2.view(-1, self.num_speculative_tokens)
            input_ids_reshaped[:, 1:] = input_ids2_reshaped
            input_ids = input_ids_reshaped.flatten()
        else:
            dp_rank_ids = kwargs.get('dp_rank_ids')
            draft_token_indices = torch.where(dp_rank_ids == self.mapping.attn_dp.rank)[0]
            if draft_token_indices.numel() > 0:
                input_ids_reshaped = input_ids.view(-1, self.num_speculative_tokens + 1)
                input_ids2_reshaped = input_ids2.view(-1, self.num_speculative_tokens)
                input_ids_reshaped[:, 1:] = input_ids2_reshaped[draft_token_indices]
                input_ids = input_ids_reshaped.flatten()

        acl_inputs_mtp = self.delete_local_tp_mtp_inputs(acl_inputs_mtp)
        if self.mapping.has_attn_inner_sp() or self.mapping.has_attn_cp():
            acl_inputs = copy.deepcopy(acl_inputs_mtp)
        else:
            acl_inputs = acl_inputs_mtp
        acl_inputs[0] = input_ids
        acl_inputs[1] = position_ids.to(torch.int64)
        acl_inputs[5] = block_tables.to(torch.int32)
        acl_inputs[6] = slots.to(torch.int32)
        acl_inputs[11] = input_lengths.to(torch.int32)
        acl_inputs[12] = lm_head_indices.to(torch.int64)
        if self.mapping.has_attn_inner_sp():
            acl_inputs[30] = kwargs.get("input_lengths_sp")
            acl_inputs[31] = self.prepare_csp_input_filter_mask(acl_inputs[30], q_lens_list)
        elif self.mapping.has_attn_cp():
            acl_inputs[30] = self.prepare_csp_input_filter_mask(acl_inputs[11], q_lens_list)
        del acl_inputs[18]

        if self.eplb_level in [EPLBType.STATIC_EPLB, EPLBType.DYNAMIC_EPLB]:
            map_id = -3 if self.mix_shared_routing else -1 # -3: eplb routing map id
            acl_inputs[map_id] = self.expert_routing_map
            EplbExpertDataCollect().collect_routing_map(self.expert_routing_map, self.mapping.rank)

        acl_inputs = self.add_local_tp_mtp_inputs(acl_inputs)
        logits, hidden_states = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        if not (ENV.enable_dp_partition_up or bool(self.distributed_enable)):
            hidden_states = hidden_states[lm_head_indices]
        if self.mapping.has_attn_cp():
            logits = logits[:logits.size(0) // self.mapping.attn_cp.group_size]
            hidden_states = hidden_states[:hidden_states.size(0) // self.mapping.attn_cp.group_size]

        def prof_collect_exec_attr():
            return {
                "input_ids": tensor_attr(input_ids),
                "hidden_states": [tensor_attr(x) for x in hidden_states]
            }
        prof = span_attr(prof, "execute_ascend_operator", prof_collect_exec_attr)
        span_end(prof)

        return (logits, hidden_states, input_ids2)

    def mtp_iter_slot_calc(self, slot_input):
        slot_list = []
        if self.num_speculative_tokens == 1:
            slot_list.append(slot_input)
            return slot_list
        slot_num_per_batch = self.num_speculative_tokens * 2
        used_slot_num_per_iter = self.num_speculative_tokens + 1
        offsets = torch.arange(used_slot_num_per_iter).npu()
        shift_num = (slot_input.size(0) - used_slot_num_per_iter) // slot_num_per_batch + 1
        shift_values = torch.arange(shift_num).npu()
        for mtp_idx in range(self.num_speculative_tokens):
            if slot_input.size(0) > 1:
                starts = mtp_idx + shift_values * slot_num_per_batch
                valid_starts = starts[starts + used_slot_num_per_iter <= slot_input.size(0)]
                indices = valid_starts.view(-1, 1) + offsets
                slot_new = slot_input[indices].flatten()
                slot_list.append(slot_new)
            else:
                slot_list.append(slot_input)
        return slot_list

    def init_kvcache(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
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
            if (self.soc_info.need_nz or self.enable_nz) and torch_npu.get_npu_format(k_caches[0]) != 29:
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
                else:
                    start_cache_num = self.layerwise.load_list.index(self.layerwise.edge_start_layer_count)
                    end_cache_num = self.layerwise.load_list.index(
                        self.config.num_hidden_layers - self.layerwise.edge_end_layer_count - 1) + 1
                    k_caches_sp2 = [k_caches[i] for i in range(start_cache_num, end_cache_num)]
                    v_caches_sp2 = [v_caches[i] for i in range(start_cache_num, end_cache_num)]
            if not self.layerwise_disaggregated:
                if self.num_speculative_tokens:
                    if self.acl_encoder_operation is not None:
                        self.acl_encoder_operation.set_kv_cache(k_caches[:-1],
                                                                v_caches[:-1])
                    if self.acl_decoder_operation is not None:
                        self.acl_decoder_operation.set_kv_cache(k_caches[:-1],
                                                                v_caches[:-1])
                    if self.acl_dap_operation is not None:
                        self.acl_dap_operation.set_kv_cache(k_caches[:-1],
                                                            v_caches[:-1])
                    if self.prefix_cache_enable:
                        self.acl_encoder_operation_prefixcache.set_kv_cache(k_caches[:-1], 
                                                                            v_caches[:-1])
                    self.mtp_k_caches = k_caches[-1:]
                    self.mtp_v_caches = v_caches[-1:]
                else:
                    if self.acl_encoder_operation is not None:
                        self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
                    if self.acl_decoder_operation is not None:
                        self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
                    if self.acl_dap_operation is not None:
                        self.acl_dap_operation.set_kv_cache(k_caches, v_caches)
                    if self.prefix_cache_enable:
                        self.acl_encoder_operation_prefixcache.set_kv_cache(k_caches, v_caches)
            else:
                if self.layerwise.split_type == DistributedType.EDGE:
                    self.acl_head_encoder_operation.set_kv_cache(k_caches_sp1, v_caches_sp1)
                    self.acl_head_decoder_operation.set_kv_cache(k_caches_sp1, v_caches_sp1)
                    self.acl_head_encoder_operation_prefixcache.set_kv_cache(k_caches_sp1, v_caches_sp1)
                    self.acl_tail_encoder_operation.set_kv_cache(k_caches_sp3, v_caches_sp3)
                    self.acl_tail_decoder_operation.set_kv_cache(k_caches_sp3, v_caches_sp3)
                    self.acl_tail_encoder_operation_prefixcache.set_kv_cache(k_caches_sp3, v_caches_sp3)
                else:
                    self.acl_internal_decoder_operation.set_kv_cache(k_caches_sp2, v_caches_sp2)
                    for layer in range(self.config.num_hidden_layers - self.layerwise.edge_start_layer_count - \
                                       self.layerwise.edge_end_layer_count):
                        self.encode_op_list[layer].set_kv_cache([k_caches[layer]], [v_caches[layer]])
                        self.encode_op_prefix_cache_list[layer].set_kv_cache([k_caches[layer]], [v_caches[layer]])

            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def unwrap_model_state_dict(self, state_dict: dict) -> dict:
        new_state_dict = {}
        for name, tensor in state_dict.items():
            new_name = name.replace('.linear', '')
            new_name = new_name.replace('.mtp_layer.', f'.layers.{self.config.num_hidden_layers}.')
            new_name = new_name.replace('.mtp_', f'.layers.{self.config.num_hidden_layers}.')
            new_name = new_name.replace('.eh_proj', f'.layers.{self.config.num_hidden_layers}.eh_proj')
            new_name = new_name.replace('.shared_head_norm', 
                                        f'.layers.{self.config.num_hidden_layers}.shared_head.norm')
            new_state_dict[new_name] = tensor
        return new_state_dict

    def get_module_save_dir(self, tensor_name):
        if "norm" in tensor_name:
            return "norm"
        if "self_attn" in tensor_name:
            return "attn"
        layer_num = int(tensor_name.split(".")[2]) if "layers" in tensor_name else 0
        if ("mlp" in tensor_name) and layer_num > 2:
            return "moe"
        if ("mlp" in tensor_name) and layer_num <= 2:
            return "dense"
        return "model"

    def generate_description(self, save_directory: Optional[str] = None):
        """Generate description file of saved quant model."""
        model_description = {}
        state_dict = self.unwrap_model_state_dict(self.state_dict())
        quantize_type = self.quantize.upper()
        model_description['model_quant_type'] = quantize_type
        model_description['version'] = getattr(self, "quant_version", "0.0.0")

        leaf_modules = get_leaf_modules_recursive(self)
        leaf_modules = self.unwrap_model_state_dict(leaf_modules)
        for name in state_dict.keys():
            parent_module_name = ".".join(name.split(".")[:-1])
            parent_module = leaf_modules.get(parent_module_name)
            model_description[name] = get_module_quant_type(parent_module, name, quantize_type)

        model_description.update(asdict(self.config.quantization_config))
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            save_directory = os.path.realpath(save_directory)
            if self.quant_version == "0.0.0":
                save_path = os.path.join(save_directory, f'quant_model_description_{quantize_type.lower()}.json')
            else:
                save_path = os.path.join(save_directory, 'quant_model_description.json')
            with file_utils.safe_open(save_path, 'w', encoding='utf-8', is_exist_ok=True) as fw:
                json.dump(model_description, fw, indent=4)
        return model_description