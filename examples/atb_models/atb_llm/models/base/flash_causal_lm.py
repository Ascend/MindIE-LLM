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
import math
from abc import abstractmethod
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np
import torch
import torch_npu
from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.eplb_expert_data_collect import EplbExpertDataCollect
from atb_llm.utils.moe_utils import EPLBType
from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType
from .model_utils import BaseModel
from ...models import InferenceMode
from ...utils.env import ENV
from ...utils.log import logger, print_log
from ...utils.initial import load_atb_speed, NPUSocInfo, is_lcoc_enable
from ...utils.layers import PositionRotaryEmbedding, AttentionMask
from ...utils.op_backend import OpBackend
from ...utils.adapter_manager import AdapterIdsType
from ...utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from ...utils.layers.norm.fast_layer_norm import NormType
from ...utils.layers.embedding.position_rotary_embedding import PositionEmbeddingType
from ...utils.weights import Weights


class DistributedType(str, Enum):
    EDGE = "master"
    CLOUD = "slave"


class LwdLayerStatus(int, Enum):
    EDGE_START_LAYER = 0,
    CLOUD_MIDDLE_LAYER = 1,
    EDGE_END_LAYER = 2,


class LayerWiseAttr:

    __slot__ = ["edge_start_layer_count", "edge_end_layer_count", "split_type", "load_list", "ascend_weight_head",
                "ascend_weight_tail", "ascend_weight_internal", "acl_inputs_prefill", 
                "acl_inputs_decode", "acl_param_prefill", "acl_param_decode", "p_out_hidden",
                "weight_wrappers", "num_hidden_layers", "acl_inputs_prefill_queue", "acl_param_prefill_queue"]

    def __init__(self, edge_start_layer_count, edge_end_layer_count, split_type):
        self.edge_start_layer_count = edge_start_layer_count
        self.edge_end_layer_count = edge_end_layer_count
        self.split_type = split_type

    
class FlashForCausalLM(BaseModel):
    """
    Base class for causal language model using paged attention, built with Cpp graph.

    Args:
        config (PretrainedConfig): The configuration for the model.
        weights (Weights): The weights for the model.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, config: PretrainedConfig, weights: Weights, **kwargs):
        super().__init__()
        load_atb_speed()
        self.model = None
        self.lm_head = None
        self.config = config
        self.soc_info = NPUSocInfo()
        self.llm_config = kwargs.get('llm_config', None)
        self.acl_decoder_flashcomm_operation = None
        self.acl_encoder_flashcomm_operation = None

        self.layerwise_disaggregated = False
        self.layerwise = None  # A set of attributes exclusive to the layerwise_disaggregated scenario
        layerwise_disaggregated = kwargs.get("layerwise_disaggregated", False)
        if layerwise_disaggregated:
            split_type = None
            edge_start_layer_count = 1
            edge_end_layer_count = 1
            layerwise_disaggregated_role_type = kwargs.get("layerwise_disaggregated_role_type", "")
            self.layerwise_disaggregated = True
            if layerwise_disaggregated_role_type == "slave":
                split_type = DistributedType.CLOUD
            else:
                split_type = DistributedType.EDGE
            self.layerwise = LayerWiseAttr(edge_start_layer_count, edge_end_layer_count, split_type)
         
        self.inference_mode = kwargs.get("inference_mode")
        self.mempool_type: MemPoolType = kwargs.get('mempool_type', MemPoolType.DISABLED)
        self.num_attention_heads = config.num_attention_heads
        if hasattr(config, 'num_key_value_heads'):
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads

        if hasattr(config, 'rope_theta'):
            self.rope_theta = config.rope_theta
        else:
            self.rope_theta = 10000.0
        if hasattr(config, 'rope_scaling') and self.config.rope_scaling is not None:
            self.scaling_factor = self.config.rope_scaling.factor
        else:
            self.scaling_factor = 1.0
        self.hidden_size = config.hidden_size
        self.head_size = config.head_dim if ("qwen3" in config.model_type and hasattr(config, "head_dim")) \
            else self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.device = weights.device
        self.mapping = weights.mapping
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.quant_version = "0.0.0" if weights.quant_desc is None else weights.quant_desc.get("version", "0.0.0")
        self.lcoc_enable = is_lcoc_enable(self.soc_info.need_nz)
        if self.llm_config is not None:
            self.lcoc_enable = self.lcoc_enable and self.llm_config.llm.ccl.enable_mc2
        self.enable_flash_comm = False
        if (self.inference_mode is not None and
            hasattr(self.inference_mode, 'enable_decode_pa')):
            self.speculate_enable = self.inference_mode.enable_decode_pa
        else:
            self.speculate_enable = self.inference_mode == InferenceMode.SPECULATE
        if self.inference_mode and hasattr(self.inference_mode, 'enable_prefill_pa'):
            self.prefix_cache_enable = self.inference_mode.enable_prefill_pa
        else:
            self.prefix_cache_enable = self.inference_mode == InferenceMode.PREFIXCACHE
        self.compress_head_enable = ENV.compress_head_enable
        self.omni_attention_enable = ENV.omni_attention_enable
        self.enable_greedy_search_opt = ENV.enable_greedy_search_opt
        if self.tp_world_size == 1 or self.soc_info.need_nz:
            self.enable_greedy_search_opt = False
        self.split_fuse_enable = self.inference_mode == InferenceMode.SPLITFUSE
        if self.config.quantization_config.reduce_quant_type is not None:
            if self.tp_world_size <= 1:
                self.config.quantization_config.reduce_quant_type = None
            else:
                self.lcoc_enable = False
        # if num_key_value_heads is nondivisible
        if self.num_key_value_heads < self.tp_world_size:
            repeat_times = self.tp_world_size // self.num_key_value_heads
        else:
            repeat_times = 1
        self.num_attention_heads = (self.num_attention_heads + self.tp_world_size - 1) // self.tp_world_size
        self.num_key_value_heads = (self.num_key_value_heads * repeat_times + self.tp_world_size - 1) \
            // self.tp_world_size

        self.rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=self.rope_theta,
                                                               device="cpu", scaling_factor=self.scaling_factor) \
            .to(self.device)
        self.max_position_embeddings = config.max_position_embeddings
        self.quantize = config.quantize
        self.dtype = weights.dtype

        self.max_base_len = 128
        self.attn_mask = AttentionMask.static(self.max_base_len, dtype=self.dtype)

        self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")

        # for ascend init
        self.acl_encoder_operation = None
        self.acl_decoder_operation = None
        self.acl_dap_operation = None
        self.enable_dap = False if self.llm_config is None else self.llm_config.llm.stream_options.micro_batch
        if self.enable_dap:
            torch.classes.ModelTorch.Context.enable_cache_workspace()
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.ascend_kcache_shape = None
        self.ascend_vcache_shape = None

        self.acl_encoder_operation_inputs: list[None | torch.Tensor] \
            = [None] * self.get_in_tensor_size(encoder=True)
        self.acl_decoder_operation_inputs: list[None | torch.Tensor] \
            = [None] * self.get_in_tensor_size(encoder=False)

        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int).to(self.device)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).to(self.device)
        self.attn_mask_fake = self.attn_mask \
            .get_attn_mask(1, dtype=self.dtype, device="cpu") \
            .to(self.device)

        self.acl_param = None
        self.cos_embed = None
        self.sin_embed = None

        self.adapter_manager = None
        self.num_lora_weight_per_layer = 14
        self.adapter_ids = None

        self.attn_wrapper = AttnWrapper(
            norm_name='input_layernorm',
            wrapper_name='self_attn',
            pack_name='query_key_value',
            sep_names=['q_proj', 'k_proj', 'v_proj'],
            o_name='o_proj'
        )
        self.mlp_wrapper = MlpWrapper(
            norm_name='post_attention_layernorm',
            wrapper_name='mlp',
            pack_name='gate_up_proj',
            sep_names=['gate_proj', 'up_proj'],
            down_name='down_proj'
        )
        self.attn_decode_backend = OpBackend.ACLNN if self.config.quantization_config.kv_quant_type is not None \
                                   else OpBackend.ATB
        if self.attn_decode_backend == OpBackend.ACLNN:
            print_log(self.tp_rank, logger.warning,
                      "If the model's max_positional_embedding is large, "
                      "AclNN attention backend may result in NPU out of memory.")
        if self.omni_attention_enable:
            patten_path = ENV.omni_attention_pattern_file
            self.pattern = np.loadtxt(patten_path).astype(bool)
            self.pattern_mask = self.pattern[:, self.tp_rank:self.tp_rank + 1].flatten().tolist()
        self.warmup_is_end = True
        self.kvcache_quant_layers = []
        self.total_prefill_token_num_per_expert = None
        self.total_decode_token_num_per_expert = None
        self.enable_swiglu_quant = False

        self.enable_model_obfuscation = False if self.llm_config is None \
            else self.llm_config.llm.pmcc_obfuscation_options.enable_model_obfuscation
        self.obfuscation_fd = 0 # handle for pmcc model obfuscation
        if self.enable_model_obfuscation:
            obfuscation_setup_operation = torch.classes.OperationTorch.OperationTorch("ObfuscationSetupOperation")
            obfuscation_setup_param = json.dumps({
                "dataType": 1 if self.dtype == torch.float16 else 27,  # 1: float16, 27: bfloat16
                "hiddenSizePerRank": self.num_attention_heads * self.head_size,
                "tpRank": self.mapping.attn_tp.rank,
                "cmd": 1, # 1: float mode; 2: w8a8 mode
                "obfCoefficient": kwargs.get('obfCoefficient', 1.0)
            })
            obfuscation_setup_operation.set_param(obfuscation_setup_param)
            self.obfuscation_fd = int(obfuscation_setup_operation.execute([])[0].cpu().numpy()[0])
        self.is_multimodal = False

    @staticmethod
    def update_adapter_weights(adapter_weights: list[torch.Tensor], in_tensor: list[torch.Tensor], start_idx: int):
        """Update adapter weights."""
        for i, weight in enumerate(adapter_weights):
            # 这里+1是需要跳过seq_len_cum_sum
            in_tensor[start_idx + 1 + i] = weight

    def update_adapter_manager(self):
        """Update adapter manager."""
        self.adapter_manager.base_model = self
        # +1 是因为Lora旁路需要多一个seq_len_cum_sum入参
        self.acl_encoder_operation_inputs.extend([None] * (self.num_lora_weight_per_layer * self.num_layers + 1))
        self.acl_decoder_operation_inputs.extend([None] * (self.num_lora_weight_per_layer * self.num_layers + 1))

    def get_in_tensor_size(self, encoder: bool = True) -> int:
        """Get input tensor size."""
        return 9

    def weight_format_cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast weight to nz format if based on SOC info."""
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        print_log(self.tp_rank, logger.info, f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def process_adapter_ids(self, adapter_ids: None | List[str | None]) -> List[str]:
        """Preprocess adapter ids."""
        if self.adapter_manager is None:
            return []
        effective_adapter_ids = self.adapter_manager.preprocess_adapter_ids(adapter_ids)
        return effective_adapter_ids

    def prepare_adapter_weights(self, adapter_ids: None | List[str | None]) -> List[torch.Tensor]:
        """Prepare adapter weights."""
        need_update = self.adapter_manager.update_adapter(adapter_ids)
        # 不需要更新adapter weights的场景
        if not need_update:
            return []
        # 更新adapter weights
        return self.adapter_manager.get_adapters(adapter_ids)

    def calculate_adapter_group_size(
            self, adapter_ids: None | List[str | None],
            input_lengths: torch.Tensor, is_prefill: bool = False
    ) -> torch.Tensor:
        """Calculate the adapter group size."""
        if len(adapter_ids) == 1:
            return self.placeholder
        elif self.adapter_manager.previous_adapter_ids.record_type == AdapterIdsType.MIXED:
            if is_prefill:
                cum_group_size = torch.cumsum(input_lengths, dim=0, dtype=torch.int64)
            else:
                cum_group_size = torch.arange(1, input_lengths.shape[0] + 1, dtype=torch.int64, device=self.device)
        else:
            active_adapters_count = len(self.adapter_manager.adapter_info_registry) - 1  # exclude *sort
            adapter_indexes = []
            for adapter_id in adapter_ids:
                adapter_indexes.append(self.adapter_manager.adapter_info_registry.get(adapter_id).idx)
            labels = torch.tensor(adapter_indexes, device=self.device, dtype=torch.int64)
            unique_labels = torch.arange(0, active_adapters_count, dtype=torch.int64, device=self.device)
            group = torch.zeros_like(unique_labels).scatter_add_(0, labels, input_lengths.to(torch.int64))
            cum_group_size = torch.cumsum(group, dim=0, dtype=torch.int64)
        return cum_group_size

    @abstractmethod
    def init_ascend_operations(self, config: PretrainedConfig):
        """Abstract method to initialize Ascend operations."""
        pass

    @abstractmethod
    def init_ascend_weight(self):
        """Abstract method to initialize Ascend weights."""
        pass

    def get_weight_wrapper(self) -> WeightWrapper:
        """Get weight and regist embedding, layer, quant (if needed), norm and lmhead."""
        weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank, self.attn_wrapper, self.mlp_wrapper)
        weight_wrapper.register_embedding(self.model.embed_tokens)
        for i in range(self.num_layers):
            layer = self.model.layers[i]
            weight_wrapper.register_layer(layer, self.quantize)
            if self.config.quantization_config.kv_quant_type is not None:
                weight_wrapper.register_layer_kvquant(layer)
            if self.config.quantization_config.reduce_quant_type is not None:
                weight_wrapper.register_layer_reducequant(layer)
        weight_wrapper.register_model_norm(self.model.norm)
        weight_wrapper.register_model_lmhead(self.lm_head)
        return weight_wrapper

    def get_coder_param(self) -> Tuple[dict, dict]:
        """Set coder param and get encoder/decoder params."""
        weight_wrapper = self.get_weight_wrapper()
        self.ascend_weight = weight_wrapper.weights
        linear_types = weight_wrapper.linear_type
        pack_quant_configs = weight_wrapper.pack_quant_type
        linear_transpose_types = weight_wrapper.linear_transpose_types
        # 设置模型参数
        rank_table_file = ENV.rank_table_file
        if self.position_embedding_type == "ROPE":
            position_embedding_type = PositionEmbeddingType.ROPE
        else:
            position_embedding_type = PositionEmbeddingType.ALIBI
        coder_param = {
            "normEps": self.config.rms_norm_eps,
            "enableAddNorm": False,
            "normType": NormType.RMS_NORM,
            "numAttentionHeadsPerRank": self.num_attention_heads,
            "hiddenSizePerAttentionHead": self.head_dim,
            "numHiddenLayers": self.config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isBF16": self.dtype == torch.bfloat16,
            "packQuantType": pack_quant_configs,
            "linearQuantType": linear_types,
            "linearTransposeType": linear_transpose_types,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": True,
            "lmHeadTransposeType": self.lm_head.linear.trans_flag,
            "supportSwiGLU": False,
            "kvQuant": self.config.quantization_config.kv_quant_type is not None,
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "hccl" if self.soc_info.need_nz or rank_table_file else "lccl",
            "rankTableFile": rank_table_file,
            "positionEmbeddingType": position_embedding_type,
            "isUnpadInputs": True,
        }

        encoder_param = {
            **coder_param, "isPrefill": True,
            "supportLcoc": self.lcoc_enable,
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "supportLcoc": False
        }
        return encoder_param, decoder_param

    def get_adapter_ids(self, **kwargs):
        """Get adapter ids from keywords."""
        if self.adapter_manager is not None:
            self.adapter_ids = kwargs.get("adapter_ids")

    def init_position_rotary_embedding(self, position_ids: torch.Tensor, max_seq_len: int):
        """Initialze rope."""
        self.rotary_embedding.update_cos_sin_cache_total(self.dtype, self.device, max_seq_len)
        if self.num_attention_heads == self.num_key_value_heads:
            self.cos_embed, self.sin_embed = self.rotary_embedding.get_cos_sin_cached_total(position_ids)
        else:
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()

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
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            if self.acl_dap_operation is not None:
                self.acl_dap_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            self.ascend_kcache_shape = kv_cache[0][0].shape
            self.ascend_vcache_shape = kv_cache[0][1].shape
            print_log(self.tp_rank, logger.info,
                      f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None,
                                  **kwargs) -> Tuple[list, str]:
        """
        Prepare the inputs for Ascend acl operation graph.

        Args:
            input_ids (torch.Tensor): The input tensor.
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
            list: A list of Ascend acl encoder operation inputs.
            str: A json formatted string contains operation parameters.
        """
        self.init_position_rotary_embedding(position_ids, max_seq_len)
        if is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = atten_mask.view(1, pad_maxs, pad_maxs // 16, 16).transpose(1, 2)
                torch_npu.npu_format_cast_(atten_mask, 29)
            else:
                atten_mask = self.attn_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                          kv_cache[0][0].device)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = self.cos_embed
            self.acl_encoder_operation_inputs[3] = self.sin_embed
            if self.dtype == torch.bfloat16:
                self.acl_encoder_operation_inputs[4] = torch.where(atten_mask == -torch.inf, 1, atten_mask)
            else:
                self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[8] = lm_head_indices.to(torch.int64)
            return self.acl_encoder_operation_inputs, self.acl_param
        else:
            self.acl_param = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = self.cos_embed
            self.acl_decoder_operation_inputs[3] = self.sin_embed
            if self.dtype == torch.bfloat16:
                self.acl_decoder_operation_inputs[4] = torch.zeros(input_lengths.size(0),
                                                                   self.num_attention_heads,
                                                                   1, input_lengths.max(),
                                                                   dtype=self.dtype,
                                                                   device=input_ids.device)
            else:
                self.acl_decoder_operation_inputs[4] = self.attn_mask_fake
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param

    def execute_ascend_operator(self,
                                acl_inputs: list,
                                acl_param: str,
                                is_prefill: bool) -> torch.Tensor:
        """Execute the Ascend acl operator."""
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        eplb_level = getattr(self.config, "eplb_level", 0)
        if ENV.enable_expert_hotpot_gather or eplb_level == EPLBType.DYNAMIC_EPLB:
            acl_model_out = EplbExpertDataCollect().split_eplb_expert_data(acl_model_out)
        try:
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
        acl_model_out = self.acl_dap_operation.execute(acl_inputs, acl_param)
        if len(acl_model_out) != 2:
            raise RuntimeError("Number of output tensors is not equal to the expected value.")
        return acl_model_out

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
            reset=False,
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


        if reset:
            print_log(self.tp_rank, logger.info, f"flash_causal_lm reset: {reset}")
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices, **kwargs)

        logits = self.execute_ascend_operator(acl_inputs, acl_param, is_prefill)
        return logits

    def dap_forward(
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
        preceder_inputs = self.prepare_inputs_for_ascend(
            input_ids[0], position_ids[0], is_prefill[0], kv_cache,
            block_tables[0], slots[0], input_lengths[0], max_seq_len[0],
            lm_head_indices[0], **dap_kwargs[0])
        acl_inputs, acl_param = preceder_inputs[0], preceder_inputs[1]
        successor_inputs = self.prepare_inputs_for_ascend(
            input_ids[1], position_ids[1], is_prefill[1], kv_cache,
            block_tables[1], slots[1], input_lengths[1], max_seq_len[1],
            lm_head_indices[1], **dap_kwargs[1])
        acl_inputs_successor, acl_param_successor = successor_inputs[0], successor_inputs[1]
        # When successor's inputs is less than preceder, successor will be rolled back as preceder
        if len(acl_inputs_successor) < len(acl_inputs):
            acl_inputs = acl_inputs[:len(acl_inputs_successor)]
            acl_param = dict(
                (k, acl_param[k])
                for k in acl_param_successor.keys()
            )
        all_inputs.extend(acl_inputs)
        all_inputs.extend(acl_inputs_successor)
        acl_param_dict = json.loads(acl_param)
        for k, v in json.loads(acl_param_successor).items():
            acl_param_dict[f"{k}_successor"] = v
        logits = self.execute_dap_ascend_operator(
            all_inputs, json.dumps(acl_param_dict), is_prefill[0])
        return logits

    def wait_model_event(self, event_pipe_key: str):
        torch.classes.ModelTorch.Event.wait(event_pipe_key)

    def record_model_event(self, event_pipe_key: str):
        torch.classes.ModelTorch.Event.record(event_pipe_key)
