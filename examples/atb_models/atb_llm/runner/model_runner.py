# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Dict, Iterable, List, Optional, Union
from enum import Enum
import os
import json
import time
import inspect
from functools import wraps

import torch
from torch import nn
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

import atb_llm.nn.distributed as dist
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.functional import gather, split
from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType

from ..models import get_model
from ..models.base.config import LoraModelConfig
from ..models.base.mindie_llm_config import MindIELLMConfig
from ..utils import file_utils
from ..utils.configuration_utils import GraphType
from ..utils.dist import initialize_distributed
from ..utils.weights import Weights
from ..utils.loader.safetensor_file_loader import SafetensorFileLoader
from ..utils.env import ENV
from ..utils.cpu_binding import bind_cpus
from ..utils.initial import NPUSocInfo, load_atb_speed
from ..utils.log import logger, print_log, message_filter
from ..utils.log.error_code import ErrorCode
from ..utils.adapter_manager import AdapterManager
from ..utils.mapping import Mapping
from ..utils.memory_utils import check_npu_mem
from ..utils.quantize.quant_type import QuantType
from ..utils.layerwise_disaggregated.edge_cloud_data_comm import EdgeCloudDataComm
from ..utils.layerwise_disaggregated.edge_cloud_ctrl_comm import EdgeCloudCtrlComm
from ..utils.layerwise_disaggregated.cloud_cut_policy import CloudCutPolicy
from ..utils.layerwise_disaggregated.chunk_prefill_policy import ChunkPrefilPolicy


MAX_POSITION_NUM = 2048
OUT_HIDDEN = "out_hidden"
TLS_ENABLE = "tls_enable"
TLS_CA_PATH = "tls_ca_path"
TLS_CA_FILE = "tls_ca_file"
TLS_CERT = "tls_cert"
TLS_PK = "tls_pk"
TLS_CRL_PATH = "tls_crl_path"
TLS_CRL_FILES = "tls_crl_files"


PREALLOC_SUPPORTED_QUANT_TYPES = {
    QuantType.FLOAT,
    QuantType.W8A8,
    QuantType.W8A8_DYNAMIC,
    QuantType.W8A8_PDMIX,
    QuantType.W8A8_MIX,
    QuantType.W8A8SC,
}


# Allow tensor initialization and casting with internal format(e.g., NZ)
torch.npu.config.allow_internal_format = True


class TruncationSide(int, Enum):
    DISABLE = 0
    LEFT = 1
    RIGHT = -1


# 专用于mempool异步分层传输特性的event pipe_key
def generate_mem_pool_event_key(only_save_kv: bool) -> str:
    return "only_save_kv" if only_save_kv else "both_save_kv"


def exception_handler(cls):
    """
    Class decorator for ModelRunner that applies various handlers to methods.
    Currently applies:
    1. _torch_oom_handler: Catches and logs PyTorch OOM errors.
    """

    def _torch_oom_handler(func):
        """Handler specifically for PyTorch OOM errors."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle PyTorch OOM (Only supports Torch >= 2.6 native exception)
                # If torch version is 2.1 or lower, please check exception message directly.
                if hasattr(torch, "OutOfMemoryError") and isinstance(e, torch.OutOfMemoryError):
                    error_msg = (
                        "Device out of memory (OOM) reported by PyTorch, but it can possibly triggered by HCCL. "
                        "Enable logs: export ASCEND_SLOG_PRINT_TO_STDOUT=1, "
                        "export ASCEND_GLOBAL_LOG_LEVEL=3 to check if there's HCCL error messages"
                    )
                    error_code = ErrorCode.ATB_MODELS_OUT_OF_MEMORY
                    logger.error(error_msg, error_code)
                    raise RuntimeError(f"{error_msg}. Error_code: {error_code}") from e
                raise

        return wrapper

    def _apply_handlers(func):
        """Apply the chain of handlers to a function."""
        return _torch_oom_handler(func)

    def _is_target_method(name):
        """Filter methods that need handling."""
        if name == "generate_position_ids":
            return False
        elif name.startswith("__"):
            return False
        return name.startswith("forward") or name.startswith("dap_forward") or name in ["generate", "load_weights"]

    for name, method in list(cls.__dict__.items()):
        if not _is_target_method(name):
            continue

        if inspect.isfunction(method):
            setattr(cls, name, _apply_handlers(method))
        elif isinstance(method, classmethod):
            setattr(cls, name, classmethod(_apply_handlers(method.__func__)))
        elif isinstance(method, staticmethod):
            setattr(cls, name, staticmethod(_apply_handlers(method.__func__)))
    return cls


@exception_handler
class ModelRunner:
    """
    Class for running model.

    Class attributes:
        model (nn.Module, optional): Model instance, defaults to None.
        soc_info (NPUSocInfo, optional): SOC info instance, defaults to None.
        head_size (int, optional): Head size of multi-head attention, defaults to None.
        num_heads (int, optional): Number of head of multi-head attention, defaults to None.
        num_kv_heads (int, optional): Number of key-value heads, defaults to None.
        num_layers (int, optional): Number of layers, defaults to None.
        device (torch.device, optional): Device to run the model, defaults to None.
        dtype (torch.dtype, optional): Dtype of data, defaults to None.
        k_head_size (int, optional): Head size of key head in multi-head attention, defaults to None.
        v_head_size (int, optional): Head size of value head in multi-head attention, defaults to None.

    Args:
        model_name_or_path (str): Model name or path.
        rank (int): Rank of current process.
        world_size (int): World size of multi process.
        npu_id (int, optional): NPU id of current process, defaults to None.
        local_rank (int, optional): Local rank of current process, defaults to None.
        is_flash_causal_lm (bool, optional): Whether to use flash causal lm, defaults to True.
        load_tokenizer (bool, optional): Whether to load tokenizer, defaults to True.
        max_position_embeddings (int, optional): Max positionembeddings, defaults to None.
        tokenizer_path (str, optional): Tokenizer path, defaults to None.
        llm_config_path: (str, optional): Llm config path, defaults to None.
        **kwargs (dict, optional): Additional keyword arguments.
    """

    model: Optional[nn.Module] = None
    soc_info: Optional[NPUSocInfo] = None
    head_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    num_layers: Optional[int] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    k_head_size: Optional[int] = None
    v_head_size: Optional[int] = None

    def __init__(
        self,
        model_name_or_path: str,
        rank: int,
        world_size: int,
        npu_id: Optional[int] = None,
        local_rank: Optional[int] = None,
        is_flash_causal_lm: bool = True,
        load_tokenizer: bool = True,
        max_position_embeddings: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        enable_edge: bool = False,
        llm_config_path: str = None,
        models_dict: dict = None,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.world_size = world_size
        self.enable_edge = enable_edge
        self.inference_mode = kwargs.get("inference_mode")
        self.plugin_params = kwargs.get("plugin_params", None)
        self.enable_atb_torch = kwargs.get("enable_atb_torch", False)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.num_speculative_tokens = kwargs.get("num_speculative_tokens", None)
        self.distributed_enable = kwargs.get("distributed_enable", False)
        self.max_batch_size = kwargs.get("max_batch_size", -1)
        self.model_role = kwargs.get("model_role", "standard")

        self.process_group, self.device = initialize_distributed(self.rank, self.npu_id, world_size)

        self.layerwise_disaggregated = kwargs.get("layerwise_disaggregated", False)

        if self.layerwise_disaggregated:
            self.isqwenvl = False
            if "Qwen2.5-VL" in model_name_or_path:
                print_log(rank, logger.info, "is qwenvl")
                self.isqwenvl = True
            self.layerwise_disaggregated_role_type = kwargs.get("layerwise_disaggregated_role_type", "")
            self.models_dict = models_dict
            self.deserialized_model_data = None

            try:
                self.deserialized_model_data = json.loads(self.models_dict)
            except json.JSONDecodeError:
                print_log(rank, logger.info, "deserialized json decode error")
            except Exception:
                print_log(rank, logger.info, "deserialized unknow error")

        self.mempool_type = kwargs.get("mempool_type", MemPoolType.DISABLED)
        load_atb_speed()

        if ENV.bind_cpu:
            try:
                bind_cpus(self.npu_id, ratio=1.0)
            except ValueError:
                # ValueError indicates a likely configuration issue;
                # use a stronger message to alert the user.
                logger.warning("CPU binding failed, please check configuration. Reluctantly skip binding cpu.")
            except Exception as _:
                logger.warning("Skip binding cpu.")

        router_ins = get_model(
            model_name_or_path,
            is_flash_causal_lm=is_flash_causal_lm,
            load_tokenizer=load_tokenizer,
            max_position_embeddings=max_position_embeddings,
            revision=None,
            tokenizer_path=tokenizer_path,
            trust_remote_code=self.trust_remote_code,
            enable_atb_torch=self.enable_atb_torch,
            enable_edge=self.enable_edge,
            llm_config_path=llm_config_path,
            models_dict=models_dict,
            sub_model_path=f"part{self.local_rank}-of-{self.world_size}",
        )

        if self.layerwise_disaggregated:
            setattr(router_ins, "layerwise_disaggregated", True)

        self.model_cls = router_ins.model_cls
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.postprocessor = router_ins.postprocessor
        self.config_dict = router_ins.config_dict
        self.enable_atb_torch = router_ins.enable_atb_torch
        self.llm_config = router_ins.llm_config

        if hasattr(self.config, "max_position_embeddings"):
            self.max_position_embeddings = self.config.max_position_embeddings
        else:
            self.max_position_embeddings = MAX_POSITION_NUM
        self.dtype = self.config.torch_dtype
        self.quantize = self.config.quantize
        self.is_gqa = (
            hasattr(self.config, "num_key_value_heads")
            and hasattr(self.config, "num_attention_heads")
            and self.config.num_attention_heads != self.config.num_key_value_heads
        )
        self.kv_quant_type = self.config.quantization_config.kv_quant_type
        self.fa_quant_type = self.config.quantization_config.fa_quant_type
        self.kv_cache_dtype = (
            torch.int8 if self.kv_quant_type is not None or self.fa_quant_type == "FAQuant" else self.dtype
        )
        self.prealloc_weight_mem_on_npu = (
            router_ins.prealloc_weight_mem_on_npu  # Router allows memory preallocation
            and not self.layerwise_disaggregated  # Layerwise disaggregation is disabled
            and self._is_quantization_supported()  # Quantization configuration check
            and (self.kv_cache_dtype != torch.int8)  # KV cache is not int8 type
        )
        self.enable_nz = self.llm_config.llm.kv_cache_options.enable_nz
        self.truncation_method = getattr(self.llm_config.models, "truncation", TruncationSide.RIGHT)

        if self.dtype not in [torch.float16, torch.bfloat16]:
            error_msg = (
                "`torch_dtype` is only supported for type `float16` and"
                " `bfloat16`, loaded from config.json -> torch_dtype. "
                "The specific types supported by each model are different, please refer to the model README file."
            )
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)

        if self.layerwise_disaggregated:
            if self.llm_config.llm.plugins and self.llm_config.llm.plugins.plugin_type:
                if self.llm_config.llm.plugins.plugin_type != "":
                    error_msg = "layerwiseDisaggregated not incompatible with llm_config.plugins.plugin_type"
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(f"{error_msg} ")

            if self.llm_config.llm.stream_options and self.llm_config.llm.stream_options.micro_batch:
                if self.llm_config.llm.stream_options.micro_batch != "false":
                    error_msg = "layerwiseDisaggregated not incompatible with lllm_config.stream_options.micro_batch"
                    logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_INVALID)
                    raise ValueError(f"{error_msg} ")

        logger.info(message_filter(f"model_runner.config: {self.config}"))

        self.adapter_manager = None
        self.lora_model_config = None
        self.lora_adapter = None
        self.attn_mask = None
        lora_modules = kwargs.get("lora_modules", None)
        lora_adapter_json_path = os.path.join(model_name_or_path, "lora_adapter.json")
        if lora_modules is not None:
            self.lora_adapter = json.loads(lora_modules)
        elif os.path.exists(lora_adapter_json_path):
            print_log(
                rank,
                logger.warning,
                "The usage of lora_adapter.json will be depreciated by 2026/06/30. "
                "Please specifying LoRA modules within ${MINDIE_LLM_HOME_PATH}/conf/config.json.",
            )
            lora_adapter_json_path = file_utils.standardize_path(lora_adapter_json_path, check_link=False)
            file_utils.check_file_safety(lora_adapter_json_path)
            with file_utils.safe_open(lora_adapter_json_path, mode="r", encoding="utf-8") as f:
                self.lora_adapter = json.load(f)
        max_loras = kwargs.get("max_loras", 0)
        max_lora_rank = kwargs.get("max_lora_rank", 0)
        if self.lora_adapter or max_loras > 0:
            max_loras = max_loras if max_loras > 0 else len(self.lora_adapter)
            self.lora_model_config = LoraModelConfig(max_loras=max_loras, max_lora_rank=max_lora_rank)
        self.mapping = Mapping(world_size=world_size, rank=rank, llm_config=self.llm_config, **kwargs)
        if self.prealloc_weight_mem_on_npu:
            from mindie_llm.runtime.utils.distributed import set_parallel_info_manager

            set_parallel_info_manager(self.mapping)

        if not NPUSocInfo().support_bf16 and self.dtype == torch.bfloat16:
            error_msg = (
                "This device does not support bfloat16."
                "Please change the data type(i.e. `torch_dtype`) to float16 in config.json from model weights."
            )
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)

        torch.npu.set_compile_mode(jit_compile=False)
        self.kvcache_quant_layers = []
        print_log(rank, logger.info, "init tokenizer done")

        self.dp_all_gather_engine = None
        # An operation used for clearing internal tensors only
        self.dummy_operation = torch.classes.ModelTorch.ModelTorch("llama_LlamaDecoderModel")

        if self.layerwise_disaggregated:
            tls_config = {
                TLS_ENABLE: kwargs.get(TLS_ENABLE, "0"),
                TLS_CA_PATH: kwargs.get(TLS_CA_PATH, ""),
                TLS_CA_FILE: kwargs.get(TLS_CA_FILE, ""),
                TLS_CERT: kwargs.get(TLS_CERT, ""),
                TLS_PK: kwargs.get(TLS_PK, ""),
                TLS_CRL_PATH: kwargs.get(TLS_CRL_PATH, ""),
                TLS_CRL_FILES: kwargs.get(TLS_CRL_FILES, ""),
            }
            self.batch_p_num = kwargs.get("batch_p_num", 1)
            moe_quantize = getattr(self.config, "moe_quantize", None)
            self.data_comm = EdgeCloudDataComm(self.dtype, self.batch_p_num)
            self.ctrl_comm = EdgeCloudCtrlComm(tls_config)
            self.time_counter = CloudCutPolicy(
                self.layerwise_disaggregated_role_type, model_name_or_path, self.batch_p_num, moe_quantize
            )
            self.chunk_prefill_manager = ChunkPrefilPolicy(model_name_or_path, self.batch_p_num, moe_quantize)
            self.prefill_input_lengths = None
            lwd_comm_args = kwargs.get("lwd_comm_args", None)
            lwd_comm_args.update({"npu_id": self.npu_id})
            self.data_comm.set_comm_args(
                rank=self.local_rank, role=self.layerwise_disaggregated_role_type, comm_args=lwd_comm_args
            )

    @classmethod
    def resume_hccl_comm(cls):
        torch.classes.ModelTorch.Context.resume_hccl_comm()

    @classmethod
    def wait_event(cls, pipe_key: str):
        torch.classes.ModelTorch.Event.wait(pipe_key)

    def load_weights(self, **kwargs):
        """Load weights from file."""
        enable_v3 = False
        if self.llm_config.llm.engine.graph == GraphType.PYTHON:
            self.mapping.init_python_comm_process_group()
            weights = SafetensorFileLoader(self.model_name_or_path, self.device, mapping=self.mapping)
            mindie_llm_config = MindIELLMConfig(self.config, self.llm_config, weights.mapping)
            if self.lora_model_config is not None:
                mindie_llm_config.lora_config = self.lora_model_config
            enable_v3 = True
        else:
            weights = Weights(
                self.model_name_or_path,
                self.device,
                self.dtype,
                process_group=self.process_group,
                quantize=self.quantize,
                is_gqa=self.is_gqa,
                extension=".safetensors",
                mapping=self.mapping,
                lazy_loading_file_handlers=self.prealloc_weight_mem_on_npu,
                **kwargs,
            )
        if "OMP_NUM_THREADS" not in os.environ and self.world_size > 1:
            os.environ["OMP_NUM_THREADS"] = "1"
        try:
            if self.layerwise_disaggregated:
                self.model = self.model_cls(
                    self.config if not enable_v3 else mindie_llm_config,
                    weights,
                    inference_mode=self.inference_mode,
                    plugin_params=self.plugin_params,
                    trust_remote_code=self.trust_remote_code,
                    lora_adapter=self.lora_adapter,
                    num_speculative_tokens=self.num_speculative_tokens,
                    distributed_enable=self.distributed_enable,
                    max_batch_size=self.max_batch_size,
                    llm_config=self.llm_config,
                    model_role=self.model_role,
                    layerwise_disaggregated=self.layerwise_disaggregated,
                    layerwise_disaggregated_role_type=self.layerwise_disaggregated_role_type,
                )
            elif self.prealloc_weight_mem_on_npu:
                from mindie_llm.runtime.utils.torch_utils import set_default_torch_dtype
                from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig as MindIELLMConfigV2

                mindie_llm_config_v2 = MindIELLMConfigV2(
                    self.model_name_or_path, self.config, self.llm_config, generation_config=None
                )
                with set_default_torch_dtype(self.config.torch_dtype):
                    with self.device:
                        self.init_model(
                            self.config,
                            weights,
                            quant_config=mindie_llm_config_v2.quant_config,
                            prealloc_weight_mem_on_npu=True,
                            mempool_type=self.mempool_type,
                        )
            else:
                self.init_model(
                    self.config if not enable_v3 else mindie_llm_config, weights, mempool_type=self.mempool_type
                )

        except TypeError as e:
            logger.warning(
                f"{self.model_cls.__name__} initialization failed with extended kwargs: {e}. "
                f"Retrying with reduced argument list (config, weights)."
            )
            self.model = self.model_cls(self.config, weights)
        if hasattr(self.model, "kvcache_quant_layers"):
            self.kvcache_quant_layers = self.model.kvcache_quant_layers
        logger.info(f"Initialized model: {self.model_cls.__name__}")

        if not enable_v3 and not self.prealloc_weight_mem_on_npu:  # FlashForCausalLM
            if self.lora_adapter is not None:
                self.model.adapter_manager = AdapterManager(weights)
                self.model.update_adapter_manager()
                self.model.adapter_manager.preload_adapter(self.lora_adapter)
                self.adapter_manager = self.model.adapter_manager
        elif enable_v3 and getattr(self.model, "adapter_manager", None):  # FlashForCausalLMV3
            self.adapter_manager = self.model.adapter_manager
            self.adapter_manager.preload_adapter(self.lora_adapter)

        if enable_v3:
            self.attn_mask = self.model.attn_mask_generator.attention_mask_ins
        else:
            self.attn_mask = getattr(self.model, "attn_mask", None)
        weights_options = self.llm_config.get("llm").get("weights_options")

        if self.prealloc_weight_mem_on_npu:
            from atb_llm.utils.data.quant_method_adapter import MethodSupportAtbGraph
            from atb_llm.utils.data.layer_adapter import MergedColumnParallelLinear
            from mindie_llm.runtime.utils.loader.default_model_loader import DefaultModelLoader

            DefaultModelLoader().load_weights(self.model, self.model_name_or_path)
            # NOTE: Since `QuantizationMethodBase` objects are replaced with corresponding adapter objects,
            # `process_weights_after_loading` will not take effect when calling `DefaultModelLoader().load_weights`.
            modules_dict = dict(self.model.named_modules())
            for _, module in modules_dict.items():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, MethodSupportAtbGraph):
                    quant_method.process_weights_after_loading(module)
                # Handle MergedColumnParallelLinear with multiple linear_modules: when sub-layers have different
                # quant types they are split into independent linear_modules, each requiring quant weight processing
                if isinstance(module, MergedColumnParallelLinear) and len(module.linear_modules) > 1:
                    for m in module.linear_modules:
                        quant_method = getattr(m, "quant_method", None)
                        if isinstance(quant_method, MethodSupportAtbGraph):
                            quant_method.process_weights_after_loading(m)
            print_log(self.rank, logger.info, "load weight done.")
        else:
            if weights_options is not None and not weights_options.low_cpu_memory_mode:
                if NPUSocInfo.is_rc_device():
                    logger.info("NPU device is working in Root Complex (RC) mode.")
                else:
                    self.check_total_npu_mem()
            logger.info(f"Start transferring model to device {weights.device}")
            self.model.to(weights.device)
            logger.info(f"Model successfully transferred to device {weights.device}")
            weights.release_file_handler()

        # FlashForCausalLMV3 will get and destroy weights loader when performing dynamic-load.
        # Below func will be called if not enable v3
        if self.adapter_manager is not None and isinstance(self.adapter_manager, AdapterManager):
            if self.adapter_manager.lora_weights_loader is not None:
                self.adapter_manager.lora_weights_loader.release_file_handler()
            self.adapter_manager.prepare_adapter_weights()

        if not enable_v3:  # FlashForCausalLM
            if self.prealloc_weight_mem_on_npu and self.lora_model_config is not None:
                self.model.adapter_manager = self.model.lora_modifier.adapter_manager
                self.adapter_manager = self.model.adapter_manager
                self.adapter_manager.wrap_get_base_weight_shape_for_atb_graph()
                self.adapter_manager.preload_adapter(self.lora_adapter)
                self.adapter_manager.wrap_lora_module_for_atb_graph()

        if self.enable_atb_torch:
            self.model.init_graph()

        self.soc_info = self.model.soc_info
        if enable_v3:
            self.head_size = self.model.model_status.head_dim
            self.num_heads = self.model.model_status.num_attention_heads
            self.num_kv_heads = self.model.model_status.num_key_value_heads
            self.num_layers = self.model.model_status.num_hidden_layers
        else:
            self.head_size = self.model.head_size
            self.num_heads = self.model.num_attention_heads
            self.num_kv_heads = self.model.num_key_value_heads
            self.num_layers = self.model.num_layers
        if self.layerwise_disaggregated and self.deserialized_model_data:
            start_layer_num = self.deserialized_model_data.get("startLayerNum", 1)
            end_layer_num = self.deserialized_model_data.get("endLayerNum", 1)
            start_end_sum_layer = start_layer_num + end_layer_num
            logger.info(
                f"[layerwiseDisaggregated] {self.__class__.__name__} rank {self.rank} "
                f"startLayerNum {start_layer_num} endLayerNum {end_layer_num}"
            )
            if self.layerwise_disaggregated_role_type == "master":
                self.num_layers = start_end_sum_layer
            elif self.layerwise_disaggregated_role_type == "slave":
                self.num_layers -= start_end_sum_layer
        # not equal k v length for mla
        if hasattr(self.model, "kv_lora_rank") and hasattr(self.model, "qk_rope_head_dim"):  # deepseekv2/v3/r1
            self.num_kv_heads = 1
            self.k_head_size = self.model.kv_lora_rank
            self.v_head_size = self.model.qk_rope_head_dim
        else:
            self.k_head_size = self.head_size
            self.v_head_size = self.head_size

        logger.info(f"Successfully loaded model: {self.model_cls.__name__}\n{self.model}")

    def init_model(self, config, weights, **kwargs):
        """
        Args:
            kwargs:
                quant_config: Includes content from `quant_model_description.json`.
                    Used only when `prealloc_weight_mem_on_npu` is enabled.
        """
        self.model = self.model_cls(
            config,
            weights,
            inference_mode=self.inference_mode,
            plugin_params=self.plugin_params,
            trust_remote_code=self.trust_remote_code,
            lora_model_config=self.lora_model_config,
            lora_adapter=self.lora_adapter,
            num_speculative_tokens=self.num_speculative_tokens,
            distributed_enable=self.distributed_enable,
            max_batch_size=self.max_batch_size,
            llm_config=self.llm_config,
            model_role=self.model_role,
            **kwargs,
        )

    def init_gather_dp_graph(self):
        # gather
        padded = gather(Tensor("accept_lens_next_tokens"), 0, Tensor("padding"))
        # allgather
        all_gather_out = dist.all_gather(send_tensor=padded, process_group=self.mapping.attn_dp.process_group)
        all_gather_out_ = all_gather_out.reshape(lambda org_shape: [org_shape[0] * org_shape[1], org_shape[2]])
        # gather
        unpad = gather(all_gather_out_, 0, Tensor("unpadding"))
        # split
        reordered_accept_lens, reordered_next_tokens = split(
            unpad, split_size_or_sections=[1, (ENV.deepseek_mtp + 1)], dim=1
        )
        get_default_net().mark_output(reordered_accept_lens, "reordered_accept_lens")
        get_default_net().mark_output(reordered_next_tokens, "reordered_next_tokens")
        self.dp_all_gather_engine = get_default_net().build_engine()

    def build_inputs(self, conversations: List[List[Dict[str, str]]], **kwargs) -> list:
        """Build inputs for model."""
        return [self.input_builder.make_context(self.rank, conversation, **kwargs) for conversation in conversations]

    def forward_layerwise_disaggregated_edge_qwenvl(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        logger.info(
            f"[layerwiseDisaggregated] edge rank {self.rank} start_layer {exe_stage.start_exec_layer} "
            f"end_layer {exe_stage.end_exec_layer} {kwargs.keys()}"
        )

        if not exe_stage.is_prefill:  # decode
            if exe_stage.start_exec_layer == 0:  # decode首层
                # 云侧decode计算图输入与边侧计算图输入产生了差异（多模态场景，迭代1中并未出现此情况，
                # 迭代二中出现，具体原因需要进一步定位），针对此情况，decode阶段需要将云边不一致的计算图输入上传云端，
                # 具体是position ids
                position_ids = kwargs.get("position_ids")
                hidden = self.model.forward(**kwargs)
                # 暂时只支持qwenvl模型，隐变量维度为5120
                temp_position = torch.zeros([1, hidden.shape[1]], dtype=torch.bfloat16, device=self.device)
                if position_ids is not None:
                    # 基于组decode组batch不可能超过5120
                    temp_position[:, : len(position_ids)] = position_ids
                # 为当前复用隐变量传输逻辑，需要在标准的隐变量基础之横向拼接一个用于传输positionid的向量。
                # 同时为支持逻辑，所以为多模态定制传输逻辑。需要在request_router_cloud中针对qwenvl做特殊处理
                new_hidden = torch.cat([hidden, temp_position], dim=0)
                self.data_comm.decode_batch_size_queue.put(new_hidden.shape[0])
                self.data_comm.send_hidden("d", new_hidden)

                self.ctrl_comm.decode_send_msg = self.ctrl_comm.shape_to_msg(new_hidden.shape)
                self.ctrl_comm.send_decode()
                # Receive hidden state in advance.
                self.data_comm.d_shape = self.data_comm.decode_batch_size_queue.get()
                self.data_comm.recv_hidden("d", self.data_comm.d_shape)
                return hidden
            else:  # decode last layer
                tmp = self.data_comm.data_wait_after_recv("d")
                hidden = self.data_comm.broadcast_hidden(tmp, self.data_comm.d_shape, "d")
                # To maintain consistent presentation logic for the cloud side,
                # the decode final layer also receives a shape identical to that of the decode initial layer;
                # however, prior to entering the computational graph, the actual hidden dimension must be restored.
                real_hidden_length = int(hidden.shape[0] - 1)
                hidden = hidden[:real_hidden_length]
                new_params = {OUT_HIDDEN: hidden}
                kwargs.update(new_params)
                res = self.model.forward(**kwargs)
                return res
        else:
            p_comm_index = exe_stage.request_key % self.batch_p_num
            if exe_stage.start_exec_layer == 0:
                # 多模态模型场景下，部分计算图输入是需要经过vit模型计算后才可以获取的，因此必须从边缘上传云端。
                hidden = self.model.forward(**kwargs)

                # 分别针对3个需要额外传输的数据进行的处理，临时方案，后续reshape后传输，positionID疑似不需要后续去除
                if len(hidden) == 3:
                    temp_cos = torch.zeros(hidden[0].shape, dtype=torch.bfloat16, device=self.device)
                    temp_cos[:, : hidden[1].shape[1]] = hidden[1]
                    temp_cos[:, hidden[1].shape[1] : hidden[1].shape[1] + hidden[2].shape[1]] = hidden[2]
                    new_hidden = torch.cat([hidden[0], temp_cos], dim=0)
                else:
                    temp_cos = torch.zeros(hidden.shape, dtype=torch.bfloat16, device=self.device)
                    new_hidden = torch.cat([hidden, temp_cos], dim=0)

                self.data_comm.prefill_seq_len_queue.put(int(new_hidden.shape[0]))

                self.data_comm.send_hidden("p", new_hidden, send_index=p_comm_index)
                self.ctrl_comm.prefill_send_msg = self.ctrl_comm.shape_to_msg(new_hidden.shape)
                self.ctrl_comm.send_prefill()

                self.data_comm.p_shape[p_comm_index] = self.data_comm.prefill_seq_len_queue.get()
                self.data_comm.recv_hidden("p", self.data_comm.p_shape, recv_index=p_comm_index)
                return hidden
            else:
                tmp = self.data_comm.data_wait_after_recv("p", wait_recv_index=p_comm_index)
                hidden = self.data_comm.broadcast_hidden(tmp, self.data_comm.p_shape, "p", wait_recv_index=p_comm_index)
                # 为适配当前decode隐变量传输逻辑，边缘返回同样大小的数据，需要还原真实的隐变量大小
                hidden = hidden[: int(hidden.shape[0] / 2)]
                new_params = {OUT_HIDDEN: hidden}
                kwargs.update(new_params)
                res = self.model.forward(**kwargs)
                return res

    def forward_layerwise_disaggregated_edge_default(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        logger.info(
            f"[layerwiseDisaggregated] edge rank {self.rank} start_layer {exe_stage.start_exec_layer} end_layer"
            f" {exe_stage.end_exec_layer} {kwargs.keys()}"
        )

        if not exe_stage.is_prefill:
            if exe_stage.start_exec_layer == 0:
                hidden = self.model.forward(**kwargs)
                logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} decode batch size putted {hidden.shape}")
                self.data_comm.decode_batch_size_queue.put(hidden.shape[0])
                logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} decode send {hidden.shape}")
                self.data_comm.npu_net_host_hidden_sync(hidden)
                self.data_comm.send_hidden("d", hidden)

                self.ctrl_comm.decode_send_msg = self.ctrl_comm.shape_to_msg(hidden.shape)
                self.ctrl_comm.send_decode()

                self.data_comm.d_shape = self.data_comm.decode_batch_size_queue.get()
                self.data_comm.recv_hidden("d", self.data_comm.d_shape)
                return hidden
            else:
                tmp = self.data_comm.data_wait_after_recv("d")
                hidden = self.data_comm.broadcast_hidden(tmp, self.data_comm.d_shape, "d")
                logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} decode recv {hidden.shape}")
                new_params = {OUT_HIDDEN: hidden}
                kwargs.update(new_params)
                res = self.model.forward(**kwargs)
                logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} decode logits {res.shape}")
                return res
        else:
            p_comm_index = exe_stage.request_key % self.batch_p_num
            if exe_stage.start_exec_layer == 0:
                hidden = self.model.forward(**kwargs)
                if not exe_stage.is_long_seq:
                    self.data_comm.prefill_seq_len_queue.put(int(hidden.shape[0]))
                    logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} prefill seq len putted {hidden.shape}")
                else:
                    if len(exe_stage.long_seq_recv_list) > 0:
                        recv_len = exe_stage.long_seq_recv_list[0][1]
                        self.data_comm.prefill_seq_len_queue.put(recv_len)
                        logger.info(f"[layerwiseDisaggregated] edge rank {self.rank}, prefill len putted {recv_len}")

                logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} prefill send {hidden.shape}")
                self.data_comm.npu_net_host_hidden_sync(hidden)
                self.data_comm.send_hidden("p", hidden, send_index=p_comm_index)

                self.ctrl_comm.prefill_send_msg = self.ctrl_comm.shape_to_msg(hidden.shape)
                self.ctrl_comm.send_prefill()

                if not exe_stage.is_long_seq or len(exe_stage.long_seq_recv_list) > 0:
                    self.data_comm.p_shape[p_comm_index] = self.data_comm.prefill_seq_len_queue.get()
                    self.data_comm.recv_hidden("p", self.data_comm.p_shape, recv_index=p_comm_index)
                    logger.info(
                        f"[layerwiseDisaggregated] edge rank {self.rank} prefill recv start part, "
                        f"the data length is: {self.data_comm.p_shape[p_comm_index]}"
                    )
                return hidden
            else:
                hidden = None
                if exe_stage.hidden_start_pos == 0:
                    tmp = self.data_comm.data_wait_after_recv("p", wait_recv_index=p_comm_index)
                    hidden = self.data_comm.broadcast_hidden(
                        tmp, self.data_comm.p_shape, "p", wait_recv_index=p_comm_index
                    )  # 这里需要保证p_shape是本次要接收的
                    logger.info(
                        f"[layerwiseDisaggregated] edge rank {self.rank} prefill recv {hidden.shape} p_shape: "
                        f"{self.data_comm.p_shape[p_comm_index]}"
                    )
                if exe_stage.is_long_seq:
                    hidden_len = exe_stage.long_seq_end_idx - exe_stage.long_seq_start_idx
                    end_idx = exe_stage.hidden_start_pos + hidden_len
                    hidden = self.data_comm.get_cached_data(
                        exe_stage.hidden_start_pos, end_idx, batch_index=p_comm_index
                    )
                    logger.info(
                        f"[layerwiseDisaggregated] edge rank {self.rank} prefill get hidden"
                        f" {hidden.shape} hidden_len:{hidden_len}"
                    )
                new_params = {OUT_HIDDEN: hidden}
                kwargs.update(new_params)
                res = self.model.forward(**kwargs)
                logger.info(f"[layerwiseDisaggregated] edge rank {self.rank} prefill logits {res.shape}")

                if len(exe_stage.long_seq_recv_list) > 0:
                    self.data_comm.p_shape[p_comm_index] = exe_stage.long_seq_recv_list[0][1]
                    self.data_comm.recv_hidden("p", self.data_comm.p_shape, recv_index=p_comm_index)
                    logger.info(
                        f"[layerwiseDisaggregated] edge rank {self.rank} prefill recv start post part, "
                        f"p_shape: {self.data_comm.p_shape[p_comm_index]}"
                    )

                return res

    def forward_layerwise_disaggregated_edge(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        if not self.isqwenvl:
            return self.forward_layerwise_disaggregated_edge_default(**kwargs)
        else:
            return self.forward_layerwise_disaggregated_edge_qwenvl(**kwargs)

    def forward_layerwise_disaggregated_cloud_qwenvl(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        logger.info(
            f"[layerwiseDisaggregated] cloud rank {self.rank} start_layer {exe_stage.start_exec_layer} "
            f"end_layer {exe_stage.end_exec_layer} {kwargs.keys()}"
        )

        if not exe_stage.is_prefill:  # Decode
            tmp = self.data_comm.data_wait_after_recv("d")
            hidden = self.data_comm.broadcast_hidden(tmp, self.data_comm.d_shape, "d")
            # decode场景下，传输来的tensor，最后一行向量为position_ids,分离真实的hidden与position_ids
            real_len = int(hidden.shape[0] - 1)
            real_hidden = hidden[:real_len, :]

            # 更新计算图输入的position ids，当前实现只能支持decode batch为1的场景，待验证后更新
            position_ids_edge = hidden[real_len:, :]
            position_ids = kwargs.get("position_ids")

            position_ids.copy_(position_ids_edge[0, : position_ids.shape[0]])

            logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} new decode position_ids:{position_ids}")
            kwargs.update({"position_ids": position_ids})
            new_params = {OUT_HIDDEN: real_hidden}
            kwargs.update(new_params)

            res = self.model.forward(**kwargs)

            # 当前实现只支持decode batchsize为1的场景，待验证后更新

            temp_values_tosend = torch.zeros([1, res.shape[1]], dtype=torch.bfloat16, device=self.device)
            new_res = torch.cat([res, temp_values_tosend], dim=0)
            self.data_comm.send_hidden("d", new_res)
            # 发送tcp信号
            self.ctrl_comm.decode_send_msg = self.ctrl_comm.shape_to_msg(new_res.shape)
            self.ctrl_comm.send_decode()
            # 此处做判断，如果有req，且没有收操作，则挂出接收操作
            with self.data_comm.lock:
                logger.info(
                    "[layerwiseDisaggregated] model_runner, pre recv "
                    f"{self.data_comm.decode_batch_size_queue.qsize()} {self.data_comm.flag_pre_recv}"
                )
                if not self.data_comm.decode_batch_size_queue.empty():
                    self.data_comm.d_shape = self.data_comm.decode_batch_size_queue.get()
                    self.data_comm.recv_hidden("d", self.data_comm.d_shape)
                    self.data_comm.flag_pre_recv = False
                else:
                    self.data_comm.flag_pre_recv = True

            # 处理res，本层为云侧中间层的尾层，返回hidden，此处应依据hidden构造logits
            batch_size = int(res.shape[0])
            res = torch.ones([batch_size, self.model.config.vocab_size], dtype=torch.float16, device=self.device)
            logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} deocde shape {res.shape}")
            return res
        else:  # Prefill
            p_comm_index = exe_stage.request_key % self.batch_p_num
            if exe_stage.start_exec_layer == 0:  # prefill首层
                self.prefill_input_lengths = kwargs.get("input_lengths")
                tmp = self.data_comm.data_wait_after_recv("p", wait_recv_index=p_comm_index)
                hidden = self.data_comm.broadcast_hidden(tmp, self.data_comm.p_shape, "p", wait_recv_index=p_comm_index)
                real_len = int(hidden.shape[0] / 2)
                real_hidden = hidden[:real_len, :]
                sin_list = hidden[real_len:, 128:256]
                cos_list = hidden[real_len:, :128]

                if not torch.all(cos_list == 0).item():
                    kwargs.update({"cos_list": cos_list})
                    kwargs.update({"sin_list": sin_list})
                new_params = {OUT_HIDDEN: real_hidden}
                kwargs.update(new_params)
                res = self.model.forward(**kwargs)
                logger.info("[layerwiseDisaggregated] isqwenvl prefill end!")
                return res
            else:
                if exe_stage.end_exec_layer < exe_stage.cloud_total_layer:
                    res = self.model.forward(**kwargs)
                    return res
                else:
                    res = self.model.forward(**kwargs)
                    # 扩展为原始返回的四倍返回
                    new_res = torch.cat([res, res], dim=0)
                    self.data_comm.send_hidden("p", new_res, send_index=p_comm_index)
                    # 发送tcp信号
                    self.ctrl_comm.prefill_send_msg = self.ctrl_comm.shape_to_msg(new_res.shape)
                    self.ctrl_comm.send_prefill()
                    # 这个逻辑不能动，返回是1倍
                    res = torch.ones(
                        [len(self.prefill_input_lengths), self.model.config.vocab_size],
                        dtype=torch.float16,
                        device=self.device,
                    )
                    logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} prefill logits shape {res.shape}")
                    return res

    def forward_layerwise_disaggregated_cloud_default(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        logger.info(
            f"[layerwiseDisaggregated] cloud rank {self.rank} start_layer {exe_stage.start_exec_layer} "
            f"end_layer {exe_stage.end_exec_layer} {kwargs.keys()}"
        )

        if not exe_stage.is_prefill:
            tmp = self.data_comm.data_wait_after_recv("d")
            hidden = self.data_comm.broadcast_hidden(tmp, self.data_comm.d_shape, "d")
            logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} decode recv {hidden.shape}")
            new_params = {OUT_HIDDEN: hidden}
            kwargs.update(new_params)

            # 同步 + decode 计时打点，为了decode统计时间准确执行hidden_sync
            hidden_sync = torch.clone(hidden[0][0])
            hidden_sync.cpu()
            self.time_counter.set_decode_start_time(exe_stage.is_prefill, time.time())

            res = self.model.forward(**kwargs)
            logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} decode send {res.shape}")
            self.data_comm.send_hidden("d", res)
            # 发送tcp信号
            self.ctrl_comm.decode_send_msg = self.ctrl_comm.shape_to_msg(res.shape)
            self.ctrl_comm.send_decode()

            # 此处做判断，如果有req，且没有收操作，则挂出接收操作
            with self.data_comm.lock:
                logger.info(
                    f"[layerwiseDisaggregated] model_runner, pre recv \
                    {self.data_comm.decode_batch_size_queue.qsize()} {self.data_comm.flag_pre_recv}"
                )
                if not self.data_comm.decode_batch_size_queue.empty():
                    self.data_comm.d_shape = self.data_comm.decode_batch_size_queue.get()
                    self.data_comm.recv_hidden("d", self.data_comm.d_shape)
                    self.data_comm.flag_pre_recv = False
                else:
                    self.data_comm.flag_pre_recv = True

            if self.plugin_params is not None:
                # 处理res，本层为云侧中间层的尾层，返回hidden，此处应依据hidden构造logits
                batch_size = int(res.shape[0])
                res = torch.ones([batch_size, self.model.config.vocab_size], dtype=self.dtype, device=self.device)
            logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} deocde shape {res.shape}")
            return res
        else:  # Prefill
            p_comm_index = exe_stage.request_key % self.batch_p_num
            self.prefill_input_lengths = kwargs.get("input_lengths")
            logger.info(
                f"[layerwiseDisaggregated]:is_long_seq {exe_stage.is_long_seq} in {exe_stage.start_exec_layer} "
                f"{exe_stage.end_exec_layer}; {exe_stage.long_seq_start_idx} {exe_stage.long_seq_end_idx}; "
                f"self.prefill_input_lengths: {self.prefill_input_lengths}"
            )
            if exe_stage.start_exec_layer == 0:
                tmp = self.data_comm.data_wait_after_recv("p", wait_recv_index=p_comm_index)
                hidden = tmp
                if exe_stage.is_long_seq:
                    hidden_len = exe_stage.long_seq_end_idx - exe_stage.long_seq_start_idx
                    if tmp is not None:
                        hidden = self.data_comm.get_cached_data(0, hidden_len, batch_index=p_comm_index)

                    # 长序列需要广播边侧发过来的很多段拼成的一大段
                    self.data_comm.p_shape[p_comm_index] = hidden_len
                    hidden = self.data_comm.broadcast_hidden(
                        hidden, self.data_comm.p_shape, "p", 0, wait_recv_index=p_comm_index
                    )
                    logger.info(
                        f"[layerwiseDisaggregated] cloud rank {self.rank} prefill recv {hidden.shape} "
                        f"hidden_len:{hidden_len} hidden_start_pos:{0} p_comm_index: {p_comm_index}"
                    )
                else:
                    hidden = self.data_comm.broadcast_hidden(
                        tmp, self.data_comm.p_shape, "p", wait_recv_index=p_comm_index
                    )
                    logger.info(
                        f"[layerwiseDisaggregated] cloud rank {self.rank} prefill recv {hidden.shape} "
                        f"p_shape:{self.data_comm.p_shape[p_comm_index]}"
                    )

                recv_list = exe_stage.long_seq_recv_list
                if len(recv_list) > 1:
                    self.data_comm.prefill_chunk_recv(recv_list, p_comm_index)
                    logger.info(
                        f"[layerwiseDisaggregated] cloud rank {self.rank}, queue input is recv_list: {recv_list}"
                    )
                elif len(recv_list) == 1:
                    self.data_comm.prefill_seq_len_queue.put(recv_list[0][1])
                    self.data_comm.p_shape[p_comm_index] = self.data_comm.prefill_seq_len_queue.get()
                    self.data_comm.recv_hidden("p", self.data_comm.p_shape, recv_index=p_comm_index)
                    logger.info(
                        f"[layerwiseDisaggregated] cloud rank {self.rank}, "
                        f"queue input is {recv_list[0][1]}, recv_list: {recv_list}"
                    )

                new_params = {OUT_HIDDEN: hidden}
                kwargs.update(new_params)
                res = self.model.forward(**kwargs)
                return res
            else:
                if exe_stage.end_exec_layer < exe_stage.cloud_total_layer:
                    res = self.model.forward(**kwargs)
                    return res
                else:
                    # 发送tcp信号
                    res = self.model.forward(**kwargs)
                    logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} prefill send {res.shape}")
                    self.data_comm.send_hidden("p", res, send_index=p_comm_index)
                    # 发送tcp信号
                    self.ctrl_comm.prefill_send_msg = self.ctrl_comm.shape_to_msg(res.shape)
                    self.ctrl_comm.send_prefill()
                    if self.plugin_params is not None:
                        res = torch.ones(
                            [len(self.prefill_input_lengths), self.model.config.vocab_size],
                            dtype=self.dtype,
                            device=self.device,
                        )
                    logger.info(f"[layerwiseDisaggregated] cloud rank {self.rank} prefill logits shape {res.shape}")
                    return res

    def forward_layerwise_disaggregated_cloud(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        if not self.isqwenvl:
            return self.forward_layerwise_disaggregated_cloud_default(**kwargs)
        else:
            return self.forward_layerwise_disaggregated_cloud_qwenvl(**kwargs)

    def forward_layerwise_disaggregated(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage")
        role_str = "edge" if self.layerwise_disaggregated_role_type == "master" else "cloud"
        if self.data_comm.init_finish and self.ctrl_comm.init_finish:
            logger.info(f"[layerwiseDisaggregated] {role_str} rank {self.rank} forward exe_stage:{exe_stage}")
            if role_str == "edge":
                res = self.forward_layerwise_disaggregated_edge(**kwargs)
            else:
                res = self.forward_layerwise_disaggregated_cloud(**kwargs)
        else:
            if not self.data_comm.init_finish and self.data_comm.get_lwd_rank_file() is not None:
                self.data_comm.init_hccl()
                if self.data_comm.init_finish:
                    self.data_comm.hccl_comm_warmup(self.model.hidden_size)

            # warmup处理
            if exe_stage is None:
                input_lengths = kwargs.get("input_lengths")
                input_ids = kwargs.get("input_ids")
                is_prefill = kwargs.get("is_prefill")
                if is_prefill:
                    batch_size = len(input_lengths) * self.mapping.attn_dp.group_size
                else:
                    batch_size = len(input_ids) * self.mapping.attn_dp.group_size
                out_dict = {
                    OUT_HIDDEN: torch.ones(
                        [len(input_ids), self.model.hidden_size], dtype=self.dtype, device=self.device
                    )
                }
                kwargs.update(out_dict)
                if self.data_comm.global_comm is not None:
                    self.mapping.set_lwd_global_comm(self.data_comm.global_comm)

            """Call model's forward pass."""
            res = self.model.forward(**kwargs)
            # warmup时exe_stage为None
            if exe_stage is None:
                res = torch.ones([batch_size, self.model.config.vocab_size], dtype=self.dtype, device=self.device)
        if ENV.enable_dp_partition_up and self.mapping.has_dp() and self.dp_all_gather_engine is None:
            self.init_gather_dp_graph()
        return res

    def forward(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        if self.layerwise_disaggregated:
            return self.forward_layerwise_disaggregated(**kwargs)

        """Call model's forward pass."""
        res = self.model.forward(**kwargs)
        if ENV.enable_dp_partition_up and self.mapping.has_dp() and self.dp_all_gather_engine is None:
            self.init_gather_dp_graph()
        return res

    def dap_forward(self, **kwargs):
        return self.model.dap_forward(**kwargs)

    def generate(self, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        """Generate output text via calling model's generate method."""
        return self.model.generate(**kwargs)

    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        """Generate position ids."""
        if self.layerwise_disaggregated and self.layerwise_disaggregated_role_type == "slave" and self.isqwenvl:
            return self.input_builder.generate_position_ids_for_cloud(input_ids)
        else:
            position_ids = self.input_builder.generate_position_ids(input_ids)
            return position_ids

    def save_pretrained(self, **kwargs):
        """Save pretrained model."""
        save_directory_key = "save_directory"
        if save_directory_key not in kwargs:
            raise ValueError(f"{save_directory_key} is required")
        kwargs[save_directory_key] = os.path.join(kwargs[save_directory_key], f"part{self.rank}-of-{self.world_size}")
        self.model.save_pretrained(**kwargs)

    def save_sharded(self, **kwargs):
        self.model.save_sharded(**kwargs)

    def check_total_npu_mem(self):
        total_weight_size = 0
        for param in list(self.model.parameters()) + list(self.model.buffers()):
            if param.device.type == "cpu":
                total_weight_size += param.numel() * param.element_size()
        check_npu_mem(rank=self.rank, total_weight_size=total_weight_size)

    def clear_internal_tensors(self):
        self.dummy_operation.clear_internal_tensors()

    def reset_execution_status(self):
        self.dummy_operation.reset_execution_status()

    def _is_quantization_supported(self):
        """Check if current quantization configuration is supported"""
        return self.config.quantize is None or self.config.quantize in PREALLOC_SUPPORTED_QUANT_TYPES
