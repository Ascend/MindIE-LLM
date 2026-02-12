# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import time
from itertools import product

import torch
from torch import nn
from tqdm.auto import tqdm

from mindie_llm.runtime.layers.fused_moe.fused_moe import FusedMoE
from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler
from mindie_llm.runtime.layers.quantization.quantization_method_base import QuantizationMethodBase
from mindie_llm.runtime.layers.attention.backend.abstract import AttentionImpl
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager
from mindie_llm.utils.log.logging import logger


_BAR_FORMAT = "{desc}: {l_bar}{bar}| Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


_global_param_dict: dict | None = None


class DefaultModelLoader:
    """Model loader for safetensors checkpoint files."""
    def __init__(self):
        self._counter_before_loading_weights: float = 0.0
        self._counter_after_loading_weights: float = 0.0
        self._loaded_weight_names = []
        self._weight_file_handler = None

    def load_weights(self, model: nn.Module, model_path: str) -> None:
        """Load model weights from checkpoint."""
        self._counter_before_loading_weights = time.perf_counter()

        # Traverse module and map to corresponding weight
        self._weight_file_handler = WeightsFileHandler(model_path, ".safetensors")
        self._load_modules(model)
        self._weight_file_handler.release_file_handler()

        self._counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self._counter_after_loading_weights - self._counter_before_loading_weights,
        )

    def _get_total_leaf_modules(self, module: nn.Module, prefix: str = "") -> dict[str, nn.Module]:
        """Get leaf modules with full names."""
        if len(list(module.children())) == 0:
            return {prefix: module}
        leaf_modules_dict = {}
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            leaf_modules_dict.update(self._get_total_leaf_modules(child, child_prefix))
        return leaf_modules_dict

    def _load_multi_prefix_module(self, module: nn.Module) -> None:
        """Load weights for multi-prefix modules.(e.g., QKVParallelLinear)."""
        for shard_id, weight_prefix in enumerate(module.prefix):
            for weight_suffix, param in module.named_parameters():
                if param is None:
                    continue

                full_param_name = f"{weight_prefix}.{weight_suffix}"
                if check_and_reuse_global_param_dict(param, full_param_name):
                    continue

                loaded_weight = self._weight_file_handler.get_tensor(full_param_name)
                param.weight_loader(param, loaded_weight, shard_id)
                update_global_param_dict(full_param_name, param)

    def _load_single_prefix_module(self, module: nn.Module, prefix: str) -> None:
        """Load weights for single-prefix modules."""
        for weight_suffix, param in module.named_parameters():
            if param is None:
                continue
            full_param_name = f"{prefix}.{weight_suffix}"
            if check_and_reuse_global_param_dict(param, full_param_name):
                continue

            try:
                loaded_weight = self._weight_file_handler.get_tensor(full_param_name)
            except ValueError as e:
                # Try module-specific prefix for weight files(e.g., used for tie_word_embedding function)
                if "Weight file was not found" in str(e) and hasattr(module, "prefix"):
                    full_param_name = f"{module.prefix}.{weight_suffix}"
                    loaded_weight = self._weight_file_handler.get_tensor(full_param_name)
            param.weight_loader(param, loaded_weight)
            update_global_param_dict(full_param_name, param)
    
    def _load_modules_with_progress(self, modules_dict: dict, pbar: tqdm,) -> None:
        """Load weights for modules with progress."""
        for prefix, module in modules_dict.items():
            # Handling multi-prefix modules
            if hasattr(module, "prefix") and isinstance(module.prefix, list):
                self._load_multi_prefix_module(module)
            elif isinstance(module, FusedMoE):
                # Handle weights (and optionally scale/bias) for every local expert.
                # Note: Will be optimized later, should not depend on specific classes.
                expert_list = module.expert_list
                for expert_id, module_suffix, weight_name in product(
                    expert_list,
                    module.suffix,
                    module.weight_list
                ):
                    full_param_name = f"{prefix}.{expert_id}.{module_suffix}.{weight_name}"
                    loaded_weight = self._weight_file_handler.get_tensor(full_param_name)
                    module.weight_loader(loaded_weight, expert_id, module_suffix, weight_name)
            else:  # Processing single prefix Module
                self._load_single_prefix_module(module, prefix)

            # Apply post-processing after pretrained-weights are loaded onto NPU device.
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizationMethodBase):
                quant_method.process_weights_after_loading(module)

            # Note: a common process_weights_after_loading method for module is needed.
            attn_impl = getattr(module, "impl", None)
            if isinstance(attn_impl, AttentionImpl):
                attn_impl.process_weights_after_loading()
        
            pbar.update(1)

    def _load_modules(self, model: nn.Module) -> None:
        """Load model weights for leaf modules."""
        leaf_modules_dict = self._get_total_leaf_modules(model)
        
        pbar = tqdm(total=len(leaf_modules_dict),
                    desc="Loading safetensors checkpoint shards in the lazy mode",
                    disable=get_parallel_info_manager().rank,
                    bar_format=_BAR_FORMAT,
                    unit="module")
        
        self._load_modules_with_progress(leaf_modules_dict, pbar)
        pbar.close()


def update_global_param_dict(full_param_name: str, param):
    # Note: rm _global_param_dict after load weight
    global _global_param_dict
    _global_param_dict[full_param_name] = param


def check_and_reuse_global_param_dict(target_param, full_param_name: str):
    global _global_param_dict
    if _global_param_dict is None:
        _global_param_dict = {}
    if full_param_name in _global_param_dict:
        source_param = _global_param_dict[full_param_name]
        target_param.data = source_param.data
        return True
    return False
