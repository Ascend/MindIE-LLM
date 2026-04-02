# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import Any, Dict
import json

import mindspore as ms
from mindspore import mutable
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import Zero
import numpy as np

from .generator_backend import GeneratorBackend
from ..utils.model_input import ModelInput
from ...utils.env import ENV
from ...utils.decorators.time_decorator import timer


class GeneratorMS(GeneratorBackend):
    """The interface class for using `mindspore` backend.

    The interface class exposed to third-party service frameworks in scenarios where the `mindspore` backend is used. It
    mainly provides forward inference and sampling functions. Its sampling function is implemented by the `sample`
    method of the base class `GeneratorBackend`.

    Args:
        model_config: A dictionary containing the model configuration as detailed in
            `mindie_llm.text_generator.utils.config.ModelConfig`.
    """
    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__(model_config)
        self.tokenizer = self.model_wrapper.tokenizer
        self.max_position_embeddings = self.model_wrapper.max_position_embeddings
        plugin_params = model_config.get('plugin_params', {})
        try:
            plugin_params = json.loads(plugin_params) if isinstance(plugin_params, str) else plugin_params
        except json.JSONDecodeError:
            plugin_params = {}
        self.plugin_type = plugin_params.get('plugin_type')
        self.npu_mem = model_config['npu_mem']
        self.key_cache = None
        self.value_cache = None
        self.model_role = model_config.get('model_role', "standard")
        if self.npu_mem == -1 and self.model_role != 'standard':
            raise Exception("pd detached does not support npu_mem_size=-1")

    def to_tensor(self, data):
        return ms.Tensor(data) if data.size > 0 else None

    @timer.track_time('forward')
    def forward(self, model_inputs: ModelInput, **kwargs) -> Any:
        """Call the `forward` method of the model wrapper, which should return a Tensor of corresponding backend."""
        result = self.model_wrapper.forward(model_inputs,
                                            key_cache=self.key_cache, value_cache=self.value_cache, **kwargs)
        return result

    def update_cache_policy(self, kvcache_settings, sepd_worker=None):
        """The method for updating the kv cache, which is used in warm-up stage."""
        if sepd_worker is None:
            if self.npu_mem == -1:
                self.alloc_kv_cache(kvcache_settings)
            return
        use_mb_swapper = ENV.use_mb_swapper
        if use_mb_swapper and len(sepd_worker.addrs) != 1:
            return
        if not use_mb_swapper and len(sepd_worker.addrs) <= 1:
            return

        if use_mb_swapper:
            kv_cache_addr_offset = 0
            for i in range(self.model_wrapper.model_runner.num_layers):
                key_cache, value_cache = self.model_wrapper.model_runner.model.kvcache(i)
                key_cache.set_device_address(sepd_worker.addrs[0] + kv_cache_addr_offset,
                                             key_cache.shape, key_cache.dtype)
                kv_cache_addr_offset += kvcache_settings.num_npu_blocks * kvcache_settings.mini_block_bytes
                key_cache.set_device_address(sepd_worker.addrs[0] + kv_cache_addr_offset,
                                             value_cache.shape, value_cache.dtype)
                kv_cache_addr_offset += kvcache_settings.num_npu_blocks * kvcache_settings.mini_block_bytes
        else:
            for i in range(self.model_wrapper.model_runner.num_layers):
                key_cache, value_cache = self.model_wrapper.model_runner.model.kvcache(i)
                key_cache.set_device_address(sepd_worker.addrs[i * 2],
                                             key_cache.shape, key_cache.dtype)
                value_cache.set_device_address(sepd_worker.addrs[i * 2 + 1],
                                               value_cache.shape, value_cache.dtype)
    
    def alloc_kv_cache(self, kvcache_settings):
        kv_shape = (int(kvcache_settings.num_npu_blocks), *kvcache_settings.block_shape)
        key_cache = []
        value_cache = []
        for _ in range(kvcache_settings.num_layers):
            key_block = Tensor(shape=kv_shape, dtype=kvcache_settings.dtype, init=Zero())
            value_block = Tensor(shape=kv_shape, dtype=kvcache_settings.dtype, init=Zero())
            key_block = key_block.new_empty(kv_shape, dtype=kvcache_settings.dtype, device="Ascend")
            value_block = key_block.new_empty(kv_shape, dtype=kvcache_settings.dtype, device="Ascend")
            key_cache.append(key_block)
            value_cache.append(value_block)
        key_cache, value_cache = mutable(key_cache), mutable(value_cache)
        self.key_cache, self.value_cache = key_cache, value_cache

    def clear_kv_cache(self):
        self.key_cache = None
        self.value_cache = None
        import gc
        gc.collect()

    def update_cache_after_switch_pd_role(self):
        pass

    def swap_cache(self, swap_decision):
        self.model_wrapper.swap_cache(swap_decision)
