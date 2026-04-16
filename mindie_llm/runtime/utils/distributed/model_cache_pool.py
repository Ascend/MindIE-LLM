# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import copy
import math
import gc
import threading
from collections import namedtuple

import torch
import torch_npu

from mindie_llm.runtime.utils.cache_spec import CacheGroupInfo, CacheType
from mindie_llm.runtime.layers.attention import get_global_attn_dict
from mindie_llm.utils.log.logging import logger

NZ_KV_CACHE_FORMAT = 16  # NZ排布的KVCache做了16位对齐
GroupKey = namedtuple("group", ["ratio", "block_size", "type"])


class ModelCachePool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        self._groups = None
        self._caches = None
        self._model = None
        self._device = None
        self._device_caches_addrs = None
        self._group_keys = None
        self._max_batch_size = None
        self.initialized = False

    @staticmethod
    def _clear() -> None:
        attn_dict = get_global_attn_dict()
        for attn in attn_dict.values():
            attn.clear()
        gc.collect()

    @staticmethod
    def _create_aligned_tensor(target_shape, dtype, device, acl_format):
        from functools import reduce
        import operator

        if acl_format == 2:  # 2: ND
            element_size = torch.tensor([], dtype=dtype).element_size()
            desired_num_elements = reduce(operator.mul, target_shape, 1)
            tensor_size = desired_num_elements * element_size
            additional_memory = 2 * 1024 * 1024
            total_size = tensor_size + additional_memory

            kv_tensor = torch.empty(size=(total_size // element_size,), dtype=dtype, device=device)
            tensor_ptr = kv_tensor.data_ptr()
            alignment = 2 * 1024 * 1024
            offset = tensor_ptr % alignment
            if offset != 0:
                cut_size = alignment - offset
            else:
                cut_size = 0
            # 切片对齐 tensor
            cut_elements = cut_size // element_size
            aligned_tensor = kv_tensor[cut_elements : cut_elements + desired_num_elements]
            aligned_tensor = aligned_tensor.contiguous()
            aligned_tensor = aligned_tensor.view(*target_shape)
            return aligned_tensor

        else:  # NZ
            additional_memory_kb = 2 * 1024
            kv_tensor = torch_npu.empty_with_format(
                size=target_shape,
                dtype=dtype,
                device=device,
                acl_format=acl_format,
                base_addr_aligned_kb=additional_memory_kb,
            )
            return kv_tensor

    def initialize(self, model, device, max_batch_size: int = 8):
        if self.initialized:
            return

        self._groups: dict[GroupKey, CacheGroupInfo] = {}
        # 储存每一层的cacheInfo，[num_blocks, tuple[block_size, num_heads, head_dim], group_id, dtype, acl_format]
        self._caches: list[list[tuple[int, tuple[int, int, int], int, int, int]]] = []
        self._model = model
        self._device = device
        self._device_caches_addrs = []
        # init
        self._group_keys = []
        self._update_caches_info()
        self.initialized = True
        self._max_batch_size = max_batch_size

    def get_caches_info(self):
        return self._caches

    def get_caches_addrs(self) -> list[list[int]]:
        return self._device_caches_addrs

    def get_groups_info(self):
        return list(self._groups.values())

    def warmup_device_cache(self, device_mem) -> int:
        cache_size = 0  # bytes
        attn_dict = get_global_attn_dict()
        attns = iter(attn_dict.values())
        for layer_caches_info in self._caches:
            layer_caches = []
            layer_caches_addrs = []
            for _, cache_info in enumerate(layer_caches_info):
                num_blocks, shape, _, dtype, acl_format = cache_info
                cache_size += math.prod(shape) * dtype.itemsize  # 计算num_blocks=1时的最小cache
                if acl_format == 29:
                    shape = (
                        math.ceil(math.prod(shape) / shape[0] / NZ_KV_CACHE_FORMAT),
                        shape[0],
                        NZ_KV_CACHE_FORMAT,
                    )
                cache = torch_npu.empty_with_format(
                    size=(num_blocks, *shape),
                    dtype=dtype,
                    device=self._device,
                    acl_format=acl_format,
                )
                cache.fill_(0)
                layer_caches.append(cache)
                layer_caches_addrs.append(cache.data_ptr())
            attn = next(attns)
            attn.bind_model_cache(layer_caches)
            self._device_caches_addrs.append(layer_caches_addrs)
        self._cal_num_blocks(device_mem=device_mem)  # 在 groups中更新真实的num_blocks用于auto_warmup计算
        return cache_size

    def allocate_device_cache(self, device_mem, is_dmi=False):
        self._clear()
        self._cal_num_blocks(device_mem=device_mem)
        self._device_caches_addrs = []
        attn_dict = get_global_attn_dict()
        attns = iter(attn_dict.values())
        for layer_id, layer_caches_info in enumerate(self._caches):
            layer_caches = []
            layer_caches_addrs = []
            for cache_id, cache_info in enumerate(layer_caches_info):
                _, shape, group_id, dtype, acl_format = cache_info
                group_key = self._group_keys[group_id]
                num_blocks = self._groups.get(group_key).num_blocks
                self._caches[layer_id][cache_id] = (
                    num_blocks,
                    shape,
                    group_id,
                    dtype,
                    acl_format,
                )  # update num_blocks
                if is_dmi:
                    cache = self._create_aligned_tensor(
                        target_shape=(num_blocks, *shape),
                        dtype=dtype,
                        device=self._device,
                        acl_format=acl_format,
                    )
                else:
                    cache = torch_npu.empty_with_format(
                        size=(num_blocks, *shape),
                        dtype=dtype,
                        device=self._device,
                        acl_format=acl_format,
                    )
                cache.fill_(0)
                layer_caches.append(cache)
                layer_caches_addrs.append(cache.data_ptr())
            attn = next(attns)
            attn.bind_model_cache(layer_caches)
            self._device_caches_addrs.append(layer_caches_addrs)

    def calculate_groups_info(self, device_mem: int):
        total = 0.0
        constant_device_mem = 0.0
        for _, cache_group_info in self._groups.items():
            if cache_group_info.type == CacheType.TOKEN:
                bytes_of_blocks = cache_group_info.bytes_of_blocks
                ratio = cache_group_info.ratio
                block_size = cache_group_info.block_size
                total += bytes_of_blocks / (ratio * block_size)
            elif cache_group_info.type == CacheType.SLIDING_WINDOW:
                bytes_of_blocks = cache_group_info.bytes_of_blocks
                constant_device_mem += (12 * self._max_batch_size) * bytes_of_blocks

        terminate = (device_mem - constant_device_mem) / total
        # cal num_blocks per group
        groups_info = []
        for _, cache_group_info in self._groups.items():
            cache_group_info = copy.deepcopy(cache_group_info)
            if cache_group_info.type == CacheType.TOKEN:
                ratio = cache_group_info.ratio
                block_size = cache_group_info.block_size
                cache_group_info.num_blocks = int(terminate / (ratio * block_size))
            elif cache_group_info.type == CacheType.SLIDING_WINDOW:
                cache_group_info.num_blocks = 12 * self._max_batch_size
            groups_info.append(cache_group_info)
        return groups_info

    def _update_caches_info(self):
        attn_dict = get_global_attn_dict()
        for attn in attn_dict.values():
            cache_spec = attn.get_cache_spec()
            cache_info = []
            acl_format = cache_spec.format
            for ratio, cache_type, dtype, shape in zip(
                cache_spec.ratio, cache_spec.type, cache_spec.dtype, cache_spec.shape
            ):
                group_key = GroupKey(ratio=ratio, type=cache_type, block_size=shape[0])
                if group_key not in self._groups.keys():
                    self._group_keys.append(group_key)
                    group_id = len(self._group_keys) - 1
                else:
                    group_id = self._group_keys.index(group_key)
                group_info = self._groups.setdefault(
                    group_key,
                    CacheGroupInfo(ratio=ratio, block_size=shape[0], type=cache_type),
                )
                group_info.bytes_of_blocks += math.prod(shape) * dtype.itemsize
                num_blocks = self._groups[group_key].num_blocks
                cache_info.append((num_blocks, shape, group_id, dtype, acl_format))
            self._caches.append(cache_info)

    def _cal_num_blocks(self, device_mem: int):
        # cal: terminate = sum(bytes_of_blocks_i / (ratio_i * block_size_i))  -> (num_blocks * terminate = device_mem)
        total = 0.0
        constant_device_mem = 0.0
        for _, cache_group_info in self._groups.items():
            if cache_group_info.type == CacheType.TOKEN:
                bytes_of_blocks = cache_group_info.bytes_of_blocks
                ratio = cache_group_info.ratio
                block_size = cache_group_info.block_size
                total += bytes_of_blocks / (ratio * block_size)
            elif cache_group_info.type == CacheType.SLIDING_WINDOW:
                bytes_of_blocks = cache_group_info.bytes_of_blocks
                constant_device_mem += (12 * self._max_batch_size + 2) * bytes_of_blocks
            elif cache_group_info.type == CacheType.SEQUENCE:
                bytes_of_blocks = cache_group_info.bytes_of_blocks
                constant_device_mem += (self._max_batch_size + 2) * bytes_of_blocks

        terminate = (device_mem - constant_device_mem) / total
        # cal num_blocks per group
        for group, cache_group_info in self._groups.items():
            if cache_group_info.type == CacheType.TOKEN:
                ratio = cache_group_info.ratio
                block_size = cache_group_info.block_size
                num_blocks = int(terminate / (ratio * block_size))
                if num_blocks <= 0:
                    logger.error(
                        f"Npu out of memory, OOM.\n"
                        f"The calculated num_blocks is a negative number: {num_blocks}.\n"
                        f"Please check if: 1) the available NPU memory is sufficient or the mem_allocate_ratio is \
                            appropriate, 2) the max_batch_size is appropriate."
                    )
                    raise RuntimeError(
                        f"Npu out of memory, OOM.\n"
                        f"The calculated num_blocks is a negative number: {num_blocks}.\n"
                        f"Please check if: 1) the available NPU memory is sufficient or the mem_allocate_ratio is \
                            appropriate, 2) the max_batch_size is appropriate."
                    )
                self._groups[group].num_blocks = num_blocks
            elif cache_group_info.type == CacheType.SLIDING_WINDOW:
                self._groups[group].num_blocks = 12 * self._max_batch_size + 2  # NOTE: 上层会预留头尾两个空位
            elif cache_group_info.type == CacheType.SEQUENCE:
                self._groups[group].num_blocks = self._max_batch_size + 2


def get_model_cache_pool(model=None, device=None, max_batch_size=None) -> ModelCachePool:
    instance = ModelCachePool()

    if not instance.initialized:
        instance.initialize(model, device, max_batch_size)
    return ModelCachePool()
