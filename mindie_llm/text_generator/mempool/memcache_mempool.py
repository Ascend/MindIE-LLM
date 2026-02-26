#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
from typing import Union, List
from enum import Enum
import torch
from mindie_llm.utils.log.logging import logger
from .base import MemPool
from .utils import flatten_tensors


class MmcDirect(Enum):
    COPY_L2G = 0
    COPY_G2L = 1
    COPY_G2H = 2
    COPY_H2G = 3

    
class MemcacheMempool(MemPool):
    def __init__(self, config_path: str, role: str, **kwargs):
        local_config = os.environ.get("MMC_LOCAL_CONFIG_PATH", None)
        if not local_config:
            os.environ["MMC_LOCAL_CONFIG_PATH"] = config_path
        try:
            from memcache_hybrid import DistributedObjectStore
        except ImportError as e:
            raise ImportError("Please install memcache.") from e
        
        device_id = kwargs.get("device_id", 0)
        init_bm = True if role == "worker" else False
        try:
            self.store = DistributedObjectStore()
            ret = self.store.init(device_id, init_bm)
        except Exception as e:
            msg = f"Initialize memcache failed with error, {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        if ret != 0:
            msg = f"Initialize memcache failed, ret: {ret}."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"Initialize {role} memcache mempool success.")
    
    def exists(self, key: str, **kwargs) -> bool:
        if not isinstance(key, str):
            logger.error(f"MemcacheMempool `exists()` expects `key` to be a string, but got {type(key)}.")
            return False
        return self.store.is_exist(key) == 1
    
    def batch_exist(self, keys: List[str]) -> List[bool]:
        if not isinstance(keys, list):
            logger.error(f"MemcacheMempool `batch_is_exist()` expects `key` to be a List[str], but got {type(keys)}.")
            return [False]
        ret = self.store.batch_is_exist(keys)
        all_result = [x == 1 for x in ret]
        return all_result

    def put(self, keys: Union[str, List[str]], tensors: Union[torch.Tensor, List], **kwargs) -> List[bool]:
        if isinstance(keys, str):
            keys = [keys]
            tensors = [tensors]
        elif isinstance(keys, list) and isinstance(keys[0], str):
            if not isinstance(tensors, List) or len(tensors) != len(keys):
                logger.error(f"The tensors should be Union[torch.Tensor, List]: got {type(tensors)} " \
                    "And the lengths of keys and tensors must be equal: " \
                    f"got {len(keys)} keys and {len(tensors)} tensors.")
                return [False]
        else:
            logger.error(f'The keys should be of type Union[str, List[str]], but got {type(keys)}.')
            return [False]
        
        all_addrs = []
        all_sizes = []
        for single_tensor in tensors:
            addr, size = self._get_addr_size(single_tensor)
            all_addrs.append(addr[:])
            all_sizes.append(size[:])
       
        try:
            ret = self.store.batch_put_from_layers(keys, all_addrs, all_sizes, MmcDirect.COPY_L2G.value)
            all_result = [x == 0 for x in ret]
        except Exception as e:
            logger.error(f"Failed to put keys to memcache mempool: {e}")
            all_result = [False] * len(keys)
        return all_result

    def get(self, keys: Union[str, List[str]], tensors: Union[torch.Tensor, List], **kwargs) -> List[bool]:
        if isinstance(keys, str):
            keys = [keys]
            tensors = [tensors]
        elif isinstance(keys, list) and isinstance(keys[0], str):
            if not isinstance(tensors, List) or len(tensors) != len(keys):
                logger.error(f"The tensors should be Union[torch.Tensor, List]: got {type(tensors)} " \
                    "And the lengths of keys and tensors must be equal: " \
                    f"got {len(keys)} keys and {len(tensors)} tensors.")
                return [False]
        else:
            logger.error(f'The keys should be of type Union[str, List[str]], but got {type(keys)}.')
            return [False]
        
        all_addrs = []
        all_sizes = []
        for single_tensor in tensors:
            addr, size = self._get_addr_size(single_tensor)
            all_addrs.append(addr[:])
            all_sizes.append(size[:])
       
        try:
            ret = self.store.batch_get_into_layers(keys, all_addrs, all_sizes, MmcDirect.COPY_G2L.value)
            all_result = [x == 0 for x in ret]
        except Exception as e:
            logger.error(f"Failed to get keys from memcache mempool: {e}")
            all_result = [False] * len(keys)
        return all_result
        
    def close(
        self
    ) -> None:
        self.store.close()
        logger.info("Closed the memcache store connection")

    def _get_addr_size(self, tensors):
        addrs = []
        sizes = []
        flat_tensors = flatten_tensors(tensors)
        for tensor in flat_tensors:
            addrs.append(tensor.data_ptr())
            sizes.append(tensor.numel() * tensor.element_size())

        return addrs, sizes