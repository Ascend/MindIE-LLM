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

import json
from dataclasses import dataclass
from typing import Union, List
import threading
import torch
import torch_npu
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.file_utils import safe_open
from .base import MemPool
from .utils import flatten_tensors, parse_global_segment_size


BASE_PORT = 8790
DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB
_global_te = None
_global_te_lock = threading.Lock()


def get_global_te(hostname: str, device_name: str):
    from mooncake.engine import TransferEngine
    global _global_te
    if _global_te is None:
        with _global_te_lock:
            # Double-Checked Locking
            if _global_te is None:
                if TransferEngine is None:
                    raise RuntimeError("mooncake is not available")
                transfer_engine = TransferEngine()
                device_name = device_name if device_name is not None else ""
                ret_value = transfer_engine.initialize(hostname,
                                                       "P2PHANDSHAKE",
                                                       "ascend", device_name)
                if ret_value != 0:
                    raise RuntimeError(
                        f"TransferEngine initialization failed with ret_value: {ret_value}"
                    )
                _global_te = transfer_engine
    return _global_te


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    use_ascend_direct: bool

    @classmethod
    def parse_config(cls, config_path: str, role: str):
        with safe_open(config_path) as fin:
            config = json.load(fin)
        required = ["local_hostname", "metadata_server", "master_server_address"]
        missing = [k for k in required if not str(config.get(k, "")).strip()]
        if missing:
            msg = f"Configuration missing or empty fileds: {', '.join(missing)}."
            raise ValueError(msg)
        ASCEND_PROTOCOL = "ascend"
        protocol = str(config.get("protocol", ASCEND_PROTOCOL)).strip().lower()
        if protocol != ASCEND_PROTOCOL:
            raise ValueError(f"The current mooncake protocol supports `{ASCEND_PROTOCOL}`, but got `{protocol}`.")
        pool_config = MooncakeStoreConfig(
            local_hostname=str(config.get("local_hostname")).strip(),
            metadata_server=str(config.get("metadata_server")).strip(),
            global_segment_size=parse_global_segment_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)),
            local_buffer_size=config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE),
            protocol=protocol,
            device_name=str(config.get("device_name", "")).strip(),
            master_server_address=str(config.get("master_server_address")).strip(),
            use_ascend_direct=bool(config.get("use_ascend_direct", True))
        )
        if role == "scheduler":
            pool_config.local_buffer_size = 0
            pool_config.global_segment_size = 0
            pool_config.protocol = "ascend_no_transport"
            torch_npu.npu.set_device(0)
        return pool_config


class MooncakeMempool(MemPool):
    def __init__(self, config_path: str, role: str, **kwargs):
        try:
            self.config = MooncakeStoreConfig.parse_config(config_path, role)
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError("Please install mooncake.") from e
        device_id = kwargs.get("device_id", 0)
        self.store = MooncakeDistributedStore()
        self.rep_config = None
        if self.config.protocol in ["ascend", "ascend_no_transport"] and not self.config.use_ascend_direct:
            self.config.local_hostname = self.config.local_hostname + ":" + str(BASE_PORT + device_id) + \
                ":" + "npu_" + str(device_id)
            ret = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol, self.config.device_name,
                self.config.master_server_address
            )
        else:
            transfer_engine = get_global_te(self.config.local_hostname, device_name=None)
            self.local_seg = self.config.local_hostname + ":" + str(
                transfer_engine.get_rpc_port())
            ret = self.store.setup(self.local_seg, self.config.metadata_server,
                                   self.config.global_segment_size,
                                   self.config.local_buffer_size,
                                   self.config.protocol,
                                   self.config.device_name,
                                   self.config.master_server_address,
                                   transfer_engine.get_engine())
            from mooncake.store import ReplicateConfig
            self.rep_config = ReplicateConfig()
            self.rep_config.preferred_segment = self.local_seg
            self.rep_config.prefer_alloc_in_same_node = True

        if ret != 0:
            msg = f"Initialize mooncake failed, ret: {ret}"
            logger.error(msg)
            raise RuntimeError(msg)
        
        if self.config.use_ascend_direct and role == "worker":
            kv_caches = kwargs.get("kv_caches", None)
            if kv_caches is None:
                msg = "Mooncake init failed, expected kv caches to register, but it is None."
                raise RuntimeError(msg)
            self._register(kv_caches)

        logger.info(f"Initialize {role} mooncake mempool success.")
    
    def exists(self, key: str, **kwargs) -> bool:
        if not isinstance(key, str):
            logger.error(f"MooncakeMempool `exists()` expects `key` to be a string, but got {type(key)}.")
            return False
        return self.store.is_exist(key) == 1
    
    def batch_exist(self, keys: List[str]) -> List[bool]:
        if not isinstance(keys, list):
            logger.error(f"MooncakeMempool `batch_is_exist()` expects `key` to be a List[str], but got {type(keys)}.")
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
        
        if not self.config.use_ascend_direct:
            return self._ascend_transport_put(keys, tensors, **kwargs)
        else:
            return self._put(keys, tensors, **kwargs)

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
        
        if not self.config.use_ascend_direct:
            return self._ascend_transport_get(keys, tensors, **kwargs)
        else:
            return self._get(keys, tensors, **kwargs)
        
    def close(
        self
    ) -> None:
        self.store.close()
        logger.info("Closed the mooncake store connection")

    def _get_addr_size(self, tensors):
        addrs = []
        sizes = []
        flat_tensors = flatten_tensors(tensors)
        for tensor in flat_tensors:
            addrs.append(tensor.data_ptr())
            sizes.append(tensor.numel() * tensor.element_size())

        return addrs, sizes

    def _register(self, kv_caches):
        for kv in kv_caches:
            k = kv[0]
            v = kv[1]
            res = self.store.register_buffer(k.data_ptr(), k.numel() * k.element_size())
            if res != 0:
                logger.error(f"Failed to register key buffer, ret: {res}")
            res = self.store.register_buffer(v.data_ptr(), v.numel() * v.element_size())
            if res != 0:
                logger.error(f"Failed to register value buffer, ret: {res}")
    
    def _put(self, keys: Union[str, List[str]], tensors: Union[torch.Tensor, List], **kwargs) -> List[bool]:
        all_addrs = []
        all_sizes = []
        for single_tensor in tensors:
            addr, size = self._get_addr_size(single_tensor)
            all_addrs.append(addr[:])
            all_sizes.append(size[:])
       
        try:
            ret = self.store.batch_put_from_multi_buffers(keys, all_addrs, all_sizes, self.rep_config)
            all_result = [x == 0 for x in ret]
        except Exception as e:
            logger.error(f"Failed to put keys to mooncake mempool: {e}")
            all_result = [False] * len(keys)
        return all_result
    
    def _ascend_transport_put(
        self, keys: Union[str, List[str]],
        tensors: Union[torch.Tensor, List],
        **kwargs
    ) -> List[bool]:
        all_result = []
        for key, tensor in zip(keys, tensors):        
            if self.exists(key):
                all_result.append(True)
                continue
            try:
                addrs, sizes = self._get_addr_size(tensor)
                ret = self.store.batch_put_from_ascend(key, addrs, sizes)
                if ret[0] != 0:
                    logger.error(f"Failed to put key {key}, ret: {ret[0]}")
                all_result.append(ret[0] == 0)
            except Exception:
                logger.error(f"Failed to put key {key}.")
                all_result.append(False)
        return all_result
    
    def _get(self, keys: Union[str, List[str]], tensors: Union[torch.Tensor, List], **kwargs) -> List[bool]:
        all_addrs = []
        all_sizes = []
        all_expect = []
        for single_tensor in tensors:
            addr, size = self._get_addr_size(single_tensor)
            expect = sum(size)
            all_addrs.append(addr[:])
            all_sizes.append(size[:])
            all_expect.append(expect)
       
        try:
            ret = self.store.batch_get_into_multi_buffers(keys, all_addrs, all_sizes, True)
            all_result = [x == y for x, y in zip(ret, all_expect)]
        except Exception as e:
            logger.error(f"Failed to get keys from mooncake mempool: {e}")
            all_result = [False] * len(keys)
        return all_result
        
    def _ascend_transport_get(self,
        keys: Union[str, List[str]], 
        tensors: Union[torch.Tensor, List], 
        **kwargs
    ) -> List[bool]:

        all_result = []
        for key, tensor in zip(keys, tensors):
            if not self.exists(key):
                all_result.append(False)
                logger.error(f"The key does not exist: [{key}].")
                continue
            try:
                addrs, sizes = self._get_addr_size(tensor)
                expect_res = sum(sizes)
                res = self.store.batch_get_into_ascend(key, addrs, sizes)
                if res[0] != expect_res:
                    logger.error(f"Failed to get key: [{key}], ret: {res}.")
                all_result.append(res[0] == expect_res)
            except Exception as e:
                logger.error(f"Failed to get key: [{key}]. {e}")
                all_result.append(False)
        return all_result