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

def CreatePool(backend, config_path, role, **kwargs):
    match backend:
        case "unifiedcache":
            from .unifiedcache_mempool import UnifiedCacheMempool
            engine = UnifiedCacheMempool(config_path, role, **kwargs)
        case "mooncake":
            from .mooncake_mempool import MooncakeMempool
            engine = MooncakeMempool(config_path, role, **kwargs)
        case "memcache":
            from .memcache_mempool import MemcacheMempool
            engine = MemcacheMempool(config_path, role, **kwargs)
        case _:
            raise ValueError(f"Unknown cache engine type {backend} (role is: {role})")
    return engine