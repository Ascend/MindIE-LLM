# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from typing import List, Tuple
from functools import reduce
import operator
import torch
import torch_npu

from ....utils.log.error_code import ErrorCode
from ....utils.log.logging import logger
from ....utils.env import ENV
from ....text_generator.utils.kvcache_settings import KVCacheSettings

# memory bridge swapper
if ENV.use_mb_swapper:
    from _memory_bridge import _Swapper

    MBSWAPPER = _Swapper.get_instance()


class KVCachePool:
    """KVCachePool manages physical kv cache with layered kv tensors. originally CachePool"""

    __slots__ = [
        "kvcache_settings",
        "device",
        "npu_cache",
        "cpu_cache",
        "npu_blocks_addrs",
        "cpu_blocks_addrs",
        "k_blocks_addrs",
        "v_blocks_addrs",
        "k_blocks_quant_addrs",
        "index_blocks_addrs",
        "swapper",
        "swap_kvcache_method",
        "use_mb_swapper",  # memory bridges swapper
        "acl_format",
        "total_h2d_swap_count",
        "total_d2h_swap_count",
        "enable_kv_pool",
    ]

    def __init__(self, kvcache_settings: KVCacheSettings, device, enable_kv_pool=False):

        self.npu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = (
            []
        )  # for each layer, first tensor is K, second tensor is V
        self.cpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = (
            []
        )  # for each layer, first tensor is K, second tensor is V
        # use kvcache setting to replace original member copy
        self.kvcache_settings = kvcache_settings
        self.device = device
        self.enable_kv_pool = enable_kv_pool

        need_nz = kvcache_settings.npu_info.need_nz
        self.acl_format = 29 if need_nz else 2  # 29: NZ; 2: ND
        self.total_h2d_swap_count = 0
        self.total_d2h_swap_count = 0
        swap_func_mapping = {"MB": self.sync_swap_mb, "BASE": self.sync_swap_base}
        self.use_mb_swapper = ENV.use_mb_swapper
        # for mb, it is first layer's every block's address: List[Tuple(k_block_addr, v_block_addr)]
        # for base, it is each layers's first block's address
        self.cpu_blocks_addrs = []
        # first layer npu blocks address: List[Tuple(k_block_addr, v_block_addr)]
        self.npu_blocks_addrs = []
        self.k_blocks_addrs = []
        self.v_blocks_addrs = []
        self.k_blocks_quant_addrs = []
        self.index_blocks_addrs = []
        if self.use_mb_swapper:
            logger.debug(">>>You are using mb swapper.")
            self.npu_blocks_addrs = []  # 各个block的 K起始地址和V起始地址 [(k0,v0),(k1,v1),...]
            self.cpu_blocks_addrs = []  # 各个block的 K起始地址和V起始地址 [(k0,v0),(k1,v1),...]
            self.swapper = MBSWAPPER  # 初始化kv cache的swapper
            self.swap_kvcache_method = swap_func_mapping.get("MB")
        else:
            logger.debug(">>>You are using torch swapper.")
            self.swap_kvcache_method = swap_func_mapping.get("BASE")

    # model id 0: k_blocks_addrs, npu_blocks_addrs
    # model id 1: v_blocks_addrs
    # model id 2: k_blocks_quant_addrs
    def get_npu_blocks_addrs(self):  # TBD: for KVMover block address registration !!!!
        return self.npu_blocks_addrs

    def allocate_npu_kvcache(self):
        self.npu_cache.clear()
        self.npu_blocks_addrs.clear()
        self.k_blocks_addrs.clear()
        self.v_blocks_addrs.clear()
        self.k_blocks_quant_addrs.clear()
        self.index_blocks_addrs.clear()
        if self.kvcache_settings.num_npu_blocks < 0:
            message = (
                "Num_npu_blocks must be non-negative.\n"
                "Debugging suggestions:\n"
                "1. If the error occurred in the prefill:"
                " Try to reduce 'max_prefill_tokens' to decrease memory usage\n"
                "2. If the error occurred in the decoder:"
                " Try to Reduce 'max_batch_size' to lower concurrent request load\n"
            )
            logger.error(message, ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY)
            raise ValueError(
                f"Invalid number of NPU blocks: {self.kvcache_settings.num_npu_blocks}. "
                "It must be non-negative. This is due to warm_up out of memory."
            )
        if self.use_mb_swapper:
            self._allocate_npu_kvcache_mb()
        else:
            self._allocate_npu_kvcache_base()

    def allocate_cpu_kvcache(self):
        if self.use_mb_swapper:
            raw_blocks = torch.empty(
                size=(
                    self.kvcache_settings.num_layers,
                    2,
                    self.kvcache_settings.num_cpu_blocks,
                    *self.kvcache_settings.block_shape,
                ),
                dtype=self.kvcache_settings.dtype,
            ).pin_memory()
            logger.debug(f"raw_blocks: {raw_blocks.shape}, {raw_blocks.dtype}, {raw_blocks.device}")

            # 更新各个block的首miniblock地址列表(cache矩阵的首行，各列地址差值为 self.mini_block_bytes)
            # miniblock 属于新概念， 缺乏明确定义？？
            for j in range(self.kvcache_settings.num_cpu_blocks):
                self.cpu_blocks_addrs.append(
                    (
                        raw_blocks.data_ptr() + j * self.kvcache_settings.mini_block_bytes,
                        raw_blocks.data_ptr()
                        + (j + self.kvcache_settings.num_cpu_blocks) * self.kvcache_settings.mini_block_bytes,
                    )
                )

            # 更新self.cpu_cache，新方案的各行的K和V仍然为全局连续，K为偶数行，V为奇数行
            for i in range(self.kvcache_settings.num_layers):
                key_blocks = raw_blocks[i][0]
                value_blocks = raw_blocks[i][1]
                self.cpu_cache.append((key_blocks, value_blocks))
        else:
            for _ in range(self.kvcache_settings.num_layers):
                key_blocks = (
                    torch.empty(
                        size=(self.kvcache_settings.num_cpu_blocks, *self.kvcache_settings.k_block_shape),
                        dtype=self.kvcache_settings.dtype,
                    )
                    if self.kvcache_settings.k_head_size > 0
                    else torch.empty((1,), dtype=self.kvcache_settings.dtype)
                )
                value_blocks = (
                    torch.empty(
                        size=(self.kvcache_settings.num_cpu_blocks, *self.kvcache_settings.v_block_shape),
                        dtype=self.kvcache_settings.dtype,
                    )
                    if self.kvcache_settings.v_head_size > 0
                    else torch.empty((1,), dtype=self.kvcache_settings.dtype)
                )
                self.cpu_cache.append((key_blocks, value_blocks))
                self.cpu_blocks_addrs.append(key_blocks.data_ptr())
                self.cpu_blocks_addrs.append(value_blocks.data_ptr())

    def sync_swap_mb(self, swap_decision: List[Tuple]):
        """use memory bridge swapper, format of swap_decision: [swap_type, src_block, dst_block]??"""
        """ never used """
        # `swap_decision_tensor是一组swap请求，每个请求有三个分量：in/out、srcBlockId、dstBlockId
        if self.kvcache_settings.is_separated_pd:
            raise Exception("SeparateDeploymentEngine not support mb_swapper")
        params = [
            self.kvcache_settings.cpu_row_bytes,
            self.kvcache_settings.npu_row_bytes,
            self.kvcache_settings.mini_block_bytes,
            self.kvcache_settings.num_layers,
        ]
        for decision in swap_decision:
            kind = int(decision[0].item())
            src_block_id = int(decision[1])
            dst_block_id = int(decision[2])
            src_in_cpu_block_flag = src_block_id >= 0 and src_block_id < len(self.cpu_blocks_addrs)
            src_in_npu_block_flag = src_block_id >= 0 and src_block_id < len(self.npu_blocks_addrs)
            dst_in_npu_block_flag = dst_block_id >= 0 and dst_block_id < len(self.npu_blocks_addrs)
            dst_in_cpu_block_flag = dst_block_id >= 0 and dst_block_id < len(self.cpu_blocks_addrs)
            if kind == 0 and src_in_cpu_block_flag and dst_in_npu_block_flag:  # swap in
                cpu_key_block_addr = self.cpu_blocks_addrs[decision[1]][0]
                npu_key_block_addr = self.npu_blocks_addrs[decision[2]][0]
                cpu_value_block_addr = self.cpu_blocks_addrs[decision[1]][1]
                npu_value_block_addr = self.npu_blocks_addrs[decision[2]][1]
                self.swapper.h2d_swap(cpu_key_block_addr, npu_key_block_addr, params)
                self.swapper.h2d_swap(cpu_value_block_addr, npu_value_block_addr, params)
                self.total_h2d_swap_count += 1
            elif kind == 1 and src_in_npu_block_flag and dst_in_cpu_block_flag:
                npu_key_block_addr = self.npu_blocks_addrs[decision[1]][0]
                cpu_key_block_addr = self.cpu_blocks_addrs[decision[2]][0]
                npu_value_block_addr = self.npu_blocks_addrs[decision[1]][1]
                cpu_value_block_addr = self.cpu_blocks_addrs[decision[2]][1]
                self.swapper.d2h_swap(cpu_key_block_addr, npu_key_block_addr, params)
                self.swapper.d2h_swap(cpu_value_block_addr, npu_value_block_addr, params)
                self.total_d2h_swap_count += 1
            else:
                continue

    def sync_swap_base(self, swap_decision):
        """use torch swapper, format of swap_decision: [swap_type, src_block, dst_block]??"""
        """ never used """
        swap_cnt = 0
        for decision in swap_decision:
            logger.debug("[CacheManager]\t>>> Swap direction %s from %s to %s", decision[0], decision[1], decision[2])
            decision_type = decision[0]
            if decision_type == 0 or decision_type == 1:
                swap_cnt += 1
                for i in range(self.kvcache_settings.num_layers):
                    device_key: torch.Tensor = self.npu_cache[i][0]
                    device_value: torch.Tensor = self.npu_cache[i][1]
                    host_key: torch.Tensor = self.cpu_cache[i][0]
                    host_value: torch.Tensor = self.cpu_cache[i][1]

                    src, dst = decision[1], decision[2]
                    if decision_type == 1:
                        # swap out
                        device_block_no = src
                        host_block_no = dst

                        if self.kvcache_settings.k_head_size > 0:
                            tmp_k = device_key[device_block_no, ...]
                            host_key[host_block_no, ...] = tmp_k

                        if self.kvcache_settings.v_head_size > 0:
                            tmp_v = device_value[device_block_no, ...]
                            host_value[host_block_no, ...] = tmp_v
                    elif decision_type == 0:
                        # swap in
                        device_block_no = dst
                        host_block_no = src

                        if self.kvcache_settings.k_head_size > 0:
                            tmp_k = host_key[host_block_no, ...]
                            device_key[device_block_no, ...] = tmp_k

                        if self.kvcache_settings.v_head_size > 0:
                            tmp_v = host_value[host_block_no, ...]
                            device_value[device_block_no, ...] = tmp_v

    def _create_aligned_tensor(self, target_shape, dtype):
        if self.acl_format == 2:  # 2: ND
            element_size = torch.tensor([], dtype=dtype).element_size()
            desired_num_elements = reduce(operator.mul, target_shape, 1)
            tensor_size = desired_num_elements * element_size
            additional_memory = 2 * 1024 * 1024
            total_size = tensor_size + additional_memory

            kv_tensor = torch.empty(size=(total_size // element_size,), dtype=dtype, device=self.device)
            tensor_ptr = kv_tensor.data_ptr()
            alignment = 2 * 1024 * 1024
            offset = tensor_ptr % alignment
            if offset != 0:
                cut_size = alignment - offset
            else:
                cut_size = 0
            # 切片对齐 tensor
            cut_elements = cut_size // element_size
            aligned_tensor = kv_tensor[cut_elements: cut_elements + desired_num_elements]
            aligned_tensor = aligned_tensor.contiguous()
            aligned_tensor = aligned_tensor.view(*target_shape)
            return aligned_tensor

        else:  # NZ
            additional_memory_kb = 2 * 1024
            kv_tensor = torch_npu.empty_with_format(
                size=target_shape,
                dtype=dtype,
                device=self.device,
                acl_format=self.acl_format,
                base_addr_aligned_kb=additional_memory_kb,
            )

            return kv_tensor

    # for memory bridge
    def _allocate_npu_kvcache_mb(self):
        if self.kvcache_settings.is_separated_pd:
            raw_blocks = self._create_aligned_tensor(
                (
                    self.kvcache_settings.num_layers,
                    2,
                    self.kvcache_settings.num_npu_blocks,
                    *self.kvcache_settings.block_shape,
                ),
                self.kvcache_settings.dtype,
            )
        else:
            raw_blocks = torch_npu.empty_with_format(
                size=(
                    self.kvcache_settings.num_layers,
                    2,
                    self.kvcache_settings.num_npu_blocks,
                    *self.kvcache_settings.block_shape,
                ),
                dtype=self.kvcache_settings.dtype,
                device=self.device,
                acl_format=self.acl_format,
            )
        logger.debug(f"raw_blocks: {raw_blocks.shape}, {raw_blocks.dtype}, {raw_blocks.device}")

        # 更新各个block的首miniblock地址列表(cache矩阵的首行，各列地址差值为 self.mini_block_bytes)
        for j in range(self.kvcache_settings.num_npu_blocks):
            self.npu_blocks_addrs.append(
                (
                    raw_blocks.data_ptr() + j * self.kvcache_settings.mini_block_bytes,
                    raw_blocks.data_ptr()
                    + (j + self.kvcache_settings.num_npu_blocks) * self.kvcache_settings.mini_block_bytes,
                )
            )

        # move self.sepd_worker.set_npu_cache(0, self.npu_blocks_addrs) to generator !!!!

        # 更新self.npu_cache，新方案的各行的K和V仍然为全局连续，K为偶数行，V为奇数行
        for i in range(self.kvcache_settings.num_layers):
            key_blocks = raw_blocks[i][0]
            value_blocks = raw_blocks[i][1]
            self.npu_cache.append((key_blocks, value_blocks))

    def _allocate_npu_kvcache_base(self):
        logger.info(f"Allocating npu_cache, acl format is {'NZ' if self.acl_format == 29 else 'ND'}")  # 29: NZ
        per_layer_k_cache_dtype = [self.kvcache_settings.dtype] * self.kvcache_settings.num_layers
        if self.kvcache_settings.kvcache_quant_layers:
            for i, kvcache_quant in enumerate(self.kvcache_settings.kvcache_quant_layers):
                if kvcache_quant:
                    per_layer_k_cache_dtype[i] = torch.int8

        if self.kvcache_settings.is_separated_pd:
            if self.kvcache_settings.k_head_size != self.kvcache_settings.v_head_size:
                for dtype in per_layer_k_cache_dtype:
                    k_block_shape = (
                        self.kvcache_settings.k_block_shape
                        if dtype != torch.int8
                        else self.kvcache_settings.k_block_quant_shape
                    )
                    if self.kvcache_settings.k_head_size > 0:
                        key_blocks = self._create_aligned_tensor(
                            (self.kvcache_settings.num_npu_blocks, *k_block_shape), dtype
                        )
                        if dtype == torch.int8:
                            self.k_blocks_quant_addrs.append(key_blocks.data_ptr())
                        else:
                            self.k_blocks_addrs.append(key_blocks.data_ptr())
                    else:
                        key_blocks = torch.empty((1,), dtype=dtype, device=self.device)
                    if self.kvcache_settings.v_head_size > 0:
                        value_blocks = self._create_aligned_tensor(
                            (self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.v_block_shape),
                            self.kvcache_settings.dtype
                        )
                        self.v_blocks_addrs.append(value_blocks.data_ptr())
                    else:
                        value_blocks = torch.empty((1,), dtype=self.kvcache_settings.dtype, device=self.device)

                    if self.kvcache_settings.index_head_dim is not None:
                        index_blocks = self._create_aligned_tensor(
                            (self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.index_block_shape),
                            self.kvcache_settings.dtype
                        ) 
                        self.index_blocks_addrs.append(index_blocks.data_ptr())
                        self.npu_cache.append((key_blocks, value_blocks, index_blocks))
                    else:
                        self.npu_cache.append((key_blocks, value_blocks))
                # K npu_cache model_id is 0, V npu_cache model_id is 1, K int8 npu_cache model_id is 2
                # move sepd worker set in generator !!!!
            else:
                # k_head_size is equal to v_head_size
                for _ in range(self.kvcache_settings.num_layers):
                    key_blocks = self._create_aligned_tensor(
                        (self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.k_block_shape),
                        self.kvcache_settings.dtype
                    )
                    value_blocks = self._create_aligned_tensor(
                        (self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.v_block_shape),
                        self.kvcache_settings.dtype
                    )
                    self.npu_cache.append((key_blocks, value_blocks))
                    self.npu_blocks_addrs.append(key_blocks.data_ptr())
                    self.npu_blocks_addrs.append(value_blocks.data_ptr())
                # move sepd worker (kvmover) setting to generator !!!!
        elif self.enable_kv_pool:
            for _, dtype in zip(range(self.kvcache_settings.num_layers), per_layer_k_cache_dtype):
                k_block_shape = self.kvcache_settings.k_block_shape if dtype != torch.int8 \
                    else self.kvcache_settings.k_block_quant_shape
                key_blocks = self._create_aligned_tensor(
                    (self.kvcache_settings.num_npu_blocks, *k_block_shape),
                    dtype) \
                    if self.kvcache_settings.k_head_size > 0 \
                    else torch.empty((1,), dtype=self.kvcache_settings.dtype, device=self.device)
                value_blocks = self._create_aligned_tensor(
                    (self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.v_block_shape),
                    self.kvcache_settings.dtype) \
                    if self.kvcache_settings.v_head_size > 0 \
                    else torch.empty((1,), dtype=self.kvcache_settings.dtype, device=self.device)
                if self.kvcache_settings.index_head_dim is not None:
                    index_blocks = self._create_aligned_tensor(
                        (self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.index_block_shape),
                        self.kvcache_settings.dtype
                    )
                    self.npu_cache.append((key_blocks, value_blocks, index_blocks))
                else:
                    self.npu_cache.append((key_blocks, value_blocks))
        else:
            for _, dtype in zip(range(self.kvcache_settings.num_layers), per_layer_k_cache_dtype):
                k_block_shape = self.kvcache_settings.k_block_shape if dtype != torch.int8 \
                    else self.kvcache_settings.k_block_quant_shape
                key_blocks = torch_npu.empty_with_format(
                    size=(self.kvcache_settings.num_npu_blocks, *k_block_shape),
                    dtype=dtype,
                    device=self.device,
                    acl_format=self.acl_format
                ) if self.kvcache_settings.k_head_size > 0 \
                    else torch.empty((1,), dtype=self.kvcache_settings.dtype, device=self.device)
                value_blocks = torch_npu.empty_with_format(
                    size=(self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.v_block_shape),
                    dtype=self.kvcache_settings.dtype,
                    device=self.device,
                    acl_format=self.acl_format
                ) if self.kvcache_settings.v_head_size > 0 \
                    else torch.empty((1,), dtype=self.kvcache_settings.dtype, device=self.device)
                if self.kvcache_settings.index_head_dim is not None:
                    index_blocks = torch_npu.empty_with_format(
                        size=(self.kvcache_settings.num_npu_blocks, *self.kvcache_settings.index_block_shape),
                        dtype=self.kvcache_settings.dtype,
                        device=self.device,
                        acl_format=self.acl_format
                    )
                    self.npu_cache.append((key_blocks, value_blocks, index_blocks))
                else:
                    self.npu_cache.append((key_blocks, value_blocks))
