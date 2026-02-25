# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import patch
from dataclasses import dataclass
import torch
import mindspore

from mindie_llm.text_generator.utils.npu_mem_tool import (
    calc_block_mem,
    calc_npu_mem,
    gb,
    NpuMemoryWatcher,
)


# 模拟ModelInfo类
@dataclass
class MockModelInfo:
    num_kv_heads: int
    head_size: int
    k_head_size: int = 0
    v_head_size: int = 0
    num_layers: int = 1
    data_byte_size: int = 2
    kvcache_quant_layers: list = None
    dtype: type = torch.float16
    enable_nz: bool = False
    index_head_dim: int = None
    num_index_heads: int = None

    def __post_init__(self):
        if self.kvcache_quant_layers is None:
            self.kvcache_quant_layers = []


class TestMemoryCalculation(unittest.TestCase):
    def get_base_model_info(self):
        return MockModelInfo(
            num_kv_heads=4,
            head_size=64,
            num_layers=2,
            kvcache_quant_layers=[False, False]
        )

    def test_calc_block_mem_basic(self):
        # 基础计算测试
        base_model_info = self.get_base_model_info()
        block_size = 16
        mem = calc_block_mem(base_model_info, block_size)

        expected = 2 * (4 * 64 + 4 * 64) * 2 * 16
        self.assertEqual(mem, expected)

    def test_calc_block_mem_with_quant(self):
        base_model_info = self.get_base_model_info()
        base_model_info.kvcache_quant_layers = [True, False]
        block_size = 16
        mem = calc_block_mem(base_model_info, block_size)

        layer1 = (4 * 64) * 1 + (4 * 64) * 2
        layer2 = (4 * 64) * 2 + (4 * 64) * 2
        expected = (layer1 + layer2) * 16
        self.assertEqual(mem, expected)

    def test_calc_block_mem_speculative(self):
        base_model_info = self.get_base_model_info()
        mem = calc_block_mem(base_model_info, 16, num_speculative_tokens=1)

        expected = 3 * (4 * 64 + 4 * 64) * 2 * 16
        self.assertEqual(mem, expected)

    def test_calc_npu_mem(self):
        base_model_info = self.get_base_model_info()
        block_nums = 10
        block_size = 16
        npu_mem = calc_npu_mem(block_nums, base_model_info, block_size)
        block_mem = calc_block_mem(base_model_info, block_size)
        self.assertEqual(npu_mem, block_nums * block_mem)

    def test_gb_conversion(self):
        # 测试GB转换
        self.assertEqual(gb(1024 ** 3), 1.0)  # 1GB
        self.assertEqual(gb(2 * 1024 ** 3), 2.0)  # 2GB
        self.assertAlmostEqual(gb(512 * 1024 ** 2), 0.5, places=2)  # 512MB


class Test_watch_npu_mem(unittest.TestCase):
    @patch("acl.rt.get_mem_info")
    @patch("mindie_llm.text_generator.utils.kvcache_settings.npu.synchronize")
    def test_watch_npu_mem(self, mock_sync, mock_get_mem):
        watch = NpuMemoryWatcher()
        mock_get_mem.return_value = (512 * 1024 ** 2, 1024 ** 3, 0)
        total, peak = watch.watch_npu_mem(0, "success", False, 65536)
        self.assertEqual(total, 1024 ** 3)
        self.assertEqual(peak, 512 * 1024 ** 2)
        warmup_mem = 128 * 1024 ** 2
        watch._set_warmup_mem_(warmup_mem)
        mock_get_mem.return_value = (256 * 1024 ** 2, 1024 ** 3, 0)
        total, peak = watch.watch_npu_mem(0, "success", False, 65536, 0)
        total, peak = watch.watch_npu_mem(0, "success", False, 65536, 0)
        mock_get_mem.return_value = (128 * 1024 ** 2, 1024 ** 3, 0)
        total, peak = watch.watch_npu_mem(0, "success", False, 65536, 0)



if __name__ == "__main__":
    unittest.main()