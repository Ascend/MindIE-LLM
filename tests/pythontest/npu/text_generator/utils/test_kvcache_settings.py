# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import unittest
from unittest.mock import patch
from dataclasses import dataclass
import torch
import mindspore

from mindie_llm.text_generator.utils.kvcache_settings import (
    NPUSocInfo,
    KVCacheSettings,
)
from mindie_llm.modeling.backend_type import BackendType


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


class TestNPUSocInfo(unittest.TestCase):
    @patch("acl.get_soc_name")
    def test_support_nz_positive(self, mock_get_soc):
        # 测试支持NZ的SOC名称
        for name in ["910PremiumA", "910ProA", "910A", "910ProB", "910B", "310P"]:
            mock_get_soc.return_value = name
            soc_info = NPUSocInfo()
            self.assertTrue(soc_info.support_nz())

    @patch("acl.get_soc_name")
    def test_support_nz_negative(self, mock_get_soc):
        # 测试不支持NZ的SOC名称
        for name in ["910", "310", "other"]:
            mock_get_soc.return_value = name
            soc_info = NPUSocInfo()
            self.assertFalse(soc_info.support_nz())

    @patch("acl.get_soc_name")
    def test_soc_name_none(self, mock_get_soc):
        mock_get_soc.return_value = None
        soc_info = NPUSocInfo()
        assert soc_info.support_nz() is False
        

class TestKVCacheSettings(unittest.TestCase):
    def get_base_model_info(self):
        return MockModelInfo(
            num_kv_heads=4,
            head_size=64,
            num_layers=2,
            kvcache_quant_layers=[False, False]
        )

    @patch("acl.get_soc_name")
    def test_init_basic(self, mock_get_soc):
        mock_get_soc.return_value = "other"  # 不支持NZ
        base_model_info = self.get_base_model_info()
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )

        # 验证基础属性
        self.assertEqual(settings.num_layers, 2)
        self.assertEqual(settings.num_heads, 4)
        self.assertFalse(settings.need_nz)

    @patch("acl.get_soc_name")
    def test_cal_kv_total_head_size(self, mock_get_soc):
        mock_get_soc.return_value = "910A"  # 支持NZ
        base_model_info = self.get_base_model_info()
        base_model_info.enable_nz = True
        base_model_info.index_head_dim = 16
        base_model_info.num_index_heads = 2
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )

        total, k_total, v_total, k_quant_total, index_total = settings._cal_kv_total_head_size()
        self.assertEqual(total, 256)
        self.assertEqual(k_total, 256)
        self.assertEqual(v_total, 256)
        self.assertEqual(k_quant_total, 256)
        self.assertEqual(index_total, 32)

    @patch("acl.get_soc_name")
    def test_cal_set_kv_block_shapes(self, mock_get_soc):
        mock_get_soc.return_value = "910A"
        base_model_info = self.get_base_model_info()
        base_model_info.enable_nz = True
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )
        settings._cal_set_kv_block_shapes()

        self.assertEqual(settings.block_shape, (16, 16, 16))
        self.assertEqual(settings.k_block_shape, (16, 16, 16))

    @patch("acl.get_soc_name")
    def test_cal_set_kv_block_shapes_with_index_head_dim(self, mock_get_soc):
        mock_get_soc.return_value = "910A"
        base_model_info = self.get_base_model_info()
        base_model_info.enable_nz = True
        base_model_info.index_head_dim = 16
        base_model_info.num_index_heads = 2
        settings = KVCacheSettings(
            rank=0,
            model_info=base_model_info,
            cpu_mem=1024 ** 3,
            npu_mem=2 * 1024 ** 3,
            block_size=16,
            backend_type=BackendType.ATB,
            is_separated_pd=False
        )
        settings._cal_set_kv_block_shapes()

        self.assertEqual(settings.block_shape, (16, 16, 16))
        self.assertEqual(settings.k_block_shape, (16, 16, 16))
        self.assertEqual(settings.index_block_shape, (2, 16, 16))

    def test_dtype_to_str(self):
        # 测试数据类型转换
        self.assertEqual(
            KVCacheSettings.dtype_to_str(BackendType.ATB, torch.float16),
            "float16"
        )
        # 测试不匹配的后端和数据类型
        with self.assertRaises(Exception):
            KVCacheSettings.dtype_to_str(BackendType.ATB, mindspore.float16)



if __name__ == "__main__":
    unittest.main()