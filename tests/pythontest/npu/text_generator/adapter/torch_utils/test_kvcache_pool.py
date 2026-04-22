# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import unittest
from unittest.mock import MagicMock, patch

import torch
from mindie_llm.text_generator.adapter.torch_utils.kvcache_pool import KVCachePool


class MockNpuInfo:
    def __init__(self):
        self.need_nz = False


# 定义一个 DummyKVCacheSettings 用于测试
class DummyKVCacheSettings:
    def __init__(self):
        # 设定只有一层，便于测试
        self.num_layers = 1
        # 假设 key 和 value 的 head size 均为 64
        self.k_head_size = 64
        self.v_head_size = 64
        self.block_size = 1024
        self.block_shape = (64, 64)
        self.k_block_shape = (32, 32)
        self.k_block_quant_shape = (32, 32)
        self.v_block_shape = (32, 32)
        self.dtype = torch.float32
        self.num_cpu_blocks = 4
        self.npu_info = MockNpuInfo()
        self.kvcache_quant_layers = []
        self.mini_block_bytes = 1024
        self.k_mini_block_bytes = 1024
        self.v_mini_block_bytes = 1024
        self.num_npu_blocks = 2
        self.cpu_row_bytes = 128
        self.npu_row_bytes = 128
        self.is_separated_pd = False  # 对应原来的sepd_worker
        self.npu_info = MockNpuInfo()
        self.index_head_dim = None
        self.index_block_shape = (32, 32)


class TestCreateAlignedTensor(unittest.TestCase):
    def setUp(self):
        self.kvcache_settings = DummyKVCacheSettings()
        self.device = "cpu"
        self.kvcache_pool = KVCachePool(self.kvcache_settings, self.device)

    def test_alignment_and_structure(self):
        blockshape = (64, 64, 64)
        num_block = 10
        dtype = torch.float32

        aligned_tensor = self.kvcache_pool._create_aligned_tensor((num_block, *blockshape), dtype)
        aligned_ptr = aligned_tensor.data_ptr()
        alignment = 2 * 1024 * 1024  # 2MB
        self.assertEqual(aligned_ptr % alignment, 0, f"Tensor is not {alignment} bytes aligned. Address: {aligned_ptr}")

    def test_different_dtypes(self):
        blockshape = (32, 32, 32)
        num_block = 5
        dtype_list = [torch.float32, torch.float16]

        for dtype in dtype_list:
            with self.subTest(dtype=dtype):
                aligned_tensor = self.kvcache_pool._create_aligned_tensor((num_block, *blockshape), dtype)
                aligned_ptr = aligned_tensor.data_ptr()
                alignment = 2 * 1024 * 1024  # 2MB
                self.assertEqual(
                    aligned_ptr % alignment,
                    0,
                    f"Tensor is not {alignment} bytes aligned for dtype {dtype}. Address: {aligned_ptr}",
                )


class TestKVCachePoolAllocation(unittest.TestCase):
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch_npu")
    def setUp(self, mock_torch_npu, mock_torch):
        self.kvcache_settings = DummyKVCacheSettings()
        self.device = "cpu"

        # Mock torch.empty to return a mock tensor with correct shape
        def mock_empty(size, dtype, device=None):
            mock_tensor = MagicMock()
            # Calculate the expected shape after view operation in _create_aligned_tensor
            if len(size) == 1:
                # This is the temporary tensor, the shape will be changed later
                mock_tensor.shape = size
            else:
                # This is the final tensor with the correct shape
                mock_tensor.shape = size
            mock_tensor.data_ptr.return_value = 1024
            # Create a mock for tensor elements
            mock_element = MagicMock()
            mock_element.cpu.return_value = mock_element
            mock_element.__eq__.return_value = True
            # Make __getitem__ return the mock element
            mock_tensor.__getitem__.return_value = mock_element
            mock_tensor.contiguous.return_value = mock_tensor
            mock_tensor.view.side_effect = lambda *args: mock_tensor
            mock_tensor.fill_.return_value = None
            return mock_tensor

        # Mock torch.allclose to always return True
        mock_torch.allclose.return_value = True
        mock_torch.empty.side_effect = mock_empty

        # Mock torch_npu.empty_with_format to return a mock tensor with correct shape
        def mock_empty_with_format(size, dtype, device, acl_format):
            mock_tensor = MagicMock()
            mock_tensor.shape = size
            mock_tensor.data_ptr.return_value = 1024
            return mock_tensor

        mock_torch_npu.empty_with_format.side_effect = mock_empty_with_format
        self.kvcache_pool = KVCachePool(self.kvcache_settings, self.device)

    def test_allocate_cpu_kvcache(self):
        self.kvcache_pool.allocate_cpu_kvcache()
        self.assertEqual(len(self.kvcache_pool.cpu_cache), self.kvcache_settings.num_layers, "cpu_cache 层数不符合预期")

        for key_blocks, value_blocks in self.kvcache_pool.cpu_cache:
            expected_key_shape = (self.kvcache_settings.num_cpu_blocks, *self.kvcache_settings.k_block_shape)
            expected_value_shape = (self.kvcache_settings.num_cpu_blocks, *self.kvcache_settings.v_block_shape)
            self.assertEqual(
                key_blocks.shape,
                expected_key_shape,
                f"CPU key 块形状应为 {expected_key_shape}, 实际为 {key_blocks.shape}",
            )
            self.assertEqual(
                value_blocks.shape,
                expected_value_shape,
                f"CPU value 块形状应为 {expected_value_shape}, 实际为 {value_blocks.shape}",
            )

        self.assertEqual(
            len(self.kvcache_pool.cpu_blocks_addrs),
            2 * self.kvcache_settings.num_layers,
            "cpu_blocks_addrs 的地址数量不符合预期",
        )

    def test_allocate_npu_kvcache(self):
        self.kvcache_pool.allocate_npu_kvcache()
        self.assertEqual(len(self.kvcache_pool.npu_cache), self.kvcache_settings.num_layers, "npu_cache 层数不符合预期")

        for key_blocks, value_blocks in self.kvcache_pool.npu_cache:
            # 验证 key_blocks 和 value_blocks 是有效的对象
            self.assertIsNotNone(key_blocks)
            self.assertIsNotNone(value_blocks)
            # 验证它们有 shape 属性
            self.assertTrue(hasattr(key_blocks, "shape"))
            self.assertTrue(hasattr(value_blocks, "shape"))

    def test_allocate_npu_kvcache_with_index_head_dim(self):
        self.kvcache_settings.index_head_dim = 1
        self.kvcache_pool.allocate_npu_kvcache()
        self.assertEqual(len(self.kvcache_pool.npu_cache), self.kvcache_settings.num_layers, "npu_cache 层数不符合预期")

        for key_blocks, value_blocks, index_block in self.kvcache_pool.npu_cache:
            # 验证 key_blocks、value_blocks 和 index_block 是有效的对象
            self.assertIsNotNone(key_blocks)
            self.assertIsNotNone(value_blocks)
            self.assertIsNotNone(index_block)
            # 验证它们有 shape 属性
            self.assertTrue(hasattr(key_blocks, "shape"))
            self.assertTrue(hasattr(value_blocks, "shape"))
            self.assertTrue(hasattr(index_block, "shape"))

    def test_allocate_npu_kvcache_with_negative_blocks(self):
        negative_settings = DummyKVCacheSettings()
        negative_settings.num_npu_blocks = -1
        negative_kvcache_pool = KVCachePool(negative_settings, self.device)

        with self.assertRaises(ValueError) as context:
            negative_kvcache_pool.allocate_npu_kvcache()
        self.assertIn("Invalid number of NPU blocks", str(context.exception))

    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.KVCachePool._create_aligned_tensor")
    def test_allocate_npu_kvcache_base_diff_head_sizes(self, mock_create_aligned):
        self.kvcache_settings.is_separated_pd = True
        self.kvcache_settings.k_head_size = 64
        self.kvcache_settings.v_head_size = 32
        self.kvcache_settings.kvcache_quant_layers = [True, False]
        self.kvcache_settings.num_layers = 2

        mock_key_int8 = MagicMock()
        mock_key_int8.data_ptr.return_value = 1024
        mock_key_float32 = MagicMock()
        mock_key_float32.data_ptr.return_value = 2048
        mock_value = MagicMock()
        mock_value.data_ptr.return_value = 3072
        mock_create_aligned.side_effect = [mock_key_int8, mock_value, mock_key_float32, mock_value]

        kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
        kvcache_pool._allocate_npu_kvcache_base()

        self.assertEqual(mock_create_aligned.call_count, 4)
        self.assertEqual(kvcache_pool.k_blocks_quant_addrs, [1024])
        self.assertEqual(kvcache_pool.k_blocks_addrs, [2048])
        self.assertEqual(kvcache_pool.v_blocks_addrs, [3072, 3072])
        self.assertEqual(len(kvcache_pool.npu_cache), 2)
        self.assertEqual(kvcache_pool.npu_cache[0], (mock_key_int8, mock_value))
        self.assertEqual(kvcache_pool.npu_cache[1], (mock_key_float32, mock_value))

    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.KVCachePool._create_aligned_tensor")
    def test_allocate_npu_kvcache_base_same_head_sizes(self, mock_create_aligned):
        self.kvcache_settings.is_separated_pd = True
        self.kvcache_settings.k_head_size = 64
        self.kvcache_settings.v_head_size = 64
        self.kvcache_settings.num_layers = 2

        mock_key = MagicMock()
        mock_key.data_ptr.return_value = 1024
        mock_value = MagicMock()
        mock_value.data_ptr.return_value = 2048
        mock_create_aligned.side_effect = [mock_key, mock_value, mock_key, mock_value]

        kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
        kvcache_pool._allocate_npu_kvcache_base()

        self.assertEqual(mock_create_aligned.call_count, 4)
        self.assertEqual(kvcache_pool.npu_blocks_addrs, [1024, 2048, 1024, 2048])
        self.assertEqual(len(kvcache_pool.npu_cache), 2)
        self.assertEqual(kvcache_pool.npu_cache[0], (mock_key, mock_value))
        self.assertEqual(kvcache_pool.npu_cache[1], (mock_key, mock_value))


class TestKVCachePoolSwap(unittest.TestCase):
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch_npu")
    def setUp(self, mock_torch_npu, mock_torch):
        self.kvcache_settings = DummyKVCacheSettings()
        self.device = "cpu"

        # Mock torch.empty to return a mock tensor with correct shape
        def mock_empty(size, dtype, device=None):
            mock_tensor = MagicMock()
            # Calculate the expected shape after view operation in _create_aligned_tensor
            if len(size) == 1:
                # This is the temporary tensor, the shape will be changed later
                mock_tensor.shape = size
            else:
                # This is the final tensor with the correct shape
                mock_tensor.shape = size
            mock_tensor.data_ptr.return_value = 1024
            # Create a mock for tensor elements
            mock_element = MagicMock()
            mock_element.cpu.return_value = mock_element
            mock_element.__eq__.return_value = True
            # Make __getitem__ return the mock element
            mock_tensor.__getitem__.return_value = mock_element
            mock_tensor.contiguous.return_value = mock_tensor
            mock_tensor.view.side_effect = lambda *args: mock_tensor
            mock_tensor.fill_.return_value = None
            return mock_tensor

        # Mock torch.allclose to always return True
        mock_torch.allclose.return_value = True
        mock_torch.empty.side_effect = mock_empty

        # Mock torch_npu.empty_with_format to return a mock tensor with correct shape
        def mock_empty_with_format(size, dtype, device, acl_format):
            mock_tensor = MagicMock()
            mock_tensor.shape = size
            mock_tensor.data_ptr.return_value = 1024
            return mock_tensor

        mock_torch_npu.empty_with_format.side_effect = mock_empty_with_format
        self.kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
        self.kvcache_pool.allocate_cpu_kvcache()
        self.kvcache_pool.allocate_npu_kvcache()

    @patch.object(torch, "allclose", return_value=True)
    def test_swap_in(self, mock_allclose):
        for layer in range(self.kvcache_settings.num_layers):
            cpu_key, cpu_value = self.kvcache_pool.cpu_cache[layer]
            cpu_key[1].fill_(123.0)
            cpu_value[1].fill_(456.0)
            npu_key, npu_value = self.kvcache_pool.npu_cache[layer]
            npu_key.fill_(0.0)
            npu_value.fill_(0.0)

        swap_decision = [[0, 1, 0]]
        self.kvcache_pool.swap_kvcache_method(swap_decision)

        for layer in range(self.kvcache_settings.num_layers):
            cpu_key, cpu_value = self.kvcache_pool.cpu_cache[layer]
            npu_key, npu_value = self.kvcache_pool.npu_cache[layer]
            self.assertTrue(torch.allclose(npu_key[0].cpu(), cpu_key[1]), "Swap in 操作失败：key 块数据不一致")
            self.assertTrue(torch.allclose(npu_value[0].cpu(), cpu_value[1]), "Swap in 操作失败：value 块数据不一致")

    @patch.object(torch, "allclose", return_value=True)
    def test_swap_out(self, mock_allclose):
        for layer in range(self.kvcache_settings.num_layers):
            npu_key, npu_value = self.kvcache_pool.npu_cache[layer]
            npu_key[0].fill_(789.0)
            npu_value[0].fill_(1011.0)
            cpu_key, cpu_value = self.kvcache_pool.cpu_cache[layer]
            cpu_key.fill_(0.0)
            cpu_value.fill_(0.0)

        swap_decision = [[1, 0, 1]]
        self.kvcache_pool.swap_kvcache_method(swap_decision)

        for layer in range(self.kvcache_settings.num_layers):
            npu_key, npu_value = self.kvcache_pool.npu_cache[layer]
            cpu_key, cpu_value = self.kvcache_pool.cpu_cache[layer]
            self.assertTrue(torch.allclose(cpu_key[1], npu_key[0].cpu()), "Swap out 操作失败：key 块数据不一致")
            self.assertTrue(torch.allclose(cpu_value[1], npu_value[0].cpu()), "Swap out 操作失败：value 块数据不一致")


class TestKVCachePoolAllocationWithMBSwapper(unittest.TestCase):
    def setUp(self):
        self.kvcache_settings = DummyKVCacheSettings()
        self.kvcache_settings.num_npu_blocks = 2
        self.kvcache_settings.num_cpu_blocks = 3
        self.device = "cpu"

    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.ENV")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch")
    def test_allocate_cpu_kvcache_mb_mode(self, mock_torch, mock_ENV):
        mock_ENV.use_mb_swapper = True
        # Mock torch.empty to return a mock tensor with pin_memory method
        mock_tensor = MagicMock()
        mock_tensor.pin_memory.return_value = mock_tensor
        mock_torch.empty.return_value = mock_tensor
        with patch(
            "mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.MBSWAPPER", create=True
        ) as mock_MBSWAPPER:
            mock_MBSWAPPER.return_value = MagicMock()
            kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
            kvcache_pool.allocate_cpu_kvcache()

            self.assertEqual(len(kvcache_pool.cpu_cache), self.kvcache_settings.num_layers)
            self.assertEqual(len(kvcache_pool.cpu_blocks_addrs), self.kvcache_settings.num_cpu_blocks)
            self.assertIsInstance(kvcache_pool.cpu_blocks_addrs[0], tuple)

    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.ENV")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch_npu")
    def test_allocate_npu_kvcache_mb_mode(self, mock_torch_npu, mock_ENV):
        mock_ENV.use_mb_swapper = True
        with patch(
            "mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.MBSWAPPER", create=True
        ) as mock_MBSWAPPER:
            mock_MBSWAPPER.return_value = MagicMock()
            mock_raw_blocks = MagicMock()
            mock_raw_blocks.data_ptr.return_value = 1024
            mock_torch_npu.empty_with_format.return_value = mock_raw_blocks

            kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
            kvcache_pool.allocate_npu_kvcache()

            self.assertEqual(len(kvcache_pool.npu_cache), self.kvcache_settings.num_layers)
            self.assertEqual(len(kvcache_pool.npu_blocks_addrs), self.kvcache_settings.num_npu_blocks)
            self.assertIsInstance(kvcache_pool.npu_blocks_addrs[0], tuple)
            mock_torch_npu.empty_with_format.assert_called_once()

    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.ENV")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.KVCachePool._create_aligned_tensor")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch_npu")
    def test_allocate_npu_kvcache_mb_separated_pd_true(self, mock_torch_npu, mock_create_aligned, mock_ENV):
        mock_ENV.use_mb_swapper = True
        with patch(
            "mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.MBSWAPPER", create=True
        ) as mock_MBSWAPPER:
            mock_MBSWAPPER.return_value = MagicMock()
            mock_raw_blocks = MagicMock()
            mock_raw_blocks.data_ptr.return_value = 1024
            mock_raw_blocks.shape = (
                self.kvcache_settings.num_layers,
                2,
                self.kvcache_settings.num_npu_blocks,
                *self.kvcache_settings.block_shape,
            )
            mock_create_aligned.return_value = mock_raw_blocks

            self.kvcache_settings.is_separated_pd = True
            kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
            kvcache_pool._allocate_npu_kvcache_mb()

            mock_create_aligned.assert_called_once_with(
                (
                    self.kvcache_settings.num_layers,
                    2,
                    self.kvcache_settings.num_npu_blocks,
                    *self.kvcache_settings.block_shape,
                ),
                self.kvcache_settings.dtype,
            )
            mock_torch_npu.empty_with_format.assert_not_called()
            expected_block_addrs = [
                (
                    1024 + j * self.kvcache_settings.mini_block_bytes,
                    1024 + (j + self.kvcache_settings.num_npu_blocks) * self.kvcache_settings.mini_block_bytes,
                )
                for j in range(self.kvcache_settings.num_npu_blocks)
            ]
            self.assertEqual(kvcache_pool.npu_blocks_addrs, expected_block_addrs)
            self.assertEqual(len(kvcache_pool.npu_cache), self.kvcache_settings.num_layers)

    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.ENV")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.KVCachePool._create_aligned_tensor")
    @patch("mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.torch_npu")
    def test_allocate_npu_kvcache_mb_separated_pd_false(self, mock_torch_npu, mock_create_aligned, mock_ENV):
        mock_ENV.use_mb_swapper = True
        with patch(
            "mindie_llm.text_generator.adapter.torch_utils.kvcache_pool.MBSWAPPER", create=True
        ) as mock_MBSWAPPER:
            mock_MBSWAPPER.return_value = MagicMock()
            mock_raw_blocks = MagicMock()
            mock_raw_blocks.data_ptr.return_value = 2048
            mock_raw_blocks.shape = (
                self.kvcache_settings.num_layers,
                2,
                self.kvcache_settings.num_npu_blocks,
                *self.kvcache_settings.block_shape,
            )
            mock_torch_npu.empty_with_format.return_value = mock_raw_blocks

            self.kvcache_settings.is_separated_pd = False
            kvcache_pool = KVCachePool(self.kvcache_settings, self.device)
            kvcache_pool._allocate_npu_kvcache_mb()

            mock_torch_npu.empty_with_format.assert_called_once_with(
                size=(
                    self.kvcache_settings.num_layers,
                    2,
                    self.kvcache_settings.num_npu_blocks,
                    *self.kvcache_settings.block_shape,
                ),
                dtype=self.kvcache_settings.dtype,
                device=self.device,
                acl_format=kvcache_pool.acl_format,
            )
            mock_create_aligned.assert_not_called()
            expected_block_addrs = [
                (
                    2048 + j * self.kvcache_settings.mini_block_bytes,
                    2048 + (j + self.kvcache_settings.num_npu_blocks) * self.kvcache_settings.mini_block_bytes,
                )
                for j in range(self.kvcache_settings.num_npu_blocks)
            ]
            self.assertEqual(kvcache_pool.npu_blocks_addrs, expected_block_addrs)
            self.assertEqual(len(kvcache_pool.npu_cache), self.kvcache_settings.num_layers)


if __name__ == "__main__":
    unittest.main()
