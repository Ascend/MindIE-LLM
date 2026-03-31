# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
import copy
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from ddt import ddt, data, unpack
import torch
import numpy as np

from mindie_llm.text_generator.adapter.generator_torch import GeneratorTorch, reorder_array, reorder_tensor
from mindie_llm.text_generator.adapter.generator_torch import check_model_config
from mindie_llm.text_generator.utils.model_input import ModelInput
from mindie_llm.text_generator.adapter.recovery_utils import (
    is_uce_error_addr_overlap_tensor_addr,
    get_tensor_address_range,
    check_and_recover_uce_in_cache,
)

MOCKED_INIT_METHOD = "mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch.__init__"
MOCKED_GET_MODEL_WRAPPER = "mindie_llm.text_generator.adapter.generator_backend.get_model_wrapper"
MOCKED_GET_OBF_FUNC = "mindie_llm.text_generator.adapter.generator_torch.GeneratorTorch._get_obfuscation_func"
TOKEN_SIZE_PER_DP_GROUP = "token_size_per_dp_group"
SHARD_EFFECTIVE_TOKEN_INDICES = "shard_effective_token_indices"
TOKEN_INDEX_WITH_PADDING = "token_index_with_padding"
SKIP_PADDING_TOKEN_INDICES = "skip_padding_token_indices"
MAX_POSITION_EMBEDDINGS = "max_position_embeddings"
MODEL_NAME = "model_name"
INFERENCE_MODEL = "inference_mode"

FAKE_INPUT_IDS = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
    4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
    9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
    15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
    2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
    17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
    18, 13, 22, 19, 17, 10532, 881, 254, 2616,
    39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
    12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
    7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
    473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
    25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
    5424, 30, 185, 32349, 25, 100000, 2640, 6, 82,
    4399, 4526, 30, 100000, 26503, 335, 279, 33396, 37576,
    1593, 429, 39166, 4403, 643, 895, 1377, 4712, 8254,
    285, 33396, 433, 9017, 418, 839, 3430, 276, 2622,
    742, 2616, 11010, 15821, 11, 5802, 1094, 12191, 536,
    441, 463, 276, 2622, 254, 11010, 3675, 9880, 4712,
    13, 685, 207, 17, 15, 15, 24, 11, 33396,
    37576, 6972, 363, 18, 13, 22, 19, 17, 10532,
    881, 254, 2616, 39696, 13, 79073, 280, 33396, 37576,
    2622, 881, 9798, 12178, 11, 285, 418, 4117, 19336,
    327, 9798, 12178, 7462, 2065, 20234, 13, 3159, 11,
    657, 418, 25541, 473, 254, 41198, 266, 12178, 47570,
    13, 185, 23853, 25, 317, 11010, 9880, 4712, 254,
    1246, 372, 3613, 5424, 30, 185, 32349, 25, 100000,
    2819, 418, 340, 30])
FAKE_POSITION_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 0, 1, 2, 3, 4])
FAKE_BLOCK_TABLES = np.array([[0, 1], [0, 0], [0, 1], [0, 0]])
FAKE_SLOTS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 0, 1, 2, 3, 4])
FAKE_CONTEXT_LENGTH = np.array([113, 7, 113, 5])


@dataclass
class MockConfig:
    max_position_embeddings: int = 0


class MockModelWrapper:
    def __init__(self):
        self.config = MockConfig()
        self.config_dict = {MAX_POSITION_EMBEDDINGS: 0}
        self.model_info = None
        self.max_position_embeddings = None
        self.tokenizer = None
        self.device = None
        self.rank = None
        self.adapter_manager = None
        self.mapping = MagicMock()
        self.model_runner = MagicMock()
        self.model_runner.llm_config = MagicMock()


@ddt
class TestGeneratorTorch(unittest.TestCase):
    def setUp(self):
        attn_dp = MagicMock()
        attn_dp.rank = 0
        attn_dp.group_size = 4

        attn_o_proj_tp = MagicMock()
        attn_o_proj_tp.rank = 0
        attn_o_proj_tp.group_size = 1

        self.mapping = MagicMock()
        self.mapping.attn_dp = attn_dp
        self.mapping.attn_o_proj_tp = attn_o_proj_tp
        self.mapping.has_dp = MagicMock(return_value=True)
        self.mapping.has_attn_inner_sp.return_value = False
        self.mapping.has_attn_cp.return_value = False

        self.cache_pool = MagicMock()
        self.cache_pool.num_npu_blocks = 10
        self.cache_pool.block_size = 128
        self.cache_pool.npu_cache = [(torch.Tensor([1, 2, 3]), torch.Tensor([1, 2, 3]))]
        self.cache_pool.kvcache_settings.num_layers = 1

        self.model_wrapper = MagicMock()
        self.model_wrapper.model_runner = MagicMock()
        self.model_wrapper.model_runner.num_speculative_tokens = 0

        self.default_model_config = {
            'backend_type': 'atb', 'npu_device_id': '3',
            'rank': '0', 'local_rank': '0', 'world_size': '1',
            'trust_remote_code': 'False',
            'num_speculative_tokens': '0'
        }

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_partition_data(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch.model_wrapper = self.model_wrapper

        dp_rank_ids = np.array([0, 1, 2, 3])
        dp_rank_ids_per_token = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        input_ids = FAKE_INPUT_IDS
        position_ids = FAKE_POSITION_IDS
        is_prefill = True
        block_tables = FAKE_BLOCK_TABLES
        slots = FAKE_SLOTS
        input_lengths = FAKE_CONTEXT_LENGTH
        lm_head_indices = np.cumsum(input_lengths) - 1
        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)
        input_token_num_per_batch = model_input.context_length
        output_token_num_per_batch = [1] * len(model_input.context_length)
        input_token_num_per_batch = np.array(input_token_num_per_batch)
        output_token_num_per_batch = np.array(output_token_num_per_batch)
        slot_dp_rank_id = np.repeat(dp_rank_ids, input_token_num_per_batch)
        generator_torch._partition_data(slot_dp_rank_id, dp_rank_ids_per_token, model_input,
                                        input_token_num_per_batch, output_token_num_per_batch)

        golden_input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
            4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
            9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
            15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
            2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
            17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
            18, 13, 22, 19, 17, 10532, 881, 254, 2616,
            39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
            12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
            7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
            473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
            25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
            5424, 30, 185, 32349, 25])
        self.assertTrue((model_input.input_ids == golden_input_ids).all())
        golden_position_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112])
        self.assertTrue((model_input.position_ids == golden_position_ids).all())
        self.assertTrue(model_input.is_prefill)
        self.assertTrue((model_input.block_tables == np.array([[0, 1]])).all())
        golden_slots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112])
        self.assertTrue((model_input.slots == golden_slots).all())
        self.assertTrue((model_input.context_length == np.array([113])).all())
        self.assertEqual(model_input.max_seq_len, 113)
        self.assertTrue((model_input.prefill_head_indices == np.array([112, 119, 232, 237])).all())

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_sp_partition_data(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.sp_size = 4
        generator_torch.sp_rank = 0
        generator_torch.cp_size = 1
        generator_torch.cp_rank = 0
        
        generator_torch.cache_pool = self.cache_pool

        input_ids = np.array([1, 2, 3, 4])
        position_ids = np.array([0])
        is_prefill = False
        block_tables = np.array([[0]])
        slots = np.array([0])
        input_lengths = np.array([4])
        sp_tokens = np.array([[4, 0, 0, 0]])
        model_input = ModelInput(input_ids, position_ids, block_tables, slots, input_lengths, 0,
                                 None, is_prefill, None, None, None, sp_tokens)
        kwargs = {'input_lengths_sp': None}
        kwargs['input_lengths_sp'] = generator_torch._sp_partition_data(model_input)
        self.assertEqual(kwargs['input_lengths_sp'], [4])

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_partition_data_dummy(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.mapping.attn_dp.rank = 3
        generator_torch.cache_pool = self.cache_pool

        dp_rank_ids = np.array([0])
        dp_rank_ids_per_token = np.array([0])
        input_ids = np.array([185])
        position_ids = np.array([7])
        is_prefill = False
        block_tables = np.array([[0]])
        slots = np.array([7])
        input_lengths = np.array([8])

        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, None,
            is_prefill, None, None, dp_rank_ids)
        input_token_num_per_batch = [1] * len(model_input.input_ids)
        batch_size = len(model_input.context_length)
        output_token_num_per_batch = [1] * batch_size
        input_token_num_per_batch = np.array(input_token_num_per_batch)
        output_token_num_per_batch = np.array(output_token_num_per_batch)
        slot_dp_rank_id = np.repeat(dp_rank_ids, input_token_num_per_batch)
        generator_torch._partition_data(slot_dp_rank_id, dp_rank_ids_per_token, model_input,
                                        input_token_num_per_batch, output_token_num_per_batch)

        self.assertTrue((model_input.input_ids == np.array([1])).all())
        self.assertTrue((model_input.position_ids == np.array([0], dtype=np.int32)).all())
        self.assertFalse(model_input.is_prefill)
        self.assertTrue((model_input.block_tables == np.array([[9]], dtype=np.int32)).all())
        self.assertTrue((model_input.slots == np.array([1152], dtype=np.int32)).all())
        self.assertTrue((model_input.context_length == np.array([1], dtype=np.int32)).all())
        self.assertEqual(model_input.max_seq_len, 1)
        self.assertTrue((model_input.prefill_head_indices == np.array([0])).all())

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_gather_dp_data(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping

        dp_rank_ids_per_token = np.array([0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3])
        res = generator_torch._gather_dp_data(dp_rank_ids_per_token)
        golden = {
            TOKEN_SIZE_PER_DP_GROUP: np.array([4, 1, 2, 5]),
            SHARD_EFFECTIVE_TOKEN_INDICES: np.array([0, 1, 2, 3]),
            TOKEN_INDEX_WITH_PADDING: np.array([0, 1, 2, 3, 0]),
            SKIP_PADDING_TOKEN_INDICES: np.array([0, 1, 2, 3, 5, 10, 11, 15, 16, 17, 18, 19])
        }
        self.assertTrue((res.get(TOKEN_SIZE_PER_DP_GROUP) == golden.get(TOKEN_SIZE_PER_DP_GROUP)).all())
        self.assertTrue(
            (res.get(SHARD_EFFECTIVE_TOKEN_INDICES) == golden.get(SHARD_EFFECTIVE_TOKEN_INDICES)).all())
        self.assertTrue((res.get(TOKEN_INDEX_WITH_PADDING) == golden.get(TOKEN_INDEX_WITH_PADDING)).all())
        self.assertTrue((res.get(SKIP_PADDING_TOKEN_INDICES) == golden.get(SKIP_PADDING_TOKEN_INDICES)).all())

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_gather_dp_data_dummy(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping

        dp_rank_ids_per_token = np.array([0, 0, 0, 0, 2, 2])
        res = generator_torch._gather_dp_data(dp_rank_ids_per_token)

        golden = {
            TOKEN_SIZE_PER_DP_GROUP: np.array([4, 1, 2, 1]),
            SHARD_EFFECTIVE_TOKEN_INDICES: np.array([0, 1, 2, 3]),
            TOKEN_INDEX_WITH_PADDING: np.array([0, 1, 2, 3]),
            SKIP_PADDING_TOKEN_INDICES: np.array([0, 1, 2, 3, 4, 8, 9, 12])
        }
        self.assertTrue((res.get(TOKEN_SIZE_PER_DP_GROUP) == golden.get(TOKEN_SIZE_PER_DP_GROUP)).all())
        self.assertTrue(
            (res.get(SHARD_EFFECTIVE_TOKEN_INDICES) == golden.get(SHARD_EFFECTIVE_TOKEN_INDICES)).all())
        self.assertTrue((res.get(TOKEN_INDEX_WITH_PADDING) == golden.get(TOKEN_INDEX_WITH_PADDING)).all())
        self.assertTrue((res.get(SKIP_PADDING_TOKEN_INDICES) == golden.get(SKIP_PADDING_TOKEN_INDICES)).all())

    @data((None, [0, 1, 2], None, np.array([])),
          (np.array([1, 2, 3, 4, 5]), [0, 1, 2, 3, 4], None, np.array([1, 2, 3, 4, 5])),
          (np.array([1, 2, 4, 5, 3]), [0, 1, 4, 2, 3], None, np.array([1, 2, 3, 4, 5])),
          (np.array([1, 2, 3, 4, 5]), [0, 1], [2, 5], np.array([1, 2, 3, 4, 5])),
          (np.array([1, 2, 3, 4, 5]), [1, 0], [2, 5], np.array([3, 4, 5, 1, 2])))
    @unpack
    def test_reorder_array(self, array, order, position, golden):
        res = reorder_array(array, order, position=position)
        self.assertTrue((res == golden).all())

    @data((torch.tensor([1, 2, 3, 4, 5]), [0, 1, 2, 3, 4], 0, None, torch.tensor([1, 2, 3, 4, 5])),
          (torch.tensor([1, 2, 4, 5, 3]), [0, 1, 4, 2, 3], 0, None, torch.tensor([1, 2, 3, 4, 5])),
          (torch.tensor([1, 2, 3, 4, 5]), [0, 1], 0, [2, 3], torch.tensor([1, 2, 3, 4, 5])),
          (torch.tensor([1, 2, 3, 4, 5]), [1, 0], 0, [2, 3], torch.tensor([3, 4, 5, 1, 2])),
          (torch.tensor([[0, 1], [2, 3], [4, 5]]), [1, 0], 1, [2, 1], torch.tensor([[4, 5], [0, 1], [2, 3]])))
    @unpack
    def test_reorder_tensor(self, tensor, order, dim, position, golden):
        res = reorder_tensor(tensor, order, dim, position)
        self.assertTrue(torch.equal(res, golden))

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_forward_tensor_raise(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool

        input_ids = torch.tensor([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166])
        position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 60, 0, 1])
        is_prefill = True
        kv_cache = [(torch.zeros([10, 128, 1, 128]),), (torch.ones([10, 128, 1, 128]),)]
        block_tables = torch.tensor([[0, 1], [0, 0]])
        slots = torch.tensor([0, 1, 2, 3, 4, 5, 60, 0, 1])
        input_lengths = torch.tensor([6, 2])
        adapter_ids = [None, None, None]
        with self.assertRaises(ValueError) as context:
            generator_torch.forward_tensor(
                input_ids, position_ids, is_prefill, kv_cache, block_tables, slots, input_lengths, 20,
                adapter_ids=adapter_ids)
        self.assertIn("The length of `adapter_ids` should not be larger than batch size.",
                      str(context.exception))

    @patch("mindie_llm.text_generator.adapter.generator_torch.reorder_tensor", return_value=torch.tensor([0, 1]))
    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_forward(self, _, mock_reorder_tensor):
        placeholder = np.array([1])
        generator_torch = GeneratorTorch({})
        generator_torch.device = 'cpu'
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch._sort_model_inputs_by_adapter_ids = MagicMock(return_value=(True, [1, 0]))
        generator_torch._partition_data = MagicMock()
        dp_additional_args = {
            "token_size_per_dp_group": placeholder,
            "shard_effective_token_indices": placeholder,
            "token_index_with_padding": placeholder,
            "skip_padding_token_indices": placeholder,
        }
        generator_torch._gather_dp_data = MagicMock(return_value=dp_additional_args)
        model_wrapper = MagicMock()
        model_wrapper.model_runner = MagicMock()
        model_wrapper.model_runner.llm_config = MagicMock()
        model_wrapper.model_runner.num_speculative_tokens = 0
        generator_torch.llm_config = MagicMock()
        generator_torch.llm_config.llm.stream_options.micro_batch = False
        generator_torch.enable_dap = False
        generator_torch.model_wrapper = model_wrapper
        generator_torch.adapter_manager = MagicMock()
        generator_torch.model_wrapper.forward = MagicMock(return_value=torch.tensor([1, 0]))

        dp_rank_ids = np.array([0, 2])
        input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166])
        position_ids = np.array([0, 1, 2, 3, 4, 5, 60, 0, 1])
        is_prefill = True
        block_tables = np.array([[0, 1], [0, 0]])
        slots = np.array([0, 1, 2, 3, 4, 5, 60, 0, 1])
        input_lengths = np.array([6, 2])
        adapter_ids = ["base", "adapter1"]
        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, None,
            is_prefill, None, adapter_ids, dp_rank_ids)

        res = generator_torch.forward(model_input)
        generator_torch._sort_model_inputs_by_adapter_ids.assert_called_once()
        golden_dp_rank_ids_per_token = np.array([0, 0, 0, 0, 0, 0, 2, 2])
        generator_torch._partition_data.assert_called_once()
        args, _ = generator_torch._partition_data.call_args
        self.assertTrue((args[1] == golden_dp_rank_ids_per_token).all())
        self.assertEqual(args[2], model_input)
        generator_torch._gather_dp_data.assert_called_once()
        args, _ = generator_torch._gather_dp_data.call_args
        self.assertTrue((args[0] == golden_dp_rank_ids_per_token).all())
        generator_torch.model_wrapper.forward.assert_called_once()
        args, kwargs = generator_torch.model_wrapper.forward.call_args
        self.assertEqual(args[0], model_input)
        self.assertTrue(kwargs, dp_additional_args)
        mock_reorder_tensor.assert_called_once()
        args, _ = mock_reorder_tensor.call_args
        self.assertTrue(torch.equal(args[0], torch.tensor([1, 0])))
        self.assertListEqual(args[1], [1, 0])
        self.assertTrue(torch.equal(res, torch.tensor([0, 1])))
    
    @patch(MOCKED_INIT_METHOD, return_value=None)
    @data((np.array([0, 0, 1, 1]),
           [torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]])],
           torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])),
          (np.array([1, 0, 1, 0]),
           [torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]])],
           torch.tensor([[9, 10, 11, 12], [1, 2, 3, 4], [13, 14, 15, 16], [5, 6, 7, 8]])))
    @unpack
    def test_dap_reorder_tensor(self, dap_stream_id_mask, dap_logits, golden_reordered_logits, _):
        generator_torch = GeneratorTorch({})

        reordered_logits = generator_torch._dap_reorder_tensor(dap_stream_id_mask, dap_logits)

        self.assertTrue(torch.equal(reordered_logits, golden_reordered_logits))
    
    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_dap_partition_data(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = MagicMock()
        generator_torch.mapping.has_dp = MagicMock()
        generator_torch.mapping.has_dp.return_value = True
        generator_torch.llm_config = MagicMock()
        generator_torch.llm_config.llm.stream_options.micro_batch = False
        generator_torch.enable_dap = False
        partition_id = 1
        partition_mask = np.array([1, 0, 0, 1])
        input_ids = FAKE_INPUT_IDS
        position_ids = FAKE_POSITION_IDS
        block_tables = FAKE_BLOCK_TABLES
        slots = FAKE_SLOTS
        context_length = FAKE_CONTEXT_LENGTH
        max_seq_len = 0
        prefill_head_indices = None
        is_prefill = True
        query_length = None
        adapter_ids = None
        dp_rank_ids = np.array([0, 0, 1, 1])
        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, context_length, max_seq_len,
            prefill_head_indices, is_prefill, query_length, adapter_ids, dp_rank_ids)
        dap_model_input = [model_input, copy.deepcopy(model_input)]
        
        q_lens = [113, 7, 113, 5]
        fake_dap_kwargs = [{"q_lens": q_lens}, {"q_lens": copy.deepcopy(q_lens)}]
        generator_torch._dap_partition_data(partition_id, partition_mask, dap_model_input, fake_dap_kwargs)

        golden_input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
            4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
            9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
            15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
            2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
            17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
            18, 13, 22, 19, 17, 10532, 881, 254, 2616,
            39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
            12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
            7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
            473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
            25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
            5424, 30, 185, 32349, 25, 100000, 2819, 418, 340, 30])
        golden_position_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 0, 1, 2, 3, 4])
        golden_block_tables = np.array([[0, 1], [0, 0]])
        golden_slots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 0, 1, 2, 3, 4])
        golden_context_length = np.array([113, 5])
        golden_max_seq_len = 113
        golden_prefill_head_indices = np.array([112, 117])
        golden_dp_rank_ids = np.array([0, 1])
        self.assertTrue((dap_model_input[1].input_ids == golden_input_ids).all())
        self.assertTrue((dap_model_input[1].position_ids == golden_position_ids).all())
        self.assertTrue((dap_model_input[1].block_tables == golden_block_tables).all())
        self.assertTrue((dap_model_input[1].slots == golden_slots).all())
        self.assertTrue((dap_model_input[1].context_length == golden_context_length).all())
        self.assertEqual(dap_model_input[1].max_seq_len, golden_max_seq_len)
        self.assertTrue((dap_model_input[1].prefill_head_indices == golden_prefill_head_indices).all())
        self.assertTrue((dap_model_input[1].dp_rank_ids == golden_dp_rank_ids).all())
    
    @patch('mindie_llm.text_generator.adapter.generator_torch.np.sum', return_value=4097)
    @patch(MOCKED_INIT_METHOD, return_value=None)
    @data((False,
           False,
           None,
           False,
           np.array([])),
          (True,
           False,
           None,
           True,
           np.array([0, 0, 1, 1])),
          (True,
           True,
           np.array([1, 0, 0, 0]),
           True,
           np.array([0, 1, 0, 1])),
          (True,
           True,
           np.array([1, 1, 0, 0]),
           True,
           np.array([0, 1, 0, 1])))
    @unpack
    def test_partition_dap_stream_by_rank(self, input_is_prefill, has_dp, input_dp_rank_ids,
                                          golden_is_dap, golden_dap_stream_id_mask,
                                          mock_init_method, mock_np_sum):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = MagicMock()
        generator_torch.mapping.has_dp = MagicMock()
        generator_torch.mapping.has_dp.return_value = has_dp
        generator_torch.mapping.attn_dp = MagicMock()
        generator_torch.mapping.attn_dp.group_size = 2
        generator_torch.llm_config = MagicMock()
        generator_torch.llm_config.llm.stream_options.micro_batch = True
        generator_torch.enable_dap = True
        input_ids = FAKE_INPUT_IDS
        position_ids = FAKE_POSITION_IDS
        block_tables = FAKE_BLOCK_TABLES
        slots = FAKE_SLOTS
        context_length = FAKE_CONTEXT_LENGTH
        max_seq_len = 0
        prefill_head_indices = None
        is_prefill = input_is_prefill
        query_length = None
        adapter_ids = None
        dp_rank_ids = input_dp_rank_ids
        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, context_length, max_seq_len,
            prefill_head_indices, is_prefill, query_length, adapter_ids, dp_rank_ids)
        
        is_dap, dap_stream_id_mask = generator_torch._partition_dap_stream_by_rank(model_input)

        self.assertEqual(is_dap, golden_is_dap)
        self.assertTrue((dap_stream_id_mask == golden_dap_stream_id_mask).all())
    
    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_dap_forward(self, _):
        def side_effect(**kwargs):
            self.assertEqual(len(kwargs.get('input_ids')), 2)
            self.assertEqual(len(kwargs.get('position_ids')), 2)
            self.assertEqual(len(kwargs.get('is_prefill')), 2)
            self.assertEqual(len(kwargs.get('block_tables')), 2)
            self.assertEqual(len(kwargs.get('slots')), 2)
            self.assertEqual(len(kwargs.get('input_lengths')), 2)
            self.assertEqual(len(kwargs.get('max_seq_len')), 2)
            self.assertEqual(len(kwargs.get('lm_head_indices')), 2)
            self.assertEqual(len(kwargs.get('dap_kwargs')), 2)
        generator_torch = GeneratorTorch({})
        generator_torch.model_wrapper = MagicMock()
        generator_torch.model_wrapper.model_runner = MagicMock()
        generator_torch.cache_pool = MagicMock()
        generator_torch.cache_pool.npu_cache = MagicMock()
        generator_torch.model_wrapper.model_runner.num_speculative_tokens = 0
        generator_torch.model_wrapper.model_runner.llm_config = MagicMock()
        generator_torch.llm_config = MagicMock()
        generator_torch.llm_config.llm.stream_options.micro_batch = True
        generator_torch.enable_dap = True
        generator_torch.mapping = MagicMock()
        generator_torch.mapping.has_dp = MagicMock()
        generator_torch.mapping.has_dp.return_value = False
        generator_torch._dap_reorder_tensor = MagicMock()
        dap_stream_id_mask = np.array([0, 1, 0, 1])
        kwargs = {}
        input_ids = FAKE_INPUT_IDS
        position_ids = FAKE_POSITION_IDS
        block_tables = FAKE_BLOCK_TABLES
        slots = FAKE_SLOTS
        context_length = FAKE_CONTEXT_LENGTH
        max_seq_len = 0
        prefill_head_indices = None
        is_prefill = True
        query_length = None
        adapter_ids = None
        dp_rank_ids = np.array([1, 1, 0, 0])
        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, context_length, max_seq_len,
            prefill_head_indices, is_prefill, query_length, adapter_ids, dp_rank_ids)
        generator_torch.model_wrapper.model_runner.dap_forward = MagicMock()
        generator_torch.model_wrapper.model_runner.dap_forward.side_effect = side_effect
        
        _ = generator_torch._dap_forward(dap_stream_id_mask, model_input, **kwargs)

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_get_dp_ep_padding_inputs_dp(self, _):
        token_size_per_dp_group = np.array([2, 3, 4, 5])
        kwargs = {"token_size_per_dp_group": token_size_per_dp_group}

        generator_torch = GeneratorTorch({})
        self.mapping.attn_tp.group_size = 2
        self.mapping.attn_tp.rank = 0
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        model_wrapper = MagicMock()
        model_wrapper.model_runner.num_speculative_tokens = 0
        generator_torch.model_wrapper = model_wrapper

        generator_torch._get_dp_ep_padding_inputs(True, kwargs)
        golden_dep_inputs = [
            np.array([0, 1, 0, 0, 0, 0], dtype=np.int32),
            np.array([0, 1, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20, 21, 22], dtype=np.int32),
            np.array([0, 1, 0, 0, 0, 0, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 13, 0], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 6, 7, 8, 12, 13, 14, 15, 18, 19, 20, 21, 22], dtype=np.int32),
            np.array([0, 1, 0], dtype=np.int32), np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32)
        ]
        dep_inputs_none = [np.array([0], dtype=np.int32) for _ in range(len(golden_dep_inputs))]
        generate_dep_inputs = kwargs.get('dep_inputs', dep_inputs_none)
        generate_max_dp_batch_size = kwargs.get('max_dp_batch_size', -1)
        self.assertTrue(np.allclose(generate_dep_inputs[0], golden_dep_inputs[0]))
        self.assertEqual(generate_max_dp_batch_size, 5)

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_get_all2all_buffer_factor(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        model_wrapper = MagicMock()
        delattr(model_wrapper.config, 'length_thresholds')
        generator_torch.model_wrapper = model_wrapper
        length = 10000
        factor = generator_torch._get_all2all_buffer_factor(length)
        self.assertAlmostEqual(factor, 1.0)

    @data((4, np.array([0, 1], dtype=np.int32),
           [np.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=np.int32), np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), np.array([0, 1, 2, 3], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32), 
            np.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=np.int32),
            np.array([1], dtype=np.int32), np.array([1], dtype=np.int32), np.array([0, 1], dtype=np.int32)]
          ),
          (5, np.array([0, 1], dtype=np.int32),
           [np.array([0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32), np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), np.array([0, 1, 2, 3, 4], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32),
            np.array([1], dtype=np.int32), np.array([1], dtype=np.int32), np.array([0, 1], dtype=np.int32)]
          ),
          (8, np.array([0, 1], dtype=np.int32),
           [np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32), 
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            np.array([1], dtype=np.int32), np.array([1], dtype=np.int32), np.array([0, 1], dtype=np.int32)]
          ))
    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_get_dp_ep_inputs_lmheadtp(self, test_cacse, _):
        batch_size, input_ids, golden_dep_inputs = test_cacse
        generator_torch = GeneratorTorch({})
        attn_tp = MagicMock()
        attn_tp.rank = 0
        attn_tp.group_size = 1

        lm_head_tp = MagicMock()
        lm_head_tp.rank = 0
        lm_head_tp.group_size = 2

        self.mapping = MagicMock()
        self.mapping.attn_tp = attn_tp
        self.mapping.lm_head_tp = lm_head_tp
        self.mapping.enable_lm_head_local_tp = True
        self.mapping.enable_o_proj_local_tp = False
        model_wrapper = MagicMock()
        config = MagicMock()
        config.ep_level = 2          
        model_wrapper.config = config
        generator_torch.model_wrapper = model_wrapper
        generator_torch.mapping = self.mapping
        generator_torch.max_batch_size = 8
        generator_torch.num_speculative_tokens = 0
        generate_dep_inputs = generator_torch._get_dp_ep_inputs(batch_size, input_ids)
        for array, golden_array in zip(generate_dep_inputs, golden_dep_inputs):
            self.assertTrue(np.allclose(array, golden_array))
    
    @data((4,np.array([0, 1], dtype=np.int32),
        [
            np.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=np.int32),np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),np.array([0, 1, 2, 3], dtype=np.int32),
            np.arange(16, dtype=np.int32),np.array([0, 1, 2, 3, 0, 0, 0, 0], dtype=np.int32),
            np.array([1], dtype=np.int32),np.array([1], dtype=np.int32),np.array([0, 1], dtype=np.int32),
        ]),
        (5,np.array([0, 1], dtype=np.int32),
        [
            np.array([0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32),np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
            np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),np.array([0, 1, 2, 3, 4], dtype=np.int32),
            np.arange(16, dtype=np.int32),np.array([0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32),
            np.array([1], dtype=np.int32),np.array([1], dtype=np.int32),np.array([0, 1], dtype=np.int32),
        ]),
        (8,np.array([0, 1], dtype=np.int32),
        [
            np.arange(8, dtype=np.int32),np.arange(8, dtype=np.int32),
            np.arange(8, dtype=np.int32),np.arange(8, dtype=np.int32),
            np.arange(16, dtype=np.int32),np.arange(8, dtype=np.int32),
            np.array([1], dtype=np.int32),np.array([1], dtype=np.int32),np.array([0, 1], dtype=np.int32),
        ]),
    )
    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_get_dp_ep_inputs_static_ep(self, test_case, _):
        batch_size, input_ids, golden_dep_inputs = test_case
        generator_torch = GeneratorTorch({})
        attn_tp = MagicMock()
        attn_tp.rank = 0
        attn_tp.group_size = 1

        lm_head_tp = MagicMock()
        lm_head_tp.rank = 0
        lm_head_tp.group_size = 2
        
        attn_dp = MagicMock()
        attn_dp.group_size = 1  

        self.mapping = MagicMock()
        self.mapping.attn_tp = attn_tp
        self.mapping.attn_dp = attn_dp
        self.mapping.lm_head_tp = lm_head_tp
        self.mapping.enable_lm_head_local_tp = True
        self.mapping.enable_o_proj_local_tp = False
        self.mapping.has_moe_ep.return_value = True   # <<< 触发读取 ep_level
        generator_torch.mapping = self.mapping

        model_wrapper = MagicMock()
        config = MagicMock()
        config.ep_level = 1                      # <<< expert_parallel_degree = 1 → 走 else 分支
        model_wrapper.config = config
        generator_torch.model_wrapper = model_wrapper

        generator_torch.max_batch_size = 8
        generator_torch.num_speculative_tokens = 0
        

        out = generator_torch._get_dp_ep_inputs(batch_size, input_ids)
        self.assertEqual(len(out), len(golden_dep_inputs))
        for i, (a, b) in enumerate(zip(out, golden_dep_inputs)):
            self.assertTrue(np.array_equal(a, b), msg=f"Mismatch at idx {i}: {a} vs {b}")
            self.assertEqual(a.dtype, np.int32)
    
    def test_check_model_config_model_name_error(self):
        model_config = {
            'model_name': '',
            'max_position_embeddings': 0,
            'num_lccl_comm_shards': 67000,
            'lccl_comm_shard_id': 67001
        }
        expect_msg = ("The length of `model_name` should be in range of [1, 256]. If you are using MindIE as a"
                      " service framework, `model_name` is loaded from "
                      "$BackendConfig.ModelDeployConfig.ModelConfig.modelName "
                      "in ${MINDIE_LLM_HOME_PATH}/conf/config.json.")
        with self.assertRaises(ValueError) as cm:
            check_model_config(model_config)
        self.assertIn(expect_msg, str(cm.exception))
    
    def test_check_model_config_embeddings_error(self):
        model_config = {
            'model_name': 'qwen',
            'max_position_embeddings': 0,
            'num_lccl_comm_shards': 67000,
            'lccl_comm_shard_id': 67001
        }
        expect_msg = ("`max_position_embeddings` must be greater than 0. If you are using MindIE as a service framework,"
                      " `max_position_embeddings` is derived from "
                      "$BackendConfig.ModelDeployConfig.ModelConfig.max_position_embeddings in "
                      "${MINDIE_LLM_HOME_PATH}/conf/config.json.")
        with self.assertRaises(ValueError) as cm:
            check_model_config(model_config)
        self.assertIn(expect_msg, str(cm.exception))

    def test_check_model_config_num_lccl_comm_shards_error(self):
        model_config = {
            'model_name': 'qwen',
            'max_position_embeddings': 2,
            'num_lccl_comm_shards': 67000,
            'lccl_comm_shard_id': 67001
        }
        expect_msg = ("`num_lccl_comm_shards` must be in the range of [0, 65536].")
        with self.assertRaises(ValueError) as cm:
            check_model_config(model_config)
        self.assertIn(expect_msg, str(cm.exception))

    def test_check_model_config_lccl_comm_shard_id_error(self):
        model_config = {
            'model_name': 'qwen',
            'max_position_embeddings': 2,
            'num_lccl_comm_shards': 2,
            'lccl_comm_shard_id': 67001
        }
        expect_msg = ("`lccl_comm_shard_id` must be in the range of [0, `num_lccl_comm_shards`).")
        with self.assertRaises(ValueError) as cm:
            check_model_config(model_config)
        self.assertIn(expect_msg, str(cm.exception))

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_dp_partition_data_with_qlen_dummy(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        model_wrapper = MagicMock()
        generator_torch.model_wrapper = model_wrapper
        generator_torch.device = 'cpu'
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch.model_wrapper = self.model_wrapper
        
        dp_rank_ids = np.array([1, 1, 2, 3])
        input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429], dtype=np.int64)
        position_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        is_prefill = False
        block_tables = FAKE_BLOCK_TABLES
        slots = np.array([12, 13, 12, 13, 12, 13, 12, 13])
        input_lengths = np.array([13, 13, 13, 13])
        lm_head_indices = np.array([1, 2, 5, 7])
        q_lens = [2, 2, 2, 2]
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 13, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)
        spec_mask = torch.tensor(np.zeros((8, 13), dtype=np.int32))
        hidden_states = torch.tensor(np.zeros((8, 13), dtype=np.int32))
        kwargs = {
            'q_lens': q_lens,
            'spec_mask': spec_mask,
            'hidden_states': hidden_states,
            'sub_model': False
        }
        generator_torch._dp_partition_data(model_inputs, kwargs)

        expected_qlen_new = [1]
        q_lens_new = kwargs.get("q_lens")
        spec_mask_new = kwargs.get("spec_mask")
        hidden_states_new = kwargs.get("hidden_states") 
        self.assertTrue(np.array_equal(q_lens_new, expected_qlen_new),
                        f"Expected qlen {expected_qlen_new}, but got {q_lens_new}")
        self.assertEqual(spec_mask_new.shape, (13,))
        self.assertEqual(hidden_states_new.shape, (1, 13))

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_dp_partition_data_with_qlen(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        model_wrapper = MagicMock()
        generator_torch.model_wrapper = model_wrapper
        generator_torch.device = 'cpu'
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch.model_wrapper = self.model_wrapper

        dp_rank_ids = np.array([0, 1, 2, 3])
        input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429], dtype=np.int64)
        position_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        is_prefill = False
        block_tables = FAKE_BLOCK_TABLES
        slots = np.array([12, 13, 12, 13, 12, 13, 12, 13])
        input_lengths = np.array([13, 13, 13, 13])
        lm_head_indices = np.array([1, 2, 5, 7])
        q_lens = [2, 2, 2, 2]
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 13, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)
        spec_mask = torch.tensor(np.zeros((8, 13), dtype=np.int32))
        hidden_states = torch.tensor(np.zeros((8, 13), dtype=np.int32))
        kwargs = {
            'q_lens': q_lens,
            'spec_mask': spec_mask,
            'hidden_states': hidden_states,
            'sub_model': False
        }
        generator_torch._dp_partition_data(model_inputs, kwargs)

        expected_qlen_new = [2]
        q_lens_new = kwargs.get("q_lens")
        spec_mask_new = kwargs.get("spec_mask")
        hidden_states_new = kwargs.get("hidden_states") 
        
        self.assertTrue(np.array_equal(q_lens_new, expected_qlen_new),
                        f"Expected qlen {expected_qlen_new}, but got {q_lens_new}")
        self.assertEqual(spec_mask_new.shape, (2, 13))
        self.assertEqual(hidden_states_new.shape, (2, 13))

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_prepare_model_inputs_distribute_prefill(self, _): 
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = True
        generator_torch.model_wrapper = self.model_wrapper
        generator_torch.mapping.has_dp = MagicMock(return_value=False)
        generator_torch.mapping.attn_tp.group_size = 4

        dp_rank_ids = np.array([0])
        input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
            4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
            9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
            15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
            2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
            17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
            18, 13, 22, 19, 17, 10532, 881, 254, 2616,
            39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
            12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
            7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
            473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
            25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
            5424, 30, 185, 32349, 25])
        position_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112])
        is_prefill = True
        block_tables = np.array([[0, 1]])
        slots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112])
        input_lengths = np.array([113])
        lm_head_indices = np.cumsum(input_lengths) - 1
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)
        kwargs = {}
        do_reorder_requests, revert_adapter_idx = generator_torch._prepare_model_inputs(model_inputs, kwargs)
        
        token_size_per_dp_group_out = kwargs.get('token_size_per_dp_group')
        lm_head_indices_dp_rank_ids_out = kwargs.get('lm_head_indices_dp_rank_ids')
        shard_effective_token_indices_out = kwargs.get('shard_effective_token_indices')
        dep_inputs_out = kwargs.get('dep_inputs')
        expect_token_size_per_dp_group_out = np.array([113,113,113,113])
        self.assertTrue(np.array_equal(token_size_per_dp_group_out, expect_token_size_per_dp_group_out))
        self.assertTrue(np.array_equal(lm_head_indices_dp_rank_ids_out, np.array([0])))
        self.assertEqual(shard_effective_token_indices_out.shape, (113,))
        self.assertEqual(len(dep_inputs_out), 9)

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_prepare_model_inputs_distribute_decode(self, _): 
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = True
        generator_torch.model_wrapper = self.model_wrapper
        generator_torch.mapping.has_dp = MagicMock(return_value=False)
        generator_torch.mapping.attn_tp.group_size = 4
        generator_torch.num_speculative_tokens = 1
        generator_torch.max_batch_size = 4

        dp_rank_ids = np.array([0])
        input_ids = np.array([100000, 26503], dtype=np.int64)
        position_ids = np.array([0, 1])
        is_prefill = False
        block_tables = FAKE_BLOCK_TABLES
        slots = np.array([12, 13])
        input_lengths = np.array([13])
        lm_head_indices = np.array([1])
        q_lens = [2]
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 13, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)

        hidden_states = torch.tensor(np.zeros((2, 13), dtype=np.int32))
        kwargs = {
            'q_lens': q_lens,
            'hidden_states': hidden_states
        }
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)
        kwargs = {}
        do_reorder_requests, revert_adapter_idx = generator_torch._prepare_model_inputs(model_inputs, kwargs)
        dep_inputs_out = kwargs.get('dep_inputs')
        self.assertEqual(len(dep_inputs_out), 9)

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_prepare_model_inputs_cp_prefill(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.cp_size = 2
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch.model_wrapper = self.model_wrapper
        generator_torch.mapping.has_dp = MagicMock(return_value=False)
        generator_torch.mapping.has_attn_cp = MagicMock(return_value=True)
        generator_torch.mapping.attn_tp.group_size = 4
        generator_torch.mapping.attn_cp.group_size = 2
        generator_torch.mapping.attn_cp.rank = 0

        dp_rank_ids = np.array([0])
        input_ids = np.array([100000, 26503, 335, 279, 33396, 37576, 1593, 429, 39166,
            4403, 643, 895, 1377, 4712, 8254, 285, 33396, 433,
            9017, 418, 839, 3430, 276, 2622, 742, 2616, 11010,
            15821, 11, 5802, 1094, 12191, 536, 441, 463, 276,
            2622, 254, 11010, 3675, 9880, 4712, 13, 685, 207,
            17, 15, 15, 24, 11, 33396, 37576, 6972, 363,
            18, 13, 22, 19, 17, 10532, 881, 254, 2616,
            39696, 13, 79073, 280, 33396, 37576, 2622, 881, 9798,
            12178, 11, 285, 418, 4117, 19336, 327, 9798, 12178,
            7462, 2065, 20234, 13, 3159, 11, 657, 418, 25541,
            473, 254, 41198, 266, 12178, 47570, 13, 185, 23853,
            25, 317, 11010, 9880, 4712, 254, 1246, 372, 3613,
            5424, 30, 185, 32349, 25, 0, 0, 0])
        position_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115])
        is_prefill = True
        block_tables = np.array([[0, 1]])
        slots = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, -1, -1, -1])
        input_lengths = np.array([116])
        lm_head_indices = np.cumsum(input_lengths) - 1
        cp_tokens = np.array([[58, 58]], dtype=np.int32)
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, lm_head_indices,
            is_prefill, None, None, dp_rank_ids, cp_tokens=cp_tokens, pad_token_count=np.array([3]))
        kwargs = {}        
        do_reorder_requests, revert_adapter_idx = generator_torch._prepare_model_inputs(model_inputs, kwargs)
        self.assertTrue(np.array_equal(model_inputs.context_length, np.array([58])))
        self.assertTrue(np.array_equal(model_inputs.prefill_head_indices, np.array([54])))
        self.assertEqual(len(model_inputs.input_ids), 58)
        self.assertEqual(len(model_inputs.position_ids), 58)

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_prepare_model_inputs_cp_decode(self, _): 
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch.model_wrapper = self.model_wrapper
        generator_torch.mapping.has_dp = MagicMock(return_value=False)
        generator_torch.mapping.has_attn_cp = MagicMock(return_value=True)
        generator_torch.mapping.attn_tp.group_size = 4
        generator_torch.mapping.attn_cp.group_size = 2
        generator_torch.mapping.attn_cp.rank = 0
        generator_torch.num_speculative_tokens = 1
        generator_torch.max_batch_size = 4
        generator_torch.cp_rank = generator_torch.mapping.attn_cp.rank
        generator_torch.cp_size = generator_torch.mapping.attn_cp.group_size
        generator_torch.sp_rank = 0
        generator_torch.sp_size = 1

        dp_rank_ids = np.array([0])
        input_ids = np.array([100000, 26503], dtype=np.int64)
        position_ids = np.array([0, 1])
        is_prefill = False
        block_tables = FAKE_BLOCK_TABLES
        slots = np.array([12, 13])
        input_lengths = np.array([13])
        lm_head_indices = np.array([1])
        q_lens = [2]
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 13, lm_head_indices,
            is_prefill, None, None, dp_rank_ids)

        hidden_states = torch.tensor(np.zeros((2, 13), dtype=np.int32))
        kwargs = {
            'q_lens': q_lens,
            'hidden_states': hidden_states
        }
        cp_tokens = np.array([[5, 8]], dtype=np.int32)
        sp_tokens = np.array([[5, 8]], dtype=np.int32)
        model_inputs = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, lm_head_indices,
            is_prefill, None, None, dp_rank_ids, cp_tokens=cp_tokens, sp_tokens=sp_tokens)
        kwargs = {}
        do_reorder_requests, revert_adapter_idx = generator_torch._prepare_model_inputs(model_inputs, kwargs)
        self.assertTrue(np.array_equal(model_inputs.context_length, np.array([5])))

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_sort_model_inputs_by_adapter_ids(self, _):
        generator_torch = GeneratorTorch({})
        generator_torch.mapping = self.mapping
        generator_torch.cache_pool = self.cache_pool
        generator_torch.distributed_enable = False
        generator_torch.model_wrapper = self.model_wrapper
        generator_torch.model_wrapper.model_name = 'qwen'
        generator_torch.model_wrapper.model_id = [0, 1, 2, 3]
        generator_torch.adapter_manager = MagicMock()
        generator_torch.adapter_manager.preprocess_adapter_ids = MagicMock(return_value=[0, 1, 2, 3])
        generator_torch.adapter_manager.check_adapter_ids_is_sorted = MagicMock(return_value=False)
        generator_torch.adapter_manager.sort_adapter_ids = MagicMock(return_value=([1, 0, 2, 3], [1, 0, 2, 3]))

        dp_rank_ids = np.array([0, 1, 2, 3])
        input_ids = FAKE_INPUT_IDS
        position_ids = FAKE_POSITION_IDS
        is_prefill = True
        block_tables = FAKE_BLOCK_TABLES
        slots = FAKE_SLOTS
        input_lengths = FAKE_CONTEXT_LENGTH
        lm_head_indices = np.cumsum(input_lengths) - 1
        adapter_ids = np.array([1, 0, 2, 3])
        model_input = ModelInput(
            input_ids, position_ids, block_tables, slots, input_lengths, 0, lm_head_indices,
            is_prefill, None, adapter_ids, dp_rank_ids)

        do_reorder_requests, revert_adapter_idx = generator_torch._sort_model_inputs_by_adapter_ids(model_input)

        self.assertTrue(do_reorder_requests)
        self.assertEqual(revert_adapter_idx, [1, 0, 2, 3])
        self.assertTrue(np.array_equal(model_input.prefill_head_indices, np.array([6, 119, 232, 237])))

    @patch(MOCKED_INIT_METHOD, return_value=None)
    def test_update_lm_head_indices(self, _):
        model_input = MagicMock()
        model_input.seq_lens = [[10, 20], [5], [], [15]]
        model_kwargs = {}

        generator = GeneratorTorch({})
        generator._update_lm_head_indices(model_input, model_kwargs)

        # flatten_seq_len = [10, 20, 5, 1, 15] -> first element -1 -> [9, 20, 5, 1, 15]
        # cumsum -> [9, 29, 34, 35, 50]
        expected_prefill = np.array([9, 29, 34, 35, 50])
        # dp_logits_num = [2, 1, 1, 1] -> cumsum -> [2, 3, 4, 5]
        expected_dp_logits_num = np.array([2, 3, 4, 5])

        np.testing.assert_array_equal(model_input.prefill_head_indices, expected_prefill)
        np.testing.assert_array_equal(model_kwargs.get("dp_logits_num", None), expected_dp_logits_num)

    @patch(MOCKED_INIT_METHOD, return_value=None)
    @patch("mindie_llm.text_generator.adapter.generator_torch.standardize_path")
    @patch("mindie_llm.text_generator.adapter.generator_torch.check_file_safety")
    def test_get_obfuscation_func(self, mock_check, mock_std_path, _):
        # Arrange
        generator = GeneratorTorch({})
        generator.llm_config = MagicMock()
        generator.llm_config.vocab_size = 1000
        ca_dir = "/fake/ca"
        generator.llm_config.llm.pmcc_obfuscation_options = MagicMock(
            data_obfuscation_ca_dir=ca_dir,
            kms_agent_port=8443
        )
        
        mock_std_path.side_effect = [f"{ca_dir}/file{i}" for i in range(6)]

        # Act
        with self.assertRaises(RuntimeError):
            result = generator._get_obfuscation_func()

    @patch(MOCKED_INIT_METHOD, return_value=None)
    @patch("torch_npu.npu.check_uce_in_memory")
    @patch("torch_npu.npu._get_uce_addr")
    def test_handle_uce_error_no_overlap(self, mock_get_uce_addr, mock_check_uce, _):
        # Mock the return values
        mock_check_uce.return_value = 2
        mock_get_uce_addr.return_value = [{"ptr": 100, "size": 50}]

        generator = GeneratorTorch({})
        generator.cache_pool = self.cache_pool
        generator.npu_device_id = 0

        result, error_msg = generator._handle_uce_error()
        self.assertEqual(result, 1)
        self.assertEqual(error_msg, "HBM uce address not overlap kvcache address, should trigger reschedule")
    
    @patch(MOCKED_INIT_METHOD, return_value=None)
    @patch("torch_npu.npu.check_uce_in_memory")
    @patch("torch_npu.npu._get_uce_addr")
    def test_handle_with_no_uce_error(self, mock_get_uce_addr, mock_check_uce, _):
        # Mock the return values
        mock_check_uce.return_value = 0
        mock_get_uce_addr.return_value = [{"ptr": 100, "size": 50}]

        generator = GeneratorTorch({})
        generator.cache_pool = self.cache_pool
        generator.npu_device_id = 0

        result, error_msg = generator._handle_uce_error()
        self.assertEqual(result, 0)
        self.assertEqual(error_msg, "")
    
    @patch(MOCKED_INIT_METHOD, return_value=None)
    @patch("torch_npu.npu.check_uce_in_memory")
    @patch("torch_npu.npu._get_uce_addr")
    def test_handle_uce_error_empty(self, mock_get_uce_addr, mock_check_uce, _):
        # Mock the return values
        mock_check_uce.return_value = 0
        mock_get_uce_addr.return_value = []

        generator = GeneratorTorch({})
        generator.cache_pool = self.cache_pool
        generator.npu_device_id = 0

        result, error_msg = generator._handle_uce_error()
        self.assertEqual(result, 0)
        self.assertEqual(error_msg, "")
    
    @patch(MOCKED_INIT_METHOD, return_value=None)
    @patch("torch_npu.npu.check_uce_in_memory")
    @patch("torch_npu.npu._get_uce_addr")
    def test_handle_uce_error_overlap(self, mock_get_uce_addr, mock_check_uce, _):
        # Mock the return values
        mock_check_uce.return_value = 3
        addr = self.cache_pool.npu_cache[0][0].data_ptr()
        mock_get_uce_addr.return_value = [{"ptr": addr, "size": 12}]

        generator = GeneratorTorch({})
        generator.cache_pool = self.cache_pool
        generator.npu_device_id = 0

        result, error_msg = generator._handle_uce_error()
        self.assertEqual(result, 2)
        self.assertEqual(error_msg, "")

        addr = self.cache_pool.npu_cache[0][1].data_ptr()
        mock_get_uce_addr.return_value = [{"ptr": addr, "size": 12}]
        result, error_msg = generator._handle_uce_error()
        self.assertEqual(result, 2)
        self.assertEqual(error_msg, "")
    
    @patch("torch_npu.npu._recovery.update_npu_tensor_to_safe")
    def test_is_uce_error_addr_overlap_tensor_addr(self, mock_update):
        # Test overlapping address ranges
        self.assertTrue(is_uce_error_addr_overlap_tensor_addr(100, 200, 100, 200))
        self.assertFalse(is_uce_error_addr_overlap_tensor_addr(50, 180, 100, 200))
        self.assertFalse(is_uce_error_addr_overlap_tensor_addr(150, 400, 100, 200))
        self.assertFalse(is_uce_error_addr_overlap_tensor_addr(300, 400, 100, 200))

    @patch("torch.Tensor")
    def test_get_tensor_address_range(self, mock_tensor):
        # Mock tensor properties
        mock_tensor.data_ptr.return_value = 1000
        mock_tensor.numel.return_value = 10
        mock_tensor.element_size.return_value = 4

        addr_start, addr_end = get_tensor_address_range(mock_tensor)
        self.assertEqual(addr_start, 1000)
        self.assertEqual(addr_end, 1040)

    @patch("torch_npu.npu._recovery.update_npu_tensor_to_safe")
    @patch("mindie_llm.text_generator.adapter.recovery_utils.get_tensor_address_range")
    def test_check_and_recover_uce_in_cache(self, mock_get_range, mock_update):
        # Mock tensor address range
        mock_get_range.return_value = (100, 200)

        # Test recovery
        cache_tensor = MagicMock()
        result = check_and_recover_uce_in_cache(150, 180, cache_tensor, 0, "kcache")
        self.assertTrue(result)
        mock_update.assert_called_once_with(cache_tensor)

        # Test no recovery
        mock_update.reset_mock()
        result = check_and_recover_uce_in_cache(300, 400, cache_tensor, 0, "kcache")
        self.assertFalse(result)
        mock_update.assert_not_called()

if __name__ == "__main__":
    unittest.main()