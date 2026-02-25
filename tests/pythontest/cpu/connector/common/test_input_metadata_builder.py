# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import array
import unittest
from unittest.mock import Mock, patch
from dataclasses import dataclass
import struct
import numpy as np

from mindie_llm.connector.common.model_execute_data_pb2 import (
    SequenceGroupMetadata,
    SamplingParams,
    ExecuteModelRequest,
    PullKVRequest
)
from mindie_llm.text_generator.utils.input_metadata import SIMULATE_SEQUENCE_ID
from mindie_llm.connector.common.input_metadata_builder import (
    convert_bytes_to_list,
    parse_all_dp_batches_seq_lens,
    parse_sampling_parameters,
    parse_swap_blocks,
    generate_lora_strings,
    make_dummy_input_metadata,
    make_dummy_input_metadata_dmi_decoder,
    convert_execute_model_request_to_input_metadata_composite,
    convert_pull_kv_request_to_input_metadata_composite,
    get_attribute_info,
    REPETITION_PENALTY_INDEX,
    ConvertPara,
)


class PDRole:
    PREFILL_ROLE = 1
    DECODE_ROLE = 2
    UNKNOWN_ROLE = 3


@dataclass
class MockModelConfig:
    max_seq_len: int
    cache_block_size: int
    rank: int
    tp_size: int
    dp_size: int
    p_inst_enable_sp_cp: bool
    sp_size: int
    cp_size: int
    speculation_gamma: int
    enable_mtp: bool


class TestInputMetadataBuilder(unittest.TestCase):

    def setUp(self):
        self.execute_model_request = ExecuteModelRequest()
        seq_group_metadata = SequenceGroupMetadata()
        seq_group_metadata.request_id = "1"
        seq_group_metadata.is_prompt = True
        seq_group_metadata.sampling_params.repetition_penalty = 1.05
        seq_group_metadata.sampling_params.frequency_penalty = 0
        seq_group_metadata.sampling_params.presence_penalty = 0
        seq_group_metadata.sampling_params.temperature = 0.7
        seq_group_metadata.sampling_params.top_k = 20
        seq_group_metadata.sampling_params.top_p = 0.80000001
        seq_group_metadata.sampling_params.top_logprobs = 0
        seq_group_metadata.sampling_params.n = 1
        seq_group_metadata.do_sample = True
        seq_group_metadata.sampling_params.seed = 52516453
        s64_array = array.array('q',
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        seq_group_metadata.block_tables = s64_array.tobytes()

        seq_group_metadata.seqIds = struct.pack('<1q', 1)

        prompt_len_array = array.array('q', [34])
        seq_group_metadata.prompt_lens = prompt_len_array.tobytes()
        prompt_array = array.array('q',
                                   [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553])
        seq_group_metadata.prompt_token_ids = prompt_array.tobytes()
        self.execute_model_request.seq_group_metadata_list.append(seq_group_metadata)

        self.sp_seq_group_metadata = SequenceGroupMetadata()
        self.sp_seq_group_metadata.request_id = "2"
        self.sp_seq_group_metadata.sp_rank_token_num.extend([10, 20, 30])
        self.sp_seq_group_metadata.sp_rank_block_num.extend([2, 2, 2])
        sp_block_array = array.array('q', [1, 2, 3, 4, 5, 6])
        self.sp_seq_group_metadata.block_tables = sp_block_array.tobytes()
        self.sp_seq_group_metadata.seqIds = struct.pack('<2q', 2, 3)
        self.sp_seq_group_metadata.prompt_lens = struct.pack('<2q', 10, 20)
        self.sp_seq_group_metadata.sampling_params.seed = 12345

        self.num_npu_blocks = 8
        self.block_size = 128
        self.empty_execute_model_request = ExecuteModelRequest()

        self.host_info = Mock()
        self.host_info.host_ip = "192.168.1.1"
        self.host_info.cluster_id = "100"
        self.host_info.HasField = Mock(return_value=False)

        self.device_info = Mock()
        self.device_info.device_ip = "10.0.0.1"
        self.device_info.physical_id = 0
        self.device_info.HasField = Mock(return_value=False)

        self.remote_info = Mock()
        self.remote_info.host_info = [self.host_info]
        self.remote_info.device_info = [self.device_info]

        self.pd_link_info = Mock()
        self.pd_link_info.pd_role = PDRole.UNKNOWN_ROLE
        self.pd_link_info.change_role = False
        self.pd_link_info.link_num = 1
        self.pd_link_info.unlink_num = 0
        self.pd_link_info.link_info = [self.remote_info]
        self.pd_link_info.unlink_info = []
        self.pd_link_info.instance2sp = {}
        self.pd_link_info.instance2cp = {}
        self.pd_link_info.host_ip_num = 1
        self.pd_link_info.super_id_num = 0
        self.pd_link_info.contains_dp_instance_ids = 0

        self.link_request = Mock()
        self.link_request.pd_link_info = [self.pd_link_info]

        self.pull_kv_request = PullKVRequest()
        pull_kv_info = self.pull_kv_request.pull_kv_infos.add()
        pull_kv_info.seq_group_metadata.CopyFrom(seq_group_metadata)
        pull_kv_info.cluster_id = "100"
        sp_pull_kv_info = self.pull_kv_request.pull_kv_infos.add()
        sp_pull_kv_info.seq_group_metadata.CopyFrom(self.sp_seq_group_metadata)
        sp_pull_kv_info.cluster_id = "200"

        self.mock_input_metadata = Mock()
        self.mock_input_metadata.split_end_position = np.array([10, 20, 30])
        self.mock_input_metadata.split_start_position = np.array([0, 10, 20])
        self.mock_input_metadata.block_tables = np.array([[1, -1, -1], [2, 3, -1], [4, 5, 6]])
        self.mock_input_metadata.input_ids = np.array([100, 200])
        self.mock_input_metadata_composite = Mock()
        self.mock_input_metadata_composite.input_metadata = self.mock_input_metadata

    def test_convert_bytes_to_list(self):
        byte_data = struct.pack("<2q", 100, 200)
        self.assertEqual(convert_bytes_to_list(byte_data), [100, 200])
        self.assertEqual(convert_bytes_to_list(b""), [])

    def test_parse_all_dp_batches_seq_lens(self):
        class MockDPBatch:
            def __init__(self, seq_lens):
                self.seq_lens = seq_lens

        all_dp_batches = [
            MockDPBatch([10, 20]),
            MockDPBatch([30, 40])
        ]
        result = parse_all_dp_batches_seq_lens(all_dp_batches)
        self.assertEqual(result, [[10, 20], [30, 40]])

    def test_parse_sampling_parameters(self):
        sampling_params = SamplingParams(
            repetition_penalty=1.2,
            temperature=0.8,
            top_k=50
        )
        seq_group_metadata = SequenceGroupMetadata(do_sample=True, sampling_params=sampling_params)
        result = parse_sampling_parameters(seq_group_metadata)
        self.assertAlmostEqual(result[0][REPETITION_PENALTY_INDEX], 1.2, places=4)
        self.assertAlmostEqual(result[0]["temperature"], 0.8, places=4)
        self.assertAlmostEqual(result[0]["do_sample"], 1.0, places=4)
        self.assertAlmostEqual(result[0]["top_k"], 50, places=4)

    def test_parse_swap_blocks(self):
        class MockSwapBlock:
            def __init__(self, num1, num2):
                self.num1 = num1
                self.num2 = num2

        swap_in = [MockSwapBlock(10, 20)]
        swap_out = [MockSwapBlock(30, 40)]
        result = parse_swap_blocks(swap_in, swap_out)
        self.assertEqual(result, [[[0, 10, 20], [1, 30, 40]]])
        self.assertIsNone(parse_swap_blocks([], []))

    def test_generate_lora_strings(self):
        meta = SequenceGroupMetadata(lora_id="lora_123")
        self.assertEqual(generate_lora_strings(meta), "lora_123")
        meta = SequenceGroupMetadata(lora_id="None")
        self.assertIsNone(generate_lora_strings(meta))

    def test_make_dummy_input_metadata(self):
        dp_batch_seq_lens_mock = Mock()
        dp_batch_seq_lens_mock.seq_lens = [10, 20]
        execute_model_request_mock = Mock()
        execute_model_request_mock.all_dp_batches_seq_lens = [dp_batch_seq_lens_mock]
        self.execute_request = Mock()
        self.execute_request.execute_model_request = execute_model_request_mock

        model_config = MockModelConfig(
            max_seq_len=1024,
            cache_block_size=64,
            rank=0,
            tp_size=1,
            dp_size=1,
            p_inst_enable_sp_cp=True,
            sp_size=3,
            cp_size=1,
            speculation_gamma=0,
            enable_mtp=False
        )
        num_npu_blocks = 50

        metadata = make_dummy_input_metadata(
            execute_request=self.execute_request,
            num_npu_blocks=num_npu_blocks,
            model_config=model_config
        )

        block_padding = model_config.max_seq_len // model_config.cache_block_size
        self.assertEqual(metadata.block_tables.shape, (1, model_config.sp_size, block_padding))
        self.assertEqual(metadata.block_tables[0][0][0], num_npu_blocks - 1)
        self.assertTrue(all(x == -1 for x in metadata.block_tables[0][0][1:]))
        for slice_idx in range(1, model_config.sp_size):
            self.assertTrue(all(x == -1 for x in metadata.block_tables[0][slice_idx]))
        self.assertIsNotNone(metadata.sp_tokens)
        self.assertEqual(metadata.sp_tokens.shape, (1, model_config.sp_size))
        self.assertTrue(np.array_equal(metadata.sp_tokens[0], [1, 0, 0]))

    def test_make_dummy_input_metadata_dmi_decoder(self):
        source_metadata = Mock()
        source_metadata.batch_dp_rank_ids = np.array([2, 3])
        model_config = MockModelConfig(
            max_seq_len=512,
            cache_block_size=32,
            rank=1,
            tp_size=1,
            dp_size=4,
            p_inst_enable_sp_cp=False,
            sp_size=2,
            cp_size=1,
            speculation_gamma=0,
            enable_mtp=False
        )
        num_npu_blocks = 20

        metadata = make_dummy_input_metadata_dmi_decoder(
            source_input_metadata=source_metadata,
            num_npu_blocks=num_npu_blocks,
            model_config=model_config
        )

        block_padding = model_config.max_seq_len // model_config.cache_block_size
        self.assertEqual(metadata.block_tables.shape, (1, block_padding))
        self.assertEqual(metadata.block_tables[0][0], num_npu_blocks - 1)
        self.assertTrue(all(x == -1 for x in metadata.block_tables[0][1:]))
        self.assertEqual(metadata.batch_dp_rank_ids.tolist(), source_metadata.batch_dp_rank_ids.tolist())
        self.assertFalse(metadata.has_sampling)
        self.assertFalse(metadata.is_prefill)
        self.assertTrue(metadata.is_dummy_batch)

    def test_convert_proto_normal_prefill(self):
        composite = convert_execute_model_request_to_input_metadata_composite(
            request=self.execute_model_request,
            num_npu_blocks=self.num_npu_blocks,
            block_size=self.block_size
        )

        self.assertTrue(hasattr(composite, "input_metadata"))
        self.assertTrue(hasattr(composite, "block_copy"))
        self.assertTrue(hasattr(composite, "block_op"))
        input_metadata = composite.input_metadata

        self.assertEqual(input_metadata.batch_size, 1)
        self.assertTrue(input_metadata.is_prefill)
        self.assertEqual(input_metadata.max_block_size, self.block_size)

        self.assertEqual(input_metadata.batch_request_ids[0], "1")
        self.assertEqual(input_metadata.batch_sequence_ids[0].tolist(), [1])
        self.assertEqual(input_metadata.batch_seq_len.tolist(), [34])

        self.assertTrue(input_metadata.has_sampling)
        self.assertAlmostEqual(input_metadata.batch_sampling_params[0]["repetition_penalty"], 1.05, places=4)
        self.assertAlmostEqual(input_metadata.batch_sampling_params[0]["temperature"], 0.7, places=4)
        self.assertEqual(input_metadata.input_ids.tolist(),
                         [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553])

        self.assertEqual(input_metadata.batch_block_tables.shape, (1, 1))

        request2 = ExecuteModelRequest()
        seq_group_metadata = SequenceGroupMetadata()
        seq_group_metadata.request_id = "1"
        seq_group_metadata.is_prompt = True
        seq_group_metadata.sampling_params.temperature = 0
        s64_array = array.array('q',
                                [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        seq_group_metadata.block_tables = s64_array.tobytes()
        seq_group_metadata.seqIds = struct.pack('<1q', 1)
        prompt_len_array = array.array('q', [34])
        seq_group_metadata.prompt_lens = prompt_len_array.tobytes()
        prompt_array = array.array('q',
                                   [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553])
        seq_group_metadata.prompt_token_ids = prompt_array.tobytes()
        request2.seq_group_metadata_list.append(seq_group_metadata)
        composite2 = convert_execute_model_request_to_input_metadata_composite(
            request=request2,
            num_npu_blocks=self.num_npu_blocks,
            block_size=self.block_size
        )
        self.assertEqual(composite2.input_metadata.batch_logprobs[0], None)

    def test_convert_proto_sp_cp_branch(self):
        # 构造一个 SP 场景的 ExecuteModelRequest
        request = ExecuteModelRequest()

        sp_seq_group_metadata = SequenceGroupMetadata()
        sp_seq_group_metadata.request_id = "1"
        prompt_array = array.array('q',
                                   [151644, 8948, 198, 2610, 525, 1207, 16948, 11])
        sp_seq_group_metadata.prompt_token_ids = prompt_array.tobytes()
        config = MockModelConfig(
            max_seq_len=1024,
            cache_block_size=64,
            rank=0,
            tp_size=1,
            dp_size=1,
            p_inst_enable_sp_cp=False,
            sp_size=2,
            cp_size=2,
            speculation_gamma=0,
            enable_mtp=False,
        )
        config.cp_size = 2
        config.sp_size = 2
        sp_seq_group_metadata.do_sample = False
        sp_seq_group_metadata.sp_rank_token_num.extend([1, 1, 1, 1, 1, 1, 1, 1])
        sp_seq_group_metadata.sp_rank_block_num.extend([1, 1, 1, 1, 1, 1, 1, 1])
        sp_block_array = array.array('q', [0] * 8)
        sp_seq_group_metadata.block_tables = sp_block_array.tobytes()
        prompt_len_array = array.array('q', [8])
        sp_seq_group_metadata.prompt_lens = prompt_len_array.tobytes()
        sp_seq_group_metadata.seqIds = struct.pack('<1q', 100)

        request.seq_group_metadata_list.append(sp_seq_group_metadata)

        # 调用转换函数
        composite = convert_execute_model_request_to_input_metadata_composite(
            request=request,
            num_npu_blocks=self.num_npu_blocks,
            block_size=self.block_size,
            config=config
        )

        self.assertTrue(hasattr(composite, "input_metadata"))
        self.assertTrue(hasattr(composite, "block_copy"))
        self.assertTrue(hasattr(composite, "block_op"))
        input_metadata = composite.input_metadata

        self.assertEqual(input_metadata.batch_size, 1)
        self.assertTrue(input_metadata.is_prefill)
        self.assertEqual(input_metadata.max_block_size, self.block_size)

        self.assertEqual(input_metadata.batch_request_ids[0], "1")
        self.assertEqual(input_metadata.batch_sequence_ids[0].tolist(), [100])
        self.assertEqual(input_metadata.batch_seq_len.tolist(), [8])
        self.assertEqual(input_metadata.input_ids.tolist(),
                         [151644, 8948, 198, 2610, 525, 1207, 16948, 11])

        self.assertEqual(input_metadata.batch_block_tables.shape, (1, 8, 1))

    def test_convert_pull_kv_request_to_input_metadata_composite(self):
        for pull_kv_info in self.pull_kv_request.pull_kv_infos:
            pull_kv_info.seq_group_metadata.sp_rank_block_num.clear()

        from mindie_llm.text_generator.utils.input_metadata import InputMetadata

        with patch.object(InputMetadata, '__post_init__', new=lambda self: None):
            composite = convert_pull_kv_request_to_input_metadata_composite(
                request=self.pull_kv_request,
                num_npu_blocks=self.num_npu_blocks,
                block_size=self.block_size
            )

            self.assertEqual(composite.input_metadata.batch_size, 2)
            self.assertTrue(composite.input_metadata.is_prefill)
            self.assertEqual(composite.input_metadata.batch_request_ids.tolist(), ["1", "2"])

            for pull_kv_info in self.pull_kv_request.pull_kv_infos:
                pull_kv_info.seq_group_metadata.computed_block_lens = struct.pack('<2q', 0, 0)
                pull_kv_info.seq_group_metadata.remote_computed_block_lens = struct.pack('<2q', 0, 0)

            composite_empty = convert_pull_kv_request_to_input_metadata_composite(
                request=self.pull_kv_request,
                num_npu_blocks=self.num_npu_blocks,
                block_size=self.block_size
            )
            self.assertIsNone(composite_empty.input_metadata.computed_blocks)
            self.assertIsNone(composite_empty.input_metadata.remote_computed_blocks)

    def test_convert_proto_mix_mode(self):
        seq_meta2 = self.execute_model_request.seq_group_metadata_list[0]
        seq_meta2.request_id = "test_req_002"
        seq_meta2.do_sample = False

        composite = convert_execute_model_request_to_input_metadata_composite(
            request=self.execute_model_request,
            num_npu_blocks=self.num_npu_blocks,
            block_size=self.block_size,
            convert_para=ConvertPara(is_prefill=False, is_mix=True),
            is_mix_model=True
        )
        input_metadata = composite.input_metadata

        self.assertTrue(input_metadata.is_mix)
        self.assertEqual(input_metadata.mix_decode_bs, 1)
        self.assertEqual(input_metadata.batch_size, 1)
        self.assertEqual(input_metadata.split_end_position.tolist(), [])

    def test_convert_proto_empty_request(self):
        with self.assertRaises(ValueError, msg="No sequence group metadata in request"):
            convert_execute_model_request_to_input_metadata_composite(
                request=self.empty_execute_model_request,
                num_npu_blocks=self.num_npu_blocks,
                block_size=self.block_size
            )

    def test_prefill_role_no_super_id(self):
        self.pd_link_info.pd_role = PDRole.PREFILL_ROLE

        attribute_info, device_data, policy = get_attribute_info(self.link_request)

        self.assertTrue(np.array_equal(attribute_info, np.array([[1, 0, 1, 0, 1, 0, 0]], dtype=np.int64)))
        self.assertEqual(device_data.shape, (1, 2, 9))
        self.assertTrue(np.array_equal(device_data[0, 0], [192, 168, 1, 1, -1, -1, -1, -1, 100]))
        self.assertTrue(np.array_equal(device_data[0, 1], [10, 0, 0, 1, -1, -1, -1, -1, 0]))

    def test_decode_role_with_super_id(self):
        self.pd_link_info.pd_role = PDRole.DECODE_ROLE
        self.pd_link_info.super_id_num = 1
        self.pd_link_info.change_role = True
        self.pd_link_info.instance2sp = {1: 8, 2: 16}
        self.pd_link_info.instance2cp = {1: 1, 2: 1}

        self.host_info.HasField = Mock(side_effect=lambda x: x == "super_pod_id")
        self.host_info.super_pod_id = 10
        self.device_info.HasField = Mock(side_effect=lambda x: x == "super_device_id")
        self.device_info.super_device_id = 20

        attribute_info, device_data, policy = get_attribute_info(self.link_request)

        self.assertTrue(np.array_equal(attribute_info, np.array([[2, 1, 1, 0, 1, 1, 0]], dtype=np.int64)))
        self.assertEqual(device_data.shape, (1, 2, 10))
        self.assertTrue(np.array_equal(device_data[0, 0], [192, 168, 1, 1, -1, -1, -1, -1, 100, 10]))
        self.assertTrue(np.array_equal(device_data[0, 1], [10, 0, 0, 1, -1, -1, -1, -1, 0, 20]))
        self.assertTrue(np.array_equal(policy, np.array([[1, 8, 1], [2, 16, 1]], dtype=np.int64)))
        
    def test_convert_proto_simulate_inference(self):
        """Test simulate inference with special seqId SIMULATE_SEQUENCE_ID"""
        # Modify the seqId to simulate inference value
        seq_group_metadata = self.execute_model_request.seq_group_metadata_list[0]
        seq_group_metadata.seqIds = struct.pack('<1q', 9223372036854774)

        composite = convert_execute_model_request_to_input_metadata_composite(
            request=self.execute_model_request,
            num_npu_blocks=self.num_npu_blocks,
            block_size=self.block_size
        )

        input_metadata = composite.input_metadata

        # Verify simulate inference uses fixed block id (7 = 8 - 1)
        self.assertEqual(input_metadata.batch_size, 1)
        self.assertTrue(input_metadata.is_prefill)
        self.assertEqual(input_metadata.batch_sequence_ids[0].tolist(), [SIMULATE_SEQUENCE_ID])
        self.assertEqual(input_metadata.block_tables[0][0], self.num_npu_blocks - 1)

    def test_convert_proto_simulate_inference_sp_cp_with_normal_request_batch(self):
        """Test simulate inference in SP/CP scenario batched with normal requests.
        
        This test verifies that when a simulate inference request (with SIMULATE_SEQUENCE_ID)
        is batched together with normal SP/CP requests, the numpy array dimensions align correctly.
        The virtual block table should have the correct length matching sp_rank_block_num.
        """
        # SP/CP config with sp_size=4
        sp_config = MockModelConfig(
            max_seq_len=1024,
            cache_block_size=64,
            rank=0,
            tp_size=1,
            dp_size=1,
            p_inst_enable_sp_cp=True,
            sp_size=4,
            cp_size=1,
            speculation_gamma=0,
            enable_mtp=False
        )

        normal_sp_request = SequenceGroupMetadata()
        normal_sp_request.request_id = "normal_sp_1"
        normal_sp_request.sp_rank_id = 0
        normal_sp_request.sp_rank_token_num.extend([10, 20, 30, 0])  # sp_size = 4
        normal_sp_request.sp_rank_block_num.extend([2, 1, 2, 0])  # total_blocks = 5
        normal_block_array = array.array('q', [1, 2, 3, 4, 5])  # 5 blocks
        normal_sp_request.block_tables = normal_block_array.tobytes()
        normal_sp_request.seqIds = struct.pack('<1q', 100)
        normal_sp_request.prompt_lens = struct.pack('<1q', 60)
        normal_prompt_array = array.array('q', [1, 2, 3, 4, 5])  # non-empty prompt tokens
        normal_sp_request.prompt_token_ids = normal_prompt_array.tobytes()
        normal_sp_request.sampling_params.seed = 12345
        normal_sp_request.sampling_params.max_output_len = 100

        simulate_sp_request = SequenceGroupMetadata()
        simulate_sp_request.request_id = "simulate_sp_1"
        simulate_sp_request.sp_rank_id = 0
        simulate_sp_request.sp_rank_token_num.extend([10, 20, 30, 0])  # sp_size = 4, same as normal
        simulate_sp_request.sp_rank_block_num.extend([2, 1, 2, 0])  # total_blocks = 5, same as normal
        simulate_sp_request.block_tables = b''
        simulate_sp_request.seqIds = struct.pack('<1q', SIMULATE_SEQUENCE_ID)  # Use simulate sequence id
        simulate_sp_request.prompt_lens = struct.pack('<1q', 60)
        simulate_prompt_array = array.array('q', [1, 2, 3])  # non-empty prompt tokens
        simulate_sp_request.prompt_token_ids = simulate_prompt_array.tobytes()
        simulate_sp_request.sampling_params.seed = 54321
        simulate_sp_request.sampling_params.max_output_len = 100

        mixed_request = ExecuteModelRequest()
        mixed_request.seq_group_metadata_list.append(normal_sp_request)
        mixed_request.seq_group_metadata_list.append(simulate_sp_request)

        composite = convert_execute_model_request_to_input_metadata_composite(
            request=mixed_request,
            num_npu_blocks=self.num_npu_blocks,
            block_size=self.block_size,
            config=sp_config
        )

        input_metadata = composite.input_metadata
        self.assertEqual(input_metadata.batch_size, 2)
        self.assertEqual(len(input_metadata.block_tables.shape), 3)
        self.assertEqual(input_metadata.block_tables.shape[0], 2)  # batch size = 2
        self.assertEqual(input_metadata.block_tables.shape[1], 4)  # sp_size = 4

        normal_block_table = input_metadata.block_tables[0]
        self.assertEqual(normal_block_table[0][0], 1)  # rank 0, first block
        self.assertEqual(normal_block_table[0][1], 2)  # rank 0, second block
        self.assertEqual(normal_block_table[1][0], 3)  # rank 1, first block
        self.assertEqual(normal_block_table[2][0], 4)  # rank 2, first block
        self.assertEqual(normal_block_table[2][1], 5)  # rank 2, second block

        simulate_block_table = input_metadata.block_tables[1]
        virtual_block_id = self.num_npu_blocks - 1  # 7
        self.assertEqual(simulate_block_table[0][0], virtual_block_id)
        self.assertEqual(simulate_block_table[0][1], -1)
        self.assertEqual(simulate_block_table[1][0], -1)
        self.assertEqual(simulate_block_table[2][0], -1)
        self.assertEqual(simulate_block_table[2][1], -1)

        self.assertEqual(input_metadata.batch_sequence_ids[0].tolist(), [100])
        self.assertEqual(input_metadata.batch_sequence_ids[1].tolist(), [SIMULATE_SEQUENCE_ID])


if __name__ == "__main__":
    unittest.main()