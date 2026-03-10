# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest.mock import patch, MagicMock, Mock
import pytest
import torch
import numpy as np

from mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp import AclGraphModelWrapperExp


@pytest.fixture
def mock_model_runner_exp():
    """Mock ModelRunnerExp with essential attributes."""
    mock_runner = MagicMock()
    mock_runner.config = {"model_type": "qwen2"}
    mock_runner.config_dict = {"hidden_size": 4096}
    mock_runner.tokenizer = MagicMock()
    mock_runner.device = torch.device("npu:0")
    mock_runner.kv_cache_dtype = torch.float16
    mock_runner.num_layers = 28
    mock_runner.num_kv_heads = 8
    mock_runner.head_size = 128
    mock_runner.k_head_size = 128
    mock_runner.v_head_size = 128
    mock_runner.enable_nz = False
    mock_runner.kvcache_quant_layers = []
    mock_runner.index_head_dim = 128
    mock_runner.num_index_heads = 0
    mock_runner.max_position_embeddings = 32768
    mock_runner.max_seq_len = -1
    mock_runner.adapter_manager = None
    mock_runner.model = MagicMock()
    mock_runner.model.is_multimodal = False
    mock_runner.build_forward_context = MagicMock(return_value=MagicMock())
    mock_runner.generate_position_ids = MagicMock(return_value=[0, 1, 2])
    mock_runner.input_builder = MagicMock()
    mock_runner.input_builder.make_context = MagicMock(return_value=[1, 2, 3])
    return mock_runner


@pytest.fixture
def mock_parallel_info_manager():
    """Mock parallel info manager with required parallel types."""
    mock_manager = MagicMock()
    
    # Mock ParallelInfo objects
    mock_dp_info = MagicMock()
    mock_dp_info.group_size = 2
    mock_sp_info = MagicMock()
    mock_sp_info.group_size = 1
    mock_cp_info = MagicMock()
    mock_cp_info.group_size = 1
    
    mock_manager.get.side_effect = lambda pt: {
        "ParallelType.ATTN_DP": mock_dp_info,
        "ParallelType.ATTN_INNER_SP": mock_sp_info,
        "ParallelType.ATTN_CP": mock_cp_info,
    }.get(str(pt), MagicMock(group_size=1))
    
    return mock_manager


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_init_success(
    mock_get_parallel_info,
    mock_model_runner_class,
    mock_model_runner_exp,
    mock_parallel_info_manager
):
    """Test successful initialization of AclGraphModelWrapperExp."""
    # Setup mocks
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    # Initialize wrapper
    wrapper = AclGraphModelWrapperExp(
        rank=0,
        local_rank=0,
        world_size=2,
        npu_device_id=0,
        model_id="qwen2-7b",
        trust_remote_code=True,
        load_tokenizer=True,
        max_batch_size=8,
        tp=2,
        dp=2
    )

    # Assertions
    assert wrapper.config == mock_model_runner_exp.config
    assert wrapper.tokenizer == mock_model_runner_exp.tokenizer
    assert wrapper.device == mock_model_runner_exp.device
    assert wrapper.rank == 0
    assert wrapper.dp_size == 2
    assert wrapper.sp_size == 1
    assert wrapper.cp_size == 1
    assert wrapper.model_info is not None
    assert wrapper.max_position_embeddings == 32768
    assert wrapper.is_multimodal is False

    # Verify ModelRunnerExp was called with correct args
    mock_model_runner_class.assert_called_once_with(
        model_name_or_path="qwen2-7b",
        rank=0,
        local_rank=0,
        npu_id=0,
        world_size=2,
        trust_remote_code=True,
        load_tokenizer=True,
        tokenizer_path=None,
        max_position_embeddings=None,
        num_speculative_tokens=None,
        max_batch_size=8,
        models_dict=None,
        tp=2,
        dp=2,
        cp=-1,
        moe_tp=-1,
        moe_ep=-1,
        role='standard',
        max_seq_len=-1,
        sampler_config=None,
        distributed_enable=False,
    )


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs(
    mock_get_parallel_info,
    mock_model_runner_class,
    mock_model_runner_exp,
    mock_parallel_info_manager
):
    """Test prepare_model_inputs method."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test"
    )

    # Create mock model_inputs
    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2, 3]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = [[10, 11], [12, 13]]
    mock_inputs.input_lengths = None  # Will be set by prepare_model_inputs

    result = wrapper.prepare_model_inputs(mock_inputs)

    # Check tensor conversion
    assert torch.is_tensor(result.input_ids)
    assert result.input_ids.device == wrapper.device
    assert torch.is_tensor(result.position_ids)
    assert result.position_ids.dtype == torch.int64

    # Check block tables assignment
    assert result.block_tables_array == mock_inputs.block_tables

    # Check input_lengths binding
    assert result.input_lengths is not None
    assert result.forward_context is not None


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_forward_success(
    mock_get_parallel_info,
    mock_model_runner_class,
    mock_model_runner_exp,
    mock_parallel_info_manager
):
    """Test forward method success path."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test"
    )

    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2, 3]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    # Mock forward result
    mock_result = {"logits": torch.tensor([1.0])}
    mock_model_runner_exp.forward.return_value = mock_result

    result = wrapper.forward(mock_inputs, npu_cache="dummy_cache")

    # Verify calls
    wrapper.model_runner.forward.assert_called_once()
    assert result == mock_result


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_generate_position_ids(
    mock_get_parallel_info,
    mock_model_runner_class,
    mock_model_runner_exp,
    mock_parallel_info_manager
):
    """Test generate_position_ids method."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test"
    )

    input_ids = np.array([1, 2, 3])
    result = wrapper.generate_position_ids(input_ids)

    wrapper.model_runner.generate_position_ids.assert_called_once_with(input_ids)
    assert result == [0, 1, 2]


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_make_context(
    mock_get_parallel_info,
    mock_model_runner_class,
    mock_model_runner_exp,
    mock_parallel_info_manager
):
    """Test make_context method."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test"
    )

    conversation = [{"role": "user", "content": "Hello"}]
    result = wrapper.make_context(conversation, add_generation_prompt=True)

    wrapper.model_runner.input_builder.make_context.assert_called_once_with(
        0, conversation, add_generation_prompt=True
    )
    assert result == [1, 2, 3]


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_resume_hccl_comm_raises(
    mock_get_parallel_info,
    mock_model_runner_class,
    mock_model_runner_exp,
    mock_parallel_info_manager
):
    """Test resume_hccl_comm raises NotImplementedError."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test"
    )

    with pytest.raises(NotImplementedError):
        wrapper.resume_hccl_comm()