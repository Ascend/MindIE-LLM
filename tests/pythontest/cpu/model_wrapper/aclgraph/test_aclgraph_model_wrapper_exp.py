# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import sys
from unittest.mock import patch, MagicMock, Mock
import pytest
import torch
import numpy as np

# Mock mie_ops before any mindie imports to avoid NPU hardware detection
mock_mie_ops = MagicMock()
sys.modules["mindie_llm.runtime.ops.mie_ops"] = mock_mie_ops

from mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp import AclGraphModelWrapperExp  # noqa: E402


@pytest.fixture
def mock_model_runner_exp():
    """Mock ModelRunnerExp with essential attributes."""
    mock_runner = MagicMock()
    mock_runner.config = {"model_type": "qwen2"}
    mock_runner.config_dict = {"hidden_size": 4096}
    mock_runner.tokenizer = MagicMock()
    mock_runner.device = torch.device("cpu")
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
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
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
        dp=2,
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
        role="standard",
        plugin_params="",
        max_seq_len=-1,
        block_size=-1,
        sampler_config=None,
        distributed_enable=False,
    )


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs method."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    # Create mock model_inputs
    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2, 3]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = [[10, 11], [12, 13]]
    mock_inputs.input_lengths = None  # Will be set by prepare_model_inputs

    result, _ = wrapper.prepare_model_inputs(mock_inputs)

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
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test forward method success path."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

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
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test generate_position_ids method."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    input_ids = np.array([1, 2, 3])
    result = wrapper.generate_position_ids(input_ids)

    wrapper.model_runner.generate_position_ids.assert_called_once_with(input_ids)
    assert result == [0, 1, 2]


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_make_context(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test make_context method."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    conversation = [{"role": "user", "content": "Hello"}]
    result = wrapper.make_context(conversation, add_generation_prompt=True)

    wrapper.model_runner.input_builder.make_context.assert_called_once_with(0, conversation, add_generation_prompt=True)
    assert result == [1, 2, 3]


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_resume_hccl_comm_raises(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test resume_hccl_comm raises NotImplementedError."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    with pytest.raises(NotImplementedError):
        wrapper.resume_hccl_comm()


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_forward_from_model_inputs_success(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test forward_from_model_inputs success path."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_result = {"logits": torch.tensor([2.0])}
    mock_model_runner_exp.forward.return_value = mock_result

    result = wrapper.forward_from_model_inputs(
        npu_cache="cache",
        input_ids=torch.tensor([1, 2, 3]),
        position_ids=torch.tensor([0, 1, 2]),
        forward_context=MagicMock(),
        extra_param="test",
    )

    mock_model_runner_exp.forward.assert_called_once()
    call_kwargs = mock_model_runner_exp.forward.call_args
    assert call_kwargs[0][0] == "cache"  # npu_cache
    assert torch.equal(call_kwargs[0][1], torch.tensor([1, 2, 3]))  # input_ids
    assert result == mock_result


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_forward_from_model_inputs_raises(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test forward_from_model_inputs re-raises exceptions."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_model_runner_exp.forward.side_effect = RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        wrapper.forward_from_model_inputs(
            npu_cache="cache",
            input_ids=torch.tensor([1]),
            position_ids=torch.tensor([0]),
        )


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_forward_error_handling(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test forward method propagates errors."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_model_runner_exp.forward.side_effect = RuntimeError("infer error")

    mock_inputs = MagicMock()
    mock_inputs.input_ids = [1, 2]
    mock_inputs.position_ids = [0, 1]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    with pytest.raises(RuntimeError, match="infer error"):
        wrapper.forward(mock_inputs, npu_cache="cache")


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_init_with_role_prefill(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test initialization with prefill role."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test", role="prefill"
    )
    assert wrapper.rank == 0


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_init_with_plugin_params(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test initialization with plugin_params and num_speculative_tokens."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    AclGraphModelWrapperExp(
        rank=0,
        local_rank=0,
        world_size=2,
        npu_device_id=0,
        model_id="test",
        plugin_params='{"plugin_type":"mtp, prefix_cache","num_speculative_tokens":2}',
        num_speculative_tokens=2,
    )

    kwargs = mock_model_runner_class.call_args[1]
    assert kwargs["plugin_params"] == '{"plugin_type":"mtp, prefix_cache","num_speculative_tokens":2}'
    assert kwargs["num_speculative_tokens"] == 2


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_generate_position_ids_error(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test generate_position_ids error handling."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")
    wrapper.model_runner.generate_position_ids.side_effect = ValueError("invalid ids")

    input_ids = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="invalid ids"):
        wrapper.generate_position_ids(input_ids)


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_make_context_error(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test make_context error handling."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")
    wrapper.model_runner.input_builder.make_context.side_effect = RuntimeError("make ctx failed")

    with pytest.raises(RuntimeError, match="make ctx failed"):
        wrapper.make_context([{"role": "user", "content": "Hi"}])


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_q_lens(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with q_lens kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [10, 20, 30]
    mock_inputs.position_ids = [5, 6, 7]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    q_lens = [3]
    result, result_kwargs = wrapper.prepare_model_inputs(mock_inputs, q_lens=q_lens)

    assert torch.is_tensor(result.q_lens)
    assert torch.equal(result.q_lens.cpu(), torch.tensor(q_lens))
    assert torch.is_tensor(result_kwargs["q_lens"])


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_mtp_indices(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with mtp_logits_gather_indices kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2, 3]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    mtp_indices = torch.tensor([0, 1])
    _, result_kwargs = wrapper.prepare_model_inputs(mock_inputs, mtp_logits_gather_indices=mtp_indices)

    assert torch.is_tensor(result_kwargs["mtp_logits_gather_indices"])


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_shard_indices(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with shard_effective_token_indices kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [4, 5, 6]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    shard_indices = [1, 2]
    _, result_kwargs = wrapper.prepare_model_inputs(mock_inputs, shard_effective_token_indices=shard_indices)

    assert torch.is_tensor(result_kwargs["shard_effective_token_indices"])


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_lm_head_local_dp(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with lm_head_local_dp kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [7, 8, 9]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    lm_head_dp = [0]
    _, result_kwargs = wrapper.prepare_model_inputs(mock_inputs, lm_head_local_dp=lm_head_dp)

    assert torch.is_tensor(result_kwargs["lm_head_local_dp"])


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_sub_model_inputs(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with sub_model_inputs kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2]
    mock_inputs.position_ids = [0, 1]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    sub_inputs = Mock()
    sub_inputs.input_ids = [5, 6]
    sub_inputs.position_ids = [3, 4]
    sub_inputs.slots = [10, 11]
    sub_inputs.context_length = [2]
    sub_inputs.prefill_head_indices = [1]
    sub_inputs.block_tables = [[0, 1]]

    _, result_kwargs = wrapper.prepare_model_inputs(mock_inputs, sub_model_inputs=sub_inputs)

    assert result_kwargs["sub_model_inputs"] is sub_inputs
    assert torch.is_tensor(result_kwargs["sub_model_inputs"].input_ids)
    assert torch.is_tensor(result_kwargs["sub_model_inputs"].slots)


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_hidden_states(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with hidden_states kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2, 3]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    hs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result, result_kwargs = wrapper.prepare_model_inputs(mock_inputs, hidden_states=hs)

    assert torch.is_tensor(result.last_hidden_states)
    assert torch.equal(result_kwargs["hidden_states"], result.last_hidden_states)


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_init_with_full_config(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test initialization with full configuration parameters."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(
        rank=1,
        local_rank=0,
        world_size=4,
        npu_device_id=0,
        model_id="deepseek_v3.2",
        trust_remote_code=False,
        load_tokenizer=True,
        max_batch_size=16,
        tp=2,
        dp=1,
        cp=16,
        moe_tp=1,
        moe_ep=32,
        role="prefill",
        max_seq_len=65536,
        block_size=128,
        plugin_params='{"plugin_type":"mtp","num_speculative_tokens":2}',
        num_speculative_tokens=2,
    )
    assert wrapper.rank == 1
    assert wrapper.is_multimodal is False
    assert wrapper.model_runner is not None

    kwargs = mock_model_runner_class.call_args[1]
    assert kwargs["cp"] == 16
    assert kwargs["moe_ep"] == 32
    assert kwargs["max_seq_len"] == 65536
    assert kwargs["role"] == "prefill"
    assert "mtp" in kwargs["plugin_params"]


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_forward_with_q_lens(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test forward with q_lens kwarg."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = MagicMock()
    mock_inputs.input_ids = [1, 2, 3]
    mock_inputs.position_ids = [0, 1, 2]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    mock_model_runner_exp.forward.return_value = {"logits": torch.tensor([0.5])}
    result = wrapper.forward(mock_inputs, npu_cache="cache", q_lens=[3])

    assert result == {"logits": torch.tensor([0.5])}


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_forward_with_empty_input_ids(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test forward with empty input_ids."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = MagicMock()
    mock_inputs.input_ids = []
    mock_inputs.position_ids = []
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    mock_model_runner_exp.forward.return_value = {"logits": torch.tensor([])}
    result = wrapper.forward(mock_inputs, npu_cache="cache")

    assert torch.equal(result["logits"], torch.tensor([]))


@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.ModelRunnerExp")
@patch("mindie_llm.modeling.model_wrapper.aclgraph.aclgraph_model_wrapper_exp.get_parallel_info_manager")
def test_prepare_model_inputs_with_all_kwargs(
    mock_get_parallel_info, mock_model_runner_class, mock_model_runner_exp, mock_parallel_info_manager
):
    """Test prepare_model_inputs with all kwargs combined."""
    mock_model_runner_class.return_value = mock_model_runner_exp
    mock_get_parallel_info.return_value = mock_parallel_info_manager

    wrapper = AclGraphModelWrapperExp(rank=0, local_rank=0, world_size=2, npu_device_id=0, model_id="test")

    mock_inputs = Mock()
    mock_inputs.input_ids = [1, 2]
    mock_inputs.position_ids = [0, 1]
    mock_inputs.block_tables = []
    mock_inputs.input_lengths = None

    sub_inputs = Mock()
    sub_inputs.input_ids = [3, 4]
    sub_inputs.position_ids = [2, 3]
    sub_inputs.slots = [5, 6]
    sub_inputs.context_length = [2]
    sub_inputs.prefill_head_indices = [1]
    sub_inputs.block_tables = [[0, 1]]

    hs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    _, result_kwargs = wrapper.prepare_model_inputs(
        mock_inputs,
        q_lens=[2],
        mtp_logits_gather_indices=torch.tensor([0]),
        shard_effective_token_indices=[0],
        lm_head_local_dp=[0],
        sub_model_inputs=sub_inputs,
        hidden_states=hs,
    )

    assert "q_lens" in result_kwargs
    assert "mtp_logits_gather_indices" in result_kwargs
    assert "shard_effective_token_indices" in result_kwargs
    assert "lm_head_local_dp" in result_kwargs
    assert "sub_model_inputs" in result_kwargs
    assert "hidden_states" in result_kwargs
