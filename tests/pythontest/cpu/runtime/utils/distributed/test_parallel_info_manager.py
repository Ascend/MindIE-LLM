# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.distributed as dist

from mindie_llm.runtime.utils.distributed.parallel_info_manager import (
    ParallelInfoManager,
    ParallelType,
    ParallelInfo,
    DEFAULT_BUFFER_SIZE,
    HCCL_BACKEND,
    GLOO_BACKEND
)


@pytest.fixture(autouse=True)
def mock_torch_distributed():
    """Mock torch.distributed to avoid real initialization."""
    with patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.get_world_size", return_value=8), \
         patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.get_rank", return_value=3), \
         patch("mindie_llm.runtime.utils.distributed.parallel_info_manager.dist.new_group") as mock_new_group:
        yield mock_new_group


@pytest.fixture
def mock_server_config():
    return {
        "tp": 2,
        "dp": 4,
        "cp": 2,
        "moe_tp": 2,
        "moe_ep": 4,  # Must satisfy moe_tp * moe_ep = world_size (2*4=8)
        "sp": 2
    }


def test_get_current_group_id():
    rank_per_group = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 4) == 1
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 0) == 0
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 8) == 2
    assert ParallelInfoManager._get_current_group_id(rank_per_group, 9) is None


def test_init_tp_parallel_info_world_size(mock_torch_distributed):
    """Test TP with group_size = world_size (single group)."""
    manager = ParallelInfoManager(local_rank=0, llm_config=None, server_config={})
    info = manager._init_tp_parallel_info(group_size=8)

    assert info.group_size == 8
    assert info.num_group == 1
    assert info.rank_per_group == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert info.current_group_id == 0
    assert info.rank == 3  # Current rank is 3
    assert info.is_reusable is True
    assert info._pg_factory is not None


def test_init_tp_parallel_info_group_size_2(mock_torch_distributed):
    """Test TP with group_size=2, world_size=8."""
    manager = ParallelInfoManager(local_rank=0, llm_config=None, server_config={})
    info = manager._init_tp_parallel_info(group_size=2)

    assert info.group_size == 2
    assert info.num_group == 4
    assert info.rank_per_group == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert info.current_group_id == 1  # Rank 3 is in group [2,3] → index 1
    assert info.rank == 1  # Local rank within [2,3]


def test_init_dp_parallel_info_group_size_4(mock_torch_distributed):
    """Test DP with group_size=4, world_size=8 → num_group=2, strided groups."""
    manager = ParallelInfoManager(local_rank=0, llm_config=None, server_config={})
    info = manager._init_dp_parallel_info(group_size=4)

    # With world_size=8, group_size=4 → num_group = 8/4 = 2
    # Groups: [0,2,4,6], [1,3,5,7]
    assert info.group_size == 4
    assert info.num_group == 2
    assert info.rank_per_group == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert info.current_group_id == 1  
    assert info.rank == 1  


def test_parallel_info_manager_initialization(mock_torch_distributed, mock_server_config):
    """Test full initialization with server config."""
    llm_config = None
    local_rank = 0

    manager = ParallelInfoManager(local_rank, llm_config, mock_server_config)

    # Check world
    assert manager.world.group_size == 8
    assert manager.world.num_group == 1

    # Check attn_tp (tp=2)
    tp = manager.attn_tp
    assert tp.group_size == 2
    assert tp.num_group == 4
    assert tp.rank_per_group == [[0,1],[2,3],[4,5],[6,7]]

    # Check attn_dp (dp=4 → 2 groups of 4)
    dp = manager.attn_dp
    assert dp.group_size == 4
    assert dp.num_group == 2
    assert dp.rank_per_group == [[0,2,4,6], [1,3,5,7]]

    # Check MoE validation passes (2*4=8)
    assert manager.moe_tp.group_size == 2
    assert manager.moe_ep.group_size == 4

    # Check aliases
    assert manager.word_embed_tp is manager.attn_tp
    assert manager.lm_head_tp is manager.mlp_tp

    # Check buffer sizes for MoE
    assert manager.moe_tp.buffer_size == 64  # default when llm_config is None
    assert manager.moe_ep.buffer_size == 512


def test_get_method_valid_types(mock_torch_distributed, mock_server_config):
    """Test get() returns correct ParallelInfo for valid types."""
    manager = ParallelInfoManager(0, None, mock_server_config)

    # Test a few
    assert manager.get(ParallelType.ATTN_TP) is manager.attn_tp
    assert manager.get(ParallelType.ATTN_DP) is manager.attn_dp
    assert manager.get(ParallelType.MOE_TP) is manager.moe_tp
    assert manager.get(ParallelType.WORLD) is manager.world


def test_get_method_invalid_type(mock_torch_distributed, mock_server_config):
    """Test get() raises KeyError for unsupported type."""
    manager = ParallelInfoManager(0, None, mock_server_config)

    with pytest.raises(KeyError, match="Unsupported ParallelType"):
        manager.get("invalid_type")


def test_deprecated_has_methods(mock_torch_distributed, mock_server_config):
    """Ensure deprecated has_* methods still work (delegate to .get().is_enabled())."""
    manager = ParallelInfoManager(0, None, mock_server_config)

    # tp=2 → enabled
    assert manager.has_attn_tp() is True
    # dp=4 → enabled
    assert manager.has_dp() is True
    # sp=2 → enabled
    assert manager.has_attn_inner_sp() is True
    # pp not implemented → False
    assert manager.has_pp() is False


def test_moe_validation_failure(mock_torch_distributed):
    """Test MoE validation fails when tp*ep != world_size."""
    invalid_config = {"moe_tp": 2, "moe_ep": 2}  # 2*2=4 ≠ 8
    
    with pytest.raises(ValueError, match="MoE parallel strategy process number mismatch"):
        ParallelInfoManager(0, None, invalid_config)


def test_parallel_info_lazy_initialization(mock_torch_distributed):
    """Test ParallelInfo lazily initializes process groups via factory."""
    manager = ParallelInfoManager(0, None, {})
    
    # Create ParallelInfo instance
    info = manager._init_tp_parallel_info(group_size=2)
    
    # Process groups not created yet
    assert info._process_group is None
    assert info._cpu_process_group is None
    
    # Mock _get_or_create_process_group to capture calls
    with patch.object(manager, '_get_or_create_process_group') as mock_get_pg:
        mock_hccl_pg = MagicMock()
        mock_gloo_pg = MagicMock()
        mock_get_pg.side_effect = lambda ranks, backend, **kwargs: mock_hccl_pg if backend == HCCL_BACKEND else mock_gloo_pg
        
        # Access NPU process group
        pg = info.process_group
        assert pg == mock_hccl_pg
        mock_get_pg.assert_any_call(
            ranks=[2, 3],  # Group containing rank 3
            backend=HCCL_BACKEND,
            hccl_buffer_size=DEFAULT_BUFFER_SIZE,
            is_reusable=True
        )
        
        # Access CPU process group
        cpu_pg = info.cpu_process_group
        assert cpu_pg == mock_gloo_pg
        mock_get_pg.assert_any_call(
            ranks=[2, 3],
            backend=GLOO_BACKEND,
            hccl_buffer_size=None,
            is_reusable=True
        )


def test_has_lm_head_local_tp(mock_torch_distributed, mock_server_config):
    """Test has_lm_head_local_tp logic."""
    manager = ParallelInfoManager(0, None, mock_server_config)
    
    # lm_head_tp.group_size = 2, world_size = 8 → 2 < 8 → True
    assert manager.has_lm_head_local_tp() is True
    
    # Test with full world size
    full_tp_config = {"tp": 8}
    manager_full = ParallelInfoManager(0, None, full_tp_config)
    assert manager_full.has_lm_head_local_tp() is False


def test_non_reusable_process_group(mock_torch_distributed):
    """Test non-reusable process group creation."""
    # Must provide both moe_tp and moe_ep so that moe_tp * moe_ep = world_size (8)
    config = {"moe_tp": 2, "moe_ep": 4}
    manager = ParallelInfoManager(0, None, config)
    
    # moe_ep_mc2 is created with is_reusable=False
    info = manager.moe_ep_mc2
    assert info.is_reusable is False
    
    with patch.object(manager, '_get_or_create_process_group') as mock_get_pg:
        mock_pg = MagicMock()
        mock_get_pg.return_value = mock_pg
        
        # Access process group
        pg = info.process_group
        
        # Verify called with is_reusable=False
        # Rank 3 belongs to moe_ep group: strided with group_size=4 → [1,3,5,7]
        mock_get_pg.assert_called_with(
            ranks=[1, 3, 5, 7],
            backend=HCCL_BACKEND,
            hccl_buffer_size=512,
            is_reusable=False
        )