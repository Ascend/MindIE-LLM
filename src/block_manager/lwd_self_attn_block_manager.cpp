/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <numeric>
#include "self_attn_block_manager.h"
#include "lwd_self_attn_block_manager.h"

#include "cpu_npu_block_allocator.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "math_utils.h"

namespace mindie_llm {
LwdSelfAttnBlockManager::LwdSelfAttnBlockManager(const BlockManagerConfig &config, size_t localDPRank)
    : SelfAttnBlockManager(config, localDPRank)
{}

void LwdSelfAttnBlockManager::LwdInitCloudBlockManager(const BlockManagerConfig &lwdCloudConfig, size_t localDPRank)
{
    lwdCloudBlockManager_ = std::make_shared<SelfAttnBlockManager>(lwdCloudConfig, localDPRank);
}

void LwdSelfAttnBlockManager::LwdGetCloudRankedBlockIds(SequenceId seqId,
    std::vector<std::vector<BlockId>> &rankedBlockIds) const
{
    lwdCloudBlockManager_->GetRankedBlockIds(seqId, rankedBlockIds);
}

void LwdSelfAttnBlockManager::AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime)
{
    this->SelfAttnBlockManager::AccessAllblocksInSeq(seq, accessTime);
    lwdCloudBlockManager_->AccessAllblocksInSeq(seq, accessTime);
}

void LwdSelfAttnBlockManager::Free(SequenceId seqId)
{
    this->SelfAttnBlockManager::Free(seqId);
    lwdCloudBlockManager_->Free(seqId);
}

AllocStatus LwdSelfAttnBlockManager::CanAllocate(const SequenceGroupSPtr &seqGroup) const
{
    AllocStatus lwdEdgeCanAllocate = this->SelfAttnBlockManager::CanAllocate(seqGroup);
    AllocStatus lwdCloudCanAllocate = lwdCloudBlockManager_->CanAllocate(seqGroup);
    if (lwdEdgeCanAllocate == AllocStatus::NEVER || lwdCloudCanAllocate == AllocStatus::NEVER) {
        return AllocStatus::NEVER;
    } else if (lwdEdgeCanAllocate == AllocStatus::LATER || lwdCloudCanAllocate == AllocStatus::LATER) {
        return AllocStatus::LATER;
    }
    return AllocStatus::OK;
}

bool LwdSelfAttnBlockManager::Allocate(const SequenceGroupSPtr &seqGroup)
{
    bool lwdEdgeAllocateSucc = this->SelfAttnBlockManager::Allocate(seqGroup);
    bool lwdCloudAllocateSucc = lwdCloudBlockManager_->Allocate(seqGroup);
    return lwdEdgeAllocateSucc && lwdCloudAllocateSucc;
}

bool LwdSelfAttnBlockManager::CanAppendSlot(const SequenceGroupSPtr &seqGroup) const
{
    bool lwdEdgeCanAppendSlot = this->SelfAttnBlockManager::CanAppendSlot(seqGroup);
    bool lwdCloudCanAppendSlot = lwdCloudBlockManager_->CanAppendSlot(seqGroup);
    return lwdEdgeCanAppendSlot && lwdCloudCanAppendSlot;
}

bool LwdSelfAttnBlockManager::CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const
{
    bool lwdEdgeCanAppendSlotNew = this->SelfAttnBlockManager::CanAppendSlotNew(seqGroup);
    bool lwdCloudCanAppendSlotNew = lwdCloudBlockManager_->CanAppendSlotNew(seqGroup);
    return lwdEdgeCanAppendSlotNew && lwdCloudCanAppendSlotNew;
}

void LwdSelfAttnBlockManager::AppendSlotNew(const SequenceGroupSPtr &seqGroup)
{
    this->SelfAttnBlockManager::AppendSlotNew(seqGroup);
    lwdCloudBlockManager_->AppendSlotNew(seqGroup);
}

size_t LwdSelfAttnBlockManager::LwdGetCloudLatestAppendedRankId(SequenceId seqId) const
{
    return lwdCloudBlockManager_->GetLatestAppendedRankId(seqId);
}

size_t LwdSelfAttnBlockManager::LwdGetCloudAppendedBlockRankId(SequenceId seqId) const
{
    return lwdCloudBlockManager_->GetAppendedBlockRankId(seqId);
}

std::vector<size_t> LwdSelfAttnBlockManager::LwdGetCloudTokenCountPerRank(SequenceId seqId) const
{
    return lwdCloudBlockManager_->GetTokenCountPerRank(seqId);
}

} // namespace mindie_llm
