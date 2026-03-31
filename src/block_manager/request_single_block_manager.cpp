/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "request_single_block_manager.h"

#include <stdexcept>

namespace mindie_llm {

RequestSingleBlockManager::RequestSingleBlockManager(const BlockManagerConfig &config, size_t localDPRank)
    : blockSize_(config.cacheBlockSize),
      cpuBlockNum_(config.cpuBlockNum),
      npuBlockNum_(config.npuBlockNum),
      reservedBlockNum_(config.reservedBlockNum),
      enableCaching_(config.enableCaching),
      rankSize_(config.rankSize),
      hostSize_(config.hostSize),
      localDPRank_(localDPRank)
{
    if (rankSize_ == 0 || hostSize_ == 0) {
        throw std::invalid_argument("The rank and host size must be greater than 0");
    }
    if (npuBlockNum_ == 0) {
        throw std::invalid_argument("npuBlockNum must be > 0");
    }
    if (reservedBlockNum_ > npuBlockNum_) {
        throw std::invalid_argument("reservedBlockNum must be <= npuBlockNum");
    }

    AllocatorConfig allocatorConfig;
    allocatorConfig.allocatorType = enableCaching_ ? BlockAllocatorType::PREFIXCACHING : BlockAllocatorType::HASHLESS;
    allocatorConfig.numCpuBlocks = cpuBlockNum_;
    allocatorConfig.numNpuBlocks = npuBlockNum_ - reservedBlockNum_;
    allocatorConfig.blockSize = (blockSize_ == 0 ? 1 : blockSize_);
    allocatorConfig.rankSize = rankSize_;
    allocatorConfig.hostSize = hostSize_;
    blockAllocator_ = std::make_shared<CpuNpuBlockAllocator>(allocatorConfig);
}

const RequestId* RequestSingleBlockManager::GetRequestIdBySeqId_(SequenceId seqId) const
{
    auto it = seqId2RequestId_.find(seqId);
    if (it == seqId2RequestId_.end()) {
        return nullptr;
    }
    return &it->second;
}

RequestSingleBlockManager::RequestEntry* RequestSingleBlockManager::GetEntryByRequestId_(const RequestId &rid)
{
    auto it = requestEntries_.find(rid);
    if (it == requestEntries_.end()) {
        return nullptr;
    }
    return &it->second;
}

const RequestSingleBlockManager::RequestEntry* RequestSingleBlockManager::GetEntryByRequestId_(
    const RequestId &rid) const
{
    auto it = requestEntries_.find(rid);
    if (it == requestEntries_.end()) {
        return nullptr;
    }
    return &it->second;
}

AllocStatus RequestSingleBlockManager::CanAllocate(const SequenceGroupSPtr &seqGroup) const
{
    if (!seqGroup) {
        return AllocStatus::NEVER;
    }
    if (seqGroup->GetFirstSequence(SequenceStatus::WAITING).empty()) {
        // No waiting sequences to allocate for.
        return AllocStatus::NEVER;
    }

    // If this request already owns a block, allocating more sequences is always OK.
    if (GetEntryByRequestId_(seqGroup->requestId) != nullptr) {
        return AllocStatus::OK;
    }

    // Each request needs exactly 1 NPU block.
    if ((npuBlockNum_ - reservedBlockNum_) < 1) {
        return AllocStatus::NEVER;
    }
    // We always allocate from rank0 and let other ranks reuse rank0's block table.
    return (blockAllocator_->GetNumFreeBlock(DeviceType::NPU, 0) >= 1) ? AllocStatus::OK : AllocStatus::LATER;
}

bool RequestSingleBlockManager::Allocate(const SequenceGroupSPtr &seqGroup)
{
    if (!seqGroup) {
        return false;
    }

    const auto waitingSeqs = seqGroup->GetFirstSequence(SequenceStatus::WAITING);
    if (waitingSeqs.empty()) {
        return false;
    }

    const RequestId &rid = seqGroup->requestId;
    auto &entry = requestEntries_[rid];

    if (!entry.block) {
        // First time for this request: allocate a single mutable block with empty tokens.
        std::vector<TokenId> emptyTokens;
        entry.block = blockAllocator_->AllocateMutableBlock(DeviceType::NPU, emptyTokens, nullptr, 0, 0);
        entry.refCount = 0;
    }

    // Bind all waiting sequences in this group to the request's single block.
    for (const auto &seq : waitingSeqs) {
        if (!seq) continue;
        const SequenceId sid = seq->seqId_;
        if (seqId2RequestId_.find(sid) != seqId2RequestId_.end()) {
            // Already allocated for this seqId.
            continue;
        }
        seqId2RequestId_[sid] = rid;
        entry.refCount++;
    }

    return true;
}

bool RequestSingleBlockManager::CanAppendSlot(const SequenceGroupSPtr &seqGroup) const
{
    if (!seqGroup) {
        return false;
    }
    // Append is a no-op for request-fixed manager; as long as block exists, it's "appendable".
    return (GetEntryByRequestId_(seqGroup->requestId) != nullptr);
}

std::vector<std::pair<BlockId, BlockId>> RequestSingleBlockManager::AppendSlot(const SequenceSPtr &seq)
{
    (void)seq;
    // No-op: the request always reuses the same blockId; no COW mapping.
    return {};
}

bool RequestSingleBlockManager::CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const
{
    return CanAppendSlot(seqGroup);
}

void RequestSingleBlockManager::AppendSlotNew(const SequenceGroupSPtr &seqGroup)
{
    (void)seqGroup;
}

void RequestSingleBlockManager::AppendTokenToLatestRank(SequenceId seqId, const std::vector<TokenId>& tokens)
{
    (void)seqId;
    (void)tokens;
}

void RequestSingleBlockManager::Fork(SequenceSPtr &parentSeq, SequenceSPtr &childSeq)
{
    if (!parentSeq || !childSeq) {
        throw std::invalid_argument("parentSeq/childSeq cannot be null");
    }
    const RequestId *rid = GetRequestIdBySeqId_(parentSeq->seqId_);
    if (!rid) {
        throw std::runtime_error("parentSeq has no bound requestId in RequestSingleBlockManager::Fork");
    }
    auto *entry = GetEntryByRequestId_(*rid);
    if (!entry || !entry->block) {
        throw std::runtime_error("request entry missing in RequestSingleBlockManager::Fork");
    }
    seqId2RequestId_[childSeq->seqId_] = *rid;
    entry->refCount++;
}

bool RequestSingleBlockManager::CanSwapOut(const SequenceGroupSPtr &seqGroup)
{
    (void)seqGroup;
    // Not supported.
    return false;
}

std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> RequestSingleBlockManager::SwapOut(
    const SequenceGroupSPtr &seqGroup)
{
    (void)seqGroup;
    // Not supported.
    return {};
}

AllocStatus RequestSingleBlockManager::CanSwapIn(const SequenceGroupSPtr &seqGroup, size_t numLookheadSlots)
{
    (void)numLookheadSlots;
    (void)seqGroup;
    // Not supported.
    return AllocStatus::NEVER;
}

std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> RequestSingleBlockManager::SwapIn(
    const SequenceGroupSPtr &seqGroup)
{
    (void)seqGroup;
    // Not supported.
    return {};
}

void RequestSingleBlockManager::Free(SequenceId seqId)
{
    const RequestId *rid = GetRequestIdBySeqId_(seqId);
    if (!rid) {
        return;
    }
    auto itEntry = requestEntries_.find(*rid);
    if (itEntry == requestEntries_.end()) {
        seqId2RequestId_.erase(seqId);
        return;
    }

    RequestEntry &entry = itEntry->second;
    if (entry.refCount > 0) {
        entry.refCount--;
    }
    seqId2RequestId_.erase(seqId);

    if (entry.refCount == 0) {
        if (entry.block) {
            blockAllocator_->Free(entry.block);
        }
        requestEntries_.erase(itEntry);
    }
}

std::vector<BlockIds> RequestSingleBlockManager::GetBlockIds(SequenceId seqId) const
{
    const RequestId *rid = GetRequestIdBySeqId_(seqId);
    if (!rid) {
        return {};
    }
    const auto *entry = GetEntryByRequestId_(*rid);
    if (!entry || !entry->block) {
        return {};
    }
    return {{entry->block->GetBlockId()}};
}

void RequestSingleBlockManager::GetRankedBlockIds(SequenceId seqId,
    std::vector<RankedBlockId> &rankedBlockIds) const
{
    rankedBlockIds.clear();
    const auto allIds = GetBlockIds(seqId);
    if (allIds.empty() || allIds[0].empty()) {
        return;
    }
    const auto &ids = allIds[0];
    // Reuse rank0's single blockId for all ranks.
    rankedBlockIds.reserve(rankSize_);
    for (size_t r = 0; r < rankSize_; ++r) {
        rankedBlockIds.push_back(RankedBlockId{ids[0], r});
    }
}

void RequestSingleBlockManager::GetRankedBlockIds(SequenceId seqId,
    std::vector<std::vector<BlockId>> &rankedBlockIds) const
{
    rankedBlockIds.clear();
    rankedBlockIds.resize(rankSize_);
    const auto allIds = GetBlockIds(seqId);
    if (allIds.empty() || allIds[0].empty()) {
        return;
    }
    const auto &ids = allIds[0];
    // Reuse rank0's single blockId for all ranks.
    for (size_t r = 0; r < rankSize_; ++r) {
        rankedBlockIds[r].push_back(ids[0]);
    }
}

std::vector<std::vector<HashValue>> RequestSingleBlockManager::GetRankedHashValues(SequenceId seqId) const
{
    (void)seqId;
    return {};
}

std::vector<HashValue> RequestSingleBlockManager::GetSeqHashValues(SequenceId seqId) const
{
    (void)seqId;
    return {};
}

std::vector<size_t> RequestSingleBlockManager::GetTokenCountPerRank(SequenceId seqId) const
{
    (void)seqId;
    // Keep shape consistent with sp/cp callers.
    return std::vector<size_t>(rankSize_, 0);
}

size_t RequestSingleBlockManager::GetLatestAppendedRankId(SequenceId seqId) const
{
    (void)seqId;
    return 0;
}

size_t RequestSingleBlockManager::GetAppendedBlockRankId(SequenceId seqId) const
{
    (void)seqId;
    return 0;
}

bool RequestSingleBlockManager::IsAppendBlock(SequenceId seqId)
{
    (void)seqId;
    return false;
}

size_t RequestSingleBlockManager::GetNumFreeNpuBlocks() const
{
    // We only allocate from rank0 (others reuse rank0), so rank0 is the effective capacity.
    return blockAllocator_->GetNumFreeBlock(DeviceType::NPU, 0);
}

size_t RequestSingleBlockManager::GetNumFreeCpuBlocks() const
{
    return blockAllocator_->GetNumFreeBlock(DeviceType::CPU);
}

void RequestSingleBlockManager::AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime)
{
    (void)seq;
    (void)accessTime;
    // No prefix-cache tracking in this manager.
}

std::vector<BlockId> RequestSingleBlockManager::GetCommonComputedBlockIds(const std::vector<SequenceSPtr> &seqs)
{
    (void)seqs;
    return {};
}

std::vector<size_t> RequestSingleBlockManager::GetAllrankComputedBlockNum(const std::vector<SequenceSPtr> &seqs)
{
    (void)seqs;
    return {};
}

std::vector<BlockId> RequestSingleBlockManager::GetRemoteComputedBlockIds(const std::vector<SequenceSPtr> &seqs,
    size_t computedLens, uint32_t tpSize, std::string modelName)
{
    (void)seqs;
    (void)computedLens;
    (void)tpSize;
    (void)modelName;
    return {};
}

std::vector<size_t> RequestSingleBlockManager::GetAllRankRemoteComputedBlockIds(
    const std::vector<SequenceSPtr> &seqs, std::vector<size_t> &computedBlocksNum, std::string modelName)
{
    (void)seqs;
    (void)computedBlocksNum;
    (void)modelName;
    return {};
}

void RequestSingleBlockManager::MarkBlocksAsComputed()
{
    // No-op; keep allocator state unchanged.
}

float RequestSingleBlockManager::GetPrefixCacheHitRate() const
{
    return blockAllocator_->GetPrefixCacheHitRate();
}

bool RequestSingleBlockManager::ResetPrefixCache() const
{
    return blockAllocator_->ResetPrefixCache();
}

size_t RequestSingleBlockManager::GetNumCachedTokens(const SequenceSPtr &seq)
{
    (void)seq;
    return 0;
}

size_t RequestSingleBlockManager::GetSeqNumCachedTokens(const SequenceSPtr &seq)
{
    (void)seq;
    return 0;
}

void RequestSingleBlockManager::ReplaceTrailingPlaceHolder(const SequenceSPtr &seq, size_t trailingPlaceHolderNum,
    size_t replacedPlaceHolderNum)
{
    (void)seq;
    (void)trailingPlaceHolderNum;
    (void)replacedPlaceHolderNum;
}

void RequestSingleBlockManager::LwdInitCloudBlockManager(const BlockManagerConfig &lwdCloudConfig,
    size_t localDPRank)
{
    (void)lwdCloudConfig;
    (void)localDPRank;
}

void RequestSingleBlockManager::LwdGetCloudRankedBlockIds(SequenceId seqId,
    std::vector<std::vector<BlockId>> &rankedBlockIds) const
{
    (void)seqId;
    rankedBlockIds.clear();
}

size_t RequestSingleBlockManager::LwdGetCloudLatestAppendedRankId(SequenceId seqId) const
{
    (void)seqId;
    return 0;
}

size_t RequestSingleBlockManager::LwdGetCloudAppendedBlockRankId(SequenceId seqId) const
{
    (void)seqId;
    return 0;
}

std::vector<size_t> RequestSingleBlockManager::LwdGetCloudTokenCountPerRank(SequenceId seqId) const
{
    (void)seqId;
    return {};
}

} // namespace mindie_llm
