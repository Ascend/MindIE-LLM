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

#include "self_attn_block_manager.h"

#include <numeric>

#include "log.h"
#include "cpu_npu_block_allocator.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "math_utils.h"

namespace mindie_llm {
SelfAttnBlockManager::SelfAttnBlockManager(const BlockManagerConfig &config, size_t localDPRank)
    : blockSize_(config.cacheBlockSize), cpuBlockNum_(config.cpuBlockNum), npuBlockNum_(config.npuBlockNum),
      reservedBlockNum_(config.reservedBlockNum), speculativeSlots_(config.speculativeSlots),
      enableCaching_(config.enableCaching), localDPRank_(localDPRank), rankSize_(config.rankSize),
      hostSize_(config.hostSize), allocationMode_(config.allocationMode)
{
    if (reservedBlockNum_ > npuBlockNum_) {
        throw std::invalid_argument("The num of reserved block is larger than npu block num");
    }

    if (rankSize_ == 0 || hostSize_ == 0) {
        throw std::invalid_argument("The rank and host size must be greater than 0");
    }

    size_t allocatableNpuBlockNum = npuBlockNum_ - reservedBlockNum_;
    PROF(INFO, AddMetaInfo("allocatableBlockNum", allocatableNpuBlockNum));
    PROF(INFO, AddMetaInfo("npuBlockNum", npuBlockNum_));
    PROF(INFO, AddMetaInfo("reservedBlockNum", reservedBlockNum_));

    AllocatorConfig allocatorConfig;
    // 生效prefixcache，则初始化PrefixCacheBlockAllocator 否则初始化HashLessBlockAllocator
    allocatorConfig.allocatorType = enableCaching_ ? BlockAllocatorType::PREFIXCACHING : BlockAllocatorType::HASHLESS;
    allocatorConfig.numCpuBlocks = cpuBlockNum_;
    allocatorConfig.numNpuBlocks = allocatableNpuBlockNum;
    allocatorConfig.blockSize = blockSize_;
    allocatorConfig.rankSize = rankSize_;
    blockAllocator_ = std::make_shared<CpuNpuBlockAllocator>(allocatorConfig);

    seqsBlockComputedTracker_ = SeqsBlocksComputedTracker(blockAllocator_, blockSize_, enableCaching_, rankSize_);
    seqsLastAccessBlocksTracker_ = SeqsLastAccessBlocksTracker(blockAllocator_);

    if (config.enableKvPool) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        py::object memPoolCls_ = py::module_::import("mindie_llm.text_generator.mempool").attr("MemPool");
        py::object memPool_ = memPoolCls_.attr("create_pool")(config.cachePoolBackend, config.cachePoolConfigPath);
        memPoolInstance_ = std::make_shared<MemPool>(std::make_shared<py::object>(memPool_));
        PyGILState_Release(gstate);
    }

    MINDIE_LLM_LOG_INFO("SelfAttnBlockManager init success!");
}

BlockSpaceManagerSPtr BlockManagerFactory::CreateBlockSpaceManager(BlockManagerType type,
                                                                   const BlockManagerConfig &config, size_t localDPRank)
{
    switch (type) {
        case BlockManagerType::SELFATTNBLOCKMANGER: return std::make_shared<SelfAttnBlockManager>(config, localDPRank);
        default: throw std::invalid_argument("Invalid block manager type");
    }
}

size_t SelfAttnBlockManager::GetNumRequiredBlocks(size_t seqLen, size_t blockSize) const
{
    if (blockSize == 0) {
        throw std::runtime_error("the blockSize should not be zero");
    }
    size_t num = (seqLen + blockSize - 1) / blockSize;
    if (rankSize_ > 1) {
        num += 1;
    }
    return num;
}

AllocStatus SelfAttnBlockManager::CanAllocate(const SequenceGroupSPtr &seqGroup) const
{
    std::vector<SequenceSPtr> waitingSeqs = seqGroup->GetFirstSequence(SequenceStatus::WAITING);
    if (waitingSeqs.empty()) {
        return AllocStatus::NEVER;
    }
    const std::vector<TokenId> &tokenIds = waitingSeqs.at(0)->GetTokenIds();
    size_t numRequiredBlocks = GetNumRequiredBlocks(tokenIds.size(), blockSize_);
    // 当有多个rank时，保守计算，以负载最重的rank的剩余block * ranksize 作为free block数量
    size_t numFreeNpuBlocks = blockAllocator_->GetNumFreeBlock(DeviceType::NPU);

    if (reservedBlockNum_ > npuBlockNum_) {
        throw std::runtime_error("The num of reserved block is larger than npu block num");
    }

    if ((npuBlockNum_ * rankSize_ - reservedBlockNum_) < numRequiredBlocks) {
        return AllocStatus::NEVER;
    } else if (numFreeNpuBlocks >= numRequiredBlocks) {
        return AllocStatus::OK;
    }
    return AllocStatus::LATER;
}

bool SelfAttnBlockManager::Allocate(const SequenceGroupSPtr &seqGroup)
{
    std::vector<SequenceSPtr> waitingSeqs = seqGroup->GetFirstSequence(SequenceStatus::WAITING);

    if (waitingSeqs.empty()) {
        return false;
    }
    // 在allocate完成前 sequence对应的blocktable不应该已经存在
    for (const auto &sequence : waitingSeqs) {
        SequenceId sequenceId = sequence->seqId_;
        auto it = seqId2BlockTable_.find(sequenceId);
        if (it != seqId2BlockTable_.end()) {
            return false;
        }
    }

    // 在一个group中，所有sequence都是有相同的prompt
    SequenceSPtr sequence = waitingSeqs.at(0);
    BlockTable blockTable = BlockTable(blockSize_, blockAllocator_, rankSize_);
    const std::vector<TokenId> &tokenIds = sequence->GetTokenIds();
    if (tokenIds.empty()) {
        return false;
    }

    if (rankSize_ == 1) {
        blockTable.Allocate(tokenIds, DeviceType::NPU, sequence->GetExtraHash());
    } else {
        // in sp cp scenario,  allocate from the smallest rank
        blockTable.AllocateSmallRankFirst(tokenIds, DeviceType::NPU, sequence->GetExtraHash());
    }

    seqId2BlockTable_[sequence->seqId_] = blockTable;

    seqsLastAccessBlocksTracker_.AddSeq(sequence->seqId_);
    for (auto it = waitingSeqs.begin() + 1; it != waitingSeqs.end(); ++it) {
        // only for multi-sequence SequenceGroup
        SequenceId seqId = (*it)->seqId_;
        seqId2BlockTable_[seqId] = blockTable.Fork();
        seqsLastAccessBlocksTracker_.AddSeq(seqId);
    }
    return true;
}

/// 确定NPU的KV缓存中是否有足够的空间为指定sequence group生成序列，
/// 约定每个被Append slot的block都需要新分配。如果append的block数量少于空闲的block数量，则可以Append多个slot。
bool SelfAttnBlockManager::CanAppendSlot(const SequenceGroupSPtr &seqGroup) const
{
    size_t numRelatedBlocks = 0;
    std::vector<SequenceSPtr> runningSeqs = seqGroup->GetSequences(SequenceStatus::RUNNING);
    for (auto &sequence : runningSeqs) {
        SequenceId seqId = sequence->seqId_;
        auto it = seqId2BlockTable_.find(seqId);
        if (it == seqId2BlockTable_.end()) {
            return false;
        }
        const BlockTable &blockTable = it->second;
        const std::vector<TokenId> &tokenIds = sequence->GetTokenIds();
        size_t tokenIdSize = blockTable.GetNewGenTokenIds(tokenIds).size();
        numRelatedBlocks += blockTable.GetNumRelatedBlocks(tokenIdSize, speculativeSlots_);
    }
    size_t numFreeNpuBlocks = blockAllocator_->GetNumFreeBlock(DeviceType::NPU);
    return numRelatedBlocks <= numFreeNpuBlocks;
}

std::vector<std::pair<BlockId, BlockId>> SelfAttnBlockManager::AppendSlot(const SequenceSPtr &sequence)
{
    if (rankSize_ > 1) {
        throw std::runtime_error("throw not supported exception.");
    }
    SequenceId seqId = sequence->seqId_;
    BlockTable &blockTable = seqId2BlockTable_.at(seqId);
    const std::vector<TokenId> &tokenIds = sequence->GetTokenIds();
    blockTable.AppendTokenIds(blockTable.GetNewGenTokenIds(tokenIds), sequence->GetExtraHash(), speculativeSlots_);
    std::vector<std::pair<BlockId, BlockId>> block2Copy = blockAllocator_->ClearCopyOnWrites();
    return block2Copy;
}

// CanAppendSlotNew和AppendSlotNew为sp专用接口，避免对现有流程影响
bool SelfAttnBlockManager::CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const
{
    std::vector<SequenceSPtr> runningSeqs = seqGroup->GetFirstSequence(SequenceStatus::RUNNING);
    if (runningSeqs.empty()) {
        return false;
    }
    const SequenceSPtr sequence = runningSeqs.at(0);
    const BlockTable &blockTable = seqId2BlockTable_.at(sequence->seqId_);
    const std::vector<TokenId> &tokenIds = sequence->GetTokenIds();
    std::vector<size_t> numFullSlotsPerRank = GetTokenCountPerRank(sequence->seqId_);
    size_t fullSlotsNum = std::accumulate(numFullSlotsPerRank.begin(), numFullSlotsPerRank.end(), size_t(0));
    std::vector<TokenId> appendTokenIds(tokenIds.begin() + fullSlotsNum, tokenIds.end());
    return blockTable.CanAppendNewTokens(appendTokenIds, speculativeSlots_);
}

void SelfAttnBlockManager::AppendSlotNew(const SequenceGroupSPtr &seqGroup)
{
    std::vector<SequenceSPtr> runningSeqs = seqGroup->GetFirstSequence(SequenceStatus::RUNNING);
    if (runningSeqs.empty()) {
        return;
    }

    const SequenceSPtr sequence = runningSeqs.at(0);
    BlockTable &blockTable = seqId2BlockTable_.at(sequence->seqId_);
    const std::vector<TokenId> &tokenIds = sequence->GetTokenIds();
    std::vector<size_t> numFullSlotsPerRank = GetTokenCountPerRank(sequence->seqId_);
    size_t fullSlotsNum = std::accumulate(numFullSlotsPerRank.begin(), numFullSlotsPerRank.end(), size_t(0));
    std::vector<TokenId> appendTokenIds(tokenIds.begin() + fullSlotsNum, tokenIds.end());

    blockTable.AppendNewTokens(appendTokenIds, sequence->GetExtraHash(), speculativeSlots_);
}

void SelfAttnBlockManager::AppendTokenToLatestRank(SequenceId seqId, const std::vector<TokenId> &tokens)
{
    BlockTable &blockTable = seqId2BlockTable_[seqId];

    blockTable.AppendToSpRank(blockTable.GetLatestAppendedRankId(), tokens);
}

bool SelfAttnBlockManager::IsAppendBlock(SequenceId seqId)
{
    BlockTable &blockTable = seqId2BlockTable_[seqId];
    return blockTable.IsAppendBlock();
}

void SelfAttnBlockManager::Free(SequenceId seqId)
{
    auto it = seqId2BlockTable_.find(seqId);
    if (it == seqId2BlockTable_.end()) {
        return; // Already freed or haven't been scheduled yet.
    }

    // 更新序列块最新访问时间
    BlockTable &blockTable = it->second;

    std::vector<std::vector<BlockId>> rankedBlockIds;
    GetRankedBlockIds(seqId, rankedBlockIds);
    seqsLastAccessBlocksTracker_.UpdateSeqBlocksLastAccess(seqId, rankedBlockIds);

    // 释放序列块的Track
    seqsLastAccessBlocksTracker_.RemoveSeq(seqId);
    seqsBlockComputedTracker_.RemoveSeq(seqId);

    // 释放block并从seqId2BlockTable_中去除对应blocktable
    blockTable.Free();
    seqId2BlockTable_.erase(it);
}

std::vector<BlockId> SelfAttnBlockManager::GetBlockIds(SequenceId seqId) const
{
    return seqId2BlockTable_.at(seqId).GetBlockIds();
}

void SelfAttnBlockManager::GetRankedBlockIds(SequenceId seqId, std::vector<RankedBlockId> &rankedBlockIds) const
{
    rankedBlockIds.clear();
    std::vector<BlockObjSPtr> blocks = seqId2BlockTable_.at(seqId).GetBlockObjs();
    for (const auto &block : blocks) {
        RankedBlockId rankedBlockId = {block->GetBlockId(), block->GetRankIdx()};
        rankedBlockIds.push_back(rankedBlockId);
    }
}

// rankedBlockIds 第一维度是rank 第二维度是block id
void SelfAttnBlockManager::GetRankedBlockIds(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds) const
{
    rankedBlockIds.clear();
    rankedBlockIds.resize(rankSize_);

    std::vector<BlockObjSPtr> blocks = seqId2BlockTable_.at(seqId).GetBlockObjs();
    for (const auto &block : blocks) {
        size_t rankIdx = block->GetRankIdx();
        BlockId blockId = block->GetBlockId();
        rankedBlockIds[rankIdx].push_back(blockId);
    }
}

// rankedHashValues 第一维度是rank 第二维度是block hashvalue
std::vector<std::vector<HashValue>> SelfAttnBlockManager::GetRankedHashValues(SequenceId seqId) const
{
    std::vector<std::vector<HashValue>> rankedHashValues;
    rankedHashValues.resize(rankSize_);
    std::vector<BlockObjSPtr> blocks = seqId2BlockTable_.at(seqId).GetBlockObjs();
    for (const auto &block : blocks) {
        size_t rankIdx = block->GetRankIdx();
        HashValue hashValue = block->GetHashValue();
        rankedHashValues[rankIdx].push_back(hashValue);
    }
    return rankedHashValues;
}

std::vector<HashValue> SelfAttnBlockManager::GetSeqHashValues(SequenceId seqId) const
{
    std::vector<HashValue> seqHashValues;
    std::vector<BlockObjSPtr> blocks = seqId2BlockTable_.at(seqId).GetBlockObjs();
    for (const auto &block : blocks) {
        HashValue hashValue = block->GetHashValue();
        seqHashValues.push_back(hashValue);
    }
    return seqHashValues;
}

// rankedBlockIds 第一维度是rank 第二维度是block 分配顺序
std::vector<size_t> SelfAttnBlockManager::GetPrefixBlockOrder(SequenceId seqId,
    std::vector<size_t> &computedBlocksNum) const
{
    std::vector<size_t> prefixBlockOrder;
    std::vector<std::vector<size_t>> rankedPrefixBlockOrder;
    seqId2BlockTable_.at(seqId).GetRankedPrefixBlockOrder(rankedPrefixBlockOrder);

    for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        prefixBlockOrder.insert(prefixBlockOrder.end(),
            rankedPrefixBlockOrder[rankIdx].begin(),
            rankedPrefixBlockOrder[rankIdx].begin() + computedBlocksNum[rankIdx]);
    }
    return prefixBlockOrder;
}

void SelfAttnBlockManager::AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime)
{
    if (enableCaching_) {
        seqsLastAccessBlocksTracker_.UpdateSeqLastAccess(seq->seqId_, accessTime);
    }
}

void SelfAttnBlockManager::MarkBlocksAsComputed() { blockAllocator_->MarkBlocksAsComputed(); }

std::vector<size_t> SelfAttnBlockManager::GetAllrankComputedBlockNum(const std::vector<SequenceSPtr> &seqs)
{
    std::vector<std::vector<std::vector<BlockId>>> rankedComputedSeqBlockIds;  /// rank_size, seq_num, block_num
    for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        std::vector<std::vector<BlockId>> computedSeqBlockIds;
        for (const auto &seq : seqs) {
            std::vector<std::vector<HashValue>> rankedHashValues = GetRankedHashValues(seq->seqId_);

            const std::vector<BlockId> &allBlocks = seqId2BlockTable_.at(seq->seqId_).GetBlockIds();
            size_t numCachedBlocks = seqsBlockComputedTracker_.GetCachedTokensNum(
                seq, rankIdx, rankedHashValues[rankIdx], seq->IsPrefill()) / blockSize_;
            std::vector<BlockId> computedBlockIds(allBlocks.begin(), allBlocks.begin() + numCachedBlocks);
            computedSeqBlockIds.push_back(computedBlockIds);
        }
        rankedComputedSeqBlockIds.push_back(computedSeqBlockIds);
    }
    std::vector<size_t> allrankComputedBlockNum =
        blockAllocator_->GetAllRankCommonComputedBlockNum(rankedComputedSeqBlockIds);
    size_t maxCachedBlocksPreRank = std::min(allrankComputedBlockNum.back() + 1, allrankComputedBlockNum.front());
    bool falg = true;
    if (maxCachedBlocksPreRank == 0) {
        falg = false;
    }
    for (auto& computedBlockNum: allrankComputedBlockNum) {
        if (falg && computedBlockNum < maxCachedBlocksPreRank) {
            falg = false;
            maxCachedBlocksPreRank--;
        }
        computedBlockNum = maxCachedBlocksPreRank;
    }

    return allrankComputedBlockNum;
}

std::vector<BlockId> SelfAttnBlockManager::GetCommonComputedBlockIds(const std::vector<SequenceSPtr> &seqs)
{
    std::vector<std::vector<BlockId>> computedSeqBlockIds;
    for (const auto &seq : seqs) {
        const std::vector<BlockId> &allBlocks = seqId2BlockTable_.at(seq->seqId_).GetBlockIds();
        size_t numCachedTokens = GetNumCachedTokens(seq);
        if (numCachedTokens % blockSize_ != 0) {
            throw std::runtime_error("The number of cached tokens is not a multiple of block size.");
        }
        size_t numCachedBlocks = numCachedTokens / blockSize_;
        std::vector<BlockId> computedBlockIds(allBlocks.begin(), allBlocks.begin() + numCachedBlocks);
        computedSeqBlockIds.push_back(computedBlockIds);
    }
    return blockAllocator_->GetCommonComputedBlockIds(computedSeqBlockIds);
}

std::vector<size_t> SelfAttnBlockManager::GetRemoteComputedBlockIds(
    const std::vector<SequenceSPtr> &seqs, size_t computedLens, uint32_t tpSize, std::string modelName)
{
    std::vector<std::vector<BlockId>> remoteComputedSeqBlockIds;
    for (const auto &seq : seqs) {
        const std::vector<BlockId> &allBlocks = seqId2BlockTable_.at(seq->seqId_).GetBlockIds();
        size_t numCachedBlocks = computedLens;
        std::vector<HashValue> hashValues = GetSeqHashValues(seq->seqId_);
        std::vector<std::string> allKeys;
        for (size_t i = computedLens; i < hashValues.size(); i++) {
            HashValue hashValue = hashValues[i];
            for (uint32_t tpIds = 0; tpIds < tpSize; tpIds++) {
                std::string key = std::to_string(hashValue) + "_"
                                + std::to_string(tpIds) + "_"
                                + std::to_string(tpSize) + "_"
                                + modelName;
                allKeys.push_back(key);
            }
        }
        if (!allKeys.empty() && tpSize > 0) {
            std::vector<bool> lookRes = memPoolInstance_->LookUp(allKeys);
            size_t numElem = 0;
            while (numElem < lookRes.size() && lookRes[numElem]) {
                ++numElem;
            }
            numCachedBlocks += numElem / tpSize;
        }

        std::vector<BlockId> remoteComputedBlockIds(allBlocks.begin(), allBlocks.begin() + numCachedBlocks);
        remoteComputedSeqBlockIds.push_back(remoteComputedBlockIds);
    }
    return blockAllocator_->GetCommonComputedBlockIds(remoteComputedSeqBlockIds);
}

std::vector<size_t> SelfAttnBlockManager::GetAllRankRemoteComputedBlockIds(
    const std::vector<SequenceSPtr> &seqs, std::vector<size_t> &computedBlocksNum, std::string modelName)
{
    if (seqs.size() > 1) {
        throw std::runtime_error("`Kv pool` do not support `splitfuse`!");
    }
    const auto &seq = seqs[0];

    size_t rankIdx = 0;
    while (rankIdx < rankSize_ && computedBlocksNum.back() < computedBlocksNum[rankIdx]) {
        rankIdx = (rankIdx + 1) % rankSize_;
    }

    std::vector<size_t> remoteComputedBlocksNum(computedBlocksNum);
    std::vector<std::vector<HashValue>> rankedHashValues = GetRankedHashValues(seq->seqId_);
    std::vector<std::string> allKeys;
    for (size_t i = remoteComputedBlocksNum[rankIdx]; i < rankedHashValues[0].size(); i++) {
        size_t curRankIdx = i == remoteComputedBlocksNum[rankIdx] ? rankIdx : 0;
        while (curRankIdx < rankSize_ && i < rankedHashValues[curRankIdx].size()) {
            HashValue hashValue = rankedHashValues[curRankIdx][i];
            std::string key = std::to_string(hashValue) + "_" + std::to_string(curRankIdx) + "_"
                            + std::to_string(rankSize_) + "_" + modelName;
            allKeys.push_back(key);
            curRankIdx++;
        }
    }
    if (!allKeys.empty()) {
        std::vector<bool> batchLookupResult = memPoolInstance_->LookUp(allKeys);
        size_t startIdx = 0;
        for (auto num : computedBlocksNum) {
            startIdx += num;
        }
        for (size_t idx = 0; idx < batchLookupResult.size() && batchLookupResult[idx]; idx++) {
            remoteComputedBlocksNum[(idx + startIdx) % rankSize_]++;
        }
    }

    return remoteComputedBlocksNum;
}

void SelfAttnBlockManager::Fork(SequenceSPtr &parentSeq, SequenceSPtr &childSeq)
{
    auto it = seqId2BlockTable_.find(parentSeq->seqId_);
    if (it == seqId2BlockTable_.end()) {
        throw std::runtime_error("SequenceId not found in seqId2BlockTable_ When Fork sequence");
    }
    BlockTable &blockTable = it->second;
    seqId2BlockTable_[childSeq->seqId_] = blockTable;
    seqsLastAccessBlocksTracker_.AddSeq(childSeq->seqId_);
}

AllocStatus SelfAttnBlockManager::CanSwap(const SequenceGroupSPtr &seqGroup, DeviceType dstDeviceType,
                                          SequenceStatus status, size_t numLookaheads)
{
    size_t numRelatedBlocks = 0;
    for (const auto &seq : seqGroup->GetFirstSequence(status)) {
        BlockTable &blockTable = seqId2BlockTable_.at(seq->seqId_);
        std::vector<BlockObjSPtr> blocksObj = blockTable.GetBlockObjs();
        numRelatedBlocks += CeilDiv(seq->GetLen() + numLookaheads, blockSize_);
    }
    size_t numTotalBlocks = blockAllocator_->GetNumTotalBlocks(dstDeviceType);
    size_t numFreeBlocks = blockAllocator_->GetNumFreeBlock(dstDeviceType);
    if (numRelatedBlocks > numTotalBlocks) {
        return AllocStatus::NEVER;
    } else if (numFreeBlocks >= numRelatedBlocks) {
        return AllocStatus::OK;
    }
    return AllocStatus::LATER;
}

AllocStatus SelfAttnBlockManager::CanSwapIn(const SequenceGroupSPtr &seqGroup, size_t numLookheadSlots)
{
    return CanSwap(seqGroup, DeviceType::NPU, SequenceStatus::SWAPPED, numLookheadSlots);
}

std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SelfAttnBlockManager::SwapIn(const SequenceGroupSPtr &seqGroup)
{
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> physicalBlockIdMapping;
    for (const auto &seq : seqGroup->GetFirstSequence(SequenceStatus::SWAPPED)) {
        std::vector<BlockObjSPtr> blocks = seqId2BlockTable_.at(seq->seqId_).GetBlockObjs();
        std::vector<std::pair<BlockId, BlockId>> seqSwapMapping =
            blockAllocator_->Swap(blocks, DeviceType::CPU, DeviceType::NPU);
        seqId2BlockTable_[seq->seqId_].Update(blocks);
        for (const auto &seqSwap : seqSwapMapping) {
            physicalBlockIdMapping.push_back({blockAllocator_->GetPhysicalBlockId(seqSwap.first),
                                              blockAllocator_->GetPhysicalBlockId(seqSwap.second)});
        }
    }
    return physicalBlockIdMapping;
}

bool SelfAttnBlockManager::CanSwapOut(const SequenceGroupSPtr &seqGroup)
{
    AllocStatus status = CanSwap(seqGroup, DeviceType::CPU, SequenceStatus::RUNNING);
    return status == AllocStatus::OK;
}

/// 将给定的sequence group swap out后 返回BlockId的一个map（从NPU到CPU）
/// \return: std::pair<PhysicalBlockId, PhysicalBlockId> {NPU BlockId, CPU BlockId}
std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SelfAttnBlockManager::SwapOut(
    const SequenceGroupSPtr &seqGroup)
{
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> physicalBlockIdMapping;

    for (const auto &seq : seqGroup->GetFirstSequence(SequenceStatus::RUNNING)) {
        std::vector<BlockObjSPtr> blocks = seqId2BlockTable_.at(seq->seqId_).GetBlockObjs();
        // swap from NPU to CPU，first为npu blockId second为cpu blockId
        std::vector<std::pair<BlockId, BlockId>> seqSwapMapping =
            blockAllocator_->Swap(blocks, DeviceType::NPU, DeviceType::CPU);
        seqId2BlockTable_[seq->seqId_].Update(blocks);
        for (const auto &seqSwap : seqSwapMapping) {
            // 在GetPhysicalBlockId接口中会根据blockId找到deviceType, 返回对应的physicalBlockId
            physicalBlockIdMapping.push_back({blockAllocator_->GetPhysicalBlockId(seqSwap.first),
                                              blockAllocator_->GetPhysicalBlockId(seqSwap.second)});
        }
    }
    return physicalBlockIdMapping;
}

size_t SelfAttnBlockManager::GetNumFreeNpuBlocks() const { return blockAllocator_->GetNumFreeBlock(DeviceType::NPU); }

size_t SelfAttnBlockManager::GetNumFreeCpuBlocks() const { return blockAllocator_->GetNumFreeBlock(DeviceType::CPU); }

float SelfAttnBlockManager::GetPrefixCacheHitRate() const { return blockAllocator_->GetPrefixCacheHitRate(); }

bool SelfAttnBlockManager::ResetPrefixCache() const { return blockAllocator_->ResetPrefixCache(); }

size_t SelfAttnBlockManager::GetNumCachedTokens(const SequenceSPtr &seq)
{
    size_t numCachedTokens = 0;
    std::vector<std::vector<HashValue>> rankedHashValues = GetRankedHashValues(seq->seqId_);
    for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        numCachedTokens += seqsBlockComputedTracker_.GetCachedTokensNum(
            seq, rankIdx, rankedHashValues[rankIdx], seq->IsPrefill());
    }
    return numCachedTokens;
}

size_t SelfAttnBlockManager::GetSeqNumCachedTokens(const SequenceSPtr &seq)
{
    size_t numCachedTokens = 0;
    numCachedTokens += seqsBlockComputedTracker_.GetCachedTokensNum(seq);
    return numCachedTokens;
}

/*
 eg. trailingPlaceHolderNum = 3 replacedPlaceHolderNum = 2
 替换 尾部trailingPlaceHolderNum个 中的 前replacedPlaceHolderNum个 为有效值
 替换前：[x x x x -1] [-1 -1]
 替换后：[x x x x  r] [ r -1]
*/
void SelfAttnBlockManager::ReplaceTrailingPlaceHolder(const SequenceSPtr &seq, size_t trailingPlaceHolderNum,
                                                      size_t replacedPlaceHolderNum)
{
    // sp并行场景下暂不支持prefix cache，所有不需要替换block中的place holder
    if (rankSize_ != 1) {
        return;
    }

    const std::vector<TokenId> &allTokenIds = seq->GetTokenIds();
    if (trailingPlaceHolderNum < replacedPlaceHolderNum || allTokenIds.size() < trailingPlaceHolderNum) {
        throw std::runtime_error("tokenIds size is less than replacement size");
    }

    size_t begin = allTokenIds.size() - trailingPlaceHolderNum;
    size_t end = allTokenIds.size() - trailingPlaceHolderNum + replacedPlaceHolderNum;
    std::vector<TokenId> newTokenIds(allTokenIds.begin() + begin, allTokenIds.begin() + end);

    BlockTable &blockTable = seqId2BlockTable_.at(seq->seqId_);
    blockTable.ReplaceTrailingPlaceHolder(newTokenIds, trailingPlaceHolderNum, replacedPlaceHolderNum);

    return;
}
size_t SelfAttnBlockManager::GetLocalDPRank() const { return localDPRank_; }

std::vector<size_t> SelfAttnBlockManager::GetTokenCountPerRank(SequenceId seqId) const
{
    const BlockTable &blockTable = seqId2BlockTable_.at(seqId);
    std::vector<size_t> tokenCounts(rankSize_, 0);

    for (const auto &blockObj : blockTable.GetBlockObjs()) {
        size_t rank = blockObj->GetRankIdx();
        tokenCounts[rank] += blockObj->GetTokenIds().size();
    }

    return tokenCounts;
}

size_t SelfAttnBlockManager::GetLatestAppendedRankId(SequenceId seqId) const
{
    const BlockTable &blockTable = seqId2BlockTable_.at(seqId);
    return blockTable.GetLatestAppendedRankId();
}

size_t SelfAttnBlockManager::GetAppendedBlockRankId(SequenceId seqId) const
{
    const BlockTable &blockTable = seqId2BlockTable_.at(seqId);
    return blockTable.GetAppendedBlockRankId();
}

} // namespace mindie_llm
