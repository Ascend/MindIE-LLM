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
 
#include "block_tracker.h"
#include "request_response/request_id.h"


namespace mindie_llm {
constexpr HashValue NONE_HASH = 0;

void BlockComputedAttr::Reset(BlockId blockId)
{
    SetComputed(blockId, false);
    UpdateAccessTime(blockId, DEFAULT_LAST_ACCESSED_TIME);
}

void BlockComputedAttr::Enable(BlockId blockId)
{
    if (IsActive(blockId)) {
        throw std::runtime_error("activeFlag_ is true, illegal to run Enable function!");
    }
    SetActive(blockId, true);
    Reset(blockId);
}

void BlockComputedAttr::Disable(BlockId blockId)
{
    if (!IsActive(blockId)) {
        throw std::runtime_error("activeFlag_ is true, illegal to run Enable function!");
    }
    SetActive(blockId, false);
    Reset(blockId);
}

void BlockComputedAttr::SetActive(BlockId blockId, bool active)
{
    if (IsValidBlockId(blockId)) {
        blockInfos_[blockId - beginBlockId_].active = active;
    }
}

bool BlockComputedAttr::IsActive(BlockId blockId) const
{
    return IsValidBlockId(blockId) && blockInfos_[blockId - beginBlockId_].active;
}

void BlockComputedAttr::SetComputed(BlockId blockId, bool computed)
{
    if (IsValidBlockId(blockId)) {
        blockInfos_[blockId - beginBlockId_].computed = computed;
    }
}

bool BlockComputedAttr::IsComputed(BlockId blockId) const
{
    return IsValidBlockId(blockId) && blockInfos_[blockId - beginBlockId_].computed;
}

void BlockComputedAttr::UpdateAccessTime(BlockId blockId, TimeStamp now)
{
    if (IsValidBlockId(blockId)) {
        blockInfos_[blockId - beginBlockId_].lastAccessed = now;
    }
}

TimeStamp BlockComputedAttr::LastAccessed(BlockId blockId) const
{
    if (IsValidBlockId(blockId)) {
        return blockInfos_[blockId - beginBlockId_].lastAccessed;
    }
    return -1;
}

HashValue ComputeHashValueForSeq(HashValue prevBlockHash, std::vector<TokenId> &tokenIds, HashValue extraHash)
{
    HashValue seed = 0;
    if (prevBlockHash != NONE_HASH) {
        HashCombine(seed, prevBlockHash);
    }

    for (const TokenId token : tokenIds) {
        HashCombine(seed, token);
    }

    HashCombine(seed, extraHash);
    return seed;
}

// Most important API of SeqsBlocksComputedTracker, used to get cached and COMPUTED tokens for whole token sequences
// At first update token sequence's prefix hashes, then use binary search to find the longest cached and COMPUTED prefix
// blocks.
size_t SeqsBlocksComputedTracker::GetCachedTokensNum(
    const SequenceSPtr &seq, size_t rankIdx, std::vector<HashValue> &blockHashes, bool seqPrefillFlag)
{
    if (!enableCaching_) {
        return 0;
    }

    SequenceId seqId = seq->seqId_;
    bool hasComputedTokens = (seqIdToNumComputedTokens_.find({seqId, rankIdx}) != seqIdToNumComputedTokens_.end());
    if (seqPrefillFlag && hasComputedTokens) {
        return seqIdToNumComputedTokens_.at({seqId, rankIdx});
    }

    size_t numCachedTokens = 0;

    // ChunkedPrefill只有在首块时计算numCachedTokens并写表
    bool isFirstChunk = (seq->data_.numComputedTokens_ == 0);
    if (isFirstChunk) {
        size_t numCachedBlocks = (allocator_->FindCachedBlocksPrefix(rankIdx, blockHashes)).size();
        numCachedTokens = numCachedBlocks * blockSize_;
    }

    seqIdToNumComputedTokens_[{seqId, rankIdx}] = numCachedTokens;
    return numCachedTokens;
}

/// Used by scheduler to enforce budget.
size_t SeqsBlocksComputedTracker::GetCachedTokensNum(const SequenceSPtr &seq)
{
    const std::vector<TokenId> tokenIds = seq->GetTokenIds();
    HashValue extraHash = seq->GetExtraHash();
    bool seqPrefillFlag = seq->IsPrefill();
    if (!enableCaching_ || !seqPrefillFlag) {
        return 0;
    }
    // ChunkedPrefill请求只有在初次调度即首块时计算CachedTokens
    bool isFirstChunk = (seq->data_.numComputedTokens_ == 0);
    if (!isFirstChunk) {
        size_t cachedTokensNum = 0;
        for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            if ((seqIdToNumComputedTokens_.find({seq->seqId_, rankIdx}) != seqIdToNumComputedTokens_.end())) {
                cachedTokensNum += seqIdToNumComputedTokens_.at({seq->seqId_, rankIdx});
            }
        }
        return cachedTokensNum;
    }

    size_t cachedTokensNum = 0;
    std::vector<HashValue> blockHashes;
    HashValue prevBlockHash = NONE_HASH;
    size_t numFullBlocks = tokenIds.size() / blockSize_;
    size_t rankIdx = 0;
    for (size_t blockIdx = 0; blockIdx < numFullBlocks; blockIdx++) {
        // 1. 计算一个block的hash value
        std::vector<TokenId> blockTokenIds;
        for (size_t tokenIdx = blockIdx * blockSize_; tokenIdx < (blockIdx + 1) * blockSize_; tokenIdx++) {
            blockTokenIds.push_back(tokenIds[tokenIdx]);
        }
        HashValue blockHash = ComputeHashValueForSeq(prevBlockHash, blockTokenIds, extraHash);
        blockHashes.push_back(blockHash);
        prevBlockHash = blockHash;
        
        // 2. 查询该block 是否命中
        bool cacheBlockFlag = allocator_->FindCachedBlockPrefix(rankIdx, blockHash);
        if (cacheBlockFlag) {
            cachedTokensNum += blockSize_;
        } else {
            break;
        }
        rankIdx = (rankIdx + 1) % rankSize_;
    }

    return cachedTokensNum;
}

void SeqsBlocksComputedTracker::RemoveSeq(SequenceId seqId)
{
    // 虚推请求跳过 PrefixCache，没有注册到 seqIdToNumComputedTokens_，直接返回
    if (!enableCaching_ || seqId == SIMULATE_SEQUENCE_ID) {
        return;
    }
    for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (seqIdToNumComputedTokens_.find({seqId, rankIdx}) == seqIdToNumComputedTokens_.end()) {
            throw std::runtime_error(
                "seqId is not recorded in the number of computed tokens table, "
                "cannot remove seqId!"
            );
        }
        seqIdToNumComputedTokens_.erase({seqId, rankIdx});
    }
}

void SeqsLastAccessBlocksTracker::AddSeq(SequenceId seqId)
{
    // make sure seqId has not been in the Tracker
    if (seqIdToLastAccessTime_.find(seqId) != seqIdToLastAccessTime_.end()) {
        throw std::runtime_error("seqId is already recorded the last access time table, add seqId fail!");
    }
    seqIdToLastAccessTime_[seqId] = -1;
}

void SeqsLastAccessBlocksTracker::RemoveSeq(SequenceId seqId)
{
    if (seqIdToLastAccessTime_.find(seqId) == seqIdToLastAccessTime_.end()) {
        throw std::runtime_error("seqId is not recorded the last access time table, cannot remove seqId!");
    }
    seqIdToLastAccessTime_.erase(seqId);
}

void SeqsLastAccessBlocksTracker::UpdateSeqLastAccess(SequenceId seqId, TimeStamp time)
{
    if (seqIdToLastAccessTime_.find(seqId) == seqIdToLastAccessTime_.end()) {
        throw std::runtime_error(
            "seqId is not recorded the last access time table, cannot update last access time to seqId!");
    }
    seqIdToLastAccessTime_[seqId] = time;
}

void SeqsLastAccessBlocksTracker::UpdateSeqBlocksLastAccess(
    SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds)
{
    if (seqIdToLastAccessTime_.find(seqId) == seqIdToLastAccessTime_.end()) {
        throw std::runtime_error(
            "seqId is not recorded the last access time table, cannot update last access time to blocks of seqId!");
    }
    TimeStamp lastAccessTime = seqIdToLastAccessTime_[seqId];
    for (size_t rankId = 0; rankId < rankedBlockIds.size(); rankId++) {
        allocator_->MarkBlocksAsAccessed(rankId, rankedBlockIds[rankId], lastAccessTime);
    }
}

} // namespace mindie_llm