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
 
#include "block_table.h"
#include <algorithm>
#include <stdexcept>
#include "math_utils.h"


namespace mindie_llm {
// Split the token ids into block-sized chunks so they can be easily allocated.
std::vector<std::vector<TokenId>> BlockTable::ChunkTokensForAllocate(const std::vector<TokenId> &tokenIds,
                                                                     size_t chunkSize)
{
    if (chunkSize == 0) {
        throw std::runtime_error("Chunk size cannot be zero.");
    }
    std::vector<std::vector<TokenId>> tokenBlocks;
    for (size_t i = 0; i < tokenIds.size(); i += chunkSize) {
        size_t min = std::min(i + chunkSize, tokenIds.size());
        tokenBlocks.emplace_back(tokenIds.begin() + i, tokenIds.begin() + min);
    }
    return tokenBlocks;
}

BlockTable::BlockTable(size_t blockSize, DeviceAwareBlockAllocatorSPtr &blockAllocator, size_t rankSize)
    : blockObjs_(rankSize), blockSize_(blockSize), blockAllocator_(blockAllocator), rankSize_(rankSize)
{
    if (blockSize_ == 0) {
        throw std::runtime_error("blockSize can't be zero.");
    }
    // 初始化每个rank的blockObjs_为空vector
    for (std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        rankBlocks = std::vector<BlockObjSPtr>();
    }
    numFullSlotsPerRank_ = std::vector<size_t>(rankSize, 0);
}

BlockTable::BlockTable(size_t blockSize, DeviceAwareBlockAllocatorSPtr &blockAllocator,
                       std::vector<std::vector<BlockObjSPtr>> &blockObjs, size_t rankSize)
    : blockObjs_(blockObjs), blockSize_(blockSize), blockAllocator_(blockAllocator), rankSize_(rankSize)
{
    if (blockSize_ == 0) {
        throw std::runtime_error("blockSize can't be zero.");
    }
    // 遍历每个rank的blockObjs
    for (const std::vector<BlockObjSPtr> &rankBlocks : blockObjs) {
        // 遍历当前rank的所有block
        for (const BlockObjSPtr &blockObj : rankBlocks) {
            blockIds_.push_back(blockObj->GetBlockId());
        }
    }

    numFullSlotsPerRank_ = std::vector<size_t>(rankSize, 0);
    for (const std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        for (const BlockObjSPtr &blockObj : rankBlocks) {
            numFullSlotsPerRank_[blockObj->GetRankIdx()] += blockObj->GetTokenIds().size();
        }
    }
}

void BlockTable::Allocate(const std::vector<TokenId> &tokenIds, DeviceType device, HashValue extraHash)
{
    if (IsAllocated()) {
        throw std::runtime_error("Blocks of this block table are already allocated.");
    }
    std::vector<BlockObjSPtr> newBlockObjs;

    newBlockObjs = AllocateBlocksForTokenIds(tokenIds, device, extraHash);
    Update(newBlockObjs); // owned block objs and block ids are invalid, so safe to update directly.
    numFullSlotsPerRank_[0] = tokenIds.size();
}

void BlockTable::AllocateSmallRankFirst(const std::vector<TokenId> &tokenIds, DeviceType device, HashValue extraHash)
{
    if (IsAllocated()) {
        throw std::runtime_error("Blocks of this block table are already allocated.");
    }
    std::vector<BlockObjSPtr> newBlockObjs;

    newBlockObjs = AllocateBlocksForTokenIdsSmallRankFirst(tokenIds, device, extraHash);
    Update(newBlockObjs); // owned block objs and block ids are invalid, so safe to update directly.
    // 更新每个rank的填充slot数
    for (BlockObjSPtr blockObj : newBlockObjs) {
        numFullSlotsPerRank_[blockObj->GetRankIdx()] += blockObj->GetTokenIds().size();
    }
}

// rebuild block id list using allocated block objects.
void BlockTable::Update(const std::vector<BlockObjSPtr> &blockObjs)
{
    for (std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        rankBlocks.clear();
    }
    blockIds_.clear();
    for (const BlockObjSPtr &blockObj : blockObjs) {
        blockObjs_[blockObj->GetRankIdx()].push_back(blockObj);
        blockIds_.push_back(blockObj->GetBlockId());
    }
}

bool BlockTable::CanAppendNewTokens(const std::vector<TokenId> &tokenIds, size_t numLookaheadSlots) const
{
    if (rankSize_ > 1) {
        size_t numTokenIds = tokenIds.size() + numLookaheadSlots;
        size_t numEmptySlots = GetNumEmptySlots(currentSpRank_);
        size_t nextSpRank = currentSpRank_;
        if ((numEmptySlots == 0 && numLookaheadSlots == 0) || (numLookaheadSlots != 0 && tokenIds.size() > numEmptySlots)) {
            nextSpRank = (nextSpRank + 1) % rankSize_;
            numEmptySlots = GetNumEmptySlots(nextSpRank);
        }
        size_t needBlocks = 1;
        if (numTokenIds > numEmptySlots) {
            needBlocks = 1 + CeilDiv(numTokenIds - numEmptySlots, blockSize_);
        }
        return blockAllocator_->GetNumFreeBlock(DeviceType::NPU, nextSpRank) >= needBlocks;
    } else {
        throw std::runtime_error("rank size == 1 need call other function.");
    }
}

void BlockTable::AppendNewTokens(const std::vector<TokenId> &newTokenIds, HashValue extraHash, size_t numLookaheadSlots)
{
    if (!IsAllocated()) {
        throw std::runtime_error("No blocks have been allocated.");
    }
    if (newTokenIds.size() == 0) {
        return;
    }
    if (numLookaheadSlots == 0) {
        // 检查是否需要申请新的block
        size_t numEmptySlots = GetNumEmptySlots(currentSpRank_);
        // 用于计算这一轮的rankid
        if (numEmptySlots == 0) {
            currentSpRank_ = (currentSpRank_ + 1) % rankSize_;
            numEmptySlots = GetNumEmptySlots(currentSpRank_);
        }
        if (numEmptySlots < 1 + numLookaheadSlots) {
            auto &rankBlockObjs = blockObjs_[currentSpRank_];
            size_t numBlocksToAllocate = CeilDiv(1 + numLookaheadSlots - numEmptySlots, blockSize_);
            for (size_t i = 0; i < numBlocksToAllocate; ++i) {
                std::vector<TokenId> emptyTokenIds;
                BlockObjSPtr newBlockObj = blockAllocator_->AllocateMutableBlock(
                    DeviceType::NPU, emptyTokenIds, rankBlockObjs.empty() ? nullptr : rankBlockObjs.back(), extraHash,
                    currentSpRank_);
                rankBlockObjs.push_back(newBlockObj);
                blockIds_.push_back(newBlockObj->GetBlockId());
            }
        }
        std::vector<TokenId> currentSpToken{newTokenIds.back()};
        // append token
        AppendToSpRank(currentSpRank_, currentSpToken);
    } else {
        std::vector<TokenId> tokenIds;
        for (size_t i = 0; i < newTokenIds.size(); i++) {
            if (newTokenIds[i] != -1) {
                tokenIds.push_back(newTokenIds[i]);
            }
        }
        size_t numEmptySlots = GetNumEmptySlots(currentSpRank_);
        size_t nextSpRank = currentSpRank_;
        isAppendBlock_ = false;
        if (newTokenIds.size() > numEmptySlots) {
            nextSpRank = (currentSpRank_ + 1) % rankSize_;
            size_t reservedSlots = newTokenIds.size() - numEmptySlots;
            size_t nextRankEmptySlots = GetNumEmptySlots(nextSpRank);
            if (nextRankEmptySlots < reservedSlots) {
                auto &rankBlockObjs = blockObjs_[nextSpRank];
                size_t numBlocksToAllocate = CeilDiv(reservedSlots, blockSize_);
                for (size_t i = 0; i < numBlocksToAllocate; ++i) {
                    std::vector<TokenId> emptyTokenIds;
                    BlockObjSPtr newBlockObj = blockAllocator_->AllocateMutableBlock(
                        DeviceType::NPU, emptyTokenIds, rankBlockObjs.empty() ? nullptr : rankBlockObjs.back(), extraHash,
                        nextSpRank);
                    rankBlockObjs.push_back(newBlockObj);
                    blockIds_.push_back(newBlockObj->GetBlockId());
                }
                isAppendBlock_ = true;
                appendBlockRankId_ = nextSpRank;
            }
        }
        if (tokenIds.size() > 0) {
            size_t segIndex = std::min(numEmptySlots, tokenIds.size() - 1);
            std::vector<TokenId> lastRankToken(tokenIds.begin(), tokenIds.begin() + segIndex);
            AppendToSpRank(currentSpRank_, lastRankToken);
            if (numEmptySlots < tokenIds.size()) {
                currentSpRank_ = nextSpRank;
            }
            std::vector<TokenId> currentRankToken(tokenIds.begin() + segIndex, tokenIds.end());
            AppendToSpRank(currentSpRank_, currentRankToken);
        }
    }
}

void BlockTable::AppendToSpRank(size_t spRank, const std::vector<TokenId>& newTokenIds)
{
    size_t spTokenNum = numFullSlotsPerRank_[spRank];
    auto &rankBlockObjs = blockObjs_[spRank];
    size_t firstBlockIdx = spTokenNum / blockSize_;
    size_t firstTokenIdx = spTokenNum % blockSize_;

    for (size_t i = 0; i < newTokenIds.size(); ++i) {
        size_t blockIdx = firstBlockIdx + (firstTokenIdx + i) / blockSize_;

        if (blockIdx >= rankBlockObjs.size()) {
            throw std::runtime_error("Block index out of range");
        }

        std::vector<TokenId> singleToken{newTokenIds[i]};
        blockAllocator_->AppendTokenIds(rankBlockObjs[blockIdx], singleToken);

        // 更新blockIds_以防block变化
        if (rankBlockObjs[blockIdx]->IsFull()) {
            blockIds_[blockIdx] = rankBlockObjs[blockIdx]->GetBlockId();
        }
    }
    numFullSlotsPerRank_[spRank] += newTokenIds.size();
}

// append tokens ids right after filled slots. If not enough empty slots, allocate new blocks.
void BlockTable::AppendTokenIds(const std::vector<TokenId> &tokenIds, HashValue extraHash, size_t numLookaheadSlots)
{
    if (!IsAllocated()) {
        throw std::runtime_error("No blocks have been allocated.");
    }

    // 固定使用rank 0来保持向后兼容
    const size_t firstRank = 0;
    std::vector<BlockObjSPtr> &firstRankBlockObjs = blockObjs_[firstRank]; // 先获取rank 0的blockObjs

    EnsureEnoughSlots(tokenIds.size() + numLookaheadSlots, extraHash);

    size_t firstBlockIdx = numFullSlotsPerRank_[firstRank] / blockSize_;
    std::vector<std::vector<TokenId>> tokenBlocks = ChunkTokensForAppend(tokenIds);

    for (size_t i = 0; i < tokenBlocks.size(); ++i) {
        // tokens appended to block must be shorter than empty slots in that block.
        blockAllocator_->AppendTokenIds(firstRankBlockObjs.at(firstBlockIdx + i), tokenBlocks.at(i));
        if (i + 1 < tokenBlocks.size()) {
            if (!firstRankBlockObjs[firstBlockIdx + i]->IsFull()) {
                throw std::runtime_error("a block table has not-full block in the middle, this should not happend.");
            }
        }
        blockIds_.at(firstBlockIdx + i) = firstRankBlockObjs.at(firstBlockIdx + i)->GetBlockId();
    }

    numFullSlotsPerRank_[firstRank] += tokenIds.size();
}

// Determine how many blocks are related given new tokenids with length of tokenIdsSize + numLookaheadSlots,
// including this blocktable's last block if it is not full. This is used for the scheduler to determine whether a
// sequence can continue generation, or it must be preempted.
size_t BlockTable::GetNumRelatedBlocks(size_t tokenIdsSize, size_t numLookaheadSlots) const
{
    size_t numTokenIds = tokenIdsSize + numLookaheadSlots;
    size_t numLastBlockEmptySlots = blockSize_ - (numFullSlotsPerRank_[0] % blockSize_);
    if (numLastBlockEmptySlots == blockSize_) { // the last block is full.
        return CeilDiv(numTokenIds, blockSize_);
    }
    if (numLastBlockEmptySlots >= numTokenIds) { // the last block has enough empty slots.
        return 0;
    }
    return CeilDiv(numTokenIds - numLastBlockEmptySlots, blockSize_);
}

BlockTable BlockTable::Fork()
{
    if (!IsAllocated()) {
        throw std::runtime_error("Empty blocks can't be forked.");
    }

    // 只支持单rank场景
    if (rankSize_ != 1) {
        throw std::runtime_error("Fork only supports single rank scenario");
    }

    // fork最后一个block
    BlockObjSPtr lastBlockObj = blockObjs_[0].back();
    std::vector<BlockObjSPtr> forkedBlockObjs = blockAllocator_->Fork(lastBlockObj);

    // 创建新的blockObjs结构
    std::vector<std::vector<BlockObjSPtr>> newBlockObjs;
    newBlockObjs.push_back(forkedBlockObjs);

    return BlockTable(blockSize_, blockAllocator_, newBlockObjs, rankSize_);
}

void BlockTable::Free()
{
    for (std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        for (BlockObjSPtr &blockObj : rankBlocks) {
            blockAllocator_->Free(blockObj); // free both block id and block obj
        }
    }
    blockObjs_.clear();
    blockIds_.clear();
    std::fill(numFullSlotsPerRank_.begin(), numFullSlotsPerRank_.end(), 0);
    lastUsedRanks_ = std::queue<size_t>();
    currentSpRank_ = 0;
}

const std::vector<BlockId> &BlockTable::GetBlockIds() const { return blockIds_; }

void BlockTable::GetRankedPrefixBlockOrder(std::vector<std::vector<size_t>> &rankedPrefixBlockOrder) const
{
    rankedPrefixBlockOrder.resize(rankSize_);
    for (size_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        for (auto &blockObj: blockObjs_[rankIdx]) {
            rankedPrefixBlockOrder[rankIdx].push_back(blockObj->GetAllocateOrder());
        }
    }
}

// std::vector<BlockObjSPtr> &BlockTable::GetBlockObjs() { return blockObjs_; }
// 合并所有rank的blockObjs并返回
std::vector<BlockObjSPtr> BlockTable::GetBlockObjs() const
{
    std::vector<BlockObjSPtr> mergedBlocks;
    for (const std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        mergedBlocks.insert(mergedBlocks.end(), rankBlocks.begin(), rankBlocks.end());
    }
    return mergedBlocks;
}
size_t BlockTable::GetNumFullSlots() const { return numFullSlotsPerRank_[0]; }

// allocate new blocks to ensure there is enough empty slots
void BlockTable::EnsureEnoughSlots(size_t numRequiredEmptySlots, HashValue extraHash, size_t rankId)
{
    if (!IsAllocated()) {
        throw std::runtime_error("No blocks have been allocated, init blockTable before using it");
    }
    if (rankId >= rankSize_) {
        throw std::runtime_error("Invalid rank id");
    }

    DeviceType device = DeviceType::NPU; // Currently only supports appending tokens to NPU blocks.

    size_t numEmptySlots = GetNumEmptySlots(rankId);
    if (numRequiredEmptySlots <= numEmptySlots) { // Current empty slots are enough.
        return;
    }
    size_t numBlocksToAllocate = CeilDiv(numRequiredEmptySlots - numEmptySlots, blockSize_);

    std::vector<BlockObjSPtr> &rankBlocks = blockObjs_[rankId];
    for (size_t i = 0; i < numBlocksToAllocate; ++i) {
        std::vector<TokenId> emptyTokenIds;
        BlockObjSPtr newBlockObj = blockAllocator_->AllocateMutableBlock(
            device, emptyTokenIds, rankBlocks.empty() ? nullptr : rankBlocks.back(), extraHash);
        rankBlocks.push_back(newBlockObj);
        blockIds_.push_back(newBlockObj->GetBlockId());
    }
}

// Get the number of tokens in the sequence that are corresponding to this block table, but not yet appended to this
// block table. input tokenIds include prompts and previous generations.
std::vector<TokenId> BlockTable::GetNewGenTokenIds(const std::vector<TokenId> &tokenIds) const
{
    if (rankSize_ > 1) {
        throw std::runtime_error("GetNewGenTokenIds only supports single rank");
    }
    return std::vector<TokenId>(tokenIds.begin() + numFullSlotsPerRank_[0], tokenIds.end());
}

// given tokens, allocate new block objs with block ids, which will be used to initialize block table
std::vector<BlockObjSPtr> BlockTable::AllocateBlocksForTokenIds(const std::vector<TokenId> &tokenIds, DeviceType device,
                                                                HashValue extraHash)
{
    std::vector<BlockObjSPtr> newBlockObjs;
    std::vector<std::vector<TokenId>> fullTokenBlocks;
    std::vector<TokenId> nonFullTokenBlock;
    std::vector<std::vector<TokenId>> blockedTokenIds = ChunkTokensForAllocate(tokenIds, blockSize_);

    for (const std::vector<TokenId> &block : blockedTokenIds) {
        if (block.size() == blockSize_) {
            fullTokenBlocks.push_back(block);
        } else {
            nonFullTokenBlock = block;
        }
    }

    BlockObjSPtr prevBlock = nullptr;

    if (!fullTokenBlocks.empty()) {
        std::vector<BlockObjSPtr> newBlocks =
            blockAllocator_->AllocateImmutableBlocks(device, fullTokenBlocks, prevBlock, extraHash);
        newBlockObjs.insert(newBlockObjs.end(), newBlocks.begin(), newBlocks.end());
        prevBlock = newBlockObjs.back();
    }

    if (!nonFullTokenBlock.empty()) {
        BlockObjSPtr block = blockAllocator_->AllocateMutableBlock(device, nonFullTokenBlock, prevBlock, extraHash);
        newBlockObjs.push_back(block);
    }

    return newBlockObjs;
}

std::vector<BlockObjSPtr> BlockTable::AllocateBlocksForTokenIdsSmallRankFirst(const std::vector<TokenId> &tokenIds,
                                                                              DeviceType device, HashValue extraHash)
{
    std::vector<BlockObjSPtr> newBlockObjs;
    BlockObjSPtr prevBlock = nullptr;
    size_t remainingTokens = tokenIds.size();
    size_t currentPos = 0;
    size_t rankIdx = 0; // 当前分配的rank索引

    while (remainingTokens > 0) {
        // 计算当前rank能分配的token数量
        size_t tokensForThisRank = std::min(remainingTokens, blockSize_);

        // 直接分配tokensForThisRank个token到当前rank
        std::vector<TokenId> rankTokens(tokenIds.begin() + currentPos,
                                        tokenIds.begin() + currentPos + tokensForThisRank);
        currentPos += tokensForThisRank;
        remainingTokens -= tokensForThisRank;

        BlockObjSPtr blockObj;
        if (rankTokens.size() == blockSize_) {
            blockObj = blockAllocator_->AllocateImmutableBlock(device, rankTokens, prevBlock, extraHash, rankIdx);
        } else {
            blockObj = blockAllocator_->AllocateMutableBlock(device, rankTokens, prevBlock, extraHash, rankIdx);
        }

        newBlockObjs.push_back(blockObj);
        prevBlock = blockObj;

        if (remainingTokens == 0) {
            currentSpRank_ = rankIdx;
            appendBlockRankId_ = currentSpRank_;
        }
        // 注意：model要求，如果下一个要分配的是最后一个block，这个block需要分配在已经分配过的最后一个
        // eg1: 1305 tokens, sp = 8 ranks, blockSize = 128 => [0 1] [0 1] [0] [0] [0] [0] [0] [0 1]
        // eg2: 380 tokens, sp = 8 ranks, blockSize = 128 => [0] [0] [0]
        if (remainingTokens <= blockSize_ && newBlockObjs.size() >= rankSize_) {
            rankIdx = rankSize_ - 1;
        } else {
            rankIdx = (rankIdx + 1) % rankSize_;
        }
    }

    return newBlockObjs;
}

size_t BlockTable::GetNumTokenIds() const
{
    size_t res = 0;
    for (const std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        for (const BlockObjSPtr &blockObj : rankBlocks) {
            res += blockObj->GetTokenIds().size();
        }
    }
    return res;
}

bool BlockTable::IsAllocated() const
{
    // 检查每个rank的blockObjs是否至少有一个block
    for (const std::vector<BlockObjSPtr> &rankBlocks : blockObjs_) {
        if (!rankBlocks.empty()) {
            return true;
        }
    }
    return false;
}

size_t BlockTable::GetNumEmptySlots(size_t rankId) const
{
    if (!IsAllocated()) {
        throw std::runtime_error("No blocks have been allocated.");
    }
    if (rankId >= rankSize_) {
        throw std::runtime_error("Invalid rank id");
    }
    size_t rankBlockCount = blockObjs_[rankId].size();
    size_t rankFullSlots = numFullSlotsPerRank_[rankId];
    return rankBlockCount * blockSize_ - rankFullSlots;
}

// Split the token ids into block-sized chunks so they can be easily appended to blocks. The first "token block"
// may have less token ids than the block size, since the last allocated block may be partially full.
std::vector<std::vector<TokenId>> BlockTable::ChunkTokensForAppend(const std::vector<TokenId> &tokenIds) const
{
    std::vector<std::vector<TokenId>> tokenBlocks{};
    if (tokenIds.empty()) {
        return tokenBlocks;
    }
    size_t numLastBlockEmptySlots = blockSize_ - (numFullSlotsPerRank_[0] % blockSize_);
    tokenBlocks.push_back(
        std::vector<TokenId>(tokenIds.begin(), tokenIds.begin() + std::min(numLastBlockEmptySlots, tokenIds.size())));
    for (size_t i = numLastBlockEmptySlots; i < tokenIds.size(); i += blockSize_) {
        size_t end = std::min(i + blockSize_, tokenIds.size());
        tokenBlocks.push_back(std::vector<TokenId>(tokenIds.begin() + i, tokenIds.begin() + end));
    }
    return tokenBlocks;
}

/*
 eg. trailingPlaceHolderNum = 3 replacedPlaceHolderNum = 2
 替换 尾部trailingPlaceHolderNum个 中的 前replacedPlaceHolderNum个 为有效值
 替换前：[x x x x -1] [-1 -1]
 替换后：[x x x x  r] [ r -1]
*/
void BlockTable::ReplaceTrailingPlaceHolder(const std::vector<TokenId> &tokenIds, size_t trailingPlaceHolderNum,
                                            size_t replacedPlaceHolderNum, size_t rankId)
{
    size_t totalTokens = GetNumTokenIds();
    if (trailingPlaceHolderNum >= totalTokens) {
        throw std::runtime_error("replace start index cannot be greater than the total number of tokens.");
    }

    if (replacedPlaceHolderNum > trailingPlaceHolderNum) {
        throw std::runtime_error("number of tokens to replace cannot be greater than the number of tokens start");
    }

    // 计算开始替换的块索引
    size_t startBlockIndex = (totalTokens - trailingPlaceHolderNum) / blockSize_;
    // 计算在开始块中的局部索引
    size_t startLocalIndex = (totalTokens - trailingPlaceHolderNum) % blockSize_;

    // replace的是decode产生的尾部token，在整个sequence中不会是第0个
    if (startBlockIndex == 0 && startLocalIndex == 0) {
        throw std::runtime_error("replace start index cannot be the first token.");
    }
    size_t prevTokenOffset = startLocalIndex == 0 ? blockSize_ - 1 : startLocalIndex - 1;
    size_t prevBlockOffset = startLocalIndex == 0 ? startBlockIndex - 1 : startBlockIndex;

    std::vector<BlockObjSPtr> &rankBlocks = blockObjs_[rankId]; // 使用指定rank的blocks

    for (size_t i = 0; i < replacedPlaceHolderNum; ++i) {
        size_t blockOffset = startBlockIndex + (startLocalIndex + i) / blockSize_;
        size_t tokenOffset = (startLocalIndex + i) % blockSize_;
        BlockObjSPtr currentBlock = rankBlocks[blockOffset];
        TokenId newToken = tokenIds[i];

        if (rankBlocks[prevBlockOffset]->GetTokenIds()[prevTokenOffset] == PLACEHOLDER_TOKEN) {
            throw std::runtime_error("preceding token can't be PLACEHOLDER_TOKEN");
        }

        if (currentBlock->GetTokenIds()[tokenOffset] != PLACEHOLDER_TOKEN) {
            throw std::runtime_error("only PLACEHOLDER_TOKEN can be replaced.");
        }
        blockAllocator_->ReplaceToken(currentBlock, tokenOffset, newToken);

        // ReplaceToken后block可能变化，需要更新blockIds_
        if (rankBlocks[blockOffset]->IsFull()) {
            blockIds_[blockOffset] = rankBlocks[blockOffset]->GetBlockId();
        }

        prevBlockOffset = blockOffset;
        prevTokenOffset = tokenOffset;
    }
}
size_t BlockTable::GetLatestAppendedRankId() const
{
    return currentSpRank_;
}

size_t BlockTable::GetAppendedBlockRankId() const
{
    return appendBlockRankId_;
}

bool BlockTable::IsAppendBlock() const
{
    return isAppendBlock_;
}
} // namespace mindie_llm