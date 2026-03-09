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
 
#ifndef PREFIX_CACHE_BLOCK_ALLOCATOR_H
#define PREFIX_CACHE_BLOCK_ALLOCATOR_H

#include <unordered_set>

#include "block_tracker.h"
#include "prefix_cache_block.h"
#include "block_allocator.h"
#include "hashless_block_allocator.h"
#include "hit_rate_calculator.h"
#include "evictor.h"

namespace mindie_llm {

class PrefixCacheBlockAllocator : public BlockAllocator,
                                  public std::enable_shared_from_this<PrefixCacheBlockAllocator> {
public:
    PrefixCacheBlockAllocator(BlockId beginBlockId, size_t numBlocks, size_t blockSize, BlockObjPoolSPtr blockObjPool);

    ~PrefixCacheBlockAllocator() override = default;

    BlockObjSPtr AllocateMutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock = nullptr,
                                      HashValue extraHash = 0) override;

    BlockObjSPtr AllocateImmutableBlock(std::vector<TokenId> &tokenIds, BlockObjSPtr prevBlock = nullptr,
                                        HashValue extraHash = 0) override;

    std::vector<BlockObjSPtr> AllocateImmutableBlocks(std::vector<std::vector<TokenId>> &tokenIds,
                                                      BlockObjSPtr prevBlock = nullptr,
                                                      HashValue extraHash = 0) override;

    size_t GetCachedBlockNum(std::vector<HashValue> &hashValues) override;

    BlockId AllocateBlockId() override;

    void FreeBlockId(BlockObjSPtr &block) override;

    BlockId PromoteToImmutableBlock(const BlockObjSPtr &block) override;

    void Free(BlockObjSPtr &block, bool keepBlockObj) override;

    std::vector<BlockObjSPtr> Fork(BlockObjSPtr &lastBlockObj) override;

    void MarkBlocksAsAccessed(const std::vector<BlockId> &blockIds, TimeStamp now) override;

    void MarkBlocksAsComputed() override;

    BlockId CowBlockIfNotAppendable(BlockObjSPtr &block) override;

    std::vector<BlockId>
    GetCommonComputedBlockIds(const std::vector<std::vector<BlockId>> &computedSeqBlockIds) override;

    size_t GetNumFullBlocksTouched(const std::vector<BlockObjSPtr> &blocks) override;

    size_t GetNumTotalBlocks() const override;

    size_t GetNumFreeBlock() const override;

    void SwapOut(std::vector<BlockObjSPtr> &blocks) override;

    void SwapIn(std::vector<BlockObjSPtr> &blocks) override;

    bool FindCachedBlockPrefix(HashValue blockHash) const override;

    std::vector<BlockId> FindCachedBlocksPrefix(std::vector<HashValue> &blockHashes) const override;

    float GetPrefixCacheHitRate() const override;

    bool ResetPrefixCache() override;

    std::vector<std::pair<BlockId, BlockId>> ClearCopyOnWrites() override;

    void AppendTokenIds(BlockObjSPtr blockObj, const std::vector<TokenId> &tokenIds) override;

    void ReplaceToken(BlockObjSPtr blockObj, size_t startIndex, TokenId newToken) override;
    
    // PrefixCacheBlockAllocator专有方法

    bool IsBlockCached(const BlockObjSPtr &block) const;

    bool IsBlockCached(const HashValue prefixHash) const;

    bool IsBlockComputed(BlockId blockId) const;

private:
    BlockId beginBlockId_;

    size_t blockSize_;

    size_t numBlocks_;

    std::vector<BlockId> allBlockIndices_;

    std::deque<BlockId> freeBlockIndices_;

    CopyOnWriteTracker cowTracker_;

    BlockComputedAttr blockComputedAttr_;

    BlockObjPoolSPtr blockObjPool_;

    // 前缀hash到blockid的映射
    std::unordered_map<HashValue, BlockId> cachedBlocks_;

    std::unordered_set<BlockId> touchedBlocks_;

    EvictorPtr evictor_;

    RefCounterProtocolSPtr refCounter_;

    std::shared_ptr<HitRateCalculator> hitRateCalculator_;

    void IncrRefCountCacheBlock(BlockObjSPtr block);

    void DecrRefCountCacheBlock(BlockObjSPtr block);

    void TrackBlockId(BlockId blockId, bool computed);

    void UntrackBlockId(BlockId blockId);

    BlockId MayBeAllocateEvictedBlockId();
};

} // namespace mindie_llm

#endif