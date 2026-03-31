/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#define private public
#include "request_single_block_manager.h"
#include "sequence_group.h"
#undef private

namespace mindie_llm {

static SequenceGroupSPtr MakeGroup(const RequestId &rid, const std::vector<SequenceSPtr> &seqs)
{
    RequestId ridCopy = rid;
    return std::make_shared<SequenceGroup>(ridCopy, seqs, /*sampling*/ nullptr);
}

TEST(RequestSingleBlockManagerTest, AllocateReuseForkFreeAndRankedBlockIds)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 4, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 2, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    auto s1 = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1, 2, 3, 4});
    auto g1 = MakeGroup("rq", {s1});

    EXPECT_EQ(mgr.CanAllocate(g1), AllocStatus::OK);
    EXPECT_TRUE(mgr.Allocate(g1));
    ASSERT_EQ(mgr.requestEntries_.size(), 1u);
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 3u);

    const auto ids1 = mgr.GetBlockIds(1);
    ASSERT_EQ(ids1.size(), 1u);
    ASSERT_EQ(ids1[0].size(), 1u);
    const BlockId bid = ids1[0][0];

    // Reuse same request: allocate more sequences without consuming new blocks.
    auto s2 = std::make_shared<Sequence>(/*seqId*/ 2, /*blockSize*/ 8, std::vector<TokenId>{9, 10});
    auto g2 = MakeGroup("rq", {s2});
    EXPECT_EQ(mgr.CanAllocate(g2), AllocStatus::OK);
    EXPECT_TRUE(mgr.Allocate(g2));
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 3u);
    EXPECT_EQ(mgr.GetBlockIds(2)[0][0], bid);

    // Ranked block ids should duplicate rank0's block for all ranks.
    std::vector<RankedBlockId> ranked;
    mgr.GetRankedBlockIds(1, ranked);
    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_EQ(ranked[0], (RankedBlockId{bid, 0}));
    EXPECT_EQ(ranked[1], (RankedBlockId{bid, 1}));

    std::vector<std::vector<BlockId>> ranked2;
    mgr.GetRankedBlockIds(1, ranked2);
    ASSERT_EQ(ranked2.size(), 2u);
    EXPECT_EQ(ranked2[0], (std::vector<BlockId>{bid}));
    EXPECT_EQ(ranked2[1], (std::vector<BlockId>{bid}));

    // Unbound seqId should be safe and return empty.
    std::vector<RankedBlockId> rankedMissing;
    mgr.GetRankedBlockIds(999, rankedMissing);
    EXPECT_TRUE(rankedMissing.empty());

    // Fork should share the same request-scoped handle.
    auto child = std::make_shared<Sequence>(/*seqId*/ 3, /*blockSize*/ 8, std::vector<TokenId>{});
    mgr.Fork(s1, child);
    EXPECT_EQ(mgr.GetBlockIds(3)[0][0], bid);
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 3u);

    // Free all seqs -> release the request entry and return the block.
    mgr.Free(1);
    mgr.Free(2);
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 3u);
    mgr.Free(3);
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 4u);
    EXPECT_TRUE(mgr.requestEntries_.empty());
}

TEST(RequestSingleBlockManagerTest, ForkWithoutAllocateThrows)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 2, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    auto parent = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto child = std::make_shared<Sequence>(/*seqId*/ 2, /*blockSize*/ 8, std::vector<TokenId>{});
    EXPECT_THROW(mgr.Fork(parent, child), std::runtime_error);
}

TEST(RequestSingleBlockManagerTest, ConstructorValidatesConfig)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 1, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 0, /*host*/ 1};
    EXPECT_THROW((void)RequestSingleBlockManager(cfg), std::invalid_argument);

    cfg.rankSize = 1;
    cfg.hostSize = 0;
    EXPECT_THROW((void)RequestSingleBlockManager(cfg), std::invalid_argument);

    cfg.hostSize = 1;
    cfg.npuBlockNum = 0;
    EXPECT_THROW((void)RequestSingleBlockManager(cfg), std::invalid_argument);

    cfg.npuBlockNum = 1;
    cfg.reservedBlockNum = 2;
    EXPECT_THROW((void)RequestSingleBlockManager(cfg), std::invalid_argument);
}

TEST(RequestSingleBlockManagerTest, CanAllocateAndAllocateCoverEdgeBranches)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 2, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    EXPECT_EQ(mgr.CanAllocate(nullptr), AllocStatus::NEVER);
    EXPECT_FALSE(mgr.Allocate(nullptr));

    // No WAITING sequences => NEVER / false.
    auto running = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    running->status_ = SequenceStatus::RUNNING;
    auto gNoWaiting = MakeGroup("rq_nowait", {running});
    EXPECT_EQ(mgr.CanAllocate(gNoWaiting), AllocStatus::NEVER);
    EXPECT_FALSE(mgr.Allocate(gNoWaiting));
}

TEST(RequestSingleBlockManagerTest, CanAllocateReturnsNeverWhenAllNpuReserved)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 1, /*reserved*/ 1, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    auto s1 = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g1 = MakeGroup("rq", {s1});
    EXPECT_EQ(mgr.CanAllocate(g1), AllocStatus::NEVER);
}

TEST(RequestSingleBlockManagerTest, CanAllocateReturnsLaterWhenNoFreeBlockForNewRequest)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 1, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    auto s1 = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g1 = MakeGroup("rq1", {s1});
    EXPECT_TRUE(mgr.Allocate(g1));
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 0u);

    auto s2 = std::make_shared<Sequence>(/*seqId*/ 2, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g2 = MakeGroup("rq2", {s2});
    EXPECT_EQ(mgr.CanAllocate(g2), AllocStatus::LATER);
}

TEST(RequestSingleBlockManagerTest, AppendAndSwapApisCoverBranches)
{
    BlockManagerConfig cfg{4, /*cpu*/ 2, /*npu*/ 2, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    EXPECT_FALSE(mgr.CanAppendSlot(nullptr));
    EXPECT_FALSE(mgr.CanAppendSlotNew(nullptr));
    EXPECT_TRUE(mgr.AppendSlot(nullptr).empty());
    mgr.AppendSlotNew(nullptr);
    mgr.AppendTokenToLatestRank(0, std::vector<TokenId>{1});

    auto s1 = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g1 = MakeGroup("rq", {s1});
    EXPECT_FALSE(mgr.CanAppendSlot(g1)); // not allocated yet

    EXPECT_TRUE(mgr.Allocate(g1));
    EXPECT_TRUE(mgr.CanAppendSlot(g1));
    EXPECT_TRUE(mgr.CanAppendSlotNew(g1));
    EXPECT_TRUE(mgr.AppendSlot(s1).empty());

    EXPECT_FALSE(mgr.CanSwapOut(g1));
    EXPECT_TRUE(mgr.SwapOut(g1).empty());
    EXPECT_EQ(mgr.CanSwapIn(g1, 0), AllocStatus::NEVER);
    EXPECT_TRUE(mgr.SwapIn(g1).empty());

    EXPECT_EQ(mgr.GetNumFreeCpuBlocks(), 2u);
}

TEST(RequestSingleBlockManagerTest, FreeCoversRefCountAndInconsistentStateBranch)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 2, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    // Unknown seqId is safe.
    mgr.Free(999);

    // Allocate with two waiting sequences => refCount==2, free one keeps entry.
    auto s1 = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto s2 = std::make_shared<Sequence>(/*seqId*/ 2, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g = MakeGroup("rq", {s1, s2});
    EXPECT_TRUE(mgr.Allocate(g));
    ASSERT_EQ(mgr.requestEntries_.count("rq"), 1u);
    EXPECT_EQ(mgr.requestEntries_.at("rq").refCount, 2u);
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 1u);

    mgr.Free(1);
    ASSERT_EQ(mgr.requestEntries_.count("rq"), 1u);
    EXPECT_EQ(mgr.requestEntries_.at("rq").refCount, 1u);
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 1u);

    mgr.Free(2);
    EXPECT_TRUE(mgr.requestEntries_.empty());
    EXPECT_EQ(mgr.GetNumFreeNpuBlocks(), 2u);

    // Inconsistent state: seqId bound, but request entry missing => erase mapping branch.
    mgr.seqId2RequestId_[123] = "rq_missing";
    EXPECT_TRUE(mgr.requestEntries_.empty());
    mgr.Free(123);
    EXPECT_TRUE(mgr.seqId2RequestId_.empty());
}

TEST(RequestSingleBlockManagerTest, ForkCoversInvalidArgsAndMissingBlockBranch)
{
    BlockManagerConfig cfg{4, /*cpu*/ 0, /*npu*/ 2, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 1, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    SequenceSPtr nullSeq;
    auto child = std::make_shared<Sequence>(/*seqId*/ 2, /*blockSize*/ 8, std::vector<TokenId>{});
    EXPECT_THROW(mgr.Fork(nullSeq, child), std::invalid_argument);

    auto parent = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g = MakeGroup("rq", {parent});
    EXPECT_TRUE(mgr.Allocate(g));

    // Force missing block to cover error branch.
    mgr.requestEntries_.at("rq").block.reset();
    EXPECT_THROW(mgr.Fork(parent, child), std::runtime_error);
}

TEST(RequestSingleBlockManagerTest, TrivialApisAreCallableForCoverage)
{
    BlockManagerConfig cfg{4, /*cpu*/ 2, /*npu*/ 2, /*reserved*/ 0, /*spec*/ 0, /*caching*/ false, /*rank*/ 2, /*host*/ 1};
    RequestSingleBlockManager mgr(cfg);

    auto s1 = std::make_shared<Sequence>(/*seqId*/ 1, /*blockSize*/ 8, std::vector<TokenId>{1});
    auto g1 = MakeGroup("rq", {s1});
    EXPECT_TRUE(mgr.Allocate(g1));

    EXPECT_FALSE(mgr.IsAppendBlock(1));
    EXPECT_EQ(mgr.GetLatestAppendedRankId(1), 0u);
    EXPECT_EQ(mgr.GetAppendedBlockRankId(1), 0u);
    EXPECT_EQ(mgr.GetTokenCountPerRank(1).size(), 2u);

    EXPECT_TRUE(mgr.GetRankedHashValues(1).empty());
    EXPECT_TRUE(mgr.GetSeqHashValues(1).empty());
    EXPECT_TRUE(mgr.GetCommonComputedBlockIds({s1}).empty());
    EXPECT_TRUE(mgr.GetAllrankComputedBlockNum({s1}).empty());
    EXPECT_TRUE(mgr.GetRemoteComputedBlockIds({s1}, 0, 1, "m").empty());
    std::vector<size_t> computed;
    EXPECT_TRUE(mgr.GetAllRankRemoteComputedBlockIds({s1}, computed, "m").empty());

    mgr.AccessAllblocksInSeq(s1, 0.0f);
    mgr.MarkBlocksAsComputed();
    const float hitRate = mgr.GetPrefixCacheHitRate();
    if (cfg.enableCaching) {
        EXPECT_GE(hitRate, 0.0f);
    } else {
        // Hashless allocator returns -1 to indicate prefix cache is not enabled / not applicable.
        EXPECT_EQ(hitRate, -1.0f);
    }
    (void)mgr.ResetPrefixCache();
    mgr.ReplaceTrailingPlaceHolder(s1, 0, 0);

    EXPECT_EQ(mgr.GetNumCachedTokens(s1), 0u);
    EXPECT_EQ(mgr.GetSeqNumCachedTokens(s1), 0u);

    // Missing seqId should be safe for ranked block ids APIs.
    std::vector<RankedBlockId> ranked;
    mgr.GetRankedBlockIds(999, ranked);
    EXPECT_TRUE(ranked.empty());
    std::vector<std::vector<BlockId>> ranked2;
    mgr.GetRankedBlockIds(999, ranked2);
    ASSERT_EQ(ranked2.size(), 2u);
    EXPECT_TRUE(ranked2[0].empty());
    EXPECT_TRUE(ranked2[1].empty());
}

} // namespace mindie_llm

