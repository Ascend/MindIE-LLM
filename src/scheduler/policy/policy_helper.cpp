/*
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
#include "policy_helper.h"

#include "log.h"
#include "msServiceProfiler/msServiceProfiler.h"

#include <stdexcept>

namespace mindie_llm {
PolicyHelper::PolicyHelper(std::shared_ptr<SchedulerConfig> schedulerConfig, BlockSpaceManagerSPtr blockManager)
    : schedulerConfig_(schedulerConfig), blockManager_(blockManager)
{
    if (!schedulerConfig_) {
        throw std::invalid_argument("schedulerConfig cannot be null");
    }
    
    if (!blockManager_) {
        throw std::invalid_argument("blockManager cannot be null");
    }
}

std::pair<size_t, size_t> PolicyHelper::GetNumComputeNewUnCachedAndCachedTokens(const SequenceGroupSPtr seqGroup,
                                                                                const SequenceStatus status,
                                                                                bool enableChunking,
                                                                                SchedulingBudget &budget)
{
    validateSequenceGroup(seqGroup, "GetNumComputeNewUnCachedAndCachedTokens");

    size_t numCachedNewTokens = 0;
    size_t numUncachedNewTokens = 0;

    std::vector<SequenceSPtr> seqs = seqGroup->GetSequences(status);
    for (SequenceSPtr &seq : seqs) {
        if (!seq) {
            throw std::runtime_error("Found null sequence in sequence group");
        }

        if (!seq->IsPrefill()) {
            numUncachedNewTokens += 1;
            continue;
        }

        const size_t numComputedTokenSeq = seq->GetNumComputedTokens();
        if (seq->GetLen() < numComputedTokenSeq) {
            throw std::runtime_error("Computed token count exceeds sequence length");
        }
        const size_t allNumNewTokensSeq = seq->GetLen() - numComputedTokenSeq;
        if (!schedulerConfig_->enablePrefixCache) {
            numUncachedNewTokens += allNumNewTokensSeq;
            continue;
        }

        const size_t numCachedTokensSeq = blockManager_->GetSeqNumCachedTokens(seq);
        if (allNumNewTokensSeq == numCachedTokensSeq) {
            seqGroup->metrics_.prefixCachedTokenNum_ = numCachedTokensSeq - schedulerConfig_->cacheBlockSize;
        } else {
            seqGroup->metrics_.prefixCachedTokenNum_ = numCachedTokensSeq;
        }

        if (schedulerConfig_->spSize > 1 || schedulerConfig_->cpSize > 1) {
            numUncachedNewTokens += allNumNewTokensSeq;
            continue;
        }

        const size_t numCachedNewTokensSeq =
            numCachedTokensSeq >= numComputedTokenSeq ? numCachedTokensSeq - numComputedTokenSeq : 0;
        const size_t numUncachedNewTokensSeq = allNumNewTokensSeq - numCachedNewTokensSeq;
        numUncachedNewTokens += numUncachedNewTokensSeq;
        numCachedNewTokens += numCachedNewTokensSeq;
    }

    if (numUncachedNewTokens == 0 && numCachedNewTokens > 0) {
        numUncachedNewTokens = 1;
        numCachedNewTokens -= 1;
    }

    // Chunk if a running request cannot fit in the given budget
    if (enableChunking) {
        numUncachedNewTokens = GetNumComputeChunkSize(budget, numUncachedNewTokens);
    }

    return {numUncachedNewTokens, numCachedNewTokens};
}

size_t PolicyHelper::GetNumComputeChunkSize(SchedulingBudget &budget, size_t numNewTokens) const
{
    size_t remainingTokenBudget = budget.RemainingTokenBudget();

    // Get the number of tokens to allocate to this prefill slot
    size_t prefillSlotBudget = budget.GetPrefillSlots();

    return std::min({numNewTokens, remainingTokenBudget, prefillSlotBudget});
}

size_t PolicyHelper::GetPromptLimit(SequenceGroupSPtr seqGroup, SchedulingBudget &budget) const
{
    validateSequenceGroup(seqGroup, "GetPromptLimit");
    seqGroup = seqGroup;
    if (schedulerConfig_->enableChunkedPrefill) {
        return schedulerConfig_->maxSeqLen;
    } else {
        return std::min(schedulerConfig_->maxSeqLen, budget.maxNumBatchedTokens_);
    }
}

void PolicyHelper::AllocateAndSetRunning(SequenceGroupSPtr seqGroup) const
{
    validateSequenceGroup(seqGroup, "AllocateAndSetRunning");

    PROF(INFO, Domain("KVCache")
                    .Resource(seqGroup->requestId)
                    .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                    .MetricScope("dp", blockManager_->GetLocalDPRank())
                    .Event("Allocate"));
    blockManager_->Allocate(seqGroup);
    auto waitingSeqs = seqGroup->GetFirstSequence(SequenceStatus::WAITING);
    for (const auto &seq : waitingSeqs) {
        seq->status_ = SequenceStatus::RUNNING;
    }
}

bool PolicyHelper::CanAppendSlots(SequenceGroupSPtr seqGroup) const
{
    validateSequenceGroup(seqGroup, "CanAppendSlots");
    
    if (!seqGroup->sampling->enableParallelSampling) {
        return (schedulerConfig_->spSize * schedulerConfig_->cpSize > 1) ? blockManager_->CanAppendSlotNew(seqGroup)
                                                                         : blockManager_->CanAppendSlot(seqGroup);
    }

    return blockManager_->CanAppendSlot(seqGroup);
}

void PolicyHelper::AppendSlotForSeqs(std::vector<SequenceSPtr> parallelSeqs, RequestId reqId,
                                     std::vector<std::pair<BlockId, BlockId>> &blockToCopy)
{
    reqId = reqId;
    for (auto seq : parallelSeqs) {
        PROF(INFO, Domain("KVCache")
                        .Resource(reqId)
                        .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                        .MetricScope("dp", blockManager_->GetLocalDPRank())
                        .Event("AppendSlot"));
        const auto cows = blockManager_->AppendSlot(seq);
        if (!cows.empty()) {
            blockToCopy.insert(blockToCopy.end(), cows.begin(), cows.end());
        }
    }
}

void PolicyHelper::AppendSlots(const SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToCopy)
{
    validateSequenceGroup(seqGroup, "AppendSlots");

    // beam search 和SP不支持叠加
    if (schedulerConfig_->spSize * schedulerConfig_->cpSize > 1 && !seqGroup->sampling->enableParallelSampling) {
        blockManager_->AppendSlotNew(seqGroup);
    } else {
        std::vector<SequenceSPtr> Seqs = seqGroup->GetSequences(SequenceStatus::RUNNING);
        AppendSlotForSeqs(Seqs, seqGroup->requestId, blockToCopy);
    }
}

bool PolicyHelper::CanSwapOut(SequenceGroupSPtr seqGroup)
{
    validateSequenceGroup(seqGroup, "CanSwapOut");

    if (!seqGroup->sampling->enableParallelSampling) {
        return blockManager_->CanSwapOut(seqGroup);
    }

    // beam search 分支
    std::vector<SequenceGroupSPtr> seqGrps = seqGroup->GetParallelSeqGrp();
    for (auto seqGrp : seqGrps) {
        if (!blockManager_->CanSwapOut(seqGrp)) {
            return false;
        }
    }
    return true;
}

AllocStatus PolicyHelper::CanSwapIn(SequenceGroupSPtr seqGroup)
{
    validateSequenceGroup(seqGroup, "CanSwapIn");

    if (!seqGroup->sampling->enableParallelSampling) {
        return blockManager_->CanSwapIn(seqGroup, 0);
    }

    // beam search分支
    std::vector<SequenceGroupSPtr> seqGrps = seqGroup->GetParallelSeqGrp();
    for (auto seqGrp : seqGrps) {
        const auto allocStatus = blockManager_->CanSwapIn(seqGrp, 0);
        if (allocStatus == AllocStatus::LATER) {
            return allocStatus;
        } else if (allocStatus == AllocStatus::NEVER) {
            throw std::runtime_error("Sequence too long.");
        }
    }
    return AllocStatus::OK;
}

void PolicyHelper::SeqsSwapIn(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapIn) const
{
    validateSequenceGroup(seqGroup, "SeqsSwapIn");

    PROF(INFO, Domain("KVCache")
           .Resource(seqGroup->requestId)
           .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
           .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
           .MetricScope("dp", blockManager_->GetLocalDPRank())
           .Event("SwapIn"));
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> mapping = blockManager_->SwapIn(seqGroup);
    blockToSwapIn.insert(blockToSwapIn.end(), mapping.begin(), mapping.end());
    std::vector<SequenceSPtr> swappedSeqs = seqGroup->GetFirstSequence(SequenceStatus::SWAPPED);
    for (const auto &seq : swappedSeqs) {
        seq->status_ = SequenceStatus::RUNNING;
    }
}

void PolicyHelper::SwapIn(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapIn) const
{
    validateSequenceGroup(seqGroup, "SwapIn");

    PROF(INFO, Domain("KVCache")
                   .Resource(seqGroup->requestId)
                   .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                   .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
                   .MetricScope("dp", blockManager_->GetLocalDPRank())
                   .Event("SwapIn"));

    if (!seqGroup->sampling->enableParallelSampling) {
        SeqsSwapIn(seqGroup, blockToSwapIn);
        return;
    }

    // beam search 分支
    std::vector<SequenceGroupSPtr> seqGrps = seqGroup->GetParallelSeqGrp();
    for (auto seqGrp : seqGrps) {
        SeqsSwapIn(seqGroup, blockToSwapIn);
    }
}

void PolicyHelper::SeqsSwapOut(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut)
{
    validateSequenceGroup(seqGroup, "SeqsSwapOut");

    PROF(INFO, Domain("KVCache")
               .Resource(seqGroup->requestId)
               .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
               .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
               .MetricScope("dp", blockManager_->GetLocalDPRank())
               .Event("SwapOut"));
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> mapping = blockManager_->SwapOut(seqGroup);
    blockToSwapOut.insert(blockToSwapOut.end(), mapping.begin(), mapping.end());

    auto swappedSeqs = seqGroup->GetFirstSequence(SequenceStatus::RUNNING);
    for (const auto &seq : swappedSeqs) {
        seq->status_ = SequenceStatus::SWAPPED;
    }
}

void PolicyHelper::SwapOut(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut)
{
    validateSequenceGroup(seqGroup, "SwapOut");

    if (!CanSwapOut(seqGroup)) {
        throw std::runtime_error("Aborted due to the lack of CPU swap space. Please increase "
                                 "the swap space to avoid this error.");
    }
    PROF(INFO, Domain("KVCache")
                   .Resource(seqGroup->requestId)
                   .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                   .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
                   .MetricScope("dp", blockManager_->GetLocalDPRank())
                   .Event("SwapOut"));

    if (!seqGroup->sampling->enableParallelSampling) {
        SeqsSwapOut(seqGroup, blockToSwapOut);
        return;
    }

    // beam search 分支
    std::vector<SequenceGroupSPtr> seqGrps = seqGroup->GetParallelSeqGrp();
    for (auto seqGrp : seqGrps) {
        SeqsSwapOut(seqGrp, blockToSwapOut);
    }
}

void PolicyHelper::ForkSeq(SequenceSPtr parentSeq, SequenceSPtr &childSeq) const
{
    blockManager_->Fork(parentSeq, childSeq);
}

void PolicyHelper::FreeSeq(SequenceSPtr seq) const
{
    PROF(INFO, Domain("KVCache")
                    .Resource(std::to_string(seq->seqId_))
                    .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                    .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
                    .MetricScope("dp", blockManager_->GetLocalDPRank())
                    .Event("Free"));
    blockManager_->Free(seq->seqId_);
}

void PolicyHelper::FreeSeqGroup(SequenceGroupSPtr seqGroup) const
{
    validateSequenceGroup(seqGroup, "FreeSeqGroup");

    for (auto seq : seqGroup->seqs_) {
        PROF(INFO, Domain("KVCache")
                    .Resource(std::to_string(seq->seqId_))
                    .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                    .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
                    .MetricScope("dp", blockManager_->GetLocalDPRank())
                    .Event("Free"));
        blockManager_->Free(seq->seqId_);
    }
}

void PolicyHelper::validateSequenceGroup(const SequenceGroupSPtr& seqGroup, const std::string& context) const
{
    if (!seqGroup) {
        throw std::invalid_argument("SequenceGroup cannot be null in " + context);
    }
}
} // namespace mindie_llm
