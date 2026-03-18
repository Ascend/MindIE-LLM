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
#include "fcfs_policy.h"

#include "log.h"
#include "math_utils.h"
#include "request_response/request_id.h"
#include "msServiceProfiler/msServiceProfiler.h"

using namespace std::chrono;

namespace mindie_llm {
void RunningOutputs::Clear()
{
    blocksToSwapOut_.clear();
    blocksToCopy_.clear();
    decodeSeqGroups_.clear();
    chunkedPrefillSeqGroups_.clear();
    preempted_.clear();
    swappedOut_.clear();

    decodeSeqGroupsList_.clear();
    prefillSeqGroupsList_.clear();
}

FcfsPolicy::FcfsPolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig, BlockSpaceManagerSPtr &blockManager)
    : policyHelper_(schedulerConfig, blockManager), schedulerConfig_(schedulerConfig), blockManager_(blockManager)
{
}

PolicyOutput FcfsPolicy::Apply(SchedulingBudget &budget, ISeqGroupCollectionSPtr &collection)
{
    // 1. preprocess collection
    PreprocessCollection(collection);

    // 2. schedule according to pdPriorityType
    switch (collection->pdPriorityType_) {
        case PDPriorityType::PREFILL_FIRST: return SchedulePrefill(budget);
        case PDPriorityType::DECODE_FIRST: return ScheduleDecode(budget);
        case PDPriorityType::MIX: return ScheduleChunkedPrefill(budget);
        default:
            throw std::runtime_error("the pdPriorityType is not supported. pdPriorityType = " +
                                     std::to_string(static_cast<uint8_t>(collection->pdPriorityType_)));
    }
}

void FcfsPolicy::PreprocessCollection(const ISeqGroupCollectionSPtr &collection) { queuesCollection_ = collection; }

PolicyOutput FcfsPolicy::SchedulePrefill(SchedulingBudget &budget)
{
    PrefillOutputs prefillOut;
    RunningOutputs runningOut;
    SwappedInOutputs swappedInOut;

    // 1. do fcfs in waiting queue
    auto spanApplyWaiting = PROF(L2, Domain("Schedule").SpanStart("ApplyToWaiting"));
    prefillOut = ApplyToWaitingQueue(budget);
    PROF(spanApplyWaiting.SpanEnd());

    // 2. build policy out
    return BuildPolicyOutput(prefillOut, runningOut, swappedInOut, budget);
}

PolicyOutput FcfsPolicy::ScheduleDecode(SchedulingBudget &budget)
{
    PrefillOutputs prefillOut;
    RunningOutputs runningOut;
    SwappedInOutputs swappedInOut;

    // 1. do fcfs in different queue
    auto spanApplyRunning = PROF(L2, Domain("Schedule").SpanStart("ApplyToRunning"));

    runningOut = ApplyToRunningQueue(budget);
    PROF(spanApplyRunning.SpanEnd());

    if (runningOut.preempted_.size() + runningOut.swappedOut_.size() == 0) {
        auto spanApplySwapped = PROF(L2, Domain("Schedule").SpanStart("ApplyToSwapped"));
        swappedInOut = ApplyToSwappedQueue(budget);
        PROF(spanApplySwapped.SpanEnd());
    }

    // 2. build policy out
    return BuildPolicyOutput(prefillOut, runningOut, swappedInOut, budget);
}

PolicyOutput FcfsPolicy::ScheduleChunkedPrefill(SchedulingBudget &budget)
{
    PrefillOutputs prefillOut;
    RunningOutputs runningOut;
    SwappedInOutputs swappedInOut;

    // 1. 创建 Statistics4PartialPrefill, 统计其中的长序列个数
    budget.statistics4PartialPrefill_ = Statistics4PartialPrefill::FromQueues(
        queuesCollection_->running_, queuesCollection_->waiting_, schedulerConfig_);

    // 2. chunked prefill是decode优先的，所以先处理running队列
    runningOut = ApplyToRunningQueue(budget, true);
    if (runningOut.preempted_.size() + runningOut.swappedOut_.size() == 0) {
        swappedInOut = ApplyToSwappedQueue(budget, true);
    }

    // 3. 然后处理waiting队列
    prefillOut = ApplyToWaitingQueue(budget, true);

    // build policy out
    return BuildPolicyOutput(prefillOut, runningOut, swappedInOut, budget);
}

PolicyOutput FcfsPolicy::BuildPolicyOutput(PrefillOutputs &prefillOut, RunningOutputs &runningOut,
                                           SwappedInOutputs &swappedInOut, SchedulingBudget &budget) const
{
    PolicyOutput policyOut;

    // leftovers in policy input collection should be withdrawn
    WithdrawLeftovers(policyOut.withdrewSeqGroups_, queuesCollection_->waiting_);
    WithdrawLeftovers(policyOut.withdrewSeqGroups_, queuesCollection_->swapped_);
    WithdrawLeftovers(policyOut.withdrewSeqGroups_, queuesCollection_->running_);

    policyOut.preemptedSeqGroups_ = runningOut.preempted_;
    policyOut.prefillSeqGroups_ = prefillOut.seqGroups_;

    // 对于chunked prefill，swappedInOut.prefillSeqGroups_和runningOut.prefillSeqGroupsList也会有数据
    if (!swappedInOut.chunkedPrefillSeqGroups_.empty()) {
        policyOut.prefillSeqGroups_.insert(policyOut.prefillSeqGroups_.end(),
                                           swappedInOut.chunkedPrefillSeqGroups_.begin(),
                                           swappedInOut.chunkedPrefillSeqGroups_.end());
    }
    if (!runningOut.prefillSeqGroupsList_.empty()) {
        policyOut.prefillSeqGroups_.insert(policyOut.prefillSeqGroups_.end(),
                                           runningOut.chunkedPrefillSeqGroups_.begin(),
                                           runningOut.chunkedPrefillSeqGroups_.end());
    }

    policyOut.decodeSeqGroups_ = runningOut.decodeSeqGroups_;
    if (!swappedInOut.decodeSeqGroups_.empty()) {
        policyOut.decodeSeqGroups_.insert(policyOut.decodeSeqGroups_.end(), swappedInOut.decodeSeqGroups_.begin(),
                                          swappedInOut.decodeSeqGroups_.end());
    }

    policyOut.swappedOutSeqGroups_ = runningOut.swappedOut_;
    policyOut.ignoredSeqGroups_ = prefillOut.ignoredSeqGroups_;
    if (!swappedInOut.infeasibleSeqGroups_.empty()) {
        policyOut.ignoredSeqGroups_.insert(policyOut.ignoredSeqGroups_.end(), swappedInOut.infeasibleSeqGroups_.begin(),
                                           swappedInOut.infeasibleSeqGroups_.end());
    }

    policyOut.blocksToSwapIn_ = std::move(swappedInOut.blocksToSwapIn_);
    policyOut.blocksToSwapOut_ = runningOut.blocksToSwapOut_;
    policyOut.blocksToCopy_ = runningOut.blocksToCopy_;
    if (!swappedInOut.blocksToCopy_.empty()) {
        policyOut.blocksToCopy_.insert(policyOut.blocksToCopy_.end(), swappedInOut.blocksToCopy_.begin(),
                                       swappedInOut.blocksToCopy_.end());
    }

    policyOut.numPrefillGroups = prefillOut.seqGroups_.size();
    policyOut.numBatchedTokens = budget.numBatchedTokens_ + budget.numCachedTokens_;
    policyOut.numPreempted = runningOut.preempted_.size() + runningOut.swappedOut_.size();

    return policyOut;
}

PrefillOutputs FcfsPolicy::ApplyToWaitingQueue(SchedulingBudget &budget, const bool enableChunking)
{
    std::vector<SequenceGroupSPtr> ignoredSeqGroups;
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> seqGroups;
    std::deque<SequenceGroupSPtr> leftOverSeqGroups;

    while (!queuesCollection_->waiting_.empty()) {
        SequenceGroupSPtr seqGroup = queuesCollection_->waiting_.front();
        if (CeilDiv(static_cast<uint32_t>(seqGroup->firstSeq->GetLen()), schedulerConfig_->cacheBlockSize) ==
            blockManager_->GetNumFreeNpuBlocks() && seqGroup->firstSeq->data_.outputTokenIds.size() > 0) {
            break;
        }
        std::vector<SequenceSPtr> waitingSeqs = seqGroup->GetFirstSequence(SequenceStatus::WAITING);

        if (budget.statistics4PartialPrefill_ && !budget.statistics4PartialPrefill_->CanSchedule(seqGroup)) {
            leftOverSeqGroups.push_front(seqGroup);
            queuesCollection_->waiting_.pop_front();
            continue;
        }

        const auto [numNewTokensUncached, numNewTokensCached] = policyHelper_.GetNumComputeNewUnCachedAndCachedTokens(
            seqGroup, SequenceStatus::WAITING, enableChunking, budget);
        const size_t numNewTokens = numNewTokensUncached + numNewTokensCached;

        // 1. check the prompt limit
        const size_t promptLimit = policyHelper_.GetPromptLimit(seqGroup, budget);
        if (seqGroup->firstSeq->data_.outputTokenIds.size() == 0 && numNewTokens > promptLimit) {
            for (const auto &seq : waitingSeqs) {
                seq->status_ = SequenceStatus::FINISH_IGNORED;
            }
            ignoredSeqGroups.push_back(seqGroup);
            queuesCollection_->waiting_.pop_front();
            // 在Add request的时候已经做过一次长度不超过最大值的判断，理论在prefill过程中不应该再出现这种情况
            MINDIE_LLM_LOG_ERROR("Prompt token exceeds limit. seq id:" << seqGroup->firstSeq->seqId_
                                                                       << "; prompt len:" << numNewTokens
                                                                       << "; Prompt limit:" << promptLimit);
            continue;
        }

        // 2. check the prompt can be allocate
        const auto canAllocate = blockManager_->CanAllocate(seqGroup);
        if (canAllocate == AllocStatus::LATER) {
            // Allow new requests to preempt the current queue based on the configured maximum first token wait time.
            auto currentTime = std::chrono::high_resolution_clock::now();
            size_t reqArriveTime = static_cast<size_t>(duration_cast<milliseconds>(currentTime -
                seqGroup->arriveTime).count());
            size_t maxFirstTokenWaitTime = schedulerConfig_->maxFirstTokenWaitTime;
            if (seqGroup->firstSeq->data_.outputTokenIds.size() == 0 && reqArriveTime > maxFirstTokenWaitTime) {
                MINDIE_LLM_LOG_DEBUG_REQUEST("Request(requestId: " << seqGroup->metrics_.inferReqId_
                    << ") exceeded first token wait time (" << reqArriveTime << " > " << maxFirstTokenWaitTime
                    << ", preempting current request, Only effective in the PD mix scenario.");
                newRequestFirst_ = true;
            }
            break;
        } else if (canAllocate == AllocStatus::NEVER) {
            /** 当前mindie逻辑应该不存在这种场景
             * 1、如果prompt长度很长，则在add request的时候就做长度校验，返回添加失败。
             * 2、decode阶段，text generator判断长度等于maxSeqLen时就结束请求。maxSeqLen不可能比npu最大可用的内存大。
             *  */
            MINDIE_LLM_LOG_ERROR("Sequence can never be scheduled due to too long sequence len. seq id:"
                                 << seqGroup->firstSeq->seqId_ << "; seq len:" << seqGroup->firstSeq->GetLen());
            throw std::runtime_error("Prompt sequence too long.");
        }

        // 3. check the tokens and seqs
        const int numNewSeqs = seqGroup->GetMaxNumRunningSeqs();
        if (numNewTokensUncached == 0 || !budget.CanSchedule(numNewTokensUncached, numNewSeqs)) {
            break;
        }

        // 4. pop seqgroup and allocate, then update sequence status
        queuesCollection_->waiting_.pop_front();
        policyHelper_.AllocateAndSetRunning(seqGroup);

        if (budget.statistics4PartialPrefill_) {
            budget.statistics4PartialPrefill_->MaybeIncrementPartialPrefills(seqGroup);
        }

        // 5. update budget
        seqGroups.push_back(std::make_shared<ScheduledSequenceGroup>(seqGroup, numNewTokens, enableChunking));
        budget.AddNumBatchedTokens(seqGroup->requestId, numNewTokensUncached, numNewTokensCached);
        budget.AddNumSeqs(seqGroup->requestId, numNewSeqs);
    }

    // 6. withdraw leftovers data
    WithdrawLeftovers(queuesCollection_->waiting_, leftOverSeqGroups);

    return PrefillOutputs({std::move(seqGroups), std::move(ignoredSeqGroups)});
}

/**
 * beam search 新增加的采样选择seqgrp 分配block空间，TBC:不占用budget是否合理？
 */
bool FcfsPolicy::AllocBlocks4ParallelSeqGrp(SequenceGroupSPtr seqGroup,
                                            std::vector<std::pair<BlockId, BlockId>> &blockToCopy)
{
    if (!seqGroup->sampling->enableParallelSampling) {
        return true;
    }

    std::vector<SequenceId> parallelSeqIds = seqGroup->seqId2ParallelSeqGroup_.KeySet();
    for (auto seqId : parallelSeqIds) {
        std::optional<SequenceGroupSPtr> seqGrpOpt = seqGroup->seqId2ParallelSeqGroup_.Get(seqId);
        if (seqGrpOpt.has_value()) {
            SequenceGroupSPtr seqGrpSPtr = seqGrpOpt.value();
            if (!seqGrpSPtr->needUpdate_) {
                continue;
            }
            if (!seqGrpSPtr->isNewSeqGroup_) {
                // 第一次分裂的 seq 由于没有分配过，因此不需要 free，后续复用的 seq 需要 free 后重新分配，确保
                // blockmanager 中的 token 是正确的
                blockManager_->Free(seqGrpSPtr->firstSeq->seqId_);
                seqGrpSPtr->firstSeq->status_ = SequenceStatus::WAITING;
            }

            // 为需要更新的 seq 重新分配 block 空间
            auto canAllocate = blockManager_->CanAllocate(seqGrpSPtr);
            if (canAllocate == AllocStatus::LATER) {
                return false; // fcfs
            } else if (canAllocate == AllocStatus::NEVER) {
                throw std::runtime_error("Beam search sequence too long.");
            }
            seqGrpSPtr->isNewSeqGroup_ = false;
            policyHelper_.AllocateAndSetRunning(seqGrpSPtr);

            // 将 seq 加入到 SeqsBlocksComputedTracker 的 seqIdToBlocksHashes_ 和 seqIdToNumComputedTokens_ 中
            blockManager_->GetCommonComputedBlockIds({seqGrpSPtr->firstSeq});

            // 将父亲节点的block内容拷贝给子sequence
            std::vector<BlockId> parentBlockIds = seqGrpSPtr->parentBlockIds_[0];
            const auto allIds = blockManager_->GetBlockIds(seqGrpSPtr->firstSeq->seqId_);
            if (allIds.empty() || allIds[0].empty()) {
                throw std::runtime_error("Child sequence has no block ids after allocation.");
            }
            std::vector<BlockId> blockIds = allIds[0];
            if (parentBlockIds.size() > blockIds.size()) {
                throw std::runtime_error("Parent and child sequence block ids size mismatch. parent size: " +
                                         std::to_string(parentBlockIds.size()) +
                                         ", child size: " + std::to_string(blockIds.size()));
            }
            for (size_t i = 0; i < parentBlockIds.size(); ++i) {
                if (parentBlockIds[i] != blockIds[i]) {
                    blockToCopy.emplace_back(parentBlockIds[i], blockIds[i]);
                }
            }
            seqGrpSPtr->needUpdate_ = false;
        }
    }

    return true;
}

void FcfsPolicy::ScheduleRunningSeqGroup(const SequenceGroupSPtr &seqGroup, size_t numUncachedNewTokens,
                                         bool enableChunking, RunningOutputs &runningOutput, SchedulingBudget &budget)
{
    bool isSimulateInferenceSeq = seqGroup->IsSimulateRequest();
    if (!isSimulateInferenceSeq) {
        policyHelper_.AppendSlots(seqGroup, runningOutput.blocksToCopy_);
    }

    auto scheduledSeqGroup =
        std::make_shared<ScheduledSequenceGroup>(seqGroup, numUncachedNewTokens, enableChunking);
    if (seqGroup->IsPrefill()) {
        scheduledSeqGroup->tokenChunkSize_ = numUncachedNewTokens;
        runningOutput.chunkedPrefillSeqGroups_.emplace_back(scheduledSeqGroup);
        runningOutput.prefillSeqGroupsList_.emplace_back(seqGroup);
    } else {
        scheduledSeqGroup->tokenChunkSize_ = 1;
        runningOutput.decodeSeqGroups_.emplace_back(scheduledSeqGroup);
        runningOutput.decodeSeqGroupsList_.emplace_back(seqGroup);
    }

    budget.AddNumBatchedTokens(seqGroup->requestId, numUncachedNewTokens);
    budget.AddNumSeqs(seqGroup->requestId, seqGroup->GetMaxNumRunningSeqs());
}

RunningOutputs FcfsPolicy::ApplyToRunningQueue(SchedulingBudget &budget, const bool enableChunking)
{
    RunningOutputs runningOutput;
    size_t swapNum = 0;

    while (!queuesCollection_->running_.empty()) {
        SequenceGroupSPtr seqGroup = queuesCollection_->running_.front();

        const auto numUncachedNewTokens = std::get<0>(policyHelper_.GetNumComputeNewUnCachedAndCachedTokens(
            seqGroup, SequenceStatus::RUNNING, enableChunking, budget));
        if (!budget.CanSchedule(numUncachedNewTokens, seqGroup->GetMaxNumRunningSeqs())) {
            break;
        }

        queuesCollection_->running_.pop_front();
        bool canAppend = true;
        bool isSimulateInferenceSeq = seqGroup->IsSimulateRequest();
        // 是否需要抢占分两种情况：1. beamsearch请求新的seq是否可以alloc 2. 其他seq是否可以append
        while (!isSimulateInferenceSeq && (!AllocBlocks4ParallelSeqGrp(seqGroup, runningOutput.blocksToCopy_) ||
               !policyHelper_.CanAppendSlots(seqGroup) || newRequestFirst_)) {
            // 2. try to pop back seqgroup of running queue to preempt
            SequenceGroupSPtr victmSeqGroup;
            if (!queuesCollection_->running_.empty()) {
                victmSeqGroup = queuesCollection_->running_.back();
                queuesCollection_->running_.pop_back();
            } else {
                victmSeqGroup = seqGroup;
                canAppend = false;
            }

            // 3. do preempt
            const auto preemptedMode = Preempt(victmSeqGroup, runningOutput.blocksToSwapOut_, swapNum);
            if (preemptedMode == PreemptionMode::RECOMPUTE) {
                runningOutput.preempted_.emplace_back(victmSeqGroup);
            } else {
                runningOutput.swappedOut_.emplace_back(victmSeqGroup);
            }
            newRequestFirst_ = false;

            if (!canAppend) {
                break;
            }
        }

        // 4. append slot and update budget
        if (canAppend) {
            ScheduleRunningSeqGroup(seqGroup, numUncachedNewTokens, enableChunking, runningOutput, budget);
        }
    }

    return runningOutput;
}

SwappedInOutputs FcfsPolicy::ApplyToSwappedQueue(SchedulingBudget &budget, const bool enableChunking)
{
    SwappedInOutputs outputs;

    while (!queuesCollection_->swapped_.empty()) {
        SequenceGroupSPtr seqGroup = queuesCollection_->swapped_.front();

        // 1. check can be swap in
        const auto allocStatus = policyHelper_.CanSwapIn(seqGroup);
        if (allocStatus == AllocStatus::LATER) {
            break;
        } else if (allocStatus == AllocStatus::NEVER) {
            /** 当前mindie逻辑应该不存在这种场景
             * 1、如果prompt长度很长，则在add request的时候就做长度校验，返回添加失败。
             * 2、decode阶段，text generator判断长度等于maxSeqLen时就结束请求。maxSeqLen不可能比npu最大可用的内存大。
             *  */
            throw std::runtime_error("Sequence too long.");
        }

        if (!AllocBlocks4ParallelSeqGrp(seqGroup, outputs.blocksToCopy_)) {
            break;
        }

        const auto [numNewTokensUncached, numNewTokensCached] = policyHelper_.GetNumComputeNewUnCachedAndCachedTokens(
            seqGroup, SequenceStatus::SWAPPED, enableChunking, budget);

        // 2. check can be schedule
        int numNewSeqs = seqGroup->GetMaxNumRunningSeqs();
        if (numNewTokensUncached == 0 || !budget.CanSchedule(numNewTokensUncached, numNewSeqs)) {
            break;
        }

        // 3. do swap in
        queuesCollection_->swapped_.pop_front();
        policyHelper_.SwapIn(seqGroup, outputs.blocksToSwapIn_);
        policyHelper_.AppendSlots(seqGroup, outputs.blocksToCopy_);

        // 4. add to seqgroups
        if (seqGroup->IsPrefill()) {
            outputs.chunkedPrefillSeqGroups_.emplace_back(std::make_shared<ScheduledSequenceGroup>(
                seqGroup, numNewTokensUncached + numNewTokensCached, enableChunking));
        } else {
            outputs.decodeSeqGroups_.emplace_back(
                std::make_shared<ScheduledSequenceGroup>(seqGroup, 1, enableChunking));
        }
        budget.AddNumBatchedTokens(seqGroup->requestId, numNewTokensUncached, numNewTokensCached);
    }

    return outputs;
}

PreemptionMode FcfsPolicy::Preempt(SequenceGroupSPtr &seqGroup,
                                   std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut,
                                   size_t &swapNum)
{
    PreemptionMode preemptionMode = PreemptionMode::RECOMPUTE;
    if (swapNum < schedulerConfig_->maxPreemptCount && policyHelper_.CanSwapOut(seqGroup)) {
        preemptionMode = PreemptionMode::SWAP;
        swapNum++;
    }

    MINDIE_LLM_LOG_INFO_REQUEST("Preemption is triggered. CumulativePreemptionNum:"
                                << (numCumulativePreemption_ + 1)
                                << "; seqId: " << seqGroup->firstSeq->seqId_
                                << "; requestId: " << seqGroup->metrics_.inferReqId_
                                << "; preempt mode:" << static_cast<int>(preemptionMode)
                                << "; maxPreemptCount config:" << schedulerConfig_->maxPreemptCount
                                << "; swapNum:" << swapNum);
    numCumulativePreemption_ += 1;

    if (newRequestFirst_) {
        MINDIE_LLM_LOG_WARN_REQUEST("Preemption is triggered to ensure that some requests with long waiting time "
                                    "can be scheduled with priority.");
    }

    // do preempt
    if (preemptionMode == PreemptionMode::RECOMPUTE) {
        PreemptByRecompute(seqGroup);
    } else {
        PreemptBySwap(seqGroup, blockToSwapOut);
    }

    return preemptionMode;
}

void FcfsPolicy::PreemptByRecompute(const SequenceGroupSPtr &seqGroup)
{
    std::vector<SequenceSPtr> seqs = seqGroup->GetSequences(SequenceStatus::RUNNING);
    if (!seqGroup->sampling->enableParallelSampling && seqs.size() != 1) {
        throw std::runtime_error("the size of seqs must be 1.");
    }

    for (auto &seq : seqs) {
        seq->status_ = SequenceStatus::WAITING;
        policyHelper_.FreeSeq(seq);
        seq->ResetStateForCompute();
    }
}

void FcfsPolicy::PreemptBySwap(SequenceGroupSPtr &seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut)
{
    policyHelper_.SwapOut(seqGroup, blockToSwapOut);
}

void FcfsPolicy::WithdrawLeftovers(std::deque<SequenceGroupSPtr> &dst, const std::deque<SequenceGroupSPtr> &src)
{
    for (auto &seqGroup : src) {
        dst.push_back(seqGroup);
    }
}
} // namespace mindie_llm
