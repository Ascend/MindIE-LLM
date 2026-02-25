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
#include <cstdlib>
#include <string>
#include "log.h"
#include "math_utils.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "layerwise_fcfs_policy.h"

using namespace std::chrono;

namespace mindie_llm {
LayerwiseFcfsPolicy::LayerwiseFcfsPolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig,
    BlockSpaceManagerSPtr &blockManager) : FcfsPolicy(schedulerConfig, blockManager)
{
    const char* longprefillLenThreshold = std::getenv("LONG_PREFILL_LEN_THRESHOLD");
    if (longprefillLenThreshold != nullptr) {
        longprefillLenThreshold_ = static_cast<unsigned int>(std::stoi(longprefillLenThreshold));
    }

    const char* dynamicPBSEnable = std::getenv("DYNAMIC_PBS_ENABLE");
    if (dynamicPBSEnable != nullptr) {
        dynamicPBSEnable_ = std::string(dynamicPBSEnable) == "1";
    }
}

void LayerwiseFcfsPolicy::setnewRequestFirst(SequenceGroupSPtr seqGroup)
{
    auto currentTime = std::chrono::high_resolution_clock::now();
    size_t reqArriveTime = static_cast<size_t>(duration_cast<milliseconds>(currentTime -
        seqGroup->arriveTime).count());
    size_t maxFirstTokenWaitTime = schedulerConfig_->maxFirstTokenWaitTime;
    if (seqGroup->firstSeq->data_.outputTokenIds.size() == 0 && reqArriveTime > maxFirstTokenWaitTime) {
        newRequestFirst_ = true;
    }
}

int LayerwiseFcfsPolicy::HandleDynamicPBS(int promptTokenIdsLen)
{
    // 累加batch长度，如果超过限制，立刻下发
    if (promptTokenIdsLen >= middleprefillLenThreshold_) {
        curBatchLenBudget_ = middleBatchLenBudget_; // 遇到1K输入限制batch总长度为2048
    }
    curBatchLen_ += promptTokenIdsLen;
    if (dynamicPBSEnable_ && curBatchLen_ >= curBatchLenBudget_) {
        return -1; // break
    }
    return 0;
}

int LayerwiseFcfsPolicy::GeneratePrefillBatch(SchedulingBudget &budget,
    std::vector<SequenceGroupSPtr> &ignoredSeqGroups, std::deque<SequenceGroupSPtr> &leftOverSeqGroups,
    int curCount, std::vector<std::shared_ptr<ScheduledSequenceGroup>> &seqGroups)
{
    SequenceGroupSPtr seqGroup = queuesCollection_->waiting_.front();
    if (CeilDiv(static_cast<uint32_t>(seqGroup->firstSeq->GetLen()), schedulerConfig_->cacheBlockSize) ==
        blockManager_->GetNumFreeNpuBlocks() && seqGroup->firstSeq->data_.outputTokenIds.size() > 0) {
        return -1; // break
    }
    std::vector<SequenceSPtr> waitingSeqs = seqGroup->GetFirstSequence(SequenceStatus::WAITING);

    SequenceData &data = seqGroup->firstSeq->data_;
    auto promptTokenIdsLen = data.layerwiseRecompute_ ? data.GetLength() : data.promptTokenIds.size();
    MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|FcfsPolicy] " << "waiting, current count is " << curCount <<
    ", seqId is " << seqGroup->firstSeq->seqId_  << ", promptTokenIdsLen is " << promptTokenIdsLen <<
    ", waiting_.size is " << queuesCollection_->waiting_.size());
    
    MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|FcfsPolicy] " <<
    "layerwiseStage_ is " << static_cast<int>(seqGroup->firstSeq->data_.layerwiseStage_) <<
    ", layerwiseRecompute_ is " << seqGroup->firstSeq->data_.layerwiseRecompute_ <<
    ", layerwiseRecomputeReturn_ is " << seqGroup->firstSeq->data_.layerwiseRecomputeReturn_);

    if (budget.statistics4PartialPrefill_ && !budget.statistics4PartialPrefill_->CanSchedule(seqGroup)) {
        leftOverSeqGroups.push_front(seqGroup);
        queuesCollection_->waiting_.pop_front();
        return 1; // continue
    }

    return GeneratePrefillBatchInner(budget, ignoredSeqGroups, curCount, seqGroups,
        seqGroup, waitingSeqs, promptTokenIdsLen);
}

int LayerwiseFcfsPolicy::GeneratePrefillBatchInner(SchedulingBudget &budget,
    std::vector<SequenceGroupSPtr> &ignoredSeqGroups,
    int curCount, std::vector<std::shared_ptr<ScheduledSequenceGroup>> &seqGroups,
    SequenceGroupSPtr seqGroup, std::vector<SequenceSPtr> waitingSeqs, size_t promptTokenIdsLen)
{
    const auto [numNewTokensUncached, numNewTokensCached] = policyHelper_.GetNumComputeNewUnCachedAndCachedTokens(
        seqGroup, SequenceStatus::WAITING, enableChunking_, budget);
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
        return 1; // continue
    }

    // 2. check the prompt can be allocate
    const auto canAllocate = blockManager_->CanAllocate(seqGroup);
    if (canAllocate == AllocStatus::LATER) {
        // Allow new requests to preempt the current queue based on the configured maximum first token wait time.
        setnewRequestFirst (seqGroup);
        return -1; // break
    } else if (canAllocate == AllocStatus::NEVER) {
        /** 当前mindie逻辑应该不存在这种场景
            * 1、如果prompt长度很长，则在add request的时候就做长度校验，返回添加失败。
            * 2、decode阶段，text generator判断长度等于maxSeqLen时就结束请求。maxSeqLen不可能比npu最大可用的内存大。
            *  */
        throw std::runtime_error("Prompt sequence too long.");
    }

    // 3. check the tokens and seqs
    const int numNewSeqs = seqGroup->GetMaxNumRunningSeqs();
    if (numNewTokensUncached == 0 || !budget.CanSchedule(numNewTokensUncached, numNewSeqs)) {
        return -1; // break
    }

    // 4. pop seqgroup and allocate, then update sequence status
    queuesCollection_->waiting_.pop_front();

    // 缓存超长prefill
    if (curCount > 0 && promptTokenIdsLen >= longprefillLenThreshold_) {
        longprefillQueue_.emplace_back(seqGroup);
        return 1; // continue
    }

    // 缓存重计算且未return prefill的seq
    if (seqGroup->firstSeq->data_.layerwiseStage_ == SequenceStage::PREFILL && \
        seqGroup->firstSeq->data_.layerwiseRecompute_ && !seqGroup->firstSeq->data_.layerwiseRecomputeReturn_) {
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|FcfsPolicy] " << "recompute but not return");
        recomputeprefillQueue_.emplace_back(seqGroup);
        return 1; // continue
    }

    policyHelper_.AllocateAndSetRunning(seqGroup);
    if (budget.statistics4PartialPrefill_) {
        budget.statistics4PartialPrefill_->MaybeIncrementPartialPrefills(seqGroup);
    }

    // 5. update budget
    seqGroups.push_back(std::make_shared<ScheduledSequenceGroup>(seqGroup, numNewTokens, enableChunking_));
    budget.AddNumBatchedTokens(seqGroup->requestId, numNewTokensUncached, numNewTokensCached);
    budget.AddNumSeqs(seqGroup->requestId, numNewSeqs);

    // 设置超长序列prefill batchsize为1
    if (curCount == 0 && promptTokenIdsLen >= longprefillLenThreshold_) { return -1; } // break
    // 处理dynamic prefill batch size
    if (HandleDynamicPBS(promptTokenIdsLen) == -1) { return -1; } // break

    return 0;
}

// 请求调度修改,边云重写prefill调度方法
PrefillOutputs LayerwiseFcfsPolicy::ApplyToWaitingQueue(SchedulingBudget &budget, const bool enableChunking)
{
    std::vector<SequenceGroupSPtr> ignoredSeqGroups;
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> seqGroups;
    std::deque<SequenceGroupSPtr> leftOverSeqGroups;

    ((void)enableChunking);

    // 组prefill batch重置变量
    int curCount = 0;
    curBatchLen_ = 0;
    curBatchLenBudget_ = initBatchLenBudget_;

    while (!queuesCollection_->waiting_.empty()) {
        int result = GeneratePrefillBatch (budget, ignoredSeqGroups, leftOverSeqGroups, curCount, seqGroups);
        if (result == -1) {
            break;
        } else if (result == 1) {
            continue;
        }

        // 请求调度修改,设置超长序列prefill batchsize为1
        curCount++;
    }

    // 6. withdraw leftovers data
    WithdrawLeftovers(queuesCollection_->waiting_, leftOverSeqGroups);

    // 请求调度修改,回填超长prefill
    for (auto it = longprefillQueue_.rbegin(); it != longprefillQueue_.rend(); ++it) {
        queuesCollection_->waiting_.push_front(*it);
    }
    longprefillQueue_.clear();

    // 请求调度修改,回填重计算prefill
    for (auto it = recomputeprefillQueue_.rbegin(); it != recomputeprefillQueue_.rend(); ++it) {
        queuesCollection_->waiting_.push_front(*it);
    }
    recomputeprefillQueue_.clear();

    return PrefillOutputs({std::move(seqGroups), std::move(ignoredSeqGroups)});
}

void LayerwiseFcfsPolicy::appendSlots(SequenceGroupSPtr seqGroup, RunningOutputs &runningOutput,
    const size_t numUncachedNewTokens, const bool enableChunking)
{
    policyHelper_.AppendSlots(seqGroup, runningOutput.blocksToCopy_);

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
}

RunningOutputs LayerwiseFcfsPolicy::ApplyToRunningQueue(SchedulingBudget &budget, const bool enableChunking)
{
    RunningOutputs runningOutput;
    size_t swapNum = 0;

    std::deque<SequenceGroupSPtr> prefillRunningQueue;
    while (!queuesCollection_->running_.empty()) {
        SequenceGroupSPtr seqGroup = queuesCollection_->running_.front();

        const auto numUncachedNewTokens = std::get<0>(policyHelper_.GetNumComputeNewUnCachedAndCachedTokens(
            seqGroup, SequenceStatus::RUNNING, enableChunking, budget));
        if (!budget.CanSchedule(numUncachedNewTokens, seqGroup->GetMaxNumRunningSeqs())) {
            break;
        }

        queuesCollection_->running_.pop_front();
        bool canAppend = true;
        while (!policyHelper_.CanAppendSlots(seqGroup) ||
               !AllocBlocks4ParallelSeqGrp(seqGroup, runningOutput.blocksToCopy_)) {
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
            if (!canAppend) {
                break;
            }
        }

        // 如果是isprefill，缓存下来
        if (seqGroup != nullptr && seqGroup->IsLayerwisePrefill() && canAppend) {
            prefillRunningQueue.emplace_back(seqGroup);
            continue;
        }

        // 4. append slot and update budget
        if (canAppend) {
            appendSlots(seqGroup, runningOutput, numUncachedNewTokens, enableChunking);

            budget.AddNumBatchedTokens(seqGroup->requestId, numUncachedNewTokens);
            budget.AddNumSeqs(seqGroup->requestId, seqGroup->GetMaxNumRunningSeqs());
        }
    }

    for (auto it = prefillRunningQueue.rbegin(); it != prefillRunningQueue.rend(); ++it) {
        queuesCollection_->running_.push_front(*it);
    }

    return runningOutput;
}

}