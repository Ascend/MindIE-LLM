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
#include "scheduling_budget.h"
#include "utils/log.h"

namespace mindie_llm {
SchedulingBudget::SchedulingBudget(const size_t maxNumBatchedTokens, const size_t maxNumSeqs,
                                   const SchedulerConfigSPtr &schedulerConfig)
    : maxNumSeqs_(maxNumSeqs), maxNumBatchedTokens_(maxNumBatchedTokens), schedulerConfig_(schedulerConfig)
{
    // chunked prefill开启时，需更改maxNumSeqs_ maxNumBatchedTokens_并初始化partialPrefillOccupiedBudgetPerSlot_
    if (schedulerConfig_ != nullptr && schedulerConfig_->enableChunkedPrefill) {
        maxNumBatchedTokens_ = schedulerConfig_->maxPrefillTokens;
        maxNumSeqs_ = schedulerConfig_->maxBatchSize;
        const size_t size = schedulerConfig_->maxNumPartialPrefills + 1;
        partialPrefillOccupiedBudgetPerSlot_.resize(size, 0);
        partialPrefillOccupiedBudgetPerSlot_[0] = schedulerConfig_->maxPrefillTokens;
        for (size_t i = 1; i < size; i++) {
            partialPrefillOccupiedBudgetPerSlot_[i] = schedulerConfig_->maxPrefillTokens / i;
        }
    }
}

bool SchedulingBudget::CanSchedule(const size_t numNewTokens, const size_t numNewSeqs) const
{
    return numBatchedTokens_ + numNewTokens <= maxNumBatchedTokens_ && numCurSeqs_ + numNewSeqs <= maxNumSeqs_;
}

size_t SchedulingBudget::RemainingTokenBudget() const
{
    if (numBatchedTokens_ > maxNumBatchedTokens_) {
        return 0; // 防止下溢出，返回0
    }
    return maxNumBatchedTokens_ - numBatchedTokens_;
}

size_t SchedulingBudget::GetPrefillSlots()
{
    size_t remainingTokenBudget = RemainingTokenBudget();

    // 非chunked_
    if (schedulerConfig_ == nullptr || !schedulerConfig_->enableChunkedPrefill) {
        return remainingTokenBudget;
    }

    // 固定切分长度
    if (schedulerConfig_->prefillChunkSize != 0) { return schedulerConfig_->prefillChunkSize; }

    // 动态切分长度
    if (statistics4PartialPrefill_ == nullptr) {
        return remainingTokenBudget;
    }

    size_t numSchedulablePrefills = statistics4PartialPrefill_->SchedulablePrefills();
    if (partialPrefillOccupiedBudgetPerSlot_.size() <= numSchedulablePrefills) {
        MINDIE_LLM_LOG_ERROR("index is out of range of partialPrefillOccupiedBudgetPerSlot_. index="
                             << numSchedulablePrefills);
        throw std::runtime_error("index is out of range of partialPrefillOccupiedBudgetPerSlot_. index=" +
                                 std::to_string(numSchedulablePrefills));
    }

    return partialPrefillOccupiedBudgetPerSlot_[numSchedulablePrefills];
}

void SchedulingBudget::AddNumBatchedTokens(RequestId &reqId, const size_t numBatchedTokens,
                                           const size_t numCachedTokens)
{
    if (requestIdsNumBatchedTokens_.count(reqId) != 0) {
        return;
    }
    requestIdsNumBatchedTokens_.emplace(reqId);
    numBatchedTokens_ += numBatchedTokens;
    numCachedTokens_ += numCachedTokens;
}

void SchedulingBudget::AddNumSeqs(RequestId &reqId, const size_t numCurSeqs)
{
    if (requestIdsNumCurSeqs_.find(reqId) != requestIdsNumCurSeqs_.end()) {
        return;
    }

    requestIdsNumCurSeqs_.emplace(reqId);
    numCurSeqs_ += numCurSeqs;
}

void SchedulingBudget::SubtractNumBatchedTokens(const RequestId &reqId, const size_t numBatchedTokens)
{
    if (requestIdsNumBatchedTokens_.find(reqId) != requestIdsNumBatchedTokens_.end()) {
        requestIdsNumBatchedTokens_.erase(reqId);
        numBatchedTokens_ -= numBatchedTokens;
    }
}

void SchedulingBudget::SubtractNumSeqs(const RequestId &reqId, const size_t numCurSeqs)
{
    if (requestIdsNumCurSeqs_.find(reqId) != requestIdsNumCurSeqs_.end()) {
        requestIdsNumCurSeqs_.erase(reqId);
        numCurSeqs_ -= numCurSeqs;
    }
}

Statistics4PartialPrefill::Statistics4PartialPrefill(int numSchedulablePrefills, int numLongPrefills,
                                                     std::shared_ptr<SchedulerConfig> schedulerConfig)
    : numSchedulablePrefills_(numSchedulablePrefills), numLongPrefills_(numLongPrefills),
      schedulerConfig_(schedulerConfig)
{
}

bool Statistics4PartialPrefill::CanSchedule(const std::shared_ptr<SequenceGroup> &seqGroup) const
{
    // 如果是短序列，可以被调度
    if (seqGroup->firstSeq->GetNumUncomputedTokens() <= schedulerConfig_->longPrefillTokenThreshold) {
        return true;
    }

    // 如果是长序列，检查是否还有剩余的长prefill调度配额
    return numLongPrefills_ < schedulerConfig_->maxLongPartialPrefills;
}

void Statistics4PartialPrefill::MaybeIncrementPartialPrefills(const std::shared_ptr<SequenceGroup> &seqGroup)
{
    if (seqGroup->firstSeq->GetNumUncomputedTokens() > schedulerConfig_->longPrefillTokenThreshold) {
        numLongPrefills_++;
    }
}

Statistics4PartialPrefillPtr Statistics4PartialPrefill::FromQueues(
    const std::deque<std::shared_ptr<SequenceGroup>> &running,
    const std::deque<std::shared_ptr<SequenceGroup>> &waiting,
    std::shared_ptr<SchedulerConfig> schedulerConfig)
{
    size_t numPrefills = 0;
    size_t numLongPrefills = 0;
    size_t numWaitingLongPrefills = 0;

    for (const SequenceGroupSPtr &seqGroup : running) {
        if (seqGroup->firstSeq->data_.stage_ == SequenceStage::PREFILL) {
            numPrefills++;
            if (seqGroup->firstSeq->GetNumUncomputedTokens() > schedulerConfig->longPrefillTokenThreshold) {
                numLongPrefills++;
            }
        }
    }

    for (const SequenceGroupSPtr &seqGroup : waiting) {
        if (numPrefills >= schedulerConfig->maxNumPartialPrefills) {
            break;
        }

        if (seqGroup->firstSeq->GetNumUncomputedTokens() > schedulerConfig->longPrefillTokenThreshold) {
            if (numLongPrefills + numWaitingLongPrefills >= schedulerConfig->maxLongPartialPrefills) {
                continue;
            }
            numWaitingLongPrefills++;
        }
        numPrefills++;
    }

    return std::make_unique<Statistics4PartialPrefill>(
        std::min(numPrefills, static_cast<size_t>(schedulerConfig->maxNumPartialPrefills)),
        numLongPrefills,
        schedulerConfig);
}

size_t Statistics4PartialPrefill::SchedulablePrefills() const { return numSchedulablePrefills_; }

size_t Statistics4PartialPrefill::LongPrefills() const { return numLongPrefills_; }
} // namespace mindie_llm