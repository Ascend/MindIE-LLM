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

#include "log.h"
#include "policy/seq_group_collection.h"
#include "stage_policy/stage_policy.h"
#include "dynamic_batch_size.h"

namespace mindie_llm {
DynamicBatchSize::DynamicBatchSize(const SchedulerConfigSPtr schedulerConfig,
                                   std::shared_ptr<LatencyPredictor> predictor,
                                   std::shared_ptr<BlockSpaceManager> blockManager)
    : schedulerConfig_(schedulerConfig), predictor_(predictor), blockManager_(blockManager)
{
    batchSizeUpperBound_ = schedulerConfig_->maxBatchSize;
    const uint64_t minBatchSize = 5U;
    batchSizeLowerBound_ = minBatchSize;
    batchSizeUpper_ = batchSizeUpperBound_;
    batchSizeLower_ = batchSizeLowerBound_;
    batchTrackerWindowSize_ = 20000;
    decodeBatchSizeQueue_ = std::make_unique<DecodeBatchSizeTracker>(batchTrackerWindowSize_);
}

// 通过策略修改最大batchsize大小
void DynamicBatchSize::ApplyDynamicBatchSize(Role role, SchedulerOutputs &schedulerOut, size_t waitingSize,
                                             size_t runningSize, size_t swappedSize)
{
    if (schedulerOut.IsEmpty() && schedulerConfig_->maxPrefillBatchSize != 0) {
        return;
    }
    size_t previousDecodeBatchSize = 0;
    if (schedulerOut.forwardMode_ == ForwardMode::DECODE) {
        previousDecodeBatchSize = schedulerOut.scheduledSeqGroups_.size();
        previousStage_ = 1;
    }

    // PD混布场景且开启动态batchsize特性
    if (role == Role::PnD && schedulerConfig_->dynamicBatchSizeEnable) {
        AdjustBatchSize(previousStage_, previousDecodeBatchSize, waitingSize, runningSize, swappedSize);
    }
}

void DynamicBatchSize::AdjustBatchSize(size_t previousStage, size_t previousDecodeBatchSize, size_t waitingSize,
                                       size_t runningSize, size_t swappedSize)
{
    const uint64_t currentPrefillRequestNum = static_cast<uint64_t>(waitingSize);
    const uint64_t currentDecodeRequestNum = static_cast<uint64_t>(runningSize + swappedSize);

    if (previousDecodeBatchSize > 0) {
        decodeBatchSizeQueue_->AddDataPoint(previousDecodeBatchSize);
    }
    if (currentPrefillRequestNum == 0 && currentDecodeRequestNum == 0) {
        return;
    }

    auto avgDecodeLatency = predictor_->GetDecodeRecentAvgLatency(batchTrackerWindowSize_);
    auto avgBatchSize = decodeBatchSizeQueue_->GetRecentAvgBatchSize(batchTrackerWindowSize_);
    if (avgBatchSize == 0 || std::fabs(avgDecodeLatency) < 1e-6f || previousStage != 1) {
        return;
    }

    MINDIE_LLM_LOG_INFO_REQUEST("[DynamicBatchSize] avg decode latency: " << avgDecodeLatency
                        << "ms, avg batch size:" << avgBatchSize);
    const uint16_t stageModulo = 2;
    stage_ = (stage_ + 1) % stageModulo;
    MINDIE_LLM_LOG_INFO_REQUEST("[DynamicBatchSize] waiting size="
                        << waitingSize << ", running size=" << runningSize << ", swapped size=" << swappedSize);

    if (stage_ % stageModulo == 0) {
        BinarySearchBatchSize(currentDecodeRequestNum, avgDecodeLatency, avgBatchSize);
    } else {
        SetMinimalBatchSize(currentDecodeRequestNum, avgDecodeLatency);
    }
}

void DynamicBatchSize::BinarySearchBatchSize(uint64_t currentDecodeRequestNum, double avgDecodeLatency, uint64_t avgBatchSize)
{
    const uint32_t deltaAdjustUpper = 10;
    const uint32_t deltaAdjustLower = 5;
    const double deltaMs = 5.0;
    const uint64_t deltaBatchSize = 2;

    // Adjust batch size bounds based on latency
    if (avgDecodeLatency > schedulerConfig_->decodeExpectedTime + deltaMs) {
        batchSizeUpper_ = std::max(avgBatchSize, batchSizeLower_ + deltaAdjustUpper);
        batchSizeLower_ = std::max(batchSizeLower_ - deltaBatchSize, batchSizeLowerBound_);
    } else if (avgDecodeLatency < schedulerConfig_->decodeExpectedTime - deltaMs) {
        batchSizeLower_ = std::min(avgBatchSize, batchSizeUpper_ - deltaAdjustUpper);
        batchSizeUpper_ = std::min(batchSizeUpper_ + deltaBatchSize, batchSizeUpperBound_);
    } else {
        batchSizeUpper_ = std::min(avgBatchSize + deltaAdjustUpper, batchSizeUpperBound_);
        batchSizeLower_ = std::max(avgBatchSize - deltaAdjustLower, batchSizeLowerBound_);
    }

    // Ensure bounds are valid
    batchSizeUpper_ = std::clamp(batchSizeUpper_, batchSizeLowerBound_, batchSizeUpperBound_);
    batchSizeLower_ = std::clamp(batchSizeLower_, batchSizeLowerBound_, batchSizeUpperBound_);

    // Calculate new batch sizes
    uint64_t newDecodeMaxBatchSize = (batchSizeUpper_ + batchSizeLower_) / 2;
    newDecodeMaxBatchSize = std::clamp(newDecodeMaxBatchSize, currentDecodeRequestNum, batchSizeUpperBound_);

    uint64_t newPrefillMaxBatchSize =
        newDecodeMaxBatchSize > currentDecodeRequestNum ? (newDecodeMaxBatchSize - currentDecodeRequestNum) : 0UL;

    // Update scheduler configuration
    schedulerConfig_->maxBatchSize = newDecodeMaxBatchSize;
    schedulerConfig_->maxPrefillBatchSize = newPrefillMaxBatchSize;

    if (previousDecodeMaxBatchSize_ == newDecodeMaxBatchSize) {
        return;
    }
    MINDIE_LLM_LOG_INFO("[DynamicBatchSize] Updated maxPrefillBatchSize: "
                        << newPrefillMaxBatchSize << ", maxBatchSize: " << previousDecodeMaxBatchSize_ << " -> "
                        << newDecodeMaxBatchSize);
    previousDecodeMaxBatchSize_ = newDecodeMaxBatchSize;
}

void DynamicBatchSize::SetMinimalBatchSize(uint64_t currentDecodeRequestNum, double avgDecodeLatency)
{
    const double deltaMs = 5.0;
    // 为了抑制prefill，将maxPrefillBatchSize设置为0，陪跑一次decode。
    if (currentDecodeRequestNum != 0 && (avgDecodeLatency > schedulerConfig_->decodeExpectedTime + deltaMs)) {
        schedulerConfig_->maxBatchSize = currentDecodeRequestNum;
        schedulerConfig_->maxPrefillBatchSize = 0UL;
        MINDIE_LLM_LOG_INFO("[DynamicBatchSize] Updated maxPrefillBatchSize: 0, maxBatchSize: " << currentDecodeRequestNum);
        previousDecodeMaxBatchSize_ = currentDecodeRequestNum;
    }
}

void DecodeBatchSizeTracker::AddDataPoint(uint64_t batchSize)
{
    if (queue_.size() == windowSize_) {
        queue_.pop_front();
    }
    queue_.push_back(batchSize);
}

// 计算当前schedulerOut占用的block数
uint32_t DynamicBatchSize::GetScheduledOutBlockNum(SchedulerOutputs schedulerOut)
{
    uint32_t blockNum = 0;
    for (size_t i = 0; i < schedulerOut.scheduledSeqGroups_.size(); ++i) {
        const auto scheSeqGroup = schedulerOut.scheduledSeqGroups_[i];
        const auto seqGroup = scheSeqGroup->seqGroup_;
        SequenceId seqId = seqGroup->seqs_[0]->seqId_;
        const auto allIds = blockManager_->GetBlockIds(seqId);
        if (allIds.empty()) {
            continue;
        }
        blockNum += static_cast<uint32_t>(allIds[0].size());
    }
    return blockNum;
}

void DynamicBatchSize::RecordPredictorMetrics(const SchedulerOutputs &schedulerOut, const SchedulingBudget &budget)
{
    if (!schedulerConfig_->dynamicBatchSizeEnable &&
        schedulerConfig_->stageSelectPolicy != static_cast<uint32_t>(StagePolicyType::LATENCY_FIRST)) {
        return;
    }
    if (schedulerOut.scheduledSeqGroups_.size() == 0) {
        return;
    }
    auto batchStatsPtr = std::make_shared<BatchStats>();
    batchStatsPtr->forwardMode = schedulerOut.forwardMode_;
    batchStatsPtr->numBatchedTokens = budget.numBatchedTokens_;
    batchStatsPtr->kvCacheBlockNum = GetScheduledOutBlockNum(schedulerOut);
    predictor_->SaveBatchStats(batchStatsPtr);
}

uint64_t DecodeBatchSizeTracker::GetRecentAvgBatchSize(uint64_t forwardNum)
{
    if (queue_.empty()) {
        return 0;
    }
    std::vector<uint64_t> dataPoint;
    std::vector<uint64_t> dataCnt;
    uint64_t sumVal = 0;
    // Traverse the data points in reverse order (from most recent to oldest)
    // to calculate the average batch size of recent data
    for (auto it = queue_.rbegin(); it != queue_.rend(); ++it) {
        sumVal += *it;
        dataPoint.push_back(*it);
        if (sumVal < forwardNum) {
            dataCnt.push_back(*it);
        } else if (sumVal == forwardNum) {
            dataCnt.push_back(*it);
            break;
        } else { // sumVal > forwardNum
            dataCnt.push_back(sumVal - forwardNum);
            break;
        }
    }

    // calculate weighted average
    uint64_t weightedSum = 0;
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < dataPoint.size(); ++i) {
        weightedSum += dataPoint[i] * dataCnt[i];
        cnt += dataCnt[i];
    }
    if (cnt == 0) {
        return 0;
    }
    return weightedSum / cnt;
}
} // namespace mindie_llm