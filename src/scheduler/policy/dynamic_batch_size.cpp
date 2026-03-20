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
#include "dynamic_batch_recorder.h"
#include "dynamic_batch_size.h"

namespace mindie_llm {
DynamicBatchSize::DynamicBatchSize(const SchedulerConfigSPtr schedulerConfig,
                                   std::shared_ptr<LatencyPredictor> predictor,
                                   std::shared_ptr<BlockSpaceManager> blockManager,
                                   size_t localDPRank)
    : schedulerConfig_(schedulerConfig), predictor_(predictor), blockManager_(blockManager), localDPRank_(localDPRank)
{
    batchSizeUpperBound_ = schedulerConfig_->maxBatchSize;
    batchSizeLowerBound_ = 3U;
    batchSizeUpper_ = batchSizeUpperBound_;
    batchSizeLower_ = batchSizeLowerBound_;
    batchTrackerWindowSize_ = 20000;
    decodeBatchSizeQueue_ = std::make_shared<DecodeBatchSizeTracker>(batchTrackerWindowSize_);

    auto &recorder = DynamicBatchRecorder::GetInstance(localDPRank_);
    recorder.SetDecodeBatchSizeTracker(decodeBatchSizeQueue_);
    recorder.SetLatencyPredictor(predictor_);
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
        // Aggregate data from all DPs before adjusting batch size
        AdjustBatchSize(previousStage_, previousDecodeBatchSize, waitingSize, runningSize, swappedSize);
    }
}

void DynamicBatchSize::AdjustBatchSize(size_t previousStage, size_t previousDecodeBatchSize, size_t waitingSize,
                                       size_t runningSize, size_t swappedSize)
{
    const uint64_t currentPrefillRequestNum = static_cast<uint64_t>(waitingSize);
    const uint64_t localDecodeRequestNum = static_cast<uint64_t>(runningSize + swappedSize);

    // Add current batch size data point
    if (previousDecodeBatchSize > 0) {
        decodeBatchSizeQueue_->AddDataPoint(previousDecodeBatchSize);
    }
    if (currentPrefillRequestNum == 0 && localDecodeRequestNum == 0) {
        return;
    }

    // Update running size for current DP before aggregation
    auto &recorder = DynamicBatchRecorder::GetInstance(localDPRank_);
    recorder.SetRunningSize(runningSize + swappedSize);

    // Aggregate all data from all DPs in a single call for consistency
    double maxDecodeLatency = 0.0;
    uint64_t maxBatchSize = 0;
    size_t maxDecodeRequestNum = 0;
    size_t validDPCount = DynamicBatchRecorder::AggregateAllFromAllDPs(
        batchTrackerWindowSize_, maxDecodeLatency, maxBatchSize, maxDecodeRequestNum);

    if (maxBatchSize == 0 || std::fabs(maxDecodeLatency) < 1e-6f || previousStage != 1) {
        schedulerConfig_->maxBatchSize = batchSizeLowerBound_;
        batchSizeLower_ = maxBatchSize;
        batchSizeUpper_ = maxBatchSize;
    }

    // Log local and aggregated input data for the algorithm
    MINDIE_LLM_LOG_INFO_REQUEST("[DynamicBatchSize|DP" << localDPRank_ << "] Inputs: " <<
         "localDecodeRequestNum=" << localDecodeRequestNum << ", " <<
         "aggregated(max)DecodeRequestNum=" << maxDecodeRequestNum << ", " <<
         "aggregated(max)DecodeLatency=" << maxDecodeLatency << "ms, " <<
         "aggregated(max)BatchSize=" << maxBatchSize << ", " <<
         "validDPCount=" << validDPCount);

    // Use maxDecodeRequestNum (aggregated from all DPs) to ensure consistent results
    // This also guarantees that running requests on any DP won't be stopped
    const uint64_t aggregatedDecodeRequestNum = static_cast<uint64_t>(maxDecodeRequestNum);

    // Each DP maintains its own stage counter and search bounds
    // Since input data is identical (aggregated max values), results will be consistent
    const uint32_t stageModulo = 2;
    stage_ = (stage_ + 1) % stageModulo;

    if (stage_ % stageModulo == 0) {
        BinarySearchBatchSize(aggregatedDecodeRequestNum, maxDecodeLatency, maxBatchSize);
    } else {
        SetMinimalBatchSize(aggregatedDecodeRequestNum, maxDecodeLatency);
    }
}

void DynamicBatchSize::BinarySearchBatchSize(uint64_t currentDecodeRequestNum, double avgDecodeLatency, uint64_t avgBatchSize)
{
    const uint32_t deltaAdjustUpper = 3;
    const uint32_t deltaAdjustLower = 2;
    const double deltaMs = 5.0;
    const uint64_t deltaBatchSize = 2;

    // Each DP uses its own search bounds (batchSizeUpper_, batchSizeLower_)
    // Since input data is identical (aggregated max values), results will be consistent across DPs

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
    uint64_t newDecodeMaxBatchSize = std::clamp((batchSizeUpper_ + batchSizeLower_) / 2,
                                                schedulerConfig_->maxBatchSize - deltaBatchSize,
                                                schedulerConfig_->maxBatchSize + deltaBatchSize);
    newDecodeMaxBatchSize = std::clamp(newDecodeMaxBatchSize, currentDecodeRequestNum, batchSizeUpperBound_);

    uint64_t newPrefillMaxBatchSize =
        newDecodeMaxBatchSize > currentDecodeRequestNum ? (newDecodeMaxBatchSize - currentDecodeRequestNum) : 0UL;

    // Log algorithm output
    MINDIE_LLM_LOG_DEBUG("[DynamicBatchSize|DP" << localDPRank_ << "] BinarySearch result: " <<
        "batchSizeUpper=" << batchSizeUpper_ << ", batchSizeLower=" << batchSizeLower_ << ", " <<
        "newDecodeMaxBatchSize=" << newDecodeMaxBatchSize << ", " <<
        "newPrefillMaxBatchSize=" << newPrefillMaxBatchSize);

    // Apply the updated batch sizes
    ApplyUpdatedBatchSize(newDecodeMaxBatchSize, newPrefillMaxBatchSize);
}

void DynamicBatchSize::SetMinimalBatchSize(uint64_t currentDecodeRequestNum, double avgDecodeLatency)
{
    const double deltaMs = 5.0;
    // 为了抑制prefill，将maxPrefillBatchSize设置为0，陪跑一次decode。
    if (currentDecodeRequestNum != 0 && (avgDecodeLatency > schedulerConfig_->decodeExpectedTime + deltaMs)) {
        uint64_t newDecodeMaxBatchSize = currentDecodeRequestNum;
        uint64_t newPrefillMaxBatchSize = 0UL;
        MINDIE_LLM_LOG_DEBUG("[DynamicBatchSize|DP" << localDPRank_ << "] SetMinimal: newDecodeMaxBatchSize=" <<
            newDecodeMaxBatchSize << ", newPrefillMaxBatchSize=" << newPrefillMaxBatchSize);

        ApplyUpdatedBatchSize(newDecodeMaxBatchSize, newPrefillMaxBatchSize);
    } else {
        // Condition not met, keep current settings
        uint64_t currentDecodeMaxBatchSize = schedulerConfig_->maxBatchSize;
        uint64_t currentPrefillMaxBatchSize = schedulerConfig_->maxPrefillBatchSize;

        MINDIE_LLM_LOG_DEBUG("[DynamicBatchSize|DP" << localDPRank_ << "] SetMinimal: condition not met, " <<
            "keeping current maxBatchSize=" << currentDecodeMaxBatchSize << ", " <<
            "maxPrefillBatchSize=" << currentPrefillMaxBatchSize);
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
    // Always save batch stats even for empty schedulerOut to ensure UpdateBatchStats can find the stats.
    // This is needed for multi-DP scenarios where some DPs may have empty scheduling results.
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

// Apply the updated batch sizes to scheduler config
void DynamicBatchSize::ApplyUpdatedBatchSize(uint64_t newDecodeMaxBatchSize, uint64_t newPrefillMaxBatchSize)
{
    schedulerConfig_->maxBatchSize = newDecodeMaxBatchSize;
    schedulerConfig_->maxPrefillBatchSize = newPrefillMaxBatchSize;

    if (previousDecodeMaxBatchSize_ == newDecodeMaxBatchSize) {
        return;
    }
    MINDIE_LLM_LOG_INFO_REQUEST("[DynamicBatchSize|DP" << localDPRank_ << "] Updated maxPrefillBatchSize: "  <<
                        newPrefillMaxBatchSize << ", maxBatchSize: " << previousDecodeMaxBatchSize_ << " -> "  <<
                        newDecodeMaxBatchSize);
    previousDecodeMaxBatchSize_ = newDecodeMaxBatchSize;
}

} // namespace mindie_llm