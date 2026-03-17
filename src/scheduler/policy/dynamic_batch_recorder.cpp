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

#include "dynamic_batch_recorder.h"
#include "log.h"
#include "dynamic_batch_size.h"
#include "latency_predictor/latency_predictor.h"

namespace mindie_llm {

// Static member initialization
std::unordered_map<size_t, std::unique_ptr<DynamicBatchRecorder>> DynamicBatchRecorder::instances_;
std::mutex DynamicBatchRecorder::mutex_;

DynamicBatchRecorder::DynamicBatchRecorder(size_t localDPRank) : localDPRank_(localDPRank) {}

DynamicBatchRecorder &DynamicBatchRecorder::GetInstance(size_t localDPRank)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(localDPRank);
    if (it == instances_.end()) {
        auto recorder = std::unique_ptr<DynamicBatchRecorder>(new DynamicBatchRecorder(localDPRank));
        auto *ptr = recorder.get();
        instances_.emplace(localDPRank, std::move(recorder));
        MINDIE_LLM_LOG_DEBUG("[DynamicBatchRecorder] Created new instance for DP rank: " << localDPRank);
        return *ptr;
    }
    MINDIE_LLM_LOG_DEBUG("[DynamicBatchRecorder] Returned existing instance for DP rank: " << localDPRank);
    return *(it->second);
}

void DynamicBatchRecorder::SetLatencyPredictor(const std::shared_ptr<LatencyPredictor> &predictor)
{
    predictor_ = predictor;
    MINDIE_LLM_LOG_DEBUG("[DynamicBatchRecorder] Set LatencyPredictor for DP rank: " << localDPRank_);
}

std::shared_ptr<LatencyPredictor> DynamicBatchRecorder::GetLatencyPredictor() const
{
    return predictor_;
}

void DynamicBatchRecorder::SetDecodeBatchSizeTracker(const std::shared_ptr<DecodeBatchSizeTracker> &tracker)
{
    decodeBatchSizeTracker_ = tracker;
    MINDIE_LLM_LOG_DEBUG("[DynamicBatchRecorder] Set DecodeBatchSizeTracker for DP rank: " << localDPRank_);
}

std::shared_ptr<DecodeBatchSizeTracker> DynamicBatchRecorder::GetDecodeBatchSizeTracker() const
{
    return decodeBatchSizeTracker_;
}

size_t DynamicBatchRecorder::GetLocalDPRank() const
{
    return localDPRank_;
}

void DynamicBatchRecorder::SetRunningSize(size_t runningSize)
{
    runningSize_ = runningSize;
}

size_t DynamicBatchRecorder::GetRunningSize() const
{
    return runningSize_;
}

size_t DynamicBatchRecorder::AggregateAllFromAllDPs(size_t forwardNum, double &maxDecodeLatency,
                                                    uint64_t &maxBatchSize, size_t &maxDecodeRequestNum)
{
    maxDecodeLatency = 0.0;
    maxBatchSize = 0;
    maxDecodeRequestNum = 0;
    size_t validDPCount = 0;

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &pair : instances_) {
        const auto &recorder = pair.second;
        auto predictor = recorder->predictor_;
        auto tracker = recorder->decodeBatchSizeTracker_;

        if (predictor == nullptr || tracker == nullptr) {
            continue;
        }

        double dpLatency = predictor->GetDecodeRecentAvgLatency(forwardNum);
        uint64_t dpBatchSize = tracker->GetRecentAvgBatchSize(forwardNum);
        size_t dpRunningSize = recorder->runningSize_;

        MINDIE_LLM_LOG_INFO_REQUEST("[DynamicBatchRecorder|Aggregate] DP" << pair.first  <<
            ": latency=" << dpLatency << "ms, batchSize=" << dpBatchSize << ", " <<
            "runningSize=" << dpRunningSize);

        if (dpLatency > maxDecodeLatency) {
            maxDecodeLatency = dpLatency;
        }
        if (dpBatchSize > maxBatchSize) {
            maxBatchSize = dpBatchSize;
        }
        if (dpRunningSize > maxDecodeRequestNum) {
            maxDecodeRequestNum = dpRunningSize;
        }
        ++validDPCount;
    }

    MINDIE_LLM_LOG_INFO_REQUEST("[DynamicBatchRecorder|Aggregate] Final aggregated (max) values: " <<
        "latency=" << maxDecodeLatency << "ms, batchSize=" << maxBatchSize << ", " <<
        "decodeRequestNum=" << maxDecodeRequestNum << ", validDPCount=" << validDPCount);

    return validDPCount;
}

} // namespace mindie_llm
