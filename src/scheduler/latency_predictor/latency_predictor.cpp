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
 
#include <utility>
#include "log.h"
#include "latency_predictor.h"

namespace mindie_llm {
void LatencyPredictor::UpdateBatchStats()
{
    int preBatchId = batchId_.load();
    std::optional<BatchStatsPtr> batchStatsOpt = batchStatsMap_.Get(preBatchId);
    if (!batchStatsOpt.has_value()) {
        throw std::runtime_error("previous batch not found");
    }
    BatchStatsPtr batchStatsPtr = batchStatsOpt.value();
    auto now = std::chrono::high_resolution_clock::now();
    auto batchSpendTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - batchStatsPtr->batchStartTime);
    batchStatsPtr->batchSpendTimeFloat = static_cast<float>(batchSpendTime.count());
    if (batchStatsPtr->forwardMode == ForwardMode::PREFILL) {
        prefillRegression_.AddDataPoint(batchStatsPtr->numBatchedTokens, batchStatsPtr->batchSpendTimeFloat);
        prefillLatency_.AddDataPoint(batchStatsPtr->batchSpendTimeFloat);
    } else if (batchStatsPtr->forwardMode == ForwardMode::DECODE) {
        decodeRegression_.AddDataPoint(batchStatsPtr->numBatchedTokens, batchStatsPtr->kvCacheBlockNum,
                                       batchStatsPtr->batchSpendTimeFloat);
        decodeLatency_.AddDataPoint(batchStatsPtr->batchSpendTimeFloat);
    }
    batchId_.fetch_sub(1);
    MINDIE_LLM_LOG_DEBUG("UpdateBatchStats batch info: forwardMode: "
                         << (batchStatsPtr->forwardMode == ForwardMode::PREFILL ? "prefill" : "decode")
                         << ", numBatchedTokens: " << batchStatsPtr->numBatchedTokens
                         << ", kvCacheBlockNum: " << batchStatsPtr->kvCacheBlockNum
                         << ", batchSpendTime: " << batchStatsPtr->batchSpendTimeFloat << "ms");
}

float LatencyPredictor::PredictBatchExecTime(BatchStats &batchStats)
{
    if (batchStats.forwardMode == ForwardMode::PREFILL) {
        return prefillRegression_.Predict(batchStats.numBatchedTokens);
    } else {
        return decodeRegression_.Predict(batchStats.numBatchedTokens, batchStats.kvCacheBlockNum);
    }
}

void LatencyPredictor::AddPercentileData(SequenceGroupSPtr &seqGroup, std::shared_ptr<SchedulerConfig> &schedulerConfig)
{
    // avoid nullptr
    if (seqGroup == nullptr) {
        return;
    }

    // // avoid intermediate prefill chunk
    if (schedulerConfig->enableChunkedPrefill) {
        return;
    }

    // add latency
    bool isPrefill = seqGroup->iterTimes <= 1;
    int64_t latencyMs = 0;
    auto now = std::chrono::high_resolution_clock::now();
    if (isPrefill) {
        latencyMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - seqGroup->arriveTime).count();
        prefillLatency_.AddDataPoint(latencyMs);
    } else {
        latencyMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - seqGroup->lastCompletionTime).count();
        decodeLatency_.AddDataPoint(latencyMs);
    }
    seqGroup->lastCompletionTime = now;
}

double LatencyPredictor::GetDecodeRecentAvgLatency(size_t forwardNum)
{
    return decodeLatency_.GetRecentAvgLatency(forwardNum);
}

void LatencyPredictor::SaveBatchStats(BatchStatsPtr batchStats)
{
    if (batchId_ >= INT_MAX) {
        batchId_.store(0);
    }
    batchId_.fetch_add(1);
    batchStatsMap_.Insert(batchId_.load(), batchStats);
}

void LatencyPredictor::SetBatchExecuteStartTime(TimePoint batchStartTime)
{
    std::optional<BatchStatsPtr> batchStatsOpt = batchStatsMap_.Get(batchId_.load());
    if (!batchStatsOpt.has_value()) {
        throw std::runtime_error("previous batch not found");
    }
    BatchStatsPtr batchStatsPtr = batchStatsOpt.value();
    batchStatsPtr->batchStartTime = batchStartTime;
}

} // namespace mindie_llm
