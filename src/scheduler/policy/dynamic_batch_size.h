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

#ifndef DYNAMIC_BATCH_SIZE_H
#define DYNAMIC_BATCH_SIZE_H

#include "block_manager_interface.h"
#include "policy/seq_group_collection.h"
#include "dataclass/metric.h"
#include "concurrent_deque.h"
#include "sequence_group.h"
#include "scheduling_budget.h"
#include "latency_predictor/latency_predictor.h"

namespace mindie_llm {

class DecodeBatchSizeTracker {
public:
    explicit DecodeBatchSizeTracker(uint64_t windowSize): windowSize_(windowSize) {}
    void AddDataPoint(uint64_t batchSize);
    uint64_t GetRecentAvgBatchSize(uint64_t forwardNum);
    ~DecodeBatchSizeTracker() = default;
private:
    uint64_t windowSize_;
    std::deque<uint64_t> queue_;
};

class DynamicBatchSize {
public:
    explicit DynamicBatchSize(const SchedulerConfigSPtr schedulerConfig, std::shared_ptr<LatencyPredictor> predictor,
    std::shared_ptr<BlockSpaceManager> blockManager);

    void ApplyDynamicBatchSize(Role role, SchedulerOutputs& schedulerOut, size_t waitingSize,
    size_t runningSize, size_t swappedSize);
    
    void RecordPredictorMetrics(const SchedulerOutputs& schedulerOut, const SchedulingBudget& budget);

    virtual ~DynamicBatchSize() = default;

private:
    SchedulerConfigSPtr schedulerConfig_;
    std::shared_ptr<LatencyPredictor> predictor_;
    BlockSpaceManagerSPtr blockManager_; // kv cache manager

    // dynamic batch size
    uint32_t stage_ = 0;
    size_t previousStage_ = 0; // 0 : prefill, 1 : decode
    size_t previousDecodeBatchSize_ = 0;
    size_t previousDecodeMaxBatchSize_ = 0;
    uint64_t batchSizeUpper_ = 0;
    uint64_t batchSizeLower_ = 0;
    uint64_t batchSizeUpperBound_ = 0;
    uint64_t batchSizeLowerBound_ = 0;
    uint64_t batchTrackerWindowSize_ = 0;
    std::unique_ptr<DecodeBatchSizeTracker> decodeBatchSizeQueue_;

    void AdjustBatchSize(size_t previousStage, size_t previousDecodeBatchSize,
        size_t waitingSize, size_t runningSize, size_t swappedSize);

    void BinarySearchBatchSize(uint64_t currentDecodeRequestNum, double avgDecodeLatency, uint64_t avgBatchSize);

    void SetMinimalBatchSize(uint64_t currentDecodeRequestNum, double avgDecodeLatency);

    uint32_t GetScheduledOutBlockNum(SchedulerOutputs schedulerOut);
};
} // namespace mindie_llm

#endif