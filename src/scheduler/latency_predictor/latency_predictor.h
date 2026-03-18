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
 
#ifndef LATENCY_PREDICTOR
#define LATENCY_PREDICTOR

#include <atomic>
#include <numeric>
#include <deque>
#include <set>
#include <map>

#include "batch_latency_tracker.h"
#include "decode_regression.h"
#include "prefill_regression.h"
#include "sequence_group.h"
#include "policy/seq_group_collection.h"
#include "config_info.h"
#include "concurrent_map.h"

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace mindie_llm {
struct BatchStats {
    ForwardMode forwardMode;
    uint32_t numBatchedTokens = 0;
    uint32_t kvCacheBlockNum = 0;
    float batchSpendTimeFloat = 0;
    TimePoint batchStartTime = std::chrono::high_resolution_clock::now();
};

using BatchStatsPtr = std::shared_ptr<BatchStats>;

class LatencyPredictor {
public:
    LatencyPredictor() = default;

    LatencyPredictor(const LatencyPredictor &) = delete;

    LatencyPredictor &operator=(const LatencyPredictor &) = delete;

    ~LatencyPredictor() = default;

    std::atomic_int64_t remainPrefillSlots = 0;

    void UpdateBatchStats();

    float PredictBatchExecTime(BatchStats &batchStats);

    void AddPercentileData(SequenceGroupSPtr &seqGroup, std::shared_ptr<SchedulerConfig> &schedulerConfig);

    double GetDecodeRecentAvgLatency(size_t forwardNum);

    void SaveBatchStats(BatchStatsPtr batchStats);

    void SetBatchExecuteStartTime(TimePoint batchStartTime);

    void SaveEdgeCloudBatchStats(ForwardMode forwardMode);

    void UpdateEdgeCloudBatchStats(ForwardMode forwardMode);

    int GetPrefillProcessing() const {return prefillProcessing_;};
   
    int GetDecodeProcessing() const {return decodeProcessing_;};

private:
    // moving average for decode time regression
    DecodeRegression decodeRegression_ = DecodeRegression();
    PrefillRegression prefillRegression_ = PrefillRegression();

    // latency profilers
    BatchLatencyTracker prefillLatency_ = BatchLatencyTracker();
    BatchLatencyTracker decodeLatency_ = BatchLatencyTracker();

    std::atomic<int> batchId_{0};
    ConcurrentMap<int, BatchStatsPtr> batchStatsMap_; // batchId->BatchStatsPtr

    int prefillProcessing_{0};
    int decodeProcessing_{0};
};
} // namespace mindie_llm

#endif // LATENCY_PREDICTOR