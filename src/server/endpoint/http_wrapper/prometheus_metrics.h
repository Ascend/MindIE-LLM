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

#ifndef PROMETHEUS_METRICS_H
#define PROMETHEUS_METRICS_H

#include <cstdint>
#include <chrono>
#include <memory>
#include <thread>

#include <prometheus/registry.h>
#include <prometheus/counter.h>
#include <prometheus/histogram.h>
#include <prometheus/gauge.h>
#include <prometheus/text_serializer.h>
#include "status.h"

namespace mindie_llm {
class PrometheusMetrics {
public:
    PrometheusMetrics(const PrometheusMetrics &) = delete;
    ~PrometheusMetrics();

    static std::shared_ptr<PrometheusMetrics> GetInstance();
    PrometheusMetrics &operator=(const PrometheusMetrics &) = delete;

    Status InitPrometheusMetrics(std::string modelName);

    void GetMetricsResult(std::string &metricsResult);

    void TTFTObserve(uint64_t prefillTime);
    void TBTObserve(uint64_t decodeTime);
    void E2EObserve(uint64_t e2eTime);

    void RequestNumberCount();
    void ResponseNumberCount();
    void FailedResponseNumberCount();

    void PrefillThroughputGaugeCollect(float prefillThroughput);
    void DecodeThroughputGaugeCollect(float decodeThroughput);

    void FailedRequestRateGaugeCollect();

    void RequestInputTokenHistogramCollect(uint32_t tokenNum);
    void ResponseOutputTokenHistogramCollect(uint32_t tokenNum);
    void RequestInputTokenCount(uint32_t tokenNum);
    void ResponseOutputTokenCount(uint32_t tokenNum);

    void CacheBlockDataCollect(uint32_t freeNpuBlockNums, uint32_t freeCpuBlockNums,
        uint32_t totalNpuBlockNums, uint32_t totalCpuBlockNums);

    void RadixMatchDataCollect(uint64_t allRadixMatchNum, uint64_t npuRadixMatchHitNum);

    void PreemptNumCount(uint64_t preemptNum);

    void RequestNumsGaugeCollect(uint32_t runningRequestNum, uint32_t waitingRequestNum, uint32_t swappedRequestNum);
    void CollectMetricDate();
    void RecordRequestNums();
    void RecordCacheBlockData();
    void StartCollectMetricDate();
    void RecordRadixMatchData();
    void GetCumulativePreemptCount();
    void RecordStatusData();

    bool IsActivate();
private:
    explicit PrometheusMetrics();

    static constexpr int tokenArraySize = 15;
    static constexpr int tokenBucketLastIndex = tokenArraySize - 1;

    std::shared_ptr<prometheus::Registry> registry;
    std::shared_ptr<prometheus::TextSerializer> serializer;
    uint32_t tokenArray[tokenArraySize][tokenArraySize];
    uint32_t inputTokens;
    void PrintTokenDistribution();

    prometheus::Counter* requestNumCounter_ = nullptr;  // request number counter
    prometheus::Counter* responseNumCounter_ = nullptr; // response number counter
    prometheus::Counter* failedResponseNumCounter_ = nullptr; // failed response number counter
    prometheus::Gauge *runningRequestNumGauge_ = nullptr; // Number of requests is running and pending status
    prometheus::Gauge *waitingRequestNumGauge_ = nullptr; // Number of requests is waiting status
    prometheus::Gauge *swappedRequestNumGauge_ = nullptr; // Number of requests swapped to CPU

    prometheus::Gauge* prefillThroughputGauge_ = nullptr;   // request prefill throughput using average
    prometheus::Gauge* decodeThroughputGauge_ = nullptr;    // request decode throughput

    prometheus::Histogram* requestTTFTHistogram_ = nullptr; // request time to first token, after all prefill
    prometheus::Histogram* requestTBTHistogram_ = nullptr;  // request time between tokens, in decoding
    prometheus::Histogram* requestE2EHistogram_ = nullptr;  // request end to end time, from request to response

    // failed request rate, recoding when http response code is not 200
    prometheus::Gauge* failedRequestRateGauge_ = nullptr;

    prometheus::Histogram* requestInputTokenHistogram_ = nullptr;   // request input token length
    prometheus::Histogram* responseOutputTokenHistogram_ = nullptr; // response output token length
    prometheus::Counter* requestInputTokenCounter_ = nullptr;
    prometheus::Counter* responseOutputTokenCounter_ = nullptr;

    prometheus::Gauge* npuCacheUsedRateGauge_ = nullptr;   // npu cache used rate
    prometheus::Gauge* cpuCacheUsedRateGauge_ = nullptr;   // cpu cache used rate

    prometheus::Gauge* npuPrefixCacheHitRate_ = nullptr;   // npu cache hit rate
    prometheus::Counter* requestNumPreemptionsTotal_ = nullptr;   // Cumulative number of preemption from the engine

    // 新增：上报原始数据，供 Coordinator 加权计算命中率
    prometheus::Gauge* allRadixMatchNumGauge_ = nullptr;      // 总的 radix match 尝试次数
    prometheus::Gauge* npuRadixMatchHitNumGauge_ = nullptr;   // radix match 命中次数

    std::chrono::steady_clock::time_point TTFTStartTime_ = std::chrono::steady_clock::time_point();
    std::chrono::steady_clock::time_point TTFTEndTime_ = std::chrono::steady_clock::time_point();
    std::chrono::steady_clock::time_point TBTStartTime_ = std::chrono::steady_clock::time_point();
    std::chrono::steady_clock::time_point TBTEndTime_ = std::chrono::steady_clock::time_point();
    std::chrono::steady_clock::time_point E2EStartTime_ = std::chrono::steady_clock::time_point();
    std::chrono::steady_clock::time_point E2EEndTime_ = std::chrono::steady_clock::time_point();

    uint64_t totalRequestNum_ = 0;
    uint64_t failedRequestNum_ = 0;

    uint64_t lastPreemptNum_ = 0;

    bool isActivate_ = false;
    bool shutdown_ = false;

    std::thread collectThread_;
    std::mutex varMutex;
};
} // namespace mindie_llm

#endif // PROMETHEUS_METRICS_H