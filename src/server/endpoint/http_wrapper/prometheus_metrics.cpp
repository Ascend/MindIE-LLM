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
#include <algorithm>
#include <cctype>
#include <iomanip>
#include "config_manager.h"
#include "infer_instances.h"
#include "env_util.h"
#include "log.h"
#include "log_utils.h"
#include "endpoint_def.h"
#include "config_manager_impl.h"
#include "prometheus_metrics.h"

namespace mindie_llm {
// 手动递增的改动标识：每次你确认要发版本/验证生效时，把数字 +1。
// 目的：仅通过日志即可确认当前运行二进制是否包含最新改动。
static constexpr int TABLEIDLEN = 2;

constexpr uint32_t MILLISEC_TO_SEC = 1000;
const prometheus::Histogram::BucketBoundaries TOKEN_BUCKETS = {10, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
    16000, 20000, 32000, 50000, 64000, 128000};
const prometheus::Histogram::BucketBoundaries TTFT_BUCKETS = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0};
const prometheus::Histogram::BucketBoundaries TBT_BUCKETS = {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4,
    0.5, 0.75, 1.0, 2.5};
const prometheus::Histogram::BucketBoundaries E2E_BUCKETS = {1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0};

constexpr int COLLECT_SLEEP = 1000;

constexpr int TABLE_MIN_COL_WIDTH = 6;
constexpr int TABLE_MIN_LABEL_WIDTH = 11;
constexpr int TABLE_COL_SEP_WIDTH = 3;

std::shared_ptr<PrometheusMetrics> PrometheusMetrics::GetInstance()
{
    static std::shared_ptr<PrometheusMetrics> instance(new PrometheusMetrics());
    return instance;
}

PrometheusMetrics::PrometheusMetrics() : inputTokens(0)
{
    int size = std::min(static_cast<int>(TOKEN_BUCKETS.size()), tokenArraySize);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            tokenArray[i][j] = 0;
        }
    }
    const std::string serviceMonitorMode = EnvUtil::GetInstance().Get("MIES_SERVICE_MONITOR_MODE");
    if (!serviceMonitorMode.empty()) {
        try {
            if (std::stoi(serviceMonitorMode) == 1) {
                isActivate_ = true;
            }
        } catch (const std::exception& e) {
            ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_MANAGE_REQUEST,
                CHECK_ERROR), "Please set the environment variable: export MIES_SERVICE_MONITOR_MODE=1");
        }
    }

    if (isActivate_) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
            std::string("[Metrics][PatchId] prometheus_metrics_patch_id=") + std::to_string(TABLEIDLEN));

        const std::vector<ModelDeployConfig> &modelConfig = GetModelDeployConfig();
        std::string modelName = "";
        for (size_t i = 0; i < modelConfig.size(); ++i) {
            modelName += modelConfig[i].modelName;
            if (i != modelConfig.size() - 1) {
                modelName += " & ";
            }
        }

        registry = std::make_shared<prometheus::Registry>();
        serializer = std::make_shared<prometheus::TextSerializer>();
        auto ret = InitPrometheusMetrics(modelName);
        if (!ret.IsOk()) {
            isActivate_ = false;
        }
        StartCollectMetricDate();
    }
}

PrometheusMetrics::~PrometheusMetrics()
{
    shutdown_ = true;
    if (collectThread_.joinable()) {
        collectThread_.join();
    }
}

Status PrometheusMetrics::InitPrometheusMetrics(std::string modelName)
{
    try {
        requestNumCounter_ = &prometheus::BuildCounter()
            .Name("request_received_total").Help("Count of received requests.").Register(*registry)
            .Add({{"model_name", modelName}});
        responseNumCounter_ = &prometheus::BuildCounter()
            .Name("request_success_total").Help("Count of successfully processed requests.").Register(*registry)
            .Add({{"model_name", modelName}});
        failedResponseNumCounter_ = &prometheus::BuildCounter()
            .Name("request_failed_total").Help("Count of failed requests.").Register(*registry)
            .Add({{"model_name", modelName}});
        runningRequestNumGauge_ = &prometheus::BuildGauge()
            .Name("num_requests_running").Help("Number of requests currently running on NPU.")
            .Register(*registry).Add({ { "model_name", modelName } });
        waitingRequestNumGauge_ = &prometheus::BuildGauge()
            .Name("num_requests_waiting").Help("Number of requests waiting to be processed.")
            .Register(*registry).Add({ { "model_name", modelName } });
        swappedRequestNumGauge_ = &prometheus::BuildGauge()
            .Name("num_requests_swapped").Help("Number of requests swapped to CPU.")
            .Register(*registry).Add({ { "model_name", modelName } });

        prefillThroughputGauge_ = &prometheus::BuildGauge()
            .Name("avg_prompt_throughput_toks_per_s").Help("Average prefill throughput in tokens/s.")
            .Register(*registry).Add({{"model_name", modelName}});
        decodeThroughputGauge_ = &prometheus::BuildGauge()
            .Name("avg_generation_throughput_toks_per_s").Help("Average generation throughput in tokens/s.")
            .Register(*registry).Add({{"model_name", modelName}});

        requestTTFTHistogram_ = &prometheus::BuildHistogram()
            .Name("time_to_first_token_seconds").Help("Histogram of time to first token in seconds.")
            .Register(*registry).Add({{"model_name", modelName}}, TTFT_BUCKETS);
        requestTBTHistogram_ = &prometheus::BuildHistogram()
            .Name("time_per_output_token_seconds").Help("Histogram of time per output token in seconds.")
            .Register(*registry).Add({{"model_name", modelName}}, TBT_BUCKETS);
        requestE2EHistogram_ = &prometheus::BuildHistogram()
            .Name("e2e_request_latency_seconds").Help("Histogram of end to end request latency in seconds.")
            .Register(*registry).Add({{"model_name", modelName}}, E2E_BUCKETS);

        failedRequestRateGauge_ = &prometheus::BuildGauge()
            .Name("failed_request_perc").Help("Requests failure rate. 1 means 100 percent usage.")
            .Register(*registry).Add({{"model_name", modelName}});

        requestInputTokenHistogram_ = &prometheus::BuildHistogram()
            .Name("request_prompt_tokens").Help("Number of prefill tokens processed.").Register(*registry)
            .Add({{"model_name", modelName}}, TOKEN_BUCKETS);
        responseOutputTokenHistogram_ = &prometheus::BuildHistogram()
            .Name("request_generation_tokens").Help("Number of generation tokens processed.").Register(*registry)
            .Add({{"model_name", modelName}}, TOKEN_BUCKETS);
        requestInputTokenCounter_ = &prometheus::BuildCounter()
            .Name("prompt_tokens_total").Help("Number of prefill tokens processed.").Register(*registry)
            .Add({{"model_name", modelName}});
        responseOutputTokenCounter_ = &prometheus::BuildCounter()
            .Name("generation_tokens_total").Help("Number of generation tokens processed.").Register(*registry)
            .Add({{"model_name", modelName}});

        npuCacheUsedRateGauge_ = &prometheus::BuildGauge()
            .Name("npu_cache_usage_perc").Help("NPU KV-cache usage. 1 means 100 percent usage.").Register(*registry)
            .Add({{"model_name", modelName}});
        cpuCacheUsedRateGauge_ = &prometheus::BuildGauge()
            .Name("cpu_cache_usage_perc").Help("CPU KV-cache usage. 1 means 100 percent usage.").Register(*registry)
            .Add({{"model_name", modelName}});

        npuPrefixCacheHitRate_ = &prometheus::BuildGauge()
            .Name("npu_prefix_cache_hit_rate").Help("NPU prefix cache block hit rate.").Register(*registry)
            .Add({{"model_name", modelName}});
        requestNumPreemptionsTotal_ = &prometheus::BuildCounter()
            .Name("num_preemptions_total").Help("Cumulative number of preemption from the engine.").Register(*registry)
            .Add({{"model_name", modelName}});
        // 新增：上报原始数据，供 Coordinator 加权计算命中率
        allRadixMatchNumGauge_ = &prometheus::BuildGauge()
            .Name("all_radix_match_num")
            .Help("Total number of prefill prompt tokens participating in prefix cache radix match.")
            .Register(*registry).Add({{"model_name", modelName}});
        npuRadixMatchHitNumGauge_ = &prometheus::BuildGauge()
            .Name("npu_radix_match_hit_num")
            .Help("Number of prefill prompt tokens hit by prefix cache radix match.")
            .Register(*registry).Add({{"model_name", modelName}});
    } catch (const std::exception &e) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_MANAGE_REQUEST,
            STATUS_WARNING), "Init prometheus metrics error!");
        return Status(Error::Code::ERROR, "Failed to init prometheus metrics");
    }

    return Status(Error::Code::OK, "Success");
}

void PrometheusMetrics::GetMetricsResult(std::string &metricsResult)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        PrintTokenDistribution();
        auto collectMetricsResult = registry->Collect();
        std::ostringstream os;
        serializer->Serialize(os, collectMetricsResult);

        metricsResult = os.str();
    }
}

void PrometheusMetrics::TTFTObserve(uint64_t prefillTime)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        float tmpPrefillTime = static_cast<float>(prefillTime) / MILLISEC_TO_SEC;
        requestTTFTHistogram_->Observe(tmpPrefillTime);
    }
}

void PrometheusMetrics::TBTObserve(uint64_t decodeTime)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        float tmpDecodeTime = static_cast<float>(decodeTime) / MILLISEC_TO_SEC;
        requestTBTHistogram_->Observe(tmpDecodeTime);
    }
}

void PrometheusMetrics::E2EObserve(uint64_t e2eTime)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        float tmpE2ETime = static_cast<float>(e2eTime) / MILLISEC_TO_SEC;
        requestE2EHistogram_->Observe(tmpE2ETime);
    }
}

void PrometheusMetrics::RequestNumberCount()
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        requestNumCounter_->Increment();
        ++totalRequestNum_;
    }
}

void PrometheusMetrics::ResponseNumberCount()
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        responseNumCounter_->Increment();
    }
}

void PrometheusMetrics::FailedResponseNumberCount()
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        ++failedRequestNum_;
        failedResponseNumCounter_->Increment();
    }
}

void PrometheusMetrics::PrefillThroughputGaugeCollect(float prefillThroughput)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        prefillThroughputGauge_->Set(prefillThroughput * MILLISEC_TO_SEC);
    }
}

void PrometheusMetrics::DecodeThroughputGaugeCollect(float decodeThroughput)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        decodeThroughputGauge_->Set(decodeThroughput * MILLISEC_TO_SEC);
    }
}

void PrometheusMetrics::FailedRequestRateGaugeCollect()
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        float failedRequestRate = 0.0;
        if (totalRequestNum_ > 0) {
            failedRequestRate = static_cast<float>(failedRequestNum_) / static_cast<float>(totalRequestNum_);
        } else if (totalRequestNum_ == 0 && failedRequestNum_ > 0) {
            failedRequestRate = 1.0;
        }

        failedRequestRateGauge_->Set(failedRequestRate);
    }
}

void PrometheusMetrics::RequestInputTokenHistogramCollect(uint32_t tokenNum)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        requestInputTokenHistogram_->Observe(tokenNum);
    }
}

void PrometheusMetrics::ResponseOutputTokenHistogramCollect(uint32_t tokenNum)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        responseOutputTokenHistogram_->Observe(tokenNum);
    }
}

void PrometheusMetrics::RequestInputTokenCount(uint32_t tokenNum)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        requestInputTokenCounter_->Increment(static_cast<double>(tokenNum));
        inputTokens = tokenNum;
    }
}

void PrometheusMetrics::ResponseOutputTokenCount(uint32_t tokenNum)
{
    if (isActivate_) {
        std::lock_guard<std::mutex> lock(varMutex);
        responseOutputTokenCounter_->Increment(static_cast<double>(tokenNum));
        int inputIndex = tokenBucketLastIndex;
        int outputIndex = tokenBucketLastIndex;
        for (size_t i = 0; i < TOKEN_BUCKETS.size(); i++) {
            if (inputTokens <= TOKEN_BUCKETS[i]) {
                inputIndex = static_cast<int>(i);
                break;
            }
        }
        for (size_t i = 0; i < TOKEN_BUCKETS.size(); i++) {
            if (tokenNum <= TOKEN_BUCKETS[i]) {
                outputIndex = static_cast<int>(i);
                break;
            }
        }
        if (inputIndex >= 0 && outputIndex >= 0) {
            tokenArray[inputIndex][outputIndex]++;
        }
    }
}

void PrometheusMetrics::PrintTokenDistribution()
{
    std::string pdRole = GetInferInstance()->GetPDRole();
    if (pdRole == "prefill" || pdRole == "decode") {
        return;
    }
    int maxNumWidth = 0;
    for (size_t i = 0; i < TOKEN_BUCKETS.size(); i++) {
        int width = std::to_string(static_cast<int>(TOKEN_BUCKETS[i])).length();
        if (width > maxNumWidth) {
            maxNumWidth = width;
        }
    }
    uint32_t maxCellValue = 0;
    for (size_t i = 0; i < TOKEN_BUCKETS.size(); i++) {
        for (size_t j = 0; j < TOKEN_BUCKETS.size(); j++) {
            if (tokenArray[i][j] > maxCellValue) {
                maxCellValue = tokenArray[i][j];
            }
        }
    }
    int maxValueDigits = static_cast<int>(std::to_string(maxCellValue).length());
    int colWidth = std::max({maxNumWidth, maxValueDigits, TABLE_MIN_COL_WIDTH});
    int maxLabelWidth = TABLE_MIN_LABEL_WIDTH;
    for (size_t i = 0; i < TOKEN_BUCKETS.size(); i++) {
        int labelWidth = 5 + std::to_string(static_cast<int>(TOKEN_BUCKETS[i])).length();
        if (labelWidth > maxLabelWidth) {
            maxLabelWidth = labelWidth;
        }
    }
    int totalWidth = maxLabelWidth + TOKEN_BUCKETS.size() * (colWidth + TABLE_COL_SEP_WIDTH);
    std::string startLine(totalWidth, '=');
    std::string sepLine(totalWidth, '-');
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, std::string("[Metrics][Seq_Len_Table] ") + startLine);
    std::stringstream header;
    header << std::left << std::setw(maxLabelWidth) << "OUTPUT";
    for (size_t out = 0; out < TOKEN_BUCKETS.size(); out++) {
        header << " | " << std::setw(colWidth) << static_cast<int>(TOKEN_BUCKETS[out]);
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, std::string("[Metrics][Seq_Len_Table] ") + header.str());
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, std::string("[Metrics][Seq_Len_Table] ") + sepLine);
    for (size_t in = 0; in < TOKEN_BUCKETS.size(); in++) {
        std::stringstream row;
        std::string label = "INPUT" + std::to_string(static_cast<int>(TOKEN_BUCKETS[in]));
        row << std::left << std::setw(maxLabelWidth) << label;
        for (size_t out = 0; out < TOKEN_BUCKETS.size(); out++) {
            row << " | " << std::setw(colWidth) << tokenArray[in][out];
        }
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, std::string("[Metrics][Seq_Len_Table] ") + row.str());
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, std::string("[Metrics][Seq_Len_Table] ") + startLine);
}

void PrometheusMetrics::CacheBlockDataCollect(uint32_t freeNpuBlockNums, uint32_t freeCpuBlockNums,
    uint32_t totalNpuBlockNums, uint32_t totalCpuBlockNums)
{
    if (isActivate_ && totalNpuBlockNums > 0 && totalNpuBlockNums >= freeNpuBlockNums) {
        float npuCacheUsedRate =
            static_cast<float>(totalNpuBlockNums - freeNpuBlockNums) / static_cast<float>(totalNpuBlockNums);
        npuCacheUsedRateGauge_->Set(npuCacheUsedRate);
    }
    if (isActivate_ && totalCpuBlockNums > 0 && totalCpuBlockNums >= freeCpuBlockNums) {
        float cpuCacheUsedRate =
            static_cast<float>(totalCpuBlockNums - freeCpuBlockNums) / static_cast<float>(totalCpuBlockNums);
        cpuCacheUsedRateGauge_->Set(cpuCacheUsedRate);
    }
}

void PrometheusMetrics::RadixMatchDataCollect(uint64_t allRadixMatchNum, uint64_t npuRadixMatchHitNum)
{
    if (isActivate_) {
        // 上报原始数据（供 Coordinator 聚合计算加权命中率）
        allRadixMatchNumGauge_->Set(static_cast<double>(allRadixMatchNum));
        npuRadixMatchHitNumGauge_->Set(static_cast<double>(npuRadixMatchHitNum));

        // 本地计算命中率（保持兼容性，单 endpoint 场景仍可直接使用此指标）
        float npuCacheHitRate = 0.0;
        if (allRadixMatchNum > 0) {
            npuCacheHitRate = static_cast<float>(npuRadixMatchHitNum) / static_cast<float>(allRadixMatchNum);
        } else if (allRadixMatchNum == 0 && npuRadixMatchHitNum > 0) {
            npuCacheHitRate = 1.0;
        }
        npuPrefixCacheHitRate_->Set(npuCacheHitRate);
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "RadixMatchDataCollect: allRadixMatchNum=" << allRadixMatchNum
            << ", npuRadixMatchHitNum=" << npuRadixMatchHitNum
            << ", npuCacheHitRate=" << npuCacheHitRate);
    }
}

void PrometheusMetrics::PreemptNumCount(uint64_t preemptNum)
{
    if (isActivate_) {
        requestNumPreemptionsTotal_->Increment(static_cast<double>(preemptNum - lastPreemptNum_));
        lastPreemptNum_ = preemptNum;
    }
}

void PrometheusMetrics::RequestNumsGaugeCollect(uint32_t runningRequestNum, uint32_t waitingRequestNum,
    uint32_t swappedRequestNum)
{
    if (isActivate_) {
        runningRequestNumGauge_->Set(runningRequestNum);
        waitingRequestNumGauge_->Set(waitingRequestNum);
        swappedRequestNumGauge_->Set(swappedRequestNum);
    }
}

void PrometheusMetrics::RecordCacheBlockData()
{
    uint64_t freeNpuBlockNums = 0;
    uint64_t freeCpuBlockNums = 0;
    uint64_t totalNpuBlockNums = 0;
    uint64_t totalCpuBlockNums = 0;

    Status status =
        GetInferInstance()->GetCacheBlockNums(freeNpuBlockNums, freeCpuBlockNums, totalNpuBlockNums, totalCpuBlockNums);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_MANAGE_REQUEST,
              CHECK_ERROR), "Failed to get cache block nums. Maybe the model instance is not ready. "
              << "Please wait the model instance(coordinator) is ready and use the Prometheus API /metrics later.");
        return;
    }

    CacheBlockDataCollect(freeNpuBlockNums, freeCpuBlockNums, totalNpuBlockNums, totalCpuBlockNums);
}

void PrometheusMetrics::RecordStatusData()
{
    float prefillThroughput = 0.0;
    float decodeThroughput = 0.0;
    Status status = GetInferInstance()->GetThroughput(prefillThroughput, decodeThroughput);
    if (!status.IsOk()) {
        std::string msg = "Can't to get throughput, prefillThroughput: " +
            std::to_string(prefillThroughput) + "decodeThroughput: " + std::to_string(decodeThroughput);
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_MANAGE_REQUEST,
            LOCAL_INVOKING_ERROR), msg);
        return;
    }

    PrefillThroughputGaugeCollect(prefillThroughput);
    DecodeThroughputGaugeCollect(decodeThroughput);
}

void PrometheusMetrics::RecordRequestNums()
{
    std::map<std::string, uint64_t> batchSchedulerMetrics{};
    Status status = GetInferInstance()->GetBatchSchedulerMetrics(batchSchedulerMetrics);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_MANAGE_REQUEST,
            CHECK_ERROR), "Failed to get request nums");
        return;
    }
    uint64_t runningRequestNum = 0;
    uint64_t waitingRequestNum = 0;
    uint64_t swappedRequestNum = 0;
    if (batchSchedulerMetrics.find("runningInferRequestNum") != batchSchedulerMetrics.end()) {
        runningRequestNum = batchSchedulerMetrics["runningInferRequestNum"];
    }
    if (batchSchedulerMetrics.find("waitingInferRequestNum") != batchSchedulerMetrics.end()) {
        waitingRequestNum = batchSchedulerMetrics["waitingInferRequestNum"];
    }
    if (batchSchedulerMetrics.find("swappedInferRequestNum") != batchSchedulerMetrics.end()) {
        swappedRequestNum = batchSchedulerMetrics["swappedInferRequestNum"];
    }
    RequestNumsGaugeCollect(runningRequestNum, waitingRequestNum, swappedRequestNum);
}

void PrometheusMetrics::RecordRadixMatchData()
{
    uint64_t allRadixMatchNum = 0;
    uint64_t npuRadixMatchHitNum = 0;

    Status status = GetInferInstance()->GetRadixMatchNums(allRadixMatchNum, npuRadixMatchHitNum);
    if (!status.IsOk()) {
        std::string msg = "Can't to get radix match data, allRadixMatchNum: " +
            std::to_string(allRadixMatchNum) + "npuRadixMatchHitNum: " + std::to_string(npuRadixMatchHitNum);
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_MANAGE_REQUEST,
            CHECK_ERROR), msg);
        return;
    }

    RadixMatchDataCollect(allRadixMatchNum, npuRadixMatchHitNum);
}

void PrometheusMetrics::GetCumulativePreemptCount()
{
    uint64_t cumulativePreemptCount = 0;

    Status status = GetInferInstance()->GetCumulativePreemptCount(cumulativePreemptCount);
    if (!status.IsOk()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
            GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_MANAGE_REQUEST, CHECK_ERROR),
            "Can't to get cumulative preempt count!");
        return;
    }

    PreemptNumCount(cumulativePreemptCount);
}

void PrometheusMetrics::StartCollectMetricDate()
{
    shutdown_ = false;
    collectThread_ = std::thread(&PrometheusMetrics::CollectMetricDate, this);
}

void PrometheusMetrics::CollectMetricDate()
{
    pthread_setname_np(pthread_self(), "CollectMetric");
    while (!shutdown_) {
        RecordCacheBlockData();
        RecordRequestNums();
        RecordRadixMatchData();
        GetCumulativePreemptCount();
        RecordStatusData();
        std::this_thread::sleep_for(std::chrono::milliseconds(COLLECT_SLEEP));
    }
}

bool PrometheusMetrics::IsActivate()
{
    return isActivate_;
}
} // namespace mindie_llm