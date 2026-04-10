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

#include "simulate_task_runner.h"
#include <algorithm>
#include "log.h"
#include "dcmi_wrapper.h"
#include "grpc_communicator.h"

namespace mindie_llm {

constexpr int POLLING_INTERVAL_MS = 1000;
constexpr int NPU_SAMPLES = 3;
constexpr int NPU_WINDOW_MS = 5000;
constexpr int NPU_CHECK_COMPLETE_WAIT_SECONDS = 10;
constexpr int SIMULATE_SKIP_AICORE_PERCENT = 50;
constexpr int SIMULATE_SKIP_EVERY_N_LOOPS = 18; // 18 X 5S = 1.5min，一分半内至少执行一次虚推


SimulateTaskRunner::SimulateTaskRunner()
{
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Instance created, waiting for Init.");
}

bool SimulateTaskRunner::Init(std::shared_ptr<ISimulateExecutor> executor,
                              const std::vector<std::pair<int, int>>& npuDeviceCardIds, int npuThreshold,
                              RunMode runMode, int chipPerCard)
{
    if (isValid_) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
            "SimulateTaskRunner::Init: Already initialized");
        return true;
    }
    if (!executor && runMode != RunMode::NPU_ONLY) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "SimulateTaskRunner::Init: executor must not be null");
        return false;
    }
    if (npuDeviceCardIds.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "SimulateTaskRunner::Init: npuDeviceCardIds must not be empty");
        return false;
    }
    if (npuThreshold == 0) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "SimulateTaskRunner::Init: NPU health check threshold cannot be zero.");
        return false;
    }

    npuThreshold_ = npuThreshold;
    executor_ = std::move(executor);
    runMode_ = runMode;
    chipPerCard_ = (chipPerCard > 0) ? chipPerCard : 1;
    npuDeviceCardIds_ = npuDeviceCardIds;
    isValid_ = true;

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner::Init: Initialized with "
        << npuDeviceCardIds_.size() << " NPU DCMI target(s), runMode="
        << static_cast<int>(runMode_) << ", chipPerCard(config)=" << chipPerCard_);
    return true;
}

SimulateTaskRunner::~SimulateTaskRunner()
{
    Stop();
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Instance destroyed.");
}

void SimulateTaskRunner::Start(uint32_t intervalSeconds)
{
    if (!isValid_) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "SimulateTaskRunner::Start: Not initialized, call Init() first.");
        return;
    }
    if (running_.load()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
            "SimulateTaskRunner: Task is already running.");
        return;
    }

    intervalSeconds_ = intervalSeconds;
    stopRequested_.store(false);
    npuCheckStopRequested_.store(false);
    running_.store(true);
    paused_.store(false);
    lastAicoreUtil_ = -1;
    loopsSinceSimulate_ = 0;

    // 更新初始状态
    {
        std::unique_lock<std::shared_mutex> lock(statusMutex_);
        healthStatus_.isRunning = true;
        healthStatus_.lastMessage = "task started";
        healthStatus_.lastUpdateTime = std::chrono::steady_clock::now();
    }

    // 启动 NPU 检测线程
    npuCheckThread_ = std::thread([this]() {
        NpuCheckLoop();
    });

    // 启动任务线程
    taskThread_ = std::thread([this]() {
        TaskLoop();
    });

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
        "SimulateTaskRunner: Started with interval=" << intervalSeconds << "s");
}

void SimulateTaskRunner::Stop()
{
    if (!running_.load()) {
        return;
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Stopping...");

    stopRequested_.store(true);
    npuCheckStopRequested_.store(true);
    running_.store(false);

    // 唤醒 NPU 检测线程使其退出
    {
        std::lock_guard<std::mutex> locker(npuCheckMutex_);
        npuCheckCv_.notify_one();
    }

    // 等待任务线程结束
    if (taskThread_.joinable()) {
        taskThread_.join();
    }

    // 等待 NPU 检测线程结束
    if (npuCheckThread_.joinable()) {
        npuCheckThread_.join();
    }

    // 更新状态
    {
        std::unique_lock<std::shared_mutex> lock(statusMutex_);
        healthStatus_.isRunning = false;
        healthStatus_.lastMessage = "task stopped";
        healthStatus_.lastUpdateTime = std::chrono::steady_clock::now();
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Stopped.");
}

void SimulateTaskRunner::Pause()
{
    if (!running_.load()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
            "SimulateTaskRunner: Cannot pause, task is not running.");
        return;
    }

    if (paused_.load()) {
        ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Already paused.");
        return;
    }

    paused_.store(true);

    {
        std::unique_lock<std::shared_mutex> lock(statusMutex_);
        healthStatus_.lastMessage = "task paused";
        healthStatus_.lastUpdateTime = std::chrono::steady_clock::now();
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Paused.");
}

void SimulateTaskRunner::Resume()
{
    if (!running_.load()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
            "SimulateTaskRunner: Cannot resume, task is not running.");
        return;
    }

    if (!paused_.load()) {
        ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Not paused.");
        return;
    }

    paused_.store(false);
    lastAicoreUtil_ = -1;
    loopsSinceSimulate_ = 0;

    {
        std::unique_lock<std::shared_mutex> lock(statusMutex_);
        healthStatus_.lastMessage = "task resumed";
        healthStatus_.lastUpdateTime = std::chrono::steady_clock::now();
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Resumed.");
}

SimulateHealthStatus SimulateTaskRunner::GetHealthStatus() const
{
    std::shared_lock<std::shared_mutex> lock(statusMutex_);
    return healthStatus_;
}

void SimulateTaskRunner::TaskLoop()
{
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Task loop started.");

    while (!stopRequested_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        TriggerNpuCheck();
        // slave节点只需要统计npu利用率
        if (runMode_ == RunMode::NPU_ONLY) {
            WaitForNpuCheckComplete();
            continue;
        }
        if (!ShouldRunSimulate()) {
            WaitForNpuCheckComplete();
            lastAicoreUtil_ = GetNpuUtilization();
            SimulateResult skipResult;
            skipResult.status = SimulateResult::Status::SUCCESS;
            skipResult.message = "Simulate skipped";
            UpdateHealthStatus(skipResult);
            continue;
        }

        SimulateResult result = executor_->RunSimulateOnce();
        WaitForNpuCheckComplete();
        if (result.status == SimulateResult::Status::TIMEOUT) {
            int npuUtil = GetNpuUtilization();
            if (npuUtil > npuThreshold_) {
                result.status = SimulateResult::Status::BUSY;
                result.message = "Timeout but AICore usage > " + std::to_string(npuThreshold_) + "%";
            } else if (slaveNpuReportTimeoutThisRound_.load()) {
                ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                    GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
                    "SimulateTaskRunner: gRPC communication abnormal (slave NPU report exceeded freshness window). "
                    "Skip abnormal judgement in this health-check cycle.");
                result.status = SimulateResult::Status::SUCCESS;
                result.message = "Timeout ignored due to stale slave NPU report";
            }
        }

        UpdateHealthStatus(result);
        lastAicoreUtil_ = GetNpuUtilization();
        ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
            "SimulateTaskRunner: Completed. status=" << static_cast<int>(result.status)
            << ", message=" << result.message);
    }
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Task loop exited.");
}

bool SimulateTaskRunner::ShouldRunSimulate()
{
    if (lastAicoreUtil_ < SIMULATE_SKIP_AICORE_PERCENT) {
        loopsSinceSimulate_ = 0;
        return true;
    }
    
    if (lastAicoreUtil_ >= SIMULATE_SKIP_AICORE_PERCENT) {
        if (++loopsSinceSimulate_ < SIMULATE_SKIP_EVERY_N_LOOPS) {
            return false;
        }
        loopsSinceSimulate_ = 0;
    }
    
    return true;
}

void SimulateTaskRunner::UpdateHealthStatus(const SimulateResult& result)
{
    std::unique_lock<std::shared_mutex> lock(statusMutex_);
    healthStatus_.lastStatus = result.status;
    healthStatus_.lastMessage = result.message;
    healthStatus_.lastUpdateTime = std::chrono::steady_clock::now();

    if (result.status == SimulateResult::Status::SUCCESS ||
        result.status == SimulateResult::Status::BUSY) {
        healthStatus_.successCount++;
    } else {
        healthStatus_.failureCount++;
    }
}

void SimulateTaskRunner::TriggerNpuCheck()
{
    npuUtil_.store(-1);
    {
        std::lock_guard<std::mutex> locker(npuCheckMutex_);
        npuCheckRequested_.store(true);
    }
    npuCheckCv_.notify_one();
}

void SimulateTaskRunner::WaitForNpuCheckComplete()
{
    if (npuUtil_.load() < 0) {
        auto lastTimePoint = std::chrono::steady_clock::now() + std::chrono::seconds(NPU_CHECK_COMPLETE_WAIT_SECONDS);
        std::unique_lock<std::mutex> locker(npuResultMutex_);
        npuResultCv_.wait_until(locker, lastTimePoint);
    }
}

void SimulateTaskRunner::NpuCheckLoop()
{
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: NPU check loop started.");
    DCMIWrapper& dcmiWrapper = DCMIWrapper::GetInstance();

    while (!npuCheckStopRequested_.load()) {
        {
            std::unique_lock<std::mutex> locker(npuCheckMutex_);
            npuCheckCv_.wait(locker, [this]() {
                return npuCheckRequested_.load() || npuCheckStopRequested_.load();
            });
            npuCheckRequested_.store(false);
        }

        if (npuCheckStopRequested_.load()) {
            break;
        }

        if (!dcmiWrapper.IsInitialized() && !dcmiWrapper.Initialize()) {
            continue;
        }

        CheckAicoreUtilization();
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: NPU check loop exited.");
}

void SimulateTaskRunner::CheckAicoreUtilization()
{
    DCMIWrapper &dcmiWrapper = DCMIWrapper::GetInstance();
    unsigned int maxUtilAcrossCards = 0;
    const int sampleCount = NPU_SAMPLES;

    auto getUtilizationFunc = dcmiWrapper.GetFunction<int (*)(int, int, int, unsigned int *)>(
        "dcmi_get_device_utilization_rate");
    if (!getUtilizationFunc) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "SimulateTaskRunner: Failed to get utilization function");
        return;
    }

    // Keep one NPU check window around 5s: scan and compensate sleep.
    const int perSampleTargetMs = std::max(POLLING_INTERVAL_MS, NPU_WINDOW_MS / sampleCount);
    for (int i = 0; i < sampleCount; i++) {
        const auto roundStart = std::chrono::steady_clock::now();
        for (const auto& target : npuDeviceCardIds_) {
            const int npuCardId = target.first;
            const int chipIdx = target.second;
            unsigned int utilizationRate = 0;
            // chipIdx: 芯片索引；2: 查询AICore 利用率
            int ret = getUtilizationFunc(npuCardId, chipIdx, 2, &utilizationRate);
            if (ret != 0) {
                ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                    GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                    "SimulateTaskRunner: DCMI get AICore failed, card=" << npuCardId
                                                                         << ", chip=" << chipIdx
                                                                         << ", error=" << ret);
                continue;
            }
            maxUtilAcrossCards = std::max(maxUtilAcrossCards, utilizationRate);
        }
        const auto elapsedMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - roundStart);
        if (elapsedMs.count() < perSampleTargetMs) {
            std::this_thread::sleep_for(std::chrono::milliseconds(perSampleTargetMs) - elapsedMs);
        }
    }

    ProcessAndReportNpuUtilization(static_cast<uint32_t>(maxUtilAcrossCards));
}

void SimulateTaskRunner::ProcessAndReportNpuUtilization(uint32_t localMax)
{
    const bool isSlave = (runMode_ == RunMode::NPU_ONLY);
    auto grpc = GRPCCommunicator::TryGetInstance();
    const bool hasGrpc = (grpc != nullptr);
    const bool isMasterNode = (hasGrpc && grpc->IsMaster());
    slaveNpuReportTimeoutThisRound_.store(false);
    uint32_t clusterMax = localMax;
    if (isSlave) {
        if (hasGrpc) {
            (void)grpc->SendNpuUtilizationReport(localMax);
        }
        npuUtil_.store(static_cast<int>(localMax));
        // Slave side: only print local NPU utilization each 5s window.
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
            "SimulateTaskRunner: slave local AICore usage(5s max)=" << localMax << '%');
    } else {
        if (isMasterNode) {
            const uint32_t slaveMax = grpc->GetSlaveMaxNpuUtilizationPercent();
            const bool timeoutFlag = grpc->ConsumeSlaveNpuReportTimeoutFlag();
            slaveNpuReportTimeoutThisRound_.store(timeoutFlag);
            clusterMax = std::max(localMax, slaveMax);
        }
        npuUtil_.store(static_cast<int>(clusterMax));
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
            "SimulateTaskRunner: AICore usage local(max over cards)="
            << localMax << "%, cluster(max with slaves if exist)="
            << clusterMax << '%');
    }

    std::unique_lock<std::mutex> locker(npuResultMutex_);
    npuResultCv_.notify_one();
}

} // namespace mindie_llm

