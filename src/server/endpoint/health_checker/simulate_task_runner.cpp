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
#include "log.h"
#include "dcmi_wrapper.h"

namespace mindie_llm {

constexpr int POLLING_INTERVAL_MS = 100;

SimulateTaskRunner::SimulateTaskRunner()
{
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Instance created, waiting for Init.");
}

bool SimulateTaskRunner::Init(std::shared_ptr<ISimulateExecutor> executor,
                              const std::set<int>& npuDeviceCardIds, int npuThreshold)
{
    if (isValid_) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
            "SimulateTaskRunner::Init: Already initialized");
        return true;
    }
    if (!executor) {
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
    npuDeviceCardIds_ = npuDeviceCardIds;
    isValid_ = true;

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner::Init: Initialized with "
        << npuDeviceCardIds_.size() << " NPU device(s).");
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
        SimulateResult result = executor_->RunSimulateOnce();
        WaitForNpuCheckComplete();

        // 超时时结合 NPU 利用率判断：高利用率视为繁忙而非故障
        if (result.status == SimulateResult::Status::TIMEOUT) {
            int npuUtil = GetNpuUtilization();
            if (npuUtil > npuThreshold_) {
                result.status = SimulateResult::Status::BUSY;
                result.message = "Timeout but AICore usage > " + std::to_string(npuThreshold_) + "%";
            }
        }

        UpdateHealthStatus(result);

        ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
            "SimulateTaskRunner: Completed. status=" << static_cast<int>(result.status)
            << ", message=" << result.message);

        for (uint32_t i = 0; i < intervalSeconds_ && !stopRequested_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (paused_.load()) {
                break;
            }
        }
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "SimulateTaskRunner: Task loop exited.");
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
        auto lastTimePoint = std::chrono::steady_clock::now() + std::chrono::seconds(1);
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
    DCMIWrapper& dcmiWrapper = DCMIWrapper::GetInstance();
    int firstNpuId = *(npuDeviceCardIds_.begin());
    unsigned int maxUtil = 0;
    unsigned int utilizationRate = 0;
    constexpr int nSamples = 4;

    auto getUtilizationFunc = dcmiWrapper.GetFunction<int(*)(int, int, int, unsigned int*)>(
        "dcmi_get_device_utilization_rate");
    if (!getUtilizationFunc) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "SimulateTaskRunner: Failed to get utilization function");
        return;
    }

    // 多次采样取最大值
    for (int i = 0; i < nSamples; i++) {
        // 0: 芯片索引，表示查询NPU的第一个芯片
        // 2: DCMI参数项，表示查询AI Core利用率
        int ret = getUtilizationFunc(firstNpuId, 0, 2, &utilizationRate);
        if (ret != 0) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                "SimulateTaskRunner: DCMI get AICore failed, error: " << ret);
            break;
        }
        maxUtil = std::max(maxUtil, utilizationRate);
        if (i < nSamples - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
        }
    }

    npuUtil_.store(static_cast<int>(maxUtil));
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
        "SimulateTaskRunner: AICore usage of card " << firstNpuId << " is " << maxUtil << '%');

    std::unique_lock<std::mutex> locker(npuResultMutex_);
    npuResultCv_.notify_one();
}

} // namespace mindie_llm

