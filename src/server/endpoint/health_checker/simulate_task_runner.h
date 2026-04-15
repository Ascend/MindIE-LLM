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

#ifndef SIMULATE_TASK_RUNNER_H
#define SIMULATE_TASK_RUNNER_H

#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <string>
#include <set>
#include <vector>
#include <utility>
#include <mutex>
#include <condition_variable>

namespace mindie_llm {

/// @brief 虚推执行结果
struct SimulateResult {
    enum class Status {
        SUCCESS,  // 成功
        BUSY,     // 繁忙（超时但 AICore 使用率高，视为健康）
        TIMEOUT,  // 超时
        ERROR     // 错误
    };
    Status status;
    std::string message;
};

/// @brief 虚推健康状态统计
struct SimulateHealthStatus {
    SimulateResult::Status lastStatus{SimulateResult::Status::SUCCESS};
    std::string lastMessage{"not started"};
    std::chrono::steady_clock::time_point lastUpdateTime;
    uint64_t successCount{0};
    uint64_t failureCount{0};
    bool isRunning{false};

    bool IsHealthy() const { return lastStatus == SimulateResult::Status::SUCCESS; }

    uint64_t GetSecondsSinceLastUpdate() const
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdateTime).count();
    }
};

/// @brief 虚推执行器接口
class ISimulateExecutor {
public:
    virtual ~ISimulateExecutor() = default;
    virtual SimulateResult RunSimulateOnce() = 0;
};

/// @brief 虚推任务运行器，管理定时虚推任务和 NPU 利用率检测
class SimulateTaskRunner {
public:
    enum class RunMode {
        SIMULATE_AND_NPU = 0, // 默认：虚推 + NPU 采样
        NPU_ONLY = 1          // 仅 NPU 采样（集中式 slave）
    };

    SimulateTaskRunner();
    ~SimulateTaskRunner();

    SimulateTaskRunner(const SimulateTaskRunner&) = delete;
    SimulateTaskRunner& operator=(const SimulateTaskRunner&) = delete;

    bool Init(std::shared_ptr<ISimulateExecutor> executor,
              const std::vector<std::pair<int, int>>& npuDeviceCardIds, int npuThreshold,
              RunMode runMode = RunMode::SIMULATE_AND_NPU, int chipPerCard = 1);

    /// @brief 检查是否初始化成功
    bool IsValid() const { return isValid_; }

    void Start(uint32_t intervalSeconds = 5);
    void Stop();
    void Pause();
    void Resume();

    SimulateHealthStatus GetHealthStatus() const;
    bool IsRunning() const { return running_.load(); }
    bool IsPaused() const { return paused_.load(); }
    int GetNpuUtilization() const { return npuUtil_.load(); }

private:
    void TaskLoop();
    void NpuCheckLoop();
    void CheckAicoreUtilization();
    void ProcessAndReportNpuUtilization(uint32_t localMax);
    void TriggerNpuCheck();
    void WaitForNpuCheckComplete();
    void UpdateHealthStatus(const SimulateResult& result);
    bool ShouldRunSimulate();

    std::shared_ptr<ISimulateExecutor> executor_;
    std::vector<std::pair<int, int>> npuDeviceCardIds_;  // NPU 设备卡 ID 集合（Init 时传入，不可为空）
    bool isValid_{false};  // 是否初始化成功
    int npuThreshold_ = 10; // 适配大部分场景的检测阈值
    RunMode runMode_{RunMode::SIMULATE_AND_NPU};
    int chipPerCard_{1}; // A2=1, A3=2

    // 虚推任务线程
    std::thread taskThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stopRequested_{false};
    uint32_t intervalSeconds_{5};

    // NPU 检测线程
    std::thread npuCheckThread_;
    std::atomic<bool> npuCheckStopRequested_{false};
    std::atomic<bool> npuCheckRequested_{false};  // 触发标志，避免虚假唤醒
    std::atomic<int> npuUtil_{-1};  // -1 表示未检测
    std::atomic<bool> slaveNpuReportTimeoutThisRound_{false};

    // NPU 检测同步原语
    std::mutex npuCheckMutex_;
    std::condition_variable npuCheckCv_;
    std::mutex npuResultMutex_;
    std::condition_variable npuResultCv_;

    mutable std::shared_mutex statusMutex_;
    SimulateHealthStatus healthStatus_;

    int lastAicoreUtil_{-1};
    int loopsSinceSimulate_{0};
};

} // namespace mindie_llm

#endif // SIMULATE_TASK_RUNNER_H

