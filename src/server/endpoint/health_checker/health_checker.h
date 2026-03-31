
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

#ifndef OCK_HEALTH_CHECKER_H
#define OCK_HEALTH_CHECKER_H

#include <vector>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <set>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "log.h"
#include "error_queue.h"
#include "infer_instances.h"
#include "config_manager.h"
#include "simulate_task_runner.h"

namespace mindie_llm {

enum ServiceStatus : uint32_t {
    SERVICE_READY = 0,
    SERVICE_NORMAL = 1,
    SERVICE_ABNORMAL = 2,
    SERVICE_PAUSE = 3,
    SERVICE_INIT = 4,
    SERVICE_BUSY = 5
};

class HealthChecker {
public:
    static HealthChecker &GetInstance();
    ~HealthChecker();
    ServiceStatus GetServiceStatus();
    void GetStatusAndErrorList(ServiceStatus &status, std::vector<ErrorItem> &errorList);
    void UpdateNpuDeviceIds(const std::set<int> &npuDeviceIds);
    void UpdateStatus(const ServiceStatus &status);
    void EnqueueErrorMessage(
        const std::string &errCode, const std::string &createdBy,
        const std::chrono::time_point<std::chrono::system_clock> &timestamp = std::chrono::system_clock::now());
    void PrintNpuDeviceIds();
    void SetSendingDecodeMessageStatus(bool sendingDecodeMessageStatus) noexcept;
    std::string StatusToString(const ServiceStatus &status) const;
    bool Start();
    void Stop();
    bool IsEnabled() const noexcept;
    SimulateResult RunHttpTimedHealthCheck(uint32_t waitTime);
    HealthChecker(const HealthChecker &) = delete;
    HealthChecker &operator=(const HealthChecker &) = delete;
    HealthChecker(HealthChecker &&) = delete;
    HealthChecker &operator=(HealthChecker &&) = delete;

private:
    std::atomic<ServiceStatus> mServiceStatus;
    int mChipPerCard = 1;    // A2: 1, A3: 2
    std::set<int> mNpuDeviceCardIds;
    std::string mEngineName;
    mutable std::shared_mutex mNpuDevicesMutex;
    std::thread mCheckerThread;
    std::atomic<bool> mRunning;
    std::atomic<bool> mSendingDecodeMessage{false};
    std::mutex mStatusMutex; // 状态锁
    std::unordered_map<int, std::vector<int>> statusTransferMap;
    static constexpr int checkIntervalSeconds = 5;
    
    void CheckServiceStatus();
    bool WaitForLlmEngineReady();
    void PerformPeriodicHealthCheck();
    void GetChipPerCard();
    void UpdateErrorList(
        const std::string &errCode, const std::string &createdBy, const std::string &deviceIP, const int &deviceID,
        const std::chrono::time_point<std::chrono::system_clock> &timestamp = std::chrono::system_clock::now());

    bool CheckErrorListEmpty();
    // npu monitor for self health check
    std::string ExecuteCommand(const std::string &cmd) const;
    ServiceStatus CheckSimulateTask();
    void HandleHealthStatus();
    bool IsValidStatusTransition(const ServiceStatus &from, const ServiceStatus &to);

    // 虚推相关方法
    std::shared_ptr<ISimulateExecutor> CreateSimulateExecutor();
    bool StartSimulateTask();
    bool CreateAndInitSimulateRunner();
    bool InitNpuDeviceCardIds();

    // 虚推执行器和任务运行器
    std::shared_ptr<ISimulateExecutor> mSimulateExecutor;
    std::unique_ptr<SimulateTaskRunner> mSimulateRunner;
    std::atomic<bool> mSimulateTaskStarted{false};

    // 虚推探测开关和阈值
    int mNPUThreshold = 0;  // 默认为0，表示不开启虚推健康探测
    std::atomic<bool> mSimulateTaskEnable{false};

private:
    HealthChecker();
};

class SendingDecodeMessageScope {
public:
    explicit SendingDecodeMessageScope(HealthChecker &checker) noexcept;
    ~SendingDecodeMessageScope() noexcept;
    SendingDecodeMessageScope(const SendingDecodeMessageScope &) = delete;
    SendingDecodeMessageScope &operator=(const SendingDecodeMessageScope &) = delete;
private:
    HealthChecker &checker_;
};

} // namespace mindie_llm

#endif