/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */
#include "health_checker.h"

#include <cstdio>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>

#include "config_manager.h"
#include "config_manager_impl.h"
#include "dmi_role.h"
#include "endpoint_def.h"
#include "infer_instances.h"
#include "simulate_request_executor.h"

namespace mindie_llm {

constexpr size_t EXECUTE_COMMAND_BUFFER_SIZE = 128;
constexpr size_t MAX_ENGINE_NOT_READY_COUNT = 120;  // 10分钟不调度

HealthChecker::HealthChecker() : mRunning(false) {
    const ServerConfig &serverConfig = GetServerConfig();
    mNPUThreshold = serverConfig.npuUsageThreshold;
    mIsCentralizedNode = ConfigManager::GetInstance().IsMultiNodeInfer() && !serverConfig.distDPServerEnabled;
    if (mIsCentralizedNode) {
        mIsCentralizedMaster = GetRanktableParam().isMaster;
    }

    if (mNPUThreshold != 0) {
        mSimulateTaskEnable.store(true);
    }
    mServiceStatus.store(SERVICE_INIT);
    GetChipPerCard();
    statusTransferMap = {
        {SERVICE_INIT, {SERVICE_NORMAL, SERVICE_BUSY}},
        {SERVICE_NORMAL, {SERVICE_PAUSE, SERVICE_ABNORMAL, SERVICE_BUSY}},
        {SERVICE_BUSY, {SERVICE_PAUSE, SERVICE_ABNORMAL, SERVICE_NORMAL}},
        {SERVICE_PAUSE, {SERVICE_READY, SERVICE_NORMAL}},
        {SERVICE_ABNORMAL, {SERVICE_NORMAL, SERVICE_BUSY}},
        {SERVICE_READY, {SERVICE_NORMAL, SERVICE_BUSY}},
        // other transfers are invalid
    };
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Healthchecker instance created.");
}

void HealthChecker::GetChipPerCard() {
    std::string cmd = "npu-smi info -t usages -i 0 | awk '/Chip Count/ {print $NF}'";
    try {
        std::string output = ExecuteCommand(cmd);
        mChipPerCard = std::stoi(output);
        if (mChipPerCard <= 0) {
            ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                      GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                      "HealthChecker: Invalid Chip Count value from npu-smi: " << output << ". Defaulting to 1.");
            mChipPerCard = 1;
        } else {
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Detected Chip Count: " << mChipPerCard);
        }
    } catch (const std::exception &e) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: Failed to parse Chip Count from npu-smi "
                  "output. Exception: "
                      << e.what() << ". Defaulting to 1.");
        mChipPerCard = 1;
    }
}

void HealthChecker::PrintNpuDeviceIds() {
    std::shared_lock<std::shared_mutex> lock(mNpuDevicesMutex);
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto &t : mNpuDeviceCardIds) {
        if (!first) {
            ss << ", ";
        }
        first = false;
        ss << t.first << ":" << t.second;
    }
    ss << "}";
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Device Card IDs (card:chip): " << ss.str());
}

std::string HealthChecker::StatusToString(const ServiceStatus &status) const {
    switch (status) {
        case SERVICE_READY:
            return "SERVICE_READY";
        case SERVICE_NORMAL:
            return "SERVICE_NORMAL";
        case SERVICE_ABNORMAL:
            return "SERVICE_ABNORMAL";
        case SERVICE_PAUSE:
            return "SERVICE_PAUSE";
        case SERVICE_INIT:
            return "SERVICE_INIT";
        case SERVICE_BUSY:
            return "SERVICE_BUSY";
        default:
            return "UNKNOWN";
    }
}

HealthChecker::~HealthChecker() {
    Stop();
    // 停止虚推任务
    {
        if (mSimulateRunner != nullptr && mSimulateTaskStarted.load()) {
            mSimulateRunner->Stop();
            mSimulateTaskStarted.store(false);
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Stopped simulate task.");
        }
        mSimulateRunner.reset();
        mSimulateExecutor.reset();
    }
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Healthchecker instance destroyed.");
}

bool HealthChecker::Start() {
    if (!mRunning.load()) {
        mRunning.store(true);
        mCheckerThread = std::thread(&HealthChecker::CheckServiceStatus, this);
        ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Health check thread started.");
        return true;
    } else {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, STATUS_WARNING),
                  "HealthChecker: Attempted to start already running health "
                  "check thread");
        return false;
    }
}

bool HealthChecker::IsEnabled() const noexcept { return mRunning.load(); }

void HealthChecker::Stop() {
    if (mRunning.load()) {
        mRunning.store(false);  // 修复：应该设置为 false 以停止线程
        if (mCheckerThread.joinable()) {
            mCheckerThread.join();
        }
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Health check thread stopped.");
    } else {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, STATUS_WARNING),
                  "HealthChecker: Attempted to stop non-running health check thread");
    }
}

HealthChecker &HealthChecker::GetInstance() {
    static HealthChecker instance;
    return instance;
}

ServiceStatus HealthChecker::GetServiceStatus() { return mServiceStatus.load(); }

bool HealthChecker::CheckErrorListEmpty() { return ErrorQueue::GetInstance().Size() == 0; }

void HealthChecker::GetStatusAndErrorList(ServiceStatus &status, std::vector<ErrorItem> &errorList) {
    status = mServiceStatus.load();
    ErrorItem item;
    while (ErrorQueue::GetInstance().PopError(item)) {
        errorList.push_back(item);
    }
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: GetStatusAndErrorList called. Status: "
                                                << status << ", ErrorList size: " << errorList.size());
}

bool HealthChecker::WaitForLlmEngineReady() {
    // 集中式多机 slave 不持有本地 LlmEngine，若等待 engine ready 会一直阻塞，
    // 导致 NPU_ONLY 探测任务无法启动。该场景直接进入后续健康检查流程。
    if (mIsCentralizedNode && !mIsCentralizedMaster) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                  "HealthChecker: centralized slave skip LlmEngine ready wait, "
                  "enter NPU_ONLY health check.");
        UpdateStatus(SERVICE_NORMAL);
        return true;
    }
    while (mRunning.load()) {
        // 检查当前状态是否为INIT，如果已被外部修改则跳过初始化等待
        ServiceStatus currentStatus = GetServiceStatus();
        if (currentStatus != SERVICE_INIT) {
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status already changed to "
                                                       << StatusToString(currentStatus) << ", exit init waiting.");
            return true;
        }

        if (!GetInferInstance()->IsLlmEngineReady()) {
            // Service not init
            std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
            continue;
        }

        if (!IsDmiInitWaitSatisfied()) {
            std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
            continue;
        }

        // 额外休眠一次，确保初始化完成
        std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Init finished, update status to normal");
        UpdateStatus(SERVICE_NORMAL);
        return true;
    }
    return false;
}

bool HealthChecker::IsDmiInitWaitSatisfied() {
    if (GetServerConfig().inferMode != INFER_MODE_DMI) {
        return true;
    }
    if (GetInferInstance()->GetPDRoleStatus() != PDRoleStatus::READY || !DmiRole::GetInstance()->IsHealthy()) {
        return false;
    }
    return true;
}

void HealthChecker::PerformPeriodicHealthCheck() {
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
              "HealthChecker: Starting health check loop with interval: " << checkIntervalSeconds << " seconds");
    ServiceStatus previousStatus = GetServiceStatus();
    while (mRunning.load()) {
        std::unique_lock lock(mStatusMutex);
        ServiceStatus currentStatus = GetServiceStatus();

        // 检测状态变化，处理虚推任务的暂停/恢复
        bool wasPauseOrReady = (previousStatus == SERVICE_PAUSE || previousStatus == SERVICE_READY);
        bool isPauseOrReady = (currentStatus == SERVICE_PAUSE || currentStatus == SERVICE_READY);

        if (!wasPauseOrReady && isPauseOrReady) {
            // 从正常状态进入 PAUSE/READY 状态，暂停虚推任务
            if (mSimulateRunner != nullptr && mSimulateTaskStarted.load()) {
                mSimulateRunner->Pause();
                ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                          "HealthChecker: Paused simulate task due to status "
                          "change to "
                              << StatusToString(currentStatus));
            }
        } else if (wasPauseOrReady && !isPauseOrReady) {
            // 从 PAUSE/READY 状态恢复，恢复虚推任务
            if (mSimulateRunner != nullptr && mSimulateTaskStarted.load()) {
                mSimulateRunner->Resume();
                ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                          "HealthChecker: Resumed simulate task due to status "
                          "change to "
                              << StatusToString(currentStatus));
            }
        }

        // PAUSE/READY状态下跳过健康检查
        previousStatus = currentStatus;
        if (isPauseOrReady) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
            continue;
        }

        // 检查是否处于跳过健康检查的场景
        if (ShouldSkipHealthCheck()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
            continue;
        }

        // 持锁状态下更新状态信息
        HandleHealthStatus();
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
    }
}

bool HealthChecker::ShouldSkipHealthCheck() {
    // 此时虚推阻塞，因为LLM侧的资源不足，非异常
    if (!GetInferInstance()->IsLlmEngineReady()) {
        llmNotReadyCount++;

        // 连续堵塞10分钟以上，告警
        if (llmNotReadyCount >= MAX_ENGINE_NOT_READY_COUNT) {
            ULOG_WARN(
                SUBMODLE_NAME_HEALTHCHECKER,
                GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                "HealthChecker: LLM engine not ready for " << llmNotReadyCount << " times, will report health status.");
        }

        mServiceStatus.store(SERVICE_NORMAL);
        std::string errCode = GenerateHealthCheckerErrCode(INFO, SUBMODLE_FEATURE_SECURE, SIMULATE_NORMAL);
        ErrorQueue::GetInstance().EnqueueErrorMessage(errCode, SUBMODLE_NAME_HEALTHCHECKER);

        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: LLM engine not ready, skip health check");
        return true;
    }

    llmNotReadyCount = 0;
    return false;
}

void HealthChecker::SetSendingMessageStatus(bool sendingMessageStatus) noexcept {
    mSendingMessage.store(sendingMessageStatus);
}

void HealthChecker::HandleHealthStatus() {
    ServiceStatus status = CheckSimulateTask();
    std::string errCode;
    const bool isSending = mSendingMessage.load();

    if (status == SERVICE_ABNORMAL) {
        if (!mFirstSimulateAbnormalSuppressed) {
            mFirstSimulateAbnormalSuppressed = true;
            status = SERVICE_NORMAL;
            errCode = GenerateHealthCheckerErrCode(INFO, SUBMODLE_FEATURE_SECURE, SIMULATE_NORMAL);
        } else if (isSending) {
            status = SERVICE_NORMAL;
            errCode = GenerateHealthCheckerErrCode(INFO, SUBMODLE_FEATURE_SECURE, SIMULATE_NORMAL);
            ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                      GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                      "P node is sending a request to D. gRPC may block."
                      "Simulate health check does not mark the service as abnormal.");
        } else {
            errCode = GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, STATUS_WARNING);
        }
    } else if (status == SERVICE_NORMAL || status == SERVICE_BUSY) {
        mFirstSimulateAbnormalSuppressed = false;
        errCode = GenerateHealthCheckerErrCode(INFO, SUBMODLE_FEATURE_SECURE, SIMULATE_NORMAL);
    }

    if (!errCode.empty()) {
        ErrorQueue::GetInstance().EnqueueErrorMessage(errCode, SUBMODLE_NAME_HEALTHCHECKER);
    }
    mServiceStatus.store(status);
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
              "HealthChecker: The simulate infer health check result is " << StatusToString(status));
}

void HealthChecker::CheckServiceStatus() {
    // 等待LLM引擎启动完成
    if (!WaitForLlmEngineReady()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: Exiting health check loop during init.");
        return;
    }

    // 检查是否被停止
    if (!mRunning.load()) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: Exiting health check loop during init.");
        return;
    }

    // 虚推健康探测是否开启
    if (!mSimulateTaskEnable.load()) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Simulate infer health task is not enabled");
        return;
    }

    // 边云协同场景不开启健康检查
    if (mindie_llm::ConfigManager::GetInstance().IslayerwiseDisaggregated()) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                  "HealthChecker: Simulate infer health task disabled in "
                  "layerwise-disaggregated mode");
        return;
    }

    // 启动虚推任务
    if (!StartSimulateTask()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: Failed to start simulate task.");
        return;
    }

    // 周期健康检查
    PerformPeriodicHealthCheck();

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Exiting health check loop.");
}

std::string HealthChecker::ExecuteCommand(const std::string &cmd) const {
    std::array<char, EXECUTE_COMMAND_BUFFER_SIZE> buffer;
    std::string result;
    std::unique_ptr<FILE, int (*)(FILE *)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: popen() failed!");
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::shared_ptr<ISimulateExecutor> HealthChecker::CreateSimulateExecutor() {
    // 使用独立的 SimulateRequestExecutor 进行虚推
    // 不再依赖 SingleLLMReqHandlerBase
    auto &serverConfig = GetServerConfig();
    InferReqType reqType =
        (serverConfig.inferMode == INFER_MODE_DMI) ? InferReqType::REQ_PREFILL : InferReqType::REQ_STAND_INFER;
    std::string mode = (serverConfig.inferMode == INFER_MODE_DMI) ? "DMI" : "Standard";

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Creating simulate executor for "
                                               << mode << " mode, reqType=" << static_cast<int>(reqType));

    return SimulateRequestExecutor::Create(reqType);
}

bool HealthChecker::StartSimulateTask() {
    // 如果任务已经启动，返回
    if (mSimulateTaskStarted.load() && mSimulateRunner != nullptr) {
        return true;
    }

    ServiceStatus currentStatus = GetServiceStatus();
    while (mRunning.load() && (currentStatus == SERVICE_PAUSE || currentStatus == SERVICE_READY)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        currentStatus = GetServiceStatus();
    }

    if (currentStatus == SERVICE_ABNORMAL) {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                  "HealthChecker: Abnormal status, stop start simulate task.");
        return false;
    }

    if (!CreateAndInitSimulateRunner()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: The SimulateRunner create failed");
        return false;
    }

    // 启动周期性虚推任务（间隔使用健康检查的间隔）
    mSimulateRunner->Start(checkIntervalSeconds);
    mSimulateTaskStarted.store(true);

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
              "HealthChecker: Started periodic simulate task with interval=" << checkIntervalSeconds << "s");

    return true;
}

bool HealthChecker::CreateAndInitSimulateRunner() {
    // 首次启动：创建执行器
    if (mSimulateExecutor == nullptr) {
        mSimulateExecutor = CreateSimulateExecutor();
        if (mSimulateExecutor == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                       "HealthChecker: Failed to create simulate executor");
            return false;
        }
    }

    // 创建并初始化任务运行器
    if (mSimulateRunner == nullptr) {
        if (!InitNpuDeviceCardIds()) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                       "HealthChecker: Failed to init NPU device IDs");
            return false;
        }
        mSimulateRunner = std::make_unique<SimulateTaskRunner>();
    }

    SimulateTaskRunner::RunMode runMode = (mIsCentralizedNode && !mIsCentralizedMaster)
                                              ? SimulateTaskRunner::RunMode::NPU_ONLY
                                              : SimulateTaskRunner::RunMode::SIMULATE_AND_NPU;
    if (!mSimulateRunner->Init(mSimulateExecutor, mNpuDeviceCardIds, mNPUThreshold, runMode, mChipPerCard)) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: Failed to init simulate runner");
        mSimulateRunner.reset();
        return false;
    }

    return true;
}

bool HealthChecker::InitNpuDeviceCardIds() {
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    auto &serverConfig = configManager.GetServerConfig();
    const int chipsPerCard = (mChipPerCard > 0) ? mChipPerCard : 1;
    const auto &npuDeviceIds = configManager.GetBackendConfig().npuDeviceIds;
    const auto loadNpuIdsFromBackend = [this, &npuDeviceIds, chipsPerCard]() -> bool {
        if (npuDeviceIds.empty() || npuDeviceIds[0].empty()) {
            return false;
        }
        std::unique_lock<std::shared_mutex> lock(mNpuDevicesMutex);
        mNpuDeviceCardIds.clear();
        for (const auto &id : npuDeviceIds[0]) {
            const int lid = static_cast<int>(id);
            // backend 为逻辑芯片 ID：映射为 (物理卡, 卡内 chip)。A3 同卡多 DP
            // 时每进程只采样本进程 chip。
            mNpuDeviceCardIds.emplace_back(lid / chipsPerCard, lid % chipsPerCard);
        }
        return !mNpuDeviceCardIds.empty();
    };

    // PD分离模式：NPU卡号已经被Controller下发过
    // 多机 slave 场景下 backendConfig.npuDeviceIds 已由 ranktable.local.device
    // 覆盖为本机设备。
    if (serverConfig.inferMode == INFER_MODE_DMI) {
        {
            std::shared_lock<std::shared_mutex> lock(mNpuDevicesMutex);
            if (!mNpuDeviceCardIds.empty()) {
                return true;
            }
        }
        if (!loadNpuIdsFromBackend()) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                       GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                       "HealthChecker: NPU device IDs are empty in DMI mode. "
                       "Neither UpdateNpuDeviceIds nor backendConfig provides "
                       "valid IDs.");
            return false;
        }
        return true;
    }

    // 标准/混布模式：backendConfig配置不能为空
    if (!loadNpuIdsFromBackend()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: NPU device id list is empty in config");
        return false;
    }
    return true;
}

ServiceStatus HealthChecker::CheckSimulateTask() {
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Starting simulate inference check.");

    // 如果任务已经启动，检查健康状态并返回
    if (mSimulateTaskStarted.load() && mSimulateRunner != nullptr) {
        auto healthStatus = mSimulateRunner->GetHealthStatus();
        switch (healthStatus.lastStatus) {
            case SimulateResult::Status::SUCCESS:
                ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Simulate task is healthy. "
                                                            << "successCount=" << healthStatus.successCount
                                                            << ", failureCount=" << healthStatus.failureCount);
                return SERVICE_NORMAL;
            case SimulateResult::Status::BUSY:
                ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Simulate task is busy (Aicore usage high). "
                                                            << "successCount=" << healthStatus.successCount
                                                            << ", failureCount=" << healthStatus.failureCount);
                return SERVICE_BUSY;
            default:
                ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                          GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
                          "HealthChecker: Simulate task is unhealthy. "
                              << "lastMessage=" << healthStatus.lastMessage
                              << ", failureCount=" << healthStatus.failureCount);
                return SERVICE_ABNORMAL;
        }
    }

    // 虚推任务未启动或runner为空，返回abnormal表示检查未通过
    ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
              GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_WARNING),
              "HealthChecker: Simulate task not started or runner is null.");
    return SERVICE_ABNORMAL;
}

bool HealthChecker::IsValidStatusTransition(const ServiceStatus &from, const ServiceStatus &to) {
    if (statusTransferMap.find(from) == statusTransferMap.end() ||
        std::find(statusTransferMap[from].begin(), statusTransferMap[from].end(), to) ==
            statusTransferMap[from].end()) {
        return false;
    }
    return true;
}

void HealthChecker::UpdateStatus(const ServiceStatus &status) {
    std::unique_lock lock(mStatusMutex);
    if (mServiceStatus.load() == status) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status unchanged: " << StatusToString(mServiceStatus));
        return;
    }
    if (!IsValidStatusTransition(mServiceStatus.load(), status)) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: Invalid status transition from " << StatusToString(mServiceStatus.load()) << " to "
                                                                    << StatusToString(status));
        return;
    }
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status changed from "
                                               << StatusToString(mServiceStatus.load()) << " to "
                                               << StatusToString(status));
    mServiceStatus.store(status);
}

void HealthChecker::EnqueueErrorMessage(const std::string &errCode, const std::string &createdBy,
                                        const std::chrono::time_point<std::chrono::system_clock> &timestamp) {
    ErrorQueue::GetInstance().EnqueueErrorMessage(errCode, createdBy, timestamp);
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
              "HealthChecker: New error added. Error code: " << errCode << ", createdBy: " << createdBy);

    UpdateStatus(SERVICE_ABNORMAL);
}

void HealthChecker::UpdateNpuDeviceIds(const std::set<int> &npuDeviceIds) {
    const int chipsPerCard = (mChipPerCard > 0) ? mChipPerCard : 1;
    {
        std::unique_lock<std::shared_mutex> lock(mNpuDevicesMutex);
        mNpuDeviceCardIds.clear();
        for (const auto &id : npuDeviceIds) {
            mNpuDeviceCardIds.emplace_back(id / chipsPerCard, id % chipsPerCard);
        }
    }
    PrintNpuDeviceIds();
}

SimulateResult HealthChecker::RunHttpTimedHealthCheck(uint32_t waitTime) {
    auto &serverConfig = GetServerConfig();
    InferReqType reqType =
        (serverConfig.inferMode == INFER_MODE_DMI) ? InferReqType::REQ_PREFILL : InferReqType::REQ_STAND_INFER;

    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: RunHttpTimedHealthCheck. reqType="
                                                << static_cast<int>(reqType) << ", waitTime=" << waitTime << "s");

    auto executor = SimulateRequestExecutor::Create(reqType);
    SimulateResult result = executor->RunSimulateOnce(waitTime);
    // BUSY 状态在 HTTP 接口中视为健康
    if (result.status == SimulateResult::Status::BUSY) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: RunHttpTimedHealthCheck busy but healthy");
    }

    return result;
}

SendingMessageScope::SendingMessageScope(HealthChecker &checker) noexcept : checker_(checker) {
    checker_.SetSendingMessageStatus(true);
}

SendingMessageScope::~SendingMessageScope() noexcept { checker_.SetSendingMessageStatus(false); }

}  // namespace mindie_llm
