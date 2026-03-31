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
#include <ctime>
#include <memory>
#include <cstdio>
#include <iostream>
#include <sstream>
#include "config_manager.h"
#include "config_manager_impl.h"
#include "endpoint_def.h"
#include "simulate_request_executor.h"
#include "infer_instances.h"
#include "health_checker.h"


namespace mindie_llm {

constexpr size_t EXECUTE_COMMAND_BUFFER_SIZE = 128;


HealthChecker::HealthChecker() : mRunning(false)
{
    const ServerConfig& serverConfig = GetServerConfig();
    mNPUThreshold = serverConfig.npuUsageThreshold;

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

void HealthChecker::GetChipPerCard()
{
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
                  "HealthChecker: Failed to parse Chip Count from npu-smi output. Exception: " << e.what()
                                                                                               << ". Defaulting to 1.");
        mChipPerCard = 1;
    }
}

void HealthChecker::PrintNpuDeviceIds()
{
    std::shared_lock<std::shared_mutex> lock(mNpuDevicesMutex);
    std::stringstream ss;
    ss << "{";
    for (const auto &id : mNpuDeviceCardIds) {
        if (id != *mNpuDeviceCardIds.begin()) {
            ss << ", ";
        }
        ss << id;
    }
    ss << "}";
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: NPU Device Card IDs: " << ss.str());
}

std::string HealthChecker::StatusToString(const ServiceStatus &status) const
{
    switch (status) {
        case SERVICE_READY: return "SERVICE_READY";
        case SERVICE_NORMAL: return "SERVICE_NORMAL";
        case SERVICE_ABNORMAL: return "SERVICE_ABNORMAL";
        case SERVICE_PAUSE: return "SERVICE_PAUSE";
        case SERVICE_INIT: return "SERVICE_INIT";
        case SERVICE_BUSY: return "SERVICE_BUSY";
        default: return "UNKNOWN";
    }
}

HealthChecker::~HealthChecker()
{
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

bool HealthChecker::Start()
{
    if (!mRunning.load()) {
        mRunning.store(true);
        mCheckerThread = std::thread(&HealthChecker::CheckServiceStatus, this);
        ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Health check thread started.");
        return true;
    } else {
        ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                  GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, STATUS_WARNING),
                  "HealthChecker: Attempted to start already running health check thread");
        return false;
    }
}

bool HealthChecker::IsEnabled() const noexcept
{
    return mRunning.load();
}

void HealthChecker::Stop()
{
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

HealthChecker &HealthChecker::GetInstance()
{
    static HealthChecker instance;
    return instance;
}

ServiceStatus HealthChecker::GetServiceStatus() { return mServiceStatus.load(); }

bool HealthChecker::CheckErrorListEmpty() { return ErrorQueue::GetInstance().Size() == 0; }

void HealthChecker::GetStatusAndErrorList(ServiceStatus &status, std::vector<ErrorItem> &errorList)
{
    status = mServiceStatus.load();
    ErrorItem item;
    while (ErrorQueue::GetInstance().PopError(item)) {
        errorList.push_back(item);
    }
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: GetStatusAndErrorList called. Status: "
                                                << status << ", ErrorList size: " << errorList.size());
}

bool HealthChecker::WaitForLlmEngineReady()
{
    while (mRunning.load()) {
        // 检查当前状态是否为INIT，如果已被外部修改则跳过初始化等待
        ServiceStatus currentStatus = GetServiceStatus();
        if (currentStatus != SERVICE_INIT) {
            ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                "HealthChecker: Status already changed to " << StatusToString(currentStatus) << ", exit init waiting.");
            return true;
        }

        if (!GetInferInstance()->IsLlmEngineReady()) {
            // Service not init
            std::this_thread::sleep_for(std::chrono::seconds(checkIntervalSeconds));
            continue;
        }

        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Init finished, update status to normal");
        UpdateStatus(SERVICE_NORMAL);
        return true;
    }
    return false;
}

void HealthChecker::PerformPeriodicHealthCheck()
{
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
                    "HealthChecker: Paused simulate task due to status change to "
                    << StatusToString(currentStatus));
            }
        } else if (wasPauseOrReady && !isPauseOrReady) {
            // 从 PAUSE/READY 状态恢复，恢复虚推任务
            if (mSimulateRunner != nullptr && mSimulateTaskStarted.load()) {
                mSimulateRunner->Resume();
                ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
                    "HealthChecker: Resumed simulate task due to status change to "
                    << StatusToString(currentStatus));
            }
        }

        previousStatus = currentStatus;
        if (isPauseOrReady) {
            // PAUSE/READY状态下跳过健康检查
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

void HealthChecker::SetSendingDecodeMessageStatus(bool sendingDecodeMessageStatus) noexcept
{
    mSendingDecodeMessage.store(sendingDecodeMessageStatus);
}

void HealthChecker::HandleHealthStatus()
{
    // NORMAL、ABNORMAL和BUSY可以互相转换，无需检查状态转移
    ServiceStatus status = CheckSimulateTask();
    std::string errCode;
    const bool isSendingDecode = mSendingDecodeMessage.load();

    if (status == SERVICE_ABNORMAL) {
        if (isSendingDecode) {
            ULOG_WARN(SUBMODLE_NAME_HEALTHCHECKER,
                GenerateHealthCheckerErrCode(WARNING, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                "P node is sending a request to D. gRPC may block."
                "Simulate health check does not mark the service as abnormal.");
        } else {
            errCode = GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, STATUS_WARNING);
        }
    } else if (status == SERVICE_NORMAL || status == SERVICE_BUSY) {
        // normal 和 busy 都当正常，统一上报 071120
        errCode = GenerateHealthCheckerErrCode(INFO, SUBMODLE_FEATURE_SECURE, SIMULATE_NORMAL);
    }

    if (!errCode.empty()) {
        ErrorQueue::GetInstance().EnqueueErrorMessage(errCode, SUBMODLE_NAME_HEALTHCHECKER);
    }

    if (status == SERVICE_ABNORMAL && isSendingDecode) {
        mServiceStatus.store(SERVICE_NORMAL);
    } else {
        mServiceStatus.store(status);
    }

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: The simulate infer health check result is "
        << StatusToString(status));
}

void HealthChecker::CheckServiceStatus()
{
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
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
            "HealthChecker: Simulate infer health task is not enabled");
        return;
    }

    // 边云协同场景不开启健康检查
    if (mindie_llm::ConfigManager::GetInstance().IslayerwiseDisaggregated()) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
            "HealthChecker: Simulate infer health task disabled in layerwise-disaggregated mode");
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

std::string HealthChecker::ExecuteCommand(const std::string &cmd) const
{
    std::array<char, EXECUTE_COMMAND_BUFFER_SIZE> buffer;
    std::string result;
    std::unique_ptr<FILE, int(*)(FILE *)> pipe(popen(cmd.c_str(), "r"), pclose);
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

std::shared_ptr<ISimulateExecutor> HealthChecker::CreateSimulateExecutor()
{
    // 使用独立的 SimulateRequestExecutor 进行虚推
    // 不再依赖 SingleLLMReqHandlerBase
    auto &serverConfig = GetServerConfig();
    InferReqType reqType = (serverConfig.inferMode == INFER_MODE_DMI)
        ? InferReqType::REQ_PREFILL : InferReqType::REQ_STAND_INFER;
    std::string mode = (serverConfig.inferMode == INFER_MODE_DMI) ? "DMI" : "Standard";

    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
        "HealthChecker: Creating simulate executor for " << mode << " mode, reqType="
        << static_cast<int>(reqType));

    return SimulateRequestExecutor::Create(reqType);
}

bool HealthChecker::StartSimulateTask()
{
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
        "HealthChecker: Started periodic simulate task with interval="
        << checkIntervalSeconds << "s");

    return true;
}

bool HealthChecker::CreateAndInitSimulateRunner()
{
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
    
    if (!mSimulateRunner->Init(mSimulateExecutor, mNpuDeviceCardIds, mNPUThreshold)) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "HealthChecker: Failed to init simulate runner");
        mSimulateRunner.reset();
        return false;
    }
    
    return true;
}

bool HealthChecker::InitNpuDeviceCardIds()
{
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    auto &serverConfig = configManager.GetServerConfig();
    const auto& npuDeviceIds = configManager.GetBackendConfig().npuDeviceIds;

    // PD分离模式：NPU卡号已经被Controller下发过
    if (serverConfig.inferMode == INFER_MODE_DMI) {
        if (mNpuDeviceCardIds.empty()) {
            ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                "HealthChecker: NPU device IDs are empty in request body from Controller");
            return false;
        }
        return true;
    }

    // 标准/混布模式：backendConfig配置不能为空
    if (npuDeviceIds.empty() || npuDeviceIds[0].empty()) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
            GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
            "HealthChecker: NPU device id list is empty in config");
        return false;
    }

    // 直接使用配置文件中的设备ID
    {
        std::unique_lock<std::shared_mutex> lock(mNpuDevicesMutex);
        mNpuDeviceCardIds.clear();
        for (const auto& id : npuDeviceIds[0]) {
            mNpuDeviceCardIds.insert(static_cast<int>(id));
        }
    }
    return true;
}

ServiceStatus HealthChecker::CheckSimulateTask()
{
    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Starting simulate inference check.");

    // 如果任务已经启动，检查健康状态并返回
    if (mSimulateTaskStarted.load() && mSimulateRunner != nullptr) {
        auto healthStatus = mSimulateRunner->GetHealthStatus();
        switch (healthStatus.lastStatus) {
            case SimulateResult::Status::SUCCESS:
                ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
                    "HealthChecker: Simulate task is healthy. "
                    << "successCount=" << healthStatus.successCount
                    << ", failureCount=" << healthStatus.failureCount);
                return SERVICE_NORMAL;
            case SimulateResult::Status::BUSY:
                ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
                    "HealthChecker: Simulate task is busy (Aicore usage high). "
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

bool HealthChecker::IsValidStatusTransition(const ServiceStatus &from, const ServiceStatus &to)
{
    if (statusTransferMap.find(from) == statusTransferMap.end() ||
        std::find(statusTransferMap[from].begin(), statusTransferMap[from].end(), to) ==
        statusTransferMap[from].end()) {
        return false;
    }
    return true;
}

void HealthChecker::UpdateStatus(const ServiceStatus &status)
{
    std::unique_lock lock(mStatusMutex);
    if (mServiceStatus.load() == status) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status unchanged: " << StatusToString(mServiceStatus));
        return;
    }
    if (!IsValidStatusTransition(mServiceStatus.load(), status)) {
        ULOG_ERROR(SUBMODLE_NAME_HEALTHCHECKER,
                   GenerateHealthCheckerErrCode(ERROR, SUBMODLE_FEATURE_SECURE, CHECK_ERROR),
                   "HealthChecker: Invalid status transition from " << StatusToString(mServiceStatus.load())
                                                                      << " to " << StatusToString(status));
        return;
    }
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: Status changed from "
                                               << StatusToString(mServiceStatus.load()) << " to "
                                               << StatusToString(status));
    mServiceStatus.store(status);
}

void HealthChecker::EnqueueErrorMessage(const std::string &errCode, const std::string &createdBy,
                                        const std::chrono::time_point<std::chrono::system_clock> &timestamp)
{
    ErrorQueue::GetInstance().EnqueueErrorMessage(errCode, createdBy, timestamp);
    ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER, "HealthChecker: New error added. Error code: "
                                               << errCode << ", createdBy: " << createdBy);

    UpdateStatus(SERVICE_ABNORMAL);
}

void HealthChecker::UpdateNpuDeviceIds(const std::set<int> &npuDeviceIds)
{
    {
        std::unique_lock<std::shared_mutex> lock(mNpuDevicesMutex);
        mNpuDeviceCardIds.clear();
        for (const auto &id : npuDeviceIds) {
            mNpuDeviceCardIds.insert(id / mChipPerCard);
        }
    }
    PrintNpuDeviceIds();
}

SimulateResult HealthChecker::RunHttpTimedHealthCheck(uint32_t waitTime)
{
    auto &serverConfig = GetServerConfig();
    InferReqType reqType = (serverConfig.inferMode == INFER_MODE_DMI)
        ? InferReqType::REQ_PREFILL : InferReqType::REQ_STAND_INFER;

    ULOG_DEBUG(SUBMODLE_NAME_HEALTHCHECKER,
        "HealthChecker: RunHttpTimedHealthCheck. reqType=" << static_cast<int>(reqType)
        << ", waitTime=" << waitTime << "s");

    auto executor = SimulateRequestExecutor::Create(reqType);
    SimulateResult result = executor->RunSimulateOnce(waitTime);
    // BUSY 状态在 HTTP 接口中视为健康
    if (result.status == SimulateResult::Status::BUSY) {
        ULOG_INFO(SUBMODLE_NAME_HEALTHCHECKER,
            "HealthChecker: RunHttpTimedHealthCheck busy but healthy");
    }

    return result;
}

SendingDecodeMessageScope::SendingDecodeMessageScope(HealthChecker &checker) noexcept
    : checker_(checker)
{
    checker_.SetSendingDecodeMessageStatus(true);
}

SendingDecodeMessageScope::~SendingDecodeMessageScope() noexcept
{
    checker_.SetSendingDecodeMessageStatus(false);
}

} // namespace mindie_llm