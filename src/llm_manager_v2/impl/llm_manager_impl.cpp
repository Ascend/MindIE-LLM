/**
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

#include "llm_manager_impl.h"

#include <pybind11/pybind11.h>

#include <chrono>
#include <iomanip>

#include "check_utils.h"
#include "common_util.h"
#include "config_manager.h"
#include "config_manager_impl.h"
#include "file_utils.h"
#include "infer_instances.h"
#include "llm_manager_v2.h"
#include "log.h"
#include "memory_utils.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "nlohmann/json.hpp"
#include "param_checker.h"
#include "request_response/callback.h"
#include "safe_io.h"
#include "shared_memory.h"
#include "string_utils.h"

using Json = nlohmann::json;
namespace py = pybind11;
using namespace mindie_llm;
using namespace model_execute_data;
namespace mindie_llm {
const std::string DEFAULT_HOST_IP = "127.0.0.1";
constexpr uint32_t PROCESS_GROUP_MASTER_PORT = 7777;
ConcurrentMap<mindie_llm::RequestIdNew, SendResponsesCallbackV2> mindie_llm::InferInstance::callbackMap{};

void HandleResponse(ResponseSPtr response) {
    auto spanHandleResponse = PROF(INFO, Domain("Engine").SpanStart("HandleResponse"));
    if (response == nullptr) {
        MINDIE_LLM_LOG_ERROR("[LlmManagerImpl] Response is null!");
        PROF(spanHandleResponse.SpanEnd());
        return;
    }

    PROF(INFO, Domain("Engine").Resource(response->reqId).Attr("endFlag", response->isEos).Event("sendResponse"));

    std::optional<SendResponsesCallbackV2> serverResponseCallback =
        InferInstance::GetCallbackMap().Get(response->reqId);

    // EOS or PUBLISH_KV_COMPLETE时，删除callback
    if (response->isEos || response->transferStatusFlag == TransferStatusType::PUBLISH_KV_COMPLETE) {
        InferInstance::GetCallbackMap().Erase(response->reqId);
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmManagerImpl] Remove SendResponsesCallback requestId: " + response->reqId +
                                    " when encountering EOS or PUBLISH_KV_COMPLETE.");
    }

    if (serverResponseCallback.has_value()) {
        serverResponseCallback.value()(response);
    } else if (!InferInstance::IsPaused()) {
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmManagerImpl] SendResponsesCallback of requestId: " + response->reqId +
                                    " is not exist.");
    }

    PROF(spanHandleResponse.SpanEnd());
}

struct PDLinkRequestData {
    model_execute_data::PDRole role = model_execute_data::PDRole::UNKNOWN_ROLE;
    bool needSwitch = false;
    int64_t linkNum = 0;
    int64_t unlinkNum = 0;
    int64_t hostIpNum = 0;
    int64_t superPodIdNum = 0;
    int64_t containsDpInstanceIds = 0;
    int64_t hostIpNumPerDp = 1;  // won't be serialized

    // {dpInstanceId: [host_ip1, host_ip2, ...]}, TBC 处理多个 host_ip
    std::unordered_map<InstanceId, std::vector<std::string>> dpInstance2HostIps;
    // {dpInstanceId: [(device_ip1, device_physical_id1), ...]}
    std::unordered_map<InstanceId, std::vector<std::pair<std::string, int64_t>>> dpInstance2LinkDevices;
    // {dpInstanceId: [(device_ip1, device_physical_id1), ...]}
    std::unordered_map<InstanceId, std::vector<std::pair<std::string, int64_t>>> dpInstance2UnlinkDevices;
    // {dpInstanceId: sp_size}
    std::unordered_map<InstanceId, int64_t> dpInstance2SPSize;
    // {dpInstanceId: cp_size}
    std::unordered_map<InstanceId, int64_t> dpInstance2CPSize;
    // {dpInstanceId: superPodId}
    std::unordered_map<InstanceId, int64_t> dpInstance2SuperPodId;
    // {dpInstanceId: [super_device_id1, super_device_id2, ...]}
    std::unordered_map<InstanceId, std::vector<int64_t>> dpInstance2LinkSuperDeviceIds;
    // {dpInstanceId: [super_device_id1, super_device_id2, ...]}
    std::unordered_map<InstanceId, std::vector<int64_t>> dpInstance2UnLinkSuperDeviceIds;
};

PDLinkRequestData GetPDLinkRequestDataFromInferRequest(RequestSPtr inferRequest) {
    PDLinkRequestData pdLinkRequestData;
    pdLinkRequestData.role = static_cast<model_execute_data::PDRole>(inferRequest->role);
    pdLinkRequestData.needSwitch = inferRequest->needSwitch;
    pdLinkRequestData.linkNum = inferRequest->linkNum;
    pdLinkRequestData.unlinkNum = inferRequest->unlinkNum;
    pdLinkRequestData.hostIpNum = inferRequest->hostIpNum;
    pdLinkRequestData.superPodIdNum = inferRequest->superPodIdNum;
    pdLinkRequestData.containsDpInstanceIds = inferRequest->containsDpInstanceIds;
    if (pdLinkRequestData.containsDpInstanceIds == 1) {
        pdLinkRequestData.hostIpNumPerDp =
            (pdLinkRequestData.linkNum != 0 ? pdLinkRequestData.hostIpNum / pdLinkRequestData.linkNum : 1);
    }
    pdLinkRequestData.dpInstance2HostIps = inferRequest->dpInstance2HostIps;
    pdLinkRequestData.dpInstance2SuperPodId = inferRequest->dpInstance2SuperPodId;
    pdLinkRequestData.dpInstance2LinkDevices = inferRequest->dpInstance2LinkDevices;
    pdLinkRequestData.dpInstance2UnlinkDevices = inferRequest->dpInstance2UnlinkDevices;
    pdLinkRequestData.dpInstance2LinkSuperDeviceIds = inferRequest->dpInstance2LinkSuperDeviceIds;
    pdLinkRequestData.dpInstance2UnLinkSuperDeviceIds = inferRequest->dpInstance2UnLinkSuperDeviceIds;

    for (const auto &[instanceId, spSize] : inferRequest->spInfo) {
        pdLinkRequestData.dpInstance2SPSize[instanceId] = spSize;
    }
    for (const auto &[instanceId, cpSize] : inferRequest->cpInfo) {
        pdLinkRequestData.dpInstance2CPSize[instanceId] = cpSize;
    }
    return pdLinkRequestData;
}

void AddDevices(const PDLinkRequestData &requestData,
                const std::unordered_map<InstanceId, std::vector<std::pair<std::string, int64_t>>> &dpInstance2Devices,
                const std::unordered_map<InstanceId, std::vector<int64_t>> &dpInstance2SuperDeviceIds,
                PDLinkRequest_PDLinkInfo *singlePDLinkInfo, bool isLinkInfo) {
    for (const auto &[dpInstanceId, devices] : dpInstance2Devices) {
        RemoteInfo *remoteInfo = nullptr;
        if (isLinkInfo) {
            remoteInfo = singlePDLinkInfo->add_link_info();
        } else {
            remoteInfo = singlePDLinkInfo->add_unlink_info();
        }

        // 设置Host信息, 考虑多个hostip的情况
        for (int64_t i = 0; i < requestData.hostIpNumPerDp; ++i) {
            auto *hostInfo = remoteInfo->add_host_info();
            auto hostIt = requestData.dpInstance2HostIps.find(dpInstanceId);
            if (hostIt != requestData.dpInstance2HostIps.end()) {
                hostInfo->set_host_ip(hostIt->second.at(i));
            } else {
                hostInfo->set_host_ip(DEFAULT_HOST_IP);  // 默认IP
            }
            hostInfo->set_cluster_id(std::to_string(dpInstanceId));
            auto superPodIdIt = requestData.dpInstance2SuperPodId.find(dpInstanceId);
            if (superPodIdIt != requestData.dpInstance2SuperPodId.end()) {
                hostInfo->set_super_pod_id(superPodIdIt->second);
            }
        }

        // 设置Device信息
        std::vector<int64_t> superDeviceIds;
        auto superDeviceIdIt = dpInstance2SuperDeviceIds.find(dpInstanceId);
        if (superDeviceIdIt != dpInstance2SuperDeviceIds.end()) {
            superDeviceIds = superDeviceIdIt->second;
        }
        for (size_t i = 0; i < devices.size(); ++i) {
            auto *deviceInfo = remoteInfo->add_device_info();
            deviceInfo->set_device_ip(devices.at(i).first);
            deviceInfo->set_physical_id(devices.at(i).second);
            if (i < superDeviceIds.size()) {
                deviceInfo->set_super_device_id(superDeviceIds.at(i));
            }
        }
    }
}

PDLinkRequest BuildPDLinkRequest(const PDLinkRequestData &pdLinkRequestData) {
    PDLinkRequest pdLinkRequest;
    auto *singlePDLinkInfo = pdLinkRequest.add_pd_link_info();

    singlePDLinkInfo->set_pd_role(pdLinkRequestData.role);
    singlePDLinkInfo->set_change_role(pdLinkRequestData.needSwitch);
    singlePDLinkInfo->set_link_num(pdLinkRequestData.linkNum);
    singlePDLinkInfo->set_unlink_num(pdLinkRequestData.unlinkNum);
    singlePDLinkInfo->set_host_ip_num(pdLinkRequestData.hostIpNum);
    singlePDLinkInfo->set_super_id_num(pdLinkRequestData.superPodIdNum);
    singlePDLinkInfo->set_contains_dp_instance_ids(pdLinkRequestData.containsDpInstanceIds);

    AddDevices(pdLinkRequestData, pdLinkRequestData.dpInstance2LinkDevices,
               pdLinkRequestData.dpInstance2LinkSuperDeviceIds, singlePDLinkInfo, true);
    AddDevices(pdLinkRequestData, pdLinkRequestData.dpInstance2UnlinkDevices,
               pdLinkRequestData.dpInstance2UnLinkSuperDeviceIds, singlePDLinkInfo, false);

    auto *instance2SP = singlePDLinkInfo->mutable_instance2sp();
    for (const auto &pair : pdLinkRequestData.dpInstance2SPSize) {
        instance2SP->insert({pair.first, pair.second});
    }

    auto *instance2CP = singlePDLinkInfo->mutable_instance2cp();
    for (const auto &pair : pdLinkRequestData.dpInstance2CPSize) {
        instance2CP->insert({pair.first, pair.second});
    }

    return pdLinkRequest;
}

bool SafeStoull(const std::string &str, uint64_t &outValue) {
    try {
        outValue = std::stoull(str);
        return true;
    } catch (const std::invalid_argument &e) {
        MINDIE_LLM_LOG_ERROR("Invalid number string: " << str << ", " << e.what());
        outValue = 0;
        return false;
    } catch (const std::out_of_range &e) {
        MINDIE_LLM_LOG_ERROR("Number out of range: " << str << ", " << e.what());
        outValue = UINT64_MAX;
        return false;
    }
}
}  // namespace mindie_llm

namespace mindie_llm {
uint32_t g_vocabSizeConfig = 0;
int32_t g_maxTopKConfig = 0;
int32_t g_truncation = 0;
uint32_t g_maxPositionEmbeddings = 1;
uint32_t g_maxSeqLen = 1;
uint32_t g_maxInputTokenLen = 1;
std::map<std::string, std::string> g_modelParams;
size_t g_truncLen = 1;
constexpr int PROCESS_STEP_SLEEP = 2;
constexpr int CONTROL_STEP_SLEEP = 1;
constexpr int RUNTIME_STEP_SLEEP = 50;
constexpr int RESPONSE_FLAG3 = 3;
const std::string ENV_NPU_DEVICE_IDS = "NPU_DEVICE_IDS";
static std::once_flag pool_init_flag;
LlmManagerImpl::LlmManagerImpl(const std::string &llmConfigPath, GetRequestsCallbackV2 getRequests,
                               SendResponsesCallbackV2 handleResponse, ControlSignalCallbackV2 controlCallback,
                               LlmManagerStatsCallback statusCallback,
                               SendStatusResponseCallbackV2 statusResponseCallback,
                               std::map<std::string, std::string> ipInfo) {
    getRequests_ = getRequests;
    handleResponse_ = handleResponse ? handleResponse : HandleResponse;
    controlCallback_ = controlCallback;
    statusCallback_ = statusCallback;
    statusResponseCallback_ = statusResponseCallback;
    ipInfo_ = ipInfo;
    // llmConfigPath comes from ENV or CMD args
    llmConfigPath_ = llmConfigPath;

    if (!mindie_llm::Log::GetInstance(LoggerType::MINDIE_LLM)) {
        std::call_once(pool_init_flag, []() { spdlog::init_thread_pool(LOGGER_QUEUE_SIZE, LOGGER_THREAD_NUM); });
        mindie_llm::Log::CreateInstance(LoggerType::MINDIE_LLM);
    }

    // for engine mode, we need to create configmanager and log instance
    if (!mindie_llm::ConfigManager::CreateInstance(llmConfigPath_)) {
        MINDIE_LLM_LOG_ERROR("Failed to create ConfigManager in LlmManagerImpl");
    }

    if (ipInfo.count("infer_mode") > 0 && ipInfo["infer_mode"] == "dmi") {
        isDmiInfer_ = true;
    }
    MINDIE_LLM_LOG_INFO("LLMRuntime init success!");
}

void LlmManagerImpl::Step() {
    shutdown_ = false;
    if (getRequests_ != nullptr) {
        processThread_ = std::thread(&LlmManagerImpl::ProcessStep, this);
        pthread_setname_np(processThread_.native_handle(), "ManagerProcess");
    }

    if (controlCallback_ != nullptr) {
        controlThread_ = std::thread(&LlmManagerImpl::ControlStep, this);
        pthread_setname_np(controlThread_.native_handle(), "ManagerControl");
    }

    if (statusCallback_ != nullptr) {
        sendRuntimeThread_ = std::thread(&LlmManagerImpl::SendRuntimeStep, this);
        pthread_setname_np(sendRuntimeThread_.native_handle(), "ManagerSendRT");
    }
    MINDIE_LLM_LOG_INFO("LLMRuntime thread start success!");
}

void LlmManagerImpl::Stop() {
    shutdown_ = true;
    bool have_gil = false;
    if (Py_IsInitialized() == 1) {
#ifdef PyGILState_Check
        have_gil = (PyGILState_Check() != 0);
#endif
    }
    if (have_gil) {
        py::gil_scoped_release release;
        if (processThread_.joinable()) {
            processThread_.join();
        }
        if (controlThread_.joinable()) {
            controlThread_.join();
        }
        if (sendRuntimeThread_.joinable()) {
            sendRuntimeThread_.join();
        }
        py::gil_scoped_acquire acquire;
    } else {
        if (processThread_.joinable()) {
            processThread_.join();
        }
        if (controlThread_.joinable()) {
            controlThread_.join();
        }
        if (sendRuntimeThread_.joinable()) {
            sendRuntimeThread_.join();
        }
    }
    MINDIE_LLM_LOG_INFO("LLMRuntime thread stop success!");
}

void LlmManagerImpl::Shutdown() {
    auto ret = Finalize();
    if (!ret.IsOk()) {
        MINDIE_LLM_LOG_ERROR("Shut down LLMRuntime failed!. errmsg:" << ret.StatusMsg());
    }
}

void LlmManagerImpl::ProcessStep() {
    while (!shutdown_) {
        ProcessRequests();
        std::this_thread::sleep_for(std::chrono::microseconds(PROCESS_STEP_SLEEP));
    }
}

void LlmManagerImpl::ControlStep() {
    while (!shutdown_) {
        ControlRequest();
        std::this_thread::sleep_for(std::chrono::milliseconds(CONTROL_STEP_SLEEP));
    }
}

void LlmManagerImpl::SendRuntimeStep() {
    while (!shutdown_) {
        SendRuntimeStatus();
        std::this_thread::sleep_for(std::chrono::milliseconds(RUNTIME_STEP_SLEEP));
    }
}

static void InitbackendConfig(EngineConfig &engineConfig, const BackendConfig &backendConfig) {
    engineConfig.backendName = backendConfig.backendName;
    engineConfig.tokenizerProcessNumber = backendConfig.tokenizerProcessNumber;
    engineConfig.deployType = backendConfig.deployType;
    engineConfig.executorType = backendConfig.executorType;
    engineConfig.backendBinPath = backendConfig.backendBinPath;
    engineConfig.multiNodesInferEnabled = backendConfig.multiNodesInferEnabled;
    engineConfig.interNodeTLSEnabled = backendConfig.interNodeTLSEnabled;
    engineConfig.multiNodesInferPort = backendConfig.multiNodesInferPort;
    engineConfig.interNodeTlsCaPath = backendConfig.interNodeTlsCaPath;
    engineConfig.interNodeTlsCaFiles = backendConfig.interNodeTlsCaFiles;
    engineConfig.interNodeTlsCert = backendConfig.interNodeTlsCert;
    engineConfig.interNodeTlsPk = backendConfig.interNodeTlsPk;
    engineConfig.interNodeTlsCrlPath = backendConfig.interNodeTlsCrlPath;
    engineConfig.interNodeTlsCrlFiles = backendConfig.interNodeTlsCrlFiles;
    engineConfig.kvPoolConfig = backendConfig.kvPoolConfig;
    engineConfig.lwdMultiNodesEnable = backendConfig.lwdMultiNodesEnable;
}

static bool CheckJsonDepthCallback(int depth, Json::parse_event_t ev, [[maybe_unused]] Json &obj) {
    return CheckJsonDepthWithLogger(depth, ev, [depth]() {
        MINDIE_LLM_LOG_INFO("Failed to parse json: depth is " << depth << ", limit is " << GetJsonDepthLimit());
    });
}

static void UpdateFromEnv(std::set<size_t> &npuDeviceIds, uint32_t modelInstanceId) {
    auto envPtr = std::getenv(ENV_NPU_DEVICE_IDS.c_str());
    if (envPtr == nullptr) {
        return;
    }
    std::string envNpuIds(envPtr);
    RemoveSpaces(envNpuIds);
    Json jsonData;
    try {
        jsonData["npuDeviceIds"] = Json::parse(envNpuIds, CheckJsonDepthCallback);
        MINDIE_LLM_LOG_INFO("Config data has been updated by env variable:" << ENV_NPU_DEVICE_IDS);

        npuDeviceIds.clear();
        for (auto &ele : jsonData["npuDeviceIds"][modelInstanceId]) {
            npuDeviceIds.insert(static_cast<size_t>(ele));
        }
    } catch (Json::parse_error &e) {
        MINDIE_LLM_LOG_ERROR("Env variable resolution failed: " << ENV_NPU_DEVICE_IDS
                                                                << ", Incorrect format: " << e.what());
    } catch (Json::out_of_range &e) {
        MINDIE_LLM_LOG_ERROR("modelInstanceId=" << modelInstanceId << " out of range: " << e.what());
    } catch (Json::type_error &e) {
        MINDIE_LLM_LOG_ERROR("Type error for modelInstanceId=" << modelInstanceId << ": " << e.what());
    }
    MINDIE_LLM_LOG_INFO("ModelDeployConfig::npuDeviceIds=" << jsonData["npuDeviceIds"]);
}

static void SetModelParams(ModelDeployConfig &modelDeployParam) {
    g_truncation = modelDeployParam.truncation;
    g_maxSeqLen = modelDeployParam.maxSeqLen;
    g_maxInputTokenLen = modelDeployParam.maxInputTokenLen;
    g_modelParams["maxSeqLen"] = std::to_string(modelDeployParam.maxSeqLen);
    g_modelParams["modelName"] = modelDeployParam.modelName;
    g_modelParams["maxInputTokenLen"] = std::to_string(modelDeployParam.maxInputTokenLen);
    g_modelParams["inputDatatype"] = std::to_string(modelDeployParam.inputDatatype);
    g_modelParams["outputDatatype"] = std::to_string(modelDeployParam.outputDatatype);
    g_modelParams["npuMemSize"] = std::to_string(modelDeployParam.npuMemSize);
    g_modelParams["worldSize"] = std::to_string(modelDeployParam.worldSize);
    g_modelParams["modelWeightPath"] = modelDeployParam.modelWeightPath;
    g_modelParams["modelInstanceType"] = modelDeployParam.modelInstanceType;
    if (g_truncation != 0) {
        g_truncLen = std::min(modelDeployParam.maxInputTokenLen, modelDeployParam.maxSeqLen - 1);
    }
    MINDIE_LLM_LOG_INFO("InitModelConfig: maxSeqLen="
                        << modelDeployParam.maxSeqLen << ", maxInputTokenLen=" << modelDeployParam.maxInputTokenLen
                        << ", g_truncation=" << g_truncation << ", g_truncLen=" << g_truncLen);
}

static void UpdateNpuDeviceId(std::set<size_t> &modelNpuDeviceIds, std::set<size_t> &npuDeviceIds,
                              ModelDeployConfig &modelParam, uint32_t modelInstanceId) {
    for (auto &npuID : npuDeviceIds) {
        modelNpuDeviceIds.insert(npuID);
    }
    if (modelParam.npuDeviceIds.size() != 0) {
        modelNpuDeviceIds.clear();
        for (auto &npuID : modelParam.npuDeviceIds) {
            modelNpuDeviceIds.insert(npuID);
        }
    }
    UpdateFromEnv(modelNpuDeviceIds, modelInstanceId);
}

static bool InitModelConfig(EngineConfig &engineConfig, std::vector<ModelDeployConfig> &modelDeployParam,
                            std::set<size_t> &npuDeviceIds, uint32_t modelInstanceId) {
    if (modelDeployParam.empty()) {
        return false;
    }
    SetModelParams(modelDeployParam.at(0));

    for (auto &singleModelParam : modelDeployParam) {
        ModelParam modelParam;
        modelParam.modelName = singleModelParam.modelName;
        modelParam.modelWeightPath = singleModelParam.modelWeightPath;
        UpdateNpuDeviceId(modelParam.npuDeviceIds, npuDeviceIds, singleModelParam, modelInstanceId);
        modelParam.worldSize = modelParam.npuDeviceIds.size();
        modelParam.cpuMemSize = singleModelParam.cpuMemSize;
        modelParam.trustRemoteCode = singleModelParam.trustRemoteCode;
        modelParam.npuMemSize = singleModelParam.npuMemSize;
        modelParam.modelInstanceType = singleModelParam.modelInstanceType;
        modelParam.backendType = singleModelParam.backendType;
        modelParam.maxSeqLen = singleModelParam.maxSeqLen;
        modelParam.maxInputTokenLen = singleModelParam.maxInputTokenLen;
        modelParam.maxPositionEmbeddings = singleModelParam.maxPositionEmbeddings;
        modelParam.vocabSize = singleModelParam.vocabSize;
        modelParam.maxTopK = singleModelParam.maxTopK;
        modelParam.inputDatatype = singleModelParam.inputDatatype;
        modelParam.outputDatatype = singleModelParam.outputDatatype;
        modelParam.speculationGamma = singleModelParam.speculationGamma;
        modelParam.modelConfig = singleModelParam.modelConfig;
        modelParam.loraModules = singleModelParam.loraModules;
        modelParam.useLora = singleModelParam.useLora;
        modelParam.maxLoras = singleModelParam.maxLoras;
        modelParam.maxLoraRank = singleModelParam.maxLoraRank;
        engineConfig.modelDeployParam.push_back(modelParam);
    }
    return true;
}

static bool GetPluginEnable(std::string pluginName, std::vector<ModelDeployConfig> &modelDeployParam) {
    for (auto &modelParam : modelDeployParam) {
        auto it = modelParam.modelConfig.find("plugin_params");
        std::string pluginParam;
        if (it != modelParam.modelConfig.end()) {
            pluginParam = it->second;
        } else {
            MINDIE_LLM_LOG_INFO("Input plugin_params is empty or only contains whitespace.");
            return false;
        }
        if (!CheckStringInputLength(pluginParam, MAX_STRING_LENGTH)) {
            MINDIE_LLM_LOG_ERROR("The 'pluginParam' is too long.");
            return false;
        }
        nlohmann::json jstring;
        try {
            jstring = nlohmann::json::parse(pluginParam, CheckJsonDepthCallback);
        } catch (const nlohmann::json::parse_error &e) {
            std::stringstream errMsg;
            errMsg << "Invalid plugin parameters. "
                   << "Please check the 'plugin_params' field in the "
                      "'config.json' file for the service, "
                   << "and ensure it adheres to the JSON format. The error "
                      "info is: "
                   << e.what();
            MINDIE_LLM_LOG_ERROR(errMsg.str());
            throw std::invalid_argument(errMsg.str());
        }
        if (jstring.contains("plugin_type")) {
            std::string pluginType = jstring["plugin_type"];

            // 分割字符串为多个词
            std::istringstream iss(pluginType);
            std::vector<std::string> words;
            std::string word;
            while (getline(iss, word, ',')) {
                // 去除可能存在的前导和尾随空格
                word.erase(0, word.find_first_not_of(" "));
                word.erase(word.find_last_not_of(" ") + 1);
                words.push_back(word);
            }

            // 检查列表中是否可以全词匹配上
            for (const auto &w : words) {
                if (w == pluginName) {
                    return true;
                }
            }
        } else {
            MINDIE_LLM_LOG_ERROR(
                "'plugin_type' field not found in plugin_params, please "
                "check!");
            return false;
        }
    }
    return false;
}

static bool CheckEngineConfig(EngineConfig &engineConfig) {
    bool checkRes = true;
    if (engineConfig.enableSplit && engineConfig.templateType != "Mix") {
        MINDIE_LLM_LOG_ERROR("templateType must be Mix when enableSplit is True, but is " << engineConfig.templateType);
        checkRes = false;
    }
    if (engineConfig.enableSplit && engineConfig.splitChunkTokens == 0) {
        MINDIE_LLM_LOG_ERROR(
            "splitChunkTokens should be larger than 0 when splitfuse is "
            "enabled, but is "
            << engineConfig.splitChunkTokens);
        checkRes = false;
    }
    if (engineConfig.enableChunkedPrefill && engineConfig.supportSelectBatch) {
        MINDIE_LLM_LOG_WARN("Both splitfuse and supportSelectBatch are configured, the "
                            << "scheduling strategy will be executed according to splitfuse, "
                               "supportSelectBatch will be disabled.");
    }
    if (ConfigManager::GetInstance().IslayerwiseDisaggregated() && engineConfig.enablePrefixCache) {
        MINDIE_LLM_LOG_ERROR("Prefix cache isn't supported in layerwise-disaggregated mode.");
        checkRes = false;
    }
    return checkRes;
}

static void UpdateEngineConfig(EngineConfig &engineConfig) {
    auto &deployParam = engineConfig.modelDeployParam[0];

    uint32_t cpSize = 1;
    const auto it = deployParam.modelConfig.find("cp");
    if (it != deployParam.modelConfig.end()) {
        // cpSize has been checked in model_deploy_config.cpp. It can be safely
        // assigned here.
        cpSize = static_cast<uint32_t>(std::stoi(it->second));
    }

    const bool isCpEnabled = (cpSize > 1);
    const uint32_t loadBalanceCpSize = cpSize * 2;

    uint32_t maxSeqLen = deployParam.maxSeqLen;
    if (isCpEnabled && (maxSeqLen % loadBalanceCpSize != 0)) {
        const uint32_t base = maxSeqLen / loadBalanceCpSize;
        maxSeqLen = (base + 1) * loadBalanceCpSize;
        MINDIE_LLM_LOG_INFO("CP enabled: maxSeqLen increased to multiple of 2 * cpsize. " << "New value: "
                                                                                          << maxSeqLen);
    }

    const uint32_t maxLen = std::min(maxSeqLen, deployParam.maxInputTokenLen + engineConfig.maxIterTimes);
    uint32_t maxPrefillTokens = engineConfig.maxPrefillTokens;

    if ((maxPrefillTokens < maxLen) && !engineConfig.enableSplit) {
        maxPrefillTokens = maxLen;
        MINDIE_LLM_LOG_INFO(
            "maxPrefillTokens is smaller than maxLen. It will be replaced by "
            "maxLen. "
            << "maxLen = min(maxSeqLen, maxInputTokenLen + maxIterTimes)");
    }

    if (isCpEnabled && (maxPrefillTokens % loadBalanceCpSize != 0)) {
        const uint32_t base = maxPrefillTokens / loadBalanceCpSize;
        maxPrefillTokens = (base + 1) * loadBalanceCpSize;
        MINDIE_LLM_LOG_INFO("CP enabled: maxPrefillTokens increased to multiple of 2 * cpsize. " << "New value: "
                                                                                                 << maxPrefillTokens);
    }

    if (engineConfig.maxBatchSize > maxPrefillTokens) {
        engineConfig.maxBatchSize = maxPrefillTokens;
        MINDIE_LLM_LOG_INFO(
            "maxBatchSize is greater than maxPrefillTokens. It will be "
            "replaced by maxPrefillTokens.");
    }

    engineConfig.maxPrefillTokens = maxPrefillTokens;
    deployParam.maxSeqLen = maxSeqLen;
}

static void InitScheduleConfig(EngineConfig &engineConfig, const ScheduleConfig &scheduleConfig, bool enableSplit,
                               bool enablePrefixCache) {
    // schedule config
    engineConfig.templateType = scheduleConfig.templateType;
    engineConfig.templateName = scheduleConfig.templateName;
    // prefill
    engineConfig.maxPrefillBatchSize = scheduleConfig.maxPrefillBatchSize;
    engineConfig.maxPrefillTokens = scheduleConfig.maxPrefillTokens;
    engineConfig.prefillTimeMsPerReq = scheduleConfig.prefillTimeMsPerReq;
    engineConfig.prefillPolicyType = scheduleConfig.prefillPolicyType;
    engineConfig.minPrefillBatchSize = scheduleConfig.minPrefillBatchSize;
    engineConfig.maxFirstTokenWaitTime = scheduleConfig.maxFirstTokenWaitTime;
    g_modelParams["prefillPolicyType"] = std::to_string(scheduleConfig.prefillPolicyType);
    g_modelParams["maxPrefillBatchSize"] = std::to_string(scheduleConfig.maxPrefillBatchSize);
    g_modelParams["maxPrefillTokens"] = std::to_string(scheduleConfig.maxPrefillTokens);
    // kvcache
    engineConfig.cacheBlockSize = scheduleConfig.cacheBlockSize;
    engineConfig.cpuBlockNum = scheduleConfig.cpuBlockNum;
    engineConfig.npuBlockNum = scheduleConfig.npuBlockNum;
    // decode
    engineConfig.decodeTimeMsPerReq = scheduleConfig.decodeTimeMsPerReq;
    engineConfig.decodePolicyType = scheduleConfig.decodePolicyType;
    g_modelParams["decodePolicyType"] = std::to_string(scheduleConfig.decodePolicyType);
    // batch common
    engineConfig.maxBatchSize = scheduleConfig.maxBatchSize;
    engineConfig.maxPreemptCount = scheduleConfig.maxPreemptCount;
    engineConfig.supportSelectBatch = scheduleConfig.supportSelectBatch;
    engineConfig.maxQueueDelayMicroseconds = scheduleConfig.maxQueueDelayMicroseconds;
    engineConfig.maxN = scheduleConfig.maxN;
    // policy config
    engineConfig.policyType = scheduleConfig.policyType;
    engineConfig.maxIterTimes = scheduleConfig.maxIterTimes;
    engineConfig.dpScheduling = scheduleConfig.dpScheduling;
    engineConfig.activateAsyncInference = scheduleConfig.activateAsyncInference;
    engineConfig.distributedEnable = scheduleConfig.distributedEnable;

    // slo
    engineConfig.stageSelectPolicy = scheduleConfig.stageSelectPolicy;
    engineConfig.dynamicBatchSizeEnable = scheduleConfig.dynamicBatchSizeEnable;

    // mix config
    engineConfig.enableSplit = enableSplit;
    if (engineConfig.enableSplit) {
        engineConfig.splitType = scheduleConfig.splitType;
        engineConfig.splitStartType = scheduleConfig.splitStartType;
        engineConfig.splitChunkTokens = scheduleConfig.splitChunkTokens;
        engineConfig.splitStartBatchSize = scheduleConfig.splitStartBatchSize;
    }

    // chunked prefill
    engineConfig.enableChunkedPrefill = enableSplit;
    if (engineConfig.enableChunkedPrefill) {
        engineConfig.prefillChunkSize = scheduleConfig.prefillChunkSize;
        engineConfig.maxNumPartialPrefills = scheduleConfig.maxNumPartialPrefills;
        engineConfig.maxLongPartialPrefills = scheduleConfig.maxLongPartialPrefills;
        engineConfig.longPrefillTokenThreshold = scheduleConfig.longPrefillTokenThreshold;
    }

    // prefix cache
    engineConfig.enablePrefixCache = enablePrefixCache;

    // buffer response
    engineConfig.bufferResponseEnabled = scheduleConfig.bufferResponseEnabled;
    engineConfig.prefillExpectedTime = scheduleConfig.prefillExpectedTime;
    engineConfig.decodeExpectedTime = scheduleConfig.decodeExpectedTime;
}

static bool InitEngineConfig(EngineConfig &engineConfig, std::vector<ModelDeployConfig> &modelDeployParam,
                             std::set<size_t> &npuDeviceIds, uint32_t modelInstanceId,
                             std::map<std::string, std::string> extendInfo) {
    const ScheduleConfig &scheduleConfig = GetScheduleConfig();
    const BackendConfig &backendConfig = GetBackendConfig();
    const RanktableParam &ranktableParam = GetRanktableParam();
    // modelDeployParam
    if (!InitModelConfig(engineConfig, modelDeployParam, npuDeviceIds, modelInstanceId)) {
        return false;
    }

    // 提取需要从modelconfig中传递给SchedulerConfig的参数
    bool enableSplit = GetPluginEnable("splitfuse", modelDeployParam);
    bool enablePrefixCache = GetPluginEnable("prefix_cache", modelDeployParam);

    InitScheduleConfig(engineConfig, scheduleConfig, enableSplit, enablePrefixCache);

    // backendconfig
    InitbackendConfig(engineConfig, backendConfig);

    // 一些在更新完enginconfig之后才可以做的校验
    if (!CheckEngineConfig(engineConfig)) {
        return false;
    }

    if (ConfigManager::GetInstance().IslayerwiseDisaggregated()) {
        auto &serverConfig = GetServerConfig();
        engineConfig.layerwiseDisaggregated = true;
        std::string role = serverConfig.layerwiseDisaggregatedRoleType;
        engineConfig.isMaster = (role == "master");
        engineConfig.masterIP = serverConfig.layerwiseDisaggregatedMasterIpAddress;
        engineConfig.localIP = serverConfig.ipAddress;
        engineConfig.slaveIPs = serverConfig.layerwiseDisaggregatedSlaveIpAddress;
        for (auto &modelParam : engineConfig.modelDeployParam) {
            modelParam.npuDeviceIds = npuDeviceIds;
            modelParam.worldSize = modelParam.npuDeviceIds.size();
        }

        // 云侧真多机的rankTable处理; 边侧就是dp=n的实现, 不用传特殊参数
        if (backendConfig.lwdMultiNodesEnable && !engineConfig.isMaster) {
            engineConfig.isLwdMultiNodesMaster = ranktableParam.isMaster;

            engineConfig.globalWorldSize = ranktableParam.globalWorldSize;
            for (auto device : ranktableParam.local.device) {
                engineConfig.globalRankIds.emplace_back(device.rankId);
            }
            std::vector<size_t> npuIds;
            for (auto device : ranktableParam.local.device) {
                npuIds.push_back(stoul(device.deviceId));
            }
            // 多机场景下对 npuids 进行 替换
            for (auto &modelParam : engineConfig.modelDeployParam) {
                modelParam.npuDeviceIds.clear();
                for (auto id : npuIds) {
                    modelParam.npuDeviceIds.insert(id);
                }
                modelParam.worldSize = modelParam.npuDeviceIds.size();
            }
        }
    }

    UpdateEngineConfig(engineConfig);

    if (ConfigManager::GetInstance().IsMultiNodeInfer()) {
        engineConfig.isMaster = ranktableParam.isMaster;
        engineConfig.globalWorldSize = ranktableParam.globalWorldSize;
        engineConfig.masterIP = ranktableParam.master.containerIp.empty() ? ranktableParam.master.serverId
                                                                          : ranktableParam.master.containerIp;
        engineConfig.localIP =
            ranktableParam.local.containerIp.empty() ? ranktableParam.local.serverId : ranktableParam.local.containerIp;
        for (auto &slave : ranktableParam.slaves) {
            if (slave.containerIp.empty()) {
                engineConfig.slaveIPs.emplace_back(slave.serverId);
            } else {
                engineConfig.slaveIPs.emplace_back(slave.containerIp);
            }
        }
        for (auto device : ranktableParam.local.device) {
            engineConfig.globalRankIds.emplace_back(device.rankId);
        }
        std::vector<size_t> npuIds;
        for (auto device : ranktableParam.local.device) {
            npuIds.push_back(stoul(device.deviceId));
        }
        // 多机场景下对 npuids 进行 替换
        for (auto &modelParam : engineConfig.modelDeployParam) {
            modelParam.npuDeviceIds.clear();
            for (auto id : npuIds) {
                modelParam.npuDeviceIds.insert(id);
            }
            modelParam.worldSize = modelParam.npuDeviceIds.size();
        }
    }

    // P节点分布式需要做进程集合通信（为了构造陪跑和padding），需要各个节点的IP信息
    if (engineConfig.distributedEnable) {
        engineConfig.masterIP = ranktableParam.master.containerIp.empty() ? ranktableParam.master.serverId
                                                                          : ranktableParam.master.containerIp;
        for (auto &slave : ranktableParam.slaves) {
            if (slave.containerIp.empty()) {
                engineConfig.slaveIPs.emplace_back(slave.serverId);
            } else {
                engineConfig.slaveIPs.emplace_back(slave.containerIp);
            }
        }
    }

    // 刷新modelConfigs_ 刷新engineConfig_
    std::string rankIdList = extendInfo.count("local_rank_ids") > 0 ? extendInfo["local_rank_ids"] : "";
    MINDIE_LLM_LOG_INFO("The rankIdList is " << rankIdList << "In InitEngineConfig paras");
    if (!rankIdList.empty() && engineConfig.distributedEnable) {
        // localRankIds service传递的局部rankids
        std::vector<std::string> localRankIds;
        mindie_llm::Split(rankIdList, ',', localRankIds);
        engineConfig.globalWorldSize = ranktableParam.globalWorldSize;
        MINDIE_LLM_LOG_INFO("The globalWorldSize is " << engineConfig.globalWorldSize << "In InitEngineConfig paras");
        engineConfig.globalRankIds.clear();
        std::vector<size_t> npuIds;
        for (auto device : ranktableParam.local.device) {
            if (std::find(localRankIds.begin(), localRankIds.end(), device.rankId) != localRankIds.end()) {
                engineConfig.globalRankIds.emplace_back(device.rankId);
                npuIds.push_back(stoul(device.deviceId));
            }
        }

        for (auto &modelParam : engineConfig.modelDeployParam) {
            modelParam.npuDeviceIds.clear();
            for (auto id : npuIds) {
                modelParam.npuDeviceIds.insert(id);
            }
            modelParam.worldSize = modelParam.npuDeviceIds.size();
            MINDIE_LLM_LOG_INFO("The world_size is " << modelParam.worldSize << "In InitEngineConfig");
        }
    }

    g_vocabSizeConfig = engineConfig.modelDeployParam[0].vocabSize;
    g_maxTopKConfig = engineConfig.modelDeployParam[0].maxTopK;
    return true;
}

static void LLMSetMultiNodeConfig(std::map<std::string, std::string> &modelConfig, const EngineConfig &engineConfig) {
    modelConfig["multiNodesInferPort"] = std::to_string(engineConfig.multiNodesInferPort);
    modelConfig["interNodeTLSEnabled"] = std::to_string(engineConfig.interNodeTLSEnabled);
    modelConfig["multiNodesInferEnabled"] = std::to_string(engineConfig.multiNodesInferEnabled);
    modelConfig["interNodeTlsCert"] = engineConfig.interNodeTlsCert;
    modelConfig["interNodeTlsCrlPath"] = engineConfig.interNodeTlsCrlPath;
    modelConfig["interNodeTlsCrlFiles"] = engineConfig.interNodeTlsCrlFiles;
    modelConfig["interNodeTlsPk"] = engineConfig.interNodeTlsPk;
    modelConfig["interNodeTlsCaPath"] = engineConfig.interNodeTlsCaPath;
    modelConfig["interNodeTlsCaFiles"] = engineConfig.interNodeTlsCaFiles;

    modelConfig["globalWorldSize"] = std::to_string(engineConfig.globalWorldSize);
}

static void LLMSetModelParam(std::map<std::string, std::string> &modelConfig, const ModelParam &modelParam) {
    modelConfig["speculation_gamma"] = std::to_string(modelParam.speculationGamma);

    for (auto it = modelParam.modelConfig.begin(); it != modelParam.modelConfig.end(); ++it) {
        modelConfig.insert(make_pair(it->first, it->second));
    }
    modelConfig["backend_type"] = modelParam.backendType;
    modelConfig["world_size"] = std::to_string(modelParam.worldSize);
    modelConfig["max_seq_len"] = std::to_string(modelParam.maxSeqLen);
    modelConfig["model_name"] = modelParam.modelName;
    modelConfig["model_id"] = modelParam.modelWeightPath;
    modelConfig["cpu_mem"] = std::to_string(modelParam.cpuMemSize);
    modelConfig["trust_remote_code"] = std::to_string(modelParam.trustRemoteCode);
    modelConfig["npu_mem"] = std::to_string(modelParam.npuMemSize);
    modelConfig["model_instance_number"] = std::to_string(1);
    modelConfig["model_instance_type"] = modelParam.modelInstanceType;
    modelConfig["max_input_len"] = std::to_string(modelParam.maxInputTokenLen);
    modelConfig["max_loras"] = std::to_string(modelParam.maxLoras);
    modelConfig["max_lora_rank"] = std::to_string(modelParam.maxLoraRank);
}

static std::string IsAsyncBatchscheduler(const EngineConfig &engineConfig) {
    const char *mindieAsyncSchedulingEnable = std::getenv("MINDIE_ASYNC_SCHEDULING_ENABLE");
    if (mindieAsyncSchedulingEnable == nullptr) {
        return "false";
    }
    if (std::string(mindieAsyncSchedulingEnable) != "1") {
        return "false";
    }
    if (engineConfig.enableSplit) {
        return "false";
    }

    return "true";
}

static void LLMSetLayerwiseDisaggregatedModelConfig(std::map<std::string, std::string> &modelConfig,
                                                    const EngineConfig &engineConfig) {
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    auto &serverConfig = configManager.GetServerConfig();
    auto &backendConfig = configManager.GetBackendConfig();
    auto &scheduleConfig = configManager.GetScheduleConfig();

    modelConfig["layerwiseDisaggregated"] = "true";
    modelConfig["layerwiseDisaggregatedRoleType"] = serverConfig.layerwiseDisaggregatedRoleType;
    modelConfig["layerwiseDisaggregatedMasterIpAddress"] = serverConfig.layerwiseDisaggregatedMasterIpAddress;
    modelConfig["layerwiseDisaggregatedDataPort"] = std::to_string(serverConfig.layerwiseDisaggregatedDataPort);
    std::string mergeSlaveIpAddress = "";
    for (auto &slaveIp : serverConfig.layerwiseDisaggregatedSlaveIpAddress) {
        mergeSlaveIpAddress += static_cast<std::string>(slaveIp) + ",";
    }
    if (!mergeSlaveIpAddress.empty()) {
        mergeSlaveIpAddress.erase(mergeSlaveIpAddress.length() - 1);
    }
    modelConfig["layerwiseDisaggregatedSlaveIpAddress"] = mergeSlaveIpAddress;

    std::string mergeCrtlPort = "";
    for (auto &crtlPort : serverConfig.layerwiseDisaggregatedCrtlPort) {
        mergeCrtlPort += std::to_string(crtlPort) + ",";
    }
    if (!mergeCrtlPort.empty()) {
        mergeCrtlPort.erase(mergeCrtlPort.length() - 1);
    }
    modelConfig["layerwiseDisaggregatedCrtlPort"] = mergeCrtlPort;

    modelConfig["interNodeTLSEnabled"] = std::to_string(backendConfig.interNodeTLSEnabled);
    modelConfig["interNodeTlsCaPath"] = backendConfig.interNodeTlsCaPath;
    modelConfig["interNodeTlsCaFiles"] = backendConfig.interNodeTlsCaFiles;
    modelConfig["interNodeTlsCert"] = backendConfig.interNodeTlsCert;
    modelConfig["interNodeTlsPk"] = backendConfig.interNodeTlsPk;
    modelConfig["interNodeTlsCrlPath"] = backendConfig.interNodeTlsCrlPath;
    modelConfig["interNodeTlsCrlFiles"] = backendConfig.interNodeTlsCrlFiles;

    if (backendConfig.lwdMultiNodesEnable) {
        modelConfig["layerwiseDisaggregatedMultiNodesInferEnabled"] = "true";
        modelConfig["layerwiseDisaggregatedMultiNodesCtrlPort"] = std::to_string(backendConfig.lwdMultiNodesCtrlPort);
        modelConfig["lwd_multi_nodes_enable"] = "true";
    }

    modelConfig["lwdNextPHeadPrior"] = scheduleConfig.lwdNextPHeadPrior ? "true" : "false";
    if (modelConfig["lwd_multi_nodes_enable"] == "true") {
        modelConfig["lwd_multi_nodes_is_master"] = engineConfig.isLwdMultiNodesMaster ? "true" : "false";
        modelConfig["layerwiseDisaggregatedMultiNodesMaster"] = engineConfig.isLwdMultiNodesMaster ? "true" : "false";
        if (configManager.GetLwdRoleType() != "master") {
            modelConfig["localIP"] =
                engineConfig.isLwdMultiNodesMaster ? engineConfig.slaveIPs[0] : engineConfig.slaveIPs[1];
        }
    }
}

static void LLMSetModelConfig(std::map<std::string, std::string> &modelConfig, const std::string &homePath,
                              const EngineConfig &engineConfig, const ModelParam &modelParam, bool isDmiInfer = false) {
    LLMSetModelParam(modelConfig, modelParam);
    modelConfig["max_prefill_tokens"] = std::to_string(engineConfig.maxPrefillTokens);
    modelConfig["max_prefill_batch_size"] = std::to_string(engineConfig.maxPrefillBatchSize);
    modelConfig["deploy_type"] = engineConfig.deployType;
    modelConfig["executor_type"] = engineConfig.executorType;
    modelConfig["max_iter_times"] = std::to_string(engineConfig.maxIterTimes);
    modelConfig["backend_bin_path"] = homePath + engineConfig.backendBinPath;
    modelConfig["model_instance_number"] = std::to_string(1);
    modelConfig["block_size"] = std::to_string(engineConfig.cacheBlockSize);
    modelConfig["is_dmi_infer"] = isDmiInfer ? "1" : "0";
    modelConfig["async_inference"] = engineConfig.activateAsyncInference ? "true" : "false";
    modelConfig["distributed_enable"] = engineConfig.distributedEnable ? "true" : "false";
    modelConfig["max_batch_size"] = std::to_string(engineConfig.maxBatchSize);
    modelConfig["max_prefill_batch_size"] = std::to_string(engineConfig.maxPrefillBatchSize);
    modelConfig["max_n"] = std::to_string(engineConfig.maxN);
    modelConfig["kv_pool_backend"] = engineConfig.kvPoolConfig.backend;
    modelConfig["kv_pool_config_path"] = engineConfig.kvPoolConfig.configPath;
    modelConfig["kv_pool_async_write"] = engineConfig.kvPoolConfig.asyncWrite ? "true" : "false";

    std::string npuIds;
    if (!modelParam.npuDeviceIds.empty()) {
        for (auto &item : modelParam.npuDeviceIds) {
            npuIds += std::to_string(item) + ",";
        }
        npuIds.pop_back();
    }
    modelConfig["npu_device_ids"] = npuIds;

    LLMSetMultiNodeConfig(modelConfig, engineConfig);
    std::string slaveIPs;
    if (!engineConfig.slaveIPs.empty()) {
        for (auto &ip : engineConfig.slaveIPs) {
            slaveIPs += ip + ",";
        }
        slaveIPs.pop_back();
    }
    modelConfig["slaveIPs"] = slaveIPs;
    modelConfig["masterIP"] = engineConfig.masterIP;
    modelConfig["localIP"] = engineConfig.localIP;
    modelConfig["isMaster"] = std::to_string(engineConfig.isMaster);
    std::string globalRankIds;
    if (!engineConfig.globalRankIds.empty()) {
        for (auto &id : engineConfig.globalRankIds) {
            globalRankIds += id + ",";
        }
        globalRankIds.pop_back();
    }
    modelConfig["globalRankIds"] = globalRankIds;
    g_modelParams["maxIterTimes"] = std::to_string(engineConfig.maxIterTimes);
    modelConfig["asyncBatchscheduler"] = IsAsyncBatchscheduler(engineConfig);
    modelConfig["threadNum"] = (modelConfig["asyncBatchscheduler"] == "true") ? "2" : "1";

    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    if (configManager.IslayerwiseDisaggregated()) LLMSetLayerwiseDisaggregatedModelConfig(modelConfig, engineConfig);
}

static void InitPolicyConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    // policy config
    schedulerConfig.prefillPolicyType = engineConfig.prefillPolicyType;
    schedulerConfig.decodePolicyType = engineConfig.decodePolicyType;
    schedulerConfig.policyType = engineConfig.policyType;

    // batch config
    schedulerConfig.maxPreemptCount = engineConfig.maxPreemptCount;
    schedulerConfig.supportSelectBatch = engineConfig.supportSelectBatch;
    schedulerConfig.prefillTimeMsPerReq = engineConfig.prefillTimeMsPerReq;
    schedulerConfig.decodeTimeMsPerReq = engineConfig.decodeTimeMsPerReq;
    schedulerConfig.maxPrefillBatchSize = engineConfig.maxPrefillBatchSize;
    schedulerConfig.maxPrefillTokens = engineConfig.maxPrefillTokens;
    schedulerConfig.minPrefillBatchSize = engineConfig.minPrefillBatchSize;
    schedulerConfig.maxFirstTokenWaitTime = engineConfig.maxFirstTokenWaitTime;
    schedulerConfig.maxBatchSize = engineConfig.maxBatchSize;
    schedulerConfig.maxQueueDelayMicroseconds = engineConfig.maxQueueDelayMicroseconds;

    // slo
    schedulerConfig.stageSelectPolicy = engineConfig.stageSelectPolicy;
    schedulerConfig.dynamicBatchSizeEnable = engineConfig.dynamicBatchSizeEnable;
}

static void InitDeviceAndInstanceConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig,
                                        const std::map<std::string, std::string> &ipInfo) {
    for (auto &paramObj : engineConfig.modelDeployParam) {
        schedulerConfig.npuDeviceIds.push_back(paramObj.npuDeviceIds);
    }

    auto ipInfoIt = ipInfo.find("local_instance_id");
    if (ipInfoIt != ipInfo.end()) {
        if (!StrToUint64(schedulerConfig.instanceId, ipInfoIt->second)) {
            MINDIE_LLM_LOG_INFO("Get instanceId failed: out of range.");
            return;
        }
        MINDIE_LLM_LOG_INFO("schedulerConfig.instanceId" << schedulerConfig.instanceId);
    }
}

static void InitBlockConfig(SchedulerConfig &schedulerConfig, BlockNum blockNum, const EngineConfig &engineConfig) {
    schedulerConfig.maxSeqLen = engineConfig.modelDeployParam[0].maxSeqLen;
    schedulerConfig.maxIterTimes = engineConfig.maxIterTimes;
    schedulerConfig.cpuBlockNum = blockNum.cpuBlockNum;
    schedulerConfig.npuBlockNum = blockNum.npuBlockNum;
    schedulerConfig.speculationGamma = engineConfig.modelDeployParam[0].speculationGamma;
    schedulerConfig.cacheBlockSize = engineConfig.cacheBlockSize;
    schedulerConfig.maxLoras = engineConfig.modelDeployParam[0].maxLoras;
    schedulerConfig.maxLoraRank = engineConfig.modelDeployParam[0].maxLoraRank;
}

static void InitParallelConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    const auto &modelConfig = engineConfig.modelDeployParam[0].modelConfig;
    auto setParallelValue = [&](const std::string &parallelInfo, uint32_t &schedulerConfigValue) {
        auto it = modelConfig.find(parallelInfo);
        if (it != modelConfig.end()) {
            // parallelValue has been checked in model_deploy_config.cpp.
            // It can be safely assigned here.
            schedulerConfigValue = static_cast<uint32_t>(std::stoi(it->second));
        }
    };
    setParallelValue("dp", schedulerConfig.dpSize);
    setParallelValue("sp", schedulerConfig.spSize);
    setParallelValue("cp", schedulerConfig.cpSize);
}

static void InitDistributedConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    schedulerConfig.globalWorldSize = engineConfig.globalWorldSize;
    schedulerConfig.globalRankIds = engineConfig.globalRankIds;
    schedulerConfig.worldSize = engineConfig.modelDeployParam[0].worldSize;
    schedulerConfig.activateAsyncInference = engineConfig.activateAsyncInference;
    schedulerConfig.distributedEnable = engineConfig.distributedEnable;
}

static void InitWorkflowConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    schedulerConfig.modelName = engineConfig.modelDeployParam[0].modelName;
    schedulerConfig.templateType = engineConfig.templateType;
    schedulerConfig.templateName = engineConfig.templateName;
    schedulerConfig.pipelineNumber = 1;
    schedulerConfig.maxInputTokenLen = engineConfig.modelDeployParam[0].maxInputTokenLen;
}

static void InitSplitFuseConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    schedulerConfig.enableSplit = engineConfig.enableSplit;
    if (schedulerConfig.enableSplit) {
        schedulerConfig.splitType = engineConfig.splitType;
        schedulerConfig.splitStartType = engineConfig.splitStartType;
        schedulerConfig.splitChunkTokens = engineConfig.splitChunkTokens;
        schedulerConfig.splitStartBatchSize = engineConfig.splitStartBatchSize;
    }

    schedulerConfig.enableChunkedPrefill = engineConfig.enableSplit;
    if (schedulerConfig.enableChunkedPrefill) {
        schedulerConfig.prefillChunkSize = engineConfig.prefillChunkSize;
        schedulerConfig.maxNumPartialPrefills = engineConfig.maxNumPartialPrefills;
        schedulerConfig.maxLongPartialPrefills = engineConfig.maxLongPartialPrefills;
        schedulerConfig.longPrefillTokenThreshold = engineConfig.longPrefillTokenThreshold;
    }
}

static void InitPrefixCacheConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    schedulerConfig.enablePrefixCache = engineConfig.enablePrefixCache;
}

static void InitKvPoolConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    if (engineConfig.kvPoolConfig.backend != "" && engineConfig.kvPoolConfig.configPath != "") {
        schedulerConfig.enableKvPool = true;
    } else {
        schedulerConfig.enableKvPool = false;
    }
    schedulerConfig.kvPoolConfig = engineConfig.kvPoolConfig;
}

static void InitBufferResponseConfig(SchedulerConfig &schedulerConfig, const EngineConfig &engineConfig) {
    schedulerConfig.bufferResponseEnabled = engineConfig.bufferResponseEnabled;
    schedulerConfig.prefillExpectedTime = engineConfig.prefillExpectedTime;
    schedulerConfig.decodeExpectedTime = engineConfig.decodeExpectedTime;
    schedulerConfig.isMultiNodeInfer = engineConfig.multiNodesInferEnabled;
}

static void InitLayerwiseDisaggregated(SchedulerConfig &schedulerConfig) {
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    auto &serverConfig = configManager.GetServerConfig();
    auto &scheduleConfig = configManager.GetScheduleConfig();
    if (serverConfig.layerwiseDisaggregated) {
        schedulerConfig.stageSelectPolicy = 3;                                 // 边云策略为3
        schedulerConfig.batchPnum = scheduleConfig.lwdNextPHeadPrior ? 2 : 1;  // 2是允许下发P batch最大数量
        // 下发batchsize为P batch数目（batchPnum）加D batch数目（1）
        schedulerConfig.maxDispatchBatchNum = schedulerConfig.batchPnum + 1;
        schedulerConfig.layerwiseDisaggregated = true;
    }
}

static void InitThinkingConfig(SchedulerConfig &schedulerConfig, ThinkingConfig &thinkingConfig) {
    schedulerConfig.earlyStoppingIds = thinkingConfig.earlyStoppingIds;
    schedulerConfig.startThinkingId = thinkingConfig.startThinkingId;
    schedulerConfig.stopThinkingId = thinkingConfig.stopThinkingId;
}

static void LLMInitSchedulerConfig(SchedulerConfig &schedulerConfig, BlockNum blockNum,
                                   const EngineConfig &engineConfig, const std::map<std::string, std::string> &ipInfo,
                                   ThinkingConfig &thinkingConfig) {
    InitDeviceAndInstanceConfig(schedulerConfig, engineConfig, ipInfo);
    InitPolicyConfig(schedulerConfig, engineConfig);
    InitParallelConfig(schedulerConfig, engineConfig);
    InitBlockConfig(schedulerConfig, blockNum, engineConfig);
    InitDistributedConfig(schedulerConfig, engineConfig);
    InitWorkflowConfig(schedulerConfig, engineConfig);
    InitSplitFuseConfig(schedulerConfig, engineConfig);
    InitPrefixCacheConfig(schedulerConfig, engineConfig);
    InitBufferResponseConfig(schedulerConfig, engineConfig);
    InitLayerwiseDisaggregated(schedulerConfig);
    InitKvPoolConfig(schedulerConfig, engineConfig);
    InitThinkingConfig(schedulerConfig, thinkingConfig);
    // Fill multi-kv-cache descriptors from executor (populated via
    // RemoteModelInitResults.kv_cache_descs). Backward compatible: if empty,
    // scheduler will create a single block manager using cacheBlockSize.
    {
        std::lock_guard<std::mutex> lock(IExecutor::kvCacheOverview_.updateValueMutex);
        schedulerConfig.kvCacheDescs.clear();
        for (const auto &d : IExecutor::kvCacheOverview_.kvCacheDescs) {
            SchedulerConfig::KVCacheDesc sd;
            sd.npuBlockNum = d.npuBlockNum;
            sd.blockSize = d.blockSize;
            sd.compressionRatio = d.compressionRatio;
            sd.cacheType = d.cacheType;
            schedulerConfig.kvCacheDescs.push_back(sd);
        }
    }
}

static void LlmSetLoraConfig(const std::map<std::string, std::string> &loraModules,
                             std::map<std::string, std::string> &modelConfig) {
    Json loraJson(loraModules);
    std::string loraString = loraJson.dump();
    modelConfig["lora_modules"] = loraString;
}

static bool LlmSetModelConfig(const EngineConfig &engineConfig,
                              std::vector<std::map<std::string, std::string>> &modelConfigs,
                              const std::map<std::string, std::string> &ipInfo = std::map<std::string, std::string>(),
                              bool isDmiInfer = false) {
    std::string homePath;
    for (auto &modelParam : engineConfig.modelDeployParam) {
        std::map<std::string, std::string> modelConfig{ipInfo};
        LLMSetModelConfig(modelConfig, homePath, engineConfig, modelParam, isDmiInfer);
        if (modelParam.useLora) {
            LlmSetLoraConfig(modelParam.loraModules, modelConfig);
        }
        modelConfigs.push_back(modelConfig);
    }
    if (modelConfigs.empty()) {
        return false;
    }
    return true;
}

Role LlmManagerImpl::GetRoleFromString(std::string &pdRole) const {
    if ("decoder" == pdRole) {
        return Role::D;
    }

    if ("prefill" == pdRole) {
        return Role::P;
    }

    if ("flex" == pdRole) {
        return Role::FlexP;
    }

    return Role::PnD;
}

void LlmManagerImpl::InitEngineDPProcessGroup(SchedulerConfig &schedulerConfig) {
    std::vector<NodeInfo> nodeInfos;
    if (engineConfig_.distributedEnable && schedulerConfig.dpSize > 1) {
        // DataParallel部署形态，需要初始化进程组用于DP间通信
        nodeInfos.push_back({engineConfig_.masterIP, engineConfig_.masterIP});
        for (const auto &slaveIp : engineConfig_.slaveIPs) {
            nodeInfos.push_back({slaveIp, slaveIp});
        }
        llmEnginePtr_->InitProcessGroup(nodeInfos, engineConfig_.masterIP, PROCESS_GROUP_MASTER_PORT);
    }
}

BlockNum LlmManagerImpl::GetMinBlockNumFromExecutors() {
    // CpuBlockNum和NpuBlockNum是Executor类的静态成员变量，所有Executor实例共享这两个值，其值为所有NPU中最小的blockNum
    // 因此只需获取第一个Executor实例的数值即可
    uint32_t minCpuBlockNum = iExecutorSPtrs_.front()->GetCpuBlockNum();
    uint32_t minNpuBlockNum = iExecutorSPtrs_.front()->GetNpuBlockNum();
    MINDIE_LLM_LOG_INFO("CpuBlockNum:" << minCpuBlockNum << "; NpuBlockNum: " << minNpuBlockNum);
    BlockNum blockNum{.cpuBlockNum = minCpuBlockNum, .npuBlockNum = minNpuBlockNum};

    return blockNum;
}

ThinkingConfig LlmManagerImpl::GetThinkingConfigFromExecutors() {
    ThinkingConfig thinkingConfig;
    if (iExecutorSPtrs_.size() != 0 && iExecutorSPtrs_.front() != nullptr) {
        thinkingConfig = iExecutorSPtrs_.front()->GetThinkingConfig();
    }
    return thinkingConfig;
}

Status LlmManagerImpl::LaunchLlmEngine(Role pdRole) {
    if (iExecutorSPtrs_.size() == 0) {
        MINDIE_LLM_LOG_ERROR("LlmManagerImpl::LaunchLlmEngine:iExecutorSPtrs_ is empty");
        return Status(Error::Code::ERROR, "Executors is empty.");
    }

    if ((engineConfig_.multiNodesInferEnabled || engineConfig_.layerwiseDisaggregated) && !engineConfig_.isMaster) {
        MINDIE_LLM_LOG_INFO(
            "In centralized inter-node PD co-locating, the slave node does not "
            "hold its own LlmEngine, "
            "it shares the same LlmEngine with the master node.");
        return Status(Error::Code::OK, "Success");
    }

    BlockNum blockNum = GetMinBlockNumFromExecutors();
    ThinkingConfig thinkingConfig = GetThinkingConfigFromExecutors();
    SchedulerConfig schedulerConfig;
    LLMInitSchedulerConfig(schedulerConfig, blockNum, engineConfig_, ipInfo_, thinkingConfig);
    if (schedulerConfig.layerwiseDisaggregated && schedulerConfig.cpSize * schedulerConfig.spSize > 1) {
        schedulerConfig.lwdCloudNpuBlockNum = iExecutorSPtrs_.front()->GetLwdCloudNpuBlockNum();
    }
    schedulerConfig.activateAsyncInference = modelConfigs_[0]["asyncBatchscheduler"] == "true";
    // 当前pdRole_ 与给定pdRole角色不同
    if (pdRole_ != pdRole) {
        schedulerConfig.templateType = (pdRole == Role::D ? "DmiDecode" : "DmiPrefill");
    }

    llmEnginePtr_ = MakeLlmEngine(schedulerConfig, iExecutorSPtrs_, handleResponse_, pdRole, &llmEngineReady_);
    std::vector<ModelParam> modelParamVec = engineConfig_.modelDeployParam;
    llmEnginePtr_->InitStaticLoras(modelParamVec, iExecutorSPtrs_.size());  // 初始化lora_manager中静态lora
    InitEngineDPProcessGroup(schedulerConfig);                              // 初始化分布式多DP进程通信资源
    llmEnginePtr_->StartEngineThread();                                     // 启动Engine调度线程

    // 标记Engine已就绪。请求会进入Scheduler队列，调度线程会按正常流程处理
    // 无需等待线程启动，因为队列和调度机制本身是线程安全的
    llmEngineReady_.store(true, std::memory_order_release);
    MINDIE_LLM_LOG_INFO("[LaunchLlmEngine] Engine started and ready to accept requests.");

    // 注意，一定要在BatchSchduler初始化成功后再改变pdRole，以保证Scheduler初始化失败时，角色信息保持不变
    pdRole_ = pdRole;
    if (pdRole_ == Role::FlexP) {
        llmEnginePtr_->SetPrefillPercentage(std::stoi(ipInfo_["local_prefill_percentage"]));
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmManagerImpl::InitModelForMultiPd(const std::map<std::string, std::string> pdInfo,
                                           [[maybe_unused]] uint32_t modelInstanceId) {
    if (iExecutorSPtrs_.size() == 0) {
        return Status(Error::Code::ERROR, "iExecutorSPtrs_ in InitModelForMultiPd is empty");
    }
    if (modelConfigs_[0].size() == 0) {
        return Status(Error::Code::ERROR, "modelConfigs_ Size is Zero");
    }
    // 根据pdInfo写入到modelConfigs_
    modelConfigs_[0].insert(pdInfo.begin(), pdInfo.end());
    // 初始化Model
    std::vector<std::thread> threads;
    threads.reserve(iExecutorSPtrs_.size());
    for (size_t i = 0; i < iExecutorSPtrs_.size(); i++) {
        threads.emplace_back([&, i]() {
            if (!iExecutorSPtrs_[i]->MasterAndSlaveModelInit(pdInfo)) {
                throw std::runtime_error("MasterAndSlaveModelInit failed for executor idx " + std::to_string(i));
            }
        });
    }
    // 等待所有线程完成
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    maxPositionEmbeddings_ = iExecutorSPtrs_.front()->GetMaxPositionEmbeddings();
    g_maxPositionEmbeddings = maxPositionEmbeddings_;
    ipInfo_ = pdInfo;
    std::string curRole = pdInfo.count("role") > 0 ? pdInfo.at("role") : inferModeStandard;
    Role role = GetRoleFromString(curRole);
    Status res = LaunchLlmEngine(role);
    return res;
}

bool LlmManagerImpl::GetMultiNodesInferEnabled() const { return multiNodesInferEnabled_; }

bool LlmManagerImpl::GetDmiInferEnabled() const { return isDmiInfer_; }

Status LlmManagerImpl::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds,
                            [[maybe_unused]] std::map<std::string, std::string> extendInfo) {
    if (handleResponse_ == nullptr) {
        return Status(Error::Code::ERROR, "callback function is nullptr");
    }

    std::vector<ModelDeployConfig> modelParamVec;
    try {
        modelParamVec = mindie_llm::ConfigManager::GetInstance().GetModelDeployConfig();
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_ERROR("Config manager init exception: " << e.what());
        return Status(Error::Code::ERROR, "Get configManagerInstance failed. modelParamVec get failed.");
    }
    if (!InitEngineConfig(engineConfig_, modelParamVec, npuDeviceIds, modelInstanceId, extendInfo)) {
        return Status(Error::Code::ERROR, "llmmanager init InitEngineConfig failed.");
    }
    multiNodesInferEnabled_ = engineConfig_.multiNodesInferEnabled;
    isMaster_ = engineConfig_.isMaster;

    if (!LlmSetModelConfig(engineConfig_, modelConfigs_, ipInfo_, isDmiInfer_)) {
        MINDIE_LLM_LOG_ERROR("Malloc modelBackends_ failed.");
        return Status(Error::Code::ERROR, "Engine init model failed: new modelBackends_ failed");
    }

    // 1、集中式的组网：只有主DP节点接收请求，由主节点做分发
    //  1.1如果是主节点上的进程，则创建DP数量个executor：0到DP/nodeNum管主节点下的卡，剩余的executor通过grpc和从节点通信
    //  1.2如果是从节点，就创建DP/nodeNum个executor，管理自己的卡，接收主节点的grpc请求
    // 2、分布式组网：每个节点都能接收到请求
    //  2.1只创建1个executor
    size_t executorNum = 1;
    // 表示当前机器上需要创建几份共享内存
    size_t shmCount = 1;
    auto it = engineConfig_.modelDeployParam[0].modelConfig.find("dp");

    if (engineConfig_.layerwiseDisaggregated) {  // 边云特性的逻辑在这个分支
        size_t dp = 1;
        if (it != engineConfig_.modelDeployParam[0].modelConfig.end()) {
            dp = std::stoul(it->second);
        }
        auto &configManager = mindie_llm::ConfigManager::GetInstance();
        if (configManager.GetLwdRoleType() == "master") {
            executorNum = dp;  // master起dp个
        } else {
            std::vector<std::string> slaveIPs;
            mindie_llm::Split(modelConfigs_[0].at("slaveIPs"), ",", slaveIPs);
            size_t slaveNum = slaveIPs.size();
            // slave在多dp下起DP/slaveNum个,单dp下起1个;至少起1个
            if (slaveNum == 0) {
                throw std::runtime_error("slaveIPs must be greater than 0");
            } else {
                executorNum = std::max(dp / slaveNum, size_t(1));
            }
        }
    } else if (engineConfig_.distributedEnable && !multiNodesInferEnabled_) {
        executorNum = 1;
    } else if (it != engineConfig_.modelDeployParam[0].modelConfig.end()) {
        // 以下是集中式组网的逻辑
        const size_t dp = std::stoul(it->second);
        if (multiNodesInferEnabled_ && dp > 1) {
            std::vector<std::string> slaveIPs;
            mindie_llm::Split(modelConfigs_[0].at("slaveIPs"), ",", slaveIPs);
            size_t nodeNum = slaveIPs.size() + 1;
            executorNum = isMaster_ ? dp : dp / nodeNum;
            shmCount = dp / nodeNum;
        } else {
            executorNum = dp;
            shmCount = dp;
        }
    }

    // 在多DP场景下，NPU之间需要建立通信，需要保证多个executor同时创建并初始化
    std::vector<std::thread> threads;
    threads.reserve(executorNum);
    iExecutorSPtrs_.resize(executorNum);
    if (!SharedMemorySizeCheck(TOTAL_SHARED_MEMORY_PER_DP * shmCount)) {
        MINDIE_LLM_LOG_ERROR(
            "Available shared memory size is not enough for all executors. "
            "Please increase the "
            "available shared memory. The least required size is " +
            std::to_string(TOTAL_SHARED_MEMORY_PER_DP * shmCount));
        return Status(Error::Code::ERROR, "Shared memory size is not enough for all executors.");
    }
    for (size_t i = 0; i < executorNum; i++) {
        threads.emplace_back([&, i]() {
            IExecutorSPtr iExecutorSPtr = CreateExecutor();
            if (multiNodesInferEnabled_ && isDmiInfer_) {
                // 需要主从通信的PD分离场景下，第一次只需要初始化GRPC通信，不需要初始化模型
                if (!iExecutorSPtr->ExecutorParseConfigAndInitGRPC(modelConfigs_[0], multiNodesInferEnabled_, i)) {
                    throw std::runtime_error("ExecutorParseConfigAndInitGRPC failed for rank " + std::to_string(i));
                }
            } else {
                // 需要主从通信的混步场景需要初始化GRPC通信和模型，其他场景直接初始化模型
                if (!iExecutorSPtr->ExecutorInstanceInit(modelConfigs_[0], multiNodesInferEnabled_, i)) {
                    throw std::runtime_error("ExecutorInstanceInit failed for rank " + std::to_string(i));
                }
            }
            iExecutorSPtrs_[i] = iExecutorSPtr;
        });
    }
    // 等待所有线程完成
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // 集中式PD分离场景，不进行GMIS初始化，等待后续InitModelForMultiPd调用
    if (multiNodesInferEnabled_ && isDmiInfer_) {
        return Status(Error::Code::OK, "Success");
    }

    g_maxPositionEmbeddings = iExecutorSPtrs_.front()->GetMaxPositionEmbeddings();
    maxPositionEmbeddings_ = g_maxPositionEmbeddings;

    std::string roleStr = ipInfo_.count("role") > 0 ? ipInfo_["role"] : "standard";
    Role role = GetRoleFromString(roleStr);
    return LaunchLlmEngine(role);
}

Status LlmManagerImpl::ProcessRequests(RequestSPtr request) {
    MINDIE_LLM_LOG_WARN_REQUEST("Get a new inferRequest from server, requestId: " << request->requestId);
    return ForwardRequest(request);
}

Status LlmManagerImpl::ProcessRequests() {
    if (getRequests_ == nullptr) {
        return Status(Error::Code::ERROR, "getRequests_ is nullptr");
    }

    std::vector<RequestSPtr> requests = getRequests_();
    for (auto req : requests) {
        RequestIdNew reqId = req->requestId;
        if (req == nullptr) {
            MINDIE_LLM_LOG_ERROR("Error: Request is null!");
            continue;
        }
        MINDIE_LLM_LOG_INFO("Get a new inferRequest from server, requestId: " << req->requestId);

        Status ret = ForwardRequest(req);
        if (!ret.IsOk()) {
            MINDIE_LLM_LOG_ERROR("Error: Process is notOK!" << ret.StatusMsg());
        }
        if (statusResponseCallback_ != nullptr) {
            statusResponseCallback_(req->requestId, ret, StatusResponseTypeV2::REQUEST_ENQUEUE_STATUS);
        }
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmManagerImpl::ForwardRequest(RequestSPtr request) {
    Status ret = ProcessReqInputIds(request);
    if (!ret.IsOk()) {
        return ret;
    }

    if (!llmEnginePtr_->AddRequest(request)) {
        return Status(Error::Code::ERROR, "Engine has been stopped. Cannot add request.");
    }

    MINDIE_LLM_LOG_INFO_REQUEST("Insert a new inferRequest, requestId: " << request->requestId);
    return Status(Error::Code::OK, "Success");
}

Status VerifyInputTokenSize(int64_t inputTokenSize, uint32_t maxInputTokenSize) {
    if (inputTokenSize > g_maxPositionEmbeddings && g_maxPositionEmbeddings > 0) {
        std::string errorMsg =
            "This model's maximum input ids length cannot be greater than "
            "maxPositionEmbeddings " +
            std::to_string(g_maxPositionEmbeddings) + "," + "the input ids length is " + std::to_string(inputTokenSize);
        MINDIE_LLM_LOG_ERROR(errorMsg);
        return Status(Error::Code::INVALID_ARG, errorMsg);
    }

    if (inputTokenSize > maxInputTokenSize) {
        std::string errorMsg = "This model's maximum input ids length cannot be greater than " +
                               std::to_string(maxInputTokenSize) + "," + "the input ids length is " +
                               std::to_string(inputTokenSize);
        MINDIE_LLM_LOG_ERROR(errorMsg);
        return Status(Error::Code::INVALID_ARG, errorMsg);
    }

    if (inputTokenSize > g_maxSeqLen) {
        std::string errorMsg =
            "This model's maximum input ids length cannot be greater than "
            "maxSeqLen " +
            std::to_string(g_maxSeqLen) + "," + "the input ids length is " + std::to_string(inputTokenSize);
        MINDIE_LLM_LOG_ERROR(errorMsg);
        return Status(Error::Code::INVALID_ARG, errorMsg);
    }
    return Status(Error::Code::OK, "Success");
}

Status VerifyTopK(RequestSPtr &request) {
    int32_t topK = request->topK.value();
    if (g_vocabSizeConfig > static_cast<uint32_t>(INT32_MAX)) {
        std::string errorMsg =
            "The value of g_vocabSizeConfig exceeds the maximum limit "
            "INT32_MAX.";
        MINDIE_LLM_LOG_ERROR(errorMsg);
        return Status(Error::Code::INVALID_ARG, errorMsg);
    }
    int32_t signedVocabSizeConfig = static_cast<int32_t>(g_vocabSizeConfig);
    if (topK < 0 || topK > std::numeric_limits<int32_t>::max()) {
        std::string errorMsg = "The value of topK must be in [0, 2147483647], but the topK is " + std::to_string(topK) +
                               ", please set topK in [0, 2147483647]";
        MINDIE_LLM_LOG_ERROR(errorMsg);
        return Status(Error::Code::INVALID_ARG, errorMsg);
    }
    if (topK > signedVocabSizeConfig || topK > g_maxTopKConfig) {
        request->topK = std::min(signedVocabSizeConfig, g_maxTopKConfig);
        MINDIE_LLM_LOG_INFO_REQUEST("Request topK value has been set to "
                                    << request->topK.value()
                                    << ". Config the `top_k` value in the `generation_config.json` "
                                       "file of the model.");
    }
    return Status(Error::Code::OK, "Success");
}

static Status CheckReqInputIds(RequestSPtr &request, const uint32_t vocabSize) {
    if (vocabSize == 0) {  // 有些配置下（如多模态），vocabSize可能为0，表示不检查input_ids
        return Status(Error::Code::OK, "Success");
    }
    MINDIE_LLM_LOG_DEBUG_REQUEST("Checking input ids from request in CheckReqInputIds function.");
    for (auto id : request->input_ids) {
        if (id >= vocabSize) {
            MINDIE_LLM_LOG_ERROR("Unexpect Input Id: " << id << ", vocab size: " << vocabSize);
            return Status(Error::Code::INVALID_ARG, "Invalid Input Ids");
        }
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmManagerImpl::ProcessReqInputIds(RequestSPtr &request) const {
    if (!request) {
        MINDIE_LLM_LOG_ERROR("CheckReqInputIds: request is nullptr.");
        return Status(Error::Code::ERROR, "CheckReqInputIds: request is nullptr.");
    }
    Status ret = CheckReqInputIds(request, g_vocabSizeConfig);
    if (!ret.IsOk()) {
        return ret;
    }

    if (g_truncation != 0 && request->input_ids.size() > g_truncLen) {
        request->input_ids.resize(g_truncLen);
    }

    int64_t inputTokenSize = request->input_token_num;
    uint32_t maxInputTokenSize;
    if (request->isRecompute) {
        maxInputTokenSize = g_maxSeqLen - 1;
    } else {
        maxInputTokenSize = g_maxInputTokenLen < g_maxSeqLen ? g_maxInputTokenLen : g_maxSeqLen - 1;
    }

    if (g_truncation == 0) {
        ret = VerifyInputTokenSize(inputTokenSize, maxInputTokenSize);
        if (!ret.IsOk()) {
            return ret;
        }
    }
    if (request->topK.has_value()) {
        ret = VerifyTopK(request);
        if (!ret.IsOk()) {
            return ret;
        }
    }
    return Status(Error::Code::OK, "Success");
}

void LlmManagerImpl::ControlRequest(const RequestIdNew &requestId, OperationV2 operation) {
    RequestId reqId = requestId;
    std::unordered_set<RequestId> reqIds = {reqId};
    MINDIE_LLM_LOG_INFO_REQUEST("Get a new ControlRequest from server, requestId: " << reqId << ", with operation:"
                                                                                    << static_cast<int>(operation));
    if (operation == OperationV2::STOP) {
        llmEnginePtr_->AbortRequests(reqIds);
    } else if (operation == OperationV2::RELEASE_KV) {
        llmEnginePtr_->ReleaseKvCache(reqIds);
    } else {
        throw std::runtime_error("Unknown operation");
    }
}

void LlmManagerImpl::ControlRequest() {
    auto stopReqPairs = controlCallback_();
    for (auto reqPair : stopReqPairs) {
        RequestId reqId = reqPair.first;
        MINDIE_LLM_LOG_INFO("Get a new ControlRequest from server, requestId: " << reqId << ", with operation:"
                                                                                << static_cast<int>(reqPair.second));
        std::unordered_set<RequestId> reqIds = {reqId};
        if (reqPair.second == OperationV2::STOP) {
            llmEnginePtr_->AbortRequests(reqIds);
        } else if (reqPair.second == OperationV2::RELEASE_KV) {
            llmEnginePtr_->ReleaseKvCache(reqIds);
        } else {
            throw std::runtime_error("Unknown operation");
        }

        Status status(Error::Code::OK, "ControlRequest success");
        if (statusResponseCallback_ != nullptr) {
            statusResponseCallback_(RequestIdNew(reqPair.first), status, StatusResponseTypeV2::CONTROL_SIGNAL_STATUS);
        }
    }
}

void LlmManagerImpl::SendRuntimeStatus() {
    if ((engineConfig_.multiNodesInferEnabled || engineConfig_.layerwiseDisaggregated) && !engineConfig_.isMaster) {
        return;
    }

    // 收集所有 DP Rank 的聚合指标
    EngineMetric engineMetric = llmEnginePtr_->CollectAllDpEngineMetric();
    SendJsonData(engineMetric);
}

void LlmManagerImpl::SendJsonData(EngineMetric &engineMetric) {
    enum class HealthStatus { READY, ABNORMAL };
    std::map<std::string, HealthStatus> healthStatus{};
    Json jsonData = {{"slaves_status", healthStatus},
                     {"remain_blocks", engineMetric.schedulerInfo.blockInfo.freeNpuBlockNum_},
                     {"remain_prefill_slots", remainPrefillSlots_},
                     {"dp_remain_blocks", dpRemainBlocks_},
                     {"remain_prefill_tokens", remainPrefill_},
                     {"processing_request_num", engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_ +
                                                    engineMetric.schedulerInfo.reqsInfo.runningRequestNum_ +
                                                    engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_},
                     {"waiting_request_num", engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_},
                     {"running_request_num", engineMetric.schedulerInfo.reqsInfo.runningRequestNum_},
                     {"swapped_request_num", engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_},
                     {"free_npu_block_num", engineMetric.schedulerInfo.blockInfo.freeNpuBlockNum_},
                     {"free_cpu_block_num", engineMetric.schedulerInfo.blockInfo.freeCpuBlockNum_},
                     {"total_npu_block_num", engineMetric.schedulerInfo.blockInfo.totalNpuBlockNum_},
                     {"total_cpu_block_num", engineMetric.schedulerInfo.blockInfo.totalCpuBlockNum_},
                     {"all_radix_match_num", engineMetric.schedulerInfo.reqsInfo.allRadixMatchNum_},
                     {"npu_radix_match_hit_num", engineMetric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_},
                     {"cumulative_preempt_count", engineMetric.schedulerInfo.reqsInfo.cumulativePreemptCount_},
                     {"prefill_throughput", engineMetric.prefillThroughput_},
                     {"decode_throughput", engineMetric.decodeThroughput_}};
    std::string strData = jsonData.dump();
    if (statusCallback_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("The statusCallback_ function is nullptr");
    } else {
        statusCallback_(strData);
    }
}

Status LlmManagerImpl::Finalize() {
    // 0. finalize threads
    Stop();

    // 1. finalize engine
    if (multiNodesInferEnabled_ && !isMaster_) {
        MINDIE_LLM_LOG_INFO("Multi Nodes inference slave instance need not finalize.");
    } else {
        llmEnginePtr_->Stop();
    }

    for (auto executor : iExecutorSPtrs_) {
        if (!executor->ExecutorInstanceFinalize()) {
            return Status(Error::Code::ERROR, "Finalize executor failed.");
        }
    }

    return Status(Error::Code::OK, "Success.");
}

Status LlmManagerImpl::FinalizeLlmEngine() const {
    if (multiNodesInferEnabled_ && !isMaster_) {
        MINDIE_LLM_LOG_INFO("Multi Nodes inference slave instance need not finalize.");
    } else {
        llmEnginePtr_->Stop();
    }

    return Status(Error::Code::OK, "imis finalize success.");
}

uint32_t LlmManagerImpl::GetMaxPositionEmbeddings() const { return maxPositionEmbeddings_; }

std::map<std::string, std::string> LlmManagerImpl::GetModelParams() { return g_modelParams; }

Status LlmManagerImpl::RelaunchLlmEngine(int64_t roleIndex) {
    constexpr int MIN_ROLE_INDEX = 1;
    constexpr int MAX_ROLE_INDEX = 3;
    if (roleIndex < MIN_ROLE_INDEX || roleIndex > MAX_ROLE_INDEX) {
        MINDIE_LLM_LOG_ERROR("[RelaunchLlmEngine] Switch PD role error: P/D role is not set.");
        return Status(Error::Code::ERROR, "Switch P/D role error: P/D role is not set.");
    }

    std::unordered_map<std::int64_t, Role> indexRoleMap{{1, Role::P}, {2, Role::D}, {3, Role::FlexP}};

    Role pdRole = indexRoleMap[roleIndex];

    auto res = FinalizeLlmEngine();
    if (!res.IsOk()) {
        MINDIE_LLM_LOG_ERROR("[RelaunchLlmEngine] Failed to finalize LlmEngine.");
        return res;
    }
    res = LaunchLlmEngine(pdRole);
    if (!res.IsOk()) {
        MINDIE_LLM_LOG_ERROR("[RelaunchLlmEngine] Failed to relaunch LlmEngine.");
        return res;
    }
    return Status(Error::Code::OK, "Switch P/D role successfully!");
}

bool LlmManagerImpl::SwitchPdRole(RequestSPtr &runtimeRequest) {
    int64_t roleInt = static_cast<int64_t>(runtimeRequest->role);
    bool needSwitch = runtimeRequest->needSwitch;

    // 非0，为切换场景
    if (needSwitch) {
        Stop();
        Status res = RelaunchLlmEngine(roleInt);
        if (!res.IsOk()) {
            return false;
        }
        Step();
    }
    return true;
}

bool LlmManagerImpl::SetExecuteConfig(bool isForceRelease, std::map<std::string, std::string> &executeConfig,
                                      RequestSPtr &runtimeRequest) {
    if (!isForceRelease) {
        if (!SwitchPdRole(runtimeRequest)) {
            return false;
        }
        executeConfig.insert(std::make_pair("EXECUTE_TYPE", "4"));
    } else {
        executeConfig.insert(std::make_pair("EXECUTE_TYPE", "5"));
    }
    return true;
}

bool LlmManagerImpl::UpdateEngineInfo(RequestSPtr &runtimeRequest, bool isForceRelease) {
    if (pdRole_ == Role::FlexP && isFlexInitialized_) {
        MINDIE_LLM_LOG_INFO(
            "[LlmManager::LlmManagerImpl::UpdateEngineInfo] Only set flex "
            "prefill percentage.");
        return true;
    }

    std::map<std::string, std::string> executeConfig;
    // 身份切换，需要重启LlmEngine
    if (!SetExecuteConfig(isForceRelease, executeConfig, runtimeRequest)) {
        return false;
    }
    MINDIE_LLM_LOG_INFO("[LlmManagerImpl::UpdateEngineInfo] EXECUTE_TYPE is " << executeConfig["EXECUTE_TYPE"]);

    PDLinkRequestData pdLinkRequestData = GetPDLinkRequestDataFromInferRequest(runtimeRequest);
    PDLinkRequest pdLinkRequest = BuildPDLinkRequest(pdLinkRequestData);
    // set pdlink
    std::vector<std::thread> threads;
    threads.reserve(iExecutorSPtrs_.size());
    for (size_t i = 0; i < iExecutorSPtrs_.size(); i++) {
        threads.emplace_back([&, i]() {
            if (!iExecutorSPtrs_[i]->SetupPDLink(pdLinkRequest)) {
                throw std::runtime_error("SetupPDLink failed for executor idx " + std::to_string(i));
            }
        });
    }
    // 等待所有线程完成
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    MINDIE_LLM_LOG_INFO("[LlmManagerV2::LlmManagerImpl::UpdateEngineInfo] Success.");
    return true;
}

bool LlmManagerImpl::LlmManagerImpl::QueryPDLinkStatus(model_execute_data::PDLinkStatusResponse &response) {
    if (pdRole_ == Role::FlexP && isFlexInitialized_) {
        MINDIE_LLM_LOG_INFO(
            "[LlmManager::LlmManagerImpl::QueryPDLinkStatus] Flex mode, "
            "skipping query.");
        return true;
    }

    PDLinkStatusRequest pdLinkStatusRequest;
    // Query all executors for link status
    std::vector<std::thread> threads;
    threads.reserve(iExecutorSPtrs_.size());
    for (size_t i = 0; i < iExecutorSPtrs_.size(); i++) {
        threads.emplace_back([&, i]() {
            if (!iExecutorSPtrs_[i]->QueryPDLinkStatus(pdLinkStatusRequest)) {
                throw std::runtime_error("QueryPDLinkStatus failed for executor idx " + std::to_string(i));
            }
        });
    }
    // Wait for all threads to complete
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Aggregate responses from all executors
    for (size_t i = 0; i < iExecutorSPtrs_.size(); i++) {
        const auto &executorResponse = iExecutorSPtrs_[i]->GetPDLinkStatusResponse();
        response.mutable_failed_link_info()->MergeFrom(executorResponse.failed_link_info());
        response.mutable_success_link_info()->MergeFrom(executorResponse.success_link_info());
        response.mutable_running_link_info()->MergeFrom(executorResponse.running_link_info());
        response.mutable_waiting_link_info()->MergeFrom(executorResponse.waiting_link_info());
    }

    MINDIE_LLM_LOG_INFO(
        "[LlmManager::LlmManagerImpl::QueryPDLinkStatus] Query completed "
        "successfully.");
    return true;
}

EngineMetric LlmManagerImpl::CollectEngineMetric(size_t localDPRank) {
    EngineMetric engineMetric = {};
    if (engineConfig_.multiNodesInferEnabled && !engineConfig_.isMaster) {
        return engineMetric;
    }

    if (llmEnginePtr_ == nullptr) {
        return engineMetric;
    }

    engineMetric = llmEnginePtr_->CollectEngineMetric(localDPRank);

    engineMetric.schedulerInfo.reqsInfo.remainBlocks_ = engineMetric.schedulerInfo.blockInfo.freeNpuBlockNum_;
    engineMetric.schedulerInfo.reqsInfo.remainPrefillSlots_ = remainPrefillSlots_;
    engineMetric.schedulerInfo.reqsInfo.remainPrefillTokens_ = remainPrefill_;

    return engineMetric;
}

Status LlmManagerImpl::HandleLoraImpl(const LoraOperation loraOperation, std::vector<LoraParamSPtr> &loraInfo) {
    Status ret;
    size_t dpSize = iExecutorSPtrs_.size();
    if (loraOperation == mindie_llm::LoraOperation::LORA_QUERY) {
        ret = llmEnginePtr_->LoraGetLoaded(loraInfo, dpSize);
        return ret;
    }
    if (pdRole_ != Role::PnD && pdRole_ != Role::FlexP) {
        MINDIE_LLM_LOG_ERROR(
            "[LlmManager::LlmManagerImpl::HandleLoraImpl] Multi Lora does not "
            "support PD separation.");
        return Status(Error::Code::ERROR, "Multi Lora does not support PD separation!");
    }
    if (loraOperation == mindie_llm::LoraOperation::LORA_LOAD) {
        ret = llmEnginePtr_->LoraLoad(loraInfo, dpSize);
    } else if (loraOperation == mindie_llm::LoraOperation::LORA_UNLOAD) {
        ret = llmEnginePtr_->LoraUnLoad(loraInfo, dpSize);
    }
    return ret;
}

bool LlmManagerImpl::UpdateFlexSwitchInfo(const std::shared_ptr<FlexSwitchInfo> flexSwitchInfo) {
    if (flexSwitchInfo == nullptr) {
        MINDIE_LLM_LOG_ERROR("[UpdateFlexSwitchInfo] flexSwitchInfo is nullptr.");
        return false;
    }
    llmEnginePtr_->SetPrefillPercentage(flexSwitchInfo->flexPrefillPercentage);
    return true;
}

bool LlmManagerImpl::ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) const {
    std::string command = commandInfo.command;

    if (command == "CMD_PAUSE_ENGINE" || command == "CMD_PAUSE_ENGINE_ROCE") {
        llmEnginePtr_->PauseScheduling();
        llmEnginePtr_->ExecuteRecoverCommand(commandInfo);
        RecoverCommandInfo clearCommandInfo("CMD_CLEAR_TRANSER");
        llmEnginePtr_->ExecuteRecoverCommand(clearCommandInfo);
    } else if (command == "CMD_REINIT_NPU") {
        llmEnginePtr_->ExecuteRecoverCommand(commandInfo);
    } else if (command == "CMD_START_ENGINE") {
        llmEnginePtr_->ExecuteRecoverCommand(commandInfo);
        llmEnginePtr_->ResumeScheduling();
    } else {
        MINDIE_LLM_LOG_ERROR("Unknown recover command: " + command);
    }
    return true;
}

}  // namespace mindie_llm
