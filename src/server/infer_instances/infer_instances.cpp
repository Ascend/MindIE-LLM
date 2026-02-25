/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <iostream>
#include <sstream>
#include "config_manager.h"
#include "common_util.h"
#include "check_utils.h"
#include "data_type.h"
#include "config_manager_impl.h"
#include "infer_instances.h"

namespace mindie_llm {
const std::string DEFAULT_HOST_IP = "127.0.0.1";
std::shared_ptr<InferInstance> InferInstance::GetInstance()
{
    static std::shared_ptr<InferInstance> instance = std::make_shared<InferInstance>();
    return instance;
}

Status InferInstance::InitFromEndpointCall(const std::string &configPath)
{
    if (!ConfigManager::CreateInstance(configPath)) {
        return Status(Error::Code::ERROR, "Failed to create config manager");
    }
    auto &modelParams = GetModelDeployConfig();
    auto &backendConfig = GetBackendConfig();
    auto &serverConfig = GetServerConfig();

    if (modelParams.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE,
            GenerateInferInstanceErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
            "Failed to init InferInstance. The count of modelDeploy is 0.");
        return Status(Error::Code::ERROR, "The count of modelDeploy is 0.");
    }

    if (backendConfig.modelInstanceNumber == 0 ||
        backendConfig.modelInstanceNumber != backendConfig.npuDeviceIds.size()) {
        std::stringstream err;
        err << "Invalid modelInstanceNumber: " << backendConfig.modelInstanceNumber
            << " or modelInstanceNumber not equal to npuDeviceIds size: " << backendConfig.npuDeviceIds.size();
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE,
            GenerateInferInstanceErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR), err.str());
        return Status(Error::Code::ERROR, err.str());
    }

    for (uint32_t i = 0; i < backendConfig.modelInstanceNumber; ++i) {
        std::map<std::string, std::string> modelConfig;
        modelConfig["configPath"] = configPath;
        modelConfig["npuDeviceIds"] = SerializeSet(backendConfig.npuDeviceIds[i]);
        modelConfig["inferMode"] = serverConfig.inferMode;

        Status modelInstanceInitStatus = InitSingleInferInstance(modelConfig, i);
        if (!modelInstanceInitStatus.IsOk()) {
            ULOG_AUDIT("system", MINDIE_SERVER, "Start InferInstance(" + backendConfig.backendName + ")", "fail");
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE,
                       GenerateInferInstanceErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                       "Failed to init InferInstance id " << i << ", backendName " << backendConfig.backendName);
            return modelInstanceInitStatus;
        }
    }

    ULOG_AUDIT("system", MINDIE_SERVER, "Start InferInstance(" + backendConfig.backendName + ")", "success");
    if (serverConfig.inferMode == "standard") {
        std::string optionStr = "Load model" + modelParams.begin()[0].modelName;
        ULOG_AUDIT("system", MINDIE_SERVER, optionStr, "success");
        if (!llmManagers_.empty() && llmManagers_[0] != nullptr) {
            ConfigManager::GetInstance().SetMaxPositionEmbeddings(llmManagers_[0]->GetMaxPositionEmbeddings());
        }
    }

    started_.store(true);
    ULOG_AUDIT("system", MINDIE_SERVER, "Start InferInstance", "success");
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::InitSingleInferInstance(std::map<std::string, std::string> modelConfig, uint32_t modelInstanceId)
{
    configPath_ = modelConfig["configPath"];
    // for dmi mode and singleNode infer, model instance will be init later by initPD
    if (modelConfig["inferMode"] == "dmi" && !GetBackendConfig().multiNodesInferEnabled) {
        ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE,
                  "In DMI & single machine infer scene, return directly after launching main process;"
                  << "And will wait to initialize model while assigning pd role");
        return Status(Error::Code::OK, "Success");
    }

    std::shared_ptr<LlmManagerV2> llmManager;
    try {
        std::map<std::string, std::string> ipInfo = {{"infer_mode", modelConfig["inferMode"]}};
        llmManager = std::make_shared<LlmManagerV2>(configPath_, nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
    } catch (const std::runtime_error &e) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]", "create an llmManager fail, error : " << e.what());
        return Status(Error::Code::ERROR, e.what());
    }

    if (!llmManager->Init(modelInstanceId, DeserializeSet(modelConfig["npuDeviceIds"])).IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]", "llmManager init fail!");
        return Status(Error::Code::ERROR, "llmManager init fail!");
    }

    llmManagers_.emplace_back(std::move(llmManager));
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::Finalize()
{
    if (!started_.load()) {
        return Status(Error::Code::OK, "Success");
    }
    for (auto &llmManager : llmManagers_) {
        if (llmManager != nullptr) {
            llmManager->Shutdown();
        }
    }
    llmManagers_.clear();

    // clear callbackMap
    for (const auto &key : callbackMap.KeySet()) {
        callbackMap.Erase(key);
    }
    started_.store(false);
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::Process(RequestSPtr request)
{
    CHECK_INITIALIZATION();
    if (callbackMap.Count(request->requestId) != 0) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                   "ReqId" + request->requestId + " already exists in the callbackMap.");
        return Status(Error::Code::ERROR, "Runtime request has been used before.");
    }

    // Select the manager with the most remaining blocks.
    uint64_t maxRemainBlocks = 0;
    std::vector<size_t> candidateIdx;
    for (size_t i = 0; i < llmManagers_.size(); ++i) {
        EngineMetric engineMetric = llmManagers_[i]->CollectEngineMetric();
        const uint64_t remain = engineMetric.schedulerInfo.reqsInfo.remainBlocks_;
        if (remain > maxRemainBlocks) {
            maxRemainBlocks = remain;
            candidateIdx.clear();
            candidateIdx.push_back(i);
        } else if (remain == maxRemainBlocks) {
            candidateIdx.push_back(i);
        }
    }
    if (candidateIdx.empty()) {
        return Status(Error::Code::ERROR, "No available LlmManager.");
    }
    // If multiple managers have the same max remaining blocks, randomly select one.
    size_t chosen = (candidateIdx.size() == 1) ? candidateIdx[0] : candidateIdx[RandomNumber(candidateIdx.size() - 1)];
    callbackMap.Insert(request->requestId, request->serverResponseCallback_);
    auto status = llmManagers_[chosen]->AddRequest(request);
    if (!status.IsOk()) {
        callbackMap.Erase(request->requestId);
        return status;
    }
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::ControlRequest(const RequestIdNew &requestId, OperationV2 operation)
{
    ULOG_DEBUG(SUBMODLE_NAME_INFERINSTANCE, requestId + " Operation " + std::to_string(static_cast<int>(operation)));
    CHECK_INITIALIZATION();
    Status status(Error::Code::ERROR, "ControlRequest not handled by any manager");

    // Send the control request to all llmManagers to ensure the request is found.
    for (auto &llmManager : llmManagers_) {
        Status managerStatus = llmManager->ControlRequest(requestId, operation);
        if (managerStatus.IsOk()) {
            status = managerStatus;
            break;
        }
    }

    if (callbackMap.Count(requestId) != 0) {
        callbackMap.Erase(requestId);
    }

    return status;
}

Status InferInstance::ControlInferInstance(mindie_llm::RecoverCommandInfo &info)
{
    ULOG_DEBUG(SUBMODLE_NAME_INFERINSTANCE, "Operation" + info.command);
    if (!started_.load()) {
        return Status(Error::Code::ERROR, "Infer instance has been finalized or not initialized.");
    }
    if (llmManagers_.empty()) {
        return Status(Error::Code::ERROR, "llmManagers_ is not initialized!");
    }
    if (info.command == "CMD_PAUSE_ENGINE") {
        isPaused_.store(true);
    }
    for (auto &llmManager : llmManagers_) {
        llmManager->ExecuteRecoverCommand(info);
    }
    if (info.command == "CMD_START_ENGINE") {
        isPaused_.store(false);
    }
    bool allSuccess = true;
    for (auto res : info.results) {
        if (res.commandResult != 0) {
            allSuccess = false;
            break;
        }
    }
    return allSuccess ? Status(Error::Code::OK, "Success") :
        Status(Error::Code::ERROR, "Some NPU execute command failed");
}

Status InferInstance::CheckInferInstanceStarted(bool &isStarted)
{
    isStarted = started_.load();
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetProcessingRequest(uint64_t &num)
{
    CHECK_INITIALIZATION();
    uint64_t total = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        total += engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_ +
                 engineMetric.schedulerInfo.reqsInfo.runningRequestNum_ +
                 engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_;
    }
    num = total;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetWaitingRequest(uint64_t &num)
{
    CHECK_INITIALIZATION();
    uint64_t total = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        total += engineMetric.schedulerInfo.reqsInfo.waitingRequestNum_;
    }
    num = total;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetRunningRequest(uint64_t &num)
{
    CHECK_INITIALIZATION();
    uint64_t total = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        total += engineMetric.schedulerInfo.reqsInfo.runningRequestNum_;
    }
    num = total;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetSwappedRequest(uint64_t &num)
{
    CHECK_INITIALIZATION();
    uint64_t total = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        total += engineMetric.schedulerInfo.reqsInfo.swappedRequestNum_;
    }
    num = total;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetCacheBlockNums(uint64_t &freeNpuBlockNums, uint64_t &freeCpuBlockNums,
    uint64_t &totalNpuBlockNums, uint64_t &totalCpuBlockNums)
{
    CHECK_INITIALIZATION();
    uint64_t accumulatedFreeNpuBlocks = 0;
    uint64_t accumulatedFreeCpuBlocks = 0;
    uint64_t accumulatedTotalNpuBlocks = 0;
    uint64_t accumulatedTotalCpuBlocks = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        accumulatedFreeNpuBlocks += engineMetric.schedulerInfo.blockInfo.freeNpuBlockNum_;
        accumulatedFreeCpuBlocks += engineMetric.schedulerInfo.blockInfo.freeCpuBlockNum_;
        accumulatedTotalNpuBlocks += engineMetric.schedulerInfo.blockInfo.totalNpuBlockNum_;
        accumulatedTotalCpuBlocks += engineMetric.schedulerInfo.blockInfo.totalCpuBlockNum_;
    }
    freeNpuBlockNums = accumulatedFreeNpuBlocks;
    freeCpuBlockNums = accumulatedFreeCpuBlocks;
    totalNpuBlockNums = accumulatedTotalNpuBlocks;
    totalCpuBlockNums = accumulatedTotalCpuBlocks;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetRadixMatchNums(uint64_t &allRadixMatchNum, uint64_t &npuRadixMatchHitNum)
{
    CHECK_INITIALIZATION();
    uint64_t totalRadixMatchCount = 0;
    uint64_t npuRadixMatchHitCount = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        totalRadixMatchCount += engineMetric.schedulerInfo.reqsInfo.allRadixMatchNum_;
        npuRadixMatchHitCount += engineMetric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_;
    }
    allRadixMatchNum = totalRadixMatchCount;
    npuRadixMatchHitNum = npuRadixMatchHitCount;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetCumulativePreemptCount(uint64_t &cumulativePreemptCount)
{
    CHECK_INITIALIZATION();
    uint64_t total = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        total += engineMetric.schedulerInfo.reqsInfo.cumulativePreemptCount_;
    }
    cumulativePreemptCount = total;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetThroughput(float &prefillThroughput, float &decodeThroughput)
{
    CHECK_INITIALIZATION();
    float totalPrefillThroughput = 0.0f;
    float totalDecodeThroughput = 0.0f;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        totalPrefillThroughput += engineMetric.prefillThroughput_;
        totalDecodeThroughput += engineMetric.decodeThroughput_;
    }
    prefillThroughput = totalPrefillThroughput;
    decodeThroughput = totalDecodeThroughput;
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetRequestBlockQuotas(uint64_t &remainBlocks, uint64_t &remainPrefillSlots,
    uint64_t &remainPrefillTokens,
    std::map<uint32_t, uint64_t> &dpRemainBlocks)
{
    CHECK_INITIALIZATION();
    uint64_t totalRemainBlocks = 0;
    uint64_t totalRemainPrefillSlots = 0;
    uint64_t totalRemainPrefillTokens = 0;
    for (auto &llmManager : llmManagers_) {
        EngineMetric engineMetric = llmManager->CollectEngineMetric();
        totalRemainBlocks += engineMetric.schedulerInfo.reqsInfo.remainBlocks_;
        totalRemainPrefillSlots += engineMetric.schedulerInfo.reqsInfo.remainPrefillSlots_;
        totalRemainPrefillTokens += engineMetric.schedulerInfo.reqsInfo.remainPrefillTokens_;
    }
    remainBlocks = totalRemainBlocks;
    remainPrefillSlots = totalRemainPrefillSlots;
    remainPrefillTokens = totalRemainPrefillTokens;
    dpRemainBlocks = dpRemainBlocks_;

    ULOG_DEBUG(SUBMODLE_NAME_INFERINSTANCE, "Backend manager get processing request. total remain blocks: "
            << remainBlocks << ", total remain prefill tokens: "
            << remainPrefillSlots << " , total Remain prefill slots: " << remainPrefillTokens);
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetNodeStatus(std::map<std::string, NodeHealthStatus> &slaveStatus)
{
    CHECK_INITIALIZATION();
    slaveStatus = slavesStatus_;
    return Status(Error::Code::OK, "get node status success");
}

static bool ProcessFailLinkIp(RequestSPtr request, GlobalIpInfo &globalIpInfo)
{
    globalIpInfo.failLinkInstanceIDAndReason.clear();

    if (request->failedLinkInfos.size() == 0) {
        ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "[ProcessFailLinkIp] All link ips succeed. ");
        return true;
    }

    std::vector<uint64_t> failInstanceIdVec; // <failInstanceId1, failInstanceId2, ...>
    for (auto &failedLinkInfo : request->failedLinkInfos) {
        failInstanceIdVec.emplace_back(failedLinkInfo.cluster_id);
        globalIpInfo.failLinkInstanceIDAndReason[failedLinkInfo.cluster_id] = failedLinkInfo.failReason;
        globalIpInfo.retryLinkIpInfo[failedLinkInfo.cluster_id] = globalIpInfo.linkIpInfo[failedLinkInfo.cluster_id];
        ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE,
                  "[ProcessFailLinkIp] add retry link instance id : " + std::to_string(failedLinkInfo.cluster_id));
    }

    globalIpInfo.linkIpInfo = RemoveMapElements(globalIpInfo.linkIpInfo, failInstanceIdVec);
    ULOG_DEBUG(SUBMODLE_NAME_INFERINSTANCE, "[ProcessFailLinkIp] Process link ips done. ");
    return true;
}

static void AddAttributeToRequest(const GlobalIpInfo &globalIpInfo, RequestSPtr request)
{
    std::string role = globalIpInfo.role;
    PDRole roleInt = PDRole::UNKNOWN;
    if (role == "prefill") {
        roleInt = PDRole::PREFILL;
    } else if (role == "decode") {
        roleInt = PDRole::DECODE;
    } else if (role == "flex") {
        roleInt = PDRole::Flex;
    } else {
        roleInt = PDRole::UNKNOWN;
    }

    // 修改 hostIpNum 计算逻辑：统计所有 DP 实例中 host IP 的总数量
    int64_t hostIpNum = 0;
    for (const auto& pair : globalIpInfo.hostIpInfo) {
        int64_t currentHostIpCount = static_cast<int64_t>(pair.second.size());
        hostIpNum += currentHostIpCount;
    }
    request->role = roleInt;
    request->needSwitch = globalIpInfo.needSwitch;
    request->linkNum = globalIpInfo.linkIpInfo.size();
    request->unlinkNum = globalIpInfo.unlinkIpInfo.size();
    request->hostIpNum = globalIpInfo.hostIpInfo.size();
    request->superPodIdNum = globalIpInfo.superPodIdInfo.size();
    request->containsDpInstanceIds = globalIpInfo.localDpInstanceIds.empty() ? 0 : 1;
}

static bool ProcessDevice(const std::pair<uint64_t, std::vector<DeviceInfo>> &pair, RequestSPtr request,
    std::unordered_map<InstanceId, std::vector<std::pair<std::string, int64_t>>> &dpInstance2Devices,
    std::unordered_map<InstanceId, std::vector<int64_t>> &dpInstance2SuperDeviceIds,
    const std::map<uint64_t, std::vector<std::string>> &hostIpInfo,
    std::map<uint64_t, std::string> &superPodIdInfo)
{
    try {
        const uint64_t &instanceId = pair.first;
        const std::vector<DeviceInfo> &devicesIP = pair.second;
        ULOG_DEBUG(SUBMODLE_NAME_INFERINSTANCE, "hostIpInfo dpinstanceId" << instanceId);
        std::vector<std::string> hostIps;
        auto it = hostIpInfo.find(instanceId);
        if (it != hostIpInfo.cend() && !it->second.empty()) {
            hostIps = it->second;
        } else {
            hostIps = {DEFAULT_HOST_IP};
        }
        std::string superPodId = "";
        if (superPodIdInfo.find(instanceId) != superPodIdInfo.end()) {
            superPodId = superPodIdInfo[instanceId];
        }
        // Process host
        request->dpInstance2HostIps[instanceId] = hostIps;
        if (!superPodId.empty()) {
            request->dpInstance2SuperPodId[instanceId] = std::stoll(superPodId);
        }

        // Process device IPs
        for (const auto &device : devicesIP) {
            std::string deviceIpAddress = device.deviceIp;
            int64_t devicePhysicalId = device.devicePhysicalId;
            int64_t superDeviceId = device.superDeviceId;
            dpInstance2Devices[instanceId].push_back({deviceIpAddress, devicePhysicalId});
            if (superDeviceId != -1) {
                dpInstance2SuperDeviceIds[instanceId].push_back(superDeviceId);
            }
        }
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "MIE05E040001", "[AddDevicesToRequest] Failed to parse device ip");
        return false;
    }
    return true;
}

static bool AddDevicesToRequest(GlobalIpInfo &globalIpInfo, RequestSPtr request)
{
    ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "AddDevicesToRequest" << "GlobalIpInfo" << globalIpInfo.ToString());
    for (auto &pair : std::as_const(globalIpInfo.linkIpInfo)) {
        ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "hostIpInfo dpInstanceId is " << pair.first);
        if (!ProcessDevice(pair, request, request->dpInstance2LinkDevices, request->dpInstance2LinkSuperDeviceIds,
                           globalIpInfo.hostIpInfo, globalIpInfo.superPodIdInfo)) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[AddDevicesToRequest] Failed to AddDevicesToRequest for lack of some link devices.");
            return false;
        }
    }
    // build unlinkDevices IP
    for (auto& pair : std::as_const(globalIpInfo.unlinkIpInfo)) {
        if (!ProcessDevice(pair, request, request->dpInstance2UnlinkDevices, request->dpInstance2UnLinkSuperDeviceIds,
                           globalIpInfo.hostIpInfo, globalIpInfo.superPodIdInfo)) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[AddDevicesToRequest] Failed to AddDevicesToRequest for lack of some unlink devices.");
            return false;
        }
    }
    return true;
}

static void AddPolicyToRequest(GlobalIpInfo &globalIpInfo, RequestSPtr request)
{
    request->spInfo = globalIpInfo.spInfo;
    request->cpInfo = globalIpInfo.cpInfo;

    // Backward-compatible defaults if caller passed empty maps
    if (request->spInfo.empty()) {
        request->spInfo.emplace(0, 1);
    }
    if (request->cpInfo.empty()) {
        request->cpInfo.emplace(0, 1);
    }
}

static void CreateIpInfo(const GlobalIpInfo &globalIpInfo, std::map<std::string, std::string> &ipInfo)
{
    // Since there is no config manager, it is hardcoded for now. Only `dmi` will enter this function.
    ipInfo["infer_mode"] = "dmi";
    if (globalIpInfo.role == "decode") {
        ipInfo["role"] = "decoder";
    } else {
        ipInfo["role"] = globalIpInfo.role;
    }
    ipInfo["needSwitch"] = (globalIpInfo.needSwitch ? "true" : "false");
    ipInfo["local_instance_id"] = std::to_string(globalIpInfo.localInstanceId);

    ipInfo["local_host_ip"] = JoinStrings(globalIpInfo.localHostIpList, ","); // ip1, ip2, ip3
    if (!globalIpInfo.localSuperPodId.empty()) {
        ipInfo["local_super_pod_id"] = globalIpInfo.localSuperPodId;
    }
    ipInfo["local_device_ip"] = JoinStrings(globalIpInfo.localDeviceIps, ",");
    ipInfo["local_logic_device_id"] = JoinStrings(globalIpInfo.localDeviceLogicalIds, ",");
    ipInfo["local_physical_device_id"] = JoinStrings(globalIpInfo.localDevicePhysicalIds, ",");
    ipInfo["local_rank_ids"] = JoinStrings(globalIpInfo.localDeviceRankIds, ",");
    ipInfo["local_prefill_percentage"] = std::to_string(globalIpInfo.flexPrefillPercentage);
    if (!globalIpInfo.localSuperDeviceIds.empty()) {
        ipInfo["local_super_device_id"] = JoinStrings(globalIpInfo.localSuperDeviceIds, ",");
    }
    if (globalIpInfo.isSingleContainer) {
        ipInfo["lccl_comm_shard_id"] = std::to_string(globalIpInfo.instanceIdxInPod);
        ipInfo["num_lccl_comm_shards"] = std::to_string(globalIpInfo.numInstancesPerPod);
    }

    std::ostringstream tmpOss;
    for (auto &pair : ipInfo) {
        tmpOss << pair.first << ":" << pair.second << ",";
    }
    std::string ipInfoLog = tmpOss.str();
    ipInfoLog = "IpInfo As: " + ipInfoLog;
    ULOG_DEBUG(SUBMODLE_NAME_INFERINSTANCE, ipInfoLog);
}

// Pass IpInfo, including local ip infos and whethe to link or unlink remote
Status InferInstance::AssignDmiRole(GlobalIpInfo &globalIpInfo)
{
    // Init P/D node
    if (globalIpInfo.needInit) {
        auto res = InitPDNode(globalIpInfo);
        if (!res.IsOk()) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[InferInstance::AssignDmiRole] InitPDNode fail!");
            return res;
        }
        globalIpInfo.needSwitch = false;
    }

    RequestSPtr runtimeRequest = std::make_shared<Request>(RequestIdNew{"0"});
    AddAttributeToRequest(globalIpInfo, runtimeRequest);

    // If both linkIpInfo and unlinkIpInfo are empty, skip AddDevicesToRequest
    if (!(globalIpInfo.linkIpInfo.empty() && globalIpInfo.unlinkIpInfo.empty())) {
        if (!AddDevicesToRequest(globalIpInfo, runtimeRequest)) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "MIE05E040001",
                       "[AssignDmiRole] Failed to add DEVICES to request.");
            return Status(Error::Code::ERROR, "ERROR: Failed to add DEVICES to request.");
        }
    }
    AddPolicyToRequest(globalIpInfo, runtimeRequest);

    // Update every manager (multi-instance)
    for (auto &llmManager : llmManagers_) {
        if (!llmManager->UpdateEngineInfo(runtimeRequest, false /* forcePRelease */)) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[LLMAssignDmiRole] UpdateEngineInfo failed on one instance.");
            return Status(Error::Code::ERROR, "UpdateEngineInfo failed.");
        }
    }

    if (globalIpInfo.role == "flex") {
        auto flexSwitchInfo = std::make_shared<FlexSwitchInfo>();
        flexSwitchInfo->flexPrefillPercentage = globalIpInfo.flexPrefillPercentage;
        void UpdateFlexSwitchInfo(const std::shared_ptr<FlexSwitchInfo> flexSwitchInfo);
    }
    if (globalIpInfo.role != "flex" && !ProcessFailLinkIp(runtimeRequest, globalIpInfo)) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                   "[LLMAssignDmiRole] Failed to delete fail ip add from globalIpInfo.");
        return Status(Error::Code::ERROR, "Fail to delete fail ip add from globalIpInfo.");
    }
    ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "[InferInstance::AssignDmiRole] Success.");
    ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "system update pd role to " + globalIpInfo.role +"success");
    ConfigManager::GetInstance().SetMaxPositionEmbeddings(llmManagers_.at(0)->GetMaxPositionEmbeddings());
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::InitPDNode(GlobalIpInfo &globalIpInfo)
{
    std::map<std::string, std::string> ipInfo;
    CreateIpInfo(globalIpInfo, ipInfo);

    if (!GetBackendConfig().multiNodesInferEnabled) {
        for (uint32_t i = 0; i < GetBackendConfig().modelInstanceNumber; ++i) {
            std::shared_ptr<LlmManagerV2> llmManager;
            try {
                llmManager =
                    std::make_shared<LlmManagerV2>(configPath_, nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
            } catch (const std::runtime_error &e) {
                ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "MIE05E040001",
                           "[InferInstance::AssignDmiRole] create an llmManager fail, error : " << e.what());
                return Status(Error::Code::ERROR, e.what());
            }
            llmManagers_.emplace_back(std::move(llmManager));
        }
    }

    for (uint32_t idx = 0; idx < llmManagers_.size(); ++idx) {
        std::shared_ptr<LlmManagerV2> llmManager = llmManagers_[idx];
        std::set<size_t> deviceIds;
        for (auto &id : Split(ipInfo["local_logic_device_id"], ',')) {
            deviceIds.insert(static_cast<size_t>(stoi(id)));
        }
        if (llmManager == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[InferInstance::AssignDmiRole] llmManager is nullptr!");
            return Status(Error::Code::ERROR, "llmManager is nullptr!");
        }

        Status status(Error::Code::OK, "Success");
        if (GetBackendConfig().multiNodesInferEnabled) {
            status = llmManager->InitModelForMultiPd(ipInfo, idx);
        } else if (GetServerConfig().distDPServerEnabled) {
            status = llmManager->Init(idx, deviceIds, ipInfo);
        } else {
            status = llmManager->Init(idx, deviceIds);
        }
        if (!status.IsOk()) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[InferInstance::InitPDNode] llmManager init fail!");
            return Status(Error::Code::ERROR, "llmManager init fail!");
        }
    }
    globalIpInfo.needInit = false;
    started_.store(true);
    ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "system init pd role success");
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::HandleLora(
    const LoraOperation& loraOperation,
    std::vector<LoraParamSPtr>& loraInfo
)
{
    ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "Start HandleLora");
    if (llmManagers_[0] == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]", "llmManager is nullptr");
        return Status(Error::Code::ERROR, "llmManager is nullptr");
    }
    auto stats = llmManagers_[0]->HandleLora(loraOperation, loraInfo);

    return stats;
}

Status InferInstance::ForcePRelease()
{
    RequestSPtr llmInferRequest = std::make_shared<Request>(RequestIdNew{"0"});
    for (auto &llmManager : llmManagers_) {
        if (!llmManager->UpdateEngineInfo(llmInferRequest, true)) {
            ULOG_ERROR(SUBMODLE_NAME_INFERINSTANCE, "[MIE05E040001]",
                       "[LLMForcePRelease] Failed to update llmInferEngine info.");
            return Status(Error::Code::ERROR, "llmInferEngine update engine info fail!");
        }
    }
    ULOG_INFO(SUBMODLE_NAME_INFERINSTANCE, "[InferInstance::ForcePRelease] Success.");
    return Status(Error::Code::OK, "Success");
}

Status InferInstance::GetBatchSchedulerMetrics(std::map<std::string, uint64_t> &batchSchedulerMetrics)
{
    CHECK_INITIALIZATION();
    uint64_t processingRequestNum = 0;
    uint64_t waitingRequestNum = 0;
    uint64_t runningRequestNum = 0;
    uint64_t swappedRequestNum = 0;
    GetProcessingRequest(processingRequestNum);
    GetWaitingRequest(waitingRequestNum);
    GetRunningRequest(runningRequestNum);
    GetSwappedRequest(swappedRequestNum);

    uint64_t remainBlocks = 0;
    uint64_t remainPrefillSlots = 0;
    uint64_t remainPrefillTokens = 0;
    std::map<uint32_t, uint64_t> dpRemainBlocks;
    GetRequestBlockQuotas(remainBlocks, remainPrefillSlots, remainPrefillTokens, dpRemainBlocks);

    batchSchedulerMetrics["waitingInferRequestNum"] = waitingRequestNum;
    batchSchedulerMetrics["processingInferRequestNum"] = processingRequestNum;
    batchSchedulerMetrics["runningInferRequestNum"] = runningRequestNum;
    batchSchedulerMetrics["swappedInferRequestNum"] = swappedRequestNum;
    batchSchedulerMetrics["remainBlocks"] = remainBlocks;

    return Status(Error::Code::OK, "Success");
}

std::string InferInstance::GetPDRole() const
{
    if (pdRole_ == PDRole::PREFILL) {
        return "prefill";
    } else if (pdRole_ == PDRole::DECODE) {
        return "decode";
    } else {
        return "none";
    }
}

PDRoleStatus InferInstance::GetPDRoleStatus() const { return pdRoleStatus_; }

void InferInstance::SetPDRoleStatus(PDRoleStatus status) { pdRoleStatus_ = status; }

void InferInstance::UpdatePDRole(const std::string &role)
{
    // 当前角色是从环境变量里读取的，只要是P或D则认为status是ready
    // 目标态是动态指定，待进化到目标态后，这里的状态不能写死
    if (role == "decode") {
        pdRole_ = PDRole::DECODE;
    } else if (role == "prefill") {
        pdRole_ = PDRole::PREFILL;
    } else {
        pdRole_ = PDRole::UNKNOWN;
    }
}

bool InferInstance::IsLlmEngineReady() const
{
    if (llmManagers_.empty()) {
        return false;
    }
    // 检查所有 LlmManager 是否都已就绪
    for (const auto &manager : llmManagers_) {
        if (manager == nullptr || !manager->IsLlmEngineReady()) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<InferInstance> GetInferInstance() { return InferInstance::GetInstance(); }
} // namespace mindie_llm