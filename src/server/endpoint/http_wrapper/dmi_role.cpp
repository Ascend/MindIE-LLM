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
#include "dmi_role.h"

#include <unordered_map>

#include "grpc_communication_mng.h"
#include "json_util.h"
#include "log.h"
#include "parse_protocol.h"

using OrderedJson = nlohmann::ordered_json;
constexpr uint32_t MAX_INSTANCE_PER_POD = 64;  // max instances per pod
constexpr uint32_t MAX_P_PERCENTAGE = 100;     // max p percentage is 100
constexpr uint32_t MIN_P_PERCENTAGE = 0;       // min p percentage is 0

namespace mindie_llm {

constexpr uint32_t QUERY_INTERVAL_SECONDS = 10;

DmiRole::DmiRole() {}

DmiRole::~DmiRole() {
    try {
        keepAlive.store(false);  // 关闭保活响应
        taskQueue_.Push(nullptr);
    } catch (...) {
        // safe destruct
    }
}
std::shared_ptr<DmiRole> DmiRole::GetInstance() {
    static std::shared_ptr<DmiRole> dmiRoleInstance = std::make_shared<DmiRole>();
    return dmiRoleInstance;
}

void DmiRole::RunTaskThread() {
#ifndef UT_ENABLED
    if (!taskThread_.joinable()) {
        taskThread_ = std::thread([this]() { this->TaskThread(); });
    }
#endif
}

void DmiRole::RunQueryThread() {
#ifndef UT_ENABLED
    if (!queryThread_.joinable()) {
        queryThread_ = std::thread([this]() {
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Start periodic link status query thread (every 10 seconds).");
            while (!queryTerminate_.load()) {
                QueryLinkStatus();
                // Sleep for 10 seconds before next query
                std::this_thread::sleep_for(std::chrono::seconds(QUERY_INTERVAL_SECONDS));
            }
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Query thread has been stopped.");
        });
    }
#endif
}

void DmiRole::TaskThread() {
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Start task thread.");
    while (!taskTerminate_.load()) {
        auto task = taskQueue_.Take();
        if (task == nullptr) {
            break;
        }

        // Check if we need to stop current task
        if (taskRunning_.load()) {
            // Stop current task - this is a simple implementation
            // In a real scenario, you might need more sophisticated
            // cancellation
            taskRunning_.store(false);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Stopped current task.");
        }

        // Execute new task
        task();
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Finish a task.");
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Task thread has been stopped.");
}

void DmiRole::StopCurrentTask() {
    if (taskRunning_.load()) {
        taskRunning_.store(false);
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Requested to stop current task.");
    }
}

// this function will be used to establish links between devices
void DmiRole::ExecuteLinkTask(GlobalIpInfo globalIpInfo) {
    taskRunning_.store(true);

    // Check if task was stopped before starting
    if (!taskRunning_.load()) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Task was stopped before execution.");
        taskRunning_.store(false);
        return;
    }

    // Set linking status for the devices being linked
    {
        std::lock_guard<std::mutex> lock(mtx_);
        linkingLinkIP_ = globalIpInfo.linkIpInfo;
        linkingHostIP_ = globalIpInfo.hostIpInfo;
        // Initialize link status for all linking instances
        for (const auto &linkPair : linkingLinkIP_) {
            remoteNodeLinkStatus_[linkPair.first] = {"linking", false};
        }
    }

    Status result = GetInferInstance()->AssignDmiRole(globalIpInfo);
    if (!result.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                   "Update PD role failed. Error code: " << static_cast<int>(result.StatusCode())
                                                         << ", Error message: " << result.StatusMsg());
        // Clear linking status on failure
        std::lock_guard<std::mutex> lock(mtx_);
        linkingLinkIP_.clear();
        linkingHostIP_.clear();
    } else {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "AssignDmiRole called successfully.");
        assignedRole_ = true;
    }
    taskRunning_.store(false);
}

void DmiRole::QueryLinkStatus() {
    // Protect all member variable access during the operation
    std::lock_guard<std::mutex> lock(mtx_);

    // Skip query if assignDmiRole has not been called yet
    if (!assignedRole_) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Skipping link status query - assignDmiRole not called yet.");
        return;
    }

    // Skip query if there are no linking connections
    if (linkingLinkIP_.empty() && runningLinkIP_.empty() && waitingLinkIP_.empty()) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Skipping link status query - no linking connections.");
        return;
    }

    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Sending periodic link status query to backend.");

    // Query link status from backend
    model_execute_data::PDLinkStatusResponse response;
    Status result = GetInferInstance()->QueryPDLinkStatus(response);
    if (!result.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                   "Query link status failed. Error message: " + result.StatusMsg());
        return;
    }

    // Process the response and update linking/success tables
    const auto &failedLinks = response.failed_link_info();
    const auto &successLinks = response.success_link_info();
    const auto &runningLinks = response.running_link_info();
    const auto &waitingLinks = response.waiting_link_info();

    ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
              "Processing link status: failed_links=" << failedLinks.size() << ", success_links=" << successLinks.size()
                                                      << ", running_links=" << runningLinks.size()
                                                      << ", waiting_links=" << waitingLinks.size());

    runningLinkIP_.clear();
    waitingLinkIP_.clear();
    for (const auto &runningLink : runningLinks) {
        runningLinkIP_.emplace_back(runningLink);
    }
    for (const auto &waitingLink : waitingLinks) {
        waitingLinkIP_.emplace_back(waitingLink);
    }

    // Process failed links - remove from linking
    ProcessFailedLinks(failedLinks);

    // Process success links - move from linking to success
    ProcessSuccessfulLinks(successLinks);

    // Set role to READY if linking table is empty and success table is not
    // empty
    CheckAllLinksCompleted();

    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Periodic link status query completed successfully.");
}

template <typename T>
void DmiRole::ProcessFailedLinks(const T &failedLinks) {
    for (const auto &failedInfo : failedLinks) {
        try {
            uint64_t instanceId = std::stoull(failedInfo.cluster_id());
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Processing failed link for instanceId: " << instanceId);

            // Always update the status for failed instances, regardless of
            // linking state This handles cases where backend reports failures
            // for instances not currently tracked
            std::string failedReason = "failed : " + std::to_string(static_cast<int>(failedInfo.pd_error_code()));
            remoteNodeLinkStatus_[instanceId] = {failedReason, true};

            // Remove from linking state if it was actually linking
            if (linkingLinkIP_.find(instanceId) != linkingLinkIP_.end()) {
                linkingLinkIP_.erase(instanceId);
                linkingHostIP_.erase(instanceId);

                std::string failMsg =
                    "Link failed for instance id: " + std::to_string(instanceId) +
                    " with error code: " + std::to_string(static_cast<int>(failedInfo.pd_error_code()));
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, failMsg);
            }
        } catch (const std::exception &e) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                       "Invalid cluster_id in failed link info: " << failedInfo.cluster_id());
        }
    }
}

template <typename T>
void DmiRole::ProcessSuccessfulLinks(const T &successLinks) {
    // Process success links - move from linking to success
    for (const auto &linkingPair : linkingLinkIP_) {
        uint64_t instanceId = linkingPair.first;
        const std::vector<DeviceInfo> &deviceInfos = linkingPair.second;

        // Check if this instance has any successful device IPs
        bool hasSuccessfulDevice = false;
        for (const DeviceInfo &deviceInfo : deviceInfos) {
            if (std::find(successLinks.begin(), successLinks.end(), deviceInfo.deviceIp) != successLinks.end()) {
                hasSuccessfulDevice = true;
                break;
            }
        }

        if (hasSuccessfulDevice) {
            // This link succeeded - move to success list
            successLinkIP_[instanceId] = linkingLinkIP_[instanceId];
            successHostIP_[instanceId] = linkingHostIP_[instanceId];
            remoteNodeLinkStatus_[instanceId] = {"ok", true};
            std::string successMsg = "Link succeeded for instance id: " + std::to_string(instanceId);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, successMsg);
        }
    }

    // Remove successful instances from linking
    auto it = linkingLinkIP_.begin();
    while (it != linkingLinkIP_.end()) {
        uint64_t instanceId = it->first;
        if (successLinkIP_.find(instanceId) != successLinkIP_.end()) {
            it = linkingLinkIP_.erase(it);
            linkingHostIP_.erase(instanceId);
        } else {
            ++it;
        }
    }
}

void DmiRole::CheckAllLinksCompleted() {
    if (linkingLinkIP_.empty() && !successLinkIP_.empty() && runningLinkIP_.empty() && waitingLinkIP_.empty()) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "All links completed successfully. Setting role status to READY.");
        GetInferInstance()->SetPDRoleStatus(PDRoleStatus::READY);
    }
}

bool DmiRole::PDParseRequestBodyToJson(const ReqCtxPtr &reqCtx, ordered_json &body) const noexcept {
    try {
        std::string msgBody = reqCtx->MsgBody();
        if (!ordered_json::accept(msgBody)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                       "Convert string to json object failed, CallbackId is " << reqCtx->CallbackId());
            return false;
        }
        body = ordered_json::parse(msgBody, CheckOrderedJsonDepthCallback);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Convert string to json object exception, CallbackId is " << reqCtx->CallbackId());
        return false;
    }
    return true;
}

bool DmiRole::UpdatePDInfo(const std::string &roleName, const std::string &preRole, const ordered_json &body,
                           GlobalIpInfo &globalIpInfo) {
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Previous role is " << preRole << ". role in request is " << roleName);
    bool res = true;
    if (preRole == "none") {  // 初始化
        res = UpdatePDSwitchInfo(roleName, body, globalIpInfo, true);
        // flex 节点不初始化
    } else if (preRole != roleName && preRole != "flex" && roleName != "flex") {
        res = UpdatePDSwitchInfo(roleName, body, globalIpInfo, false);
    } else if (preRole == roleName) {
        res = UpdatePDNotSwitchInfo(roleName, body, globalIpInfo);
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Whether the node's role needs to be switched: " << globalIpInfo.needSwitch);
    return res;
}

bool DmiRole::UpdatePDInfoV2(const std::string &roleName, const std::string &preRole, const ordered_json &body,
                             GlobalIpInfo &globalIpInfo) {
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Previous role is : " << preRole << ". role in request is : " << roleName);
    bool res = true;
    if (preRole == "none") {  // 初始化
        res = UpdatePDSwitchInfoV2(roleName, body, globalIpInfo, true);
    } else if (preRole != roleName) {
        res = UpdatePDSwitchInfoV2(roleName, body, globalIpInfo, false);
    } else if (preRole == roleName) {
        res = UpdatePDNotSwitchInfoV2(roleName, body, globalIpInfo);
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Whether the node's role needs to be switched: " << globalIpInfo.needSwitch);
    return res;
}

void DmiRole::ProcessInitInfo(const ordered_json &body, GlobalIpInfo &globalIpInfo) {
    try {
        globalIpInfo.needInit = true;
        globalIpInfo.localInstanceId = body["local"]["id"];
        globalIpInfo.localHostIpList.emplace_back(body["local"]["host_ip"]);
        if (body["local"].contains("super_pod_id")) {
            globalIpInfo.localSuperPodId = body["local"]["super_pod_id"];
        }
        if (body["local"].contains("instance_idx_in_pod")) {
            globalIpInfo.instanceIdxInPod = body["local"]["instance_idx_in_pod"];
        }
        if (body["local"].contains("num_instances_per_pod")) {
            globalIpInfo.numInstancesPerPod = body["local"]["num_instances_per_pod"];
        }
        if (body["local"].contains("is_single_container")) {
            globalIpInfo.isSingleContainer = body["local"]["is_single_container"];
        }
        globalIpInfo.hostIpInfo[globalIpInfo.localInstanceId] = globalIpInfo.localHostIpList;
        localInstanceId_ = globalIpInfo.localInstanceId;
        for (const auto &deviceInfo : body["local"]["device"]) {
            globalIpInfo.localDeviceIps.emplace_back(deviceInfo["device_ip"]);
            globalIpInfo.localDeviceLogicalIds.emplace_back(deviceInfo["device_logical_id"]);
            globalIpInfo.localDevicePhysicalIds.emplace_back(deviceInfo["device_id"]);
            if (deviceInfo.contains("super_device_id")) {
                globalIpInfo.localSuperDeviceIds.emplace_back(deviceInfo["super_device_id"]);
            }
        }
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred while processing dmi role init information: " << e.what());
        throw std::runtime_error("DmiRole::ProcessInitInfo" + std::string(e.what()));
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred while processing dmi role init information.");
        throw std::runtime_error("DmiRole::ProcessInitInfo Error");
    }
}

// update local ip info
void DmiRole::ProcessInitInfoV2(const ordered_json &body, GlobalIpInfo &globalIpInfo) {
    try {
        // update local info
        globalIpInfo.needInit = true;
        auto firstNodeInfo = body["local"][0];
        auto dpInstanceId = firstNodeInfo["dp_inst_list"][0]["dp_inst_id"];
        globalIpInfo.numInstancesPerPod = MAX_INSTANCE_PER_POD;
        auto ret = ReverseDpInstId(dpInstanceId);
        globalIpInfo.localInstanceId = ret.first;
        localInstanceId_ = globalIpInfo.localInstanceId;
        if (firstNodeInfo.contains("super_pod_id")) {
            globalIpInfo.localSuperPodId = firstNodeInfo["super_pod_id"];
        }

        for (const auto &nodeInfo : body["local"]) {
            globalIpInfo.localHostIpList.emplace_back(nodeInfo["host_ip"]);
            for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                globalIpInfo.localDpInstanceIds.emplace_back(dpGroupInfo["dp_inst_id"]);
                for (const auto &deviceInfo : dpGroupInfo["device"]) {
                    globalIpInfo.localDeviceIps.emplace_back(deviceInfo["device_ip"]);
                    globalIpInfo.localDeviceLogicalIds.emplace_back(deviceInfo["device_logical_id"]);
                    globalIpInfo.localDevicePhysicalIds.emplace_back(deviceInfo["device_id"]);
                    globalIpInfo.localDeviceRankIds.emplace_back(deviceInfo["rank_id"]);
                    if (deviceInfo.contains("super_device_id")) {
                        globalIpInfo.localSuperDeviceIds.emplace_back(deviceInfo["super_device_id"]);
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred : " << e.what());
        throw std::runtime_error("DmiRole::ProcessInitInfoV2" + std::string(e.what()));
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Unknown error occurred.");
        throw std::runtime_error("DmiRole::ProcessInitInfoV2 Unknown Error");
    }
}

bool DmiRole::UpdatePDSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
                                 bool needInit) {
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = true;
    if (needInit) {  // 初始化
        ProcessInitInfo(body, globalIpInfo);
        // 初始化的时候P、D或flex的p_percentage为0、100时，peers不允许为空
        if (body["peers"].size() == 0 &&
            (roleName != "flex" || globalIpInfo.flexPrefillPercentage == MIN_P_PERCENTAGE ||
             globalIpInfo.flexPrefillPercentage == MAX_P_PERCENTAGE)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                       "[DmiRole::UpdatePDSwitchLinkInfo] Parse req json "
                       "failed, while role is P, D, "
                       "or flex's p_percentage is "
                           << MIN_P_PERCENTAGE << " or " << MAX_P_PERCENTAGE << ", peers can't be empty.");
            return false;
        }
        if (body["local"].contains("p_percentage")) {
            globalIpInfo.flexPrefillPercentage = body["local"]["p_percentage"];
            FlexPPercentageProcessor::GetInstance().SetPdRoleFlexPPercentage(globalIpInfo.flexPrefillPercentage);
        }
    }

    // 根据peers信息更新linkIpInfo,
    // 更新链接信息时，根据当前链接信息，只更新新增的链接 和 需要断开的链接
    try {
        std::map<uint64_t, std::vector<DeviceInfo>> currentLinkIpInfo{};
        std::string superPodId;
        for (const auto &serverInfo : body["peers"]) {
            if (serverInfo.contains("super_pod_id")) {
                superPodId = serverInfo["super_pod_id"];
                globalIpInfo.superPodIdInfo[serverInfo["id"]] = serverInfo["super_pod_id"];
            }
            uint32_t instanceId = serverInfo["id"];
            globalIpInfo.localHostIpList.emplace_back(body["local"]["host_ip"]);
            globalIpInfo.hostIpInfo[instanceId] = {serverInfo["host_ip"]};
            instanceIdToServerIp_[instanceId] = serverInfo["server_ip"];
            std::vector<DeviceInfo> linkDeviceIp;
            for (const auto &deviceInfo : serverInfo["device"]) {
                DeviceInfo device;
                device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                if (deviceInfo.contains("super_device_id")) {
                    device.superDeviceId = std::stoi(deviceInfo["super_device_id"].get<std::string>());
                }
                linkDeviceIp.emplace_back(device);
            }
            currentLinkIpInfo.insert({instanceId, linkDeviceIp});
        }
        UpdateIpInfo(globalIpInfo, currentLinkIpInfo, superPodId);
        return true;
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred. " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Unknown error occurred.");
        return false;
    }
}

bool DmiRole::UpdatePDSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
                                   bool needInit) {
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = true;
    if (needInit) {  // 初始化
        ProcessInitInfoV2(body, globalIpInfo);
    } else {  // PD 转换
        globalIpInfo.unlinkIpInfo = this->successLinkIP_;
    }
    try {
        std::map<uint64_t, std::vector<DeviceInfo>> currentLinkIpInfo{};
        std::string superPodId = "";
        for (const auto &nodeInfo : body["local"]) {
            for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                globalIpInfo.localDpInstanceIds.emplace_back(dpGroupInfo["dp_inst_id"]);
            }
        }
        for (const auto &peerInfo : body["peers"]) {
            for (const auto &nodeInfo : peerInfo) {
                if (nodeInfo.contains("super_pod_id")) {
                    superPodId = nodeInfo["super_pod_id"];
                }
                auto ret = ReverseDpInstId(nodeInfo["dp_inst_list"][0]["dp_inst_id"]);
                uint32_t instanceId = ret.first;
                spSize = nodeInfo.value("sp_size", 1);
                cpSize = nodeInfo.value("cp_size", 1);
                globalIpInfo.spInfo[instanceId] = spSize;
                globalIpInfo.cpInfo[instanceId] = cpSize;
                instanceIdToServerIp_[instanceId] = nodeInfo["server_ip"];
                for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                    auto dpInstanceId = dpGroupInfo["dp_inst_id"];
                    if (globalIpInfo.hostIpInfo.count(dpInstanceId) != 0) {
                        globalIpInfo.hostIpInfo[dpInstanceId].emplace_back(nodeInfo["host_ip"]);
                    } else {
                        globalIpInfo.hostIpInfo[dpInstanceId] = {nodeInfo["host_ip"]};
                    }
                    if (nodeInfo.contains("super_pod_id")) {
                        globalIpInfo.superPodIdInfo[dpInstanceId] = nodeInfo["super_pod_id"];
                    }
                    remoteNodeLinkStatus_[dpInstanceId] = {"None", false};
                    std::vector<DeviceInfo> linkDeviceIp;
                    for (const auto &deviceInfo : dpGroupInfo["device"]) {
                        DeviceInfo device;
                        device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                        device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                        if (deviceInfo.contains("super_device_id")) {
                            device.superDeviceId = std::stoi(deviceInfo["super_device_id"].get<std::string>());
                        }
                        linkDeviceIp.emplace_back(device);
                    }
                    if (currentLinkIpInfo.find(dpInstanceId) != currentLinkIpInfo.end()) {
                        auto &existingDevices = currentLinkIpInfo[dpInstanceId];
                        existingDevices.insert(existingDevices.end(), linkDeviceIp.begin(), linkDeviceIp.end());
                    } else {
                        currentLinkIpInfo.insert({dpInstanceId, linkDeviceIp});
                    }
                }
            }
        }
        UpdateIpInfo(globalIpInfo, currentLinkIpInfo, superPodId);
        return true;
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred : " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Unknown error occurred.");
        return false;
    }
}

bool DmiRole::UpdatePDNotSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo) {
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = false;
    std::string superPodId = "";
    std::map<uint64_t, std::vector<DeviceInfo>> currentLinkIpInfo{};
    std::map<uint64_t, std::vector<std::string>> currentLinkHostIpInfo{};
    try {
        for (const auto &serverInfo : body["peers"]) {
            uint32_t instanceId = serverInfo["id"];
            globalIpInfo.localHostIpList.emplace_back(body["local"]["host_ip"]);
            globalIpInfo.hostIpInfo[instanceId] = {serverInfo["host_ip"]};
            instanceIdToServerIp_[instanceId] = serverInfo["server_ip"];
            std::vector<DeviceInfo> linkDeviceIp;
            for (const auto &deviceInfo : serverInfo["device"]) {
                DeviceInfo device;
                device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                linkDeviceIp.emplace_back(device);
            }
            currentLinkIpInfo.insert({instanceId, linkDeviceIp});
            currentLinkHostIpInfo.insert({instanceId, {serverInfo["host_ip"]}});
        }
        if (body["local"].contains("p_percentage")) {
            globalIpInfo.flexPrefillPercentage = body["local"]["p_percentage"];
            FlexPPercentageProcessor::GetInstance().SetPdRoleFlexPPercentage(globalIpInfo.flexPrefillPercentage);
        }
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred. " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Unknown error occurred.");
        return false;
    }

    this->UpdateIpInfo(globalIpInfo, currentLinkIpInfo, superPodId);
    this->UpdateHostIpInfo(globalIpInfo, currentLinkHostIpInfo);
    return true;
}

// v2 interface
bool DmiRole::UpdatePDNotSwitchInfoV2(const std::string &roleName, const ordered_json &body,
                                      GlobalIpInfo &globalIpInfo) {
    std::lock_guard<std::mutex> lock(mtx_);
    globalIpInfo.role = roleName;
    globalIpInfo.needSwitch = false;
    std::string superPodId = "";
    (void)roleName;  // 显式使用一次roleName，以消除编译器警告（编译器误报）

    // 首先读"peers"字段的值存在currentLinkIpInfo中
    std::map<uint64_t, std::vector<DeviceInfo>> currentLinkIpInfo{};
    std::map<uint64_t, std::vector<std::string>> currentLinkHostIpInfo{};
    try {
        for (const auto &nodeInfo : body["local"]) {
            for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                globalIpInfo.localDpInstanceIds.emplace_back(dpGroupInfo["dp_inst_id"]);
            }
        }
        for (const auto &peerInfo : body["peers"]) {
            for (const auto &nodeInfo : peerInfo) {
                auto ret = ReverseDpInstId(nodeInfo["dp_inst_list"][0]["dp_inst_id"]);
                uint32_t instanceId = ret.first;
                spSize = nodeInfo.value("sp_size", 1);
                cpSize = nodeInfo.value("cp_size", 1);
                globalIpInfo.spInfo[instanceId] = spSize;
                globalIpInfo.cpInfo[instanceId] = cpSize;
                instanceIdToServerIp_[instanceId] = nodeInfo["server_ip"];

                for (const auto &dpGroupInfo : nodeInfo["dp_inst_list"]) {
                    std::vector<DeviceInfo> linkDeviceIp;
                    auto dpInstanceId = dpGroupInfo["dp_inst_id"];
                    if (nodeInfo.contains("super_pod_id")) {
                        superPodId = nodeInfo["super_pod_id"];
                        globalIpInfo.superPodIdInfo[dpInstanceId] = superPodId;
                    }
                    for (const auto &deviceInfo : dpGroupInfo["device"]) {
                        DeviceInfo device;
                        device.deviceIp = deviceInfo["device_ip"].get<std::string>();
                        device.devicePhysicalId = std::stoi(deviceInfo["device_id"].get<std::string>());
                        if (deviceInfo.contains("super_device_id")) {
                            device.superDeviceId = std::stoi(deviceInfo["super_device_id"].get<std::string>());
                        }
                        linkDeviceIp.emplace_back(device);
                    }
                    if (currentLinkIpInfo.find(dpInstanceId) != currentLinkIpInfo.end()) {
                        // 如果dpInstanceId已存在，则合并设备信息
                        auto &existingDevices = currentLinkIpInfo[dpInstanceId];
                        existingDevices.insert(existingDevices.end(), linkDeviceIp.begin(), linkDeviceIp.end());
                    } else {
                        currentLinkIpInfo.insert({dpInstanceId, linkDeviceIp});
                    }
                    if (currentLinkHostIpInfo.find(dpInstanceId) != currentLinkHostIpInfo.end()) {
                        currentLinkHostIpInfo[dpInstanceId].emplace_back(nodeInfo["host_ip"]);
                    } else {
                        currentLinkHostIpInfo[dpInstanceId] = {nodeInfo["host_ip"]};
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Error occurred : " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "Unknown error occurred.");
        return false;
    }
    this->UpdateIpInfo(globalIpInfo, currentLinkIpInfo, superPodId);
    this->UpdateHostIpInfo(globalIpInfo, currentLinkHostIpInfo);
    return true;
}

void DmiRole::UpdateIpInfo(GlobalIpInfo &globalIpInfo, std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo,
                           std::string &superPodId) {
    // Process all unlinking operations
    ProcessAllUnlinks(globalIpInfo, currentLinkIpInfo, superPodId);

    // Process new links to establish
    ProcessNewLinks(globalIpInfo, currentLinkIpInfo);

    // Clean up linking links that are no longer needed
    CleanupLinkingLinks(currentLinkIpInfo);

    // Clean up remote node status
    CleanupRemoteNodeStatus();

    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "linkingLinkIP_ size is " << std::to_string(this->linkingLinkIP_.size()));
}

void DmiRole::ProcessAllUnlinks(GlobalIpInfo &globalIpInfo,
                                const std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo,
                                const std::string &superPodId) {
    // The unlink item mathematical operation is as below:
    // this->successLinkIP_ - currentLinkIpInfo = unlink
    for (auto it = this->successLinkIP_.cbegin(); it != this->successLinkIP_.cend();) {
        auto key = it->first;
        if (currentLinkIpInfo.find(key) == currentLinkIpInfo.cend()) {
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                      "Unlink successLinkIP_ key: " << key << " value: " << this->successLinkIP_[key].size());
            globalIpInfo.unlinkIpInfo[key] = this->successLinkIP_[key];
            it = this->successLinkIP_.erase(it);
            if (superPodId != "") {
                globalIpInfo.superPodIdInfo[key] = superPodId;
            }
            globalIpInfo.cpInfo[key] = cpSize;
            globalIpInfo.spInfo[key] = spSize;
        } else {
            ++it;
        }
    }

    // links being established also need to check if they need to be
    // disconnected this->linkingLinkIP_ - currentLinkIpInfo = unlink
    for (auto it = this->linkingLinkIP_.cbegin(); it != this->linkingLinkIP_.cend(); ++it) {
        auto key = it->first;
        if (currentLinkIpInfo.find(key) == currentLinkIpInfo.cend()) {
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                      "Unlink linkingLinkIP_ key: " << key << " value: " << this->linkingLinkIP_[key].size());
            globalIpInfo.unlinkIpInfo[key] = this->linkingLinkIP_[key];
            if (superPodId != "") {
                globalIpInfo.superPodIdInfo[key] = superPodId;
            }
            globalIpInfo.cpInfo[key] = cpSize;
            globalIpInfo.spInfo[key] = spSize;
        }
    }
}

void DmiRole::ProcessNewLinks(GlobalIpInfo &globalIpInfo,
                              const std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo) {
    // The link item mathematical operation is as below:
    // currentLinkIpInfo - this->successLinkIP_ - this->linkingLinkIP_ = link
    for (auto it = currentLinkIpInfo.cbegin(); it != currentLinkIpInfo.cend(); ++it) {
        auto key = it->first;
        if (this->successLinkIP_.find(key) == this->successLinkIP_.cend()) {
            auto &existingDevices = globalIpInfo.linkIpInfo[key];
            existingDevices.insert(existingDevices.end(), it->second.begin(), it->second.end());
            auto &existingLinkIP = this->linkingLinkIP_[key];
            existingLinkIP.insert(existingLinkIP.end(), it->second.begin(), it->second.end());
        }
    }
}

void DmiRole::CleanupLinkingLinks(const std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo) {
    // Remove unlinked entries from linkingLinkIP_ after processing,
    // put it after link process.
    auto it = this->linkingLinkIP_.begin();
    while (it != this->linkingLinkIP_.end()) {
        auto key = it->first;
        if (currentLinkIpInfo.find(key) == currentLinkIpInfo.cend()) {
            it = this->linkingLinkIP_.erase(it);
        } else {
            ++it;
        }
    }
}

void DmiRole::CleanupRemoteNodeStatus() {
    // Clean up remoteNodeLinkStatus_ for instances that are no longer in
    // success or linking
    auto statusIt = this->remoteNodeLinkStatus_.begin();
    while (statusIt != this->remoteNodeLinkStatus_.end()) {
        uint64_t instanceId = statusIt->first;
        bool stillExists = (this->successLinkIP_.find(instanceId) != this->successLinkIP_.end()) ||
                           (this->linkingLinkIP_.find(instanceId) != this->linkingLinkIP_.end());
        if (!stillExists) {
            statusIt = this->remoteNodeLinkStatus_.erase(statusIt);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Cleaned up remoteNodeLinkStatus_ for instanceId: " << instanceId);
        } else {
            ++statusIt;
        }
    }
}

void DmiRole::UpdateHostIpInfo(GlobalIpInfo &globalIpInfo,
                               std::map<uint64_t, std::vector<std::string>> &currentLinkHostIpInfo) {
    // The unlink item mathematical operation is as below:
    // this->successHostIP_ - currentLinkHostIpInfo = unlink
    for (auto it = this->successHostIP_.cbegin(); it != this->successHostIP_.cend(); ++it) {
        auto key = it->first;
        if (currentLinkHostIpInfo.find(key) == currentLinkHostIpInfo.cend()) {
            globalIpInfo.unlinkHostIpInfo[key] = this->successHostIP_[key];
        }
    }

    // this->linkingHostIP_ - currentLinkHostIpInfo = unlink
    for (auto it = this->linkingHostIP_.cbegin(); it != this->linkingHostIP_.cend(); ++it) {
        auto key = it->first;
        if (currentLinkHostIpInfo.find(key) == currentLinkHostIpInfo.cend()) {
            globalIpInfo.unlinkHostIpInfo[key] = this->linkingHostIP_[key];
        }
    }

    // The link item mathematical operation is as below:
    // currentLinkHostIpInfo - this->successHostIP_ - this->linkingHostIP_ =
    // link
    for (auto it = currentLinkHostIpInfo.cbegin(); it != currentLinkHostIpInfo.cend(); ++it) {
        auto key = it->first;
        if (this->successHostIP_.find(key) == this->successHostIP_.cend() &&
            this->linkingHostIP_.find(key) == this->linkingHostIP_.cend()) {
            globalIpInfo.hostIpInfo[key] = currentLinkHostIpInfo[key];
            this->linkingHostIP_[key] = currentLinkHostIpInfo[key];
        }
    }

    // Remove unlinked entries from linkingHostIP_ after processing,
    // put it after link process.
    auto it = this->linkingHostIP_.begin();
    while (it != this->linkingHostIP_.end()) {
        auto key = it->first;
        if (currentLinkHostIpInfo.find(key) == currentLinkHostIpInfo.cend()) {
            it = this->linkingHostIP_.erase(it);
        } else {
            ++it;
        }
    }

    auto successIt = this->successHostIP_.begin();
    while (successIt != this->successHostIP_.end()) {
        auto key = successIt->first;
        if (currentLinkHostIpInfo.find(key) == currentLinkHostIpInfo.cend()) {
            successIt = this->successHostIP_.erase(successIt);
        } else {
            ++successIt;
        }
    }
}

void DmiRole::HandlePDRoleV1(const ReqCtxPtr &ctx, const std::string &roleName) {
    ordered_json body;
    GlobalIpInfo globalIpInfo;
    if (!PDParseRequestBodyToJson(ctx, body) || !JsonParse::CheckPDRoleReqJson(body)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, JSON_PARSE_ERROR),
                   "[DmiRole::HandlePDRole] Req body converts to json fail. "
                   "Reset to previous node status.");
        HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::UnprocessableContent_422,
                                           HttpRestResource::WrapperJson("Req body converts to json fail. "
                                                                         "Reset to previous node status.",
                                                                         "Input validation error"));
        return;
    }

    std::string preRole = GetInferInstance()->GetPDRole();
    if (!UpdatePDInfo(roleName, preRole, body, globalIpInfo)) {
        HttpRestResource::ResponseJsonBody(
            ctx, httplib::StatusCode::ServiceUnavailable_503,
            HttpRestResource::WrapperJson("Update pd info failed. Reset to previous node status.",
                                          "Service Unavailable"));
        return;
    }
    ProcessPDRoleSwitch(ctx, roleName, globalIpInfo);
}

void DmiRole::HandlePDRoleV2(const ReqCtxPtr &ctx, const std::string &roleName) {
    ordered_json body;
    GlobalIpInfo globalIpInfo;
    try {
        if (!PDParseRequestBodyToJson(ctx, body) || !JsonParse::CheckPDRoleV2ReqJson(body)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_PREFIX_CACHE, STATUS_WARNING),
                       "[DmiRole::HandlePDRole] Req body converts to json fail. Reset "
                       "to previous node status. ");
            HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::UnprocessableContent_422,
                                               HttpRestResource::WrapperJson("Req body converts to json fail. "
                                                                             "Reset to previous node status.",
                                                                             "Input validation error"));
            return;
        }

        std::string preRole = GetInferInstance()->GetPDRole();
        if (!UpdatePDInfoV2(roleName, preRole, body,
                            globalIpInfo)) {  // Json -> globalIpInfo
            HttpRestResource::ResponseJsonBody(
                ctx, httplib::StatusCode::ServiceUnavailable_503,
                HttpRestResource::WrapperJson("Parse req json failed. Reset to previous node status.",
                                              "Service Unavailable"));
            return;
        }
        ProcessPDRoleSwitch(ctx, roleName, globalIpInfo);
    } catch (const std::exception &e) {
        // 捕获标准异常
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                   "Standard exception caught: " << e.what());
        return;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                   "Unknown exception caught.");
        return;
    }
}

void DmiRole::ProcessPDRoleSwitch(const ReqCtxPtr &ctx, const std::string &roleName, GlobalIpInfo &globalIpInfo) {
    // Allow concurrent processing but protect shared resources with locks
    // Each request can proceed independently but must coordinate on shared
    // state
    {
        std::lock_guard<std::mutex> lock(mtx_);
        // Critical section: update global state while holding lock
        GetInferInstance()->SetPDRoleStatus(PDRoleStatus::SWITCHING);
        GetInferInstance()->UpdatePDRole(roleName);
    }
    OrderedJson jsonObj;
    jsonObj["result"] = "ok";

    if (roleName == "prefill") {
        keepAlive.store(false);  // prefill role does not need to keep alive
    }

    // Execute link task only when there are actual link/unlink operations
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!globalIpInfo.linkIpInfo.empty() || !globalIpInfo.unlinkIpInfo.empty()) {
            auto task = [this, globalIpInfo = globalIpInfo]() { this->ExecuteLinkTask(globalIpInfo); };
            taskQueue_.Push(std::move(task));
        }
    }
    // Always set response for successful role switch
    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, jsonObj.dump());
}

void DmiRole::ModifyPullKVFailId(const uint32_t &instanceId) {
    std::lock_guard<std::mutex> lock(mtx_);
    this->successLinkIP_.erase(instanceId);
    this->successHostIP_.erase(instanceId);
    remoteNodeLinkStatus_[instanceId].first = "failed : pull kv failed.";
    this->abnormalLink_ = true;
}

const std::map<uint64_t, std::vector<DeviceInfo>> &DmiRole::GetSuccessLinkIp() { return successLinkIP_; }

const std::map<uint64_t, std::vector<std::string>> &DmiRole::GetSuccessHostIp() { return successHostIP_; }

const std::map<uint64_t, std::vector<DeviceInfo>> &DmiRole::GetLinkingLinkIp() { return linkingLinkIP_; }

const std::map<uint64_t, std::vector<std::string>> &DmiRole::GetLinkingHostIp() { return linkingHostIP_; }

const std::map<uint64_t, std::pair<std::string, bool>> &DmiRole::GetRemoteNodeLinkStatus() {
    std::lock_guard<std::mutex> lock(mtx_);
    return this->remoteNodeLinkStatus_;
}

std::map<uint64_t, std::pair<std::string, bool>> GetInstanceStatus(
    const std::map<uint64_t, std::pair<std::string, bool>> &dpInstanceStatusMap) {
    std::map<uint64_t, std::pair<std::string, bool>> instanceStatus;
    // 遍历所有dpInstance的状态
    for (const auto &entry : dpInstanceStatusMap) {
        auto dpInstanceId = entry.first;
        bool dpInstanceStatus = entry.second.second;

        // 反推instanceId
        auto instanceId = dpInstanceId / 10000;
        // 如果该instance状态还没有被设置，先设置为true
        if (instanceStatus.find(instanceId) == instanceStatus.end()) {
            instanceStatus[instanceId] = {"ok", true};
        }

        if (!dpInstanceStatus) {
            instanceStatus[instanceId] = {"dp instance id : " + std::to_string(dpInstanceId) + entry.second.first,
                                          false};
        }
    }

    return instanceStatus;
}

std::map<uint64_t, std::pair<std::string, bool>> DmiRole::GetRemoteNodeLinkStatusV2() {
    std::lock_guard<std::mutex> lock(mtx_);
    // summarize status for each dp group -> instance status
    auto ret = GetInstanceStatus(this->remoteNodeLinkStatus_);
    return ret;
}

const std::map<uint32_t, std::string> &DmiRole::GetInstanceIdToServerIp() {
    std::lock_guard<std::mutex> lock(mtx_);
    return this->instanceIdToServerIp_;
}

const uint32_t &DmiRole::GetLocalInstanceId() {
    std::lock_guard<std::mutex> lock(mtx_);
    return this->localInstanceId_;
}

// Unhealthy P node: all links ought to be linked, so it is always healthy
// Unhealthy D node: any link is abnormal and no success linked peers
bool DmiRole::IsHealthy() {
    std::lock_guard<std::mutex> lock(mtx_);
    return !(this->abnormalLink_ && this->successLinkIP_.size() == 0);
}
}  // namespace mindie_llm
