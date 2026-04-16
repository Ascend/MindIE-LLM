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

#ifndef DMI_ROLE_H
#define DMI_ROLE_H

#include <map>
#include <unordered_set>
#include <vector>

#include "blocking_queue.h"
#include "common_util.h"
#include "config_manager.h"
#include "global_ip_info.h"
#include "http_rest_resource.h"
#include "httplib.h"
#include "infer_instances.h"

using ordered_json = nlohmann::ordered_json;

namespace mindie_llm {
constexpr uint32_t DEFAULT_PD_ROLE_FLEX_P_PERCENTAGE = 50;  // pdRole flex p_percentage 默认值50
class FlexPPercentageProcessor {
   public:
    static FlexPPercentageProcessor &GetInstance() {
        static FlexPPercentageProcessor instance;
        return instance;
    }
    uint32_t GetPdRoleFlexPPercentage() const { return this->pdRoleFlexPPercentage; }
    void SetPdRoleFlexPPercentage(const uint32_t pPercentage) { this->pdRoleFlexPPercentage = pPercentage; }

   private:
    FlexPPercentageProcessor() = default;
    ~FlexPPercentageProcessor() = default;
    uint32_t pdRoleFlexPPercentage{DEFAULT_PD_ROLE_FLEX_P_PERCENTAGE};
};

class DmiRole {
   public:
    DmiRole();
    ~DmiRole();
    void HandlePDRoleV1(const ReqCtxPtr &ctx, const std::string &roleName);
    void HandlePDRoleV2(const ReqCtxPtr &ctx, const std::string &roleName);
    const std::map<uint64_t, std::vector<DeviceInfo>> &GetSuccessLinkIp();
    const std::map<uint64_t, std::vector<std::string>> &GetSuccessHostIp();
    const std::map<uint64_t, std::vector<DeviceInfo>> &GetLinkingLinkIp();
    const std::map<uint64_t, std::vector<std::string>> &GetLinkingHostIp();
    const std::map<uint64_t, std::pair<std::string, bool>> &GetRemoteNodeLinkStatus();
    std::map<uint64_t, std::pair<std::string, bool>> GetRemoteNodeLinkStatusV2();
    const std::map<uint32_t, std::string> &GetInstanceIdToServerIp();
    const uint32_t &GetLocalInstanceId();
    void ModifyPullKVFailId(const uint32_t &instanceId);
    void RunTaskThread();
    void StopCurrentTask();
    void RunQueryThread();
    void QueryLinkStatus();
    bool IsHealthy();
    static std::shared_ptr<DmiRole> GetInstance();

   private:
    bool UpdatePDInfo(const std::string &roleName, const std::string &preRole, const ordered_json &body,
                      GlobalIpInfo &globalIpInfo);
    bool UpdatePDInfoV2(const std::string &roleName, const std::string &preRole, const ordered_json &body,
                        GlobalIpInfo &globalIpInfo);
    bool UpdatePDSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
                            bool needInit);
    bool UpdatePDSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
                              bool needInit);
    bool UpdatePDNotSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo);
    bool UpdatePDNotSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo);
    bool PDParseRequestBodyToJson(const ReqCtxPtr &reqCtx, ordered_json &body) const noexcept;
    void ProcessInitInfo(const ordered_json &body, GlobalIpInfo &globalIpInfo);
    void ProcessInitInfoV2(const ordered_json &body, GlobalIpInfo &globalIpInfo);
    void UpdateIpInfo(GlobalIpInfo &globalIpInfo, std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo,
                      std::string &superPodId);
    void ProcessAllUnlinks(GlobalIpInfo &globalIpInfo,
                           const std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo,
                           const std::string &superPodId);
    void ProcessNewLinks(GlobalIpInfo &globalIpInfo,
                         const std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo);
    void CleanupLinkingLinks(const std::map<uint64_t, std::vector<DeviceInfo>> &currentLinkIpInfo);
    void CleanupRemoteNodeStatus();
    void UpdateHostIpInfo(GlobalIpInfo &globalIpInfo,
                          std::map<uint64_t, std::vector<std::string>> &currentLinkHostIpInfo);
    void ProcessPDRoleSwitch(const ReqCtxPtr &ctx, const std::string &roleName, GlobalIpInfo &globalIpInfo);
    template <typename T>
    void ProcessFailedLinks(const T &failedLinks);
    template <typename T>
    void ProcessSuccessfulLinks(const T &successLinks);
    void CheckAllLinksCompleted();
    void TaskThread();
    void ExecuteLinkTask(GlobalIpInfo globalIpInfo);

    // already successful linked server ip address
    std::map<uint64_t, std::vector<DeviceInfo>> successLinkIP_;
    std::map<uint64_t, std::vector<std::string>> successHostIP_;
    // currently linking server ip address
    std::map<uint64_t, std::vector<DeviceInfo>> linkingLinkIP_;
    std::map<uint64_t, std::vector<std::string>> linkingHostIP_;

    std::vector<std::string> runningLinkIP_;
    std::vector<std::string> waitingLinkIP_;

    // [key] is <localDpInstanceId, remoteDpInstanceId> and [value] is
    // {linkStatus, isProcessed} [record remote node status]: key is instanceId,
    // [value] is {linkStatus, isProcessed}
    std::map<uint64_t, std::pair<std::string, bool>> remoteNodeLinkStatus_;
    // example: [10001, 0k] [10002, not ok] will cause [1: not ok]
    std::map<uint32_t, std::string> instanceIdToServerIp_;
    std::thread taskThread_;
    std::atomic<bool> taskTerminate_{false};
    std::atomic<bool> taskRunning_{false};
    BlockingQueue<std::function<void()>> taskQueue_;
    std::thread queryThread_;
    std::atomic<bool> queryTerminate_{false};
    std::mutex mtx_;
    // Attention: localInstanceId_ will be updated by [DmiRole::ProcessInitInfo]
    uint32_t localInstanceId_{0};
    // Attention: localDpInstanceIds_ will be updated by
    // [DmiRole::ProcessInitInfo]
    std::vector<uint64_t> localDpInstanceIds_;
    bool abnormalLink_{false};
    std::vector<uint64_t> dpInstanceList;
    // Flag to indicate if assignDmiRole has been called at least once
    bool assignedRole_{false};
    int32_t cpSize{1};
    int32_t spSize{1};
};
extern std::atomic<bool> keepAlive;
}  // namespace mindie_llm

#endif  // OCK_ENDPOINT_HTTP_HANDLER_H
