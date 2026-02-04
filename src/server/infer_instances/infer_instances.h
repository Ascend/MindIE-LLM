/**
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
/*
 * @brief Main responsibilities of this module
 *
 * 1. **Multi-manager orchestration & request dispatch**
 *    Maintain multiple LlmManager instances and route inference requests to the best target.
 *
 * 2. **Request–callback routing**
 *    Map RequestId to SendResponsesCallback for response delivery and lifecycle tracking.
 *
 * 3. **Engine metrics & status aggregation**
 *    Collect and aggregate throughput, queue sizes, cache blocks, and other runtime stats.
 *
 * 4. **PD role assignment & management**
 *    Assign and manage Prefill/Decode roles, including initialization and updates.
 */

#pragma once
#include <shared_mutex>
#include <queue>
#include <condition_variable>
#include "log.h"
#include "llm_manager_v2.h"
#include "status.h"
#include "pd_role.h"
#include "config_manager.h"
#include "concurrent_deque.h"
#include "concurrent_map.h"
#include "global_ip_info.h"
#include "request_response/response.h"
#include "request_response/request.h"
#include "data_type.h"

namespace mindie_llm {
#define CHECK_INITIALIZATION()                                                                                         \
    do {                                                                                                               \
        if (!started_.load()) {                                                                                        \
            return Status(Error::Code::ERROR, "Model instance has been finalized or not initialized.");                \
        }                                                                                                              \
        if (llmManagers_.empty()) {                                                                                    \
            return Status(Error::Code::ERROR, "llmInferEngine is not initialized!");                                   \
        }                                                                                                              \
    } while (0)
class InferInstance {
public:
    InferInstance() = default;

    static std::shared_ptr<InferInstance> GetInstance();

    // delete copy / move
    InferInstance(const InferInstance &) = delete;
    InferInstance &operator=(const InferInstance &) = delete;
    InferInstance(InferInstance &&) = delete;
    InferInstance &operator=(InferInstance &&) = delete;

    Status InitFromEndpointCall(const std::string &configPath = "");

    Status InitSingleInferInstance(std::map<std::string, std::string> modelConfig, uint32_t modelInstanceId);

    Status Finalize();

    Status Process(RequestSPtr request);

    Status ControlRequest(const RequestIdNew &requestId, OperationV2 operation);

    // NHM fault recover interface
    Status ControlInferInstance(mindie_llm::RecoverCommandInfo &info);

    Status CheckInferInstanceStarted(bool &isStarted);

    Status GetRequestBlockQuotas(uint64_t &remainBlocks, uint64_t &remainPrefillSlots, uint64_t &remainPrefillTokens,
                                 std::map<uint32_t, uint64_t> &dpRemainBlocks);

    Status GetNodeStatus(std::map<std::string, NodeHealthStatus> &slaveStatus);

    Status GetProcessingRequest(uint64_t &num);

    Status GetWaitingRequest(uint64_t &num);

    Status GetRunningRequest(uint64_t &num);

    Status GetSwappedRequest(uint64_t &num);

    Status GetCacheBlockNums(uint64_t &freeNpuBlockNums, uint64_t &freeCpuBlockNums, uint64_t &totalNpuBlockNums,
                             uint64_t &totalCpuBlockNums);

    Status GetRadixMatchNums(uint64_t &allRadixMatchNum, uint64_t &npuRadixMatchHitNum);

    Status GetCumulativePreemptCount(uint64_t &cumulativePreemptCount);

    Status GetThroughput(float &prefillThroughput, float &decodeThroughput);

    Status AssignDmiRole(GlobalIpInfo &globalIpInfo);

    Status ForcePRelease();

    Status GetBatchSchedulerMetrics(std::map<std::string, uint64_t> &batchSchedulerMetrics);

    std::string GetPDRole() const;

    PDRoleStatus GetPDRoleStatus() const;

    bool IsLlmEngineReady() const;
    
    void SetPDRoleStatus(PDRoleStatus status);

    void UpdatePDRole(const std::string &role);

    static ConcurrentMap<mindie_llm::RequestIdNew, SendResponsesCallbackV2>& GetCallbackMap() { return callbackMap; }

    Status HandleLora(const LoraOperation& loraOperation, std::vector<LoraParamSPtr>& loraInfo);

    static bool IsPaused() { return isPaused_.load(); }

private:
    Status InitPDNode(GlobalIpInfo &globalIpInfo);

    PDRole pdRole_ = PDRole::UNKNOWN;

    PDRoleStatus pdRoleStatus_ = PDRoleStatus::UNKNOWN;

    // Hold multiple LlmManager instances (one per model instance)
    std::vector<std::shared_ptr<mindie_llm::LlmManagerV2>> llmManagers_;

    std::map<uint32_t, uint64_t> dpRemainBlocks_;
    
    std::map<std::string, NodeHealthStatus> slavesStatus_{};

    std::atomic<bool> started_{false};

    inline static std::atomic<bool> isPaused_{false};

    std::string configPath_;

    static ConcurrentMap<mindie_llm::RequestIdNew, SendResponsesCallbackV2> callbackMap;
};
std::shared_ptr<InferInstance> GetInferInstance();
} // namespace mindie_llm
