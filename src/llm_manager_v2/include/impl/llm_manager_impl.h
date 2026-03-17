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

#ifndef MINDIE_LLM_LLMMANGER_IMPL_H
#define MINDIE_LLM_LLMMANGER_IMPL_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>

#include "request_response/callback.h"
#include "request_response/response.h"
#include "llm_manager_v2.h"
#include "basic_types.h"
#include "engine/illm_engine.h"
#include "metric.h"
#include "config_info.h"

namespace mindie_llm {
struct BlockNum {
    uint32_t cpuBlockNum;
    uint32_t npuBlockNum;
};

enum class FlexInstanceUpdateType {
    ONLY_UPDATE_P_PERCENTAGE = 0, // P_Percentage变更
    UPDATE_ALL = 1 // P_Percentage及其他信息变更
};

class LlmManagerImpl {
public:
    LlmManagerImpl(const std::string &llmConfigPath, GetRequestsCallbackV2 getRequests,
                   SendResponsesCallbackV2 handleResponse, ControlSignalCallbackV2 controlCallback,
                   LlmManagerStatsCallback statusCallback, SendStatusResponseCallbackV2 statusResponseCallback,
                   std::map<std::string, std::string> ipInfo = std::map<std::string, std::string>());

    uint32_t GetMaxPositionEmbeddings() const;

    void Shutdown();

    bool UpdateEngineInfo(RequestSPtr &runtimeRequest, bool isForceRelease);

    static std::map<std::string, std::string> GetModelParams();

    void Step();

    void Stop();

    Status Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds,
                std::map<std::string, std::string> extendInfo = {});

    Status InitModelForMultiPd(const std::map<std::string, std::string> pdInfo, uint32_t modelInstanceId);

    bool ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) const;

    bool GetMultiNodesInferEnabled() const;

    bool GetDmiInferEnabled() const;

    bool IsLlmEngineReady() const { return llmEngineReady_.load(std::memory_order_acquire); }
    
    Status ProcessRequests(RequestSPtr request);

    Status ProcessRequests();

    void ControlRequest();

    void ControlRequest(const RequestIdNew &requestId, OperationV2 operation);

    EngineMetric CollectEngineMetric(size_t localDPRank = 0);

    Status HandleLoraImpl(const LoraOperation loraOperation, std::vector<LoraParamSPtr> &loraInfo);

    bool UpdateFlexSwitchInfo(const std::shared_ptr<FlexSwitchInfo> flexSwitchInfo);
private:
    void ProcessStep();

    void ControlStep();

    void SendRuntimeStep();

    void SendRuntimeStatus();

    void SendJsonData(EngineMetric &engineMetric);

    bool SwitchPdRole(RequestSPtr &runtimeRequest);

    bool SetExecuteConfig(bool isForceRelease, std::map<std::string, std::string> &executeConfig,
                          RequestSPtr &runtimeRequest);

    Status LaunchLlmEngine(Role pdRole);

    Status RelaunchLlmEngine(int64_t roleIndex);

    Status ForwardRequest(RequestSPtr request);

    Status Finalize();

    Status FinalizeLlmEngine() const;

private:
    Status ProccessReqInputIds(RequestSPtr &request) const;

    Role GetRoleFromString(std::string &pdRole) const;

    void InitEngineDPProcessGroup(SchedulerConfig &schedulerConfig);

    BlockNum GetMinBlockNumFromExecutors();

    ThinkingConfig GetThinkingConfigFromExecutors();

    GetRequestsCallbackV2 getRequests_ = nullptr;

    SendResponsesCallbackV2 handleResponse_ = nullptr;

    ControlSignalCallbackV2 controlCallback_ = nullptr;

    LlmManagerStatsCallback statusCallback_ = nullptr;

    SendStatusResponseCallbackV2 statusResponseCallback_ = nullptr;

    std::map<uint64_t, uint64_t> dpRemainBlocks_ = {};

    uint64_t remainPrefillSlots_ = 0;

    uint64_t remainPrefill_ = 0;

    std::atomic<bool> shutdown_ = false;

    const std::string inferModeStandard{"standard"};

    bool multiNodesInferEnabled_ = false;
    bool isMaster_ = false;
    uint32_t maxPositionEmbeddings_;
    bool started_{false};
    std::string llmConfigPath_;
    std::thread processThread_;
    std::thread controlThread_;
    std::thread sendRuntimeThread_;
    Role pdRole_{Role::PnD};
    bool isDmiInfer_ = false;

    // Engine就绪标志：标识Engine调度线程已启动并可以接收请求
    std::atomic<bool> llmEngineReady_{false};

    // for init dp
    std::string homePath_;
    EngineConfig engineConfig_;
    std::map<std::string, std::string> ipInfo_;
    std::vector<std::map<std::string, std::string>> modelConfigs_;

    LlmEnginePtr llmEnginePtr_ = nullptr;
    std::vector<IExecutorSPtr> iExecutorSPtrs_;
    bool isFlexInitialized_{false};
};
} // namespace mindie_llm

#endif