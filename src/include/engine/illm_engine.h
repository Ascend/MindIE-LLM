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

#ifndef LLM_ENGINE_INTERFACE_H
#define LLM_ENGINE_INTERFACE_H

#include <functional>
#include <memory>
#include "basic_types.h"
#include "executor/executor_interface.h"
#include "request_response/request.h"
#include "request_response/response.h"
#include "config_info.h"
#include "metric.h"
#include "lora/loraops_mixin.h"

namespace mindie_llm {
using ForwardRespToManagerCall = std::function<void(ResponseSPtr response)>;

class ILlmEngine : public LoraOpsMixin {
public:
    virtual ~ILlmEngine() = default;

    virtual void InitProcessGroup(const std::vector<NodeInfo> &nodeInfos, std::string &processGroupMasterIP,
        uint32_t processGroupMasterPort) = 0;

    virtual void StartEngineThread() = 0;

    virtual bool AddRequest(RequestSPtr request) = 0;

    virtual void AbortRequests(std::unordered_set<RequestId> &requestIds) = 0;

    virtual void ReleaseKvCache(std::unordered_set<RequestId> &requestIds) = 0;

    virtual void Stop() = 0;

    virtual void PauseScheduling() = 0;

    virtual void ResumeScheduling() = 0;

    virtual void ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) = 0;

    virtual EngineMetric CollectEngineMetric(size_t localDPRank = 0) = 0;

    /// @brief 收集所有 DP Rank 的聚合指标
    /// @return EngineMetric 所有 DP Rank 的聚合指标
    virtual EngineMetric CollectAllDpEngineMetric() = 0;

    virtual void SetPrefillPercentage(uint32_t prefillPercentage) = 0;
};
using LlmEnginePtr = std::unique_ptr<ILlmEngine>;

LlmEnginePtr MakeLlmEngine(SchedulerConfig schedulerConfig, std::vector<IExecutorSPtr> executors,
    ForwardRespToManagerCall cb, Role pdRole);

} // namespace mindie_llm

#endif
