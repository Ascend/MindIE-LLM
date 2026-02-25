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

#include "llm_manager_v2.h"
#include "llm_manager_impl.h"
#include "infer_instances.h"

namespace mindie_llm {
constexpr uint32_t MAX_MODEL_INSTANCE_ID = 10;
LlmManagerV2::LlmManagerV2(const std::string &llmConfigPath, GetRequestsCallbackV2 getRequests,
    SendResponsesCallbackV2 sendResponse, ControlSignalCallbackV2 controlCallback,
    LlmManagerStatsCallback statusCallback, SendStatusResponseCallbackV2 statusResponseCallback,
    std::map<std::string, std::string> ipInfo)
{
    impl_ = std::make_shared<LlmManagerImpl>(llmConfigPath, getRequests, sendResponse, controlCallback, statusCallback,
                                             statusResponseCallback, ipInfo);
}

void LlmManagerV2::Shutdown()
{
    if (impl_ == nullptr) {
        return;
    }
    impl_->Shutdown();
}

uint32_t LlmManagerV2::GetMaxPositionEmbeddings() const
{
    if (impl_ == nullptr) {
        throw std::runtime_error("LlmManager impl_ is a nullptr!");
    }
    return impl_->GetMaxPositionEmbeddings();
}

std::map<std::string, std::string> LlmManagerV2::GetModelParams() const
{
    return LlmManagerImpl::GetModelParams();
}


bool LlmManagerV2::UpdateEngineInfo(RequestSPtr &runtimeRequest, bool isForceRelease)
{
    if (impl_ == nullptr) {
        return false;
    }
    if (runtimeRequest == nullptr) {
        return false;
    }
    if (!impl_->UpdateEngineInfo(runtimeRequest, isForceRelease)) {
        return false;
    }
    return true;
}

bool LlmManagerV2::UpdateFlexSwitchInfo(const std::shared_ptr<FlexSwitchInfo> flexSwitchInfo)
{
    if (impl_ == nullptr || flexSwitchInfo == nullptr) {
        return false;
    }
    impl_->UpdateFlexSwitchInfo(flexSwitchInfo);
    return true;
}

Status LlmManagerV2::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    if (modelInstanceId > MAX_MODEL_INSTANCE_ID) {
        return Status(Error::Code::INVALID_ARG, "modelInstanceId is invalid in LlmManagerV2::Init");
    }
    if (npuDeviceIds.empty()) {
        return Status(Error::Code::INVALID_ARG, "npuDeviceIds is empty");
    }
    Status ret = impl_->Init(modelInstanceId, npuDeviceIds);
    if (!ret.IsOk()) {
        return ret;
    }
    if (!(impl_->GetDmiInferEnabled() && impl_->GetMultiNodesInferEnabled())) {
        impl_->Step();
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmManagerV2::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds,
    std::map<std::string, std::string> extendInfo)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    if (modelInstanceId > MAX_MODEL_INSTANCE_ID) {
        return Status(Error::Code::INVALID_ARG, "modelInstanceId is invalid in LlmManagerV2::Init");
    }
    if (npuDeviceIds.empty()) {
        return Status(Error::Code::INVALID_ARG, "npuDeviceIds is empty");
    }
    Status ret = impl_->Init(modelInstanceId, npuDeviceIds, extendInfo);
    if (!ret.IsOk()) {
        return ret;
    }
    if (!(impl_->GetDmiInferEnabled() && impl_->GetMultiNodesInferEnabled())) {
        impl_->Step();
    }
    return Status(Error::Code::OK, "Success");
}

Status LlmManagerV2::InitModelForMultiPd(std::map<std::string, std::string> pdInfo, uint32_t modelInstanceId)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    Status ret = impl_->InitModelForMultiPd(pdInfo, modelInstanceId);
    if (!ret.IsOk()) {
        return ret;
    }
    impl_->Step();
    return Status(Error::Code::OK, "Success");
}

// LlmManagerV2 new api
Status LlmManagerV2::AddRequest(RequestSPtr request)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    if (request != nullptr) {
        return impl_->ProcessRequests(request);
    } else {
        return impl_->ProcessRequests();
    }
}

Status LlmManagerV2::ControlRequest(const RequestIdNew &requestId, OperationV2 operation)
{
    std::optional<SendResponsesCallbackV2> serverResponseCallback = InferInstance::GetCallbackMap().Get(requestId);
    if (operation == OperationV2::STOP && !serverResponseCallback.has_value()) {
        return Status(Error::Code::ERROR, "Invalid RequestId");
    }
    impl_->ControlRequest(requestId, operation);
    return Status(Error::Code::OK, "Success");
}

EngineMetric LlmManagerV2::CollectEngineMetric(size_t localDPRank) { return impl_->CollectEngineMetric(localDPRank); }

Status LlmManagerV2::HandleLora(const LoraOperation loraOperation, std::vector<LoraParamSPtr> &loraInfo)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    Status ret = impl_->HandleLoraImpl(loraOperation, loraInfo);
    return ret;
}

bool LlmManagerV2::ExecuteRecoverCommand(RecoverCommandInfo &commandInfo)
{
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->ExecuteRecoverCommand(commandInfo);
}

bool LlmManagerV2::IsLlmEngineReady() const
{
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->IsLlmEngineReady();
}

LlmManagerV2::~LlmManagerV2() = default;
} // namespace mindie_llm