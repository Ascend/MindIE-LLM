/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "llm_manager_v2/llm_manager_v2.h"
#include "src/llm_manager_v2/include/impl/llm_manager_impl.h"
#include "llm_manager_adapter.h"
#include "config_manager.h"
#include "config_manager_impl.h"
#include "llm_manager/llm_manager.h"

namespace mindie_llm {
constexpr uint32_t MAX_MODEL_INSTANCE_ID = 10;
LlmManager::LlmManager(const std::string &llmConfigPath, GetRequestsCallback getRequest,
                       SendResponsesCallback sendResponse, ControlSignalCallback controlCallback,
                       LlmManagerStatsCallback statusCallback, SendStatusResponseCallback statusResponseCallback,
                       std::map<std::string, std::string> ipInfo)
{
    ConfigManager::CreateInstance(llmConfigPath);
    auto getRequestV2 = std::bind(AdaptGetRequestV1ToV2, getRequest);
    auto sendResponseV1 = std::bind(AdaptSendResponseV2ToV1, sendResponse, std::placeholders::_1);
    auto controlSignalCallbackV2 = std::bind(AdaptControlSignalCallbackV1ToV2, controlCallback);
    auto statusResponseCallback2to1 = std::bind(AdaptStatusResponseCallbackV2ToV1, statusResponseCallback,
                                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    impl_ = std::make_shared<LlmManagerImpl>(llmConfigPath, getRequestV2, sendResponseV1, controlSignalCallbackV2,
                                             statusCallback, statusResponseCallback2to1, ipInfo);
}

void LlmManager::Shutdown()
{
    if (impl_ == nullptr) {
        return;
    }
    impl_->Shutdown();
}

uint32_t LlmManager::GetMaxPositionEmbeddings() const
{
    if (impl_ == nullptr) {
        throw std::runtime_error("LlmManager impl_ is a nullptr!");
    }
    return impl_->GetMaxPositionEmbeddings();
}

std::map<std::string, std::string> LlmManager::GetModelParams() const
{
    return LlmManagerImpl::GetModelParams();
}

bool LlmManager::UpdateEngineInfo(std::shared_ptr<mindie_llm::InferRequest> &runtimeRequest, bool isForceRelease)
{
    if (impl_ == nullptr) {
        return false;
    }
    if (runtimeRequest == nullptr) {
        return false;
    }
    RequestSPtr runtimeRequestNew = std::make_shared<Request>();
    if (!impl_->UpdateEngineInfo(runtimeRequestNew, isForceRelease)) {
        return false;
    }
    return true;
}

Status LlmManager::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    if (modelInstanceId > MAX_MODEL_INSTANCE_ID) {
        return Status(Error::Code::INVALID_ARG, "modelInstanceId is invalid in LlmManager::Init");
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

Status LlmManager::Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds,
                        std::map<std::string, std::string> extendInfo)
{
    if (impl_ == nullptr) {
        return Status(Error::Code::INVALID_ARG, "llmImpl is null ptr");
    }
    if (modelInstanceId > MAX_MODEL_INSTANCE_ID) {
        return Status(Error::Code::INVALID_ARG, "modelInstanceId is invalid in LlmManager::Init");
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

Status LlmManager::InitModelForMultiPd(std::map<std::string, std::string> pdInfo, uint32_t modelInstanceId)
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

bool LlmManager::ExecuteRecoverCommand(RecoverCommandInfo &commandInfo)
{
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->ExecuteRecoverCommand(commandInfo);
}

LlmManager::~LlmManager() = default;
} // namespace mindie_llm