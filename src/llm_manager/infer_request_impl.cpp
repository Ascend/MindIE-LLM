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

#include "infer_request_impl.h"

#include "system_log.h"
#include "check_utils.h"
using namespace mindie_llm;

namespace mindie_llm {
InferRequestImpl::InferRequestImpl(InferRequestId requestId) : requestId_(requestId)
{
}

void InferRequestImpl::SetTensor(const std::string& tensorName, TensorPtr &tensor)
{
    if (!CheckStringInputLength(tensorName, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The length of tensor name: " << tensorName << "is too long.";
        return;
    }
    inputs_[tensorName] = tensor;
}

Status InferRequestImpl::AddTensor(const std::string& tensorName, TensorPtr &tensor)
{
    if (!CheckStringInputLength(tensorName, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The length of tensor name: " << tensorName << "is too long in 'AddTensor'.";
        return Status(Error::Code::INVALID_ARG, "The length of tensor name: " + tensorName + " is too long");
    }
    if (tensor == nullptr) {
        return Status(Error::Code::INVALID_ARG, "tensor is nullptr in 'AddTensor' parameter");
    }
    const auto &pr = inputs_.insert(std::make_pair(tensorName, tensor));
    if (!pr.second) {
        return Status(Error::Code::INVALID_ARG, "input '" + tensorName + "' already exists in request");
    }
    return Status(Error::Code::OK, "Success");
}

Status InferRequestImpl::GetTensorByName(const std::string& tensorName, TensorPtr &tensor)
{
    if (!CheckStringInputLength(tensorName, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The length of tensor name: " << tensorName << " is too long in 'GetTensorByName'.";
        return Status(Error::Code::INVALID_ARG, "The length of tensor name: " + tensorName + " is too long");
    }
    auto iter = inputs_.find(tensorName);
    if (iter == inputs_.end()) {
        return Status(Error::Code::NOT_FOUND, "input '" + tensorName + "' not found in request");
    }
    tensor = iter->second;
    if (tensor == nullptr) {
        return Status(Error::Code::INVALID_ARG, "tensor is nullptr in 'GetTensorByName' parameter");
    }
    return Status(Error::Code::OK, "Success");
}

Status InferRequestImpl::DelTensorByName(const std::string &name)
{
    if (!CheckStringInputLength(name, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The length of tensor name: " << name << "is too long in 'DelTensorByName'.";
        return Status(Error::Code::INVALID_ARG, "The length of tensor name: " + name + " is too long");
    }
    if (inputs_.erase(name) != 1) {
        return Status(Error::Code::INVALID_ARG, "input '" + name + "' does not exist in request");
    }
    return Status(Error::Code::OK, "Success");
}

InferRequestId InferRequestImpl::GetRequestId() const
{
    return requestId_;
}

Status InferRequestImpl::SetMaxOutputLen(uint32_t maxOutputLen)
{
    if (maxOutputLen > 0) {
        maxOutputLen_ = maxOutputLen;
        return Status(Error::Code::OK, "Success");
    } else {
        LOG_ERROR_LLM << "InferRequest SetMaxOutputLen failed due to invalid parameter";
        return Status(Error::Code::ERROR, "output length is invalid parameter");
    }
}

uint32_t InferRequestImpl::GetMaxOutputLen() const
{
    return maxOutputLen_;
}

void InferRequestImpl::SetSendResponseCallback(const mindie_llm::SendResponseCallback4Request &callback)
{
    responseCallback_ = callback;
}

mindie_llm::SendResponseCallback4Request &InferRequestImpl::GetSendResponseCallback()
{
    return responseCallback_;
}

void InferRequestImpl::SetReleaseCallback(const mindie_llm::ReleaseCallback &callback)
{
    releaseCallback_ = callback;
}

mindie_llm::ReleaseCallback &InferRequestImpl::GetReleaseCallback()
{
    return releaseCallback_;
}

void InferRequestImpl::SetEngineResponseCallback(const mindie_llm::SendResponseCallback4Request &callback)
{
    engineResponseCallback_ = callback;
}

mindie_llm::SendResponseCallback4Request &InferRequestImpl::GetEngineResponseCallback()
{
    return engineResponseCallback_;
}

const mindie_llm::TensorMap &InferRequestImpl::ImmutableInputs() const
{
    return inputs_;
}

void InferRequestImpl::SetReqType(mindie_llm::InferReqType reqType)
{
    reqType_ = reqType;
}

mindie_llm::InferReqType InferRequestImpl::GetReqType() const
{
    return reqType_;
}

void InferRequestImpl::SetRecompute(bool isRecompute)
{
    isRecompute_ = isRecompute;
}

bool InferRequestImpl::IsRecompute() const
{
    return isRecompute_;
}

bool InferRequestImpl::IsPrefillReq() const
{
    return reqType_ == mindie_llm::InferReqType::REQ_PREFILL;
}

bool InferRequestImpl::IsDecodeReq() const
{
    return reqType_ == mindie_llm::InferReqType::REQ_DECODE;
}

void InferRequestImpl::SetDTarget(std::string &dTarget)
{
    if (!CheckStringInputLength(dTarget, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The Length of dTarget: " << dTarget << " is too long in SetDTarget.";
        return;
    }
    dTarget_ = dTarget;
}

std::string InferRequestImpl::GetDTarget() const
{
    return dTarget_;
}

void InferRequestImpl::SetPrefillAddr(std::string &prefillAddr)
{
    if (!CheckStringInputLength(prefillAddr, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The Length of dTarget: prefillAddr is too long";
        return;
    }
    prefillAddr_ = prefillAddr;
}

std::string InferRequestImpl::GetPrefillAddr() const
{
    return prefillAddr_;
}

void InferRequestImpl::SetSrcBlockTable(const std::vector<int64_t> &srcBlockTable)
{
    srcBlockTable_ = srcBlockTable;
}

std::vector<int64_t> InferRequestImpl::GetSrcBlockTable() const
{
    return srcBlockTable_;
}

void InferRequestImpl::SetDpInstanceIds(const std::vector<uint64_t> &dpInstanceIds)
{
    dpInstanceIds_ = dpInstanceIds;
}

std::vector<uint64_t> InferRequestImpl::GetDpInstanceIds() const
{
    return dpInstanceIds_;
}

void InferRequestImpl::SetSrcHmoTable(const std::vector<std::vector<int64_t>> &srcHmoTable)
{
    srcHmoTable_ = srcHmoTable;
}

std::vector<std::vector<int64_t>> InferRequestImpl::GetSrcHmoTable() const
{
    return srcHmoTable_;
}

}