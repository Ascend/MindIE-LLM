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

#include "llm_manager/infer_request.h"

#include "llm_manager/infer_request_impl.h"
#include "system_log.h"
namespace mindie_llm {
InferRequest::InferRequest(InferRequestId requestId)
{
    impl_ = std::make_shared<InferRequestImpl>(requestId);
}

Status InferRequest::AddTensor(const std::string& tensorName, TensorPtr &tensor)
{
    return impl_->AddTensor(tensorName, tensor);
}

void InferRequest::SetTensor(const std::string& tensorName, TensorPtr &tensor)
{
    impl_->SetTensor(tensorName, tensor);
}

Status InferRequest::GetTensorByName(const std::string& tensorName, TensorPtr &tensor)
{
    return impl_->GetTensorByName(tensorName, tensor);
}

Status InferRequest::DelTensorByName(const std::string &name)
{
    return impl_->DelTensorByName(name);
}

InferRequestId InferRequest::GetRequestId() const
{
    return impl_->GetRequestId();
}

Status InferRequest::SetMaxOutputLen(uint32_t maxOutputLen)
{
    return impl_->SetMaxOutputLen(maxOutputLen);
}

uint32_t InferRequest::GetMaxOutputLen() const
{
    return impl_->GetMaxOutputLen();
}

std::shared_ptr<mindie_llm::InferRequestImpl> InferRequest::GetRequestInner() const
{
    if (impl_ == nullptr) {
        LOG_ERROR_LLM << "RequestInner in InferRequest is null";
    }
    return impl_;
}

const TensorMap &InferRequest::ImmutableInputs() const
{
    return impl_->ImmutableInputs();
}

void InferRequest::SetReqType(mindie_llm::InferReqType reqType)
{
    impl_->SetReqType(reqType);
}

mindie_llm::InferReqType InferRequest::GetReqType() const
{
    return impl_->GetReqType();
}

void InferRequest::SetRecompute(bool isRecompute)
{
    impl_->SetRecompute(isRecompute);
}

bool InferRequest::IsRecompute() const
{
    return impl_->IsRecompute();
}

bool InferRequest::IsPrefillReq() const
{
    return impl_->IsPrefillReq();
}

bool InferRequest::IsDecodeReq() const
{
    return impl_->IsDecodeReq();
}

void InferRequest::SetDTarget(std::string &dTarget)
{
    impl_->SetDTarget(dTarget);
}

std::string InferRequest::GetDTarget() const
{
    return impl_->GetDTarget();
}

void InferRequest::SetPrefillAddr(std::string &prefillAddr)
{
    impl_->SetPrefillAddr(prefillAddr);
}

std::string InferRequest::GetPrefillAddr() const
{
    return impl_->GetPrefillAddr();
}

void InferRequest::SetSrcBlockTable(const std::vector<int64_t> &srcBlockTable)
{
    impl_->SetSrcBlockTable(srcBlockTable);
}

std::vector<int64_t> InferRequest::GetSrcBlockTable() const
{
    return impl_->GetSrcBlockTable();
}

void InferRequest::SetDpInstanceIds(const std::vector<uint64_t> &dpInstanceIds)
{
    impl_->SetDpInstanceIds(dpInstanceIds);
}

std::vector<uint64_t> InferRequest::GetDpInstanceIds() const { return impl_->GetDpInstanceIds(); }

InferRequest::~InferRequest() = default;
}