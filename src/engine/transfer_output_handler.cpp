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
 
#include "transfer_output_handler.h"
#include "model_exec_output_handler.h"
#include "log.h"

using namespace model_execute_data;
namespace mindie_llm {

TransferOutputHandler::TransferOutputHandler(ForwardRespToManagerCall cb, size_t localDPRank)
    : forwardRespToManagerCall_(cb), localDPRank_(localDPRank)
{
}

void TransferOutputHandler::Entry4Executor(PullKVResponseSPtr pullKvResponse)
{
    // 先处理 successPulledRequestIds_，不阻塞调度
    for (int i = 0; i < pullKvResponse->pull_kv_results_size(); ++i) {
        const auto &result = pullKvResponse->pull_kv_results(i);
        PDErrorCode errorCode = result.pd_error_code();
        if (errorCode == PDErrorCode::SUCCESS) {
            kvPulledReqIds_.PushBack(result.request_id());
        }
    }
    for (int i = 0; i < pullKvResponse->pull_kv_results_size(); ++i) {
        const auto &result = pullKvResponse->pull_kv_results(i);
        // RequestId => InferRequestId
        SequenceGroupSPtr seqGroup = LiveInferContext::GetInstance(localDPRank_)->GetSeqGroup(result.request_id());
        if (seqGroup == nullptr) {
            MINDIE_LLM_LOG_INFO("Pull kv done while seggrp is aborted:" << result.request_id());
            continue;
        }
        RequestIdNew inferRequestId = seqGroup->metrics_.inferReqId_;
        PDErrorCode errorCode = result.pd_error_code();

        // former PULL_KV_RESULT
        ResponseSPtr response = std::make_shared<Response>(inferRequestId);
        response->responseContents.resize(1);
        response->responseContents[0].pdErrorCode = errorCode; // pdErrorCode

        // 返回推理结果给上层的回调函数
        response->transferStatusFlag = TransferStatusType::PULL_KV_COMPLETE;
        if (errorCode != PDErrorCode::SUCCESS) {
            response->isEos = true;
            response->inferStatusFlag = InferStatusType::PULL_KV_ERROR;
        }
        forwardRespToManagerCall_(response);
    }
}

ConcurrentDeque<RequestId> &TransferOutputHandler::GetPulledReqIds() { return kvPulledReqIds_; }
} // namespace mindie_llm
