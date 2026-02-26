/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef DECODE_WRAPPER_H
#define DECODE_WRAPPER_H

#include <vector>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/chrono.hpp>
#include <shared_mutex>
#include "prefillAndDecodeCommunication.grpc.pb.h"
#include "http_rest_resource.h"
#include "dresult_event_dispatcher.h"
#include "grpc_context.h"
#include "single_llm_req_handler_base.h"

namespace mindie_llm {
class SingleLLMDecodeReqHandler final : public SingleLLMReqHandlerBase,
                                        public std::enable_shared_from_this<SingleLLMDecodeReqHandler> {
public:
    using SingleLLMReqHandlerBase::GetContextJsonBody;
    SingleLLMDecodeReqHandler(ReqCtxPtr& ctx, std::shared_ptr<DResultEventDispatcher>& dResultDispatcher,
        std::shared_ptr<GrpcContext>& grpcContext);
    ~SingleLLMDecodeReqHandler() override;

    void Process([[maybe_unused]] RequestSPtr request, [[maybe_unused]] const std::string &inputId,
                 [[maybe_unused]] const uint64_t &timestamp = 0) override
    {
    }
    void ProcessGrpcReq(RequestSPtr request, const std::string &inputId,
                        prefillAndDecodeCommunication::DecodeRequestResponse &response) override;
    bool GetContextJsonBody(InferParamSPtr param, RequestSPtr request, std::vector<int64_t> &reqTokens,
                            std::vector<int64_t> &respTokens) override;
    bool GetContextRequestId(std::string& requestId) override;
    void UpdateInferRequest(const std::vector<int64_t> &reqTokens,
                            const int64_t &oriReqTokenLen, RequestSPtr request) override;
    void SetBackManagerCallBack(RequestSPtr request) override;
    void SendResponseInfo(int code, const std::string& responseStr, bool needMetricsCollect = true) override;
    void SendResponse(int code, const std::string& responseStr) override;
    void SendResponseStream(bool isEnd, const std::string& responseStr) override;
    void GetContextSamplingParamsFirst(RequestSPtr request);
    void GetContextSamplingParamsNext(InferParamSPtr param, RequestSPtr request);
    void GetContextSamplingStopWords(InferParamSPtr param, RequestSPtr request);
    void GetContextInferParams(InferParamSPtr param, RequestSPtr request);
    void GetContextMetrics();

private:
    bool SendKvRelease(const std::string& reqId);
    void ResponseReCompute(const std::string& body);
    void SendDResult(const std::string &msg, std::string reqId);
    void SendDError(const std::string &errorMsg);

    std::shared_ptr<DResultEventDispatcher> dResultDispatcher_{nullptr};
    std::string pNodeAddr_;
    // pInstanceId_ will be updated by GetContextJsonBody
    uint32_t pInstanceId_{0};
    std::string reqId_;
    std::string tritonReqId_;
    std::shared_ptr<GrpcContext> grpcContext_{nullptr};
    std::vector<BestNTokens> respTokens_;
    std::thread controlThread_;
    // 多个response并发call back使用；TBC_后续会做P/D分离Response处理优化
    std::shared_mutex decodeCbMutex;
};
} // namespace mindie_llm
#endif // DECODE_WRAPPER_H