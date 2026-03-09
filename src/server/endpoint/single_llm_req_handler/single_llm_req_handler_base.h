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
 
#ifndef BASE_WRAPPER_H
#define BASE_WRAPPER_H

#include <codecvt>
#include <vector>
#include <functional>
#include <string>
#include <memory>
#include <queue>
#include <atomic>

#include "prefillAndDecodeCommunication.grpc.pb.h"
#include "httplib.h"
#include "http_rest_resource.h"
#include "endpoint_def.h"
#include "event_dispatcher.h"
#include "infer_param.h"
#include "nlohmann/json.hpp"
#include "infer_instances.h"

namespace mindie_llm {
using ConstructOneResponseCallBack =
    std::function<bool(const ResponseSPtr &response, const std::vector<BestNTokens> &, RespBodyQueue &)>;

class SingleLLMReqHandlerBase {
public:
    using DMIReComputeBuildMethod = std::function<std::string(const std::vector<BestNTokens>&)>;

    explicit SingleLLMReqHandlerBase(ReqCtxPtr& ctx);
    virtual ~SingleLLMReqHandlerBase() = default;
    void SetConstructOneResponseCallBack(const ConstructOneResponseCallBack &constructOneResponseCallBack);
    void SetDMIReComputeBuildCallBack(const DMIReComputeBuildMethod &reComputeBuildMethod);
    void SetStreamMode(bool streamMode);
    bool ParseTokensFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken);
    // Prometheus
    Metrics &GetMetrics();
    void SetRecomputeMetrics(Metrics &reMetrics);

    void MetricsCallback(const ResponseSPtr &response);
    void SetMetricParams(const ResponseSPtr &response);

    InferReqType GetReqType() const {return reqType_;};

    void ProcessFailedResponsePrometheusMetrics() const;
    void InsertStopRequest(std::string stopReqId) const;

    void DumpInferParam(const RequestSPtr request);

    virtual void Process(RequestSPtr request, const std::string &inputId, const uint64_t &timestamp = 0) = 0;
    virtual void ProcessGrpcReq([[maybe_unused]] RequestSPtr request, [[maybe_unused]] const std::string &inputId,
                                [[maybe_unused]] prefillAndDecodeCommunication::DecodeRequestResponse &response){};
    virtual bool GetContextJsonBody(nlohmann::ordered_json &body);
    virtual bool GetContextJsonBody(InferParamSPtr inputParam, RequestSPtr request, std::vector<int64_t> &reqTokens,
                                    std::vector<int64_t> &respTokens);
    virtual bool GetContextRequestId(std::string& requestId) = 0;
    virtual void UpdateInferRequest(const std::vector<int64_t> &reqTokens,
                                    const int64_t &oriReqTokenLen, RequestSPtr request) = 0;
    virtual void UpdateInferParam(RequestSPtr request, const InferParamSPtr &inferParam);
    virtual void SetBackManagerCallBack(RequestSPtr request) = 0;
    virtual void SendResponseInfo(int code, const std::string& responseStr, bool needMetricsCollect = true) = 0;
    virtual void SendResponse(int code, const std::string& responseStr) = 0;
    virtual void SendResponseStream(bool isEnd, const std::string& responseStr) = 0;

protected:
    boost::mutex lock;
    boost::condition_variable cv;
    std::queue<ResponseSPtr> inferResponseQueue_;

    uint32_t seqEndCount_{0};
    std::atomic_bool isFinish_ {false};
    bool streamMode_ { false };

    const ReqCtxPtr ctx;
    ConstructOneResponseCallBack constructOneResponseCallBack_{};
    DMIReComputeBuildMethod dmiReCompBuildMethod_{};
    uint64_t simulatedSeqIndex{1};
    std::shared_ptr<EventDispatcher> eventDispatcher{nullptr};
    InferParamSPtr inferParam_{nullptr};
    RequestSPtr request_{nullptr};

    Metrics metrics;
    bool isRecompute_ {false};

    std::vector<int64_t> respTokens{};
    static std::unordered_set<std::string> stopReqSet_;

    InferReqType reqType_ =  InferReqType::REQ_STAND_INFER;

private:
    bool ParseSeqIdFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken);
    bool ParseParentSeqIdFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken) const;
    bool ParseOutTokenIdFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken) const;
    bool ParseCumLogProbsFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken) const;
    bool ParseFinishReasonAndTruncationIdFromResponse(const ResponseSPtr &response,
        std::vector<BestNTokens> &postToken) const;
    bool ParseOutLogprobFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken) const;
    bool ParseTopLogProbsFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken) const;
    bool ParseLogProbsFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken) const;
};
} // namespace mindie_llm
#endif // BASE_WRAPPER_H