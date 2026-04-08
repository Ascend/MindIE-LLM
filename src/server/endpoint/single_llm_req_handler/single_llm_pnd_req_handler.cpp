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
#include "single_llm_pnd_req_handler.h"
#include <atomic>
#include "http_metrics.h"
#include "config_manager.h"
#include "prometheus_metrics.h"
#include "log.h"
#include "config_manager_impl.h"
#include "infer_tokenizer.h"
#include "safe_io.h"

using ordered_json = nlohmann::ordered_json;

// Count the number of created and deleted SingleLLMPnDReqHandler objects for debug.
std::atomic<size_t> g_numCreateSingleLLMPnDReqHandler = 0;
std::atomic<size_t> g_numDeleteSingleLLMPnDReqHandler = 0;

namespace mindie_llm {

SingleLLMPnDReqHandler::SingleLLMPnDReqHandler(ReqCtxPtr& ctx, bool isFlexLocalReq)
    : SingleLLMReqHandlerBase(ctx), isFlexLocalReq_(isFlexLocalReq)
{
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Create SingleLLMPnDReqHandler #" << (++g_numCreateSingleLLMPnDReqHandler));
}

SingleLLMPnDReqHandler::~SingleLLMPnDReqHandler()
{
    if (eventDispatcher != nullptr) {
        eventDispatcher->Clear();
    }
    try {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Delete SingleLLMPnDReqHandler #" << (++g_numDeleteSingleLLMPnDReqHandler));
    } catch (...) {
        // Forbid to throw exceptions in destructor function
    }
}

bool SingleLLMPnDReqHandler::GetContextJsonBody(ordered_json &body)
{
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> convertor;
        auto converted = convertor.to_bytes(convertor.from_bytes(ctx->MsgBody()));
        if (!ordered_json::accept(converted)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                       "Convert string to json object exception, cbId is " << ctx->CallbackId());
            return false;
        }
        body = ordered_json::parse(converted, CheckOrderedJsonDepthCallback);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   "Convert string to json object exception, cbId is " << ctx->CallbackId());
        return false;
    }
    return true;
}

bool SingleLLMPnDReqHandler::GetContextRequestId(std::string &requestId)
{
    requestId = ctx->Req().get_header_value("req-id");
    return !requestId.empty();
}

void SingleLLMPnDReqHandler::UpdateInferRequest([[maybe_unused]] const std::vector<int64_t> &reqTokens,
                                                [[maybe_unused]] const int64_t &oriReqTokenLen,
                                                [[maybe_unused]] RequestSPtr request)
{
}

void SingleLLMPnDReqHandler::UpdateInferParam(RequestSPtr request, const InferParamSPtr &inferParam)
{
    request_ = request;
    inferParam_ = inferParam;
}

void SingleLLMPnDReqHandler::Process(RequestSPtr request, const std::string &inputId, const uint64_t &timestamp)
{
    if (isFlexLocalReq_) {
        request->reqType = InferReqType::REQ_FLEX_LOCAL;
    }
    metrics.startingTime = std::chrono::steady_clock::now();
    SetBackManagerCallBack(request);
    Status status = GetInferInstance()->Process(request);
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "request_id=" << inputId << " enter backend forward and return result");
    if (!status.IsOk()) {
        auto errMsg = "Failed to enqueue inferRequest. " + status.StatusMsg();
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                  "request_id=" << request->requestId << ". " << errMsg);
        int errCode = status.StatusCode() == Error::Code::INVALID_ARG ? httplib::StatusCode::FailedDependency_424
                                                                      : httplib::StatusCode::InternalServerError_500;
        std::string errType = status.StatusCode() == Error::Code::INVALID_ARG
                                  ? g_exceptionInfo.at(httplib::StatusCode::FailedDependency_424)
                                  : g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500);
        SendResponse(errCode, HttpRestResource::WrapperJson(errMsg, errType));
        ProcessFailedResponsePrometheusMetrics();
        constructOneResponseCallBack_ = nullptr;
        return;
    }

    if (streamMode_) {
        ProcessStreamModeRequest(inputId, timestamp);
    } else {
        ProcessNonStreamModeRequest(inputId, timestamp);
    }
}

void SingleLLMPnDReqHandler::SetBackManagerCallBack(RequestSPtr request)
{
    // 使用weak_ptr避免因request与handler之间循环引用导致的内存泄漏
    std::weak_ptr<SingleLLMPnDReqHandler> weakSelf = shared_from_this();
    auto requestId = request->requestId;
    request->serverResponseCallback_ = [weakSelf, requestId](ResponseSPtr response) {
        auto self = weakSelf.lock();
        if (!self) {
            return;
        }
        if (response == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Invoke callback failed: response is null. request_id=" << requestId);
            return;
        }

        boost::unique_lock<boost::mutex> locker(self->lock);
        if (self->isFinish_.load()) {
            return;
        }
        std::vector<BestNTokens> bestNTokens;
        if (!self->ParseTokensFromResponse(response, bestNTokens)) {
            self->ProcessFailedResponsePrometheusMetrics();
        } else {
            self->ProcessOneResponsePrometheusMetrics(response);
            std::for_each(bestNTokens.begin(), bestNTokens.end(), [self](BestNTokens &item) {
                self->respTokens.insert(self->respTokens.end(), item.tokens.begin(), item.tokens.end());
            });
            if (response->isEos) {
                PrometheusMetrics::GetInstance()->ResponseOutputTokenHistogramCollect(self->respTokens.size());
                PrometheusMetrics::GetInstance()->ResponseOutputTokenCount(self->respTokens.size());
            }
        }

        self->inferResponseQueue_.push(response);
        self->cv.notify_one();
    };
}

// 发送响应信息
void SingleLLMPnDReqHandler::SendResponseInfo(int code, const std::string &responseStr, bool needMetricsCollect)
{
    // set failed request prometheus metrics
    if (needMetricsCollect && code != httplib::StatusCode::OK_200) {
        ProcessFailedResponsePrometheusMetrics();
    }
    if (streamMode_) {
        return SendResponseStream(true, responseStr);
    }
    return SendResponse(code, responseStr);
}

// 发送非流式响应
void SingleLLMPnDReqHandler::SendResponse(int code, const std::string &responseStr)
{
    HttpRestResource::ResponseJsonBody(ctx, code, responseStr);
}

// 发送流式响应
void SingleLLMPnDReqHandler::SendResponseStream(bool isEnd, const std::string &responseStr)
{
    if (responseStr == "") {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Response of the streaming request is empty");
        return;
    }
    eventDispatcher->SendEvent(responseStr, isEnd);
}

void SingleLLMPnDReqHandler::ProcessNonStreamModeRequest(const std::string &inputId, const uint64_t &timestamp)
{
    bool responseFinished = false;
    boost::cv_status res = boost::cv_status::no_timeout;
    std::vector<ResponseSPtr> inferResponses;

    auto clientStartTime = boost::chrono::steady_clock::now();
    BoostTimePoint lastTimePoint;
    std::string timeoutDesc;

    while (!responseFinished) {
        boost::unique_lock<boost::mutex> locker(lock);

        auto tokenStartTime = boost::chrono::steady_clock::now();
        GetTimeoutPoint(clientStartTime, lastTimePoint, timeoutDesc);

        if (inferResponseQueue_.empty()) {
            res = cv.wait_until(locker, lastTimePoint);
            if (res == boost::cv_status::timeout) {
                isFinish_.store(true);
                locker.unlock();
                break;
            }
        }

        while (!inferResponseQueue_.empty()) {
            inferResponses.push_back(inferResponseQueue_.front());
            inferResponseQueue_.pop();
        }
        locker.unlock();

        responseFinished = ProcessNonStreamBatchResponse(inferResponses);
        inferResponses.clear();

        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                   "Token spend time is " << GetElapsedTimeMillis(tokenStartTime) << " ms. request_id=" << inputId);
    }

    if (res == boost::cv_status::timeout) {
        std::string timeoutMsg = "Engine callback timeout: " + timeoutDesc;
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, TIMEOUT_WARNING),
                  timeoutMsg << ". request_id=" << inputId);

        RequestIdNew requestId{inputId};
        Status status = GetInferInstance()->ControlRequest(requestId, OperationV2::STOP);
        if (status.StatusCode() != Error::Code::OK) {
            ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                      GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, STATUS_WARNING),
                      "Failed stop inference for timeout. request_id=" << inputId << ". " << status.StatusMsg());
        } else {
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Stop inference for timeout successfully. request_id=" << inputId);
            TokenizerProcessPool::GetInstance().RemoveMultimodalCache(timestamp);
        }
        SendResponseInfo(httplib::StatusCode::InternalServerError_500,
                         HttpRestResource::WrapperJson(
                             timeoutMsg, g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
        constructOneResponseCallBack_ = nullptr;
        return;
    }

    if (responseFinished) {
        TokenizerProcessPool::GetInstance().RemoveMultimodalCache(timestamp);
    }

    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "E2E spend time is " << GetElapsedTimeMillis(GetMetrics().e2eStartingTime)
                                                            << " ms. request_id=" << inputId);
}

bool SingleLLMPnDReqHandler::ProcessNonStreamBatchResponse(
    const std::vector<ResponseSPtr> &responseImpls)
{
    for (auto &responseImpl : responseImpls) {
        if (ProcessOneNonStreamResponse(responseImpl)) {
            boost::unique_lock<boost::mutex> locker(lock);
            isFinish_.store(true);
            return true;
        }
    }

    return false;
}

bool SingleLLMPnDReqHandler::ProcessOneNonStreamResponse(const ResponseSPtr &response)
{
    std::vector<BestNTokens> bestNTokens;
    if (!ParseTokensFromResponse(response, bestNTokens)) {
        constructOneResponseCallBack_ = nullptr;
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to get engine response");
        SendResponse(httplib::StatusCode::InternalServerError_500,
                     HttpRestResource::WrapperJson("Failed to get engine response.",
                                                   g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
        return true;
    }

    RespBodyQueue responseJsonQueue;
    auto decodeStatus = false;
    if (constructOneResponseCallBack_ != nullptr) {
        decodeStatus = constructOneResponseCallBack_(response, bestNTokens, responseJsonQueue);
    }
    if (!decodeStatus) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to decode tokenId to response. requestId: " << response->reqId);
    }
    if (response->isEos) {
        constructOneResponseCallBack_ = nullptr;
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                  "ResponseCallback begin to send last response. requestId: " << response->reqId);
        while (!responseJsonQueue.empty()) {
            std::string msgTmp = responseJsonQueue.front();
            responseJsonQueue.pop();
            if (msgTmp.empty()) {
                ULOG_ERROR(
                    SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to send empty http body in non-stream mode. requestId: "
                        << response->reqId);
                continue;
            }
            SendResponse(httplib::StatusCode::OK_200, msgTmp);
        }
        return true;
    }

    return false;
}

void SingleLLMPnDReqHandler::ProcessStreamModeRequest(const std::string &inputId, const uint64_t &timestamp)
{
    auto &httpResp = ctx->Res();
    auto self = shared_from_this();
    httpResp.set_chunked_content_provider("text/event-stream",
                                          [self, inputId, timestamp](size_t /* offset */, httplib::DataSink &sink) {
                                              self->StreamResponseFillHttpSink(inputId, sink, timestamp);
                                              return true;
                                          });
}

void SingleLLMPnDReqHandler::StreamResponseFillHttpSink(const std::string &inputId, httplib::DataSink &sink,
                                                        const uint64_t &timestamp)
{
    auto tokenStartTime = boost::chrono::steady_clock::now();
    auto clientStartTime = tokenStartTime;

    BoostTimePoint lastTimePoint;
    std::string timeoutDesc;
    GetTimeoutPoint(clientStartTime, lastTimePoint, timeoutDesc);

    boost::unique_lock<boost::mutex> locker(lock);
    while (inferResponseQueue_.empty()) {
        auto res = cv.wait_until(locker, lastTimePoint);
        if (res == boost::cv_status::timeout) {
            isFinish_.store(true);
            locker.unlock();

            std::string timeoutMsg = "Engine callback timeout: " + timeoutDesc;
            ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                      GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, TIMEOUT_WARNING),
                      timeoutMsg << ". request_id=" << inputId);

            RequestIdNew requestId{inputId};
            Status status = GetInferInstance()->ControlRequest(requestId, OperationV2::STOP);
            if (status.StatusCode() != Error::Code::OK) {
                ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                          GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE, STATUS_WARNING),
                          "Failed stop inference for timeout. request_id=" << inputId << ". " << status.StatusMsg());
            } else {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Stop inference for timeout successfully. request_id=" << inputId);
                TokenizerProcessPool::GetInstance().RemoveMultimodalCache(timestamp);
            }
            ctx->SetResponseFinished(true);
            sink.write(timeoutMsg.data(), timeoutMsg.size());
            sink.done();
            ProcessFailedResponsePrometheusMetrics();
            constructOneResponseCallBack_ = nullptr;
            return;
        }
    }

    std::vector<ResponseSPtr> inferResponses;
    inferResponses.reserve(inferResponseQueue_.size());
    while (!inferResponseQueue_.empty()) {
        inferResponses.push_back(inferResponseQueue_.front());
        inferResponseQueue_.pop();
    }
    locker.unlock();

    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
               "Token spend time is " << GetElapsedTimeMillis(tokenStartTime) << " ms. request_id=" << inputId);

    bool finished = false;
    for (auto &inferResp : inferResponses) {
        RespBodyQueue responseJsonQueue;
        finished = ProcessOneStreamResponse(inferResp, responseJsonQueue) || finished;
        while (!responseJsonQueue.empty()) {
            std::string msgTmp = responseJsonQueue.front();
            responseJsonQueue.pop();
            if (msgTmp.empty()) {
                continue;
            }
            sink.write(msgTmp.data(), msgTmp.size()); // write one token into http response stream
        }

        if (finished) {
            sink.done();
            locker.lock();
            isFinish_.store(true);
            locker.unlock();
            // Update Stream Response Finished Flag
            if (inferResp->isEos) {
                ctx->SetResponseFinished(true);
            }
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "E2E spend time is "
                                                   << GetElapsedTimeMillis(GetMetrics().e2eStartingTime)
                                                   << " ms. request_id=" << inputId);
            TokenizerProcessPool::GetInstance().RemoveMultimodalCache(timestamp);
            break;
        }
    }
}

bool SingleLLMPnDReqHandler::ProcessOneStreamResponse(const ResponseSPtr &response, RespBodyQueue &text)
{
    std::vector<BestNTokens> bestNTokens;
    if (!ParseTokensFromResponse(response, bestNTokens)) {
        constructOneResponseCallBack_ = nullptr;
        std::string err = "Failed to get engine response";
        text.push(err);
        return true;
    }

    RespBodyQueue responseJsonQueue;
    auto decodeStatus = false;
    if (constructOneResponseCallBack_ != nullptr) {
        decodeStatus = constructOneResponseCallBack_(response, bestNTokens, responseJsonQueue);
    }
    if (!decodeStatus) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to decode tokenId to response. requestId: " << response->reqId);
    }
    auto isEnd = response->isEos;
    if (isEnd) {
        constructOneResponseCallBack_ = nullptr;
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                  "ResponseCallback begin to send last response. requestId: " << response->reqId);
    }

    text = std::move(responseJsonQueue);
    return isEnd;
}

void SingleLLMPnDReqHandler::ProcessOneResponsePrometheusMetrics(const ResponseSPtr &response)
{
    SetMetricParams(response);
    if (response->isEos) {
        PrometheusMetrics::GetInstance()->ResponseNumberCount();
        PrometheusMetrics::GetInstance()->FailedRequestRateGaugeCollect();
        size_t e2eTime = static_cast<size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                 std::chrono::steady_clock::now() - GetMetrics().e2eStartingTime)
                                                 .count());
        PrometheusMetrics::GetInstance()->E2EObserve(e2eTime);
    }
}

void SingleLLMPnDReqHandler::GetTimeoutPoint(const BoostTimePoint &tokenStartTime, BoostTimePoint &lastTimePoint,
    std::string &desc)
{
    auto &serverConfig = GetServerConfig();
    // 打印出tokenTimeout 和 e2eTimeout的值，方便排查超时问题
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "server tokenTimeout is " << serverConfig.tokenTimeout
        << " seconds, server e2eTimeout is " << serverConfig.e2eTimeout << " seconds.");
    BoostTimePoint clientTimeoutPoint = tokenStartTime + boost::chrono::seconds(inferParam_->timeout);
    BoostTimePoint serverTokenTimeoutPoint = tokenStartTime + boost::chrono::seconds(serverConfig.tokenTimeout);
    // e2e means endpoint to endpoint, e2eTimeoutPoint = e2eStartingTime + e2eTimeoutDuration
    BoostTimePoint e2eTimeoutPoint(boost::chrono::nanoseconds(
        (GetMetrics().e2eStartingTime + std::chrono::seconds(serverConfig.e2eTimeout)).time_since_epoch().count()));

    // The timeout moment is the earliest of the three timeout constraints
    lastTimePoint = std::min({clientTimeoutPoint, serverTokenTimeoutPoint, e2eTimeoutPoint});
    if (lastTimePoint == clientTimeoutPoint) {
        desc = "client timeout";
    } else if (lastTimePoint == serverTokenTimeoutPoint) {
        desc = "server tokenTimeout";
    } else {
        desc = "server e2eTimeout";
    }
}

int64_t SingleLLMPnDReqHandler::GetElapsedTimeMillis(const std::any &startTimePoint) const
{
    if (startTimePoint.type() == typeid(std::chrono::steady_clock::time_point)) {
        auto tp = std::any_cast<std::chrono::steady_clock::time_point>(startTimePoint);
        auto duration = std::chrono::steady_clock::now() - tp;
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    } else if (startTimePoint.type() == typeid(boost::chrono::steady_clock::time_point)) {
        auto tp = std::any_cast<boost::chrono::steady_clock::time_point>(startTimePoint);
        auto duration = boost::chrono::steady_clock::now() - tp;
        return boost::chrono::duration_cast<boost::chrono::milliseconds>(duration).count();
    } else {
        throw std::runtime_error("Unsupported time point type.");
    }
}
} // namespace mindie_llm