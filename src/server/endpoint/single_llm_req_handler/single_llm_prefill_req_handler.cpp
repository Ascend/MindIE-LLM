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
#include "single_llm_prefill_req_handler.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/chrono.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>

#include "httplib.h"
#include "prometheus_metrics.h"
#include "log.h"
#include "endpoint_def.h"
#include "env_util.h"
#include "dmi_role.h"
#include "config_manager_impl.h"
#include "safe_io.h"

using namespace prefillAndDecodeCommunication;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
SingleLLMPrefillReqHandler::SingleLLMPrefillReqHandler(ReqCtxPtr& ctx, uint16_t msgType, bool isRecompute)
    : SingleLLMReqHandlerBase(ctx)
{
    msgType_ = msgType;
    isRecompute_ = isRecompute;
    reqType_ =  InferReqType::REQ_PREFILL;
}

void SingleLLMPrefillReqHandler::Process(RequestSPtr request, const std::string &requestId, const uint64_t &timestamp)
{
    (void)timestamp;
    PROF(INFO, Domain("Communication").Resource(requestId.c_str()).Event("receiveReq"));
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[P Node] Receive a request. requestId: " << requestId);
    reqId_ = requestId;
    metrics.startingTime = std::chrono::steady_clock::now();
    metrics.e2eStartingTime = std::chrono::steady_clock::now();
    metrics.sysE2eStartingTime = std::chrono::system_clock::now();
    SetBackManagerCallBack(request);
    Status status = GetInferInstance()->Process(request);
    if (!status.IsOk()) {
        auto errMsg = "Failed to enqueue inferRequest. " + status.StatusMsg();
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "requestId: " << requestId << ". " << errMsg);
        int errCode = status.StatusCode() == Error::Code::INVALID_ARG ?
            httplib::StatusCode::FailedDependency_424 : httplib::StatusCode::InternalServerError_500;
        std::string errType = status.StatusCode() == Error::Code::INVALID_ARG ?
            g_exceptionInfo.at(httplib::StatusCode::FailedDependency_424) :
            g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500);
        SendResponseInfo(errCode, HttpRestResource::WrapperJson(errMsg, errType));
        constructOneResponseCallBack_ = nullptr;
        return;
    }
    auto &serverConfig = GetServerConfig();
    boost::unique_lock<boost::mutex> locker(lock);
    auto res = cv.wait_until(locker,
        boost::chrono::steady_clock::now() + boost::chrono::seconds(serverConfig.tokenTimeout));
    if (res == boost::cv_status::timeout) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            TIMEOUT_WARNING), "Call response callback timeout! requestId: " << requestId);
        bool expect = false;
        // if has not send control request
        if (!AtomicReadWriteFinish(expect)) {
            return;
        }

        Status status = GetInferInstance()->ControlRequest(requestId, OperationV2::STOP);
        if (status.StatusCode() != Error::Code::OK) {
            ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
                TIMEOUT_WARNING), "Fail to stop timeout. requestId: " << requestId);
        } else {
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Succeed to stop timeout. requestId: " << requestId);
        }
        SendResponseInfo(httplib::StatusCode::InternalServerError_500,
            HttpRestResource::WrapperJson("Engine callback timeout.",
            g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
    }
}

bool SingleLLMPrefillReqHandler::GetContextJsonBody(OrderedJson& body)
{
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> convertor;
        auto converted = convertor.to_bytes(convertor.from_bytes(ctx->MsgBody()));
        if (!OrderedJson::accept(converted)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                JSON_PARSE_ERROR), "Convert string to json object exception. CallbackId is " << ctx->CallbackId());
            return false;
        }
        body = OrderedJson::parse(converted, CheckOrderedJsonDepthCallback);
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            JSON_PARSE_ERROR), "Convert string to json object exception. CallbackId is " << ctx->CallbackId());
        return false;
    }
    return true;
}

bool SingleLLMPrefillReqHandler::GetContextRequestId(std::string& requestId)
{
    requestId = ctx->Req().get_header_value("req-id");
    return !requestId.empty();
}

void SingleLLMPrefillReqHandler::UpdateInferRequest(const std::vector<int64_t> &reqTokens, const int64_t &tokenLen,
                                                    RequestSPtr request)
{
    inputTokens_ = reqTokens;
    oriReqTokenLen = tokenLen;
    request->reqType = mindie_llm::InferReqType::REQ_PREFILL;
    request->isRecompute = isRecompute_;
}

void SingleLLMPrefillReqHandler::UpdateInferParam(RequestSPtr request, const InferParamSPtr &inferParam)
{
    request_ = request;
    inferParam_ = inferParam;
    modelName_ = inferParam->model_;
}

bool SingleLLMPrefillReqHandler::GenerateFirstToken(ResponseSPtr response, bool expect)
{
    firstToken_.clear();
    // 1.1 get first token from response
    std::vector<BestNTokens> firstToken;
    if (!ParseTokensFromResponse(response, firstToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "[P Node] Failed to get tokenIds from inferResponse. requestId: "
            << reqId_);
        if (!AtomicReadWriteFinish(expect)) {
            constructOneResponseCallBack_ = nullptr;
            return false;
        }
        SendResponseInfo(httplib::StatusCode::InternalServerError_500,
            HttpRestResource::WrapperJson("[P Node] Failed to get tokenIds from inferResponse.",
            g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
        cv.notify_one();
        constructOneResponseCallBack_ = nullptr;
        return false;
    }
    SetMetricParams(response);

    // 1.2 detokenizer first token
    if (firstToken.empty()) {
        return false;
    }
    firstToken_ = firstToken[0].tokens;
    if (constructOneResponseCallBack_ == nullptr || !constructOneResponseCallBack_(response, firstToken, respStr_)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ENCODE_DECODE_ERROR), "[P Node] Failed to decode tokenIds. requestId: " << reqId_);
        if (!AtomicReadWriteFinish(expect)) {
            constructOneResponseCallBack_ = nullptr;
            return false;
        }
        SendResponseInfo(httplib::StatusCode::InternalServerError_500,
            HttpRestResource::WrapperJson("[P Node] Failed to decode tokenIds to respStr.",
            g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
        cv.notify_one();
        constructOneResponseCallBack_ = nullptr;
        return false;
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "[P Node] Finish decode first token. requestId: " << reqId_);
    constructOneResponseCallBack_ = nullptr;
    return true;
}

void SingleLLMPrefillReqHandler::GetPNodeAddr(bool containPort)
{
    auto &serverConfig = GetServerConfig();
    prefillNodeAddr_ = serverConfig.ipAddress;
    if (containPort) {
        prefillNodeAddr_.append(IP_PORT_DELIMITER).append(std::to_string(serverConfig.interCommPort));
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[SingleLLMPrefillReqHandler] GetPNodeAddr success. Prefill node ip is "
        << prefillNodeAddr_);
}

void SingleLLMPrefillReqHandler::GetSingleLLMPrefillReqHandlerId()
{
    pInstanceId_ = DmiRole::GetInstance()->GetLocalInstanceId();
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
               "[SingleLLMPrefillReqHandler] GetSingleLLMPrefillReqHandlerId success. Prefill instance id is "
                   << pInstanceId_);
}

void SingleLLMPrefillReqHandler::SetBackManagerCallBack(RequestSPtr request)
{
    // 使用weak_ptr避免因request与handler之间循环引用导致的内存泄漏
    std::weak_ptr<SingleLLMPrefillReqHandler> weakSelf = shared_from_this();
    std::string dTargetAddr = ctx->Req().get_header_value("d-target");
    bool containPort = dTargetAddr.find(IP_PORT_DELIMITER) != std::string::npos;
    GetPNodeAddr(containPort);
    GetSingleLLMPrefillReqHandlerId();
    request->serverResponseCallback_ = [weakSelf, dTargetAddr](ResponseSPtr response) {
        auto self = weakSelf.lock();
        if (!self) {
            return;
        }
        std::unique_lock lock(self->prefillCbMutex);
        if (response == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                CHECK_ERROR), "[P Node] Invoke callback failed: response is null");
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }

        bool expect = false;
        /* 3. receive a cancel response, with a high possibility to be a release kvcache response */
        if (response->inferStatusFlag == InferStatusType::RELEASE_KV_COMPLETE) {
            PROF(INFO, Domain("Communication").Resource(self->reqId_.c_str()).Event("releaseKvcacheRes"));
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[P Node] Got a release kvcache response. requestId: "
                << self->reqId_);
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }

        /* 4. receive a cancel response, it is possible to be a stop response */
        if (response->inferStatusFlag == InferStatusType::ILLEGAL_INPUT) {
            PROF(INFO, Domain("Communication").Resource(self->reqId_.c_str()).Event("timeoutRes"));
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[P Node] Got a stop response. requestId: " << self->reqId_);
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }

        if (self->isFinish_.load()) {
            ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
                CHECK_WARNING), "[P Node] has finished! requestId: " << self->reqId_);
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }

        /* send to coordinator if it is the last token */
        if (response->isEos) {
            if (!self->GenerateFirstToken(response, expect)) {
                return;
            }
            PrometheusMetrics::GetInstance()->ResponseNumberCount();
            PrometheusMetrics::GetInstance()->FailedRequestRateGaugeCollect();
            uint64_t e2eTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                self->GetMetrics().e2eStartingTime).count();
            PrometheusMetrics::GetInstance()->E2EObserve(e2eTime);
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                "[P Node] Send first end token to coordinator. requestId: " << self->reqId_);
            if (!self->AtomicReadWriteFinish(expect)) {
                return;
            }
            self->ProcessResponseStream(response->isEos);
            self->cv.notify_one();
            return;
        }

        if (response->transferStatusFlag == TransferStatusType::PREFILL_COMPLETE) {
            /* 1. generate first token */
            if (!self->GenerateFirstToken(response, expect)) {
                return;
            }
        } else if (response->transferStatusFlag == TransferStatusType::PUBLISH_KV_COMPLETE) {
            /* note: send to D and coordinator.
            */
            auto isReqStopped = self->stopReqSet_.find(self->reqId_) != self->stopReqSet_.end() ? true : false;
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "P requestId: " << self->reqId_
                                                              << " is ready for kv-cache publish, stop flag is "
                                                              << isReqStopped);
            PROF(INFO, Domain("Communication").Resource(self->reqId_.c_str()).Event("sendReqToD"));
            self->constructOneResponseCallBack_ = nullptr;
            DecodeParameters decodeParams;
            self->BuildDecodeParameters(response, decodeParams);
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                       "[P Node] Start to get dTargetAddr before send decode msg. prefillNodeAddr is "
                           << self->prefillNodeAddr_ << ", singleLLMPrefillReqHandlerId is " << self->pInstanceId_
                           << ", dTargetAddr is " << dTargetAddr);
            std::string errMsg;
            if (isReqStopped || !GrpcCommunicationMng::GetInstance().SendDecodeRequest(
                decodeParams, dTargetAddr, self->reqId_, errMsg)) {
                if (isReqStopped) {
                    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "[P Node] requestId:  "<< self->reqId_ <<
                        " has stop, do not send to D and coordinator.");
                    self->stopReqSet_.erase(self->reqId_);
                    return;
                }
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "[P Node] Failed to send decode msg to decode node:" << errMsg
                    << " requestId: " << self->reqId_);
                if (!self->AtomicReadWriteFinish(expect)) {
                    return;
                }
                self->SendResponseInfo(httplib::StatusCode::InternalServerError_500,
                    HttpRestResource::WrapperJson("[P Node] Send D request failed:" + errMsg,
                    g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
                self->cv.notify_one();
                return;
            }
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[P Node] Send request to success. requestId: " << self->reqId_);
            PROF(INFO, Domain("Communication").Resource(self->reqId_.c_str()).Event("sendReqToDSucc"));
            self->ProcessResponseStream(response->isEos);
            if (!self->AtomicReadWriteFinish(expect)) {
                return;
            }
            self->cv.notify_one();
        } else {
            self->constructOneResponseCallBack_ = nullptr;
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                UNKNOWN_ERROR), "[P Node] Unknown transfer flag. requestId: " << self->reqId_);
            if (!self->AtomicReadWriteFinish(expect)) {
                return;
            }
            self->SendResponseInfo(httplib::StatusCode::InternalServerError_500,
                HttpRestResource::WrapperJson("[P Node] Unknown transfer flag",
                g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)));
            self->cv.notify_one();
        }
    };
}

// 发送响应信息
void SingleLLMPrefillReqHandler::SendResponseInfo(int code, const std::string &responseStr,
                                                  [[maybe_unused]] bool needMetricsCollect)
{
    // set failed request prometheus metrics
    if (needMetricsCollect && code != httplib::StatusCode::OK_200) {
        ProcessFailedResponsePrometheusMetrics();
    }
    SendResponse(code, responseStr);
}

void SingleLLMPrefillReqHandler::SendResponse(int code, const std::string& responseStr)
{
    HttpRestResource::ResponseJsonBody(ctx, code, responseStr);
}

void SingleLLMPrefillReqHandler::SendResponseStream(bool isEnd, const std::string& responseStr)
{
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Send first token for requestId: " << reqId_ << " which is "
        << (isEnd ? "" : "not") << " the end");
    OrderedJson respBody;
    respBody["reqId"] = this->reqId_;
    respBody["isStream"] = this->streamMode_;
    if (this->streamMode_ || isEnd) {
        respBody["isLastResp"] = isEnd;
        respBody["output"] = responseStr;
    }
    HttpRestResource::ResponseJsonBody(ctx, httplib::StatusCode::OK_200, respBody.dump());
    PROF(INFO, Domain("Communication").Resource(reqId_.c_str()).Event("prefillRes"));
}

void SingleLLMPrefillReqHandler::BuildDecodeParameters(ResponseSPtr response, DecodeParameters& params)
{
    params.set_pnodeaddr(prefillNodeAddr_);
    params.set_pinstanceid(pInstanceId_);
    params.set_reqid(reqId_);
    params.set_msgtype(msgType_);
    params.set_isstream(streamMode_);
    auto prof = PROF(L2, Domain("Communication").Resource(reqId_.c_str()).SpanStart("BuildDecodeParameters"));
    PROF(prof.Attr("InstanceId", pInstanceId_));

    // adaptor recompute
    oriReqTokenLen = std::min(oriReqTokenLen, inputTokens_.size());
    for (size_t i = 0; i < oriReqTokenLen ; i++) {
        params.add_tokens(inputTokens_[i]);
    }
    PROF(prof.NumArrayAttr("tokens", &inputTokens_[0] + 0, &inputTokens_[0] + oriReqTokenLen));

    for (size_t i = oriReqTokenLen; i < inputTokens_.size() ; i++) {
        params.add_firsttoken(inputTokens_[i]);
    }
    PROF(prof.NumArrayAttr("first_tokens", &inputTokens_[0] + oriReqTokenLen, &inputTokens_[0] + inputTokens_.size()));

    for (int64_t token : firstToken_) {
        params.add_firsttoken(token);
    }
    PROF(prof.NumArrayAttr("firsttoken", firstToken_.begin(), firstToken_.end()));

    params.set_prefilltokennum(firstToken_.size());

    for (std::string outputName: inferParam_->outputNames) {
        params.add_outputnames(outputName);
    }
    PROF(prof.ArrayAttr("outputnames", inferParam_->outputNames.begin(), inferParam_->outputNames.end(),
        [](decltype(prof)* pColl, decltype(inferParam_->outputNames.begin()) item) -> void {
            pColl->Attr("name", *item);
        }));

    params.set_batchsize(inferParam_->batchSize);
    // The number of tokens needed for inference at node d
    if (inferParam_->isMaxNewTokensSetByUser) {
        params.set_maxnewtoken(inferParam_->maxNewTokens);
        PROF(prof.Attr("maxnewtoken", inferParam_->maxNewTokens));
    } else {
        params.set_maxnewtoken(0);
        PROF(prof.Attr("maxnewtoken", 0));
    }
    params.set_truncate(inferParam_->truncate);
    params.set_tools(inferParam_->tools);
    params.set_toolchoice(inferParam_->toolChoice);
    params.set_loraid(request_->loraId);
    params.set_returnfulltext(inferParam_->returnFullText);
    params.set_decoderinputdetails(inferParam_->decoderInputDetails);
    params.set_modelname(modelName_);
    params.set_details(inferParam_->showDetails);
    params.set_id(request_->requestId);
    params.set_usetoolcall(inferParam_->useToolsCall);
    
    PROF(prof.Attr("truncate", inferParam_->truncate));
    PROF(prof.Attr("tools", inferParam_->tools));
    PROF(prof.Attr("toolchoice", inferParam_->toolChoice));
    PROF(prof.Attr("loraid", request_->loraId));
    PROF(prof.Attr("returnfulltext", inferParam_->returnFullText));
    PROF(prof.Attr("decoderinputdetails", inferParam_->decoderInputDetails));
    PROF(prof.Attr("modelname", modelName_));
    PROF(prof.Attr("details", inferParam_->showDetails));
    PROF(prof.Attr("id", request_->requestId));
    PROF(prof.Attr("usetoolscall", inferParam_->useToolsCall));

    auto prefillE2EStartTime = GetMetrics().sysE2eStartingTime;
    auto durationInMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        prefillE2EStartTime.time_since_epoch());
    uint64_t e2eStartTimeNs = static_cast<uint64_t>(durationInMicroseconds.count());
    params.set_e2estarttime(e2eStartTimeNs);
    PROF(prof.Attr("e2estarttime", e2eStartTimeNs));

    std::vector<std::vector<int64_t>> blockTables = response->responseContents[0].srcBlockTable;
    for (const auto& blockTable : blockTables) {
        auto *blockIds = params.add_blocktable();
        for (int64_t block : blockTable) {
            blockIds->add_blockid(block);
        }
    }
    if (!blockTables.empty()) {
        PROF(prof.NumArrayAttr("blocktable", blockTables[0].begin(), blockTables[0].end()));
    }
    std::vector<uint64_t> dpInstanceIds = {
        static_cast<unsigned int>(response->responseContents[0].singleLLMPrefillReqHandlerId)};
    for (uint64_t dpId : dpInstanceIds) {
        params.add_dpinstanceids(dpId);
    }
    PROF(prof.NumArrayAttr("dpinstanceids", dpInstanceIds.begin(), dpInstanceIds.end()));

    params.set_prevdecodeindex(0);
    if (!inferParam_->prevDecodeIndex.empty()) {
        params.set_prevdecodeindex(inferParam_->prevDecodeIndex.begin()->second);
        PROF(prof.Attr("prevdecodeindex", inferParam_->prevDecodeIndex.begin()->second));
    }
    params.set_currentdecodeindex(0);
    if (!inferParam_->currentDecodeIndex.empty()) {
        params.set_currentdecodeindex(inferParam_->currentDecodeIndex.begin()->second);
        PROF(prof.Attr("currentdecodeindex", inferParam_->currentDecodeIndex.begin()->second));
    }
    params.set_preoutputtokennum(inferParam_->preOutputTokenNum);

    params.set_postsingletext("");
    if (!inferParam_->postSingleText.empty()) {
        params.set_postsingletext(inferParam_->postSingleText.begin()->second);
        PROF(prof.Attr("postsingletext", inferParam_->postSingleText.begin()->second));
    }
    if (inferParam_->userInputId.has_value()) {
        params.mutable_inputid()->set_value(inferParam_->userInputId.value());
        PROF(prof.Attr("inputid", inferParam_->userInputId.value()));
    }
    params.mutable_textinput()->set_value(inferParam_->textInput);
    PROF(prof.Attr("textinput", inferParam_->textInput));

    BuildSamplingParametersFirst(params);
    BuildSamplingParametersNext(params);
    BuildInferParameters(params);
    BuildMetricsParameters(params);
}

void SingleLLMPrefillReqHandler::BuildSamplingParametersFirst(DecodeParameters& params)
{
    if (request_->temperature.has_value()) {
        params.mutable_samplingparams()->mutable_temperature()->set_value(request_->temperature.value());
    }
    if (request_->topK.has_value()) {
        params.mutable_samplingparams()->mutable_topk()->set_value(request_->topK.value());
    }
    if (request_->topP.has_value()) {
        params.mutable_samplingparams()->mutable_topp()->set_value(request_->topP.value());
    }
    if (request_->typicalP.has_value()) {
        params.mutable_samplingparams()->mutable_typicalp()->set_value(request_->typicalP.value());
    }
    if (request_->doSample.has_value()) {
        params.mutable_samplingparams()->mutable_dosample()->set_value(request_->doSample.value());
    }
    if (request_->seed.has_value()) {
        params.mutable_samplingparams()->mutable_seed()->set_value(request_->seed.value());
    }
    if (request_->repetitionPenalty.has_value()) {
        params.mutable_samplingparams()->mutable_repetitionpenalty()->set_value(request_->repetitionPenalty.value());
    }
    if (request_->watermark.has_value()) {
        params.mutable_samplingparams()->mutable_watermark()->set_value(request_->watermark.value());
    }
}

void SingleLLMPrefillReqHandler::BuildSamplingParametersNext(DecodeParameters& params)
{
    if (request_->frequencyPenalty.has_value()) {
        params.mutable_samplingparams()->mutable_frequencypenalty()->set_value(request_->frequencyPenalty.value());
    }
    if (request_->presencyPenalty.has_value()) {
        params.mutable_samplingparams()->mutable_presencypenalty()->set_value(request_->presencyPenalty.value());
    }
    if (request_->stopTokenIds.has_value()) {
        for (int64_t token : request_->stopTokenIds.value()) {
            params.mutable_samplingparams()->mutable_stoptokenids()->add_value(token);
        }
    }
    if (request_->stopStrings.has_value()) {
        params.mutable_samplingparams()->mutable_stopstrings()->set_value(request_->stopStrings.value());
    }
    if (request_->stopStrList.has_value()) {
        for (std::string strVal : request_->stopStrList.value()) {
            params.mutable_samplingparams()->mutable_stopstrings()->add_list(strVal);
        }
    }
    if (request_->includeStopStrInOutput.has_value()) {
        params.mutable_samplingparams()->mutable_includestopstrinoutput()
            ->set_value(request_->includeStopStrInOutput.value());
    }
    if (request_->skipSpecialTokens.has_value()) {
        params.mutable_samplingparams()->mutable_skipspecialtokens()->set_value(request_->skipSpecialTokens.value());
    }
    if (request_->ignoreEos.has_value()) {
        params.mutable_samplingparams()->mutable_ignoreeos()->set_value(request_->ignoreEos.value());
    }
    if (request_->logprobs.has_value()) {
        params.mutable_samplingparams()->mutable_logprobs()->set_value(request_->logprobs.value());
    }
    if (request_->topLogprobs.has_value()) {
        params.mutable_samplingparams()->mutable_toplogprobs()->set_value(request_->topLogprobs.value());
    }
    if (inferParam_->enableThinking.has_value()) {
        params.mutable_samplingparams()->mutable_enablethinking()->set_value(inferParam_->enableThinking.value());
    }
}

void SingleLLMPrefillReqHandler::BuildInferParameters(DecodeParameters& params)
{
    // infer parameter
    params.mutable_inferparams()->set_priority(request_->priority);
    params.mutable_inferparams()->set_timeout(inferParam_->timeout);
}

void SingleLLMPrefillReqHandler::BuildMetricsParameters(DecodeParameters &params)
{
    params.mutable_metrics()->set_firsttokencost(metrics.firstTokenCost);
    params.mutable_metrics()->set_lasttokencost(metrics.lastTokenCost);
    for (size_t time : metrics.decodeTime) {
        params.mutable_metrics()->add_decodetime(time);
    }
    *params.mutable_metrics()->mutable_batchsize() = {metrics.batchSize.begin(), metrics.batchSize.end()};
    *params.mutable_metrics()->mutable_queuewaittime() = {metrics.queueWaitTime.begin(), metrics.queueWaitTime.end()};
    *params.mutable_metrics()->mutable_prefixcachedtokennums() = {
        metrics.prefixCachedTokenNums.begin(), metrics.prefixCachedTokenNums.end()};
    params.mutable_metrics()->set_callbackindex(metrics.callbackIndex);
}

void SingleLLMPrefillReqHandler::ProcessResponseStream(bool isEnd)
{
    // post a empty message to coordinator which would be ignored by coordinator
    // because coordinator need to record the first token
    if (respStr_.empty()) {
        std::string invalidResponse = "";
        SendResponseStream(isEnd, invalidResponse);
    }
    std::string totalMsg;
    while (!respStr_.empty()) {
        std::string msgTmp = respStr_.front();
        respStr_.pop();
        if (msgTmp.empty()) {
            continue;
        }
        totalMsg += msgTmp;
    }
    SendResponseStream(isEnd, totalMsg);
}

bool SingleLLMPrefillReqHandler::AtomicReadWriteFinish(bool &expect)
{
    auto ret = isFinish_.compare_exchange_strong(expect, true,
        std::memory_order_release, std::memory_order_relaxed);
    if (!ret) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_WARNING), "Has finished before! requestId: " << this->reqId_);
    } else {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Set finished flag. requestId: " << this->reqId_);
    }
    return ret;
}
} // namespace mindie_llm