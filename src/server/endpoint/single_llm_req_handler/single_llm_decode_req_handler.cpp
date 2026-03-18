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
 
#include <map>
#include "httplib.h"
#include "grpc_communication_mng.h"
#include "dmi_role.h"
#include "prometheus_metrics.h"
#include "log.h"
#include "common_util.h"
#include "config_manager_impl.h"
#include "health_checker.h"
#include "single_llm_decode_req_handler.h"

using ordered_json = nlohmann::ordered_json;

namespace mindie_llm {
SingleLLMDecodeReqHandler::SingleLLMDecodeReqHandler(ReqCtxPtr &ctx,
                                                     std::shared_ptr<DResultEventDispatcher> &dResultDispatcher,
                                                     std::shared_ptr<GrpcContext> &grpcContext)
    : SingleLLMReqHandlerBase(ctx), dResultDispatcher_(dResultDispatcher), grpcContext_(grpcContext)
{
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[SingleLLMDecodeReqHandler] Init SingleLLMDecodeReqHandler. DmiServerInfo is "
        << grpcContext_->GetDmiServerInfo().ToString());
    pNodeAddr_ = grpcContext_->GetDmiServerInfo().pNodeAddr;
    reqId_ = grpcContext_->GetDecodeParams().reqid();

    tritonReqId_ = grpcContext_->GetProtocalTyep() == MsgType::MSG_TYPE_TRITON ?
        grpcContext_->GetTritonTextInfo().userSepcId : "";
    reqType_ =  InferReqType::REQ_DECODE;
}

SingleLLMDecodeReqHandler::~SingleLLMDecodeReqHandler() {}

bool SingleLLMDecodeReqHandler::GetContextJsonBody(InferParamSPtr inputParam, RequestSPtr request,
                                                   std::vector<int64_t> &reqTokens, std::vector<int64_t> &respTokens)
{
    pInstanceId_ = grpcContext_->GetDecodeParams().pinstanceid();

    inputParam->batchSize = grpcContext_->GetDecodeParams().batchsize();
    inputParam->showDetails = grpcContext_->GetDecodeParams().details();
    if (grpcContext_->GetDecodeParams().maxnewtoken() == 0) {
        const ScheduleConfig &scheduleParam = GetScheduleConfig();
        inputParam->maxNewTokens = scheduleParam.maxIterTimes;
    } else {
        inputParam->maxNewTokens = grpcContext_->GetDecodeParams().maxnewtoken();
    }
    inputParam->streamMode = grpcContext_->GetDecodeParams().isstream();
    inputParam->returnFullText = grpcContext_->GetDecodeParams().returnfulltext();
    inputParam->decoderInputDetails = grpcContext_->GetDecodeParams().decoderinputdetails();
    request->requestId = grpcContext_->GetDecodeParams().id();
    inputParam->tools = grpcContext_->GetDecodeParams().tools();
    inputParam->toolChoice = grpcContext_->GetDecodeParams().toolchoice();
    request->loraId = grpcContext_->GetDecodeParams().loraid();
    inputParam->model_ = grpcContext_->GetDecodeParams().modelname();
    inputParam->prevDecodeIndex = {{SPECIAL_SEQ_ID_PRESET, grpcContext_->GetDecodeParams().prevdecodeindex()}};
    inputParam->currentDecodeIndex = {{SPECIAL_SEQ_ID_PRESET, grpcContext_->GetDecodeParams().currentdecodeindex()}};
    inputParam->preOutputTokenNum = grpcContext_->GetDecodeParams().preoutputtokennum();
    inputParam->respStreamStr = {{SPECIAL_SEQ_ID_PRESET, grpcContext_->GetDecodeParams().postsingletext()}};
    inputParam->useToolsCall = grpcContext_->GetDecodeParams().usetoolcall();
    // PD: derive isChatReq from msgType so decode uses reasoning/tool-call parsing (bfcl-simple etc.)
    const uint32_t msgType = grpcContext_->GetDecodeParams().msgtype();
    inputParam->isChatReq = (msgType == static_cast<uint32_t>(MsgType::MSG_TYPE_OPENAI) ||
                            msgType == static_cast<uint32_t>(MsgType::MSG_TYPE_VLLM_OPENAI));

    int64_t tokenNum = grpcContext_->GetDecodeParams().tokens_size();
    for (int64_t i = 0; i < tokenNum; i++) {
        reqTokens.emplace_back(grpcContext_->GetDecodeParams().tokens()[i]);
    }
    tokenNum = grpcContext_->GetDecodeParams().firsttoken_size();
    for (int64_t i = 0; i < tokenNum; i++) {
        respTokens.emplace_back(grpcContext_->GetDecodeParams().firsttoken()[i]);
    }
    tokenNum = grpcContext_->GetDecodeParams().outputnames_size();
    for (int64_t i = 0; i < tokenNum; i++) {
        inputParam->outputNames.emplace_back(grpcContext_->GetDecodeParams().outputnames()[i]);
    }

    GetContextSamplingParamsFirst(request);
    GetContextSamplingParamsNext(inputParam, request);
    GetContextInferParams(inputParam, request);
    if (grpcContext_->GetDecodeParams().inputid().has_value()) {
        inputParam->userInputId = grpcContext_->GetDecodeParams().inputid().value();
    }
    if (grpcContext_->GetDecodeParams().textinput().has_value()) {
        inputParam->textInput = grpcContext_->GetDecodeParams().textinput().value();
    }
    // D节点，已输出的token数，注意，需要考虑P节点推理出来的第一个token，因此需要-1
    inputParam->outputLenOffset = respTokens.size() - grpcContext_->GetDecodeParams().prefilltokennum();
    GetContextMetrics();
    return true;
}

void SingleLLMDecodeReqHandler::GetContextSamplingParamsFirst(RequestSPtr request)
{
    const auto& samplingParams = grpcContext_->GetDecodeParams().samplingparams();
    if (samplingParams.temperature().has_value()) {
        request->temperature = samplingParams.temperature().value();
    }
    if (samplingParams.topk().has_value()) {
        request->topK = samplingParams.topk().value();
    }
    if (samplingParams.topp().has_value()) {
        request->topP = samplingParams.topp().value();
    }
    if (samplingParams.typicalp().has_value()) {
        request->typicalP = samplingParams.typicalp().value();
    }
    if (samplingParams.dosample().has_value()) {
        request->doSample = samplingParams.dosample().value();
    }
    if (samplingParams.seed().has_value()) {
        request->seed = samplingParams.seed().value();
    }
    if (samplingParams.repetitionpenalty().has_value()) {
        request->repetitionPenalty = samplingParams.repetitionpenalty().value();
    }
}

void SingleLLMDecodeReqHandler::GetContextSamplingParamsNext(InferParamSPtr inputParam, RequestSPtr request)
{
    const auto& samplingParams = grpcContext_->GetDecodeParams().samplingparams();
    if (samplingParams.watermark().has_value()) {
        request->watermark = samplingParams.watermark().value();
    }
    if (samplingParams.frequencypenalty().has_value()) {
        request->frequencyPenalty = samplingParams.frequencypenalty().value();
    }
    if (samplingParams.presencypenalty().has_value()) {
        request->presencyPenalty = samplingParams.presencypenalty().value();
    }
    GetContextSamplingStopWords(inputParam, request);
    if (samplingParams.skipspecialtokens().has_value()) {
        request->skipSpecialTokens = samplingParams.skipspecialtokens().value();
    }
    if (samplingParams.ignoreeos().has_value()) {
        request->ignoreEos = samplingParams.ignoreeos().value();
    }
    if (samplingParams.logprobs().has_value()) {
        request->logprobs = samplingParams.logprobs().value();
    }
    if (samplingParams.toplogprobs().has_value()) {
        request->topLogprobs = samplingParams.toplogprobs().value();
    }
    if (samplingParams.enablethinking().has_value()) {
        inputParam->enableThinking = samplingParams.enablethinking().value();
    }
}

void SingleLLMDecodeReqHandler::GetContextSamplingStopWords(InferParamSPtr inputParam, RequestSPtr request)
{
    const auto& samplingParams = grpcContext_->GetDecodeParams().samplingparams();
    int64_t tokenNum = samplingParams.stoptokenids().value_size();
    if (tokenNum > 0) {
        request->stopTokenIds = std::vector<TokenId>();
    }
    for (int64_t i = 0; i < tokenNum; i++) {
        request->stopTokenIds.value().emplace_back(
            samplingParams.stoptokenids().value().at(i));
    }
    if (samplingParams.stopstrings().has_value()) {
        request->stopStrings = samplingParams.stopstrings().value();
    }
    request->windowSize = 0;
    int64_t listNum = samplingParams.stopstrings().list_size();
    if (listNum > 0) {
        request->stopStrList = std::vector<std::string>();
    }
    for (int64_t i = 0; i < listNum; i++) {
        auto s = samplingParams.stopstrings().list().at(i);
        request->stopStrList.value().emplace_back(s);
        request->windowSize = std::max(request->windowSize, (uint32_t)s.size());
    }
    if (samplingParams.includestopstrinoutput().has_value()) {
        request->includeStopStrInOutput =
            samplingParams.includestopstrinoutput().value();
    }
    if (inputParam->streamMode && request->HasStopWords()) {
        inputParam->prevDecodeIndex[SPECIAL_SEQ_ID_PRESET] = inputParam->preOutputTokenNum;
        inputParam->currentDecodeIndex[SPECIAL_SEQ_ID_PRESET] = inputParam->preOutputTokenNum;
    }
}

void SingleLLMDecodeReqHandler::GetContextInferParams(InferParamSPtr inputParam, RequestSPtr request)
{
    request->priority = grpcContext_->GetDecodeParams().inferparams().priority();
    inputParam->timeout = grpcContext_->GetDecodeParams().inferparams().timeout();
}

void SingleLLMDecodeReqHandler::GetContextMetrics()
{
    metrics.firstTokenCost = grpcContext_->GetDecodeParams().metrics().firsttokencost();
    metrics.lastTokenCost = grpcContext_->GetDecodeParams().metrics().lasttokencost();
    uint32_t decodeTimeNum = static_cast<uint32_t>(grpcContext_->GetDecodeParams().metrics().decodetime_size());
    for (uint32_t i = 0; i < decodeTimeNum; i++) {
        metrics.decodeTime.push_back(grpcContext_->GetDecodeParams().metrics().decodetime()[i]);
    }
    auto& grpcBatchSize = grpcContext_->GetDecodeParams().metrics().batchsize();
    metrics.batchSize.reserve(metrics.batchSize.size() + grpcBatchSize.size());
    metrics.batchSize.insert(metrics.batchSize.end(), grpcBatchSize.begin(), grpcBatchSize.end());
    auto& grpcQueueWaitTime = grpcContext_->GetDecodeParams().metrics().queuewaittime();
    metrics.queueWaitTime.reserve(metrics.queueWaitTime.size() + grpcQueueWaitTime.size());
    metrics.queueWaitTime.insert(metrics.queueWaitTime.end(), grpcQueueWaitTime.begin(), grpcQueueWaitTime.end());
    auto& grpcPrefixCachedTokenNums = grpcContext_->GetDecodeParams().metrics().prefixcachedtokennums();
    metrics.prefixCachedTokenNums.reserve(metrics.prefixCachedTokenNums.size() + grpcPrefixCachedTokenNums.size());
    metrics.prefixCachedTokenNums.insert(metrics.prefixCachedTokenNums.end(), grpcPrefixCachedTokenNums.begin(),
        grpcPrefixCachedTokenNums.end());
    metrics.callbackIndex = grpcContext_->GetDecodeParams().metrics().callbackindex();
}

bool SingleLLMDecodeReqHandler::GetContextRequestId(std::string& requestId)
{
    requestId = grpcContext_->GetDecodeParams().reqid();
    return !requestId.empty();
}

void SingleLLMDecodeReqHandler::UpdateInferRequest([[maybe_unused]] const std::vector<int64_t> &reqTokens,
                                                   [[maybe_unused]] const int64_t &oriReqTokenLen,
                                                   RequestSPtr request)
{
    // 配置 id，p节点地址， d请求类型， blocktable
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
               "[SingleLLMDecodeReqHandler] Generate infer request success. singleLLMPrefillReqHandlerId is "
                   << pInstanceId_ << " , pNodeAddr is " << pNodeAddr_);
    request->reqType = InferReqType::REQ_DECODE;
    request->pInstanceId = pInstanceId_;
    request->srcBlockTable = grpcContext_->GetDmiServerInfo().kvCacheInfo.blockTable;
    std::vector<uint64_t> dpInsts;
    for (uint64_t dpRank : grpcContext_->GetDmiServerInfo().kvCacheInfo.dpInstanceIds) {
        // D节点pull kv时候，作为dpinstanceid去索引p节点信息
        // kvCacheInfo.dpInstanceIds 面放的是dp rank id
        // dpinstance = pInstanceId_ * 10000 + dpRank
        uint64_t dpInst = pInstanceId_ * 10000 + dpRank;
        dpInsts.emplace_back(dpInst);
    }
    request->dpInstanceIds = dpInsts;

    request->ignoreEos = grpcContext_->GetDecodeParams().samplingparams().ignoreeos().value();
}

void SingleLLMDecodeReqHandler::ProcessGrpcReq(RequestSPtr request, [[maybe_unused]] const std::string &inputId,
                                               prefillAndDecodeCommunication::DecodeRequestResponse &response)
{
    metrics.startingTime = std::chrono::steady_clock::now();
    metrics.lastTokenTime = std::chrono::steady_clock::now();
    metrics.isPrefill = false;
    SetBackManagerCallBack(request);
    Status status = GetInferInstance()->Process(request);
    if (!status.IsOk()) {
        std::string strMsg = "Failed forward in for infer engine.";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), strMsg);
        response.set_errormessage(strMsg);
        response.set_isvaliddecodeparameters(true);
        SendDError(strMsg);
        constructOneResponseCallBack_ = nullptr;
        return;
    }
}

bool GetPullKVFlag(ResponseSPtr &response, uint16_t &pullKVFlag)
{
    if (response == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            PULL_KV_ERROR), "Failed to get pull kv result or ptr invalid.");
        return false;
    }
    pullKVFlag = response->responseContents[0].pdErrorCode;    // pdErrorCode
    return true;
}

void SingleLLMDecodeReqHandler::SetBackManagerCallBack(RequestSPtr request)
{
    // 使用weak_ptr避免因request与handler之间循环引用导致的内存泄漏
    std::weak_ptr<SingleLLMDecodeReqHandler> weakSelf = shared_from_this();

    request->serverResponseCallback_ = [weakSelf](ResponseSPtr response) {
        auto self = weakSelf.lock();
        if (!self) {
            return;
        }
        std::unique_lock lock(self->decodeCbMutex);
        if (self == nullptr || response == nullptr || self->isFinish_.load()) { return; }
        self->reqId_ = response->reqId;
        // 场景1: decode response is pull, 通知p释放kvcache
        if (response->transferStatusFlag == TransferStatusType::PULL_KV_COMPLETE) {
            uint16_t pullKVFlag = 0;
            if (!GetPullKVFlag(response, pullKVFlag)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to get pull kv flag.");
                return;
            }
            int64_t pdErrorCode = response->responseContents[0].pdErrorCode;
            if (pullKVFlag == static_cast<uint16_t>(PULL_KV_FAIL_IRREVERSIBLY)) {
                DmiRole::GetInstance()->ModifyPullKVFailId(self->pInstanceId_);
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    PULL_KV_ERROR), "Pull kv failed irreversibly. " << self->reqId_);
                self->SendDError("Pull kv failed irreversibly.");
                self->isFinish_.store(true);
                self->constructOneResponseCallBack_ = nullptr;
            } else if (pullKVFlag == static_cast<uint16_t>(PULL_KV_FAIL_REVERSIBLY)) {
                DmiRole::GetInstance()->ModifyPullKVFailId(self->pInstanceId_);
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    PULL_KV_ERROR), "Pull kv failed reversibly. " << self->reqId_);
                HealthChecker::GetInstance().EnqueueErrorMessage("MIE05E01001B", "SINGLE_LLM_DECODE_REQ_HANDLER");
                self->SendDError("Pull kv failed reversibly.");
                self->isFinish_.store(true);
                self->constructOneResponseCallBack_ = nullptr;
            }

            if (!self->SendKvRelease(self->reqId_)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to send kv release. requestId: " << self->reqId_);
            }
            return;
        }

        // 场景2 重计算场景
        if (response->transferStatusFlag == TransferStatusType::RECOMPUTED_TRIGGERED) {
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Do recompute. requestId: " << self->reqId_);
            if (!self->dmiReCompBuildMethod_) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Recompute is not supported. requestId: " << self->reqId_);
                self->SendDError("Try to do recompute but it is not supported.");
                self->isFinish_.store(true);
                self->constructOneResponseCallBack_ = nullptr;
                return;
            }
            std::string reCompBody = self->dmiReCompBuildMethod_(self->respTokens_);
            self->ResponseReCompute(reCompBody);
            self->isFinish_.store(true);
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }

        // 场景3：produce token
        std::vector<BestNTokens> bestNTokens{};
        if (!self->ParseTokensFromResponse(response, bestNTokens)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to get token id from infer response for repId is " << self->reqId_);
            self->SendDError("Failed to get token id from infer response.");
            self->isFinish_.store(true);
            self->cv.notify_one();
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }
        self->SetMetricParams(response);

        if (!self->streamMode_) {
            std::for_each(bestNTokens.begin(), bestNTokens.end(), [self](auto token) {
                self->respTokens_.emplace_back(token);
            });
            if (!response->isEos) {
                return;
            }
        }
        RespBodyQueue responseJsonQueue;
        std::vector<BestNTokens>& tokenIds = self->streamMode_ ? bestNTokens : self->respTokens_;
        if (self->constructOneResponseCallBack_ == nullptr ||
            !self->constructOneResponseCallBack_(response, tokenIds, responseJsonQueue)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                       "Failed to get json response for requestId: " << self->reqId_);
            self->SendDError("Failed to decode token id to json response.");
            self->isFinish_.store(true);
            self->cv.notify_one();
            self->constructOneResponseCallBack_ = nullptr;
            return;
        }
        while (!responseJsonQueue.empty()) {
            std::string msgTmp = responseJsonQueue.front();
            responseJsonQueue.pop();
            if (!self->streamMode_) {
                self->SendResponse(httplib::StatusCode::OK_200, msgTmp);
            } else {
                auto eos = response->isEos;
                if (!responseJsonQueue.empty()) {
                    eos = false;
                }
                self->SendResponseStream(eos, msgTmp);
            }
        }

        if (response->isEos) {
            PrometheusMetrics::GetInstance()->ResponseNumberCount();
            uint64_t e2eStartTimeMs = self->grpcContext_->GetDecodeParams().e2estarttime();
            if (e2eStartTimeMs != 0) {
                auto e2eStartTime = std::chrono::system_clock::time_point(std::chrono::microseconds(e2eStartTimeMs));
                uint64_t decodeE2ETime = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now() - e2eStartTime)
                                             .count();
                PrometheusMetrics::GetInstance()->E2EObserve(decodeE2ETime);
            }
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                "D-Node ResponseCallback begin to send last response. requestId: " << self->reqId_);
            PROF(INFO, Domain("Communication").Resource(self->reqId_.c_str()).Event("decodeRes"));
            self->isFinish_.store(true);
            self->cv.notify_one();
            self->constructOneResponseCallBack_ = nullptr;
        }
    };
}

// 发送响应信息
void SingleLLMDecodeReqHandler::SendResponseInfo([[maybe_unused]] int code,
                                                 [[maybe_unused]] const std::string &responseStr,
                                                 [[maybe_unused]] bool needMetricsCollect)
{
}

// 发送非流式响应
// D节点中，所有的正常响应无论是否流式都走长连接返回，错误都都走SendResponseInfo，因此这里的code用不到
// 非流式的响应以last token的形式返回
void SingleLLMDecodeReqHandler::SendResponse(int /* code */, const std::string& responseStr)
{
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Wrap non-streaming D response in chunk for reqId is " << reqId_);
    if (responseStr.empty()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_WARNING), "The response message is empty. requestId: " << reqId_);
        return;
    }
    std::string msg = "";
    DResultWrapParam param { responseStr, "lastData:", this->reqId_, this->tritonReqId_ };
    DResultEventDispatcher::WrapChunkedDResponse(msg, param);
    this->SendDResult(msg, this->reqId_);
}

// 发送流式响应
void SingleLLMDecodeReqHandler::SendResponseStream(bool isEnd, const std::string& responseStr)
{
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Wrap chunk D response for " << reqId_);
    if (responseStr.empty()) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_WARNING), "The response message is empty. requestId: " << reqId_);
        return;
    }
    std::string msg = "";
    DResultWrapParam param { responseStr, isEnd ? "lastData:" : "data:", this->reqId_, this->tritonReqId_ };
    DResultEventDispatcher::WrapChunkedDResponse(msg, param);
    this->SendDResult(msg, this->reqId_);
}

bool SingleLLMDecodeReqHandler::SendKvRelease([[maybe_unused]] const std::string& reqId)
{
    (void)reqId;

    prefillAndDecodeCommunication::RequestId id;
    id.set_reqid(reqId_);
    return GrpcCommunicationMng::GetInstance().SendKvReleaseMsg(id, pNodeAddr_);
}

void SingleLLMDecodeReqHandler::ResponseReCompute(const std::string& body)
{
    auto reComputeSpan = PROF(INFO, Domain("Communication")
        .Resource(this->reqId_.c_str()).SpanStart("decodeReCompute"));
    std::string msg = "";
    DResultWrapParam param { body, "retry:", this->reqId_, this->tritonReqId_ };
    DResultEventDispatcher::WrapChunkedDResponse(msg, param);
    this->SendDResult(msg, this->reqId_);
    PROF(reComputeSpan.SpanEnd());
}

void SingleLLMDecodeReqHandler::SendDResult(const std::string &msg, std::string reqId)
{
    if (this->dResultDispatcher_ == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
            CHECK_ERROR), "The dResultDispatcher_ is nullptr. requestId: " << this->reqId_);
        this->cv.notify_one();
    } else {
        this->dResultDispatcher_->SendEvent(msg, false, reqId);
    }
}

void SingleLLMDecodeReqHandler::SendDError(const std::string &errorMsg)
{
    ProcessFailedResponsePrometheusMetrics();
    std::string msg = "";
    DResultWrapParam param { errorMsg, "error:", this->reqId_, this->tritonReqId_ };
    DResultEventDispatcher::WrapChunkedDResponse(msg, param);
    this->SendDResult(msg, this->reqId_);
}
} // namespace mindie_llm