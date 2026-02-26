/*
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
#include "single_req_infer_interface_base.h"
#include <atomic>
#include <chrono>
#include <codecvt>
#include <cstdlib>
#include <thread>
#include <utility>
#include "endpoint_def.h"
#include "http_metrics.h"
#include "http_rest_resource.h"
#include "infer_tokenizer.h"
#include "parse_protocol.h"
#include "prometheus_metrics.h"
#include "random_generator.h"
#include "parameters_checker.h"
#include "config_manager_impl.h"
#include "safe_io.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {

std::atomic<bool> StopServiceOption::stopServiceFlag{false};

// Count the number of created and deleted SingleReqInferInterfaceBase objects for debug.
std::atomic<size_t> g_numCreateSingleReqInferInterfaceBase = 0;
std::atomic<size_t> g_numDeleteSingleReqInferInterfaceBase = 0;

SingleReqInferInterfaceBase::SingleReqInferInterfaceBase(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : singleLLMReqHandlerBase_{singleLLMReqHandlerBase}, isReCompute_{isReCompute}, loraConfigs_{loraConfigs}
{
    if (singleLLMReqHandlerBase == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, INIT_ERROR),
                   "singleLLMReqHandlerBase cannot be null");
        // 设置一个默认的requestId，避免后续代码崩溃
        requestId_ = GetNextInferRequestId();
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                   "Create INVALID singleLLMReqHandlerBase #" << (++g_numCreateSingleReqInferInterfaceBase));
        return;  // 提前返回
    }
    inputParam = std::make_shared<InferParam>();
    if (singleLLMReqHandlerBase_->GetContextRequestId(requestId_)) { // get request id from header "req-id" field
        ULOG_DEBUG(
            SUBMODLE_NAME_ENDPOINT,
            "[SingleReqInferInterfaceBase::SingleReqInferInterfaceBase] Set requestId from singleLLMReqHandlerBase "
            "GetContextRequestId, requestId is "
                << requestId_);
    } else {
        requestId_ = GetNextInferRequestId();
        ULOG_DEBUG(
            SUBMODLE_NAME_ENDPOINT,
            "[SingleReqInferInterfaceBase::SingleReqInferInterfaceBase] Set InputId from generator, requestId is "
                << requestId_);
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
               "Create SingleReqInferInterfaceBase #" << (++g_numCreateSingleReqInferInterfaceBase));
}

RequestIdNew SingleReqInferInterfaceBase::GetRequestId() { return requestId_; }

SingleReqInferInterfaceBase::~SingleReqInferInterfaceBase()
{
    singleLLMReqHandlerBase_ = nullptr;
    try {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                   "Delete SingleReqInferInterfaceBase #" << (++g_numDeleteSingleReqInferInterfaceBase));
        if (inputParam->userInputId.has_value() && userInputIdMap_.Count(inputParam->userInputId.value()) != 0) {
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
                      "End user id " << inputParam->userInputId.value() << ", request id " << requestId_);
            userInputIdMap_.Erase(inputParam->userInputId.value());
        }
    } catch (...) {
        // Forbid to throw exceptions in destructor function
    }
}

template <typename ValueType>
void DumpSequenceCache(std::map<uint64_t, ValueType> &map, uint64_t sourceKey, uint64_t targetKey,
                       std::function<void(const ValueType &source, ValueType &target)> dumpFunc)
{
    if (map.count(sourceKey) == 0 || sourceKey == targetKey) {
        return;
    }
    dumpFunc(map[sourceKey], map[targetKey]);
}

template <typename T> inline void DumpValue(const T &source, T &target) { target = source; }

template <typename T> inline void DumpVector(const std::vector<T> &source, std::vector<T> &target)
{
    target = std::vector<T>(source.begin(), source.end());
}

void SingleReqInferInterfaceBase::ClearUnusedDecodeCache(const std::set<uint64_t> &usedSeqIds)
{
    std::set<uint64_t> unUsedSeqIds;
    for (const auto &[seqId, _] : respTokenMap) {
        // Beam Search: do not meet eos but be eliminated
        static_cast<void>(_);
        if (eosMap[seqId] == InferStatusType::ITERATION_CONTINUE && usedSeqIds.count(seqId) == 0) {
            unUsedSeqIds.insert(seqId);
        }
    }
    for (const auto seqId : unUsedSeqIds) {
        respTokenMap.erase(seqId);
        eosMap.erase(seqId);
        probesMap.erase(seqId);
        finishReasonMap.erase(seqId);
        postTokenIdMap.erase(seqId);
        logprobsTokensMap.erase(seqId);
        logprobsMap.erase(seqId);
        pickedLogprobMap.erase(seqId);
        fullTextMap.erase(seqId);
        postSingleTokenMap.erase(seqId);
        reasoningContentFullTextMap.erase(seqId);
        reasoningContentStreamMap.erase(seqId);
        toolsCallObjectMap.erase(seqId);
        toolsCallContentMap.erase(seqId);
        toolsCallObjectStreamMap.erase(seqId);
        toolsCallContentStreamMap.erase(seqId);
        skipCurrentRoundMap.erase(seqId);
        windowText.erase(seqId);
        parsingContentFlag.erase(seqId);
        for (auto &[_, textVec] : windowJsonTextVec) {
            static_cast<void>(_);
            textVec.erase(seqId);
        }
        for (auto &[_, textOut] : windowJsonTextOut) {
            static_cast<void>(_);
            textOut.erase(seqId);
        }
        truncationIdMap.erase(seqId);
        inputParam->prevDecodeIndex.erase(seqId);
        inputParam->currentDecodeIndex.erase(seqId);
        inputParam->postSingleText.erase(seqId);
        inputParam->respStreamStr.erase(seqId);
    }
}

// copy historical data of parent seqence to child seqence
void SingleReqInferInterfaceBase::DumpDecodeCache(const std::map<uint64_t, std::vector<uint64_t>> &currentSeqIds)
{
    for (const auto &[parentSeqId, seqIds] : currentSeqIds) {
        for (size_t i = 0; i < seqIds.size(); ++i) {
            uint64_t source = i == 0 ? parentSeqId : seqIds[0];
            uint64_t target = i == 0 ? seqIds[0] : seqIds[i];

            DumpSequenceCache<std::vector<int64_t>>(respTokenMap, source, target, DumpVector<int64_t>);
            DumpSequenceCache<InferStatusType>(eosMap, source, target, DumpValue<InferStatusType>);
            DumpSequenceCache<double>(probesMap, source, target, DumpValue<double>);
            DumpSequenceCache<std::string>(finishReasonMap, source, target, DumpValue<std::string>);
            DumpSequenceCache<std::vector<int64_t>>(postTokenIdMap, source, target, DumpVector<int64_t>);
            DumpSequenceCache<std::vector<int64_t>>(logprobsTokensMap, source, target, DumpVector<int64_t>);
            DumpSequenceCache<std::vector<float>>(logprobsMap, source, target, DumpVector<float>);
            DumpSequenceCache<std::vector<float>>(pickedLogprobMap, source, target, DumpVector<float>);
            DumpSequenceCache<std::string>(fullTextMap, source, target, DumpValue<std::string>);
            DumpSequenceCache<std::vector<int64_t>>(postSingleTokenMap, source, target, DumpVector<int64_t>);
            DumpSequenceCache<int64_t>(truncationIdMap, source, target, DumpValue<int64_t>);
            DumpSequenceCache<std::string>(reasoningContentFullTextMap, source, target, DumpValue<std::string>);
            DumpSequenceCache<std::string>(reasoningContentStreamMap, source, target, DumpValue<std::string>);
            DumpSequenceCache<Json>(toolsCallObjectMap, source, target, DumpValue<Json>);
            DumpSequenceCache<std::string>(toolsCallContentMap, source, target, DumpValue<std::string>);
            DumpSequenceCache<Json>(toolsCallObjectStreamMap, source, target, DumpValue<Json>);
            DumpSequenceCache<std::string>(toolsCallContentStreamMap, source, target, DumpValue<std::string>);
            DumpSequenceCache<bool>(skipCurrentRoundMap, source, target, DumpValue<bool>);
            DumpSequenceCache<std::string>(windowText, source, target, DumpValue<std::string>);
            DumpSequenceCache<std::pair<bool, bool>>(parsingContentFlag, source, target,
                                                     DumpValue<std::pair<bool, bool>>);
            for (auto &[_, textVec] : windowJsonTextVec) {
                static_cast<void>(_);
                DumpSequenceCache<std::string>(textVec, source, target, DumpValue<std::string>);
            }
            for (auto &[_, textOut] : windowJsonTextOut) {
                static_cast<void>(_);
                DumpSequenceCache<std::string>(textOut, source, target, DumpValue<std::string>);
            }
            DumpSequenceCache<uint32_t>(inputParam->prevDecodeIndex, source, target, DumpValue<uint32_t>);
            DumpSequenceCache<uint32_t>(inputParam->currentDecodeIndex, source, target, DumpValue<uint32_t>);
            DumpSequenceCache<std::string>(inputParam->postSingleText, source, target, DumpValue<std::string>);
            DumpSequenceCache<std::string>(inputParam->respStreamStr, source, target, DumpValue<std::string>);
        }
    }
}

void SingleReqInferInterfaceBase::ConvertTokenToMap(const std::vector<BestNTokens> &decodeResult)
{
    if (std::count_if(decodeResult.begin(), decodeResult.end(),
                      [](auto res) { return res.finishReason == InferStatusType::ILLEGAL_INPUT; }) != 0) {
        // abort all sequence once there is an illegal input.
        for (auto &[seqId, finishReason] : eosMap) {
            if (finishReason == InferStatusType::ITERATION_CONTINUE) {
                finishReason = InferStatusType::ABORT;
                respTokenMap[seqId].emplace_back(-1);
            }
        }
        return;
    }
    // if there is no parent id then use SPECIAL_SEQ_ID_PRESET as parent id
    if (respTokenMap.count(SPECIAL_SEQ_ID_PRESET) != 0) {
        std::map<uint64_t, std::vector<uint64_t>> seqIdMap = {{SPECIAL_SEQ_ID_PRESET, {}}};
        for (const auto &item : decodeResult) {
            seqIdMap[SPECIAL_SEQ_ID_PRESET].emplace_back(item.seqId);
        }
        this->DumpDecodeCache(seqIdMap);
    }

    std::map<uint64_t, std::vector<uint64_t>> stoppedSeqIdMap;
    std::map<uint64_t, std::vector<uint64_t>> beamSeqIdMap;
    std::set<uint64_t> usedSeqIds;

    for (const auto &item : decodeResult) {
        // The ids larger than 2 ** 63 - 1 has been reserved for the eos sequences of beam search.
        if (item.seqId > (std::numeric_limits<SequenceId>::max())) {
            if (stoppedSeqIdMap.count(item.parentSeqId) == 0) {
                stoppedSeqIdMap[item.parentSeqId] = {};
            }
            stoppedSeqIdMap[item.parentSeqId].emplace_back(item.seqId);
        } else {
            if (beamSeqIdMap.count(item.parentSeqId) == 0) {
                beamSeqIdMap[item.parentSeqId] = {};
            }
            beamSeqIdMap[item.parentSeqId].emplace_back(item.seqId);
        }
        usedSeqIds.insert(item.seqId);
    }

    this->DumpDecodeCache(stoppedSeqIdMap);
    this->DumpDecodeCache(beamSeqIdMap);
    this->ClearUnusedDecodeCache(usedSeqIds);

    for (const auto &tokens : decodeResult) {
        if (this->respTokenMap.count(tokens.seqId) == 0) {
            this->respTokenMap[tokens.seqId] = {};
        }
        for (const auto &token : tokens.tokens) {
            this->respTokenMap[tokens.seqId].emplace_back(token);
        }
        for (const auto &logprobs : tokens.logprobs) {
            this->logprobsMap[tokens.seqId].emplace_back(logprobs);
        }
        for (const auto &logprobsIds : tokens.logprobsTokens) {
            this->logprobsTokensMap[tokens.seqId].emplace_back(logprobsIds);
        }
        this->pickedLogprobMap[tokens.seqId].emplace_back(tokens.logprob);
        this->eosMap[tokens.seqId] = tokens.finishReason;
        this->probesMap[tokens.seqId] = tokens.cumLogprobs;
        this->truncationIdMap[tokens.seqId] = tokens.truncationIndex;
    }
}

void SingleReqInferInterfaceBase::DecodeProcess(prefillAndDecodeCommunication::DecodeRequestResponse &response) noexcept
{
    std::string strMsg = "";
    inputParam = std::make_shared<InferParam>();
    request_ = std::make_shared<Request>(RequestIdNew{"0"});

    // decode struct to infer paramerter and request
    if (!singleLLMReqHandlerBase_->GetContextJsonBody(inputParam, request_, reqTokens_,
                                                      respTokenMap[SPECIAL_SEQ_ID_PRESET])) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   "Failed to parse context to infer param for id" << requestId_);
        strMsg = "Failed to parse context to infer param.";
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(strMsg, g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }
    PROF(INFO, Domain("Communication").Resource(request_->requestId.c_str()).Event("decodeReq"));
    this->model = inputParam->model_;
    oriReqTokenLen_ = static_cast<size_t>(reqTokens_.size());
    this->requestId_ = request_->requestId;
    for (auto token : respTokenMap[SPECIAL_SEQ_ID_PRESET]) {
        this->postTokenIdMap[SPECIAL_SEQ_ID_PRESET].emplace_back(token);
    }

    // 请求相关参数存入request 和 inputParam
    if (!GenerateInferRequest(strMsg)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, INFERENCE_GENERATE_REQUEST_ERROR),
                   "Failed to generate infer request. The requestId is " << requestId_);
        strMsg = "Failed to generate infer request";
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::FailedDependency_424,
            HttpRestResource::WrapperJson(strMsg, g_exceptionInfo.at(httplib::StatusCode::FailedDependency_424)));
        return;
    }

    // set stream mode
    singleLLMReqHandlerBase_->SetStreamMode(inputParam->streamMode);

    // set decode callback
    auto self = shared_from_this();
    singleLLMReqHandlerBase_->SetConstructOneResponseCallBack([self](const ResponseSPtr &response,
                                                                     const std::vector<BestNTokens> &tokenIdList,
                                                                     RespBodyQueue &jsonStrings) {
        self->ConvertTokenToMap(tokenIdList);

        if (response->isEos) {
            size_t tokenNum = std::accumulate(self->respTokenMap.cbegin(), self->respTokenMap.cend(), 0,
                                              [](size_t sum, const auto &item) { return sum + item.second.size(); });
            PrometheusMetrics::GetInstance()->ResponseOutputTokenHistogramCollect(tokenNum);
            PrometheusMetrics::GetInstance()->ResponseOutputTokenCount(tokenNum);
            PROF(INFO, Domain("Request")
                           .Resource(self->requestId_.c_str())
                           .Metric("replyTokenSize", tokenNum)
                           .Event("httpRes"));
        }

        auto copyResp = response;
        return self->BuildResponseJson(copyResp, tokenIdList, jsonStrings);
    });

    // forward
    singleLLMReqHandlerBase_->ProcessGrpcReq(request_, requestId_, response);
}

void SingleReqInferInterfaceBase::Process() noexcept
{
    singleLLMReqHandlerBase_->GetMetrics().e2eStartingTime = std::chrono::steady_clock::now();
    auto httpReqSpan = PROF(INFO, Domain("Request").SpanStart("httpReq"));
    PrometheusMetrics::GetInstance()->RequestNumberCount();
    std::string strMsg = "";
    
    // parse request context into JSON: RequestContext => reqJsonBody
    if (!singleLLMReqHandlerBase_->GetContextJsonBody(this->reqJsonBody_)) {
        strMsg = "Failed to parse request context to json";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   strMsg << ". The requestId is " << requestId_);
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(strMsg,
                g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }

    if (StopServiceOption::stopServiceFlag.load()) {
        strMsg = "The service has been stopped";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   strMsg << ". The requestId is " << requestId_);
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::ServiceUnavailable_503,
            HttpRestResource::WrapperJson(strMsg,
                                          g_exceptionInfo.at(httplib::StatusCode::ServiceUnavailable_503)));
        return;
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    uint64_t timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    auto tid = std::this_thread::get_id();
    uint64_t threadHash = std::hash<std::thread::id>{}(tid);

    timestamp ^= (threadHash & 0xFFFFFFFF);
    timestamp_ = timestamp;

    // validate input text and encode it into tokens
    if (!ValidateAndPrepareReqToken(reqJsonBody_, strMsg, timestamp)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   strMsg << ". The requestId is " << requestId_);
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(strMsg, g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }
    PROF(httpReqSpan.Resource(requestId_.c_str()).SpanEnd());

    // construct inference request
    request_ = std::make_shared<Request>(requestId_);

    // parse JSON contents into inference request and inference parameters: reqJsonBody => request & inputParam
    if (!SetupInferParams(request_, strMsg)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, INFERENCE_GENERATE_REQUEST_ERROR),
                   strMsg << ". The requestId is " << requestId_);
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(strMsg, g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)));
        return;
    }
    if (!GenerateInferRequest(strMsg)) {
        strMsg = "Failed to generate infer request";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, INFERENCE_GENERATE_REQUEST_ERROR),
                   strMsg << ". The requestId is " << requestId_);
        singleLLMReqHandlerBase_->SendResponseInfo(httplib::StatusCode::FailedDependency_424,
            HttpRestResource::WrapperJson(strMsg, g_exceptionInfo.at(httplib::StatusCode::FailedDependency_424)));
        return;
    }

    InferReqType reqType = singleLLMReqHandlerBase_->GetReqType();
    if (reqType != InferReqType::REQ_DECODE) {
        if (PrometheusMetrics::GetInstance() != nullptr) {
            PrometheusMetrics::GetInstance()->RequestInputTokenHistogramCollect(reqTokens_.size());
            PrometheusMetrics::GetInstance()->RequestInputTokenCount(reqTokens_.size());
        }
    }

    // set stream mode
    singleLLMReqHandlerBase_->SetStreamMode(inputParam->streamMode);

    // set decode callback
    auto self = shared_from_this();
    singleLLMReqHandlerBase_->SetConstructOneResponseCallBack([self, reqType](const ResponseSPtr &response,
                                                       const std::vector<BestNTokens> &tokenIdList,
                                                       RespBodyQueue &jsonStrings) {
        self->ConvertTokenToMap(tokenIdList);
        if (response->isEos) {
            size_t tokenNum = std::accumulate(self->respTokenMap.cbegin(), self->respTokenMap.cend(), 0,
                                              [](size_t sum, const auto &item) { return sum + item.second.size(); });
            HttpMetrics::GetInstance().CollectStatisticsRequest(self);

            PROF(INFO, Domain("Request")
                           .Resource(self->requestId_.c_str())
                           .Metric("replyTokenSize", tokenNum)
                           .Event("DecodeEnd"));
        }

        auto respCopy = response;
        return self->BuildResponseJson(respCopy, tokenIdList, jsonStrings, self->timestamp_);
    });

    // forward
    singleLLMReqHandlerBase_->GetMetrics().startingTime = std::chrono::steady_clock::now();
    singleLLMReqHandlerBase_->Process(request_, requestId_, self->timestamp_);

    PROF(INFO, Domain("Request").Resource(requestId_.c_str()).Event("httpRes"));
}

bool SingleReqInferInterfaceBase::GenerateInferRequest(std::string &msg) noexcept
{
    std::vector<int64_t> inferTokens = reqTokens_; // 在D节点和重计算场景，本次推理的请求id需要加上respTokens
    if (respTokenMap.size() == 1) {
        inferTokens.insert(inferTokens.end(), respTokenMap.begin()->second.begin(), respTokenMap.begin()->second.end());
    }
    request_->input_ids = inferTokens;
    request_->input_token_num = static_cast<int64_t>(reqTokens_.size());

    // supplement inference request and inference parameters
    inputParam->model_ = this->model;
    request_->requestId = requestId_;
    singleLLMReqHandlerBase_->UpdateInferRequest(inferTokens, oriReqTokenLen_, request_);
    singleLLMReqHandlerBase_->UpdateInferParam(request_, inputParam);
    singleLLMReqHandlerBase_->DumpInferParam(request_);
    if (inputParam->maxNewTokens > 0) {
        int maxOutputLen = inputParam->maxNewTokens - static_cast<int>(inputParam->outputLenOffset);
        if (maxOutputLen < 0) {
            msg = "MaxNewTokens is less than already generated tokens. The requestId is " + std::string(requestId_);
            MINDIE_LLM_LOG_ERROR(msg);
            return false;
        }
        request_->maxOutputLen = static_cast<uint64_t>(maxOutputLen);
    }
    return true;
}

void SingleReqInferInterfaceBase::Stop() noexcept
{
    OrderedJson reqBody;
    std::string errorMessage = "";
    if (!singleLLMReqHandlerBase_->GetContextJsonBody(reqBody)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   "Failed to parse request context to json. The requestId is " << requestId_);
        errorMessage = "Failed to parse context to json body.";
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(errorMessage,
                                          g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)),
            false);
        return;
    }

    if (!JsonParse::JsonContainItemWithType(reqBody, "id", OrderedJson::value_t::string, errorMessage)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, JSON_PARSE_ERROR),
                   "Failed to parse json body");
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(errorMessage,
                                          g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)),
            false);
        return;
    }

    std::string stopReqId = reqBody["id"];
    std::regex pattern("^[a-zA-Z0-9_-]{1,256}$");
    if (!std::regex_match(stopReqId, pattern)) {
        errorMessage = std::string("The id can contain only digits, letters, underscores(_), hyphens(-) and ")
            .append("no more than ").append(std::to_string(MAX_INPUT_ID_LENGTH)).append(" words in length.");
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), errorMessage);
        singleLLMReqHandlerBase_->SendResponseInfo(httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(errorMessage,
                                          g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)),
            false);
        return;
    }
    if (stopReqId != requestId_) {
        stopReqId = userInputIdMap_.Get(stopReqId).value_or(stopReqId);
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
              "Begin stop infer request. requestId: " << stopReqId << ", requestIdString: " << requestId_);
    singleLLMReqHandlerBase_->InsertStopRequest(stopReqId);
    RequestIdNew requestId{stopReqId};
    Status status = GetInferInstance()->ControlRequest(requestId, OperationV2::STOP);
    if (status.StatusCode() == Error::Code::OK) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Stop success. requestId: " << stopReqId);
        singleLLMReqHandlerBase_->SendResponseInfo(httplib::StatusCode::OK_200, reqBody.dump(), false);
        return;
    }

    if (status.StatusCode() == Error::Code::NOT_FOUND) {
        errorMessage = std::string("RequestId is ").append(stopReqId).append(" not found.");
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT,
                  GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), errorMessage);
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::NotFound_404,
            HttpRestResource::WrapperJson(errorMessage,
                g_exceptionInfo.at(httplib::StatusCode::NotFound_404)), false);
        return;
    }

    if (status.StatusCode() == Error::Code::INVALID_ARG) {
        errorMessage = std::string("To stop status is invalid. RequestId is ").append(stopReqId);
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), errorMessage);
        singleLLMReqHandlerBase_->SendResponseInfo(
            httplib::StatusCode::UnprocessableContent_422,
            HttpRestResource::WrapperJson(errorMessage,
                                          g_exceptionInfo.at(httplib::StatusCode::UnprocessableContent_422)),
            false);
        return;
    }
    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
               GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
               "requestId: " << stopReqId << " failed. " << status.StatusMsg());
    errorMessage = "Stop request failed, internal failed.";
    singleLLMReqHandlerBase_->SendResponseInfo(
        httplib::StatusCode::InternalServerError_500,
        HttpRestResource::WrapperJson(errorMessage, g_exceptionInfo.at(httplib::StatusCode::InternalServerError_500)),
        false);
    return;
}

bool SingleReqInferInterfaceBase::ProcessResponseSingle(ResponseSPtr response, const uint64_t &timestamp) noexcept
{
    if (!response->isEos) {
        return true;
    }
    for (const auto &[seqId, respTokens] : respTokenMap) {
        finishReasonMap[seqId] = GetFinishReasonStr(eosMap[seqId]);
        auto ret = PostProcess(respTokens, fullTextMap[seqId], seqId, false, true, 0, 0, timestamp);
        if (!ret) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
                       "Convert token to text output failed. The requestId is " << requestId_);
            return false;
        }
    }

    for (const auto &[seqId, _] : respTokenMap) {
        // if function call is used, the finish reason is tool_calls
        static_cast<void>(_);
        if (toolsCallObjectMap.find(seqId) != toolsCallObjectMap.end() && !toolsCallObjectMap[seqId].is_null()) {
            finishReasonMap[seqId] = "tool_calls";
        }
        auto &fullText = fullTextMap[seqId];
        std::string errorMsg = "";
        std::u16string utf16 = GetU16Str(fullText, &errorMsg);
        if (!errorMsg.empty()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
                       errorMsg);
            return false;
        }
        if (truncationIdMap.count(seqId) != 0 && truncationIdMap[seqId] < 0 &&
            utf16.length() >= static_cast<size_t>(std::abs(truncationIdMap[seqId]))) {
            fullText = TransformTruncation(utf16, 0, utf16.length() + truncationIdMap[seqId], &errorMsg);
            if (!errorMsg.empty()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                           GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
                           errorMsg);
                return false;
            }
        }

        postTokenIdMap[seqId] = std::move(respTokenMap[seqId]);
        if (inputParam->returnFullText) {
            fullText = inputParam->textInput + " " + fullText;
        }
    }
    return true;
}

bool SingleReqInferInterfaceBase::GetUniqueSequenceId(uint64_t &seqId)
{
    if (respTokenMap.size() != 1) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to get sequence id which must exist and be unique, got count of sequence id "
                       << respTokenMap.size());
        return false;
    }
    seqId = respTokenMap.begin()->first;
    return true;
}

bool SingleReqInferInterfaceBase::PushLatestCache(std::string &errMsg)
{
    StreamCache cache;
    cache.probesMap = std::map<uint64_t, double>{probesMap.begin(), probesMap.end()};
    cache.fullTextMap = std::map<uint64_t, std::string>{fullTextMap.begin(), fullTextMap.end()};
    cache.reasoningContentFullTextMap =
        std::map<uint64_t, std::string>{reasoningContentFullTextMap.begin(), reasoningContentFullTextMap.end()};
    cache.reasoningContentStreamMap =
        std::map<uint64_t, std::string>{reasoningContentStreamMap.begin(), reasoningContentStreamMap.end()};
    cache.eosMap = std::map<uint64_t, InferStatusType>{eosMap.begin(), eosMap.end()};
    cache.postTokenIdMap = std::map<uint64_t, std::vector<int64_t>>{postTokenIdMap.begin(), postTokenIdMap.end()};
    cache.postSingleText =
        std::map<uint64_t, std::string>{inputParam->postSingleText.begin(), inputParam->postSingleText.end()};
    cache.postSingleTokenMap =
        std::map<uint64_t, std::vector<int64_t>>{postSingleTokenMap.begin(), postSingleTokenMap.end()};
    cache.finishReasonMap = std::map<uint64_t, std::string>{finishReasonMap.begin(), finishReasonMap.end()};
    cache.logprobsTokensMap =
        std::map<uint64_t, std::vector<int64_t>>{logprobsTokensMap.begin(), logprobsTokensMap.end()};
    cache.logprobsMap = std::map<uint64_t, std::vector<float>>{logprobsMap.begin(), logprobsMap.end()};
    cache.pickedLogprobMap = std::map<uint64_t, std::vector<float>>{pickedLogprobMap.begin(), pickedLogprobMap.end()};
    cache.prevDecodeIndex =
        std::map<uint64_t, uint32_t>{inputParam->prevDecodeIndex.begin(), inputParam->prevDecodeIndex.end()};
    cache.currentDecodeIndex =
        std::map<uint64_t, uint32_t>{inputParam->currentDecodeIndex.begin(), inputParam->currentDecodeIndex.end()};
    cache.parsingContentFlag =
        std::map<uint64_t, std::pair<bool, bool>>{parsingContentFlag.begin(), parsingContentFlag.end()};

    uint64_t seqId = 0;
    if (GetUniqueSequenceId(seqId)) {
        cache.curTokenNum = respTokenMap[seqId].size();
    }

    for (const auto &[seqId, _] : cache.eosMap) {
        static_cast<void>(_);
        cache.u16TokenText[seqId] = GetU16Str(cache.postSingleText[seqId], &errMsg);
        if (!errMsg.empty()) {
            return false;
        }
        cache.canOutput[seqId] = false;
    }

    streamCache.emplace_back(cache);
    return true;
}

bool SingleReqInferInterfaceBase::ProcessStreamCacheTruncationId(std::string &errMsg)
{
    if (streamCache.empty()) {
        return true;
    }
    StreamCache latestData = streamCache[streamCache.size() - 1];
    for (const auto &[seqId, truncationIndex] : this->truncationIdMap) {
        int64_t restTruncation = truncationIndex;

        for (auto rit = streamCache.rbegin(); rit != streamCache.rend(); ++rit) {
            if (rit->u16TokenText.count(seqId) == 0 || truncationIndex >= 0 || restTruncation >= 0) {
                continue;
            }
            std::u16string &u16Text = rit->u16TokenText[seqId];
            if (static_cast<size_t>(std::abs(restTruncation)) > u16Text.length()) {
                restTruncation = restTruncation + static_cast<int64_t>(u16Text.length());
                rit->canOutput.erase(seqId); // sequence has ended before
                continue;
            }
            // text truncation of current sequence will occur in this cache round
            // eos/fullText/finishReason of current sequence should be updated to the latest data
            // caches of sequence with current sequence id after this round should be updated to data of this round
            rit->postSingleText[seqId] =
                TransformTruncation(u16Text, 0, static_cast<int64_t>(u16Text.length()) + restTruncation, &errMsg);
            if (!errMsg.empty()) {
                return false;
            }
            rit->eosMap[seqId] = latestData.eosMap[seqId];
            rit->fullTextMap[seqId] = latestData.fullTextMap[seqId];
            rit->finishReasonMap[seqId] = latestData.finishReasonMap[seqId];
            rit->canOutput[seqId] = true;
            for (auto subRit = rit - 1; rit != streamCache.rbegin() && subRit >= streamCache.rbegin(); --subRit) {
                subRit->probesMap[seqId] = rit->probesMap[seqId];
                subRit->fullTextMap[seqId] = rit->fullTextMap[seqId];
                subRit->reasoningContentFullTextMap[seqId] = rit->reasoningContentFullTextMap[seqId];
                subRit->reasoningContentStreamMap[seqId] = rit->reasoningContentStreamMap[seqId];
                subRit->eosMap[seqId] = rit->eosMap[seqId];
                subRit->postTokenIdMap[seqId] = rit->postTokenIdMap[seqId];
                subRit->postSingleText[seqId] = rit->postSingleText[seqId];
                subRit->postSingleTokenMap[seqId] = rit->postSingleTokenMap[seqId];
                subRit->finishReasonMap[seqId] = rit->finishReasonMap[seqId];
                subRit->logprobsTokensMap[seqId] = rit->logprobsTokensMap[seqId];
                subRit->logprobsMap[seqId] = rit->logprobsMap[seqId];
                subRit->pickedLogprobMap[seqId] = rit->pickedLogprobMap[seqId];
                subRit->prevDecodeIndex[seqId] = rit->prevDecodeIndex[seqId];
                subRit->currentDecodeIndex[seqId] = rit->currentDecodeIndex[seqId];
                subRit->parsingContentFlag[seqId] = rit->parsingContentFlag[seqId];
                subRit->canOutput.erase(seqId); // sequence has ended before
            }
            restTruncation = restTruncation + static_cast<int64_t>(u16Text.length());
        }
    }
    return true;
}

bool SingleReqInferInterfaceBase::ProcessStreamCacheWindowSize()
{
    // if sum length of last few text >= window size, mark output flag of sequences before to true
    std::map<uint64_t, int64_t> windowSize{};
    std::map<uint64_t, bool> ignoreAndMarkTrue{};
    for (auto rit = streamCache.rbegin(); rit != streamCache.rend(); ++rit) {
        for (const auto &[seqId, _] : rit->canOutput) {
            static_cast<void>(_);
            if (ignoreAndMarkTrue[seqId]) {
                if (rit->canOutput.count(seqId) != 0) {
                    rit->canOutput[seqId] = true;
                }
                continue;
            }
            if (windowSize.count(seqId) == 0) {
                windowSize[seqId] = request_->windowSize;
            }
            windowSize[seqId] -= static_cast<int64_t>(rit->u16TokenText[seqId].length());

            if (windowSize[seqId] <= 0) {
                ignoreAndMarkTrue[seqId] = true;
            }

            if (rit->eosMap[seqId] != InferStatusType::ITERATION_CONTINUE) {
                if (rit->canOutput.count(seqId) != 0) {
                    rit->canOutput[seqId] = true;
                }
                ignoreAndMarkTrue[seqId] = true;
            }
        }
    }
    return true;
}

bool SingleReqInferInterfaceBase::GetAvailableOutputCache(std::vector<StreamCache> &cacheArr)
{
    // Get cached output
    std::string errMsg = "";
    if (!PushLatestCache(errMsg) || !ProcessStreamCacheTruncationId(errMsg) || !ProcessStreamCacheWindowSize()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
                   "Failed to process stream cache. " << errMsg);
        return false;
    }
    if (streamCache.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            LOCAL_INVOKING_ERROR), "Nothing could be dumped in response for request");
        return false;
    }
    try {
        // If not using stop feature, do not need to filter output content
        if (!request_->HasStopWords()) {
            cacheArr.emplace_back(streamCache[0]);
            streamCache.erase(streamCache.begin());
            return true;
        }

        uint32_t popCount = 0;
        for (auto &item : streamCache) {
            // for this round of cache, exists a sequence that
            // can be returned or is already finished (not SCHEDULER_RESPONSE_CONTINUE)
            bool hasOutputReadyOrFinished =
                std::any_of(item.canOutput.begin(), item.canOutput.end(), [&item](auto canOut) {
                    return canOut.second || item.eosMap.at(canOut.first) != InferStatusType::ITERATION_CONTINUE;
                });
            if (hasOutputReadyOrFinished) {
                // if canOutput value of sequence is absent in cache means sequence has ended before
                // if canOutput is empty means cache of this round can be dropped
                if (!item.canOutput.empty()) {
                    cacheArr.emplace_back(item);
                }
                popCount++;
            } else {
                break;
            }
        }
        for (uint32_t i = 0; i < popCount; ++i) {
            if (streamCache[0].curTokenNum.has_value()) {
                inputParam->preOutputTokenNum = streamCache[0].curTokenNum.value();
            }
            streamCache.erase(streamCache.begin());
        }
    } catch (const std::out_of_range &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            LOCAL_INVOKING_ERROR), "Get available output failed for out of range exception: " << e.what());
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            LOCAL_INVOKING_ERROR), "Get available output failed: " << e.what());
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            LOCAL_INVOKING_ERROR), "Get available output failed");
    }
    return true;
}

bool SingleReqInferInterfaceBase::ProcessResponseStream(ResponseSPtr response,
    const std::vector<BestNTokens> &bestNTokens, RespBodyQueue &jsonObjs, const uint64_t &timestamp) noexcept
{
    for (auto &[_, textStr] : inputParam->postSingleText) {
        static_cast<void>(_);
        textStr.clear();
    }

    for (const auto &[seqId, finishReason] : eosMap) {
        if (eosMap[seqId] != InferStatusType::ITERATION_CONTINUE) {
            finishReasonMap[seqId] = GetFinishReasonStr(finishReason);
        }
    }

    bool requestEndFlag = false;
    requestEndFlag = response->isEos;

    auto &serverConfig = GetServerConfig();
    for (const auto &item : bestNTokens) {
        auto tempTokens = item.tokens;
        if (tempTokens.empty()) {
            continue;
        }
        std::string curRespTokenText{}; // single token text
        postSingleTokenMap[item.seqId] = item.tokens;
        postTokenIdMap[item.seqId].insert(postTokenIdMap[item.seqId].end(), tempTokens.begin(), tempTokens.end());
        auto ret = PostProcess(postTokenIdMap[item.seqId], curRespTokenText, item.seqId, true, requestEndFlag,
            inputParam->prevDecodeIndex[item.seqId], inputParam->currentDecodeIndex[item.seqId], timestamp);
        inputParam->postSingleText[item.seqId].append(curRespTokenText);
        if (!ret) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                LOCAL_INVOKING_ERROR), "Failed to post process.");
            return false;
        }
        // normal content output and reasoning content output is not empty at the same time
        if (updateIndexMap[item.seqId]) {
            inputParam->prevDecodeIndex[item.seqId] = inputParam->currentDecodeIndex[item.seqId];
            inputParam->currentDecodeIndex[item.seqId] = postTokenIdMap[item.seqId].size();
            if (serverConfig.fullTextEnabled) {
                inputParam->respStreamStr[item.seqId] += curRespTokenText;
            }
        }
        if (serverConfig.fullTextEnabled) {
            // the returned result is the same as last time if inputParam->postSingleText is empty
            inputParam->postSingleText[item.seqId] = inputParam->respStreamStr[item.seqId];
        }

        if (item.finishReason != InferStatusType::ITERATION_CONTINUE) {
            postTokenIdMap[item.seqId] = std::move(respTokenMap[item.seqId]);
            ret = PostProcess(postTokenIdMap[item.seqId], fullTextMap[item.seqId], item.seqId, false,
                requestEndFlag, 0, 0, timestamp);
            if (!ret) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    LOCAL_INVOKING_ERROR), "Failed to post process.");
                return false;
            }
            std::string errorMsg = "";
            std::u16string utf16Full = GetU16Str(fullTextMap[item.seqId], &errorMsg);
            if (truncationIdMap.count(item.seqId) != 0 && truncationIdMap[item.seqId] < 0 &&
                utf16Full.length() >= static_cast<size_t>(std::abs(truncationIdMap[item.seqId]))) {
                fullTextMap[item.seqId] = TransformTruncation(utf16Full, 0,
                    utf16Full.length() + truncationIdMap[item.seqId], &errorMsg);
            }
            if (!errorMsg.empty()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    LOCAL_INVOKING_ERROR), errorMsg);
                return false;
            }
            if (inputParam->returnFullText) {
                fullTextMap[item.seqId] = inputParam->textInput + " " + fullTextMap[item.seqId];
            }
        }
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[commonInfer:ProcessResponseStream] postSingleText is " <<
        inputParam->postSingleText[item.seqId] << ". tempTokens.size is " << tempTokens.size() << ".");
    }
    isEnd = response->isEos;
    SendStreamResponse(jsonObjs);
    return true;
}

Metrics &SingleReqInferInterfaceBase::GetMetrics()
{
    return singleLLMReqHandlerBase_->GetMetrics();
}

InferParam::ValidationContext SingleReqInferInterfaceBase::BuildValidationContext() const
{
    InferParam::ValidationContext ctx;
    // 端点能力（虚函数，子类覆写）
    ctx.endpoint = GetFeatureSupport();
    // 运行形态
    ctx.streamMode = inputParam->streamMode;
    ctx.isDmiMode = (GetServerConfig().inferMode == mindie_llm::INFER_MODE_DMI);
    ctx.isVllmEntrance = isVllmEntrance;
    // 请求参数
    if (inputParam != nullptr) {
        ctx.enableThinking = inputParam->enableThinking.value_or(false);
        ctx.reqUseToolsCall = inputParam->useToolsCall;
        ctx.reqUseBeamSearch = request_->useBeamSearch.value_or(false);
        ctx.reqLogprobs = request_->logprobs.value_or(false);
        ctx.reqTopLogprobsSet = request_->topLogprobs.has_value();
        ctx.reqStop = request_->stopStrings.has_value();
        ctx.reqStopTokenIds = request_->stopTokenIds.has_value();
        ctx.reqIncludeStopStrInOutput = request_->includeStopStrInOutput.has_value();
        uint32_t bestOfValue = request_->bestOf.value_or(1);
        uint32_t nValue = request_->n.value_or(1);
        // 检查bestof/n中是否有未设置的值，若有则将未设置的同步为另一个相同数值
        if (request_->bestOf.has_value() && !request_->n.has_value()) {
            nValue = bestOfValue;
        } else if (!request_->bestOf.has_value() && request_->n.has_value()) {
            bestOfValue = nValue;
        }
        ctx.reqBestOf = bestOfValue;
        ctx.reqN = nValue;
        ctx.reqTemperature = request_->temperature.value_or(0.0f);
    }
    // 服务/模型级能力（从 ServerConfig 的 pluginEnabled 获取）
    auto &serverConfig = GetServerConfig();
    ctx.pluginEnabled = serverConfig.pluginEnabled;
    ctx.deepseekEnabled = serverConfig.deepseekEnabled;
    return ctx;
}

bool SingleReqInferInterfaceBase::PostProcess(const std::vector<int64_t> &tokenIds, std::string &inferResult,
    const uint64_t &seqId, bool decodeOneToken, bool requestEndFlag, uint32_t prevDecodeIndexLocal,
    uint32_t currentDecodeIndexLocal, const uint64_t &timestamp) noexcept
{
    TokenizerProcessPool &pool = TokenizerProcessPool::GetInstance();
    auto copyTokens = tokenIds;
    Status status;
    auto decodeSpan = PROF(INFO, Domain("Request").Resource(requestId_.c_str()).SpanStart("decode"));
    if (decodeOneToken) {
        if (detokenizeExtraInfo[seqId].has_value()) {
            status = pool.DecodeOne(copyTokens, inferResult, prevDecodeIndexLocal, currentDecodeIndexLocal, timestamp,
                                    inputParam->useToolsCall, request_->skipSpecialTokens.value_or(false),
                                    requestEndFlag,
                                    {detokenizeExtraInfo[seqId].value().isCurrentToolNameSent,
                                     detokenizeExtraInfo[seqId].value().isCurrentArgumentSent,
                                     detokenizeExtraInfo[seqId].value().currentToolId, inputParam->isChatReq,
                                     inputParam->enableThinking, inputParam->toolCallsJson});
        } else {
            status = pool.DecodeOne(copyTokens, inferResult, prevDecodeIndexLocal, currentDecodeIndexLocal, timestamp,
                                    inputParam->useToolsCall, request_->skipSpecialTokens.value_or(false),
                                    requestEndFlag,
                                    {std::nullopt, std::nullopt, std::nullopt, inputParam->isChatReq,
                                    inputParam->enableThinking, inputParam->toolCallsJson});
        }
    } else {
        status = pool.Decode(copyTokens, inferResult, timestamp, inputParam->useToolsCall,
                             request_->skipSpecialTokens.value_or(false),
                             {std::nullopt, std::nullopt, std::nullopt, inputParam->isChatReq,
                             inputParam->enableThinking, inputParam->toolCallsJson});
    }
    PROF(decodeSpan.SpanEnd());
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
                   "Failed post process by " << status.StatusMsg());
        return false;
    }
    ParseDetokenizedOutput(inferResult, seqId, decodeOneToken);
    return true;
}

void SingleReqInferInterfaceBase::ParseDetokenizedOutput(std::string &inferResult,
    const uint64_t &seqId, const bool &decodeOneToken)
{
    // functioncall in stream mode exists skipping step
    TokenizerContents contents = JsonParse::ParseContentFromJson(inferResult);
    bool noContent =
        !contents.content.has_value() && !contents.reasoningContent.has_value() && !contents.toolCalls.has_value();
    // if tools have been called, then finish reason should be "tool_calls"
    if (toolsCalled.find(seqId) != toolsCalled.end()) {
        toolsCalled[seqId] = contents.toolCalls.has_value();
    } else {
        toolsCalled[seqId] = toolsCalled[seqId] || contents.toolCalls.has_value();
    }
    bool enableReasoningContent = inputParam->enableThinking.value_or(inputParam->isChatReq);
    skipCurrentRoundMap[seqId] = noContent;
    if (decodeOneToken == inputParam->streamMode) {
        parsingContentFlag[seqId] = std::make_pair(
            contents.reasoningContent.has_value(), contents.content.has_value());
    }
    inferResult = contents.content.value_or("");
    detokenizeExtraInfo[seqId] = contents.detokenizeStatus;
    if (!decodeOneToken) {
        toolsCallObjectMap[seqId] = contents.toolCalls.value_or(nullptr);
        toolsCallContentMap[seqId] = contents.content.value_or("");
        if (enableReasoningContent) {
            reasoningContentFullTextMap[seqId] = contents.reasoningContent.value_or("");
        }
    } else {
        toolsCallObjectStreamMap[seqId] = contents.toolCalls.value_or(nullptr);
        toolsCallContentStreamMap[seqId] = contents.content.value_or("");
        if (enableReasoningContent) {
            reasoningContentStreamMap[seqId] = contents.reasoningContent.value_or("");
            reasoningContentFullTextMap[seqId] += contents.reasoningContent.value_or("");
        }
        updateIndexMap[seqId] = contents.needUpdateIndex.value_or(true);
    }
    reasoningTokens[seqId] = contents.detokenizeStatus.has_value() ?
        contents.detokenizeStatus.value().reasoningTokens.value_or(-1) : -1;
}

bool SingleReqInferInterfaceBase::GetTokensFromInput(const std::string &input,
    std::vector<std::int64_t> &requestTokens, std::vector<std::int64_t> &responseTokens, std::string &errorMsg)
{
    std::istringstream iss(input);
    std::string token = "";
    uint64_t idx = 0;
    while (std::getline(iss, token, ',')) {
        try {
            if (idx == 0) {
                oriReqTokenLen_ = static_cast<uint64_t>(std::stoll(token));
                idx++;
                continue;
            }
            if (idx <= oriReqTokenLen_) {
                requestTokens.push_back(std::stoll(token));
            } else {
                responseTokens.push_back(std::stoll(token));
            }
            idx++;
        } catch (const std::exception &e) {
            errorMsg = e.what();
            return false;
        } catch (...) {
            errorMsg = "Unknown error occur when get token from input";
            return false;
        }
    }
    // 重计算场景
    inputParam->preOutputTokenNum = responseTokens.size();
    inputParam->outputLenOffset = inputParam->preOutputTokenNum; // maxNewTokens需要减去已输出的token数
    return true;
}

// decode one token corresponding to a specific logprob
bool SingleReqInferInterfaceBase::DecodeSingleToken(std::vector<int64_t> &tokenIds, std::string &output,
    const uint32_t &prevDecodeIndex, const uint32_t &currentDecodeIndex,
    const bool &skipSpecialTokens)
{
    std::string inferResult;
    std::string err;
    auto status =
        TokenizerProcessPool::GetInstance().DecodeOne(tokenIds, inferResult, prevDecodeIndex, currentDecodeIndex,
                                                      this->timestamp_, inputParam->useToolsCall, skipSpecialTokens);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to detokenize one single token for logprobs feature");
        return false;
    }
    try {
        Json resultJson = Json::parse(inferResult, CheckJsonDepthCallbackUlog);
        if (JsonParse::JsonContainItemWithType(resultJson, "content", Json::value_t::string, err) &&
            !resultJson["content"].get<std::string>().empty()) {
            output = resultJson["content"];
        } else if (JsonParse::JsonContainItemWithType(resultJson, "reasoning_content", Json::value_t::string, err) &&
                   !resultJson["reasoning_content"].get<std::string>().empty()) {
            output = resultJson["reasoning_content"];
        } else {
            output = "";
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Find no infer result from tokenizer");
        }
    } catch (Json::parse_error &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Parse one single infer result from tokenizer failed");
        return false;
    }
    return true;
}

Status SingleReqInferInterfaceBase::InsertPerfInfoIntoJson(nlohmann::ordered_json &body,
    const std::vector<int32_t> perfInfoTypeList,
    const std::vector<std::string> keyList)
{
    std::string error;
    auto &metrics = singleLLMReqHandlerBase_->GetMetrics();
    if (perfInfoTypeList.size() != keyList.size()) {
        error = "Invalid performance index";
        return Status(Error::Code::ERROR, error);
    }
    for (size_t i = 0; i < keyList.size(); ++i) {
        const std::vector<int64_t> *perfInfo = nullptr;
        switch (perfInfoTypeList[i]) {
            case PerfInfoType::PERF_BATCH_SZIE: perfInfo = &metrics.batchSize; break;
            case PerfInfoType::PERF_QUEUE_WAIT_TIME: perfInfo = &metrics.queueWaitTime; break;
            default: error = "Perfermance information type is not supported"; return Status(Error::Code::ERROR, error);
        }
        // Maybe json type is object or array
        if (body.is_object()) {
            body[keyList[i]] = *perfInfo;
        } else if (body.is_array()) {
            OrderedJson tempJsonObj;
            tempJsonObj[keyList[i]] = *perfInfo;
            body.emplace_back(tempJsonObj);
        } else {
            error = "Invalid json type to insert performance information";
            return Status(Error::Code::ERROR, error);
        }
    }
    return Status(Error::Code::OK, "Success");
}

bool SingleReqInferInterfaceBase::ParseChatTemplate(const nlohmann::ordered_json &jsonObj,
    std::string &error) const
{
    if (!jsonObj.contains("chat_template_kwargs")) {
        return true;
    }

    const auto &kwargs = jsonObj["chat_template_kwargs"];
    if (!kwargs.is_object()) {
        error = "chat_template_kwargs must be a JSON object";
        return false;
    }

    if (kwargs.contains("enable_thinking")) {
        try {
            inputParam->enableThinking = kwargs.at("enable_thinking").get<bool>();
        } catch (const nlohmann::json::exception &e) {
            std::string actual_type = kwargs["enable_thinking"].type_name();
            error = "enable_thinking must be boolean, got " + actual_type;
            return false;
        }
    }
    return true;
}

std::string SingleReqInferInterfaceBase::BuildReComputeInput()
{
    std::stringstream ssInputs;
    ssInputs << oriReqTokenLen_ << ",";
    StreamAppend(ssInputs, reqTokens_, oriReqTokenLen_);
    uint64_t seqId;
    if (GetUniqueSequenceId(seqId)) {
        size_t copyLen = this->respTokenMap[seqId].size();
        if (inputParam->streamMode && request_->HasStopWords()) {
            copyLen = std::min(copyLen, inputParam->preOutputTokenNum);
        }
        StreamAppend(ssInputs, this->respTokenMap[seqId], copyLen, true);
    }
    return ssInputs.str();
}
} // namespace mindie_llm