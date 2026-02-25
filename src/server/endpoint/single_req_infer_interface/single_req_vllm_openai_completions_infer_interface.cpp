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
#include "single_req_vllm_openai_completions_infer_interface.h"
#include "endpoint_def.h"
#include "parse_protocol.h"
#include "http_rest_resource.h"
#include "parameters_checker.h"
#include "infer_tokenizer.h"
#include "common_util.h"
#include "base64_util.h"
#include "config_manager_impl.h"
#include "safe_io.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
static constexpr double MAX_VLLM_OPENAI_REPETITION_PENALTY = 2.0;
static constexpr uint64_t MAX_VLLM_OPENAI_TOP_LOGPROBS = 5;
SingleReqVllmOpenAiCompletionsInferInterface::SingleReqVllmOpenAiCompletionsInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs}
{
}

static bool AssignLogprobs(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string keyLogprobs = "logprobs";
    return ParametersChecker::OptionalUint32JsonCheck(
        jsonObj, keyLogprobs, tmpReq->topLogprobs, error, [&](auto value, auto &ss) {
            if (value > MAX_VLLM_OPENAI_TOP_LOGPROBS) {
                ss << "logprobs should be in range [0, " << MAX_VLLM_OPENAI_TOP_LOGPROBS << "], ";
                ss << "got " << jsonObj[keyLogprobs] <<  ".";
                return false;
            }
            tmpReq->logprobs = true;
            tmpReq->logprobs = true;
            return true;
        });
}

// 1. validate and pass original value to ibis
// 2. calculate the number of sequences to be returned to the user
bool SingleReqVllmOpenAiCompletionsInferInterface::SetReturnSeqCount(RequestSPtr req, std::string& msg)
{
    // when use beam search, it's no need to check best_of
    if (req->useBeamSearch.value_or(false)) {
        returnSeqCount_ = req->n.value_or(1);
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "The number of sequences that would be returned to user is "
            << returnSeqCount_ << ". The logid is " << requestId_ << ".");
        return true;
    }

    // not use beam search
    if (inputParam->streamMode) {
        if (req->bestOf.has_value()) {
            // best_of must be equal to n when both best_of and n are provided
            if (req->n.has_value() && req->bestOf.value() != req->n.value()) {
                std::stringstream ss;
                ss << "best_of must be equal to n, but best_of is " << req->bestOf.value() << " and n is "
                    << req->n.value() << ".";
                msg = ss.str();
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    CHECK_ERROR), "Failed to check parameters by " << msg << " The logid is " << requestId_ << ".");
                return false;
            }
            // n shouldn't be null when best_of is provided
            if (!req->n.has_value()) {
                std::stringstream ss;
                ss << "best_of must be equal to n in stream mode, but best_of is " << req->bestOf.value()
                   << ", n is None.";
                msg = ss.str();
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    CHECK_ERROR), "Failed to check parameters by " << msg << " The logid is " << requestId_ << ".");
                return false;
            }
        }
        returnSeqCount_ = req->n.value_or(1);
    } else {
        // best_of must be greater than or equal to n when both best_of and n are provided
        if (req->bestOf.has_value() && req->n.has_value()
            && req->bestOf.value() < req->n.value()) {
            std::stringstream ss;
            ss << "Best_of must be greater than or equal to n, but best_of is " << req->bestOf.value()
                << " and n is " << req->n.value() << ".";
            msg = ss.str();
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "Failed to check parameters by " << msg << " The logid is " << requestId_ << ".");
            return false;
        }
        returnSeqCount_ = std::max(req->bestOf.value_or(1), req->n.value_or(1));
    }
    if (!CheckReturnSeqCount(req, msg)) {
        return false;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "The number of sequences that would be returned to user is "
        << returnSeqCount_ << ". The logid is " << requestId_ << ".");
    return true;
}

// when beam search is not avaliable, returnSeqCount_ can be greater than 1 only when sampling is enabled
bool SingleReqVllmOpenAiCompletionsInferInterface::CheckReturnSeqCount(RequestSPtr req, std::string &msg) const
{
    if ((!req->temperature.has_value() || IsFloatEquals(req->temperature.value(), 0.0f)) && returnSeqCount_ > 1) {
        std::stringstream ss;
        ss << "Best_of and n shouldn't be greater than 1 when sampling is disabled, but best_of is "
           << (req->bestOf.has_value() ? std::to_string(req->bestOf.value()) : "null") << " and n is "
           << (req->n.has_value() ? std::to_string(req->n.value()) : "null") << ".";
        msg = ss.str();
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "Failed to check parameters by " << msg << " The logid is " << requestId_  << ".");
        return false;
    }
    return true;
}
bool SingleReqVllmOpenAiCompletionsInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg)
{
    if (!(AssignIgnoreEos(reqJsonBody_, tmpReq, msg) &&
          AssignStopStrings(reqJsonBody_, tmpReq, msg) && // 保留一次即可
          AssignIncludeStopStrInOutput(reqJsonBody_, tmpReq, msg) &&
          AssignStopTokenIds(reqJsonBody_, tmpReq, msg) &&
          AssignTemperature(reqJsonBody_, tmpReq, msg, true) &&
          AssignTopK(reqJsonBody_, tmpReq, msg, false, true) &&
          AssignTopP(reqJsonBody_, tmpReq, msg) &&
          AssignSeed(reqJsonBody_, tmpReq, msg) &&
          AssignRepetitionPenalty(reqJsonBody_, tmpReq, msg, MAX_VLLM_OPENAI_REPETITION_PENALTY) &&
          AssignFrequencyPenalty(reqJsonBody_, tmpReq, msg) &&
          AssignPresencePenalty(reqJsonBody_, tmpReq, msg) &&
          AssignSkipSpecialTokens(reqJsonBody_, tmpReq, msg) &&
          AssignBestOf(reqJsonBody_, tmpReq, msg) &&
          AssignN(reqJsonBody_, tmpReq, msg) &&
          AssignBeamSearch(reqJsonBody_, tmpReq, msg) &&
          AssignLogprobs(reqJsonBody_, tmpReq, msg) &&
          AssignMaxTokens(reqJsonBody_, inputParam, msg) &&
          AssignStream(reqJsonBody_, inputParam, msg) &&
          AssignLoraId(reqJsonBody_, tmpReq, this->model, msg))) {
        return false;
    }
    if (!SetReturnSeqCount(tmpReq, msg)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Body in request format invalid for id=" << requestId_);
        return false;
    }
    auto ctx = BuildValidationContext();
    if (!inputParam->ValidateFeatureCompatibility(ctx, msg)) {
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::CheckModelName(const std::string &modelName,
                                                                  std::string &foundModelName)
{
    auto &modelParam = GetModelDeployConfig();

    for (auto &mParam : modelParam) {
        if (mParam.modelName == modelName) {
            foundModelName = modelName;
            return true;
        }
    }
    for (const auto& lora_config : loraConfigs_) {
        if (modelName == lora_config->loraName) {
            foundModelName = lora_config->loraName;
            return true;
        }
    }
    return false;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::AssignModel(nlohmann::ordered_json &body, std::string &model,
                                                               std::string &error)
{
    const std::string key = "model";
    if (!body.contains(key) || body[key].is_null()) {
        error = "Request contains not model or model null";
        return false;
    }
    if (!body[key].is_string()) {
        error = "Request param model must be string";
        return false;
    }
    std::string inputValue = body[key];
    if (inputValue.length() == 0) {
        error = "Request param model must be not empty";
        return false;
    }
    if (!CheckModelName(inputValue, model)) {
        error = "Not found given model, please check modelName in config file.";
        return false;
    }

    return true;
}

static std::string ValidateLength(size_t length, size_t maxLen, const std::string& type)
{
    if (length == 0 || length > maxLen) {
        return "Length of " + type + " should be greater than 0 and less than or equal to " + std::to_string(maxLen);
    }
    return "";
}

static bool AssignPrompt(const OrderedJson &jsonObj, std::vector<int64_t> &tokenIds, std::string &text,
    uint64_t &timestamp, std::string &error) noexcept
{
    const std::string key = "prompt";
    if (!jsonObj.contains(key) || jsonObj[key].is_null()) {
        error = std::string("Contains not prompt or prompt null");
        return false;
    }
    if (jsonObj[key].type() == OrderedJson::value_t::string) {
        text = jsonObj[key];
        std::u16string utf16 = GetU16Str(text);
        auto lenError = ValidateLength(utf16.length(), GetMaxInputLen(), "Prompt text");
        if (!lenError.empty()) {
            error = lenError + ", but got " + std::to_string(utf16.length());
            return false;
        }
        auto status = TokenizerProcessPool::GetInstance().Encode(text, tokenIds, ENCODE_FLAG, timestamp);
        if (!status.IsOk()) {
            error = status.StatusMsg();
            return false;
        }
    } else if (jsonObj[key].type() == OrderedJson::value_t::array) {
        if (jsonObj[key].empty()) {
            error = "Prompt array cannot be empty.";
            return false;
        }
        for (auto& item : jsonObj[key]) {
            if (!item.is_number_integer()) {
                error = "All elements of the prompt array must be integers.";
                return false;
            }
            tokenIds.push_back(item);
        }
        auto status = TokenizerProcessPool::GetInstance().Decode(tokenIds, text, timestamp, false);
        if (!status.IsOk()) {
            error = status.StatusMsg();
            return false;
        }
    } else {
        error = "The type of prompt is abnormal,prompt must be str or integer list.";
        return false;
    }
    auto lenError = ValidateLength(tokenIds.size(), MAX_TOKENS_NUM, "Prompt token");
    if (!lenError.empty()) {
        error = lenError + ", but got " + std::to_string(tokenIds.size());
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::ValidateAndPrepareReqToken(nlohmann::ordered_json &body,
                                                                              std::string &msg, uint64_t &timestamp)
{
    try {
        // check model
        if (!AssignModel(body, model, msg)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), msg);
            return false;
        }
        if (this->isReCompute_) {
            // when recompute, tokenids can be directly parsed from input, no need to do tokenize
            if (!this->GetTokensFromInput(inputParam->textInput, reqTokens_, this->respTokenMap[SPECIAL_SEQ_ID_PRESET],
                                          msg)) {
                msg = "Failed to get token from input: " + msg;
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ABNORMAL_TRANSMISSION_ERROR), msg);
                return false;
            }
        } else if (!AssignPrompt(body, reqTokens_, inputParam->textInput, timestamp, msg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    ABNORMAL_TRANSMISSION_ERROR), msg);
                return false;
        }
        return true;
    } catch (...) {
        msg = "Check open ai req parameter error";
        return false;
    }
    return true;
}

void SingleReqVllmOpenAiCompletionsInferInterface::SetDMIReComputeBuilder()
{
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqVllmOpenAiCompletionsInferInterface::BuildReComputeBody, this, std::placeholders::_1));
}

bool SingleReqVllmOpenAiCompletionsInferInterface::BuildResponseJson(ResponseSPtr response,
                                                                     const std::vector<BestNTokens> &tempTokens,
                                                                     RespBodyQueue &jsonStrings,
                                                                     const uint64_t &timestamp)
{
    bool res = true;
    if (inputParam->streamMode) {
        if (!ProcessResponseStream(response, tempTokens, jsonStrings, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process openAI response stream");
            return false;
        }
    } else {
        if (!ProcessResponseSingle(response, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process openAI response single");
            return false;
        }
        res = EncodeResponse(jsonStrings);
    }
    return res;
}

void SingleReqVllmOpenAiCompletionsInferInterface::SendStreamResponse(RespBodyQueue &jsonStrings)
{
    if (request_->useBeamSearch.value_or(false) && !isEnd) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "use beam search and Inference didn't finished");
        return;
    }
    if (!EncodeStreamResponse(jsonStrings)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to encode buffer");
        return;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Response has ended is " << isEnd << ", requestId is " << requestId_);
}

std::string SingleReqVllmOpenAiCompletionsInferInterface::ChangeUtf8Str(std::string &input) const
{
    try {
        return CleanStringForJson(input);
    } catch (const std::exception &e) {
        // 处理转换错误
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to change str to utf8. " << e.what());
        return " ";
    }
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeSingleTokenLogrobs(OrderedJson &singleToken,
    const uint64_t &baseIndex, const uint64_t seqId, const StreamCache* cache) noexcept
{
    std::vector<int64_t> oneToken(1);
    for (uint64_t j = 0; j < request_->topLogprobs.value(); j++) {
        std::string tokenStr;
        const auto& logprobsTokensMapRef = (cache == nullptr ? logprobsTokensMap : cache->logprobsTokensMap);
        if (!mindie_llm::SafeGetMapVectorValue(logprobsTokensMapRef, seqId, baseIndex + j,
                                               oneToken[0], "LogprobsTokensMap")) {
            return false;
        }
        if (!DecodeSingleToken(oneToken, tokenStr, 0, 0, false)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ENCODE_DECODE_ERROR), "Convert logprobs token to string failed");
            return false;
        }
        float logprobsValueFloat;
        const auto& logprobsMapRef = (cache == nullptr ? logprobsMap : cache->logprobsMap);
        if (!mindie_llm::SafeGetMapVectorValue(logprobsMapRef, seqId, baseIndex + j,
                                               logprobsValueFloat, "LogprobsMap")) {
            return false;
        }
        double logprobsValue = static_cast<double>(logprobsValueFloat);
        if (singleToken.contains(ChangeUtf8Str(tokenStr))) {
            continue;
        }
        singleToken[ChangeUtf8Str(tokenStr)] = logprobsValue;
    }
    return true;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeLogprobsFullText(OrderedJson &choiceJsonObj,
                                                                          const uint64_t seqId) noexcept
{
    // condition: do not need logprobs information in response
    if (!request_->logprobs.has_value() || !request_->logprobs.value() || !request_->topLogprobs.has_value()) {
        choiceJsonObj["logprobs"] = nullptr;
        return true;
    }
    // condition: need logprobs information in response
    choiceJsonObj["logprobs"]["text_offset"] = OrderedJson::array();
    choiceJsonObj["logprobs"]["token_logprobs"] = OrderedJson::array();
    choiceJsonObj["logprobs"]["tokens"] = OrderedJson::array();
    choiceJsonObj["logprobs"]["top_logprobs"] = OrderedJson::array();

    for (uint64_t i = 0; i < postTokenIdMap[seqId].size(); ++i) {
        // current output token
        std::string curToken;
        std::vector<int64_t> oneToken(1);
        if (!mindie_llm::SafeGetMapVectorValue(postTokenIdMap, seqId, i, oneToken[0],
                                               "PostTokenIdMap")) {
            return false;
        }
        uint64_t baseIndex = i * request_->topLogprobs.value();
        float curLogprobsFloat;
        if (!mindie_llm::SafeGetMapVectorValue(pickedLogprobMap, seqId, i, curLogprobsFloat,
                                               "PickedLogprobMap")) {
            return false;
        }
        double curLogprobs = static_cast<double>(curLogprobsFloat);
        if (!DecodeSingleToken(oneToken, curToken, 0, 0, request_->skipSpecialTokens.value_or(false))) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ENCODE_DECODE_ERROR), "Convert logprobs token to string failed");
            return false;
        }
        choiceJsonObj["logprobs"]["text_offset"].emplace_back(wordOffset);
        wordOffset += curToken.size();
        choiceJsonObj["logprobs"]["token_logprobs"].emplace_back(curLogprobs);
        choiceJsonObj["logprobs"]["tokens"].emplace_back(ChangeUtf8Str(curToken));
        OrderedJson singleTokenToplogprobs;
        singleTokenToplogprobs[ChangeUtf8Str(curToken)] = curLogprobs;
        if (!EncodeSingleTokenLogrobs(singleTokenToplogprobs, baseIndex, seqId)) {
            return false;
        }
        choiceJsonObj["logprobs"]["top_logprobs"].emplace_back(singleTokenToplogprobs);
    }
    return true;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeLogprobsStream(OrderedJson &choiceJsonObj,
    const uint64_t seqId, const StreamCache& cache) noexcept
{
    // condition: do not need logprobs information in response
    if (!request_->logprobs.has_value() || !request_->logprobs.value() || !request_->topLogprobs.has_value()) {
        choiceJsonObj["logprobs"] = nullptr;
        return true;
    }
    // condition: need logprobs information in response
    choiceJsonObj["logprobs"]["text_offset"] = OrderedJson::array();
    choiceJsonObj["logprobs"]["token_logprobs"] = OrderedJson::array();
    choiceJsonObj["logprobs"]["tokens"] = OrderedJson::array();
    choiceJsonObj["logprobs"]["top_logprobs"] = OrderedJson::array();

    std::string curToken;
    std::vector<int64_t> oneToken(1);
    auto prevDecodeIndexIt = cache.prevDecodeIndex.find(seqId);
    if (prevDecodeIndexIt == cache.prevDecodeIndex.end()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
            SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
            "PrevDecodeIndex sequence id " << seqId << " not found.");
        return false;
    }
    uint32_t prevIndex = prevDecodeIndexIt->second;
    
    if (!mindie_llm::SafeGetMapVectorValue(cache.postTokenIdMap, seqId, prevIndex, oneToken[0],
                                           "PostTokenIdMap")) {
        return false;
    }
    uint64_t logprobsIdx = static_cast<uint64_t>(prevIndex) * request_->topLogprobs.value();
    float curLogprobsFloat;
    if (!mindie_llm::SafeGetMapVectorValue(cache.pickedLogprobMap, seqId, prevIndex,
                                           curLogprobsFloat, "PickedLogprobMap")) {
        return false;
    }
    double curLogprobs = static_cast<double>(curLogprobsFloat);
    if (!DecodeSingleToken(oneToken, curToken, 0, 0, request_->skipSpecialTokens.value_or(false))) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Convert logprobs token to string failed");
        return false;
    }
    choiceJsonObj["logprobs"]["text_offset"].emplace_back(wordOffset);
    wordOffset += curToken.size();
    choiceJsonObj["logprobs"]["token_logprobs"].emplace_back(curLogprobs);
    choiceJsonObj["logprobs"]["tokens"].emplace_back(curToken);
    OrderedJson singleTokenToplogprobs;
    singleTokenToplogprobs[ChangeUtf8Str(curToken)] = curLogprobs;
    if (!EncodeSingleTokenLogrobs(singleTokenToplogprobs, logprobsIdx, seqId, &cache)) {
        return false;
    }
    choiceJsonObj["logprobs"]["top_logprobs"].emplace_back(singleTokenToplogprobs);
    return true;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeNonStreamJsonObject(RespBodyQueue &jsonStrs,
    std::map<uint64_t, std::string>& responseTextMap, const bool &needSort)
{
    try {
        OrderedJson tmpJsonObj;
        tmpJsonObj["id"] = requestId_;
        tmpJsonObj["object"] = "text_completion";
        tmpJsonObj["created"] = time(nullptr);
        tmpJsonObj["model"] = model;
        tmpJsonObj["choices"] = OrderedJson::array();

        std::vector<std::pair<uint64_t, std::string>> responseText;
        if (needSort) {
            // sort response text by accumulated probes and truncate by returnSeqCount_
            std::vector<std::pair<uint64_t, double>> accuLogprobs(probesMap.cbegin(), probesMap.cend());
            std::sort(accuLogprobs.begin(), accuLogprobs.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            if (returnSeqCount_ < accuLogprobs.size()) {
                accuLogprobs.resize(returnSeqCount_);
            }
            responseText.reserve(accuLogprobs.size());
            for (const auto& [seqId, _] : accuLogprobs) {
                responseText.emplace_back(seqId, responseTextMap[seqId]);
            }
        } else {
            responseText.insert(responseText.end(), responseTextMap.begin(), responseTextMap.end());
        }

        uint32_t index = 0;
        size_t completeTokenCount = 0;
        for (auto& [seqId, fullText] : responseText) {
            OrderedJson choiceJsonObj;
            choiceJsonObj["index"] = index++;
            choiceJsonObj["text"] = ChangeUtf8Str(fullText);
            if (!EncodeLogprobsFullText(choiceJsonObj, seqId)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    ENCODE_DECODE_ERROR), "Failed to encode vllm completions logprobs for single response");
            }
            choiceJsonObj["stop_reason"] = nullptr;
            choiceJsonObj["finish_reason"] = finishReasonMap[seqId] == "length" ? "length" : "stop";
            tmpJsonObj["choices"].emplace_back(choiceJsonObj);
            completeTokenCount += postTokenIdMap[seqId].size();
        }
        tmpJsonObj["usage"]["prompt_tokens"] = reqTokens_.size();
        if (!singleLLMReqHandlerBase_->GetMetrics().prefixCachedTokenNums.empty()) {
            tmpJsonObj["usage"]["prompt_tokens_details"]["cached_tokens"] =
                singleLLMReqHandlerBase_->GetMetrics().prefixCachedTokenNums.front();
        }
        tmpJsonObj["usage"]["completion_tokens"] = completeTokenCount;
        if (responseText.size() > 0) {
            uint64_t seqId = responseText.front().first;
            if ((reasoningTokens.find(seqId) != reasoningTokens.end()) && (reasoningTokens[seqId] != -1)) {
                tmpJsonObj["usage"]["completion_tokens_details"]["reasoning_tokens"] = reasoningTokens[seqId];
            }
        }
        tmpJsonObj["usage"]["total_tokens"] = reqTokens_.size() + completeTokenCount;
        // 根据 MINDIE_LLM_BENCHMARK_ENABLE 环境变量判断是否开启性能数据采集
        // MINDIE_LLM_BENCHMARK_ENABLE取值含义 1:同步enable；2:异步enable；其他取值:关闭
        const int benchmarkVal = EnvUtil::GetInstance().GetInt("MINDIE_LLM_BENCHMARK_ENABLE", 0);
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "mindieLlmBenchmarkEnable value is " << benchmarkVal);
        if (benchmarkVal == BENCHMARK_ENABLE_SYNC || benchmarkVal == BENCHMARK_ENABLE_ASYNC) {
            auto status = InsertPerfInfoIntoJson(tmpJsonObj["usage"],
                {PerfInfoType::PERF_BATCH_SZIE, PerfInfoType::PERF_QUEUE_WAIT_TIME},
                {"batch_size", "queue_wait_time"});
            if (!status.IsOk()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                    SUBMODLE_FEATURE_SINGLE_INFERENCE, ENCODE_DECODE_ERROR),
                    "Failed to insert performance informations for requestId " << requestId_ << ", error msg is "
                    << status.StatusMsg());
            }
        }
        jsonStrs.push(tmpJsonObj.dump());
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Failed to encode open ai generate response");
        return false;
    }
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeResponse(RespBodyQueue &jsonStrs)
{
    // when candidate count is 1, no need to sort response text by accumulated probes
    if (returnSeqCount_ == 1 && fullTextMap.size() == 1) {
        return EncodeNonStreamJsonObject(jsonStrs, fullTextMap);
    }

    return EncodeNonStreamJsonObject(jsonStrs, fullTextMap, true);
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeStreamJsonObject(RespBodyQueue &jsonStrings,
    std::map<uint64_t, std::string>& responseTextMap, const bool &needSort) noexcept
{
    bool useBeamSearch = request_->useBeamSearch.value_or(false);
    auto encodeLogprobsFunc = [this, useBeamSearch](OrderedJson &choiceJsonObj, const uint64_t seqId,
        const StreamCache& cache) -> bool {
        if (useBeamSearch) {
            return this->EncodeLogprobsFullText(choiceJsonObj, seqId);
        } else {
            return this->EncodeLogprobsStream(choiceJsonObj, seqId, cache);
        }
    };
    std::vector<uint64_t> responseSeqIds;
    if (needSort) {
        // sort response text by accumulated probes and truncate by returnSeqCount_
        std::vector<std::pair<uint64_t, double>> accuLogprobs(probesMap.cbegin(), probesMap.cend());
        std::sort(accuLogprobs.begin(), accuLogprobs.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        if (returnSeqCount_ < accuLogprobs.size()) {
            accuLogprobs.resize(returnSeqCount_);
        }
        responseSeqIds.reserve(accuLogprobs.size());
        for (const auto& [seqId, _] : accuLogprobs) {
            responseSeqIds.emplace_back(seqId);
        }
    } else {
        responseSeqIds.reserve(responseTextMap.size());
        for (const auto& [seqId, _] : responseTextMap) {
            responseSeqIds.emplace_back(seqId);
        }
    }

    std::vector<StreamCache> canOutCache;
    if (!GetAvailableOutputCache(canOutCache)) {
        return false;
    }
    for (auto& item: canOutCache) {
        try {
            uint32_t index = 0;
            for (auto& seqId : responseSeqIds) {
                OrderedJson tmpJsonObj;
                tmpJsonObj["id"] = requestId_;
                tmpJsonObj["object"] = "text_completion";
                tmpJsonObj["created"] = time(nullptr);
                tmpJsonObj["model"] = model;
                tmpJsonObj["choices"] = OrderedJson::array();
                if (endedSeqIds.find(seqId) != endedSeqIds.end()) {
                    index++;
                    continue;
                }
                OrderedJson choiceJsonObj;
                choiceJsonObj["index"] = index++;
                choiceJsonObj["text"] =
                    ChangeUtf8Str(useBeamSearch ? item.fullTextMap[seqId] : item.postSingleText[seqId]);
                if (!encodeLogprobsFunc(choiceJsonObj, seqId, item)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                        ENCODE_DECODE_ERROR), "Failed to encode vllm completions logprobs for stream response");
                }
                choiceJsonObj["stop_reason"] = nullptr;
                if (item.eosMap[seqId] == InferStatusType::ITERATION_CONTINUE) {
                    choiceJsonObj["finish_reason"] = nullptr;
                } else {
                    choiceJsonObj["finish_reason"] = item.finishReasonMap[seqId] == "length" ? "length" : "stop";
                    tmpJsonObj["usage"]["prompt_tokens"] = reqTokens_.size();
                    if (!singleLLMReqHandlerBase_->GetMetrics().prefixCachedTokenNums.empty()) {
                        tmpJsonObj["usage"]["prompt_tokens_details"]["cached_tokens"] =
                            singleLLMReqHandlerBase_->GetMetrics().prefixCachedTokenNums.front();
                    }
                    tmpJsonObj["usage"]["completion_tokens"] = item.postTokenIdMap[seqId].size();
                    if (reasoningTokens[seqId] != -1) {
                        tmpJsonObj["usage"]["completion_tokens_details"]["reasoning_tokens"] = reasoningTokens[seqId];
                    }
                    tmpJsonObj["usage"]["total_tokens"] = reqTokens_.size() + item.postTokenIdMap[seqId].size();
                    // 根据 MINDIE_LLM_BENCHMARK_ENABLE 环境变量判断是否开启性能数据采集
                    const int benchmarkVal = EnvUtil::GetInstance().GetInt("MINDIE_LLM_BENCHMARK_ENABLE", 0);
                    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "mindieLlmBenchmarkEnable value is " << benchmarkVal);
                    if (benchmarkVal == BENCHMARK_ENABLE_SYNC || benchmarkVal == BENCHMARK_ENABLE_ASYNC) {
                        auto status = InsertPerfInfoIntoJson(tmpJsonObj["usage"],
                            {PerfInfoType::PERF_BATCH_SZIE, PerfInfoType::PERF_QUEUE_WAIT_TIME},
                            {"batch_size", "queue_wait_time"});
                        if (!status.IsOk()) {
                            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                                SUBMODLE_FEATURE_SINGLE_INFERENCE, ENCODE_DECODE_ERROR),
                                "Failed to insert performance informations for requestId " << requestId_
                                << ", error msg is " << status.StatusMsg());
                        }
                    }
                    endedSeqIds.insert(seqId);
                }
                tmpJsonObj["choices"].emplace_back(choiceJsonObj);
                jsonStrings.push("data: " + tmpJsonObj.dump() + "\n\n");
            }
            if (std::all_of(item.eosMap.begin(), item.eosMap.end(),
                            [](auto eos) { return eos.second != InferStatusType::ITERATION_CONTINUE; })) {
                jsonStrings.push("data: [DONE]\n\n");
            }
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ENCODE_DECODE_ERROR), "Failed to encode open ai stream response");
            return false;
        }
    }
    return true;
}

bool SingleReqVllmOpenAiCompletionsInferInterface::EncodeStreamResponse(RespBodyQueue &jsonStrings) noexcept
{
    // when beamsearch isn't used, return true stream
    if (!request_->useBeamSearch.value_or(false)) {
        return EncodeStreamJsonObject(jsonStrings, inputParam->postSingleText);
    }

    // when using beamsearch, return only at the end of request
    if (!isEnd) {
        return true;
    }

    // when candidate count is 1, no need to sort response text by accumulated probes
    if (returnSeqCount_ == 1 && fullTextMap.size() == 1) {
        return EncodeStreamJsonObject(jsonStrings, fullTextMap);
    }

    // using beam search in stream mode, response format is just like the format of non-stream mode
    bool status = EncodeNonStreamJsonObject(jsonStrings, fullTextMap, true);
    std::string strTmp = jsonStrings.front();
    jsonStrings.pop();
    strTmp.insert(0, "data: ");
    strTmp.append("\n\n");
    jsonStrings.push(strTmp);
    jsonStrings.push("data: [DONE]\n\n");
    return status;
}

std::string SingleReqVllmOpenAiCompletionsInferInterface::BuildReComputeBody(const std::vector<BestNTokens>& tokens)
{
    OrderedJson newReqJsonObj;
    // Get tokens in non-stream mode
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    newReqJsonObj["model"] = this->model;
    newReqJsonObj["prompt"] = this->BuildReComputeInput();
    if (request_->frequencyPenalty.has_value()) {
        newReqJsonObj["frequency_penalty"] = request_->frequencyPenalty.value();
    }
    if (this->inputParam->maxNewTokens != MAX_NEW_TOKENS_DFT) {
        newReqJsonObj["max_tokens"] = this->inputParam->maxNewTokens;
    }
    if (request_->presencyPenalty.has_value()) {
        newReqJsonObj["presence_penalty"] = request_->presencyPenalty.value();
    }
    if (request_->seed.has_value()) {
        newReqJsonObj["seed"] = request_->seed.value();
    }
    ParseStopString(newReqJsonObj);
    newReqJsonObj["stream"] = this->inputParam->streamMode;
    if (request_->temperature.has_value()) {
        newReqJsonObj["temperature"] = request_->temperature.value();
    }
    if (request_->topP.has_value()) {
        newReqJsonObj["top_p"] = request_->topP.value();
    }
    if (request_->topK.has_value()) {
        newReqJsonObj["top_k"] = request_->topK.value();
    }
    if (request_->repetitionPenalty.has_value()) {
        newReqJsonObj["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (request_->stopTokenIds.has_value() && request_->stopTokenIds.value().size() != 0) {
        newReqJsonObj["stop_token_ids"] = request_->stopTokenIds.value();
    }
    if (request_->ignoreEos.has_value()) {
        newReqJsonObj["ignore_eos"] = request_->ignoreEos.value();
    }
    if (request_->includeStopStrInOutput.has_value()) {
        newReqJsonObj["include_stop_str_in_output"] = request_->includeStopStrInOutput.value();
    }
    if (request_->skipSpecialTokens.has_value()) {
        newReqJsonObj["skip_special_tokens"] = request_->skipSpecialTokens.value();
    }
    return newReqJsonObj.dump();
}

void SingleReqVllmOpenAiCompletionsInferInterface::ParseStopString(nlohmann::ordered_json& newReqJsonObj)
{
    std::string stopStr = request_->stopStrings.has_value() ? request_->stopStrings.value() : "";
    if (stopStr != "") {
        try {
            newReqJsonObj["stop"] = nlohmann::json::parse(stopStr, CheckJsonDepthCallbackUlog);
        } catch(...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Failed to parse stopStrings");
        }
    }
}
} // namespace mindie_llm