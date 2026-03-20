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

#include "single_llm_req_handler_base.h"
#include "endpoint_def.h"
#include "single_req_infer_interface_base.h"
#include "prometheus_metrics.h"
#include "log.h"

namespace mindie_llm {
std::unordered_set<std::string> SingleLLMReqHandlerBase::stopReqSet_ = {};

SingleLLMReqHandlerBase::SingleLLMReqHandlerBase(ReqCtxPtr &ctx) : ctx{ctx} {}

void SingleLLMReqHandlerBase::SetConstructOneResponseCallBack(
    const ConstructOneResponseCallBack &constructOneResponseCallBack)
{
    constructOneResponseCallBack_ = constructOneResponseCallBack;
}

void SingleLLMReqHandlerBase::SetDMIReComputeBuildCallBack(const DMIReComputeBuildMethod &reComputeBuildMethod)
{
    this->dmiReCompBuildMethod_ = reComputeBuildMethod;
}

void SingleLLMReqHandlerBase::SetStreamMode(bool streamMode) { streamMode_ = streamMode; }

void SingleLLMReqHandlerBase::UpdateInferParam(RequestSPtr request, const InferParamSPtr &inferParam)
{
    request_ = request;
    inferParam_ = inferParam;
}

bool SingleLLMReqHandlerBase::GetContextJsonBody([[maybe_unused]] nlohmann::ordered_json &body) { return true; };

bool SingleLLMReqHandlerBase::GetContextJsonBody([[maybe_unused]] InferParamSPtr inputParam,
                                                 [[maybe_unused]] RequestSPtr request,
                                                 [[maybe_unused]] std::vector<int64_t> &reqTokens,
                                                 [[maybe_unused]] std::vector<int64_t> &respTokens)
{
    return true;
};

bool SingleLLMReqHandlerBase::ParseSeqIdFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken)
{
    auto seqId = 0;
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        seqId = response->responseContents[i].seqId;
        if (seqId == 0) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "seqId is null in Response.");
            return false;
        } else {
            postToken.at(i).seqId = seqId;
            if (static_cast<uint64_t>(postToken.at(i).seqId) == std::numeric_limits<uint64_t>::max()) {
                postToken.at(i).seqId = (std::numeric_limits<uint64_t>::max() >> 1) + simulatedSeqIndex++;
            }
        }
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseParentSeqIdFromResponse(const ResponseSPtr &response,
                                                           std::vector<BestNTokens> &postToken) const
{
    auto parentSeqId = 0;
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        parentSeqId = response->responseContents[i].parentSeqId;
        if (parentSeqId == 0) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "parentSeqId is null in Response.");
            return false;
        } else {
            postToken.at(i).parentSeqId = parentSeqId;
        }
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseOutTokenIdFromResponse(const ResponseSPtr &response,
                                                          std::vector<BestNTokens> &postToken) const
{
    auto outTokenIds = std::vector<TokenId>();
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        outTokenIds = response->responseContents[i].outTokenIds;
        if (outTokenIds.size() == 0) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "outTokenIds is emtpy in Response.");
            return false;
        }
        for (const auto &tokenId : outTokenIds) {
            if (tokenId != -1) {
                // valid tokens
                postToken.at(i).tokens.push_back(tokenId);
            }
        }
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseCumLogProbsFromResponse(const ResponseSPtr &response,
                                                           std::vector<BestNTokens> &postToken) const
{
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        postToken.at(i).cumLogprobs = response->responseContents[i].cumLogProb;
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseFinishReasonAndTruncationIdFromResponse(const ResponseSPtr &response,
                                                                           std::vector<BestNTokens> &postToken) const
{
    bool hasEndedSequence = false;
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        postToken.at(i).finishReason = response->responseContents[i].finishReason;
        hasEndedSequence = hasEndedSequence || postToken.at(i).finishReason != InferStatusType::ITERATION_CONTINUE;
    }

    if (!hasEndedSequence) { // truncate only when sequence has not ended yet !
        return true;
    }
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        if (response->responseContents[i].truncationIndex > 0) {
            ULOG_ERROR(
                SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                "truncationIndex should be smaller than 0, got " << response->responseContents[i].truncationIndex);
            return false;
        }
        postToken.at(i).truncationIndex = response->responseContents[i].truncationIndex;
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseOutLogprobFromResponse(const ResponseSPtr &response,
                                                          std::vector<BestNTokens> &postToken) const
{
    for (size_t i = 0; i < response->responseContents.size(); i++) {
        postToken.at(i).logprob = response->responseContents[i].outLogProbs;
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseTopLogProbsFromResponse(const ResponseSPtr &response,
                                                           std::vector<BestNTokens> &postToken) const
{
    size_t topLogprobs = request_->topLogprobs.value();
    for (size_t i = 0; i < response->responseContents.size(); ++i) {
        const ResponseContent &content = response->responseContents[i];

        // Sanity check for consistent vector sizes
        if (content.topLogProbTokenIds.size() != topLogprobs * content.speculativeTokenNum) {
            std::string err = std::string("content.topLogProbTokenIds.size() = ") +
                              std::to_string(content.topLogProbTokenIds.size()) +
                              std::string(" does not match topLogprobs * content.speculativeTokenNum = ") +
                              std::to_string(topLogprobs * content.speculativeTokenNum);
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    ABNORMAL_TRANSMISSION_ERROR), err);
            return false;
        }
        if (content.topLogProbs.size() != topLogprobs * content.speculativeTokenNum) {
            std::string err = std::string("content.topLogProbs.size() = ") +
                              std::to_string(content.topLogProbs.size()) +
                              std::string(" does not match topLogprobs * content.speculativeTokenNum = ") +
                              std::to_string(topLogprobs * content.speculativeTokenNum);
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    ABNORMAL_TRANSMISSION_ERROR), err);
            return false;
        }

        postToken[i].logprobsTokens = content.topLogProbTokenIds;
        postToken[i].logprobs = content.topLogProbs;
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseLogProbsFromResponse(const ResponseSPtr &response,
                                                        std::vector<BestNTokens> &postToken) const
{
    // check parameters
    if (!request_->logprobs.has_value() || !request_->logprobs.value() || !request_->topLogprobs.has_value()) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Do not need logprobs in response.");
        return true;
    }

    // parse contents from Response
    if (!ParseOutLogprobFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse outLogProbs from Response.");
        return false;
    }
    if (request_->topLogprobs.value() == 0) {
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Do not need top_logprobs in response.");
        return true;
    }
    if (!ParseTopLogProbsFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse topLogProbTokenIds and topLogProbs from Response.");
        return false;
    }
    return true;
}

bool SingleLLMReqHandlerBase::ParseTokensFromResponse(const ResponseSPtr &response, std::vector<BestNTokens> &postToken)
{
    size_t parallelResponseSize = response->responseContents.size();
    // check whether the response contains sequences
    if (parallelResponseSize == 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                   "size of Response is 0.");
        return false;
    }

    // Resize postToken to fit the number of responseContents
    try {
        postToken.resize(parallelResponseSize);
    } catch (const std::bad_alloc& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Memory allocation failed during postToken resize: " << e.what());
        return false;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Unknown exception during postToken resize");
        return false;
    }

    // parse contents from Response
    if (!ParseSeqIdFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse seqId from Response.");
        return false;
    }
    if (!ParseParentSeqIdFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse parentSeqId from Response.");
        return false;
    }
    if (!ParseOutTokenIdFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse outTokenIds from Response.");
        return false;
    }
    if (!ParseCumLogProbsFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse cumLogProb from Response.");
        return false;
    }
    if (!ParseFinishReasonAndTruncationIdFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse finishReason and truncationIndex from Response.");
        return false;
    }

    if (std::count_if(postToken.begin(), postToken.end(),
                      [](BestNTokens &item) { return item.finishReason == InferStatusType::ILLEGAL_INPUT; }) != 0) {
        return true;
    }

    // parse logprobs-contents from Response
    if (!ParseLogProbsFromResponse(response, postToken)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to parse logprobs-related and toplogprobs-related contents from Response.");
        return false;
    }

    return true;
}

void SingleLLMReqHandlerBase::InsertStopRequest(std::string stopReqId) const { stopReqSet_.insert(stopReqId); }

Metrics &SingleLLMReqHandlerBase::GetMetrics() { return metrics; }

void SingleLLMReqHandlerBase::SetRecomputeMetrics(Metrics &reMetrics)
{
    uint64_t decodeTimeNum = reMetrics.decodeTime.size();
    for (uint64_t i = 0; i < decodeTimeNum; i++) {
        metrics.decodeTime.push_back(reMetrics.decodeTime[i]);
    }
    metrics.firstTokenCost = reMetrics.firstTokenCost;
}

void SingleLLMReqHandlerBase::MetricsCallback(const ResponseSPtr &response)
{
    // 生成token数
    auto reqId = response->reqId;
    uint64_t decodeTime = 0;
    size_t outputLen = 0;

    // parse metrics
    metrics.batchSize.emplace_back(response->metrics.batchSize);
    metrics.queueWaitTime.emplace_back(response->metrics.queueWaitTime);
    metrics.prefixCachedTokenNums.emplace_back(response->metrics.prefixCachedTokenNum);

    // parse EOS and outputlen
    // TBC: not sure whether using item 0 is suitbale
    // in principle speculativeTokeNum can vary from sequence to sequence in Parallel Decoding.
    outputLen = response->responseContents.at(0).speculativeTokenNum;

    if (metrics.isPrefill) {
        metrics.isPrefill = false;
        // prefill 记录首token时间
        decodeTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - metrics.startingTime)
                                               .count());
        if (isRecompute_) {
            metrics.decodeTime.push_back(decodeTime);
            metrics.callbackIndexQue.push(DECODE_CALLBACK_METRICS_TAG); // decode
        } else {
            metrics.firstTokenCost = decodeTime;
            metrics.callbackIndexQue.push(PREFILL_CALLBACK_METRICS_TAG); // prefill
        }
        if (reqType_ != InferReqType::REQ_DECODE) {
            PrometheusMetrics::GetInstance()->TTFTObserve(decodeTime);
        }
    } else {
        metrics.callbackIndexQue.push(DECODE_CALLBACK_METRICS_TAG); // decode
        // decode 记录每次decode的时间
        decodeTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - metrics.lastTokenTime)
                                               .count());
        // 针对投机场景适配，decode返回小于等于gamma个token，四舍五入
        auto avgDecodeTime = (decodeTime + outputLen / 2) / outputLen;
        for (size_t i = 0; i < outputLen; ++i) {
            metrics.decodeTime.push_back(avgDecodeTime);
            PrometheusMetrics::GetInstance()->TBTObserve(avgDecodeTime);
        }
    }
    metrics.lastTokenTime = std::chrono::steady_clock::now();

    if (response->isEos) {
        metrics.endingTime = std::chrono::steady_clock::now();
        // 最后一个Token耗时
        metrics.lastTokenCost = decodeTime;
    }
}

void SingleLLMReqHandlerBase::SetMetricParams(const ResponseSPtr &response) { MetricsCallback(response); }

void SingleLLMReqHandlerBase::ProcessFailedResponsePrometheusMetrics() const
{
    PrometheusMetrics::GetInstance()->FailedResponseNumberCount();
    PrometheusMetrics::GetInstance()->FailedRequestRateGaugeCollect();
}

void SingleLLMReqHandlerBase::DumpInferParam(const RequestSPtr request)
{
    OrderedJson paramJson;

    auto setParam = [&paramJson](const std::string &key, const auto &param) {
        if (param.has_value()) {
            paramJson[key] = param.value();
        } else {
            paramJson[key] = nullptr;
        }
    };

    setParam("temperature", request->temperature);
    setParam("top_k", request->topK);
    setParam("top_p", request->topP);
    setParam("enable_thinking", request->enableThinking);
    setParam("thinking_budget", request->thinkingBudget);
    setParam("typical_p", request->typicalP);
    setParam("do_sample", request->doSample);
    setParam("seed", request->seed);
    setParam("repetition_penalty", request->repetitionPenalty);
    setParam("watermark", request->watermark);
    setParam("frequency_penalty", request->frequencyPenalty);
    setParam("presence_penalty", request->presencyPenalty);
    setParam("stop_token_ids", request->stopTokenIds);
    setParam("stop", request->stopStrings);
    setParam("skip_special_tokens", request->skipSpecialTokens);
    setParam("include_stop_str_in_output", request->includeStopStrInOutput);
    setParam("ignore_eos", request->ignoreEos);

    std::string msg = "Sampling parameters for request id: " + request->requestId + "\n" + paramJson.dump(4);
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, msg);
}
} // namespace mindie_llm