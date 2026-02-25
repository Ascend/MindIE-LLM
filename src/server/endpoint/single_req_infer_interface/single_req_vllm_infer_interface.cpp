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
#include "single_req_vllm_infer_interface.h"
#include <chrono>
#include <tuple>

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
static constexpr double MAX_VLLM_REPETITION_PENALTY = 2.0;
SingleReqVllmInferInterface::SingleReqVllmInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs}
{
}

static bool AssignVLLMLoraID(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error)
{
    const std::string key = "model";
    if (!jsonObj.contains(key) || jsonObj[key].is_null()) {
        tmpReq->loraId = std::string("None");
        return true;
    }
    if (jsonObj[key].type() != OrderedJson::value_t::string) {
        error = "Model must be string type.";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "vLLM input parameter [model] not string type");
        return false;
    }
    std::string loraId = jsonObj[key];
    if (GetU16Str(loraId).length() < 1 || GetU16Str(loraId).length() > MAX_LORA_ID_LENGTH) {
        error = "Input validation error: length of `model` must be in [1, " + std::to_string(MAX_LORA_ID_LENGTH) +
        "], but got " + std::to_string(GetU16Str(loraId).length());
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), error);
        return false;
    }
    tmpReq->loraId = jsonObj[key];
    return true;
}

bool SingleReqVllmInferInterface::SetReturnSeqCount(RequestSPtr req, std::string &errMsg)
{
    if (!req->bestOf.has_value() && !req->n.has_value()) {
        returnSeqCount_ = 1;
    } else if (req->bestOf.has_value() && !req->n.has_value()) {
        returnSeqCount_ = req->bestOf.value();
    } else if (!req->bestOf.has_value() && req->n.has_value()) {
        returnSeqCount_ = req->n.value();
    } else {
        if (req->bestOf.value() < req->n.value()) {
            std::stringstream ss;
            ss << "best_of should greater than or equal to n, but best_of is " << req->bestOf.value()
                << ", n is " << req->n.value() << ".";
            errMsg = ss.str();
            return false;
        }
        returnSeqCount_ = req->bestOf.value();
    }
    if (IsFloatEquals(req->temperature.value_or(0.0f), 0.0f) && returnSeqCount_ > 1) {
        std::stringstream ss;
        ss << "Best_of/N can not greater than 1 when temperature is 0 or not set, but ";
        ss << "best_of is " << (req->bestOf.has_value() ? std::to_string(req->bestOf.value()) : "None");
        ss << ", n is " << (req->n.has_value() ? std::to_string(req->n.value()) : "None") << ".";
        errMsg = ss.str();
        return false;
    }
    return true;
}
bool SingleReqVllmInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg)
{
    // 设置vllm入口标识位
    isVllmEntrance = true;
    if (!(AssignIgnoreEos(reqJsonBody_, tmpReq, msg) &&
    AssignStopStrings(reqJsonBody_, tmpReq, msg) &&
          AssignStopTokenIds(reqJsonBody_, tmpReq, msg) &&
          AssignIncludeStopStrInOutput(reqJsonBody_, tmpReq, msg) &&
          AssignTemperature(reqJsonBody_, tmpReq, msg, true) &&
          AssignTopK(reqJsonBody_, tmpReq, msg, false, true) &&
          AssignTopP(reqJsonBody_, tmpReq, msg) &&
          AssignSeed(reqJsonBody_, tmpReq, msg) &&
          AssignRepetitionPenalty(reqJsonBody_, tmpReq, msg) &&
          AssignFrequencyPenalty(reqJsonBody_, tmpReq, msg) &&
          AssignPresencePenalty(reqJsonBody_, tmpReq, msg) &&
          AssignSkipSpecialTokens(reqJsonBody_, tmpReq, msg) &&
          AssignBestOf(reqJsonBody_, tmpReq, msg) &&
          AssignN(reqJsonBody_, tmpReq, msg) &&
          AssignVLLMLoraID(reqJsonBody_, tmpReq, msg) &&
          AssignMaxTokens(reqJsonBody_, inputParam, msg) &&
          AssignStream(reqJsonBody_, inputParam, msg))) {
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

bool SingleReqVllmInferInterface::ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg,
                                                             uint64_t &timestamp)
{
    try {
        const std::string key = "prompt";
        if (!body.contains(key) || body[key].is_null()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "Prompt not found for requestId is " << requestId_);
            msg = std::string("Contains not prompt or prompt null");
            return false;
        }
        if (body[key].type() == OrderedJson::value_t::string) {
            inputParam->textInput = body[key];
        } else if (body[key].type() == OrderedJson::value_t::array) {
            if (body[key].empty()) {
                msg = "Prompt arr is empty";
                return false;
            }
            if (!CheckMultimodalUrlFromJson(body[key], msg)) {
                return false;
            }
            inputParam->textInput = body[key].dump();
        } else {
            msg = "The type of prompt is abnormal";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), msg);
            return false;
        }
        std::string errorMsg = "";
        std::u16string utf16 = GetU16Str(inputParam->textInput, &errorMsg);
        if (!errorMsg.empty()) {
            msg = "Failed to check the input text. ";
            msg += errorMsg;
            return false;
        }
        if (utf16.length() == 0 || utf16.length() > GetMaxInputLen()) {
            msg = "Prompt must be necessary and data type must be string and length in (0, " +
                std::to_string(GetMaxInputLen()) + "]";
            msg += ", but the length of inputs is " + std::to_string(utf16.length());
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), msg);
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
            if (!body.contains("origin_inputs") || body["origin_inputs"].is_null()) {
                msg = "Failed to get parameter `origin_inputs` in given json body";
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ABNORMAL_TRANSMISSION_ERROR), msg);
                return false;
            }
            inputParam->textInput = body["origin_inputs"];
        } else {
            auto encodeSpan = PROF(INFO, Domain("Request").Resource(requestId_.c_str()).SpanStart("encode"));
            auto status =
                TokenizerProcessPool::GetInstance().Encode(inputParam->textInput, reqTokens_, ENCODE_FLAG, timestamp);
            if (!status.IsOk()) {
                msg = status.StatusMsg();
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    LOCAL_INVOKING_ERROR), msg << " for requestId is " << requestId_);
                return false;
            }
            PROF(encodeSpan.Metric("recvTokenSize", reqTokens_.size()));
        }

        if (reqTokens_.size() == 0 || reqTokens_.size() > MAX_TOKENS_NUM) {
            msg = "Messages token length must be in (0, " + std::to_string(MAX_TOKENS_NUM) +
                "], but got " + std::to_string(reqTokens_.size()) +
                ". This could be caused by invalid input format or empty content.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), msg);
            return false;
        }
        return true;
    } catch (...) {
        msg = "Check vLLM req parameter error";
        return false;
    }
}

bool SingleReqVllmInferInterface::BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens,
    RespBodyQueue &jsonStrings, const uint64_t &timestamp)
{
    bool res = true;
    if (inputParam->streamMode) {
        if (!ProcessResponseStream(response, tempTokens, jsonStrings, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process vLLM response stream");
            return false;
        }
    } else {
        if (!ProcessResponseSingle(response, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process vLLM response single");
            return false;
        }
        res = EncodeVllmResponse(jsonStrings);
    }
    return res;
}

std::string SingleReqVllmInferInterface::ChangeUtf8Str(std::string &input) const
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

static bool IsValidJson(const std::string& str)
{
    try {
        return Json::accept(str);
    } catch (const Json::parse_error& e) {
        // cannot parse json
        return false;
    }
}

bool SingleReqVllmInferInterface::EncodeVllmResponse(RespBodyQueue &jsonStrs)
{
    try {
        OrderedJson tmpJsonObj;
        std::string space = " ";
        std::string parsedPrompt;
        if (IsValidJson(inputParam->textInput)) {
            Json multiModalInput = Json::parse(inputParam->textInput, CheckJsonDepthCallbackUlog);
            for (const auto& item: multiModalInput) {
                if (item.contains("type") && item["type"] == "text" && item.contains("text")) {
                    parsedPrompt = item["text"];
                    break;
                }
            }
        } else {
            parsedPrompt = inputParam->textInput;
        }
        std::string prefixText = ChangeUtf8Str(parsedPrompt) + ChangeUtf8Str(space);
        tmpJsonObj["text"] = OrderedJson::array();
        std::vector<std::pair<uint64_t, double>> bestSeqs{probesMap.begin(), probesMap.end()};
        std::sort(bestSeqs.begin(), bestSeqs.end(), [](auto left, auto right) {
            return left.second > right.second;
        });
        bestSeqs.resize(std::min<size_t>(returnSeqCount_, bestSeqs.size()));
        for (const auto &[seqId, _] : bestSeqs) {
            tmpJsonObj["text"].push_back(prefixText + ChangeUtf8Str(fullTextMap[seqId]));
        }
        jsonStrs.push(tmpJsonObj.dump());
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Failed to encode vLLM generate response");
        return false;
    }
}

bool SingleReqVllmInferInterface::EncodeVllmStreamResponse(RespBodyQueue &jsonStrings) noexcept
{
    std::vector<StreamCache> canOutCache{};
    if (!GetAvailableOutputCache(canOutCache)) {
        return false;
    }
    for (auto &cache : canOutCache) {
        try {
            OrderedJson tmpJsonObj;
            tmpJsonObj["text"] = OrderedJson::array();
            auto &serverConfig = GetServerConfig();
            uint32_t index = 0;
            for (auto &[seqId, _] : cache.postSingleText) {
                if (index == returnSeqCount_) {
                    break;
                }
                if (endedSeqIds.find(seqId) != endedSeqIds.end()) {
                    tmpJsonObj["text"].push_back(serverConfig.fullTextEnabled ?
                        ChangeUtf8Str(cache.fullTextMap[seqId]) : "");
                } else {
                    tmpJsonObj["text"].push_back(ChangeUtf8Str(cache.postSingleText[seqId]));
                }
                if (cache.eosMap[seqId] != InferStatusType::ITERATION_CONTINUE) {
                    endedSeqIds.insert(seqId);
                }
            }
            std::string outputStr = tmpJsonObj.dump();
            jsonStrings.push(outputStr + '\0');
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ENCODE_DECODE_ERROR), "Failed to encode vLLM stream response");
            return false;
        }
    }
    return true;
}

void SingleReqVllmInferInterface::SendStreamResponse(RespBodyQueue &jsonStrings)
{
    if (!EncodeVllmStreamResponse(jsonStrings)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Failed to encode buffer");
        return;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Response has ended is " << isEnd << ", requestId is " << requestId_);
}

void SingleReqVllmInferInterface::SetDMIReComputeBuilder()
{
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqVllmInferInterface::BuildVllmReComputeBody, this, std::placeholders::_1));
}

std::string SingleReqVllmInferInterface::BuildVllmReComputeBody(const std::vector<BestNTokens>& tokens)
{
    OrderedJson newReqJsonObj;
    // Get tokens in non-stream mode
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    newReqJsonObj["prompt"] = this->BuildReComputeInput();
    newReqJsonObj["origin_inputs"] = this->inputParam->textInput;
    newReqJsonObj["stream"] = this->inputParam->streamMode;

    if (this->inputParam->maxNewTokens != VLLM_MAX_NEW_TOKENS_DFT) {
        newReqJsonObj["max_tokens"] = this->inputParam->maxNewTokens;
    }
    if (request_->repetitionPenalty.has_value()) {
        newReqJsonObj["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (request_->presencyPenalty.has_value()) {
        newReqJsonObj["presence_penalty"] = request_->presencyPenalty.value();
    }
    if (request_->frequencyPenalty.has_value()) {
        newReqJsonObj["frequency_penalty"] = request_->frequencyPenalty.value();
    }
    if (request_->temperature.has_value()) {
        newReqJsonObj["temperature"] = request_->temperature.value();
    }
    if (request_->topK.has_value()) {
        newReqJsonObj["top_k"] = request_->topK.value();
    }
    if (request_->topP.has_value()) {
        newReqJsonObj["top_p"] = request_->topP.value();
    }
    if (request_->seed.has_value()) {
        newReqJsonObj["seed"] = request_->seed.value();
    }
    std::string stopStr = request_->stopStrings.has_value() ? request_->stopStrings.value() : "";
    if (stopStr != "") {
        try {
            newReqJsonObj["stop"] = nlohmann::json::parse(stopStr, CheckJsonDepthCallbackUlog);
        } catch(...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Failed to parse stopStrings");
        }
    }
    if (request_->stopTokenIds.has_value() && request_->stopTokenIds.value().size() != 0) {
        newReqJsonObj["stop_token_ids"] = request_->stopTokenIds.value();
    }
    if (request_->skipSpecialTokens.has_value()) {
        newReqJsonObj["skip_special_tokens"] = request_->skipSpecialTokens.value();
    }
    newReqJsonObj["model"] = request_->loraId;

    return newReqJsonObj.dump();
}
} // namespace mindie_llm