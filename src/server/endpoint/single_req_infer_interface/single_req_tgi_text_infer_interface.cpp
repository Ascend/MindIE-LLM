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
#include "single_req_tgi_text_infer_interface.h"
#include <utility>
#include <sstream>
#include <regex>

#include "httplib.h"
#include "endpoint_def.h"
#include "infer_tokenizer.h"
#include "parameters_checker.h"
#include "http_rest_resource.h"
#include "common_util.h"
#include "base64_util.h"
#include "safe_io.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
SingleReqTgiTextInferInterface::SingleReqTgiTextInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute, bool stream,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs}
{
    inputParam->streamMode = stream;
}

static bool AssignReturnFullText(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "return_full_text";
    return ParametersChecker::BooleanJsonCheck(jsonObj, key, param->returnFullText, error);
}

static bool AssignDecoderInputDetail(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "decoder_input_details";
    return ParametersChecker::BooleanJsonCheck(jsonObj, key, param->decoderInputDetails, error);
}

static bool AssignTruncate(const OrderedJson &body, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "truncate";
    auto res = JsonParse::CheckOptionalItemType(body, key, OrderedJson::value_t::number_integer, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (!res.isPresent) {
        return true;
    }

    uint32_t inputNum = body[key];
    if (inputNum == 0 || inputNum > MAX_INT32_VALUE) {
        std::stringstream ss;
        ss << "truncate must be in (0," << MAX_INT32_VALUE << "], got " << body[key] << ".";
        error = ss.str();
        return false;
    }
    param->truncate = inputNum;
    return true;
}

bool SingleReqTgiTextInferInterface::AssignAdapterId(const OrderedJson &body, RequestSPtr tmpReq,
                                                     std::string &error) const
{
    auto res = JsonParse::CheckOptionalItemType(body, "adapter_id", OrderedJson::value_t::string, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (res.isPresent) {
        std::string adapterId = body["adapter_id"];
        std::regex regexPattern("^[a-zA-Z0-9_./-]{1,256}$");
        if (!std::regex_match(adapterId, regexPattern)) {
            error = "The format of adapter_id is invalid.";
            return false;
        }
        tmpReq->loraId = adapterId;
    }
    return true;
}

void SingleReqTgiTextInferInterface::TruncateReqTokens()
{
    // assign reqToken
    if (inputParam->truncate != 0 && reqTokens_.size() > inputParam->truncate) {
        uint32_t tokenIdx = reqTokens_.size() - inputParam->truncate;
        reqTokens_.erase(reqTokens_.begin(),
                         reqTokens_.begin() + tokenIdx); // discard head tokens when input is too long
    }
}

bool SingleReqTgiTextInferInterface::ValidTGIParameterSpec(std::string &msg)
{
    if (inputParam->streamMode && inputParam->decoderInputDetails) {
        msg = "The decoder_input_details must not be true when stream is true.";
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), msg);
        return false;
    }
    return true;
}

bool SingleReqTgiTextInferInterface::ValidateAndPrepareReqToken(OrderedJson &body, std::string &msg,
                                                                uint64_t &timestamp)
{
    try {
        if (!body.contains("inputs") || body["inputs"].is_null()) {
            msg = "Request should contain 'inputs' and the type should be string.";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), msg);
            return false;
        }
        if (body["inputs"].type() == OrderedJson::value_t::string) {
            inputParam->textInput = body["inputs"];
        } else if (body["inputs"].type() == OrderedJson::value_t::array) {
            if (body["inputs"].empty()) {
                msg = "Inputs arr is empty";
                return false;
            }
            if (!CheckMultimodalUrlFromJson(body["inputs"], msg)) {
                return false;
            }
            inputParam->textInput = body["inputs"].dump();
        } else {
            msg = "The type of inputs is abnormal";
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
            msg = "Inputs length must be in (0, " + std::to_string(GetMaxInputLen());
            msg += "], but got " + std::to_string(utf16.length());
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
                msg = "Failed to get parameter `origin_inputs` in json body";
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    CHECK_ERROR), msg);
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
                    LOCAL_INVOKING_ERROR), msg << ". The requestId is " << requestId_);
                return false;
            }
            PROF(encodeSpan.Metric("recvTokenSize", reqTokens_.size()));
        }

        if (reqTokens_.size() == 0 || reqTokens_.size() > MAX_TOKENS_NUM) {
            msg = "Inputs token length must be in (0, " + std::to_string(MAX_TOKENS_NUM) + "]";
            msg += ", but got " + std::to_string(reqTokens_.size());
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), msg);
            return false;
        }
        return true;
    } catch (...) {
        msg = "Check tgi req parameter error";
        return false;
    }
}
bool SingleReqTgiTextInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg)
{
    auto paramCheckRet =
        JsonParse::CheckOptionalItemType(reqJsonBody_, "parameters", OrderedJson::value_t::object, msg);
    if (!paramCheckRet.isCorrectType) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Parameters in request param invalid for requestId is " << requestId_);
        return false;
    }
    const auto& params = reqJsonBody_["parameters"];
    if (tmpReq == nullptr) {
        tmpReq = std::make_shared<mindie_llm::Request>();
    }
    if (!(AssignStopStrings(params, tmpReq, msg, true, MAX_STOP_STRING_LEN) &&
          AssignTemperature(params, tmpReq, msg, false) &&
          AssignTopK(params, tmpReq, msg) &&
          AssignTopP(params, tmpReq, msg, false) &&
          AssignTypicalP(params, tmpReq, msg) &&
          AssignDoSample(params, tmpReq, msg) &&
          AssignSeed(params, tmpReq, msg) &&
          AssignRepetitionPenalty(params, tmpReq, msg) &&
          AssignWatermark(params, tmpReq, msg) &&
          AssignAdapterId(params, tmpReq, msg) &&
          AssignMaxNewTokens(params, inputParam, msg) &&
          AssignDetails(params, inputParam, msg) &&
          AssignDecoderInputDetail(params, inputParam, msg) &&
          AssignReturnFullText(params, inputParam, msg) &&
          AssignTruncate(params, inputParam, msg) &&
          ValidTGIParameterSpec(msg))) {
        return false;
    }
    auto ctx = BuildValidationContext();
    if (!inputParam->ValidateFeatureCompatibility(ctx, msg)) {
        return false;
    }
    TruncateReqTokens();
    return true;
}

bool SingleReqTgiTextInferInterface::BuildResponseJson(ResponseSPtr response,
                                                       const std::vector<BestNTokens> &tempTokens,
                                                       RespBodyQueue &jsonObjs, const uint64_t &timestamp)
{
    bool res = true;
    if (inputParam->streamMode) {
        if (!ProcessResponseStream(response, tempTokens, jsonObjs, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process TGI response stream");
            return false;
        }
    } else {
        if (!ProcessResponseSingle(response, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process TGI response single");
            return false;
        }
        res = EncodeTGIResponse(jsonObjs);
    }
    return res;
}

void SingleReqTgiTextInferInterface::SendStreamResponse(RespBodyQueue &jsonStrs)
{
    auto ret = EncodeTGIStreamResponse(jsonStrs);
    if (ret != 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to encode buffer");
        return;
    }
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Should not finish, isFinish is " << isEnd);
    return;
}

std::string SingleReqTgiTextInferInterface::ChangeUtf8Str(std::string &input) const
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

int32_t SingleReqTgiTextInferInterface::GenerateRspDetailJsonStr(OrderedJson &jsonObj, std::string &jsonStr)
{
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return -1;
    }
    try {
        jsonObj["details"]["prompt_tokens"] = reqTokens_.size();
        jsonObj["details"]["finish_reason"] = finishReasonMap[seqId];
        jsonObj["details"]["generated_tokens"] = postTokenIdMap[seqId].size();
        if (inputParam->decoderInputDetails) {
            for (long &i : reqTokens_) {
                OrderedJson tmp = Json{ { "id", i }, { "logprob", nullptr },
                    { "special", nullptr }, { "text", nullptr } };
                jsonObj["details"]["prefill"].emplace_back(tmp);
            }
        } else {
            jsonObj["details"]["prefill"] = OrderedJson::array();
        }
        if (request_->seed.has_value()) {
            jsonObj["details"]["seed"] = request_->seed.value();
        } else {
            jsonObj["details"]["seed"] = nullptr;
        }
        for (long &i : postTokenIdMap[seqId]) {
            OrderedJson tmp = Json{ { "id", i }, { "logprob", nullptr }, { "special", nullptr }, { "text", nullptr } };
            jsonObj["details"]["tokens"].emplace_back(tmp);
        }

        jsonObj["generated_text"] = ChangeUtf8Str(fullTextMap[seqId]);
        jsonStr = jsonObj.dump();
        return 0;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to generate response details");
        return -1;
    }
}

bool SingleReqTgiTextInferInterface::EncodeTGIResponse(RespBodyQueue &jsonStrs)
{
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return false;
    }
    std::string jsonStr;
    try {
        OrderedJson tmpJsonObj;
        if (!inputParam->showDetails && !inputParam->decoderInputDetails) {
            tmpJsonObj["generated_text"] = ChangeUtf8Str(fullTextMap[seqId]);
            jsonStr = tmpJsonObj.dump();
        } else {
            if (GenerateRspDetailJsonStr(tmpJsonObj, jsonStr) != 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    JSON_PARSE_ERROR), "Generate tgi response detail json string failed");
                return false;
            }
        }
        jsonStrs.push(jsonStr);
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Failed to encode tgi generate response");
        return false;
    }
}

int32_t SingleReqTgiTextInferInterface::EncodeTGIStreamResponse(RespBodyQueue &jsonStrs)
{
    uint64_t seqId;
    if (!GetUniqueSequenceId(seqId)) {
        return -1;
    }
    std::vector<StreamCache> canOutCache;
    if (!GetAvailableOutputCache(canOutCache)) {
        return -1;
    }

    try {
        for (auto& item: canOutCache) {
            OrderedJson output;
            output["token"]["id"] = item.postSingleTokenMap[seqId];
            output["token"]["text"] = item.postSingleText[seqId];
            output["token"]["logprob"] = nullptr;
            output["token"]["special"] = nullptr;
            bool isFinalResponse = item.eosMap[seqId] != InferStatusType::ITERATION_CONTINUE;
            if (!isFinalResponse) {
                output["generated_text"] = nullptr;
                output["details"] = nullptr;
                std::string outputStr = "data: " + output.dump() + "\n\n";
                jsonStrs.push(outputStr);
                continue;
            }
            output["generated_text"] = ChangeUtf8Str(item.fullTextMap[seqId]);
            if (inputParam->showDetails) {
                output["details"]["prompt_tokens"] = reqTokens_.size();
                output["details"]["finish_reason"] = item.finishReasonMap[seqId];
                output["details"]["generated_tokens"] = item.postTokenIdMap[seqId].size();
                if (request_->seed.has_value()) {
                    output["details"]["seed"] = request_->seed.value();
                } else {
                    output["details"]["seed"] = nullptr;
                }
            } else {
                output["details"] = nullptr;
            }
            endedSeqIds.insert(seqId);
            std::string outputStr = "data: " + output.dump() + "\n\n";
            jsonStrs.push(outputStr);
        }
        return 0;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Failed to encode tgi stream response");
        return -1;
    }
}

void SingleReqTgiTextInferInterface::SetDMIReComputeBuilder()
{
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqTgiTextInferInterface::BuildTgiReComputeBody, this, std::placeholders::_1));
}

std::string SingleReqTgiTextInferInterface::BuildTgiReComputeBody(
    const std::vector<BestNTokens>& tokens)
{
    OrderedJson newReqJsonObj;
    // Get tokens in non-stream mode
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    newReqJsonObj["inputs"] = this->BuildReComputeInput();
    newReqJsonObj["origin_inputs"] = this->inputParam->textInput;
    newReqJsonObj["parameters"]["details"] = this->inputParam->showDetails;
    newReqJsonObj["stream"] = this->inputParam->streamMode;
    newReqJsonObj["parameters"]["return_full_text"] = this->inputParam->returnFullText;
    newReqJsonObj["parameters"]["decoder_input_details"] = this->inputParam->decoderInputDetails;
    if (this->truncate != 0) {
        newReqJsonObj["parameters"]["truncate"] = this->truncate;
    }
    newReqJsonObj["parameters"]["adapter_id"] = request_->loraId;
    if (request_->doSample.has_value()) {
        newReqJsonObj["parameters"]["do_sample"] = request_->doSample.value();
    }
    if (request_->repetitionPenalty.has_value()) {
        newReqJsonObj["parameters"]["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (request_->temperature.has_value()) {
        newReqJsonObj["parameters"]["temperature"] = request_->temperature.value();
    }
    if (this->inputParam->maxNewTokens != MAX_NEW_TOKENS_DFT) {
        newReqJsonObj["parameters"]["max_new_tokens"] = this->inputParam->maxNewTokens;
    }
    if (request_->topK.has_value()) {
        newReqJsonObj["parameters"]["top_k"] = request_->topK.value();
    }
    if (request_->topP.has_value()) {
        newReqJsonObj["parameters"]["top_p"] = request_->topP.value();
    }
    if (request_->seed.has_value()) {
        newReqJsonObj["parameters"]["seed"] = request_->seed.value();
    }
    if (request_->typicalP.has_value()) {
        newReqJsonObj["parameters"]["typical_p"] = request_->typicalP.value();
    }
    if (request_->watermark.has_value()) {
        newReqJsonObj["parameters"]["watermark"] = request_->watermark.value();
    }
    ParseStopString(newReqJsonObj);
    return newReqJsonObj.dump();
}

void SingleReqTgiTextInferInterface::ParseStopString(nlohmann::ordered_json& newReqJsonObj)
{
    std::string stopStr = request_->stopStrings.has_value() ? request_->stopStrings.value() : "";
    if (stopStr != "") {
        try {
            newReqJsonObj["parameters"]["stop"] = nlohmann::json::parse(stopStr, CheckJsonDepthCallbackUlog);
        } catch(...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "Failed to parse stopStrings");
        }
    }
}

bool SingleReqGeneralTgiTextInferInterface::EncodeTGIResponse(RespBodyQueue &jsonStrs)
{
    uint64_t seqId;
    std::string jsonStr;
    if (!GetUniqueSequenceId(seqId)) {
        return false;
    }
    try {
        nlohmann::ordered_json tmpJsonObj;
        nlohmann::ordered_json rstJsonObj;
        if (!inputParam->showDetails && !inputParam->decoderInputDetails) {
            tmpJsonObj["generated_text"] = ChangeUtf8Str(fullTextMap[seqId]);
            rstJsonObj.emplace_back(tmpJsonObj);
            jsonStr = rstJsonObj.dump();
            jsonStrs.push(jsonStr);
            return true;
        }
        if (GenerateRspDetailJsonStr(tmpJsonObj, jsonStr) != 0) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Generate general tgi response detail json string failed");
            return false;
        }
        rstJsonObj.emplace_back(tmpJsonObj);
        jsonStr = rstJsonObj.dump();
        jsonStrs.push(jsonStr);
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            PARAM_PARSE_ERROR), "Failed to encode tgi generate response");
        return false;
    }
}
} // namespace mindie_llm