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
#include "single_req_vllm_openai_infer_interface.h"
#include <regex>

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
static constexpr double MAX_OPENAI_REPETITION_PENALTY = 2.0;
static constexpr auto FUNCTION_NAME_PATTERN = L"^[a-zA-Z0-9_\u4e00-\u9fa5-]{1,64}$";

SingleReqVllmOpenAiInferInterface::SingleReqVllmOpenAiInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase>& singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqInferInterfaceBase{singleLLMReqHandlerBase, isReCompute, loraConfigs}
{
    inputParam->isChatReq = true;
}

bool SingleReqVllmOpenAiInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg)
{
    if (!(AssignIgnoreEos(reqJsonBody_, tmpReq, msg) &&
          AssignStopStrings(reqJsonBody_, tmpReq, msg) &&
          AssignStopTokenIds(reqJsonBody_, tmpReq, msg) &&
          AssignIncludeStopStrInOutput(reqJsonBody_, tmpReq, msg) &&
          AssignTemperature(reqJsonBody_, tmpReq, msg, true) &&
          AssignTopK(reqJsonBody_, tmpReq, msg, false, true) &&
          AssignTopP(reqJsonBody_, tmpReq, msg) &&
          AssignSeed(reqJsonBody_, tmpReq, msg) &&
          AssignRepetitionPenalty(reqJsonBody_, tmpReq, msg, MAX_OPENAI_REPETITION_PENALTY) &&
          AssignFrequencyPenalty(reqJsonBody_, tmpReq, msg) &&
          AssignPresencePenalty(reqJsonBody_, tmpReq, msg) &&
          AssignSkipSpecialTokens(reqJsonBody_, tmpReq, msg) &&
          AssignBestOf(reqJsonBody_, tmpReq, msg) &&
          AssignN(reqJsonBody_, tmpReq, msg) &&
          AssignBeamSearch(reqJsonBody_, tmpReq, msg) &&
          AssignOpenAILogprobs(reqJsonBody_, tmpReq, msg) &&
          AssignMaxTokens(reqJsonBody_, inputParam, msg) &&
          AssignStream(reqJsonBody_, inputParam, msg) &&
          AssignLoraId(reqJsonBody_, tmpReq, this->model, msg) &&
          AssignResponseFormat(reqJsonBody_, tmpReq, msg))) {
        return false;
    }

    if (!SetReturnSeqCount(tmpReq, msg)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Body in request format invalid for id=" << requestId_);
        return false;
    }
    auto ctx = BuildValidationContext();
    if (!inputParam->ValidateFeatureCompatibility(ctx, msg, true)) {
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::ParseTextInput(nlohmann::ordered_json &body, std::string &msg)
{
    const std::string messageKey = "messages";
    const std::string roleKey = "role";
    const std::string contentKey = "content";
    if (this->isReCompute_) {
        // 重计算的请求中，messages为token字符串, 如：1,2,3,4,5。不需要解析数组
        if (!body.contains(messageKey) || !body[messageKey].is_string()) {
            msg = "Messages MUST be a string for recompute";
            return false;
        }
        inputParam->textInput = body[messageKey];
        return true;
    }
    if (!body.contains(messageKey) || body[messageKey].is_null()) {
        msg = "Request param contains not messages or messages null";
        return false;
    }
    if (!body[messageKey].is_array() || body[messageKey].empty()) {
        msg = "Request param messages not arr or arr is empty";
        return false;
    }
    OrderedJson paramObj = body[messageKey];
    for (OrderedJson &paramItem : paramObj) {
        if (!paramItem.contains(roleKey) || paramItem[roleKey].is_null()) {
            msg = "Request param contains not role or role null";
            return false;
        }

         // 指定了tool_call时sassistant的content可以为空
        if (paramItem[roleKey] == "tool" && !ValidToolCallID(paramItem, msg)) {
            return false;
        }
        if (paramItem[roleKey] == "assistant") {
            if (!ValidAssistantMessage(paramItem, msg)) {
                return false;
            }
        } else {
            if (!paramItem.contains(contentKey) || paramItem[contentKey].is_null()) {
                msg = "Request param contains not content or content null";
                return false;
            }
            if (paramItem[contentKey].type() == OrderedJson::value_t::string) {
                std::string contents = paramItem[contentKey];
            } else if (paramItem[contentKey].type() == OrderedJson::value_t::array) {
                if (paramItem[contentKey].empty()) {
                    msg = "Inputs arr is empty";
                    return false;
                }
                if (!CheckMultimodalUrlFromJson(paramItem[contentKey], msg)) {
                    return false;
                }
            } else {
                msg = "The type of inputs is abnormal";
                return false;
            }
        }
    }
    inputParam->textInput = paramObj.dump();
    return CheckTextInputLen(inputParam->textInput, msg);
}

bool SingleReqVllmOpenAiInferInterface::SetReturnSeqCount(RequestSPtr req, std::string &errMsg)
{
    if (req->useBeamSearch.value_or(false)) {
        returnSeqCount_ = req->n.value_or(1);
        return true;
    }
    if (!req->bestOf.has_value() && !req->n.has_value()) {
        returnSeqCount_ = 1;
    } else if (req->bestOf.has_value() && !req->n.has_value()) {
        if (inputParam->streamMode) {
            std::stringstream ss;
            ss << "best_of must be equal to n in stream mode, but best_of is " << req->bestOf.value()
                << ", n is None.";
            errMsg = ss.str();
            return false;
        }
        returnSeqCount_ = req->bestOf.value();
    } else if (!req->bestOf.has_value() && req->n.has_value()) {
        returnSeqCount_ = req->n.value();
    } else {
        if (!inputParam->streamMode && req->bestOf < req->n) {
            std::stringstream ss;
            ss << "best_of must be greater than or equal to n, but best_of is " << req->bestOf.value()
                << ", n is " << req->n.value() << ".";
            errMsg = ss.str();
            return false;
        }
        if (inputParam->streamMode && req->bestOf != req->n) {
            std::stringstream ss;
            ss << "best_of must be equal to n in stream mode, but best_of is " << req->bestOf.value()
                << ", n is " << req->n.value() << ".";
            errMsg = ss.str();
            return false;
        }
        returnSeqCount_ = req->bestOf.value();
    }
    if (IsFloatEquals(req->temperature.value_or(0.0f), 0.0f) && returnSeqCount_ > 1) {
        std::stringstream ss;
        ss << "best_of/n can not greater than 1 without using beam search when temperature is 0 or not set, but ";
        ss << "best_of is " << (req->bestOf.has_value() ? std::to_string(req->bestOf.value()) : "None");
        ss << ", n is " << (req->n.has_value() ? std::to_string(req->n.value()) : "None") << ".";
        errMsg = ss.str();
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::CheckTextInputLen(std::string &textInput, std::string &msg) const
{
    std::string errorMsg = "";
    std::u16string utf16 = GetU16Str(textInput, &errorMsg);
    if (!errorMsg.empty()) {
        msg = "Failed to check the input text. " + errorMsg;
        return false;
    }
    if (utf16.length() == 0 || utf16.length() > GetMaxInputLen()) {
        msg = "Messages len not in (0, " + std::to_string(GetMaxInputLen()) + ", but the length of inputs is " +
            std::to_string(utf16.length());
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::ValidToolCall(OrderedJson &toolCalls, std::string &msg) const
{
    for (OrderedJson &toolCall : toolCalls) {
        if (!toolCall.contains("id") || toolCall["id"].is_null()) {
            msg = "Tool call param contains no id.";
            return false;
        }
        if (toolCall["id"].type() != OrderedJson::value_t::string) {
            msg = "Tool call param \"id\" arguments expected to be a string";
            return false;
        }
        if (!toolCall.contains("type") || toolCall["type"] != "function") {
            msg = "The type of the tool. Currently, only function is supported.";
            return false;
        }
        if (!toolCall.contains("function") || !toolCall["function"].is_object()) {
            msg = "Tool call param contains no function that the model called.";
            return false;
        }
        OrderedJson &function = toolCall["function"];
        if (!function.contains("name") || function["name"].is_null()) {
            msg = "Tool call param contains no function name";
            return false;
        }
        if (function["name"].type() != OrderedJson::value_t::string) {
            msg = "Tool call param function name expected to be a string";
            return false;
        }
        std::string functionName = function["name"].get<std::string>();
        if (!std::regex_match(String2Wstring(functionName), std::wregex(FUNCTION_NAME_PATTERN))) {
            msg = "The name of function must be a-z, A-Z, 0-9, common chinese characters, underscores and dashe "
            "within max length of 64, unexpected function name: " + functionName;
            return false;
        }
        if (!function.contains("arguments") || function["arguments"].is_null()) {
            msg = "Tool call param contains no function arguments";
            return false;
        }
        if (function["arguments"].type() != OrderedJson::value_t::string) {
            msg = "Tool call param function arguments expected to be a string";
            return false;
        }
    }

    return true;
}
bool SingleReqVllmOpenAiInferInterface::ValidAssistantMessage(OrderedJson &message, std::string &msg) const
{
    if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
        if (ValidToolCall(message["tool_calls"], msg)) {
            // if tool calls is valid, content can be none or length is 0
            return true;
        }
        return false;
    }
    if (!message.contains("content") || message["content"].is_null()) {
        msg = "Request param contains not content or content null";
        return false;
    }
    if (message["content"].type() != OrderedJson::value_t::string) {
        msg = "Request param \"content\" arguments expected to be a string";
        return false;
    }
    std::string contents = message["content"];
    if (contents.empty()) {
        msg = "Request param \"content\" should not be empty";
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::ValidToolCallID(OrderedJson &paramItem, std::string &msg) const
{
    const std::string toolCallID = "tool_call_id";
    if (!paramItem.contains(toolCallID) || paramItem[toolCallID].is_null()) {
        msg = "Request param contains not tool_call_id or tool_call_id null while role is tool";
        return false;
    }
    if (paramItem[toolCallID].type() == OrderedJson::value_t::string) {
        std::string contents = paramItem[toolCallID];
        if (contents.length() == 0) {
            msg = "Request param tool_call_id len should not be 0 while role is tool";
            return false;
        }
    } else {
        msg = "The type of tool_call_id is abnormal";
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::ValidMessagesArray(OrderedJson &body, OrderedJson &messges,
                                                           std::string &msg) const
{
    const std::string messageKey = "messages";
    const std::string roleKey = "role";
    const std::string contentKey = "content";
    const std::string assistanKey = "assistant";
    if (!body.contains(messageKey) || body[messageKey].is_null()) {
        msg = "Request param messages is null";
        return false;
    }
    if (!body[messageKey].is_array() || body[messageKey].empty()) {
        msg = "Request param messages is not arr or arr is empty";
        return false;
    }
    messges = body[messageKey];
    for (OrderedJson &paramItem : messges) {
        if (!paramItem.contains(roleKey) || paramItem[roleKey].is_null()) {
            msg = "Request param contains not role or role null";
            return false;
        } else if (paramItem[roleKey] != "system" && paramItem[roleKey] != "assistant" &&
            paramItem[roleKey] != "user"  && paramItem[roleKey] != "tool") {
            msg = "Request param contains role must be system, assistant, user or tool";
            return false;
        }
        if (paramItem[roleKey] == "tool" && !ValidToolCallID(paramItem, msg)) {
            return false;
        }
        if (paramItem[roleKey] == assistanKey) {
            if (!ValidAssistantMessage(paramItem, msg)) {
                return false;
            }
        } else {
            if (paramItem[roleKey] == "tool") {
                if (!paramItem.contains("tool_call_id") || paramItem["tool_call_id"].is_null()) {
                    msg = "Request param contains no tool_call_id";
                    return false;
                }
            }
            if (!paramItem.contains(contentKey) || paramItem[contentKey].is_null()) {
                msg = "Request param contains not content or content null";
                return false;
            }
            std::string contents = paramItem[contentKey];
            if (paramItem[roleKey] != "tool" && contents.length() == 0) {
                msg = "Request param content len should not be 0";
                return false;
            }
        }
    }
    return true;
}

void SingleReqVllmOpenAiInferInterface::FilterToolChoice(OrderedJson &filterTools)
{
    const std::string toolChoiceStr = inputParam->toolChoice;
    const OrderedJson toolChoiceObject = inputParam->toolChoiceObject;
    OrderedJson toolsObject = inputParam->toolsObject;

    if (toolChoiceStr == "none") {
        return;
    } else if (toolChoiceStr == "auto" || toolChoiceStr =="required" || toolChoiceStr == "") {
        filterTools = toolsObject;
    } else if (toolChoiceObject.is_object()) {
        const OrderedJson function = toolChoiceObject["function"];
        if (!function.is_object()) {
            return;
        }
        const std::string functionName = function["name"];
        for (OrderedJson &tool : toolsObject) {
            if (tool.is_object() && tool["function"]["name"] == functionName) {
                filterTools = OrderedJson::array();
                filterTools.push_back(tool);
                break;
            }
        }
    }
    return;
}

bool SingleReqVllmOpenAiInferInterface::PrepareMessageArray(
    nlohmann::ordered_json &body, std::string &msg, OrderedJson& messageArray)
{
    const std::string toolsKey = "tools";
    const std::string toolChoiceKey = "tool_choice";
    const std::string roleKey = "role";
    const std::string contentKey = "content";
    const std::string toolRoleKey = "tools_call";
    if (!ValidMessagesArray(body, messageArray, msg)) {
        return false;
    }

    if (inputParam->tools.empty()) {
        msg = "Tools is empty";
        return false;
    }
    if (!messageArray.is_array()) {
        msg = "Messages parsed not is a array";
        return false;
    }

    OrderedJson filterTools;
    FilterToolChoice(filterTools);
    if (!filterTools.is_null()) {
        inputParam->toolCallsJson = filterTools.dump();
        OrderedJson toolsInputJson;
        toolsInputJson[roleKey] = toolRoleKey;
        toolsInputJson[toolsKey] = filterTools;
        toolsInputJson[toolChoiceKey] = inputParam->toolChoice;
        toolsInputJson[contentKey] = "";
        messageArray.push_back(toolsInputJson);
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::ParseToolCall(nlohmann::ordered_json &body, std::string &msg)
{
    const std::string messageKey = "messages";
    if (this->isReCompute_) {
        // 重计算的请求中，messages为token字符串, 如：1,2,3,4,5。不需要解析数组
        if (!body.contains(messageKey) || !body[messageKey].is_string()) {
            msg = "Messages must be a string for recompute";
            return false;
        }
        inputParam->textInput = body[messageKey];
        return true;
    }
    OrderedJson messageArray;
    if (!PrepareMessageArray(body, msg, messageArray)) {
        return false;
    }
    inputParam->textInput = messageArray.dump();
    std::string errorMsg = "";
    std::u16string utf16 = GetU16Str(inputParam->textInput, &errorMsg);
    if (!errorMsg.empty()) {
        msg = "Failed to check the input text. ";
        msg += errorMsg;
        return false;
    }
    if (utf16.length() == 0 || utf16.length() > GetMaxInputLen()) {
        msg = "Messages len not in (0, " + std::to_string(GetMaxInputLen()) + "]";
        msg += ", but the length of inputs is " + std::to_string(utf16.length());
        return false;
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::CheckToolType(const OrderedJson &toolParam, std::string &error) const
{
    if (!toolParam.contains("type")) {
        error = "Tool param invalid, tool type not exist";
        return false;
    }
    if (toolParam["type"].is_null()) {
        error = "Tool param invalid tool type";
        return false;
    } else if (!toolParam["type"].is_string()) {
        error = "Tool param invalid tool string type";
        return false;
    } else if (toolParam["type"] != "function") {
        error = "Tool param not function type";
        return false;
    } else {
        return true;
    }
}

bool SingleReqVllmOpenAiInferInterface::CheckFunction(const OrderedJson &toolParam, std::string &error) const
{
    if (!toolParam.contains("function")) {
        error = "Tool param invalid, tool function not exist";
        return false;
    }
    if (toolParam["function"].is_null() || !toolParam["function"].is_object()) {
        error = "Tool param invalid function object";
        return false;
    }

    auto func = toolParam["function"];
    if (!func.contains("name")) {
        error = "Function object invalid, function name not exist";
        return false;
    }
    if (func["name"].is_null() || !func["name"].is_string()) {
        error = "Function object invalid name type";
        return false;
    }
    std::string functionName = func["name"].get<std::string>();
    if (!std::regex_match(String2Wstring(functionName), std::wregex(FUNCTION_NAME_PATTERN))) {
        error = "The name of function must be a-z, A-Z, 0-9, common chinese characters, underscores and dashe "
        "within max length of 64, unexpected function name: " + functionName;
        return false;
    }

    return true;
}

bool SingleReqVllmOpenAiInferInterface::CheckToolObject(const OrderedJson &tool, std::string &error)
{
    auto result = CheckToolType(tool, error);
    if (!result) {
        return result;
    }

    return CheckFunction(tool, error);
}

bool SingleReqVllmOpenAiInferInterface::GetToolChoiceAsString(const std::string &toolChoice, std::string &error)
{
    if (toolChoice == "none" || toolChoice == "auto" || toolChoice == "required") {
        inputParam->toolChoice = toolChoice;
        return true;
    }

    error = "Parameter tool_choice string not in [none, auto, required]";
    return false;
}

bool SingleReqVllmOpenAiInferInterface::GetToolChoiceAsObject(const OrderedJson &toolChoice, std::string &error)
{
    auto result = CheckToolObject(toolChoice, error);
    if (!result) {
        return result;
    }

    inputParam->toolChoice = toolChoice.dump();
    inputParam->toolChoiceObject = toolChoice;
    return true;
}

bool SingleReqVllmOpenAiInferInterface::ParseOpenAiTools(const OrderedJson &jsonObj, std::string &error)
{
    // Initialize useToolsCall to false by default
    inputParam->useToolsCall = false;

    // Parse tool_choice first
    if (!AssignOpenAiToolChoice(jsonObj, error)) {
        return false;
    }
    // Then parse tools
    if (!AssignOpenAiTools(jsonObj, error)) {
        return false;
    }

    return true;
}

bool SingleReqVllmOpenAiInferInterface::AssignOpenAiTools(const OrderedJson &jsonObj, std::string &error)
{
    if (!jsonObj.contains("tools") || jsonObj["tools"].is_null() || jsonObj["tools"].empty()) {
        return true;
    }

    if (!jsonObj["tools"].is_array()) {
        error = "Tools not array";
        return false;
    }

    OrderedJson paramObj = jsonObj["tools"];
    for (OrderedJson &tool : paramObj) {
        auto result = CheckToolObject(tool, error);
        if (!result) {
            return false;
        }
    }

    inputParam->tools = paramObj.dump();
    inputParam->toolsObject = paramObj;
    inputParam->useToolsCall = true;
    return true;
}

bool SingleReqVllmOpenAiInferInterface::AssignOpenAiToolChoice(const OrderedJson &jsonObj, std::string &error)
{
    if (!jsonObj.contains("tool_choice") || jsonObj["tool_choice"].is_null()) {
        return true;
    }

    if (jsonObj["tool_choice"].is_string()) {
        return GetToolChoiceAsString(jsonObj["tool_choice"], error);
    } else if (jsonObj["tool_choice"].is_object()) {
        return GetToolChoiceAsObject(jsonObj["tool_choice"], error);
    } else {
        error = "Parameter tool_choice not string or object";
        return false;
    }
}

bool SingleReqVllmOpenAiInferInterface::CheckModelName(const std::string &modelName, std::string &foundModelName) const
{
    auto &modelParam = GetModelDeployConfig();
    for (auto &mParam : modelParam) {
        if (mParam.modelName == modelName) {
            foundModelName = modelName;
            return true;
        }
    }
    // 如果没有在模型中找到，转向去lora列表中找
    for (const auto& lora_config : loraConfigs_) {
        if (modelName == lora_config->loraName) {
            foundModelName = lora_config->loraName;
            return true;
        }
    }
    return false;
}

bool SingleReqVllmOpenAiInferInterface::ParseModelName(nlohmann::ordered_json &body, std::string &outModel,
                                                       std::string &err)
{
    const std::string key = "model";
    if (!body.contains(key) || body[key].is_null()) {
        err = "Request contains not model or model null";
        return false;
    }
    if (!body[key].is_string()) {
        err = "Request param model must be string";
        return false;
    }
    std::string inputValue = body[key];
    if (inputValue.length() == 0) {
        err = "Request param model must be not empty";
        return false;
    }
    if (!CheckModelName(inputValue, outModel)) {
        err = "Model not found";
        return false;
    }
    return true;
}


static bool CheckHasMedia(const OrderedJson &jsonObj) noexcept
{
    const std::string imageKey = "image_url";
    const std::string videoKey = "video_url";
    const std::string audioKey = "audio_url";

    bool hasMedia = false;

    for (const auto& param : jsonObj) {
        if (param.contains(imageKey) || param.contains(videoKey) || param.contains(audioKey)) {
            hasMedia = true;
            break;
        }
    }
    return hasMedia;
}

bool SingleReqVllmOpenAiInferInterface::ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg,
                                                                   uint64_t &timestamp)
{
    try {
        const std::string messageKey = "messages";
        const std::string roleKey = "role";
        const std::string contentKey = "content";
        const std::string textKey = "text";
        const std::string typeKey = "type";

        nlohmann::ordered_json jsonBody = body;
        // content list
        if (jsonBody.contains(messageKey) && jsonBody[messageKey].is_array()) {
            for (auto& message : jsonBody[messageKey]) {
                if (message.contains(contentKey) && message[contentKey] != nullptr &&
                    message[contentKey].is_array()) {
                    if (CheckHasMedia(message[contentKey])) {
                        break;
                    }
                    std::string combineText;
                    for (const auto& contentItem : message[contentKey]) {
                        if (contentItem.is_object() && contentItem.contains(typeKey) &&
                            contentItem[typeKey] == textKey && contentItem.contains(textKey)) {
                            combineText += contentItem[textKey].get<std::string>();
                        }
                    }
                    message[contentKey] = combineText;
                }
            }
        }
        body = jsonBody;
        // check model
        if (!ParseModelName(body, model, msg)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
            return false;
        }
        if (!ParseOpenAiTools(body, msg)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
            return false;
        }
        if (!ParseChatTemplate(body, msg)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       msg << ". The requestId is " << requestId_);
            return false;
        }
        if (!ParseChatTemplateRequest(body, msg)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       msg << ". The requestId is " << requestId_);
            return false;
        }
        if (inputParam->useToolsCall) {
            if (!ParseToolCall(body, msg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                    SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
                return false;
            }
        } else {
            if (!ParseTextInput(body, msg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                    SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), msg);
                return false;
            }
        }
        if (this->isReCompute_) {
            // when recompute, tokenids can be directly parsed from input, no need to do tokenize
            if (!this->GetTokensFromInput(inputParam->textInput, reqTokens_, this->respTokenMap[SPECIAL_SEQ_ID_PRESET],
                                          msg)) {
                msg = "Failed to get token from input: " + msg;
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                    SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR), msg);
                return false;
            }
        } else {
            auto encodeSpan = PROF(INFO, Domain("Request").Resource(requestId_.c_str()).SpanStart("encode"));
            std::string chatTemplate;
            if (body.contains("chat_template") && !body["chat_template"].is_null()) {
                chatTemplate = body["chat_template"].get<std::string>();
            } else {
                chatTemplate = "";
            }
            auto status = TokenizerProcessPool::GetInstance().Encode(inputParam->textInput, reqTokens_,
                                                                     ENCODE_CHAT_FLAG, timestamp,
                                                                     inputParam->enableThinking,
                                                                     chatTemplate);
            if (!status.IsOk()) {
                msg = status.StatusMsg();
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    LOCAL_INVOKING_ERROR), msg << ". The requestId is " << requestId_);
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
        msg = "Check open ai req parameter error";
        return false;
    }
}

void SingleReqVllmOpenAiInferInterface::SetDMIReComputeBuilder()
{
    singleLLMReqHandlerBase_->SetDMIReComputeBuildCallBack(
        std::bind(&SingleReqVllmOpenAiInferInterface::BuildVllmOpenAIReComputeBody, this, std::placeholders::_1));
}

bool SingleReqVllmOpenAiInferInterface::BuildResponseJson(ResponseSPtr response,
    const std::vector<BestNTokens> &tempTokens, RespBodyQueue &jsonStrings, const uint64_t &timestamp)
{
    bool res = true;
    if (inputParam->streamMode) {
        if (!ProcessResponseStream(response, tempTokens, jsonStrings, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process OpenAI response stream");
            return false;
        }
    } else {
        if (!ProcessResponseSingle(response, timestamp)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to process OpenAI response single");
            return false;
        }
        res = EncodeResponse(jsonStrings);
    }
    return res;
}

void SingleReqVllmOpenAiInferInterface::GetUtf8CodeFromStr(const std::string &input, OrderedJson &bytes) const
{
    for (unsigned char c: input) {
        bytes.emplace_back(static_cast<int>(c));
    }
}

std::string SingleReqVllmOpenAiInferInterface::ChangeUtf8Str(std::string &input) const
{
    try {
        return CleanStringForJson(input);
    } catch (const std::exception &e) {
        // 处理转换错误
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to change str to utf8. " << e.what());
        return " ";
    }
}

void SingleReqVllmOpenAiInferInterface::SendStreamResponse(RespBodyQueue &jsonStrings)
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

bool SingleReqVllmOpenAiInferInterface::EncodeSingleTokenLogrobs(OrderedJson &singleToken, const uint64_t &baseIndex,
    const uint64_t seqId, const StreamCache* cache) noexcept
{
    std::vector<int64_t> oneToken(1);
    for (uint64_t j = 0; j < request_->topLogprobs.value(); j++) {
        // content
        OrderedJson tokenLogProbs;
        std::string tokenStr;
        const auto& logprobsTokensMapRef = (cache == nullptr ? logprobsTokensMap : cache->logprobsTokensMap);
        if (!mindie_llm::SafeGetMapVectorValue(logprobsTokensMapRef, seqId, baseIndex + j,
                                               oneToken[0], "LogprobsTokensMap")) {
            return false;
        }
        if (!DecodeSingleToken(oneToken, tokenStr, 0, 0, false)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ABNORMAL_TRANSMISSION_ERROR), "Convert logprobs token to string failed. LogId is " << requestId_);
            return false;
        }
        float logprobsValueFloat;
        const auto& logprobsMapRef = (cache == nullptr ? logprobsMap : cache->logprobsMap);
        if (!mindie_llm::SafeGetMapVectorValue(logprobsMapRef, seqId, baseIndex + j,
                                               logprobsValueFloat, "LogprobsMap")) {
            return false;
        }
        double logprobsValue = static_cast<double>(logprobsValueFloat);
        tokenLogProbs["token"] = ChangeUtf8Str(tokenStr);
        tokenLogProbs["logprob"] = logprobsValue;
        tokenLogProbs["bytes"] = OrderedJson::array();
        GetUtf8CodeFromStr(tokenStr, tokenLogProbs["bytes"]);
        singleToken["top_logprobs"].emplace_back(tokenLogProbs);
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::EncodeLogprobsFullText(OrderedJson &choiceJsonObj,
    const uint64_t seqId) noexcept
{
    // no logprobs info need to be encoded in response
    if (!request_->logprobs.has_value() || !request_->logprobs.value() || !request_->topLogprobs.has_value()) {
        choiceJsonObj["logprobs"] = nullptr;
        return true;
    }
    choiceJsonObj["logprobs"]["content"] = OrderedJson::array();
    for (uint64_t i = 0; i < postTokenIdMap[seqId].size(); ++i) {
        OrderedJson singleToken;
        // current output token
        std::string curToken;
        std::vector<int64_t> oneToken(1);
        if (!mindie_llm::SafeGetMapVectorValue(postTokenIdMap, seqId, i, oneToken[0],
                                               "PostTokenIdMap")) {
            return false;
        }
        if (!DecodeSingleToken(oneToken, curToken, 0, 0, request_->skipSpecialTokens.value_or(false))) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                       "Convert logprobs token to string failed. LogId is " << requestId_);
            return false;
        }
        uint64_t baseIndex = i * request_->topLogprobs.value();
        float curLogprobsFloat;
        if (!mindie_llm::SafeGetMapVectorValue(pickedLogprobMap, seqId, i, curLogprobsFloat,
                                               "PickedLogprobMap")) {
            return false;
        }
        double curLogprobs = static_cast<double>(curLogprobsFloat);
        // content
        singleToken["token"] = ChangeUtf8Str(curToken);
        singleToken["logprob"] = curLogprobs;
        singleToken["bytes"] = OrderedJson::array();
        GetUtf8CodeFromStr(curToken, singleToken["bytes"]);
        singleToken["top_logprobs"] = OrderedJson::array();
        if (!EncodeSingleTokenLogrobs(singleToken, baseIndex, seqId)) {
            return false;
        }
        choiceJsonObj["logprobs"]["content"].emplace_back(singleToken);
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::EncodeLogprobsStream(OrderedJson &choiceJsonObj, const uint64_t seqId,
    const StreamCache& cache) noexcept
{
    // no logprobs info need to be encoded in response
    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "EncodeLogprobsStream cache input : " << cache.ToString());
    if (!request_->logprobs.has_value() || !request_->logprobs.value() || !request_->topLogprobs.has_value()) {
        choiceJsonObj["logprobs"] = nullptr;
        return true;
    }
    choiceJsonObj["logprobs"]["content"] = OrderedJson::array();
    OrderedJson singleToken;
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
    if (!DecodeSingleToken(oneToken, curToken, 0, 0, request_->skipSpecialTokens.value_or(false))) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, ABNORMAL_TRANSMISSION_ERROR),
                   "Convert logprobs token to string failed. LogId is " << requestId_);
        return false;
    }
    uint64_t logprobsIdx = static_cast<uint64_t>(prevIndex) * request_->topLogprobs.value();
    float curLogprobsFloat;
    if (!mindie_llm::SafeGetMapVectorValue(cache.pickedLogprobMap, seqId, prevIndex,
                                           curLogprobsFloat, "PickedLogprobMap")) {
        return false;
    }
    double curLogprobs = static_cast<double>(curLogprobsFloat);
    // content
    singleToken["token"] = ChangeUtf8Str(curToken);
    singleToken["logprob"] = curLogprobs;
    singleToken["bytes"] = OrderedJson::array();
    GetUtf8CodeFromStr(curToken, singleToken["bytes"]);
    singleToken["top_logprobs"] = OrderedJson::array();
    if (!EncodeSingleTokenLogrobs(singleToken, logprobsIdx, seqId, &cache)) {
        return false;
    }
    choiceJsonObj["logprobs"]["content"].emplace_back(singleToken);
    return true;
}

bool SingleReqVllmOpenAiInferInterface::EncodeNonStreamJsonObject(RespBodyQueue &jsonStrs,
    std::map<uint64_t, std::string> &responseTextMap,
    const bool &needSort)
{
    try {
        OrderedJson tmpJsonObj;
        tmpJsonObj["id"] = requestId_;
        tmpJsonObj["object"] = "chat.completion";
        tmpJsonObj["created"] = time(nullptr);
        tmpJsonObj["model"] = model;
        tmpJsonObj["choices"] = OrderedJson::array();

        std::vector<std::pair<uint64_t, std::string>> responseText;
        if (needSort) {
            // sort response text by accumulated probes and truncate by returnSeqCount_
            std::vector<std::pair<uint64_t, double>> accuLogprobs(probesMap.cbegin(), probesMap.cend());
            std::sort(accuLogprobs.begin(), accuLogprobs.end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });
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
            choiceJsonObj["message"]["role"] = "assistant";
            choiceJsonObj["message"]["content"] = ChangeUtf8Str(fullText);
            if (parsingContentFlag[seqId].first) {
                choiceJsonObj["message"]["reasoning_content"] = ChangeUtf8Str(reasoningContentFullTextMap[seqId]);
            }
            if (toolsCallObjectMap[seqId].is_null()) {
                choiceJsonObj["message"]["tool_calls"] = OrderedJson::array();
            } else {
                choiceJsonObj["message"]["tool_calls"] = toolsCallObjectMap[seqId];
            }
            if (!EncodeLogprobsFullText(choiceJsonObj, seqId)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ENCODE_DECODE_ERROR), "Failed to encode open ai logprobs for single response");
            }
            choiceJsonObj["finish_reason"] = finishReasonMap[seqId] == "length" ? "length" : \
                (finishReasonMap[seqId] == "tool_calls" ? "tool_calls" : "stop");
            if (choiceJsonObj["finish_reason"] == "tool_calls") {
                choiceJsonObj["message"]["content"] = ChangeUtf8Str(toolsCallContentMap[seqId]);
            } else {
                choiceJsonObj["message"]["content"] = ChangeUtf8Str(fullText);
            }
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
            tmpJsonObj["prefill_time"] = singleLLMReqHandlerBase_->GetMetrics().firstTokenCost;
            tmpJsonObj["decode_time_arr"] = singleLLMReqHandlerBase_->GetMetrics().decodeTime;
        }
        jsonStrs.push(tmpJsonObj.dump());
        return true;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ENCODE_DECODE_ERROR), "Failed to encode open ai generate response");
        return false;
    }
}

bool SingleReqVllmOpenAiInferInterface::EncodeResponse(RespBodyQueue &jsonStrs)
{
    // when candidate count is 1, no need to sort response text by accumulated probes
    if (returnSeqCount_ == 1 && fullTextMap.size() == 1) {
        return EncodeNonStreamJsonObject(jsonStrs, fullTextMap);
    }

    return EncodeNonStreamJsonObject(jsonStrs, fullTextMap, true);
}

/*
 vllm generate empty content in first response for stream mode, the "choice" in response body is shown below
 "choices":[{
        "index":0,
        "delta":{"role":"assistant","content":"","tool_calls":[]},
        "logprobs":null,
        "finish_reason":null
    }]
 */
static void EncodeOpenAiFirstStreamResponse(RespBodyQueue &jsonStrings,
    const size_t seqNum, const std::string inputId, const std::string model, const bool neverSendResponse) noexcept
{
    if (!neverSendResponse) {
        return;
    }
    for (size_t i = 0; i < seqNum; i++) {
        OrderedJson tmpJsonObj;
        tmpJsonObj["id"] = inputId;
        tmpJsonObj["object"] = "chat.completion.chunk";
        tmpJsonObj["created"] = time(nullptr);
        tmpJsonObj["model"] = model;
        OrderedJson choiceJsonObj;
        choiceJsonObj["index"] = i;
        choiceJsonObj["delta"]["role"] = "assistant";
        choiceJsonObj["delta"]["content"] = "";
        choiceJsonObj["logprobs"] = nullptr;
        choiceJsonObj["finish_reason"] = nullptr;
        tmpJsonObj["choices"].emplace_back(choiceJsonObj);
        jsonStrings.push("data: " + tmpJsonObj.dump() + "\n\n");
    }
}

bool SingleReqVllmOpenAiInferInterface::EncodeStreamJsonObject(RespBodyQueue &jsonStrings,
    std::map<uint64_t, std::string> &responseTextMap,
    const bool &needSort) noexcept
{
    bool useBeamSearch = request_->useBeamSearch.value_or(false);
    auto encodeLogprobsFunc = [this, useBeamSearch](OrderedJson &choiceJsonObj, const uint64_t seqId,
                                                    const StreamCache &cache) -> bool {
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
        EncodeOpenAiFirstStreamResponse(jsonStrings, responseSeqIds.size(), requestId_, model, neverSendResponse_);
        try {
            uint32_t index = 0;
            for (auto &seqId : responseSeqIds) {
                if (endedSeqIds.find(seqId) != endedSeqIds.end()) {
                    index++;
                    continue;
                }
                OrderedJson tmpJsonObj;
                tmpJsonObj["id"] = requestId_;
                tmpJsonObj["object"] = "chat.completion.chunk";
                tmpJsonObj["created"] = time(nullptr);
                tmpJsonObj["model"] = model;
                OrderedJson choiceJsonObj;
                choiceJsonObj["index"] = index++;
                choiceJsonObj["delta"]["role"] = "assistant";
                // when we use beamsearch feature in stream mode,
                // we send all things in one response like what we do in non-stream mode
                std::string outputContent = useBeamSearch ? item.fullTextMap[seqId] : item.postSingleText[seqId];
                std::string reasoninContent = useBeamSearch ? item.reasoningContentFullTextMap[seqId] :
                    item.reasoningContentStreamMap[seqId];
                if (useBeamSearch || item.parsingContentFlag[seqId].first) {
                    choiceJsonObj["delta"]["reasoning_content"] = ChangeUtf8Str(reasoninContent);
                }
                if (useBeamSearch || item.parsingContentFlag[seqId].second) {
                    choiceJsonObj["delta"]["content"] = ChangeUtf8Str(outputContent);
                }
                if (useBeamSearch && !toolsCallObjectMap[seqId].is_null()) {
                    choiceJsonObj["delta"]["tool_calls"] = toolsCallObjectMap[seqId];
                } else if (!toolsCallObjectStreamMap[seqId].is_null()) {
                    choiceJsonObj["delta"]["tool_calls"] = toolsCallObjectStreamMap[seqId];
                }
                try {
                    if (skipCurrentRoundMap.at(seqId) && inputParam->streamMode &&
                        item.eosMap.at(seqId) == InferStatusType::ITERATION_CONTINUE) {
                        continue;
                    }
                } catch (const std::out_of_range&) {
                    // skipCurrentRoundMap 或 eosMap 中不存在该 seqId，继续处理
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                        ENCODE_DECODE_ERROR), "Missing seqId in skipCurrentRoundMap or eosMap. SeqId is " << seqId);
                    return false;
                }
                if (!choiceJsonObj["delta"].contains("reasoning_content")
                    && !choiceJsonObj["delta"].contains("content") && !choiceJsonObj["delta"].contains("tool_calls")) {
                    choiceJsonObj["delta"]["content"] = "";
                }
                if (!encodeLogprobsFunc(choiceJsonObj, seqId, item)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    ENCODE_DECODE_ERROR), "Failed to encode open ai logprobs for stream response");
                }
                if (item.eosMap[seqId] == InferStatusType::ITERATION_CONTINUE) {
                    choiceJsonObj["finish_reason"] = nullptr;
                } else {
                    choiceJsonObj["finish_reason"] = item.finishReasonMap[seqId] == "length" ? "length" :
                        (toolsCalled[seqId] ? "tool_calls" : "stop");
                    if (GetServerConfig().fullTextEnabled) {
                        tmpJsonObj["full_text"] = item.fullTextMap[seqId];
                    }
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
            if (std::all_of(item.eosMap.begin(), item.eosMap.end(), [](auto eos) {
                return eos.second != InferStatusType::ITERATION_CONTINUE;
                })) {
                jsonStrings.push("data: [DONE]\n\n");
            }
            neverSendResponse_ = false;
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                ENCODE_DECODE_ERROR), "Failed to encode open ai stream response");
            return false;
        }
    }
    return true;
}

bool SingleReqVllmOpenAiInferInterface::EncodeStreamResponse(RespBodyQueue &jsonStrings) noexcept
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

    return EncodeStreamJsonObject(jsonStrings, fullTextMap, true);
}

std::string SingleReqVllmOpenAiInferInterface::BuildVllmOpenAIReComputeBody(const std::vector<BestNTokens>& tokens)
{
    OrderedJson newReqJsonObj;
    if (tokens.size() != 0) {
        ConvertTokenToMap(tokens);
    }
    newReqJsonObj["model"] = this->model;
    newReqJsonObj["messages"] = this->BuildReComputeInput();
    newReqJsonObj["stream"] = this->inputParam->streamMode;
    if (this->inputParam->maxNewTokens != MAX_NEW_TOKENS_DFT) {
        newReqJsonObj["max_tokens"] = this->inputParam->maxNewTokens;
    }
    if (request_->presencyPenalty.has_value()) {
        newReqJsonObj["presence_penalty"] = request_->presencyPenalty.value();
    }
    if (request_->frequencyPenalty.has_value()) {
        newReqJsonObj["frequency_penalty"] = request_->frequencyPenalty.value();
    }
    if (request_->seed.has_value()) {
        newReqJsonObj["seed"] = request_->seed.value();
    }
    if (request_->temperature.has_value()) {
        newReqJsonObj["temperature"] = request_->temperature.value();
    }
    if (request_->topP.has_value()) {
        newReqJsonObj["top_p"] = request_->topP.value();
    }
    if (request_->repetitionPenalty.has_value()) {
        newReqJsonObj["repetition_penalty"] = request_->repetitionPenalty.value();
    }
    if (request_->topK.has_value()) {
        newReqJsonObj["top_k"] = request_->topK.value();
    }
    BuildStopWords(newReqJsonObj);
    if (request_->skipSpecialTokens.has_value()) {
        newReqJsonObj["skip_special_tokens"] = request_->skipSpecialTokens.value();
    }
    if (request_->ignoreEos.has_value()) {
        newReqJsonObj["ignore_eos"] = request_->ignoreEos.value();
    }
    if (request_->logprobs.has_value()) {
        newReqJsonObj["logprobs"] = request_->logprobs.value();
    }
    if (request_->topLogprobs.has_value()) {
        newReqJsonObj["top_logprobs"] = request_->topLogprobs.value();
    }
    return newReqJsonObj.dump();
}

void SingleReqVllmOpenAiInferInterface::BuildStopWords(nlohmann::ordered_json& newReqJsonObj)
{
    if (request_->stopStrList.has_value()) {
        newReqJsonObj["stop"] = nlohmann::ordered_json(request_->stopStrList.value());
    }
    if (request_->stopTokenIds.has_value() && request_->stopTokenIds.value().size() != 0) {
        newReqJsonObj["stop_token_ids"] = request_->stopTokenIds.value();
    }
    if (request_->includeStopStrInOutput.has_value()) {
        newReqJsonObj["include_stop_str_in_output"] = request_->includeStopStrInOutput.value();
    }
}

} // namespace mindie_llm