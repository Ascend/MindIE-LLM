/*
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
#include <regex>

#include "endpoint_def.h"
#include "parse_protocol.h"
#include "http_rest_resource.h"
#include "parameters_checker.h"
#include "infer_tokenizer.h"
#include "common_util.h"
#include "base64_util.h"
#include "config_manager_impl.h"
#include "single_req_openai_infer_interface.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
static constexpr double MAX_VLLM_OPENAI_REPETITION_PENALTY = 2.0;
static constexpr double MAX_OPENAI_TEMPERATURE = 2.0;

SingleReqOpenAiInferInterface::SingleReqOpenAiInferInterface(
    const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute,
    const std::vector<LoraParamSPtr> loraConfigs) noexcept
    : SingleReqVllmOpenAiInferInterface{singleLLMReqHandlerBase, isReCompute, loraConfigs}
{
    inputParam->isChatReq = true;
}

bool SingleReqOpenAiInferInterface::CheckModelName(const std::string &modelName) const
{
    auto &modelParam = GetModelDeployConfig();
    for (auto& mParam : modelParam) {
        if (mParam.modelName == modelName) {
            return true;
        }
    }
    for (const auto& lora_config : loraConfigs_) {
        if (modelName == lora_config->loraName) {
            return true;
        }
    }
    return false;
}

bool SingleReqOpenAiInferInterface::ParseModelName(nlohmann::ordered_json &body, std::string &outModel,
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
    if (!CheckModelName(inputValue)) {
        err = "Got model " + inputValue + ", but not found in server";
        return false;
    }

    outModel = inputValue;
    return true;
}

bool SingleReqOpenAiInferInterface::SetupInferParams(RequestSPtr tmpReq, std::string &msg)
{
    if (!(AssignIgnoreEos(reqJsonBody_, tmpReq, msg) &&
          AssignStopStrings(reqJsonBody_, tmpReq, msg, true, MAX_STOP_STRING_LEN) &&
          AssignStopTokenIds(reqJsonBody_, tmpReq, msg) &&
          AssignIncludeStopStrInOutput(reqJsonBody_, tmpReq, msg) &&
          AssignTemperature(reqJsonBody_, tmpReq, msg, true, MAX_OPENAI_TEMPERATURE) &&
          AssignTopK(reqJsonBody_, tmpReq, msg, true, false) &&
          AssignTopP(reqJsonBody_, tmpReq, msg) &&
          AssignSeed(reqJsonBody_, tmpReq, msg) &&
          AssignRepetitionPenalty(reqJsonBody_, tmpReq, msg, MAX_VLLM_OPENAI_REPETITION_PENALTY) &&
          AssignFrequencyPenalty(reqJsonBody_, tmpReq, msg) &&
          AssignPresencePenalty(reqJsonBody_, tmpReq, msg) &&
          AssignSkipSpecialTokens(reqJsonBody_, tmpReq, msg) &&
          AssignBestOf(reqJsonBody_, tmpReq, msg) &&
          AssignN(reqJsonBody_, tmpReq, msg) &&
          AssignBeamSearch(reqJsonBody_, tmpReq, msg) &&
          AssignOpenAILogprobs(reqJsonBody_, tmpReq, msg) &&
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
    if (!inputParam->ValidateFeatureCompatibility(ctx, msg, true)) {
        return false;
    }
    return true;
}
} // namespace mindie_llm