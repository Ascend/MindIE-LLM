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
#include <sstream>
#include "memory_utils.h"
#include "base64_util.h"
#include "random_generator.h"
#include "parse_protocol.h"
#include "endpoint_def.h"
#include "http_rest_resource.h"
#include "parameters_checker.h"
#include "infer_tokenizer.h"
#include "log.h"
#include "common_util.h"
#include "basic_types.h"
#include "config_manager_impl.h"
#include "infer_param.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
static constexpr int64_t MAX_SIGNED_32BITS_NUM = 0x7FFFFFFFL;
static constexpr int32_t MAX_NUM_NEW_TOKENS_OF_BEAM_SEARCH = 128;
static constexpr double MIN_PRESENCE_PENALTY = -2.0;
static constexpr double MAX_PRESENCE_PENALTY = 2.0;
static constexpr double MIN_FREQUENCY_PENALTY = -2.0;
static constexpr double MAX_FREQUENCY_PENALTY = 2.0;
static constexpr uint64_t MAX_OPENAI_TOP_LOGPROBS = 20;

bool InferParam::ValidateFeatureCompatibility(const ValidationContext &ctx, std::string &error) const noexcept
{
    // 规则一：DMI模式限制（pd 架构）
    if (ctx.isDmiMode) {
        if (ctx.reqUseBeamSearch) {
            error = "use_beam_search is not supported in dmi mode";
            return false;
        }
        if (ctx.reqLogprobs) {
            error = "logprobs is not supported in dmi mode";
            return false;
        }
        if (ctx.reqTopLogprobsSet) {
            error = "top_logprobs is not supported in dmi mode";
            return false;
        }
        if (ctx.reqStop) {
            error = "stop is not supported in dmi mode";
            return false;
        }
        if (ctx.reqStopTokenIds) {
            error = "stop_token_ids is not supported in dmi mode";
            return false;
        }
        if (ctx.reqIncludeStopStrInOutput) {
            error = "include_stop_str_in_output is not supported in dmi mode";
            return false;
        }
        if (ctx.reqN != 1) {
            error = "n must be 1 in dmi mode";
            return false;
        }
        if (ctx.reqBestOf != 1) {
            error = "best_of must be 1 in dmi mode";
            return false;
        }
    }

    // 规则二：接口能力限制
    if (!ctx.endpoint.useBeamSearch && ctx.reqUseBeamSearch) {
        error = "use_beam_search is not supported by this endpoint";
        return false;
    }
    if (!ctx.endpoint.useLogprobs && (ctx.reqLogprobs || ctx.reqTopLogprobsSet)) {
        error = "logprobs/top_logprobs are not supported by this endpoint";
        return false;
    }
    if (!ctx.reqUseBeamSearch) {
        if (!ctx.endpoint.useBestOfN) {
            if (ctx.reqBestOf > 1 || ctx.reqN > 1) {
                error = "best_of/n greater than 1 are not supported by this endpoint";
                return false;
            }
        }
    }
    if (!ctx.endpoint.useFunctionCall && ctx.reqUseToolsCall) {
        error = "function/tool calls are not supported by this endpoint";
        return false;
    }

    // 规则三：功能间组合限制
    if (!ctx.reqUseBeamSearch) {
        if (ctx.streamMode) {
            // 流式模式下的best_of和n校验
            if (ctx.isVllmEntrance) {
                // vllm入口：允许best_of >= n
                if (ctx.reqBestOf < ctx.reqN) {
                    error = "in stream mode for vllm entrance, best_of must be greater than or equal to n";
                    return false;
                }
            } else {
                // 其他入口：best_of 必须等于 n
                if (ctx.reqBestOf != ctx.reqN) {
                    error = "in stream mode, best_of must be equal to n";
                    return false;
                }
            }
        } else {
            if (ctx.reqBestOf < ctx.reqN) {
                error = "best_of must be greater than or equal to n";
                return false;
            }
        }
    } else {  // safety specification of beam search
        if (maxNewTokens > MAX_NUM_NEW_TOKENS_OF_BEAM_SEARCH) {
            std::stringstream ss;
            ss << "Please set the max_tokens to be not larger than "
               << std::to_string(MAX_NUM_NEW_TOKENS_OF_BEAM_SEARCH)
               << " when the use_beam_search is on.";
            error = ss.str();
            return false;
        }
    }

    // 当关闭采样（temperature==0或未设置），且未启用beam search时，不允许返回多条
    if (!ctx.reqUseBeamSearch) {
        if ((IsFloatEquals(ctx.reqTemperature, 0.0f)) && (ctx.reqBestOf > 1 || ctx.reqN > 1)) {
            error = "when sampling disabled (temperature=0), best_of/n must be 1 without beam search";
            return false;
        }
    }

    // 环境变量，当异步推理环境变量开启不支持bestof n, beamsearch
    const char *mindieAsyncSchedulingEnable = std::getenv("MINDIE_ASYNC_SCHEDULING_ENABLE");
    bool asyncEnvVariable = mindieAsyncSchedulingEnable != nullptr && std::string(mindieAsyncSchedulingEnable) == "1";
    if (asyncEnvVariable) {
        if (ctx.reqUseBeamSearch) {
            error = "use_beam_search is not supported while MINDIE_ASYNC_SCHEDULING_ENABLE is on";
            return false;
        }
        if (ctx.reqN != 1) {
            error = "n must be 1 while MINDIE_ASYNC_SCHEDULING_ENABLE is on";
            return false;
        }
        if (ctx.reqBestOf != 1) {
            error = "best_of must be 1 while MINDIE_ASYNC_SCHEDULING_ENABLE is on";
            return false;
        }
    }

    // 叠加限制：参考规格“该参数不支持与 MTP、Function Call、SplitFuse、Prefix Cache、并行解码、PD 分离等叠加”
    auto forbid_when = [&](bool cond, const char *msg) {
        if (cond) {
            error = msg;
            return true;
        }
        return false;
    };

    // 插件启用时的限制规则
    if (ctx.pluginEnabled) {
        if (forbid_when(ctx.reqUseBeamSearch, "beam search cannot be used with plugins")) {
            return false;
        }
        if (forbid_when(ctx.reqLogprobs || ctx.reqTopLogprobsSet, "logprobs cannot be used with plugins")) {
            return false;
        }
        if (forbid_when(ctx.reqBestOf > 1 || ctx.reqN > 1, "best_of/n > 1 cannot be used with plugins")) {
            return false;
        }
    }

    // 插件启用时的限制规则
    if (ctx.deepseekEnabled) {
        if (forbid_when(ctx.reqUseBeamSearch, "beam search cannot be used with deepseek")) {
            return false;
        }
        if (forbid_when(ctx.reqBestOf > 1 || ctx.reqN > 1, "best_of/n > 1 cannot be used with deepseek")) {
            return false;
        }
    }
    return true;
}

bool AssignDoSample(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "do_sample";
    return ParametersChecker::OptionalBooleanJsonCheck(jsonObj, key, tmpReq->doSample, error);
}

bool AssignRepetitionPenalty(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error,
                             double maxValue) noexcept
{
    const std::string key = "repetition_penalty";
    return ParametersChecker::OptionalFloatJsonCheck(
        jsonObj, key, tmpReq->repetitionPenalty, error, [&](auto value, auto &ss) {
            if (!(value > 0.0 && value <= maxValue)) {
                ss << "Parameter '" << key << "' must be in (0.0, " << maxValue << "], got "
                   << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignSeed(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "seed";
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, OrderedJson::value_t::number_unsigned, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (!res.isPresent) {
        // 用户没有输入seed时，要生成随机seed传入
        auto random = RandomGenerator::GetInstance();
        if (random == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Failed to create random generator");
            return false;
        }
        tmpReq->seed = random->GetRand();
        return true;
    }
    uint64_t value = jsonObj[key];
    if (value > static_cast<uint64_t>(INT64_MAX)) {
        error = "Parameter seed must be in [0, " + std::to_string(INT64_MAX) + "], got " + jsonObj[key].dump() + ".";
        return false;
    }
    tmpReq->seed = value;
    return true;
}

bool AssignStopStrings(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error, bool isNumStopStrLimited,
                       uint32_t maxLength) noexcept
{
    const std::string key = "stop";
    if (!jsonObj.contains(key) || jsonObj[key].is_null()) {
        return true;
    }
    auto stopStrings = jsonObj[key];
    bool validFlag = true;
    uint32_t totalLength = 0;
    switch (stopStrings.type()) {
        case OrderedJson::value_t::array: {
            for (size_t i = 0; i < stopStrings.size(); i++) {
                auto stopStrLen = GetU16Str(stopStrings[i]).length();
                if (stopStrings[i].is_null() || !stopStrings[i].is_string() || stopStrLen < 1 ||
                    stopStrLen > MAX_STOP_STRING_LEN) {
                    validFlag = false;
                    break;
                }
                tmpReq->windowSize = std::max(tmpReq->windowSize, static_cast<uint32_t>(stopStrLen));
                totalLength += stopStrLen;
            }
            if (!validFlag || totalLength > MAX_TOTAL_STOP ||
                (isNumStopStrLimited && stopStrings.size() > MAX_STOP_STRING_NUM)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                           GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                           "Request param stop content is invalid");
                error = std::string("Input validation error: `stop` must be list[string] if list, "
                    "and item length in [1, ") + std::to_string(MAX_STOP_STRING_LEN) +
                    std::string("] with total <= ") + std::to_string(MAX_TOTAL_STOP);
                return false;
            }
            if (totalLength == 0) {
                return true;
            }
            tmpReq->stopStrings = Base64Util::Encode(stopStrings.dump());
            break;
        }
        case OrderedJson::value_t::string: {
            auto stopStrLen = GetU16Str(stopStrings).length();
            if (stopStrLen < 1 || stopStrLen > maxLength) {
                error = "Input validation error: length of `stop` must be in [1, " + std::to_string(maxLength) +
                        "], but got " + std::to_string(stopStrLen);
                return false;
            }
            tmpReq->windowSize = std::max(tmpReq->windowSize, static_cast<uint32_t>(stopStrLen));
            tmpReq->stopStrings = Base64Util::Encode(OrderedJson::array({stopStrings}).dump());
            break;
        }
        default: error = "Input validation error: param stop must be string or list[string]"; return false;
    }
    return true;
}

bool AssignPresencePenalty(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "presence_penalty";
    return ParametersChecker::OptionalFloatJsonCheck(
        jsonObj, key, tmpReq->presencyPenalty, error, [&](auto value, auto &ss) {
            if (!(value >= MIN_PRESENCE_PENALTY && value <= MAX_PRESENCE_PENALTY)) {
                ss << "Parameter presence_penalty not in [-2.0, 2.0], but got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignN(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "n";
    return ParametersChecker::OptionalUint32JsonCheck(
        jsonObj, key, tmpReq->n, error, [&](auto value, auto &ss) {
        if (value != 1 &&
            GetServerConfig().inferMode == mindie_llm::INFER_MODE_DMI) {
            ss << "Paramter [n] can only be set to 1 in pd disaggregation mode, got " << value << ".";
            return false;
        }
            if (!(value >= MIN_N && value <= MAX_N)) {
                ss << "Parameter [n] should be in [" << MIN_N << ", " << MAX_N << "], got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignBestOf(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "best_of";
    return ParametersChecker::OptionalUint32JsonCheck(
        jsonObj, key, tmpReq->bestOf, error, [&](auto value, auto &ss) {
        if (value != 1 &&
            GetServerConfig().inferMode == mindie_llm::INFER_MODE_DMI) {
            ss << "Paramter [best_of] can only be set to 1 in pd disaggregation mode, got " << value << ".";
            return false;
        }
        if (!(value >= MIN_BEST_OF && value <= MAX_BEST_OF)) {
            ss << "Parameter [best_of] should be in [" << MIN_BEST_OF << ", " << MAX_BEST_OF << "], got "
               << jsonObj[key] << ".";
            return false;
        }
        return true;
    });
}

bool AssignBeamSearch(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "use_beam_search";
    if (!ParametersChecker::OptionalBooleanJsonCheck(jsonObj, key, tmpReq->useBeamSearch, error)) {
        return false;
    }
    if (GetServerConfig().inferMode == mindie_llm::INFER_MODE_DMI &&
        tmpReq->useBeamSearch.has_value() && tmpReq->useBeamSearch.value()) {
        error = "Request param use_beam_search can not be true in pd disaggregation mode.";
        return false;
    }
    return true;
}

bool AssignFrequencyPenalty(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "frequency_penalty";
    return ParametersChecker::OptionalFloatJsonCheck(
        jsonObj, key, tmpReq->frequencyPenalty, error, [&](auto value, auto &ss) {
            if (!(value >= MIN_FREQUENCY_PENALTY && value <= MAX_FREQUENCY_PENALTY)) {
                ss << "Parameter frequency_penalty should be in [-2.0, 2.0], got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignSkipSpecialTokens(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "skip_special_tokens";
    return ParametersChecker::OptionalBooleanJsonCheck(jsonObj, key, tmpReq->skipSpecialTokens, error);
}

bool AssignIncludeStopStrInOutput(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error)
{
    if (!jsonObj.contains("stop") && !jsonObj.contains("stop_token_ids")) {
        return true;
    }
    const std::string key = "include_stop_str_in_output";
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, OrderedJson::value_t::boolean, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (GetServerConfig().inferMode == INFER_MODE_DMI) {
        tmpReq->includeStopStrInOutput = false;
        if (res.isPresent) {
            error = "Request param include_stop_str_in_output is not support in dmi mode";
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR), error);
            return false;
        } else {
            return true;
        }
    }
    if (res.isPresent) {
        tmpReq->includeStopStrInOutput = jsonObj[key];
    }
    return true;
}

bool AssignTemperature(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error, bool allowLowerBound,
                       double maxValue) noexcept
{
    const std::string key = "temperature";
    return ParametersChecker::OptionalFloatJsonCheck(
        jsonObj, key, tmpReq->temperature, error, [&](auto value, auto &ss) {
            const bool isValid = allowLowerBound
                ? (value >= 0.0 && value <= maxValue)
                : (value > 0.0 && value <= maxValue);
            if (!isValid) {
                ss << "Parameter temperature must be in (0.0," << maxValue << "], got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignMaxTokens(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error)
{
    const std::string key = "max_tokens";
    auto &scheduleParam = GetScheduleConfig();
    if (!jsonObj.contains(key) || jsonObj[key].is_null()) {
        param->isMaxNewTokensSetByUser = false;
        param->maxNewTokens = scheduleParam.maxIterTimes;
        return true;
    }
    if (!jsonObj[key].is_number_integer()) {
        error = "Parameter max_tokens not number_integer";
        return false;
    }

    int64_t value = jsonObj[key];
    if (value <= 0 || value > MAX_INT32_VALUE) {
        error = "Parameter max_tokens must be (0," + std::to_string(MAX_INT32_VALUE) + "], got " + jsonObj[key].dump();
        return false;
    }
    param->maxNewTokens = static_cast<int32_t>(value);
    return true;
}

bool AssignTopK(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error, bool allowLowerBound,
                bool allowNegativeOne) noexcept
{
    const std::string key = "top_k";
    auto res = JsonParse::CheckOptionalItemType(jsonObj, key, OrderedJson::value_t::number_integer, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (!res.isPresent) {
        return true;
    }
    int64_t value = jsonObj[key];
    if (value == -1 && allowNegativeOne) {
        tmpReq->topK = 0;
        return true;
    }
    bool isValid = (value <= MAX_INT32_VALUE) && (allowLowerBound ? value >= 0 : value > 0);
    if (!isValid) {
        error =
            "Parameter top_k must be in -1 || (0," + std::to_string(MAX_INT32_VALUE) + "], got " + jsonObj[key].dump();
        return false;
    }
    tmpReq->topK = static_cast<int32_t>(value);
    return true;
}

bool AssignTopP(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error, bool allowUpperBound) noexcept
{
    const std::string key = "top_p";
    return ParametersChecker::OptionalFloatJsonCheck(
        jsonObj, key, tmpReq->topP, error, [&](auto topPValue, auto &ss) {
            if (!(topPValue > 0.0 && (allowUpperBound ? topPValue <= 1.0 : topPValue < 1.0))) {
                ss << "Parameter top_p must be in (0.0,1.0], got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignIgnoreEos(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "ignore_eos";
    return ParametersChecker::OptionalBooleanJsonCheck(jsonObj, key, tmpReq->ignoreEos, error);
}

bool AssignStopTokenIds(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "stop_token_ids";
    if (!jsonObj.contains(key) || jsonObj[key].is_null()) {
        return true;
    }
    auto stopTokenIdsStr = jsonObj[key];
    if (!stopTokenIdsStr.is_array()) {
        error = "Parameter stop_token_ids not array";
        return false;
    }
    if (stopTokenIdsStr.empty()) {
        return true;
    }
    if (!tmpReq->stopTokenIds.has_value()) {
        tmpReq->stopTokenIds = std::vector<TokenId>();
    }
    for (const auto &tokenId : stopTokenIdsStr) {
        if (!tokenId.is_number_integer()) {
            error = "Parameter stop_token_ids not list[int]";
            return false;
        }
        int64_t value = tokenId;
        if (value < 0 || value > MAX_INT32_VALUE) {
            continue;
        }
        tmpReq->stopTokenIds.value().push_back(tokenId);
    }
    if (!tmpReq->stopTokenIds.has_value()) {
        return true;
    }
    std::set<TokenId> tempStopTokenIds(tmpReq->stopTokenIds.value().begin(), tmpReq->stopTokenIds.value().end());
    tmpReq->stopTokenIds.value().clear();
    tmpReq->stopTokenIds.value().assign(tempStopTokenIds.begin(), tempStopTokenIds.end());
    return true;
}

bool AssignTypicalP(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "typical_p";
    return ParametersChecker::OptionalFloatJsonCheck(
        jsonObj, key, tmpReq->typicalP, error, [&](auto value, auto &ss) {
            if (!(value > 0.0 && value <= 1.0)) {
                ss << "Parameter typical_p must be in (0.0,1.0], got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignWatermark(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "watermark";
    return ParametersChecker::OptionalBooleanJsonCheck(jsonObj, key, tmpReq->watermark, error);
}

bool AssignBatchSize(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "batch_size";
    return ParametersChecker::Int32JsonCheck(jsonObj, key, param->batchSize, error, [&](auto value, auto &ss) {
        if (!(value > 0 && value <= MAX_INT32_VALUE)) {
            ss << "Parameter batch_size must be in (0," << MAX_INT32_VALUE << "], got " << jsonObj[key] << ".";
            return false;
        }
        return true;
    });
}

bool AssignMaxNewTokens(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "max_new_tokens";
    return ParametersChecker::Int32JsonCheck(
        jsonObj, key, param->maxNewTokens, error, [&](auto value, auto &ss) {
            if (!(value > 0 && value <= MAX_INT32_VALUE)) {
                ss << "Parameter max_new_tokens must be in (0," << MAX_INT32_VALUE << "], got " << jsonObj[key] << ".";
                return false;
            }
            return true;
        });
}

bool AssignBoolValue(const OrderedJson &jsonObj, const std::string &key, bool &value, std::string &error) noexcept
{
    return ParametersChecker::BooleanJsonCheck(jsonObj, key, value, error);
}

bool AssignPriority(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string key = "priority";
    return ParametersChecker::UInt64JsonCheck(
        jsonObj, key, tmpReq->priority, error, [&](auto value, auto &ss) {
            if (value < 1 || value > MAX_PRIORITY) {
                ss << "Parameter priority must be in [1," << MAX_PRIORITY << "], got " << value << ".";
                return false;
            }
            return true;
        });
}

bool AssignTimeout(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "timeout";
    return ParametersChecker::UInt64JsonCheck(jsonObj, key, param->timeout, error, [&](auto value, auto &ss) {
        if (value == 0 || value > MAX_TIMEOUT_SECOND) {
            ss << "Parameter timeout must be strictly positive and small than 65535 seconds, got "
            << jsonObj[key] << ".";
            return false;
        }
        return true;
    });
}

bool AssignDetails(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept
{
    const std::string key = "details";
    return ParametersChecker::BooleanJsonCheck(jsonObj, key, param->showDetails, error);
}

bool AssignOpenAILogprobs(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept
{
    const std::string logprobsKey = "logprobs";
    const std::string topLogprobsKey = "top_logprobs";
    if (!ParametersChecker::OptionalBooleanJsonCheck(jsonObj, logprobsKey, tmpReq->logprobs, error)) {
        return false;
    }
    if (!ParametersChecker::OptionalUint32JsonCheck(
        jsonObj, topLogprobsKey, tmpReq->topLogprobs, error, [&](auto value, auto &ss) {
        if (!(value <= MAX_OPENAI_TOP_LOGPROBS)) {
            ss << "Parameter top logprobs must be in [0, 20], got " << jsonObj[topLogprobsKey] << ".";
            return false;
        }
        return true;
        })) {
        return false;
    }
    if (tmpReq->logprobs.has_value() && !tmpReq->logprobs.value() && tmpReq->topLogprobs.has_value()) {
        error = "Value error, when using top_logprobs, logprobs must be true or null in chat completions";
        return false;
    }
    if (tmpReq->logprobs.has_value() && tmpReq->logprobs.value() && !tmpReq->topLogprobs.has_value()) {
        tmpReq->topLogprobs = 0;
    }
    if (!tmpReq->logprobs.has_value() && tmpReq->topLogprobs.has_value()) {
        tmpReq->logprobs = true;
    }
    return true;
}

bool AssignStream(const OrderedJson &jsonObj, InferParamSPtr inferParam, std::string &error) noexcept
{
    const std::string key = "stream";
    return ParametersChecker::BooleanJsonCheck(jsonObj, key, inferParam->streamMode, error);
}

bool CheckMultimodalUrlFromJson(const OrderedJson &jsonObj, std::string &error) noexcept
{
    const std::string imageKey = "image_url";
    const std::string videoKey = "video_url";
    const std::string audioKey = "audio_url";
    const std::string textKey = "text";

    uint32_t allMediaNum = 0;
    for (auto &param : jsonObj) {
        if (param.contains(textKey) && !param[textKey].is_string()) {
            error = "Request param text must be string";
            return false;
        }
        if (!param.contains(imageKey) && !param.contains(videoKey) && !param.contains(audioKey)) {
            continue;
        }
        if (param.contains(imageKey) && param[imageKey].is_null()) {
            error = "Request param image_url null";
            return false;
        }
        if (param.contains(audioKey) && param[audioKey].is_null()) {
            error = "Request param audio_url null";
            return false;
        }
        if (param.contains(videoKey) && param[videoKey].is_null()) {
            error = "Request param video_url null";
            return false;
        }

        allMediaNum++;
        if (allMediaNum > MAX_MULTIMODAL_URL_NUM) {
            error = "The number of multimodal url should be no more than " + std::to_string(MAX_MULTIMODAL_URL_NUM);
            return false;
        }
    }
    return true;
}

bool AssignLoraId(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &modelName, std::string &error)
{
    auto res = JsonParse::CheckOptionalItemType(jsonObj, "model", OrderedJson::value_t::string, error);
    if (!res.isCorrectType) {
        return false;
    }
    if (res.isPresent) {
        const std::vector<ModelDeployConfig> &configs = GetModelDeployConfig();
        const std::string masterModelName = (!configs.empty() ? configs[0].modelName : std::string{});
        if (modelName != masterModelName) {
            tmpReq->loraId = jsonObj["model"];
        } else {
            tmpReq->loraId = std::string("None");
        }
    }
    return true;
}
} // namespace mindie_llm