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

#ifndef ENDPOINT_INFER_PARAM_H
#define ENDPOINT_INFER_PARAM_H

#include <cstdint>
#include <optional>
#include <sstream>
#include "httplib.h"
#include "nlohmann/json.hpp"
#include "http_rest_resource.h"
#include "endpoint_def.h"
#include "base64_util.h"

namespace mindie_llm {
struct InferParam {
#ifdef UT_ENABLED
    uint64_t timeout = 1;
#else
    uint64_t timeout = 65535;
#endif
    // user request id
    std::optional<std::string> userInputId;
    // origin prompt
    std::string textInput;
    // tools call json
    std::optional<std::string> toolCallsJson;

    bool streamMode = false;
    int32_t maxNewTokens = MAX_NEW_TOKENS_DFT; // The number of tokens remaining for inference
    size_t outputLenOffset = 0; // D节点或重计算场景，maxNewTokens需要减去已输出的token数

    // Defaults to true because non-vLLM/OpenAI interfaces handle max_new_tokens by default.
    // Only vLLM and OpenAI interfaces need to decide between user input or config parameters.
    bool isMaxNewTokensSetByUser = true;

    bool showDetails = false;
    bool returnFullText = false;
    bool decoderInputDetails = false;
    int32_t batchSize = -1;
    uint32_t truncate = 0;
    std::string tools;
    std::string toolChoice;
    nlohmann::ordered_json toolsObject;
    nlohmann::ordered_json toolChoiceObject;
    bool useToolsCall = false;
    std::optional<bool> enableThinking;
    bool isChatReq = false;
    std::string adapterId;
    std::string model_;
    std::vector<std::string> outputNames; // triton token
    std::map<uint64_t, uint32_t> prevDecodeIndex{};
    std::map<uint64_t, uint32_t> currentDecodeIndex{};
    size_t preOutputTokenNum = 0;
    std::map<uint64_t, std::string> postSingleText{}; // text produced in current round
    std::map<uint64_t, std::string> respStreamStr{}; // accumulated text produced for current request
    Metrics metrics;

    struct FeatureSupport {
        bool useBeamSearch = false;
        bool useLogprobs = false;
        bool useBestOfN = true;
        bool useFunctionCall = false;
    };

    struct ValidationContext {
        // 端点能力
        FeatureSupport endpoint;
        // 服务/模型级能力（从配置/插件解析）
        bool pluginEnabled = false;
        bool deepseekEnabled = false;
        // 运行形态
        bool streamMode = false;
        bool isDmiMode = false;
        bool enableThinking = false;
        // 入口标识
        bool isVllmEntrance = false;
        // 本次请求参数（与组合规则相关）
        bool reqUseToolsCall = false;
        bool reqUseBeamSearch = false;
        bool reqLogprobs = false;
        bool reqTopLogprobsSet = false;
        bool reqStop = false;
        bool reqStopTokenIds = false;
        bool reqIncludeStopStrInOutput = false;
        uint32_t reqBestOf = 1;
        uint32_t reqN = 1;
        float reqTemperature = 0.0f;
        
        std::string ToString() const
        {
            std::ostringstream oss;
            oss << "ValidationContext {\n";
            oss << "  endpoint: {\n";
            oss << "    useBeamSearch: " << (endpoint.useBeamSearch ? "true" : "false") << ",\n";
            oss << "    useLogprobs: " << (endpoint.useLogprobs ? "true" : "false") << ",\n";
            oss << "    useBestOfN: " << (endpoint.useBestOfN ? "true" : "false") << ",\n";
            oss << "    useFunctionCall: " << (endpoint.useFunctionCall ? "true" : "false") << "\n";
            oss << "  },\n";
            oss << "  pluginEnabled: " << (pluginEnabled ? "true" : "false") << ",\n";
            oss << "  streamMode: " << (streamMode ? "true" : "false") << ",\n";
            oss << "  isDmiMode: " << (isDmiMode ? "true" : "false") << ",\n";
            oss << "  enableThinking: " << (enableThinking ? "true" : "false") << ",\n";
            oss << "  isVllmEntrance: " << (isVllmEntrance ? "true" : "false") << ",\n";
            oss << "  reqUseToolsCall: " << (reqUseToolsCall ? "true" : "false") << ",\n";
            oss << "  reqUseBeamSearch: " << (reqUseBeamSearch ? "true" : "false") << ",\n";
            oss << "  reqLogprobs: " << (reqLogprobs ? "true" : "false") << ",\n";
            oss << "  reqTopLogprobsSet: " << (reqTopLogprobsSet ? "true" : "false") << ",\n";
            oss << "  reqStop: " << (reqStop ? "true" : "false") << ",\n";
            oss << "  reqStopTokenIds: " << (reqStopTokenIds ? "true" : "false") << ",\n";
            oss << "  reqIncludeStopStrInOutput: " << (reqIncludeStopStrInOutput ? "true" : "false") << ",\n";
            oss << "  reqBestOf: " << reqBestOf << ",\n";
            oss << "  reqN: " << reqN << ",\n";
            oss << "  reqTemperature: " << reqTemperature << "\n";
            oss << "}";
            return oss.str();
        }
    };

    // 统一能力校验：根据端点能力、服务能力、请求参数/运行形态，拦截不支持的组合
    // 返回true表示通过；返回false并在error中携带原因。
    bool ValidateFeatureCompatibility(const ValidationContext &ctx, std::string &error,
        bool dmiSupportStopWords = false) const noexcept;
private:
    bool ValidateFeatureDmi(const ValidationContext &ctx, std::string &error,
        bool dmiSupportStopWords = false) const noexcept;
    bool ValidateFeatureBeamSearch(const ValidationContext &ctx, std::string &error) const noexcept;
    bool ValidateFeatureOverlay(const ValidationContext &ctx, std::string &error) const noexcept;
};

using InferParamSPtr = std::shared_ptr<InferParam>;
using OrderedJson = nlohmann::ordered_json;

bool AssignDoSample(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignRepetitionPenalty(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error,
                             double maxValue = MAX_REPETITION_PENALTY) noexcept;
bool AssignSeed(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignStopStrings(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error,
                       bool isNumStopStrLimited = false, uint32_t maxLength = MAX_TOTAL_STOP) noexcept;
bool AssignStopStringList(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error,
                          bool isNumStopStrLimited = false, uint32_t maxLength = MAX_TOTAL_STOP) noexcept;
bool AssignStopSingleString(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error,
                            uint32_t maxLength = MAX_TOTAL_STOP) noexcept;
bool AssignPresencePenalty(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignN(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignBestOf(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignBeamSearch(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignFrequencyPenalty(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignSkipSpecialTokens(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignIncludeStopStrInOutput(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error);
bool AssignTemperature(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error, bool allowLowerBound = false,
                       double maxValue = MAX_TEMPERATURE) noexcept;
bool AssignMaxTokens(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error);
bool AssignTopK(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error, bool allowLowerBound = false,
                bool allowNegativeOne = false) noexcept;
bool AssignTopP(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error,
                bool allowUpperBound = true) noexcept;
bool AssignIgnoreEos(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignStopTokenIds(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignTypicalP(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignWatermark(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignBatchSize(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept;
bool AssignMaxNewTokens(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept;
bool AssignBoolValue(const OrderedJson &jsonObj, const std::string &key, bool &value, std::string &error) noexcept;
bool AssignPriority(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignTimeout(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept;
bool AssignDetails(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) noexcept;
bool AssignOpenAILogprobs(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
bool AssignStream(const OrderedJson &jsonObj, InferParamSPtr inferParam, std::string &error) noexcept;
bool CheckMultimodalUrlFromJson(const OrderedJson &jsonObj, std::string &error) noexcept;
bool AssignLoraId(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &modelName, std::string &error);
bool AssignResponseFormat(const OrderedJson &jsonObj, RequestSPtr tmpReq, std::string &error) noexcept;
} // namespace mindie_llm

#endif // ENDPOINT_INFER_PARAM_H
