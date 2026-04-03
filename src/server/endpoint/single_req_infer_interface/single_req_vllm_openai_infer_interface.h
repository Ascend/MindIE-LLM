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

#ifndef ENDPOINT_VLLM_OPENAI_INFER_H
#define ENDPOINT_VLLM_OPENAI_INFER_H

#include <cstdint>
#include <vector>
#include "httplib.h"
#include "single_req_infer_interface_base.h"
#include "infer_param.h"

using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
/**
 * @brief Vllm OpenAi 格式的推理请求处理类
 */
class SingleReqVllmOpenAiInferInterface : public SingleReqInferInterfaceBase {
public:
    explicit SingleReqVllmOpenAiInferInterface(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                               bool isReCompute = false,
                                               const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens,
                           RespBodyQueue &jsonStrings, const uint64_t &timestamp = 0) override;
    void SetDMIReComputeBuilder() override;
    const InferParam::FeatureSupport &GetFeatureSupport() const override
    {
        static constexpr InferParam::FeatureSupport kSupport{true, true, true, true};
        return kSupport;
    }
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;
protected:
    // Model resolution. Default: check model exists in deploy config (LoRA mapping).
    // Openai overrides this to not support LoRA->base model resolution.
    virtual bool ParseModelName(OrderedJson &body, std::string &outModel, std::string &err);
    bool CheckModelName(const std::string &modelName, std::string &foundModelName) const;
    bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) override;
    void SendStreamResponse(RespBodyQueue &jsonStrings) override;
    std::string ChangeUtf8Str(std::string &input) const;
    bool EncodeNonStreamJsonObject(RespBodyQueue &jsonStrs, std::map<uint64_t, std::string> &responseTextMap,
                                   const bool &needSort = false);
    bool EncodeResponse(RespBodyQueue &jsonStrs);
    bool EncodeStreamJsonObject(RespBodyQueue &jsonStrings, std::map<uint64_t, std::string> &responseTextMap,
                                const bool &needSort = false) noexcept;
    bool EncodeStreamResponse(RespBodyQueue &jsonStrings) noexcept;
    bool ParseTextInput(nlohmann::ordered_json &body, std::string &msg);
    bool SetReturnSeqCount(RequestSPtr req, std::string &errMsg);
    std::string BuildVllmOpenAIReComputeBody(const std::vector<BestNTokens> &tokens);
    void BuildStopWords(nlohmann::ordered_json& newReqJsonObj);
    void BuildThinkingConfig(nlohmann::ordered_json& newReqJsonObj);
    bool ParseToolCall(nlohmann::ordered_json &body, std::string &msg);
    bool ValidMessagesArray(OrderedJson &body, OrderedJson &messges, std::string &msg) const;
    void FilterToolChoice(OrderedJson &filterTools);
    bool PrepareMessageArray(nlohmann::ordered_json &body, std::string &msg, OrderedJson& messageArray);
    bool ValidAssistantMessage(OrderedJson &message, std::string &msg) const;
    bool ValidToolCall(OrderedJson &toolCalls, std::string &msg) const;
    bool ValidToolCallID(OrderedJson &paramItem, std::string &msg) const;
    bool CheckTextInputLen(std::string &textInput, std::string &msg) const;
    // encode logprobs
    void GetUtf8CodeFromStr(const std::string &input, nlohmann::ordered_json &bytes) const;
    bool EncodeLogprobsFullText(nlohmann::ordered_json &choiceJsonObj, const uint64_t seqId) noexcept;
    bool EncodeLogprobsStream(nlohmann::ordered_json &choiceJsonObj, const uint64_t seqId,
                              const StreamCache &cache) noexcept;
    bool EncodeSingleTokenLogrobs(nlohmann::ordered_json &singleToken, const uint64_t &baseIndex, const uint64_t seqId,
                                  const StreamCache *cache = nullptr) noexcept;
    bool ParseOpenAiTools(const OrderedJson &jsonObj, std::string &error);
    bool AssignOpenAiTools(const OrderedJson &jsonObj, std::string &error);
    bool AssignOpenAiToolChoice(const OrderedJson &jsonObj, std::string &error);
    bool GetToolChoiceAsObject(const OrderedJson &toolChoice, std::string &error);
    bool GetToolChoiceAsString(const std::string &toolChoice, std::string &error);
    bool CheckToolObject(const OrderedJson &tool, std::string &error);
    bool CheckFunction(const OrderedJson &toolParam, std::string &error) const;
    bool CheckToolType(const OrderedJson &toolParam, std::string &error) const;
    void AssignLora(const OrderedJson &jsonObj, RequestSPtr tmpReq) const;
};
} // namespace mindie_llm

#endif // ENDPOINT_VLLM_OPENAI_INFER_H
