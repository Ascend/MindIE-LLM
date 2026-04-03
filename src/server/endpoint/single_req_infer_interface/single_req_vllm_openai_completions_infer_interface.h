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

#ifndef ENDPOINT_VLLM_OPENAI_COMPLETIOINS_INFER_H
#define ENDPOINT_VLLM_OPENAI_COMPLETIOINS_INFER_H

#include <cstdint>
#include <vector>
#include "httplib.h"
#include "single_req_infer_interface_base.h"
#include "infer_param.h"
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
class SingleReqVllmOpenAiCompletionsInferInterface : public SingleReqInferInterfaceBase {
public:
    explicit SingleReqVllmOpenAiCompletionsInferInterface(
        const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase, bool isReCompute = false,
        const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens,
                           RespBodyQueue &jsonStrings, const uint64_t &timestamp = 0) override;
    void SetDMIReComputeBuilder() override;
    const InferParam::FeatureSupport &GetFeatureSupport() const override
    {
        static constexpr InferParam::FeatureSupport kSupport{true, true, true, false};
        return kSupport;
    }
    bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) override;
protected:
    bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) override;
    void SendStreamResponse(RespBodyQueue &jsonStrings) override;
private:
    bool CheckModelName(const std::string &modelName, std::string &foundModelName);
    bool AssignModel(nlohmann::ordered_json &body, std::string &model, std::string &error);
    bool SetReturnSeqCount(RequestSPtr req, std::string& msg);
    bool CheckReturnSeqCount(RequestSPtr req, std::string& msg) const;
    std::string ChangeUtf8Str(std::string &input) const;
    bool EncodeNonStreamJsonObject(RespBodyQueue &jsonStrs,
        std::map<uint64_t, std::string>& responseTextMap, const bool &needSort = false);
    bool EncodeResponse(RespBodyQueue &jsonStrs);
    bool EncodeStreamJsonObject(RespBodyQueue &jsonStrings,
        std::map<uint64_t, std::string>& responseTextMap, const bool &needSort = false) noexcept;
    bool EncodeStreamResponse(RespBodyQueue &jsonStrings) noexcept;
    std::string BuildReComputeBody(const std::vector<BestNTokens>& tokens);
    void BuildStopWords(nlohmann::ordered_json& newReqJsonObj);
    std::map<uint64_t, std::string> cachedStreamTextMap{};
    // encode logprobs
    bool EncodeLogprobsFullText(nlohmann::ordered_json &choiceJsonObj, const uint64_t seqId) noexcept;
    bool EncodeLogprobsStream(nlohmann::ordered_json &choiceJsonObj, const uint64_t seqId,
        const StreamCache& cache) noexcept;
    bool EncodeSingleTokenLogrobs(nlohmann::ordered_json &singleToken, const uint64_t &baseIndex,
        const uint64_t seqId, const StreamCache* cache = nullptr) noexcept;
    uint64_t wordOffset = 0;
};
} // namespace mindie_llm

#endif  // ENDPOINT_VLLM_OPENAI_COMPLETIOINS_INFER_H