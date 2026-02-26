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

#ifndef ENDPOINT_COMMON_INFER_H
#define ENDPOINT_COMMON_INFER_H

#include <string>
#include <atomic>
#include <memory>
#include <ostream>
#include <limits>
#include <unordered_map>
#include <set>
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/chrono.hpp>
#include "httplib.h"
#include "random_generator.h"
#include "single_llm_req_handler_base.h"
#include "http_rest_resource.h"
#include "endpoint_def.h"
#include "infer_instances.h"
#include "infer_param.h"
#include "log.h"
namespace mindie_llm {

struct StopServiceOption {
    static std::atomic<bool> stopServiceFlag;
};
/**
 * @brief 推理请求处理基类，完成请求解析，调用推理引擎，回复响应等
 * @note 使用此类以及子类，不要直接调用构造函数创建对象，要使用 std::make_shared 创建对象
 */

class SingleReqInferInterfaceBase : public std::enable_shared_from_this<SingleReqInferInterfaceBase> {
public:
    explicit SingleReqInferInterfaceBase(const std::shared_ptr<SingleLLMReqHandlerBase> &singleLLMReqHandlerBase,
                                bool isReCompute = false, const std::vector<LoraParamSPtr> loraConfigs = {}) noexcept;
    ~SingleReqInferInterfaceBase();

public:
    void DecodeProcess(prefillAndDecodeCommunication::DecodeRequestResponse &response) noexcept;
    void Process() noexcept;
    bool GenerateInferRequest(std::string &msg) noexcept;
    void Stop() noexcept;
    bool ProcessResponseSingle(ResponseSPtr response, const uint64_t &timestamp) noexcept;
    bool ProcessResponseStream(ResponseSPtr response, const std::vector<BestNTokens> &bestNTokens,
                               RespBodyQueue &jsonObjs, const uint64_t &timestamp) noexcept;
    bool PostProcess(const std::vector<int64_t> &tokenIds, std::string &inferResult, const uint64_t &seqId,
                     bool decodeOneToken, bool requestEndFlag, uint32_t prevDecodeIndexLocal,
                     uint32_t currentDecodeIndexLocal, const uint64_t &timestamp = 0) noexcept;
    virtual bool BuildResponseJson(ResponseSPtr response, const std::vector<BestNTokens> &tempTokens,
                                    RespBodyQueue &jsonStrings, const uint64_t &timestamp = 0) = 0;
    RequestIdNew GetRequestId();
    Metrics &GetMetrics();
    virtual bool SetupInferParams(RequestSPtr tmpReq, std::string &msg) = 0;

public:
    virtual void SetDMIReComputeBuilder() = 0;
    // 端点能力声明：默认全 false，需要能力的端点自行覆写
    virtual const InferParam::FeatureSupport &GetFeatureSupport() const
    {
        static constexpr InferParam::FeatureSupport
            kDefault{}; // 所有位默认 false / useBestOfN 默认为 true? 我们设为默认初始化: useBestOfN=true在结构体定义中
        return kDefault;
    }

protected:
    virtual bool ValidateAndPrepareReqToken(nlohmann::ordered_json &body, std::string &msg, uint64_t &timestamp) = 0;
    virtual void SendStreamResponse([[maybe_unused]] RespBodyQueue &jsonStrs){};
    bool GetTokensFromInput(const std::string &input, std::vector<std::int64_t> &requestTokens,
                            std::vector<std::int64_t> &responseTokens, std::string &errorMsg);
    bool GetUniqueSequenceId(uint64_t &seqId);
    bool DecodeSingleToken(std::vector<int64_t> &tokenIds, std::string &output,
        const uint32_t &prevDecodeIndex, const uint32_t &currentDecodeIndex, const bool &skipSpecialTokens);
    Status InsertPerfInfoIntoJson(nlohmann::ordered_json &body, const std::vector<int32_t> perfInfoTypeList,
        const std::vector<std::string> keyList);
    bool ValidateChatTemplateKwargs(const nlohmann::ordered_json& jsonObj, InferParam& param,
            std::string& error) const;
    void ParseDetokenizedOutput(std::string &inferResult, const uint64_t &seqId, const bool &decodeOneToken);
    void ConvertTokenToMap(const std::vector<BestNTokens> &decodeResult);
    bool ParseChatTemplate(const nlohmann::ordered_json &jsonObj, std::string &error) const;
    std::string BuildReComputeInput();
    // userInputId -> requestId_
    inline static ConcurrentMap<std::string, std::string> userInputIdMap_;

protected:
    RequestIdNew requestId_;
    std::shared_ptr<SingleLLMReqHandlerBase> singleLLMReqHandlerBase_{nullptr};
    bool isReCompute_{false};
    std::vector<LoraParamSPtr> loraConfigs_;

    nlohmann::ordered_json reqJsonBody_;
    std::vector<int64_t> reqTokens_{};
    size_t oriReqTokenLen_ = std::numeric_limits<size_t>::max();
    std::map<uint64_t, std::vector<int64_t>> respTokenMap{};

    std::map<uint64_t, InferStatusType> eosMap{};
    std::map<uint64_t, int64_t> truncationIdMap{};
    std::map<uint64_t, double> probesMap{}; // <sequenceId, accumulatedLogprobs>
    static const std::string requestIdPrefix;
    
    // parsingContentFlag: first -- there is reasoning content; second -- there is content
    std::map<uint64_t, std::pair<bool, bool>> parsingContentFlag{};
    std::map<uint64_t, std::optional<DetokenizeExtraInfo>> detokenizeExtraInfo{};
    std::map<uint64_t, std::string> finishReasonMap{};

    std::map<uint64_t, std::vector<int64_t>> postTokenIdMap{};    // accumulate tokenIds
    std::map<uint64_t, std::vector<int64_t>> logprobsTokensMap{}; // candidate tokens of picked tokenId
    std::map<uint64_t, std::vector<float>> logprobsMap{};         // logprobs of candidate tokens
    std::map<uint64_t, std::vector<float>> pickedLogprobMap{};

    // save response text in stream mode
    std::map<uint64_t, std::string> fullTextMap{}; // accumulate response text
    std::map<uint64_t, std::string> reasoningContentFullTextMap{};
    std::map<uint64_t, std::string> reasoningContentStreamMap{};
    std::map<uint64_t, int64_t> reasoningTokens{};
    std::map<uint64_t, Json> toolsCallObjectMap{};
    std::map<uint64_t, std::string> toolsCallContentMap{};
    std::map<uint64_t, Json> toolsCallObjectStreamMap{};
    std::map<uint64_t, std::string> toolsCallContentStreamMap{};
    std::map<uint64_t, bool> toolsCalled{};
    std::map<uint64_t, bool> skipCurrentRoundMap{};
    std::map<uint64_t, bool> updateIndexMap{};
    std::map<uint64_t, std::vector<int64_t>> postSingleTokenMap{}; // response text for current round
    std::set<uint64_t> endedSeqIds{}; // seqId of ended sequence
    bool isEnd = false; // stream response finished

    std::map<uint64_t, std::string> windowText{};
    std::vector<std::pair<std::string, std::map<uint64_t, std::string>>> windowJsonTextVec{};
    std::vector<std::pair<std::string, std::map<uint64_t, std::string>>> windowJsonTextOut{};

    InferParamSPtr inputParam{nullptr};
    RequestSPtr request_{nullptr};
    // vllm入口标识位，用于标识请求是否来自vllm_infer入口
    bool isVllmEntrance = false;

    uint64_t timestamp_ = 0;
    std::string model;
    // calculated by use_beam_search, best_of, n
    size_t returnSeqCount_ = 1;
    // For openai impl, we need to send a special response for the first time in stream mode
    bool neverSendResponse_ = true;

    struct StreamCache {
        std::map<uint64_t, double> probesMap{};
        std::map<uint64_t, std::string> fullTextMap{};
        std::map<uint64_t, InferStatusType> eosMap{};
        std::map<uint64_t, std::vector<int64_t>> postTokenIdMap{};
        std::map<uint64_t, std::string> postSingleText{};
        std::map<uint64_t, std::vector<int64_t>> postSingleTokenMap{};
        std::map<uint64_t, std::string> finishReasonMap{};
        std::map<uint64_t, std::vector<int64_t>> logprobsTokensMap{};
        std::map<uint64_t, std::vector<float>> logprobsMap{};
        std::map<uint64_t, std::vector<float>> pickedLogprobMap{};
        std::map<uint64_t, uint32_t> prevDecodeIndex{};
        std::map<uint64_t, uint32_t> currentDecodeIndex{};
        std::map<uint64_t, std::string> reasoningContentFullTextMap{};
        std::map<uint64_t, std::string> reasoningContentStreamMap{};
        std::map<uint64_t, Json> toolsCallObjectMap{};
        std::map<uint64_t, std::string> toolsCallContentMap{};
        std::map<uint64_t, Json> toolsCallObjectStreamMap{};
        std::map<uint64_t, std::string> toolsCallContentStreamMap{};
        std::map<uint64_t, bool> skipCurrentRoundMap{};
        std::map<uint64_t, std::u16string> u16TokenText{};
        std::map<uint64_t, bool> canOutput{};
        std::map<uint64_t, std::pair<bool, bool>> parsingContentFlag{};
        std::optional<size_t> curTokenNum;

        template <typename T> static std::string FormatValue(const T &value)
        {
            if constexpr (std::is_arithmetic_v<T>) {
                std::ostringstream oss;
                oss << value;
                return oss.str();
            } else if constexpr (std::is_same_v<T, std::string>) {
                return "\"" + value + "\"";
            } else if constexpr (std::is_same_v<T, std::u16string>) {
                return "\"" + std::string(value.begin(), value.end()) + "\"";
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>> || std::is_same_v<T, std::vector<float>>) {
                std::ostringstream oss;
                oss << "[ ";
                for (const auto &elem : value) {
                    oss << elem << " ";
                }
                oss << "]";
                return oss.str();
            } else {
                return "<unsupported type>";
            }
        }

        template <typename K, typename V>
        static std::string FormatMap(const std::map<K, V> &map, const std::string &name)
        {
            std::ostringstream oss;
            oss << name << ":\n";
            for (const auto &[key, value] : map) {
                oss << "  [" << key << "] -> " << FormatValue(value) << "\n";
            }
            return oss.str();
        }

        std::string ToString() const
        {
            std::ostringstream oss;
            oss << FormatMap(probesMap, "probesMap");
            oss << FormatMap(fullTextMap, "fullTextMap");
            oss << FormatMap(eosMap, "eosMap");
            oss << FormatMap(postTokenIdMap, "postTokenIdMap");
            oss << FormatMap(postSingleText, "postSingleText");
            oss << FormatMap(finishReasonMap, "finishReasonMap");
            oss << FormatMap(logprobsTokensMap, "logprobsTokensMap");
            oss << FormatMap(logprobsMap, "logprobsMap");
            oss << FormatMap(pickedLogprobMap, "pickedLogprobMap");
            oss << FormatMap(prevDecodeIndex, "prevDecodeIndex");
            oss << FormatMap(currentDecodeIndex, "currentDecodeIndex");
            oss << FormatMap(u16TokenText, "u16TokenText");
            oss << FormatMap(canOutput, "canOutput");
            return oss.str();
        }
    };
    std::vector<StreamCache> streamCache{};
    bool GetAvailableOutputCache(std::vector<StreamCache> &cacheArr);
    // 组装统一校验上下文，避免各端点重复代码
    InferParam::ValidationContext BuildValidationContext() const;

private:
    bool PushLatestCache(std::string &errMsg);
    bool ProcessStreamCacheTruncationId(std::string &errMsg);
    bool ProcessStreamCacheWindowSize();
    void ClearUnusedDecodeCache(const std::set<uint64_t> &usedSeqIds);
    void DumpDecodeCache(const std::map<uint64_t, std::vector<uint64_t>> &currentSeqIds);
};
} // namespace mindie_llm

#endif // ENDPOINT_COMMON_INFER_H
