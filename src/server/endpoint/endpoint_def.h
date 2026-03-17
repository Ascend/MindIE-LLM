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

#ifndef ENDPOINT_DEF_H
#define ENDPOINT_DEF_H

#ifndef CPPHTTPLIB_HEADER_MAX_LENGTH
#define CPPHTTPLIB_HEADER_MAX_LENGTH 16384
#endif

#include <mutex>
#include <memory>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <queue>
#include <map>
#include <limits>
#include <optional>
#include <nlohmann/json.hpp>
#include "httplib.h"
#include "dresult_event_dispatcher.h"
#include "data_type.h"
#include "basic_types.h"
#include "env_util.h"
#include "infer_tokenizer.h"
#include "request_response/response.h"

#ifdef WITH_PROF
#include "msServiceProfiler/msServiceProfiler.h"
#else
#define PROF(...) 0
#endif
using InferDataType = mindie_llm::InferDataType;

namespace mindie_llm {
class EndpointDef {};

struct BestNTokens {
    SequenceId seqId = 0;
    SequenceId parentSeqId = 0;
    InferStatusType finishReason = InferStatusType::ITERATION_CONTINUE;
    std::vector<int64_t> tokens{};
    double cumLogprobs = 0.0;
    std::vector<float> logprob{};
    std::vector<int64_t> logprobsTokens{};
    std::vector<float> logprobs{};
    int64_t truncationIndex = 0;
};
    // sequence with id = PRESET will append to front of the new sequences
    constexpr uint64_t SPECIAL_SEQ_ID_PRESET = 0;
    uint32_t GetMaxInputLen();
    extern std::shared_ptr<DResultEventDispatcher> dResultDispatcher;
    constexpr uint32_t CV_WAIT_TIME = 600;
    constexpr uint32_t SIMULATE_CV_WAIT_TIME = 10;
    constexpr uint32_t MAX_TOKENS_NUM = 1024 * 1024; // Token id<=1M
    constexpr uint32_t MAX_STOP_STRING_LEN = 1024;
    constexpr uint32_t MAX_STOP_STRING_NUM = 1024;
    constexpr uint32_t MAX_TOTAL_STOP = 32 * 1024; // 32k
    constexpr uint32_t MAX_LORA_ID_LENGTH = 256; // LORA ID 长度 不允许超过256
    constexpr uint32_t MAX_INPUT_ID_LENGTH = 256; // triton ID 长度 不允许超过256
    constexpr auto IP_PORT_DELIMITER = ";"; // 使用分号来分割ip和端口

    constexpr int32_t MAX_NEW_TOKENS_DFT = 20;
    constexpr int32_t VLLM_MAX_NEW_TOKENS_DFT = 16;
    constexpr float PRESENCY_PENALTY_DFT = 0.0;
    constexpr float FREQUENCY_PENALTY_DFT = 0.0;
    constexpr uint64_t SEED_DFT = 1;
    constexpr float TEMPERATURE_DFT = -1.0;
    constexpr float TOP_P_DFT = 1.0;
    constexpr float REPETITION_PENALTY_DFT = 1.0;
    constexpr int32_t TOP_K_DFT = -1;
    constexpr float TYPICAL_P_DFT = -1.0;
    constexpr uint64_t PRIORITY_DFT = 5;
    constexpr uint32_t MAX_MULTIMODAL_URL_NUM = 20;  // multimodal support max multimodal url number
    constexpr uint64_t MAX_PRIORITY = 5;
    constexpr uint64_t MAX_TIMEOUT_SECOND = 65535; // 65535 seconds
    constexpr uint32_t MIN_BEST_OF = 1;
    constexpr uint32_t MAX_BEST_OF = 128;
    constexpr uint32_t MIN_N = 1;
    constexpr uint32_t MAX_N = 8192;

    constexpr float MAX_FLOAT_VALUE = std::numeric_limits<float>::max();
    constexpr float MAX_TEMPERATURE = std::numeric_limits<float>::max();
    constexpr float MAX_REPETITION_PENALTY = std::numeric_limits<float>::max();
    
    constexpr int32_t MAX_INT32_VALUE = std::numeric_limits<int32_t>::max();
    constexpr uint64_t MAX_UINT64_VALUE = std::numeric_limits<uint64_t>::max();

    constexpr uint8_t PREFILL_CALLBACK_METRICS_TAG = 1;
    constexpr uint8_t DECODE_CALLBACK_METRICS_TAG = 2;

    // MINDIE_LLM_BENCHMARK_ENABLE 环境变量取值常量
    // 1: 同步enable; 2: 异步enable; 其他取值: 关闭
    constexpr int32_t BENCHMARK_ENABLE_SYNC = 1;
    constexpr int32_t BENCHMARK_ENABLE_ASYNC = 2;

    std::u16string GetU16Str(const std::string &inputStr, std::string *error = nullptr);
    std::wstring String2Wstring(const std::string& str, std::string *error = nullptr);
    std::string TransformTruncation(std::u16string inputStr, int64_t truncationStart, int64_t truncationEnd,
        std::string *error = nullptr);
    std::string GetUriParameters(const httplib::Request &request, uint32_t index);
    std::string GetFinishReasonStr(InferStatusType finishReason);
    extern std::atomic<bool> g_health;
    class HealthManager {
    public:
        // 获取全局健康状态的静态方法
        static std::atomic<bool>& GetHealth();
        static void UpdateHealth(bool healthStatus);
        // 禁止拷贝和赋值
        HealthManager(const HealthManager&) = delete;
        HealthManager& operator=(const HealthManager&) = delete;
    };

    struct HttpReqHeadersOption {
        InferReqType reqType;
        bool isReCompute = false;
        bool isFlexLocal = false;
    };

    enum MsgType : uint16_t {
        MSG_TYPE_INVALID,
        MSG_TYPE_TGI,
        MSG_TYPE_GENERAL_TGI,
        MSG_TYPE_VLLM,
        MSG_TYPE_OPENAI,
        MSG_TYPE_VLLM_OPENAI,
        MSG_TYPE_VLLM_OPENAI_COMP,
        MSG_TYPE_KSERVE,
        MSG_TYPE_INFER,
        MSG_TYPE_TRITON,
        MSG_TYPE_TRITON_TOKEN
    };

    struct Metrics {
        std::chrono::steady_clock::time_point e2eStartingTime;
        std::chrono::system_clock::time_point sysE2eStartingTime;
        std::chrono::steady_clock::time_point startingTime;  // 开始时间
        std::chrono::steady_clock::time_point endingTime;    // 结束时间
        std::chrono::steady_clock::time_point lastTokenTime; // 最后一个Token的生成时间
        uint64_t firstTokenCost{0}; // 首Token耗时
        uint64_t lastTokenCost{0};  // 最后Token耗时
        std::vector<uint64_t> decodeTime;      // 每一轮decode的耗时
        std::vector<int64_t> batchSize;  // batch size
        std::vector<int64_t> queueWaitTime;  // 队列等待时间
        std::vector<int64_t> prefixCachedTokenNums; // 前缀缓存的Token数
        bool isPrefill{ true }; // 判断这一轮记录的是否是第一个token
        std::queue<uint8_t> callbackIndexQue{};
        uint64_t callbackIndex{0};
    };

    struct TokenizerContents {
        std::optional<std::string> content{};
        std::optional<std::string> reasoningContent{};
        std::optional<nlohmann::ordered_json> toolCalls{};
        std::optional<DetokenizeExtraInfo> detokenizeStatus{};
        std::optional<bool> needUpdateIndex{};

        TokenizerContents() = default;

        TokenizerContents(std::string contentVal, std::string reasoningContentVal,
            nlohmann::ordered_json toolCallsVal) : content(std::move(contentVal)),
            reasoningContent(std::move(reasoningContentVal)),
            toolCalls(std::move(toolCallsVal)) {}

        TokenizerContents(std::string contentVal, std::string reasoningContentVal) : content(std::move(contentVal)),
            reasoningContent(std::move(reasoningContentVal)) {}

        explicit TokenizerContents(std::string contentVal) : content(std::move(contentVal)) {}
    };

    enum PerfInfoType : int32_t {
        PERF_BATCH_SZIE = 0,
        PERF_QUEUE_WAIT_TIME = 1
    };

    enum EpCode : int32_t {
        // P.S. error codes for End Point
        EP_OK = 0,
        EP_ERROR = 1,
        EP_INVALID_PARAM = 2,
        EP_ALLOC_FAIL = 3,
        EP_NEW_OBJ_FAIL = 4,
        EP_NOT_INITIALIZED = 5,
        EP_INVALID_CONFIG = 6,
        EP_ALREADY_DONE = 7,
        EP_ALREADY_EXISTS = 8,
        EP_NOT_EXISTS = 9,
        EP_NOT_READY = 10,
        EP_NULL_PTR = 11,
        EP_SEND_HTTP_HEAD_ERR = 12,
        EP_PARSE_JSON_ERR = 13,
        EP_HTTP_COPY_ERR = 14,
        EP_CALLBACK_ERR = 15,
        EP_GET_PARAM_ERR = 16,
        EP_PARSE_NO_PARAM_ERR = 17,

        EP_PARSE_NO_INPUTS_ERR = 18,
        EP_PARSE_NO_MAX_TOKEN_ERR = 19,
        EP_PARSE_NO_REPETITION_PENALTY_ERR = 20,
        EP_PARSE_NO_STREAM_ERR = 21,
        /* add error code ahead of this */
        EP_MAX,
    };

    enum PullKVFlag : uint16_t {
        PULL_KV_SUCCESS = 0,
        PULL_KV_FAIL_REVERSIBLY = 2003,
        PULL_KV_FAIL_IRREVERSIBLY = 2004,
    };
} // namespace mindie_llm

#endif // OCK_ENDPOINT_DEF_H
