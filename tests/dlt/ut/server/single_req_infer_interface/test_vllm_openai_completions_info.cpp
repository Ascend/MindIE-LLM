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
#include <gtest/gtest.h>

#include "config_manager/config_manager_impl.h"
#include "mock_util.h"
#include "mockcpp/mockcpp.hpp"
#include "request.h"
#include "response.h"
#include "single_llm_pnd_req_handler.h"
#include "single_req_vllm_openai_completions_infer_interface.h"

using namespace mindie_llm;
namespace mindie_llm {
MOCKER_CPP_OVERLOAD_EQ(ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
class VllmOpenAiCompletionsInferTest : public testing::Test {
   protected:
    VllmOpenAiCompletionsInferTest() = default;
    void SetUp() {
        mockScheduleConfig_.maxBatchSize = 128;
        mockScheduleConfig_.maxPrefillBatchSize = 128;
        mockScheduleConfig_.maxN = 128;
        mockScheduleConfig_.maxIterTimes = 0;
        MOCKER_CPP(GetScheduleConfig, const ScheduleConfig &(*)()).stubs().will(returnValue(mockScheduleConfig_));
        MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(mockServerConfig_));

        httpRequest = httplib::Request();
        httpResponse = httplib::Response();
        request = std::make_shared<Request>(RequestIdNew("mockRequest"));
        response = std::make_shared<Response>(RequestIdNew("mockResponse"));
        requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
        pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
        inferInterface = std::make_shared<SingleReqVllmOpenAiCompletionsInferInterface>(pndReqHandler, false);
        inferInterface->inputParam = std::make_shared<InferParam>();
        inferInterface->request_ = std::make_shared<Request>();
    }

    void TearDown() { GlobalMockObject::verify(); }

    SingleReqInferInterfaceBase::StreamCache CreateTestStreamCache(uint64_t seqId) {
        SingleReqInferInterfaceBase::StreamCache cache;
        cache.postTokenIdMap[seqId] = {1001};
        cache.prevDecodeIndex[seqId] = 0;
        cache.pickedLogprobMap[seqId] = {-0.5};
        cache.logprobsTokensMap[seqId] = {1001, 1002, 1003};
        cache.logprobsMap[seqId] = {-0.5, -0.8, -1.2};
        cache.canOutput[seqId] = true;
        cache.eosMap[seqId] = InferStatusType::ITERATION_CONTINUE;  // 非结束
        return cache;
    }

    httplib::Request httpRequest;
    httplib::Response httpResponse;
    RequestSPtr request;
    ResponseSPtr response;
    std::string error;
    std::shared_ptr<RequestContext> requestContext;
    std::shared_ptr<SingleLLMPnDReqHandler> pndReqHandler;
    std::shared_ptr<SingleReqVllmOpenAiCompletionsInferInterface> inferInterface;
    ScheduleConfig mockScheduleConfig_;
    ServerConfig mockServerConfig_;
};

TEST_F(VllmOpenAiCompletionsInferTest, TestValidateAndPrepareReqToken) {
    OrderedJson body;
    std::string errorMsg, input;
    uint64_t timestamp;
    auto stubs = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool *, const std::string &,
                                                                     std::vector<int64_t> &, HeadFlag, uint64_t &))
                     .stubs();
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    // invalid values
    //      should return false given "model" is not string
    body["model"] = 1;
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given "model" is empty
    body["model"] = "";
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given "model" is not found in ModelDeployConfig
    body["model"] = "mockModel";
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    std::vector<ModelDeployConfig> mockDeployConfig = {{}};
    mockDeployConfig[0].modelName = "mockModel";
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    //      should return false given "prompt" is empty
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    body["prompt"] = "mockPrompt";
    EXPECT_TRUE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given tokenizer fails
    stubs.will(invoke(&MockTokenizerEncodeFail));
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    //      should return false given "prompt" is not string or array
    body["prompt"] = 1;
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given "prompt" is empty string
    body["prompt"] = "";
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given "prompt" is empty array
    body["prompt"] = OrderedJson::array();
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given "prompt" is not integer list
    body["prompt"] = OrderedJson::array({1, 2, "a"});
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return false given tokenizer fails
    auto encodeStubs =
        MOCKER_CPP(&TokenizerProcessPool::Decode,
                   Status(*)(TokenizerProcessPool *, std::vector<int64_t> &, std::string &, const uint64_t &, bool))
            .stubs();
    encodeStubs.will(invoke(&MockTokenizerDecodeFail));
    body["prompt"] = OrderedJson::array({1, 2, 3});
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    //      should return true given tokenize succeeds
    encodeStubs.will(invoke(&MockTokenizerDecodeSuccess));
    EXPECT_TRUE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
    inferInterface->isReCompute_ = true;
    MOCKER_CPP(&SingleReqInferInterfaceBase::GetTokensFromInput,
               bool (*)(const std::string &, std::vector<std::int64_t> &, std::vector<std::int64_t> &, std::string &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(inferInterface->ValidateAndPrepareReqToken(body, errorMsg, timestamp));
}

TEST_F(VllmOpenAiCompletionsInferTest, TestProcess) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "stream": false,
        "timeout": 1,
        "temperature": 1.0,
        "top_k": 1,
        "stop_token_ids": [1, 2, 3],
        "top_p": 0.5,
        "ignore_eos": false,
        "repetition_penalty": 1.0,
        "seed": 1,
        "typical_p": 0.5,
        "watermark": false
    })");
    std::string errorMsg;
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    inferInterface->Process();
    EXPECT_TRUE(request->temperature.has_value());
    EXPECT_NEAR(request->temperature.value(), 1.0, 1e-6);
    EXPECT_EQ(request->topK.value(), 1);
    EXPECT_NEAR(request->topP.value(), 0.5, 1e-6);
    EXPECT_EQ(request->seed, 1);
    EXPECT_NEAR(request->repetitionPenalty.value(), 1.0, 1e-6);
    EXPECT_EQ(inferInterface->inputParam->streamMode, false);
    EXPECT_EQ(inferInterface->inputParam->timeout, 1);
}

TEST_F(VllmOpenAiCompletionsInferTest, TestBuildResponseJson) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "stream": false,
        "timeout": 1,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 0.5,
        "ignore_eos": false,
        "repetition_penalty": 1.0,
        "seed": 1,
        "typical_p": 0.5,
        "watermark": false
    })");
    std::string errorMsg;
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    inferInterface->Process();
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    EXPECT_TRUE(inferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // should return false ProcessResponseSingle fails
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseSingle, bool (*)(ResponseSPtr, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(inferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(VllmOpenAiCompletionsInferTest, TestBuildStreamResponseJson) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "stream": true,
        "timeout": 1,
        "temperature": 1.0,
        "top_k": 1,
        "stop_token_ids": [1, 2, 3],
        "top_p": 0.5,
        "ignore_eos": false,
        "repetition_penalty": 1.0,
        "seed": 1,
        "typical_p": 0.5,
        "watermark": false
    })");
    std::string errorMsg;
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    inferInterface->Process();
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    // should return false ProcessResponseStream fails
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseStream,
               bool (*)(ResponseSPtr, const std::vector<BestNTokens> &, RespBodyQueue &, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(inferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(VllmOpenAiCompletionsInferTest, TestSetupInferParams) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    std::string errorMsg;
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "stream": true,
        "timeout": 1,
        "temperature": 1.0,
        "top_k": 1,
        "stop_token_ids": [1, 2, 3],
        "top_p": 0.5,
        "ignore_eos": false,
        "repetition_penalty": 1.0,
        "seed": 1,
        "typical_p": 0.5,
        "watermark": false
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    // invalid values
    //      should return false given "ignore_eos" is not bool
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({"ignore_eos": "a"})");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
    //      should return false given "best_of" is not equal to "n"
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "ignore_eos": false,
        "best_of": 3,
        "n": 2
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
    //      should return true given "best_of" is equal to "n"
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "ignore_eos": false,
        "best_of": 3,
        "n": 3
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    //      should return false given "best_of" is less than "n" and stream is false
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "stream": false,
        "ignore_eos": false,
        "best_of": 3,
        "n": 5
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(VllmOpenAiCompletionsInferTest, TestSetupInferParamsCase2) {
    std::string errorMsg;
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "logprobs": 5,
        "use_beam_search": true
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    // should return false given "logprobs" is greater than MAX_VLLM_OPENAI_TOP_LOGPROBS
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "logprobs": 6
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
    // should return false given "best_of" is not null and "n" is null
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "stream": true,
        "best_of": 3,
        "use_beam_search": false
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "stream": false,
        "best_of": 5,
        "n": 3
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(VllmOpenAiCompletionsInferTest, TestSetupInferParamsWithResponseFormat) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    std::string errorMsg;

    // valid response_format with json_object
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "response_format": {"type": "json_object"}
    })");
    EXPECT_TRUE(inferInterface->SetupInferParams(request, errorMsg));
    EXPECT_TRUE(request->responseFormat.has_value());
    EXPECT_EQ(request->responseFormat.value(), R"({"type":"json_object"})");

    // valid response_format with json_schema and valid name
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "my_test-schema_123",
                "schema": {"type": "object"}
            }
        }
    })");
    EXPECT_TRUE(inferInterface->SetupInferParams(request, errorMsg));
    EXPECT_TRUE(request->responseFormat.has_value());

    // invalid response_format - type is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "response_format": {"type": "text"}
    })");
    EXPECT_FALSE(inferInterface->SetupInferParams(request, errorMsg));

    // invalid response_format - json_schema.name is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        }
    })");
    EXPECT_FALSE(inferInterface->SetupInferParams(request, errorMsg));

    // valid response_format - json_schema.name
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "invalid@name",
                "schema": {"type": "object"}
            }
        }
    })");
    EXPECT_TRUE(inferInterface->SetupInferParams(request, errorMsg));

    // invalid response_format - json_schema.name is empty
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "",
                "schema": {"type": "object"}
            }
        }
    })");
    EXPECT_FALSE(inferInterface->SetupInferParams(request, errorMsg));
}

TEST_F(VllmOpenAiCompletionsInferTest, TestSendStreamResponse) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "mock test",
        "model": "llama_65b",
        "stream": false,
        "timeout": 1,
        "temperature": 1.0,
        "top_k": 1,
        "stop_token_ids": [1, 2, 3],
        "top_p": 0.5,
        "ignore_eos": false,
        "repetition_penalty": 1.0,
        "seed": 1,
        "typical_p": 0.5,
        "watermark": false
    })");
    std::string errorMsg;
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    inferInterface->Process();
    inferInterface->request_->useBeamSearch = false;
    RespBodyQueue jsonStrings;
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::GetAvailableOutputCache,
                            bool (*)(std::vector<SingleReqInferInterfaceBase::StreamCache> &))
                     .stubs();
    stubs.will(returnValue(false));
    inferInterface->SendStreamResponse(jsonStrings);
    stubs.will(returnValue(true));
    inferInterface->SendStreamResponse(jsonStrings);
    inferInterface->request_->useBeamSearch = true;
    inferInterface->isEnd = true;
    inferInterface->probesMap[0] = 0;
    inferInterface->probesMap[1] = 1;
    inferInterface->fullTextMap[0] = "zero";
    inferInterface->fullTextMap[1] = "one";
    inferInterface->SendStreamResponse(jsonStrings);
    uint64_t seqId = 100;
    inferInterface->request_->logprobs = true;
    inferInterface->request_->topLogprobs = 1;
    inferInterface->inputParam->streamMode = true;
    inferInterface->inputParam->postSingleText[seqId] = "test";
    inferInterface->postTokenIdMap[1] = {1};
    inferInterface->pickedLogprobMap[1] = {2.34};
    inferInterface->logprobsTokensMap[1] = {5};
    inferInterface->logprobsMap[1] = {-0.1};
    SingleReqInferInterfaceBase::StreamCache testCache = CreateTestStreamCache(seqId);
    inferInterface->streamCache.push_back(testCache);
    inferInterface->reqTokens_ = {101, 102};
    MOCKER_CPP(&SingleReqInferInterfaceBase::DecodeSingleToken,
               bool (*)(std::vector<int64_t> &, std::string &, const uint32_t &, const uint32_t &, const bool &))
        .stubs()
        .will(returnValue(true));
    inferInterface->SendStreamResponse(jsonStrings);
}

TEST_F(VllmOpenAiCompletionsInferTest, BuildReComputeBody) {
    using OrderedJson = nlohmann::ordered_json;

    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->frequencyPenalty = 0.21f;
    request->presencyPenalty = 0.31f;
    request->seed = static_cast<uint64_t>(1234);
    request->temperature = 0.5f;
    request->topP = 0.95f;
    request->topK = 10;
    request->repetitionPenalty = 1.03f;
    request->ignoreEos = true;
    request->includeStopStrInOutput = true;
    request->skipSpecialTokens = true;
    request->stopStrList = {"\n\n", "<|end|>"};
    std::vector<TokenId> stopIds{7, 8};
    request->stopTokenIds = stopIds;

    auto inferParam = std::make_shared<InferParam>();
    inferParam->streamMode = true;
    inferParam->maxNewTokens = 128;

    inferInterface->inputParam = inferParam;
    inferInterface->request_ = request;
    inferInterface->model = "mockModelCompletions";
    inferInterface->isReCompute_ = true;

    std::vector<BestNTokens> tokens;
    std::string ret = inferInterface->BuildReComputeBody(tokens);
    OrderedJson obj = OrderedJson::parse(ret);

    ASSERT_TRUE(obj.contains("model"));
    EXPECT_EQ(obj["model"], "mockModelCompletions");
    ASSERT_TRUE(obj.contains("prompt"));
    EXPECT_TRUE(obj["prompt"].is_string());
    ASSERT_TRUE(obj.contains("frequency_penalty"));
    EXPECT_NEAR(obj["frequency_penalty"].get<double>(), 0.21, 1e-6);
    ASSERT_TRUE(obj.contains("presence_penalty"));
    EXPECT_NEAR(obj["presence_penalty"].get<double>(), 0.31, 1e-6);
    ASSERT_TRUE(obj.contains("seed"));
    EXPECT_EQ(obj["seed"].get<uint64_t>(), 1234ULL);
    ASSERT_TRUE(obj.contains("temperature"));
    EXPECT_NEAR(obj["temperature"].get<double>(), 0.5, 1e-6);
    ASSERT_TRUE(obj.contains("top_p"));
    EXPECT_NEAR(obj["top_p"].get<double>(), 0.95, 1e-6);
    ASSERT_TRUE(obj.contains("top_k"));
    EXPECT_EQ(obj["top_k"].get<int>(), 10);
    ASSERT_TRUE(obj.contains("repetition_penalty"));
    EXPECT_NEAR(obj["repetition_penalty"].get<double>(), 1.03, 1e-6);
    ASSERT_TRUE(obj.contains("ignore_eos"));
    EXPECT_TRUE(obj["ignore_eos"].get<bool>());
    ASSERT_TRUE(obj.contains("include_stop_str_in_output"));
    EXPECT_TRUE(obj["include_stop_str_in_output"].get<bool>());
    ASSERT_TRUE(obj.contains("skip_special_tokens"));
    EXPECT_TRUE(obj["skip_special_tokens"].get<bool>());
    ASSERT_TRUE(obj.contains("stream"));
    EXPECT_TRUE(obj["stream"].get<bool>());
    ASSERT_TRUE(obj.contains("max_tokens"));
    EXPECT_EQ(obj["max_tokens"].get<int>(), 128);
    ASSERT_TRUE(obj.contains("stop"));
    ASSERT_TRUE(obj["stop"].is_array());
    ASSERT_EQ(obj["stop"].size(), 2);
    EXPECT_EQ(obj["stop"][0].get<std::string>(), "\n\n");
    EXPECT_EQ(obj["stop"][1].get<std::string>(), "<|end|>");
    ASSERT_TRUE(obj.contains("stop_token_ids"));
    ASSERT_TRUE(obj["stop_token_ids"].is_array());
    ASSERT_EQ(obj["stop_token_ids"].size(), 2);
    EXPECT_EQ(obj["stop_token_ids"][0].get<long long>(), 7);
    EXPECT_EQ(obj["stop_token_ids"][1].get<long long>(), 8);
}

}  // namespace mindie_llm
