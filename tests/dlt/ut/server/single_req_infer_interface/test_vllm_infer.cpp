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

#include <gtest/gtest.h>

#include "base64_util.h"
#include "config_manager/config_manager_impl.h"
#include "mock_util.h"
#include "mockcpp/mockcpp.hpp"
#include "request.h"
#include "response.h"
#include "single_llm_pnd_req_handler.h"
#include "single_req_vllm_infer_interface.h"

using namespace mindie_llm;
using OrderedJson = nlohmann::ordered_json;

namespace mindie_llm {
MOCKER_CPP_OVERLOAD_EQ(ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)

class VllmInferTest : public testing::Test {
   protected:
    VllmInferTest() = default;
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
        vllmInferInterface = std::make_shared<SingleReqVllmInferInterface>(pndReqHandler, false);
        vllmInferInterface->inputParam = std::make_shared<InferParam>();
        vllmInferInterface->request_ = std::make_shared<Request>();
    }
    void TearDown() { GlobalMockObject::verify(); }

    httplib::Request httpRequest;
    httplib::Response httpResponse;
    RequestSPtr request;
    ResponseSPtr response;
    std::shared_ptr<RequestContext> requestContext;
    std::shared_ptr<SingleLLMPnDReqHandler> pndReqHandler;
    std::shared_ptr<SingleReqVllmInferInterface> vllmInferInterface;
    ScheduleConfig mockScheduleConfig_;
    ServerConfig mockServerConfig_;
};

TEST_F(VllmInferTest, testProcess) {
    MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool *, const std::string &,
                                                        std::vector<int64_t> &, HeadFlag, uint64_t &, const bool))
        .stubs()
        .will(invoke(&MockTokenizerEncodeSuccess));

    // HTTP request
    httpRequest.method = "mockMethod";
    httpRequest.path = "mockPath";
    httpRequest.version = "mockVersion";
    httpRequest.body = R"({
        "prompt": "mock test 1", "stream": false, "repetition_penalty": 1.0,
        "top_p": 1.0, "top_k": 10, "max_tokens": 16, "temperature": 1.0})";
    requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
    pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    vllmInferInterface = std::make_shared<SingleReqVllmInferInterface>(pndReqHandler, false);
    // InferInterface::Process: HTTP request => JSON format => MindIE Request
    EXPECT_NO_THROW(vllmInferInterface->Process());
    EXPECT_EQ(vllmInferInterface->request_->input_ids, std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(vllmInferInterface->inputParam->streamMode, false);
    EXPECT_NEAR(vllmInferInterface->request_->repetitionPenalty.value(), 1.0, 1e-6);
    EXPECT_NEAR(vllmInferInterface->request_->topP.value(), 1.0, 1e-6);
    EXPECT_EQ(vllmInferInterface->request_->topK.value(), 10);
    EXPECT_EQ(vllmInferInterface->request_->maxOutputLen, 16);
    EXPECT_EQ(vllmInferInterface->request_->temperature.has_value(), true);
    EXPECT_NEAR(vllmInferInterface->request_->temperature.value(), 1.0, 1e-6);

    // HTTP request
    httpRequest.body = R"({
        "prompt": "mock test 2", "stream": true,
        "seed": 1234, "stop": ["test"], "timeout": 1, "presence_penalty": 0,
        "frequency_penalty": 0, "length_penalty": 1.0, "skip_special_tokens": false,
        "stop_token_ids": [999], "ignore_eos": false
    })";
    requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
    pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    vllmInferInterface = std::make_shared<SingleReqVllmInferInterface>(pndReqHandler, false);
    // InferInterface::Process: HTTP request => JSON format => MindIE Request
    EXPECT_NO_THROW(vllmInferInterface->Process());
    EXPECT_EQ(vllmInferInterface->request_->input_ids, std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(vllmInferInterface->inputParam->streamMode, true);
    EXPECT_EQ(vllmInferInterface->request_->seed, 1234);
    EXPECT_EQ(vllmInferInterface->request_->stopStrings, Base64Util::Encode("[\"test\"]"));
    EXPECT_EQ(vllmInferInterface->inputParam->timeout, 1);
    EXPECT_EQ(vllmInferInterface->request_->presencyPenalty.value(), 0);
    EXPECT_EQ(vllmInferInterface->request_->frequencyPenalty.value(), 0);
    EXPECT_EQ(vllmInferInterface->request_->skipSpecialTokens.value(), false);
    EXPECT_TRUE(vllmInferInterface->request_->stopTokenIds.has_value());
    EXPECT_EQ(vllmInferInterface->request_->stopTokenIds.value(), std::vector<int64_t>({999}));
    EXPECT_EQ(vllmInferInterface->request_->ignoreEos.value(), false);

    // HTTP request
    httpRequest.body = R"({"prompt": [{"image_url": "mock"}, {}]})";
    requestContext = std::make_shared<RequestContext>(httpRequest, httpResponse);
    pndReqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    vllmInferInterface = std::make_shared<SingleReqVllmInferInterface>(pndReqHandler, false);
    // InferInterface::Process: HTTP request => JSON format => MindIE Request
    EXPECT_NO_THROW(vllmInferInterface->Process());
    EXPECT_EQ(vllmInferInterface->request_->input_ids, std::vector<int64_t>({1, 2, 3}));
}

TEST_F(VllmInferTest, testSetupInferParams) {
    // valid values
    vllmInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "prompt": "Hi?",
        "max_tokens": 100,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.03,
        "frequency_penalty": 1.0,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "seed": 1234,
        "stop": [
            "stop1",
            "stop2"
        ],
        "stop_token_ids": [
            2,
            13
        ],
        "model": "llama_65b",
        "include_stop_str_in_output": false,
        "skip_special_tokens": true,
        "ignore_eos": false,
        "best_of": 3,
        "n": 2
    })");

    // SetupInferParams should return true given valid reqJsonBody
    std::string errorMsg;
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), true);
    vllmInferInterface->Process();

    // validate request's items with reqJsonBody
    // input_ids, maxOutputLen are not validated since they are assigned outside SetupInferParams
    EXPECT_NEAR(request->repetitionPenalty.value(), 1.0, 1e-6);
    EXPECT_NEAR(request->presencyPenalty.value(), 1.03, 1e-6);
    EXPECT_NEAR(request->frequencyPenalty.value(), 1.0, 1e-6);
    ASSERT_TRUE(request->temperature.has_value());
    EXPECT_NEAR(request->temperature.value(), 0.5, 1e-6);
    EXPECT_NEAR(request->topP.value(), 0.95, 1e-6);
    EXPECT_EQ(request->topK.value(), 40);
    EXPECT_EQ(request->seed, 1234);
    EXPECT_TRUE(request->stopStrings.has_value());
    EXPECT_EQ(*request->stopStrings, Base64Util::Encode("[\"stop1\",\"stop2\"]"));
    EXPECT_TRUE(request->stopTokenIds.has_value());
    EXPECT_EQ(*request->stopTokenIds, std::vector<long>({2, 13}));
    EXPECT_TRUE(request->includeStopStrInOutput.has_value());
    EXPECT_FALSE(*request->includeStopStrInOutput);
    EXPECT_TRUE(request->skipSpecialTokens.value());
    EXPECT_FALSE(request->ignoreEos.value());
    EXPECT_TRUE(request->bestOf.has_value());
    EXPECT_EQ(*request->bestOf, 3U);
    EXPECT_TRUE(request->n.has_value());
    EXPECT_EQ(*request->n, 2U);

    // invalid values
    //      should return false given repetition_penalty exceeds limits
    vllmInferInterface->reqJsonBody_["repetition_penalty"] = -1;
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), false);
    //      should return false given model is not string
    vllmInferInterface->reqJsonBody_["repetition_penalty"] = 1.0;
    vllmInferInterface->reqJsonBody_["model"] = 1;
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(VllmInferTest, testSetReturnSeqCount) {
    std::string errorMsg;
    // valid values
    vllmInferInterface->reqJsonBody_ = OrderedJson::parse(R"({})");
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), true);
    // invalid values
    //      should return false when temperature not set and best_of is greater than 1
    vllmInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "best_of": 3
    })");
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), false);
    vllmInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "best_of": 3,
        "n": 2
    })");
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), false);
    //      should return false when best_of less than n
    vllmInferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "best_of": 2,
        "n": 3
    })");
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(VllmInferTest, testBuildResponseJson) {
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    std::string errorMsg;
    EXPECT_TRUE(vllmInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    vllmInferInterface->inputParam->textInput = R"({
        "inputs": "[{\"type\": \"text\", \"text\": \"mockText\"}]"
    })";
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), true);
    EXPECT_TRUE(vllmInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // Failed to process vLLM response single, return false
    MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseSingle, bool (*)(ResponseSPtr, const uint64_t &))
        .stubs()
        .will(returnValue(false));
    EXPECT_FALSE(vllmInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    vllmInferInterface->reqJsonBody_ = OrderedJson::parse(R"({"stream": true})");
    EXPECT_EQ(vllmInferInterface->SetupInferParams(request, errorMsg), true);
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::ProcessResponseStream,
                            bool (*)(ResponseSPtr, const std::vector<BestNTokens> &, RespBodyQueue &, const uint64_t &))
                     .stubs();
    // succeed to process vLLM response stream, return true
    stubs.will(returnValue(true));
    EXPECT_TRUE(vllmInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
    // Failed to process vLLM response stream, return false
    stubs.will(returnValue(false));
    EXPECT_FALSE(vllmInferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(VllmInferTest, testSendStreamResponse) {
    RespBodyQueue jsonStrings;
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::GetAvailableOutputCache,
                            bool (*)(std::vector<SingleReqInferInterfaceBase::StreamCache> &))
                     .stubs();
    stubs.will(returnValue(false));
    vllmInferInterface->SendStreamResponse(jsonStrings);
    EXPECT_TRUE(jsonStrings.empty());
}

// NOTE: the following unit tests come from the code repo before refactor

TEST_F(VllmInferTest, ValidateAndPrepareReqToken) {
    OrderedJson body;
    std::string msg;
    uint64_t timestamp;

    OrderedJson mockMultiModelUrlArr = {};
    for (size_t i = 0; i < MAX_MULTIMODAL_URL_NUM + 1; ++i) {
        mockMultiModelUrlArr.push_back(
            {{"image_url", "mockImageUrl"}, {"audio_url", "mockAudioUrl"}, {"video_url", "mockVideoUrl"}});
    }
    auto testPromptArr = OrderedJson::array(
        {nullptr, 1, "\xFF", "", OrderedJson::array(), OrderedJson::array({1, 2}),
         OrderedJson::array({{{"image_url", nullptr}}}),
         OrderedJson::array({{{"image_url", "mockImageUrl"}, {"audio_url", nullptr}}}),
         OrderedJson::array({{{"image_url", "mockImageUrl"}, {"audio_url", "mockAudioUrl"}}}),
         OrderedJson::array({{{"image_url", "mockImageUrl"}, {"audio_url", "mockAudioUrl"}, {"video_url", nullptr}}}),
         OrderedJson::array(
             {{{"image_url", "mockImageUrl"}, {"audio_url", "mockAudioUrl"}, {"video_url", "mockVideoUrl"}}}),
         mockMultiModelUrlArr, "hello"});

    auto stubs = MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool *, const std::string &,
                                                                     std::vector<int64_t> &, HeadFlag, uint64_t &))
                     .stubs();

    stubs.will(invoke(&MockTokenizerEncodeFail));
    for (size_t i = 0; i < testPromptArr.size(); ++i) {
        body["prompt"] = testPromptArr[i];
        EXPECT_FALSE(vllmInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    }

    stubs.will(invoke(&MockTokenizerResultsEmpty));
    EXPECT_FALSE(vllmInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp));

    stubs.will(invoke(&MockTokenizerEncodeSuccess));
    EXPECT_TRUE(vllmInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    // case when recompute is true
    vllmInferInterface->isReCompute_ = true;
    EXPECT_FALSE(vllmInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
    MOCKER_CPP(&SingleReqInferInterfaceBase::GetTokensFromInput,
               bool (*)(const std::string &, std::vector<std::int64_t> &, std::vector<std::int64_t> &, std::string &))
        .stubs()
        .will(returnValue(true));
    EXPECT_FALSE(vllmInferInterface->ValidateAndPrepareReqToken(body, msg, timestamp));
}

TEST_F(VllmInferTest, BuildVllmReComputeBody) {
    auto request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->repetitionPenalty = 1.03f;
    request->presencyPenalty = 0.10f;
    request->frequencyPenalty = 0.20f;
    request->temperature = 0.5f;
    request->topK = 10;
    request->topP = 0.95f;
    request->seed = static_cast<uint64_t>(1234);
    request->stopStrings = std::string(R"(["<|end|>", "</s>"])");
    request->stopTokenIds = std::vector<TokenId>{7, 8};
    request->skipSpecialTokens = true;
    request->loraId = "mockLoraId";

    auto inferParam = std::make_shared<InferParam>();
    inferParam->textInput = "mock origin text";
    inferParam->streamMode = true;
    inferParam->maxNewTokens = 128;

    vllmInferInterface->request_ = request;
    vllmInferInterface->inputParam = inferParam;
    vllmInferInterface->isReCompute_ = true;

    std::vector<BestNTokens> tokens;
    std::string ret = vllmInferInterface->BuildVllmReComputeBody(tokens);
    OrderedJson j = OrderedJson::parse(ret);

    ASSERT_TRUE(j.contains("origin_inputs"));
    EXPECT_EQ(j["origin_inputs"], "mock origin text");
    ASSERT_TRUE(j.contains("stream"));
    EXPECT_TRUE(j["stream"].get<bool>());
    ASSERT_TRUE(j.contains("max_tokens"));
    EXPECT_EQ(j["max_tokens"], 128);
    ASSERT_TRUE(j.contains("repetition_penalty"));
    EXPECT_NEAR(j["repetition_penalty"].get<double>(), 1.03, 1e-6);
    ASSERT_TRUE(j.contains("presence_penalty"));
    EXPECT_NEAR(j["presence_penalty"].get<double>(), 0.10, 1e-6);
    ASSERT_TRUE(j.contains("frequency_penalty"));
    EXPECT_NEAR(j["frequency_penalty"].get<double>(), 0.20, 1e-6);
    ASSERT_TRUE(j.contains("temperature"));
    EXPECT_NEAR(j["temperature"].get<double>(), 0.5, 1e-6);
    ASSERT_TRUE(j.contains("top_k"));
    EXPECT_EQ(j["top_k"], 10);
    ASSERT_TRUE(j.contains("top_p"));
    EXPECT_NEAR(j["top_p"].get<double>(), 0.95, 1e-6);
    ASSERT_TRUE(j.contains("seed"));
    EXPECT_EQ(j["seed"], 1234);
    ASSERT_TRUE(j.contains("skip_special_tokens"));
    EXPECT_TRUE(j["skip_special_tokens"].get<bool>());
    ASSERT_TRUE(j.contains("stop"));
    ASSERT_TRUE(j["stop"].is_array());
    ASSERT_EQ(j["stop"].size(), 2);
    EXPECT_EQ(j["stop"][0], "<|end|>");
    EXPECT_EQ(j["stop"][1], "</s>");
    ASSERT_TRUE(j.contains("stop_token_ids"));
    std::vector<int64_t> expectedStopIds{7, 8};
    EXPECT_EQ(j["stop_token_ids"].get<std::vector<int64_t>>(), expectedStopIds);
    ASSERT_TRUE(j.contains("model"));
    EXPECT_EQ(j["model"], "mockLoraId");
    ASSERT_TRUE(j.contains("prompt"));
    EXPECT_TRUE(j["prompt"].is_string());
}

}  // namespace mindie_llm
