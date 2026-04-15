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
#include <gtest/gtest.h>

#include "base_config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "mock_util.h"
#include "mockcpp/mockcpp.hpp"
#include "request.h"
#include "response.h"
#include "single_llm_pnd_req_handler.h"
#include "single_req_vllm_openai_infer_interface.h"

using namespace mindie_llm;

namespace mindie_llm {
MOCKER_CPP_OVERLOAD_EQ(ScheduleConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
class VllmOpenAiInferTest : public testing::Test {
   protected:
    VllmOpenAiInferTest() = default;
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
        inferInterface = std::make_shared<SingleReqVllmOpenAiInferInterface>(pndReqHandler, false);
        inferInterface->inputParam = std::make_shared<InferParam>();
        inferInterface->request_ = std::make_shared<Request>();
    }

    void TearDown() { GlobalMockObject::verify(); }
    httplib::Request httpRequest;
    httplib::Response httpResponse;
    RequestSPtr request;
    ResponseSPtr response;
    std::string error;
    std::shared_ptr<RequestContext> requestContext;
    std::shared_ptr<SingleLLMPnDReqHandler> pndReqHandler;
    std::shared_ptr<SingleReqVllmOpenAiInferInterface> inferInterface;
    ScheduleConfig mockScheduleConfig_;
    ServerConfig mockServerConfig_;
};

TEST_F(VllmOpenAiInferTest, TestSetupInferParams) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    std::string errorMsg;
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "model": "llama_65b",
        "stream": false,
        "timeout": 123,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 0.5,
        "repetition_penalty": 1.0,
        "seed": 1,
        "typical_p": 0.5,
        "watermark": false
    })");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), true);
    EXPECT_TRUE(request->temperature.has_value());
    EXPECT_NEAR(request->temperature.value(), 1.0, 1e-6);
    EXPECT_EQ(request->topK.value(), 1);
    EXPECT_NEAR(request->topP.value(), 0.5, 1e-6);
    EXPECT_EQ(request->seed, 1);
    EXPECT_NEAR(request->repetitionPenalty.value(), 1.0, 1e-6);
    EXPECT_EQ(inferInterface->inputParam->streamMode, false);
    EXPECT_EQ(inferInterface->inputParam->timeout, 1);
    // invalid values
    //      should return false given "ignore_eos" is not bool
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({"ignore_eos": "a"})");
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
    //      should return when SetReturnSeqCount return false
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "model": "llama_65b",
        "ignore_eos": false,
        "best_of": 3,
        "n": 2
    })");
    MOCKER_CPP(&SingleReqVllmOpenAiInferInterface::SetReturnSeqCount, bool (*)(ResponseSPtr, std::string &))
        .stubs()
        .will(returnValue(false));
    EXPECT_EQ(inferInterface->SetupInferParams(request, errorMsg), false);
}

TEST_F(VllmOpenAiInferTest, TestSetupInferParamsWithResponseFormat) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(&ConfigManager::GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    std::string errorMsg;

    // valid response_format with json_object
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "model": "llama_65b",
        "response_format": {"type": "json_object"}
    })");
    EXPECT_TRUE(inferInterface->SetupInferParams(request, errorMsg));
    EXPECT_TRUE(request->responseFormat.has_value());
    EXPECT_EQ(request->responseFormat.value(), R"({"type":"json_object"})");

    // valid response_format with json_schema
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "model": "llama_65b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "my_schema",
                "schema": {"type": "object"}
            }
        }
    })");
    EXPECT_TRUE(inferInterface->SetupInferParams(request, errorMsg));
    EXPECT_TRUE(request->responseFormat.has_value());

    // invalid response_format - type is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
        "model": "llama_65b",
        "response_format": {"type": "invalid_type"}
    })");
    EXPECT_FALSE(inferInterface->SetupInferParams(request, errorMsg));

    // invalid response_format - json_schema.name is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->reqJsonBody_ = OrderedJson::parse(R"({
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
        "model": "llama_65b",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "invalid name",
                "schema": {"type": "object"}
            }
        }
    })");
    EXPECT_TRUE(inferInterface->SetupInferParams(request, errorMsg));
}

TEST_F(VllmOpenAiInferTest, TestParseModelName) {
    OrderedJson body;
    std::string errorMsg;
    std::string model;
    // valid values
    //      should return true given "model" is found in ModelDeployConfig
    std::vector<ModelDeployConfig> mockConfigs;
    ModelDeployConfig mockConfig;
    mockConfig.modelName = "mockName";
    mockConfigs.push_back(mockConfig);
    auto stubsGetParam =
        MOCKER_CPP(&ModelDeployConfigManager::GetParam, std::vector<ModelDeployConfig> & (*)()).stubs();
    stubsGetParam.will(returnValue(mockConfigs));
    body["model"] = "mockName";
    EXPECT_EQ(inferInterface->ParseModelName(body, model, errorMsg), true);
    EXPECT_EQ(model, "mockName");
    //      should return true given "model" is found in LoraConfig
    std::vector<LoraParamSPtr> mockLoraConfigs;
    LoraParamSPtr mockLoraConfig = std::make_shared<LoraParam>("mockLoraName", "mockLoraPath", "mockMasterModel");
    mockLoraConfigs.push_back(mockLoraConfig);
    inferInterface->loraConfigs_ = mockLoraConfigs;
    body["model"] = "mockLoraName";
    EXPECT_EQ(inferInterface->ParseModelName(body, model, errorMsg), true);
    EXPECT_EQ(model, "mockLoraName");
    // invalid values
    //      should return false given "model" is null
    body.clear();
    EXPECT_EQ(inferInterface->ParseModelName(body, model, errorMsg), false);
    //      should return false given "model" is empty
    body["model"] = "";
    EXPECT_EQ(inferInterface->ParseModelName(body, model, errorMsg), false);
    //      should return false given "model" is not found in ModelDeployConfig or LoraConfig
    body["model"] = "otherMockName";
    EXPECT_EQ(inferInterface->ParseModelName(body, model, errorMsg), false);
    //      should return false given "model" is nou string
    body["model"] = 123;
    EXPECT_EQ(inferInterface->ParseModelName(body, model, errorMsg), false);
}

}  // namespace mindie_llm
