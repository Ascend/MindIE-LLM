/**
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
#include "mockcpp/mockcpp.hpp"
#include "request.h"
#include "basic_types.h"
#include "base64_util.h"
#include <nlohmann/json.hpp>
#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "mock_util.h"
#include "infer_param.h"

using OrderedJson = nlohmann::ordered_json;

MOCKER_CPP_OVERLOAD_EQ(ServerConfig)

namespace mindie_llm {

class InferParamTest : public testing::Test {
protected:
    InferParamTest() = default;
    void SetUp() override
    {
        request = std::make_shared<Request>(RequestIdNew("mockRequest"));
        jsonObj = OrderedJson::object();
        inferParam = std::make_shared<InferParam>();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
    }

    RequestSPtr request;
    InferParamSPtr inferParam;
    std::string error;
    OrderedJson jsonObj;
};

TEST_F(InferParamTest, testAssignDoSample)
{
    // valid values
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    //      doSample should be true given "do_sample" is true
    jsonObj["do_sample"] = true;
    EXPECT_EQ(AssignDoSample(jsonObj, request, error), true);
    EXPECT_EQ(request->doSample.value(), true);
    //      doSample should be false given "do_sample" is false
    jsonObj["do_sample"] = false;
    EXPECT_EQ(AssignDoSample(jsonObj, request, error), true);
    EXPECT_EQ(request->doSample.value(), false);

    // invalid values
    //      should return false given "do_sample" is not a boolean
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["do_sample"] = "something-not-boolean";
    EXPECT_EQ(AssignDoSample(jsonObj, request, error), false);

    // contents missing
    //      doSample should be missing given "do_sample" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_EQ(AssignDoSample(jsonObj, request, error), true);
    EXPECT_EQ(request->doSample.has_value(), false);
}

TEST_F(InferParamTest, testAssignRepetitionPenalty)
{
    // valid values
    //      should return true given "repetition_penalty" is valid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["repetition_penalty"] = 1.0;
    EXPECT_TRUE(AssignRepetitionPenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.0);
    //      corner case 1
    jsonObj["repetition_penalty"] = 1.5;
    EXPECT_TRUE(AssignRepetitionPenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.5);
    //      corner case 2
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["repetition_penalty"] = 0.0;
    EXPECT_FALSE(AssignRepetitionPenalty(jsonObj, request, error));
    EXPECT_FALSE(request->repetitionPenalty.has_value());

    // invalid values
    //      repetitionPenalty should be a default value given "repetition_penalty" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["repetition_penalty"] = -1;
    EXPECT_FALSE(AssignRepetitionPenalty(jsonObj, request, error));
    EXPECT_FALSE(request->repetitionPenalty.has_value());
    
    // contents missing
    //      repetitionPenaly should be a default value given "repetition_penalty" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignRepetitionPenalty(jsonObj, request, error));
    EXPECT_FALSE(request->repetitionPenalty.has_value());
}

TEST_F(InferParamTest, testAssignSeed)
{
    // valid values
    //      should return true given "seed" is valid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["seed"] = static_cast<uint64_t>(1234);
    EXPECT_TRUE(AssignSeed(jsonObj, request, error));
    EXPECT_EQ(request->seed, 1234);
    
    // invalid values
    //      should return false given "seed" is invalid
    jsonObj["seed"] = -1;
    EXPECT_FALSE(AssignSeed(jsonObj, request, error));
    
    // contents missing
    //      should return true given "seed" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignSeed(jsonObj, request, error));
}

TEST_F(InferParamTest, testAssignStopStrings)
{
    // valid values
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    //      should return true given "stop" is an array
    jsonObj["stop"] = {"stop", "end"};
    EXPECT_TRUE(AssignStopStrings(jsonObj, request, error));
    EXPECT_TRUE(request->stopStrings.has_value());
    EXPECT_EQ(request->stopStrings.value(), Base64Util::Encode("[\"stop\",\"end\"]"));
    //      should return true given "stop" is a string
    jsonObj["stop"] = "stop";
    EXPECT_TRUE(AssignStopStrings(jsonObj, request, error));
    EXPECT_TRUE(request->stopStrings.has_value());
    EXPECT_EQ(request->stopStrings.value(), Base64Util::Encode("[\"stop\"]"));

    // invalid values
    //      stopStrings should have no value given "stop" is an empty array
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["stop"] = {};
    EXPECT_TRUE(AssignStopStrings(jsonObj, request, error));
    EXPECT_FALSE(request->stopStrings.has_value());
    //      stopStrings should have no value given "stop" is an empty string
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["stop"] = "";
    EXPECT_FALSE(AssignStopStrings(jsonObj, request, error));
    EXPECT_FALSE(request->stopStrings.has_value());
    //      stopStrings should have no value given "stop" exceeds length limit
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["stop"] = std::string(48 * 1024, 'a'); // 48K > MAX_TOTAL_STOP = 32K
    EXPECT_FALSE(AssignStopStrings(jsonObj, request, error));
    EXPECT_FALSE(request->stopStrings.has_value());
}

TEST_F(InferParamTest, testAssignPresencePenalty)
{
    // valid values
    //      should return true given "presence_penalty" is valid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["presence_penalty"] = 0.8;
    EXPECT_TRUE(AssignPresencePenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->presencyPenalty.value(), 0.8);
    //      corner case 1
    jsonObj["presence_penalty"] = -2.0;
    EXPECT_TRUE(AssignPresencePenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->presencyPenalty.value(), -2.0);
    //      corner case 2
    jsonObj["presence_penalty"] = 2.0;
    EXPECT_TRUE(AssignPresencePenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->presencyPenalty.value(), 2.0);
    
    // invalid values
    //      case 1
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["presence_penalty"] = 2.1;
    EXPECT_FALSE(AssignPresencePenalty(jsonObj, request, error));
    EXPECT_FALSE(request->presencyPenalty.has_value());
    //      case 2
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["presence_penalty"] = -2.1;
    EXPECT_FALSE(AssignPresencePenalty(jsonObj, request, error));
    EXPECT_FALSE(request->presencyPenalty.has_value());
}

TEST_F(InferParamTest, testAssignN)
{
    ServerConfig mockServerConfig;
    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(mockServerConfig));
    // valid values
    //      should return true given "n" is valid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["n"] = static_cast<uint32_t>(2);
    EXPECT_TRUE(AssignN(jsonObj, request, error));
    EXPECT_TRUE(request->n.has_value());
    EXPECT_EQ(request->n.value(), 2);
    //      corner case 1
    jsonObj["n"] = static_cast<uint32_t>(1);
    EXPECT_TRUE(AssignN(jsonObj, request, error));
    EXPECT_TRUE(request->n.has_value());
    EXPECT_EQ(request->n.value(), 1);
    //      corner case 2
    jsonObj["n"] = static_cast<uint32_t>(128);
    EXPECT_TRUE(AssignN(jsonObj, request, error));
    EXPECT_TRUE(request->n.has_value());
    EXPECT_EQ(request->n.value(), 128);

    // invalid values
    //      n should have no value given "n" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["n"] = 0;
    EXPECT_FALSE(AssignN(jsonObj, request, error));
    EXPECT_FALSE(request->n.has_value());

    // contents missing
    //      n should have no value given "n" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignN(jsonObj, request, error));
    EXPECT_FALSE(request->n.has_value());
}

TEST_F(InferParamTest, testAssignBestOf)
{
    ServerConfig mockServerConfig;
    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(mockServerConfig));
    // valid values
    //      should return true given "best_of" is valid
    jsonObj["best_of"] = static_cast<uint32_t>(3);
    EXPECT_TRUE(AssignBestOf(jsonObj, request, error));
    EXPECT_TRUE(request->bestOf.has_value());
    EXPECT_EQ(request->bestOf.value(), 3);
    //      corner case
    jsonObj["best_of"] = static_cast<uint32_t>(1);
    EXPECT_TRUE(AssignBestOf(jsonObj, request, error));
    EXPECT_TRUE(request->bestOf.has_value());
    EXPECT_EQ(request->bestOf.value(), 1);

    // invalid values
    //      bestOf should have no value given "best_of" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["best_of"] = 0;
    EXPECT_FALSE(AssignBestOf(jsonObj, request, error));
    EXPECT_FALSE(request->bestOf.has_value());

    // contents missing
    //      bestOf should have no value given "best_of" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignBestOf(jsonObj, request, error));
    EXPECT_FALSE(request->bestOf.has_value());
}

TEST_F(InferParamTest, testAssignFrequencyPenalty)
{
    // valid values
    //      should return true given "frequency_penalty" is valid
    jsonObj["frequency_penalty"] = 0.5;
    EXPECT_TRUE(AssignFrequencyPenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->frequencyPenalty.value(), 0.5);
    //      corner case
    jsonObj["frequency_penalty"] = 2.0;
    EXPECT_TRUE(AssignFrequencyPenalty(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->frequencyPenalty.value(), 2.0);

    // invalid values
    //      frequencyPenalty should be a default value given "frequency_penalty" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["frequency_penalty"] = 2.1;
    EXPECT_FALSE(AssignFrequencyPenalty(jsonObj, request, error));
    EXPECT_FALSE(request->frequencyPenalty.has_value());

    // contents missing
    //      frequencyPenalty should be a default value given "frequency_penalty" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignFrequencyPenalty(jsonObj, request, error));
    EXPECT_FALSE(request->frequencyPenalty.has_value());
}

TEST_F(InferParamTest, testAssignTemperature)
{
    // valid values
    //      should return true given "temperature" is valid
    jsonObj["temperature"] = 0.7;
    EXPECT_TRUE(AssignTemperature(jsonObj, request, error));
    EXPECT_TRUE(request->temperature.has_value());
    EXPECT_FLOAT_EQ(request->temperature.value(), 0.7);
    //      corner case 1
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["temperature"] = 0.0;
    EXPECT_FALSE(AssignTemperature(jsonObj, request, error));
    EXPECT_FALSE(request->temperature.has_value());
    //      corner case 2
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["temperature"] = 0.0;
    EXPECT_TRUE(AssignTemperature(jsonObj, request, error, true));
    EXPECT_TRUE(request->temperature.has_value());
    EXPECT_FLOAT_EQ(request->temperature.value(), 0.0);

    // invalid values
    //      temperature should have no value given "temperature" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["temperature"] = 2.1;
    EXPECT_TRUE(AssignTemperature(jsonObj, request, error));
    EXPECT_TRUE(request->temperature.has_value());

    // contents missing
    //      temperature should have no value given "temperature" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignTemperature(jsonObj, request, error));
    EXPECT_FALSE(request->temperature.has_value());
}

TEST_F(InferParamTest, testAssignTopK)
{
    // valid values
    //      should return true given "top_k" is valid
    jsonObj["top_k"] = 50;
    EXPECT_TRUE(AssignTopK(jsonObj, request, error));
    EXPECT_EQ(request->topK.value(), 50);
    //      corner case 1
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["top_k"] = 0;
    EXPECT_FALSE(AssignTopK(jsonObj, request, error));
    EXPECT_FALSE(request->topK.has_value());
    //      corner case 2
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["top_k"] = 0;
    EXPECT_TRUE(AssignTopK(jsonObj, request, error, true));
    EXPECT_EQ(request->topK.value(), 0);

    // invalid values
    //      topK should be a default value given "top_k" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["top_k"] = -1;
    EXPECT_FALSE(AssignTopK(jsonObj, request, error));
    EXPECT_FALSE(request->topK.has_value());

    // contents missing
    //      topK should be a default value given "top_k" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignTopK(jsonObj, request, error));
    EXPECT_FALSE(request->topK.has_value());
}

TEST_F(InferParamTest, testAssignTopP)
{
    // valid values
    //      should return true given "top_p" is valid
    jsonObj["top_p"] = 0.9;
    EXPECT_TRUE(AssignTopP(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->topP.value(), 0.9);
    //      corner case
    jsonObj["top_p"] = 1.0;
    EXPECT_TRUE(AssignTopP(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->topP.value(), 1.0);

    // invalid values
    //      topP should be a default value given "top_p" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["top_p"] = 1.1;
    EXPECT_FALSE(AssignTopP(jsonObj, request, error));
    EXPECT_FALSE(request->topP.has_value());

    // contents missing
    //      topP should be a default value given "top_p" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignTopP(jsonObj, request, error));
    EXPECT_FALSE(request->topP.has_value());
}

TEST_F(InferParamTest, testAssignStopTokenIds)
{
    // valid values
    //      should return true given "stop_token_ids" is valid
    jsonObj["stop_token_ids"] = std::vector<int64_t>{1, 2, 3};
    EXPECT_TRUE(AssignStopTokenIds(jsonObj, request, error));
    EXPECT_TRUE(request->stopTokenIds.has_value());
    EXPECT_EQ(request->stopTokenIds.value(), std::vector<int64_t>({1, 2, 3}));

    // invalid values
    //      stopTokenIds should have no value given "stop_token_ids" is invalid
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["stop_token_ids"] = "[1, 2, 3]";
    EXPECT_FALSE(AssignStopTokenIds(jsonObj, request, error));
    EXPECT_FALSE(request->stopTokenIds.has_value());

    // contents missing
    //      stopTokenIds should have no value given "stop_token_ids" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.erase("stop_token_ids");
    EXPECT_TRUE(AssignStopTokenIds(jsonObj, request, error));
    EXPECT_FALSE(request->stopTokenIds.has_value());
}

TEST_F(InferParamTest, testAssignStream)
{
    // valid values
    //      streamMode should be true given "stream" is true
    inferParam = std::make_shared<InferParam>();
    jsonObj["stream"] = true;
    EXPECT_TRUE(AssignStream(jsonObj, inferParam, error));
    EXPECT_TRUE(inferParam->streamMode);
    //      streamMode should be false given "stream" is false
    inferParam = std::make_shared<InferParam>();
    jsonObj["stream"] = false;
    EXPECT_TRUE(AssignStream(jsonObj, inferParam, error));
    EXPECT_FALSE(inferParam->streamMode);

    // contents missing
    //      streamMode should be false given "stream" is missing
    inferParam = std::make_shared<InferParam>();
    jsonObj.erase("stream");
    EXPECT_TRUE(AssignStream(jsonObj, inferParam, error));
    EXPECT_FALSE(inferParam->streamMode);

    // invalid values
    //      streamMode should be false given "stream" is not boolean
    inferParam = std::make_shared<InferParam>();
    jsonObj["stream"] = "something-not-boolean";
    EXPECT_FALSE(AssignStream(jsonObj, inferParam, error));
    EXPECT_FALSE(inferParam->streamMode);
}

TEST_F(InferParamTest, testCheckMultimodalUrlFromJson)
{
    // single url
    jsonObj = OrderedJson::array({{"image_url", "http://example.com/image.jpg"}});
    EXPECT_TRUE(CheckMultimodalUrlFromJson(jsonObj, error));
    EXPECT_TRUE(error.empty());
    jsonObj = OrderedJson::array({{"video_url", "http://example.com/video.mp4"}});
    EXPECT_TRUE(CheckMultimodalUrlFromJson(jsonObj, error));
    EXPECT_TRUE(error.empty());
    jsonObj = OrderedJson::array({{"audio_url", "http://example.com/audio.mp3"}});
    EXPECT_TRUE(CheckMultimodalUrlFromJson(jsonObj, error));
    EXPECT_TRUE(error.empty());

    // multiple urls
    jsonObj = OrderedJson::array({
        {"image_url", "http://example.com/image1.jpg"},
        {"video_url", "http://example.com/video1.mp4"},
        {"audio_url", "http://example.com/audio1.mp3"}
    });
    EXPECT_TRUE(CheckMultimodalUrlFromJson(jsonObj, error));
    EXPECT_TRUE(error.empty());

    // contents missing
    jsonObj.clear();
    EXPECT_TRUE(CheckMultimodalUrlFromJson(jsonObj, error));
    EXPECT_TRUE(error.empty());
}

TEST_F(InferParamTest, testDoSample)
{
    // valid values
    jsonObj["do_sample"] = true;
    EXPECT_TRUE(AssignDoSample(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->doSample.value(), true);

    // invalid values
    jsonObj["do_sample"] = "true";
    EXPECT_FALSE(AssignDoSample(jsonObj, request, error));
}

TEST_F(InferParamTest, testTypicalP)
{
    // valid values
    jsonObj["typical_p"] = 0.9;
    EXPECT_TRUE(AssignTypicalP(jsonObj, request, error));
    EXPECT_FLOAT_EQ(request->typicalP.value(), 0.9);

    // invalid values
    jsonObj["typical_p"] = 1.1;
    EXPECT_FALSE(AssignTypicalP(jsonObj, request, error));
}

TEST_F(InferParamTest, testAssignMaxNewTokens)
{
    // valid values
    //      should return true given "max_new_tokens" is valid
    inferParam = std::make_shared<InferParam>();
    jsonObj["max_new_tokens"] = 100;
    EXPECT_TRUE(AssignMaxNewTokens(jsonObj, inferParam, error));
    EXPECT_EQ(inferParam->maxNewTokens, 100);

    // invalid values
    //      maxNewTokens should be a default value given "max_new_tokens" is invalid
    inferParam = std::make_shared<InferParam>();
    jsonObj["max_new_tokens"] = -1;
    EXPECT_FALSE(AssignMaxNewTokens(jsonObj, inferParam, error));
    EXPECT_EQ(inferParam->maxNewTokens, MAX_NEW_TOKENS_DFT);
}

TEST_F(InferParamTest, testAssignResponseFormat)
{
    // contents missing
    //      should return true given "response_format" is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj.clear();
    EXPECT_TRUE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());

    // null value
    //      should return true given "response_format" is null
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = nullptr;
    EXPECT_TRUE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());

    // valid values - type "json_object"
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", "json_object"}};
    EXPECT_TRUE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_TRUE(request->responseFormat.has_value());
    EXPECT_EQ(request->responseFormat.value(), R"({"type":"json_object"})");

    // valid values - type "json_schema" with valid schema and name
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", "test_schema"}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_TRUE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_TRUE(request->responseFormat.has_value());

    // valid values - type "json_schema" with name containing hyphen and underscore
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", "my-test_schema_123"}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_TRUE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_TRUE(request->responseFormat.has_value());

    // invalid values - response_format is not an object
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = "not_an_object";
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format must be an object");

    // invalid values - response_format is an array
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = OrderedJson::array({"text"});
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());

    // invalid values - type field is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"other_field", "value"}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format must contain 'type' field");

    // invalid values - type field is null
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", nullptr}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format must contain 'type' field");

    // invalid values - type field is not a string
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", 123}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.type must be a string");

    // invalid values - type field has invalid value
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", "invalid_type"}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.type must be 'json_object', or 'json_schema', got 'invalid_type'");

    // invalid values - type "text" is not supported
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", "text"}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.type must be 'json_object', or 'json_schema', got 'text'");

    // invalid values - json_schema type without json_schema field
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", "json_schema"}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema is required when type is 'json_schema'");

    // invalid values - json_schema type with null json_schema field
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", "json_schema"}, {"json_schema", nullptr}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema is required when type is 'json_schema'");

    // invalid values - json_schema type with non-object json_schema field
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {{"type", "json_schema"}, {"json_schema", "not_an_object"}};
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema must be an object");

    // invalid values - json_schema.name is missing
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name is required");

    // invalid values - json_schema.name is null
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", nullptr}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name is required");

    // invalid values - json_schema.name is not a string
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", 123}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name must be a string");

    // invalid values - json_schema.name is empty
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", ""}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name must be 1-64 characters");

    // invalid values - json_schema.name exceeds 64 characters
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", std::string(65, 'a')}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name must be 1-64 characters");

    // valid values - json_schema.name with exactly 64 characters
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", std::string(64, 'a')}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_TRUE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_TRUE(request->responseFormat.has_value());

    // invalid values - json_schema.name contains invalid character (space)
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", "invalid name"}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name must contain only 0-9, a-z, A-Z, -, _");

    // invalid values - json_schema.name contains invalid character (dot)
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", "invalid.name"}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name must contain only 0-9, a-z, A-Z, -, _");

    // invalid values - json_schema.name contains invalid character (special char)
    request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    jsonObj["response_format"] = {
        {"type", "json_schema"},
        {"json_schema", {{"name", "invalid@name"}, {"schema", {{"type", "object"}}}}}
    };
    EXPECT_FALSE(AssignResponseFormat(jsonObj, request, error));
    EXPECT_FALSE(request->responseFormat.has_value());
    EXPECT_EQ(error, "Parameter response_format.json_schema.name must contain only 0-9, a-z, A-Z, -, _");
}

} // namespace mindie_llm