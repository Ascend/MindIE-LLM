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
#include "mockcpp/mockcpp.hpp"
#include "single_req_infer_interface_base.h"
#include "single_req_openai_infer_interface.h"
#include "single_llm_req_handler_base.h"
#include "request.h"
#include "response.h"
#include "grpc_context.h"
#include "single_llm_decode_req_handler.h"
#include "single_req_tgi_text_infer_interface.h"
#include "custom_exception.h"
#include "prefillAndDecodeCommunication.grpc.pb.h"
#include "config_manager.h"
#include "mock_util.h"
#include "single_llm_pnd_req_handler.h"
#include "base_config_manager.h"

MOCKER_CPP_OVERLOAD_EQ(mindie_llm::ServerConfig);
MOCKER_CPP_OVERLOAD_EQ(mindie_llm::Error);

using namespace mindie_llm;
using OrderedJson = nlohmann::ordered_json;

long long MockStoll(const std::string& str, size_t* pos = nullptr, int base = 10)
{
    throw CustomException("A custom error occurred.");
}

namespace mindie_llm {
class InferInterfaceBaseTest : public testing::Test {
protected:
    InferInterfaceBaseTest() = default;
    void SetUp()
    {
        // mock ConfigManager
        MOCKER_CPP(&CanonicalPath, bool (*)(std::string &)).stubs().will(returnValue(true));
        MOCKER_CPP(&GetConfigPath, Error (*)(std::string &)).stubs().will(returnValue(Error(Error::Code::OK)));
        MOCKER_CPP(&ServerConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&BackendConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&ScheduleConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&ModelDeployConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&LogConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&ConfigManager::CheckAllParam, bool (*)()).stubs().will(returnValue(true));
        EXPECT_EQ(ConfigManager::CreateInstance("mockPath"), true);
        
        requestContext = std::make_shared<RequestContext>(req, resp);
        reqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
        inferInterface = std::make_shared<SingleReqOpenAiInferInterface>(reqHandler, isReCompute);
        inferInterface->inputParam = std::make_shared<InferParam>();
        inferInterface->request_ = std::make_shared<Request>();
        testSeqId = 100;
    }
    void TearDown()
    {
        GlobalMockObject::verify();
    }

    SingleReqInferInterfaceBase::StreamCache CreateStreamCache(uint64_t seqId, const std::u16string &u16Text,
                                                               InferStatusType eosStatus, bool initCanOutput = false)
    {
        SingleReqInferInterfaceBase::StreamCache cache;
        cache.canOutput[seqId] = initCanOutput;
        cache.u16TokenText[seqId] = u16Text;
        cache.eosMap[seqId] = eosStatus;
        return cache;
    }

    httplib::Request req;
    httplib::Response resp;
    std::shared_ptr<RequestContext> requestContext;
    bool isReCompute = false;
    std::string errorMsg;
    OrderedJson body;
    std::shared_ptr<SingleLLMPnDReqHandler> reqHandler;
    std::shared_ptr<SingleReqOpenAiInferInterface> inferInterface;
    const InferStatusType EOS_CONTINUE = InferStatusType::ITERATION_CONTINUE;
    const InferStatusType EOS_ABORT = InferStatusType::ABORT;
    uint64_t testSeqId;
};

TEST_F(InferInterfaceBaseTest, testParseChatTemplate)
{
    // valid values
    //      should return true given chat_template_kwargs is empty
    body.clear();
    EXPECT_EQ(inferInterface->ParseChatTemplate(body, errorMsg), true);
    body["chat_template_kwargs"] = nlohmann::json::object();
    EXPECT_EQ(inferInterface->ParseChatTemplate(body, errorMsg), true);
    //      should return true given enable_thinking is boolean
    body["chat_template_kwargs"]["enable_thinking"] = true;
    EXPECT_EQ(inferInterface->ParseChatTemplate(body, errorMsg), true);
    body["chat_template_kwargs"]["enable_thinking"] = false;
    EXPECT_EQ(inferInterface->ParseChatTemplate(body, errorMsg), true);

    // invalid values
    //      should return false given chat_template_kwargs is not a JSON object
    body["chat_template_kwargs"] = nlohmann::json::array();
    EXPECT_EQ(inferInterface->ParseChatTemplate(body, errorMsg), false);
    //      should return false given enable_thinking is not a boolean
    body["chat_template_kwargs"] = nlohmann::json::object();
    body["chat_template_kwargs"]["enable_thinking"] = 1;
    EXPECT_EQ(inferInterface->ParseChatTemplate(body, errorMsg), false);
}

TEST_F(InferInterfaceBaseTest, testProcess)
{
    // case 1: invalid reqJsonBody
    EXPECT_NO_THROW(inferInterface->Process());
    // case 2: valid reqJsonBody
    req.method = "mockMethod";
    req.path = "mockPath";
    req.version = "mockVersion";
    req.body = R"({
        "prompt": "mock test 1",
        "stream": false,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "top_k": 10,
        "max_tokens": 16,
        "temperature": 1.0
    })";
    requestContext = std::make_shared<RequestContext>(req, resp);
    reqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    inferInterface = std::make_shared<SingleReqOpenAiInferInterface>(reqHandler, false);
    EXPECT_NO_THROW(inferInterface->Process());
    // case 3: failed to generate infer request
    MOCKER_CPP(&SingleReqOpenAiInferInterface::ValidateAndPrepareReqToken,
               bool (*)(OrderedJson &, std::string &, uint64_t &))
        .stubs()
        .will(returnValue(true));
    MOCKER_CPP(&SingleReqInferInterfaceBase::GenerateInferRequest, bool (*)(std::string &))
        .stubs()
        .will(returnValue(false));
    EXPECT_NO_THROW(inferInterface->Process());
}

// testSimulateProcess removed: method moved to HealthChecker::RunHttpTimedHealthCheck
// testSimulateGenerateInferRequest removed: method moved to SimulateRequestExecutor

TEST_F(InferInterfaceBaseTest, testStop)
{
    // case 1: invalid reqJsonBody
    EXPECT_NO_THROW(inferInterface->Stop());
    // case 2: reqBody not contain id
    req.method = "mockMethod";
    req.path = "mockPath";
    req.version = "mockVersion";
    req.body = R"({
        "prompt": "mock test 1"
    })";
    requestContext = std::make_shared<RequestContext>(req, resp);
    reqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    inferInterface = std::make_shared<SingleReqOpenAiInferInterface>(reqHandler, false);
    EXPECT_NO_THROW(inferInterface->Stop());
    // case 2: id length is morn than 256
    std::string longId(MAX_INPUT_ID_LENGTH + 1, 'a');
    req.body = R"({
        "id": ")" + longId + R"(",
        "prompt": "mock test 1"
    })";
    requestContext = std::make_shared<RequestContext>(req, resp);
    reqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    inferInterface = std::make_shared<SingleReqOpenAiInferInterface>(reqHandler, false);
    EXPECT_NO_THROW(inferInterface->Stop());
    // case 3: id length is morn than 256
    req.body = R"({
        "id": "mockId",
        "prompt": "mock test 1"
    })";
    requestContext = std::make_shared<RequestContext>(req, resp);
    reqHandler = std::make_shared<SingleLLMPnDReqHandler>(requestContext);
    inferInterface = std::make_shared<SingleReqOpenAiInferInterface>(reqHandler, false);
    EXPECT_NO_THROW(inferInterface->Stop());
    // case 4: stop success
    auto stubs = MOCKER_CPP(&InferInstance::ControlRequest,
                            Status (*)(const RequestIdNew &, OperationV2)).stubs();
    stubs.will(returnValue(Status(Error::Code::OK, "Success")));
    EXPECT_NO_THROW(inferInterface->Stop());
    // case 5: requestId not found
    stubs.will(returnValue(Status(Error::Code::NOT_FOUND, "RequestId not found")));
    EXPECT_NO_THROW(inferInterface->Stop());
    // case 6: stop status is invalid
    stubs.will(returnValue(Status(Error::Code::INVALID_ARG, "Stop status is invalid")));
    EXPECT_NO_THROW(inferInterface->Stop());
}

TEST_F(InferInterfaceBaseTest, testToolCall)
{
    // add a unit test here
}

TEST_F(InferInterfaceBaseTest, testProcessResponseSingle)
{
    uint64_t timestamp = 1000;
    ResponseSPtr response = std::make_shared<Response>(RequestIdNew("1"));
    
    // ProcessResponseSingle should return true given isEos == false
    response->isEos = false;
    EXPECT_EQ(inferInterface->ProcessResponseSingle(response, timestamp), true);

    response->isEos = true;
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::PostProcess, bool(*)(SingleReqInferInterfaceBase*, const std::vector<int64_t> &,
        std::string &, const uint64_t &, bool, uint32_t, uint32_t, const uint64_t &)).stubs();
    
    // ProcessResponseSingle should return false given PostProcess == false
    stubs.will(returnValue(false));
    std::vector<int64_t> respToken = {100, 101, 102, 103, 104};
    for (size_t i = 0; i < respToken.size(); i++) {
        inferInterface->respTokenMap[0].push_back(respToken[i]);
        EXPECT_EQ(inferInterface->ProcessResponseSingle(response, timestamp), false);
    }

    // ProcessResponseSingle should return true given valid fullText, like versa
    stubs.will(returnValue(true));
    inferInterface->fullTextMap[0] = "\xFF \xFF \xFF \xFF \xFF";
    EXPECT_EQ(inferInterface->ProcessResponseSingle(response, timestamp), false);
    inferInterface->fullTextMap[0] = "zero one two three four";
    EXPECT_EQ(inferInterface->ProcessResponseSingle(response, timestamp), true);

    // ProcessResponseSingle should return true given valid truncationIndex
    stubs.will(returnValue(true));
    respToken = {105, 106, 107, 108, 109};
    for (size_t i = 0; i < respToken.size(); i++) {
        inferInterface->truncationIdMap[0] = 5 + i;
        inferInterface->respTokenMap[0].push_back(respToken[i]);
        EXPECT_EQ(inferInterface->ProcessResponseSingle(response, timestamp), true);
    }
}

TEST_F(InferInterfaceBaseTest, testGetTokensFromInput)
{
    std::string input;
    std::vector<std::int64_t> requestTokens;
    std::vector<std::int64_t> responseTokens;
    
    // valid values
    //      should return true when given valid input (tokens)
    //      the first item (5) stands for the length of requestTokens
    errorMsg = "";
    input = "5,0,1,2,3,4,5,6,7,8,9";
    EXPECT_EQ(inferInterface->GetTokensFromInput(input, requestTokens, responseTokens, errorMsg), true);
    EXPECT_EQ(requestTokens, std::vector<std::int64_t>({0, 1, 2, 3, 4}));
    EXPECT_EQ(responseTokens, std::vector<std::int64_t>({5, 6, 7, 8, 9}));
    EXPECT_EQ(errorMsg, "");
    // invalid values
    //      should return false when input is not made of tokens but characters
    input = "apple,banana,cherry,watermelon";
    EXPECT_EQ(inferInterface->GetTokensFromInput(input, requestTokens, responseTokens, errorMsg), false);
    EXPECT_EQ(errorMsg, "stoll");
    
    //      should return false when stoll throws an exception
    errorMsg = ""; // reset
    long long (*ptrStoll)(const std::string&, size_t*, int) = &std::stoll;
    MOCKER_CPP(ptrStoll, long long (*)(const std::string&, size_t*, int)).stubs().will(invoke(MockStoll));
    EXPECT_EQ(inferInterface->GetTokensFromInput(input, requestTokens, responseTokens, errorMsg), false);
    EXPECT_EQ(errorMsg, "Unknown error occur when get token from input");
}

TEST_F(InferInterfaceBaseTest, testInsertPerfInfoIntoJson)
{
    Metrics testMetrics;
    testMetrics.batchSize = {100, 100, 200, 250, 300};
    testMetrics.queueWaitTime = {10, 10, 20, 25, 30};
    reqHandler->metrics = testMetrics;
    OrderedJson tmpJsonObj = OrderedJson::object();
    // InsertPerfInfoIntoJson should return true given valid metrics and json body
    auto status = inferInterface->InsertPerfInfoIntoJson(tmpJsonObj,
        {PerfInfoType::PERF_BATCH_SZIE, PerfInfoType::PERF_QUEUE_WAIT_TIME},
        {"batch_size", "queue_wait_time"});
    EXPECT_TRUE(status.IsOk());
    //  should return false given invalid performance index
    status = inferInterface->InsertPerfInfoIntoJson(tmpJsonObj,
        {PerfInfoType::PERF_BATCH_SZIE, PerfInfoType::PERF_QUEUE_WAIT_TIME},
        {"batch_size"});
    EXPECT_FALSE(status.IsOk());
    //  should return true given body is array
    OrderedJson tmpJsonArr = OrderedJson::array();
    status = inferInterface->InsertPerfInfoIntoJson(tmpJsonArr,
        {PerfInfoType::PERF_BATCH_SZIE, PerfInfoType::PERF_QUEUE_WAIT_TIME},
        {"batch_size", "queue_wait_time"}
    );
    EXPECT_TRUE(status.IsOk());
    //  should return false given invalid json type
    OrderedJson tmpJsonStr = "not_object_or_array";
    status = inferInterface->InsertPerfInfoIntoJson(tmpJsonStr,
        {PerfInfoType::PERF_BATCH_SZIE},
        {"batch_size"}
    );
    EXPECT_FALSE(status.IsOk());
}

TEST_F(InferInterfaceBaseTest, testDecodeProcess)
{
    KvCacheInfo kvcacheInfo;
    DmiServerInfo serverInfo("", "", "", kvcacheInfo, InferReqType::REQ_DECODE);
    auto grpcContext = std::make_shared<GrpcContext>(serverInfo);
    std::shared_ptr<RequestContext> requestContext{nullptr};
    auto dResultDispatcher =  std::make_shared<DResultEventDispatcher>();
    auto tmpDispatcher = atomic_load(&dResultDispatcher);
    auto decodeReqHandler = std::make_shared<SingleLLMDecodeReqHandler>(requestContext, tmpDispatcher, grpcContext);
    auto decoderInferInterface = std::make_shared<SingleReqTgiTextInferInterface>(decodeReqHandler);
    auto* response = new (std::nothrow) prefillAndDecodeCommunication::DecodeRequestResponse();
    decoderInferInterface->DecodeProcess(*response);
    // invalid values
    //      case 1: failed to generate infer request
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::GenerateInferRequest,
                            bool (*)(std::string &))
                     .stubs();
    stubs.will(returnValue(false));
    decoderInferInterface->DecodeProcess(*response);
}

// NOTE: the following unit tests come from the code repo before refactor

TEST_F(InferInterfaceBaseTest, testParseDetokenizedOutput)
{
    std::string inferResult;
    uint64_t seqId = 0;
    bool isStream = false;

    inferResult = R"({"tool_calls": "mockToolCalls"})";
    inferInterface->ParseDetokenizedOutput(inferResult, seqId, isStream);
    EXPECT_EQ(inferResult, "");

    inferResult = R"({"origintext": "mockOriginText"})";
    inferInterface->ParseDetokenizedOutput(inferResult, seqId, isStream);
    EXPECT_EQ(inferResult, "mockOriginText");

    inferResult = R"({"content": "mockGeneratedContent"})";
    inferInterface->ParseDetokenizedOutput(inferResult, seqId, isStream);
    EXPECT_EQ(inferResult, "mockGeneratedContent");

    inferResult = R"({"reasoning_content": "mockReasoningContent"})";
    inferInterface->ParseDetokenizedOutput(inferResult, seqId, isStream);
    EXPECT_EQ(inferResult, "");

    inferResult = R"({})"; // empty
    inferInterface->ParseDetokenizedOutput(inferResult, seqId, isStream);
    EXPECT_EQ(inferResult, "");

    inferResult = "mockJson"; // Json::parse_error
    inferInterface->ParseDetokenizedOutput(inferResult, seqId, isStream);
    EXPECT_EQ(inferResult, "mockJson");
}

TEST_F(InferInterfaceBaseTest, testBuildResponseJson)
{
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    EXPECT_TRUE(inferInterface->BuildResponseJson(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(InferInterfaceBaseTest, testSendStreamResponse)
{
    RespBodyQueue jsonStrings;
    auto stubs = MOCKER_CPP(&SingleReqInferInterfaceBase::GetAvailableOutputCache,
                            bool (*)(std::vector<SingleReqInferInterfaceBase::StreamCache> &))
                     .stubs();
    stubs.will(returnValue(false));
    inferInterface->respTokenMap[0] = {0};
    inferInterface->SendStreamResponse(jsonStrings);
}

TEST_F(InferInterfaceBaseTest, testGetRequestId)
{
    ResponseSPtr response = std::make_shared<Response>(RequestIdNew("mockRequestId"));
    EXPECT_NE(inferInterface->GetRequestId().find("endpoint_common"), std::string::npos);
}

TEST_F(InferInterfaceBaseTest, testConvertTokenToMap)
{
    std::vector<BestNTokens> tempTokens;
    // case 1: invalid input
    BestNTokens illegalToken;
    illegalToken.finishReason = InferStatusType::ILLEGAL_INPUT;
    illegalToken.seqId = 100;
    illegalToken.parentSeqId = 0;
    inferInterface->eosMap[100] = InferStatusType::ITERATION_CONTINUE;
    tempTokens.push_back(illegalToken);
    inferInterface->ConvertTokenToMap(tempTokens);
    // case 2: clear unused decode cache
    tempTokens[0].finishReason = InferStatusType::ITERATION_CONTINUE;
    tempTokens[0].seqId = 201;
    tempTokens[0].parentSeqId = 200;
    tempTokens[0].tokens = {4, 5, 6};
    tempTokens[0].cumLogprobs = 0.5;
    tempTokens[0].logprob = 0.8;
    inferInterface->respTokenMap[0] = {0};
    inferInterface->respTokenMap[100] = {1, 2, 3};
    inferInterface->respTokenMap[200] = {10, 20, 30};
    inferInterface->eosMap[100] = InferStatusType::ITERATION_CONTINUE;
    inferInterface->eosMap[200] = InferStatusType::ITERATION_CONTINUE;
    inferInterface->ConvertTokenToMap(tempTokens);
}

TEST_F(InferInterfaceBaseTest, testPushLatestCache)
{
    std::string errMsg;
    // valid values
    uint64_t seqId = 100;
    inferInterface->eosMap[seqId] = InferStatusType::ITERATION_CONTINUE;
    inferInterface->probesMap[seqId] = 3.14;
    inferInterface->inputParam->postSingleText[seqId] = "test_text";
    inferInterface->fullTextMap[seqId] = "full_text";
    EXPECT_TRUE(inferInterface->PushLatestCache(errMsg));
    // invalid values
    std::string invalidUtf8(1, '\x80');
    inferInterface->inputParam->postSingleText[seqId] = invalidUtf8;
    EXPECT_FALSE(inferInterface->PushLatestCache(errMsg));
}

TEST_F(InferInterfaceBaseTest, testProcessStreamCacheTruncationId)
{
    std::string errMsg;
    // valid values
    //      should return true streamCache is empty
    EXPECT_TRUE(inferInterface->ProcessStreamCacheTruncationId(errMsg));
    //      should return true streamCache is valid
    std::u16string oldU16Text = u"old_text";
    std::u16string newU16Text = u"new_text_abcdef";
    const uint64_t seqId = 100;
    SingleReqInferInterfaceBase::StreamCache cacheA;
    cacheA.u16TokenText[seqId] = u"abcde";
    cacheA.eosMap[seqId] = InferStatusType::END_OF_SENTENCE;
    cacheA.fullTextMap[seqId] = "A_full";
    cacheA.finishReasonMap[seqId] = "A_reason";
    cacheA.postSingleText[seqId] = "A_post";
    cacheA.canOutput[seqId] = false;
    SingleReqInferInterfaceBase::StreamCache cacheB;
    cacheB.u16TokenText[seqId] = u"67890";
    cacheB.eosMap[seqId] = InferStatusType::END_OF_SENTENCE;
    cacheB.fullTextMap[seqId] = "B_full";
    cacheB.finishReasonMap[seqId] = "B_reason";
    cacheB.postSingleText[seqId] = "B_post";
    cacheB.canOutput[seqId] = false;
    SingleReqInferInterfaceBase::StreamCache cacheC;
    cacheC.u16TokenText[seqId] = u"12345";
    cacheC.eosMap[seqId] = InferStatusType::END_OF_SENTENCE;
    cacheC.fullTextMap[seqId] = "C_full";
    cacheC.finishReasonMap[seqId] = "C_reason";
    cacheC.postSingleText[seqId] = "C_post";
    cacheC.canOutput[seqId] = false;
    inferInterface->streamCache.push_back(cacheC);
    inferInterface->streamCache.push_back(cacheB);
    inferInterface->streamCache.push_back(cacheA);
    inferInterface->truncationIdMap[seqId] = -12;
    EXPECT_TRUE(inferInterface->ProcessStreamCacheTruncationId(errMsg));
}

TEST_F(InferInterfaceBaseTest, testProcessStreamCacheWindowSize)
{
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    request->windowSize = 100;
    inferInterface->request_ = request;
    std::u16string u16Text1 = u"text1";
    SingleReqInferInterfaceBase::StreamCache cache1 =
        CreateStreamCache(testSeqId, u16Text1, EOS_CONTINUE, false);
    std::u16string u16Text2 = u"text2";
    SingleReqInferInterfaceBase::StreamCache cache2 =
        CreateStreamCache(testSeqId, u16Text2, EOS_ABORT, false);
    inferInterface->streamCache.push_back(cache1);
    inferInterface->streamCache.push_back(cache2);
    EXPECT_TRUE(inferInterface->ProcessStreamCacheWindowSize());
}

TEST_F(InferInterfaceBaseTest, testGetAvailableOutputCache)
{
    // valid values
    //      should return true given stopStrings is empty
    RequestSPtr request = std::make_shared<Request>(RequestIdNew("mockRequest"));
    inferInterface->request_ = request;
    std::vector<SingleReqInferInterfaceBase::StreamCache> cacheArr;
    EXPECT_TRUE(inferInterface->GetAvailableOutputCache(cacheArr));
    //      should return true streamCache is valid
    inferInterface->request_->stopStrings = std::optional<std::string>("stop");
    std::u16string u16Text = u"test_text";
    SingleReqInferInterfaceBase::StreamCache cache0 = CreateStreamCache(testSeqId, u16Text, EOS_CONTINUE, false);
    SingleReqInferInterfaceBase::StreamCache cache1 = CreateStreamCache(testSeqId, u16Text, EOS_CONTINUE, true);
    SingleReqInferInterfaceBase::StreamCache cache2 = CreateStreamCache(testSeqId, u16Text, EOS_ABORT, false);
    inferInterface->streamCache.push_back(cache0);
    inferInterface->streamCache.push_back(cache1);
    inferInterface->streamCache.push_back(cache2);
    EXPECT_TRUE(inferInterface->GetAvailableOutputCache(cacheArr));
}

TEST_F(InferInterfaceBaseTest, testProcessResponseStream)
{
    MOCKER_CPP(&SingleReqOpenAiInferInterface::SendStreamResponse, void (*)(RespBodyQueue &))
        .stubs();
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> tempTokens;
    RespBodyQueue jsonStrings;
    uint64_t timestamp = 0;
    inferInterface->eosMap[testSeqId] = InferStatusType::ITERATION_CONTINUE;
    inferInterface->updateIndexMap[testSeqId] = true;
    inferInterface->respTokenMap[testSeqId] = {101, 102, 103};
    inferInterface->truncationIdMap[testSeqId] = -5;
    BestNTokens item;
    item.seqId = testSeqId;
    item.tokens = {1001, 1002};
    item.finishReason = InferStatusType::ITERATION_CONTINUE;
    tempTokens.push_back(item);
    auto stubsTokenizer = MOCKER_CPP(&TokenizerProcessPool::DecodeOne,
        Status(*)(TokenizerProcessPool*, std::vector<int64_t> &, std::string &, uint32_t, uint32_t,
                  const uint64_t &, const bool &, const bool &, const bool, const DetokenizeExtraInfo &)).stubs();
    // should return true given tokenizer success
    stubsTokenizer.will(invoke(&MockTokenizerDecodeOneSuccess));
    EXPECT_TRUE(inferInterface->ProcessResponseStream(resp, tempTokens, jsonStrings, timestamp));
    // should return false given tokenizer fails
    stubsTokenizer.will(invoke(&MockTokenizerDecodeOneFail));
    EXPECT_FALSE(inferInterface->ProcessResponseStream(resp, tempTokens, jsonStrings, timestamp));
    // case when finishReason is not continue
    item.finishReason = InferStatusType::END_OF_SENTENCE;
    tempTokens.clear();
    tempTokens.push_back(item);
    MOCKER_CPP(&SingleReqInferInterfaceBase::PostProcess, bool(*)(SingleReqInferInterfaceBase*, const std::vector<int64_t> &,
        std::string &, const uint64_t &, bool, uint32_t, uint32_t, const uint64_t &)).stubs().will(returnValue(true));
    stubsTokenizer.will(invoke(&MockTokenizerDecodeOneSuccess));
    EXPECT_TRUE(inferInterface->ProcessResponseStream(resp, tempTokens, jsonStrings, timestamp));
}

TEST_F(InferInterfaceBaseTest, testGetGetMetrics)
{
    Metrics& reqMetrics = reqHandler->GetMetrics();
    Metrics& infMetrics = inferInterface->GetMetrics();
    reqMetrics.callbackIndex++;
    EXPECT_EQ(infMetrics.callbackIndex, reqMetrics.callbackIndex);
}

TEST_F(InferInterfaceBaseTest, testDecodeSingleToken)
{
    std::vector<int64_t> tokenIds = {100, 200, 300};
    uint32_t prevDecodeIndex = 0;
    uint32_t currentDecodeIndex = 1;
    bool skipSpecialTokens = false;
    std::string output;
    auto stubsTokenizer = MOCKER_CPP(&TokenizerProcessPool::DecodeOne,
        Status(*)(TokenizerProcessPool*, std::vector<int64_t> &, std::string &, uint32_t, uint32_t,
                  const uint64_t &, const bool &, const bool &, const bool, const DetokenizeExtraInfo &)).stubs();
    // should return false given tokenizer fails
    stubsTokenizer.will(invoke(&MockTokenizerDecodeOneFail));
    EXPECT_FALSE(inferInterface->DecodeSingleToken(tokenIds, output, prevDecodeIndex, currentDecodeIndex, skipSpecialTokens));
    // should return false parse result from tokenizer failed
    stubsTokenizer.will(invoke(&MockTokenizerDecodeOneSuccess));
    EXPECT_FALSE(inferInterface->DecodeSingleToken(tokenIds, output, prevDecodeIndex, currentDecodeIndex, skipSpecialTokens));
}
} // namespace mindie_llm
