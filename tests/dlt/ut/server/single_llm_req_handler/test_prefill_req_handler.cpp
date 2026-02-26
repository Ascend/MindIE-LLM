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
#include <libgen.h>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "single_llm_prefill_req_handler.h"
#include "param_checker.h"
#include "response.h"
#include "env_util.h"

using namespace mindie_llm;
using namespace prefillAndDecodeCommunication;

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

class PrefillReqHandlerTestF : public ::testing::Test {
protected:
    httplib::Request req{};
    httplib::Response res{};
    std::shared_ptr<RequestContext> ctx;
    ResponseSPtr response;
    std::function<bool(const ResponseSPtr &response, const std::vector<BestNTokens> &, RespBodyQueue &)> func;
    void SetUp() override
    {
        req.method = "POST";
        req.path = "/prefill";
        req.version = "HTTP/1.1";
        ctx = std::make_shared<RequestContext>(req, res);
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_grpc.json");
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        // Silence HTTP responses in tests
        MOCKER_CPP(&HttpRestResource::ResponseJsonBody, int (*)(const ReqCtxPtr &, int, const std::string &))
            .stubs()
            .will(returnValue(0));
        RequestIdNew reqId(std::to_string(300));
        response = std::make_shared<Response>(reqId);
        ResponseContent content;
        content.seqId = 1001;
        content.parentSeqId = 1000;
        content.outTokenIds = {42};
        content.cumLogProb = 0.95f;
        content.finishReason = InferStatusType::ITERATION_CONTINUE;
        content.truncationIndex = -1;
        content.outLogProbs = {0.9f};
        content.speculativeTokenNum = 1;
        content.topLogProbTokenIds = {42, 43, 44};
        content.topLogProbs = {0.9f, 0.8f, 0.7f};
        response->responseContents.push_back(content);
        response->metrics.batchSize = 1;
        response->metrics.queueWaitTime = 10;
        func = [](const ResponseSPtr &response, const std::vector<BestNTokens> &, RespBodyQueue &queue) {
                    queue.push("Start");
                    queue.push("End");
                    return true;
                };
    }

    void TearDown() override
    {
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
    }

    std::string GetParentDirectory()
    {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }

        std::unique_ptr<char, decltype(&std::free)> temp(strdup(buffer), &std::free);
        
        if (temp == nullptr) {
            std::cerr << "Memory allocation failed" << std::endl;
            return "";
        }

        char* parent = dirname(temp.get());
        std::string result(parent);
        return result;
    }
};

TEST_F(PrefillReqHandlerTestF, Constructor_SetsFlags)
{
    SingleLLMPrefillReqHandler handler(ctx, /*msgType*/ 123, /*isRecompute*/ true);
    EXPECT_EQ(handler.msgType_, 123);
    EXPECT_TRUE(handler.isRecompute_);
    EXPECT_EQ(handler.reqType_, InferReqType::REQ_PREFILL);
}

TEST_F(PrefillReqHandlerTestF, GetContextJsonBody_SuccessAndFailure)
{
    // Success
    req.body = R"({"k":1,"s":"ok"})";
    ctx = std::make_shared<RequestContext>(req, res);
    SingleLLMPrefillReqHandler handler(ctx, 0, false);
    nlohmann::ordered_json jsonBody;
    EXPECT_TRUE(handler.GetContextJsonBody(jsonBody));
    EXPECT_EQ(jsonBody["k"], 1);

    // Failure (bad JSON)
    req.body = "{bad json";
    ctx = std::make_shared<RequestContext>(req, res);
    SingleLLMPrefillReqHandler handler2(ctx, 0, false);
    nlohmann::ordered_json jsonBody2;
    EXPECT_FALSE(handler2.GetContextJsonBody(jsonBody2));
}

TEST_F(PrefillReqHandlerTestF, GetContextRequestId_HeaderPresentAndMissing)
{
    // Present
    req.headers.emplace("req-id", "RID-001");
    ctx = std::make_shared<RequestContext>(req, res);
    SingleLLMPrefillReqHandler handler(ctx, 0, false);
    std::string requestId;
    EXPECT_TRUE(handler.GetContextRequestId(requestId));
    EXPECT_EQ(requestId, "RID-001");

    // Missing
    httplib::Request req2;
    httplib::Response resp2;
    auto ctx2 = std::make_shared<RequestContext>(req2, resp2);
    SingleLLMPrefillReqHandler handler2(ctx2, 0, false);
    std::string requestId2;
    EXPECT_FALSE(handler2.GetContextRequestId(requestId2));
    EXPECT_TRUE(requestId2.empty());
}

TEST_F(PrefillReqHandlerTestF, UpdateInferRequest_SetsFields)
{
    SingleLLMPrefillReqHandler handler(ctx, /*msgType*/ 9, /*isRecompute*/ true);

    auto param = std::make_shared<InferParam>();
    param->model_ = "m-1";

    RequestSPtr request = std::make_shared<Request>();
    std::vector<int64_t> tokens{10, 11, 12};
    int64_t tokenLen = 2;

    EXPECT_NO_THROW(handler.UpdateInferRequest(tokens, tokenLen, request));
    EXPECT_NO_THROW(handler.UpdateInferParam(request, param));
    ASSERT_EQ(handler.inferParam_.get(), param.get());
    EXPECT_EQ(handler.inputTokens_, tokens);
    EXPECT_EQ(handler.oriReqTokenLen, static_cast<size_t>(tokenLen));
    EXPECT_EQ(request->reqType, InferReqType::REQ_PREFILL);
    EXPECT_TRUE(request->isRecompute);
}

TEST_F(PrefillReqHandlerTestF, BuildSamplingParameters_FirstAndNext)
{
    SingleLLMPrefillReqHandler handler(ctx, 0, false);
    auto param = std::make_shared<InferParam>();
    RequestSPtr request = std::make_shared<Request>("");

    // First batch
    request->temperature = 1.5f;
    request->topK = 42;
    request->topP = 0.7f;
    request->typicalP = 0.6f;
    request->doSample = true;
    request->seed = 123u;
    request->repetitionPenalty = 1.1f;
    request->watermark = true;

    // Next batch
    request->frequencyPenalty = 0.2f;
    request->presencyPenalty = 0.3f;
    request->stopTokenIds = std::vector<int64_t>{7, 8};
    request->stopStrings = std::string("STOP");
    request->stopStrList = {"STOP"};
    request->skipSpecialTokens = true;
    request->includeStopStrInOutput = false;
    request->ignoreEos = true;
    request->logprobs = true;
    request->topLogprobs = 2u;
    param->enableThinking = true;

    handler.inferParam_ = param;
    handler.request_ = request;

    DecodeParameters decodeParam;
    handler.BuildSamplingParametersFirst(decodeParam);
    handler.BuildSamplingParametersNext(decodeParam);

    const auto &samplingParam = decodeParam.samplingparams();
    EXPECT_FLOAT_EQ(samplingParam.temperature().value(), 1.5f);
    EXPECT_EQ(samplingParam.topk().value(), 42);
    EXPECT_FLOAT_EQ(samplingParam.topp().value(), 0.7f);
    EXPECT_FLOAT_EQ(samplingParam.typicalp().value(), 0.6f);
    EXPECT_TRUE(samplingParam.dosample().value());
    EXPECT_EQ(samplingParam.seed().value(), 123u);
    EXPECT_FLOAT_EQ(samplingParam.repetitionpenalty().value(), 1.1f);
    EXPECT_TRUE(samplingParam.watermark().value());

    EXPECT_FLOAT_EQ(samplingParam.frequencypenalty().value(), 0.2f);
    EXPECT_FLOAT_EQ(samplingParam.presencypenalty().value(), 0.3f);
    ASSERT_EQ(samplingParam.stoptokenids().value_size(), 2);
    EXPECT_EQ(samplingParam.stoptokenids().value(0), 7);
    EXPECT_EQ(samplingParam.stoptokenids().value(1), 8);
    EXPECT_EQ(samplingParam.stopstrings().value(), "STOP");
    EXPECT_EQ(samplingParam.stopstrings().list_size(), 1);
    EXPECT_EQ(samplingParam.stopstrings().list(0), "STOP");
    EXPECT_TRUE(samplingParam.skipspecialtokens().value());
    EXPECT_FALSE(samplingParam.includestopstrinoutput().value());
    EXPECT_TRUE(samplingParam.ignoreeos().value());
    EXPECT_TRUE(samplingParam.logprobs().value());
    EXPECT_EQ(samplingParam.toplogprobs().value(), 2u);
    EXPECT_TRUE(samplingParam.enablethinking().value());
}

TEST_F(PrefillReqHandlerTestF, BuildInferParameters_SetsPriorityTimeout)
{
    SingleLLMPrefillReqHandler handler(ctx, 0, false);
    auto param = std::make_shared<InferParam>();
    param->timeout = 9000;
    handler.inferParam_ = param;
    auto request = std::make_shared<Request>();
    request->priority = 5;
    handler.request_ = request;

    DecodeParameters decodeParam;
    handler.BuildInferParameters(decodeParam);

    EXPECT_EQ(decodeParam.inferparams().priority(), 5);
    EXPECT_EQ(decodeParam.inferparams().timeout(), 9000);
}

TEST_F(PrefillReqHandlerTestF, BuildMetricsParameters_PacksMetrics)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    handler->metrics.firstTokenCost = 12u;
    handler->metrics.lastTokenCost = 34u;
    handler->metrics.decodeTime = {3u, 4u, 5u};
    handler->metrics.batchSize = {1, 2};
    handler->metrics.queueWaitTime = {7, 8, 9};
    handler->metrics.callbackIndex = 99u;

    DecodeParameters decodeParam;
    handler->BuildMetricsParameters(decodeParam);

    const auto &metrics = decodeParam.metrics();
    EXPECT_EQ(metrics.firsttokencost(), 12u);
    EXPECT_EQ(metrics.lasttokencost(), 34u);
    ASSERT_EQ(metrics.decodetime_size(), 3);
    EXPECT_EQ(metrics.decodetime(0), 3u);
    EXPECT_EQ(metrics.decodetime(2), 5u);

    ASSERT_EQ(metrics.batchsize_size(), 2);
    EXPECT_EQ(metrics.batchsize(1), 2);

    ASSERT_EQ(metrics.queuewaittime_size(), 3);
    EXPECT_EQ(metrics.queuewaittime(2), 9);

    EXPECT_EQ(metrics.callbackindex(), 99u);
}

TEST_F(PrefillReqHandlerTestF, BuildDecodeParameters_PacksAllKeyFields)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, /*msgType*/ 555, false);

    // Prefill handler internal state
    handler->prefillNodeAddr_ = "10.0.0.1:7000";
    handler->pInstanceId_ = 7;
    handler->reqId_ = "RID-X";
    handler->streamMode_ = true;
    handler->inputTokens_ = {10, 11, 12, 13};
    handler->oriReqTokenLen = 2; // split boundary
    handler->firstToken_ = {99};

    // Metrics timestamp -> stable for comparison
    handler->metrics.sysE2eStartingTime = std::chrono::system_clock::time_point{}; // epoch -> 0us

    // InferParam setup
    auto inferParam = std::make_shared<InferParam>();
    RequestSPtr request = std::make_shared<Request>("");
    inferParam->batchSize = 3;
    inferParam->isMaxNewTokensSetByUser = true;
    inferParam->maxNewTokens = 50;
    inferParam->truncate = 1;
    inferParam->tools = "[]";
    inferParam->toolChoice = "auto";
    request->loraId = "lora-1";
    inferParam->returnFullText = true;
    inferParam->decoderInputDetails = true;
    inferParam->outputNames = {"logits", "probs"};
    inferParam->showDetails = true;
    request->requestId= "uid-123";
    inferParam->useToolsCall = false;
    inferParam->prevDecodeIndex = {{0u, 42u}};
    inferParam->currentDecodeIndex = {{0u, 77u}};
    inferParam->postSingleText = {{0u, std::string("post")}};
    inferParam->userInputId = std::string("u-1");
    inferParam->textInput = std::string("hello");
    request->temperature = 1.0f;
    request->topK = 5;
    request->topP = 0.9f;
    request->ignoreEos = false;
    handler->inferParam_ = inferParam;
    handler->request_ = request;

    // Response to supply src block table and producer instance id
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    resp->responseContents.resize(1);
    resp->responseContents[0].srcBlockTable = {1, 2};
    resp->responseContents[0].singleLLMPrefillReqHandlerId = 8;

    DecodeParameters out;
    handler->BuildDecodeParameters(resp, out);

    // Basic check
    EXPECT_EQ(out.pnodeaddr(), "10.0.0.1:7000");
    EXPECT_EQ(out.pinstanceid(), 7u);
    EXPECT_EQ(out.reqid(), "RID-X");
    EXPECT_EQ(out.msgtype(), 555u);
    EXPECT_TRUE(out.isstream());

    // Token packing:
    // - first `oriReqTokenLen` go to tokens
    // - remaining input tokens + firstToken_ go to firsttoken
    ASSERT_EQ(out.tokens_size(), 2);
    EXPECT_EQ(out.tokens(0), 10);
    EXPECT_EQ(out.tokens(1), 11);

    ASSERT_EQ(out.firsttoken_size(), 3);
    EXPECT_EQ(out.firsttoken(0), 12);
    EXPECT_EQ(out.firsttoken(1), 13);
    EXPECT_EQ(out.firsttoken(2), 99);

    // Output names & batch size
    ASSERT_EQ(out.outputnames_size(), 2);
    EXPECT_EQ(out.outputnames(0), "logits");
    EXPECT_EQ(out.batchsize(), 3u);

    // Max new token logic
    EXPECT_EQ(out.maxnewtoken(), 50u);

    // Misc infer params
    EXPECT_EQ(out.truncate(), 1);
    EXPECT_EQ(out.tools(), "[]");
    EXPECT_EQ(out.toolchoice(), "auto");
    EXPECT_EQ(out.loraid(), "lora-1");
    EXPECT_TRUE(out.returnfulltext());
    EXPECT_TRUE(out.decoderinputdetails());
    EXPECT_TRUE(out.details());
    EXPECT_EQ(out.id(), "uid-123");
    EXPECT_FALSE(out.usetoolcall());

    // E2E start time (epoch -> 0)
    EXPECT_EQ(out.e2estarttime(), 0u);

    // Block table
    ASSERT_EQ(out.blocktable_size(), 2);
    EXPECT_EQ(out.blocktable(1), 2);

    // Prev/current decode index + post single text
    EXPECT_EQ(out.prevdecodeindex(), 42u);
    EXPECT_EQ(out.currentdecodeindex(), 77u);
    EXPECT_EQ(out.postsingletext(), "post");

    // Optional inputs
    EXPECT_TRUE(out.has_inputid());
    EXPECT_EQ(out.inputid().value(), "u-1");
    EXPECT_TRUE(out.has_textinput());
    EXPECT_EQ(out.textinput().value(), "hello");

    // DP instance ids from response content
    ASSERT_EQ(out.dpinstanceids_size(), 1);
    EXPECT_EQ(out.dpinstanceids(0), 8u);

    // Check a few sampling params
    const auto &s = out.samplingparams();
    EXPECT_EQ(s.topk().value(), 5);
    EXPECT_FLOAT_EQ(s.topp().value(), 0.9f);
    EXPECT_FLOAT_EQ(s.temperature().value(), 1.0f);
    EXPECT_FALSE(s.ignoreeos().value());
}

TEST_F(PrefillReqHandlerTestF, AtomicReadWriteFinish_OnlyTransitionsOnce)
{
    SingleLLMPrefillReqHandler handler(ctx, 0, false);
    bool expect = false;
    EXPECT_TRUE(handler.AtomicReadWriteFinish(expect)); // flips false -> true
    expect = false;
    EXPECT_FALSE(handler.AtomicReadWriteFinish(expect)); // already finished; no flip
}

static inline Status Ok() { return Status(Error::Code::OK); }
static inline Status InvalidArg() { return Status(Error::Code::INVALID_ARG); }
static inline Status InternalErr() { return Status(Error::Code::ERROR); }

TEST_F(PrefillReqHandlerTestF, Process_EnqueueInvalidArg_Sends424)
{
    // Stub GetInferInstance()->Process -> INVALID_ARG
    MOCKER_CPP(&InferInstance::Process, Status (*)(RequestSPtr)).stubs().will(returnValue(InvalidArg()));
    MOCKER_CPP(&SingleLLMPrefillReqHandler::GetPNodeAddr, void (*)(bool)).stubs();
    MOCKER_CPP(&HttpRestResource::ResponseJsonBody, int (*)(const ReqCtxPtr &, int, const std::string &))
        .stubs()
        .will(returnValue(0));

    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    auto req = std::make_shared<Request>();
    handler->constructOneResponseCallBack_ = func;
    EXPECT_NO_THROW(handler->Process(req, "RID-P1", /*timeout*/ 1000));
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
}

TEST_F(PrefillReqHandlerTestF, Process_TimesOut_StopAnd500)
{
    // Stub GetInferInstance()->Process -> OK
    MOCKER_CPP(&InferInstance::Process, Status (*)(RequestSPtr)).stubs().will(returnValue(Ok()));
    MOCKER_CPP(&SingleLLMPrefillReqHandler::GetPNodeAddr, void (*)(bool)).stubs();
    MOCKER_CPP(&InferInstance::ControlRequest, Status (*)(const RequestIdNew &, OperationV2))
        .stubs()
        .will(returnValue(Ok()));
    MOCKER_CPP(&HttpRestResource::ResponseJsonBody, int (*)(const ReqCtxPtr &, int, const std::string &))
        .stubs()
        .will(returnValue(0));
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    auto req = std::make_shared<Request>();

    EXPECT_NO_THROW(handler->Process(req, "RID-P2", /*timeout*/ 1000));
    EXPECT_TRUE(handler->isFinish_.load());
}

TEST_F(PrefillReqHandlerTestF, Process_CallBack)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    auto req = std::make_shared<Request>();
    handler->constructOneResponseCallBack_ = func;
    handler->SetBackManagerCallBack(req);
    auto callback = req->serverResponseCallback_;
    callback(nullptr);
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
    EXPECT_FALSE(handler->isFinish_.load());
    handler->constructOneResponseCallBack_ = func;
    response->inferStatusFlag = InferStatusType::RELEASE_KV_COMPLETE;
    callback(response);
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
    EXPECT_FALSE(handler->isFinish_.load());
    handler->constructOneResponseCallBack_ = func;
    response->inferStatusFlag = InferStatusType::ILLEGAL_INPUT;
    callback(response);
    EXPECT_FALSE(handler->isFinish_.load());
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
    handler->constructOneResponseCallBack_ = func;
    response->inferStatusFlag = InferStatusType::ITERATION_CONTINUE;
    handler->isFinish_.store(true);
    callback(response);
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
}

TEST_F(PrefillReqHandlerTestF, Process_EosTrue_CallBack)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    auto req = std::make_shared<Request>();
    handler->SetBackManagerCallBack(req);
    auto callback = req->serverResponseCallback_;
    response->inferStatusFlag = InferStatusType::ITERATION_CONTINUE;
    handler->isFinish_.store(false);
    response->isEos = true;
    MOCKER_CPP(&SingleLLMPrefillReqHandler::GenerateFirstToken,
              bool (*)(ResponseSPtr, bool)).stubs().will(returnValue(false));
    handler->constructOneResponseCallBack_ = func;
    callback(response);
    EXPECT_NE(handler->constructOneResponseCallBack_, nullptr);
    EXPECT_FALSE(handler->isFinish_.load());
    GlobalMockObject::verify();
    MOCKER_CPP(&SingleLLMPrefillReqHandler::GenerateFirstToken,
              bool (*)(ResponseSPtr, bool)).stubs().will(returnValue(true));
    callback(response);
    EXPECT_NE(handler->constructOneResponseCallBack_, nullptr);
    EXPECT_TRUE(handler->isFinish_.load());
}

TEST_F(PrefillReqHandlerTestF, TransferStatusFlag_CallBack)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    auto req = std::make_shared<Request>();
    handler->SetBackManagerCallBack(req);
    auto callback = req->serverResponseCallback_;
    response->inferStatusFlag = InferStatusType::ITERATION_CONTINUE;
    handler->isFinish_.store(false);
    response->isEos = false;
    MOCKER_CPP(&SingleLLMPrefillReqHandler::GenerateFirstToken,
              bool (*)(ResponseSPtr, bool)).stubs().will(returnValue(false));
    response->transferStatusFlag = TransferStatusType::PREFILL_COMPLETE;
    handler->constructOneResponseCallBack_ = func;
    callback(response);
    EXPECT_NE(handler->constructOneResponseCallBack_, nullptr);

    response->transferStatusFlag = TransferStatusType::PUBLISH_KV_COMPLETE;
    MOCKER_CPP(&SingleLLMPrefillReqHandler::BuildDecodeParameters,
              void (*)(ResponseSPtr, DecodeParameters&)).stubs();
    callback(response);
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
    EXPECT_TRUE(handler->isFinish_.load());

    handler->isFinish_.store(false);
    handler->constructOneResponseCallBack_ = func;
    MOCKER_CPP(&GrpcCommunicationMng::SendDecodeRequest,
              bool (*)(DecodeParameters&, const std::string&, const std::string&, std::string&)).stubs()
              .will(returnValue(false));
    callback(response);
    EXPECT_TRUE(handler->isFinish_.load());
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);

    handler->isFinish_.store(false);
    handler->constructOneResponseCallBack_ = func;
    response->transferStatusFlag = TransferStatusType::NOT_TRANSFER;
    callback(response);
    EXPECT_TRUE(handler->isFinish_.load());
    EXPECT_EQ(handler->constructOneResponseCallBack_, nullptr);
}

TEST_F(PrefillReqHandlerTestF, GetPNodeAddr)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    handler->GetPNodeAddr(true);
    EXPECT_EQ(handler->prefillNodeAddr_, "127.0.0.1;1121");
    
}

TEST_F(PrefillReqHandlerTestF, GenerateFirstToken_FailCase)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseLogProbsFromResponse,
              bool (*)(const ResponseSPtr,
              std::vector<BestNTokens>)).stubs().will(returnValue(true));
    EXPECT_FALSE(handler->GenerateFirstToken(response, true));
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseTokensFromResponse,
              bool (*)(const ResponseSPtr,
              std::vector<BestNTokens>)).stubs().will(returnValue(false));
    EXPECT_FALSE(handler->GenerateFirstToken(response, true));
}

TEST_F(PrefillReqHandlerTestF, GenerateFirstToken_SuccessCase)
{
    auto handler = std::make_shared<SingleLLMPrefillReqHandler>(ctx, 0, false);
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseLogProbsFromResponse, bool (*)(const ResponseSPtr, std::vector<BestNTokens>)).stubs().will(returnValue(true));
    handler->SetConstructOneResponseCallBack(
        [](const ResponseSPtr &,
           const std::vector<BestNTokens> &,
           RespBodyQueue &respQueue) -> bool {
            respQueue.push("decoded_text");
            return true;
        }
    );
    EXPECT_TRUE(handler->GenerateFirstToken(response, true));
    ASSERT_FALSE(handler->respStr_.empty());
    std::string front = handler->respStr_.front();
    handler->respStr_.pop();
    EXPECT_EQ(front, "decoded_text");
}