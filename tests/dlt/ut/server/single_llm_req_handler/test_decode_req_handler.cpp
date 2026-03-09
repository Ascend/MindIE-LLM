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
#include "single_llm_decode_req_handler.h"
#include "env_util.h"

using namespace mindie_llm;
using namespace prefillAndDecodeCommunication;

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

// ------------------------------ Helpers ------------------------------
static inline Status Ok() { return Status(Error::Code::OK); }
static inline Status InvalidArg() { return Status(Error::Code::INVALID_ARG); }
static inline Status InternalErr() { return Status(Error::Code::ERROR); }

struct TestBacking {
    DecodeParameters params;
    DmiServerInfo dmi;
    TritonTextInfo triton;
    TestBacking()
        : dmi(/*reqId*/ "",
              /*pNodeAddr*/ "",
              /*dtargetAddr*/ "",
              /*kv*/ KvCacheInfo{},
              /*reqType*/ InferReqType::REQ_DECODE)
    {
    }
};

static TestBacking TB;

void FillDecodeParams(DecodeParameters &params)
{
    // Reset all fields to empty/default to avoid cross-test accumulation.
    params.Clear();

    params.set_reqid("RID-DEC-001");
    params.set_batchsize(3);
    params.set_details(true);
    params.set_maxnewtoken(7);
    params.set_isstream(false);
    params.set_returnfulltext(true);
    params.set_decoderinputdetails(false);
    params.set_id("decode-id");
    params.set_tools("t-json");
    params.set_toolchoice("auto");
    params.set_loraid("lora-x");
    params.set_modelname("model-d");
    params.set_prevdecodeindex(11);
    params.set_currentdecodeindex(12);
    params.set_postsingletext("accum");
    params.set_usetoolcall(true);

    // tokens/firstToken/outputNames
    params.add_tokens(101);
    params.add_tokens(102);
    params.add_firsttoken(7);
    params.add_outputnames("logits");
    params.add_outputnames("ids");

    // sampling first-part
    auto *samplingParams = params.mutable_samplingparams();
    samplingParams->mutable_temperature()->set_value(1.25f);
    samplingParams->mutable_topk()->set_value(33);
    samplingParams->mutable_topp()->set_value(0.66f);
    samplingParams->mutable_typicalp()->set_value(0.5f);
    samplingParams->mutable_dosample()->set_value(true);
    samplingParams->mutable_seed()->set_value(42);
    samplingParams->mutable_repetitionpenalty()->set_value(1.1f);
    // sampling next-part
    samplingParams->mutable_watermark()->set_value(true);
    samplingParams->mutable_frequencypenalty()->set_value(0.2f);
    samplingParams->mutable_presencypenalty()->set_value(0.3f);
    samplingParams->mutable_stoptokenids()->add_value(9);
    samplingParams->mutable_stopstrings()->set_value("STOP");
    samplingParams->mutable_skipspecialtokens()->set_value(true);
    samplingParams->mutable_includestopstrinoutput()->set_value(false);
    samplingParams->mutable_ignoreeos()->set_value(false);
    samplingParams->mutable_logprobs()->set_value(true);
    samplingParams->mutable_toplogprobs()->set_value(2);
    samplingParams->mutable_enablethinking()->set_value(true);

    // infer params
    params.mutable_inferparams()->set_priority(5);
    params.mutable_inferparams()->set_timeout(9000);

    // input/text
    params.mutable_inputid()->set_value("UIN-1");
    params.mutable_textinput()->set_value("hello");

    // metrics
    auto *metrics = params.mutable_metrics();
    metrics->set_firsttokencost(12);
    metrics->set_lasttokencost(34);
    metrics->add_decodetime(3);
    metrics->add_decodetime(5);
    metrics->add_batchsize(1);
    metrics->add_batchsize(2);
    metrics->add_queuewaittime(7);
    metrics->add_queuewaittime(8);
    metrics->set_callbackindex(99);

    params.set_e2estarttime(0);
}

void PrepareDmiInfo(TestBacking &tb)
{
    tb.dmi.pNodeAddr = "10.0.0.1:50051";
    tb.dmi.kvCacheInfo.blockTable = {1, 2, 3};
    tb.dmi.kvCacheInfo.dpInstanceIds = {1001, 1002};
}

// Silence HTTP JSON response writes
void SilenceHttp()
{
    MOCKER_CPP(&HttpRestResource::ResponseJsonBody, int (*)(const ReqCtxPtr &, int, const std::string &))
        .stubs()
        .will(MOCKCPP_NS::returnValue(0));
}

void StubGrpcContext(std::shared_ptr<GrpcContext> &grpc)
{
    TB.triton.userSepcId = "TRI-SEP-1";
    grpc = std::make_shared<GrpcContext>(TB.dmi, TB.triton);
    grpc->SetDecodeParams(TB.params);
}

// ------------------------------ Tests ------------------------------
class DecodeReqHandlerTestF : public ::testing::Test {
protected:
    httplib::Request req{};
    httplib::Response resp{};
    std::shared_ptr<RequestContext> httpCtx;

    std::shared_ptr<DResultEventDispatcher> dres;
    std::shared_ptr<GrpcContext> grpc;
    std::shared_ptr<SingleLLMDecodeReqHandler> handler;

    uint16_t pullKVFlag;

    void SetUp() override
    {
        pullKVFlag = 0;
        req.method = "POST";
        req.path = "/decode";
        req.version = "HTTP/1.1";
        httpCtx = std::make_shared<RequestContext>(req, resp);
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE", GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_grpc.json");

        SilenceHttp();

        FillDecodeParams(TB.params);
        PrepareDmiInfo(TB);
        StubGrpcContext(grpc);
        handler = std::make_shared<SingleLLMDecodeReqHandler>(httpCtx, dres, grpc);
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
        MOCKCPP_NS::GlobalMockObject::verify();
        MOCKCPP_NS::GlobalMockObject::reset();
    }

    std::function<bool(const ResponseSPtr &response, const std::vector<BestNTokens> &, RespBodyQueue &)> func;
    
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

TEST_F(DecodeReqHandlerTestF, Constructor_WiresIdsAndReqType)
{
    EXPECT_EQ(handler->reqType_, InferReqType::REQ_DECODE);
    EXPECT_EQ(handler->pNodeAddr_, "10.0.0.1:50051");
    EXPECT_EQ(handler->reqId_, "RID-DEC-001");
    EXPECT_EQ(handler->tritonReqId_, "TRI-SEP-1");
}

TEST_F(DecodeReqHandlerTestF, GetContextJsonBody_DoesNotModifyBody)
{
    nlohmann::ordered_json body = {{"key", "value"}};
    bool ret = handler->GetContextJsonBody(body);
    EXPECT_TRUE(ret);
    EXPECT_EQ(body["key"], "value");
}

TEST_F(DecodeReqHandlerTestF, GetContextRequestId_FromGrpcParams)
{
    std::string rid;
    EXPECT_TRUE(handler->GetContextRequestId(rid));
    EXPECT_EQ(rid, "RID-DEC-001");
}

TEST_F(DecodeReqHandlerTestF, GetContextJsonBody_PopulatesAllFields)
{
    auto param = std::make_shared<InferParam>();
    RequestSPtr request = std::make_shared<Request>();
    std::vector<int64_t> reqTokens, respTokens;

    EXPECT_TRUE(handler->GetContextJsonBody(param, request, reqTokens, respTokens));

    // tokens
    ASSERT_EQ(reqTokens.size(), 2u);
    EXPECT_EQ(reqTokens[0], 101);
    ASSERT_EQ(respTokens.size(), 1u);
    EXPECT_EQ(respTokens[0], 7);

    // sampling (first)
    EXPECT_FLOAT_EQ(request->temperature.value(), 1.25f);
    EXPECT_EQ(request->topK.value(), 33);
    EXPECT_FLOAT_EQ(request->topP.value(), 0.66f);
    EXPECT_FLOAT_EQ(request->typicalP.value(), 0.5f);
    EXPECT_TRUE(request->doSample.value());
    EXPECT_EQ(request->seed.value(), 42u);
    EXPECT_FLOAT_EQ(request->repetitionPenalty.value(), 1.1f);

    // sampling (next)
    EXPECT_TRUE(request->watermark.value());
    EXPECT_FLOAT_EQ(request->frequencyPenalty.value(), 0.2f);
    EXPECT_FLOAT_EQ(request->presencyPenalty.value(), 0.3f);
    ASSERT_TRUE(request->stopTokenIds.has_value());
    ASSERT_EQ(request->stopTokenIds->size(), 1u);
    EXPECT_EQ(request->stopTokenIds->at(0), 9);
    EXPECT_EQ(request->stopStrings.value(), "STOP");
    EXPECT_TRUE(request->skipSpecialTokens.value());
    EXPECT_FALSE(request->includeStopStrInOutput.value());
    EXPECT_FALSE(request->ignoreEos.value());
    EXPECT_TRUE(request->logprobs.value());
    EXPECT_EQ(request->topLogprobs.value(), 2u);
    ASSERT_TRUE(param->enableThinking.has_value());
    EXPECT_TRUE(param->enableThinking.value());

    // infer params + misc
    EXPECT_EQ(request->priority, 5u);
    EXPECT_EQ(param->timeout, 9000u);
    EXPECT_EQ(request->requestId, "decode-id");
    EXPECT_EQ(param->tools, "t-json");
    EXPECT_EQ(param->toolChoice, "auto");
    EXPECT_EQ(request->loraId, "lora-x");
    EXPECT_EQ(param->model_, "model-d");
    EXPECT_TRUE(param->useToolsCall);
    EXPECT_TRUE(param->userInputId.has_value());
    EXPECT_EQ(param->userInputId.value(), "UIN-1");
    EXPECT_EQ(param->textInput, "hello");
    EXPECT_EQ(param->maxNewTokens, 7);
    EXPECT_FALSE(param->streamMode);

    // output names
    ASSERT_EQ(param->outputNames.size(), 2u);
    EXPECT_EQ(param->outputNames[0], "logits");

    // metrics copied into handler
    EXPECT_EQ(handler->metrics.firstTokenCost, 12u);
    EXPECT_EQ(handler->metrics.lastTokenCost, 34u);
    ASSERT_EQ(handler->metrics.decodeTime.size(), 2u);
    EXPECT_EQ(handler->metrics.decodeTime[1], 5u);
    ASSERT_EQ(handler->metrics.batchSize.size(), 2u);
    EXPECT_EQ(handler->metrics.batchSize[1], 2);
    ASSERT_EQ(handler->metrics.queueWaitTime.size(), 2u);
    EXPECT_EQ(handler->metrics.queueWaitTime[0], 7);
    EXPECT_EQ(handler->metrics.callbackIndex, 99u);

    EXPECT_EQ(param->batchSize, 3u);
}

TEST_F(DecodeReqHandlerTestF, UpdateInferRequest_SetsIdsAndTables)
{
    TB.params.set_pinstanceid(7);
    grpc->SetDecodeParams(TB.params);

    auto param = std::make_shared<InferParam>();
    RequestSPtr request = std::make_shared<Request>("");
    std::vector<int64_t> reqTokens, respTokens;
    ASSERT_TRUE(handler->GetContextJsonBody(param, request, reqTokens, respTokens));

    RequestSPtr reqObj = std::make_shared<Request>();
    handler->UpdateInferRequest(/*reqTokens*/ reqTokens, /*ori*/ 0, reqObj);
    handler->UpdateInferParam(reqObj, param);

    EXPECT_EQ(reqObj->reqType, InferReqType::REQ_DECODE);
    EXPECT_EQ(reqObj->pInstanceId, 7u);
    ASSERT_EQ(reqObj->srcBlockTable.size(), 3u);
    EXPECT_EQ(reqObj->srcBlockTable[2], 3u);
    ASSERT_EQ(reqObj->dpInstanceIds.size(), 2u);
    EXPECT_EQ(reqObj->dpInstanceIds[0], 71001u);
    ASSERT_TRUE(handler->inferParam_ != nullptr);
    EXPECT_EQ(handler->inferParam_.get(), param.get());
}

TEST_F(DecodeReqHandlerTestF, ProcessGrpcReq_WhenEngineFails_SetsErrorAndFlags)
{
    // Mock InferInstance::Process to return an error
    MOCKER_CPP(&InferInstance::Process, Status (*)(RequestSPtr))
        .stubs()
        .will(MOCKCPP_NS::returnValue(InvalidArg()));

    RequestSPtr reqObj = std::make_shared<Request>();
    DecodeRequestResponse resp;
    handler->ProcessGrpcReq(reqObj, /*inputId*/ "ignored", resp);

    EXPECT_TRUE(resp.isvaliddecodeparameters());
    EXPECT_EQ(resp.errormessage(), "Failed forward in for infer engine.");

    EXPECT_FALSE(handler->metrics.isPrefill);
    EXPECT_LE(handler->metrics.startingTime, handler->metrics.lastTokenTime);
}

TEST_F(DecodeReqHandlerTestF, SendResponse_And_Stream_NoCrash)
{
    auto handler = std::make_shared<SingleLLMDecodeReqHandler>(httpCtx, dres, grpc);

    handler->SendResponse(/*code*/ 200, "");
    handler->SendResponse(/*code*/ 200, "{\"ok\":1}");
    handler->SendResponseStream(/*isEnd*/ false, "");
    handler->SendResponseStream(/*isEnd*/ true, "{\"delta\":1}");
}

TEST_F(DecodeReqHandlerTestF, Callback_RecomputePath_NoCrash)
{
    auto handler = std::make_shared<SingleLLMDecodeReqHandler>(httpCtx, dres, grpc);

    // Install a simple recompute body builder to hit ResponseReCompute
    handler->SetDMIReComputeBuildCallBack(
        [](const std::vector<BestNTokens> &) { return std::string("RECOMPUTE_BODY"); });

    RequestSPtr reqObj = std::make_shared<Request>();
    handler->SetBackManagerCallBack(reqObj);

    auto resp = std::make_shared<Response>(RequestIdNew{});
    resp->transferStatusFlag = TransferStatusType::RECOMPUTED_TRIGGERED;
    resp->reqId = "RID-DEC-001";
    ASSERT_TRUE(static_cast<bool>(reqObj->serverResponseCallback_));
    reqObj->serverResponseCallback_(resp);

    EXPECT_TRUE(handler->isFinish_.load());
}

TEST_F(DecodeReqHandlerTestF, ServerResponseCallback_NullConditionsCheck)
{
    auto handler = std::make_shared<SingleLLMDecodeReqHandler>(httpCtx, dres, grpc);
    RequestSPtr request = std::make_shared<Request>();
    handler->SetBackManagerCallBack(request);
    ResponseSPtr nullResponse = nullptr;
    handler->constructOneResponseCallBack_ = func;
    EXPECT_TRUE(static_cast<bool>(handler->constructOneResponseCallBack_));
    EXPECT_NO_THROW(request->serverResponseCallback_(nullResponse));
    EXPECT_TRUE(static_cast<bool>(handler->constructOneResponseCallBack_));
}

TEST_F(DecodeReqHandlerTestF, ServerResponseCallback_PullKVCompleteHandling)
{
    auto handler = std::make_shared<SingleLLMDecodeReqHandler>(httpCtx, dres, grpc);
    RequestSPtr request = std::make_shared<Request>();
    handler->SetBackManagerCallBack(request);
    ResponseSPtr response = std::make_shared<Response>(RequestIdNew{});
    response->transferStatusFlag = TransferStatusType::PULL_KV_COMPLETE;
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
    handler->constructOneResponseCallBack_ = func;
    request->serverResponseCallback_(response);
    EXPECT_TRUE(static_cast<bool>(handler->constructOneResponseCallBack_));
}

TEST_F(DecodeReqHandlerTestF, ServerResponseCallback_RecomputeTriggeredHandling)
{
    RequestSPtr request = std::make_shared<Request>();
    handler->SetBackManagerCallBack(request);
    ResponseSPtr response = std::make_shared<Response>(RequestIdNew{});
    response->transferStatusFlag = TransferStatusType::RECOMPUTED_TRIGGERED;
    handler->dmiReCompBuildMethod_ = nullptr;
    handler->constructOneResponseCallBack_ = func;
    request->serverResponseCallback_(response);
    EXPECT_TRUE(handler->isFinish_.load());
    EXPECT_FALSE(static_cast<bool>(handler->constructOneResponseCallBack_));
    std::function<std::string(const std::vector<BestNTokens>&)> rcb =
    [](const std::vector<BestNTokens> &) {
        return std::string("ok");
    };
    handler->dmiReCompBuildMethod_ = rcb;
    handler->constructOneResponseCallBack_ = func;
    handler->isFinish_.store(false);
    request->serverResponseCallback_(response);
    EXPECT_TRUE(handler->isFinish_.load());
    EXPECT_FALSE(static_cast<bool>(handler->constructOneResponseCallBack_));
}

TEST_F(DecodeReqHandlerTestF, ServerResponseCallback_NormalTokenGeneration)
{
    RequestSPtr request = std::make_shared<Request>();
    handler->SetBackManagerCallBack(request);
    ResponseSPtr response = std::make_shared<Response>(RequestIdNew{});
    response->transferStatusFlag = TransferStatusType::PREFILL_COMPLETE;
    MOCKER_CPP(&SingleLLMDecodeReqHandler::ParseTokensFromResponse,
              bool (*)(const ResponseSPtr,
              std::vector<BestNTokens>)).stubs().will(returnValue(false));
    request->serverResponseCallback_(response);
    EXPECT_TRUE(handler->isFinish_.load());
    handler->isFinish_.store(false);
    MOCKCPP_NS::GlobalMockObject::verify();
    MOCKER_CPP(&SingleLLMDecodeReqHandler::ParseTokensFromResponse,
              bool (*)(const ResponseSPtr,
              std::vector<BestNTokens>)).stubs().will(returnValue(true));
    MOCKER_CPP(&SingleLLMReqHandlerBase::SetMetricParams,
              void (*)(const ResponseSPtr)).stubs();
    handler->streamMode_ = false;
    response->isEos = false;
    request->serverResponseCallback_(response);
    
    response->isEos = true;
    handler->streamMode_ = true;
    handler->constructOneResponseCallBack_ = func;
    request->serverResponseCallback_(response);
    EXPECT_FALSE(static_cast<bool>(handler->constructOneResponseCallBack_));
    EXPECT_TRUE(handler->isFinish_.load());
    handler->isFinish_.store(false);
    handler->constructOneResponseCallBack_ = func;
    request->serverResponseCallback_(response);
    EXPECT_FALSE(static_cast<bool>(handler->constructOneResponseCallBack_));
    EXPECT_TRUE(handler->isFinish_.load());
}

TEST_F(DecodeReqHandlerTestF, Test_HeaderFile)
{
    RequestSPtr request = std::make_shared<Request>();
    EXPECT_NO_THROW(handler->Process(request, "1", 5));
}