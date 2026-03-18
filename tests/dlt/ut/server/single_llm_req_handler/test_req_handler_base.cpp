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
#define private public
#include "single_llm_req_handler_base.h"

using namespace mindie_llm;

class TestLLMReqHandler : public SingleLLMReqHandlerBase {
public:
    explicit TestLLMReqHandler(ReqCtxPtr &ctx) : SingleLLMReqHandlerBase(ctx) {}

    // Expose a way to set inferParam_ for tests
    void SetInferParamForTest(const InferParamSPtr &param) { inferParam_ = param; }
    void SetRequestForTest(const RequestSPtr &request) { request_ = request; }
    void SetIsRecomputeForTest(bool isRecompute) { isRecompute_ = isRecompute; }

    // Required overrides
    void Process(RequestSPtr, const std::string &, const uint64_t &) override {}
    bool GetContextJsonBody(nlohmann::ordered_json &) override { return true; }
    bool GetContextJsonBody(InferParamSPtr, RequestSPtr, std::vector<int64_t> &, std::vector<int64_t> &) override
    {
        return true;
    }
    bool GetContextRequestId(std::string &) override { return true; }
    void UpdateInferRequest(const std::vector<int64_t> &, const int64_t &, RequestSPtr) override {}
    void SetBackManagerCallBack(RequestSPtr) override {}
    void SendResponseInfo(int, const std::string &, bool) override {}
    void SendResponse(int, const std::string &) override {}
    void SendResponseStream(bool, const std::string &) override {}
};

// Helpers
static ResponseSPtr MakeBasicResponse()
{
    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    resp->metrics.batchSize = 2;
    resp->metrics.queueWaitTime = 5;
    ResponseContent content;
    content.seqId = 111;
    content.parentSeqId = 999;
    content.finishReason = InferStatusType::ITERATION_CONTINUE;
    content.truncationIndex = 0;
    content.speculativeTokenNum = 2;
    content.outTokenIds = {101, 102, -1}; // -1 should be filtered out
    content.outLogProbs = {-0.42f};
    content.cumLogProb = 1.25;
    content.topLogProbTokenIds = {201, 202, 203, 204}; // size = speculativeTokenNum * topLogprobs (=2*2)
    content.topLogProbs = {0.10f, 0.20f, 0.30f, 0.40f};
    resp->responseContents.push_back(std::move(content));
    return resp;
}

static RequestSPtr MakeRequest(bool with_logprobs, uint32_t top_logprobs)
{
    RequestSPtr request = std::make_shared<Request>();
    request->logprobs = with_logprobs;
    request->topLogprobs = top_logprobs;
    return request;
}

class SingleLLMReqHandlerBaseTest : public ::testing::Test {
protected:
    ReqCtxPtr ctx;
    void SetUp() override
    {
        httplib::Request req;
        httplib::Response resp;
        ctx = std::make_shared<RequestContext>(req, resp);
    }
};

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Success_WithLogprobsAndToplogprobs)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 4));

    ResponseSPtr resp = MakeBasicResponse();
    std::vector<BestNTokens> out;
    ASSERT_TRUE(handler.ParseTokensFromResponse(resp, out));
    ASSERT_EQ(out.size(), 1u);

    // Seq ids
    EXPECT_EQ(out[0].seqId, 111);
    EXPECT_EQ(out[0].parentSeqId, 999);

    // Tokens: -1 filtered out
    ASSERT_EQ(out[0].tokens.size(), 2u);
    EXPECT_EQ(out[0].tokens[0], 101);
    EXPECT_EQ(out[0].tokens[1], 102);

    // Cum logprob and single-token logprob
    EXPECT_DOUBLE_EQ(out[0].cumLogprobs, 1.25);
    EXPECT_FLOAT_EQ(out[0].logprob, -0.42f);

    // Toplogprobs flattened list (2 speculative tokens * 2 toplogprobs)
    EXPECT_EQ(out[0].logprobsTokens.size(), 4u);
    EXPECT_EQ(out[0].logprobs.size(), 4u);
}

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Fails_WhenResponseEmpty)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 1));

    ResponseSPtr resp = std::make_shared<Response>(RequestIdNew{});
    std::vector<BestNTokens> out;
    EXPECT_FALSE(handler.ParseTokensFromResponse(resp, out));
}

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Fails_WhenSeqIdZero)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 1));

    ResponseSPtr resp = MakeBasicResponse();
    resp->responseContents[0].seqId = 0; // invalid
    std::vector<BestNTokens> out;
    EXPECT_FALSE(handler.ParseTokensFromResponse(resp, out));
}

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Fails_WhenParentSeqIdZero)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 1));

    ResponseSPtr resp = MakeBasicResponse();
    resp->responseContents[0].parentSeqId = 0; // invalid
    std::vector<BestNTokens> out;
    EXPECT_FALSE(handler.ParseTokensFromResponse(resp, out));
}

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Fails_WhenOutTokenIdsEmpty)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 1));

    ResponseSPtr resp = MakeBasicResponse();
    resp->responseContents[0].outTokenIds.clear(); // invalid
    std::vector<BestNTokens> out;
    EXPECT_FALSE(handler.ParseTokensFromResponse(resp, out));
}

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Fails_WhenEndedAndPositiveTruncationId)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 0));

    ResponseSPtr resp = MakeBasicResponse();
    resp->responseContents[0].finishReason = InferStatusType::END_OF_SENTENCE;
    resp->responseContents[0].truncationIndex = 1; // should be <= 0 for ended sequences
    std::vector<BestNTokens> out;
    EXPECT_FALSE(handler.ParseTokensFromResponse(resp, out));
}

TEST_F(SingleLLMReqHandlerBaseTest, ParseTokens_Succeeds_AndSkipsLogprobs_OnIllegalInput)
{
    TestLLMReqHandler handler(ctx);
    handler.SetInferParamForTest(std::make_shared<InferParam>());
    handler.SetRequestForTest(MakeRequest(/*with_logprobs*/ true, /*top_logprobs*/ 1));

    ResponseSPtr resp = MakeBasicResponse();
    resp->responseContents[0].finishReason = InferStatusType::ILLEGAL_INPUT;
    resp->responseContents[0].truncationIndex = 0; // ok
    std::vector<BestNTokens> out;
    EXPECT_TRUE(handler.ParseTokensFromResponse(resp, out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].seqId, 111);
}

TEST_F(SingleLLMReqHandlerBaseTest, SetRecomputeMetrics_AppendsAndCopiesFirstTokenCost)
{
    TestLLMReqHandler handler(ctx);

    // Prepare "this->metrics" initial state
    Metrics &metrics = handler.GetMetrics();
    metrics.decodeTime = {1u, 2u};
    metrics.firstTokenCost = 9u;

    // Prepare recompute metrics to merge in
    Metrics newMetrics;
    newMetrics.decodeTime = {10u, 20u};
    newMetrics.firstTokenCost = 123u;
    handler.SetRecomputeMetrics(newMetrics);

    // DecodeTime appended, firstTokenCost replaced
    ASSERT_EQ(metrics.decodeTime.size(), 4u);
    EXPECT_EQ(metrics.decodeTime[0], 1u);
    EXPECT_EQ(metrics.decodeTime[1], 2u);
    EXPECT_EQ(metrics.decodeTime[2], 10u);
    EXPECT_EQ(metrics.decodeTime[3], 20u);
    EXPECT_EQ(metrics.firstTokenCost, 123u);
}

TEST_F(SingleLLMReqHandlerBaseTest, Metrics_SetMetricParams_PrefillPath_NoPrometheusCalls)
{
    TestLLMReqHandler handler(ctx);

    // Avoid TTFT observe by marking this as DECODE type
    handler.reqType_ = InferReqType::REQ_DECODE;

    // Set metrics to "prefill" state
    Metrics &metrics = handler.GetMetrics();
    metrics.isPrefill = true;
    metrics.startingTime = std::chrono::steady_clock::now();

    ResponseSPtr resp = MakeBasicResponse();
    resp->isEos = true;
    handler.SetMetricParams(resp);

    // firstTokenCost should be set (prefill branch, non-recompute)
    EXPECT_GE(metrics.firstTokenCost, 0u);
    // decodeTime not appended in non-recompute prefill branch
    EXPECT_TRUE(metrics.decodeTime.empty());
}

TEST_F(SingleLLMReqHandlerBaseTest, Metrics_SetMetricParams_DecodePath_AppendsAvgTimes_AndSetsLastCost)
{
    TestLLMReqHandler handler(ctx);

    // Set to decode path
    Metrics &metrics = handler.GetMetrics();
    metrics.isPrefill = false;
    // Make sure decodeTime > 0
    metrics.lastTokenTime = std::chrono::steady_clock::now() - std::chrono::milliseconds(5);

    // Response: outputLen = speculativeTokenNum (e.g., 3)
    ResponseSPtr resp = MakeBasicResponse();
    resp->responseContents.resize(1);
    resp->responseContents[0].speculativeTokenNum = 3;
    resp->isEos = true;
    handler.SetMetricParams(resp);

    // Assert: decode path pushed exactly outputLen entries (all equal to avg)
    ASSERT_EQ(metrics.decodeTime.size(), static_cast<size_t>(3));
    EXPECT_EQ(metrics.decodeTime[0], metrics.decodeTime[1]);
    EXPECT_EQ(metrics.decodeTime[1], metrics.decodeTime[2]);

    // DECODE tag queued
    ASSERT_FALSE(metrics.callbackIndexQue.empty());
    EXPECT_EQ(metrics.callbackIndexQue.front(), DECODE_CALLBACK_METRICS_TAG);

    // lastTokenTime updated; endingTime and lastTokenCost set on EOS
    EXPECT_TRUE(metrics.lastTokenTime.time_since_epoch().count() > 0);
    EXPECT_TRUE(metrics.endingTime.time_since_epoch().count() > 0);
    EXPECT_TRUE(metrics.lastTokenCost > 0u);
}

TEST_F(SingleLLMReqHandlerBaseTest, Setters_Work_And_DoNotCrash)
{
    TestLLMReqHandler handler(ctx);
    // Set stream mode and callbacks to test setters
    handler.SetStreamMode(true);

    ConstructOneResponseCallBack cb = [](const ResponseSPtr &, const std::vector<BestNTokens> &, RespBodyQueue &) {
        return true;
    };
    handler.SetConstructOneResponseCallBack(cb);

    TestLLMReqHandler::DMIReComputeBuildMethod rcb = [](const std::vector<BestNTokens> &) {
        return std::string("ok");
    };
    handler.SetDMIReComputeBuildCallBack(rcb);
}

TEST_F(SingleLLMReqHandlerBaseTest, Test_HeaderFile)
{
    TestLLMReqHandler handler(ctx);
    EXPECT_NO_THROW(handler.GetReqType());
    RequestSPtr request = std::make_shared<Request>();
    prefillAndDecodeCommunication::DecodeRequestResponse resp;
    EXPECT_NO_THROW(handler.ProcessGrpcReq(request, "1", resp));
}