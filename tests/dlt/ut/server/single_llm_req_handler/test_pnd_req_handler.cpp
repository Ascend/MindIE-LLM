/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>

#include <mockcpp/mockcpp.hpp>
#define private public
#include "nlohmann/json.hpp"
#include "param_checker.h"
#include "response.h"
#include "single_llm_pnd_req_handler.h"

using namespace mindie_llm;
using namespace prefillAndDecodeCommunication;

#define MOCKER_CPP(api, TT) \
    (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

using ordered_json = nlohmann::ordered_json;

class PndReqHandlerTestF : public ::testing::Test {
   protected:
    httplib::Request req{};
    httplib::Response res{};
    std::shared_ptr<RequestContext> ctx;
    ResponseSPtr response;
    std::function<bool(const ResponseSPtr &response,
                       const std::vector<BestNTokens> &, RespBodyQueue &)>
        func;
    std::shared_ptr<SingleLLMPnDReqHandler> handler;

    void SetUp() override {
        req.method = "POST";
        req.path = "/prefill";
        req.version = "HTTP/1.1";
        ctx = std::make_shared<RequestContext>(req, res);
        MOCKER_CPP(&HttpRestResource::ResponseJsonBody,
                   int (*)(const ReqCtxPtr &, int, const std::string &))
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
        func = [](const ResponseSPtr &response,
                  const std::vector<BestNTokens> &, RespBodyQueue &queue) {
            queue.push("Start");
            queue.push("End");
            return true;
        };
        handler = std::make_unique<SingleLLMPnDReqHandler>(ctx);
    }

    void TearDown() override { GlobalMockObject::verify(); }
};

using AcceptFuncType = bool (*)(const std::string &, bool);

TEST_F(PndReqHandlerTestF, GetContextJsonBody_InvalidJsonParsing) {
    ordered_json body;
    EXPECT_FALSE(handler->GetContextJsonBody(body));
}

TEST_F(PndReqHandlerTestF, GetContextRequestId) {
    std::string requestId;
    EXPECT_FALSE(handler->GetContextRequestId(requestId));
    EXPECT_EQ(requestId, "");
}

TEST_F(PndReqHandlerTestF, GetContextJsonBody) {
    InferParamSPtr inputParam;
    RequestSPtr request = std::make_shared<Request>("");
    std::vector<int64_t> reqTokens;
    std::vector<int64_t> respTokens;
    EXPECT_TRUE(handler->GetContextJsonBody(inputParam, request, reqTokens,
                                            respTokens));
}

TEST_F(PndReqHandlerTestF, UpdateInferParam_NullParam) {
    handler->inferParam_ = std::make_shared<InferParam>();
    handler->UpdateInferParam(nullptr, nullptr);
    EXPECT_EQ(handler->inferParam_, nullptr);
}

static inline Status Ok() { return Status(Error::Code::OK); }
static inline Status InvalidArg() { return Status(Error::Code::INVALID_ARG); }

TEST_F(PndReqHandlerTestF, Process) {
    MOCKER_CPP(&SingleLLMPnDReqHandler::SetBackManagerCallBack,
               void (*)(RequestSPtr))
        .stubs();
    RequestSPtr request = std::make_shared<Request>();
    MOCKER_CPP(&InferInstance::Process, Status(*)(RequestSPtr))
        .stubs()
        .will(returnValue(InvalidArg()));
    MOCKER_CPP(&TokenizerProcessPool::RemoveMultimodalCache,
               void (*)(const uint64_t &))
        .stubs();
    handler->constructOneResponseCallBack_ = func;
    handler->Process(request, "1", 5);
    EXPECT_FALSE(static_cast<bool>(handler->constructOneResponseCallBack_));
    GlobalMockObject::verify();
    MOCKER_CPP(&SingleLLMPnDReqHandler::SetBackManagerCallBack,
               void (*)(RequestSPtr))
        .stubs();
    MOCKER_CPP(&InferInstance::Process, Status(*)(RequestSPtr))
        .stubs()
        .will(returnValue(Ok()));
    MOCKER_CPP(&SingleLLMPnDReqHandler::ProcessNonStreamModeRequest,
               void (*)(const std::string &inputId, const uint64_t &))
        .stubs();
    handler->constructOneResponseCallBack_ = func;
    handler->Process(request, "1", 5);
    EXPECT_TRUE(static_cast<bool>(handler->constructOneResponseCallBack_));
    handler->streamMode_ = true;
    handler->Process(request, "1", 5);
    EXPECT_TRUE(static_cast<bool>(handler->constructOneResponseCallBack_));
}

TEST_F(PndReqHandlerTestF, SetBackManagerCallBack) {
    RequestSPtr request = std::make_shared<Request>();
    request->requestId = "1";
    handler->SetBackManagerCallBack(request);
    auto callback = request->serverResponseCallback_;
    handler->constructOneResponseCallBack_ = func;
    callback(nullptr);
    EXPECT_TRUE(static_cast<bool>(handler->constructOneResponseCallBack_));
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseTokensFromResponse,
               bool (*)(const ResponseSPtr, std::vector<BestNTokens>))
        .stubs()
        .will(returnValue(true));
    response->isEos = true;
    callback(response);
    EXPECT_EQ(handler->inferResponseQueue_.size(), 1);
    GlobalMockObject::verify();
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseTokensFromResponse,
               bool (*)(const ResponseSPtr, std::vector<BestNTokens>))
        .stubs()
        .will(returnValue(false));
    callback(response);
    EXPECT_EQ(handler->inferResponseQueue_.size(), 2);
}

TEST_F(PndReqHandlerTestF, SendResponseStream) {
    EXPECT_NO_THROW(handler->SendResponseStream(true, ""));
    std::shared_ptr<EventDispatcher> eventDispatcher =
        std::make_shared<EventDispatcher>(1, "1", 5);
    MOCKER_CPP(&EventDispatcher::SendEvent,
               void (*)(const std::string, bool, std::string))
        .stubs();
    EXPECT_NO_THROW(handler->SendResponseStream(true, "1"));
}

TEST_F(PndReqHandlerTestF, ProcessNonStreamModeRequest) {
    handler->request_ = std::make_shared<Request>();
    handler->inferParam_ = std::make_shared<InferParam>();
    handler->ProcessNonStreamModeRequest("1", 5);
    EXPECT_TRUE(handler->isFinish_.load());
    handler->isFinish_.store(false);
    handler->inferResponseQueue_.push(response);
    handler->ProcessNonStreamModeRequest("1", 5);
    EXPECT_TRUE(handler->isFinish_.load());
}

TEST_F(PndReqHandlerTestF, ProcessNonStreamBatchResponse) {
    handler->request_ = std::make_shared<Request>();
    handler->inferParam_ = std::make_shared<InferParam>();
    const std::vector<ResponseSPtr> responses = {response};
    response->isEos = true;
    handler->constructOneResponseCallBack_ = func;
    handler->inferResponseQueue_.push(response);
    EXPECT_TRUE(handler->ProcessNonStreamBatchResponse(responses));
    response->isEos = false;
    EXPECT_FALSE(handler->ProcessNonStreamBatchResponse(responses));
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseTokensFromResponse,
               bool (*)(const ResponseSPtr, std::vector<BestNTokens>))
        .stubs()
        .will(returnValue(false));
    EXPECT_TRUE(handler->ProcessNonStreamBatchResponse(responses));
}

TEST_F(PndReqHandlerTestF, StreamResponseFillHttpSink) {
    MOCKER_CPP(&TokenizerProcessPool::RemoveMultimodalCache,
               void (*)(const uint64_t &))
        .stubs();
    RequestSPtr request = std::make_shared<Request>();
    httplib::DataSink sink;
    bool writeCalled = false;
    bool doneCalled = false;
    sink.write = [&](const char *data, size_t size) -> bool {
        std::cout << "write called with size: " << size << std::endl;
        writeCalled = true;
        return true;
    };
    sink.done = [&]() {
        doneCalled = true;
        std::cout << "done called" << std::endl;
    };
    handler->inferParam_ = std::make_shared<InferParam>();
    MOCKER_CPP(&InferInstance::ControlRequest,
               Status(*)(const RequestIdNew, OperationV2))
        .stubs()
        .will(returnValue(InvalidArg()));
    handler->StreamResponseFillHttpSink(
        "1", reinterpret_cast<httplib::DataSink &>(sink), 5);
    EXPECT_TRUE(writeCalled);
    EXPECT_TRUE(doneCalled);
    handler->inferResponseQueue_.push(response);
    writeCalled = false;
    doneCalled = false;
    MOCKER_CPP(&SingleLLMReqHandlerBase::ParseTokensFromResponse,
               bool (*)(const ResponseSPtr &, std::vector<BestNTokens>))
        .stubs()
        .will(returnValue(false));
    handler->StreamResponseFillHttpSink(
        "1", reinterpret_cast<httplib::DataSink &>(sink), 5);
    EXPECT_TRUE(writeCalled);
    EXPECT_TRUE(doneCalled);
}

TEST_F(PndReqHandlerTestF, GetElapsedTimeMillis_StdChrono) {
    auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 等待50ms
    int64_t elapsed = handler->GetElapsedTimeMillis(start);
    EXPECT_GE(elapsed, 50);   // 至少45ms
    EXPECT_LE(elapsed, 150);  // 最多55ms
}

TEST_F(PndReqHandlerTestF, GetElapsedTimeMillis_BoostChrono) {
    auto start = boost::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // 等待50ms
    int64_t elapsed = handler->GetElapsedTimeMillis(start);

    EXPECT_GE(elapsed, 45);  // 至少45ms
    EXPECT_LE(elapsed, 55);  // 最多55ms
}

TEST_F(PndReqHandlerTestF, GetElapsedTimeMillis_UnsupportedType) {
    std::any invalidType = 42;
    EXPECT_THROW(handler->GetElapsedTimeMillis(invalidType),
                 std::runtime_error);
    std::any empty;
    EXPECT_THROW(handler->GetElapsedTimeMillis(empty), std::runtime_error);
}

TEST_F(PndReqHandlerTestF, GetElapsedTimeMillis_ZeroDuration) {
    auto start = std::chrono::steady_clock::now();
    int64_t elapsed = handler->GetElapsedTimeMillis(start);
    EXPECT_GE(elapsed, 0);
    EXPECT_LE(elapsed, 5);  // 最多5ms
}
