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
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#define private public
#define protected public
#include "llm_engine.h"
#include "model_exec_output_handler.h"
#include "scheduler.h"
#include "../crash_handler.h"
#include "request.h"
using namespace mindie_llm;

// ------------------------------- Utility function -------------------------------//

SchedulerConfigSPtr createDefaultSchedulerConfig()
{
    auto config = std::make_shared<SchedulerConfig>();
    config->policyType = 0;
    config->maxSeqLen = 100;
    config->maxPrefillTokens = 100;
    config->maxPrefillBatchSize = 4;
    config->maxBatchSize = 4;
    config->maxQueueDelayMicroseconds = 5000; // 5ms
    config->cacheBlockSize = 4;
    config->cpuBlockNum = 300;
    config->npuBlockNum = 300;
    config->spSize = 1;
    config->tpSize = 1;
    return config;
}

RequestSPtr createInferRequest(int promptLength, std::string id = "1")
{
    auto request = std::make_shared<Request>(RequestIdNew(id));

    // Set maximum output length
    request->maxOutputLen = 1024;

    // Set prompt tokens
    std::vector<int64_t> promptTokens;
    for (int i = 0; i < promptLength; ++i) {
        promptTokens.push_back(i);
    }
    request->input_ids = promptTokens;
    return request;
}

ModelBatchResultSPtr CreateModelBatchResult(const std::vector<SequenceId> &seqIds, const std::vector<int> &tokenIds,
                                            const std::vector<int> &finishReasons)
{
    // build response
    model_execute_data::ExecuteModelResponse response;
    for (size_t i = 0; i < seqIds.size(); ++i) {
        model_execute_data::CompletionSequenceGroupOutput *outputs = response.add_outputs();
        model_execute_data::SequenceOutput *sample = outputs->add_samples();
        sample->set_seq_id(seqIds[i]);
        sample->set_parent_seq_id(seqIds[i]);
        sample->set_truncation_index(100);
        sample->add_output_token(tokenIds[i]);
        sample->add_logprob(0.9);
        sample->set_finish_reason(finishReasons[i]);
        sample->set_num_parallel_tokens(1);
        sample->set_num_speculative_tokens(1);
    }

    return std::make_shared<model_execute_data::ExecuteModelResponse>(response);
}

PullKVResponseSPtr CreatePullKVResponse(const std::vector<RequestId> &reqIds,
                                        const std::vector<model_execute_data::PDErrorCode> &errorCodes)
{
    model_execute_data::PullKVResponse response;
    for (size_t i = 0; i < reqIds.size(); ++i) {
        model_execute_data::PullKVResponse_PullKVResult *result = response.add_pull_kv_results();
        result->set_request_id(reqIds[i]);
        result->set_pd_error_code(errorCodes[i]);
    }
    return std::make_shared<model_execute_data::PullKVResponse>(response);
}

// ------------------------ Mock Executor ------------------------ //

class MockModelExecutor : public IExecutor {
public:
    void ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) override {}
    
    bool ExecutorInstanceInit(std::map<std::string, std::string> &config, bool isMultiNodesInfer,
                              size_t dpIdx = 0) override
    {
        return true;
    }
    ThinkingConfig GetThinkingConfig() const override
    {
        ThinkingConfig conf;
        return conf;
    }
    bool AsyncExecuteModel(ExecuteModelRequestPtr &modelExecRequest,
                           std::function<void(ModelBatchResultSPtr)> callback = nullptr) override
    {
        return true; // Always return true
    }
    bool AsyncTGCleanup(TGCleanupRequestPtr &TGCleanupRequest) override
    {
        return true; // Always return true
    }
    bool ExecutorParseConfigAndInitGRPC(std::map<std::string, std::string> &configFromManager, bool isMultiNodesInfer,
                                        size_t rankIdx) override
    {
        return true;
    }
    bool MasterAndSlaveModelInit(const std::map<std::string, std::string> &pdInfo) override { return true; }
    bool SetupPDLink(model_execute_data::PDLinkRequest &pdLinkRequest) override { return true; }
    bool ExecuteKVTransfer(PullKVRequestPtr &pullKVRequest,
                           std::function<void(PullKVResponseSPtr)> callback = nullptr) override
    {
        return true;
    }

    bool ExecutorInstanceFinalize() override { return true; }
    uint32_t GetCpuBlockNum() const override { return 1; }
    uint32_t GetNpuBlockNum() const override { return 1; }
    uint32_t GetLwdCloudNpuBlockNum() const override { return 1; }
    uint32_t GetMaxPositionEmbeddings() const override { return 4096; }
    model_execute_data::PDLinkResponse GetPDLinkResponse() const override
    {
        return model_execute_data::PDLinkResponse();
    }
    bool ExecutLoraRequest(LoraOperationRequest &loraOperationRequest) override
    {
        return true;
    }

    bool AsyncEOSCleanup(TGCleanupRequestPtr &TGCleanupRequest) override
    {
        return true;
    }

    model_execute_data::LoraOperationResponse GetLoraOperationResponse() const override
    {
        return model_execute_data::LoraOperationResponse();
    }
};

// ------------------------ Test Fixture ------------------------ //

class LlmEngineTest : public ::testing::Test {
protected:
    void SetUp() override {}

    static void SetUpTestSuite()
    {
        mindie_llm::test::InitCrashHandler();
    }

    void InitLlmEngine(SchedulerConfig schedulerConfig, IExecutorSPtr executor = nullptr, Role pdRole = Role::PnD)
    {
        LiveInferContext::GetInstance(0)->reqId2SeqGroupMap_ = std::unordered_map<RequestId, SequenceGroupSPtr>();
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_ = std::unordered_map<SequenceId, SequenceGroupSPtr>();
        executor_ = executor;
        std::vector<IExecutorSPtr> executors;
        executors.push_back(executor);
        engine_ = std::make_shared<LlmEngine>(schedulerConfig, executors, [](ResponseSPtr response) {}, pdRole);
        scheduler_ = std::shared_ptr<Scheduler>(dynamic_cast<Scheduler *>(engine_->enginePerDPs_[0]->scheduler.get()),
                                                [](Scheduler *) {});
        engine_->StartEngineThread();
        ASSERT_TRUE(engine_ != nullptr);
        ASSERT_TRUE(scheduler_ != nullptr);
        ASSERT_TRUE(engine_->enginePerDPs_[0]->schedulerThread.joinable());
    }

    void TearDown() override
    {
        engine_->Stop();
        if (engine_->enginePerDPs_[0]->schedulerThread.joinable()) {
            engine_->enginePerDPs_[0]->schedulerThread.join();
        }
    }

    void CreateRequest(std::string &reqId, SequenceId &seqId, int promptLength = 10, Role role = Role::P)
    {
        auto request = createInferRequest(promptLength, reqId);
        engine_->AddRequest(request);

        seqId = LiveInferContext::GetInstance(0)->reqId2SeqGroupMap_.at(reqId)->seqs_.at(0)->seqId_;
    }

    std::shared_ptr<Scheduler> scheduler_;
    std::shared_ptr<LlmEngine> engine_;
    IExecutorSPtr executor_;
};

// ---------------------------- Tests --------------------------------//

TEST_F(LlmEngineTest, IgnoreRequestAfterStop)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    engine_->Stop();

    auto request = createInferRequest(/*promptLength=*/10);
    engine_->AddRequest(request);
    EXPECT_EQ(engine_->enginePerDPs_[0]->scheduler->GetUnFinishedSeqGroups(), 0u);
}

TEST_F(LlmEngineTest, AsyncBatchNumTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    auto request1 = createInferRequest(/*promptLength=*/10, "1");
    engine_->AddRequest(request1);

    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_EQ(engine_->enginePerDPs_[0]->modelExecOutputHandler->GetAsyncBatchNum(),
              static_cast<size_t>(MAX_ASYNC_SCHEDULE_TIMES));
    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 1u);

    auto request2 = createInferRequest(/*promptLength=*/10, "2");
    engine_->AddRequest(request2);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(engine_->enginePerDPs_[0]->modelExecOutputHandler->GetAsyncBatchNum(),
              static_cast<size_t>(MAX_ASYNC_SCHEDULE_TIMES));
    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 2u);
}

TEST_F(LlmEngineTest, AddResponseTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    std::string reqId = "reqId";
    SequenceId seqId = 100;
    CreateRequest(reqId, seqId);
    ModelBatchResultSPtr result = CreateModelBatchResult({seqId}, {100}, {0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));
}

TEST_F(LlmEngineTest, UpdatePlaceholderWithNextTokenTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    std::string reqId = "reqId";
    SequenceId seqId = 100;
    CreateRequest(reqId, seqId);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ModelBatchResultSPtr result = CreateModelBatchResult({seqId}, {100}, {0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    SequenceGroupSPtr seqGroup = scheduler_->running_.Front();
    EXPECT_EQ(seqGroup->seqs_[0]->data_.outputTokenIds[0], 100);
}

TEST_F(LlmEngineTest, ContinueResponseTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    std::string reqId = "reqId";
    SequenceId seqId = 100;
    CreateRequest(reqId, seqId);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ModelBatchResultSPtr result = CreateModelBatchResult({seqId}, {100}, {0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 1u);
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->seqIdToOutputTokenQueue_.Empty());
}

TEST_F(LlmEngineTest, FinishedRequestsTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    std::string reqId = "reqId";
    SequenceId seqId = 100;
    CreateRequest(reqId, seqId);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ModelBatchResultSPtr result = CreateModelBatchResult({seqId}, {100}, {1});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 0u);
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->seqIdToOutputTokenQueue_.Empty());
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->finishedSeqIds_.Empty());
}

TEST_F(LlmEngineTest, ExceptionRequestsTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    std::string reqId = "reqId";
    SequenceId seqId = 100;
    CreateRequest(reqId, seqId);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ModelBatchResultSPtr result = CreateModelBatchResult({seqId}, {100}, {2});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 0u);
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->seqIdToOutputTokenQueue_.Empty());
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->execExceptionSeqIds_.Empty());
}

TEST_F(LlmEngineTest, MultiRequestMultiResponseTest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    int numRequests = 2;
    int numDecodeTokens = 3;

    // Add multiple requests
    std::vector<SequenceId> seqIds;
    for (int i = 0; i < numRequests; ++i) {
        std::string reqId = std::to_string(i);
        SequenceId seqId;
        CreateRequest(reqId, seqId);
        seqIds.push_back(seqId);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Simulate multiple responses
    for (const SequenceId seqId : seqIds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ModelBatchResultSPtr result = CreateModelBatchResult({seqId}, {100}, {0});
        EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));
    }

    for (int token = 1; token < numDecodeTokens; ++token) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        ModelBatchResultSPtr result = CreateModelBatchResult(seqIds, {token + 100, token + 100}, {0, 0});
        EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Check if the tokens are correctly replaced
    SequenceGroupSPtr seqGroup = scheduler_->running_.Front();
    for (int j = 0; j < numDecodeTokens; ++j) {
        EXPECT_EQ(seqGroup->seqs_[0]->data_.outputTokenIds.at(j), j + 100);
    }

    // Simulate finishing and aborting requests)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ModelBatchResultSPtr result = CreateModelBatchResult(seqIds, {200, 200}, {1, 2});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Check all requests are finished
    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 0u);
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->seqIdToOutputTokenQueue_.Empty());
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->finishedSeqIds_.Empty());
    EXPECT_TRUE(engine_->enginePerDPs_[0]->modelExecOutputHandler->execExceptionSeqIds_.Empty());
}

TEST_F(LlmEngineTest, RecomputeAndAbortTest)
{
    int blockSize = 4;
    auto schedulerConfig = createDefaultSchedulerConfig();
    schedulerConfig->cacheBlockSize = blockSize;
    schedulerConfig->cpuBlockNum = 2;
    schedulerConfig->npuBlockNum = 5;
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>());

    std::unordered_set<RequestId> abortedReqIds = {};
    std::vector<SequenceId> seqIds;
    for (int i = 0; i < 2; ++i) {
        std::string reqId = std::to_string(i);
        SequenceId seqId;
        CreateRequest(reqId, seqId, blockSize * 2);
        abortedReqIds.insert(reqId);
        seqIds.push_back(seqId);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    ModelBatchResultSPtr result1 = CreateModelBatchResult({seqIds[0]}, {100}, {0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result1));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ModelBatchResultSPtr result2 = CreateModelBatchResult({seqIds[1]}, {100}, {0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result2));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // The preemption happens, and the scheduler will recompute the second request
    EXPECT_EQ(scheduler_->swapped_.Size(), 0u);
    EXPECT_EQ(scheduler_->waiting_.Size(), 1u);
    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 2u);
    // Simulate an abort for the first and second requests
    engine_->AbortRequests(abortedReqIds);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ModelBatchResultSPtr result3 = CreateModelBatchResult({seqIds[0]}, {101}, {0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result3));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Check all requests are finished
    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 0u);
}

TEST_F(LlmEngineTest, PDSeperationPrefillInP)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>(), Role::P);

    int numRequests = 2;

    // Add requests
    std::unordered_set<RequestId> allReqIds = {};
    std::vector<SequenceId> seqIds;
    for (int i = 0; i < numRequests; ++i) {
        std::string reqId = std::to_string(i);
        SequenceId seqId;
        CreateRequest(reqId, seqId);
        allReqIds.insert(reqId);
        seqIds.push_back(seqId);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Simulate responses
    ModelBatchResultSPtr result = CreateModelBatchResult(seqIds, {100, 100}, {0, 0});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    for (SequenceId seqId : seqIds) {
        EXPECT_EQ(scheduler_->transferringMap_.Count(seqId), 1u);
        EXPECT_TRUE(LiveInferContext::GetInstance(0)->GetSeqGroup(seqId) != nullptr);
    }

    // Abort the requests
    engine_->AbortRequests(allReqIds);
    engine_->enginePerDPs_[0]->modelExecOutputHandler->GetAsyncBatchNum().fetch_sub(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (SequenceId seqId : seqIds) {
        EXPECT_EQ(scheduler_->transferringMap_.Count(seqId), 0u);
        EXPECT_TRUE(LiveInferContext::GetInstance(0)->GetSeqGroup(seqId) == nullptr);
    }
}

TEST_F(LlmEngineTest, PDSeperationPullKVInD)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>(), Role::D);

    int numRequests = 2;

    // Add requests
    std::vector<RequestId> allReqIds = {};
    std::vector<SequenceId> seqIds;
    int promptLength = 10;
    Role role = Role::D;
    for (int i = 0; i < numRequests; ++i) {
        std::string reqId = std::to_string(i);
        SequenceId seqId;
        CreateRequest(reqId, seqId, promptLength, role);
        allReqIds.push_back(reqId);
        seqIds.push_back(seqId);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    for (SequenceId seqId : seqIds) {
        EXPECT_EQ(scheduler_->transferringMap_.Count(seqId), 1u);
        EXPECT_TRUE(LiveInferContext::GetInstance(0)->GetSeqGroup(seqId) != nullptr);
    }

    // Simulate pull KV responses
    PullKVResponseSPtr pullKVResponse = CreatePullKVResponse(
        allReqIds, std::vector<model_execute_data::PDErrorCode>{model_execute_data::PDErrorCode::SUCCESS,
                                                                model_execute_data::PDErrorCode::SUCCESS});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->transferOutputHandler->Entry4Executor(pullKVResponse));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    for (int i = 0; i < numRequests; ++i) {
        EXPECT_EQ(scheduler_->transferringMap_.Count(seqIds[i]), 0u);
        EXPECT_TRUE(LiveInferContext::GetInstance(0)->GetSeqGroup(allReqIds[i]) != nullptr);
    }
    EXPECT_EQ(scheduler_->running_.Size(), 2u);

    // Semulate finishing responses
    ModelBatchResultSPtr result = CreateModelBatchResult(seqIds, {100, 100}, {1, 1});
    EXPECT_NO_THROW(engine_->enginePerDPs_[0]->modelExecOutputHandler->Entry4Executor(result));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 0u);
}

TEST_F(LlmEngineTest, SPAddRequest)
{
    auto schedulerConfig = createDefaultSchedulerConfig();
    schedulerConfig->spSize = 8;
    InitLlmEngine(*schedulerConfig, std::make_shared<MockModelExecutor>(), Role::PnD);

    // Add requests
    auto request = createInferRequest(/*promptLength=*/2, "3");
    engine_->AddRequest(request);

    SequenceGroupSPtr seqGroup = LiveInferContext::GetInstance(0)->GetSeqGroup("3");
    EXPECT_EQ(seqGroup->firstSeq->GetLen(), 2u);
}