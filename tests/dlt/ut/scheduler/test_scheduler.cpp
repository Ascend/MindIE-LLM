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
#include <unordered_set>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
#define private public
#define protected public
#include "scheduler.h"
#include "config_info.h"
#include "sequence_group.h"
#include "sequence.h"
#include "sampling.h"
#include "block_manager_interface.h"
#include "self_attn_block_manager.h"
#include "concurrent_deque.h"

using namespace mindie_llm;

//------------------------------- Utility function -------------------------------//

SequenceGroupSPtr createDummyPrompt(std::string &requestId, int promptLength, int blockSize)
{
    std::vector<TokenId> dummyInputs;
    for (int i = 0; i < promptLength; ++i) {
        dummyInputs.push_back(i);
    }

    SequenceSPtr seq = std::make_shared<Sequence>(std::stol(requestId), blockSize, dummyInputs);
    auto samplingParams = std::make_shared<SamplingParams>();
    samplingParams->bestOf = 1;

    std::vector<SequenceSPtr> seqs{seq};
    SequenceGroupSPtr seqGroup = std::make_shared<SequenceGroup>(requestId, seqs, samplingParams);
    seqGroup->seqId2ParallelSeqGroup_.Insert(seqGroup->firstSeq->seqId_, seqGroup);
    seqGroup->parentSeqId_ = seqGroup->firstSeq->seqId_;
    return seqGroup;
}

SequenceGroupSPtr createDummyBeamSearchSeqGroup(SequenceId seqId, int promptLength, int blockSize, int beamWidth = 2)
{
    std::vector<TokenId> dummyInputs;
    for (int i = 0; i < promptLength; ++i) {
        dummyInputs.push_back(i);
    }

    SequenceSPtr seq = std::make_shared<Sequence>(seqId, blockSize, dummyInputs);
    auto samplingParams = std::make_shared<SamplingParams>();
    samplingParams->useBeamsearch = true;
    samplingParams->n = beamWidth;
    samplingParams->enableParallelSampling = true;

    std::vector<SequenceSPtr> seqs{seq};
    RequestId requestId = std::to_string(seqId);
    SequenceGroupSPtr seqGroup = std::make_shared<SequenceGroup>(requestId, seqs, samplingParams);
    seqGroup->seqId2ParallelSeqGroup_.Insert(seqGroup->firstSeq->seqId_, seqGroup);
    seqGroup->parentSeqId_ = seqGroup->firstSeq->seqId_;

    return seqGroup;
}

// 为beam search创建额外的序列组（模拟beam expansion后的状态），功能与ModelExecOutputHandler::CreateNewSequenceGroup类似
void expandBeamSearchSeqGroup(SequenceGroupSPtr seqGroup, int beamWidth = 2)
{
    for (int i = 1; i < beamWidth; ++i) {
        SequenceId newSeqId = seqGroup->firstSeq->seqId_ * 100 + i; // 生成唯一的序列ID
        SequenceSPtr newSeq = std::make_shared<Sequence>(newSeqId, seqGroup->firstSeq->blockSize_,
                                                         seqGroup->firstSeq->data_.promptTokenIds);
        newSeq->data_.outputTokenIds.push_back(0); // 模拟生成的token
        std::vector<SequenceSPtr> newSeqs{newSeq};
        RequestId requestId = std::to_string(newSeqId);
        SequenceGroupSPtr newSeqGroup = std::make_shared<SequenceGroup>(requestId, newSeqs);
        newSeqGroup->isNewSeqGroup_ = true;
        newSeqGroup->needUpdate_ = true;
        newSeqGroup->parentSeqId_ = seqGroup->firstSeq->seqId_;
        newSeqGroup->UpdateNumComputedTokens(newSeqGroup->firstSeq->GetLen());
        seqGroup->seqId2ParallelSeqGroup_.Insert(newSeqId, newSeqGroup);
        LiveInferContext::GetInstance(0)->AddIntoSeqRootMap(newSeqId, seqGroup);
    }
}

std::vector<SequenceGroupSPtr> getSeqGroupsFromSchedulerOutputs(const SchedulerOutputs &out)
{
    std::vector<SequenceGroupSPtr> seqGroups;
    for (auto &scheduledSG : out.scheduledSeqGroups_) {
        seqGroups.push_back(scheduledSG->seqGroup_);
    }
    return seqGroups;
}

SchedulerConfigSPtr createDefaultSchedulerConfig(size_t blockSize = 4)
{
    auto config = std::make_shared<SchedulerConfig>();
    config->policyType = 0;
    config->maxSeqLen = 200;
    config->maxPrefillTokens = 200;
    config->maxPrefillBatchSize = 4;
    config->maxBatchSize = 4;
    config->cacheBlockSize = blockSize;
    config->cpuBlockNum = 300;
    config->npuBlockNum = 300;
    config->spSize = 1;

    return config;
}

//---------------------------- Google Test --------------------------------//

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void InitScheduler(std::shared_ptr<SchedulerConfig> schedulerConfig)
    {
        schedulerConfig_ = schedulerConfig;
        auto predictor = std::make_shared<LatencyPredictor>();
        scheduler_ = std::make_shared<Scheduler>(schedulerConfig_, predictor, Role::PnD);
        LiveInferContext::GetInstance(0)->reqId2SeqGroupMap_ = std::unordered_map<RequestId, SequenceGroupSPtr>();
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_ = std::unordered_map<SequenceId, SequenceGroupSPtr>();
        ASSERT_TRUE(scheduler_ != nullptr);
    }

    std::shared_ptr<SchedulerConfig> schedulerConfig_;
    std::shared_ptr<Scheduler> scheduler_;
};

//-------------------------------- Test ---------------------------------------//

TEST_F(SchedulerTest, AddSeqGroup)
{
    int blockSize = 4;
    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    int numSeqGroups = 4;
    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
        EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), static_cast<size_t>(i + 1));
    }
}

TEST_F(SchedulerTest, AddSeqGroupSpecialSeqIdDirectToRunning)
{
    int blockSize = 4;
    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    // 测试1：特殊seqId直接进入running队列
    {
        std::string specialReqId = "18446744073709550";
        std::vector<TokenId> dummyInputs{0, 1, 2, 3};
        SequenceSPtr specialSeq = std::make_shared<Sequence>(9223372036854774L, blockSize, dummyInputs);
        auto samplingParams = std::make_shared<SamplingParams>();
        samplingParams->bestOf = 1;
        std::vector<SequenceSPtr> seqs{specialSeq};
        SequenceGroupSPtr specialSeqGroup = std::make_shared<SequenceGroup>(specialReqId, seqs, samplingParams);

        size_t waitingBefore = scheduler_->waiting_.Size();
        size_t runningBefore = scheduler_->running_.Size();

        scheduler_->AddSeqGroup(specialSeqGroup);

        // 验证特殊seqId进入running队列
        EXPECT_EQ(scheduler_->running_.Size(), runningBefore + 1);
        EXPECT_EQ(scheduler_->waiting_.Size(), waitingBefore);  // waiting队列不变
    }

    // 测试2：普通seqId进入waiting队列
    {
        std::string normalReqId = "normal_request";
        std::vector<TokenId> dummyInputs{0, 1, 2, 3};
        SequenceSPtr normalSeq = std::make_shared<Sequence>(12345L, blockSize, dummyInputs);
        auto samplingParams = std::make_shared<SamplingParams>();
        samplingParams->bestOf = 1;
        std::vector<SequenceSPtr> seqs{normalSeq};
        SequenceGroupSPtr normalSeqGroup = std::make_shared<SequenceGroup>(normalReqId, seqs, samplingParams);

        size_t waitingBefore = scheduler_->waiting_.Size();
        size_t runningBefore = scheduler_->running_.Size();

        scheduler_->AddSeqGroup(normalSeqGroup);

        // 验证普通seqId进入waiting队列
        EXPECT_EQ(scheduler_->waiting_.Size(), waitingBefore + 1);
        EXPECT_EQ(scheduler_->running_.Size(), runningBefore);  // running队列不变
    }

    // 测试3：多个特殊seqId和普通seqId的混合
    {
        std::string specialReqId2 = "18446744073709550";
        std::vector<TokenId> dummyInputs{0, 1, 2, 3};
        SequenceSPtr specialSeq2 = std::make_shared<Sequence>(9223372036854774L, blockSize, dummyInputs);
        auto samplingParams = std::make_shared<SamplingParams>();
        samplingParams->bestOf = 1;
        std::vector<SequenceSPtr> seqs{specialSeq2};
        SequenceGroupSPtr specialSeqGroup2 = std::make_shared<SequenceGroup>(specialReqId2, seqs, samplingParams);

        size_t waitingBefore = scheduler_->waiting_.Size();
        size_t runningBefore = scheduler_->running_.Size();

        scheduler_->AddSeqGroup(specialSeqGroup2);

        // 验证第二个特殊seqId也进入running队列
        EXPECT_EQ(scheduler_->running_.Size(), runningBefore + 1);
        EXPECT_EQ(scheduler_->waiting_.Size(), waitingBefore);
    }
}

TEST_F(SchedulerTest, ReplacePlaceHolderWithTokenTest)
{
    size_t blockSize = 4;
    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    std::string reqId1 = "1";
    auto seqGroup = createDummyPrompt(reqId1, /*promptLength=*/blockSize, /*blockSize=*/blockSize);
    auto seq = seqGroup->GetSequences().front();

    // Single Placeholder Replacement.
    seq->data_.outputTokenIds = {10, 20, PLACEHOLDER_TOKEN};
    scheduler_->blockManager_->Allocate(seqGroup);
    scheduler_->predictedTokensBySeqId_[seq->seqId_] = {100};
    seq->data_.stage_ = SequenceStage::DECODE;
    scheduler_->ReplacePlaceHolderWithToken(seqGroup);
    auto tokens = seq->data_.outputTokenIds;
    EXPECT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], 10);
    EXPECT_EQ(tokens[1], 20);
    EXPECT_EQ(tokens[2], 100);
    EXPECT_EQ(scheduler_->predictedTokensBySeqId_.count(seq->seqId_), 0u);

    // Double Placeholder Replacement.
    seq->data_.outputTokenIds = {10, PLACEHOLDER_TOKEN, PLACEHOLDER_TOKEN};
    scheduler_->predictedTokensBySeqId_[seq->seqId_] = {200, 300};
    scheduler_->ReplacePlaceHolderWithToken(seqGroup);
    tokens = seq->data_.outputTokenIds;
    EXPECT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], 10);
    EXPECT_EQ(tokens[1], 200);
    EXPECT_EQ(tokens[2], 300);
    EXPECT_EQ(scheduler_->predictedTokensBySeqId_.count(seq->seqId_), 0u);

    // No Replacement When Predicted Tokens Are Absent.
    seq->data_.outputTokenIds = {10, PLACEHOLDER_TOKEN};
    scheduler_->ReplacePlaceHolderWithToken(seqGroup);
    tokens = seq->data_.outputTokenIds;
    EXPECT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 10);
    EXPECT_EQ(tokens[1], PLACEHOLDER_TOKEN);
}

TEST_F(SchedulerTest, FetchReqIdsAndTokensTest)
{
    auto config = createDefaultSchedulerConfig();
    InitScheduler(config);

    ConcurrentDeque<SequenceId> finishedSeqIds;
    for (int i = 0; i < 5; ++i) {
        SequenceId seqId = i;
        finishedSeqIds.PushBack(seqId);
    }
    scheduler_->FetchFinishedSeqIds(finishedSeqIds);
    for (int i = 0; i < 5; ++i) {
        SequenceId seqId = i;
        EXPECT_EQ(scheduler_->finishedSeqIds_.count(seqId), 1u);
    }

    InitScheduler(config);
    ConcurrentDeque<SequenceId> exceptionSeqIds;
    for (int i = 0; i < 5; ++i) {
        SequenceId seqId = i + 100;
        exceptionSeqIds.PushBack(seqId);
    }
    scheduler_->FetchExceptionSeqIds(exceptionSeqIds);
    for (int i = 0; i < 5; ++i) {
        SequenceId seqId = i + 100;
        EXPECT_EQ(scheduler_->exceptionSeqIds_.count(seqId), 1u);
    }

    InitScheduler(config);
    ConcurrentDeque<RequestId> abortedReqIds;
    for (int i = 0; i < 5; ++i) {
        std::string reqId = std::to_string(i);
        abortedReqIds.PushBack(reqId);
    }
    scheduler_->FetchAbortedReqIds(abortedReqIds);
    for (int i = 0; i < 5; ++i) {
        std::string reqId = std::to_string(i);
        EXPECT_EQ(scheduler_->abortedReqIds_.count(reqId), 1u);
    }

    InitScheduler(config);
    LiveInferContextSPtr contextSPtr = LiveInferContext::GetInstance(0);
    ConcurrentDeque<std::pair<SequenceId, TokenId>> seqIdToOutputTokenQueue;
    for (int i = 0; i < 5; ++i) {
        std::string reqId = std::to_string(i);
        SequenceGroupSPtr seqGroup = createDummyPrompt(reqId, 2, 4);
        contextSPtr->Add(seqGroup);

        std::pair<SequenceId, TokenId> pairA(i, i + 100);
        seqIdToOutputTokenQueue.PushBack(pairA);
        std::pair<SequenceId, TokenId> pairB(i, i + 200);
        seqIdToOutputTokenQueue.PushBack(pairB);
    }

    scheduler_->FetchSeqGeneratedTokens(seqIdToOutputTokenQueue);
    for (int i = 0; i < 5; ++i) {
        const auto &tokens = scheduler_->predictedTokensBySeqId_[i];
        EXPECT_EQ(tokens.size(), 2u);
        EXPECT_EQ(tokens[0], i + 100);
        EXPECT_EQ(tokens[1], i + 200);
    }
}

TEST_F(SchedulerTest, ScheduleSimple)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    // Add 4 sequence groups
    std::vector<SequenceGroupSPtr> allSeqGroups;
    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        allSeqGroups.push_back(seqGroup);
        seqGroup.reset();
    }

    // 1) Schedule prefill
    auto [seqGroupMeta, out] = scheduler_->Schedule();
    {
        EXPECT_EQ(out.scheduledSeqGroups_.size(), allSeqGroups.size());

        int expectedTokens = blockSize * numSeqGroups; // 4 groups x 4 tokens each
        EXPECT_EQ(out.numBatchedTokens_, static_cast<size_t>(expectedTokens));
    }
    scheduler_->PrepareNextSchedule(out.scheduledSeqGroups_);

    // 2) Schedule decode
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    {
        EXPECT_EQ(out2.scheduledSeqGroups_.size(), allSeqGroups.size());

        // Now we expect 4 tokens total (one for each group)
        EXPECT_EQ(out2.numBatchedTokens_, static_cast<size_t>(numSeqGroups));
        EXPECT_EQ(seqGroupMeta2.metaList.size(), static_cast<size_t>(numSeqGroups));
    }
}

TEST_F(SchedulerTest, ScheduleSimpleWithBeamSearch)
{
    int blockSize = 4;
    int beamWidth = 2;

    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    // Add a beam search sequence group with beam width 2
    auto beamSeqGroup = createDummyBeamSearchSeqGroup(100, blockSize, blockSize, beamWidth);
    scheduler_->AddSeqGroup(beamSeqGroup);

    // 1) Schedule prefill
    auto [seqGroupMeta, out] = scheduler_->Schedule();
    {
        // 只有 root seqgroup 被调度
        EXPECT_EQ(out.scheduledSeqGroups_.size(), 1);

        // prefill 不拷贝 block
        EXPECT_EQ(out.blocksToCopy_.size(), 0);
    }
    scheduler_->PrepareNextSchedule(out.scheduledSeqGroups_);

    expandBeamSearchSeqGroup(beamSeqGroup, beamWidth);

    // 2) Schedule first decode
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    {
        // 仍然只有 root seqgroup 被调度
        EXPECT_EQ(out2.scheduledSeqGroups_.size(), 1);

        // 第一次 decode 需要拷贝 block
        EXPECT_EQ(out2.blocksToCopy_.size(), 1);
    }
}

TEST_F(SchedulerTest, PrefillPrioritized)
{
    int blockSize = 4;
    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    // 1) Add small prompt (length=1)
    std::string reqId1 = "1";
    auto seqGroup1 = createDummyPrompt(reqId1, /*promptLength=*/1, blockSize);
    scheduler_->AddSeqGroup(seqGroup1);
    seqGroup1.reset();

    auto [metas, out] = scheduler_->Schedule();
    auto scheduledGroups = getSeqGroupsFromSchedulerOutputs(out);
    EXPECT_EQ(scheduledGroups.size(), 1u);
    EXPECT_EQ(scheduledGroups[0]->requestId, reqId1) << "Should schedule only seqGroup1 (the small prefill).";

    // 2) Add large prompt (length=30)
    std::string reqId2 = "2";
    auto seqGroup2 = createDummyPrompt(reqId2, /*promptLength=*/30, blockSize);
    scheduler_->AddSeqGroup(seqGroup2);
    seqGroup2.reset();

    auto [metas2, out2] = scheduler_->Schedule();
    auto scheduledGroups2 = getSeqGroupsFromSchedulerOutputs(out2);
    EXPECT_EQ(scheduledGroups2.size(), 1u);
    EXPECT_EQ(scheduledGroups2[0]->requestId, reqId2) << "The new big prefill request2 should be scheduled alone.";
}

TEST_F(SchedulerTest, SchedulePreemptAbort)
{
    int blockSize = 4;
    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 2;
    config->npuBlockNum = 2;
    InitScheduler(config);

    std::string reqIdA = "1";
    auto seqGroupA = createDummyPrompt(reqIdA, /*promptLength=*/blockSize, blockSize);
    std::string reqIdB = "2";
    auto seqGroupB = createDummyPrompt(reqIdB, /*promptLength=*/blockSize, blockSize);
    scheduler_->AddSeqGroup(seqGroupA);
    seqGroupA.reset();
    scheduler_->AddSeqGroup(seqGroupB);
    seqGroupB.reset();

    // Schedule once => both prompts should be batched.
    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    {
        auto scheduledGrps = getSeqGroupsFromSchedulerOutputs(out1);
        ASSERT_EQ(scheduledGrps.size(), 2u);
        EXPECT_EQ(scheduledGrps[0]->requestId, reqIdA);
        EXPECT_EQ(scheduledGrps[1]->requestId, reqIdB);

        EXPECT_EQ(out1.numBatchedTokens_, 8u); // 4 + 4 = 8 tokens total

        EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 2u);
    }
    scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);

    // Next schedule => we want only seqGroupA to decode; seqGroupB gets preempted.
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    {
        auto scheduledGrps = getSeqGroupsFromSchedulerOutputs(out2);
        ASSERT_EQ(scheduledGrps.size(), 1u);
        EXPECT_EQ(scheduledGrps[0]->requestId, reqIdA);

        EXPECT_EQ(out2.numBatchedTokens_, 1u); // Only 1 token is being decoded
        EXPECT_EQ(out2.numPreempted_, 1u);
        EXPECT_EQ(scheduler_->GetUnFinishedSeqGroups(), 2u); // Still 2 unfinished
    }
    scheduler_->PrepareNextSchedule(out2.scheduledSeqGroups_);
}

TEST_F(SchedulerTest, SchedulePreemptWithBeamSearch)
{
    int blockSize = 4;
    int beamWidth = 2;
    auto config = createDefaultSchedulerConfig(blockSize);
    config->npuBlockNum = 4;
    InitScheduler(config);

    SequenceId seqIdA = 100;
    // seqGroupA 初始占一个block
    auto seqGroupA = createDummyBeamSearchSeqGroup(seqIdA, /*promptLength=*/blockSize, blockSize);
    SequenceId seqIdB = 101;
    // seqGroupB 初始占两个block
    auto seqGroupB = createDummyBeamSearchSeqGroup(seqIdB, /*promptLength=*/blockSize + 1, blockSize);
    scheduler_->AddSeqGroup(seqGroupA);
    scheduler_->AddSeqGroup(seqGroupB);

    // First schedule
    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    {
        // seqGroupA 和 seqGroupB 都被调度
        EXPECT_EQ(out1.scheduledSeqGroups_.size(), 2);

        auto scheduledGrps = getSeqGroupsFromSchedulerOutputs(out1);
        EXPECT_EQ(scheduledGrps[0]->firstSeq->seqId_, seqIdA);
        EXPECT_EQ(scheduledGrps[1]->firstSeq->seqId_, seqIdB);
    }
    scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);

    // 模拟生成token，此时 seqGroupA 和 seqGroupB 都需要 2 + 2 = 4 个 block
    seqGroupA->firstSeq->data_.outputTokenIds.push_back(0);
    expandBeamSearchSeqGroup(seqGroupA, beamWidth);
    seqGroupB->firstSeq->data_.outputTokenIds.push_back(0);
    expandBeamSearchSeqGroup(seqGroupB, beamWidth);

    // Next schedule
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    {
        // seqGroupA 被调度， seqGroupB 被抢占
        EXPECT_EQ(out2.scheduledSeqGroups_.size(), 1);
        EXPECT_EQ(out2.numPreempted_, 1);

        auto scheduledGrps = getSeqGroupsFromSchedulerOutputs(out2);
        EXPECT_EQ(scheduledGrps[0]->firstSeq->seqId_, seqIdA);

        // beamsearch 被抢占的请求会被直接 abort
        EXPECT_EQ(scheduler_->GetAbortedParallelSeqGroups().size(), 1);
    }
}

TEST_F(SchedulerTest, ScheduleEmptyWithBeamSearch)
{
    int blockSize = 4;
    int beamWidth = 2;
    auto config = createDefaultSchedulerConfig(blockSize);
    config->npuBlockNum = 3;
    InitScheduler(config);

    SequenceId seqIdA = 100;
    // 初始占一个block
    auto beamSeqGroup = createDummyBeamSearchSeqGroup(seqIdA, /*promptLength=*/blockSize, blockSize);
    scheduler_->AddSeqGroup(beamSeqGroup);

    // First schedule
    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);

    // 此时需要 2 + 2 = 4 个 block
    expandBeamSearchSeqGroup(beamSeqGroup, beamWidth);

    // Next schedule
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    {
        EXPECT_TRUE(out2.IsEmpty());
        EXPECT_EQ(out2.numPreempted_, 1);
    }
}

TEST_F(SchedulerTest, ScheduleEmptyWithBeamSearchWhenSingleSeqCanSchedule)
{
    int blockSize = 4;
    int beamWidth = 3;
    auto config = createDefaultSchedulerConfig(blockSize);
    config->npuBlockNum = 4;
    InitScheduler(config);

    SequenceId seqIdA = 100;
    // 初始占一个block
    auto beamSeqGroup = createDummyBeamSearchSeqGroup(seqIdA, /*promptLength=*/blockSize - 1, blockSize);
    scheduler_->AddSeqGroup(beamSeqGroup);

    // First schedule
    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);

    // 此时需要 1 * 3 = 3 个 block
    expandBeamSearchSeqGroup(beamSeqGroup, beamWidth);

    // Next schedule
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    scheduler_->PrepareNextSchedule(out2.scheduledSeqGroups_);

    // Final schedule, 此时需要 2 * 3 = 6 个 block
    auto [seqGroupMeta3, out3] = scheduler_->Schedule();
    {
        EXPECT_TRUE(out3.IsEmpty());
        EXPECT_EQ(out3.numPreempted_, 1);
    }
}

TEST_F(SchedulerTest, MaxSeqsTest)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->maxPrefillBatchSize = 2;
    InitScheduler(config);

    std::vector<SequenceGroupSPtr> allSeqGroups;
    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        allSeqGroups.push_back(seqGroup);
    }

    scheduler_->AddSeqGroup(allSeqGroups[0]);

    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    auto scheduled1 = getSeqGroupsFromSchedulerOutputs(out1);
    ASSERT_EQ(scheduled1.size(), 1u);
    EXPECT_EQ(scheduled1[0]->requestId, allSeqGroups[0]->requestId);

    scheduler_->AddSeqGroup(allSeqGroups[1]);
    scheduler_->AddSeqGroup(allSeqGroups[2]);
    scheduler_->AddSeqGroup(allSeqGroups[3]);

    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    auto scheduled2 = getSeqGroupsFromSchedulerOutputs(out2);
    ASSERT_EQ(scheduled2.size(), 2u); // maxBatchSize is 2
    EXPECT_EQ(scheduled2[0]->requestId, allSeqGroups[1]->requestId);
    EXPECT_EQ(scheduled2[1]->requestId, allSeqGroups[2]->requestId);

    // Abort SeqGroup 1 and SeqGroup 2
    std::unordered_set<std::string> abortIds;
    abortIds.insert(allSeqGroups[1]->requestId);
    abortIds.insert(allSeqGroups[2]->requestId);

    auto [seqGroupMeta3, out3] = scheduler_->Schedule();
    auto scheduled3 = getSeqGroupsFromSchedulerOutputs(out3);
    ASSERT_EQ(scheduled3.size(), 1u);
    EXPECT_EQ(scheduled3[0]->requestId, allSeqGroups[3]->requestId);
}

TEST_F(SchedulerTest, LongPromptTest)
{
    int blockSize = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->maxSeqLen = 30;
    InitScheduler(config);

    std::string reqId = "0";
    auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/60, blockSize);
    scheduler_->AddSeqGroup(seqGroup);
    seqGroup.reset();
    auto [seqGroupMeta, out] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 0u);
}

TEST_F(SchedulerTest, ScheduleTokenBudget)
{
    int blockSize = 4;
    {
        // ----- Scenario 1: Token Budget = 0 (nothing should be scheduled) -----
        auto config = createDefaultSchedulerConfig(blockSize);
        config->maxSeqLen = 0; // Token budget is 0
        config->maxPrefillTokens = 0;
        InitScheduler(config);

        for (int i = 0; i < 2; ++i) {
            std::string reqId = std::to_string(i);
            auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/60, blockSize);
            scheduler_->AddSeqGroup(seqGroup);
        }

        auto [seqGroupMeta, out] = scheduler_->Schedule();
        EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 0u);
        EXPECT_EQ(out.ignoredSeqGroups_.size(), 2u);
        EXPECT_EQ(out.numBatchedTokens_, 0u);
    }
    {
        // ----- Scenario 2: Token Budget = 60 (only one group should be scheduled) -----
        auto config = createDefaultSchedulerConfig(blockSize);
        config->maxSeqLen = 60; // Token budget is 60
        config->maxPrefillTokens = 60;
        InitScheduler(config);

        for (int i = 0; i < 2; ++i) {
            std::string reqId = std::to_string(i);
            auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/60, blockSize);
            scheduler_->AddSeqGroup(seqGroup);
        }

        auto [seqGroupMeta, out] = scheduler_->Schedule();
        EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 1u);
        EXPECT_EQ(out.ignoredSeqGroups_.size(), 0u);
        EXPECT_EQ(out.numBatchedTokens_, 60u);
    }
}

TEST_F(SchedulerTest, NoBlockManagerCapacity)
{
    int blockSize = 4;
    int numSeqGroups = 3;
    // ----- Scenario 1: AllocStatus::LATER -----------
    {
        auto config = createDefaultSchedulerConfig(blockSize);
        config->cpuBlockNum = 30;
        config->npuBlockNum = 30;
        InitScheduler(config);

        for (int i = 0; i < numSeqGroups; ++i) {
            std::string reqId = std::to_string(i);
            auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/60, blockSize);
            scheduler_->AddSeqGroup(seqGroup);
            seqGroup.reset();
        }

        auto [seqGroupMeta, out] = scheduler_->Schedule();
        EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 2u);
        EXPECT_EQ(out.numBatchedTokens_, 120u);
        scheduler_->PrepareNextSchedule(out.scheduledSeqGroups_);

        auto [seqGroupMeta2, out2] = scheduler_->Schedule();
        EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out2).size(), 1u);
        EXPECT_EQ(out2.numBatchedTokens_, 1u);
        scheduler_->PrepareNextSchedule(out2.scheduledSeqGroups_);
    }
    // ----- Scenario 2: AllocStatus::NEVER -----------
    {
        auto config = createDefaultSchedulerConfig(blockSize);
        config->cpuBlockNum = 30;
        config->npuBlockNum = 30;
        InitScheduler(config);

        for (int i = 0; i < numSeqGroups; ++i) {
            std::string reqId = std::to_string(i);
            auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/121, blockSize);
            scheduler_->AddSeqGroup(seqGroup);
            seqGroup.reset();
        }
        // The prompt length exceeds block manager capacity, throw error
        EXPECT_THROW(scheduler_->Schedule(), std::runtime_error);
    }
}

TEST_F(SchedulerTest, SchedulerDelayTest)
{
    int blockSize = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->maxQueueDelayMicroseconds = 10000; // 10ms
    InitScheduler(config);

    std::string reqId1 = "0";
    auto seqGroup1 = createDummyPrompt(reqId1, /*promptLength=*/blockSize, blockSize);
    scheduler_->AddSeqGroup(seqGroup1);
    seqGroup1.reset();

    // Simulate a delay of 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    auto scheduled1 = getSeqGroupsFromSchedulerOutputs(out1);
    EXPECT_EQ(scheduled1.size(), 1u);
    EXPECT_EQ(scheduled1[0]->requestId, reqId1);
    scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);

    std::string reqId2 = "1";
    auto seqGroup2 = createDummyPrompt(reqId2, /*promptLength=*/blockSize, blockSize);
    scheduler_->AddSeqGroup(seqGroup2);
    seqGroup2.reset();

    // second prompt should not be scheduled
    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    auto scheduled2 = getSeqGroupsFromSchedulerOutputs(out2);
    EXPECT_EQ(scheduled2.size(), 1u);
    EXPECT_EQ(scheduled2[0]->requestId, reqId1);
    scheduler_->PrepareNextSchedule(out2.scheduledSeqGroups_);

    // Simulate a delay of 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto [seqGroupMeta3, out3] = scheduler_->Schedule();
    auto scheduled3 = getSeqGroupsFromSchedulerOutputs(out3);
    EXPECT_EQ(scheduled3.size(), 1u);
    EXPECT_EQ(scheduled3[0]->requestId, reqId2);
}

TEST_F(SchedulerTest, WaitingTimeReachedTest)
{
    int blockSize = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->maxQueueDelayMicroseconds = 10000; // 10ms
    InitScheduler(config);

    std::string reqId = "0";
    auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
    scheduler_->AddSeqGroup(seqGroup);
    seqGroup.reset();

    // sleep for 50ms
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_FALSE(scheduler_->ShouldImmediatePrefill());

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_TRUE(scheduler_->ShouldImmediatePrefill());
}

// This test needs to be adjusted according to MAX_ASYNC_SCHEDULE_TIMES.
TEST_F(SchedulerTest, MtpPlaceHolderTest)
{
    int blockSize = 4;
    {
        // ----- Scenario 1: Token Budget = 0 (nothing should be scheduled) -----
        auto config = createDefaultSchedulerConfig(blockSize);
        config->speculationGamma = 1;
        InitScheduler(config);

        std::string reqId = "1";
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/3, blockSize);
        scheduler_->AddSeqGroup(seqGroup);

        auto [seqGroupMeta, out] = scheduler_->Schedule();
        scheduler_->PrepareNextSchedule(out.scheduledSeqGroups_);

        std::vector<TokenId> &outPutTokenIds = seqGroup->firstSeq->data_.outputTokenIds;
        auto rit = std::find_if_not(outPutTokenIds.rbegin(), outPutTokenIds.rend(),
                                    [](auto token) { return token == PLACEHOLDER_TOKEN; });
        size_t placeholderCount = std::distance(outPutTokenIds.rbegin(), rit);
        EXPECT_EQ(placeholderCount, 2); // 首轮根据mtp，所以添加2个placeholder

        // 模拟response返回，mtp没有命中
        seqGroup->firstSeq->data_.outputTokenIds.clear();
        seqGroup->firstSeq->data_.outputTokenIds.push_back(222);
        seqGroup->firstSeq->data_.outputTokenIds.push_back(PLACEHOLDER_TOKEN);

        auto [seqGroupMeta1, out1] = scheduler_->Schedule();
        scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);
        rit = std::find_if_not(outPutTokenIds.rbegin(), outPutTokenIds.rend(),
                               [](auto token) { return token == PLACEHOLDER_TOKEN; });
        placeholderCount = std::distance(outPutTokenIds.rbegin(), rit);
        EXPECT_EQ(placeholderCount, 3);
    }
}

TEST_F(SchedulerTest, PDSeperationSimpleScheduleInP)
{
    int blockSize = 4;
    int numSeqGroups = 2;

    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    scheduler_->SetRole(Role::P);

    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
    }

    auto [transferSeqGroupMeta, kvTransferOut] = scheduler_->ScheduleTransfer();
    EXPECT_EQ(transferSeqGroupMeta.metaList.size(), 0u);
    EXPECT_TRUE(kvTransferOut.IsEmpty());

    auto [seqGroupMeta, out] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 2u);
    EXPECT_EQ(out.numBatchedTokens_, 8u);
    // Check if the sequence groups are in the transferring map
    EXPECT_EQ(scheduler_->transferringMap_.Count(0), 1u);
    EXPECT_EQ(scheduler_->transferringMap_.Count(1), 1u);
}

TEST_F(SchedulerTest, PDSeperationSchedulePullKVInD)
{
    int blockSize = 4;
    int numSeqGroups = 2;

    auto config = createDefaultSchedulerConfig(blockSize);
    InitScheduler(config);

    scheduler_->SetRole(Role::D);

    std::vector<SequenceGroupSPtr> allSeqGroups;
    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        seqGroup->isDecode_ = true;
        scheduler_->AddSeqGroup(seqGroup);
        allSeqGroups.push_back(seqGroup);
        seqGroup.reset();
    }

    auto [seqGroupMeta, out] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 0u);
    EXPECT_EQ(out.numBatchedTokens_, 0u);

    auto [transferSeqGroupMeta, kvTransferOut] = scheduler_->ScheduleTransfer();
    EXPECT_EQ(transferSeqGroupMeta.metaList.size(), 2u);
    EXPECT_EQ(kvTransferOut.pullSeqGroups.size(), 2u);
    EXPECT_EQ(scheduler_->transferringMap_.Count(0), 1u);
    EXPECT_EQ(scheduler_->transferringMap_.Count(1), 1u);
    scheduler_->PrepareNextSchedule(kvTransferOut.pullSeqGroups);

    ConcurrentDeque<RequestId> pulledRequestIds;
    for (const auto &seqGroup : allSeqGroups) {
        pulledRequestIds.PushBack(seqGroup->requestId);
    }
    scheduler_->KVPulledReqEnterRunningQueue(pulledRequestIds);
    EXPECT_EQ(scheduler_->running_.Size(), 2u);

    auto [transferSeqGroupMeta2, kvTransferOut2] = scheduler_->ScheduleTransfer();
    EXPECT_EQ(transferSeqGroupMeta2.metaList.size(), 0u);
    EXPECT_EQ(kvTransferOut2.pullSeqGroups.size(), 0u);

    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out2).size(), 2u);
    EXPECT_EQ(out2.numBatchedTokens_, 2u);
    EXPECT_EQ(scheduler_->transferringMap_.Count(0), 0u);
    EXPECT_EQ(scheduler_->transferringMap_.Count(1), 0u);
}

TEST_F(SchedulerTest, PDSeperationNoEnoughBlockInP)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1; // Setting CPU block count to 1; has no effect on prefill phase in process P
    config->npuBlockNum = 3; // Limit NPU block allocation to a maximum of 3
    InitScheduler(config);

    scheduler_->SetRole(Role::P);

    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
    }

    auto [seqGroupMeta1, out1] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out1).size(), 3u);
    EXPECT_EQ(out1.numBatchedTokens_, 12u);
    scheduler_->PrepareNextSchedule(out1.scheduledSeqGroups_);

    auto [seqGroupMeta2, out2] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out2).size(), 0u);
    EXPECT_EQ(out2.numBatchedTokens_, 0u);
    EXPECT_EQ(scheduler_->waiting_.Size(), 1u);
}

TEST_F(SchedulerTest, PDSeperationNoEnoughBlockInD)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1;
    config->npuBlockNum = 3;
    InitScheduler(config);

    scheduler_->SetRole(Role::D);

    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        seqGroup->isDecode_ = true;
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
    }

    auto [transferSeqGroupMeta, kvTransferOut] = scheduler_->ScheduleTransfer();
    EXPECT_EQ(transferSeqGroupMeta.metaList.size(), 2u);
    EXPECT_EQ(kvTransferOut.pullSeqGroups.size(), 2u);
    scheduler_->PrepareNextSchedule(kvTransferOut.pullSeqGroups);

    auto [seqGroupMeta, out] = scheduler_->Schedule();
    EXPECT_EQ(getSeqGroupsFromSchedulerOutputs(out).size(), 0u);
    EXPECT_EQ(out.numBatchedTokens_, 0u);

    // Test the situation where freeTokenNum <= schedulerConfig_->maxBatchSize
    config->maxBatchSize = 13;
    InitScheduler(config);
    scheduler_->SetRole(Role::D);
    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
    }

    auto [transferSeqGroupMeta2, kvTransferOut2] = scheduler_->ScheduleTransfer();
    EXPECT_EQ(transferSeqGroupMeta2.metaList.size(), 0u);
    EXPECT_EQ(kvTransferOut2.pullSeqGroups.size(), 0u);
}

TEST_F(SchedulerTest, PDFlexSetPrefillPercentage)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1;
    config->npuBlockNum = 3;
    auto predictor = std::make_shared<LatencyPredictor>();
    scheduler_ = std::make_shared<Scheduler>(config, predictor, Role::FlexP);

    uint32_t prefillPercentage = 50;
    scheduler_->SetPrefillPercentage(prefillPercentage);
}

TEST_F(SchedulerTest, PDFlexSwitchRole)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1;
    config->npuBlockNum = 3;
    auto predictor = std::make_shared<LatencyPredictor>();
    scheduler_ = std::make_shared<Scheduler>(config, predictor, Role::FlexP);

    auto role = scheduler_->SwitchRole();
    EXPECT_EQ(role, Role::FlexP);
}

TEST_F(SchedulerTest, GetStagePolicy)
{
    int blockSize = 4;
    int numSeqGroups = 4;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1;
    config->npuBlockNum = 3;
    auto predictor = std::make_shared<LatencyPredictor>();
    scheduler_ = std::make_shared<Scheduler>(config, predictor, Role::FlexP);

    auto policy = scheduler_->GetStagePolicy();
    EXPECT_NE(policy, nullptr);
    EXPECT_NE(dynamic_cast<TimeDivisionPolicy *>(policy.get()), nullptr);
}

TEST_F(SchedulerTest, PDDequeueForFlex)
{
    int blockSize = 4;
    int numSeqGroups = 1;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1;
    config->npuBlockNum = 3;
    auto predictor = std::make_shared<LatencyPredictor>();
    scheduler_ = std::make_shared<Scheduler>(config, predictor, Role::FlexP);
    LiveInferContext::GetInstance(0)->reqId2SeqGroupMap_ = std::unordered_map<RequestId, SequenceGroupSPtr>();
    LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_ = std::unordered_map<SequenceId, SequenceGroupSPtr>();

    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
    }

    std::deque<SequenceGroupSPtr> queue_;
    size_t maxNum = 1;
    auto actualNum = scheduler_->DequeueForFlex(scheduler_->waiting_, queue_, Role::FlexP, maxNum);
    EXPECT_EQ(actualNum, 1);
}

TEST_F(SchedulerTest, PDPrepCandidatesForFlex)
{
    int blockSize = 4;
    int numSeqGroups = 1;

    auto config = createDefaultSchedulerConfig(blockSize);
    config->cpuBlockNum = 1;
    config->npuBlockNum = 3;
    auto predictor = std::make_shared<LatencyPredictor>();
    scheduler_ = std::make_shared<Scheduler>(config, predictor, Role::FlexP);
    LiveInferContext::GetInstance(0)->reqId2SeqGroupMap_ = std::unordered_map<RequestId, SequenceGroupSPtr>();
    LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_ = std::unordered_map<SequenceId, SequenceGroupSPtr>();

    for (int i = 0; i < numSeqGroups; ++i) {
        std::string reqId = std::to_string(i);
        auto seqGroup = createDummyPrompt(reqId, /*promptLength=*/blockSize, blockSize);
        scheduler_->AddSeqGroup(seqGroup);
        seqGroup.reset();
    }

    PDPriorityType pdPriorityType = PDPriorityType::PREFILL_FIRST;
    size_t batchSize = config->maxPrefillBatchSize;
    SchedulingBudget budget(config->maxPrefillTokens, batchSize, config);
    ISeqGroupCollectionSPtr data = scheduler_->PrepCandidatesForFlex(pdPriorityType, budget);
}
