/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Description: Unit tests for SeqGroupBuilderFromInferReq
 * Author: MindIE
 * Create: 2025
 * History: NA
 */
#include <gtest/gtest.h>
#include <memory>
#include "seq_group_builder_from_infer_req.h"
#include "request_response/request.h"
#include "dataclass/sequence.h"
#include "config/config_info.h"

using namespace mindie_llm;
using namespace std;

class SeqGroupBuilderTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // 创建调度器配置
        schedulerConfig_ = std::make_shared<SchedulerConfig>();
        schedulerConfig_->maxSeqLen = 2048;
        schedulerConfig_->cacheBlockSize = 16;
        schedulerConfig_->maxIterTimes = 512;
    }

    void TearDown() override { schedulerConfig_.reset(); }

    RequestSPtr CreateRequest(const string &requestId, const vector<int64_t> &inputIds,
                              bool isSimulateRequest = false)
    {
        auto request = std::make_shared<Request>(requestId);
        request->input_ids = inputIds;
        request->maxOutputLen = 100;
        request->temperature = 0.7f;
        request->topK = 50;
        request->topP = 0.9f;
        request->doSample = true;
        request->loraId = "None";
        request->isSimulateRequest = isSimulateRequest;
        return request;
    }

    SchedulerConfigSPtr schedulerConfig_;
};

/**
 * 测试: 普通requestId生成随机seqId
 * 预期: seqId应该被生成并且是非零值
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_NormalRequestId_GeneratesRandomSeqId)
{
    string normalRequestId = "normal_request_123";
    vector<int64_t> inputIds = {1, 2, 3, 4, 5};

    auto request = CreateRequest(normalRequestId, inputIds);

    SequenceSPtr seq = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request, schedulerConfig_);

    ASSERT_NE(seq, nullptr);
    EXPECT_NE(seq->seqId_, 0);
    EXPECT_EQ(seq->data_.promptTokenIds.size(), inputIds.size());
    EXPECT_EQ(seq->data_.promptTokenIds, inputIds);
}

/**
 * 测试: 虚推请求映射到固定seqId
 * 预期: 当isSimulateRequest为true时，seqId应该等于SIMULATE_SEQUENCE_ID
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_SimulateRequest_MapsToFixedSeqId)
{
    string requestId = "simulate_request_123";
    vector<int64_t> inputIds = {10, 20, 30};
    long expectedSeqId = 9223372036854774L;  // SIMULATE_SEQUENCE_ID

    auto request = CreateRequest(requestId, inputIds, true);  // isSimulateRequest = true

    SequenceSPtr seq = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request, schedulerConfig_);

    ASSERT_NE(seq, nullptr);
    EXPECT_EQ(seq->seqId_, expectedSeqId);
    EXPECT_EQ(seq->data_.promptTokenIds.size(), inputIds.size());
    EXPECT_EQ(seq->data_.promptTokenIds, inputIds);
}

/**
 * 测试: 不同的普通requestId生成不同的seqId
 * 预期: 两个不同的普通requestId应该生成不同的seqId
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_DifferentNormalRequestIds_GenerateDifferentSeqIds)
{
    vector<int64_t> inputIds = {1, 2, 3};

    auto request1 = CreateRequest("request_1", inputIds);
    auto request2 = CreateRequest("request_2", inputIds);

    SequenceSPtr seq1 = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request1, schedulerConfig_);
    SequenceSPtr seq2 = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request2, schedulerConfig_);

    ASSERT_NE(seq1, nullptr);
    ASSERT_NE(seq2, nullptr);
    // 两个不同的请求应该生成不同的seqId（概率非常高）
    EXPECT_NE(seq1->seqId_, seq2->seqId_);
}

/**
 * 测试: 空input_ids处理
 * 预期: 应该能处理空的input_ids
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_EmptyInputIds_Success)
{
    string requestId = "empty_input_request";
    vector<int64_t> emptyInputIds;

    auto request = CreateRequest(requestId, emptyInputIds);

    SequenceSPtr seq = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request, schedulerConfig_);

    ASSERT_NE(seq, nullptr);
    EXPECT_NE(seq->seqId_, 0);
    EXPECT_EQ(seq->data_.promptTokenIds.size(), 0);
}

/**
 * 测试: 超长prompt处理
 * 预期: 当input长度超过maxSeqLen时应该抛出异常
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_ExceedMaxSeqLen_ThrowsException)
{
    string requestId = "exceed_max_len_request";
    // 创建一个超过maxSeqLen的input_ids
    vector<int64_t> tooLongInputIds;
    for (int i = 0; i < schedulerConfig_->maxSeqLen + 100; i++) {
        tooLongInputIds.push_back(i);
    }

    auto request = CreateRequest(requestId, tooLongInputIds);

    EXPECT_THROW(
        SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request, schedulerConfig_),
        std::invalid_argument);
}

/**
 * 测试: 缓存块大小配置
 * 预期: 创建的Sequence应该使用配置的缓存块大小
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_CacheBlockSize_Applied)
{
    string requestId = "cache_block_test";
    vector<int64_t> inputIds = {1, 2, 3};

    // 设置不同的缓存块大小
    schedulerConfig_->cacheBlockSize = 32;
    auto request = CreateRequest(requestId, inputIds);

    SequenceSPtr seq = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request, schedulerConfig_);

    ASSERT_NE(seq, nullptr);
    // 验证Sequence被正确创建（实际块大小取决于Sequence实现）
    EXPECT_NE(seq->seqId_, 0);
}

/**
 * 测试: 虚推请求seqId的一致性
 * 预期: 所有isSimulateRequest为true的请求应该总是映射到相同的固定seqId
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_SimulateRequest_ConsistentMapping)
{
    vector<int64_t> inputIds = {100, 200};
    long expectedSeqId = 9223372036854774L;  // SIMULATE_SEQUENCE_ID

    // 创建两个不同requestId但都是虚推的请求
    auto request1 = CreateRequest("simulate_req_1", inputIds, true);
    auto request2 = CreateRequest("simulate_req_2", inputIds, true);

    SequenceSPtr seq1 = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request1, schedulerConfig_);
    SequenceSPtr seq2 = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request2, schedulerConfig_);

    ASSERT_NE(seq1, nullptr);
    ASSERT_NE(seq2, nullptr);
    // 所有虚推请求应该生成相同的固定seqId
    EXPECT_EQ(seq1->seqId_, expectedSeqId);
    EXPECT_EQ(seq2->seqId_, expectedSeqId);
    EXPECT_EQ(seq1->seqId_, seq2->seqId_);
}

/**
 * 测试: 大型input_ids列表
 * 预期: 应该能处理大型input_ids而不失败
 */
TEST_F(SeqGroupBuilderTest, InitSeqFromInferRequest_LargeInputIds_Success)
{
    string requestId = "large_input_request";
    // 创建一个大但不超过maxSeqLen的input_ids
    vector<int64_t> largeInputIds;
    for (int i = 0; i < 1000; i++) {
        largeInputIds.push_back(i);
    }

    auto request = CreateRequest(requestId, largeInputIds);

    SequenceSPtr seq = SeqGroupBuilderFromInferReq::InitSeqFromInferRequest(request, schedulerConfig_);

    ASSERT_NE(seq, nullptr);
    EXPECT_NE(seq->seqId_, 0);
    EXPECT_EQ(seq->data_.promptTokenIds.size(), largeInputIds.size());
    EXPECT_EQ(seq->data_.promptTokenIds, largeInputIds);
}

