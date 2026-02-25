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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "edge_cloud_policy.h"
#include "latency_predictor.h"


#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

using namespace mindie_llm;

static SequenceGroupSPtr createDummySeqGroup(std::string &requestId, int promptLength, int blockSize)
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

class EdgeCloudPolicyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        edgeCloudPolicy_ = std::make_shared<EdgeCloudPolicy>(1);
    }
    void TearDown() override
    {
        MOCKCPP_NS::GlobalMockObject::reset();
    }
    std::shared_ptr<EdgeCloudPolicy> edgeCloudPolicy_;
};

TEST_F(EdgeCloudPolicyTest, TestApply)
{
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    ConcurrentDeque<SequenceGroupSPtr> running_;
    ConcurrentDeque<SequenceGroupSPtr> swapped_;
    int blockSize = 4;
    std::string reqId = "0";
    auto seqGroup = createDummySeqGroup(reqId, /*promptLength=*/blockSize, blockSize);

    // 计数00，waiting和running都空，下P
    auto type = edgeCloudPolicy_->Apply(waiting_, running_, swapped_);
    EXPECT_EQ(type, PDPriorityType::PREFILL_FIRST);

    // 计数00，waiting为空，running非空，下D
    running_.PushBack(seqGroup);
    type = edgeCloudPolicy_->Apply(waiting_, running_, swapped_);
    EXPECT_EQ(type, PDPriorityType::DECODE_FIRST);

    // 计数00，waiting不为空，下P
    waiting_.PushBack(seqGroup);
    type = edgeCloudPolicy_->Apply(waiting_, running_, swapped_);
    EXPECT_EQ(type, PDPriorityType::PREFILL_FIRST);

    // 计数01,下P
    ForwardMode forwardMode = ForwardMode::DECODE;
    edgeCloudPolicy_->LayerwiseAddBatchCnt(forwardMode);
    type = edgeCloudPolicy_->Apply(waiting_, running_, swapped_);
    EXPECT_EQ(type, PDPriorityType::PREFILL_FIRST);

    // 计数10,下P
    forwardMode = ForwardMode::PREFILL;
    edgeCloudPolicy_->LayerwiseAddBatchCnt(forwardMode);
    forwardMode = ForwardMode::DECODE;
    edgeCloudPolicy_->LayerwiseSubBatchCnt(forwardMode);
    type = edgeCloudPolicy_->Apply(waiting_, running_, swapped_);
    EXPECT_EQ(type, PDPriorityType::DECODE_FIRST);

    // 计数11,进入异常分支
    forwardMode = ForwardMode::DECODE;
    edgeCloudPolicy_->LayerwiseAddBatchCnt(forwardMode);
    EXPECT_THROW(edgeCloudPolicy_->Apply(waiting_, running_, swapped_), std::runtime_error);
}