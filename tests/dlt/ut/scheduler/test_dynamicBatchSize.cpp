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
#define private public
#define protected public
#include "policy/dynamic_batch_size.h"
#include "self_attn_block_manager.h"

using namespace mindie_llm;
class DynamicBatchSizeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        schedulerConfig_ = std::make_shared<SchedulerConfig>();
        predictor_ = std::make_shared<LatencyPredictor>();
        predictor_->decodeLatency_.AddDataPoint(40);
        BlockManagerConfig blockConf = {schedulerConfig_->cacheBlockSize,
                                        schedulerConfig_->cpuBlockNum,
                                        schedulerConfig_->npuBlockNum,
                                        0,
                                        0,
                                        schedulerConfig_->enablePrefixCache,
                                        schedulerConfig_->spSize * schedulerConfig_->cpSize};
        blockManager_ = std::static_pointer_cast<SelfAttnBlockManager>(BlockManagerFactory::CreateBlockSpaceManager(
            BlockManagerType::SELFATTNBLOCKMANAGER, std::move(blockConf), 0));
        blockManager_->seqId2BlockTable_[0] = BlockTable();
        blockManager_->seqId2BlockTable_[0].blockIds_.push_back(1);

        schedulerConfig_->maxBatchSize = 100;
        schedulerConfig_->decodeExpectedTime = 100.0;
        schedulerConfig_->dynamicBatchSizeEnable = true;
        schedulerConfig_->stageSelectPolicy = 2;
        schedulerConfig_->maxPrefillTokens = 1000;

        dynamicBatchSize_ = std::make_unique<DynamicBatchSize>(schedulerConfig_, predictor_, blockManager_);

        ScheduledSequenceGroupSPtr newElement = std::make_shared<ScheduledSequenceGroup>();
        auto seq = std::make_shared<Sequence>(0, 1);
        seqVec.push_back(seq);
        newElement->seqGroup_ = std::make_shared<SequenceGroup>(reqId, seqVec);
        schedulerOut_.scheduledSeqGroups_.push_back(newElement);
    }

    std::shared_ptr<SchedulerConfig> schedulerConfig_;
    std::shared_ptr<LatencyPredictor> predictor_;
    std::shared_ptr<SelfAttnBlockManager> blockManager_;
    std::unique_ptr<DynamicBatchSize> dynamicBatchSize_;
    SchedulerOutputs schedulerOut_{};
    std::vector<SequenceSPtr> seqVec;
    RequestId reqId{"0"};
};

TEST_F(DynamicBatchSizeTest, ApplyDynamicBatchSize)
{
    size_t waitingSize = 1;
    size_t runningSize = 1;
    size_t swappedSize = 1;
    EXPECT_EQ(dynamicBatchSize_->batchSizeUpperBound_, schedulerConfig_->maxBatchSize);
    EXPECT_EQ(dynamicBatchSize_->batchSizeLowerBound_, 5);
    schedulerOut_.forwardMode_ = ForwardMode::DECODE;
    dynamicBatchSize_->batchTrackerWindowSize_ = 1;

    dynamicBatchSize_->ApplyDynamicBatchSize(Role::PnD, schedulerOut_, waitingSize, runningSize, swappedSize);
    EXPECT_EQ(dynamicBatchSize_->stage_, 1); // 进入了SetMinimalBatchSize
    EXPECT_EQ(schedulerConfig_->maxPrefillBatchSize, 0);

    dynamicBatchSize_->ApplyDynamicBatchSize(Role::PnD, schedulerOut_, waitingSize, runningSize, swappedSize);
    EXPECT_EQ(dynamicBatchSize_->stage_, 0); // 进入了BinarySearchBatchSize
    EXPECT_EQ(schedulerConfig_->maxBatchSize, 52);
    EXPECT_EQ(schedulerConfig_->maxPrefillBatchSize, 50);
    EXPECT_EQ(dynamicBatchSize_->batchSizeLower_, 5);
    EXPECT_EQ(dynamicBatchSize_->batchSizeUpper_, 100);
}

TEST_F(DynamicBatchSizeTest, AdjustBatchSize)
{
    size_t previousStage = 1;
    size_t previousDecodeBatchSize = 50;
    size_t waitingSize = 10;
    size_t runningSize = 5;
    size_t swappedSize = 3;
    dynamicBatchSize_->AdjustBatchSize(previousStage, previousDecodeBatchSize, waitingSize, runningSize, swappedSize);
    EXPECT_EQ(dynamicBatchSize_->decodeBatchSizeQueue_->queue_.size(), 1);
    EXPECT_EQ(dynamicBatchSize_->decodeBatchSizeQueue_->queue_.front(), previousDecodeBatchSize);
}

TEST_F(DynamicBatchSizeTest, BinarySearchBatchSize)
{
    uint64_t currentDecodeRequestNum = 5;
    predictor_->decodeLatency_.AddDataPoint(200); // 超过预期延迟
    double avgDecodeLatency = 200.0;
    uint64_t avgBatchSize = 0;
    dynamicBatchSize_->BinarySearchBatchSize(currentDecodeRequestNum, avgDecodeLatency, avgBatchSize);
    // 验证批大小是否被调整
    EXPECT_GT(dynamicBatchSize_->batchSizeUpper_, dynamicBatchSize_->batchSizeLower_);
}

TEST_F(DynamicBatchSizeTest, SetMinimalBatchSize)
{
    uint64_t currentDecodeRequestNum = 5;
    double avgDecodeLatency = 200.0;
    dynamicBatchSize_->SetMinimalBatchSize(currentDecodeRequestNum, avgDecodeLatency);
    EXPECT_EQ(schedulerConfig_->maxBatchSize, currentDecodeRequestNum);
    EXPECT_EQ(schedulerConfig_->maxPrefillBatchSize, 0);
}

TEST_F(DynamicBatchSizeTest, RecordPredictorMetrics)
{
    SchedulingBudget budget(schedulerConfig_->maxPrefillTokens, schedulerConfig_->maxBatchSize, schedulerConfig_);
    budget.numBatchedTokens_ = 0;
    dynamicBatchSize_->RecordPredictorMetrics(schedulerOut_, budget);
    EXPECT_EQ(predictor_->batchStatsMap_.Get(1).value()->forwardMode, schedulerOut_.forwardMode_);
    EXPECT_EQ(predictor_->batchStatsMap_.Get(1).value()->numBatchedTokens, budget.numBatchedTokens_);
    EXPECT_EQ(predictor_->batchStatsMap_.Get(1).value()->kvCacheBlockNum, dynamicBatchSize_->GetScheduledOutBlockNum(schedulerOut_));
}





