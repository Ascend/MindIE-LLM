/*
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

#ifndef LAYERWISE_FCFS_POLICY_H
#define LAYERWISE_FCFS_POLICY_H
#include "fcfs_policy.h"
#include "math_utils.h"
#include "msServiceProfiler/msServiceProfiler.h"
namespace mindie_llm {
class LayerwiseFcfsPolicy : public FcfsPolicy {
public:
    LayerwiseFcfsPolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig, BlockSpaceManagerSPtr &blockManager);
protected:
    PrefillOutputs ApplyToWaitingQueue(SchedulingBudget &budget, const bool enableChunking = false) override;

    RunningOutputs ApplyToRunningQueue(SchedulingBudget &budget, const bool enableChunking = false) override;

    void setnewRequestFirst(SequenceGroupSPtr seqGroup);

    int GeneratePrefillBatch(SchedulingBudget &budget,
        std::vector<SequenceGroupSPtr> &ignoredSeqGroups, std::deque<SequenceGroupSPtr> &leftOverSeqGroups,
        int curCount, std::vector<std::shared_ptr<ScheduledSequenceGroup>> &seqGroups);

    int HandleDynamicPBS(int promptTokenIdsLen);

    void appendSlots(SequenceGroupSPtr seqGroup, RunningOutputs &runningOutput,
        const size_t numUncachedNewTokens, const bool enableChunking);
    
    std::deque<SequenceGroupSPtr> recomputeprefillQueue_;  // 重计算prefill缓存队列
    std::deque<SequenceGroupSPtr> longprefillQueue_;  // 超长prefill缓存队列
    unsigned int longprefillLenThreshold_ = 2048;  // 超过2048或者32768为超长序列
    bool dynamicPBSEnable_ = true;  // 动态PBS的开关

    const int middleprefillLenThreshold_ = 1024;  // 1K只能组2个
    const int middleBatchLenBudget_ = 2048;
    const int initBatchLenBudget_ = 4096;
    int curBatchLen_{0};
    int curBatchLenBudget_{0};

    const bool enableChunking_{false};
private:
    int GeneratePrefillBatchInner(SchedulingBudget &budget,
        std::vector<SequenceGroupSPtr> &ignoredSeqGroups,
        int curCount, std::vector<std::shared_ptr<ScheduledSequenceGroup>> &seqGroups,
        SequenceGroupSPtr seqGroup, std::vector<SequenceSPtr> waitingSeqs, size_t promptTokenIdsLen);
};
}


#endif