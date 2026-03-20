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

#ifndef FCFS_POLICY_H
#define FCFS_POLICY_H

#include "scheduler.h"
#include "policy.h"
#include "policy_helper.h"

namespace mindie_llm {
/**
 * At standard mode, if not enable prefill chunk, all sequence groups are in prefill or in decode.
 * if in decode, they are from running or swapped deque, else in waiting deque
 */
struct PrefillOutputs {
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> seqGroups_;

    std::vector<SequenceGroupSPtr> ignoredSeqGroups_;
};

struct RunningOutputs {
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> decodeSeqGroups_;

    std::vector<std::shared_ptr<ScheduledSequenceGroup>> chunkedPrefillSeqGroups_; // only used for chunked prefill

    std::vector<SequenceGroupSPtr> preempted_;

    std::vector<SequenceGroupSPtr> swappedOut_;

    std::vector<std::pair<BlockId, BlockId>> blocksToSwapOut_;

    std::vector<std::pair<BlockId, BlockId>> blocksToCopy_;

    std::vector<SequenceGroupSPtr> decodeSeqGroupsList_;

    std::vector<SequenceGroupSPtr> prefillSeqGroupsList_;

    void Clear();
};

struct SwappedInOutputs {
    std::vector<std::shared_ptr<ScheduledSequenceGroup>> decodeSeqGroups_;

    std::vector<std::shared_ptr<ScheduledSequenceGroup>> chunkedPrefillSeqGroups_; // only used for chunked prefill

    std::vector<SequenceGroupSPtr> infeasibleSeqGroups_;

    std::vector<std::pair<BlockId, BlockId>> blocksToSwapIn_;

    std::vector<std::pair<BlockId, BlockId>> blocksToCopy_;
};

class FcfsPolicy : public Policy {
public:
    FcfsPolicy(std::shared_ptr<SchedulerConfig> &schedulerConfig, BlockSpaceManagerSPtr &blockManager);

    PolicyOutput Apply(SchedulingBudget &budget, std::shared_ptr<SeqGroupCollection> &collection) override;

protected:
    void PreprocessCollection(const std::shared_ptr<SeqGroupCollection> &collection);

    PolicyOutput SchedulePrefill(SchedulingBudget &budget);

    PolicyOutput ScheduleDecode(SchedulingBudget &budget);

    [[nodiscard]] PolicyOutput BuildPolicyOutput(PrefillOutputs &prefillOut, RunningOutputs &runningOut,
                                                 SwappedInOutputs &swappedInOut, SchedulingBudget &budget) const;

    virtual PrefillOutputs ApplyToWaitingQueue(SchedulingBudget &budget, const bool enableChunking = false);

    virtual RunningOutputs ApplyToRunningQueue(SchedulingBudget &budget, const bool enableChunking = false);

    SwappedInOutputs ApplyToSwappedQueue(SchedulingBudget &budget, const bool enableChunking = false);

    PreemptionMode Preempt(SequenceGroupSPtr &seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut,
        size_t &swapNum);

    void PreemptByRecompute(const SequenceGroupSPtr &seqGroup);

    void PreemptBySwap(SequenceGroupSPtr &seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToSwapOut);

    static void WithdrawLeftovers(std::deque<SequenceGroupSPtr> &dst, const std::deque<SequenceGroupSPtr> &src);

    bool AllocBlocks4ParallelSeqGrp(SequenceGroupSPtr seqGroup, std::vector<std::pair<BlockId, BlockId>> &blockToCopy);

    virtual PolicyOutput ScheduleChunkedPrefill(SchedulingBudget &budget);

    void ScheduleRunningSeqGroup(const SequenceGroupSPtr &seqGroup, size_t numUncachedNewTokens,
                                 bool enableChunking, RunningOutputs &runningOutput, SchedulingBudget &budget);

    void UpdateStatusForRecompute(const SequenceGroupSPtr &seqGroup);

protected:
    int numCumulativePreemption_{};

    PolicyHelper policyHelper_;

    ISeqGroupCollectionSPtr queuesCollection_;

    std::shared_ptr<SchedulerConfig> schedulerConfig_;

    BlockSpaceManagerSPtr blockManager_;
};
} // namespace mindie_llm

#endif