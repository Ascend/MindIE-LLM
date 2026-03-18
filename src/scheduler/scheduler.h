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
 
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <algorithm>
#include <climits>

#include "ischeduler.h"
#include "policy/policy.h"
#include "live_infer_context.h"
#include "qps_tracker.h"
#include "latency_predictor/queue_counter.h"
#include "latency_predictor/latency_predictor.h"
#include "policy/stage_policy/stage_policy.h"
#include "policy/stage_policy/time_division_policy.h"
#include "policy/dynamic_batch_size.h"
#include "layerwise_mixin/layerwise_mixin.h"

namespace mindie_llm {

enum class PreemptionMode : int { NONE = 0, SWAP, RECOMPUTE };
constexpr size_t LOG_INTERVAL_COUNT = 100;
constexpr size_t defaultMaxQueueDelayMicroseconds = 5000;
constexpr size_t LOG_EMPTY_BATCH_INTERVAL_COUNT = 100000;

constexpr float PRESERVED_FACTOR_FOR_DECODE = 0.05;
struct SchedulerMetricsStatics {
    std::atomic<uint64_t> freeNpuBlockNum_;
    std::atomic<uint64_t> freeCpuBlockNum_;

    std::atomic<uint64_t> waitingRequestNum_;
    std::atomic<uint64_t> runningRequestNum_;
    std::atomic<uint64_t> swappedRequestNum_;

    // 最新一个完成的请求，prefill/decode单个token的计算时间
    std::atomic<uint64_t> prefillThroughput_;
    std::atomic<uint64_t> decodeThroughput_;

    // 累计的抢占次数
    std::atomic<uint64_t> cumulativePreemptCount_;

    // 统计所有请求的prompt token占用的block个数
    std::atomic<uint64_t> allRadixMatchNum_;

    // 统计所有请求的prompt cache命中的block个数
    std::atomic<uint64_t> npuRadixMatchHitNum_;
};

class Scheduler : public IScheduler {
public:
    explicit Scheduler(const SchedulerConfigSPtr &schedulerConfig, std::shared_ptr<LatencyPredictor> predictor,
                       Role pdRole, size_t localDPRank = 0);

    void AddSeqGroup(SequenceGroupSPtr &seqGroup) override;

    void StopRunningRequest() override;

    std::pair<SequenceGroupMetaDatas, SchedulerOutputs> Schedule(bool needSync = false) override;

    std::pair<SequenceGroupMetaDatas, SchedulerKVTransferOutput> ScheduleTransfer() override;

    size_t GetUnFinishedSeqGroups() override;

    // scheduler need to know which sequence reaches its life, so it need to fetch these sequences from execution
    // response handler thread or engine request control.
    std::unordered_set<SequenceId> &FetchFinishedSeqIds(ConcurrentDeque<SequenceId> &finishedSeqIds) override;
    std::unordered_set<SequenceId> &FetchExceptionSeqIds(ConcurrentDeque<SequenceId> &exceptionSeqIds) override;
    std::unordered_set<RequestId> &FetchAbortedReqIds(ConcurrentDeque<RequestId> &abortedReqIds) override;
    void KVPulledReqEnterRunningQueue(ConcurrentDeque<RequestId> &pulledReqIds) override;
    std::unordered_set<SequenceId> ClearAndReturnTerminatedSeqIds() override;

    void NotifyMeKvPulledSeqIds(SequenceId seqId) override;

    // sheduler need to get new generated tokens from execution response handler, so it can update place holder tokens
    // with real token
    void FetchSeqGeneratedTokens(ConcurrentDeque<std::pair<SequenceId, TokenId>> &seqIdToOutputTokenQueue) override;

    // mark that no batching in last schedule round
    void MarkLastScheduleEmpty() override;

    void ClearLastScheduleEmpty() override;

    // increase computed token number and add place holder tokens to output tokens
    void PrepareNextSchedule(std::vector<ScheduledSequenceGroupSPtr> &scheduledSeqGroups) override;

    SchedulerMetric CollectSchedulerMetric() override;

    void ClearSeqGrp(SequenceGroupSPtr seqGroup, SequenceStatus finalStatus) override;

    // output token of chunked prefill requests needs to be discard
    bool isDiscardOutputToken(LiveInferContextSPtr &contextSPtr, SequenceId seqId);

    std::vector<SequenceGroupSPtr> &GetAbortedParallelSeqGroups() override;

    void CollectAndClearAbortedParallelSeqGroups() override;
    void SetPrefillPercentage(uint32_t prefillPercentage) override;

    Role SwitchRole() override;

    std::shared_ptr<StagePolicy> GetStagePolicy() override;

protected:
    // dequeue sequence group concurrent deques into non-concurrent data structure which is only accessed by scheduling
    // thread. So scheduling policy can own this resulting data collection exclusively.
    std::shared_ptr<SeqGroupCollection> PrepCandidatesForPolicy(PDPriorityType pdPriorityType,
                                                                SchedulingBudget &budget);
    // kv transfer policy need to access waiting, running, transferring map according to P-D role, so dequeue them into
    // data collection for kv transfer scheduling. kv transfer should not be scheduled later and only pull kv should be
    // scheduled.
    std::shared_ptr<SeqGroupCollection> PrepCandidatesForKvTransferPolicy();

    void BackfillConcurrentQueue(PolicyOutput &policyOut);

    void BackfillConcurrentQueue(KVTransferPolicyOutput &policyOut);

    std::unordered_set<SequenceId> ReleaseKvPulledBlocks();

    [[nodiscard]] SchedulerOutputs ConvertToSchedulerOutput(const SchedulingBudget &budget, PolicyOutput &policyOut);

    [[nodiscard]] SchedulerKVTransferOutput ConvertToSchedulerTransferOutput(KVTransferPolicyOutput &policyOut) const;

private:
    void SetRole(Role role);

    void ClearSeq(SequenceId seqId);

    void ParallelSeqGroupLifeEnd(SequenceGroupSPtr seqGroup);

    [[nodiscard]] SequenceGroupMetaDatas GenerateSequenceGroupMetadata(const SchedulerOutputs &schedulerOut);

    void CollectOrAggregateComputedBlocks(std::vector<SequenceGroupMetaData> &metaList, size_t metaIndex,
 	    const std::vector<SequenceSPtr> &runningSeqSPtrs, bool isSimulateSeq);

    void CollectComputedBlocksInfo(std::vector<SequenceGroupMetaData> &metaList, size_t metaIndex,
    const std::vector<SequenceSPtr> &runningSeqSPtrs);
    
    void AggregateComputedBlocksInfo(std::vector<SequenceGroupMetaData> &metaList, size_t metaIndex,
        const std::vector<SequenceSPtr> &runningSeqSPtrs);

    [[nodiscard]] SequenceGroupMetaDatas GenSeqGroupMetadata(const SchedulerKVTransferOutput &schedulerOut) const;

    void SetBasicMetadata(SequenceGroupMetaData &metaData, const SequenceGroupSPtr seqGroup,
                          ScheduledSequenceGroupSPtr scheduledGrp) const;

    // 虚推请求入队处理：根据节点角色决定入队位置
    void EnqueueSimulateInferenceRequest(SequenceGroupSPtr &seqGroup);

    std::vector<BlockId> SetSpCpParamAndReturnAllBlocks(SequenceGroupMetaData &meta, SequenceGroupSPtr seqGrpSPtr,
                                                        SequenceId seqId, ForwardMode forwardMode) const;

    std::vector<BlockId> LwdSetSpCpParamAndReturnAllBlocks(SequenceGroupMetaData &meta, SequenceGroupSPtr seqGrpSPtr,
                                                        SequenceId seqId, ForwardMode forwardMode) const;

    std::vector<BlockIds> GetAllBlocks(SequenceGroupSPtr seqGrpSPtr, SequenceId seqId) const;
    void SetChunkedParam(SequenceSPtr seq, SequenceGroupMetaData &meta) const;

    PDPriorityType DecidePDPriority(bool needSync);

    void WaitingAvoidDummyBatch(PDPriorityType priority, bool needSync);

    // safe dequeue from concurrent deque to schedule policy owned deque
    size_t Dequeue(ConcurrentDeque<SequenceGroupSPtr> &srcQueue, std::deque<SequenceGroupSPtr> &dstDeque,
                   const size_t maxNum = 0);

    size_t DequeueWaiting(ConcurrentDeque<SequenceGroupSPtr> &srcQueue, std::deque<SequenceGroupSPtr> &dstDeque,
                          const size_t maxNum = 0);

    void Enqueue(ConcurrentDeque<SequenceGroupSPtr> &dstQueue, std::vector<SequenceGroupSPtr> &seqGroups,
                 bool front = false) const;
    // put scheduled ones to concurrent deque for continuous batching
    void Enqueue(ConcurrentDeque<SequenceGroupSPtr> &dstQueue,
                 std::vector<std::shared_ptr<ScheduledSequenceGroup>> &scheduleSeqGroups, bool front = false) const;

    void ReplacePlaceHolderWithToken(SequenceGroupSPtr seqGrpSPtr);

    void UpdatePromptAndOutputTokenIds(SequenceSPtr seq);

    void AddNextTokenPlaceHolder(std::vector<ScheduledSequenceGroupSPtr> &scheduledSeqGrps) const;

    void AccumulateComputedTokens(std::vector<ScheduledSequenceGroupSPtr> &seqGrps) const;

    bool ShouldImmediatePrefill();

    void ClearQueueAndSendAbortedResponse(ConcurrentDeque<SequenceGroupSPtr> &srcQueue);

    template <typename T> void PopAndSave_(ConcurrentDeque<T> &src, std::unordered_set<T> &dst) const;

    template <typename T> void LifeEndKVCleanup(std::unordered_set<T> &lifeEndSet);

    bool LifeEndedWrapup_(SequenceGroupSPtr &seqGroup);

    SequenceStatus FinalizeSeqGrpStatus(SequenceGroupSPtr seqGroup);

    size_t CalculatePlaceHolderNum(ScheduledSequenceGroupSPtr seqGrpSPtr) const;

    void RecordMetricsStatics(SchedulerOutputs &schedulerOut, SequenceGroupMetaDatas &seqGroupMetadata);

    SequenceGroupMetaDatas InitSequenceGroupMetaDatas(const SchedulerOutputs &schedulerOut) const;

    // flex 场景使用，应为不同类型的请求都在一个实例，出队时需要根据请求中携带的标签去过滤
    size_t DequeueForFlex(ConcurrentDeque<SequenceGroupSPtr> &srcQueue, std::deque<SequenceGroupSPtr> &dstDeque,
                          Role role, const size_t maxNum);
    // flex 场景使用，应为不同类型的请求都在一个实例，组batch时需要根据请求中携带的标签去过滤
    ISeqGroupCollectionSPtr PrepCandidatesForFlex(PDPriorityType pdPriorityType, SchedulingBudget &budget);
    // concurrent waiting queue shared by scheduling thread and main thread
    // Seq Groups in waiting_ queue have no kv blocks allocated, like prefill requests, and first decode requests after
    // prefill in separated P-D scenario
    ConcurrentDeque<SequenceGroupSPtr> waiting_;
    // concurrent running queue shared by scheduling thread and main thread
    // Seq groups in running_ queue have kv blocks allocated, like decode requests in PnD scenario, prefilling requests
    // in separated P-D scenario
    ConcurrentDeque<SequenceGroupSPtr> running_;
    // concurrent swapped queue shared by scheduling thread and main thread
    // seq groups in swapped_ queue have kv blocks swapped to host memory
    ConcurrentDeque<SequenceGroupSPtr> swapped_;

    // a sequence id to seq group map, containing all sequence groups transferring
    ConcurrentMap<SequenceId, SequenceGroupSPtr> transferringMap_;

    // P节点release kv使用
    ConcurrentDeque<SequenceId> kvCachePulledSeqIds_; // avoid concurrent to non-current copy, use concurrent directly

    std::shared_ptr<SchedulerConfig> schedulerConfig_;

    std::shared_ptr<LatencyPredictor> predictor_;

    // TBC: if one engine manages several DP ranks in one host, we need several schedulers (in EnginePerDP),
    // localDPRank_ tells the local DP rank in one host. If one engine manages one DP rank, it is always 0.
    size_t localDPRank_{0};

    QPSTracker qpsTracker;

    BlockSpaceManagerSPtr blockManager_; // kv cache manager

    std::shared_ptr<Policy> prefillPolicy_; // prefill阶段的policy，当前支持fcfs

    std::shared_ptr<Policy> decodePolicy_; // decode阶段的policy，当前支持fcfs

    std::shared_ptr<StagePolicy> stagePolicy_;

    std::shared_ptr<DynamicBatchSize> dynamicBatchSize_;

    // kv transfer scheduling policy, kv transfer needs HBM blocks allocated, so when free blocks are limited, the
    // policy decides which can be batched for kv transferring.
    std::shared_ptr<KVTransferPolicy> transferPolicy_;

    Role role_;

    /** 存储从response获取的完成的seqIds */
    std::unordered_set<SequenceId> finishedSeqIds_;

    std::unordered_map<SequenceId, std::vector<TokenId>> predictedTokensBySeqId_;

    // engine收到的都是RequestId， 执行器收到的响应都是SequenceId
    // copy from engine's aborted reqid concurrent deque
    std::unordered_set<RequestId> abortedReqIds_;

    /** 存储从response获取的exception的seqIds */
    std::unordered_set<SequenceId> exceptionSeqIds_;

    std::vector<SequenceGroupSPtr> abortedSeqGroups_;

    // Records last schedule where prefill had priority but output was empty.
    bool lastScheduleEmpty_{false};

    bool serving_{false}; // 用来避免服务中切换role， 并且此标志并不真正管用。 未来会支持服务中切换role

    std::unordered_set<SequenceId> abortedSequenceIds_;

    // 打印日志时使用
    // 调度轮次
    size_t iterTimes_{0};
    // rankid
    int dpRankId_{0};

    SchedulerMetricsStatics schedulerMetricsStatics_{};

    // 调度实际下发的批次，当前实现，同步时是1，异步（只支持异步单发）时是2
    size_t maxScheduledBatch_{1};

    // save aborted ParallelSeqGroups
    // Since RECOMPUTE is not currently supported for ParallelSeqGroups, the preempted request will be aborted.
    // abortedParallelSeqGroups_ is also used in lingqu recover
    std::vector<SequenceGroupSPtr> abortedParallelSeqGroups_;

    // 边云新增
    PDPriorityType LayerwiseDecidePDPriority(size_t freeBlocksNum, size_t reserveBlockNum4Decode);
    bool LayerwiseDiscardToken(LiveInferContextSPtr &contextSPtr, SequenceId seqId);
    LwdPDelayType LayerwiseDecidePDelay();
    std::chrono::time_point<std::chrono::high_resolution_clock> pDelayTime{INVALID_TIME};
    LayerwiseMixin layerwiseMixin_;
};

inline size_t TrailingPlaceholderTokenCount(std::vector<TokenId> &outPutTokenIds)
{
    // 计算place holder个数
    auto rit = std::find_if_not(outPutTokenIds.rbegin(), outPutTokenIds.rend(),
                                [](auto token) { return token == PLACEHOLDER_TOKEN; });
    size_t placeholderCount = static_cast<size_t>(std::distance(outPutTokenIds.rbegin(), rit));

    return placeholderCount;
}

} // namespace mindie_llm

#endif