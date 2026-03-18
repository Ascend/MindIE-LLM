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

#ifndef SEQUENCE_GROUP_H
#define SEQUENCE_GROUP_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <unordered_set>

#include "basic_types.h"
#include "sampling.h"
#include "sequence.h"
#include "request_response/request_id.h"
#include "concurrent_map.h"

namespace mindie_llm {
constexpr std::chrono::high_resolution_clock::time_point INVALID_TIME =
    std::chrono::high_resolution_clock::time_point::min();

struct RequestMetrics {
    RequestIdNew inferReqId_;
    time_t arrivalTime_{};
    TimeStamp firstTokenTime_{}; // The time when the first token was generated.
    uint64_t queueWaitTime_{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> responseTime_{INVALID_TIME};
    uint64_t prefixCachedTokenNum_{0};
};
using RequestMetricsSPtr = std::shared_ptr<RequestMetrics>;

struct SequenceGroup {
    RequestId requestId;

    std::vector<SequenceSPtr> seqs_;

    SequenceSPtr firstSeq; // no need to define field, use method instead

    SamplingParamsSPtr sampling;

    RequestMetrics metrics_; // inference metrics of this request

    std::chrono::time_point<std::chrono::high_resolution_clock> arriveTime;

    // 给LatencyPredictor使用
    std::chrono::time_point<std::chrono::high_resolution_clock> lastCompletionTime;

    // 执行的轮次，非chunked prefill情况下，第一轮是prefill阶段，后面都是decode阶段
    uint64_t iterTimes{0};

    /* 请求优先级，比如调度支持支持优先级抢占策略 */
    uint64_t priority_{0};

    /* 推理输出序列最大输出长度 */
    uint64_t maxOutputLen_{1024};

    /* TBC_推理是否同步 */
    bool isSynchronous_{false};

    /* 为PD分离预留，prefill阶段的block table。P节点需要设置，D节点上需要此数据 */
    std::vector<BlockIds> pBlockTable{};

    /* 为PD分离预留，P节点的instance id。用于计算最终传给D节点的 dpinstanceid */
    InstanceId pInstanceId = 0;

    /* 为PD分离预留，完成请求的dp instance id。D节点上需要此数据 */
    uint64_t dpInstanceId_ = 0;

    /* 为PD分离预留，KV是否pulled。PD节点上需要此数据, P节点表示可以释放KV，D节点表示可以做decode */
    std::atomic<bool> isKVPulled{false};

    /* 序列最大的推理次数，包括prefill和decode */
    uint64_t maxIterTimes_ = 0;

    std::optional<bool> skipSpecialTokens_;

    std::optional<bool> ignoreEos_;

    /* 并行采样使用，存储一次采样中生成的全部序列 */
    ConcurrentMap<SequenceId, std::shared_ptr<SequenceGroup>> seqId2ParallelSeqGroup_;

    /* 并行采样使用，是否是新生成的序列组 */
    bool isNewSeqGroup_{false};

    bool needUpdate_{false};

    /* ChunkedPrefill使用，判断当前是否是最后一个prefill块 */
    bool isLastChunk_{false};

    /* 并行采样使用，父序列的 seqId */
    SequenceId parentSeqId_{-1};

    /* 并行采样使用，保存父序列的 blockId */
    std::vector<BlockIds> parentBlockIds_{};

    std::optional<std::string> loraId_;

    WaveId waveId_{-1}; // -1 means not set

    size_t rankId_ = 0;

    SequenceGroup(RequestId &tRequestId, const std::vector<SequenceSPtr> &tSeqs);

    SequenceGroup(RequestId &tRequestId, const std::vector<SequenceSPtr> &tSeqs, const SamplingParamsSPtr &tSampling);

    SequenceGroup(RequestId &tRequestId, const std::vector<SequenceSPtr> &tSeqs, const SamplingParamsSPtr &tSampling,
                    const std::optional<std::string> &tLoraId, size_t tRankId);

    ~SequenceGroup();
    
    std::vector<SequenceSPtr> GetFirstSequence(const SequenceStatus status = SequenceStatus::ALL_STATUS);

    std::vector<SequenceSPtr> GetSequences(const SequenceStatus status = {});

    std::vector<SequenceSPtr> GetParallelSequences(const SequenceStatus status = SequenceStatus::ALL_STATUS) const;

    std::vector<std::shared_ptr<SequenceGroup>> GetParallelSeqGrp();

    void UpdateNumComputedTokens(size_t numNewComputedTokens);

    [[nodiscard]] int GetMaxNumRunningSeqs() const;

    [[nodiscard]] bool IsPrefill() const;

    [[nodiscard]] bool IsLayerwisePrefill() const;

    [[nodiscard]] bool IsFinished() const;

    /// @brief 判断当前请求组是否为虚推请求
    /// @return 如果第一个序列的 seqId 等于 SIMULATE_SEQUENCE_ID 则返回 true
    [[nodiscard]] bool IsSimulateRequest() const;

    // pd分离场景下，标记当前seqGroup是否需要pull kv
    bool isDecode_{false};
    // pd分离场景下，标记当前seqGroup是否是本地flex节点上执行的
    bool isFlexLocal_{false};
    // 边云动态切块新增, 单位ms，给TG侧传请求到达时间间隔用于切图
    int32_t requestGap_{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> recomputeArriveTime_;
};

using SequenceGroupSPtr = std::shared_ptr<SequenceGroup>;

struct ScheduledSequenceGroup {
    SequenceGroupSPtr seqGroup_{};

    size_t tokenChunkSize_{}; // uncached token size need to be computed in following execution

    ScheduledSequenceGroup() = default;

    ScheduledSequenceGroup(const SequenceGroupSPtr &tSeqGroup, const size_t tTokenChunkSize,
                           bool enableChunked = false);
};

using ScheduledSequenceGroupSPtr = std::shared_ptr<ScheduledSequenceGroup>;

enum class ForwardMode : int { PREFILL, DECODE, EXTEND, MIXED, DUMMY };

struct SchedulerOutputs {
    std::vector<ScheduledSequenceGroupSPtr> scheduledSeqGroups_;

    std::vector<SequenceGroupSPtr> ignoredSeqGroups_;

    // recompute seq id, must notify text generator to release cache data
    std::vector<SequenceId> recomputeSeqIds_;

    size_t numPrefillGroups_{};

    size_t numBatchedTokens_{};

    size_t numPreempted_{};

    size_t runningQueueSize_{};

    std::vector<std::pair<BlockId, BlockId>> blocksToSwapIn_;

    std::vector<std::pair<BlockId, BlockId>> blocksToSwapOut_;

    std::vector<std::pair<BlockId, BlockId>> blocksToCopy_;

    ForwardMode forwardMode_{ForwardMode::DECODE};

    bool IsEmpty();
    // 边云特性动态切块使用，给TG侧传当前waiting队列长度
    uint32_t curWaitQueueLen_{0};
};

struct SchedulerKVTransferOutput {
    // sequence ids already pulled in the previous round (need to clear cache)
    std::unordered_set<SequenceId> pulledSeqIds;

    // sequence groups scheduled and to be pulled in this round
    std::vector<ScheduledSequenceGroupSPtr> pullSeqGroups;

    bool IsEmpty();
};

} // namespace mindie_llm

#endif