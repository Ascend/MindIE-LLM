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
 
#include "scheduler.h"

#include <climits>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include "policy/policy_factory.h"
#include "log.h"
#include "request_response/request_id.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "pre_scheduler.h"
#include "policy/stage_policy/edge_cloud_policy.h"

using namespace std;
using namespace std::chrono;
using std::chrono::system_clock;

namespace mindie_llm {
constexpr bool USE_PREFIX_CACHE_WITH_PLACE_HOLDER = false;
constexpr size_t PREFILL_SCHEDULER_SLEEP_INTERVAL = 1;
constexpr int PREPARE_DATA_EXPANSION_FACTOR = 2;
constexpr int TIME_WINDOW_INTERVAL = 1000; // 统计1s之内的的QPS
constexpr int BUCKET_TIME_INTERVAL = 20;   // 每个bukcet的时间间隔是20ms

namespace {
BlockSpaceManagerSPtr CreateBlockManagerFromSchedulerConfig(const SchedulerConfig &cfg, size_t localDPRank)
{
    if (cfg.spSize * cfg.cpSize > 1 && cfg.kvCacheDescs.size() > 1) {
        throw std::invalid_argument("Composite block manager is not supported when sp/cp is enabled.");
    }

    // Backward compatible: if kvCacheDescs is empty, fallback to single block manager config.
    if (cfg.kvCacheDescs.empty()) {
        BlockManagerConfig blockConf = {cfg.cacheBlockSize,
                                        cfg.cpuBlockNum,
                                        cfg.npuBlockNum,
                                        0,
                                        cfg.speculationGamma,
                                        cfg.enablePrefixCache,
                                        cfg.spSize * cfg.cpSize,
                                        1,
                                        cfg.enableKvPool,
                                        cfg.kvPoolConfig.backend,
                                        cfg.kvPoolConfig.configPath};
        return BlockManagerFactory::CreateBlockSpaceManager(BlockManagerType::SELFATTNBLOCKMANAGER,
                                                            std::move(blockConf), localDPRank);
    } else if (cfg.kvCacheDescs.size() == 1) {
        auto desc = cfg.kvCacheDescs[0];
        BlockManagerConfig blockConf = {static_cast<size_t>(desc.blockSize) *
                                            static_cast<size_t>(desc.compressionRatio),
                                        cfg.cpuBlockNum,
                                        static_cast<uint32_t>(desc.npuBlockNum),
                                        0,
                                        cfg.speculationGamma,
                                        cfg.enablePrefixCache,
                                        cfg.spSize * cfg.cpSize,
                                        1,
                                        cfg.enableKvPool,
                                        cfg.kvPoolConfig.backend,
                                        cfg.kvPoolConfig.configPath};
        blockConf.cacheType = static_cast<KvCacheType>(desc.cacheType);
        return BlockManagerFactory::CreateBlockSpaceManager(BlockManagerType::SELFATTNBLOCKMANAGER,
                                                            std::move(blockConf), localDPRank);
    }

    throw std::invalid_argument("Multiple kvCacheDescs are not supported by current BlockManagerFactory.");
}
} // namespace

Scheduler::Scheduler(const std::shared_ptr<SchedulerConfig> &schedulerConfig,
                     std::shared_ptr<LatencyPredictor> predictor, Role pdRole, size_t localDPRank)
    : schedulerConfig_(schedulerConfig), predictor_(predictor), localDPRank_(localDPRank),
      qpsTracker(TIME_WINDOW_INTERVAL, BUCKET_TIME_INTERVAL)
{
    BlockManagerConfig blockConf = {schedulerConfig->cacheBlockSize,
                                    schedulerConfig->cpuBlockNum,
                                    schedulerConfig->npuBlockNum,
                                    0,
                                    schedulerConfig->speculationGamma,
                                    schedulerConfig->enablePrefixCache,
                                    schedulerConfig->spSize * schedulerConfig->cpSize,
                                    1,
                                    schedulerConfig->enableKvPool,
                                    schedulerConfig->kvPoolConfig.backend,
                                    schedulerConfig->kvPoolConfig.configPath};
    dpRankId_ = schedulerConfig_->dpRankId_;
    if (schedulerConfig_->layerwiseDisaggregated && schedulerConfig->spSize * schedulerConfig->cpSize > 1) {
        blockManager_ = BlockManagerFactory::CreateBlockSpaceManager(BlockManagerType::LWDSELFATTNBLOCKMANAGER,
                                                                     std::move(blockConf), localDPRank);
        BlockManagerConfig blockConf = {schedulerConfig->cacheBlockSize,
                                        schedulerConfig->cpuBlockNum,
                                        schedulerConfig->lwdCloudNpuBlockNum,
                                        0,
                                        schedulerConfig->speculationGamma,
                                        schedulerConfig->enablePrefixCache,
                                        16,
                                        1,
                                        schedulerConfig->enableKvPool,
                                        schedulerConfig->kvPoolConfig.backend,
                                        schedulerConfig->kvPoolConfig.configPath};
        blockManager_->LwdInitCloudBlockManager(std::move(blockConf), localDPRank);
    } else {
        blockManager_ = CreateBlockManagerFromSchedulerConfig(*schedulerConfig_, localDPRank);
    }
    // create inference scheduling policy and kv transfer scheduling policy according to role
    // If role is P only or D only, polices will be created again.
    SetRole(pdRole); // 默认是非PD分离的。因为非PD分离场景，Server不会调用SetRole方法。

    uint32_t asyncScheduleRound = schedulerConfig_->layerwiseDisaggregated ?
                                schedulerConfig_->maxDispatchBatchNum : MAX_ASYNC_SCHEDULE_TIMES;

    if (schedulerConfig_->activateAsyncInference) {
        maxScheduledBatch_ = asyncScheduleRound + 1;
        MINDIE_LLM_LOG_INFO("Scheduler enable async. maxScheduledBatch:" << maxScheduledBatch_);
    }

    stagePolicy_ = PolicyFactory::CreateStagePolicy(schedulerConfig, predictor, blockManager_, pdRole);
    dynamicBatchSize_ = std::make_shared<DynamicBatchSize>(schedulerConfig, predictor, blockManager_, localDPRank_);

    MINDIE_LLM_LOG_INFO("Scheduler init success!");
}

void Scheduler::SetRole(Role role)
{
    // 1. set Role must called before accept Request
    if (serving_) {
        throw std::runtime_error("set roles cannot be called after accept request.");
    }

    // 2. create policy by policyType
    if (schedulerConfig_->prefillPolicyType != 0 || schedulerConfig_->decodePolicyType != 0) {
        throw std::runtime_error("Not support policy type.");
    }
    prefillPolicy_ = PolicyFactory::CreatePolicy(role, schedulerConfig_, blockManager_, PDPriorityType::PREFILL_FIRST);
    decodePolicy_ = PolicyFactory::CreatePolicy(role, schedulerConfig_, blockManager_, PDPriorityType::DECODE_FIRST);
    if (role != Role::PnD) {
        transferPolicy_ = PolicyFactory::CreateTransferPolicy(role, schedulerConfig_, blockManager_);
    }
    MINDIE_LLM_LOG_INFO("Policy create success!");
    role_ = role;
    PROF(INFO, AddMetaInfo("Role", static_cast<uint8_t>(role_)));
}

void Scheduler::AddSeqGroup(SequenceGroupSPtr &seqGroup)
{
    // 1. check request id (虚推请求跳过重复检查，因为大EP场景下多个rank会同时添加同一个虚推请求)
    bool isSimulateInference = seqGroup->IsSimulateRequest();
    if (isSimulateInference) {
        MINDIE_LLM_LOG_DEBUG("[SimulateInference] Simulate inference request entering AddSeqGroup, requestId="
                            << seqGroup->requestId << ", seqId=" << seqGroup->firstSeq->seqId_);
    }
    if (!isSimulateInference && LiveInferContext::GetInstance(localDPRank_)->GetSeqGroup(seqGroup->requestId)) {
        throw std::runtime_error("the requestId exist, requestId=" + seqGroup->requestId);
    }

    // 2. add sequence group to LiveInferContext
    LiveInferContext::GetInstance(localDPRank_)->Add(seqGroup);
    if (seqGroup->sampling->enableParallelSampling) {
        LiveInferContext::GetInstance(localDPRank_)->AddIntoSeqRootMap(seqGroup->firstSeq->seqId_, seqGroup);
    }

    // 边云动态切块新增,统计新入队请求与队列中上一个入队请求的到达时间间隔,对重计算请求按照重计算重新入队时间计算
    layerwiseMixin_.LwdComputeArrTimeGap(schedulerConfig_->layerwiseDisaggregated, seqGroup, waiting_.Back());

    // 3. add sequence group to waiting queue, it will do prefill or pull kv
    if (isSimulateInference) {
        EnqueueSimulateInferenceRequest(seqGroup);
    } else {
        waiting_.PushBack(seqGroup);
        auto prof = PROF(INFO, Domain("Schedule").Resource(seqGroup->requestId));
        PROF(prof.Metric("QueueSize", waiting_.Size()).Attr("status", "waiting").Event("Enqueue"));
    }

    // 4. record for qps
    qpsTracker.Record();

    serving_ = true;
}

void Scheduler::EnqueueSimulateInferenceRequest(SequenceGroupSPtr &seqGroup)
{
    // 虚推请求根据节点角色选择入队位置：
    // - D节点/FlexD节点/PnD节点/FlexPnD节点：直接进入running队列
    // - P节点/FlexP节点：进入waiting队列，走正常prefill流程，maxOutputLen会使虚推直接返回不进入D节点
    auto prof = PROF(INFO, Domain("Schedule").Resource(seqGroup->requestId));
    if (role_ == Role::D || role_ == Role::FlexD || role_ == Role::PnD || role_ == Role::FlexPnD) {
        seqGroup->firstSeq->status_ = SequenceStatus::RUNNING;
        seqGroup->firstSeq->data_.stage_ = SequenceStage::DECODE;
        running_.PushBack(seqGroup);
        PROF(prof.Metric("QueueSize", running_.Size()).Attr("status", "running").Event("Enqueue"));
        MINDIE_LLM_LOG_DEBUG("[SimulateInference] D/PnD node: special seqId enter running queue directly, seqId="
                            << seqGroup->firstSeq->seqId_ << ", role=" << static_cast<int>(role_));
    } else {
        waiting_.PushBack(seqGroup);
        PROF(prof.Metric("QueueSize", waiting_.Size()).Attr("status", "waiting").Event("Enqueue"));
        MINDIE_LLM_LOG_DEBUG("[SimulateInference] P node: special seqId enter waiting queue, seqId="
                            << seqGroup->firstSeq->seqId_ << ", role=" << static_cast<int>(role_));
    }
}

void Scheduler::RecordMetricsStatics(SchedulerOutputs &schedulerOut, SequenceGroupMetaDatas &seqGroupMetadata)
{
    schedulerMetricsStatics_.freeNpuBlockNum_ = blockManager_->GetNumFreeNpuBlocks();
    schedulerMetricsStatics_.freeCpuBlockNum_ = blockManager_->GetNumFreeCpuBlocks();
    schedulerMetricsStatics_.waitingRequestNum_ = waiting_.Size() + transferringMap_.Size();
    schedulerMetricsStatics_.runningRequestNum_ = running_.Size();
    schedulerMetricsStatics_.swappedRequestNum_ = swapped_.Size();

    if (schedulerOut.forwardMode_ == ForwardMode::PREFILL) {
        for (SequenceGroupMetaData meta : seqGroupMetadata.metaList) {
            // prefix cache 命中率口径：cached_tokens / prefill_tokens
            // - computedLens_：命中的 block 数（可能是每个 rank 一项）
            // - 分母用 prompt token 数（用 promptLens_ 聚合，避免 tokenIds_ 在 recompute 场景包含 outputTokenIds）
            const uint64_t promptTokens =
                std::accumulate(meta.promptLens_.begin(), meta.promptLens_.end(), static_cast<uint64_t>(0));
            const uint64_t blockSize = static_cast<uint64_t>(schedulerConfig_->cacheBlockSize);
            const uint64_t hitBlocks =
                std::accumulate(meta.computedLens_.begin(), meta.computedLens_.end(), static_cast<uint64_t>(0));
            const uint64_t hitTokens = (blockSize == 0) ? 0 : std::min(promptTokens, hitBlocks * blockSize);

            schedulerMetricsStatics_.allRadixMatchNum_.fetch_add(promptTokens);
            schedulerMetricsStatics_.npuRadixMatchHitNum_.fetch_add(hitTokens);
        }
    }
    schedulerMetricsStatics_.cumulativePreemptCount_.fetch_add(schedulerOut.numPreempted_);

    // 更新wave信息，暂不考虑chunked prefill场景
    if (schedulerOut.forwardMode_ != ForwardMode::PREFILL ||
        schedulerOut.scheduledSeqGroups_[0]->seqGroup_->waveId_ < 0) {
        // waveId为-1表示没有设置waveId, 无需更新
        return;
    }

    std::unordered_map<WaveId, size_t> tempWaveInfo;
    for (const auto &item : schedulerOut.scheduledSeqGroups_) {
        WaveId waveId = item->seqGroup_->waveId_;
        if (tempWaveInfo.find(waveId) == tempWaveInfo.end()) {
            tempWaveInfo[waveId] = 1;
        }
        tempWaveInfo[waveId] = tempWaveInfo[waveId] + 1;
    }
}

void Scheduler::WaitingAvoidDummyBatch(PDPriorityType priority, bool needSync)
{
    // 不需要同步 或 不做prefill时，不需要wait
    if (!needSync || priority != PDPriorityType::PREFILL_FIRST) {
        return;
    }

    // 当qps小于阈值时，不需要wait
    if (qpsTracker.GetQPS() < schedulerConfig_->lowQPSForWaitBatch) {
        return;
    }

    // In large distributed PD disaggregation cases, different DPs need to wait
    // for a while to avoid dummy requests and improve performance.
    size_t waitTimes = schedulerConfig_->maxQueueDelayMicroseconds / 1000;
    for (size_t i = 0; i < waitTimes; i++) {
        if (waiting_.Size() < schedulerConfig_->maxPrefillBatchSize) {
            std::this_thread::sleep_for(std::chrono::milliseconds(PREFILL_SCHEDULER_SLEEP_INTERVAL));
        } else {
            MINDIE_LLM_LOG_INFO("waiting queue sleep time is " << i);
            break;
        }
    }
}

std::pair<SequenceGroupMetaDatas, SchedulerOutputs> Scheduler::Schedule(bool needSync)
{
    PDPriorityType pdPriorityType = DecidePDPriority(needSync);
    if (role_ == Role::P) {
        WaitingAvoidDummyBatch(pdPriorityType, needSync);
    }

    if (schedulerConfig_->layerwiseDisaggregated && schedulerConfig_->dpSize > 1) {
        layerwiseMixin_.LwdWaitingResponse(pdPriorityType, stagePolicy_);
    }

    LwdPDelayType pDelayType = LwdPDelayType::INVALID;
    if (schedulerConfig_->layerwiseDisaggregated &&
            schedulerConfig_->dpSize == 1 && pdPriorityType == PDPriorityType::PREFILL_FIRST) {
        pDelayType = LayerwiseDecidePDelay();
        if (pDelayType == LwdPDelayType::PREFILL_TO_DECODE) {
            pdPriorityType = PDPriorityType::DECODE_FIRST;
        }
    }

    size_t batchSize = (pdPriorityType == PDPriorityType::PREFILL_FIRST) ? schedulerConfig_->maxPrefillBatchSize
                                                                         : schedulerConfig_->maxBatchSize;
    // maxSeqLen和maxPrefillTokens哪个大，哪个作为budget的batch 最大token数。
    size_t budgetTokenNum = (schedulerConfig_->maxSeqLen > schedulerConfig_->maxPrefillTokens)
                                ? schedulerConfig_->maxSeqLen
                                : schedulerConfig_->maxPrefillTokens;
    if (pDelayType == LwdPDelayType::PREFILL_SKIP) {
        // 仅开启Lwd特性下2P调度情况下，可能决策p延迟下发，且延迟策略为跳过本轮p调度
        batchSize = 0;
        budgetTokenNum = 0;
    }
    SchedulingBudget budget(budgetTokenNum, batchSize, schedulerConfig_);

    // 2. apply policy
    // return different collection for different policies
    ISeqGroupCollectionSPtr data;
    if (static_cast<int32_t>(role_) <= 2) { // 小于 2， role为 PnD, P, D
        data = PrepCandidatesForPolicy(pdPriorityType, budget);
    } else {
        data = PrepCandidatesForFlex(pdPriorityType, budget);
    }

    PolicyOutput policyOutput;
    if (pdPriorityType == PDPriorityType::PREFILL_FIRST) {
        policyOutput = prefillPolicy_->Apply(budget, data);
        decodePolicy_->newRequestFirst_ = prefillPolicy_->newRequestFirst_;
    } else {
        policyOutput = decodePolicy_->Apply(budget, data);
        prefillPolicy_->newRequestFirst_ = decodePolicy_->newRequestFirst_;
    }

    // 3. update waiting/running/swapped queue
    BackfillConcurrentQueue(policyOutput);

    // 4. statistic meta data
    SchedulerOutputs schedulerOut = ConvertToSchedulerOutput(budget, policyOutput);
    auto seqGroupMetadata = GenerateSequenceGroupMetadata(schedulerOut);

    // 本轮调度结束，mark供下一轮调度使用
    blockManager_->MarkBlocksAsComputed();
    
    if ((!schedulerOut.IsEmpty() && schedulerOut.forwardMode_ == ForwardMode::PREFILL) ||
        (!schedulerOut.IsEmpty() && iterTimes_++ % LOG_INTERVAL_COUNT == 0) ||
        iterTimes_++ % LOG_EMPTY_BATCH_INTERVAL_COUNT == 0) {
        MINDIE_LLM_LOG_INFO_REQUEST("[Scheduler|Schedule-scheduling] DP RankId: "
                            << dpRankId_ << ". After Backfill, running size:" << running_.Size()
                            << "; waiting size: " << waiting_.Size() << "; swapped size:" << swapped_.Size()
                            << "; batch size:" << schedulerOut.scheduledSeqGroups_.size()
                            << "; transferring size:" << transferringMap_.Size()
                            << "; schedule forwardMode:" << static_cast<int>(schedulerOut.forwardMode_)
                            << "; PD PriorityType:" << static_cast<int>(pdPriorityType));
    }

    // 5.获取cpu和npu的空闲块个数，用于构造metric统计信息
    RecordMetricsStatics(schedulerOut, seqGroupMetadata);

    if (schedulerOut.scheduledSeqGroups_.size() > 0) {
        PROF(L1, Domain("Schedule").Metric("QueueSize", waiting_.Size()).Attr("status", "waiting").Event("Queue"));
        PROF(L1, Domain("Schedule").Metric("QueueSize", running_.Size()).Attr("status", "running").Event("Queue"));
        PROF(L1, Domain("Schedule").Metric("QueueSize", swapped_.Size()).Attr("status", "swapped").Event("Queue"));
    }

    // 6.统计BatchStats信息，用于时延优先和动态batchSize策略
    dynamicBatchSize_->RecordPredictorMetrics(schedulerOut, budget);

    // 7.PD混布且动态更新batchsize开关打开时，动态更新batchsize值
    dynamicBatchSize_->ApplyDynamicBatchSize(role_, schedulerOut, waiting_.Size(), running_.Size(), swapped_.Size());

    return {seqGroupMetadata, schedulerOut};
}

std::unordered_set<SequenceId> Scheduler::ReleaseKvPulledBlocks()
{
    // kvCachePulledSeqIds_ and transferringMap_ are erased only here. otherwise there will be concurrent bug
    std::unordered_set<SequenceId> pulledSeqIds;
    while (!kvCachePulledSeqIds_.Empty()) {
        SequenceId seqId = -1;
        kvCachePulledSeqIds_.PopFront(seqId);
        if (transferringMap_.Count(seqId) == 0) {
            // abort流程也会触发release kv，如果abort和release kv 并发，可能会存在这个场景。增加日志打印
            MINDIE_LLM_LOG_WARN("Try to release kv, but kv has released before. seqid:" << seqId);
        }
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Request-Release KV] DP RankId: "
                            << dpRankId_ << ". KV blocks of seqId: " << seqId << " are released.");
        blockManager_->Free(seqId);
        transferringMap_.Erase(seqId);
        LiveInferContext::GetInstance(localDPRank_)->Remove(seqId);
        pulledSeqIds.insert(seqId);
    }

    return pulledSeqIds;
}

// Decode needs to schedule kv pull, P 's kv transfer scheduling is only for kv block free
std::pair<SequenceGroupMetaDatas, SchedulerKVTransferOutput> Scheduler::ScheduleTransfer()
{
    if (role_ == Role::P || role_ == Role::FlexP) {
        std::unordered_set<SequenceId> pulledSeqIds = ReleaseKvPulledBlocks();
        return {SequenceGroupMetaDatas(), SchedulerKVTransferOutput({pulledSeqIds, {}})};
    }

    // 1. create budget
    // Some blocks are idle to avoid recompute and swap out.
    size_t freeTokenNum = blockManager_->GetNumFreeNpuBlocks() * schedulerConfig_->cacheBlockSize;
    if (freeTokenNum <= schedulerConfig_->maxBatchSize) {
        // 没有可用的block做pv pull
        return {SequenceGroupMetaDatas(), SchedulerKVTransferOutput()};
    }
    size_t maxTransferTokens = freeTokenNum - schedulerConfig_->maxBatchSize;
    SchedulingBudget budget(maxTransferTokens, schedulerConfig_->maxPrefillBatchSize, schedulerConfig_);

    // apply policy
    ISeqGroupCollectionSPtr data = PrepCandidatesForKvTransferPolicy();

    if (transferPolicy_ == nullptr) {
        throw std::runtime_error("transferPolicy_ is not initialized. Please set role before scheduling transfer.");
    }
    KVTransferPolicyOutput policyOutput = transferPolicy_->Apply(budget, data);

    // backfill waiting/running/transfer
    BackfillConcurrentQueue(policyOutput);

    // statistic meta data
    SchedulerKVTransferOutput transferOut = ConvertToSchedulerTransferOutput(policyOutput);
    auto seqGroupMetadata = GenSeqGroupMetadata(transferOut);

    // 累计computed token数量
    AccumulateComputedTokens(transferOut.pullSeqGroups);
    if (schedulerConfig_->speculationGamma > 0) {
        // PD分离在D节点的Prompt里面已经包括prefill生成的第一个token
        // mtp或者并行解码场景下，需要额外申请占位符用于slot申请
        AddNextTokenPlaceHolder(transferOut.pullSeqGroups);
    }

    if (!transferOut.IsEmpty()) {
        PROF(L1, Domain("Schedule").Metric("QueueSize", waiting_.Size()).Attr("status", "waiting").Event("Queue"));
    }

    return {seqGroupMetadata, transferOut};
}

void Scheduler::PrepareNextSchedule(std::vector<ScheduledSequenceGroupSPtr> &scheduledSeqGroups)
{
    AccumulateComputedTokens(scheduledSeqGroups);
    AddNextTokenPlaceHolder(scheduledSeqGroups);
}

size_t Scheduler::GetUnFinishedSeqGroups() { return waiting_.Size() + running_.Size() + swapped_.Size(); }

/** batch调度希望调度尽量多的batch
 *  size的请求。因此会等maxQueueDelayMicroseconds时间再取调度waiting，让请求在waiting累积一些
 */
bool Scheduler::ShouldImmediatePrefill()
{
    auto now = std::chrono::high_resolution_clock::now();
    if (waiting_.Empty()) {
        return false;
    }
    // 适用于要求首Token时延的场景，组够batch就立即调度
    if (schedulerConfig_->maxQueueDelayMicroseconds > defaultMaxQueueDelayMicroseconds &&
        waiting_.Size() >= schedulerConfig_->maxPrefillBatchSize) {
        return true;
    }
    // waiting并发队列只有调度器一个consumer， 所以之前判断不空， 这里仍然有效。
    auto earliestTime = waiting_.Front()->arriveTime;
    std::chrono::duration queueDelay = now - earliestTime;
    std::chrono::duration maxQueueDelay = std::chrono::microseconds(schedulerConfig_->maxQueueDelayMicroseconds);
    bool reached = (queueDelay >= maxQueueDelay);
    return reached;
}

PDPriorityType Scheduler::LayerwiseDecidePDPriority(size_t freeBlocksNum, size_t reserveBlockNum4Decode)
{
    PDPriorityType priority = PDPriorityType::PREFILL_FIRST;
    MINDIE_LLM_LOG_DEBUG("[layerwiseDisaggregated|scheduler] "<<"lastScheduleEmpty_: "
        << lastScheduleEmpty_ << ", running_.Empty(): " << running_.Empty() << ", swapped_.Empty(): "
        << swapped_.Empty() << ", ShouldImmediatePrefill(): " << ShouldImmediatePrefill()
        << ", freeBlocksNum < reserveBlockNum4Decode: " << (freeBlocksNum < reserveBlockNum4Decode));

    std::shared_ptr<EdgeCloudPolicy> lwdPolicy = std::static_pointer_cast<EdgeCloudPolicy>(stagePolicy_);
    if (((lastScheduleEmpty_ && !running_.Empty()) || !swapped_.Empty() ||
        freeBlocksNum < reserveBlockNum4Decode) && (lwdPolicy->GetDecodeBatchCnt() == 0)) {
        MINDIE_LLM_LOG_DEBUG("[layerwiseDisaggregated|scheduler] "
            <<"last batch empty, no decode is processing, force schedule decode!");
        priority = PDPriorityType::DECODE_FIRST;
    } else {
        priority = stagePolicy_->Apply(waiting_, running_, swapped_);
    }
    return priority;
}

LwdPDelayType Scheduler::LayerwiseDecidePDelay()
{
    static bool isDelayEnable = true;
    int32_t maxRequestIntervalTime = 2500;
    int32_t minRequestIntervalTime = 2000;
    static int32_t requestIntervalOverThresholdCount = 0;
    static int32_t requestIntervalUnderThresholdCount = 0;
    // 开启Lwd特性前提下，本函数决定本轮是否延迟下发Prefill请求，以及若延迟下发采用的处理策略
    if (isDelayEnable && schedulerConfig_->batchPnum == 2 && waiting_.Size() > 0) { // 2为最大的P batch数量
        auto currentTime = std::chrono::high_resolution_clock::now();
        int32_t pWaitTime = -1;
        if (pDelayTime != INVALID_TIME) {
            pWaitTime = static_cast<int32_t>(duration_cast<milliseconds>(currentTime - pDelayTime).count());
        }
        int32_t maxPWaitTime = 1000; // 最大等待1000ms
        if (waiting_.Front()->requestGap_ > 0) {
            bool isGapOverIntervalTime = waiting_.Front()->requestGap_ > maxRequestIntervalTime;
            requestIntervalOverThresholdCount += isGapOverIntervalTime ? 1 : 0;
            // 累计5次请求到达间隔超过2.5s, 关闭延迟下发
            if (isGapOverIntervalTime && requestIntervalOverThresholdCount >= 5) {
                isDelayEnable = false;
                requestIntervalUnderThresholdCount = 0;
            }
            // maxPWaitTime = waiting_.Front()->requestGap_ / 2;
            maxPWaitTime = std::min(maxPWaitTime, waiting_.Front()->requestGap_ / 2);
            // 这里稍微调了下，一是避免重计算情况下延迟下发太久，二是真是情况下也可能请求率就是特别低
        }
        std::shared_ptr<EdgeCloudPolicy> lwdPolicy = std::static_pointer_cast<EdgeCloudPolicy>(stagePolicy_);
        // 判断当前waiting队列中有2P，或等待已超限，则停止延迟，立即下发，延迟计时复位
        if (waiting_.Size() >= 2 || pWaitTime > maxPWaitTime) {
            pDelayTime = INVALID_TIME;
            return LwdPDelayType::PREFILL_KEEP;
        } else if (lwdPolicy->GetDecodeBatchCnt() < 1 && running_.Size() > 0) {
            // 若上述不延迟条件均不满足，P下发延迟，若D batch计数为0且有D，则尝试下发D
            // pDelayTime标识延迟初始时刻，默认为INVALID_TIME，若为INVALID_TIME说明第一次打算等，置为决策延迟下发时刻，否则保持延长计时
            pDelayTime = (pDelayTime == INVALID_TIME ? currentTime : pDelayTime);
            return LwdPDelayType::PREFILL_TO_DECODE;
        } else {
            // 若上述不延迟条件均不满足，P下发延迟，且不满足下D条件，则跳过本轮调度
            pDelayTime = (pDelayTime == INVALID_TIME ? currentTime : pDelayTime);
            return LwdPDelayType::PREFILL_SKIP;
        }
    } else {
        if (!isDelayEnable && waiting_.Size() > 0 && waiting_.Front()->requestGap_ <= minRequestIntervalTime) {
            requestIntervalUnderThresholdCount++;
            if (requestIntervalUnderThresholdCount >= 3) { // 累计3次请求到达间隔不超过2s, 开启延迟下发
                requestIntervalOverThresholdCount = 0;
                isDelayEnable = true;
            }
        }
        pDelayTime = INVALID_TIME;
        return LwdPDelayType::PREFILL_KEEP;
    }
}

// decide what to schedule in this round ,  prefill or decode
PDPriorityType Scheduler::DecidePDPriority(bool needSync)
{
    PDPriorityType priority = PDPriorityType::PREFILL_FIRST;
    switch (role_) {
        case Role::PnD:
        case Role::FlexPnD: {
            size_t freeBlocksNum = blockManager_->GetNumFreeNpuBlocks();
            size_t totalBlocksNum = blockManager_->GetTotalNpuBlocks();
            size_t reserveBlockNum4Decode = static_cast<size_t>(PRESERVED_FACTOR_FOR_DECODE * totalBlocksNum);
            if (schedulerConfig_->enableChunkedPrefill) {
                // PD混部场景，chunked prefill优先
                return PDPriorityType::MIX;
            }
            if (schedulerConfig_->layerwiseDisaggregated) {
                priority = LayerwiseDecidePDPriority(freeBlocksNum, reserveBlockNum4Decode);
            } else if (lastScheduleEmpty_ && !running_.Empty()) {
                priority = PDPriorityType::DECODE_FIRST;
            } else if (schedulerConfig_->stageSelectPolicy > static_cast<uint32_t>(StagePolicyType::PREFILL_FIRST)) {
                priority = stagePolicy_->Apply(waiting_, running_, swapped_);
            } else if (!swapped_.Empty() || !ShouldImmediatePrefill() || freeBlocksNum < reserveBlockNum4Decode) {
                priority = PDPriorityType::DECODE_FIRST;
            }

            if (needSync) {
                // cross dp sync info
                SchedulerMetric metrics = CollectSchedulerMetric();
                SchedInfo schedInfo(priority, metrics);
                std::vector<SchedInfo> globalSchedInfo =
                    PreScheduler::ShareSchedInfo(schedInfo, localDPRank_, schedulerConfig_->distributedEnable);
                // Aggregate multi dp info and make decision
                priority = PreScheduler::DecidePDPriority(globalSchedInfo);
            }
            break;
        }
        case Role::FlexP:
        case Role::P: priority = PDPriorityType::PREFILL_FIRST; break;
        case Role::FlexD:
        case Role::D: priority = PDPriorityType::DECODE_FIRST; break;
        default: throw std::runtime_error("Not a supportted role. role=" + std::to_string(static_cast<uint8_t>(role_)));
    }

    return priority;
}

ISeqGroupCollectionSPtr Scheduler::PrepCandidatesForFlex(PDPriorityType pdPriorityType, SchedulingBudget &budget)
{
    // 构造本轮可能要调度sequence group
    ISeqGroupCollectionSPtr seqGrpCollection = std::make_shared<SeqGroupCollection>(pdPriorityType);

    // get data from queues. if do decode, all data will come from running deque.
    size_t maxNumForCurIter = budget.maxNumSeqs_ * PREPARE_DATA_EXPANSION_FACTOR;
    if (role_ == Role::FlexPnD) {
        if (pdPriorityType == PDPriorityType::PREFILL_FIRST) {
            // P优先
            DequeueForFlex(waiting_, seqGrpCollection->waiting_, Role::FlexPnD, maxNumForCurIter);
            stagePolicy_->MarkInferenceStartTimeStamp(PDPriorityType::PREFILL_FIRST);
        } else if (pdPriorityType == PDPriorityType::DECODE_FIRST) {
            // D优先
            size_t queuedNum = DequeueForFlex(running_, seqGrpCollection->running_, Role::FlexPnD, maxNumForCurIter);
            Assert(maxNumForCurIter >= queuedNum);
            DequeueForFlex(swapped_, seqGrpCollection->swapped_, Role::FlexPnD, maxNumForCurIter - queuedNum);
            stagePolicy_->MarkInferenceStartTimeStamp(PDPriorityType::DECODE_FIRST);
        }
    } else if (role_ == Role::FlexP) {
        size_t queuedWaitNum = DequeueForFlex(waiting_, seqGrpCollection->waiting_, Role::FlexP, maxNumForCurIter);
        // PD分离，chunked_prefill时还需要从running队列取数据
        if (schedulerConfig_->enableChunkedPrefill) {
            DequeueForFlex(running_, seqGrpCollection->running_, Role::FlexP, maxNumForCurIter - queuedWaitNum);
        }
        stagePolicy_->MarkInferenceStartTimeStamp(PDPriorityType::PREFILL_FIRST);
    } else if (role_ == Role::FlexD) {
        size_t queuedRunNum = DequeueForFlex(running_, seqGrpCollection->running_, Role::FlexD, maxNumForCurIter);
        Assert(maxNumForCurIter >= queuedRunNum);
        DequeueForFlex(swapped_, seqGrpCollection->swapped_, Role::FlexD, maxNumForCurIter - queuedRunNum);
        stagePolicy_->MarkInferenceStartTimeStamp(PDPriorityType::DECODE_FIRST);
    }
    return seqGrpCollection;
}

ISeqGroupCollectionSPtr Scheduler::PrepCandidatesForPolicy(PDPriorityType pdPriorityType, SchedulingBudget &budget)
{
    // 构造本轮可能要调度sequence group
    ISeqGroupCollectionSPtr seqGrpCollection = std::make_shared<SeqGroupCollection>(pdPriorityType);

    // get data from queues. if do decode, all data will come from running deque.
    size_t maxNumForCurIter = budget.maxNumSeqs_ * PREPARE_DATA_EXPANSION_FACTOR;

    if (role_ == Role::PnD) {
        if (pdPriorityType == PDPriorityType::PREFILL_FIRST) {
            // P优先
            DequeueWaiting(waiting_, seqGrpCollection->waiting_, maxNumForCurIter);
        } else if (pdPriorityType == PDPriorityType::DECODE_FIRST) {
            // D优先
            size_t queuedNum = Dequeue(running_, seqGrpCollection->running_, maxNumForCurIter);
            Assert(maxNumForCurIter >= queuedNum);
            Dequeue(swapped_, seqGrpCollection->swapped_, maxNumForCurIter - queuedNum);
        } else if (pdPriorityType == PDPriorityType::MIX) {
            // chunked prefill需要从三个队列取数据
            size_t queuedRunNum = Dequeue(running_, seqGrpCollection->running_, maxNumForCurIter);
            size_t queuedWaitNum = Dequeue(waiting_, seqGrpCollection->waiting_, maxNumForCurIter - queuedRunNum);
            Dequeue(swapped_, seqGrpCollection->swapped_, maxNumForCurIter - queuedWaitNum - queuedRunNum);
        }
    } else if (role_ == Role::P) {
        size_t queuedWaitNum = Dequeue(waiting_, seqGrpCollection->waiting_, maxNumForCurIter);

        // PD分离，chunked_prefill时还需要从running队列取数据
        if (schedulerConfig_->enableChunkedPrefill) {
            Dequeue(running_, seqGrpCollection->running_, maxNumForCurIter - queuedWaitNum);
        }
    } else if (role_ == Role::D) {
        size_t queuedRunNum = Dequeue(running_, seqGrpCollection->running_, maxNumForCurIter);
        Assert(maxNumForCurIter >= queuedRunNum);
        Dequeue(swapped_, seqGrpCollection->swapped_, maxNumForCurIter - queuedRunNum);
    }
    return seqGrpCollection;
}

// only D role needs to schedule kv pulling, P role doesn't. P only needs to handle transfer events
std::shared_ptr<SeqGroupCollection> Scheduler::PrepCandidatesForKvTransferPolicy()
{
    ISeqGroupCollectionSPtr seqGrpCollection = std::make_shared<SeqGroupCollection>();

    size_t maxPrefillNumForCurIter = schedulerConfig_->maxPrefillBatchSize * 2;
    // prepare pull-kv candidates. since pull-kv needs similiar blocks as prefilling, the max candidates number is
    // same.
    if (static_cast<int32_t>(role_) <= 2) { // 小于 2， role为 PnD, P, D
        Dequeue(waiting_, seqGrpCollection->waiting_, maxPrefillNumForCurIter);
    } else {
        DequeueForFlex(waiting_, seqGrpCollection->waiting_, Role::FlexD, maxPrefillNumForCurIter);
    }
    return seqGrpCollection;
}

void Scheduler::BackfillConcurrentQueue(PolicyOutput &policyOut)
{
    // 1. update waiting request
    if (role_ == Role::PnD || role_ == Role::FlexPnD) {
        for (SequenceGroupSPtr &seqGroup : policyOut.preemptedSeqGroups_) {
            if (seqGroup->sampling->enableParallelSampling) {
                MINDIE_LLM_LOG_WARN_REQUEST(
                    "Parallel sampling does not support RECOMPUTE preemption now, request(requestId: "
                    << seqGroup->requestId << ") will be aborted!");
                abortedParallelSeqGroups_.push_back(seqGroup);
            }
            layerwiseMixin_.LwdSetRecomputeArrTime(schedulerConfig_->layerwiseDisaggregated, seqGroup,
                waiting_.Back());
        }
        // P节点不存在recompute情况；D节点Recompute需要转到P节点做
        Enqueue(waiting_, policyOut.preemptedSeqGroups_);
    }

    // 2. update new running request
    std::sort(policyOut.decodeSeqGroups_.begin(), policyOut.decodeSeqGroups_.end(),
              [](const ScheduledSequenceGroupSPtr& a, const ScheduledSequenceGroupSPtr& b) {
                  return a->seqGroup_->firstSeq->seqId_ < b->seqGroup_->firstSeq->seqId_;
              });
    Enqueue(running_, policyOut.decodeSeqGroups_);
    if (role_ == Role::P || role_ == Role::FlexP) {
        for (const ScheduledSequenceGroupSPtr &seqGroupSptr : policyOut.prefillSeqGroups_) {
            if ((schedulerConfig_->enableChunkedPrefill) && (!seqGroupSptr->seqGroup_->isLastChunk_)) {
                // if ChunkedPrefill, the unfinished prefills should backfill to running queue
                running_.PushBack(seqGroupSptr->seqGroup_);
            } else {
                // prefill seg group need to be put into transferring map, so publish and release kv is easy
                transferringMap_.Insert(seqGroupSptr->seqGroup_->firstSeq->seqId_, seqGroupSptr->seqGroup_);
            }
        }
    } else { // other cases, prefill --> running
        Enqueue(running_, policyOut.prefillSeqGroups_);
    }

    // 3. update swapped request
    Enqueue(swapped_, policyOut.swappedOutSeqGroups_);

    // 4. update remaining data to different deque
    std::sort(policyOut.withdrewSeqGroups_.begin(), policyOut.withdrewSeqGroups_.end(),
              [](const SequenceGroupSPtr& a, const SequenceGroupSPtr& b) {
                  return a->firstSeq->seqId_ < b->firstSeq->seqId_;
              });
    while (!policyOut.withdrewSeqGroups_.empty()) {
        /* withdrewSeqGroups_ 按照入队时间正序排序，所以要从队尾出放回原队列的队首，保证原队列按照入队时间排队 */
        SequenceGroupSPtr seqGroup = policyOut.withdrewSeqGroups_.back();
        switch (seqGroup->firstSeq->status_) {
            case SequenceStatus::WAITING: waiting_.PushFront(seqGroup); break;
            case SequenceStatus::RUNNING: running_.PushFront(seqGroup); break;
            case SequenceStatus::SWAPPED: swapped_.PushFront(seqGroup); break;
            case SequenceStatus::FINISH_ABORTED:
                if (!seqGroup->sampling->enableParallelSampling) {
                    throw std::runtime_error("error sequence status in remainSeqsGroups of PolictOut.");
                }
                // beam search场景下，可能原生的seqgrp没有被选中，状态已经是终止态
                running_.PushFront(seqGroup);
                break;

            default: throw std::runtime_error("error sequence status in remainSeqsGroups of PolictOut.");
        }
        policyOut.withdrewSeqGroups_.pop_back();
    }
}

/**
 * 如果mtp的推理都命中，则每轮添加place holder的个数为(1+mtp)个数
 * 如果mtp的推理没有命中，则前面调度轮次申请的mtp个数的空间没有被使用。为了防止mtp持续不命中导致的kv空间浪费，做如下处理
 *     place holder的最大个数为: 异步调度轮次 * (1+mtp)
 */
size_t Scheduler::CalculatePlaceHolderNum(ScheduledSequenceGroupSPtr seqGrpSPtr) const
{
    // 统计当前已经有的place holder个数
    std::vector<TokenId> &outPutTokenIds = seqGrpSPtr->seqGroup_->seqs_.at(0)->data_.outputTokenIds;
    size_t placeholderCount = TrailingPlaceholderTokenCount(outPutTokenIds);
    if (schedulerConfig_->speculationGamma == 0) {
        // 不支持mtp，每轮只要申请1个占位符
        return 1;
    }

    // 计算本轮需要申请的place holder的个数(1表示main module计算的token，speculationGamma表示mtp计算的token)
    size_t tokenNumPerIter = 1 + static_cast<size_t>(schedulerConfig_->speculationGamma);
    /** 支持投机解码或者mtp的时候，不明确当前轮会生成多少个token，占用多少个KV，因此提前预留一个Iter的KV空间 */

    size_t maxPlaceHolderNum = maxScheduledBatch_ * tokenNumPerIter + tokenNumPerIter;
    Assert(maxPlaceHolderNum >= placeholderCount);
    if (placeholderCount + tokenNumPerIter >= maxPlaceHolderNum) {
        return (maxPlaceHolderNum - placeholderCount);
    } else {
        return tokenNumPerIter;
    }
}

void Scheduler::BackfillConcurrentQueue(KVTransferPolicyOutput &policyOut)
{
    // add withdrewSeqGroups to different deque
    for (auto it = policyOut.withdrewSeqGroups.rbegin(); it != policyOut.withdrewSeqGroups.rend(); ++it) {
        /* withdrewSeqGroups 按照入队时间正序排序，所以要从队尾出放回原队列的队首，保证原队列按照入队时间排队 */
        switch ((*it)->firstSeq->status_) {
            case SequenceStatus::WAITING: // waiting to pull kv(in D node)
                waiting_.PushFront(*it);
                break;
            default: throw std::runtime_error("error sequence status in withdrewSeqGroups of KVTransferPolicyOutput.");
        }
    }

    // move pullSeqGroups to transferringMap
    for (ScheduledSequenceGroupSPtr kvPullSG : policyOut.pullSeqGroups) {
        transferringMap_.Insert(kvPullSG->seqGroup_->firstSeq->seqId_, kvPullSG->seqGroup_);
    }
}

/**
 * 使用异步调度，在入running队列前，将上一轮的“结果”预先更新
 * 1）插入Token id占位符。最多异步调度2个outstanding的batch，第二个batch要将前一个的batch的output token写成占位符-1
 * TBC: a sequence group may contain more than one sequence
 */
void Scheduler::AddNextTokenPlaceHolder(std::vector<ScheduledSequenceGroupSPtr> &scheduledSeqGrps) const
{
    for (auto scheduledSeqGrpSPtr : scheduledSeqGrps) {
        // ChunkedPrefill的前几块不能加占位符，最后一个chunk才可以加
        if ((schedulerConfig_->enableChunkedPrefill) && (!scheduledSeqGrpSPtr->seqGroup_->isLastChunk_)) {
            continue;
        } else {
            // 使用异步调度，上轮的token id写上占位符-1
            TokenId placeHolder = PLACEHOLDER_TOKEN;
            size_t placeHolderNum = CalculatePlaceHolderNum(scheduledSeqGrpSPtr);
            std::vector<SequenceGroupSPtr> seqGroups;
            if (scheduledSeqGrpSPtr->seqGroup_->sampling->enableParallelSampling) {
                seqGroups = scheduledSeqGrpSPtr->seqGroup_->GetParallelSeqGrp();
            } else {
                seqGroups.push_back(scheduledSeqGrpSPtr->seqGroup_);
            }
            for (auto seqGrp : seqGroups) {
                for (size_t i = 0; i < placeHolderNum; i++) {
                    seqGrp->firstSeq->data_.outputTokenIds.push_back(placeHolder);
                }
            }
        }
    }
}

/**
 * 使用异步调度，在入running队列前，将上一轮的“结果”预先更新
 * 1）累计numNewComputedTokens, 这样已经计算的tokens不会被再次计算kv， 而是使用cache里的kv
 */
void Scheduler::AccumulateComputedTokens(std::vector<ScheduledSequenceGroupSPtr> &seqGrps) const
{
    for (auto seqGrpSptr : seqGrps) {
        // 使用异步调度，直接更新numNewComputedTokens
        seqGrpSptr->seqGroup_->UpdateNumComputedTokens(seqGrpSptr->tokenChunkSize_);
    }
}

void Scheduler::UpdatePromptAndOutputTokenIds(SequenceSPtr seq)
{
    if (schedulerConfig_->enableChunkedPrefill) {
        // 重计算请求重新被调度之前，刷新promptTokenIds和outputTokenIds
        if (seq->IsPrefill() && seq->data_.outputTokenIds.size() > 0) {
            seq->data_.promptTokenIds.insert(seq->data_.promptTokenIds.end(),
                seq->data_.outputTokenIds.begin(), seq->data_.outputTokenIds.end());
            seq->data_.outputTokenIds.clear();
        }
    }
}

/**
 * 从Response tokenid map里拿到真实的tokenid，将占位符更新为正确的tokenid
 */
void Scheduler::ReplacePlaceHolderWithToken(SequenceGroupSPtr seqGrpSPtr)
{
    for (auto seq : seqGrpSPtr->GetSequences()) {
        if (schedulerConfig_->enableChunkedPrefill && !seqGrpSPtr->isLastChunk_) {
            // ChunkedPrefill的前几块不能加占位符，最后一个chunk才可以加
            continue;
        } else if (seq->IsPrefill() && predictedTokensBySeqId_.count(seq->seqId_) > 0) {
            // PD混部场景下重计算请求再次调度时，前面生成的token直接append到output tokenid里。
            // 占位符已经在重计算触发时候删除
            std::vector<TokenId> generatedTokens = predictedTokensBySeqId_[seq->seqId_];
            seq->data_.outputTokenIds.insert(seq->data_.outputTokenIds.end(), generatedTokens.begin(),
                                             generatedTokens.end());
            predictedTokensBySeqId_.erase(seq->seqId_);
            UpdatePromptAndOutputTokenIds(seq);
            continue;
        }
        UpdatePromptAndOutputTokenIds(seq);
        std::vector<TokenId> &outputTokenIds = seq->data_.outputTokenIds;
        // 第一轮prefill还没有填充占位符 或者占位符的真实token还没有回来
        if (outputTokenIds.empty() || predictedTokensBySeqId_.count(seq->seqId_) == 0) {
            continue;
        }

        // 计算place holder的个数
        size_t placeholderCount = TrailingPlaceholderTokenCount(outputTokenIds);
        std::vector<TokenId> &generatedTokens = predictedTokensBySeqId_[seq->seqId_];
        size_t numGenTokens = generatedTokens.size();
        size_t tokenNumPerIter = 1 + schedulerConfig_->speculationGamma;
        if (placeholderCount < numGenTokens ||
            placeholderCount - numGenTokens >= maxScheduledBatch_ * tokenNumPerIter + tokenNumPerIter) {
            MINDIE_LLM_LOG_ERROR("Replace place holder error, seqid: "
                                 << seq->seqId_ << ", trailingPlaceholderCount:" << placeholderCount
                                 << ", is prefill:" << seq->IsPrefill() << "; numPredictedTokens:" << numGenTokens);
            std::string newTokens = "";
            for (auto token : generatedTokens) {
                newTokens += std::to_string(token) + ", ";
            }
            MINDIE_LLM_LOG_ERROR("Replace place holder error, seqid: " << seq->seqId_
                                                                       << ", token size:" << generatedTokens.size()
                                                                       << ", new tokens:" << newTokens);
            std::string outputTokens = "";
            for (auto token : seq->data_.outputTokenIds) {
                outputTokens += std::to_string(token) + ", ";
            }
            MINDIE_LLM_LOG_ERROR("Replace place holder error, seqid: " << seq->seqId_ << ", out size:"
                                                                       << seq->data_.outputTokenIds.size()
                                                                       << ", all output tokens:" << outputTokens);
            throw std::runtime_error("The num of place holder is wrong. Check logs.");
        }

        // 替换placeholder为真实的tokenid
        Assert(outputTokenIds.size() >= placeholderCount);
        size_t placeholderStartIdx = outputTokenIds.size() - placeholderCount;
        for (size_t i = 0; i < numGenTokens; ++i) {
            outputTokenIds[placeholderStartIdx + i] = generatedTokens[i];
        }

        if (USE_PREFIX_CACHE_WITH_PLACE_HOLDER) {
            blockManager_->ReplaceTrailingPlaceHolder(seq, placeholderCount, numGenTokens);
        }
        predictedTokensBySeqId_.erase(seq->seqId_);
    }
}

SchedulerOutputs Scheduler::ConvertToSchedulerOutput(const SchedulingBudget &budget, PolicyOutput &policyOut)
{
    SchedulerOutputs schedulerOut;

    schedulerOut.numPrefillGroups_ = policyOut.numPrefillGroups;
    schedulerOut.numBatchedTokens_ = budget.numBatchedTokens_ + budget.numCachedTokens_;
    schedulerOut.numPreempted_ = policyOut.numPreempted;
    schedulerOut.runningQueueSize_ = running_.Size(); // don't know what's this for.s

    schedulerOut.blocksToSwapIn_ = std::move(policyOut.blocksToSwapIn_);
    schedulerOut.blocksToSwapOut_ = std::move(policyOut.blocksToSwapOut_);
    schedulerOut.blocksToCopy_ = std::move(policyOut.blocksToCopy_);

    // 由于text_generator的原因，chunked_prefill的时候需要把decode的数据放在前面，所以先添加decodeSeqGroups
    schedulerOut.scheduledSeqGroups_ = std::move(policyOut.decodeSeqGroups_);

    // ChunkedPrefill特性下，prefillSeqGroups需要放到scheduledSeqGroups里
    if ((schedulerConfig_->enableChunkedPrefill) &&
        (role_ == Role::P || role_ == Role::PnD || role_ == Role::FlexP || role_ == Role::FlexPnD)) {
        schedulerOut.forwardMode_ = ForwardMode::MIXED;
        schedulerOut.scheduledSeqGroups_.insert(schedulerOut.scheduledSeqGroups_.end(),
                                                policyOut.prefillSeqGroups_.begin(), policyOut.prefillSeqGroups_.end());
    } else {
        if (policyOut.numPrefillGroups != 0) {
            schedulerOut.forwardMode_ = ForwardMode::PREFILL;
            // 如果numPrefillGroups不为0，说明没有decode的sequence group，只有prefill的sequence group
            schedulerOut.scheduledSeqGroups_ = std::move(policyOut.prefillSeqGroups_);
        }
    }

    schedulerOut.ignoredSeqGroups_ = std::move(policyOut.ignoredSeqGroups_);

    schedulerOut.recomputeSeqIds_.resize(policyOut.preemptedSeqGroups_.size());
    for (size_t i = 0; i < schedulerOut.recomputeSeqIds_.size(); ++i) {
        schedulerOut.recomputeSeqIds_[i] = policyOut.preemptedSeqGroups_[i]->firstSeq->seqId_;
    }

    // 边云特性动态切块使用，给TG侧传当前waiting队列长度
    if (schedulerConfig_->layerwiseDisaggregated && schedulerOut.forwardMode_ == ForwardMode::PREFILL) {
        schedulerOut.curWaitQueueLen_ = waiting_.Size();
    }

    return schedulerOut;
}

SchedulerKVTransferOutput Scheduler::ConvertToSchedulerTransferOutput(KVTransferPolicyOutput &policyOut) const
{
    SchedulerKVTransferOutput transferOut;

    Assert(role_ == Role::D || role_ == Role::FlexD);
    transferOut.pullSeqGroups = std::move(policyOut.pullSeqGroups);

    return transferOut;
}

std::vector<BlockId> Scheduler::SetSpCpParamAndReturnAllBlocks(SequenceGroupMetaData &meta,
                                                               SequenceGroupSPtr seqGrpSPtr, SequenceId seqId,
                                                               ForwardMode forwardMode) const
{
    std::vector<BlockId> blockIds;
    std::vector<std::vector<BlockId>> allRankBlocks;
    blockManager_->GetRankedBlockIds(seqId, allRankBlocks);
    size_t maxRankBlockNum = allRankBlocks.at(0).size();
    for (std::vector<BlockId> &rankBlocks : allRankBlocks) {
        maxRankBlockNum = rankBlocks.size() > maxRankBlockNum ? rankBlocks.size() : maxRankBlockNum;
    }

    // 获取spcp rank 所有的block table的数据
    if (forwardMode == ForwardMode::PREFILL) {
        if (seqGrpSPtr->pBlockTable.empty()) {
            seqGrpSPtr->pBlockTable.resize(1);
        } else {
            seqGrpSPtr->pBlockTable[0].clear();
        }
    }

    for (auto &rankBlocks : allRankBlocks) {
        blockIds.insert(blockIds.end(), rankBlocks.begin(), rankBlocks.end());
        meta.spRankBlockNum_.push_back(rankBlocks.size());

        if (forwardMode == ForwardMode::PREFILL) {
            auto &firstBlockTable = seqGrpSPtr->pBlockTable[0];
            firstBlockTable.insert(firstBlockTable.end(), rankBlocks.begin(), rankBlocks.end());
            for (size_t i = 0; i < maxRankBlockNum - rankBlocks.size(); ++i) {
                firstBlockTable.push_back(static_cast<BlockId>(-1));
            }
        }
    }

    if (schedulerConfig_->spSize > 1) {
        meta.isSp_ = true;
    }
    if (schedulerConfig_->cpSize > 1) {
        meta.isCp_ = true;
    }
    if (schedulerConfig_->speculationGamma > 0) {
        if (forwardMode == ForwardMode::PREFILL) {
            std::vector<RankedBlockId> rankedBlockIds;
            blockManager_->GetRankedBlockIds(seqId, rankedBlockIds);
            for (RankedBlockId &rankBlocks : rankedBlockIds) {
                meta.prefillBlockRankId_.push_back(rankBlocks.rankId);
            }
        }
        meta.isMtp_ = true;
    }
    if (meta.isSp_ or meta.isCp_) {
        meta.spRankId_ = blockManager_->GetLatestAppendedRankId(seqId);
        meta.appendBlockRankId_ = blockManager_->GetAppendedBlockRankId(seqId);
        meta.spRankPromptTokenNum_ = blockManager_->GetTokenCountPerRank(seqId);
        meta.isAppendBlock_ = blockManager_->IsAppendBlock(seqId);
    }
    return blockIds;
}

std::vector<BlockId> Scheduler::LwdSetSpCpParamAndReturnAllBlocks(SequenceGroupMetaData &meta,
                                                                  SequenceGroupSPtr seqGrpSPtr, SequenceId seqId,
                                                                  ForwardMode forwardMode) const
{
    std::vector<BlockId> blockIds;
    std::vector<std::vector<BlockId>> allRankBlocks;
    blockManager_->LwdGetCloudRankedBlockIds(seqId, allRankBlocks);
    size_t maxRankBlockNum = allRankBlocks.at(0).size();
    for (std::vector<BlockId> &rankBlocks : allRankBlocks) {
        maxRankBlockNum = rankBlocks.size() > maxRankBlockNum ? rankBlocks.size() : maxRankBlockNum;
    }

    // 获取spcp rank 所有的block table的数据
    for (std::vector<BlockId> &rankBlocks : allRankBlocks) {
        blockIds.insert(blockIds.end(), rankBlocks.begin(), rankBlocks.end());
        meta.lwdCloudSpRankBlockNum_.push_back(rankBlocks.size());
    }

    if (meta.isSp_ or meta.isCp_) {
        meta.lwdCloudSpRankId_ = blockManager_->LwdGetCloudLatestAppendedRankId(seqId);
        meta.lwdCloudAppendBlockRankId_ = blockManager_->LwdGetCloudAppendedBlockRankId(seqId);
        meta.lwdCloudSpRankPromptTokenNum_ = blockManager_->LwdGetCloudTokenCountPerRank(seqId);
    }
    return blockIds;
}

std::vector<BlockIds> Scheduler::GetAllBlocks(SequenceGroupSPtr seqGrpSPtr, SequenceId seqId) const
{
    std::vector<BlockIds> blockIds;
    blockIds = blockManager_->GetBlockIds(seqId);
    if (role_ == Role::P || role_ == Role::FlexP) {
        seqGrpSPtr->pBlockTable.clear();
        seqGrpSPtr->pBlockTable = blockIds;
    }
    return blockIds;
}

void Scheduler::SetChunkedParam(SequenceSPtr seq, SequenceGroupMetaData &meta) const
{
    size_t calEndPos = seq->GetNumComputedTokens() + meta.tokenChunkSize_;
    bool isReqPrefill = (seq->GetNumComputedTokens() >= seq->data_.promptTokenIds.size()) ? false : true;
    bool isReqLastChunk = (calEndPos >= seq->data_.promptTokenIds.size()) ? true : false;

    meta.isReqPrefill_.push_back(isReqPrefill);
    meta.isReqLastChunk_.push_back(isReqLastChunk);
    meta.splitStartPos_.push_back(seq->GetNumComputedTokens());

    if (isReqPrefill) {
        size_t splitEndPos =
            (calEndPos >= seq->data_.promptTokenIds.size()) ? seq->data_.promptTokenIds.size() : calEndPos;
        meta.splitEndPos_.push_back(splitEndPos);
    } else {
        meta.splitEndPos_.push_back(calEndPos);
    }
}

void Scheduler::SetBasicMetadata(SequenceGroupMetaData &metaData, const SequenceGroupSPtr seqGroup,
                                 ScheduledSequenceGroupSPtr scheduledGrp) const
{
    metaData.requestId_ = seqGroup->requestId;
    metaData.serverid_ = seqGroup->metrics_.inferReqId_;
    metaData.samplingParams_ = seqGroup->sampling;
    metaData.ignoreEos_ = seqGroup->ignoreEos_;
    metaData.loraId_ = seqGroup->loraId_;
    metaData.skipSpecialTokens_ = seqGroup->skipSpecialTokens_;

    metaData.tokenChunkSize_ = scheduledGrp->tokenChunkSize_;
    // 边云特性动态切块使用，给TG侧传请求到达时间间隔用于切图
    if (schedulerConfig_->layerwiseDisaggregated) {
        metaData.requestGap_ = seqGroup->requestGap_;
    }
    // JSON 结构化输出约束
    metaData.responseFormat_ = seqGroup->sampling->responseFormat;
}

SequenceGroupMetaDatas Scheduler::InitSequenceGroupMetaDatas(const SchedulerOutputs &schedulerOut) const
{
    SequenceGroupMetaDatas metadatas;
    metadatas.metaList.resize(schedulerOut.scheduledSeqGroups_.size());
    metadatas.seqLenList.resize(1);
    metadatas.seqLenList[0].resize(schedulerOut.scheduledSeqGroups_.size());
    metadatas.maxBatchSize = static_cast<int64_t>(schedulerOut.scheduledSeqGroups_.size());
    return metadatas;
}

SequenceGroupMetaDatas Scheduler::GenerateSequenceGroupMetadata(const SchedulerOutputs &schedulerOut)
{
    SequenceGroupMetaDatas metadatas = InitSequenceGroupMetaDatas(schedulerOut);
    std::vector<SequenceGroupMetaData> &metaList = metadatas.metaList;

    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    for (size_t i = 0; i < schedulerOut.scheduledSeqGroups_.size(); ++i) {
        const auto scheSeqGroup = schedulerOut.scheduledSeqGroups_[i];
        const auto seqGroup = scheSeqGroup->seqGroup_;

        // 普通场景只有1个runningSeqSPtrs，是beam search会返回多个runningSeqSPtrs
        std::vector<SequenceSPtr> runningSeqSPtrs = seqGroup->GetSequences(SequenceStatus::RUNNING);
        if (runningSeqSPtrs.size() == 0) {
            MINDIE_LLM_LOG_WARN("the sequence group is not in running status. requestId=:" << seqGroup->requestId);
            continue;
        }

        SetBasicMetadata(metaList[i], seqGroup, scheSeqGroup);

        for (auto seq : runningSeqSPtrs) {
            std::vector<BlockIds> blockIds;
            std::vector<BlockId> lwdCloudBlockIds;
            bool isSimulateSeq = (seq->seqId_ == SIMULATE_SEQUENCE_ID);
            if (isSimulateSeq) {
                MINDIE_LLM_LOG_INFO("GetBlockIds called for special seqId: " << seq->seqId_);
                blockIds.push_back({static_cast<BlockId>(schedulerConfig_->npuBlockNum - 1)});
            } else if (schedulerConfig_->spSize * schedulerConfig_->cpSize <= 1) {
                blockIds = GetAllBlocks(seqGroup, seq->seqId_);
            } else {
                blockIds = {SetSpCpParamAndReturnAllBlocks(metaList[i],
                    seqGroup, seq->seqId_, schedulerOut.forwardMode_)};
                if (schedulerConfig_->layerwiseDisaggregated) {
                    lwdCloudBlockIds = LwdSetSpCpParamAndReturnAllBlocks(metaList[i],
                        seqGroup, seq->seqId_, schedulerOut.forwardMode_);
                }
            }
            std::vector<BlockIds> perSeqBlockTables = blockIds;

            // 虚推请求跳过 PrefixCache 的 LRU 访问时间更新
            if (!isSimulateSeq) {
                blockManager_->AccessAllblocksInSeq(seq, now);
            }
            metaList[i].seqIds_.push_back(seq->seqId_);
            // Accumulate per-sequence block tables into seq-group tables. Never shrink outer dimension to avoid
            // dropping previously appended manager tables in mixed/edge scenarios (e.g. simulate seq).
            const size_t targetMgrCount = std::max(metaList[i].blockIds_.size(), perSeqBlockTables.size());
            metaList[i].blockIds_.resize(targetMgrCount);
            for (size_t m = 0; m < perSeqBlockTables.size(); ++m) {
                metaList[i].blockIds_[m].reserve(metaList[i].blockIds_[m].size() + perSeqBlockTables[m].size());
                metaList[i].blockIds_[m].insert(metaList[i].blockIds_[m].end(),
                                                perSeqBlockTables[m].begin(), perSeqBlockTables[m].end());
            }
            metaList[i].lwdCloudBlockIds_.insert(metaList[i].lwdCloudBlockIds_.end(),
                lwdCloudBlockIds.begin(), lwdCloudBlockIds.end());

            if (schedulerOut.forwardMode_ == ForwardMode::MIXED ||
                (role_ == Role::P && schedulerConfig_->enableChunkedPrefill)) {
                // TBC_ 此处chunked prefill 还未适配支持beam search。
                SetChunkedParam(seq, metaList[i]);
            }

            if (schedulerOut.forwardMode_ == ForwardMode::PREFILL ||
                (schedulerOut.forwardMode_ == ForwardMode::MIXED && metaList[i].isReqPrefill_[0])) {
                // recompute 阶段要将prompt和output 的tokenid都要传下去
                metaList[i].promptLens_.push_back(seq->GetLen());
                metaList[i].tokenIds_.insert(metaList[i].tokenIds_.end(), seq->data_.promptTokenIds.begin(),
                                             seq->data_.promptTokenIds.end());
                metaList[i].tokenIds_.insert(metaList[i].tokenIds_.end(), seq->data_.outputTokenIds.begin(),
                                             seq->data_.outputTokenIds.end());
                if (seq->data_.outputTokenIds.size() > 0) {
                    // 重计算最大推理个数要减去前面已经推理的token数，
                    seqGroup->sampling->maxOutputLen =
                        seqGroup->sampling->maxOutputLen - seq->data_.outputTokenIds.size();
                }
            }
        }

        // 对于 PREFILL/MIXED 模式，计算 computed blocks 信息
        bool isSimulateSeq = seqGroup->IsSimulateRequest();
        bool needComputeBlocks = (schedulerOut.forwardMode_ == ForwardMode::PREFILL) ||
            (schedulerOut.forwardMode_ == ForwardMode::MIXED && !metaList[i].isReqPrefill_.empty());
        if (needComputeBlocks) {
            CollectOrAggregateComputedBlocks(metaList, i, runningSeqSPtrs, isSimulateSeq);
        }

        metadatas.maxSeqLen = std::max(metadatas.maxSeqLen, static_cast<int64_t>(metaList[i].tokenIds_.size()));

        metadatas.seqLenList[0][i] = 1;
        if (schedulerOut.forwardMode_ == ForwardMode::PREFILL) {
            metadatas.seqLenList[0][i] = static_cast<int64_t>(metaList[i].promptLens_[0] - \
                metaList[i].remoteComputedLens_.back() * schedulerConfig_->cacheBlockSize);
        }
    }

    return metadatas;
}

void Scheduler::CollectOrAggregateComputedBlocks(std::vector<SequenceGroupMetaData> &metaList, size_t metaIndex,
    const std::vector<SequenceSPtr> &runningSeqSPtrs, bool isSimulateSeq)
{
    // 虚推请求不参与 PrefixCache，直接填充默认值0
    if (isSimulateSeq) {
        metaList[metaIndex].computedLens_.push_back(0);
        metaList[metaIndex].remoteComputedLens_.push_back(0);
        return;
    }

    uint32_t cspSize = schedulerConfig_->spSize * schedulerConfig_->cpSize;
    if (cspSize == 1) {
        CollectComputedBlocksInfo(metaList, metaIndex, runningSeqSPtrs);
    } else {
        AggregateComputedBlocksInfo(metaList, metaIndex, runningSeqSPtrs);
    }
}

void Scheduler::CollectComputedBlocksInfo(std::vector<SequenceGroupMetaData> &metaList, size_t metaIndex,
    const std::vector<SequenceSPtr> &runningSeqSPtrs)
{
    std::vector<BlockId> computedBlocks = blockManager_->GetCommonComputedBlockIds(runningSeqSPtrs);
    std::vector<BlockId> remoteComputedBlocks;
    if (schedulerConfig_->enableKvPool) {
        remoteComputedBlocks = blockManager_->GetRemoteComputedBlockIds(runningSeqSPtrs,
            computedBlocks.size(), schedulerConfig_->tpSize, schedulerConfig_->modelName);
    } else {
        remoteComputedBlocks = computedBlocks;
    }
    metaList[metaIndex].computedLens_.push_back(computedBlocks.size());
    metaList[metaIndex].remoteComputedLens_.push_back(remoteComputedBlocks.size());
}

void Scheduler::AggregateComputedBlocksInfo(std::vector<SequenceGroupMetaData> &metaList, size_t metaIndex,
    const std::vector<SequenceSPtr> &runningSeqSPtrs)
{
    std::vector<size_t> computedBlocksNum = blockManager_->GetAllrankComputedBlockNum(runningSeqSPtrs);

    std::vector<size_t> remoteComputedBlocksNum;
    if (schedulerConfig_->enableKvPool) {
        remoteComputedBlocksNum = blockManager_->GetAllRankRemoteComputedBlockIds(runningSeqSPtrs,
            computedBlocksNum, schedulerConfig_->modelName);
    } else {
        remoteComputedBlocksNum = computedBlocksNum;
    }

    metaList[metaIndex].computedLens_.insert(metaList[metaIndex].computedLens_.end(),
        computedBlocksNum.begin(), computedBlocksNum.end());
    metaList[metaIndex].remoteComputedLens_.insert(metaList[metaIndex].remoteComputedLens_.end(),
        remoteComputedBlocksNum.begin(), remoteComputedBlocksNum.end());
}

SequenceGroupMetaDatas Scheduler::GenSeqGroupMetadata(const SchedulerKVTransferOutput &schedulerOut) const
{
    SequenceGroupMetaDatas metadatas;
    auto &scheduleSeqGroups = schedulerOut.pullSeqGroups;
    std::vector<SequenceGroupMetaData> &metaList = metadatas.metaList;
    metaList.resize(scheduleSeqGroups.size());

    for (size_t i = 0; i < scheduleSeqGroups.size(); ++i) {
        SequenceGroupSPtr seqGroup = scheduleSeqGroups[i]->seqGroup_;
        // TBC_ PD分离不支持beam search
        std::vector<SequenceSPtr> runningSeqs = seqGroup->GetSequences(SequenceStatus::RUNNING);

        SetBasicMetadata(metaList[i], seqGroup, scheduleSeqGroups[i]);

        for (auto &seq : runningSeqs) {
            std::vector<BlockIds> blockIds = blockManager_->GetBlockIds(seq->seqId_);

            metaList[i].seqIds_.push_back(seq->seqId_);

            metaList[i].blockIds_.resize(blockIds.size());
            for (size_t j = 0; j < blockIds.size(); ++j) {
                metaList[i].blockIds_[j].reserve(metaList[i].blockIds_[j].size() + blockIds[j].size());
                metaList[i].blockIds_[j].insert(metaList[i].blockIds_[j].end(), blockIds[j].begin(), blockIds[j].end());
            }

            metaList[i].promptLens_.push_back(seq->data_.promptTokenIds.size());
            metaList[i].tokenIds_.insert(metaList[i].tokenIds_.end(), seq->data_.promptTokenIds.begin(),
                                         seq->data_.promptTokenIds.end());
            metaList[i].dpInstanceId_ = seqGroup->dpInstanceId_;

            metaList[i].srcBlockIds_.resize(seqGroup->pBlockTable.size());
            for (size_t j = 0; j < seqGroup->pBlockTable.size(); ++j) {
                metaList[i].srcBlockIds_[j].reserve(
                    metaList[i].srcBlockIds_[j].size() + seqGroup->pBlockTable[j].size());
                metaList[i].srcBlockIds_[j].insert(metaList[i].srcBlockIds_[j].end(),
                                                   seqGroup->pBlockTable[j].begin(), seqGroup->pBlockTable[j].end());
            }
        }

        metadatas.maxSeqLen = std::max(metadatas.maxSeqLen, static_cast<int64_t>(metaList[i].tokenIds_.size()));
    }
    metadatas.maxBatchSize = static_cast<int64_t>(scheduleSeqGroups.size());

    return metadatas;
}

template <typename T> void Scheduler::PopAndSave_(ConcurrentDeque<T> &src, std::unordered_set<T> &dst) const
{
    while (!src.Empty()) {
        T resId = T{}; // if Request is int, it is not initialized.
        src.PopFront(resId);
        if (dst.count(resId) != 0) {
            MINDIE_LLM_LOG_INFO_REQUEST("Request(id:" << resId << ") is already in the finished/aborted queue.");
        } else {
            dst.insert(resId);
        }
    }
}

/**
 * PD分离场景下，P节点做完prefill请求就完成的场景，回收KV cache。
 * 请求结束不需要D节点执行，不需要给service上送IBISSCHEDULER_PUBLISH_COMPLETED的Response。KV的清理在P节点内完成。
 * 例如：maxIterTimes配置为1，只需要推理1个token，请求就完成。
 */
template <typename T> void Scheduler::LifeEndKVCleanup(std::unordered_set<T> &lifeEndSet)
{
    // P节点做完prefill之后，不再入队。对于做完prefill生命周期就结束的请求，要回收kv
    if (role_ != Role::P || static_cast<int32_t>(role_) > 2) { // 2 表示 role 为 FlexD, FlexP, FlexPnD
        return;
    }
    auto setSize = lifeEndSet.size();
    uint64_t processNum = 0;
    for (auto it = lifeEndSet.begin(); processNum < setSize && it != lifeEndSet.end();) {
        SequenceGroupSPtr seqGrpSPtr = LiveInferContext::GetInstance(localDPRank_)->GetSeqGroup(*it);
        if (seqGrpSPtr == nullptr || transferringMap_.Count(seqGrpSPtr->firstSeq->seqId_) == 0) {
            // 还在waiting的请求保留放在aborted Set，由调度线程调度waiting时候清理
            ++it;
            processNum++;
            continue;
        }
        // flex 场景下，此处由于跟主线程可能不同步，需要校验当前请求是否是 flexp 类型
        if (static_cast<int32_t>(role_) > 2 && // role为 FlexD, FlexP, FlexPnD
            LiveInferContext::GetInstance(localDPRank_)->GetInferInstanceFlexRole4Req(seqGrpSPtr->requestId) ==
                Role::FlexP) {
            if (seqGrpSPtr->pInstanceId != localDPRank_) {
                // flex P节点只清理本节点的请求
                ++it;
                processNum++;
                continue;
            }
        }
        // 已经在transferring map的请求加入到pulled清理队列由后面调度线程统一清理KV.
        this->kvCachePulledSeqIds_.PushBack(seqGrpSPtr->firstSeq->seqId_);
        it = lifeEndSet.erase(it);
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Life End, add to release-kv queue] Add to pulled. requestId: "
                            << seqGrpSPtr->metrics_.inferReqId_
                            << "; seqId: " << seqGrpSPtr->firstSeq->seqId_);
        processNum++;
    }
}

void Scheduler::NotifyMeKvPulledSeqIds(SequenceId seqId) { this->kvCachePulledSeqIds_.PushBack(seqId); }

std::unordered_set<SequenceId> &Scheduler::FetchFinishedSeqIds(ConcurrentDeque<SequenceId> &finishedSeqIds)
{
    PopAndSave_(finishedSeqIds, this->finishedSeqIds_); // dst is unordered set
    LifeEndKVCleanup(this->finishedSeqIds_);
    return finishedSeqIds_;
}

std::unordered_set<SequenceId> &Scheduler::FetchExceptionSeqIds(ConcurrentDeque<SequenceId> &exceptionSeqIds)
{
    PopAndSave_(exceptionSeqIds, this->exceptionSeqIds_); // dst is unordered set
    LifeEndKVCleanup(this->exceptionSeqIds_);
    return exceptionSeqIds_;
}

/**
 * 将新的aborted请求加入到abortedReqIds_队列，老的已经转换到sequence id的aborted请求返回给engine做资源清理。
 */
std::unordered_set<RequestId> &Scheduler::FetchAbortedReqIds(ConcurrentDeque<RequestId> &abortedReqIds)
{
    PopAndSave_(abortedReqIds, this->abortedReqIds_); // dst is unordered set
    LifeEndKVCleanup(this->abortedReqIds_);
    return abortedReqIds_;
}

void Scheduler::KVPulledReqEnterRunningQueue(ConcurrentDeque<RequestId> &pulledReqIds)
{
    RequestId reqId{};
    while (!pulledReqIds.Empty()) {
        bool isSucc = pulledReqIds.PopFront(reqId);
        if (!isSucc) {
            throw std::runtime_error("Pop pulledReqIds failed.");
        }

        SequenceGroupSPtr seqGrpSPtr = LiveInferContext::GetInstance(localDPRank_)->GetSeqGroup(reqId);
        if (!seqGrpSPtr) {
            MINDIE_LLM_LOG_WARN("Pull kv finished, but request has been aborted. RequestID:" << reqId);
            continue;
        }
        auto prof = PROF(INFO, Domain("Schedule").Resource(seqGrpSPtr->requestId));
        running_.PushBack(seqGrpSPtr);
        PROF(prof.Metric("QueueSize", running_.Size()).Attr("status", "running").Event("Enqueue"));
        transferringMap_.Erase(seqGrpSPtr->firstSeq->seqId_);
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Request-Enter running queue] DP RankId: "
                            << dpRankId_ << ". Pull kv ended, enter running queue. requestId: "
                            << seqGrpSPtr->metrics_.inferReqId_
                            << "; seqId: " << seqGrpSPtr->firstSeq->seqId_
                            << "; running size:" << running_.Size() << "; waiting size: " << waiting_.Size()
                            << "; swapped size:" << swapped_.Size()
                            << "; transferring size:" << transferringMap_.Size());
    }
}

void Scheduler::FetchSeqGeneratedTokens(ConcurrentDeque<std::pair<SequenceId, TokenId>> &seqIdToOutputTokenQueue)
{
    while (!seqIdToOutputTokenQueue.Empty()) {
        std::pair<SequenceId, TokenId> result;
        seqIdToOutputTokenQueue.PopFront(result);
        auto seqId = result.first;
        auto token = result.second;
        // 如果请求已经被abort，则丢弃对应的token
        // beam search或者并行采样的子seqgrp在RootMap中
        LiveInferContextSPtr contextSPtr = LiveInferContext::GetInstance(localDPRank_);
        if (contextSPtr->GetSeqGroup(seqId) || contextSPtr->GetSeqGroupFromSeqRootMap(seqId)) {
            // 边云协同场景特殊情况需要丢弃
            if (LayerwiseDiscardToken(contextSPtr, seqId)) {
                continue;
            }
            // ChunkedPrefill前几块的output token需要丢弃，最后一块才加入
            if (isDiscardOutputToken(contextSPtr, seqId)) {
                continue;
            } else {
                predictedTokensBySeqId_[seqId].push_back(token);
            }
        }
    }
}

bool Scheduler::LayerwiseDiscardToken(LiveInferContextSPtr &contextSPtr, SequenceId seqId)
{
    if (schedulerConfig_->layerwiseDisaggregated && !schedulerConfig_->enableChunkedPrefill
        && contextSPtr->GetSeqGroup(seqId) != nullptr
        && contextSPtr->GetSeqGroup(seqId)->firstSeq->data_.layerwiseDiscard_) {
        // 边云协同场景特殊情况需要丢弃
        contextSPtr->GetSeqGroup(seqId)->firstSeq->data_.layerwiseDiscard_ = false;
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|scheduler] "<<"seq id= " << seqId << ", is discarded");
        return true;
    }
    return false;
}

bool Scheduler::isDiscardOutputToken(LiveInferContextSPtr &contextSPtr, SequenceId seqId)
{
    if (!schedulerConfig_->enableChunkedPrefill) {
        // 不开chunkedprefill时，不会丢弃token，返回false
        return false;
    } else if (contextSPtr->GetSeqGroup(seqId) != nullptr) {
        // 不开beamsearch时，segGrp用GetSeqGroup获取
        return !contextSPtr->GetSeqGroup(seqId)->isLastChunk_;
    } else if (contextSPtr->GetSeqGroupFromSeqRootMap(seqId) != nullptr) {
        // 开beamsearch时，segGrp用GetSeqGroupFromSeqRootMap获取
        return !contextSPtr->GetSeqGroupFromSeqRootMap(seqId)->isLastChunk_;
    }
    return false;
}

/**
 * 获取Sequence grp的状态信息（是否被abort/是否response返回已经结束）
 */
SequenceStatus Scheduler::FinalizeSeqGrpStatus(SequenceGroupSPtr seqGroup)
{
    // 用户下发abort，生命周期要结束
    if ((abortedReqIds_.count(seqGroup->requestId) > 0)) {
        return SequenceStatus::FINISH_ABORTED;
    }

    if (seqGroup->sampling->enableParallelSampling) {
        // 开启beam search，生命周期结束是所有的parallel的seqgrp都完成
        if (seqGroup->seqId2ParallelSeqGroup_.Size() == 0) {
            return SequenceStatus::FINISH_STOPPED;
        }
        return SequenceStatus::RUNNING;
    }

    // 虚推请求使用固定seqId，需要跳过finishedSeqIds_和exceptionSeqIds_检查
    // 避免上一轮虚推的seqId残留导致当前虚推被误判为已完成或异常
    // 只在P节点（P/FlexP）应用此逻辑，D节点不需要
    bool isSimulateInference = seqGroup->IsSimulateRequest();
    bool isPNode = (role_ == Role::P || role_ == Role::FlexP);
    if (isSimulateInference && isPNode) {
        bool inFinished = finishedSeqIds_.count(seqGroup->seqs_[0]->seqId_) > 0;
        bool inException = exceptionSeqIds_.count(seqGroup->seqs_[0]->seqId_) > 0;
        if (inFinished || inException) {
            MINDIE_LLM_LOG_INFO("[SimulateInference] P node skip status check. "
                                << "seqId=" << seqGroup->seqs_[0]->seqId_
                                << ", inFinishedSeqIds=" << inFinished
                                << ", inExceptionSeqIds=" << inException
                                << ", requestId=" << seqGroup->requestId);
        }
        // P节点跳过检查，直接返回RUNNING
        return SequenceStatus::RUNNING;
    }

    if (finishedSeqIds_.count(seqGroup->seqs_[0]->seqId_) > 0) {
        return SequenceStatus::FINISH_STOPPED;
    }

    if (exceptionSeqIds_.count(seqGroup->seqs_[0]->seqId_) > 0) {
        return SequenceStatus::FINISH_ABORTED;
    }
    return SequenceStatus::RUNNING;
}

void Scheduler::ClearSeq(SequenceId seqId)
{
    exceptionSeqIds_.erase(seqId);
    finishedSeqIds_.erase(seqId);
    // 生命周期结束，则将残留的tokenid清理掉
    if (predictedTokensBySeqId_.count(seqId) > 0) {
        predictedTokensBySeqId_.erase(seqId);
    }
    blockManager_->Free(seqId);
}

void Scheduler::ClearSeqGrp(SequenceGroupSPtr seqGroup, SequenceStatus finalStatus)
{
    // sequence group处于终止态，则删除各个容器中的资源（有可能被aborted的同时Response的返回也是终止态）
    LiveInferContext::GetInstance(localDPRank_)->Remove(seqGroup->requestId);
    MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Request-Life End] Request life endup. DP RankId: "
                        << dpRankId_
                        << ". requestId: " << seqGroup->metrics_.inferReqId_
                        << "; seqId: " << seqGroup->firstSeq->seqId_
                        << "; final status:" << static_cast<int>(finalStatus));
    abortedReqIds_.erase(seqGroup->requestId);
    for (auto &seq : seqGroup->GetFirstSequence()) {
        ClearSeq(seq->seqId_);
        seq->status_ = finalStatus;
    }
    PROF(INFO, Domain("KVCache")
                    .Resource(seqGroup->requestId)
                    .Metric("deviceBlock", blockManager_->GetNumFreeNpuBlocks())
                    .Metric("hostBlock", blockManager_->GetNumFreeCpuBlocks())
                    .MetricScope("dp", blockManager_->GetLocalDPRank())
                    .Event("Free"));
}

/**
并行采样，对于没有被选中的seqId2ParallelSeqGroup_中的seqgrp，需要释放对应的资源
seqId2ParallelSeqGroup_ 只在这个函数删除，其他地方不允许做删除操作
 */
void Scheduler::ParallelSeqGroupLifeEnd(SequenceGroupSPtr seqGroup)
{
    std::vector<SequenceId> parallelSeqIds = seqGroup->seqId2ParallelSeqGroup_.KeySet();
    std::vector<SequenceId> seqIdsToClear{};
    bool isAbortedRequest = abortedReqIds_.count(seqGroup->requestId) > 0;
    for (auto seqId : parallelSeqIds) {
        optional<SequenceGroupSPtr> parallelSeqGrpOpt = seqGroup->seqId2ParallelSeqGroup_.Get(seqId);
        if (parallelSeqGrpOpt.has_value()) {
            SequenceGroupSPtr parallelSeqGrp = parallelSeqGrpOpt.value();
            if (!isAbortedRequest && parallelSeqGrp->needUpdate_) {
                // 并行采样时，释放父节点的 block 在分配子节点的 block 之前，因此必须先保存父节点的 blockIds
                parallelSeqGrp->parentBlockIds_ = blockManager_->GetBlockIds(parallelSeqGrp->parentSeqId_);
            }
            if (isAbortedRequest || finishedSeqIds_.count(seqId) > 0 || exceptionSeqIds_.count(seqId) > 0) {
                // 置位状态为了root seqgrp（在running队列中）在调度中不会申请block
                // 非root seqgrp会在从map中删除后引用计数为0而被析构
                parallelSeqGrp->firstSeq->status_ = SequenceStatus::FINISH_ABORTED;
                seqIdsToClear.push_back(seqId);
            }
        } else {
            throw std::runtime_error("ParallelSeqGroup has been deleted!");
        }
    }
    for (auto seqId : seqIdsToClear) {
        ClearSeq(seqId);
        seqGroup->seqId2ParallelSeqGroup_.Erase(seqId);
        LiveInferContext::GetInstance(localDPRank_)->RemoveFromSeqRootMap(seqId);
    }
}

// if life ended, wrap up the seqgroup into bin, and release blocks occupied
// return true if life ended, otherwise false
bool Scheduler::LifeEndedWrapup_(SequenceGroupSPtr &seqGroup)
{
    // 处理beam search或者并行采样的parallel的seqgrp的资源回收
    if (seqGroup->sampling->enableParallelSampling) {
        ParallelSeqGroupLifeEnd(seqGroup);
    }

    SequenceStatus finalStatus = FinalizeSeqGrpStatus(seqGroup);
    if (finalStatus == SequenceStatus::RUNNING) {
        return false;
    }

    if (abortedReqIds_.count(seqGroup->requestId) > 0) {
        abortedSequenceIds_.insert(seqGroup->seqs_[0]->seqId_);
    }

    ClearSeqGrp(seqGroup, finalStatus);
    return true;
}

// dequeue concurrent queue, and check if it is finished or aborted
// when seqgroup life ended, wrap up it.
size_t Scheduler::Dequeue(ConcurrentDeque<SequenceGroupSPtr> &srcQueue, std::deque<SequenceGroupSPtr> &dstDeque,
                          const size_t maxNum)
{
    size_t actualNum = 0;

    // repeat until empty or pop enough
    while (!srcQueue.Empty() && dstDeque.size() < maxNum) {
        SequenceGroupSPtr sgPtr = nullptr;
        srcQueue.PopFront(sgPtr);

        // 更新token id占位符为正确的token id
        ReplacePlaceHolderWithToken(sgPtr);
        bool terminated = LifeEndedWrapup_(sgPtr);
        if (!terminated) {
            dstDeque.push_back(sgPtr);
            actualNum++;
        }
    }

    return actualNum;
}

size_t Scheduler::DequeueWaiting(ConcurrentDeque<SequenceGroupSPtr> &srcQueue, std::deque<SequenceGroupSPtr> &dstDeque,
                                 const size_t maxNum)
{
    size_t actualNum = 0;
    std::deque<SequenceGroupSPtr> recomputeReqs;

    // repeat until empty or pop enough
    while (!srcQueue.Empty() && dstDeque.size() < maxNum) {
        SequenceGroupSPtr sgPtr = nullptr;
        srcQueue.PopFront(sgPtr);

        // 更新token id占位符为正确的token id
        ReplacePlaceHolderWithToken(sgPtr);
        bool terminated = LifeEndedWrapup_(sgPtr);
        if (!terminated) {
            if (sgPtr->seqs_.at(0)->data_.outputTokenIds.size() > 0) {
                recomputeReqs.push_back(sgPtr);
            } else {
                dstDeque.push_back(sgPtr);
            }
            actualNum++;
        }
    }
    while (!recomputeReqs.empty()) {
        dstDeque.push_back(recomputeReqs.front());
        recomputeReqs.pop_front();
    }

    return actualNum;
}

void Scheduler::Enqueue(ConcurrentDeque<SequenceGroupSPtr> &dstQueue, std::vector<SequenceGroupSPtr> &seqGroups,
                        bool front) const
{
    if (seqGroups.empty()) {
        return;
    }

    // push element to deque
    if (front) {
        for (SequenceGroupSPtr &seq : seqGroups) {
            dstQueue.PushFront(seq);
        }
    } else {
        for (SequenceGroupSPtr &seq : seqGroups) {
            dstQueue.PushBack(seq);
        }
    }
}

void Scheduler::Enqueue(ConcurrentDeque<SequenceGroupSPtr> &dstQueue,
                        std::vector<std::shared_ptr<ScheduledSequenceGroup>> &scheduleSeqGroups, bool front) const
{
    // 1. check seqGroup size
    if (scheduleSeqGroups.empty()) {
        return;
    }

    // 2. push element to deque
    if (front) {
        for (const auto &seq : scheduleSeqGroups) {
            dstQueue.PushFront(seq->seqGroup_);
        }
    } else {
        for (const auto &seq : scheduleSeqGroups) {
            dstQueue.PushBack(seq->seqGroup_);
        }
    }
}

SchedulerPtr MakeScheduler(SchedulerConfigSPtr schedulerConfig, std::shared_ptr<LatencyPredictor> latencypredictor,
                           Role pdRole, size_t localDPRank)
{
    return std::make_unique<Scheduler>(schedulerConfig, latencypredictor, pdRole, localDPRank);
}

void Scheduler::MarkLastScheduleEmpty() { lastScheduleEmpty_ = true; }

void Scheduler::ClearLastScheduleEmpty() { lastScheduleEmpty_ = false; }

void Scheduler::ClearQueueAndSendAbortedResponse(ConcurrentDeque<SequenceGroupSPtr> &srcQueue)
{
    while (!srcQueue.Empty()) {
        SequenceGroupSPtr sgPtr = nullptr;
        srcQueue.PopFront(sgPtr);
        abortedParallelSeqGroups_.push_back(sgPtr);
        ClearSeqGrp(sgPtr, SequenceStatus::FINISH_ABORTED);
    }
}

void Scheduler::StopRunningRequest()
{
    ClearQueueAndSendAbortedResponse(waiting_);

    // Clear transferring map
    std::vector<SequenceId> seqIds = transferringMap_.KeySet();
    for (auto seqId : seqIds) {
        SequenceGroupSPtr sgPtr = transferringMap_.Get(seqId).value_or(nullptr);
        if (sgPtr != nullptr) {
            transferringMap_.Erase(seqId);
            abortedParallelSeqGroups_.push_back(sgPtr);
            ClearSeqGrp(sgPtr, SequenceStatus::FINISH_ABORTED);
        }
    }
    ClearQueueAndSendAbortedResponse(running_);
    ClearQueueAndSendAbortedResponse(swapped_);
    LiveInferContext::GetInstance(localDPRank_)->RemoveAll();
    MINDIE_LLM_LOG_DEBUG(
        "[Scheduler] Cleared all running, swapped, waiting and transferring requests (status=FINISH_ABORTED).]");
}

std::unordered_set<SequenceId> Scheduler::ClearAndReturnTerminatedSeqIds()
{
    std::unordered_set<SequenceId> seqIds = abortedSequenceIds_;
    abortedSequenceIds_.clear();
    return seqIds;
}

SchedulerMetric Scheduler::CollectSchedulerMetric()
{
    SchedulerMetric schedulerMetric{};
    uint32_t cspSize = schedulerConfig_->spSize * schedulerConfig_->cpSize;
    schedulerMetric.blockInfo.totalCpuBlockNum_ = schedulerConfig_->cpuBlockNum * cspSize;
    schedulerMetric.blockInfo.totalNpuBlockNum_ = schedulerConfig_->npuBlockNum * cspSize;
    schedulerMetric.blockInfo.freeNpuBlockNum_ = schedulerMetricsStatics_.freeNpuBlockNum_;
    schedulerMetric.blockInfo.freeCpuBlockNum_ = schedulerMetricsStatics_.freeCpuBlockNum_;

    schedulerMetric.reqsInfo.allRadixMatchNum_ = schedulerMetricsStatics_.allRadixMatchNum_.load();
    schedulerMetric.reqsInfo.cumulativePreemptCount_ = schedulerMetricsStatics_.cumulativePreemptCount_.load();
    schedulerMetric.reqsInfo.npuRadixMatchHitNum_ = schedulerMetricsStatics_.npuRadixMatchHitNum_.load();
    schedulerMetric.reqsInfo.runningRequestNum_ = schedulerMetricsStatics_.runningRequestNum_.load();
    schedulerMetric.reqsInfo.swappedRequestNum_ = schedulerMetricsStatics_.swappedRequestNum_.load();
    schedulerMetric.reqsInfo.waitingRequestNum_ = schedulerMetricsStatics_.waitingRequestNum_.load();

    return schedulerMetric;
}

bool MatchesFilter(const SequenceGroupSPtr &sg, Role role)
{
    switch (role) {
        case Role::FlexP: return !sg->isDecode_ && !sg->isFlexLocal_;
        case Role::FlexD: return sg->isDecode_ && !sg->isFlexLocal_;
        case Role::FlexPnD: return sg->isFlexLocal_;
        default: return false;
    }
}

size_t Scheduler::DequeueForFlex(ConcurrentDeque<SequenceGroupSPtr> &srcQueue, std::deque<SequenceGroupSPtr> &dstDeque,
                                 Role role, const size_t maxNum)
{
    size_t actualNum = 0;
    size_t queueSize = srcQueue.Size();
    size_t processSeqNum = 0;

    while (processSeqNum < queueSize && !srcQueue.Empty() && dstDeque.size() < maxNum) {
        SequenceGroupSPtr sgPtr = nullptr;
        srcQueue.PopFront(sgPtr);

        if (MatchesFilter(sgPtr, role)) {
            ReplacePlaceHolderWithToken(sgPtr);
            bool terminated = LifeEndedWrapup_(sgPtr);
            if (!terminated) {
                LiveInferContext::GetInstance(localDPRank_)->SetInferInstanceFlexRole4Req(sgPtr->requestId, role);
                dstDeque.push_back(sgPtr);
                actualNum++;
            }
        } else {
            srcQueue.PushBack(sgPtr);
        }

        processSeqNum++;
    }

    return actualNum;
}

void Scheduler::SetPrefillPercentage(uint32_t prefillPercentage)
{
    stagePolicy_->SetPrefillPercentage(prefillPercentage);
}
Role Scheduler::SwitchRole()
{
    role_ = stagePolicy_->GetFlexRole(waiting_, running_, swapped_);
    return role_;
}

std::shared_ptr<StagePolicy> Scheduler::GetStagePolicy() { return stagePolicy_; }

void Scheduler::CollectAndClearAbortedParallelSeqGroups()
{
    for (SequenceGroupSPtr &seqGroup : abortedParallelSeqGroups_) {
        RequestId reqId = seqGroup->requestId;
        if (!abortedReqIds_.insert(reqId).second) {
            MINDIE_LLM_LOG_WARN("Request(id:" << reqId << ") is already in the abortedReqIds queue.");
        }
    }
    abortedParallelSeqGroups_.clear();
}

std::vector<SequenceGroupSPtr> &Scheduler::GetAbortedParallelSeqGroups() { return abortedParallelSeqGroups_; }

} // namespace mindie_llm
