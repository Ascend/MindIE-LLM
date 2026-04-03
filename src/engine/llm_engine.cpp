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
#include "llm_engine.h"

#include <unordered_set>

#include "construct_execute_request.h"
#include "live_infer_context.h"
#include "id_utils.h"
#include "log.h"
#include "request_response/request_id.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "thread_group_cc.h"
#include "process_group.h"
#include "common_util.h"
#include "post_scheduler.h"
#include "policy/stage_policy/stage_policy.h"
#include "policy/stage_policy/edge_cloud_policy.h"
#include "policy/dynamic_batch_recorder.h"
#include "error_queue.h"
using namespace std;
using namespace std::chrono;

namespace mindie_llm {
constexpr int LOG_TIME_THRESHOLD_MS = 1000;
constexpr int LOG_CC_TIME_THRESHOLD_MS = 10;
LlmEngine::LlmEngine(SchedulerConfig schedulerConfig, std::vector<IExecutorSPtr> executors,
                     ForwardRespToManagerCall cb, Role pdRole)
{
    if (executors.empty()) {
        throw std::invalid_argument("At lease one executor is needed");
    }
    schedulerConfig_ = std::make_shared<SchedulerConfig>(schedulerConfig);
    if (schedulerConfig_->distributedEnable) { // 只有多节点分布式才需要进程间通信，多dp集中式使用线程间通信
        uint32_t numNpuPerDp = schedulerConfig_->globalWorldSize / schedulerConfig_->dpSize;
        if (numNpuPerDp == 0) {
            throw runtime_error("NumNpuPerDp should not be zero.");
        }
        schedulerConfig_->tpSize = numNpuPerDp / schedulerConfig_->cpSize;
        // 每个P/D实例占用tpSize个device，globalRandIds是从0递增的，因此dpRankId=globalRankIds[0] / (tpSize * cpSize)
        dpRankId_ = static_cast<int>(std::stoull(schedulerConfig_->globalRankIds[0]) / numNpuPerDp);
    } else if (schedulerConfig_->worldSize != schedulerConfig_->globalWorldSize &&
        schedulerConfig_->globalWorldSize > 0) {
        schedulerConfig_->tpSize = schedulerConfig_->globalWorldSize / schedulerConfig_->dpSize;
    } else {
        schedulerConfig_->tpSize = schedulerConfig_->worldSize / schedulerConfig_->dpSize;
    }
    if (schedulerConfig_->tpSize == 0) {
        MINDIE_LLM_LOG_ERROR("`tpSize` should not be 0!");
    }
    role_ = pdRole;
    for (size_t i = 0; i < executors.size(); ++i) {
        EnginePerDPSPtr enginePerDP = std::make_shared<EnginePerDP>();
        enginePerDPs_.emplace_back(enginePerDP);

        IExecutorSPtr executor = executors[i];
        enginePerDP->modelExecutor = executor;
        enginePerDP->latencypredictor = std::make_shared<LatencyPredictor>();
        schedulerConfig_->dpRankId_ = (dpRankId_ > 0 ? dpRankId_ : static_cast<int>(i));
        enginePerDP->scheduler = MakeScheduler(schedulerConfig_, enginePerDP->latencypredictor, pdRole, i);
        enginePerDP->abortRespToManagerCall = cb;
        enginePerDP->modelExecOutputHandler =
            std::make_unique<ModelExecOutputHandler>(cb, pdRole, schedulerConfig_, enginePerDP->latencypredictor, i);
        // 配置 SetStagePolicy， 在flex场景中， modelExecOutputHandler在请求完成时记录时间戳
        enginePerDP->modelExecOutputHandler->SetStagePolicy(enginePerDP->scheduler->GetStagePolicy());
        enginePerDP->transferOutputHandler = std::make_unique<TransferOutputHandler>(cb, i);

        // 按需注册 LatencyPredictor 到 DynamicBatchRecorder
        if (schedulerConfig_->stageSelectPolicy == static_cast<uint32_t>(StagePolicyType::LATENCY_FIRST) ||
            schedulerConfig_->dynamicBatchSizeEnable) {
            auto &recorder = DynamicBatchRecorder::GetInstance(i);
            recorder.SetLatencyPredictor(enginePerDP->latencypredictor);
        }
    }

    // 分布式executors.size()只有1，
    // 集中式主节点才有Engine实例，只要size大于1，就需要初始化线程集合通信
    if (executors.size() > 1) {
        ThreadGroupCC::GetInstance(executors.size());
    }
    // TBC_暂时仅支持双机主从模式，PD分离不支持
    if (role_ == Role::PnD && schedulerConfig_->isMultiNodeInfer && enginePerDPs_.size() > 1) {
        loadBalancer_ = MakeLoadBalancer(enginePerDPs_, schedulerConfig_->maxPrefillBatchSize);
    }

    // 构造loraManager
    mindie_llm::LoraManager::Initialize(executors, schedulerConfig_->maxLoras);

    MINDIE_LLM_LOG_INFO("LlmEngine init succeeds! " << enginePerDPs_.size() << " enginePerDPs are created. "
                                                    << "Need to call StartEngineThread to start Engine thread(s).");
}

LlmEngine::~LlmEngine()
{
    Stop();
    for (std::shared_ptr<EnginePerDP> &enginePerDP : enginePerDPs_) {
        if (enginePerDP->schedulerThread.joinable()) {
            enginePerDP->schedulerThread.join();
        }
    }
}

bool LlmEngine::AddRequest(RequestSPtr request)
{
    if (stop_.load(std::memory_order_relaxed)) {
        MINDIE_LLM_LOG_DEBUG("[LlmEngine]Engine hasn't initialized. Cann't add request:" << request->requestId);
        return false;
    }

    // 存放映射，用于返回给manager层的Response构造
    static size_t rrIdx = 0;
    size_t rankId = (rrIdx++) % enginePerDPs_.size();
    auto seqGroup =
        SeqGroupBuilderFromInferReq::BuildSequenceGroup(request, schedulerConfig_, rankId);
    seqGroup->isDecode_ = request->reqType == InferReqType::REQ_DECODE;
    seqGroup->isFlexLocal_ = request->reqType == InferReqType::REQ_FLEX_LOCAL;

    // TBC_暂时仅支持双机主从模式，PD分离不支持
    if (role_ == Role::PnD && schedulerConfig_->isMultiNodeInfer && enginePerDPs_.size() > 1) {
        loadBalancer_->AddSeqGroup(seqGroup);
    } else {
        EnginePerDPSPtr &engine = enginePerDPs_.at(rankId);
        engine->scheduler->AddSeqGroup(seqGroup);
        engine->addedRequestNum++;
    }
    return true;
}

void LlmEngine::SendAbortResponse(SequenceGroupSPtr seqGroup, size_t localDPRank, InferStatusType flag) const
{
    ResponseSPtr response = std::make_shared<Response>(seqGroup->metrics_.inferReqId_);
    response->isEos = true;
    response->inferStatusFlag = flag;
    response->iterTimes = seqGroup->iterTimes;
    ResponseContent content;
    content.seqId = seqGroup->firstSeq->seqId_;
    content.parentSeqId = seqGroup->firstSeq->seqId_;
    content.finishReason = InferStatusType::ABORT;
    content.outTokenIds = std::vector<TokenId>(1, -1);
    content.speculativeTokenNum = 1;
    response->responseContents.emplace_back(content);

    enginePerDPs_.at(localDPRank)->abortRespToManagerCall(response);
}

void LlmEngine::AbortParallelSeqGroups([[maybe_unused]] size_t localDPRank) const
{
    for (SequenceGroupSPtr &seqGroup : enginePerDPs_.at(localDPRank)->scheduler->GetAbortedParallelSeqGroups()) {
        SendAbortResponse(seqGroup, localDPRank, InferStatusType::ABORT);
    }
}

void LlmEngine::AbortRequests(std::unordered_set<RequestId> &requestIds)
{
    for (RequestId reqId : requestIds) {
        auto [localDPRank, seqGrpSptr] = LiveInferContext::FindSeqGroupInAllRank(reqId);
        if (seqGrpSptr == nullptr) {
            MINDIE_LLM_LOG_WARN_REQUEST("[LlmEngine]Abort request(requestId: "
                                 << reqId << ") does not exist. This request may have been kv-released.");
            continue;
        }

        SendAbortResponse(seqGrpSptr, localDPRank, InferStatusType::ABORT);
        // PD分离场景，加入到abortedRequestIds后，后续调度线程会回收transferringMap_，
        enginePerDPs_.at(localDPRank)->abortedRequestIds.PushBack(reqId);
        enginePerDPs_.at(localDPRank)->abortedRequestNum++;
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine]Abort request(requestId: "
                            << reqId << ") successfully."
                            << " Total aborted request num is:" << enginePerDPs_.at(localDPRank)->abortedRequestNum);
    }
}

void LlmEngine::ReleaseKvCache(std::unordered_set<RequestId> &requestIds)
{
    for (auto reqId : requestIds) {
        auto [localDPRank, seqGroup] = LiveInferContext::FindSeqGroupInAllRank(reqId);
        if (seqGroup == nullptr) {
            MINDIE_LLM_LOG_WARN_REQUEST("[LlmEngine]Try to release kv.The request(" << reqId
                                                                            << ") is not exist. Maybe been aborted");
            continue;
        }

        SequenceId seqId = seqGroup->firstSeq->seqId_;
        enginePerDPs_[localDPRank]->scheduler->NotifyMeKvPulledSeqIds(seqId);
    }
}

void LlmEngine::Stop()
{
    stop_.store(true, std::memory_order_relaxed);
    MINDIE_LLM_LOG_INFO("[LlmEngine]Engine stopped successfully.");
}

void LlmEngine::StartEngineThread()
{
    // 初始化engine的线程
    for (size_t i = 0; i < enginePerDPs_.size(); ++i) {
        EnginePerDPSPtr enginePerDP = enginePerDPs_.at(i);
        if (role_ == Role::D && schedulerConfig_->distributedEnable) {
            enginePerDP->dummyQuotaManagerSPtr_ = std::make_shared<DummyQuotaManager>(dpRankId_);
            MINDIE_LLM_LOG_INFO("[LlmEngine]Start dummyQuotaManager thread(" << dpRankId_ << ") successfully!!!");
        }
        enginePerDP->schedulerThread =
            std::thread([this, localDPRank = i]() { this->SchedulerThreadEntry(localDPRank); });
        pthread_setname_np(enginePerDP->schedulerThread.native_handle(), ("scheduler-" + std::to_string(i)).c_str());
        MINDIE_LLM_LOG_INFO("[LlmEngine]Start thread(" << i << ") successfully.");
    }
    MINDIE_LLM_LOG_INFO("[LlmEngine]Engine thread(s) start successfully.");
}

void LlmEngine::InitProcessGroup(const std::vector<NodeInfo> &nodeInfos, std::string &processGroupMasterIP,
                                 uint32_t processGroupMasterPort)
{
    MINDIE_LLM_LOG_INFO("Initialize Process Group: masterIP=" << processGroupMasterIP
                                                              << ", masterPort=" << processGroupMasterPort);
    // 根据infos信息创建ProcessGroup
    std::vector<std::string> hostIps = GetHostIP();
    std::string hostIp = GetLocalHostIP(nodeInfos, hostIps);

    if (processGroupMasterIP.size() != 0 && processGroupMasterPort != 0) {
        MINDIE_LLM_LOG_INFO("Initialize Process Group: worldSize=" << schedulerConfig_->dpSize
                                                                   << ", rank=" << dpRankId_);
        bool isMaster = std::find(hostIps.begin(), hostIps.end(), processGroupMasterIP) != hostIps.end();
        isMaster = isMaster && (dpRankId_ == 0); // TBC，当前认为dpRank为0的一定是master节点
        ProcessGroup::GetInstance(processGroupMasterIP, processGroupMasterPort, hostIp, dpRankId_,
                                  schedulerConfig_->dpSize, isMaster);
        isProcessGroupInit = true;
        MINDIE_LLM_LOG_INFO("Process Group initialized successfully.");
        PROF(INFO, AddMetaInfo("isMaster", isMaster));
        PROF(INFO, AddMetaInfo("rankHostIp", hostIp));
    }
    PROF(INFO, AddMetaInfo("masterIP", processGroupMasterIP));
    for (const std::string &ip : hostIps) {
        PROF(INFO, AddMetaInfo("hostIpList", ip));
    }
    PROF(INFO, AddMetaInfo("dpRankId", dpRankId_));
}

/**
 * 当前D节点返回recompute response在调度线程中，后续如果有性能问题，需要考虑使用其他线程。
 */
void LlmEngine::SendRecomputeResponse(std::vector<SequenceId> &recomputeSeqIds, size_t localDPRank)
{
    if (role_ != Role::D) {
        return;
    }

    for (SequenceId seqId : recomputeSeqIds) {
        SequenceGroupSPtr seqGroup = LiveInferContext::GetInstance(localDPRank)->GetSeqGroup(seqId);
        if (seqGroup == nullptr) {
            MINDIE_LLM_LOG_INFO(
                "Can not find sequence group when try to send recompute response, seqId :" << seqId);
            continue;
        }

        ResponseSPtr response = std::make_shared<Response>(seqGroup->metrics_.inferReqId_);
        response->isEos = true;
        response->iterTimes = seqGroup->iterTimes;
        response->transferStatusFlag = TransferStatusType::RECOMPUTED_TRIGGERED;
        enginePerDPs_.at(localDPRank)->abortRespToManagerCall(response);
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine] DP RankId: "
                            << (dpRankId_ > 0 ? dpRankId_ : localDPRank)
                            << ". Engine(Decode Node) send recompute response successfully. seqId: " << seqId
                            << ", requestId: " << seqGroup->metrics_.inferReqId_);

        // recompute的请求不再进入waiting队列，不会再被轮询，需要结束清理资源
        enginePerDPs_.at(localDPRank)->scheduler->ClearSeqGrp(seqGroup, SequenceStatus::FINISH_RECOMPUTE);
    }
}

SchOutDataPair LlmEngine::PostScheduleSyncUp(bool needSync, SequenceGroupMetaDatas &metas, SchedulerOutputs &schOut,
                                             size_t localDPRank)
{
    std::vector<std::vector<SequenceGroupMetaDatas>> allDpMetas;
    auto syncUpBegin = high_resolution_clock::now();

    // 多DP集中式和分布式的P节点都需要同步的信息
    BatchInfo batchInfo(metas.maxBatchSize, metas.maxSeqLen);
    if (needSync) {
        PostScheduler::SyncBatchInfo(batchInfo, localDPRank, schedulerConfig_->distributedEnable);
        metas.maxBatchSize = batchInfo.maxBatchSize_;
        metas.maxSeqLen = batchInfo.maxSeqLen_;
    }
    auto syncUpAfterBatchInfo = high_resolution_clock::now();
    // 进程（分布式）集合通信需要同步信息seqlen，做DP间的dummy和padding
    if (isDistributedPNodeProcessCCReady_) { // 进程级通信修改metas，需要在赋值allDpMetas之前处理
        PostScheduler::SyncSeqLenList(metas.seqLenList, batchInfo.batchSizeList_, batchInfo.maxBatchSize_, localDPRank,
                                      schedulerConfig_->distributedEnable);
    }

    std::vector<SequenceGroupMetaDatas> tmpMeta = {metas};
    allDpMetas.push_back(tmpMeta);
    std::vector<std::vector<SchedulerOutputs>> allDpOuts;
    std::vector<SchedulerOutputs> tmpOut = {schOut};
    allDpOuts.push_back(tmpOut);

    // 线程（集中式）需要同步的信息，往所有DP发送所有DP的batch请求，TG做DP切分
    if (isCentralizedThreadCCReady_) {
        if (batchInfo.maxBatchSize_ > 0) {
            PostScheduler::AllGatherBatchesAcrossDPs(allDpMetas, allDpOuts, localDPRank);
        }
    }
    auto syncUpEnd = high_resolution_clock::now();
    auto syncSecondCost = duration_cast<milliseconds>(syncUpEnd - syncUpAfterBatchInfo).count();
    if (syncSecondCost > LOG_CC_TIME_THRESHOLD_MS) {
        auto syncFirstCost = duration_cast<milliseconds>(syncUpAfterBatchInfo - syncUpBegin).count();
        auto syncUpCost = duration_cast<milliseconds>(syncUpEnd - syncUpBegin).count();
        MINDIE_LLM_LOG_INFO_REQUEST("[Scheduler|Schedule-Sync up] PostSchedule sync too long :"
                            << syncUpCost << ", DP RankId:" << (dpRankId_ > 0 ? dpRankId_ : localDPRank)
                            << ", maxBatchSize:" << metas.maxBatchSize << ", maxSeqLen:" << metas.maxSeqLen
                            << ", syncFirstCost:" << syncFirstCost << ", syncSecondCost:" << syncSecondCost);
    }

    return {allDpMetas, allDpOuts};
}

std::pair<SequenceGroupMetaDatas, SchedulerOutputs>
LlmEngine::MakeDummySchedulerOutput(SequenceGroupMetaDatas seqGroupMetadata) const
{
    SchedulerOutputs output;
    output.forwardMode_ = ForwardMode::DUMMY;
    return {seqGroupMetadata, output};
}

void LlmEngine::ExecuteDummy(EnginePerDPSPtr enginePerDP, SequenceGroupMetaDatas &seqGroupMetadata, size_t localDPRank,
                             std::function<void(ModelBatchResultSPtr)> responseHandler) const
{
    auto [dummyMetadata, dummpOut] = MakeDummySchedulerOutput(seqGroupMetadata);
    ExecuteModelRequestPtr dummyRequest = std::make_unique<model_execute_data::ExecuteModelRequest>();
    ConstructExecuteRequest::ConstructExecuteModelRequest(dummyRequest, dummyMetadata, dummpOut, localDPRank);
    bool succ = enginePerDP->modelExecutor->AsyncExecuteModel(
        dummyRequest, std::function<void(ModelBatchResultSPtr)>(responseHandler));
    if (!succ) {
        MINDIE_LLM_LOG_ERROR("Call AsyncExecuteModel(dummy) failed.");
        throw runtime_error("The async execution failed.Check logs.");
    }
}

void LlmEngine::switchRole(size_t localDPRank)
{
    EnginePerDPSPtr enginePerDP = enginePerDPs_.at(localDPRank);
    role_ = enginePerDP->scheduler->SwitchRole();
}

bool LlmEngine::DistDecodeAcquireDummyQuota(bool isDummy, EnginePerDPSPtr enginePerDP) const
{
    if (!enginePerDP->dummyQuotaManagerSPtr_) {
        return false;
    }

    if (isDummy) {
        return enginePerDP->dummyQuotaManagerSPtr_->AcquireQuota(true);
    }

    // 真实请求
    enginePerDP->dummyQuotaManagerSPtr_->Wakeup();
    return enginePerDP->dummyQuotaManagerSPtr_->AcquireQuota(false);
}

void LlmEngine::PauseScheduling()
{
    isPauseScheduling_.store(true, std::memory_order_relaxed);
    MINDIE_LLM_LOG_INFO("[LlmEngine]Scheduling paused.");
}

void LlmEngine::ResumeScheduling()
{
    isPauseScheduling_.store(false, std::memory_order_relaxed);
    MINDIE_LLM_LOG_INFO("[LlmEngine]Scheduling resumed.");
}

void LlmEngine::LayerwiseEosClean(bool layerwiseDisaggregated, std::unordered_set<SequenceId> &eosCleanupSeqIds,
                                  EnginePerDPSPtr enginePerDP) const
{
    if (!layerwiseDisaggregated) {
        return;
    }
    // 边云特性专用，云上推不出eos，由调度器额外下发cache清理的clean batch
    const size_t eosCleanupThreshold = 10;
    if (eosCleanupSeqIds.size() > eosCleanupThreshold) {
        // 多dp扩展下是否要同步需分析
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|Scheduler]:"
            << "Clean EOS seqIds number is: " << eosCleanupSeqIds.size());
        TGCleanupRequestPtr EOSCleanupRequest = BuildTGCleanupRequest(eosCleanupSeqIds);
        bool succ = enginePerDP->modelExecutor->AsyncEOSCleanup(EOSCleanupRequest);
        if (!succ) {
            MINDIE_LLM_LOG_ERROR("Call AsyncEOSCleanup failed.");
            throw runtime_error("The async eos clean up failed. Check logs.");
        } // clear the container after send success
        eosCleanupSeqIds.clear();
    }
}

void LlmEngine::SchedulerThreadEntry(size_t localDPRank)
{
    EnginePerDPSPtr enginePerDP = enginePerDPs_.at(localDPRank);
    isCentralizedThreadCCReady_ = enginePerDPs_.size() > 1; // size大于1就表示集中式的主节点
    isDistributedPNodeProcessCCReady_ = schedulerConfig_->distributedEnable && isProcessGroupInit && role_ == Role::P;
    bool needSync = isDistributedPNodeProcessCCReady_ || isCentralizedThreadCCReady_;
    time_point<high_resolution_clock, high_resolution_clock::duration> asynExeBegin = high_resolution_clock::now();
    auto heartbeatBegin = high_resolution_clock::now();
    size_t schedulingRound = 0;
    int64_t scheduleCost = 0;

    std::unordered_set<SequenceId> eosCleanupSeqIds = {};

    while (!stop_) {
        // 暂停调度组batch
        if (isPauseScheduling_.load(std::memory_order_relaxed)) {
            enginePerDP->scheduler->StopRunningRequest();
            AbortParallelSeqGroups(localDPRank);
            enginePerDP->scheduler->CollectAndClearAbortedParallelSeqGroups();
            std::this_thread::sleep_for(milliseconds(DEFAULT_SLEEP_TIME_BETWEEN_TWO_ITER));
            continue;
        }

        // flex 场景下，先进行优先级判断切换角色后，再进行调度
        if (role_ == Role::FlexP || role_ == Role::FlexD || role_ == Role::FlexPnD) {
            switchRole(localDPRank);
        }
        // 1、PD分离场景进行KV transfer调度（P节点清理KVcache；D节点组pull kv request）
        auto transferBegin = high_resolution_clock::now();
        ScheduleExecTransfer(enginePerDP);
        auto transferEnd = high_resolution_clock::now();

        // 进行心跳检查和日志打印
        CheckAndPrintHeartbeat(heartbeatBegin, enginePerDP);

        // 2. 异步调度轮次控制
        uint32_t asyncScheduleRound = schedulerConfig_->layerwiseDisaggregated ?
                                 schedulerConfig_->maxDispatchBatchNum : MAX_ASYNC_SCHEDULE_TIMES;

        if (enginePerDP->modelExecOutputHandler->GetAsyncBatchNum() >= asyncScheduleRound ||
            enginePerDP->lastScheduleEmpty) {
            std::this_thread::sleep_for(milliseconds(DEFAULT_SLEEP_TIME_BETWEEN_TWO_ITER));
            enginePerDP->lastScheduleEmpty = false;
            continue;
        }

        // 进入本轮调度时刻 - 上轮下发完成时刻 = 推理返回response时间
        auto responseRet = high_resolution_clock::now(); // 进入本轮调度时刻
        auto responseCost = duration_cast<milliseconds>(responseRet - asynExeBegin).count();
        int64_t totalIterCost = scheduleCost + responseCost;
        if (totalIterCost > LOG_TIME_THRESHOLD_MS) {
            auto transferKVCost = duration_cast<milliseconds>(transferEnd - transferBegin).count();
            MINDIE_LLM_LOG_INFO_REQUEST(
                "[Scheduler|Schedule-Response] Response and schedule transfer cost too long. DP RankId:"
                << (dpRankId_ > 0 ? dpRankId_ : localDPRank) << ", response cost:" << responseCost
                << ", ScheduleExecTransfer cost:" << transferKVCost << ", scheduleCost:" << scheduleCost
                << ", totalIterCost:" << totalIterCost << ", schedulingRound:" << schedulingRound);
        }
        schedulingRound++;

        // 3. 从Response获取执行结果信息（output token、 finished/exception request）和aborted请求；
        auto scheduleBegin = high_resolution_clock::now();
        enginePerDP->scheduler->FetchSeqGeneratedTokens(
            enginePerDP->modelExecOutputHandler->GetSeqIdToOutputTokenQueue());
        std::unordered_set<SequenceId> finishedSeqIds =
        enginePerDP->scheduler->FetchFinishedSeqIds(enginePerDP->modelExecOutputHandler->GetFinishedSeqIds());
        enginePerDP->scheduler->FetchExceptionSeqIds(enginePerDP->modelExecOutputHandler->GetExceptionSeqIds());
        enginePerDP->scheduler->FetchAbortedReqIds(enginePerDP->abortedRequestIds);
        enginePerDP->scheduler->CollectAndClearAbortedParallelSeqGroups();
        enginePerDP->lastScheduleEmpty = false;

        // clean batch修改，记录结束的fininshedSeq
        if (schedulerConfig_->layerwiseDisaggregated) {
            eosCleanupSeqIds.insert(finishedSeqIds.begin(), finishedSeqIds.end());
        }

        // 4. 执行调度组batch
        auto spanSchedule = PROF(INFO, Domain("Schedule").SpanStart("BatchSchedule", false));
        auto [seqGroupMetadata, scheduleOut] = enginePerDP->scheduler->Schedule(needSync);
        enginePerDP->scheduler->ClearLastScheduleEmpty();
        if (scheduleOut.scheduledSeqGroups_.size() > 0) {
            PROF(spanSchedule.ArrayResource(
                scheduleOut.scheduledSeqGroups_.begin(), scheduleOut.scheduledSeqGroups_.end(),
                [](decltype(spanSchedule) *prof, decltype(scheduleOut.scheduledSeqGroups_.begin()) iter) -> void {
                    prof->Resource((*iter)->seqGroup_->requestId)
                        .Attr("iter", (*iter)->seqGroup_->firstSeq->data_.outputTokenIds.size())
                        .Metric("num_scheduled_tokens", (*iter)->tokenChunkSize_);
                }));
            PROF(spanSchedule.Attr("batchType", static_cast<int>(scheduleOut.forwardMode_)));
        if (schedulerConfig_->layerwiseDisaggregated) {
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|Scheduler]:"
                << "batchType is: " << static_cast<int>(scheduleOut.forwardMode_)
                << ", batchsize is: " << scheduleOut.scheduledSeqGroups_.size());
        }
            PROF(spanSchedule.SpanEnd());
        }

        // 5. 同步进程/线程间的batch信息
        auto spanPostSchedule = PROF(L2, Domain("Schedule").SpanStart("PostSchedule"));
        auto [allDpMetas, allDpOuts] = PostScheduleSyncUp(needSync, seqGroupMetadata, scheduleOut, localDPRank);
        PROF(spanPostSchedule.SpanEnd());
        if (schedulerConfig_->layerwiseDisaggregated) {
            for (auto tmpOut : allDpOuts) {
                for (auto scOut : tmpOut) {
                    if (scOut.forwardMode_ == ForwardMode::PREFILL) {
                        scheduleOut.forwardMode_ = ForwardMode::PREFILL;    // 陪跑也需要改成和下发一致的类型, 默认decode
                        break;
                    }
                }
            }
        }

        // 6. batch下发给Executor执行
        // 如果单dp下自身batch不空， 或者多dp下其他dp的batch不空（集中式）
        auto responseHandler = [this, enginePerDP](ModelBatchResultSPtr output) {
            if (output->has_err_msg() && output->err_msg() != "") {
                MINDIE_LLM_LOG_ERROR("Error code from executor: " << output->err_msg());
                ErrorQueue::GetInstance().EnqueueErrorMessage(output->err_msg(), "LlmEngine");
                PauseScheduling();
                return;
            }
            enginePerDP->modelExecOutputHandler->Entry4Executor(output);
        };

        if (!scheduleOut.IsEmpty() || (isCentralizedThreadCCReady_ && seqGroupMetadata.maxBatchSize > 0)) {
            for (const auto& scheduledSeqGroup : scheduleOut.scheduledSeqGroups_) {
                if (scheduledSeqGroup->seqGroup_->IsSimulateRequest()) {
                    MINDIE_LLM_LOG_DEBUG("[SimulateInference] Building ExecuteRequest, forwardMode="
                                        << static_cast<int>(scheduleOut.forwardMode_)
                                        << ", batchSize=" << scheduleOut.scheduledSeqGroups_.size()
                                        << ", requestId=" << scheduledSeqGroup->seqGroup_->requestId);
                }
            }
            ExecuteModelRequestPtr request =
                BuildExecuteModelRequest(allDpMetas, allDpOuts, schedulerConfig_->distributedEnable, dpRankId_);
            RecordEngineMetrics(scheduleOut, enginePerDP);

            auto spanExecute = PROF(INFO, Domain("Schedule").SpanStart("Execute"));
            PROF(spanExecute.ArrayResource(
                scheduleOut.scheduledSeqGroups_.begin(), scheduleOut.scheduledSeqGroups_.end(),
                [](decltype(spanExecute) *prof, decltype(scheduleOut.scheduledSeqGroups_.begin()) iter) -> void {
                    prof->Resource((*iter)->seqGroup_->requestId)
                        .Attr("iter", (*iter)->seqGroup_->firstSeq->data_.outputTokenIds.size())
                        .Metric("num_scheduled_tokens", (*iter)->tokenChunkSize_);
                }));
            if (schedulerConfig_->stageSelectPolicy == static_cast<uint32_t>(StagePolicyType::LATENCY_FIRST) ||
                schedulerConfig_->dynamicBatchSizeEnable) {
                auto batchExecuteStartTime = std::chrono::high_resolution_clock::now();
                SetupLatencyPredictor(batchExecuteStartTime, dpRankId_);
            }

            DistDecodeAcquireDummyQuota(false, enginePerDP);
            bool succ = enginePerDP->modelExecutor->AsyncExecuteModel(
                request, std::function<void(ModelBatchResultSPtr)>(responseHandler));
            if (!succ) {
                MINDIE_LLM_LOG_ERROR("Call AsyncExecuteModel failed.");
                // 异步调用失败是代码bug或者系统故障， 对于推理服务来说无法自行恢复
                PROF(spanExecute.SpanEnd());
                throw runtime_error("The async execution failed.Check logs.");
            }
            PROF(spanExecute.SpanEnd());

            enginePerDP->scheduler->PrepareNextSchedule(scheduleOut.scheduledSeqGroups_);
            enginePerDP->modelExecOutputHandler->GetAsyncBatchNum().fetch_add(1);

            // 边云协同场景，记录batch下发的类型
            layerwiseMixin_.LwdPrepareBatch(schedulerConfig_->layerwiseDisaggregated, scheduleOut);
            layerwiseMixin_.LwdEngineAddBatchCnt(schedulerConfig_->layerwiseDisaggregated,
                enginePerDP->scheduler->GetStagePolicy(), scheduleOut);
        } else {
            // 多dp下其他dp的batch不空(分布式)，需要陪跑
            if (seqGroupMetadata.maxBatchSize > 0 || DistDecodeAcquireDummyQuota(true, enginePerDP)) {
                auto spanDummy = PROF(L1, Domain("Schedule").SpanStart("ExecuteDummy"));
                ExecuteDummy(enginePerDP, seqGroupMetadata, localDPRank, responseHandler);
                enginePerDP->modelExecOutputHandler->GetAsyncBatchNum().fetch_add(1);
                PROF(spanDummy.SpanEnd());
            }
            if (scheduleOut.IsEmpty()) {
                RecordEngineMetrics(scheduleOut, enginePerDP);
            }
            // 供调度判断PFirst还是DFirst
            enginePerDP->scheduler->MarkLastScheduleEmpty();
            // 供当前线程判断是否调度为空，让出cpu时间
            enginePerDP->lastScheduleEmpty = true;
        }

        if (scheduleOut.recomputeSeqIds_.size() > 0) {
            // PD分离场景，Recompute的请求需要上送Recompute Response到coordinator，从P节点重新调度
            SendRecomputeResponse(scheduleOut.recomputeSeqIds_, localDPRank);
        }
        // 调度下发完成，connector/TG开始执行

        AbortParallelSeqGroups(localDPRank);

        asynExeBegin = high_resolution_clock::now();

        // 卸载lora
        auto loraManager = mindie_llm::LoraManager::GetInstance(localDPRank);
        if (loraManager != nullptr) {
            loraManager->TryUnLoadWaiting();
        }

        // 7. 清理TexetGenerator CPU缓存（abort和recompute和P节点release kv场景））
        std::unordered_set<SequenceId> abortedSeqIds = enginePerDP->scheduler->ClearAndReturnTerminatedSeqIds();
        enginePerDP->TGCleanupSeqIds_.insert(abortedSeqIds.begin(), abortedSeqIds.end());
        enginePerDP->TGCleanupSeqIds_.insert(scheduleOut.recomputeSeqIds_.begin(), scheduleOut.recomputeSeqIds_.end());
        
        // 集中式场景下，所有DP的请求在每个DP的TG里都缓存了上下文cache，因此cache清理也需要gather后发送给所有dp
        std::unordered_set<SequenceId> allDPCleanSeqIds =
            (!isCentralizedThreadCCReady_)
                ? enginePerDP->TGCleanupSeqIds_
                : PostScheduler::AllGatherCleanSeqIdsAcrossDPs(enginePerDP->TGCleanupSeqIds_, localDPRank);
        if (!allDPCleanSeqIds.empty()) {
            TGCleanupRequestPtr TGCleanupRequest = BuildTGCleanupRequest(allDPCleanSeqIds);
            bool succ = enginePerDP->modelExecutor->AsyncTGCleanup(TGCleanupRequest);
            if (!succ) {
                MINDIE_LLM_LOG_ERROR("Call AsyncTGCleanup failed.");
                throw runtime_error("The async text generator clean up failed. Check logs.");
            }
            enginePerDP->TGCleanupSeqIds_.clear(); // clear the container after send success
        }
        // 边云协同场景，需要额外清理EOS
        LayerwiseEosClean(schedulerConfig_->layerwiseDisaggregated, eosCleanupSeqIds, enginePerDP);

        auto scheduleEnd = high_resolution_clock::now();
        scheduleCost = duration_cast<milliseconds>(scheduleEnd - scheduleBegin).count();
        if (scheduleCost > LOG_TIME_THRESHOLD_MS) {
            MINDIE_LLM_LOG_INFO_REQUEST("[Scheduler|Schedule-Batch] Schedule too long :"
                                << scheduleCost << ", DP RankId:" << (dpRankId_ > 0 ? dpRankId_ : localDPRank)
                                << ",cur dp batch size:" << scheduleOut.scheduledSeqGroups_.size()
                                << ", all dp max batch size:" << seqGroupMetadata.maxBatchSize
                                << ", forward mode:" << static_cast<int>(scheduleOut.forwardMode_)
                                << ", schedulingRound:" << schedulingRound);
        }
    }
}

// release transferred kv blocks for P and schedule/execute kv pull for D
void LlmEngine::ScheduleExecTransfer(std::shared_ptr<EnginePerDP> &engine) const
{
    if (role_ == Role::PnD || role_ == Role::FlexPnD) {
        return;
    }

    // 1. 执行调度组transfer
    auto spanTransferSchedule = PROF(L2, Domain("Schedule").SpanStart("ScheduleExecTransfer"));
    engine->scheduler->KVPulledReqEnterRunningQueue(engine->transferOutputHandler->GetPulledReqIds());

    auto [scheduleTransferMetadatas, scheduleTransferOut] = engine->scheduler->ScheduleTransfer();
    engine->TGCleanupSeqIds_.insert(scheduleTransferOut.pulledSeqIds.begin(), scheduleTransferOut.pulledSeqIds.end());
    PROF(spanTransferSchedule.SpanEnd());

    // 2. transfer batch下发给Executor执行
    if (scheduleTransferMetadatas.metaList.empty()) {
        return;
    }

    auto spanCall = PROF(L2, Domain("Schedule").SpanStart("CallExecuteSchedulerPull"));
    auto spanBuild = PROF(L2, Domain("Schedule").SpanStart("BuildPullKVRequestCallExecute"));
    PullKVRequestPtr request = ConstructExecuteRequest::ConstructPullKVRequest(scheduleTransferMetadatas);
    PROF(spanBuild.SpanEnd());

    auto spanExecute = PROF(L2, Domain("Schedule").SpanStart("CallExecute"));
    auto kvPullResponseHandler = [this, &engine](PullKVResponseSPtr output) {
        engine->transferOutputHandler->Entry4Executor(output);
    };
    bool succ = engine->modelExecutor->ExecuteKVTransfer(
        request, std::function<void(PullKVResponseSPtr)>(kvPullResponseHandler));
    if (!succ) {
        MINDIE_LLM_LOG_ERROR("Call ExecuteKVTransfer failed.");
        // 异步调用失败是代码bug或者系统故障， 对于推理服务来说无法自行恢复
        throw runtime_error("The KV transfer failed.Check logs.");
    }

    PROF(spanExecute.SpanEnd());
    PROF(spanCall.SpanEnd());
}

void LlmEngine::SyncBatchInfoAcrossNodes(SequenceGroupMetaDatas &metadata) const
{
    // 1. 集合通信获取各节点的batchInfo
    std::vector<torch::Tensor> batchInfo;
    batchInfo.emplace_back(
        torch::tensor({metadata.maxBatchSize, metadata.maxSeqLen}, torch::dtype(torch::kInt64).device(c10::kCPU)));
    try {
        std::vector<std::vector<torch::Tensor>> batchInfos = ProcessGroup::GetInstance().AllGather(batchInfo);
        if (batchInfos.empty() || batchInfos[0].empty()) {
            return;
        }
        // 2. 获取最大batchSize和maxSeqLen
        for (const auto &info : batchInfos[0]) {
            metadata.maxBatchSize = std::max(metadata.maxBatchSize, info[0].item<int64_t>());
            metadata.maxSeqLen = std::max(metadata.maxSeqLen, info[1].item<int64_t>());
        }
    } catch (const std::exception& e) {
        MINDIE_LLM_LOG_ERROR("SyncBatchInfoAcrossNodes failed: batchInfos is invalid.");
    }
}

ExecuteModelRequestPtr LlmEngine::BuildExecuteModelRequest(std::vector<std::vector<SequenceGroupMetaDatas>> &metadatas,
                                                           std::vector<std::vector<SchedulerOutputs>> &schedulerOutputs,
                                                           bool distributedEnable, int dpRankId)
{
    ExecuteModelRequestPtr request = std::make_unique<model_execute_data::ExecuteModelRequest>();
    for (size_t i = 0; i < metadatas.size(); i++) {
        SequenceGroupMetaDatas dpMetas = metadatas.at(i).at(0);
        SchedulerOutputs dpOut = schedulerOutputs.at(i).at(0);
        int reqDpRankId = static_cast<int>(i);
        if (distributedEnable) {
            // 集中式需要下发所有的DP请求，dp rankid根据集合通信结果设置
            // 分布式下发的请求都是本节点的dp rankid
            reqDpRankId = dpRankId;
        }
        if (dpMetas.metaList.size() > 0) {
            ConstructExecuteRequest::ConstructExecuteModelRequest(request, dpMetas, dpOut, reqDpRankId);
        }
    }
    return request;
}

// TG: TextGenerator
TGCleanupRequestPtr LlmEngine::BuildTGCleanupRequest(std::unordered_set<SequenceId> &TGCleanupSeqIds)
{
    TGCleanupRequestPtr request = std::make_unique<model_execute_data::TGCleanupRequest>();
    for (SequenceId seqId : TGCleanupSeqIds) {
        request->add_seq_ids(seqId);
    }

    return request;
}

// 统计前一批的每毫秒执行的token数
void LlmEngine::CalculateThroughput(std::shared_ptr<EnginePerDP> enginePerDP) const
{
    if (enginePerDP->lastExecuteStartTime_ != INVALID_TIME) {
        auto now = std::chrono::high_resolution_clock::now();
        auto lashBatchSpendTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - enginePerDP->lastExecuteStartTime_);
        if (lashBatchSpendTime.count() == 0) {
            return;
        }
        if (enginePerDP->lastForwardMode_ == ForwardMode::PREFILL) {
            enginePerDP->engineMetricStatics_.prefillThroughput_ =
                static_cast<float>(enginePerDP->lastBatchTokenNum_) / static_cast<float>(lashBatchSpendTime.count());
        } else {
            enginePerDP->engineMetricStatics_.decodeThroughput_ =
                static_cast<float>(enginePerDP->lastBatchTokenNum_) / static_cast<float>(lashBatchSpendTime.count());
        }
    }
}

void LlmEngine::RecordEngineMetrics(SchedulerOutputs &scOut, std::shared_ptr<EnginePerDP> enginePerDP)
{
    size_t tokenCount = (scOut.forwardMode_ == ForwardMode::PREFILL)
                            ? 0
                            : scOut.scheduledSeqGroups_.size() * (1 + schedulerConfig_->speculationGamma);
    for (auto scSeqGrpSPtr : scOut.scheduledSeqGroups_) {
        SequenceGroupSPtr seqGroup = scSeqGrpSPtr->seqGroup_;
        auto now = high_resolution_clock::now();
        auto waitTime = duration_cast<MicroSeconds>(now - seqGroup->arriveTime);
        if (seqGroup->metrics_.responseTime_ != INVALID_TIME) {
            // 上次decode或者prefill结束到重新被调度的时间间隔
            waitTime = duration_cast<MicroSeconds>(now - seqGroup->metrics_.responseTime_);
        }
        seqGroup->metrics_.queueWaitTime_ = static_cast<uint64_t>(waitTime.count());

        if (scOut.forwardMode_ == ForwardMode::PREFILL) {
            tokenCount += seqGroup->seqs_.at(0)->data_.promptTokenIds.size();
        }
    }

    // 统计前一批的每毫秒执行的token数
    CalculateThroughput(enginePerDP);
    enginePerDP->lastBatchTokenNum_ = tokenCount;
    enginePerDP->lastExecuteStartTime_ = high_resolution_clock::now();
    enginePerDP->lastForwardMode_ = scOut.forwardMode_;
}

LlmEnginePtr MakeLlmEngine(SchedulerConfig schedulerConfig, std::vector<IExecutorSPtr> executors,
                           ForwardRespToManagerCall cb, Role pdRole)
{
    return std::make_unique<LlmEngine>(schedulerConfig, executors, cb, pdRole);
}

EngineMetric LlmEngine::CollectEngineMetric(size_t localDPRank)
{
    EngineMetric engineMetric{};

    EnginePerDPSPtr &engine = enginePerDPs_.at(localDPRank);
    engineMetric.schedulerInfo = engine->scheduler->CollectSchedulerMetric();
    engineMetric.prefillThroughput_ = engine->engineMetricStatics_.prefillThroughput_;
    engineMetric.decodeThroughput_ = engine->engineMetricStatics_.decodeThroughput_;

    return engineMetric;
}

void LlmEngine::AccumulateDpMetricInto(size_t dpIndex, EngineMetric &aggregatedMetric)
{
    EngineMetric metric = CollectEngineMetric(dpIndex);
    MINDIE_LLM_LOG_DEBUG("DP[" << dpIndex << "] metrics: "
        << "freeNpuBlock=" << metric.schedulerInfo.blockInfo.freeNpuBlockNum_
        << ", totalNpuBlock=" << metric.schedulerInfo.blockInfo.totalNpuBlockNum_
        << ", freeCpuBlock=" << metric.schedulerInfo.blockInfo.freeCpuBlockNum_
        << ", totalCpuBlock=" << metric.schedulerInfo.blockInfo.totalCpuBlockNum_
        << ", waitingReq=" << metric.schedulerInfo.reqsInfo.waitingRequestNum_
        << ", runningReq=" << metric.schedulerInfo.reqsInfo.runningRequestNum_
        << ", swappedReq=" << metric.schedulerInfo.reqsInfo.swappedRequestNum_
        << ", allRadixMatch=" << metric.schedulerInfo.reqsInfo.allRadixMatchNum_
        << ", npuRadixHit=" << metric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_
        << ", preemptCount=" << metric.schedulerInfo.reqsInfo.cumulativePreemptCount_
        << ", prefillTput=" << metric.prefillThroughput_
        << ", decodeTput=" << metric.decodeThroughput_);

    aggregatedMetric.schedulerInfo.blockInfo.freeNpuBlockNum_ +=
        metric.schedulerInfo.blockInfo.freeNpuBlockNum_;
    aggregatedMetric.schedulerInfo.blockInfo.freeCpuBlockNum_ +=
        metric.schedulerInfo.blockInfo.freeCpuBlockNum_;
    aggregatedMetric.schedulerInfo.blockInfo.totalNpuBlockNum_ +=
        metric.schedulerInfo.blockInfo.totalNpuBlockNum_;
    aggregatedMetric.schedulerInfo.blockInfo.totalCpuBlockNum_ +=
        metric.schedulerInfo.blockInfo.totalCpuBlockNum_;

    aggregatedMetric.schedulerInfo.reqsInfo.waitingRequestNum_ +=
        metric.schedulerInfo.reqsInfo.waitingRequestNum_;
    aggregatedMetric.schedulerInfo.reqsInfo.runningRequestNum_ +=
        metric.schedulerInfo.reqsInfo.runningRequestNum_;
    aggregatedMetric.schedulerInfo.reqsInfo.swappedRequestNum_ +=
        metric.schedulerInfo.reqsInfo.swappedRequestNum_;
    aggregatedMetric.schedulerInfo.reqsInfo.allRadixMatchNum_ +=
        metric.schedulerInfo.reqsInfo.allRadixMatchNum_;
    aggregatedMetric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_ +=
        metric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_;
    aggregatedMetric.schedulerInfo.reqsInfo.cumulativePreemptCount_ +=
        metric.schedulerInfo.reqsInfo.cumulativePreemptCount_;

    aggregatedMetric.prefillThroughput_ += metric.prefillThroughput_;
    aggregatedMetric.decodeThroughput_ += metric.decodeThroughput_;
}

EngineMetric LlmEngine::CollectAllDpEngineMetric()
{
    EngineMetric aggregatedMetric{};
    size_t dpSize = enginePerDPs_.size();
    if (dpSize == 0) {
        MINDIE_LLM_LOG_WARN("No DP ranks available to collect metrics.");
        return aggregatedMetric;
    }

    for (size_t i = 0; i < dpSize; ++i) {
        AccumulateDpMetricInto(i, aggregatedMetric);
    }

    // 吞吐量取平均值
    aggregatedMetric.prefillThroughput_ /= static_cast<float>(dpSize);
    aggregatedMetric.decodeThroughput_ /= static_cast<float>(dpSize);

    MINDIE_LLM_LOG_DEBUG("Aggregated metrics from " << dpSize << " DP ranks: "
        << "totalFreeNpuBlock=" << aggregatedMetric.schedulerInfo.blockInfo.freeNpuBlockNum_
        << ", totalNpuBlock=" << aggregatedMetric.schedulerInfo.blockInfo.totalNpuBlockNum_
        << ", totalFreeCpuBlock=" << aggregatedMetric.schedulerInfo.blockInfo.freeCpuBlockNum_
        << ", totalCpuBlock=" << aggregatedMetric.schedulerInfo.blockInfo.totalCpuBlockNum_
        << ", totalWaitingReq=" << aggregatedMetric.schedulerInfo.reqsInfo.waitingRequestNum_
        << ", totalRunningReq=" << aggregatedMetric.schedulerInfo.reqsInfo.runningRequestNum_
        << ", totalSwappedReq=" << aggregatedMetric.schedulerInfo.reqsInfo.swappedRequestNum_
        << ", totalAllRadixMatch=" << aggregatedMetric.schedulerInfo.reqsInfo.allRadixMatchNum_
        << ", totalNpuRadixHit=" << aggregatedMetric.schedulerInfo.reqsInfo.npuRadixMatchHitNum_
        << ", totalPreemptCount=" << aggregatedMetric.schedulerInfo.reqsInfo.cumulativePreemptCount_
        << ", avgPrefillTput=" << aggregatedMetric.prefillThroughput_
        << ", avgDecodeTput=" << aggregatedMetric.decodeThroughput_);
    return aggregatedMetric;
}

void LlmEngine::SetPrefillPercentage(uint32_t prefillPercentage)
{
    for (size_t i = 0; i < enginePerDPs_.size(); i++) {
        EnginePerDPSPtr enginePerDP = enginePerDPs_.at(i);
        if (enginePerDP != nullptr && enginePerDP->scheduler != nullptr) {
            enginePerDP->scheduler->SetPrefillPercentage(prefillPercentage);
            MINDIE_LLM_LOG_INFO("Set prefill ratio successfully, dp rank id:" << i);
        } else {
            MINDIE_LLM_LOG_ERROR("Set prefill ratio failed, engine or scheduler is null, dp rank id:" << i);
        }
    }
}

void LlmEngine::ExecuteRecoverCommand(RecoverCommandInfo &commandInfo)
{
    if (enginePerDPs_.empty()) {
        MINDIE_LLM_LOG_ERROR("No enginePerDp available to execute recover command.");
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(enginePerDPs_.size());

    for (size_t i = 0; i < enginePerDPs_.size(); ++i) {
        threads.emplace_back([this, i, &commandInfo]() {
            EnginePerDPSPtr enginePerDP = enginePerDPs_.at(i);
            enginePerDP->modelExecutor->ExecuteRecoverCommand(commandInfo);
            MINDIE_LLM_LOG_INFO("Execute Recover command: " << commandInfo.command);
        });
    }

    // 等待所有线程完成
    for (auto &thread : threads) {
        thread.join();
    }
}

void LlmEngine::SetupLatencyPredictor(
    const std::chrono::high_resolution_clock::time_point& batchExecuteStartTime,
    int dpRankId)
{
    auto& recorder = DynamicBatchRecorder::GetInstance(static_cast<size_t>(dpRankId));
    auto predictor = recorder.GetLatencyPredictor();
    if (predictor != nullptr) {
        predictor->SetBatchExecuteStartTime(batchExecuteStartTime);
    }
}

} // namespace mindie_llm