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
 
#ifndef LLM_ENGINE_H
#define LLM_ENGINE_H

#include <atomic>
#include <thread>
#include "engine/illm_engine.h"
#include "ischeduler.h"
#include "iload_balancer.h"
#include "model_exec_output_handler.h"
#include "transfer_output_handler.h"
#include "seq_group_builder_from_infer_req.h"
#include "process_group.h"
#include "request_response/request.h"
#include "latency_predictor/latency_predictor.h"
#include "lora_manager.h"
#include "dummy_quota_manager.h"
#include "layerwise_mixin/layerwise_mixin.h"
#include "data_type.h"

namespace mindie_llm {
// 调度完2轮batch后，让出1ms cpu时间
constexpr int DEFAULT_SLEEP_TIME_BETWEEN_TWO_ITER = 1;
constexpr int HEARTBEAT_INTERVAL_SECONDS = 60;
constexpr int METRICS_UPDATE_INTERVAL = 50; // 50ms

struct EngineMetricStatics {
    std::atomic<float> prefillThroughput_{0.0};
    std::atomic<float> decodeThroughput_{0.0};
};

// 在Engine内完成Data Parallelism (DP)的特性
struct EnginePerDP {
    std::thread schedulerThread;
    std::shared_ptr<LatencyPredictor> latencypredictor;
    SchedulerPtr scheduler;
    IExecutorSPtr modelExecutor;
    std::unique_ptr<ModelExecOutputHandler> modelExecOutputHandler;
    std::unique_ptr<TransferOutputHandler> transferOutputHandler;
    ConcurrentDeque<RequestId> abortedRequestIds;
    bool lastScheduleEmpty{false};

    ForwardRespToManagerCall abortRespToManagerCall;
    size_t addedRequestNum{0};
    size_t abortedRequestNum{0};

    ForwardMode lastForwardMode_;
    /** 统计token占用时间 */
    std::chrono::time_point<std::chrono::high_resolution_clock> lastExecuteStartTime_{INVALID_TIME};
    size_t lastBatchTokenNum_;
    EngineMetricStatics engineMetricStatics_;

    std::unordered_set<SequenceId> TGCleanupSeqIds_;

    DummyQuotaManagerSPtr dummyQuotaManagerSPtr_;
};
using EnginePerDPSPtr = std::shared_ptr<EnginePerDP>;
using SchOutDataPair =
    std::pair<std::vector<std::vector<SequenceGroupMetaDatas>>, std::vector<std::vector<SchedulerOutputs>>>;

// LlmEngine is used by mindie_llm LlmManagerV2
class LlmEngine final : public ILlmEngine {
public:
    LlmEngine(SchedulerConfig schedulerConfig, std::vector<IExecutorSPtr> executors, ForwardRespToManagerCall cb,
              Role pdRole);

    ~LlmEngine() override;

    void InitProcessGroup(const std::vector<NodeInfo> &nodeInfos, std::string &processGroupMasterIP,
                          uint32_t processGroupMasterPort) override;

    void StartEngineThread() override;

    bool DistDecodeAcquireDummyQuota(bool isDummy, EnginePerDPSPtr enginePerDP) const;

    bool AddRequest(RequestSPtr request) override;
    
    void SendAbortResponse(SequenceGroupSPtr seqGroup, size_t localDPRank, InferStatusType flag) const;

    void AbortRequests(std::unordered_set<RequestId> &requestIds) override;

    void ReleaseKvCache(std::unordered_set<RequestId> &requestIds) override;

    void Stop() override;

    EngineMetric CollectEngineMetric(size_t localDPRank = 0) override;

    EngineMetric CollectAllDpEngineMetric() override;

    void SetPrefillPercentage(uint32_t prefillPercentage) override;

    void PauseScheduling() override;

    void ResumeScheduling() override;

    void ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) override;
    
protected:
    void SchedulerThreadEntry(size_t localDPRank);

    void ScheduleExecTransfer(std::shared_ptr<EnginePerDP> &engine) const;

    void SyncBatchInfoAcrossNodes(SequenceGroupMetaDatas &metadata) const;

    static ExecuteModelRequestPtr BuildExecuteModelRequest(std::vector<std::vector<SequenceGroupMetaDatas>> &metadatas,
                                                           std::vector<std::vector<SchedulerOutputs>> &schedulerOutputs,
                                                           bool distributedEnable, int dpRankId);

    static TGCleanupRequestPtr
    BuildTGCleanupRequest(std::unordered_set<SequenceId> &TGCleanupSeqIds); // TG: TextGenerator

    SequenceId NextSeqId();

    void RecordEngineMetrics(SchedulerOutputs &scOut, std::shared_ptr<EnginePerDP> enginePerDP);

    void CalculateThroughput(std::shared_ptr<EnginePerDP> enginePerDP) const;

private:
    /// 将第 dpIndex 个 DP 的指标累加到 aggregatedMetric，并打单 DP 日志
    void AccumulateDpMetricInto(size_t dpIndex, EngineMetric &aggregatedMetric);

    void AbortParallelSeqGroups(size_t localDPRank) const;

    void SendRecomputeResponse(std::vector<SequenceId> &recomputeSeqIds, size_t localDPRank);
    template <typename T> SequenceGroupSPtr GetSequenceGroupWithoutRank(T id);

    SchOutDataPair PostScheduleSyncUp(bool needSync, SequenceGroupMetaDatas &metas, SchedulerOutputs &schOut,
                                      size_t localDPRank);

    std::pair<SequenceGroupMetaDatas, SchedulerOutputs>
    MakeDummySchedulerOutput(SequenceGroupMetaDatas seqGroupMetadata) const;

    void ExecuteDummy(EnginePerDPSPtr enginePerDP, SequenceGroupMetaDatas &seqGroupMetadata, size_t localDPRank,
                      std::function<void(ModelBatchResultSPtr)> responseHandler) const;

    SchedulerConfigSPtr schedulerConfig_ = {nullptr};

    std::vector<EnginePerDPSPtr> enginePerDPs_;

    std::atomic<bool> stop_ = {false};

    Role role_{Role::PnD}; // 默认角色为PnD

    LoadBalancerPtr loadBalancer_;

    bool isProcessGroupInit{false};

    // 是否集中式多DP并且线程资源初始化ok
    bool isCentralizedThreadCCReady_{false};

    // 是否分布式多DP，且进程资源初始化ok
    bool isDistributedPNodeProcessCCReady_{false};

    // 调度暂停标志位
    std::atomic<bool> isPauseScheduling_{false};

    // 分布式场景下记录DP rank id，用于下发batch时指定dp rank id值
    int dpRankId_{0};

    void switchRole(size_t localDPRank);

    inline void CheckAndPrintHeartbeat(std::chrono::time_point<std::chrono::high_resolution_clock> &heartbeatBegin,
                                       const EnginePerDPSPtr &enginePerDP) const
    {
        auto diffTime = std::chrono::high_resolution_clock::now() - heartbeatBegin;
        if (diffTime > std::chrono::seconds(HEARTBEAT_INTERVAL_SECONDS)) {
            heartbeatBegin = std::chrono::high_resolution_clock::now();
            auto passed_seconds = std::chrono::duration_cast<std::chrono::seconds>(diffTime).count();
            MINDIE_LLM_LOG_INFO_REQUEST("Since last schedule, pass " << passed_seconds
                << " seconds, AsyncBatchNum=" << enginePerDP->modelExecOutputHandler->GetAsyncBatchNum()
                << ", freeNpuBlockNum=" << enginePerDP->scheduler->CollectSchedulerMetric().blockInfo.freeNpuBlockNum_
                << ", freeCpuBlockNum=" << enginePerDP->scheduler->CollectSchedulerMetric().blockInfo.freeCpuBlockNum_);
        }
    }
    // 边云新增
    void LayerwiseEosClean(bool layerwiseDisaggregated, std::unordered_set<SequenceId> &eosCleanupSeqIds,
                           EnginePerDPSPtr enginePerDP) const;
    LayerwiseMixin layerwiseMixin_;
};

} // namespace mindie_llm

#endif
