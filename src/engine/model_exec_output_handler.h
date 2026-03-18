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
 
#ifndef MODEL_EXEC_OUTPUT_HANDLER_H
#define MODEL_EXEC_OUTPUT_HANDLER_H
#include <condition_variable>
#include <functional>
#include <thread>
#include <unordered_set>
#include <atomic>
#include <chrono>

#include "concurrent_deque.h"
#include "executor/executor_interface.h"
#include "engine/illm_engine.h"
#include "concurrent_map.h"
#include "memory_utils.h"
#include "live_infer_context.h"
#include "buffered_responser.h"
#include "policy/stage_policy/stage_policy.h"
#include "latency_predictor/latency_predictor.h"
#include "layerwise_mixin/layerwise_mixin.h"

namespace mindie_llm {
using MicroSeconds = std::chrono::microseconds;

class ModelExecOutputHandler {
public:
    explicit ModelExecOutputHandler(ForwardRespToManagerCall cb, Role pdRole, SchedulerConfigSPtr &config,
                                    std::shared_ptr<LatencyPredictor> latencypredictor, size_t localDPRank = 0);

    void Entry4Executor(ModelBatchResultSPtr &modelBatchResult); // for model backend agent

    ConcurrentDeque<SequenceId> &GetFinishedSeqIds();

    ConcurrentDeque<SequenceId> &GetExceptionSeqIds();

    ConcurrentDeque<std::pair<SequenceId, TokenId>> &GetSeqIdToOutputTokenQueue();

    std::atomic<size_t> &GetAsyncBatchNum();

    void SetRole(Role role) { role_ = role; }
    // flex 场景中， 需要在请求完成时记录时间戳
    void SetStagePolicy(std::shared_ptr<StagePolicy> stagePolicy) { stagePolicy_ = stagePolicy; }

private:
    Role role_{Role::PnD};
    ForwardRespToManagerCall forwardRespToManagerCall_; // from llm manager
    SchedulerConfigSPtr schedulerConfig_;
    BufferResponseConfig bufferResponseConfig_;
    std::shared_ptr<LatencyPredictor> latencypredictor_;
    size_t localDPRank_{0};
    ConcurrentDeque<std::pair<SequenceId, TokenId>>
        seqIdToOutputTokenQueue_; // this queue is fill by std::pair<SequenceId, TokenId>
    // model execution output handler will save finished req ids and scheduler will use it to finish requests.
    ConcurrentDeque<SequenceId> finishedSeqIds_;      // will be used by scheduler, seqs that are finished
    ConcurrentDeque<SequenceId> execExceptionSeqIds_; // will be used by scheduler, seqs of execution exception
    std::atomic<size_t> asyncBatchNum_{0};
    BufferedResponser bufferedResponser_;
    // store rankid for logging
    int dpRankId_{0};
    ResponseSPtr ConvertSequenceGroupOutputToResponse(const model_execute_data::CompletionSequenceGroupOutput &output,
                                                      uint64_t &queueWaitTime, uint64_t &currentPrefixCachedTokenNums);

    void AddOutputsToResponse(ResponseSPtr response,
                              const model_execute_data::CompletionSequenceGroupOutput &output) const;

    void AsyncPublishPrefilledKvCache(ModelBatchResultSPtr &modelBatchResult);

    void HandleParallelSampling(const model_execute_data::CompletionSequenceGroupOutput &output,
                                LiveInferContextSPtr &liveInferContext);

    void HandleGreedySampling(const model_execute_data::SequenceOutput &sample);

    SequenceGroupSPtr FindRootSequenceGroup(const model_execute_data::CompletionSequenceGroupOutput &output,
                                            LiveInferContextSPtr &liveInferContext) const;

    void CreateNewSequenceGroup(const model_execute_data::SequenceOutput &sample, SequenceGroupSPtr &rootSeqGrp,
                                LiveInferContextSPtr &liveInferContext) const;

    void UpdateSequenceGroup(const model_execute_data::SequenceOutput &sample, SequenceGroupSPtr &rootSeqGrp) const;

    void SetResponseFlags(const model_execute_data::CompletionSequenceGroupOutput &output, ResponseSPtr response) const;

    void ProcessSequenceStatus(SequenceId seqId, int64_t finishReason);

    std::shared_ptr<StagePolicy> stagePolicy_{nullptr};

    // 边云新增
    LayerwiseMixin layerwiseMixin_;
};
} // namespace mindie_llm
#endif