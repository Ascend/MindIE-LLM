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
 
#include "model_exec_output_handler.h"

#include <string>
#include <chrono>

#include "log.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "live_infer_context.h"
#include "policy/stage_policy/stage_policy.h"
#include "policy/stage_policy/edge_cloud_policy.h"
#include "health_checker.h"

using namespace mindie_llm;
using namespace model_execute_data;
// store total decode token num for logging
std::atomic<int> g_decodeTokenCount = 0;
ModelExecOutputHandler::ModelExecOutputHandler(ForwardRespToManagerCall cb, Role pdRole, SchedulerConfigSPtr &config,
                                               std::shared_ptr<LatencyPredictor> latencypredictor, size_t localDPRank)
    : role_(pdRole), forwardRespToManagerCall_(cb), schedulerConfig_(config),
      bufferResponseConfig_({config->bufferResponseEnabled, config->prefillExpectedTime, config->decodeExpectedTime}),
      latencypredictor_(latencypredictor), localDPRank_(localDPRank), bufferedResponser_(cb, bufferResponseConfig_),
      dpRankId_(config->dpRankId_)
{
}

void ModelExecOutputHandler::AsyncPublishPrefilledKvCache(ModelBatchResultSPtr &modelBatchResult)
{
    for (int i = 0; i < modelBatchResult->outputs_size(); i++) {
        model_execute_data::CompletionSequenceGroupOutput output = modelBatchResult->outputs(i);
        if (output.samples_size() == 0) {
            throw std::runtime_error("There is no sample in output.");
        }
        model_execute_data::SequenceOutput firstSample = output.samples(0);
        SequenceGroupSPtr seqGroup = LiveInferContext::GetInstance(localDPRank_)->GetSeqGroup(firstSample.seq_id());
        if ((schedulerConfig_->enableChunkedPrefill) && (seqGroup != nullptr) && (!seqGroup->isLastChunk_)) {
            continue;
        }
        if (firstSample.finish_reason() == static_cast<int64_t>(InferStatusType::ITERATION_CONTINUE)) {
            // 返回 continue 表示 prefill 完成了
            if (seqGroup == nullptr) {
                MINDIE_LLM_LOG_INFO("Can not find sequence group, seqId=" << firstSample.seq_id());
                continue;
            }

            ResponseSPtr response = std::make_shared<Response>(seqGroup->metrics_.inferReqId_);
            response->transferStatusFlag = TransferStatusType::PUBLISH_KV_COMPLETE;

            // response contents dedicated to PD disaggregation
            // only item 0 needs to be set since PD disaggregation does not support Parallel Sampling by far
            response->responseContents.resize(1);
            response->responseContents[0].srcBlockTable = seqGroup->pBlockTable;
            response->responseContents[0].singleLLMPrefillReqHandlerId = localDPRank_;
            MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Request-Publish Complete] DP RankId: "
                                << dpRankId_ << ". Request Prefill Complete, requestId: "
                                << seqGroup->metrics_.inferReqId_ << ", seqId: " << firstSample.seq_id()
                                << ", pInstanceId:" << seqGroup->pInstanceId << ", localDPRank_:" << localDPRank_);
            // 返回推理结果给上层的回调函数
            forwardRespToManagerCall_(response);
        }
    }
}

// 以下函数会并发调用，需要保证线程安全
void ModelExecOutputHandler::Entry4Executor(ModelBatchResultSPtr &modelBatchResult)
{
    if (modelBatchResult == nullptr) {
        throw std::runtime_error("modelBatchResult is nullptr.");
    }

    if (modelBatchResult->has_err_msg() && modelBatchResult->err_msg() != "") {
        MINDIE_LLM_LOG_ERROR("Error code from executor: " << modelBatchResult->err_msg());
        mindie_llm::HealthChecker::GetInstance().EnqueueErrorMessage(
            modelBatchResult->err_msg(), "LlmEngine");
        return;
    }

    if (modelBatchResult->outputs_size() == 0) {
        // dummy请求需要减去调度计数
        asyncBatchNum_.fetch_sub(1);
        return;
    }
    LiveInferContextSPtr liveInferContext = LiveInferContext::GetInstance(localDPRank_);
    // 遍历 responses
    std::vector<ResponseSPtr> responsesToCallback;
    std::vector<uint64_t> queueWaitTimes;
    std::vector<uint64_t> prefixCachedTokenNums;

    // 边云特性维护的局部变量
    ForwardMode lastForwardMode = ForwardMode::DUMMY;
    bool layerwiseNeedUpdate = false;
    ForwardMode lwdCurrBatchType = ForwardMode::DUMMY;
    if (schedulerConfig_->layerwiseDisaggregated) {
        lwdCurrBatchType = static_cast<ForwardMode>(!modelBatchResult->layerwise_is_prefill());
    }
    std::deque<SequenceGroupSPtr> recomputeInDBatchQueue; // recompute in decode batch
    
    for (const CompletionSequenceGroupOutput &output : modelBatchResult->outputs()) {
        uint64_t queueWaitTime = 0;
        uint64_t currentPrefixCachedTokenNums = 0;
        auto spanConvert = PROF(INFO, Domain("Engine").SpanStart("ConvertOutputToResponse"));
        if (output.samples_size() == 0) {
            throw std::runtime_error("There is no sample in output.");
        }
        ResponseSPtr response = ConvertSequenceGroupOutputToResponse(output, queueWaitTime,
            currentPrefixCachedTokenNums);
        PROF(spanConvert.SpanEnd());

        if (output.samples_size() > 1) {
            HandleParallelSampling(output, liveInferContext);
        } else if (output.samples_size() == 1) {
            HandleGreedySampling(output.samples(0));
        }

        // ChunkedPrefill特性下，除了最后一个prefill块，前几个块做forward后的输出token要丢弃
        SequenceGroupSPtr seqGroup = liveInferContext->GetSeqGroup(output.samples(0).seq_id());
        bool discardChunkedPrefillReqToken = (seqGroup != nullptr) && ((schedulerConfig_->enableChunkedPrefill) &&
            (!seqGroup->isLastChunk_));

        if (response != nullptr) {
            if (discardChunkedPrefillReqToken) {
                MINDIE_LLM_LOG_DEBUG_REQUEST("The output token of the chunked prefill request need to be discard.");
            } else {
                queueWaitTimes.push_back(queueWaitTime);
                responsesToCallback.push_back(response);
                for (size_t i = 0; i < response->responseContents.size(); i++) {
                    g_decodeTokenCount += response->responseContents[i].speculativeTokenNum;
                }
                MINDIE_LLM_LOG_INFO_TOKEN("[LlmEngine|Request-Response] DP RankId: "
                                          << dpRankId_ << ". Response generated, requestId: " << response->reqId
                                          << ", batchsize: " << modelBatchResult->outputs_size()
                                          << ", total decoded tokens: " << g_decodeTokenCount);
                prefixCachedTokenNums.push_back(currentPrefixCachedTokenNums);
            }
        }
        layerwiseNeedUpdate = layerwiseMixin_.LwdProcessResponse(schedulerConfig_->layerwiseDisaggregated, seqGroup,
            lastForwardMode, lwdCurrBatchType, recomputeInDBatchQueue);
    }
    layerwiseMixin_.LwdProcessRecomputeSeq(layerwiseNeedUpdate, lastForwardMode, recomputeInDBatchQueue);
    layerwiseMixin_.LwdHandlerSubBatchCnt(schedulerConfig_->layerwiseDisaggregated, stagePolicy_, lwdCurrBatchType);

    // 更新batch统计信息（结束时间）
    if (schedulerConfig_->stageSelectPolicy == static_cast<uint32_t>(StagePolicyType::LATENCY_FIRST) ||
        schedulerConfig_->dynamicBatchSizeEnable) {
        // 以下函数可能存在并发问题，需确保UpdateBatchStats()线程安全
        latencypredictor_->UpdateBatchStats();
    }

    asyncBatchNum_.fetch_sub(1);

    // TBC 返回推理结果给上层的回调函数, 如果性能较差需要改为线程池
    for (size_t i = 0; i < responsesToCallback.size(); i++) {
        ResponseSPtr response = responsesToCallback[i];
        response->metrics.batchSize = responsesToCallback.size();
        response->metrics.queueWaitTime = queueWaitTimes.at(i);
        response->metrics.prefixCachedTokenNum = prefixCachedTokenNums.at(i);

        if (bufferResponseConfig_.bufferResponseEnabled) {
            bufferedResponser_.TryRespond(response);
        } else {
            forwardRespToManagerCall_(response);
        }
    }
    SequenceGroupSPtr seqGroup = liveInferContext->GetSeqGroup(modelBatchResult->outputs().at(0).samples(0).seq_id());
    // 标记推理请求结束时间戳
    if (seqGroup != nullptr && stagePolicy_ != nullptr) {
        stagePolicy_->MarkInferenceEndTimeStamp();
    }
    if (role_ == Role::P ||
            (seqGroup != nullptr && liveInferContext->GetInferInstanceFlexRole4Req(seqGroup->requestId) == Role::FlexP)) {
        // notify ms coordinator that these prefilled kv cache can be pulled
        AsyncPublishPrefilledKvCache(modelBatchResult);
    }
}

SequenceGroupSPtr ModelExecOutputHandler::FindRootSequenceGroup(const CompletionSequenceGroupOutput &output,
                                                                LiveInferContextSPtr &liveInferContext) const
{
    SequenceGroupSPtr seqGroup = nullptr;
    for (const model_execute_data::SequenceOutput &sample : output.samples()) {
        seqGroup = liveInferContext->GetSeqGroupFromSeqRootMap(sample.parent_seq_id());
        if (seqGroup != nullptr) {
            break;
        }
    }
    return seqGroup;
}

void ModelExecOutputHandler::ProcessSequenceStatus(SequenceId seqId, int64_t finishReason)
{
    if (finishReason == static_cast<int64_t>(InferStatusType::ITERATION_CONTINUE)) {
        return;
    }
    MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Request-End] DP RankId: " << dpRankId_
                                                              << ". Sequence finished. seqId: " << seqId
                                                              << "; finishReason: "
                                                              << finishReason);
    if (finishReason == static_cast<int64_t>(InferStatusType::END_OF_SENTENCE)) {
        finishedSeqIds_.PushBack(seqId);
    } else {
        execExceptionSeqIds_.PushBack(seqId);
    }
}

void ModelExecOutputHandler::HandleGreedySampling(const model_execute_data::SequenceOutput &sample)
{
    auto spanGreedySampling = PROF(INFO, Domain("Engine").SpanStart("HandleGreedySampling"));
    int64_t tokenIdx = 0;
    for (int64_t output_token : sample.output_token()) {
        if (schedulerConfig_->speculationGamma > 0 && tokenIdx >= sample.num_speculative_tokens()) {
            // 并行解码/mtp返回的token 包含占位符-1和0，无需放入结果token
            break;
        }
        tokenIdx++;

        // <seqid, tokenid> 入队用于更新占位符
        if (output_token != PLACEHOLDER_TOKEN) {
            seqIdToOutputTokenQueue_.PushBack(std::pair<SequenceId, TokenId>{sample.seq_id(), output_token});
        } else if (schedulerConfig_->layerwiseDisaggregated) {
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"seq id is "
                << sample.seq_id() << ", output_token is -1");
        }
    }
    ProcessSequenceStatus(sample.seq_id(), sample.finish_reason());

    PROF(spanGreedySampling.SpanEnd());
}

void ModelExecOutputHandler::HandleParallelSampling(const model_execute_data::CompletionSequenceGroupOutput &output,
                                                    LiveInferContextSPtr &liveInferContext)
{
    auto spanParallelSampling = PROF(INFO, Domain("Engine").SpanStart("HandleParallelSampling"));

    SequenceGroupSPtr rootSeqGrp = FindRootSequenceGroup(output, liveInferContext);
    if (rootSeqGrp == nullptr) {
        throw std::runtime_error("No root seqId exists in the LiveInferContext.");
    }
    // 创建新生成的序列的 SequenceGroup
    for (const model_execute_data::SequenceOutput &sample : output.samples()) {
        SequenceId seqId = sample.seq_id();
        if (seqId == EOS_SEQUENCE_ID) {
            continue;
        }
        if (rootSeqGrp->seqId2ParallelSeqGroup_.Count(seqId) == 0) {
            CreateNewSequenceGroup(sample, rootSeqGrp, liveInferContext);
        } else if (seqId != sample.parent_seq_id()) {
            UpdateSequenceGroup(sample, rootSeqGrp);
        }
    }
    // 遍历 SequenceGroupOutput, 并行采样时会包含多个 SequenceOutput
    std::unordered_set<SequenceId> outputSeqIds;
    for (const model_execute_data::SequenceOutput &sample : output.samples()) {
        SequenceId seqId = sample.seq_id();
        // 当序列结束的时候（最后一个 token 为 EOS ），返回的 seqId 为 EOS_SEQUENCE_ID
        if (seqId == EOS_SEQUENCE_ID) {
            continue;
        }
        SequenceGroupSPtr seqGrp = rootSeqGrp->seqId2ParallelSeqGroup_.Get(seqId).value();
        if (!seqGrp->isNewSeqGroup_) {
            for (TokenId outputToken : sample.output_token()) {
                seqIdToOutputTokenQueue_.PushBack(std::pair<SequenceId, TokenId>{seqId, outputToken});
            }
        }

        ProcessSequenceStatus(sample.seq_id(), sample.finish_reason());

        outputSeqIds.insert(seqId);
    }

    // 被剪枝的序列, 加入到 execExceptionSeqIds_ 中来释放 block
    std::vector<SequenceId> parallelSeqIds = rootSeqGrp->seqId2ParallelSeqGroup_.KeySet();
    for (const auto &seqId : parallelSeqIds) {
        if (outputSeqIds.count(seqId) == 0) {
            execExceptionSeqIds_.PushBack(seqId);
        }
    }

    PROF(spanParallelSampling.SpanEnd());
}

void ModelExecOutputHandler::CreateNewSequenceGroup(const model_execute_data::SequenceOutput &sample,
                                                    SequenceGroupSPtr &rootSeqGrp,
                                                    LiveInferContextSPtr &liveInferContext) const
{
    if (!rootSeqGrp->seqId2ParallelSeqGroup_.Get(sample.parent_seq_id())) {
        MINDIE_LLM_LOG_ERROR("Can not find sequence group for parent seq id=" << sample.parent_seq_id());
        throw std::runtime_error("Can not find sequence group for parent seq id");
    }
    SequenceGroupSPtr parentSeqGrp = rootSeqGrp->seqId2ParallelSeqGroup_.Get(sample.parent_seq_id()).value();
    std::vector<TokenId> promptTokenIds = parentSeqGrp->firstSeq->data_.promptTokenIds;

    // 创建新 Sequence 和 SequenceGroup
    SequenceSPtr newSeq = std::make_shared<Sequence>(
        sample.seq_id(), parentSeqGrp->firstSeq->blockSize_, promptTokenIds);
    std::vector<TokenId> &outputTokenIds = newSeq->data_.outputTokenIds;
    std::copy(sample.output_token().begin(), sample.output_token().end(), std::back_inserter(outputTokenIds));
    SequenceGroupSPtr newSeqGrp =
        std::make_shared<SequenceGroup>(rootSeqGrp->requestId, std::vector<SequenceSPtr>{newSeq});
    newSeqGrp->isNewSeqGroup_ = true;
    newSeqGrp->needUpdate_ = true;
    newSeqGrp->parentSeqId_ = sample.parent_seq_id();

    // 添加到 seqId2ParallelSeqGroup_ 和 liveInferContext->seqId2RootSeqGroupMap_ 中
    rootSeqGrp->seqId2ParallelSeqGroup_.Insert(sample.seq_id(), newSeqGrp);
    liveInferContext->AddIntoSeqRootMap(sample.seq_id(), rootSeqGrp);
    newSeqGrp->UpdateNumComputedTokens(newSeqGrp->firstSeq->GetLen());
}

void ModelExecOutputHandler::UpdateSequenceGroup(const model_execute_data::SequenceOutput &sample,
                                                 SequenceGroupSPtr &rootSeqGrp) const
{
    if (rootSeqGrp == nullptr) {
        throw std::runtime_error("rootSeqGrp is null.");
    }
    if (!rootSeqGrp->seqId2ParallelSeqGroup_.Get(sample.parent_seq_id())) {
        MINDIE_LLM_LOG_ERROR("Can not find sequence group for parent seq id=" << sample.parent_seq_id());
        throw std::runtime_error("Can not find sequence group for parent seq id");
    }
    SequenceGroupSPtr parentSeqGrp = rootSeqGrp->seqId2ParallelSeqGroup_.Get(sample.parent_seq_id()).value();
    if (parentSeqGrp == nullptr) {
        throw std::runtime_error("parentSeqGrp is null.");
    }
    if (!rootSeqGrp->seqId2ParallelSeqGroup_.Get(sample.seq_id())) {
        MINDIE_LLM_LOG_ERROR("Can not find sequence group for seq id=" << sample.seq_id());
        throw std::runtime_error("Can not find sequence group for seq id");
    }
    SequenceGroupSPtr seqGrp = rootSeqGrp->seqId2ParallelSeqGroup_.Get(sample.seq_id()).value();
    if (seqGrp == nullptr) {
        throw std::runtime_error("seqGrp is null.");
    }
    seqGrp->needUpdate_ = true;
    seqGrp->firstSeq->data_.promptTokenIds = parentSeqGrp->firstSeq->data_.promptTokenIds;
    seqGrp->firstSeq->data_.outputTokenIds = parentSeqGrp->firstSeq->data_.outputTokenIds;
    seqGrp->parentSeqId_ = sample.parent_seq_id();
}

void ModelExecOutputHandler::AddOutputsToResponse(ResponseSPtr response,
    const model_execute_data::CompletionSequenceGroupOutput &output) const
{
    response->numParallelTokens = static_cast<size_t>(output.samples(0).num_parallel_tokens());
    // enumerate different sequences in Parallel Sampling
    for (const model_execute_data::SequenceOutput &sample : output.samples()) {
        // 统计末尾占位符数量（sample.num_speculative_tokens()表示有效的token数。output_token中包含占位符（0或者-1），不能传给用户）
        int tokenNum = sample.num_speculative_tokens();
        int trailingPlaceholderNum = 0;
        while (trailingPlaceholderNum < tokenNum &&
               sample.output_token(tokenNum - 1 - trailingPlaceholderNum) == PLACEHOLDER_TOKEN) {
            trailingPlaceholderNum++;
        }
        if (trailingPlaceholderNum == tokenNum) {
            continue; // 没有有效的 token, 不需要返回给上层
        }
        // add new ParallelResponse for this sequence sample
        response->responseContents.emplace_back(ResponseContent{
            .seqId = sample.seq_id(),
            .parentSeqId = sample.parent_seq_id(),
            .finishReason = static_cast<InferStatusType>(sample.finish_reason()),
            .speculativeTokenNum = static_cast<size_t>(sample.num_speculative_tokens()),
            .outTokenIds = std::vector<TokenId>(
                sample.output_token().begin(), sample.output_token().begin() + tokenNum),
            .outLogProbs = std::vector<float>(sample.logprob().begin(), sample.logprob().end()),
            .cumLogProb = sample.cumulative_logprobs(),
            .truncationIndex = sample.truncation_index(),
            .topLogProbTokenIds = std::vector<TokenId>(sample.top_token_ids().begin(), sample.top_token_ids().end()),
            .topLogProbs = std::vector<float>(sample.top_logprobs().begin(), sample.top_logprobs().end()),
            .srcBlockTable = {},
            .singleLLMPrefillReqHandlerId = 0,
            .pdErrorCode = 0
        });
    }
}

ResponseSPtr ModelExecOutputHandler::ConvertSequenceGroupOutputToResponse(
    const model_execute_data::CompletionSequenceGroupOutput &output, uint64_t &queueWaitTime,
    uint64_t &currentPrefixCachedTokenNums)
{
    LiveInferContextSPtr liveInferContext = LiveInferContext::GetInstance(localDPRank_);
    // 并行采样时，从 liveInferContext->seqId2RootSeqGroupMap_ 中获取 seqGroup
    // 只有 RootSeqGroup 包含创建 InferResponse 所需要的信息
    // 注意需要使用 parent_seq_id 去查找，因为某个序列结束时返回的 seqId 为 EOS_SEQUENCE_ID, 无法查到
    // 当使能 best of 时，存在某些 seq 结束导致只返回一个 seq 的情况，此时也需要用 seqId2RootSeqGroupMap_ 去查找
    SequenceGroupSPtr seqGroup = FindRootSequenceGroup(output, liveInferContext);
    if (seqGroup == nullptr) {
        // 非并行采样时, 从 liveInferContext->seqId2SeqGroupMap_ 中获取 seqGroup
        seqGroup = liveInferContext->GetSeqGroup(output.samples(0).seq_id());
    }
    if (seqGroup == nullptr) {
        // TBC 集中式时不同dp会返回全部的结果，导致找不到其他dp的seqGroup而日志刷屏，待修复
        MINDIE_LLM_LOG_DEBUG_REQUEST("Can not find sequence group.");
        return nullptr;
    }

    // 初始化 response
    ResponseSPtr response = std::make_shared<Response>(seqGroup->metrics_.inferReqId_);

    SetResponseFlags(output, response);

    // 设置 iterTimes
    response->iterTimes = seqGroup->iterTimes;
    seqGroup->iterTimes++;
    // 记录response返回时间，用于scheduler计算请求空等时间
    seqGroup->metrics_.responseTime_ = std::chrono::high_resolution_clock::now();
    queueWaitTime = seqGroup->metrics_.queueWaitTime_;
    currentPrefixCachedTokenNums = seqGroup->metrics_.prefixCachedTokenNum_;
    if (bufferResponseConfig_.bufferResponseEnabled) {
        bufferedResponser_.RecordArriveTime(seqGroup->metrics_.inferReqId_, seqGroup->arriveTime);
    }
    if (schedulerConfig_->stageSelectPolicy == static_cast<uint32_t>(StagePolicyType::LATENCY_FIRST) ||
        schedulerConfig_->dynamicBatchSizeEnable) {
        // 以下函数可能存在并发问题，需确保AddPercentileData()线程安全
        latencypredictor_->AddPercentileData(seqGroup, schedulerConfig_);
    }
    // 用 CompletionSequenceGroupOutput 的内容设置 response 的各个字段
    AddOutputsToResponse(response, output);
    if (response->responseContents.empty()) {
        return nullptr;
    }

    return response;
}

void ModelExecOutputHandler::SetResponseFlags(const model_execute_data::CompletionSequenceGroupOutput &output,
                                              ResponseSPtr response) const
{
    // 检查是否所有序列都已完成
    size_t continueSeqCount =
        static_cast<size_t>(std::count_if(output.samples().begin(), output.samples().end(), [](const auto &sample) {
            return sample.finish_reason() == static_cast<int64_t>(InferStatusType::ITERATION_CONTINUE);
        }));
    if (continueSeqCount == 0) {
        response->isEos = true;
        MINDIE_LLM_LOG_INFO_REQUEST("[LlmEngine|Request-End] DP RankId: " << dpRankId_
                                                                  << ". Send eos response. seqId: "
                                                                  << output.samples(0).seq_id());
    }
    response->inferStatusFlag = static_cast<InferStatusType>(output.samples(0).finish_reason());
    LiveInferContextSPtr liveInferContext = LiveInferContext::GetInstance(localDPRank_);
    SequenceGroupSPtr seqGroup = liveInferContext->GetSeqGroup(output.samples(0).seq_id());
    if (role_ == Role::P ||
        (seqGroup != nullptr && liveInferContext->GetInferInstanceFlexRole4Req(seqGroup->requestId) == Role::FlexP)) {
        response->transferStatusFlag = TransferStatusType::PREFILL_COMPLETE;
    }
}

ConcurrentDeque<SequenceId> &ModelExecOutputHandler::GetFinishedSeqIds() { return finishedSeqIds_; }

ConcurrentDeque<SequenceId> &ModelExecOutputHandler::GetExceptionSeqIds() { return execExceptionSeqIds_; }

ConcurrentDeque<std::pair<SequenceId, TokenId>> &ModelExecOutputHandler::GetSeqIdToOutputTokenQueue()
{
    return seqIdToOutputTokenQueue_;
}

std::atomic<size_t> &ModelExecOutputHandler::GetAsyncBatchNum() { return asyncBatchNum_; }