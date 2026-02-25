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
#include "sequence_group.h"
#include <stdexcept>
#include "src/engine/lora_manager.h"

namespace mindie_llm {
SequenceGroup::SequenceGroup(RequestId &tRequestId, const std::vector<SequenceSPtr> &tSeqs)
    : requestId(tRequestId), seqs_(tSeqs)
{
    if (seqs_.empty()) {
        throw std::invalid_argument(
            "Cannot create SequenceGroup with empty sequences, requestId = " + requestId);
    }
    arriveTime = std::chrono::high_resolution_clock::now();
    firstSeq = seqs_[0];
}

SequenceGroup::SequenceGroup(RequestId &tRequestId, const std::vector<SequenceSPtr> &tSeqs,
                             const SamplingParamsSPtr &tSampling)
    : requestId(tRequestId), seqs_(tSeqs), sampling(tSampling)
{
    if (seqs_.empty()) {
        throw std::invalid_argument(
            "Cannot create SequenceGroup with empty sequences, requestId = " + requestId);
    }
    arriveTime = std::chrono::high_resolution_clock::now();
    firstSeq = seqs_[0];
}

SequenceGroup::SequenceGroup(RequestId &tRequestId, const std::vector<SequenceSPtr> &tSeqs,
                             const SamplingParamsSPtr &tSampling, const std::optional<std::string> &tLoraId,
                             size_t tRankId)
    : requestId(tRequestId), seqs_(tSeqs), sampling(tSampling), rankId_(tRankId)
{
    if (seqs_.empty()) {
        throw std::invalid_argument(
            "Cannot create SequenceGroup with empty sequences, requestId = " + requestId);
    }
    arriveTime = std::chrono::high_resolution_clock::now();
    firstSeq = seqs_[0];
    
    auto loraManager = mindie_llm::LoraManager::GetInstance(rankId_);
    if (loraManager && loraManager->ValidateLoraId(tLoraId)) {
        loraId_ = tLoraId;
        loraManager->IncLoraRef(loraId_);
    } else {
        loraId_ = "None";
    }
}
SequenceGroup::~SequenceGroup()
{
    if (loraId_.has_value() && loraId_ != "None") {
        auto loraManager = mindie_llm::LoraManager::GetInstance(rankId_);
        if (loraManager) {
            loraManager->DecLoraRef(loraId_);
        }
    }
}

std::vector<SequenceSPtr> SequenceGroup::GetFirstSequence(const SequenceStatus status)
{
    /* 默认参数，0表示没有传入状态 */
    if (static_cast<int>(status) == 0) {
        return seqs_;
    }
    // 如果传入的状态和第一个序列的状态相同，则返回所有序列
    if (firstSeq->status_ == status) {
        return seqs_;
    }
    // 否则，返回空序列
    return {};
}

std::vector<SequenceSPtr> SequenceGroup::GetSequences(const SequenceStatus status)
{
    if (sampling && sampling->enableParallelSampling) {
        return GetParallelSequences(status);
    }
    return GetFirstSequence(status);
}

/**
获取所有的beam search的seqgrp下的所有sequence。status 0 表示获取所有状态
 */
std::vector<SequenceSPtr> SequenceGroup::GetParallelSequences(const SequenceStatus status) const
{
    std::vector<SequenceSPtr> seqs;

    std::vector<SequenceId> parallelSeqIds = seqId2ParallelSeqGroup_.KeySet();
    for (auto seqId : parallelSeqIds) {
        std::optional<SequenceGroupSPtr> seqGrpOpt = seqId2ParallelSeqGroup_.Get(seqId);
        if (seqGrpOpt.has_value()) {
            SequenceGroupSPtr seqGrpSPtr = seqGrpOpt.value();
            if (status == SequenceStatus::ALL_STATUS || seqGrpSPtr->firstSeq->status_ == status) {
                seqs.push_back(seqGrpSPtr->firstSeq);
            }
        }
    }
    return seqs;
}

std::vector<SequenceGroupSPtr> SequenceGroup::GetParallelSeqGrp()
{
    std::vector<SequenceGroupSPtr> parallelSeqGrp;
    std::vector<SequenceId> parallelSeqIds = seqId2ParallelSeqGroup_.KeySet();
    for (auto seqId : parallelSeqIds) {
        std::optional<SequenceGroupSPtr> seqGrpOpt = seqId2ParallelSeqGroup_.Get(seqId);
        if (seqGrpOpt.has_value()) {
            parallelSeqGrp.push_back(seqGrpOpt.value());
        }
    }
    return parallelSeqGrp;
}

void SequenceGroup::UpdateNumComputedTokens(size_t numNewComputedTokens)
{
    for (auto seq : seqs_) {
        if (!seq->IsFinished()) {
            seq->data_.UpdateNumComputedTokens(numNewComputedTokens);
        }
    }
}

int SequenceGroup::GetMaxNumRunningSeqs() const
{
    if (sampling && !sampling->enableParallelSampling) {
        return firstSeq->IsFinished() ? 0 : 1;
    }
    std::vector<SequenceSPtr> seqs = GetParallelSequences(SequenceStatus::RUNNING);
    return seqs.size();
}

bool SequenceGroup::IsPrefill() const { return firstSeq->IsPrefill(); }

bool SequenceGroup::IsLayerwisePrefill() const { return firstSeq->IsLayerwisePrefill(); }

bool SequenceGroup::IsFinished() const { return firstSeq->IsFinished(); }

bool SequenceGroup::IsSimulateRequest() const { return firstSeq->seqId_ == SIMULATE_SEQUENCE_ID; }

ScheduledSequenceGroup::ScheduledSequenceGroup(const SequenceGroupSPtr &tSeqGroup, const size_t tTokenChunkSize,
                                               bool enableChunked)
    : seqGroup_(tSeqGroup), tokenChunkSize_(tTokenChunkSize)
{
    if (enableChunked) {
        SequenceSPtr seq = seqGroup_->firstSeq;
        if (seq->GetNumComputedTokens() + tokenChunkSize_ >= seq->data_.promptTokenIds.size()) {
            seqGroup_->isLastChunk_ = true;
        } else {
            seqGroup_->isLastChunk_ = false;
        }
    }
}

bool SchedulerOutputs::IsEmpty()
{
    return scheduledSeqGroups_.empty() && blocksToSwapIn_.empty() && blocksToSwapOut_.empty();
}

bool SchedulerKVTransferOutput::IsEmpty() { return pullSeqGroups.empty(); }

} // namespace mindie_llm
