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

#include "live_infer_context.h"

#include "system_log.h"
#include "utils/spin_lock_guard.h"

namespace mindie_llm {
LiveInferContextSPtr LiveInferContext::GetInstance(size_t localDPRank)
{
    static std::vector<LiveInferContextSPtr> instances(MAX_DP_COUNT, nullptr);
    static std::mutex initMutex;
    if (localDPRank >= MAX_DP_COUNT) {
        throw std::runtime_error("dp num max than 16 in LiveInferContext.");
    }

    if (instances[localDPRank] == nullptr) {
        std::unique_lock<std::mutex> lock(initMutex);
        if (instances[localDPRank] == nullptr) {
            instances[localDPRank] = std::make_shared<LiveInferContext>();
        }
    }

    return instances[localDPRank];
}

LiveInferContext::LiveInferContext() { pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE); }

LiveInferContext::~LiveInferContext() { pthread_spin_destroy(&spinlock_); }

void LiveInferContext::Add(SequenceGroupSPtr &seqGroup)
{
    SpinLockGuard lockGuard(spinlock_);

    bool reqIdExists = reqId2SeqGroupMap_.find(seqGroup->requestId) != reqId2SeqGroupMap_.end();
    bool seqIdExists = seqId2SeqGroupMap_.find(seqGroup->firstSeq->seqId_) != seqId2SeqGroupMap_.end();
    if (reqIdExists) {
        LOG_WARN_LLM << "The sequence group(requestId=" << seqGroup->requestId << ", seqId="
                            << seqGroup->firstSeq->seqId_ << ") requestId already exist";
        return;
    }
    if (seqIdExists && seqGroup->firstSeq->seqId_ == SIMULATE_SEQUENCE_ID) {
        reqId2SeqGroupMap_.insert({seqGroup->requestId, seqGroup});
        LOG_DEBUG_LLM << "Simulate seqId=" << seqGroup->firstSeq->seqId_
            << " already in seqId2SeqGroupMap_, keep first entry. requestId="
            << seqGroup->requestId << " registered in reqId2SeqGroupMap_ only.";
        return;
    }
    if (seqIdExists) {
        LOG_WARN_LLM << "The sequence group(requestId=" << seqGroup->requestId << ", seqId="
                            << seqGroup->firstSeq->seqId_ << ") seqId already exist";
        return;
    }
    seqId2SeqGroupMap_.insert({seqGroup->firstSeq->seqId_, seqGroup});
    reqId2SeqGroupMap_.insert({seqGroup->requestId, seqGroup});
}

void LiveInferContext::AddIntoSeqRootMap(SequenceId seqId, SequenceGroupSPtr &rootSeqGroup)
{
    SpinLockGuard lockGuard(spinlock_);

    if (seqId2RootSeqGroupMap_.find(seqId) != seqId2RootSeqGroupMap_.end()) {
        LOG_WARN_LLM << "The mapping seqId(" << seqId << ") to root sequence group is aready exist";
        return;
    }

    seqId2RootSeqGroupMap_.insert({seqId, rootSeqGroup});
}

void LiveInferContext::Remove(SequenceId seqId)
{
    SpinLockGuard lockGuard(spinlock_);

    auto it = seqId2SeqGroupMap_.find(seqId);
    if (it == seqId2SeqGroupMap_.end()) {
        LOG_DEBUG_LLM.SetType(LogType::REQUEST) << "The sequence id(" << seqId << ") is not exist";
        return;
    }

    RequestId reqId = it->second->requestId;
    reqId2SeqGroupMap_.erase(reqId);
    reqId2UsedInstanceRoleMap_.erase(reqId);
    seqId2SeqGroupMap_.erase(seqId);
}

void LiveInferContext::Remove(RequestId reqId)
{
    SpinLockGuard lockGuard(spinlock_);

    auto it = reqId2SeqGroupMap_.find(reqId);
    if (it == reqId2SeqGroupMap_.end()) {
        LOG_DEBUG_LLM.SetType(LogType::REQUEST) << "The request id(" << reqId << ") is not exist";
        return;
    }

    SequenceId seqId = it->second->firstSeq->seqId_;
    SequenceGroupSPtr seqGroup = it->second;
    reqId2SeqGroupMap_.erase(reqId);
    reqId2UsedInstanceRoleMap_.erase(reqId);

    auto seqIt = seqId2SeqGroupMap_.find(seqId);
    if (seqIt != seqId2SeqGroupMap_.end() && seqIt->second == seqGroup) {
        seqId2SeqGroupMap_.erase(seqId);
    }
}

void LiveInferContext::RemoveFromSeqRootMap(SequenceId seqId)
{
    SpinLockGuard lockGuard(spinlock_);

    auto it = seqId2RootSeqGroupMap_.find(seqId);
    if (it == seqId2RootSeqGroupMap_.end()) {
        LOG_DEBUG_LLM.SetType(LogType::REQUEST) << "The sequence id(" << seqId << ") is not in seqId2RootSeqGroupMap_";
        return;
    }

    seqId2RootSeqGroupMap_.erase(seqId);
}

void LiveInferContext::RemoveAll()
{
    SpinLockGuard lockGuard(spinlock_);
    reqId2SeqGroupMap_.clear();
    seqId2SeqGroupMap_.clear();
    seqId2RootSeqGroupMap_.clear();
    reqId2UsedInstanceRoleMap_.clear();
}

SequenceSPtr LiveInferContext::GetSeq(SequenceId seqId)
{
    SpinLockGuard lockGuard(spinlock_);
    auto it = seqId2SeqGroupMap_.find(seqId);
    return it != seqId2SeqGroupMap_.end() ? it->second->firstSeq : nullptr;
}

SequenceGroupSPtr LiveInferContext::GetSeqGroup(RequestId reqId)
{
    SpinLockGuard lockGuard(spinlock_);
    auto it = reqId2SeqGroupMap_.find(reqId);
    return it != reqId2SeqGroupMap_.end() ? it->second : nullptr;
}

SequenceGroupSPtr LiveInferContext::GetSeqGroup(SequenceId seqId)
{
    SpinLockGuard lockGuard(spinlock_);
    auto it = seqId2SeqGroupMap_.find(seqId);
    return it != seqId2SeqGroupMap_.end() ? it->second : nullptr;
}

SequenceGroupSPtr LiveInferContext::GetSeqGroupFromSeqRootMap(SequenceId seqId)
{
    SpinLockGuard lockGuard(spinlock_);
    auto it = seqId2RootSeqGroupMap_.find(seqId);
    return it != seqId2RootSeqGroupMap_.end() ? it->second : nullptr;
}

void LiveInferContext::SetInferInstanceFlexRole4Req(RequestId reqId, Role role)
{
    SpinLockGuard lockGuard(spinlock_);
    if (reqId2UsedInstanceRoleMap_.count(reqId) != 0) {
        return;
    }
    reqId2UsedInstanceRoleMap_[reqId] = role;
}

Role LiveInferContext::GetInferInstanceFlexRole4Req(RequestId reqId)
{
    SpinLockGuard lockGuard(spinlock_);
    auto it = reqId2UsedInstanceRoleMap_.find(reqId);
    return it != reqId2UsedInstanceRoleMap_.end() ? it->second : Role::FlexPnD;
}

} // namespace mindie_llm