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
#include <chrono>
#include "policy/stage_policy/edge_cloud_policy.h"
#include "layerwise_mixin/layerwise_mixin.h"

using namespace std;
using namespace std::chrono;
using std::chrono::system_clock;

namespace mindie_llm {
void LayerwiseMixin::LwdPrepareBatch(bool layerwiseDisaggregated, SchedulerOutputs &scheduleOut) const
{
    if (!layerwiseDisaggregated) {
        return;
    }
    if (scheduleOut.forwardMode_ == ForwardMode::PREFILL) {
        for (auto prefillSeqGrpSPtr : scheduleOut.scheduledSeqGroups_) {
            SequenceGroupSPtr prefillSeqGroup = prefillSeqGrpSPtr->seqGroup_;
            auto prefillSeqId = prefillSeqGroup->firstSeq->seqId_;
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|Scheduler]:"<<"prefillSeqId in prefill batch is: "
                                << prefillSeqId << ", set prefill:1, recompute:0");

            bool isPrefill = true;
            prefillSeqGroup->firstSeq->data_.SetLayerwiseStage(isPrefill);
            if (prefillSeqGroup->firstSeq->data_.layerwiseRecompute_) {
                prefillSeqGroup->firstSeq->data_.layerwiseRecompute_ = false;
                prefillSeqGroup->firstSeq->data_.layerwiseRecomputeReturn_ = false;
                prefillSeqGroup->firstSeq->data_.layerwiseRunning_ = true;
            }
        }
    }
}

void LayerwiseMixin::LwdEngineAddBatchCnt(bool layerwiseDisaggregated, std::shared_ptr<StagePolicy> stagePolicy,
    SchedulerOutputs &scheduleOut) const
{
    if (!layerwiseDisaggregated) {
        return;
    }
    std::shared_ptr<EdgeCloudPolicy> lwdPolicy = std::static_pointer_cast<EdgeCloudPolicy>(stagePolicy);
    lwdPolicy->LayerwiseAddBatchCnt(scheduleOut.forwardMode_);
}

void LayerwiseMixin::LwdComputeArrTimeGap(bool layerwiseDisaggregated, SequenceGroupSPtr &seqGroup,
    SequenceGroupSPtr lastSeqGroup)
{
    // 边云特性动态切块使用，给TG侧传请求到达时间间隔用于切图
    if (!layerwiseDisaggregated) {
        return;
    }
    int32_t timeGap = -1;
    auto currentTime = std::chrono::high_resolution_clock::now();
    if (seqGroup->arriveTime != std::chrono::high_resolution_clock::time_point()) {
        currentTime = seqGroup->arriveTime;
    }
    if (lastSeqGroup != nullptr) {
        auto lastPArriveTime = lastSeqGroup->arriveTime;
        if (lastSeqGroup->firstSeq->data_.layerwiseRecompute_) {
            lastPArriveTime = lastSeqGroup->recomputeArriveTime_;
        }
        timeGap = static_cast<int32_t>(duration_cast<milliseconds>(currentTime - lastPArriveTime).count());
    } else if (lastArriveTime_ != std::chrono::high_resolution_clock::time_point()) { // 说明已经设置有效时间
        timeGap = static_cast<int32_t>(duration_cast<milliseconds>(currentTime - lastArriveTime_).count());
    }
    seqGroup->requestGap_ = timeGap;
    if (seqGroup->arriveTime != std::chrono::high_resolution_clock::time_point()) {
        lastArriveTime_ = seqGroup->arriveTime;
    } else {
        lastArriveTime_ = currentTime;
    }
}

void LayerwiseMixin::LwdSetRecomputeArrTime(bool layerwiseDisaggregated, SequenceGroupSPtr &seqGroup,
    SequenceGroupSPtr lastSeqGroup)
{
    if (!layerwiseDisaggregated || !seqGroup->firstSeq->data_.layerwiseRecompute_) {
        return;
    }
    int32_t timeGap = -1;
    seqGroup->recomputeArriveTime_ = std::chrono::high_resolution_clock::now();
    if (lastSeqGroup != nullptr) {
        auto lastPArriveTime = lastSeqGroup->arriveTime;
        if (lastSeqGroup->firstSeq->data_.layerwiseRecompute_) {
            lastPArriveTime = lastSeqGroup->recomputeArriveTime_;
        }
        timeGap = static_cast<int32_t>(duration_cast<milliseconds>(seqGroup->recomputeArriveTime_
            - lastPArriveTime).count());
    } else if (lastArriveTime_ != std::chrono::high_resolution_clock::time_point()) { // 说明已经设置有效时间
        timeGap = static_cast<int32_t>(duration_cast<milliseconds>(seqGroup->recomputeArriveTime_
            - lastArriveTime_).count());
    }
    seqGroup->requestGap_ = timeGap;
    lastArriveTime_ = seqGroup->recomputeArriveTime_;
}

bool LayerwiseMixin::LwdProcessResponse(bool layerwiseDisaggregated, SequenceGroupSPtr seqGroup,
    ForwardMode &lastForwardMode, ForwardMode lwdCurrBatchType,
    std::deque<SequenceGroupSPtr> &recomputeInDBatchQueue) const
{
    if (!layerwiseDisaggregated) {
        return false;
    }
    if (seqGroup == nullptr) {
        lastForwardMode = lwdCurrBatchType;
        std::string forwardModeString = lastForwardMode == ForwardMode::PREFILL ? "prefill" : "decode";
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"seqGoup is nullptr!!! "
            << forwardModeString<<" return");
        return false;
    }
    auto returnSeqId = seqGroup->firstSeq->seqId_;
    ForwardMode forwardMode = seqGroup->IsLayerwisePrefill() ? ForwardMode::PREFILL : ForwardMode::DECODE;
    if (forwardMode == ForwardMode::PREFILL) {
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"prefill return seq id:"<< returnSeqId);
        
        // recompute in decode batch
        if (seqGroup->firstSeq->data_.layerwiseRunning_) {
            recomputeInDBatchQueue.emplace_back(seqGroup);
        }

        if (!seqGroup->firstSeq->data_.layerwiseRecompute_) {
            bool isPrefill = false;
            seqGroup->firstSeq->data_.SetLayerwiseStage(isPrefill);
            seqGroup->firstSeq->data_.layerwiseRunning_ = false;
        } else {
            // 记录recompute的seq返回，之后才允许下发重计算的P
            seqGroup->firstSeq->data_.layerwiseRecomputeReturn_ = true;
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"recompute return seq id:"<< returnSeqId);
        }
    } else {
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"decode return seq id:"<< returnSeqId);
    }
    if (lastForwardMode == ForwardMode::DUMMY) {
        lastForwardMode = forwardMode;
    } else {
        if (lastForwardMode != forwardMode) {
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"P/D Type is not same in one batch!!!!!");
        }
        lastForwardMode = forwardMode;
    }
    return true;
}

void LayerwiseMixin::LwdProcessRecomputeSeq(bool layerwiseNeedUpdate, ForwardMode lastForwardMode,
    const std::deque<SequenceGroupSPtr> &recomputeInDBatchQueue) const
{
    if (!layerwiseNeedUpdate) {
        return;
    }
    // recompute in decode batch
    if (lastForwardMode == ForwardMode::DECODE && recomputeInDBatchQueue.size() > 0) {
        for (auto recomputeSeqGroup: recomputeInDBatchQueue) {
            auto recomputeSeqId = recomputeSeqGroup->firstSeq->seqId_;
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|handler] "<<"if SeqId="
                << recomputeSeqId << "return in decode batch, ignore");
            // 兜底一
            bool isPrefill = true;
            recomputeSeqGroup->firstSeq->data_.SetLayerwiseStage(isPrefill);
            recomputeSeqGroup->firstSeq->data_.layerwiseRunning_ = true;
            // 兜底二
            recomputeSeqGroup->firstSeq->data_.layerwiseDiscard_ = true;
        }
    }
}

void LayerwiseMixin::LwdHandlerSubBatchCnt(bool layerwiseNeedUpdate, std::shared_ptr<StagePolicy> stagePolicy,
    ForwardMode lastForwardMode) const
{
    if (!layerwiseNeedUpdate) {
        return;
    }
    // 边云协同场景状态机更新
    std::shared_ptr<EdgeCloudPolicy> lwdPolicy = std::static_pointer_cast<EdgeCloudPolicy>(stagePolicy);
    lwdPolicy->LayerwiseSubBatchCnt(lastForwardMode);
}
}