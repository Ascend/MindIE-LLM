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

// 边云协同新增

#include "log.h"
#include "edge_cloud_policy.h"

namespace mindie_llm {
PDPriorityType EdgeCloudPolicy::Apply(ConcurrentDeque<SequenceGroupSPtr> &waiting,
                                      ConcurrentDeque<SequenceGroupSPtr> &running,
                                      ConcurrentDeque<SequenceGroupSPtr> &swapped)
{
    ((void)swapped);
    if (prefillBatchCount_ < batchPnum_ && decodeBatchCount_ == 0) {
        // 00
        if (!waiting.Empty()) {
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|EdgeCloudPolicy] "
                <<"prefillBatchCount_: " << prefillBatchCount_ <<
                ", decodeBatchCount_: 0, waiting_ is not empty,schedule P");
            return PDPriorityType::PREFILL_FIRST;
        } else if (!running.Empty()) {
            MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|EdgeCloudPolicy] "
                <<"prefillBatchCount_: " << prefillBatchCount_ <<
                ", decodeBatchCount_: 0, running_ is not empty, schedule D");
            return PDPriorityType::DECODE_FIRST;
        } else {
            return PDPriorityType::PREFILL_FIRST;
        }
    } else if (prefillBatchCount_ < batchPnum_ && decodeBatchCount_ == 1) {
        // 01
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|EdgeCloudPolicy] "
            << "prefillBatchCount_: " << prefillBatchCount_ << ", decodeBatchCount_: 1, schedule P");
        return PDPriorityType::PREFILL_FIRST;
    } else if (prefillBatchCount_ == batchPnum_ && decodeBatchCount_ == 0) {
        // 10
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|EdgeCloudPolicy] "
            << "prefillBatchCount_: " << prefillBatchCount_ << ", decodeBatchCount_: 0, schedule D");
        return PDPriorityType::DECODE_FIRST;
    } else {
        // 11
        MINDIE_LLM_LOG_ERROR("[layerwiseDisaggregated|EdgeCloudPolicy] "
            << "P&D are scheduled, should be refused by batchtimes, exception occurred!!!");
        throw std::runtime_error("Try to schedule one more batch when prefillBatchCount_="
            + std::to_string(prefillBatchCount_) + " and decodeBatchCount_=1. "
            "It is not allowd because maxDispatchBatchNum=" + std::to_string(batchPnum_ + 1) + ".");
    }
    return PDPriorityType::PREFILL_FIRST;
}
void EdgeCloudPolicy::LayerwiseAddBatchCnt(ForwardMode forwardMode)
{
    if (forwardMode == ForwardMode::PREFILL && prefillBatchCount_ >= 0 && prefillBatchCount_ < batchPnum_) {
        prefillBatchCount_ += 1;
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|predictor] "<<"prefill change from " <<
            (prefillBatchCount_ - 1) << " to " << prefillBatchCount_);
    } else if (forwardMode == ForwardMode::DECODE && decodeBatchCount_ == 0) {
        decodeBatchCount_ = 1;
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|predictor] "<<"decode change from 0 to 1");
    } else {
        std::string forwardModeString = forwardMode == ForwardMode::PREFILL ? "prefill" : "decode";
        MINDIE_LLM_LOG_ERROR("[layerwiseDisaggregated|predictor] Wrong batch stats in EdgeCloud SaveBatchStats! "
            "batch type is: " << forwardModeString << ", prefillBatchCount_ is: " << prefillBatchCount_
            << ", decodeBatchCount_ is: " << decodeBatchCount_);
    }
}

void EdgeCloudPolicy::LayerwiseSubBatchCnt(ForwardMode forwardMode)
{
    if (forwardMode == ForwardMode::PREFILL && prefillBatchCount_ >= 1 && prefillBatchCount_ <= batchPnum_) {
        prefillBatchCount_ -= 1;
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|predictor] "<<"prefill change from " <<
            (prefillBatchCount_ + 1) << " to " << prefillBatchCount_);
    } else if (forwardMode == ForwardMode::DECODE && decodeBatchCount_ == 1) {
        decodeBatchCount_ = 0;
        MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|predictor] "<<"decode change from 1 to 0");
    } else {
        std::string forwardModeString = forwardMode == ForwardMode::PREFILL ? "prefill" : "decode";
        MINDIE_LLM_LOG_ERROR("[layerwiseDisaggregated|predictor] Wrong batch stats in EdgeCloud UpdateBatchStats! "
            "batch type is: " << forwardModeString << ", prefillBatchCount_ is: " << prefillBatchCount_
            << ", decodeBatchCount_ is " << decodeBatchCount_);
    }
}
}

