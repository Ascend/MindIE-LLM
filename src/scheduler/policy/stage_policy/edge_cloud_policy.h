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
#ifndef EDGE_CLOUD_POLICY_H
#define EDGE_CLOUD_POLICY_H

#include "sequence_group.h"
#include "policy/stage_policy/stage_policy.h"
#include "latency_predictor/latency_predictor.h"

namespace mindie_llm {
class EdgeCloudPolicy final : public StagePolicy {
public:
    explicit EdgeCloudPolicy(uint32_t batchPnum) : batchPnum_(batchPnum) {};
    PDPriorityType Apply([[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &waiting,
                         [[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &running,
                         [[maybe_unused]] ConcurrentDeque<SequenceGroupSPtr> &swapped) override;

    bool lastScheduleEmpty{false};

    void LayerwiseAddBatchCnt(ForwardMode forwardMode);

    void LayerwiseSubBatchCnt(ForwardMode forwardMode);

    int GetPrefillBatchCnt() const {return prefillBatchCount_;};

    int GetDecodeBatchCnt() const {return decodeBatchCount_;};

private:
    std::shared_ptr<LatencyPredictor> predictor_;
    int prefillBatchCount_{0};
    int decodeBatchCount_{0};
    const int batchPnum_{1};
};
} // namespace mindie_llm

#endif