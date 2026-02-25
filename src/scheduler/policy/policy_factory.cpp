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
#include <stdexcept>
#include "policy.h"
#include "fcfs_policy.h"
#include "layerwise_fcfs_policy.h"
#include "pdds_policy.h"
#include "stage_policy/latency_stage_policy.h"
#include "stage_policy/tpt_stage_policy.h"
#include "stage_policy/prefill_first_policy.h"
#include "stage_policy/edge_cloud_policy.h"
#include "policy_factory.h"


namespace mindie_llm {
std::shared_ptr<Policy> PolicyFactory::CreatePolicy(Role role, std::shared_ptr<SchedulerConfig> &schedulerConfig,
                                                    BlockSpaceManagerSPtr &blockManager, PDPriorityType pdPriorityType)
{
    std::shared_ptr<Policy> policy;
    switch (role) {
        case Role::PnD:
        case Role::FlexPnD:
        case Role::FlexP:
        case Role::FlexD:
            if (schedulerConfig->layerwiseDisaggregated) {
                policy = std::make_shared<LayerwiseFcfsPolicy>(schedulerConfig, blockManager);
                break;
            }
            if (pdPriorityType == PDPriorityType::PREFILL_FIRST) {
                // 当前Prefill只支持FCFS调度策略。后续新增策略，Prefill时候根据prefillPolicyType初始化调度策略
                policy = std::make_shared<FcfsPolicy>(schedulerConfig, blockManager);
            } else {
                // 当前Decode只支持FCFS调度策略。后续新增策略，decode时候根据decodePolicyType初始化调度策略
                policy = std::make_shared<FcfsPolicy>(schedulerConfig, blockManager);
            }
            break;
        case Role::P:
        case Role::D: policy = std::make_shared<PDDSPolicy>(schedulerConfig, blockManager, role); break;
        default: throw std::runtime_error("Not support role. role=" + std::to_string(static_cast<uint8_t>(role)));
    }
    return policy;
}

std::shared_ptr<KVTransferPolicy> PolicyFactory::CreateTransferPolicy(Role role,
                                                                      std::shared_ptr<SchedulerConfig> &schedulerConfig,
                                                                      BlockSpaceManagerSPtr &blockManager)
{
    return std::make_shared<KVTransferSchedulePolicy>(schedulerConfig, blockManager, role);
}

std::shared_ptr<StagePolicy> PolicyFactory::CreateStagePolicy(const SchedulerConfigSPtr schedulerConfig,
                                                              std::shared_ptr<LatencyPredictor> predictor,
                                                              std::shared_ptr<BlockSpaceManager> blockManager,
                                                              Role role)
{
    StagePolicyType policyType = static_cast<StagePolicyType>(schedulerConfig->stageSelectPolicy);
    std::shared_ptr<StagePolicy> stagePolicyPtr;

    if (role == Role::FlexD || role == Role::FlexP || role == Role::FlexPnD) {
        return std::make_shared<TimeDivisionPolicy>();
    }

    switch (policyType) {
        case StagePolicyType::PREFILL_FIRST: stagePolicyPtr = std::make_shared<PrefillFirstPolicy>(); break;
        case StagePolicyType::FIXED_COST_TIME_TPT_FIRST:
            stagePolicyPtr = std::make_shared<TptStagePolicy>(schedulerConfig);
            break;
        case StagePolicyType::LATENCY_FIRST:
            stagePolicyPtr = std::make_shared<LatencyStagePolicy>(schedulerConfig, predictor, blockManager);
            break;
        // 边云协同场景的StagePolicy
        case StagePolicyType::EDGE_CLOUD:
            stagePolicyPtr = std::make_shared<EdgeCloudPolicy>(schedulerConfig->batchPnum);
            break;
        default:
            throw std::runtime_error("Not support stage policy. policy=" +
                                     std::to_string(schedulerConfig->stageSelectPolicy));
    }
    return stagePolicyPtr;
}
} // namespace mindie_llm