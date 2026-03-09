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

#include "pre_scheduler.h"
#include "process_group.h"
#include "thread_group_cc.h"
#include "system_log.h"

namespace mindie_llm {
std::vector<SchedInfo> PreScheduler::ShareSchedInfo(const SchedInfo &schedInfo, size_t dpRank, bool enableDistributed)
{
    if (enableDistributed) {
        return ShareSchedInfoCrossNode(schedInfo);
    } else {
        return ShareSchedInfoCrossDP(schedInfo, dpRank);
    }
}

std::vector<SchedInfo> PreScheduler::ShareSchedInfoCrossDP(const SchedInfo &schedInfo, size_t dpRank)
{
    // 1. 使用ThreadGroupCC进行allGather通信
    std::vector<int64_t> sendData = {
        static_cast<int64_t>(schedInfo.pdPriority_),
        schedInfo.waitingSeqGroupNum_,
        schedInfo.runningSeqGroupNum_,
    };
    std::vector<std::vector<int64_t>> recvData;
    ThreadGroupCC::GetInstance().AllGather(sendData, recvData, dpRank);

    // 2. 转换成SchedInfoPerRank
    std::vector<SchedInfo> result(recvData.size());
    size_t indexNum = 3;
    for (size_t i = 0; i < recvData.size(); ++i) {
        if (recvData[i].size() < indexNum) {
            throw std::runtime_error("Invalid received data from dpRank " + std::to_string(i) +
                ": expected 3 elements, got " + std::to_string(recvData[i].size()));
        }
        result[i].pdPriority_ = static_cast<PDPriorityType>(recvData[i][0]);
        result[i].waitingSeqGroupNum_ = recvData[i][1]; // 1: waitingSeqGroupNum_
        result[i].runningSeqGroupNum_ = recvData[i][2]; // 2: runningSeqGroupNum_
    }
    return result;
}

std::vector<SchedInfo> PreScheduler::ShareSchedInfoCrossNode(const SchedInfo &schedInfo)
{
    // 1. 使用ThreadGroupCC进行allGather通信
    std::vector<int64_t> sendData = {
        static_cast<int64_t>(schedInfo.pdPriority_),
        schedInfo.waitingSeqGroupNum_,
        schedInfo.runningSeqGroupNum_,
    };
    std::vector<torch::Tensor> inputs;
    inputs.emplace_back(torch::tensor(sendData, torch::dtype(torch::kInt64).device(c10::kCPU)));
    try {
        std::vector<std::vector<torch::Tensor>> outputs = ProcessGroup::GetInstance().AllGather(inputs);
        if (outputs.empty() || outputs[0].empty()) {
            return {};
        }
        
        // 2. 转换成SchedInfoPerRank
        std::vector<SchedInfo> result(outputs[0].size());
        for (size_t i = 0; i < outputs[0].size(); ++i) {
            torch::Tensor tensor = outputs[0][i];
            result[i].pdPriority_ = static_cast<PDPriorityType>(tensor[0].item<int64_t>());
            result[i].waitingSeqGroupNum_ = tensor[1].item<int64_t>();
            result[i].runningSeqGroupNum_ = tensor[2].item<int64_t>();
        }
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR_LLM << "ShareSchedInfoCrossNode failed: outputs is invalid.";
        return {};
    }
}

PDPriorityType PreScheduler::DecidePDPriority(const std::vector<SchedInfo> &schedInfos)
{
    std::vector<SchedInfo> decideScheduleInfos;
    for (auto it = schedInfos.begin(); it != schedInfos.end(); ++it) {
        bool allQueEmpty = (it->waitingSeqGroupNum_ + it->runningSeqGroupNum_ + it->swapSeqGroupNum_) == 0;
        if (!allQueEmpty) {
            // 陪跑的节点不参与决策本轮是跑Prefill还是Decode
            decideScheduleInfos.push_back(*it);
        }
    }
    size_t numPrefill =
        static_cast<size_t>(std::count_if(decideScheduleInfos.begin(), decideScheduleInfos.end(), [](SchedInfo info) {
            return info.pdPriority_ == PDPriorityType::PREFILL_FIRST;
        }));
    return numPrefill >= (decideScheduleInfos.size() - numPrefill) ? PDPriorityType::PREFILL_FIRST
                                                                   : PDPriorityType::DECODE_FIRST;
}
} // namespace mindie_llm