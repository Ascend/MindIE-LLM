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

#include "post_scheduler.h"

#include <vector>
#include <stdexcept>

#include "process_group.h"
#include "thread_group_cc.h"
#include "system_log.h"

namespace mindie_llm {
void PostScheduler::SyncBatchInfo(BatchInfo &batchInfo, size_t dpRank, bool enableDistributed)
{
    if (enableDistributed) {
        SyncBatchInfoAcrossNodes(batchInfo);
    } else {
        SyncBatchInfoAcrossDP(batchInfo, dpRank);
    }
}

void PostScheduler::SyncSeqLenList(std::vector<std::vector<int64_t>> &tokenNumList, std::vector<int64_t> &batchSizeList,
                                   size_t paddingSize, size_t dpRank, bool enableDistributed)
{
    // 1. add padding data
    AddPaddingData(tokenNumList, paddingSize);

    // 2. syncronize tokenNum
    if (enableDistributed) {
        SyncSeqLenListAcrossNodes(tokenNumList);
    } else {
        SyncSeqLenListAcrossDP(tokenNumList, dpRank);
    }

    // 3. remove padding data
    RemovePaddingData(tokenNumList, batchSizeList);
}

void PostScheduler::AllGatherBatchesAcrossDPs(std::vector<std::vector<SequenceGroupMetaDatas>> &allDpMetas,
    std::vector<std::vector<SchedulerOutputs>> &allDpOutputs, size_t dpRank)
{
    if (allDpMetas.empty() || allDpOutputs.empty()) {
        throw std::invalid_argument("Input containers cannot be empty. "
            "allDpMetas size: " + std::to_string(allDpMetas.size()) +
            ", allDpOutputs size: " + std::to_string(allDpOutputs.size()));
    }
    std::vector<SequenceGroupMetaDatas> sendMetaData = allDpMetas[0];
    ThreadGroupCC::GetInstance().AllGather(sendMetaData, allDpMetas, dpRank);

    std::vector<SchedulerOutputs> sendOutput = allDpOutputs[0];
    ThreadGroupCC::GetInstance().AllGather(sendOutput, allDpOutputs, dpRank);
}

std::unordered_set<SequenceId> PostScheduler::AllGatherCleanSeqIdsAcrossDPs(
    std::unordered_set<SequenceId> &curCleanSeqIdSet, size_t dpRank)
{
    std::vector<std::vector<std::unordered_set<SequenceId>>> allRankSeqIds;
    std::vector<std::unordered_set<SequenceId>> curCleanSeqIds;
    curCleanSeqIds.push_back(curCleanSeqIdSet);
    allRankSeqIds.push_back(curCleanSeqIds);
    ThreadGroupCC::GetInstance().AllGather(curCleanSeqIds, allRankSeqIds, dpRank);

    std::unordered_set<SequenceId> allDPSeqIdSet;
    for (auto dpRankSeqIds : allRankSeqIds) {
        allDPSeqIdSet.insert(dpRankSeqIds[0].begin(), dpRankSeqIds[0].end());
    }

    return allDPSeqIdSet;
}

void PostScheduler::AddPaddingData(std::vector<std::vector<int64_t>> &tokenNumList, size_t paddingSize)
{
    paddingSize = std::max(static_cast<size_t>(1), paddingSize); // 确保至少同步一个元素
    tokenNumList.resize(1);                                      // 避免tokenNumList没有元素导致的coredump
    tokenNumList[0].resize(paddingSize, -1);
}

void PostScheduler::RemovePaddingData(std::vector<std::vector<int64_t>> &tokenNumList,
                                      std::vector<int64_t> batchSizeList)
{
    if (tokenNumList.size() != batchSizeList.size()) {
        throw std::runtime_error("the size of tokenNumList and batchSizeList is mismatched.");
    }

    for (size_t i = 0; i < tokenNumList.size(); ++i) {
        if (batchSizeList[i] >= 0) {
            tokenNumList[i].resize(batchSizeList[i]);
        }
    }
}

void PostScheduler::SyncBatchInfoAcrossDP(BatchInfo &batchInfo, size_t dpRank)
{
    // 1. 集合通信获取各节点的batchInfo
    std::vector<std::vector<int64_t>> recvData;
    std::vector<int64_t> sendData = {batchInfo.maxBatchSize_, batchInfo.maxSeqLen_};
    ThreadGroupCC::GetInstance().AllGather(sendData, recvData, dpRank);

    // 2. 获取maxBatchSize, maxSeqLen和tokenNum
    for (const auto &item : recvData) {
        batchInfo.batchSizeList_.emplace_back(item[0]);
        batchInfo.maxBatchSize_ = std::max(batchInfo.maxBatchSize_, item[0]);
        batchInfo.maxSeqLen_ = std::max(batchInfo.maxSeqLen_, item[1]);
    }
}

void PostScheduler::SyncBatchInfoAcrossNodes(BatchInfo &batchInfo)
{
    // 1. 集合通信获取各节点的batchInfo
    std::vector<torch::Tensor> inputs;
    inputs.emplace_back(
        torch::tensor({batchInfo.maxBatchSize_, batchInfo.maxSeqLen_}, torch::dtype(torch::kInt64).device(c10::kCPU)));
    try {
        std::vector<std::vector<torch::Tensor>> outputs = ProcessGroup::GetInstance().AllGather(inputs);
        if (outputs.empty() || outputs[0].empty()) {
            return;
        }

        // 2. 获取maxBatchSize, maxSeqLen和tokenNum
        for (const torch::Tensor &tensor : outputs[0]) {
            batchInfo.batchSizeList_.emplace_back(tensor[0].item<int64_t>());
            batchInfo.maxBatchSize_ = std::max(batchInfo.maxBatchSize_, tensor[0].item<int64_t>());
            batchInfo.maxSeqLen_ = std::max(batchInfo.maxSeqLen_, tensor[1].item<int64_t>());
        }
    } catch (const std::exception& e) {
        LOG_ERROR_LLM << "SyncBatchInfoAcrossNodes failed: outputs is invalid.";
    }
}

void PostScheduler::SyncSeqLenListAcrossDP(std::vector<std::vector<int64_t>> &tokenNumList, size_t dpRank)
{
    if (tokenNumList.empty()) {
        throw std::invalid_argument("tokenNumList cannot be empty.");
    }
    std::vector<int64_t> sendData = tokenNumList[0];
    ThreadGroupCC::GetInstance().AllGather(sendData, tokenNumList, dpRank);
}

void PostScheduler::SyncSeqLenListAcrossNodes(std::vector<std::vector<int64_t>> &tokenNumList)
{
    if (tokenNumList.size() == 0) {
        LOG_ERROR_LLM << "SyncSeqLenList failed: tokenNumList is null";
        return;
    }
    // 1. syncronize token num
    std::vector<torch::Tensor> inputs;
    inputs.emplace_back(torch::tensor(tokenNumList[0], torch::dtype(torch::kInt64).device(c10::kCPU)));
    try {
        std::vector<std::vector<torch::Tensor>> outputs = ProcessGroup::GetInstance().AllGather(inputs);
        if (outputs.empty() || outputs[0].empty()) {
            return;
        }

        // 2. reset tokenNumList to receive data
        tokenNumList = std::vector<std::vector<int64_t>>();
        for (const torch::Tensor &tensor : outputs[0]) {
            int64_t *pTextNum = tensor.data_ptr<int64_t>();
            tokenNumList.emplace_back(std::vector<int64_t>(pTextNum, pTextNum + tensor.size(0)));
        }
    } catch (const std::exception& e) {
        LOG_ERROR_LLM << "SyncSeqLenListAcrossNodes failed: outputs is invalid.";
    }
}
} // namespace mindie_llm
