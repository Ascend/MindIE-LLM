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
#include "load_balancer.h"

#include "system_log.h"

namespace mindie_llm {
LoadBalancer::LoadBalancer(const std::vector<EnginePerDPSPtr> &enginePerDPs, size_t waveNumPerDP, size_t thresholdPerDP,
                           size_t intervalMs)
    : enginePerDPs_(enginePerDPs), dpSize_(enginePerDPs.size()), waveNumPerDP_(waveNumPerDP),
      thresholdPerDP_(thresholdPerDP), waveId_(0), intervalMs_(intervalMs), timer_(io_)
{
    balancePolicyPtr_ = MakeBalancePolicy(BalancePolicyType::ROUND_ROBIN);
    if (balancePolicyPtr_ == nullptr) {
        throw std::runtime_error("LoadBalancer::LoadBalancer: Failed to create BalancePolicy");
    }
    lastIntervalTime_ = intervalMs_;
    timer_.expires_after(std::chrono::milliseconds(intervalMs_));
    timer_.async_wait(boost::bind(&LoadBalancer::ExecutePeriodicTask, this, boost::asio::placeholders::error));
    worker_ = std::thread([this]() { io_.run(); });
    pthread_setname_np(worker_.native_handle(), "LoadBalancer");
}

LoadBalancer::~LoadBalancer()
{
    Stop();
    if (worker_.joinable()) {
        worker_.join();
    }
}

// 添加sequence group
void LoadBalancer::AddSeqGroup(SequenceGroupSPtr &seqGroup)
{
    if (seqGroup == nullptr) {
        LOG_ERROR_LLM << "Attempt to add a null SequenceGroup.";
        return;
    }

    if (stop_) {
        LOG_INFO_LLM << "LoadBalancer stopped.";
        return;
    }

    seqGroups_.PushBack(seqGroup);

    // 如果当前的seqGroups_数量超过阈值，则触发分配
    if (seqGroups_.Size() >= waveNumPerDP_) {
        boost::asio::post(io_, boost::bind(&LoadBalancer::TriggerImmediatelyTask, this));
    }
}

void LoadBalancer::TriggerImmediatelyTask()
{
    timer_.cancel();
    Distribute();

    // 重新启动定时器
    timer_.expires_after(std::chrono::milliseconds(intervalMs_));
    timer_.async_wait(boost::bind(&LoadBalancer::ExecutePeriodicTask, this, boost::asio::placeholders::error));
}

void LoadBalancer::ExecutePeriodicTask(const boost::system::error_code &ec)
{
    if (ec == boost::asio::error::operation_aborted || (stop_ && seqGroups_.Empty())) {
        // 定时器被外部取消，此时直接返回，触发方调用async_wait重新启动
        return;
    }

    static int times = 0;
    if ((times++ % 3000) == 0) { // 由于周期性调度，降低刷新频率到1/3000
        LOG_DEBUG_LLM << "ExecutePeriodicTask.";
    }
    Distribute();

    // 重新启动定时器
    timer_.expires_after(std::chrono::milliseconds(intervalMs_));
    timer_.async_wait(boost::bind(&LoadBalancer::ExecutePeriodicTask, this, boost::asio::placeholders::error));
}

void LoadBalancer::Stop()
{
    stop_ = true;
    io_.stop();
}

// 分配SequenceGroup到不同的Engine
void LoadBalancer::Distribute()
{
    if (seqGroups_.Empty() && candidates_.empty()) {
        return;
    }

    // 1. 收集当前当前Scheduler中的wave信息
    std::vector<size_t> dpRankIds = GetDistributedDpRankIds();
    if (dpRankIds.empty()) {
        LOG_WARN_LLM << "There is not scheduler need to distribute sequence group.";
        return;
    }

    // 2. 准备数据
    PrepCandidates(dpRankIds.size(), waveNumPerDP_);

    // 3. 分配SequenceGroup到不同的调度器
    DistributeSeqGroups(dpRankIds);
}

std::vector<size_t> LoadBalancer::GetDistributedDpRankIds()
{
    std::vector<size_t> dpRankIds;
    uint64_t minFreeBlockNum = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < enginePerDPs_.size(); ++i) {
        SchedulerMetric metric = enginePerDPs_[i]->scheduler->CollectSchedulerMetric();
        dpRankIds.push_back(i);
        uint64_t freeBlockNum = metric.blockInfo.freeNpuBlockNum_;
        minFreeBlockNum = std::min(minFreeBlockNum, freeBlockNum);
    }
    // set limit
    BalancerConstraintParam param(waveNumPerDP_, dpRankIds.size(), minFreeBlockNum);
    balancePolicyPtr_->SetConstraint(param);
    return dpRankIds;
}

// 准备下发的数据
void LoadBalancer::PrepCandidates(size_t numDp, size_t maxNumPerDp)
{
    SequenceGroupSPtr seqGroup;
    while (seqGroups_.PopFront(seqGroup)) { // TBC_此处逻辑有点问题，应该先判断，然后放入才对。
        candidates_.push_back(seqGroup);
        if (candidates_.size() >= maxNumPerDp * numDp) {
            // 达到每个DP的最大数量，停止收集
            break;
        }
    }
}

// 可以根据seqLen的长度均分，当前使用RoundRobin的方式
void LoadBalancer::DistributeSeqGroups(const std::vector<size_t> &dpRankIds)
{
    if (this->candidates_.empty()) {
        return;
    }

    // 1. 确定分配的waveId
    waveId_++;

    // 2. 按照policy的方式分配
    std::vector<std::vector<SequenceGroupSPtr>> candidatesForScheds = balancePolicyPtr_->Apply(this->candidates_);
    for (size_t i = 0; i < candidatesForScheds.size(); i++) {
        auto candidatesForSched = candidatesForScheds[i];
        auto &enginePerDP = enginePerDPs_[dpRankIds[i]];
        for (auto seqGroupSPtr : candidatesForSched) {
            if (seqGroupSPtr == nullptr) {
                LOG_ERROR_LLM << "null seqGroup in candidates";
                continue;
            }
            seqGroupSPtr->waveId_ = waveId_;
            enginePerDP->scheduler->AddSeqGroup(seqGroupSPtr);
            LOG_INFO_LLM.SetType(LogType::REQUEST) << "Load Balance dispatch request to DP RankId: "
                                << dpRankIds[i] << ", requestId: " << seqGroupSPtr->metrics_.inferReqId_
                                << "seqId: " << seqGroupSPtr->firstSeq->seqId_ << ").";
            enginePerDP->addedRequestNum++;
        }
    }
}

LoadBalancerPtr MakeLoadBalancer(const std::vector<std::shared_ptr<EnginePerDP>> &enginePerDPs, size_t waveNumPerDP,
                                 size_t thresholdPerDP, size_t intervalMs)
{
    return std::make_unique<LoadBalancer>(enginePerDPs, waveNumPerDP, thresholdPerDP, intervalMs);
}
} // namespace mindie_llm