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
 
#include "dummy_quota_manager.h"
#include "log.h"

using namespace mindie_llm;
using namespace std;
using namespace boost::concurrent;

DummyQuotaManager::DummyQuotaManager(int rank, int quota) : initQuota_(quota), rank_(rank)
{
    quotaLeft_.store(quota);
    acrossProcSyncThread_ = thread([this]() { this->AcrossProcSyncThreadEntry_(); });
    MINDIE_LLM_LOG_INFO("DummyQuotaManager init successfully. rank:" << rank_ << ", initQuota:" << initQuota_);
}

DummyQuotaManager::~DummyQuotaManager()
{
    // cannot guarantee every process exit simultaneously, so cancel thread directly
    threadNeedStop_.store(true);
    if (acrossProcSyncThread_.joinable()) {
        acrossProcSyncThread_.join();
    }
    MINDIE_LLM_LOG_INFO("DummyQuotaManager sync thread stopped.");
}

bool DummyQuotaManager::AcquireQuota(bool isDummy)
{
    size_t reqWaitLoop = 0;
    if (threadNeedStop_.load()) { // downgraded
        return true;
    }
    if (!isDummy) {
        while (quotaLeft_.load() == 0 && !threadNeedStop_.load()) {
            Wakeup();
            this_thread::sleep_for(chrono::milliseconds(1)); // wait until wake up finish
            if (reqWaitLoop++ % REQ_WAIT_LOG_LOOP == 0) {
                MINDIE_LLM_LOG_WARN(
                    "no quota available. wait wakup to restore quota. If you keep seeing this, there might "
                    "be bug in DummyTokenManager. reqWaitLoop:"
                    << reqWaitLoop << "; dummy queue size:" << dummyBatchQueue_.size());
            }
        }
        dummyBatchQueue_.push(isDummy);
        quotaLeft_.fetch_sub(1, std::memory_order_seq_cst);
        return true;
    }
    bool succ = false;
    if (quotaLeft_.load() != 0) {
        dummyBatchQueue_.push(isDummy);
        quotaLeft_.fetch_sub(1, std::memory_order_seq_cst);
        succ = true;
    }
    return succ;
}

// Wakeup must be called before AcquireQuota, otherwise no quota will be available to dispatch dummy batch
void DummyQuotaManager::Wakeup()
{
    bool isReqComming = true;
    if (reqCommingQueue_.empty()) {
        reqCommingQueue_.push(isReqComming);
    }
}

void DummyQuotaManager::AllGather_(bool iAmDummy, bool &allDummy, int &maxQuota, bool dummyBatchSync)
{
    vector<torch::Tensor> batchInfo;
    batchInfo.emplace_back(torch::tensor({static_cast<int>(iAmDummy), quotaLeft_.load(),
                                          static_cast<int>(dummyBatchSync), static_cast<int>(dummyBatchQueue_.size())},
                                         torch::dtype(torch::kInt32).device(c10::kCPU)));
    vector<vector<torch::Tensor>> batchInfos;
    size_t cost = 0;
    static size_t allCost = 0;
    static size_t count = 0;
    try {
        auto start = high_resolution_clock::now();
        batchInfos = ProcessGroup::GetInstance().AllGather(batchInfo);
        auto end = high_resolution_clock::now();
        cost = static_cast<size_t>(duration_cast<milliseconds>(end - start).count());
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_WARN("allgather standard exception caught: " << e.what() << ". downgrade to dummy-always.");
        quotaLeft_.store(initQuota_);
        threadNeedStop_.store(true);
        return;
    } catch (...) {
        MINDIE_LLM_LOG_WARN("allgather got unknown exception, DummyTokenManager cannot work, downgrade "
                            "to dummy-always.");
        quotaLeft_.store(initQuota_);
        threadNeedStop_.store(true);
        return;
    }
    int preGatherType = 0;
    int cnt = 0;
    int maxDummyBatchQueueSize = 0;
    for (const auto &tensor : batchInfos[0]) {
        bool isDummy = tensor[0].item<bool>();
        allDummy = allDummy && isDummy;
        maxDummyBatchQueueSize = max(maxDummyBatchQueueSize, tensor[3].item<int>());
        maxQuota = max(maxQuota, tensor[1].item<int>());
        if ((cnt++ > 0 && preGatherType != tensor[2].item<int>()) ||
            maxDummyBatchQueueSize > MAX_ALLOWED_DUMMY_QUEUE_PENDINGS) {
            MINDIE_LLM_LOG_WARN("All-gather message type is not same or dummy batch too many. rank"
                                 << rank_ << ", is dummyBatchSync:" << tensor[2].item<int>() << ", err rank:" << cnt--
                                 << ", maxDummyBatchQueueSize:" << maxDummyBatchQueueSize);
            quotaLeft_.store(initQuota_);
            threadNeedStop_.store(true);
            return;
        }
        preGatherType = tensor[2].item<int>();
    }
    allCost += cost;
    count++;
    if (cost > 200) {
        MINDIE_LLM_LOG_INFO_REQUEST("dummy all gather cost too long, dp:"
                            << rank_ << ", time:" << cost << ", quotaLeft_:" << quotaLeft_.load() << ", iAmDummy:"
                            << iAmDummy << ", dummyQueue size:" << dummyBatchQueue_.size() << ", allDummy:" << allDummy
                            << ", maxQuota:" << maxQuota << ", average cost:" << (allCost / count));
    }
}
void DummyQuotaManager::QuotaAllGather_(bool &iAmDummy, bool &allDummy, int &maxQuota)
{
    iAmDummy = dummyBatchQueue_.pull();
    AllGather_(iAmDummy, allDummy, maxQuota, true);
}

void DummyQuotaManager::WakeupAllGather_(bool &iAmDummy, bool &allDummy, int &maxQuota)
{
    iAmDummy = reqCommingQueue_.empty();
    if (!iAmDummy) { // empty the queue
        bool val = false;
        while (reqCommingQueue_.try_pull(val) == queue_op_status::success) {
        }
    }
    AllGather_(iAmDummy, allDummy, maxQuota, false);
}

void DummyQuotaManager::AcrossProcSyncThreadEntry_()
{
    while (!threadNeedStop_.load()) {
        // when dummyBatchQueue_ has item, or will have item, we must do dummy sync
        // when quota is 0, and a real request comes in, AcquireQuota will not push request into dummyBatchQueue_
        // only after wakeup finished, real request will be pushed into dummyBatchQueue_
        // quotaLeft + (dummy number in dummyBatchQueue_) across all processes are always == 0 or > 0 at this point.
        // quotaLeft + (dummy number in dummyBatchQueue_) means unmatched (match is done in QuotaAllGather_) dummies.
        // When there is unmatches dummies, do QuotaAllGather_, otherwise WakeupAllGather_.
        // when quotaLeft_0 == 0, dummyBatchQueue_ can only be [Dummy*, Req, Dummy+ ], cannot be [Dummy*, Req]
        // so QuotaAllGather_ and WakeupAllGather_ will not messed up across all processes.
        bool needDummySync = (quotaLeft_.load() == 0 && dummyBatchQueue_.size() != 0) || quotaLeft_.load() != 0;
        if (needDummySync) {
            bool allDummy = true;
            bool iAmDummy = true;
            int maxQuota = 0;
            QuotaAllGather_(iAmDummy, allDummy, maxQuota);
            int quotaToFill = max(0, initQuota_ - maxQuota);
            if (!allDummy) { // restore to defaultQuota quota
                quotaLeft_.fetch_add(quotaToFill, memory_order_seq_cst);
            }
        } else { // wakeup (restore default quota) if there is request
            bool allDummy = true;
            bool iAmDummy = true;
            int maxQuota = 0;
            WakeupAllGather_(iAmDummy, allDummy, maxQuota);
            int quotaToFill = max(0, initQuota_ - maxQuota);
            if (!allDummy) { // restore to defaultQuota quota when request comming
                quotaLeft_.fetch_add(quotaToFill, memory_order_seq_cst);
            }
        }
    }
    MINDIE_LLM_LOG_INFO("DummyQuotaManager across process synchronization thread exits.");
}

// replace sleep and poll when fully tested
void DummyQuotaManager::WaitForQuota_()
{
    unique_lock<mutex> lock(notifyMutex_);
    notifyCv_.wait(lock, [this] { return quotaLeft_.load() != 0 || threadNeedStop_.load(); });
}

void DummyQuotaManager::NotifyQuotaAvailable_() { notifyCv_.notify_one(); }