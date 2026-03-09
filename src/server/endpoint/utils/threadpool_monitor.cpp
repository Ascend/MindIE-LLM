/*
 * Copyright (c) 2024 Yuji Hirose. All rights reserved.
 * MIT License.
 *
 * Implement threadpool based on httplib::TaskQueue from cpp-httplib.
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "threadpool_monitor.h"
#include <cstring>

#include "http_rest_resource.h"
#include "log.h"
#include "logger_def.h"

namespace mindie_llm {
const int connectionCheckInterval = 200;

ThreadPoolMonitor::ThreadPoolMonitor(size_t n, size_t mqr) : shutdown_(false), max_queued_requests_(mqr)
{
    while (n > 0) {
        threads_.emplace_back(Worker(*this));
        n--;
    }
    stop_monitor_flag.store(false);
    monitor_thread = std::thread(&ThreadPoolMonitor::Monitor, this);
}

ThreadPoolMonitor::~ThreadPoolMonitor()
{
    if (monitor_thread.joinable()) {
        stop_monitor_flag.store(true);
        monitor_thread.join();
    }
}

bool ThreadPoolMonitor::enqueue(std::function<void()> fn)
{
    bool enqueSuc = false;
    size_t num = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        num = jobs_.size();
        if (max_queued_requests_ > 0 && jobs_.size() >= max_queued_requests_) {
            enqueSuc = false;
        } else {
            jobs_.push_back(std::move(fn));
            enqueSuc = true;
        }
    }

    if (enqueSuc) {
        cond_.notify_one();
        return true;
    } else {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
            GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, SUBPROCESS_ERROR),
            "[ThreadPoolMonitor] enqueue fail: wait num is " << num << ", max is " << max_queued_requests_);
        return false;
    }
}

void ThreadPoolMonitor::shutdown()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cond_.notify_all();
    for (auto &t : threads_) {
        t.join();
    }
}

void ThreadPoolMonitor::CheckAndRemoveClosedConnections()
{
    std::unique_lock<std::shared_mutex> lock(monitor_map_mutex_);
    for (auto it = monitorRequests_.begin(); it != monitorRequests_.end();) {
        auto req = it->first;
        auto reqContext = it->second;
        bool isFinished = false;
        // This lock ensures that checking the response status and
        // the lambda function is_connection_closed is performed atomically.
        // Without synchronization, another thread might delete the request object,
        // leading to data racing and other undefined behaviors.
        auto checkLock = reqContext->LockAndCheckResponseFinished(isFinished);
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[ThreadPoolMonitor] Acquiring Lock for RequestContext: " << reqContext);
        if (isFinished) {
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                       "[ThreadPoolMonitor] StreamRequest Already Finished" << reqContext->GetHTTPRequestUUID());
            it = monitorRequests_.erase(it);
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[ThreadPoolMonitor] Release Lock for RequestContext: " << reqContext);
            continue;
        }
        bool connection_closed = false;
        try {
            connection_closed = reqContext->IsConnectionClosed();
        } catch (const std::exception &e) {
            // 捕获所有 std::exception 派生的异常，包括 std::bad_function_call
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST, SUBPROCESS_ERROR),
                       "[ThreadPoolMonitor] Catch Exception: " << e.what());
            connection_closed = true;
        }
        if (connection_closed) {
            reqContext->StopInferRequest();
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "[ThreadPoolMonitor] Check and Remove Closed Connection Request UUID "
                                                  << reqContext->GetHTTPRequestUUID());
            it = monitorRequests_.erase(it);
        } else {
            ++it;
        }
        ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[ThreadPoolMonitor] Release Lock for RequestContext: " << reqContext);
    }
}

std::function<void()> ThreadPoolMonitor::Worker::GetNextJob()
{
    std::unique_lock<std::mutex> lock(pool_.mutex_);
    pool_.cond_.wait(lock, [this] { return !pool_.jobs_.empty() || pool_.shutdown_; });

    if (pool_.shutdown_ && pool_.jobs_.empty()) {
        return nullptr; // No more jobs and shutdown flag set
    }

    // 获取并移除队列中的第一个任务
    auto job = std::move(pool_.jobs_.front());
    pool_.jobs_.pop_front();

    return job;
}

void ThreadPoolMonitor::Worker::operator()()
{
    while (true) {
        std::function<void()> task = GetNextJob();
        if (!task) {
            break;
        }
        task();
    }

#if defined(CPPHTTPLIB_OPENSSL_SUPPORT) && !defined(OPENSSL_IS_BORINGSSL) && !defined(LIBRESSL_VERSION_NUMBER)
    OPENSSL_thread_stop();
#endif
}

void ThreadPoolMonitor::AddRequestToMonitor(std::shared_ptr<RequestContext> requestContext)
{
    std::unique_lock<std::shared_mutex> lock(monitor_map_mutex_);
    monitorRequests_.insert({requestContext->GetHTTPRequestUUID(), requestContext});
}

void ThreadPoolMonitor::RemoveMonitorRequest(std::string reqid)
{
    std::unique_lock<std::shared_mutex> lock(monitor_map_mutex_);
    monitorRequests_.erase(reqid);
}

void ThreadPoolMonitor::Monitor()
{
    pthread_setname_np(pthread_self(), "Monitor");
    while (!stop_monitor_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(connectionCheckInterval));
        CheckAndRemoveClosedConnections();
    }
}
} // namespace mindie_llm