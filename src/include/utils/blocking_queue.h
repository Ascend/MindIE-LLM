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
 
#ifndef BLOCKING_QUEUE_H
#define BLOCKING_QUEUE_H

#include <semaphore.h>
#include <mutex>
#include <shared_mutex>
#include <queue>

namespace mindie_llm {
template <typename T> class BlockingQueue {
public:
    BlockingQueue()
    {
        sem_init(&sem_, 0, 0);
    }

    ~BlockingQueue() = default;

    int Push(T item)
    {
        std::unique_lock<std::shared_mutex> lock(queueMutex_);
        queue_.push(std::move(item));
        int result = sem_post(&sem_);
        if (result != 0) {
            ULOG_ERROR("blocking_queue",
                GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                "sem_post failed when push a item");
            throw std::runtime_error("sem_post failed");
        }
        return result;
    }

    T Take()
    {
        if (sem_wait(&sem_) != 0) {
            ULOG_ERROR("blocking_queue",
                GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, CHECK_ERROR),
                "sem_wait failed when pop a item");
            throw std::runtime_error("sem_wait failed");
        }
        std::unique_lock<std::shared_mutex> lock(queueMutex_);
        T value(std::move(queue_.front()));
        queue_.pop();
        return value;
    }

    size_t Size()
    {
        std::shared_lock<std::shared_mutex> lock(queueMutex_);
        return queue_.size();
    }

    bool Empty()
    {
        std::shared_lock<std::shared_mutex> lock(queueMutex_);
        return queue_.empty();
    }

private:
    sem_t sem_ {};

    std::queue<T> queue_;

    std::shared_mutex queueMutex_;
};
} // namespace mindie_llm
#endif