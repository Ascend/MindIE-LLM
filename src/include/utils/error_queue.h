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

#ifndef MINDIE_LLM_ERROR_QUEUE_H
#define MINDIE_LLM_ERROR_QUEUE_H

#include <string>
#include <chrono>
#include <set>
#include "concurrent_deque.h"

namespace mindie_llm {

struct ErrorItem {
    int64_t timestamp;
    std::string errCode;
    std::string createdBy;
    ErrorItem()
        : timestamp(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count()),
          errCode(""), createdBy("") {}
    ErrorItem(const std::string &errCode, const std::string &createdBy,
        const std::chrono::time_point<std::chrono::system_clock> &timestamp)
        : timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count()),
          errCode(errCode), createdBy(createdBy) {}
};

class ErrorQueue {
public:
    static ErrorQueue& GetInstance();

    void EnqueueErrorMessage(
        const std::string &errCode,
        const std::string &createdBy,
        const std::chrono::time_point<std::chrono::system_clock> &timestamp = std::chrono::system_clock::now());

    bool PopError(ErrorItem &item);

    size_t Size() const;

    ErrorQueue(const ErrorQueue &) = delete;
    ErrorQueue &operator=(const ErrorQueue &) = delete;
    ErrorQueue(ErrorQueue &&) = delete;
    ErrorQueue &operator=(ErrorQueue &&) = delete;

private:
    ErrorQueue() = default;
    mutable ConcurrentDeque<ErrorItem> errorList_;
    static constexpr int maxErrorListSize = 100;
};

} // namespace mindie_llm

#endif // MINDIE_LLM_ERROR_QUEUE_H
