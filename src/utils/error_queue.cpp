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

#include "error_queue.h"

namespace mindie_llm {

ErrorQueue& ErrorQueue::GetInstance()
{
    static ErrorQueue instance;
    return instance;
}

void ErrorQueue::EnqueueErrorMessage(
    const std::string &errCode,
    const std::string &createdBy,
    const std::chrono::time_point<std::chrono::system_clock> &timestamp)
{
    ErrorItem item(errCode, createdBy, timestamp);

    if (errorList_.Size() >= maxErrorListSize) {
        ErrorItem itemToRemove;
        errorList_.PopFront(itemToRemove);
    }
    errorList_.PushBack(item);
}

bool ErrorQueue::PopError(ErrorItem &item)
{
    return errorList_.PopFront(item);
}

size_t ErrorQueue::Size() const
{
    return errorList_.Size();
}

} // namespace mindie_llm
