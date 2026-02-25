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

#ifndef MINDIE_LLM_REQUEST_ID_H
#define MINDIE_LLM_REQUEST_ID_H

#include <string>
#include <atomic>

namespace mindie_llm {
using RequestIdNew = std::string;

/// @brief 虚推请求对应的固定 SequenceId（用于 LLM 引擎层特殊处理）
constexpr int64_t SIMULATE_SEQUENCE_ID = 9223372036854774L;

inline RequestIdNew GetNextInferRequestId()
{
    static std::atomic<uint64_t> inferRequestIdGenerator{0};
    const std::string prefix{"endpoint_common_"};
    return prefix + std::to_string(inferRequestIdGenerator.fetch_add(1, std::memory_order_relaxed));
}

}
#endif
