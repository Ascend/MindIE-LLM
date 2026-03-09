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

#include "hit_rate_calculator.h"

#include <iostream>

#include "system_log.h"

namespace mindie_llm {
/*
    Record the cache hit/miss when a cacheable block is allocated
    This should be called every time the allocator allocates an immutable block
    Only used for PrefixCachingAllocator
*/
void HitRateCalculator::Record(bool hit)
{
    static constexpr uint64_t maxHitNum = 62;
    static constexpr uint64_t limit = (1ULL << maxHitNum);
    if (hit) {
        hitNum_++;
    } else {
        missNum_++;
    }
    if (hitNum_ >= limit || missNum_ >= limit) {
        uint64_t hitExceed  = (hitNum_  >= limit) ? (hitNum_  - limit) : 0;
        uint64_t missExceed = (missNum_ >= limit) ? (missNum_ - limit) : 0;
        LOG_WARN_LLM << "HitRateCalculator may overflow! " << "hitNum=" << hitNum_ << ", missNum=" << missNum_
                     << ", limit=" << limit << ", hitExceed=" << hitExceed << ", missExceed=" << missExceed;
    }
}

// Calculate cache hit rate
double HitRateCalculator::GetHitRate() const
{
    uint64_t totalNum = hitNum_ + missNum_;
    if (totalNum == 0) {
        return 0;
    }
    return static_cast<double>(hitNum_) / static_cast<double>(totalNum);
}
} // namespace mindie_llm