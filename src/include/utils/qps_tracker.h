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

#ifndef QPS_TRACKER_H
#define QPS_TRACKER_H
#include <atomic>
#include <cstddef>
#include <vector>

namespace mindie_llm {
struct QCntBucket {
    long bucketId{0};
    std::atomic<int> counter;
};

class QPSTracker {
   public:
    QPSTracker(int timeWindowInterval, int bucketTimeInterval);
    float GetQPS();
    void Record();

   private:
    int64_t bucketTimeInterval_;  // ms
    int64_t timeWindowInterval_;  // ms
    size_t bucketsNum_;
    std::vector<QCntBucket> buckets_;
};
}  // namespace mindie_llm
#endif
