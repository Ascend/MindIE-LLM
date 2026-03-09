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

#ifndef DATACLASS_METRIC_H
#define DATACLASS_METRIC_H

#include <string>
#include <atomic>

namespace mindie_llm {

// BlockManger提供内存资源信息
struct BlockMetric {
    uint64_t freeNpuBlockNum_{0};
    uint64_t totalNpuBlockNum_{0};
    uint64_t freeCpuBlockNum_{0};
    uint64_t totalCpuBlockNum_{0};
};

// 请求队列的快照
struct ReqStatistic {
    uint64_t remainPrefillSlots_{0};  // TBC_为了快手自己做调度，当前没有用， 裁决是否要去掉
    uint64_t remainPrefillTokens_{0}; // TBC_为了快手自己做调度，当前没有用， 裁决是否要去掉
    uint64_t remainBlocks_{0};        // TBC_为了快手自己做调度，当前没有用， 裁决是否要去掉
    uint64_t dpRemainBlocks_{0};      // TBC_待明确DP实现方案后再考虑怎么落地

    uint64_t waitingRequestNum_{0};
    uint64_t runningRequestNum_{0};
    uint64_t swappedRequestNum_{0};

    uint64_t cumulativePreemptCount_{0};
    uint64_t allRadixMatchNum_{0};    // 统计所有请求的 prefill prompt token 总数
    uint64_t npuRadixMatchHitNum_{0}; // 统计所有请求的 prefix cache 命中的 token 总数
};

struct SchedulerMetric {
    ReqStatistic reqsInfo;
    BlockMetric blockInfo;
};

struct EngineMetric {
    SchedulerMetric schedulerInfo;
    float prefillThroughput_{0.0};
    float decodeThroughput_{0.0};
};

} // namespace mindie_llm

#endif // DATACLASS_METRIC_H