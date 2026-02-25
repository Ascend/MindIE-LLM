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

#ifndef ISEQ_GROUP_COLLECTION_H
#define ISEQ_GROUP_COLLECTION_H

#include <deque>

#include "sequence_group.h"

namespace mindie_llm {
enum class OrderType {
    FCFS,
    PRIORITY,
};

enum class PDPriorityType : uint8_t {
    /* 优先调度prefill的请求，且只会做prefill */
    PREFILL_FIRST,

    /* 优先调度decode的请求，且只会做decode */
    DECODE_FIRST,

    /* 由于role=PnD时，chunked_prefill=true时，同时做prefill和decode，即使chunked_prefill时decode优先 */
    MIX,
};

enum class LwdPDelayType : uint8_t {
    /* 本轮延迟调度Prefill请求，方式为改调度Prefill请求为调度Decode请求 */
    PREFILL_TO_DECODE,
    /* 本轮延迟调度Prefill请求，方式为本轮跳过prefill请求，不下发batch */
    PREFILL_SKIP,
    /* 本轮不延迟调度prefill请求 */
    PREFILL_KEEP,
    /* 默认类型无意义 */
    INVALID,
};

class SeqGroupCollection {
public:
    SeqGroupCollection() = default;

    explicit SeqGroupCollection(PDPriorityType pdPriorityType);

    virtual ~SeqGroupCollection() = default;

    // waiting queue
    std::deque<SequenceGroupSPtr> waiting_;

    // running queue
    std::deque<SequenceGroupSPtr> running_;

    // swapped queue
    std::deque<SequenceGroupSPtr> swapped_;

    PDPriorityType pdPriorityType_{PDPriorityType::PREFILL_FIRST};
};

using ISeqGroupCollectionSPtr = std::shared_ptr<SeqGroupCollection>;
} // namespace mindie_llm

#endif