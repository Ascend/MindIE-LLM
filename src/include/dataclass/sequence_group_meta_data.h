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
 
#ifndef SEQUENCE_GROUP_META_DATA_H
#define SEQUENCE_GROUP_META_DATA_H
#include "basic_types.h"
#include "sequence.h"
#include "sampling.h"

namespace mindie_llm {
struct SequenceGroupState {
    size_t numSteps_;

    size_t currentSteps_;
};

struct SequenceGroupMetaData {
    RequestId requestId_;

    std::string serverid_;

    std::shared_ptr<SamplingParams> samplingParams_;

    bool doSample_;

    size_t tokenChunkSize_;

    SequenceGroupState state_;

    std::vector<SequenceId> seqIds_;

    std::vector<size_t> promptLens_;

    std::vector<TokenId> tokenIds_; // prompt的tokenid

    std::vector<BlockIds> blockIds_; // 个数和promptLens_相关，无需额外定义block的个数

    uint64_t dpInstanceId_; // the instance ids of P Node dp

    std::vector<BlockIds> srcBlockIds_; // // the block ids in P Node

    std::vector<size_t> computedLens_;

    std::vector<size_t> remoteComputedLens_;

    std::optional<bool> skipSpecialTokens_;

    std::optional<bool> ignoreEos_;

    std::optional<std::string> loraId_;

    bool isSp_{false};

    bool isCp_{false};

    bool isMtp_{false};

    size_t spRankId_{0};

    bool isAppendBlock_{false};

    size_t appendBlockRankId_{0};

    std::vector<size_t> spRankPromptTokenNum_;

    std::vector<size_t> spRankBlockNum_;

    std::vector<size_t> prefillBlockRankId_;

    std::vector<SequenceId> reservedSeqIds_; // 并行采样时预留的seqid
    
    // ChunkedPrefill相关参数，beamsearch场景下长度>1，常规场景下长度为1
    std::vector<bool> isReqPrefill_;

    std::vector<bool> isReqLastChunk_;

    std::vector<size_t> splitStartPos_;

    std::vector<size_t> splitEndPos_;

    // 边云动态切块新增, 单位ms，给TG侧传请求到达时间间隔用于切图
    int32_t requestGap_{0};

    size_t lwdCloudSpRankId_{0};
    size_t lwdCloudAppendBlockRankId_{0};
    std::vector<size_t> lwdCloudSpRankPromptTokenNum_;
    std::vector<size_t> lwdCloudSpRankBlockNum_;
    std::vector<BlockId> lwdCloudBlockIds_;
    
    std::optional<std::string> responseFormat_; // JSON 结构化输出约束
};

struct SequenceGroupMetaDatas {
    std::vector<SequenceGroupMetaData> metaList;

    // 其中seqLens[0], 用于保存当前DP/节点batch中SeqGroup中的tokenNum信息. prefill时是prom
    std::vector<std::vector<int64_t>> seqLenList;

    int64_t maxBatchSize = 0;

    int64_t maxSeqLen = 0;
};
} // namespace mindie_llm
#endif