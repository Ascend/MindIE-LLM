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

#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <cstdint>
#include <memory>
#include <vector>

#include "basic_types.h"

namespace mindie_llm {

enum class SequenceStatus {
    ALL_STATUS = 0,

    WAITING = 1,

    RUNNING,

    SWAPPED,

    FINISH_STOPPED,

    FINISH_LENGTH_CAPPED,

    FINISH_ABORTED,

    FINISH_IGNORED,

    FINISH_RECOMPUTE // PD分离场景做recompute，当前engine请求完成，由coordinate重新下发调度
};

namespace sequence_status {
bool IsFinish(const SequenceStatus status);
}

enum class SequenceStage {
    PREFILL,

    DECODE,
};

struct SequenceData {
    std::vector<TokenId> promptTokenIds;

    std::vector<TokenId> outputTokenIds;

    SequenceStage stage_ = SequenceStage::PREFILL;

    // 边云协同新增变量
    SequenceStage layerwiseStage_ = SequenceStage::PREFILL;
    bool layerwiseRecompute_ = false;
    bool layerwiseRecomputeReturn_ = false;
    bool layerwiseRunning_ = false;
    bool layerwiseDiscard_ = false;

    size_t numComputedTokens_ = 0;

    static SequenceData FromSequence(const std::vector<TokenId> &tPromptTokenIds);

    SequenceData() = default;

    explicit SequenceData(const std::vector<TokenId> &tPromptTokenIds);

    void ResetStateForRecompute();

    /**
     * 统计prefill阶段已经计算的token数
     */
    size_t GetNumComputedTokens() const;

    /**
     * 注意，如果是重新计算，则在prefill阶段会将前面已经得到output token
     * id一起作为prefill的输入
     */
    size_t GetNumUncomputedTokens();

    /**
     * 累计已经计算的token数
     */
    void UpdateNumComputedTokens(size_t numNewComputedTokens);

    void SetLayerwiseStage(bool isPrefill);

    [[nodiscard]] size_t GetLength();
};

struct Sequence {
    SequenceId seqId_{};

    int blockSize_{};

    TokenId eosTokenId_{};

    SequenceData data_{};

    SequenceStatus status_{SequenceStatus::WAITING};

    HashValue hashValue_{INVALID_HASH_VALUE};

    Sequence(SequenceId seqId, int blockSize);

    Sequence(SequenceId seqId, int blockSize, const std::vector<TokenId> &inputs);

    [[nodiscard]] size_t GetLen();

    [[nodiscard]] size_t GetOutputLen(bool containsPlaceholder = false);

    [[nodiscard]] bool IsPrefill() const;

    [[nodiscard]] bool IsLayerwisePrefill() const;

    [[nodiscard]] bool IsFinished() const;

    size_t GetNumComputedTokens() const;

    size_t GetNumUncomputedTokens();

    void ResetStateForCompute();

    const std::vector<TokenId> GetTokenIds() const;

    HashValue GetExtraHash() const;

    void SetExtraHash(HashValue hashValue);
};

using SequencePtr = std::unique_ptr<Sequence>;
using SequenceSPtr = std::shared_ptr<Sequence>;

struct SequenceOutput {
    SequenceId parentSeqId;

    TokenId outputId;
};
using SequenceOutputSPtr = std::shared_ptr<SequenceOutput>;

struct SequenceGroupoutput {
    std::vector<SequenceOutput> outputs;
};

} // namespace mindie_llm

#endif
