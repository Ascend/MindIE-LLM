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

#include "sequence.h"

#include <algorithm>

namespace mindie_llm {
bool sequence_status::IsFinish(const SequenceStatus status) { return status > SequenceStatus::SWAPPED; }

SequenceData SequenceData::FromSequence(const std::vector<TokenId> &tPromptTokenIds)
{
    return SequenceData(tPromptTokenIds);
}

SequenceData::SequenceData(const std::vector<TokenId> &tPromptTokenIds) : promptTokenIds(tPromptTokenIds) {}

void SequenceData::ResetStateForRecompute()
{
    numComputedTokens_ = 0;
    stage_ = SequenceStage::PREFILL;
    layerwiseRecompute_ = true; // 边云特性特有状态位
    // recompute 删除output里面的占位符PLACEHOLDER_TOKEN
    auto it = std::find_if(outputTokenIds.rbegin(), outputTokenIds.rend(),
        [](int val) { return val != PLACEHOLDER_TOKEN; });
    outputTokenIds.erase(it.base(), outputTokenIds.end());
}

size_t SequenceData::GetNumComputedTokens() const { return numComputedTokens_; }

size_t SequenceData::GetNumUncomputedTokens()
{
    size_t totalLength = GetLength();
    size_t numComputedTokens = GetNumComputedTokens();
    Assert(totalLength >= numComputedTokens);
    return (totalLength - numComputedTokens);
}

void SequenceData::UpdateNumComputedTokens(size_t numNewComputedTokens)
{
    numComputedTokens_ += numNewComputedTokens;
    if (GetNumUncomputedTokens() == 0) {
        stage_ = SequenceStage::DECODE;
    }
}

void SequenceData::SetLayerwiseStage(bool isPrefill)
{
    if (isPrefill) {
        layerwiseStage_ = SequenceStage::PREFILL;
    } else {
        layerwiseStage_ = SequenceStage::DECODE;
    }
}

/**
 * 重新计算的时候，需要将前面已经重新prefill
 */
size_t SequenceData::GetLength() { return outputTokenIds.size() + promptTokenIds.size(); }

Sequence::Sequence(SequenceId seqId, int blockSize)
    : seqId_(seqId), blockSize_(blockSize), status_(SequenceStatus::WAITING)
{
}

Sequence::Sequence(SequenceId seqId, int blockSize, const std::vector<TokenId> &inputs)
    : seqId_(seqId), blockSize_(blockSize), data_(SequenceData::FromSequence(inputs)), status_(SequenceStatus::WAITING)
{
}

size_t Sequence::GetLen() { return data_.GetLength(); }

size_t Sequence::GetOutputLen(bool containsPlaceholder)
{
    if (containsPlaceholder) {
        return data_.outputTokenIds.size();
    } else {
        return std::count_if(data_.outputTokenIds.rbegin(), data_.outputTokenIds.rend(),
                             [](auto token) { return token != PLACEHOLDER_TOKEN; });
    }
}

bool Sequence::IsPrefill() const { return data_.stage_ == SequenceStage::PREFILL; }

bool Sequence::IsLayerwisePrefill() const { return data_.layerwiseStage_ == SequenceStage::PREFILL; }

bool Sequence::IsFinished() const { return sequence_status::IsFinish(status_); }

size_t Sequence::GetNumComputedTokens() const { return data_.GetNumComputedTokens(); }

size_t Sequence::GetNumUncomputedTokens()
{
    if (data_.stage_ == SequenceStage::DECODE) {
        return 1;
    } else {
        return data_.GetNumUncomputedTokens();
    }
}

void Sequence::ResetStateForCompute() { data_.ResetStateForRecompute(); }

const std::vector<TokenId> Sequence::GetTokenIds() const
{
    // merge promptTokenIds and outputTokenIds
    std::vector<TokenId> tokenIds = data_.promptTokenIds;
    tokenIds.insert(tokenIds.end(), data_.outputTokenIds.begin(), data_.outputTokenIds.end());
    return tokenIds;
}

HashValue Sequence::GetExtraHash() const
{
    return hashValue_;
}

void Sequence::SetExtraHash(HashValue hashValue)
{
    hashValue_ = hashValue;
}
} // namespace mindie_llm
