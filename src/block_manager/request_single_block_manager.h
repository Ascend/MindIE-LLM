/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "basic_types.h"
#include "block_manager_interface.h"
#include "cpu_npu_block_allocator.h"

namespace mindie_llm {
class RequestSingleBlockManager : public BlockSpaceManager {
public:
    explicit RequestSingleBlockManager(const BlockManagerConfig &config, size_t localDPRank = 0);

    AllocStatus CanAllocate(const SequenceGroupSPtr &seqGroup) const override;
    bool Allocate(const SequenceGroupSPtr &seqGroup) override;

    bool CanAppendSlot(const SequenceGroupSPtr &seqGroup) const override;
    std::vector<std::pair<BlockId, BlockId>> AppendSlot(const SequenceSPtr &seq) override;

    bool CanAppendSlotNew(const SequenceGroupSPtr &seqGroup) const override;
    void AppendSlotNew(const SequenceGroupSPtr &seqGroup) override;

    void AppendTokenToLatestRank(SequenceId seqId, const std::vector<TokenId>& tokens) override;

    void Fork(SequenceSPtr &parentSeq, SequenceSPtr &childSeq) override;

    // Not supported for request-fixed manager currently.
    bool CanSwapOut(const SequenceGroupSPtr &seqGroup) override;
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SwapOut(const SequenceGroupSPtr &seqGroup) override;

    // Not supported for request-fixed manager currently.
    AllocStatus CanSwapIn(const SequenceGroupSPtr &seqGroup, size_t numLookheadSlots) override;
    std::vector<std::pair<PhysicalBlockId, PhysicalBlockId>> SwapIn(const SequenceGroupSPtr &seqGroup) override;

    void Free(SequenceId seqId) override;

    std::vector<BlockIds> GetBlockIds(SequenceId seqId) const override;
    void GetRankedBlockIds(SequenceId seqId, std::vector<RankedBlockId> &rankedBlockIds) const override;
    void GetRankedBlockIds(SequenceId seqId, std::vector<std::vector<BlockId>> &rankedBlockIds) const override;

    std::vector<std::vector<HashValue>> GetRankedHashValues(SequenceId seqId) const override;
    std::vector<HashValue> GetSeqHashValues(SequenceId seqId) const override;

    std::vector<size_t> GetTokenCountPerRank(SequenceId seqId) const override;
    size_t GetLatestAppendedRankId(SequenceId seqId) const override;
    size_t GetAppendedBlockRankId(SequenceId seqId) const override;
    bool IsAppendBlock(SequenceId seqId) override;

    size_t GetNumFreeNpuBlocks() const override;
    size_t GetNumFreeCpuBlocks() const override;
    size_t GetTotalNpuBlocks() const override { return npuBlockNum_; }

    void AccessAllblocksInSeq(const SequenceSPtr &seq, float accessTime) override;

    std::vector<BlockId> GetCommonComputedBlockIds(const std::vector<SequenceSPtr> &seqs) override;
    std::vector<size_t> GetAllrankComputedBlockNum(const std::vector<SequenceSPtr> &seqs) override;
    std::vector<BlockId> GetRemoteComputedBlockIds(const std::vector<SequenceSPtr> &seqs,
                                                  size_t computedLens, uint32_t tpSize, std::string modelName) override;
    std::vector<size_t> GetAllRankRemoteComputedBlockIds(
        const std::vector<SequenceSPtr> &seqs, std::vector<size_t> &computedBlocksNum, std::string modelName) override;

    void MarkBlocksAsComputed() override;
    float GetPrefixCacheHitRate() const override;
    bool ResetPrefixCache() const override;

    size_t GetNumCachedTokens(const SequenceSPtr &seq) override;
    size_t GetSeqNumCachedTokens(const SequenceSPtr &seq) override;

    void ReplaceTrailingPlaceHolder(const SequenceSPtr &seq, size_t trailingPlaceHolderNum,
                                    size_t replacedPlaceHolderNum) override;

    size_t GetLocalDPRank() const override { return localDPRank_; }

    void LwdInitCloudBlockManager(const BlockManagerConfig &lwdCloudConfig, size_t localDPRank) override;

    void LwdGetCloudRankedBlockIds(SequenceId seqId,
        std::vector<std::vector<BlockId>> &rankedBlockIds) const override;

    size_t LwdGetCloudLatestAppendedRankId(SequenceId seqId) const override;

    size_t LwdGetCloudAppendedBlockRankId(SequenceId seqId) const override;

    std::vector<size_t> LwdGetCloudTokenCountPerRank(SequenceId seqId) const override;

private:
    struct RequestEntry {
        BlockObjSPtr block;   // single block object for this request
        size_t refCount{0};   // number of sequences currently referencing this request entry
    };

    const RequestId* GetRequestIdBySeqId_(SequenceId seqId) const;
    RequestEntry* GetEntryByRequestId_(const RequestId &rid);
    const RequestEntry* GetEntryByRequestId_(const RequestId &rid) const;

    size_t blockSize_{0};
    size_t cpuBlockNum_{0};
    size_t npuBlockNum_{0};
    size_t reservedBlockNum_{0};
    bool enableCaching_{false};
    size_t rankSize_{1};
    size_t hostSize_{1};
    size_t localDPRank_{0};

    DeviceAwareBlockAllocatorSPtr blockAllocator_;

    // seqId -> requestId
    std::unordered_map<SequenceId, RequestId> seqId2RequestId_;
    // requestId -> entry
    std::unordered_map<RequestId, RequestEntry> requestEntries_;
};
} // namespace mindie_llm
