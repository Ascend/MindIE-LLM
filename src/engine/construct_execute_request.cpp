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
#include <string>
#include "log.h"
#include "id_utils.h"
#include "construct_execute_request.h"

namespace mindie_llm {

model_execute_data::SequenceData ConstructExecuteRequest::MakeSequenceData(SequenceData &metaData)
{
    model_execute_data::SequenceData protoSeqData;
    for (TokenId promptTokenId : metaData.promptTokenIds) {
        protoSeqData.add_prompt_token_ids(promptTokenId);
    }
    for (TokenId outputTokenId : metaData.outputTokenIds) {
        protoSeqData.add_output_token_ids(outputTokenId);
    }

    return protoSeqData;
}

void ConstructExecuteRequest::ConstructSampleParam(model_execute_data::SamplingParams &sampleParams,
                                                   const SequenceGroupMetaData &metaData)
{
    sampleParams.set_n(metaData.samplingParams_->n);
    sampleParams.set_best_of(metaData.samplingParams_->bestOf);
    sampleParams.set_max_output_len(metaData.samplingParams_->maxOutputLen);
    if (metaData.samplingParams_->presencePenalty.has_value()) {
        sampleParams.set_presence_penalty(metaData.samplingParams_->presencePenalty.value());
    }
    if (metaData.samplingParams_->frequencyPenalty.has_value()) {
        sampleParams.set_frequency_penalty(metaData.samplingParams_->frequencyPenalty.value());
    }
    if (metaData.samplingParams_->repetitionPenalty.has_value()) {
        sampleParams.set_repetition_penalty(metaData.samplingParams_->repetitionPenalty.value());
    }
    if (metaData.samplingParams_->temperature.has_value()) {
        sampleParams.set_temperature(metaData.samplingParams_->temperature.value());
    }
    if (metaData.samplingParams_->topK.has_value()) {
        sampleParams.set_top_k(metaData.samplingParams_->topK.value());
    }
    if (metaData.samplingParams_->topP.has_value()) {
        sampleParams.set_top_p(metaData.samplingParams_->topP.value());
    }
    if (metaData.samplingParams_->seed.has_value()) {
        sampleParams.set_seed(metaData.samplingParams_->seed.value());
    }
    if (metaData.samplingParams_->logprobs.has_value()) {
        sampleParams.set_logprobs(metaData.samplingParams_->logprobs.value());
    }
    if (metaData.samplingParams_->topLogprobs.has_value()) {
        sampleParams.set_top_logprobs(metaData.samplingParams_->topLogprobs.value());
    }
    sampleParams.set_use_beam_search(metaData.samplingParams_->useBeamsearch);
}

void ConstructExecuteRequest::ConstructChunkedPrefillParam(const SequenceGroupMetaData &metaData,
                                                           model_execute_data::SequenceGroupMetadata &protoMeta)
{
    // 基础参数
    for (size_t i = 0; i < metaData.isReqPrefill_.size(); ++i) {
        protoMeta.add_is_req_prefill(metaData.isReqPrefill_[i]);
        protoMeta.add_is_req_last_chunk(metaData.isReqLastChunk_[i]);
        protoMeta.add_split_start_pos(metaData.splitStartPos_[i]);
        protoMeta.add_split_end_pos(metaData.splitEndPos_[i]);
    }

    // 对于prefill请求，需要根据分块长度更新prompt_lens和prompt_token_ids
    if (metaData.isReqPrefill_.size() == 1 && metaData.isReqPrefill_[0]) {
        std::vector<size_t> partialPromtLens = {metaData.splitEndPos_[0] - metaData.splitStartPos_[0]};
        protoMeta.set_prompt_lens(partialPromtLens.data(), sizeof(int64_t));
        protoMeta.set_prompt_token_ids(
            metaData.tokenIds_.data() + metaData.splitStartPos_[0],
            (metaData.splitEndPos_[0] - metaData.splitStartPos_[0]) * sizeof(TokenId)
        );
    }

    // ChunkedPrefill下prefill和decode都需要computed_block，以保证混合batch在TextGenerator中维度完整
    protoMeta.set_computed_block_lens(metaData.computedLens_.data(), metaData.computedLens_.size() * sizeof(int64_t));
    protoMeta.set_remote_computed_block_lens(metaData.remoteComputedLens_.data(),
                                             metaData.remoteComputedLens_.size() * sizeof(int64_t));
}

void ConstructExecuteRequest::ConstructPrefillData(const SequenceGroupMetaData &metaData,
                                                   model_execute_data::SequenceGroupMetadata &protoMeta)
{
    // 构造prompt token id
    protoMeta.set_prompt_lens(metaData.promptLens_.data(), metaData.promptLens_.size() * sizeof(int64_t));
    protoMeta.set_prompt_token_ids(metaData.tokenIds_.data(), metaData.tokenIds_.size() * sizeof(TokenId));

    // 构造computed block table
    protoMeta.set_computed_block_lens(metaData.computedLens_.data(), metaData.computedLens_.size() * sizeof(int64_t));
    protoMeta.set_remote_computed_block_lens(metaData.remoteComputedLens_.data(),
                                             metaData.remoteComputedLens_.size() * sizeof(int64_t));
    // 边云特性动态切块使用，prefill request携带请求率信息
    protoMeta.set_request_gap(metaData.requestGap_);
}

void ConstructExecuteRequest::ConstructProtoMeta(const SequenceGroupMetaData &metaData,
                                                 model_execute_data::SequenceGroupMetadata &protoMeta, bool isPrefill)
{
    protoMeta.set_request_id(metaData.requestId_);
    protoMeta.set_server_id(metaData.serverid_);
    protoMeta.set_is_prompt(isPrefill);
    if (isPrefill) {
        ConstructPrefillData(metaData, protoMeta);
    }

    protoMeta.set_seqids(metaData.seqIds_.data(), metaData.seqIds_.size() * sizeof(SequenceId));

    // 构造block table
    protoMeta.clear_block_tables();
    for (const auto &mgrBlockIds : metaData.blockIds_) {
        const size_t bytesLen = mgrBlockIds.size() * sizeof(BlockId);
        if (bytesLen == 0) {
            protoMeta.add_block_tables("");
            continue;
        }
        const char *ptr = reinterpret_cast<const char *>(mgrBlockIds.data());
        protoMeta.add_block_tables(ptr, bytesLen);
    }
    
    if (metaData.isSp_ or metaData.isCp_) {
        // sp & cp common part
        protoMeta.set_sp_rank_id(metaData.spRankId_);
        protoMeta.set_append_block_rank_id(metaData.appendBlockRankId_);
        if (metaData.isMtp_) {
            protoMeta.set_is_append_block(metaData.isAppendBlock_);
            for (size_t rankId : metaData.prefillBlockRankId_) {
                protoMeta.add_prefill_block_rank_id(rankId);
            }
        }
        for (size_t tokenNum : metaData.spRankPromptTokenNum_) {
            protoMeta.add_sp_rank_token_num(tokenNum);
        }

        for (size_t blockNum : metaData.spRankBlockNum_) {
            protoMeta.add_sp_rank_block_num(blockNum);
        }
    }
    
    protoMeta.add_stop(metaData.samplingParams_->stopStrings);
    for (TokenId stopTokenId : metaData.samplingParams_->stopTokenIds) {
        protoMeta.add_stop_token_ids(stopTokenId);
    }
    if (metaData.samplingParams_->includeStopStrInOutput.has_value()) {
        protoMeta.set_include_stop_str_in_output(metaData.samplingParams_->includeStopStrInOutput.value());
    }

    if (metaData.skipSpecialTokens_.has_value()) {
        protoMeta.set_skip_special_tokens(metaData.skipSpecialTokens_.value());
    }

    if (metaData.ignoreEos_.has_value()) {
        protoMeta.set_ignore_eos(metaData.ignoreEos_.value());
    }

    if (metaData.loraId_.has_value()) {
        protoMeta.set_lora_id(metaData.loraId_.value());
    }

    if (metaData.responseFormat_.has_value()) {
        protoMeta.set_response_format(metaData.responseFormat_.value());
    }

    // 采样参数
    if (metaData.samplingParams_->doSample.has_value()) {
        protoMeta.set_do_sample(metaData.samplingParams_->doSample.value());
    }
    model_execute_data::SamplingParams *sampleParams = protoMeta.mutable_sampling_params();
    if (sampleParams != nullptr) {
        ConstructSampleParam(*sampleParams, metaData);
    }

    // ChunkedPrefill参数
    if (!metaData.isReqPrefill_.empty()) {
        ConstructChunkedPrefillParam(metaData, protoMeta);
    }

    // 添加reserved_seq_id参数
    if (metaData.samplingParams_->n == 0 || metaData.samplingParams_->bestOf == 0) {
        throw std::runtime_error("Invalid sampling parameters: n or bestOf must be greater than 0");
    }
    uint32_t reservedSeqNum = metaData.samplingParams_->useBeamsearch ? metaData.samplingParams_->n - 1
                                                                      : metaData.samplingParams_->bestOf - 1;
    for (uint32_t i = 0; i < reservedSeqNum; i++) {
        protoMeta.add_reserved_seq_ids(IDUtils::GenerateSequenceId());
    }

    // beamsearch叠加chunkedprefill时对于非最后一个chunk需要去除beam参数
    if (metaData.samplingParams_ != nullptr &&
        !metaData.isReqLastChunk_.empty() &&
        metaData.samplingParams_->useBeamsearch &&
        !metaData.isReqLastChunk_[0] &&
        sampleParams != nullptr) {
        ClearBeamParam4ChunkedPrefill(*sampleParams, protoMeta);
    }
}

void ConstructExecuteRequest::LwdConstructCloudProtoMeta(const SequenceGroupMetaData &metaData,
    model_execute_data::SequenceGroupMetadata &protoMeta, bool isPrefill)
{
    auto* lwdMeta = protoMeta.mutable_lwd_cloud_metadata();
    lwdMeta->set_lwd_cloud_block_tables(metaData.lwdCloudBlockIds_.data(),
        metaData.lwdCloudBlockIds_.size() * sizeof(BlockId));
    if (metaData.isSp_ || metaData.isCp_) {
        lwdMeta->set_lwd_cloud_sp_rank_id(metaData.lwdCloudSpRankId_);
        lwdMeta->set_lwd_cloud_append_block_rank_id(metaData.lwdCloudAppendBlockRankId_);
        for (size_t tokenNum : metaData.lwdCloudSpRankPromptTokenNum_) {
            lwdMeta->add_lwd_cloud_sp_rank_token_num(tokenNum);
        }
        for (size_t blockNum : metaData.lwdCloudSpRankBlockNum_) {
            lwdMeta->add_lwd_cloud_sp_rank_block_num(blockNum);
        }
    }
}

void ConstructExecuteRequest::ClearBeamParam4ChunkedPrefill(model_execute_data::SamplingParams &sampleParams,
                                                            model_execute_data::SequenceGroupMetadata &protoMeta)
{
    // beamsearch叠加chunkedprefill时，对非最后一个chunk的prefill请求不能扩充为n个序列
    sampleParams.set_use_beam_search(false);
    sampleParams.set_best_of(1);
    sampleParams.set_n(1);
    protoMeta.clear_reserved_seq_ids();
}

model_execute_data::ForwardType ConstructExecuteRequest::ConvertToProtoForwardType(ForwardMode fMode)
{
    switch (fMode) {
        case ForwardMode::PREFILL: return model_execute_data::ForwardType::PREFILL;
        case ForwardMode::DECODE: return model_execute_data::ForwardType::DECODE;
        case ForwardMode::EXTEND: return model_execute_data::ForwardType::EXTEND;
        case ForwardMode::MIXED: return model_execute_data::ForwardType::MIXED;
        case ForwardMode::DUMMY: return model_execute_data::ForwardType::DUMMY;
        default: throw std::runtime_error("Not support ForwardMode");
    }
}

void ConstructExecuteRequest::ConstructExecuteModelRequest(ExecuteModelRequestPtr &modelRequest,
    SequenceGroupMetaDatas &metadatas, SchedulerOutputs &scOut, size_t dpRankId)
{
    // construct proto metadata
    bool isPrefill = scOut.forwardMode_ == ForwardMode::PREFILL;
    for (const SequenceGroupMetaData &metadata : metadatas.metaList) {
        model_execute_data::SequenceGroupMetadata *protoMeta = modelRequest->add_seq_group_metadata_list();
        if (protoMeta == nullptr) {
            continue;
        }

        ConstructProtoMeta(metadata, *protoMeta, isPrefill);
        LwdConstructCloudProtoMeta(metadata, *protoMeta, isPrefill);
        protoMeta->set_dp_rank_id(dpRankId);
    }

    modelRequest->set_forward_type(ConvertToProtoForwardType(scOut.forwardMode_));

    // 边云特性动态切块使用，给TG侧传waiting队列长度
    modelRequest->set_wait_queue_len(scOut.curWaitQueueLen_);

    for (auto pair : scOut.blocksToCopy_) {
        model_execute_data::IntPair *protoPair = modelRequest->add_blocks_to_copy();
        protoPair->set_num1(pair.first);
        protoPair->set_num2(pair.second);
    }
    for (auto pair : scOut.blocksToSwapIn_) {
        model_execute_data::IntPair *protoPair = modelRequest->add_blocks_to_swap_in();
        protoPair->set_num1(pair.first);
        protoPair->set_num2(pair.second);
    }
    for (auto pair : scOut.blocksToSwapOut_) {
        model_execute_data::IntPair *protoPair = modelRequest->add_blocks_to_swap_out();
        protoPair->set_num1(pair.first);
        protoPair->set_num2(pair.second);
    }
    modelRequest->set_running_queue_size(scOut.runningQueueSize_);

    // 分布式P节点DP间同步batch的seqlen信息传递到给后端，用于打padding
    for (size_t dp_idx = 0; dp_idx < metadatas.seqLenList.size(); ++dp_idx) {
        auto *dp_batch_seq_lens = modelRequest->add_all_dp_batches_seq_lens();
        for (size_t seq_idx = 0; seq_idx < metadatas.seqLenList[dp_idx].size(); ++seq_idx) {
            dp_batch_seq_lens->add_seq_lens(metadatas.seqLenList[dp_idx][seq_idx]);
        }
    }
}

PullKVRequestPtr ConstructExecuteRequest::ConstructPullKVRequest(SequenceGroupMetaDatas &seqGroupMetadata)
{
    PullKVRequestPtr request = std::make_unique<model_execute_data::PullKVRequest>();
    for (auto &metadata : seqGroupMetadata.metaList) {
        auto *info = request->add_pull_kv_infos();

        for (const auto& blockIds : metadata.blockIds_) {
            const size_t bytesLen = blockIds.size() * sizeof(BlockId);
            if (bytesLen == 0) {
                info->add_dst_block_tables("");
                continue;
            }
            const char *ptr = reinterpret_cast<const char *>(blockIds.data());
            info->add_dst_block_tables(ptr, bytesLen);
        }

        for (const auto& blockIds : metadata.srcBlockIds_) {
            const size_t bytesLen = blockIds.size() * sizeof(BlockId);
            if (bytesLen == 0) {
                info->add_src_block_tables("");
                continue;
            }
            const char *ptr = reinterpret_cast<const char *>(blockIds.data());
            info->add_src_block_tables(ptr, bytesLen);
        }

        info->set_cluster_id(std::to_string(metadata.dpInstanceId_));

        model_execute_data::SequenceGroupMetadata *protoMeta = info->mutable_seq_group_metadata();
        ConstructProtoMeta(metadata, *protoMeta, true);
    }

    return request;
}
} // namespace mindie_llm