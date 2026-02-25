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
#include "models/deepseekv2/model/mtp_decoder_model.h"
#include <vector>
#include "atb/comm.h"
#include "hccl/hccl.h"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/lmhead/hidden_state_slice.h"


namespace atb_speed {
namespace deepseekV2 {

// Weight count
constexpr uint32_t WEIGHT_COUNT_PER_LAYER = 84;
constexpr uint32_t WEIGHT_COUNT_BEFORE_LAYER = 4;
constexpr uint32_t WEIGHT_COUNT_POST_NORM = 1;
constexpr uint32_t WEIGHT_COUNT_LM_HEAD = 1;
constexpr uint32_t DECODER_WEIGHT_COUNT_PER_LAYER = 24; // 共享专家权重18 + 路由权重gate6
// quant linear count
constexpr uint32_t DEEPSEEKV2_LINEAR_TYPE_LENGTH = 9;

// Operation count
constexpr uint32_t OPERATION_COUNT_BEFORE_LAYER = 2;  // Word Embedding + Positional Embedding
constexpr uint32_t OPERATION_COUNT_AFTER_LAYER = 2;  // RmsNorm + LmHead

constexpr uint32_t ATTN_LINEAR_TYPE_LENGTH = 6;
constexpr uint32_t MLP_LINEAR_TYPE_LENGTH = 4;
constexpr uint32_t MOE_LINEAR_TYPE_LENGTH = 4;

constexpr uint32_t MODEL_OUT_MTP = 2;

MtpDecoderModel::MtpDecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->param.FromString(param);
    modelName_ += this->param.isPrefill ? "_Prefill" : "_Decoder";
}

static std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelInTensorCandidates = {
        {"default", {
            "in_tensor_input_ids", "in_tensor_position_ids", "in_tensor_cos_table", "in_tensor_sin_table",
            "in_tensor_attention_mask", "in_tensor_block_tables", "in_tensor_slots", "in_tensor_kvcache_idx",
            "in_final_state_model",
            "in_tensor_token_offset", "in_tensor_place_holder", "in_tensor_seq_len", "in_tensor_logits_indices",
            "in_expert_array_model", "in_expert_group_model", "in_one_hot_model", "in_zero_hot_model",
            "in_tensor_q_len",
            "in_tensor_final_hidden_state",
            }},
        {"parallel_input", {
            "in_attn_padding_idx_model", "in_attn_unpadding_idx_model",
            "in_ffn_padding_idx_model", "in_ffn_unpadding_idx_model",
            "in_lm_head_skip_padding_token_indices_model",
            "in_attention_padding_idx_slice", "in_start_expert_idx_model",
            "in_device_expert_count_model",
            "in_lty_idx_model", "in_moe_idx_model", "in_post_lmhead_unpadding_indices"}},
        {"attn_cp_prefill", {"in_seq_len_cp", "in_cp_load_balance_idx_first", "in_cp_load_balance_idx_last",
                             "in_cp_o_recover_idx", "in_cp_kv_recover_idx"}},
        {"attn_inner_sp_decode", {"in_seq_len_sp"}},
        {"sp_mtp", {"is_need_mask"}},
        {"attn_cp_sp_decode", {"in_filter_mask"}},
        {"epwb", {
            "in_expert_routing_map_model"}},
        {"mix_shared_routing", {
            "mix_shared_routing_weight",
            "mix_shared_routing_expert"
        }},
        {"prefixcache", {
            "in_history_compressed_kv", "in_history_k_rope", "ring_cur_seqlen", "ring_cache_seqlen"
        }},
        {"prefixcache_cp", {
            "in_kv_cache_padding_idx", "in_kv_cache_unpadding_idx", "in_kv_cache_len"
        }},
        {"prefixcache_c8", {"in_history_compressed_kv_int"}},
        {"dense_tp", {
            "in_dense_tp_padding_idx_model", "in_dense_tp_mlp_out_idx_model",
            "in_dense_tp_attn_add_out_idx_model", "in_dense_tp_gather_prenorm_idx_model",
            "in_dense_tp_mlp_rs_out_idx_model"
        }}
    };
    return deepseekV2ModelInTensorCandidates;
}

void MtpDecoderModel::ConstructInTensorMap()
{
    auto deepseekV2ModelInTensorCandidates = GetDeepseekV2ModelInTensorCandidates();
    atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "default", this->inTensorMap);
    atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates,
        "parallel_input", this->inTensorMap);
    if (param.mapping.Get(base::ATTN_CP).IsEnabled() && param.isPrefill) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "attn_cp_prefill", this->inTensorMap);
    }
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() && !param.isPrefill) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelInTensorCandidates, "attn_inner_sp_decode", this->inTensorMap);
        if (param.enableSpeculate) {
            atb_speed::common::AssignTensorIdx(
                deepseekV2ModelInTensorCandidates, "sp_mtp", this->inTensorMap);
        }
    }
    if ((param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) &&
        !param.isPrefill) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelInTensorCandidates, "attn_cp_sp_decode", this->inTensorMap);
    }
    if (param.enablePrefixCache) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "prefixcache", this->inTensorMap);
        if (param.enableFA3) {
            atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "prefixcache_c8", this->inTensorMap);
        }
        if (param.mapping.Get(base::ATTN_CP).IsEnabled()) {
            atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "prefixcache_cp", this->inTensorMap);
        }
    }
    if (param.enableDenseTp) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "dense_tp", this->inTensorMap);
    }
    if (param.enableEPWB) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "epwb", this->inTensorMap);
    }
    if (param.mixSharedRouting) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "mix_shared_routing", this->inTensorMap);
    }
}

static std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelInternalTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelInternalTensorCandidates = {
        {"default", {
            "internal_tensor_hidden_states", "internal_tensor_cos_emb", "internal_tensor_sin_emb",
            "enorm", "hnorm", "concat"}},
        {"last_layer", {
            "internal_tensor_last_layer"}},
        {"enable_dp_out", {"internal_lmhead_out", "internal_lmhead_out_dp", "internal_hidden_states_out_dp"}}
    };
    return deepseekV2ModelInternalTensorCandidates;
}

void MtpDecoderModel::ConstructInternalTensorMap()
{
    auto deepseekV2ModelInternalTensorCandidates = GetDeepseekV2ModelInternalTensorCandidates();
    atb_speed::common::AssignTensorIdx(
        deepseekV2ModelInternalTensorCandidates, "default", this->internalTensorMap);
    if (param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelInternalTensorCandidates, "last_layer", this->internalTensorMap);
    }
    if (param.enableDpOut && param.lmHeadLocalTp) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelInternalTensorCandidates, "enable_dp_out", this->internalTensorMap);
    }
}

void MtpDecoderModel::ConstructOutTensorMap()
{
    this->outTensorMap.clear();
    std::map<std::string, std::vector<std::string>> deepseekV2ModelOutTensorCandidates = {
        {"default", {"logits", "final_hidden_states"}},
        {"eplb_data_collection", {"activation_count_per_expert"}},
        {"topk", {"activation_topk"}},
    };
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(
        deepseekV2ModelOutTensorCandidates, "default", this->outTensorMap);
    if (param.enableExpertCumSumOutput) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelOutTensorCandidates, "eplb_data_collection", this->outTensorMap);
    }
    if (param.enableTopkOutput) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelOutTensorCandidates, "topk", this->outTensorMap);
    }
}

atb::TensorDesc MtpDecoderModel::GetLogitsDesc(
    const std::vector<atb::TensorDesc> &inTensorDescs, uint32_t logitsIndicesIdx)
{
    atb::TensorDesc logitsDesc;
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    logitsDesc.dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    logitsDesc.format = graph_.weightTensors.at(0).desc.format;
    logitsDesc.shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;

    logitsDesc.shape.dims[0] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
    if (param.isFA) {  // unpadInputs = false
        logitsDesc.shape.dims[1] = \
            param.isPrefill ? inTensorDescs.at(logitsIndicesIdx).shape.dims[0] : 1;
    }

    if (param.isLmHeadParallel) {
        if (param.mapping.Get(base::MLP_TP).IsEnabled()) {
            logitsDesc.shape.dims[logitsDesc.shape.dimNum - 1] = \
            CheckIntMulOverFlow(vocabSizePerRank,
                static_cast<int64_t>(param.mapping.Get(base::LM_HEAD_TP).rankIds.size()));
        } else {
            logitsDesc.shape.dims[logitsDesc.shape.dimNum - 1] = \
            CheckIntMulOverFlow(vocabSizePerRank,
                static_cast<int64_t>(param.mapping.Get(base::MOE_EP).rankIds.size()));
        }
    } else {
        logitsDesc.shape.dims[logitsDesc.shape.dimNum - 1] = vocabSizePerRank;
    }
    return logitsDesc;
}

atb::Status MtpDecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter MtpDecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    uint32_t outTensorIdx = 0;
    uint32_t logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices");
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum + 1);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);
    outTensorDescs.at(outTensorIdx) = GetLogitsDesc(inTensorDescs, logitsIndicesIdx);
    outTensorIdx++;

    uint32_t hiddenIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_final_state_model");
    outTensorDescs.at(outTensorIdx) = inTensorDescs.at(hiddenIndicesIdx);
    outTensorIdx++;

    if (param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();
        logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices" + suffix);
        CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);
        outTensorDescs.at(outTensorIdx) = GetLogitsDesc(inTensorDescs, logitsIndicesIdx);
        outTensorIdx++;

        hiddenIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_final_state_model" + suffix);
        outTensorDescs.at(outTensorIdx) = inTensorDescs.at(hiddenIndicesIdx);
        outTensorIdx++;
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }
    if (param.enableExpertCumSumOutput && param.enableTopkOutput) {
        atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx, param.numOfDeviceExperts);
        outTensorIdx++;
        atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx,
            param.numOfSelectedExperts.at(0), false);
        outTensorIdx++;
    } else if (param.enableExpertCumSumOutput) {
        atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx, param.numOfDeviceExperts);
        outTensorIdx++;
    } else if (param.enableTopkOutput) {
        atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx,
            param.numOfSelectedExperts.at(0), false);
        outTensorIdx++;
    }
    return atb::NO_ERROR;
}

int MtpDecoderModel::GetWeightCountPerLayer()
{
    int weightCountPerLayerTmp = WEIGHT_COUNT_PER_LAYER;
    if (param.enableFA3) {
        weightCountPerLayerTmp += 5; // 5: FA3 多5个inTensorensor
    }
    return weightCountPerLayerTmp;
}

uint32_t MtpDecoderModel::CalcWeightTensorSize()
{
    weightCountPerLayer = GetWeightCountPerLayer();
    int weightTensorSize = 0;
    if (param.hasP2DWeight) {
        weightTensorSize =
            WEIGHT_COUNT_BEFORE_LAYER + CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
            CheckIntMulOverFlow(DECODER_WEIGHT_COUNT_PER_LAYER, param.numHiddenLayers - 0) +
            WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    } else {
        weightTensorSize =
            WEIGHT_COUNT_BEFORE_LAYER + CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
            WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
    }
    return weightTensorSize;
}

atb::Status MtpDecoderModel::AddNodesBeforeLayer()
{
    CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddENorm());
    CHECK_OPERATION_STATUS_RETURN(AddHNorm());
    CHECK_OPERATION_STATUS_RETURN(AddConcat());
    CHECK_OPERATION_STATUS_RETURN(AddLinear());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddNodesAfterLayer()
{
    if (param.lmHeadLocalTp && param.enableDistributed) {
        CHECK_OPERATION_STATUS_RETURN(AddSliceFinalStateOut());
        CHECK_OPERATION_STATUS_RETURN(AddGatherFinalStateOut());
    }
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    if (param.enableDpOut && param.lmHeadLocalTp) {
        CHECK_OPERATION_STATUS_RETURN(AddGatherAfterLmhead());
        CHECK_OPERATION_STATUS_RETURN(AddIndicesGatherAfterLmhead());
    }
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddENorm()
{
    atb::Operation *op = nullptr;
    auto eNormNode = std::make_unique<atb_speed::Model::Node>();

    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));

    eNormNode->operation.reset(op);
    eNormNode->inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "internal_tensor_hidden_states")),
        &graph_.weightTensors.at(1),
    };
    eNormNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "enorm")),
    };
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddENorm");
    graph_.nodes.push_back(*eNormNode);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddHNorm()
{
    atb::Operation *op = nullptr;
    auto hNormNode = std::make_unique<atb_speed::Model::Node>();

    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));

    hNormNode->operation.reset(op);
    hNormNode->inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_final_hidden_state")),
        &graph_.weightTensors.at(2),
    };
    hNormNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "hnorm")),
    };
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddHNorm");
    graph_.nodes.push_back(*hNormNode);
    return atb::NO_ERROR;
}


atb::Status MtpDecoderModel::AddConcat()
{
    atb::Operation *op = nullptr;
    auto concatNode = std::make_unique<atb_speed::Model::Node>();

    atb::infer::ConcatParam qCatParam;
    qCatParam.concatDim = -1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(qCatParam, &op));

    concatNode->operation.reset(op);
    concatNode->inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "enorm")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hnorm")),
    };
    concatNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "concat")),
    };
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddConcat");
    graph_.nodes.push_back(*concatNode);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddLinear()
{
    atb::Operation *op = nullptr;
    auto linearNode = std::make_unique<atb_speed::Model::Node>();

    atb_speed::common::FusionLinearParam vReprojNodeParam;
    vReprojNodeParam.isBF16 = param.isBF16;
    vReprojNodeParam.hasBias = false;
    vReprojNodeParam.quantType = atb_speed::common::LinearQuantType::NO_QUANT;
    vReprojNodeParam.transposeType = true;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(vReprojNodeParam, &op));

    linearNode->operation.reset(op);
    linearNode->inTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "concat")),
        &graph_.weightTensors.at(3),
        &graph_.weightTensors.at(3),
        &graph_.weightTensors.at(3),
        &graph_.weightTensors.at(3),
        &graph_.weightTensors.at(3),
        &graph_.weightTensors.at(3),
    };
    linearNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "internal_tensor_hidden_states")),
    };
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddLinear");
    graph_.nodes.push_back(*linearNode);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    auto wordEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    wordEmbeddingParam.unpadInputs = !param.isFA;
    if (param.isEmbeddingParallel) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::WORD_EMBED_TP);
        wordEmbeddingParam.tensorParallelInfo.rank = parallelInfo.rank;
        wordEmbeddingParam.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        wordEmbeddingParam.tensorParallelInfo.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(
            wordEmbeddingParam.tensorParallelInfo.hcommInfo,
            wordEmbeddingParam.tensorParallelInfo.commDomain);
    };
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
    wordEmbeddingNode->operation.reset(op);
    wordEmbeddingNode->inTensors = {
        &graph_.weightTensors.at(0),                    // shape: [vocabSize + 1, hiddenSize]
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_input_ids"))
    };
    wordEmbeddingNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "internal_tensor_hidden_states"))
    };
    graph_.nodes.push_back(*wordEmbeddingNode);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    auto posEmbeddingNode = std::make_unique<atb_speed::Model::Node>();
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    posEmbeddingNode->operation.reset(op);
    posEmbeddingNode->inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_position_ids")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_cos_table")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_sin_table")),
    };
    posEmbeddingNode->outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "internal_tensor_cos_emb")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
                                                                   "internal_tensor_sin_emb")),
    };
    graph_.nodes.push_back(*posEmbeddingNode);
    return atb::NO_ERROR;
}

void MtpDecoderModel::SetLayerParam(DecoderLayerParam &layerParam, int64_t layerId)
{
    layerParam.isFA = param.isFA;
    layerParam.isPrefill = param.isPrefill;
    layerParam.isBF16 = param.isBF16;
    layerParam.enableSwiGLU = param.enableSwiGLU;
    layerParam.enableSwiGLUQuantForSharedExperts = param.enableSwiGLUQuantForSharedExperts;
    layerParam.enableLcoc = param.enableLcoc;
    layerParam.packQuantType = param.packQuantType[layerId];
    layerParam.attnLinearQuantType = param.attnLinearQuantType[layerId];
    layerParam.mlpLinearQuantType = param.mlpLinearQuantType[layerId];
    layerParam.moeLinearQuantType = param.moeLinearQuantType[layerId];
    layerParam.attnLinearTransposeType = param.attnLinearTransposeType[layerId];
    layerParam.mlpLinearTransposeType = param.mlpLinearTransposeType[layerId];
    layerParam.moeLinearTransposeType = param.moeLinearTransposeType[layerId];
    layerParam.normEps = param.normEps;
    layerParam.numAttentionHeadsPerRank = param.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = param.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = param.numKeyValueHeadsPerRank;
    layerParam.rank = param.rank;
    layerParam.worldSize = param.worldSize;
    layerParam.backend = param.backend;
    layerParam.rankTableFile = "";
    layerParam.layerId = layerId;
    layerParam.enableInterLayerAddNorm = param.enableInterLayerAddNorm;
    layerParam.enableIntraLayerAddNorm = param.enableIntraLayerAddNorm;
    layerParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    layerParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    layerParam.enableDpOut = param.enableDpOut;
    layerParam.lmHeadLocalTp = param.lmHeadLocalTp;
    layerParam.numHiddenLayers = param.numHiddenLayers;
    layerParam.enableSpeculate = param.enableSpeculate;
    layerParam.maskfree = param.maskfree;
    layerParam.enableGatingDp = param.enableGatingDp && layerId >= param.firstKDenseReplace;
    
    SetMlaParam(layerParam, param, layerId);
    SetMoeParam(layerParam, param, layerId);
    SetParallelParam(layerParam, param);
}

atb::Status MtpDecoderModel::AddLayerWeights(atb_speed::Model::Node &layerNode, size_t &inTensorId,
    const uint32_t layerId)
{
    weightCountPerLayer = GetWeightCountPerLayer();
    if (param.hasP2DWeight) {
        for (size_t weightTensorId = 0;
            weightTensorId < weightCountPerLayer + DECODER_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(layerId, weightCountPerLayer) \
                + CheckIntMulOverFlow(layerId - static_cast<uint32_t>(0),
                    DECODER_WEIGHT_COUNT_PER_LAYER) \
                + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
        }
    } else {
        for (size_t weightTensorId = 0; weightTensorId < weightCountPerLayer; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(layerId, weightCountPerLayer) \
                + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
        }
    }

    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddSingleLayer(uint32_t layerId)
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node layerNode;
    DecoderLayerParam layerParam;
    SetLayerParam(layerParam, layerId);
    ATB_SPEED_LOG_DEBUG("start create Decoderlayer");
    CHECK_OPERATION_STATUS_RETURN(DecoderLayer(layerParam, &op));
    ATB_SPEED_LOG_DEBUG("Decoderlayer create success");
    layerNode.operation.reset(op);
    ATB_SPEED_LOG_DEBUG("Decoderlayer inTensor number: " << layerNode.operation->GetInputNum());
    layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    size_t inTensorId = 0;
    AddLayerWeights(layerNode, inTensorId, layerId);
    ATB_SPEED_LOG_DEBUG("start add layerhostweight");
    AddLayerHostWeight(layerNode, inTensorId, layerId);
    ATB_SPEED_LOG_DEBUG("Add layerhostweight seccess");
    if (layerId == param.numHiddenLayers - 1 && \
        !(param.lmHeadLocalTp && param.enableDistributed)) {
            layerNode.outTensors = {&graph_.outTensors.at(
                atb_speed::common::GetTensorIdx(this->outTensorMap, "final_hidden_states"))};
    } else {
        const size_t layerInternalOutTensorId = atb_speed::common::GetTensorIdx(this->internalTensorMap,
            (param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) \
            && layerId == param.numHiddenLayers - 1 ? "internal_tensor_last_layer" : "internal_tensor_hidden_states");
        layerNode.outTensors = {&graph_.internalTensors.at(layerInternalOutTensorId)};
    }
    if (param.enableExpertCumSumOutput) {
        layerNode.outTensors.push_back(&graph_.outTensors.at(
            atb_speed::common::GetTensorIdx(this->outTensorMap, "activation_count_per_expert")));
    }
    if (param.enableTopkOutput) {
        layerNode.outTensors.push_back(&graph_.outTensors.at(
            atb_speed::common::GetTensorIdx(this->outTensorMap, "activation_topk")));
    }
    graph_.nodes.push_back(layerNode);
    ATB_SPEED_LOG_DEBUG("[+] add mtp layerNode num" << layerId);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddDenseTpHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId)
{
    if (param.enableDenseTp && layerId < param.firstKDenseReplace) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_dense_tp_padding_idx_model"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_dense_tp_mlp_out_idx_model"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_dense_tp_attn_add_out_idx_model"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_dense_tp_gather_prenorm_idx_model"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_dense_tp_mlp_rs_out_idx_model"));
    }
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attn_padding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attn_unpadding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_padding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_unpadding_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_lm_head_skip_padding_token_indices_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_attention_padding_idx_slice"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_start_expert_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_device_expert_count_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_lty_idx_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_moe_idx_model"));
    if (param.mapping.Get(base::ATTN_CP).IsEnabled() && param.isPrefill) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_seq_len_cp"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_cp_load_balance_idx_first"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_cp_load_balance_idx_last"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_cp_o_recover_idx"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_cp_kv_recover_idx"));
    }
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() && !param.isPrefill) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_seq_len_sp"));
        if (param.enableSpeculate) {
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                atb_speed::common::GetTensorIdx(this->inTensorMap, "is_need_mask"));
        }
    }
    if ((param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) &&
        !param.isPrefill) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_filter_mask"));
    }
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddPrefixCacheHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
{
    if (param.enablePrefixCache) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_history_compressed_kv"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_history_k_rope"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "ring_cur_seqlen"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "ring_cache_seqlen"));
        if (param.enableFA3) {
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                atb_speed::common::GetTensorIdx(this->inTensorMap, "in_history_compressed_kv_int"));
        }
    }
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddPrefixCacheCpHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
{
    if (param.enablePrefixCache && param.mapping.Get(base::ATTN_CP).IsEnabled()) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_kv_cache_padding_idx"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_kv_cache_unpadding_idx"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_kv_cache_len"));
    }
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddLayerHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_hidden_states"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_expert_array_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_expert_group_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_one_hot_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_zero_hot_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_final_state_model"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_cos_emb"));
    layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
        atb_speed::common::GetTensorIdx(this->internalTensorMap, "internal_tensor_sin_emb"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_seq_len"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_token_offset"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_kvcache_idx"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_block_tables"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_slots"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_q_len"));
    AddParallelHostWeight(layerNode, inTensorId);
    AddPrefixCacheHostWeight(layerNode, inTensorId);
    AddPrefixCacheCpHostWeight(layerNode, inTensorId);
    AddDenseTpHostWeight(layerNode, inTensorId, layerId);
    if (param.enableEPWB) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_expert_routing_map_model"));
    }
    if (param.mixSharedRouting) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "mix_shared_routing_weight"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "mix_shared_routing_expert"));
    }
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    auto finalNormNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    finalNormNode->operation.reset(op);
    const size_t finalLayerNormWeightTensorId =
        this -> graph_.weightTensors.size() - WEIGHT_COUNT_POST_NORM - WEIGHT_COUNT_LM_HEAD;
    const size_t layerOutTensorId = atb_speed::common::GetTensorIdx(this->internalTensorMap,
        param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled() ? \
        "internal_tensor_last_layer" : "internal_tensor_hidden_states");
    if (!(param.lmHeadLocalTp && param.enableDistributed)) {
        finalNormNode->inTensors = {
            &graph_.outTensors.at(
                atb_speed::common::GetTensorIdx(this->outTensorMap, "final_hidden_states")),
            &graph_.weightTensors.at(finalLayerNormWeightTensorId)
        };
    } else {
        finalNormNode->inTensors = {
            &graph_.internalTensors.at(layerOutTensorId),
            &graph_.weightTensors.at(finalLayerNormWeightTensorId)
        };
    }
    finalNormNode->outTensors = {
        // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
        &graph_.internalTensors.at(layerOutTensorId),
    };

    ATB_SPEED_LOG_DEBUG("MtpDecoderModel build graph:finalNormNode end");
    graph_.nodes.push_back(*finalNormNode);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    auto lmHeadNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param.isFA;
    lmHeadParam.gatherAhead = (param.enableDpOut && param.lmHeadLocalTp) ? false : true;
    lmHeadParam.hiddenSizePerAttentionHead = param.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.unpadInputs = !param.isFA;
    lmHeadParam.enableDpOut = param.enableDpOut;
    if (param.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
        lmHeadParam.linearParallelParam.tensorParallelInfo.rank = parallelInfo.rank;
        lmHeadParam.linearParallelParam.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        if (atb_speed::common::IsA2()) {
            lmHeadParam.linearParallelParam.tensorParallelInfo.backend = "hccl";
            parallelInfo.InitCommDomain(
                lmHeadParam.linearParallelParam.tensorParallelInfo.hcommInfo,
                lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain, "hccl");
        } else {
            lmHeadParam.linearParallelParam.tensorParallelInfo.backend = parallelInfo.defaultBackend;
            parallelInfo.InitCommDomain(
                lmHeadParam.linearParallelParam.tensorParallelInfo.hcommInfo,
                lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain);
        }
    }
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel build graph:create LMHead end");

    lmHeadNode->operation.reset(op);
    const size_t finalLinearWeightTensorId = this -> graph_.weightTensors.size() - WEIGHT_COUNT_LM_HEAD;
    const size_t finalLayerNormOutTensorId = atb_speed::common::GetTensorIdx(this->internalTensorMap,
        param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled() ? \
        "internal_tensor_last_layer" : "internal_tensor_hidden_states");
    uint32_t placeHolderIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_place_holder");
    lmHeadNode->inTensors = {
        &graph_.internalTensors.at(finalLayerNormOutTensorId),
        // shape: [vocabSizePerRank, hiddenSize]
        &graph_.weightTensors.at(finalLinearWeightTensorId),
        // LmHead未接入量化，量化权重使用placeholder代替
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(placeHolderIdx),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices"))
    };
    if (this->param.enableGreedyPostProcessing) {
        lmHeadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "logits_offset_tensor")));
    } else {
        lmHeadNode->inTensors.emplace_back(&graph_.inTensors.at(placeHolderIdx));
    }
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode->outTensors = {param.enableDpOut && param.lmHeadLocalTp ? \
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,"internal_lmhead_out")) : \
        &graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "logits"))};

    ATB_SPEED_LOG_DEBUG("MtpDecoderModel build graph success");
    graph_.nodes.push_back(*lmHeadNode);
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddSliceFinalStateOut()
{
    atb::Operation *op = nullptr;
    auto sliceNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::HiddenStateSliceParam hiddenstateSliceParam;

    hiddenstateSliceParam.rank = param.mapping.Get(base::LM_HEAD_TP).rank;
    hiddenstateSliceParam.world_size = param.mapping.Get(base::LM_HEAD_TP).rankIds.size();

    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::HiddenStateSlice(hiddenstateSliceParam, &op));

    sliceNode->inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
        this->internalTensorMap, param.mapping.Get(base::ATTN_DP).IsEnabled() ?
            "internal_tensor_last_layer" : "internal_tensor_hidden_states"))};
    sliceNode->outTensors =  {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
        this->internalTensorMap, "internal_hidden_states_out_dp"))};
    sliceNode->inTensorReshapeFuncs.resize(1);
    sliceNode->inTensorReshapeFuncs[0] = [&](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: world_size, token, hidden_state
        newShape.dims[0] = param.mapping.Get(base::LM_HEAD_TP).rankIds.size();
        newShape.dims[1] = oldShape.dims[0] / param.mapping.Get(base::LM_HEAD_TP).rankIds.size();
        newShape.dims[2] = oldShape.dims[1]; // 2: hidden_state
    };
    sliceNode->operation.reset(op);
    graph_.nodes.push_back(*sliceNode);

    ATB_SPEED_LOG_DEBUG("AddSliceFinalStateOut AddGatherFinalStateOut success");
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddGatherFinalStateOut()
{
    atb::Operation *op = nullptr;
    auto unpadNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &op));
    unpadNode->inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
        this->internalTensorMap, "internal_hidden_states_out_dp"))};
    unpadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "in_post_lmhead_unpadding_indices")));
    unpadNode->outTensors = {&graph_.outTensors.at(
        atb_speed::common::GetTensorIdx(this->outTensorMap, "final_hidden_states"))};
    unpadNode->operation.reset(op);
    graph_.nodes.push_back(*unpadNode);
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddGatherFinalStateOut success");
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddGatherAfterLmhead()
{
    atb::Operation *op = nullptr;
    auto unpadNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &op));
    unpadNode->inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
        "internal_lmhead_out"))};
    unpadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "in_post_lmhead_unpadding_indices")));
    unpadNode->outTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
        "internal_lmhead_out_dp"))};
    unpadNode->operation.reset(op);
    graph_.nodes.push_back(*unpadNode);
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddGatherAfterLmhead calculation success");
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::AddIndicesGatherAfterLmhead()
{
    atb::Operation *op = nullptr;
    auto unpadNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &op));
    unpadNode->inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
        "internal_lmhead_out_dp"))};
    unpadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "in_tensor_logits_indices")));
    unpadNode->outTensors = {&graph_.outTensors.at(0)};
    unpadNode->operation.reset(op);
    graph_.nodes.push_back(*unpadNode);
    ATB_SPEED_LOG_DEBUG("MtpDecoderModel AddIndicesGatherAfterLmhead calculation success");
    return atb::NO_ERROR;
}

atb::Status MtpDecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        // 仅需在graph的intensor中bind一次
        return atb::NO_ERROR;
    }

    if (param.enableDap) {
        BindDapHostTensor(this->seqLenForDap, "in_tensor_seq_len");
        BindDapHostTensor(this->tokenOffsetForDap, "in_tensor_token_offset");
        BindDapHostTensor(this->qLenForDap, "in_tensor_q_len");
        return atb::NO_ERROR;
    }

    const uint32_t tokenOffsetTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_token_offset");
    if (tokenOffsetTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tokenOffsetTensorIdx).hostData = tokenOffset.data();
    }

    const uint32_t seqLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_seq_len");
    if (seqLenTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(seqLenTensorIdx).hostData = seqLen.data();
    }
    if (param.mapping.Get(base::ATTN_CP).IsEnabled() && param.isPrefill) {
        const uint32_t seqLenCpTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_seq_len_cp");
        graph_.inTensors.at(seqLenCpTensorIdx).hostData = seqLenCp.Get().data();
    }
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() && !param.isPrefill) {
        const uint32_t seqLenSpTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_seq_len_sp");
        graph_.inTensors.at(seqLenSpTensorIdx).hostData = seqLenSp.Get().data();
        if (param.enableSpeculate) {
            const uint32_t isNeedMaskTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "is_need_mask");
            graph_.inTensors.at(isNeedMaskTensorIdx).hostData = isNeedMask.Get().data();
        }
    }
    // MTP固定走并行解码
    const uint32_t qLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_q_len");
    if (qLenTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(qLenTensorIdx).hostData = qLen.data();
    }
    if (param.enablePrefixCache) {
        uint32_t ringCurSeqLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "ring_cur_seqlen");
        if (ringCurSeqLenTensorIdx != UINT32_MAX) {
            graph_.inTensors.at(ringCurSeqLenTensorIdx).hostData = ringCurSeqlen.Get().data();
        }
        uint32_t ringCacheSeqLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "ring_cache_seqlen");
        if (ringCacheSeqLenTensorIdx != UINT32_MAX) {
            graph_.inTensors.at(ringCacheSeqLenTensorIdx).hostData = ringCacheSeqlen.Get().data();
        }
        if (param.mapping.Get(base::ATTN_CP).IsEnabled() && param.isPrefill) {
            uint32_t kvCachelenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_kv_cache_len");
            if (kvCachelenTensorIdx != UINT32_MAX) {
                graph_.inTensors.at(kvCachelenTensorIdx).hostData = kvCachelen.Get().data();
            }
        }
    }
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");

    return atb::NO_ERROR;
}
} // namespace deepseekV2
} // namespace atb_speed
