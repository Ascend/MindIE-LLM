/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "models/deepseekv2/model/decoder_model.h"
#include <vector>
#include "atb/comm.h"
#include "hccl/hccl.h"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "models/base/param/model_param.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/lmhead/hidden_state_slice.h"


namespace atb_speed {
namespace deepseekV2 {

// Weight count
constexpr uint32_t WEIGHT_COUNT_PER_LAYER = 84;
constexpr uint32_t WEIGHT_COUNT_WORD_EMBEDDINGNODE = 1;
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

constexpr uint32_t DIM1 = 1;  // DIM 1
constexpr uint32_t DIM2 = 2;  // DIM 2

HcclComm DeepseekV2ModelParam::dispatchAndCombineHcclComm = nullptr;
std::string DeepseekV2ModelParam::dispatchAndCombinecommDomain = "";

void DeepseekV2ModelParam::SetHcclComm() const
{
    if (!isPrefill && enableAllToAllMC2 && expertParallelDegree == 2) { // 2: dynamic ep level
        // Assign commDomain by rankIds and rank
        if (dispatchAndCombineHcclComm != nullptr) {
            ATB_SPEED_LOG_DEBUG("Reuse the hccl communication group for dispatch and combine.");
        } else {
            atb_speed::common::ParallelInfo moeEpParallelInfo = mapping.Get(base::MOE_EP);
            dispatchAndCombinecommDomain = GetSingleton<ExternalCommManager>().GetCommDomain(
                moeEpParallelInfo.groupId, moeEpParallelInfo.rankIds, moeEpParallelInfo.rank,
                moeEpParallelInfo.defaultBackend, moeEpParallelInfo.bufferSize, 0, false);  // 0: Default Stream Id

            dispatchAndCombineHcclComm = \
                GetSingleton<ExternalCommManager>().GetCommPtr(dispatchAndCombinecommDomain);
            ATB_SPEED_LOG_DEBUG("Create the hccl communication group for dispatch and combine.");
        }
    }
}

void DeepseekV2ModelParam::AddParamJsonMLA(const std::string &param)
{
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    if (paramJson.contains("qLoraRank")) {
        qLoraRank = atb_speed::base::FetchJsonParam<int>(paramJson, "qLoraRank");
    }
    if (paramJson.contains("kvLoraRank")) {
        kvLoraRank = atb_speed::base::FetchJsonParam<int>(paramJson, "kvLoraRank");
    }
    if (paramJson.contains("qkNopeHeadDim")) {
        qkNopeHeadDim = atb_speed::base::FetchJsonParam<int>(paramJson, "qkNopeHeadDim");
    }
    if (paramJson.contains("qkRopeHeadDim")) {
        qkRopeHeadDim = atb_speed::base::FetchJsonParam<int>(paramJson, "qkRopeHeadDim");
    }
    if (paramJson.contains("softmaxScale")) {
        softmaxScale = atb_speed::base::FetchJsonParam<float>(paramJson, "softmaxScale");
    }
    if (paramJson.contains("enableMlaPreprocess")) {
        enableMlaPreprocess = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableMlaPreprocess");
    }
    if (paramJson.contains("enableSpeculate")) {
        enableSpeculate = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSpeculate");
    }
    if (paramJson.contains("maskfree")) {
        maskfree = atb_speed::base::FetchJsonParam<bool>(paramJson, "maskfree");
    }
    for (auto item : paramJson["attnLinearQuantType"]) {
        attnLinearQuantType.push_back(
            atb_speed::base::FetchJsonParam<std::vector<int>>(item, "attnLinearQuantType", true));
    }
    CheckLinearParamsSufficient(attnLinearQuantType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);
    for (auto item : paramJson["attnLinearTransposeType"]) {
        attnLinearTransposeType.push_back(
            atb_speed::base::FetchJsonParam<std::vector<int>>(item, "attnLinearTransposeType", true));
    }
    CheckLinearParamsSufficient(attnLinearTransposeType, numHiddenLayers, ATTN_LINEAR_TYPE_LENGTH);
    if (paramJson.contains("isNzCache")) {
        isNzCache = atb_speed::base::FetchJsonParam<bool>(paramJson, "isNzCache");
    }
    if (paramJson.contains("enablePrefixCache")) {
        enablePrefixCache = atb_speed::base::FetchJsonParam<bool>(paramJson, "enablePrefixCache");
    }
    if (paramJson.contains("enableMlaPrefetch")) {
        enableMlaPrefetch = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableMlaPrefetch");
    }
}

void DeepseekV2ModelParam::AddParamJsonMoE(const std::string &param)
{
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    if (paramJson.contains("enableSwiGLUQuantForSharedExperts")) {
        enableSwiGLUQuantForSharedExperts = \
            atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSwiGLUQuantForSharedExperts");
    }
    if (paramJson.contains("enableAtlasGMMFused")) {
        enableAtlasGMMFused = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableAtlasGMMFused");
    }
    if (paramJson.contains("enableAllToAllMC2")) {
        enableAllToAllMC2 = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableAllToAllMC2");
    }
    if (paramJson.contains("enableLoadBalance")) {
        enableLoadBalance = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLoadBalance");
    }
    if (paramJson.contains("enableExtraOprojTp")) {
        enableExtraOprojTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableExtraOprojTp");
    }
    if (paramJson.contains("enableEPWB")) {
        enableEPWB = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableEPWB");
    }
    if (paramJson.contains("numOfRedundantExpert")) {
        numOfRedundantExpert = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfRedundantExpert");
    }
    AddParamJsonMoEGate(param);
}

void DeepseekV2ModelParam::AddParamJsonMoEGate(const std::string &param)
{
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    if (paramJson.contains("numOfGroups")) {
        numOfGroups = atb_speed::base::FetchJsonParam<int>(paramJson, "numOfGroups");
    }
    if (paramJson.contains("moePackQuantType")) {
        this->moePackQuantType = atb_speed::base::FetchJsonParam<int>(paramJson, "moePackQuantType");
    }
    if (paramJson.contains("enableATBGateMatmul")) {
        enableATBGateMatmul = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableATBGateMatmul");
    }
    if (paramJson.contains("routedScalingFactor")) {
        routedScalingFactor = atb_speed::base::FetchJsonParam<float>(paramJson, "routedScalingFactor");
    }
    if (paramJson.contains("routingMethod")) {
        routingMethod = atb_speed::base::FetchJsonParam<std::string>(paramJson, "routingMethod");
    }
    if (paramJson.contains("processLogits")) {
        processLogits = atb_speed::base::FetchJsonParam<std::string>(paramJson, "processLogits");
    }
    if (paramJson.contains("scaledTopk")) {
        scaledTopk = atb_speed::base::FetchJsonParam<int>(paramJson, "scaledTopk");
    }
    if (paramJson.contains("enableInitRoutingCutoff")) {
        enableInitRoutingCutoff = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableInitRoutingCutoff");
    }
    for (auto item : paramJson["topkGroups"]) {
        topkGroups.push_back(atb_speed::base::FetchJsonParam<int>(item, "topkGroups", true));
    }
    if (paramJson.contains("enableFusedTopk")) {
        enableFusedTopk = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableFusedTopk");
    }
    if (paramJson.contains("enableExpertCumSumOutput")) {
        enableExpertCumSumOutput = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableExpertCumSumOutput");
    }
    if (paramJson.contains("enableTopkOutput")) {
        enableTopkOutput = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableTopkOutput");
    }
    if (paramJson.contains("numDanglingSharedExperts")) {
        numDanglingSharedExperts = paramJson.at("numDanglingSharedExperts").get<int64_t>();
    }
    if (paramJson.contains("numDanglingSharedExperts")) {
        numDanglingSharedExperts = atb_speed::base::FetchJsonParam<int64_t>(paramJson, "numDanglingSharedExperts");
    }
    if (paramJson.contains("mixSharedRouting")) {
        mixSharedRouting = atb_speed::base::FetchJsonParam<bool>(paramJson, "mixSharedRouting");
    }
}

void DeepseekV2ModelParam::AddParamJsonH3P(const std::string &param)
{
    // Prefill H3P, Hierarchical & Heterogeneous & Hybrid Parallel
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    if (paramJson.contains("enableQkvdownDp")) {
        enableQkvdownDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableQkvdownDp");
    }
    if (paramJson.contains("enableSharedExpertDp")) {
        enableSharedExpertDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSharedExpertDp");
    }
    if (paramJson.contains("enableGatingDp")) {
        enableGatingDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableGatingDp");
    }
    if (paramJson.contains("enableSharedExpertOverlap")) {
        enableSharedExpertOverlap = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableSharedExpertOverlap");
    }
    if (paramJson.contains("enableLcocTp")) {
        enableLcocTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLcocTp");
    }
    if (paramJson.contains("enableLcocAll2All")) {
        enableLcocAll2All = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableLcocAll2All");
    }
    if (paramJson.contains("enableFusedMLA")) {
        enableFusedMLA = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableFusedMLA");
    }
}

void DeepseekV2ModelParam::CheckParam()
{
    if (this->enableDap && this->enableExpertCumSumOutput) {
        throw std::runtime_error("'enableDap' and 'enableExpertCumSumOutput' are incompatible, "
            "do not enable them at the same time, please check.");
    }
    if (this->enableDap && this->lmHeadLocalTp) {
        throw std::runtime_error("'enableDap' and 'lmHeadLocalTp' are incompatible, "
            "do not enable them at the same time, please check.");
    }
}

void DeepseekV2ModelParam::AddLogInfo()
{
    ATB_SPEED_LOG_DEBUG("DecoderModel param" << ", isFA:" << isFA << ", isPrefill:" << isPrefill
        << ", isBF16:" << isBF16 << ", isEmbeddingParallel: " << isEmbeddingParallel << ", isLmHeadParallel: "
        << isLmHeadParallel << ", enableSwiGLU: " << enableSwiGLU << ", enableLcoc:" << enableLcoc
        << ", lmHeadTransposeType: " << lmHeadTransposeType << ", normEps:" << normEps
        << ", numAttentionHeadsPerRank:" << numAttentionHeadsPerRank << ", hiddenSizePerAttentionHead:"
        << hiddenSizePerAttentionHead<< ", numHiddenLayers:" << numHiddenLayers << ", numKeyValueHeadsPerRank:"
        << numKeyValueHeadsPerRank << ", rank:" << rank << ", worldSize:" << worldSize << ", backend:" << backend
        << ", rankTableFile:" << rankTableFile << ", enableFusedTopk" << enableFusedTopk
        << ", numOfExperts:" << numOfExperts << ", numOfDeviceExperts:" << numOfDeviceExperts
        << ", expertParallelDegree:" << expertParallelDegree << ", deviceExpert:" << deviceExpert
        << ", maskStartIdx:" << maskStartIdx << ", numOfSelectedExperts:" << numOfSelectedExperts << ", topkGroups:"
        << topkGroups << ", processLogits:" << processLogits << ", routedScalingFactor" << routedScalingFactor
        << "firstKDenseReplace: " << firstKDenseReplace << ", numOfSharedExperts" << numOfSharedExperts
        << "routingMethod: " << routingMethod << "enableAllToAllMC2: " << enableAllToAllMC2
        << "enableExtraOprojTp: " << enableExtraOprojTp << "enableQkvdownDp: " << enableQkvdownDp
        << "finalStateOut" << finalStateOut << "enableSharedExpertDp: " << enableSharedExpertDp
        << "enableGatingDp: " << enableGatingDp << "enableSharedExpertOverlap: " << enableSharedExpertOverlap
        << "enableLcocTp: " << enableLcocTp << ", enablePrefixCache: " << enablePrefixCache
    );
}

void DeepseekV2ModelParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = atb_speed::base::StringToJson(param);
    ParseParam(paramJson);
    if (rank > worldSize) {
        std::stringstream ss;
        ss << "worldSize must be greater or equal to 0, please check." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    AddParamJsonMLA(param);
    AddParamJsonMoE(param);
    AddParamJsonH3P(param);
    if (paramJson.contains("hasP2DWeight")) {
        hasP2DWeight = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasP2DWeight");
    }
    if (paramJson.contains("enableGatherPreNorm")) {
        enableGatherPreNorm = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableGatherPreNorm");
    }
    if (paramJson.contains("finalStateOut")) {
        finalStateOut = atb_speed::base::FetchJsonParam<bool>(paramJson, "finalStateOut");
    }
    if (paramJson.contains("enableInfNan")) {
        enableInfNan = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableInfNan");
    }
    if (paramJson.contains("enableDistributed")) {
        enableDistributed = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableDistributed");
    }

    ParseLayerwiseDisaggregatedParam(paramJson);
    
    kvcacheQuantLayers.clear();
    if (paramJson.contains("kvcacheQuantLayers")) {
        for (auto item : paramJson["kvcacheQuantLayers"]) {
            kvcacheQuantLayers.push_back(item.get<bool>());
        }
    }
    if (paramJson.contains("enableDenseTp")) {
        enableDenseTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "enableDenseTp");
    }
    SetHcclComm();
    AddLogInfo();
    CheckMixParallelValid();
}

void DeepseekV2ModelParam::ParseLayerwiseDisaggregatedParam(nlohmann::json &paramJson)
{
    if (paramJson.contains("layerwiseDisaggregated")) {
        layerwiseDisaggregated = atb_speed::base::FetchJsonParam<bool>(paramJson, "layerwiseDisaggregated");
    }
    if (layerwiseDisaggregated) {
        if (paramJson.contains("skipWordEmbedding")) {
            skipWordEmbedding = atb_speed::base::FetchJsonParam<bool>(paramJson, "skipWordEmbedding");
        }
        if (paramJson.contains("layerwiseMode")) {
            layerwiseMode = atb_speed::base::FetchJsonParam<int32_t>(paramJson, "layerwiseMode");
            if (layerwiseMode == LWD_CLOUD_MIDDLE || layerwiseMode == LWD_EDGE_LAST) {
                skipWordEmbedding = true;
            }
        }
        if (paramJson.contains("hiddenSize")) {
            hiddenSize = atb_speed::base::FetchJsonParam<int32_t>(paramJson, "hiddenSize");
        }
        if (paramJson.contains("startLayerId")) {
            startLayerId = atb_speed::base::FetchJsonParam<int32_t>(paramJson, "startLayerId");
        }
        if (paramJson.contains("endLayerId")) {
            endLayerId = atb_speed::base::FetchJsonParam<int32_t>(paramJson, "endLayerId");
        }
        if (paramJson.contains("cloudLastLayerId")) {
            cloudLastLayerId = atb_speed::base::FetchJsonParam<int32_t>(paramJson, "cloudLastLayerId");
        }
        
        this->isInternalLayer = this->layerwiseDisaggregated
            && (this->layerwiseMode == LWD_EDGE_FIRST || this->layerwiseMode == LWD_CLOUD_MIDDLE);
    }
}

void DeepseekV2ModelParam::CheckMixParallelValid() const
{
    // check MLA
    if (mapping.Get(base::ATTN_O_PROJ_TP).IsEnabled() && !mapping.Get(base::ATTN_DP).IsEnabled()) {
        std::stringstream ss;
        ss << "The attention extra O proj TP should work with the attention DP, "
           << "and the attention extra O proj TP is enabled but the attention DP is disabled. "
           << "Please enable DP when using the attention extra O proj TP." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    if (mapping.Get(base::ATTN_O_PROJ_TP).IsEnabled() && mapping.Get(base::ATTN_TP).IsEnabled()) {
        std::stringstream ss;
        ss << "The attention extra O proj TP conflicts with the attention TP. "
           << "Make sure to disable one of them." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    if (mapping.Get(base::ATTN_O_PROJ_TP).IsEnabled() && expertParallelDegree != 2) {  // 2: dynamic ep
        std::stringstream ss;
        ss << "The attention extra O proj TP should work with expertParallelDegree 2. "
           << "Make sure to disable one of them." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    // check MoE
    if (enableAllToAllMC2 && expertParallelDegree != 2) { // 2: dynamic ep
        std::stringstream ss;
        ss << "The expertParallelDegree is not 2. "
           << "The MoE distribute dispatch/combine operation should work with expertParallelDegree 2."
           << "Please set expertParallelDegree or ep_level as 2 when using dispatch/combine operation." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    if (enableAllToAllMC2 && mapping.Get(base::MOE_TP).IsEnabled() && atb_speed::common::IsA2()) {
        std::stringstream ss;
        ss << "The MoE distribute dispatch/combine operation does not support MOE TP on this device." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    // check LM head
    if (lmHeadLocalTp && !enableDpOut) {
        std::stringstream ss;
        ss << "The lmHeadLocalTp should work with enableDpOut. "
           << "Please set enableDpOut = true when using lmHeadLocalTp." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
}

DecoderModel::DecoderModel(const std::string &param) : atb_speed::moe::MoeDecoderModel(param)
{
    this->param.FromString(param);
    modelName_ += this->param.isPrefill ? "_Prefill" : "_Decoder";
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelInTensorCandidates = {
        {"default", {
            "in_tensor_input_ids", "in_tensor_position_ids", "in_tensor_cos_table", "in_tensor_sin_table",
            "in_tensor_attention_mask", "in_tensor_block_tables", "in_tensor_slots", "in_tensor_kvcache_idx",
            "in_final_state_model",
            "in_tensor_token_offset", "in_tensor_place_holder", "in_tensor_seq_len", "in_tensor_logits_indices",
            "in_expert_array_model", "in_expert_group_model", "in_one_hot_model", "in_zero_hot_model",
            "in_tensor_q_len"}},
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
        {"force_load_balance", {
            "in_fake_topk_model"
        }},
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

void DecoderModel::ConstructInTensorMap()
{
    auto deepseekV2ModelInTensorCandidates = GetDeepseekV2ModelInTensorCandidates();
    if (this->param.layerwiseDisaggregated) {
        if (this->param.skipWordEmbedding) {
            deepseekV2ModelInTensorCandidates["default"].at(0) = "input_embedding";
        }
    }
    atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "default", this->inTensorMap);
    if (param.enableLoadBalance) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "force_load_balance", this->inTensorMap);
    }
    atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "parallel_input", this->inTensorMap);
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
    // new inTensors please add here before
    if (param.enableEPWB) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "epwb", this->inTensorMap);
    }
    if (param.mixSharedRouting) {
        atb_speed::common::AssignTensorIdx(deepseekV2ModelInTensorCandidates, "mix_shared_routing", this->inTensorMap);
    }
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelInternalTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelInternalTensorCandidates = {
        {"default", {
            "internal_tensor_hidden_states", "internal_tensor_cos_emb", "internal_tensor_sin_emb"}},
        {"last_layer", {
            "internal_tensor_last_layer"}},
        {"enable_lm_head_local_tp_out", {"internal_lmhead_out", "internal_hidden_states_out_dp"}},
        {"eplb_data_collection", {"internal_tensor_gmm_cumsum_list"}},
        {"qkvdown_dp", {"internal_tensor_hidden_states_slice"}}
    };
    return deepseekV2ModelInternalTensorCandidates;
}

void DecoderModel::ConstructInternalTensorMap()
{
    auto deepseekV2ModelInternalTensorCandidates = GetDeepseekV2ModelInternalTensorCandidates();
    if (this->param.layerwiseDisaggregated) {
        if (this->param.skipWordEmbedding && this->param.numHiddenLayers == 1 && this->param.isInternalLayer) {
            deepseekV2ModelInternalTensorCandidates["default"] = {
                "internal_tensor_cos_emb", "internal_tensor_sin_emb"
            };
        }
        if (param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) {
            if (this->param.skipWordEmbedding && this->param.numHiddenLayers == 1 &&
                    this->param.layerwiseMode == LWD_EDGE_LAST) {
                deepseekV2ModelInternalTensorCandidates["default"] = {
                    "internal_tensor_cos_emb", "internal_tensor_sin_emb"
                };
            }
        }
    }
    atb_speed::common::AssignTensorIdx(
        deepseekV2ModelInternalTensorCandidates, "default", this->internalTensorMap);
    if (param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) {
        if (!this->param.layerwiseDisaggregated || this->param.layerwiseMode == LWD_EDGE_LAST) {
            atb_speed::common::AssignTensorIdx(
                deepseekV2ModelInternalTensorCandidates, "last_layer", this->internalTensorMap);
        }
    }
    if (param.enableDpOut && param.lmHeadLocalTp) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelInternalTensorCandidates, "enable_lm_head_local_tp_out", this->internalTensorMap);
    }
    if (param.enableQkvdownDp) {
        if (!this->param.layerwiseDisaggregated) {
            atb_speed::common::AssignTensorIdx(
                deepseekV2ModelInternalTensorCandidates, "qkvdown_dp", this->internalTensorMap);
        } else {
            if (this->param.layerwiseMode == LWD_EDGE_LAST ||
                (this->param.startLayerId >= this->param.firstKDenseReplace &&
                    this->param.endLayerId - this->param.startLayerId >= 2)) { // moe层且中间层数大于2
                atb_speed::common::AssignTensorIdx(deepseekV2ModelInternalTensorCandidates,
                    "qkvdown_dp", this->internalTensorMap);
            }
        }
    }
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2ModelOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2ModelOutTensorCandidates = {
        {"default", {"logits"}},
        {"final_hidden_states", {"final_hidden_states"}},
    };
    return deepseekV2ModelOutTensorCandidates;
}

void DecoderModel::ConstructOutTensorMap()
{
    this->outTensorMap.clear();
    auto deepseekV2ModelOutTensorCandidates = GetDeepseekV2ModelOutTensorCandidates();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(
        deepseekV2ModelOutTensorCandidates, "default", this->outTensorMap);
    
    if (param.finalStateOut) {
        atb_speed::common::AssignTensorIdx(
            deepseekV2ModelOutTensorCandidates, "final_hidden_states", this->outTensorMap);
    }
    uint32_t currentTensorIdx = this->outTensorMap.size();
    uint32_t moeLayerNum = param.numHiddenLayers - param.firstKDenseReplace;
    if (param.layerwiseDisaggregated) {
        if (param.endLayerId < param.firstKDenseReplace) {
            moeLayerNum = 0;
        } else if (param.endLayerId >= param.firstKDenseReplace && param.startLayerId < param.firstKDenseReplace) {
            moeLayerNum = param.endLayerId + 1 - param.firstKDenseReplace;
        } else {
            moeLayerNum = param.endLayerId - param.startLayerId;
        }
    }
    for (uint32_t i = 0; i < moeLayerNum; i++) {
        if (param.enableExpertCumSumOutput && param.enableTopkOutput) {
            // 每层都采专家热度和topk数据，专家热度和topk数据整体呈交错排列
            this->outTensorMap["layer_" + std::to_string(i) + "_activation_count_per_expert"] = currentTensorIdx;
            currentTensorIdx++;
            this->outTensorMap["layer_" + std::to_string(i) + "_activation_topk"] = currentTensorIdx;
            currentTensorIdx++;
        } else if (param.enableExpertCumSumOutput) {
            this->outTensorMap["layer_" + std::to_string(i) + "_activation_count_per_expert"] = currentTensorIdx;
            currentTensorIdx++;
        } else if (param.enableTopkOutput) {
            this->outTensorMap["layer_" + std::to_string(i) + "_activation_topk"] = currentTensorIdx;
            currentTensorIdx++;
        }
    }
}

atb::TensorDesc DecoderModel::GetLogitsDesc(
    const std::vector<atb::TensorDesc> &inTensorDescs, uint32_t logitsIndicesIdx)
{
    atb::TensorDesc logitsDesc;
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    // FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSisze]
    logitsDesc.dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
    logitsDesc.format = graph_.weightTensors.at(0).desc.format;
    if (!this->param.layerwiseDisaggregated) {
        logitsDesc.shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    } else {
        if (this->param.layerwiseMode == LWD_EDGE_LAST) {
            logitsDesc.shape.dimNum = inTensorDescs.at(0).shape.dimNum;
        } else {
            logitsDesc.shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
        }
    }

    if (param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) {
        logitsDesc.shape.dims[0] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
    } else {
        logitsDesc.shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        if (param.isFA) {  // unpadInputs = false
            logitsDesc.shape.dims[1] = \
                param.isPrefill ? inTensorDescs.at(logitsIndicesIdx).shape.dims[0] : 1;
        } else {  // unpadInputs = true
            if (param.isPrefill) {
                logitsDesc.shape.dims[0] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
            }
        }
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

atb::TensorDesc DecoderModel::GetLWDLogitsDesc(
    const std::vector<atb::TensorDesc> &inTensorDescs)
{
    atb::TensorDesc logitsDesc;
    logitsDesc.dtype = this->param.isBF16 ? \
     aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
    logitsDesc.format = graph_.weightTensors.at(0).desc.format;
    if (this->param.layerwiseMode == LWD_EDGE_FIRST) {
        logitsDesc.shape.dimNum = inTensorDescs.at(0).shape.dimNum + 1;
    } else {
        logitsDesc.shape.dimNum = inTensorDescs.at(0).shape.dimNum;
    }
    logitsDesc.shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    logitsDesc.shape.dims[1] = this->param.hiddenSize;
    if (param.enableQkvdownDp && param.endLayerId > param.firstKDenseReplace &&
            param.startLayerId <= param.firstKDenseReplace) {
        logitsDesc.shape.dims[0] = inTensorDescs.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ffn_padding_idx_model")
        ).shape.dims[0];
        // 2: dynamic ep level
        if (param.expertParallelDegree != 2 && param.mapping.Get(base::MLP_TP).rankIds.size() != 0) {
            logitsDesc.shape.dims[0] /= param.mapping.Get(base::MLP_TP).rankIds.size();
        }
    }
    if (param.enableQkvdownDp && param.endLayerId == param.cloudLastLayerId + 1) {
        logitsDesc = inTensorDescs.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_final_state_model"));
    }

    return logitsDesc;
}

atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    ATB_SPEED_LOG_DEBUG("Enter DecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    uint32_t outTensorIdx = 0;
    uint32_t logitsIndicesIdx = param.enableDpOut && param.lmHeadLocalTp ? \
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_post_lmhead_unpadding_indices") : \
        atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices");
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(0).shape.dimNum + 1);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);
    if (!this->param.layerwiseDisaggregated) {
        outTensorDescs.at(outTensorIdx) = GetLogitsDesc(inTensorDescs, logitsIndicesIdx);
    } else {
        if (this->param.layerwiseMode == LWD_EDGE_FIRST || this->param.layerwiseMode == LWD_CLOUD_MIDDLE) {
            outTensorDescs.at(outTensorIdx) = GetLWDLogitsDesc(inTensorDescs);
        } else {
            outTensorDescs.at(outTensorIdx) = GetLogitsDesc(inTensorDescs, logitsIndicesIdx);
        }
    }
    outTensorIdx++;

    if (param.finalStateOut) {
        uint32_t hiddenIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_final_state_model");
        outTensorDescs.at(outTensorIdx) = inTensorDescs.at(hiddenIndicesIdx);
        outTensorIdx++;
    }

    if (param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();

        logitsIndicesIdx = param.enableDpOut && param.lmHeadLocalTp ? \
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_post_lmhead_unpadding_indices" + suffix) : \
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_logits_indices" + suffix);
        outTensorDescs.at(outTensorIdx) = GetLogitsDesc(inTensorDescs, logitsIndicesIdx);
        outTensorIdx++;

        if (param.finalStateOut) {
            uint32_t hiddenIndicesIdx = atb_speed::common::GetTensorIdx(
                this->inTensorMap, "in_final_state_model" + suffix);
            outTensorDescs.at(outTensorIdx) = inTensorDescs.at(hiddenIndicesIdx);
            outTensorIdx++;
        }
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }

    uint32_t moeLayernum = param.numHiddenLayers - static_cast<uint32_t>(param.firstKDenseReplace);
    if (param.layerwiseDisaggregated) {
        if (param.endLayerId < param.firstKDenseReplace) {
            moeLayernum = 0;
        } else if (param.endLayerId >= param.firstKDenseReplace && param.startLayerId < param.firstKDenseReplace) {
            moeLayernum = param.endLayerId + 1 - static_cast<uint32_t>(param.firstKDenseReplace);
        } else {
            moeLayernum = param.endLayerId - param.startLayerId;
        }
    }

    for (uint32_t i = 0; i < moeLayernum; i++) {
        if (param.enableExpertCumSumOutput && param.enableTopkOutput) {  // 不支持和DAP同时开启
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx, param.numOfDeviceExperts);
            outTensorIdx++;
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx,
                param.numOfSelectedExperts.at(0), false);
            outTensorIdx++;
        } else if (param.enableExpertCumSumOutput) {  // 添加热点数据
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx, param.numOfDeviceExperts);
            outTensorIdx++;
        } else if (param.enableTopkOutput) {  // 添加topk张量描述
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, outTensorIdx,
                param.numOfSelectedExperts.at(0), false);
            outTensorIdx++;
        }
    }
    return atb::NO_ERROR;
}

uint32_t DecoderModel::CalcWeightTensorSize()
{
    weightCountPerLayer = WEIGHT_COUNT_PER_LAYER;
    if (param.enableFA3) {
        weightCountPerLayer += 5; // 5: FA3 多5个inTensorensor
    }
    int weightTensorSize = 0;
    if (!this->param.layerwiseDisaggregated) {
        if (param.hasP2DWeight) {
            weightTensorSize =
                WEIGHT_COUNT_WORD_EMBEDDINGNODE + CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
                CheckIntMulOverFlow(DECODER_WEIGHT_COUNT_PER_LAYER, param.numHiddenLayers - param.firstKDenseReplace) +
                WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
        } else {
            weightTensorSize =
                WEIGHT_COUNT_WORD_EMBEDDINGNODE + CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
                WEIGHT_COUNT_POST_NORM + WEIGHT_COUNT_LM_HEAD;
        }
    } else {
        weightCountWordEmbedding = WEIGHT_COUNT_WORD_EMBEDDINGNODE;
        weightCountLmHead = WEIGHT_COUNT_LM_HEAD;
        weightCountFinalNorm = WEIGHT_COUNT_POST_NORM;
        if (this->param.skipWordEmbedding) {
            weightCountWordEmbedding = 0;
        }
        if (this->param.layerwiseMode == LWD_EDGE_FIRST || this->param.layerwiseMode == LWD_CLOUD_MIDDLE) {
            weightCountFinalNorm = 0;
            weightCountLmHead = 0;
        }
        uint32_t moeLayerNum = 0;
        if (param.hasP2DWeight) {
            if (param.endLayerId < param.firstKDenseReplace) {
                moeLayerNum = 0;
            } else if (param.endLayerId >= param.firstKDenseReplace && param.startLayerId < param.firstKDenseReplace) {
                moeLayerNum = param.endLayerId + 1 - param.firstKDenseReplace;
            } else {
                moeLayerNum = param.endLayerId - param.startLayerId;
            }
            weightTensorSize =
                weightCountWordEmbedding + CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
                CheckIntMulOverFlow(DECODER_WEIGHT_COUNT_PER_LAYER, moeLayerNum) +
                weightCountFinalNorm + weightCountLmHead;
            } else {
            weightTensorSize =
                weightCountWordEmbedding + CheckIntMulOverFlow(weightCountPerLayer, param.numHiddenLayers) +
                weightCountFinalNorm + weightCountLmHead;
            }
    }
    return weightTensorSize;
}

atb::Status DecoderModel::AddNodesBeforeLayer()
{
    if (!this->param.layerwiseDisaggregated || !this->param.skipWordEmbedding) {
        CHECK_OPERATION_STATUS_RETURN(AddWordEmbedding());
    }
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddNodesAfterLayer()
{
    if (param.lmHeadLocalTp && param.enableDistributed && param.finalStateOut) {
        CHECK_OPERATION_STATUS_RETURN(AddSliceFinalStateOut());
        CHECK_OPERATION_STATUS_RETURN(AddGatherFinalStateOut());
    }
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    if (param.enableDpOut && param.lmHeadLocalTp) {
        CHECK_OPERATION_STATUS_RETURN(AddGatherAfterLmhead());
    }
    if (param.attnOprojPrefetch || param.enableMlaPrefetch) {
        CHECK_OPERATION_STATUS_RETURN(AddCmoSync());
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddWordEmbedding()
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

atb::Status DecoderModel::AddPositionalEmbedding()
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

void SetMlaParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param, int64_t layerId)
{
    layerParam.qLoraRank = param.qLoraRank;
    layerParam.headNum = param.headNum;
    layerParam.qkNopeHeadDim = param.qkNopeHeadDim;
    layerParam.qkRopeHeadDim = param.qkRopeHeadDim;
    layerParam.kvLoraRank = param.kvLoraRank;
    layerParam.softmaxScale = param.softmaxScale;
    layerParam.enableMlaPreprocess = param.enableMlaPreprocess;
    layerParam.isNzCache = param.isNzCache;
    layerParam.enableFA3 = param.enableFA3;
    layerParam.enablePrefixCache = param.enablePrefixCache;
    if (!param.layerwiseDisaggregated) {
        if (layerId < int64_t(param.kvcacheQuantLayers.size())) {
            layerParam.enableKvQuantLayer = param.kvcacheQuantLayers.at(layerId);
        }
    } else {
        if (layerId - param.startLayerId < int64_t(param.kvcacheQuantLayers.size())) {
            layerParam.enableKvQuantLayer = param.kvcacheQuantLayers.at(layerId - param.startLayerId);
        }
    }
    layerParam.enableFusedMLA = param.enableFusedMLA && param.isPrefill && !param.enablePrefixCache;
}

void SetParallelParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param)
{
    layerParam.enableAllToAllMC2 = param.enableAllToAllMC2;
    layerParam.enableGatherPreNorm = param.enableGatherPreNorm;
    layerParam.enableExtraOprojTp = param.enableExtraOprojTp;
    layerParam.enableDenseTp = param.enableDenseTp;
    layerParam.mapping = param.mapping;
    layerParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
}

void SetMoeParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param, int64_t layerId)
{
    layerParam.hasSharedExpert = param.hasSharedExpert;
    layerParam.hasSharedExpertGate = param.hasSharedExpertGate;
    layerParam.processLogits = param.processLogits;
    layerParam.routedScalingFactor = param.routedScalingFactor;
    layerParam.numOfSelectedExperts = param.numOfSelectedExperts;
    layerParam.expertParallelDegree = param.expertParallelDegree;
    layerParam.deviceExpert = param.deviceExpert;
    layerParam.numOfExperts = param.numOfExperts;
    layerParam.numOfDeviceExperts = param.numOfDeviceExperts;
    layerParam.maskStartIdx = param.maskStartIdx;
    layerParam.firstKDenseReplace = param.firstKDenseReplace;
    layerParam.numOfSharedExperts = param.numOfSharedExperts;
    layerParam.routingMethod = param.routingMethod;
    layerParam.numOfGroups = param.numOfGroups;
    layerParam.scaledTopk = param.scaledTopk;
    layerParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    layerParam.topkGroups = param.topkGroups;
    layerParam.quantGroupSize = param.quantGroupSize;
    layerParam.hasP2DWeight = param.hasP2DWeight;
    layerParam.enableInitQuant = param.enableInitQuant;
    layerParam.enableSwigluQuant = param.enableSwigluQuant;
    layerParam.enableFusedTopk = param.enableFusedTopk;
    layerParam.enableCVOverlap = param.enableCVOverlap;
    layerParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    layerParam.enableTopkOutput = param.enableTopkOutput;
    if (param.expertParallelDegree == 2) { // 2: dynamic ep level
        layerParam.isDynamicEp = true;
    }
    if (layerId < param.firstKDenseReplace) {
        layerParam.isDenseLayer = true;
    }

    SetLayerwiseDisaggregatedParam(layerParam, param, layerId);
    layerParam.enableLoadBalance = param.enableLoadBalance;
    layerParam.enableEPWB = param.enableEPWB;
    layerParam.numOfRedundantExpert = param.numOfRedundantExpert;
    layerParam.numDanglingSharedExperts = param.numDanglingSharedExperts;
    layerParam.enableInfNan = param.enableInfNan;
    layerParam.enableATBGateMatmul = param.enableATBGateMatmul;
    layerParam.enableMlaPrefetch = param.enableMlaPrefetch &&
        param.enableAllToAllMC2 && !layerParam.isLastLayer && !layerParam.isDenseLayer;
    layerParam.dispatchAndCombineHcclComm = param.dispatchAndCombineHcclComm;
    layerParam.dispatchAndCombinecommDomain = param.dispatchAndCombinecommDomain;
    layerParam.enableDispatchCombineV2 = param.enableDispatchCombineV2;
    layerParam.enableOutLcocTp = !layerParam.isDenseLayer && param.enableLcocTp && param.isPrefill;
    layerParam.enablePreprocessLcocTp = layerId > layerParam.firstKDenseReplace &&
        param.enableLcocTp && param.enableQkvdownDp;

    layerParam.enableLcocAll2All = param.enableLcocAll2All;

    layerParam.mixSharedRouting = param.mixSharedRouting;
}

void SetLayerwiseDisaggregatedParam(DecoderLayerParam &layerParam, const DeepseekV2ModelParam &param, int64_t layerId)
{
    if (!param.layerwiseDisaggregated) {
        if (layerId == param.numHiddenLayers - 1) {
            layerParam.isLastLayer = true;
        }
    } else {
        if ((layerId - param.startLayerId == param.numHiddenLayers - 1) && (param.layerwiseMode == LWD_EDGE_LAST)) {
            layerParam.isLastLayer = true;
        }
    }
}

void DecoderModel::SetLaywiseDisaggregatedQuantParam(DecoderLayerParam &layerParam, int64_t layerId)
{
    layerParam.packQuantType = param.packQuantType[layerId - param.startLayerId];
    layerParam.attnLinearQuantType = param.attnLinearQuantType[layerId - param.startLayerId];
    layerParam.mlpLinearQuantType = param.mlpLinearQuantType[layerId - param.startLayerId];
    layerParam.moeLinearQuantType = param.moeLinearQuantType[layerId - param.startLayerId];
    layerParam.attnLinearTransposeType = param.attnLinearTransposeType[layerId - param.startLayerId];
    layerParam.mlpLinearTransposeType = param.mlpLinearTransposeType[layerId - param.startLayerId];
    layerParam.moeLinearTransposeType = param.moeLinearTransposeType[layerId - param.startLayerId];
    layerParam.isCloudLastLayer = param.enableQkvdownDp && layerId == param.cloudLastLayerId;
}

void DecoderModel::SetLayerParam(DecoderLayerParam &layerParam, int64_t layerId)
{
    layerParam.isFA = param.isFA;
    layerParam.isPrefill = param.isPrefill;
    layerParam.isBF16 = param.isBF16;
    layerParam.enableSwiGLU = param.enableSwiGLU;
    layerParam.enableSwiGLUQuantForSharedExperts = param.enableSwiGLUQuantForSharedExperts;
    layerParam.enableLcoc = param.enableLcoc;
    if (!param.layerwiseDisaggregated) {
        layerParam.packQuantType = param.packQuantType[layerId];
        layerParam.attnLinearQuantType = param.attnLinearQuantType[layerId];
        layerParam.mlpLinearQuantType = param.mlpLinearQuantType[layerId];
        layerParam.moeLinearQuantType = param.moeLinearQuantType[layerId];
        layerParam.attnLinearTransposeType = param.attnLinearTransposeType[layerId];
        layerParam.mlpLinearTransposeType = param.mlpLinearTransposeType[layerId];
        layerParam.moeLinearTransposeType = param.moeLinearTransposeType[layerId];
    } else {
        SetLaywiseDisaggregatedQuantParam(layerParam, layerId);
    }
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
    layerParam.numHiddenLayers = param.numHiddenLayers;
    layerParam.enableDpOut = param.enableDpOut;
    layerParam.lmHeadLocalTp = param.lmHeadLocalTp;
    layerParam.enableSpeculate = param.enableSpeculate;
    layerParam.enableQkvdownDp = param.enableQkvdownDp && \
        layerId >= param.firstKDenseReplace;  // h3p qkvdown dp for moe
    layerParam.enableSharedExpertDp = param.enableSharedExpertDp && layerId >= param.firstKDenseReplace;
    layerParam.enableGatingDp = param.enableGatingDp && layerId >= param.firstKDenseReplace;
    layerParam.enableSharedExpertOverlap = param.enableSharedExpertOverlap && layerParam.enableSharedExpertDp;
    layerParam.maskfree = param.maskfree;
    layerParam.enableModelConfuscation = this->param.enableModelConfuscation;
    layerParam.modelConfuscationFd = this->param.modelConfuscationFd;
    if (layerId != 1) { layerParam.enableModelConfuscation = false; }

    SetMlaParam(layerParam, param, layerId);
    SetMoeParam(layerParam, param, layerId);
    SetParallelParam(layerParam, param);
    layerParam.moePackQuantType = param.moePackQuantType;
    layerParam.attnOprojPrefetch = param.attnOprojPrefetch;
}

std::string DecoderModel::GetLayerOutName(uint32_t layerId)
{
    std::string layerOutName = "internal_tensor_hidden_states";
    if (!this->param.layerwiseDisaggregated) {
        layerOutName = (param.mapping.Get(base::ATTN_DP).IsEnabled() || \
        param.mapping.Get(base::ATTN_CP).IsEnabled()) \
        && layerId == param.numHiddenLayers - 1 ? \
        "internal_tensor_last_layer" : "internal_tensor_hidden_states";
        // h3p qkvdown dp reshape layerout for moe, without lastlayer
        if (param.enableQkvdownDp && \
            layerId >= static_cast<uint32_t>(param.firstKDenseReplace) && \
            layerId != param.numHiddenLayers - 1) {
            layerOutName = "internal_tensor_hidden_states_slice";
        }
    } else {
        layerOutName = (param.mapping.Get(base::ATTN_DP).IsEnabled() || \
        param.mapping.Get(base::ATTN_CP).IsEnabled()) \
            && layerId == param.numHiddenLayers - 1 && this->param.layerwiseMode == LWD_EDGE_LAST ? \
            "internal_tensor_last_layer" : "internal_tensor_hidden_states";
        if (param.enableQkvdownDp && \
            layerId + this->param.startLayerId >= static_cast<uint32_t>(param.firstKDenseReplace) && \
            !(layerId == param.numHiddenLayers - 1 && this->param.layerwiseMode == LWD_EDGE_LAST)) {
            layerOutName = "internal_tensor_hidden_states_slice";
        }
    }
    return layerOutName;
}

atb::Status DecoderModel::AddSingleLayer(uint32_t layerId)
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node layerNode;
    DecoderLayerParam layerParam;
    if (!this->param.layerwiseDisaggregated) {
        SetLayerParam(layerParam, layerId);
    } else {
        SetLayerParam(layerParam, layerId + this->param.startLayerId);
    }
    ATB_SPEED_LOG_DEBUG("start create Decoderlayer");
    CHECK_OPERATION_STATUS_RETURN(DecoderLayer(layerParam, &op));
    if (this->param.layerwiseDisaggregated) {   // DecoderLayer 中可能修改了 enableQkvdownDp
        param.enableQkvdownDp = layerParam.enableQkvdownDp;
    }
    ATB_SPEED_LOG_DEBUG("Decoderlayer create success");
    layerNode.operation.reset(op);
    ATB_SPEED_LOG_DEBUG("Decoderlayer inTensor number: " << layerNode.operation->GetInputNum());
    layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    size_t inTensorId = 0;
    weightCountPerLayer = WEIGHT_COUNT_PER_LAYER;
    if (param.enableFA3) {
        weightCountPerLayer += 5; // 5: FA3 多5个inTensorensor
    }
	
    if (this->param.layerwiseDisaggregated && this->param.skipWordEmbedding) {
        weightCountWordEmbedding = 0;
    } else {
        weightCountWordEmbedding = WEIGHT_COUNT_WORD_EMBEDDINGNODE;
    }

    if (param.hasP2DWeight && layerId + this->param.startLayerId >= static_cast<uint32_t>(param.firstKDenseReplace)) {
        for (size_t weightTensorId = 0;
            weightTensorId < weightCountPerLayer + DECODER_WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(layerId, weightCountPerLayer) \
                + CheckIntMulOverFlow(layerId + this->param.startLayerId \
                - static_cast<uint32_t>(param.firstKDenseReplace),
                    DECODER_WEIGHT_COUNT_PER_LAYER) + weightTensorId + weightCountWordEmbedding);
        }
    } else {
        for (size_t weightTensorId = 0; weightTensorId < weightCountPerLayer; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
                CheckIntMulOverFlow(layerId, weightCountPerLayer) \
                + weightTensorId + weightCountWordEmbedding);
        }
    }
    ATB_SPEED_LOG_DEBUG("start add layerhostweight");
    AddLayerHostWeight(layerNode, inTensorId, layerId);
    ATB_SPEED_LOG_DEBUG("Add layerhostweight seccess");
    if (layerParam.enableMlaPrefetch) {
        // next_layer_in_q_proj_a_weight
        constexpr uint32_t nextLayerInQProjAWeightId = 4;
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId + 1, WEIGHT_COUNT_PER_LAYER) \
            + nextLayerInQProjAWeightId + weightCountWordEmbedding);

        // next_layer_in_k_proj_b_for_q_weight
        constexpr uint32_t nextLayerInKProjBForQWeightId = 26;
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId + 1, WEIGHT_COUNT_PER_LAYER) \
            + nextLayerInKProjBForQWeightId + weightCountWordEmbedding);
    }
    if (param.finalStateOut && layerId == param.numHiddenLayers - 1 && \
        !(param.lmHeadLocalTp && param.enableDistributed)) {
        layerNode.outTensors = {&graph_.outTensors.at(
            atb_speed::common::GetTensorIdx(this->outTensorMap, "final_hidden_states"))};
    } else {
        bool isLayerWiseOutputLayer = this->param.layerwiseDisaggregated && this->param.isInternalLayer && \
                layerId == this->param.numHiddenLayers - 1;
        if (isLayerWiseOutputLayer) {
            ATB_SPEED_LOG_DEBUG("layerwise output layer");
            layerNode.outTensors = {&graph_.outTensors.at(0)};
        } else {
            std::string layerOutName = GetLayerOutName(layerId);
            const size_t layerInternalOutTensorId = \
                                atb_speed::common::GetTensorIdx(this->internalTensorMap, layerOutName);
            layerNode.outTensors = {&graph_.internalTensors.at(layerInternalOutTensorId)};
        }
    }
    uint32_t ExpertCumSumStartIdx = param.finalStateOut ? 2 : 1;
    if (this->param.layerwiseDisaggregated) {
        layerId += this->param.startLayerId;
    }
    if (layerId >= static_cast<uint32_t>(param.firstKDenseReplace) && \
        param.enableExpertCumSumOutput && param.enableTopkOutput) {
        uint32_t moeLayerId = layerId - static_cast<uint32_t>(param.firstKDenseReplace);
        layerNode.outTensors.push_back(&graph_.outTensors.at(ExpertCumSumStartIdx + 2 * moeLayerId)); // 专家热度
        layerNode.outTensors.push_back(&graph_.outTensors.at(ExpertCumSumStartIdx + 2 * moeLayerId + 1)); // topk
    } else if (layerId >= static_cast<uint32_t>(param.firstKDenseReplace) && param.enableExpertCumSumOutput) {
        uint32_t moeLayerId = layerId - static_cast<uint32_t>(param.firstKDenseReplace);
        layerNode.outTensors.push_back(&graph_.outTensors.at(ExpertCumSumStartIdx + moeLayerId));
    } else if (layerId >= static_cast<uint32_t>(param.firstKDenseReplace) && param.enableTopkOutput) {
        uint32_t moeLayerId = layerId - static_cast<uint32_t>(param.firstKDenseReplace);
        layerNode.outTensors.push_back(&graph_.outTensors.at(ExpertCumSumStartIdx + moeLayerId)); // topk
    }
    graph_.nodes.push_back(layerNode);
    ATB_SPEED_LOG_DEBUG("[+] add base layerNode num" << layerId);

    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddSequenceParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
{
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

atb::Status DecoderModel::AddDenseTpHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId)
{
    if (param.enableDenseTp && layerId + this->param.startLayerId < param.firstKDenseReplace) {
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

atb::Status DecoderModel::AddPrefixCacheHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
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

atb::Status DecoderModel::AddPrefixCacheCpHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
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

atb::Status DecoderModel::AddParallelHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
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
    AddSequenceParallelHostWeight(layerNode, inTensorId);
    return atb::NO_ERROR;
}

void DecoderModel::AddLayerForSkipEmbedding(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId)
{
    if (!this->param.layerwiseDisaggregated) {
        layerNode.inTensors.at(inTensorId++) =
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
            (param.enableQkvdownDp && layerId > param.firstKDenseReplace) ?
            "internal_tensor_hidden_states_slice" : "internal_tensor_hidden_states"));
    } else {
        if (this->param.skipWordEmbedding && layerId == 0) {
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
                this->inTensorMap, "input_embedding"));
        } else {
            layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap,
                (param.enableQkvdownDp && layerId + this->param.startLayerId > param.firstKDenseReplace) ?
                "internal_tensor_hidden_states_slice" : "internal_tensor_hidden_states"));
        }
    }
}

atb::Status DecoderModel::AddExpertHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId)
{
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

atb::Status DecoderModel::AddLayerHostWeight(atb_speed::Model::Node &layerNode, size_t &inTensorId, int layerId)
{
    // h3p qkvdown dp change layerin for moe, without first moe
    AddLayerForSkipEmbedding(layerNode, inTensorId, layerId);
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
    if (param.enableLoadBalance) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "in_fake_topk_model"));
    }
    AddParallelHostWeight(layerNode, inTensorId);
    AddPrefixCacheHostWeight(layerNode, inTensorId);
    AddPrefixCacheCpHostWeight(layerNode, inTensorId);
    AddDenseTpHostWeight(layerNode, inTensorId, layerId);
    // new inTensors please add here before
    AddExpertHostWeight(layerNode, inTensorId);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddFinalNorm()
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
    if (param.finalStateOut && !(param.lmHeadLocalTp && param.enableDistributed)) {
        finalNormNode->inTensors = {
            &graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "final_hidden_states")),
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

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph:finalNormNode end");
    graph_.nodes.push_back(*finalNormNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    auto lmHeadNode = std::make_unique<atb_speed::Model::Node>();
    atb_speed::common::LmHeadParam lmHeadParam;
    lmHeadParam.unpadInputs = !param.isFA;
    lmHeadParam.gatherAhead = param.isPrefill || (param.mapping.Get(base::ATTN_DP).IsEnabled() && !param.lmHeadLocalTp);
    lmHeadParam.hiddenSizePerAttentionHead = param.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = param.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = param.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.unpadInputs = !param.isFA;
    lmHeadParam.enableDpOut = param.enableDpOut && (param.mapping.Get(base::ATTN_DP).rankIds.size() > 1);
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
    ATB_SPEED_LOG_DEBUG("DecoderModel build graph:create LMHead end");

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
    if (param.enableGreedyPostProcessing) {
        lmHeadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "logits_offset_tensor")));
    } else {
        lmHeadNode->inTensors.emplace_back(&graph_.inTensors.at(placeHolderIdx));
    }
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode->outTensors = {param.enableDpOut && param.lmHeadLocalTp ? \
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,"internal_lmhead_out")) : \
        &graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "logits"))};

    ATB_SPEED_LOG_DEBUG("DecoderModel build graph success");
    graph_.nodes.push_back(*lmHeadNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddSliceFinalStateOut()
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

    ATB_SPEED_LOG_DEBUG("AddSliceFinalStateOut calculation success");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddGatherFinalStateOut()
{
    atb::Operation *op = nullptr;
    auto unpadNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &op));
    unpadNode->inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
        this->internalTensorMap, "internal_hidden_states_out_dp"))};
    unpadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "in_post_lmhead_unpadding_indices")));
    unpadNode->outTensors = {&graph_.outTensors.at(1)};
    unpadNode->operation.reset(op);
    graph_.nodes.push_back(*unpadNode);
    ATB_SPEED_LOG_DEBUG("AddGatherFinalStateOut calculation success");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddGatherAfterLmhead()
{
    atb::Operation *op = nullptr;
    auto unpadNode = std::make_unique<atb_speed::Model::Node>();
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &op));
    unpadNode->inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
        "internal_lmhead_out"))};
    unpadNode->inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "in_post_lmhead_unpadding_indices")));
    unpadNode->outTensors = {&graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "logits"))};
    unpadNode->operation.reset(op);
    graph_.nodes.push_back(*unpadNode);
    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddCmoSync()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node waitNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        op, atb_speed::EventAction::POP, atb_speed::common::CMO_SYNC));
    waitNode.inTensors = {};
    waitNode.outTensors = {};
    waitNode.operation.reset(op);
    graph_.nodes.push_back(waitNode);

    atb_speed::Model::Node recordNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        op, atb_speed::EventAction::PUSH, atb_speed::common::CMO_SYNC));
    recordNode.inTensors = {};
    recordNode.outTensors = {};
    recordNode.operation.reset(op);
    CHECK_OPERATION_STATUS_RETURN(SetNodeStreamId(recordNode, 1));
    graph_.nodes.push_back(recordNode);
    return atb::NO_ERROR;
}

atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
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

    uint32_t tokenOffsetTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_token_offset");
    if (tokenOffsetTensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tokenOffsetTensorIdx).hostData = tokenOffset.data();
    }

    uint32_t seqLenTensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_tensor_seq_len");
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

