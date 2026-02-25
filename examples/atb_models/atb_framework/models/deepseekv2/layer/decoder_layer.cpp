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
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/aclnn/ops/inplace_nan_to_num_operation.h"
#include "operations/aclnn/ops/obfuscation_calculate_operation.h"
#include "operations/aclrt/ops/aclrt_cmo_async.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/deepseekv2/operation/latent_attention.h"
#include "atb_speed/base/event_manager.h"
#include "models/deepseekv2/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseekV2 {

static const uint64_t STREAM1 = 1;
static const uint64_t NUM2 = 2;
static const float FLOAT16_MAX = 65504.0f;
static const float FLOAT16_MIN = -65504.0f;

void ObfuscationReshape(const DecoderLayerParam &param, atb::Node& node, uint32_t inTensorId)
{
    if (param.enableModelConfuscation && param.mapping.Get(base::ATTN_TP).rankIds.size() > 1) {
        node.inTensorReshapeFuncs.resize(node.inTensorIds.size());
        if (inTensorId >= node.inTensorIds.size()) {
            return;
        }
        node.inTensorReshapeFuncs[inTensorId] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim num
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: dim
        };
    }
}

void SetDeepseekV2LayerInTensorDefaultCandidates(
    std::map<std::string, std::vector<std::string>> &deepseekV2LayerInTensorCandidates)
{
    deepseekV2LayerInTensorCandidates["default"] = {
            "in_hidden_states", "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
            "in_final_state",
            "in_cos_table", "in_sin_table", "in_attention_mask", "in_k_cache", "in_k_rope_cache", "in_seq_len",
            "in_place_holder", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots", "in_q_len"};
    deepseekV2LayerInTensorCandidates["default_weight"] = {
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
            "in_q_proj_a_compress_idx", "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
            "in_q_proj_b_compress_idx", "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias",
            "in_kv_proj_with_mqa_descale", "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale",
            "in_kv_proj_with_mqa_compress_idx", "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
            "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
            "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
            "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
            "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset",
            "in_attention_out_scale", "in_attention_out_compress_idx", "in_selfattention_out_norm_weight",
            "in_selfattention_out_norm_bias", "in_selfattention_out_new_norm_weight",
            "in_selfattention_out_new_norm_bias", "in_mlp_gateup_weight_shared_expert",
            "in_mlp_gateup_bias_shared_expert", "in_mlp_gateup_descale_shared_expert",
            "in_mlp_gateup_offset_shared_expert", "in_mlp_gateup_scale_shared_expert",
            "in_mlp_gateup_compress_idx_shared_expert", "in_mlp_down_weight_shared_expert",
            "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
            "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert",
            "in_mlp_down_compress_idx_shared_expert", "in_shared_expert_gate_weight", "in_shared_expert_gate_bias",
            "in_shared_expert_gate_descale", "in_shared_expert_gate_offset", "in_shared_expert_gate_scale",
            "in_shared_expert_gate_compress_idx", "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias",
            "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale",
            "in_block_sparse_moe_gate_compress_idx", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert", "in_mlp_down_bias_expert",
            "in_mlp_down_descale_expert", "in_mlp_down_offset_expert", "in_mlp_down_scale_expert",
            "in_mlp_down_compress_idx_expert"};
    deepseekV2LayerInTensorCandidates["mla_prefetch"] = {
            "next_layer_in_q_proj_a_weight", "next_layer_in_k_proj_b_for_q_weight",
    };
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2LayerInTensorCandidates = {
        {"fa3_quant", {
            "in_q_quant_scale", "in_k_quant_scale", "in_qk_descale",
            "kv_offset", "fa3_v_quant_scale"}},
        {"parallel_input", {
            "in_attn_padding_idx", "in_attn_unpadding_idx", "in_ffn_padding_idx",
            "in_ffn_unpadding_idx", "in_lm_head_skip_padding_token_indices",
            "in_attention_padding_idx_slice", "in_start_expert_idx",
            "in_device_expert_count",
            "in_lty_idx", "in_moe_idx"}},
        {"decoder_weight", {
            "in_mlp_gateup_weight_shared_expert_tp", "in_mlp_gateup_bias_shared_expert_tp",
            "in_mlp_gateup_descale_shared_expert_tp", "in_mlp_gateup_offset_shared_expert_tp",
            "in_mlp_gateup_scale_shared_expert_tp", "in_mlp_gateup_compress_idx_shared_expert_tp",
            "in_mlp_down_weight_shared_expert_tp", "in_mlp_down_bias_shared_expert_tp",
            "in_mlp_down_descale_shared_expert_tp", "in_mlp_down_offset_shared_expert_tp",
            "in_mlp_down_scale_shared_expert_tp", "in_mlp_down_compress_idx_shared_expert_tp",
            "in_shared_expert_gate_weight_tp", "in_shared_expert_gate_bias_tp",
            "in_shared_expert_gate_descale_tp", "in_shared_expert_gate_offset_tp",
            "in_shared_expert_gate_scale_tp", "in_shared_expert_gate_compress_idx_tp",
            "in_block_sparse_moe_gate_weight_shuffled", "in_block_sparse_moe_gate_bias_shuffled",
            "in_block_sparse_moe_gate_descale_shuffled", "in_block_sparse_moe_gate_offset_shuffled",
            "in_block_sparse_moe_gate_scale_shuffled", "in_block_sparse_moe_gate_compress_idx_shuffled"
        }},
        {"attn_cp_prefill", {"in_seq_len_cp", "in_cp_load_balance_idx_first", "in_cp_load_balance_idx_last",
                             "in_cp_o_recover_idx", "in_cp_kv_recover_idx"}},
        {"attn_inner_sp_decode", {"in_seq_len_sp"}},
        {"sp_mtp", {"is_need_mask"}},
        {"attn_cp_sp_decode", {"in_filter_mask"}},
        {"force_load_balance", {"in_fake_topk"}},
        {"epwb", {"in_expert_routing_map"}},
        {"mix_shared_routing", {"mix_shared_routing_weight", "mix_shared_routing_expert"}},
        {"prefixcache", {
            "in_history_compressed_kv", "in_history_k_rope", "ring_cur_seqlen", "ring_cache_seqlen"
        }},
        {"prefixcache_cp", {
            "in_kv_cache_padding_idx", "in_kv_cache_unpadding_idx", "in_kv_cache_len"
        }},
        {"prefixcache_c8", {"in_history_compressed_kv_int"}},
        {"dense_tp", {
            "in_dense_tp_padding_idx", "in_dense_tp_mlp_out_idx",
            "in_dense_tp_attn_add_out_idx", "in_dense_tp_gather_prenorm_idx",
            "in_dense_tp_mlp_rs_out_idx"
        }}
    };
    SetDeepseekV2LayerInTensorDefaultCandidates(deepseekV2LayerInTensorCandidates);
    return deepseekV2LayerInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2LayerIntermediateTensorCandidates = {
        {"default", {
            "intermediate_attention_out", "intermediate_attention_add_out",
            "intermediate_moe_out_with_shared"}},
        {"attn_unpadding", {
            "intermediate_selfattention_norm_out"}},
        {"shared_expert", {
            "intermediate_shared_expert_out"}},
        {"attn_need_padding", {
            "intermediate_attention_out_padding",
            "intermediate_selfattention_norm_out_partial",
        }},
        {"attn_reduce_scatter", {
            "intermediate_attention_out_scatter"}},
        {"attn_allgather", {
            "intermediate_dp_attn_out_all_with_padding"}},
        {"ffn_reduce_scatter", {
            "intermediate_moe_out_with_shared_with_padding"}},
        {"ffn_allgather", {
            "intermediate_mlp_out_all"}},
        {"ffn_need_padding", {
            "intermediate_mlp_out"}},
        {"gatherprenorm", {
            "intermediate_selfattention_norm_out_fp32"}},
        {"hiddenstates_padding_slice", {
            "intermediate_hidden_states_padding", "intermediate_hidden_states_scatter"}},
        {"epwb", {
            "intermediate_expert_routing_map"
        }},
        {"enable_out_lcoc_tp", {"intermediate_selfattention_norm_out_partial"}},
        {"pmcc", {"intermediate_pmcc"}},
        {"pmcc_multi", {"intermediate_pmcc", "intermediate_pmcc_per_rank",
            "intermediate_pmcc_gather", "intermediate_pmcc_gather_out"}},
        {"dense_tp", {"intermediate_mlp_out_unpad", "intermediate_attn_add_out_unpad"}},
        {"dense_tp_oproj", {"intermediate_dense_tp_oproj_attn_pad"}},
        {"dense_tp_attn_tp", {"intermediate_dense_tp_rs_addout"}}
    };
    return deepseekV2LayerIntermediateTensorCandidates;
}

bool isGatherPreNorm(const DecoderLayerParam &param)
{
    bool ordinaryGatherNorm = (param.attnReduceScatter || param.attnAllGather) && param.enableGatherPreNorm;
    bool hasQkvdownDp = !param.enableQkvdownDp || param.layerId == param.firstKDenseReplace;
    if ((ordinaryGatherNorm || param.enableExtraOprojTp) && hasQkvdownDp) {
        return true;
    }
    return false;
}

std::vector<std::string> ConstructIntensorList(const DecoderLayerParam &param)
{
    auto deepseekV2InTensorCandidates = GetDeepseekV2LayerInTensorCandidates();
    std::vector<std::string> inTensorList = {};

    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default_weight", inTensorList);
    if (param.enableFA3) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "fa3_quant", inTensorList);
    }

    if (param.hasP2DWeight && param.layerId >= param.firstKDenseReplace) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "decoder_weight", inTensorList);
    }

    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default", inTensorList);
    if (param.enableLoadBalance) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "force_load_balance", inTensorList);
    }
    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "parallel_input", inTensorList);
    if (param.mapping.Get(base::ATTN_CP).IsEnabled() && param.isPrefill) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "attn_cp_prefill", inTensorList);
    }
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() && !param.isPrefill) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "attn_inner_sp_decode", inTensorList);
        if (param.enableSpeculate) {
            atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "sp_mtp", inTensorList);
        }
    }
    if ((param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) &&
        !param.isPrefill) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "attn_cp_sp_decode", inTensorList);
    }
    if (param.enablePrefixCache) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "prefixcache", inTensorList);
        if (param.enableFA3) {
            atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "prefixcache_c8", inTensorList);
        }
        if (param.mapping.Get(base::ATTN_CP).IsEnabled()) {
            atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "prefixcache_cp", inTensorList);
        }
    }
    if (param.hasDenseTp) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "dense_tp", inTensorList);
    }
    if (param.enableEPWB) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "epwb", inTensorList);
    }
    if (param.mixSharedRouting) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "mix_shared_routing", inTensorList);
    }
    
    return inTensorList;
}

void AddAttentionTensor(const DecoderLayerParam &param, std::vector<std::string> &intermediateTensorList,
    std::map<std::string, std::vector<std::string>> deepseekV2IntermediateCandidates)
{
    if (!param.attnAllreduce && (param.hasAttnComm)) {
        if (param.enableOutLcocTp) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "enable_out_lcoc_tp", intermediateTensorList);
        } else {
            if (!param.hasDenseTp || (param.hasDenseTp && !param.enableExtraOprojTp)) {
                atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                    "attn_need_padding", intermediateTensorList);
            }
        }
        if (!param.enableGatherPreNorm) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                                               "hiddenstates_padding_slice", intermediateTensorList);
        }

        if (param.attnReduceScatter && !param.enableOutLcocTp) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "attn_reduce_scatter", intermediateTensorList);
        }
        if (param.attnAllGather) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "attn_allgather", intermediateTensorList);
        }
    }
    if (param.enableModelConfuscation) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
        if (parallelInfo.rankIds.size() > 1) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "pmcc_multi", intermediateTensorList);
        } else {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "pmcc", intermediateTensorList);
        }
    }
}

std::vector<std::string> ConstructIntertensorList(const DecoderLayerParam &param)
{
    auto deepseekV2IntermediateCandidates = GetDeepseekV2LayerIntermediateTensorCandidates();
    std::vector<std::string> intermediateTensorList = {};

    if (param.enableEPWB and param.layerId >= param.firstKDenseReplace) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "epwb", intermediateTensorList);
    }
    atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "default", intermediateTensorList);
    if (!param.hasDenseTp || (param.hasDenseTp && param.enableExtraOprojTp)) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "attn_unpadding", intermediateTensorList);
    }
    if (param.hasSharedExpert && !param.isDenseLayer) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "shared_expert", intermediateTensorList);
    }
    AddAttentionTensor(param, intermediateTensorList, deepseekV2IntermediateCandidates);
    if (param.ffnAllreduce || param.hasFfnComm) {
        // 大ep场景下，开启h3p qkvdown dp时moe层不需要intermediate_mlp_out，最后一层除外
        if (!(param.enableQkvdownDp && !param.isLastLayer && !param.isCloudLastLayer && param.ffnStreamNum > 1)) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "ffn_need_padding", intermediateTensorList);
        }
    }
    if (param.ffnAllGather) {
        if (!param.enableQkvdownDp || param.isLastLayer || param.isCloudLastLayer) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "ffn_allgather", intermediateTensorList);
        }
    }
    if (param.hasFfnComm && !param.hasDenseTp) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
            "ffn_reduce_scatter", intermediateTensorList);
    }
    if (isGatherPreNorm(param)) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
            "gatherprenorm", intermediateTensorList);
    }
    if (param.hasDenseTp) {
        if (param.mapping.Get(base::ATTN_TP).IsEnabled()) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "dense_tp_attn_tp", intermediateTensorList);
        } else {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "dense_tp", intermediateTensorList);
        }
        if (param.enableExtraOprojTp) {
            atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates,
                "dense_tp_oproj", intermediateTensorList);
        }
    }
    return intermediateTensorList;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const DecoderLayerParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    std::vector<std::string> outTensorList = {"out_decoder_layer"};
    if (!param.isDenseLayer && param.enableExpertCumSumOutput) {
        outTensorList.push_back("out_gmm_cumsum_list");
    }

    if (!param.isDenseLayer && param.enableTopkOutput) {
        outTensorList.push_back("out_topk_list");
    }

    std::vector<std::string> inTensorList =  ConstructIntensorList(param);
    if (param.enableMlaPrefetch) {
        inTensorList.push_back("next_layer_in_q_proj_a_weight");
        inTensorList.push_back("next_layer_in_k_proj_b_for_q_weight");
    }
    std::vector<std::string> intermediateTensorList = ConstructIntertensorList(param);

    inTensorNum = inTensorList.size();
    internalTensorNum = intermediateTensorList.size();
    outTensorNum = outTensorList.size();
    ATB_SPEED_LOG_DEBUG("ConstructTensorMap done");
    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

atb::Status SetLatentAttentionInnerCommParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    latentAttentionParam.enableExtraOprojTp = param.enableExtraOprojTp;
    latentAttentionParam.selfOutLinearInnerTensorParallelInfo = param.mapping.Get(base::ATTN_O_PROJ_TP);
    latentAttentionParam.attnOprojPrefetch = param.attnOprojPrefetch;
    latentAttentionParam.enableFusedMLA = param.enableFusedMLA;

    if (param.attnAllreduce) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
        latentAttentionParam.selfOutLinearTensorParallelInfo.rank = parallelInfo.rank;
        latentAttentionParam.selfOutLinearTensorParallelInfo.worldSize = parallelInfo.rankIds.size();
        latentAttentionParam.selfOutLinearTensorParallelInfo.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(
            latentAttentionParam.selfOutLinearTensorParallelInfo.hcommInfo,
            latentAttentionParam.selfOutLinearTensorParallelInfo.commDomain);
    }
    return atb::NO_ERROR;
}

void SetRmsNormParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    latentAttentionParam.normParamType = attenRmsNormParam;
}

void SetRmsNormQuantParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    latentAttentionParam.normQuantParamType = attenRmsNormQuantParam;
}

void SetAttnCpParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    if (param.mapping.Get(base::ATTN_CP).IsEnabled()) {
        latentAttentionParam.contextParallelInfo = param.mapping.Get(base::ATTN_CP);
        latentAttentionParam.ringMLAParam.headNum = param.numAttentionHeadsPerRank;
        latentAttentionParam.ringMLAParam.kvHeadNum = param.numAttentionHeadsPerRank;
        latentAttentionParam.ringMLAParam.qkScale = param.softmaxScale;
        if (param.enablePrefixCache) {
            latentAttentionParam.prefixcacheContextParallelInfo = param.mapping.Get(base::ATTN_PREFIX_CACHE_CP);
        }
    }
}

void SetAttnInnerSpParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled()) {
        latentAttentionParam.hasAttnInnerSp = param.mapping.Get(base::ATTN_INNER_SP).IsEnabled();
        latentAttentionParam.attnSpRank = param.mapping.Get(base::ATTN_INNER_SP).rank;
        latentAttentionParam.attnSpSize = param.mapping.Get(base::ATTN_INNER_SP).rankIds.size();
        latentAttentionParam.attnSpRankTableFile = "";
        latentAttentionParam.attnSpBackend = "lccl";
        param.mapping.Get(base::ATTN_INNER_SP).InitCommDomain(
            latentAttentionParam.attnSpHcclComm,
            latentAttentionParam.attnSpDomain,
            "lccl");

        latentAttentionParam.pageAttentionParam.headNum = \
            param.numAttentionHeadsPerRank * param.mapping.Get(base::ATTN_INNER_SP).rankIds.size();
    }
}

atb::Status SetLatentAttentionParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    SetRmsNormParam(latentAttentionParam, param);
    SetRmsNormQuantParam(latentAttentionParam, param);

    latentAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    latentAttentionParam.isBF16 = param.isBF16;
    latentAttentionParam.attnLinearQuantType = param.attnLinearQuantType;
    latentAttentionParam.packQuantType = param.packQuantType.at(0);
    latentAttentionParam.quantGroupSize = param.quantGroupSize;
    latentAttentionParam.attnLinearTransposeType = param.attnLinearTransposeType;
    latentAttentionParam.enableLcoc = param.enableLcoc;
    latentAttentionParam.qLoraRank = param.qLoraRank;
    latentAttentionParam.headNum = param.headNum;
    latentAttentionParam.qkNopeHeadDim = param.qkNopeHeadDim;
    latentAttentionParam.qkRopeHeadDim = param.qkRopeHeadDim;
    latentAttentionParam.kvLoraRank = param.kvLoraRank;
    latentAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    latentAttentionParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
    latentAttentionParam.isFA = param.isFA;
    latentAttentionParam.isPrefill = param.isPrefill;
    latentAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    latentAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.selfAttentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    latentAttentionParam.selfAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        latentAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        latentAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    latentAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    latentAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.pageAttentionParam.kvHeadNum = 1;
    latentAttentionParam.pageAttentionParam.mlaVHeadSize = param.kvLoraRank;
    latentAttentionParam.pageAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    latentAttentionParam.enableMlaPreprocess = param.enableMlaPreprocess;
    if (param.enableSpeculate) {
        if (param.maskfree) {
            latentAttentionParam.pageAttentionParam.maskType = \
                atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
        } else {
            latentAttentionParam.pageAttentionParam.maskType = \
                atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
        }
        latentAttentionParam.pageAttentionParam.calcType = atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
    }
    if (param.enableFA3 && param.enableKvQuantLayer) {
        latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
            atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ;
        latentAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE;
        latentAttentionParam.pageAttentionParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    } else if (param.isNzCache) {
        latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
            atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ;
    } else {
        latentAttentionParam.reshapeCacheParm.kvCacheCfg = \
            atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE;
    }
    latentAttentionParam.isNzCache = param.isNzCache;
    // This function must be called after the pageAttentionParam is set. It will change pageAttentionParam.headNum
    SetAttnCpParam(latentAttentionParam, param);
    SetAttnInnerSpParam(latentAttentionParam, param);
    SetLatentAttentionInnerCommParam(latentAttentionParam, param);
    latentAttentionParam.enableQkvdownDp = param.enableQkvdownDp && param.layerId > param.firstKDenseReplace;
    latentAttentionParam.layerId = param.layerId;
    latentAttentionParam.firstKDenseReplace = param.firstKDenseReplace;
    latentAttentionParam.hasAttnComm = param.hasAttnComm;
    latentAttentionParam.attnTpRank = param.mapping.Get(base::ATTN_TP).rank;
    latentAttentionParam.attnTpSize = param.mapping.Get(base::ATTN_TP).rankIds.size();
    latentAttentionParam.attnTpBackend = param.mapping.Get(base::ATTN_TP).defaultBackend;
    latentAttentionParam.attnTpRankTableFile = "";
    param.mapping.Get(base::ATTN_TP).InitCommDomain(latentAttentionParam.hcclComm, latentAttentionParam.attnTpDomain);
    latentAttentionParam.ffnAllGather = param.ffnAllGather;
    latentAttentionParam.hasFfnComm = param.hasFfnComm;

    latentAttentionParam.enableOutLcocTp = param.enableOutLcocTp;
    latentAttentionParam.enablePreprocessLcocTp = param.enablePreprocessLcocTp;
    latentAttentionParam.lcocAttnTpRank = param.mapping.Get(base::ATTN_TP).rank;
    latentAttentionParam.lcocAttnTpRankSize = param.mapping.Get(base::ATTN_TP).rankIds.size();
    latentAttentionParam.lcocAttnTpBackend = "lcoc";
    param.mapping.Get(base::ATTN_TP).InitCommDomain(
        latentAttentionParam.lcocHcclComm, latentAttentionParam.lcocAttnTpDomain,
        latentAttentionParam.lcocAttnTpBackend);
    latentAttentionParam.enablePrefixCache = param.enablePrefixCache;
    return atb::NO_ERROR;
}

int64_t SetAttention(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node attentionNode;
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> latentAttentionParam;
    SetLatentAttentionParam(latentAttentionParam, param);
    CHECK_OPERATION_STATUS_RETURN(Attention(latentAttentionParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
        "in_hidden_states",
        "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
        "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale",
        "in_q_proj_a_offset", "in_q_proj_a_scale", "in_q_proj_a_compress_idx",
        "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
        "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale",
        "in_q_proj_b_offset", "in_q_proj_b_scale", "in_q_proj_b_compress_idx",
        "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
        "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
        "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
        "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
        "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
        "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
        "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
        "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
        "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx",
        "in_cos_table", "in_sin_table", "in_seq_len", "in_k_cache", "in_k_rope_cache",
        "in_attention_mask", "in_q_len", "in_token_offset", "in_layer_id", "in_block_tables",
        "in_slots", "in_attn_padding_idx"
    };
    if (param.enablePrefixCache) {
        atb_speed::common::AddTensorToList(GetDeepseekV2LayerInTensorCandidates(), "prefixcache", attnInTensorNames);
        if (param.enableFA3 && param.enableKvQuantLayer) {
            atb_speed::common::AddTensorToList(GetDeepseekV2LayerInTensorCandidates(), "prefixcache_c8", attnInTensorNames);
        }
        if (param.mapping.Get(base::ATTN_CP).IsEnabled()) {
            atb_speed::common::AddTensorToList(
                GetDeepseekV2LayerInTensorCandidates(), "prefixcache_cp", attnInTensorNames);
        }
    }
    if (param.enableFA3 && param.enableKvQuantLayer) {
        atb_speed::common::AddTensorToList(GetDeepseekV2LayerInTensorCandidates(), "fa3_quant", attnInTensorNames);
    }
    if (param.enableQkvdownDp && param.layerId > param.firstKDenseReplace) {
        // h3p qkvdown dp特性和sp特性同时开启时，需要与latent_attention.cpp模块中两个特性的inTensor顺序一致，
        // h3p qkvdown dp的inTensor需在sp的inTensor之前
        attnInTensorNames.push_back("in_ffn_unpadding_idx");
    }
    if (param.mapping.Get(base::ATTN_CP).IsEnabled() && param.isPrefill) {
        attnInTensorNames.push_back("in_seq_len_cp");
        attnInTensorNames.push_back("in_cp_load_balance_idx_first");
        attnInTensorNames.push_back("in_cp_load_balance_idx_last");
        attnInTensorNames.push_back("in_cp_o_recover_idx");
        attnInTensorNames.push_back("in_cp_kv_recover_idx");
    }
    if (param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() && !param.isPrefill) {
        attnInTensorNames.push_back("in_seq_len_sp");
        if (param.enableSpeculate) {
            attnInTensorNames.push_back("is_need_mask");
        }
    }
    if ((param.mapping.Get(base::ATTN_INNER_SP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled())
        && !param.isPrefill) {
        attnInTensorNames.push_back("in_filter_mask");
    }
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
    opGraph.nodes.push_back(attentionNode);
    ATB_SPEED_LOG_DEBUG("Attention calculation success");
    return atb::NO_ERROR;
}

atb::Status SetPadding(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap,
    const DecoderLayerParam &param)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out",
        param.hasDenseTp ? "in_dense_tp_padding_idx" : "in_attn_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {
        (param.hasDenseTp && param.enableExtraOprojTp) ?
        "intermediate_dense_tp_oproj_attn_pad" : "intermediate_attention_out_padding"});

    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("create SetPadding");
    return atb::NO_ERROR;
}

atb::Status SetAttnReduceScatter(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node rsNode;
    atb::infer::ReduceScatterParam rsParam;
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
    rsParam.rank = parallelInfo.rank;
    rsParam.rankSize = parallelInfo.rankIds.size();
    rsParam.backend = parallelInfo.defaultBackend;
    parallelInfo.InitCommDomain(rsParam.hcclComm, rsParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(rsParam, &rsNode.operation));
    rsNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {param.enableOutLcocTp ?
        "intermediate_attention_out" : "intermediate_attention_out_padding"});
    rsNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out_scatter"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(rsNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

atb::Status SetResidualPadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {param.enableModelConfuscation ?
        "intermediate_pmcc" : "in_hidden_states", "in_attn_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_padding"});
    ObfuscationReshape(param, gatherNode, 0);
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status SetResidualSliceNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::infer::SliceParam sliceParam;
    atb::Node sliceNode;

    sliceParam.offsets.resize(3); // 3: Slice offset dim
    sliceParam.offsets[0] = param.mapping.Get(base::ATTN_TP).rank;
    sliceParam.offsets[1] = 0;
    sliceParam.offsets[2] = 0; // 2: dim：2

    sliceParam.size.resize(3); // 3: Slice Size dim
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    sliceParam.size[2] = -1; // 2: dim：2
    CreateOperation(sliceParam, &sliceNode.operation);

    sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_padding"});
    sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_scatter"});

    sliceNode.inTensorReshapeFuncs.resize(sliceNode.inTensorIds.size());
    sliceNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum == 2) { // 2: dimNum
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = param.mapping.Get(base::ATTN_TP).rankIds.size();
            newShape.dims[1] = oldShape.dims[0] / param.mapping.Get(base::ATTN_TP).rankIds.size();
            newShape.dims[2] = oldShape.dims[1]; // 2: dim 2
        } else {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = param.mapping.Get(base::ATTN_TP).rankIds.size();
            newShape.dims[1] = oldShape.dims[0] * oldShape.dims[1] / param.mapping.Get(base::ATTN_TP).rankIds.size();
            newShape.dims[2] = oldShape.dims[2]; // 2: dim 2
        }
    };
    opGraph.nodes.push_back(sliceNode);
    return atb::NO_ERROR;
}

atb::Status SetSelfResidualAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    if (param.attnReduceScatter) {
        selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_hidden_states_scatter",
            param.enableOutLcocTp ? "intermediate_attention_out" : "intermediate_attention_out_scatter"});
        selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
        selfResidualAddNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 2; // 2: dimNum
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
        };
    } else {
        selfResidualAddNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(tensorMap, {param.enableModelConfuscation ?
                "intermediate_pmcc" : "in_hidden_states", "intermediate_attention_out"});
        ObfuscationReshape(param, selfResidualAddNode, 0);
    }
    selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {"intermediate_attention_add_out"});

    opGraph.nodes.push_back(selfResidualAddNode);
    ATB_SPEED_LOG_DEBUG("SelfResidualAdd calculation success");
    return atb::NO_ERROR;
}

int64_t SetAllGather(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    if (param.hasDenseTp) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::DENSE_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else if (param.attnReduceScatter) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_DP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    }

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {(param.hasDenseTp && param.enableExtraOprojTp) ?
            "intermediate_selfattention_norm_out" : "intermediate_selfattention_norm_out_partial"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_dp_attn_out_all_with_padding"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));

    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

atb::Status SetAllGatherCCOverlap(atb::GraphParam &opGraph, const DecoderLayerParam &param)
{
    if (param.enableSharedExpertOverlap) {
        if (!param.isPrefill || !param.enableGatingDp) {
            CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
                opGraph, atb_speed::EventAction::POP, atb_speed::common::COMM_CONTROL));
            CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateWaitWithoutNodeId(
                opGraph, atb_speed::EventAction::POP, atb_speed::common::COMP_CONTROL));
        } else {
            CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
                opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMM_CONTROL));
            CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateWaitWithoutNodeId(
                opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMP_CONTROL));
        }
    }
    ATB_SPEED_LOG_DEBUG("AllGather CCOverlap Event success");
    return atb::NO_ERROR;
}

int64_t SetAttnUnpadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node unpadNode;
    atb::infer::GatherParam unpadParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(unpadParam, &unpadNode.operation));
    unpadNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {param.attnAllGather ? "intermediate_dp_attn_out_all_with_padding" :
            "intermediate_selfattention_norm_out_partial", "in_attn_unpadding_idx"});
    unpadNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {"intermediate_selfattention_norm_out"});
    unpadNode.inTensorReshapeFuncs.reserve(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs.resize(unpadNode.inTensorIds.size());
    unpadNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
    newShape.dimNum = 2; // 2：新shape维度为2
        if (oldShape.dimNum == 3) { // 3：旧shape维度为3
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
            newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
        } else {
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1]; // 1, 2: 新shape最后一维不变
        }
    };
    opGraph.nodes.push_back(unpadNode);

    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

atb::Status SetNormQauntInTensors(
    std::vector<std::string> &selfNormInTensorNames,
    atb::infer::RmsNormParam &mlpRmsNormParam,
    atb::infer::RmsNormParam &mlpRmsNormQuantParam,
    const DecoderLayerParam &param,
    atb::Node &selfNormNode)
{
    if (param.mlpNormQuantType == atb::infer::QUANT_INT8) { // w8a8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormQuantParam, &selfNormNode.operation));
        if (param.isAntiOutlier) {
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
        } else {
            selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
            selfNormInTensorNames.push_back("in_selfattention_out_norm_bias");
        }
    } else if (param.normHasBias) { // FP
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
        selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
        selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
        if (param.isAntiOutlier) {
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
        } else {
            selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
        }
    }
    return atb::NO_ERROR;
}

int64_t SetSelfNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfNormNode;
    atb::infer::RmsNormParam mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    std::vector<std::string> selfNormInTensorNames;
    std::vector<std::string> selfNormOutTensorNames;
    bool obfuscation = false;
    if (!param.attnAllreduce && param.hasAttnComm) {
        selfNormOutTensorNames.push_back("intermediate_selfattention_norm_out_partial");
    } else {
        selfNormOutTensorNames.push_back("intermediate_selfattention_norm_out");
    }

    if (param.enableIntraLayerAddNorm) {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormParam.preNormParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormQuantParam.preNormParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
        if (param.attnReduceScatter) { // 2: Dynamic EP
            selfNormInTensorNames.push_back("intermediate_hidden_states_scatter");
            selfNormOutTensorNames.push_back("intermediate_attention_add_out");
        } else {
            selfNormInTensorNames.push_back(param.enableModelConfuscation ?
                "intermediate_pmcc" : "in_hidden_states");
            selfNormOutTensorNames.push_back("intermediate_attention_add_out");
            obfuscation = true;
        }
    } else {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormParam.normParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormQuantParam.normParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
        selfNormInTensorNames.push_back("intermediate_attention_add_out");
    }

    SetNormQauntInTensors(selfNormInTensorNames, mlpRmsNormParam, mlpRmsNormQuantParam, param, selfNormNode);
    selfNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormInTensorNames);
    selfNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormOutTensorNames);
    if (obfuscation) {
        ObfuscationReshape(param, selfNormNode, 0);
    }
    opGraph.nodes.push_back(selfNormNode);
    ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");
    return atb::NO_ERROR;
}

int64_t SetMlpOutUnpad(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam padParam;
    atb::CreateOperation(padParam, &gatherNode.operation);
    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_mlp_out", "in_dense_tp_mlp_out_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_mlp_out_unpad"});
    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("Set mlpOutUnpad success");
    return atb::NO_ERROR;
}

int64_t SetAttnAddOutUnpad(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam padParam;
    atb::CreateOperation(padParam, &gatherNode.operation);
    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_attention_add_out", "in_dense_tp_attn_add_out_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_attn_add_out_unpad"});
    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("Set attnAddOutUnpad success");
    return atb::NO_ERROR;
}

int64_t SetMlpExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mlpExpertNode;
    atb_speed::common::SharedExpertParam mlpExpertParam;
    mlpExpertParam.isBF16 = param.isBF16;
    mlpExpertParam.transposeGateup = param.mlpLinearTransposeType[MLP_GATEUP_LINEAR_INDEX];
    mlpExpertParam.transposeDown = param.mlpLinearTransposeType[MLP_DOWN_LINEAR_INDEX];
    mlpExpertParam.hasSharedExpertGate = false;
    mlpExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    mlpExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    mlpExpertParam.packQuantType = param.packQuantType.at(1);
    mlpExpertParam.quantGroupSize = param.quantGroupSize;
    mlpExpertParam.enableSwiGLUQuantForSharedExperts = param.enableSwiGLUQuantForSharedExperts;
    atb_speed::common::CreateSharedExpertOperation(mlpExpertParam, &mlpExpertNode.operation);
    std::vector<std::string> mlpExpertInTensorNames = {
        param.hasDenseTp ?
        "intermediate_dp_attn_out_all_with_padding" : "intermediate_selfattention_norm_out",
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    mlpExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpExpertInTensorNames);
    mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    if (param.hasDenseTp) {
        mlpExpertNode.inTensorReshapeFuncs.reserve(mlpExpertNode.inTensorIds.size());
        mlpExpertNode.inTensorReshapeFuncs.resize(mlpExpertNode.inTensorIds.size());
        mlpExpertNode.inTensorReshapeFuncs[0] = [=] (const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2：新shape维度为2
            if (oldShape.dimNum == 3) { // 3：旧shape维度为3
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
                newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
            } else {
                newShape.dims[0] = oldShape.dims[0];
                newShape.dims[1] = oldShape.dims[1]; // 1, 2: 新shape最后一维不变
            }
        };
    }
    opGraph.nodes.push_back(mlpExpertNode);
    ATB_SPEED_LOG_DEBUG("mlp expert calculation success");
    return atb::NO_ERROR;
}

int64_t SetExpertRoutingMapSlice(
    atb::GraphParam &opGraph,
    const DecoderLayerParam &param, std::map<std::string,
    uint32_t> tensorMap)
{
    atb::Node sliceNode;
    atb::infer::SliceParam sliceParam;
    sliceParam.offsets.resize(NUM2);
    sliceParam.offsets[0] = param.layerId - param.firstKDenseReplace;
    sliceParam.offsets[1] = 0;
    sliceParam.size.resize(NUM2);
    sliceParam.size[0] = 1;
    sliceParam.size[1] = -1;
    CreateOperation(sliceParam, &sliceNode.operation);
    sliceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"in_expert_routing_map"});
    sliceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_expert_routing_map"});
    opGraph.nodes.push_back(sliceNode);
    return atb::NO_ERROR;
}

atb::Status SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam, const DecoderLayerParam &param)
{
    sparseMoeParam.isBF16 = param.isBF16;
    sparseMoeParam.gateUpTransposeB = param.moeLinearTransposeType[MOE_GATEUP_LINEAR_INDEX];
    sparseMoeParam.downTransposeB = param.moeLinearTransposeType[MOE_DOWN_LINEAR_INDEX];
    sparseMoeParam.numOfExperts = param.numOfExperts;
    sparseMoeParam.numOfDeviceExperts = param.numOfDeviceExperts;
    sparseMoeParam.num = param.numOfSelectedExperts;
    sparseMoeParam.routingMethod = param.routingMethod;
    sparseMoeParam.numOfGroups = param.numOfGroups;
    sparseMoeParam.topkGroups = param.topkGroups;
    sparseMoeParam.scaledTopk = param.scaledTopk;
    sparseMoeParam.enableInitRoutingCutoff = param.enableInitRoutingCutoff;
    sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
    sparseMoeParam.isDynamicEp = param.isDynamicEp;
    sparseMoeParam.deviceExpert = param.deviceExpert;
    sparseMoeParam.routedScalingFactor = param.routedScalingFactor;
    sparseMoeParam.processLogits = param.processLogits;
    sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
    if (param.moePackQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED) {
        sparseMoeParam.packQuantType = param.packQuantType.at(1);
    } else {
        sparseMoeParam.packQuantType = param.moePackQuantType;
    }
    sparseMoeParam.quantGroupSize = param.quantGroupSize;
    sparseMoeParam.enableFusedRouting = param.enableFusedRouting;
    sparseMoeParam.enableInitQuant = param.enableInitQuant;
    sparseMoeParam.enableSwigluQuant = param.enableSwigluQuant;
    sparseMoeParam.enableFusedTopk = param.enableFusedTopk;
    sparseMoeParam.enableExpertCumSumOutput = param.enableExpertCumSumOutput;
    sparseMoeParam.enableTopkOutput = param.enableTopkOutput;
    sparseMoeParam.enableATBGateMatmul = param.enableATBGateMatmul;
    sparseMoeParam.enableFp32GateInput = isGatherPreNorm(param) && param.isDynamicEp;
    sparseMoeParam.enableGMMSwigluQuant = param.enableGMMSwigluQuant;
    sparseMoeParam.enableAtlasGMMFused = param.enableAtlasGMMFused;
    sparseMoeParam.enableCVOverlap = param.enableCVOverlap;
    sparseMoeParam.enableLoadBalance = param.enableLoadBalance;
    sparseMoeParam.enableEPWB = param.enableEPWB;
    sparseMoeParam.numOfRedundantExpert = param.numOfRedundantExpert;
    sparseMoeParam.numDanglingSharedExperts = param.numDanglingSharedExperts;
    sparseMoeParam.maxDecodeDpTokenSize = param.maxDecodeDpTokenSize;
    sparseMoeParam.enableMoeDistribute = !param.isPrefill && param.enableAllToAllMC2 && param.isDynamicEp;
    sparseMoeParam.enableDispatchCombineV2 = param.enableDispatchCombineV2;
    sparseMoeParam.enableGatingDp = param.enableGatingDp && param.isPrefill;  // h3p gatingdp for moe
    sparseMoeParam.enableGatingShift = param.enableGatingDp && !param.isPrefill && !param.mixSharedRouting;  // h3p gatingshift for decode
    sparseMoeParam.enableGatingOverlap = sparseMoeParam.enableGatingDp &&
                                        param.enableSharedExpertOverlap;  // h3p Gating overlap
    sparseMoeParam.mixSharedRouting = param.mixSharedRouting;
    return atb::NO_ERROR;
}

atb::Status SetSparseMoeCommParam(atb_speed::common::SparseMoeParam &sparseMoeParam, const DecoderLayerParam &param)
{
    sparseMoeParam.hasMoeEp = param.mapping.Get(base::MOE_EP).IsEnabled();
    sparseMoeParam.moeEpParallelInfo = param.mapping.Get(base::MOE_EP);
    sparseMoeParam.mlpTpParallelInfo = param.mapping.Get(base::MLP_TP);
    sparseMoeParam.moeEpInterNodeParallelInfo = param.mapping.Get(base::MOE_EP_INTER_NODE);
    sparseMoeParam.moeEpIntraNodeParallelInfo = param.mapping.Get(base::MOE_EP_INTRA_NODE);
    
    sparseMoeParam.enableLcocAll2All = param.enableLcocAll2All;
    if (sparseMoeParam.enableMoeDistribute) {
        sparseMoeParam.dispatchAndCombinecommDomain = param.dispatchAndCombinecommDomain;
        sparseMoeParam.dispatchAndCombineHcclComm = param.dispatchAndCombineHcclComm;
    }
    if (sparseMoeParam.enableLcocAll2All) {
        param.mapping.Get(base::MOE_EP).InitCommDomain(
            sparseMoeParam.lcclMoeEpHcclComm, sparseMoeParam.lcclMoeEpDomain, "lccl");
    }
    return atb::NO_ERROR;
}

int64_t SetMoe(atb::GraphParam &opGraph, const DecoderLayerParam &param, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node moeNode;
    atb_speed::common::SparseMoeParam sparseMoeParam;
    SetSparseMoeParam(sparseMoeParam, param);
    SetSparseMoeCommParam(sparseMoeParam, param);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation));
    std::vector<std::string> moeInTensorNames;
    moeInTensorNames = std::vector<std::string>{
        "intermediate_selfattention_norm_out", "in_block_sparse_moe_gate_weight",
        "in_block_sparse_moe_gate_bias", "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset",
        "in_block_sparse_moe_gate_scale", "in_block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
        "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
        "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
        "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert",
        "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
    };
    if (param.hasP2DWeight && !param.isPrefill) {
        for (int i = 1; i < 7; i++) {  // 7: 需要更改的最后一个变量
            moeInTensorNames[i] += "_shuffled";
        }
    }
    if (param.enableLoadBalance) {
        moeInTensorNames.push_back("in_fake_topk");
    }
    if (param.mapping.Get(base::MOE_EP).IsEnabled()) {
        moeInTensorNames.push_back("in_start_expert_idx");
        moeInTensorNames.push_back("in_device_expert_count");
        moeInTensorNames.push_back("in_ffn_padding_idx");
        if (param.isDynamicEp) {
            moeInTensorNames.push_back("in_lty_idx");
            moeInTensorNames.push_back("in_moe_idx");
        }
    }
    if (param.enableEPWB) {
        SetExpertRoutingMapSlice(opGraph, param, tensorMap);
        moeInTensorNames.push_back("intermediate_expert_routing_map");
    }
    // h3p gatingdp prefill add intensor partial
    if (param.enableGatingDp && param.isPrefill) {
        moeInTensorNames.push_back("intermediate_selfattention_norm_out_partial");
        moeInTensorNames.push_back("in_attn_unpadding_idx");
    }
    if (isGatherPreNorm(param) && param.isDynamicEp) {
        moeInTensorNames.push_back("intermediate_selfattention_norm_out_fp32");
    }
    if (param.mixSharedRouting) {
        moeInTensorNames.push_back("mix_shared_routing_weight");
        moeInTensorNames.push_back("mix_shared_routing_expert");
    }
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
    moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    if (param.enableExpertCumSumOutput) {
        moeNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, {"out_gmm_cumsum_list"}));
    }
    if (param.enableTopkOutput) {
        moeNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, {"out_topk_list"}));
    }
    opGraph.nodes.push_back(moeNode);
    ATB_SPEED_LOG_DEBUG("Moe sparse calculation success");
    return atb::NO_ERROR;
}

atb::Status SetSharedExpertParam(atb_speed::common::SharedExpertParam &sharedExpertParam,
                                 const DecoderLayerParam &param)
{
    sharedExpertParam.isBF16 = param.isBF16;
    sharedExpertParam.transposeGateup = param.mlpLinearTransposeType[MLP_GATEUP_LINEAR_INDEX];
    sharedExpertParam.transposeDown = param.mlpLinearTransposeType[MLP_DOWN_LINEAR_INDEX];
    sharedExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
    sharedExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    sharedExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    sharedExpertParam.quantGroupSize = param.quantGroupSize;
    sharedExpertParam.packQuantType = param.packQuantType.at(1);
    sharedExpertParam.enableCVOverlap = param.enableCVOverlap;
    sharedExpertParam.enableSwiGLUQuantForSharedExperts = param.enableSwiGLUQuantForSharedExperts;
    return atb::NO_ERROR;
}

int64_t SetSharedExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                        std::map<std::string, uint32_t> tensorMap)
{
    atb::Node sharedExpertNode;
    atb_speed::common::SharedExpertParam sharedExpertParam;
    SetSharedExpertParam(sharedExpertParam, param);
    atb_speed::common::CreateSharedExpertOperation(sharedExpertParam, &sharedExpertNode.operation);
    if (param.hasP2DWeight && !param.isPrefill) {
        std::vector<std::string> sharedExpertInTensorNames = {
            "intermediate_selfattention_norm_out",
            "in_mlp_gateup_weight_shared_expert_tp", "in_mlp_gateup_bias_shared_expert_tp",
            "in_mlp_gateup_descale_shared_expert_tp", "in_mlp_gateup_offset_shared_expert_tp",
            "in_mlp_gateup_scale_shared_expert_tp", "in_mlp_gateup_compress_idx_shared_expert_tp",
            "in_mlp_down_weight_shared_expert_tp", "in_mlp_down_bias_shared_expert_tp",
            "in_mlp_down_descale_shared_expert_tp", "in_mlp_down_offset_shared_expert_tp",
            "in_mlp_down_scale_shared_expert_tp", "in_mlp_down_compress_idx_shared_expert_tp",
            "in_shared_expert_gate_weight_tp", "in_shared_expert_gate_bias_tp",
            "in_shared_expert_gate_descale_tp", "in_shared_expert_gate_offset_tp",
            "in_shared_expert_gate_scale_tp", "in_shared_expert_gate_compress_idx_tp"
        };
        sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
    } else {
        std::vector<std::string> sharedExpertInTensorNames = {
            "intermediate_selfattention_norm_out",
            "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
            "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
            "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
            "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
            "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
            "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
            "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
            "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
        };
        // h3p shared expert dp first intensor partial
        if (param.enableSharedExpertDp) {
            sharedExpertInTensorNames[0] = "intermediate_selfattention_norm_out_partial";
        }
        sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
    }
    sharedExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_expert_out"});
    if (param.enableCVOverlap) {
        // set extreme stream for moe cv parallel, stream id is 2
        atb::SetExecuteStreamId(sharedExpertNode.operation, 2);
    }
    if (param.enableSharedExpertOverlap) {
        atb::SetExecuteStreamId(sharedExpertNode.operation, STREAM1);
    }
    opGraph.nodes.push_back(sharedExpertNode);
    ATB_SPEED_LOG_DEBUG("Shared expert calculation success");
    return atb::NO_ERROR;
}

int64_t AddExpertAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> tensorMap)
{
    atb::Node expertAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
    expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared",
                                                                                "intermediate_shared_expert_out"});
    expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared"});
    // h3p shared expert dp add routing expert out after reduce scatter
    if (param.enableSharedExpertDp) {
        expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out",
                                                                                    "intermediate_shared_expert_out"});
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    }
    opGraph.nodes.push_back(expertAddNode);
    ATB_SPEED_LOG_DEBUG("create add operation");
    return atb::NO_ERROR;
}

int64_t SetAllReduce(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> tensorMap)
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    if (param.isDenseLayer) {
        if (param.enableDenseTp) {
            atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::DENSE_TP);
            allReduceParam.rank = parallelInfo.rank;
            allReduceParam.rankSize = parallelInfo.rankIds.size();
            allReduceParam.backend = parallelInfo.defaultBackend;
            parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
        } else {
            atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
            allReduceParam.rank = parallelInfo.rank;
            allReduceParam.rankSize = parallelInfo.rankIds.size();
            allReduceParam.backend = parallelInfo.defaultBackend;
            parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
        }
    } else {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
        allReduceParam.rank = parallelInfo.rank;
        allReduceParam.rankSize = parallelInfo.rankIds.size();
        allReduceParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allReduceParam.hcclComm, allReduceParam.commDomain);
    }

    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {"intermediate_moe_out_with_shared"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(moeAllReduceNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

atb::Status SetMlpResidualAdd(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                              std::map<std::string, uint32_t> tensorMap)
{
    atb::Node mlpResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    std::vector<std::string> mlpResidualAddInTensorNames = {};
    std::vector<std::string> mlpResidualAddOutTensorNames = {};
    if (param.hasDenseTp) {
        if (param.mapping.Get(base::ATTN_TP).IsEnabled()) {
            mlpResidualAddInTensorNames = {"intermediate_attention_add_out", "intermediate_mlp_out"};
            mlpResidualAddOutTensorNames = {"intermediate_dense_tp_rs_addout"};
        } else {
            mlpResidualAddInTensorNames = {"intermediate_attn_add_out_unpad", "intermediate_mlp_out_unpad"};
            mlpResidualAddOutTensorNames = {"out_decoder_layer"};
        }
    } else {
        mlpResidualAddInTensorNames = {"intermediate_attention_add_out",
        param.ffnAllreduce || param.ffnReduceScatter ?
        "intermediate_mlp_out" :
        ((param.hasAttnComm) && (param.hasFfnComm) ?
            "intermediate_moe_out_with_shared_with_padding" : "intermediate_moe_out_with_shared")};
        mlpResidualAddOutTensorNames = {param.ffnAllGather || param.ffnReduceScatter ?
        ((param.enableQkvdownDp && !param.isLastLayer && !param.isCloudLastLayer) ?
         "out_decoder_layer" : "intermediate_mlp_out") : "out_decoder_layer"};
    }
    mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddInTensorNames);
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
    opGraph.nodes.push_back(mlpResidualAddNode);
    ATB_SPEED_LOG_DEBUG("create mlpResidualAdd");
    return atb::NO_ERROR;
}

int64_t SetFFNPadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node padNode;
    atb::infer::GatherParam padParam;
    atb::CreateOperation(padParam, &padNode.operation);
    padNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {param.hasAttnComm ?
        "intermediate_moe_out_with_shared" : "intermediate_mlp_out",
        param.hasAttnComm ?
        "in_ffn_padding_idx" : "in_attn_padding_idx"});
    padNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_moe_out_with_shared_with_padding"});
    opGraph.nodes.push_back(padNode);
    ATB_SPEED_LOG_DEBUG("create padNode");
    return atb::NO_ERROR;
}

int64_t SetMlpReduceScatter(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node reduceScatterNode;
    atb::infer::ReduceScatterParam reduceScatterParam;
    if (param.hasDenseTp) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::DENSE_TP);
        reduceScatterParam.rank = parallelInfo.rank;
        reduceScatterParam.rankSize = parallelInfo.rankIds.size();
        reduceScatterParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(reduceScatterParam.hcclComm, reduceScatterParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
        reduceScatterParam.rank = parallelInfo.rank;
        reduceScatterParam.rankSize = parallelInfo.rankIds.size();
        reduceScatterParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(reduceScatterParam.hcclComm, reduceScatterParam.commDomain);
    }

    CreateOperation(reduceScatterParam, &reduceScatterNode.operation);
    if (reduceScatterNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    reduceScatterNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {param.hasDenseTp ?
        "intermediate_moe_out_with_shared" : "intermediate_moe_out_with_shared_with_padding"});
    reduceScatterNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_mlp_out"});
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(reduceScatterNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

int64_t SetMlpReduceScatterNanToNum(atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node nanToNumNode;
    atb_speed::common::AclNNNanToNumParam NanToNumParam;
    NanToNumParam.posInfValue = FLOAT16_MAX;  // replaces positive infinity values in tensor elements
    NanToNumParam.negInfValue = FLOAT16_MIN;  // replaces negative infinity values in tensor elements
    nanToNumNode.operation = new atb_speed::common::InplaceNanToNumOperation("nanToNumNode", NanToNumParam);
    nanToNumNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    nanToNumNode.outTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    opGraph.nodes.push_back(nanToNumNode);
    ATB_SPEED_LOG_DEBUG("create nan to num");
    return atb::NO_ERROR;
}

int64_t SetMlpResidualAddNanToNum(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node nanToNumNode;
    atb_speed::common::AclNNNanToNumParam NanToNumParam;
    NanToNumParam.posInfValue = FLOAT16_MAX;  // replaces positive infinity values in tensor elements
    NanToNumParam.negInfValue = FLOAT16_MIN;  // replaces negative infinity values in tensor elements
    nanToNumNode.operation = new atb_speed::common::InplaceNanToNumOperation("nanToNumNode1", NanToNumParam);
    std::vector<std::string> mlpResidualAddOutTensorNames = {};
    if (param.hasDenseTp) {
        mlpResidualAddOutTensorNames = {param.mapping.Get(base::ATTN_TP).IsEnabled() ? \
            "intermediate_dense_tp_rs_addout" : "out_decoder_layer"};
    } else {
        mlpResidualAddOutTensorNames = {param.ffnAllGather || param.ffnReduceScatter ? \
                          ((param.enableQkvdownDp && !param.isLastLayer && !param.isCloudLastLayer) ? \
                          "out_decoder_layer" : "intermediate_mlp_out") : "out_decoder_layer"};
    }

    nanToNumNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
    nanToNumNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddOutTensorNames);
    opGraph.nodes.push_back(nanToNumNode);
    ATB_SPEED_LOG_DEBUG("create nan to num");
    return atb::NO_ERROR;
}

atb::Status SetTPAllGatherNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    if (param.hasDenseTp) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::DENSE_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else if (!param.isLastLayer || (param.isLastLayer && param.enableDpOut && !param.lmHeadLocalTp)) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else if (param.isLastLayer && param.enableDpOut && param.lmHeadLocalTp) {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    } else {
        atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::MLP_TP);
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    }

    CreateOperation(allGatherParam, &allGatherNode.operation);

    if (param.hasDenseTp) {
        allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
            {"intermediate_dense_tp_rs_addout"});
    } else {
        allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
            {param.hasAttnComm ?
            "intermediate_mlp_out" : "intermediate_moe_out_with_shared_with_padding"});
    }
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out_all"});

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

atb::Status SetFFNUnPadding(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    if (param.hasDenseTp) {
        gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
            {"intermediate_mlp_out_all", "in_dense_tp_mlp_rs_out_idx"});
    } else {
        gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
            {param.ffnAllGather ? "intermediate_mlp_out_all" : "intermediate_mlp_out",
            (param.isLastLayer && (!param.enableDpOut || (param.enableDpOut && param.lmHeadLocalTp))) ? \
                "in_lm_head_skip_padding_token_indices" : "in_ffn_unpadding_idx"});
    }

        gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
    // intermediate_layer_out
    if (param.ffnAllGather) {
        gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
        gatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 2; // 2: dimNum
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
        };
    }
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status CalculateDataPartition(DecoderLayerParam &param)
{
    // ATTN
    param.attnStreamNum = param.mapping.Get(base::ATTN_DP).rankIds.size() *
        param.mapping.Get(base::ATTN_CP).rankIds.size();
    // FFN
    if (param.isDenseLayer) {
        if (param.enableDenseTp) {
            param.ffnStreamNum = param.mapping.worldSize_ / param.mapping.Get(base::DENSE_TP).rankIds.size();
        } else {
            param.ffnStreamNum = param.mapping.Get(base::ATTN_DP).rankIds.size() *
                param.mapping.Get(base::ATTN_CP).rankIds.size();
        }
    } else {
        if (param.isDynamicEp) {
            param.ffnStreamNum = param.mapping.Get(base::MOE_EP).rankIds.size() *
            param.mapping.Get(base::MOE_TP).rankIds.size();
        } else {
            param.ffnStreamNum = 1; // 暂不支持MoE DP
        }
    }
    // Lmhead
    param.lmheadStreamNum = 1; // Lmhead DP使用
    ATB_SPEED_LOG_DEBUG("CalculateDataPartition done"
        << ". Attention Stream Num is " << param.attnStreamNum
        << " . FFN Stream Num is " << param.ffnStreamNum
        << " . lmheadStreamNum Stream Num is " << param.lmheadStreamNum);
    return atb::NO_ERROR;
}

atb::Status CalculateCommType(DecoderLayerParam &param)
{
    if (param.worldSize == 1) {
        return atb::NO_ERROR;
    }
    int outStreamNum = (param.isLastLayer && (!param.enableDpOut || (param.enableDpOut && param.lmHeadLocalTp))) ? \
        param.lmheadStreamNum : param.attnStreamNum;

    param.hasDenseTp = param.enableDenseTp && param.isDenseLayer;

    param.attnAllreduce = param.mapping.Get(base::ATTN_TP).IsEnabled() &&
                            param.ffnStreamNum == param.attnStreamNum ? true : false;

    param.attnReduceScatter = !param.attnAllreduce && param.mapping.Get(base::ATTN_TP).IsEnabled() ? true : false;

    param.attnAllGather = (param.attnReduceScatter && param.worldSize > param.ffnStreamNum) || \
        (param.attnStreamNum > param.ffnStreamNum) ?
        true : false;
    param.ffnAllreduce = (param.hasDenseTp && !param.mapping.Get(base::ATTN_TP).IsEnabled()) || \
        (param.attnAllreduce && param.ffnStreamNum == param.attnStreamNum) ? true : false;

    param.ffnReduceScatter = !param.ffnAllreduce && param.attnAllGather ? true : false;

    int ffnOutStreamNum = param.ffnReduceScatter ? param.mapping.worldSize_ : param.ffnStreamNum;
    param.ffnAllGather = ffnOutStreamNum > outStreamNum ? true : false;

    param.hasAttnComm = param.attnReduceScatter || param.attnAllGather;
    param.hasFfnComm = param.ffnReduceScatter || param.ffnAllGather;
    ATB_SPEED_LOG_DEBUG("CalculateCommType done"
        << ". outStreamNum is " << outStreamNum
        << ". attnAllreduce is " << param.attnAllreduce << " . attnReduceScatter is " << param.attnReduceScatter
        << " . attnAllGather is " << param.attnAllGather
        << " . ffnAllreduce is " << param.ffnAllreduce << " . ffnReduceScatter is " << param.ffnReduceScatter
        << " . ffnAllGather is " << param.ffnAllGather);
    return atb::NO_ERROR;
}

atb::Status CreateNewStreamRecordWithoutNodeId(atb::GraphParam &opGraph, atb_speed::EventAction eventAction,
    const std::string &cvKey)
{
    atb::Node recordNode;
    recordNode.inTensorIds = {};
    recordNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        recordNode.operation,
        eventAction,
        cvKey));
    atb::SetExecuteStreamId(recordNode.operation, STREAM1);
    opGraph.nodes.push_back(recordNode);
    ATB_SPEED_LOG_DEBUG("Record event success");
    return atb::NO_ERROR;
}

atb::Status CreateNewStreamWaitWithoutNodeId(atb::GraphParam &opGraph, atb_speed::EventAction eventAction,
    const std::string &cvKey)
{
    atb::Node waitNode;
    waitNode.inTensorIds = {};
    waitNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        waitNode.operation,
        eventAction,
        cvKey));
    atb::SetExecuteStreamId(waitNode.operation, STREAM1);
    opGraph.nodes.push_back(waitNode);
    ATB_SPEED_LOG_DEBUG("Wait event success");
    return atb::NO_ERROR;
}

atb::Status SetFFN(std::map<std::string, uint32_t> &tensorMap,
    const DecoderLayerParam &param, atb::GraphParam &opGraph)
{
    if (param.isDenseLayer) {
        CHECK_OPERATION_STATUS_RETURN(SetMlpExpert(opGraph, param, tensorMap));
    } else {
        if (param.hasSharedExpert && !param.enableSharedExpertOverlap) {
            CHECK_OPERATION_STATUS_RETURN(SetSharedExpert(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetMoe(opGraph, param, tensorMap));
        if (param.hasSharedExpert && !param.enableSharedExpertDp) {
            CHECK_OPERATION_STATUS_RETURN(AddExpertAdd(opGraph, param, tensorMap));
        }
    };
    return atb::NO_ERROR;
}

atb::Status SetCast(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node castNode;
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_selfattention_norm_out_fp32")};
    castNode.outTensorIds = {
        atb_speed::common::GetTensorIdx(tensorMap, param.enableExtraOprojTp ?
            "intermediate_selfattention_norm_out" : "intermediate_selfattention_norm_out_partial")};

    opGraph.nodes.push_back(castNode);
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

atb::Status SetGatherPreNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node gatherNormNode;
    atb::infer::GatherPreRmsNormParam gatherRmsNormParam;
    gatherRmsNormParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherRmsNormParam, &gatherNormNode.operation));

    std::vector<std::string> outTensorNames;
    std::vector<std::string> inTensorNames;

    outTensorNames.push_back("intermediate_selfattention_norm_out_fp32");
    outTensorNames.push_back("intermediate_attention_add_out");
    uint32_t obfuscationIdx = 0;
    if (param.enableExtraOprojTp) {
        if (param.hasDenseTp) {
            inTensorNames.push_back("intermediate_dense_tp_oproj_attn_pad");
            inTensorNames.push_back("in_hidden_states");
        } else {
            inTensorNames.push_back(param.enableModelConfuscation ?
                "intermediate_pmcc" : "in_hidden_states");
            inTensorNames.push_back("intermediate_attention_out");
        }
    } else {
        if (param.attnReduceScatter) {
            inTensorNames.push_back(param.enableOutLcocTp ?
                "intermediate_attention_out" : "intermediate_attention_out_scatter");
        } else {
            inTensorNames.push_back(param.enableOutLcocTp ?
                "intermediate_attention_out" : "intermediate_attention_out_padding");
        }
        inTensorNames.push_back(param.enableModelConfuscation ?
            "intermediate_pmcc" : "in_hidden_states");
        obfuscationIdx = 1; // 1: pmcc tensor idx
    }
    inTensorNames.push_back(param.hasDenseTp ? "in_dense_tp_gather_prenorm_idx" : "in_attention_padding_idx_slice");

    if (param.normHasBias) { // FP
        inTensorNames.push_back("in_selfattention_out_norm_weight");
        inTensorNames.push_back("in_selfattention_out_new_norm_bias");
    } else {
        if (param.isAntiOutlier) {
            inTensorNames.push_back("in_selfattention_out_new_norm_weight");
        } else {
            inTensorNames.push_back("in_selfattention_out_norm_weight");
        }
    }

    gatherNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, inTensorNames);
    gatherNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, outTensorNames);
    ObfuscationReshape(param, gatherNormNode, obfuscationIdx);

    opGraph.nodes.push_back(gatherNormNode);
    ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");

    return atb::NO_ERROR;
}

atb::Status SetPreNorm(atb::GraphParam &opGraph, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node preNormNode;
    atb::infer::RmsNormParam preRmsNormParam;
    preRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
    preRmsNormParam.preNormParam.epsilon = param.normEps;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(preRmsNormParam, &preNormNode.operation));
    
    std::vector<std::string> outTensorNames;
    std::vector<std::string> inTensorNames;

    outTensorNames.push_back("intermediate_selfattention_norm_out_partial");
    outTensorNames.push_back("intermediate_attention_add_out");

    if (param.attnReduceScatter) {
        inTensorNames.push_back(param.enableOutLcocTp ?
            "intermediate_attention_out" : "intermediate_attention_out_scatter");
    } else {
        inTensorNames.push_back(param.enableOutLcocTp ?
            "intermediate_attention_out" : "intermediate_attention_out_padding");
    }
    inTensorNames.push_back(param.enableModelConfuscation ?
        "intermediate_pmcc" : "in_hidden_states");
    if (param.normHasBias) { // FP
        inTensorNames.push_back("in_selfattention_out_norm_weight");
        inTensorNames.push_back("in_selfattention_out_new_norm_bias");
    } else {
        if (param.isAntiOutlier) {
            inTensorNames.push_back("in_selfattention_out_new_norm_weight");
        } else {
            inTensorNames.push_back("in_selfattention_out_norm_weight");
        }
    }

    preNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, inTensorNames);
    preNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, outTensorNames);
    ObfuscationReshape(param, preNormNode, 1); // 1: pmcc tensor idx
    opGraph.nodes.push_back(preNormNode);
    ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");

    return atb::NO_ERROR;
}

atb::Status AddObfuscationCalculateNode(const DecoderLayerParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::ATTN_TP);
    bool isMultiRank = parallelInfo.rankIds.size() > 1;
    int rank = parallelInfo.rank;
    int32_t hiddenSizePerRank = CheckIntMulOverFlow(param.hiddenSizePerAttentionHead,
        param.numAttentionHeadsPerRank);
    if (isMultiRank) {
        atb::Node sliceNode;
        atb::infer::SliceParam sliceParam;
        sliceParam.offsets = {0, rank * hiddenSizePerRank};
        sliceParam.size = {-1, hiddenSizePerRank};
        sliceNode.inTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states"));
        sliceNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap, "intermediate_pmcc_per_rank"));
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(sliceParam, &sliceNode.operation));
        opGraph.nodes.push_back(sliceNode);
    }
    atb::Node obfCalNode;
    atb_speed::common::ObfuscationCalculateParam obfCalParam;
    obfCalParam.fd = param.modelConfuscationFd;
    obfCalParam.hiddenSizePerRank = hiddenSizePerRank;
    obfCalParam.obfCoefficient = 0.5; // 0.5: 混淆因子
    obfCalNode.inTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap,
        isMultiRank ? "intermediate_pmcc_per_rank" : "in_hidden_states"));
    obfCalNode.outTensorIds.push_back(atb_speed::common::GetTensorIdx(tensorMap,
        isMultiRank ? "intermediate_pmcc_gather" : "intermediate_pmcc"));
    obfCalNode.operation = new atb_speed::common::ObfuscationCalculateOperation("ObfCalculateOperation", obfCalParam);
    opGraph.nodes.push_back(obfCalNode);
    if (isMultiRank) {
        atb::Node allGatherNode;
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = parallelInfo.rank;
        allGatherParam.rankSize = parallelInfo.rankIds.size();
        allGatherParam.backend = parallelInfo.defaultBackend;
        parallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
        allGatherNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_pmcc_gather")};
        allGatherNode.outTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_pmcc_gather_out")};
        opGraph.nodes.push_back(allGatherNode);

        atb::Node transposeNode;
        atb::infer::TransposeParam transposeParam;
        transposeParam.perm = {1, 0, 2};
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
        transposeNode.inTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_pmcc_gather_out")};
        transposeNode.outTensorIds = {atb_speed::common::GetTensorIdx(tensorMap, "intermediate_pmcc")};
        opGraph.nodes.push_back(transposeNode);
    }
    return atb::NO_ERROR;
}

atb::Status SetPostAttnProcess(std::map<std::string, uint32_t> &tensorMap,
    const DecoderLayerParam &param, atb::GraphParam &opGraph)
{
    if (param.enableModelConfuscation) {
        CHECK_OPERATION_STATUS_RETURN(AddObfuscationCalculateNode(param, opGraph, tensorMap));
    }
    if (param.hasAttnComm) {
        if (!param.enableOutLcocTp) {
            CHECK_OPERATION_STATUS_RETURN(SetPadding(opGraph, tensorMap, param));
        }
        if (!param.enableGatherPreNorm) {
            CHECK_OPERATION_STATUS_RETURN(SetResidualPadding(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetResidualSliceNode(opGraph, param, tensorMap));
        }
    }
    if (param.attnReduceScatter && !param.enableOutLcocTp) {
        CHECK_OPERATION_STATUS_RETURN(SetAttnReduceScatter(opGraph, param, tensorMap));
    }
    if ((param.hasAttnComm && param.enableGatherPreNorm) || param.enableExtraOprojTp) {
        // h3p qkvdown dp move moe allgather+gather to mla, without first moe
        if (param.enableQkvdownDp && param.layerId > param.firstKDenseReplace) {
            CHECK_OPERATION_STATUS_RETURN(SetPreNorm(opGraph, param, tensorMap));
        } else {
            CHECK_OPERATION_STATUS_RETURN(SetGatherPreNorm(opGraph, param, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetCast(opGraph, param, tensorMap));
        }
    } else {
        if (!param.enableIntraLayerAddNorm) {
            CHECK_OPERATION_STATUS_RETURN(SetSelfResidualAdd(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetSelfNorm(opGraph, param, tensorMap));
    }
    if (param.enableSharedExpertOverlap) {
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
            opGraph, atb_speed::EventAction::PUSH, atb_speed::common::CC_START));
        CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
            opGraph, atb_speed::EventAction::POP, atb_speed::common::CC_START));
    }
    if (param.enableSharedExpertOverlap && !param.isDenseLayer && param.hasSharedExpert) {
        CHECK_OPERATION_STATUS_RETURN(SetSharedExpert(opGraph, param, tensorMap));
        if (!param.isPrefill || !param.enableGatingDp) {
            CHECK_OPERATION_STATUS_RETURN(CreateNewStreamRecordWithoutNodeId(
                opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMP_CONTROL));
            CHECK_OPERATION_STATUS_RETURN(CreateNewStreamWaitWithoutNodeId(
                opGraph, atb_speed::EventAction::PUSH, atb_speed::common::COMM_CONTROL));
        }
    }
    if (param.attnAllGather) {
        CHECK_OPERATION_STATUS_RETURN(SetAllGather(opGraph, param, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(SetAllGatherCCOverlap(opGraph, param));
    }
    if (param.hasAttnComm) {
        if (!param.hasDenseTp) {
            CHECK_OPERATION_STATUS_RETURN(SetAttnUnpadding(opGraph, param, tensorMap));
        }
    }
    return atb::NO_ERROR;
}

atb::Status SetMlaPrefetch(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node computeRecordNode;
    computeRecordNode.inTensorIds = {};
    computeRecordNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        computeRecordNode.operation,
        atb_speed::EventAction::PUSH,
        atb_speed::common::CMO_MLAPO));
    opGraph.nodes.push_back(computeRecordNode);
 
    atb::Node commWaitNode;
    commWaitNode.inTensorIds = {};
    commWaitNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        commWaitNode.operation,
        atb_speed::EventAction::POP,
        atb_speed::common::CMO_MLAPO));
    atb::SetExecuteStreamId(commWaitNode.operation, 1);
    opGraph.nodes.push_back(commWaitNode);
 
    atb::Node cmoNode1;
    cmoNode1.operation = new atb_speed::common::AclrtCmoAsyncOperation("AclrtCmoAsync");
    cmoNode1.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {
        "next_layer_in_q_proj_a_weight"
    });
    cmoNode1.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {});
    atb::SetExecuteStreamId(cmoNode1.operation, 1);
    opGraph.nodes.push_back(cmoNode1);
 
    atb::Node cmoNode2;
    cmoNode2.operation = new atb_speed::common::AclrtCmoAsyncOperation("AclrtCmoAsync");
    cmoNode2.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {
        "next_layer_in_k_proj_b_for_q_weight"
    });
    cmoNode2.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {});
    atb::SetExecuteStreamId(cmoNode2.operation, 1);
    opGraph.nodes.push_back(cmoNode2);
 
    return atb::NO_ERROR;
}

atb::Status SetPostMoeProcess(std::map<std::string, uint32_t> &tensorMap,
    const DecoderLayerParam &param, atb::GraphParam &opGraph)
{
    if (param.hasFfnComm && param.hasAttnComm && !param.hasDenseTp) {
        CHECK_OPERATION_STATUS_RETURN(SetFFNPadding(opGraph, param, tensorMap));
    }
    if (param.ffnAllreduce) {
        CHECK_OPERATION_STATUS_RETURN(SetAllReduce(opGraph, param, tensorMap));
        if (param.hasDenseTp) {
            CHECK_OPERATION_STATUS_RETURN(SetAttnAddOutUnpad(opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetMlpOutUnpad(opGraph, tensorMap));
        }
    } else if (param.ffnReduceScatter) {
        CHECK_OPERATION_STATUS_RETURN(SetMlpReduceScatter(opGraph, param, tensorMap));
        if (param.enableInfNan) {
            CHECK_OPERATION_STATUS_RETURN(SetMlpReduceScatterNanToNum(opGraph, tensorMap));
        }
        if (param.enableSharedExpertDp) {
            CHECK_OPERATION_STATUS_RETURN(AddExpertAdd(opGraph, param, tensorMap));
        }
    }
    ATB_SPEED_LOG_DEBUG("enableMlaPrefetch: " << param.enableMlaPrefetch);
    if (param.enableMlaPrefetch) {
        CHECK_OPERATION_STATUS_RETURN(SetMlaPrefetch(opGraph, tensorMap));
        ATB_SPEED_LOG_DEBUG("set mla prefetch success");
    }
    CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAdd(opGraph, param, tensorMap));
    if (param.enableInfNan) {
        CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNanToNum(opGraph, param, tensorMap));
    }
    if (param.ffnAllGather) {
        if (!param.hasAttnComm) {
            CHECK_OPERATION_STATUS_RETURN(SetFFNPadding(opGraph, param, tensorMap));
        }
        // h3p qkvdown dp move moe allgather to mla, without last moe
        if (!param.enableQkvdownDp || param.isLastLayer || param.isCloudLastLayer) {
            CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(opGraph, param, tensorMap));
        }
    }
    if (param.hasFfnComm) {
        // h3p qkvdown dp move moe gather to mla, without last moe
        if (!param.enableQkvdownDp || param.isLastLayer || param.isCloudLastLayer) {
            CHECK_OPERATION_STATUS_RETURN(SetFFNUnPadding(opGraph, param, tensorMap));
        }
    }
    return atb::NO_ERROR;
}

atb::Status DecoderLayer(DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    CalculateDataPartition(param);
    CalculateCommType(param);
    param.enableQkvdownDp = param.enableQkvdownDp && param.ffnAllGather;
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph inTensorNum: " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph outTensorNum: " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum: " << opGraph.internalTensorNum);
    CHECK_OPERATION_STATUS_RETURN(SetAttention(opGraph, param, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(SetPostAttnProcess(tensorMap, param, opGraph));
    CHECK_OPERATION_STATUS_RETURN(SetFFN(tensorMap, param, opGraph));
    CHECK_OPERATION_STATUS_RETURN(SetPostMoeProcess(tensorMap, param, opGraph));
    opGraph.inferShapeFunc = [=] (const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (((param.mapping.Get(base::ATTN_DP).IsEnabled() || param.mapping.Get(base::ATTN_CP).IsEnabled()) && \
            param.isLastLayer && !param.enableDpOut) || (param.enableQkvdownDp && param.isCloudLastLayer)) {
            outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_final_state"));
        } else if (param.mapping.Get(base::ATTN_DP).IsEnabled() && param.isLastLayer && \
            param.enableDpOut && param.lmHeadLocalTp) {
            outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_final_state"));
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, \
                "in_lm_head_skip_padding_token_indices")).shape.dims[0];
        } else {
            outTensorDescs.at(0) = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states"));
        }

        if (param.enableQkvdownDp && param.layerId == param.firstKDenseReplace) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(
                atb_speed::common::GetTensorIdx(tensorMap, "in_ffn_padding_idx")
            ).shape.dims[0];
            if (!param.isDynamicEp && param.mapping.Get(base::MLP_TP).rankIds.size() != 0) {
                outTensorDescs.at(0).shape.dims[0] /= param.mapping.Get(base::MLP_TP).rankIds.size();
            }
        }

        if (!param.isDenseLayer && param.enableExpertCumSumOutput && param.enableTopkOutput) {
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, 1, param.numOfDeviceExperts);
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, 2,
                param.numOfSelectedExperts.at(0), false);
        } else if (!param.isDenseLayer && param.enableExpertCumSumOutput) {
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, 1, param.numOfDeviceExperts);
        } else if (!param.isDenseLayer && param.enableTopkOutput) {
            atb_speed::common::SetOutTensorDescsForEPLB(outTensorDescs, 1,
                param.numOfSelectedExperts.at(0), false);
        }
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

DecoderLayer::DecoderLayer() {}

DecoderLayer::~DecoderLayer() {}

} // namespace deepseekV2
} // namespace atb_speed

