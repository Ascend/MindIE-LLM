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
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/dequant_rope_quant_kvcache_operation.h"
#include "operations/aclnn/ops/rotary_pos_emb_operation.h"
#include "operations/fusion/attention/qkv_linear_split.h"
#include "operations/fusion/attention/self_attention.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/fusion/attention/fusion_attention.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetAttnInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> attnInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
            "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
            "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
            "in_weight_2", "in_scale_2", "in_offset_2", "in_descale_2", "in_bias_2", "in_compress_idx_2",
            "in_cos_embed", "in_sin_embed", "in_seq_len", "in_k_cache", "in_v_cache", "in_attention_mask",
            "in_token_offset", "in_layer_id", "in_block_tables",
            "in_slots_in_pa_or_logn_in_fa",
            "in_weight_dense", "in_scale_dense", "in_offset_dense", "in_descale_dense", "in_bias_dense",
            "in_compress_idx_dense"}
        },
        {"alibi_mask_compress", {"in_slopes"}},
        {"compress_head_alibi", {"in_batch_wins", "in_ra_seq_len"}},
        {"compress_head_rope",
            {"in_batch_wins", "in_ra_seq_len", "in_pffset_index", "in_ra_offset", "in_reshape_seq_len"}},
        {"speculate", {"in_q_len"}},
        {"kv_quant_scale",
            {"in_k_quant_scale", "in_k_dequant_scale", "in_v_quant_scale", "in_v_dequant_scale"}
        },
        {"kv_quant_offset",
            {"in_k_quant_offset", "in_k_dequant_offset", "in_v_quant_offset", "in_v_dequant_offset"}
        },
        {"fa3_quant",
            {"in_q_quant_scale", "in_k_quant_scale", "in_v_quant_scale", "in_qk_descale",
            "q_offset", "kv_offset", "fa3_v_quant_scale", "fa3_offset"}
        },
        {"reduce_quant",
            {"in_reduce_quant_scale", "in_reduce_quant_offset", "in_gather_quant_scale", "in_gather_quant_offset"}
        },
        {"lora", {
            "in_seq_len_cum_sum", "in_lora_a_0", "in_lora_b_0", "in_lora_a_1", "in_lora_b_1",
            "in_lora_a_2", "in_lora_b_2", "in_dense_lora_a", "in_dense_lora_b"}
        },
        {"lora_with_mask", {"in_im_mask"}},
        {"log_n_scale", {"in_log_n_scale"}},
        {"qk_norm", {"in_q_norm_weight", "in_k_norm_weight"}},
        {"add_norm", {"in_residual_add"}},
        {"add_rmsnorm_quant", {"in_qkv_scale_fill", "in_qkv_offset_fill"}},
        {"cmo_mlp_first_matmul_weight", {"in_mlp_weight_0"}},
        {"flash_comm", {
            "send_counts", "sdispls", "send_count", "recv_counts", "rdispls", "recv_count",
            "fake_rs_shape", "fake_ag_shape"}
        },
    };
    return attnInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAttnIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> attnIntermediateTensorCandidates = {
        {"default",
            {"intermediate_q", "intermediate_k", "intermediate_v", "intermediate_self_attention"}
        },
        {"kv_quant_scale",
            {"intermediate_k_int8", "intermediate_v_int8"}
        },
        {"q_quant_scale",
            {"intermediate_q_int8"}
        },
        {"dequant_rope",
            {"intermediate_qkv_rope"}
        }
    };
    return attnIntermediateTensorCandidates;
}

template <typename NormParamType>
atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>> &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList, std::vector<std::string> &intermediateTensorList)
{
    // 添加KV cache int8特性的Tensor
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        AddTensorToList(attnInTensorCandidates, "kv_quant_scale", inTensorList);
        if (!param.enableRopeQuantKvcache) {
            AddTensorToList(attnIntermediateTensorCandidates, "kv_quant_scale", intermediateTensorList);
        }
        if (param.pageAttentionParam.hasQuantOffset) {
            AddTensorToList(attnInTensorCandidates, "kv_quant_offset", inTensorList);
        }
    }

    // 添加FA3特性的Tensor
    if (!param.isPrefill && param.pageAttentionParam.quantType == \
        atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        AddTensorToList(attnIntermediateTensorCandidates, "q_quant_scale", intermediateTensorList);
    }
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        AddTensorToList(attnInTensorCandidates, "fa3_quant", inTensorList);
        AddTensorToList(attnIntermediateTensorCandidates, "kv_quant_scale", intermediateTensorList);
    }
    return atb::NO_ERROR;
}


template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto attnInTensorCandidates = GetAttnInTensorCandidates();
    auto attnIntermediateTensorCandidates = GetAttnIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(attnInTensorCandidates, "default", inTensorList);
    // 添加AddRmsNormQuant特性的Tensor
    if (param.enableAddNorm) {
        AddTensorToList(attnInTensorCandidates, "add_rmsnorm_quant", inTensorList);
    }
    AddTensorToList(attnIntermediateTensorCandidates, "default", intermediateTensorList);

    // 添加Mask Alibi特性的Tensor
    if (
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN
    ) {
        AddTensorToList(attnInTensorCandidates, "alibi_mask_compress", inTensorList);
    }

    // 添加多头压缩特性的Tensor
    if (param.pageAttentionParam.compressType == atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD) {
        AddTensorToList(attnInTensorCandidates, "compress_head_alibi", inTensorList);
    } else if (param.pageAttentionParam.compressType ==
                    atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE) {
        AddTensorToList(attnInTensorCandidates, "compress_head_rope", inTensorList);
    }
    // 添加并行解码特性的Tensor
    if (param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
        AddTensorToList(attnInTensorCandidates, "speculate", inTensorList);
    }

    ConstructAttentionQuantTensorMap(param, attnInTensorCandidates, attnIntermediateTensorCandidates,
        inTensorList, intermediateTensorList);

    // 添加lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(attnInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(attnInTensorCandidates, "lora", inTensorList);
    }
    // 添加lccl all reduce int8特性的Tensor
    if (param.selfOutLinearTensorParallelInfo.quantType != \
        atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        AddTensorToList(attnInTensorCandidates, "reduce_quant", inTensorList);
    }

    // 添加logN attention特性
    if (param.pageAttentionParam.scaleType == atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN) {
        AddTensorToList(attnInTensorCandidates, "log_n_scale", inTensorList);
    }

    // 添加 qk_norm 的 Tensor
    if (param.useQKNorm) {
        AddTensorToList(attnInTensorCandidates, "qk_norm", inTensorList);
    }

    // 添加add norm融合的Tensor
    if (param.enableAddNorm) {
        AddTensorToList(attnInTensorCandidates, "add_norm", inTensorList);
    }

    if (param.enableAddNorm) {
        outTensorList.push_back("out_add");
    }

    // 添加cmo特性的Tensor
    if (param.preFetchWeightSize) {
        AddTensorToList(attnInTensorCandidates, "cmo_mlp_first_matmul_weight", inTensorList);
    }

    // 添加 dequant rope tensor
    if (param.enableRopeQuantKvcache) {
        AddTensorToList(attnIntermediateTensorCandidates, "dequant_rope", intermediateTensorList);
    }

    // Add flashcomm1.0 Tensor
    if (param.enableFlashComm) {
        AddTensorToList(attnInTensorCandidates, "flash_comm", inTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
atb::Status AddFAttnQKVLinearSplitNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qkvLinearSplitNode;
    CHECK_OPERATION_STATUS_RETURN(QKVLinearSplit(param, &qkvLinearSplitNode.operation));
    std::vector<std::string> qkvInTensorNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
        "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
        "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
        "in_weight_2", "in_scale_2", "in_offset_2", "in_descale_2", "in_bias_2", "in_compress_idx_2",
    };
    // 添加AddRmsNormQuant特性的Tensor
    if (param.enableAddNorm) {
        qkvInTensorNames.push_back("in_qkv_scale_fill");
        qkvInTensorNames.push_back("in_qkv_offset_fill");
        qkvInTensorNames.push_back("in_residual_add");
    }
    if (param.supportLora) {
        if (param.useImMask) {
            qkvInTensorNames.push_back("in_im_mask");
        }
        qkvInTensorNames.push_back("in_seq_len_cum_sum");
        qkvInTensorNames.push_back("in_lora_a_0");
        qkvInTensorNames.push_back("in_lora_b_0");
        qkvInTensorNames.push_back("in_lora_a_1");
        qkvInTensorNames.push_back("in_lora_b_1");
        qkvInTensorNames.push_back("in_lora_a_2");
        qkvInTensorNames.push_back("in_lora_b_2");
    }
    if (param.useQKNorm) {
        qkvInTensorNames.push_back("in_q_norm_weight");
        qkvInTensorNames.push_back("in_k_norm_weight");
    }
    if (param.enableFlashComm) {
        qkvInTensorNames.push_back("send_counts");
        qkvInTensorNames.push_back("sdispls");
        qkvInTensorNames.push_back("send_count");
        qkvInTensorNames.push_back("recv_counts");
        qkvInTensorNames.push_back("rdispls");
        qkvInTensorNames.push_back("recv_count");
        qkvInTensorNames.push_back("fake_ag_shape");
    }
    qkvLinearSplitNode.inTensorIds = GetTensorIdxList(tensorMap, qkvInTensorNames);
    std::vector<std::string> qkvOutTensorNames = {"intermediate_q", "intermediate_k", "intermediate_v"};
    
    if (param.enableRopeQuantKvcache) { // 3 -> 1
        qkvOutTensorNames = {"intermediate_qkv_rope"};
    }
    if (param.enableAddNorm) {
        qkvOutTensorNames.push_back("out_add");
    }
    qkvLinearSplitNode.outTensorIds = GetTensorIdxList(tensorMap, qkvOutTensorNames);
    opGraph.nodes.push_back(qkvLinearSplitNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddRopeQuantKvcacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    atb::Node dequantRopeQuantKvcacheNode;
    AclNNDequantRopeQuantKvcacheParam aclnnParam;

    int64_t sizeSpiltsZero = CheckIntMulOverFlow(param.selfAttentionParam.headNum, param.headDim);
    int64_t sizeSpiltsOne = CheckIntMulOverFlow(param.selfAttentionParam.kvHeadNum, param.headDim);
    aclnnParam.sizeSpilts = {sizeSpiltsZero, sizeSpiltsOne, sizeSpiltsOne};

    aclnnParam.kvOutput = true;
    aclnnParam.quantMode = "static";
    aclnnParam.layout = "BSND";
    LinearQuantType quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[Q_LINEAR_INDEX], param.enableNormQuantOp);
    aclnnParam.enableDequant = (!param.isPrefill && param.isBF16 && \
                                (quantType == LINEAR_W8A8_QUANT || quantType == LINEAR_W8A8_DEQUANT));
    dequantRopeQuantKvcacheNode.operation = new atb_speed::common::DequantRopeQuantKvcacheOperation(
        "aclnnDequantRopeQuantKvcacheNode", aclnnParam
    );
    dequantRopeQuantKvcacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_qkv_rope"), // 1: input_x
        GetTensorIdx(tensorMap, "in_cos_embed"), // 2: cos
        GetTensorIdx(tensorMap, "in_sin_embed"), // 3: sin
        GetTensorIdx(tensorMap, "in_k_cache"), // 4: k_cache
        GetTensorIdx(tensorMap, "in_v_cache"), // 5: v_cache
        GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"), // 6: indices
        GetTensorIdx(tensorMap, "in_k_quant_scale"), // 7: scale_k
        GetTensorIdx(tensorMap, "in_v_quant_scale"), // 8: scale_v
        GetTensorIdx(tensorMap, "in_k_quant_offset"), // 9: offset_k
        GetTensorIdx(tensorMap, "in_v_quant_offset"), // 10: offset_v
    };
    if (aclnnParam.enableDequant) {
        dequantRopeQuantKvcacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_descale_0")); // 11: weight_scale
        dequantRopeQuantKvcacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias_0")); // 12: bias
    }
    dequantRopeQuantKvcacheNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"), // q_out // 1024, 8, 128
        GetTensorIdx(tensorMap, "intermediate_k"), // k_out // 1024, 1, 128
        GetTensorIdx(tensorMap, "intermediate_v"), // v_out // 1024, 1, 128
    };
    opGraph.nodes.push_back(dequantRopeQuantKvcacheNode);

    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddFAttnRopeNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node ropeNode;
    atb_speed::common::RotaryPositionEmbeddingParam ropeParam;
    ropeParam.rotaryType = param.rotaryType;
    ropeParam.isFA = param.isFA;
    ropeParam.headDim = param.headDim;
    ropeParam.headNum = param.selfAttentionParam.headNum;
    ropeParam.kvHeadNum = param.selfAttentionParam.kvHeadNum;
    ropeParam.ropeParam = param.ropeParam;

    if (param.ropeBackend == atb_speed::common::OpBackend::ATB) {
        RotaryPositionEmbedding(ropeParam, &ropeNode.operation);
        ropeNode.inTensorIds = {  // [B,S,N,D] PA [BS,ND]
            GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "intermediate_k"),
            GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
            GetTensorIdx(tensorMap, "in_seq_len")
        };
        if (!param.isFA) {
            ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
            ropeNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
            ropeNode.inTensorReshapeFuncs.at(1) = &SqueezeHeadNumHeadDim;
        }
        ropeNode.outTensorIds = {  // FA [B,S,N,D] PA [BS,N,D]
            GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "intermediate_k"),
        };
    } else {
        ropeNode.operation = new RotaryPosEmbOperation("RotaryPosEmbNode");
        ropeNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed")
        };
    }
    opGraph.nodes.push_back(ropeNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddKVValueQuantNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isK)
{
    atb::Node kvValueQuantNode;
    atb::infer::ElewiseParam kvValueQuantParam;
    kvValueQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CREATE_OPERATION(kvValueQuantParam, &kvValueQuantNode.operation);
    if (isK) {
        kvValueQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_k"), GetTensorIdx(tensorMap, "in_k_quant_scale"),
            GetTensorIdx(tensorMap, "in_k_quant_offset"),
        };
        kvValueQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_int8")};
    } else {
        kvValueQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_v"), GetTensorIdx(tensorMap, "in_v_quant_scale"),
            GetTensorIdx(tensorMap, "in_v_quant_offset"),
        };
        kvValueQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_int8")};
    }
    kvValueQuantNode.inTensorReshapeFuncs.resize(kvValueQuantNode.inTensorIds.size());
    kvValueQuantNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
    };
    kvValueQuantNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
    };
    opGraph.nodes.push_back(kvValueQuantNode);
    return atb::NO_ERROR;
}

atb::Status AddQKVQuantNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, std::string nodeType)
{
    atb::Node qkvQuantNode;
    atb::infer::ElewiseParam qkvQuantParam;
    qkvQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CREATE_OPERATION(qkvQuantParam, &qkvQuantNode.operation);
    if (nodeType == "Q") {
        qkvQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "in_q_quant_scale"),
            GetTensorIdx(tensorMap, "q_offset")
        };
        qkvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_int8")};
    } else if (nodeType == "K") {
        qkvQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_k"), GetTensorIdx(tensorMap, "in_k_quant_scale"),
            GetTensorIdx(tensorMap, "kv_offset")
        };
        qkvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_int8")};
    } else if (nodeType == "V") {
        qkvQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_v"), GetTensorIdx(tensorMap, "in_v_quant_scale"),
            GetTensorIdx(tensorMap, "kv_offset")
        };
        qkvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_int8")};
    }
    qkvQuantNode.inTensorReshapeFuncs.resize(qkvQuantNode.inTensorIds.size());
    opGraph.nodes.push_back(qkvQuantNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSelfOutLinearParallelNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfOutLinearParallelNode;
    atb_speed::common::LinearParallelParam selfOutLinearParam;
    if (param.enableFlashComm) {
        selfOutLinearParam.parallelType = atb_speed::common::REDUCE_SCATTER;
    } else {
        selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
    }
    selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.layerLinearQuantType[DENSE_LINEAR_INDEX], false,
        param.layerLinearDescs[DENSE_LINEAR_INDEX]);
    selfOutLinearParam.biasAfterSync = param.selfOutLinearTensorParallelInfo.worldSize > 1 && \
        selfOutLinearParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT && \
        param.selfAttnHasBias;
    selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
    selfOutLinearParam.fusionLinearParam.supportLora = param.supportLora;
    selfOutLinearParam.fusionLinearParam.useImMask = param.useImMask;
    selfOutLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    selfOutLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[DENSE_LINEAR_INDEX];
    selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    selfOutLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    selfOutLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
    selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
    selfOutLinearParam.supportLcoc = param.supportLcoc;
    selfOutLinearParam.enableMC2 = param.enableMC2;
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
    std::vector<std::string> denseInTensorNames = {
        "intermediate_self_attention", "in_weight_dense", "in_scale_dense", "in_offset_dense", "in_descale_dense",
        "in_bias_dense", "in_compress_idx_dense"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            denseInTensorNames.push_back("in_im_mask");
        }
        denseInTensorNames.push_back("in_seq_len_cum_sum");
        denseInTensorNames.push_back("in_dense_lora_a");
        denseInTensorNames.push_back("in_dense_lora_b");
    }
    if (param.selfOutLinearTensorParallelInfo.quantType != \
        atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        denseInTensorNames.push_back("in_reduce_quant_scale");
        denseInTensorNames.push_back("in_reduce_quant_offset");
        denseInTensorNames.push_back("in_gather_quant_scale");
        denseInTensorNames.push_back("in_gather_quant_offset");
    }
    if (selfOutLinearParam.parallelType == atb_speed::common::REDUCE_SCATTER) {
        denseInTensorNames.push_back("send_counts");
        denseInTensorNames.push_back("sdispls");
        denseInTensorNames.push_back("recv_count");
        denseInTensorNames.push_back("fake_rs_shape");
    }
    selfOutLinearParallelNode.inTensorIds = GetTensorIdxList(tensorMap, denseInTensorNames);
    if (!param.isFA) {
        selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
        if (param.attnBackend == atb_speed::common::OpBackend::ACLNN && !param.isPrefill) {
            selfOutLinearParallelNode.inTensorReshapeFuncs.at(0) = &SqueezeBatchAndHiddenSize;
        } else {
            selfOutLinearParallelNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
        }
    }
    selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(selfOutLinearParallelNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddQScaleNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qScaleNode;
    atb::infer::ElewiseParam qScaleParam;
    qScaleParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    qScaleParam.mulsParam.varAttr = 1.0 / sqrt(param.headDim);
    CREATE_OPERATION(qScaleParam, &qScaleNode.operation);
    qScaleNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    qScaleNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    opGraph.nodes.push_back(qScaleNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "Attention";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum " << opGraph.internalTensorNum);

    if (param.layerLinearDescs.size() != 0 && \
        CheckParamVectorSize(param.layerLinearDescs, DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearDescs is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (param.layerLinearQuantType.size() != 0 && \
        CheckParamVectorSize(param.layerLinearQuantType, DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearQuantType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (CheckParamVectorSize(param.layerLinearTransposeType, DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearTransposeType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    
    if (param.enableRopeQuantKvcache) {
        // AddQNormLinearNode only, skip others
        CHECK_OPERATION_STATUS_RETURN(AddFAttnQKVLinearSplitNode(param, opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(AddRopeQuantKvcacheOperation(opGraph, param, tensorMap));
    } else {
        // QKV Node
        CHECK_OPERATION_STATUS_RETURN(AddFAttnQKVLinearSplitNode(param, opGraph, tensorMap));
        
        // Rope Node
        if (param.rotaryType != RotaryType::NO_ROTARY) {
            CHECK_OPERATION_STATUS_RETURN(AddFAttnRopeNode(param, opGraph, tensorMap));
        }

        // QScale Node
        if (param.enableQScale) {
            CHECK_OPERATION_STATUS_RETURN(AddQScaleNode(param, opGraph, tensorMap));
        }

        bool atbAttentionDequant = param.attnBackend == atb_speed::common::OpBackend::ATB && \
            param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION;
        bool aclnnAttentionDequant = param.attnBackend == atb_speed::common::OpBackend::ACLNN && \
            param.aclnnIncreAttentionParam.hasKVQuant;
        if (atbAttentionDequant || aclnnAttentionDequant) {
            // K Quant
            CHECK_OPERATION_STATUS_RETURN(AddKVValueQuantNode(param, opGraph, tensorMap, true));
            // V Quant
            CHECK_OPERATION_STATUS_RETURN(AddKVValueQuantNode(param, opGraph, tensorMap, false));
        }
    }

    // FA3 QKV Quant Node
    if (!param.isPrefill && param.pageAttentionParam.quantType == \
        atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        CHECK_OPERATION_STATUS_RETURN(AddQKVQuantNode(opGraph, tensorMap, "Q"));
    }
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
            CHECK_OPERATION_STATUS_RETURN(AddQKVQuantNode(opGraph, tensorMap, "K"));
            CHECK_OPERATION_STATUS_RETURN(AddQKVQuantNode(opGraph, tensorMap, "V"));
    }

    // SelfAttention Node
    CHECK_OPERATION_STATUS_RETURN(AddSelfAttention(opGraph, param, tensorMap));

    // Dense Node
    CHECK_OPERATION_STATUS_RETURN(AddSelfOutLinearParallelNode(param, opGraph, tensorMap));

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

template atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>> &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList, std::vector<std::string> &intermediateTensorList);
template std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status AddFAttnRopeNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddKVValueQuantNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isK);
template atb::Status AddSelfOutLinearParallelNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddQScaleNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status Attention(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::Operation **operation);

template atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>> &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList, std::vector<std::string> &intermediateTensorList);
template std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status AddFAttnRopeNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddKVValueQuantNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isK);
template atb::Status AddSelfOutLinearParallelNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddQScaleNode(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status Attention(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::Operation **operation);
} // namespace common
} // namespace atb_speed