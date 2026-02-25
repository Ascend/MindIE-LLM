/**
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

#include <atb/atb_infer.h>
#include <cmath>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/w8a16_operation.h"
#include "operations/aclnn/ops/w4a16_operation.h"
#include "operations/aclnn/ops/w4a8_operation.h"
#include "operations/aclnn/ops/w8a8_operation.h"
#include "operations/aclnn/ops/w16a16_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"
#include "operations/aclnn/ops/w16a16sc_operation.h"
#include "operations/aclnn/utils/utils.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/linear/linear.h"

namespace atb_speed {
namespace common {

// 是否为matmulBackend开启下LINEAR_W8A8_QUANT、LINEAR_W8A8_DEQUANT场景
bool IsAclnnPerTensor(const FusionLinearParam &param)
{
    return param.matmulBackend == atb_speed::common::OpBackend::ACLNN &&
        (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_DEQUANT);
}

// 是否使用aclnn的QuantBatchMatmul接口
bool UseQuantBatchMatmul(const FusionLinearParam &param)
{
    // All机型: dynamic、pdmix
    return IsAclnnPerTensor(param) || \
           param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || \
           param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT || \
           param.quantType == LINEAR_W4A8_DYNAMIC_QUANT || \
           param.quantType == LINEAR_W4A8_DYNAMIC_DEQUANT;
}

std::map<std::string, std::vector<std::string>> GetLinearInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearInTensorCandidates = {
        {"default", {
            "in_input", "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"}
        },
        {"lora", {"in_group_list", "in_lora_a", "in_lora_b"}},
        {"lora_with_mask", {"in_im_mask"}},
        {"dynamic_quant", {"dynamic_input_scale"}},
        {"add_swiglu_quant_sacle_in", {"swiglu_quant_input_scale"}},
        {"flash_comm", {
            "send_counts", "sdispls", "send_count", "recv_counts", "rdispls", "recv_count", "fake_ag_shape"}
        },
    };
    return linearInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLinearIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearIntermediateTensorCandidates = {
        {"quant_input", {"intermediate_quant_input"}},
        {"lora", {"intermediate_base_linear_out", "intermediate_lora_a_out", "intermediate_lora_b_out"}},
        {"dynamic_quant", {"intermediate_input_scale"}},
        {"lora_with_mask", {"intermediate_im_mask_out"}},
        {"flashComm", {"intermediate_allgather_out"}},
        {"flashComm_dynamic_quant", {"intermediate_allgather_input_scale_out"}},
    };
    return linearIntermediateTensorCandidates;
}

std::map<std::string, uint32_t> ConstructLinearTensorMap(
    const FusionLinearParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto linearInTensorCandidates = GetLinearInTensorCandidates();
    auto linearIntermediateTensorCandidates = GetLinearIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(linearInTensorCandidates, "default", inTensorList);

    if (!param.enableSwigluQuant || (param.quantType != LINEAR_W8A8_DYNAMIC_DEQUANT
        && param.quantType != LINEAR_W4A8_DYNAMIC_DEQUANT)) {
        // 添加额外的中间Tensor
        if (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT
            || ((param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W4A8_DYNAMIC_QUANT)
            && !param.enableSwiGLUQuantForSharedExperts)) {
            AddTensorToList(linearIntermediateTensorCandidates, "quant_input", intermediateTensorList);
        }
        // 添加动态量化中间Tensor
        if ((param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W4A8_DYNAMIC_QUANT)
            && !param.enableSwiGLUQuantForSharedExperts) {
            AddTensorToList(linearIntermediateTensorCandidates, "dynamic_quant", intermediateTensorList);
        }
    }
    // 添加AddRmsNormDynamicQuant的输出
    if (param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT || param.quantType == LINEAR_W4A8_DYNAMIC_DEQUANT) {
        AddTensorToList(linearInTensorCandidates, "dynamic_quant", inTensorList);
    }

    // 添加SwiGLUQuant的输出
    if (param.enableSwiGLUQuantForSharedExperts) {
        AddTensorToList(linearInTensorCandidates, "add_swiglu_quant_sacle_in", inTensorList);
    }

    if (param.enableFlashComm) {
        AddTensorToList(linearInTensorCandidates, "flash_comm", inTensorList);
        AddTensorToList(linearIntermediateTensorCandidates, "flashComm", intermediateTensorList);
        if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT) {
            AddTensorToList(linearIntermediateTensorCandidates, "flashComm_dynamic_quant",
                intermediateTensorList);
        }
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddElewiseQuant(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    if (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
        // quant
        atb::Node inputQuantNode;
        atb::infer::ElewiseParam inputQuantParam;
        inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(inputQuantParam, &inputQuantNode.operation));
        inputQuantNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_scale", "in_offset"});
        inputQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_quant_input")};
        opGraph.nodes.push_back(inputQuantNode);
    }
    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W4A8_DYNAMIC_QUANT) {
        atb::Node inputDynamicQuantNode;
        inputDynamicQuantNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input"});
        inputDynamicQuantNode.outTensorIds = GetTensorIdxList(tensorMap, {"intermediate_quant_input",
                                                                          "intermediate_input_scale"});
        inputDynamicQuantNode.operation = new atb_speed::common::DynamicQuantOperation("DynamicQuantNode");
        opGraph.nodes.push_back(inputDynamicQuantNode);
    }
    return atb::NO_ERROR;
}

int64_t AddAllGather(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allGatherVNode;
    atb::infer::AllGatherVParam allGatherVParam;
    allGatherVParam.rank = param.flashCommParallelInfo.rank;
    allGatherVParam.rankSize = param.flashCommParallelInfo.worldSize;
    allGatherVParam.backend = param.flashCommParallelInfo.backend;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherVParam, &allGatherVNode.operation));
    allGatherVNode.inTensorIds = {GetTensorIdx(
        tensorMap, (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_DYNAMIC_QUANT
        || param.quantType == LINEAR_W8A8_SC_QUANT) ? "intermediate_quant_input" : "in_input")};
    allGatherVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "recv_count"));
    allGatherVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "send_counts"));
    allGatherVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "sdispls"));
    allGatherVNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "fake_ag_shape"));
    allGatherVNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_allgather_out")};

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherVNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));

    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT) {
        atb::Node allGatherInputScaleNode;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherVParam, &allGatherInputScaleNode.operation));
        allGatherInputScaleNode.inTensorIds = {GetTensorIdx(
            tensorMap, param.quantType == LINEAR_W8A8_DYNAMIC_QUANT
            ? "intermediate_input_scale" : "dynamic_input_scale")};
        allGatherInputScaleNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "send_count"));
        allGatherInputScaleNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "recv_counts"));
        allGatherInputScaleNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "rdispls"));
        allGatherInputScaleNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "fake_ag_shape"));
        allGatherInputScaleNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_allgather_input_scale_out")};
        CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
        opGraph.nodes.push_back(allGatherInputScaleNode);
        CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    }
    return atb::NO_ERROR;
}

int64_t AddAclNNWeightQuantBatchMatmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        param.enableFlashComm ? "intermediate_allgather_out" : "in_input",
        "in_weight", "in_scale", "in_offset"
    });
    AclNNWeightQuantBatchMatmulParam aclnnParam;
    aclnnParam.transposeB = param.transposeType == TRANSPOSE;
    if (param.hasBias) {
        linearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        aclnnParam.hasBias = true;
    }
    if (param.quantType == W8A16) {
        aclnnParam.quantGroupSize = param.quantGroupSize;
        linearNode.operation = new atb_speed::common::W8A16Operation("W8A16LinearNode", aclnnParam);
    } else if (param.quantType == W4A16) {
        aclnnParam.quantGroupSize = param.quantGroupSize;  // W4A16 group size默认为64，此时精度更高
        linearNode.operation = new atb_speed::common::W4A16Operation("W4A16LinearNode", aclnnParam);
    }
    if (linearNode.operation == nullptr) {
        return atb::ERROR_INVALID_GRAPH;
    }
    return atb::NO_ERROR;
}

int64_t AddW4A8Matmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    const bool containingQuant = param.quantType == LINEAR_W4A8_DYNAMIC_QUANT;
    AclNNW4A8Param aclnnParam;
    std::string key;
    if (param.enableSwigluQuant) {
        key = "in_input";
    } else {
        key = (containingQuant && !param.enableSwiGLUQuantForSharedExperts) ?
            "intermediate_quant_input" : "in_input";
    }
    std::string inputScaleKey;
    if (param.enableSwigluQuant && containingQuant) {
        inputScaleKey = "intermediate_quant_input_scale";
    } else {
        inputScaleKey = !containingQuant ? "dynamic_input_scale" : param.enableSwiGLUQuantForSharedExperts ?
            "swiglu_quant_input_scale" : "intermediate_input_scale";
    }
    std::vector<std::string> tensorNames = {key, "in_weight", inputScaleKey, "in_scale", "in_bias"};
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, tensorNames);
    ATB_SPEED_LOG_DEBUG("tensorNames: " << tensorNames << "; inTensorIds: " << linearNode.inTensorIds);
    aclnnParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    linearNode.operation = new atb_speed::common::W4A8Operation("W4A8LinearNode", aclnnParam);

    return atb::NO_ERROR;
}

int64_t AddAclNNQuantMatmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    AclNNQuantMatmulParam aclnnQuantMatmulParam;
    aclnnQuantMatmulParam.transposeB = param.transposeType == TRANSPOSE;
    std::string key;
    if (param.enableFlashComm) {
        key = "intermediate_allgather_out";
    } else {
        key = (param.quantType == LINEAR_W8A8_QUANT ||
            (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT
             && !param.enableSwiGLUQuantForSharedExperts)) ?
            "intermediate_quant_input" : "in_input";
    }
    std::string inScaleKey = (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_DEQUANT) ?
        "in_descale" : "in_scale";
    std::string inputScaleKey;
    if (param.enableFlashComm) {
        inputScaleKey = "intermediate_allgather_input_scale_out";
    } else {
        inputScaleKey = param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT ?
            "dynamic_input_scale" : param.enableSwiGLUQuantForSharedExperts ? "swiglu_quant_input_scale" :
            "intermediate_input_scale";
    }
    std::vector<std::string> tensorNames = {key, "in_weight", inScaleKey};
    // per token
    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT) {
        tensorNames.push_back(inputScaleKey);
        aclnnQuantMatmulParam.hasPerTokenScale = true;
    }

    // per tensor场景必需走bias场景
    if (param.hasBias || param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_DEQUANT) {
        tensorNames.push_back("in_bias");
        aclnnQuantMatmulParam.hasBias = true;
    }
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, tensorNames);
    ATB_SPEED_LOG_DEBUG("tensorNames: " << tensorNames << "; inTensorIds: " << linearNode.inTensorIds);
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) { // 1: input
        newShape.dimNum = 2; // dimNum: 2
        // 开启TURBO_ATTN时, w8a8_pdmix或pertoken场景, canndev算子, input会多出一维(2维)
        if (oldShape.dimNum == NUM3) {
            newShape.dims[DIM0] = oldShape.dims[DIM0] * oldShape.dims[DIM1];
            newShape.dims[DIM1] = oldShape.dims[DIM2];
        }
    };
    // dynamic的inputScaleKey转换
    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT || param.quantType == LINEAR_W8A8_DYNAMIC_DEQUANT) {
        linearNode.inTensorReshapeFuncs[3] = [=](const atb::Dims &oldShape, atb::Dims &newShape) { // 3: 3号scale
            newShape.dimNum = 1; // dimNum: 1
            // 开启TURBO_ATTN时, canndev算子, scale会多出一维(2维)
            newShape.dims[0] = oldShape.dimNum == NUM2 ? oldShape.dims[0] * oldShape.dims[1] : oldShape.dims[0];
        };
    }
    aclnnQuantMatmulParam.isBF16 = param.isBF16;
    linearNode.operation = new atb_speed::common::W8A8Operation("W8A8LinearNode", aclnnQuantMatmulParam);

    return atb::NO_ERROR;
}

int64_t AddAclNNMatmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        (param.enableFlashComm) ?
        "intermediate_allgather_out" : "in_input", "in_weight"});
    AclNNMatmulParam aclnnMatmulParam;
    aclnnMatmulParam.transposeB = param.transposeType == TRANSPOSE;
    if (param.hasBias) {
        linearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        aclnnMatmulParam.hasBias = true;
    }
    linearNode.operation = new atb_speed::common::W16A16Operation("W16A16LinearNode", aclnnMatmulParam);
    return atb::NO_ERROR;
}

int64_t AddW16A16SCMatmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        (param.enableFlashComm) ?
        "intermediate_allgather_out" : "in_input", "in_weight", "in_bias",  "in_compress_idx"});
    AclNNW16A16SCParam aclnnw16a16scParam;
    linearNode.operation = new atb_speed::common::W16A16SCOperation("W16A16SCLinearNode", aclnnw16a16scParam);
    return atb::NO_ERROR;
}

int64_t AddAclNNLinear(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    if (param.quantType == LINEAR_W4A8_DYNAMIC_QUANT || param.quantType == LINEAR_W4A8_DYNAMIC_DEQUANT) {
        CHECK_OPERATION_STATUS_RETURN(AddW4A8Matmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    if (param.quantType == W8A16 || param.quantType == W4A16) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNWeightQuantBatchMatmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    bool useQuantBatchMatmul = UseQuantBatchMatmul(param);
    if (useQuantBatchMatmul) {
        ATB_SPEED_LOG_DEBUG("AddAclNNQuantMatmul api: " << param.quantType << "," << useQuantBatchMatmul);
        CHECK_OPERATION_STATUS_RETURN(AddAclNNQuantMatmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    if (param.quantType == NO_QUANT) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNMatmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    if (param.quantType == LINEAR_W16A16_SC) {
        CHECK_OPERATION_STATUS_RETURN(AddW16A16SCMatmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    
    return atb::NO_ERROR;
}

int64_t AddLinear(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node linearNode;
    atb::infer::LinearParam linearParam;
    int matmulBackend = param.matmulBackend;
    if (param.quantType == NO_QUANT) {
        if ((param.isBF16 && IsA2()) || IsA3()) {
            matmulBackend = atb_speed::common::OpBackend::ATB;
        }
    }
    linearParam.transposeB = param.transposeType == TRANSPOSE;
    if (param.quantType != NO_QUANT) {
        linearParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    }
    if (param.enEin) {
        linearParam.matmulType = atb::infer::LinearParam::MATMUL_EIN_SUM;
    }
    // 设置LinearNode outTensor
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    // 稀疏量化
    if (param.quantType == LINEAR_W8A8_SC_DEQUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
        atb::infer::LinearSparseParam linearSparseParam;
        linearSparseParam.tilingK = 8;  // 8: 稀疏量化系数
        linearSparseParam.tilingN = 8;  // 8: 稀疏量化稀疏
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearSparseParam, &linearNode.operation));
        std::string key;
        if (param.enableFlashComm) {
            key = "intermediate_allgather_out";
        } else {
            key = param.quantType == LINEAR_W8A8_SC_DEQUANT ? "in_input" : "intermediate_quant_input";
        }
        linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
            key, "in_weight", "in_bias", "in_descale", "in_compress_idx"
        });
        opGraph.nodes.push_back(linearNode);
        return atb::NO_ERROR;
    }
    // W16A16SC SparseCompress
    if (param.quantType == LINEAR_W16A16_SC) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNLinear(linearNode, param, tensorMap));
        opGraph.nodes.push_back(linearNode);
        return atb::NO_ERROR;
    }
    // AclNN Linear (W8A16, W4A16, LINEAR_W8A8_DYNAMIC_QUANT, LINEAR_W8A8_DYNAMIC_DEQUANT)
    if (param.quantType == W8A16 || param.quantType == W4A16 || UseQuantBatchMatmul(param)) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNLinear(linearNode, param, tensorMap));
        opGraph.nodes.push_back(linearNode);
        return atb::NO_ERROR;
    }
    if (matmulBackend == atb_speed::common::OpBackend::ATB) {
        std::string key;
        if (param.enableFlashComm) {
            key = "intermediate_allgather_out";
        } else {
            key = param.quantType == LINEAR_W8A8_QUANT ? "intermediate_quant_input" : "in_input";
        }
        // 加速库Linear
        if (param.quantType == NO_QUANT && param.hasBias) {
            linearParam.hasBias = true;
            linearNode.inTensorIds = GetTensorIdxList(tensorMap, {key, "in_weight", "in_bias"});
        } else if (param.quantType == NO_QUANT && !param.hasBias) {
            linearParam.hasBias = false;
            linearNode.inTensorIds = GetTensorIdxList(tensorMap, {key, "in_weight"});
        } else {
            linearParam.hasBias = true;
            linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
                key, "in_weight", "in_bias", "in_descale"
            });
        }
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &linearNode.operation));
    } else {
        // AclNN Linear (NO_QUANT)
        CHECK_OPERATION_STATUS_RETURN(AddAclNNLinear(linearNode, param, tensorMap));
    }
    
    opGraph.nodes.push_back(linearNode);

    return atb::NO_ERROR;
}

atb::Status CreateFusionLinear(const FusionLinearParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.quantType == NO_QUANT ? "LinearNoQuant" : \
        param.quantType == LINEAR_W8A8_DEQUANT || param.quantType == LINEAR_W8A8_SC_DEQUANT ? "LinearDequantOnly" : \
        param.quantType == LINEAR_W16A16_SC ? "LinearW16A16SC" : \
        param.quantType == W8A16 ? "LinearW8A16" : \
        param.quantType == W4A16 ? "LinearW4A16" : "LinearQuant";
    std::map<std::string, uint32_t> tensorMap = ConstructLinearTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    if (param.transposeType == TRANSPOSE_INVALID) {
        ATB_SPEED_LOG_ERROR("param.transposeType is invalid");
        return atb::ERROR_INVALID_GRAPH;
    }
    // dense层: enableSwiGLUQuantForSharedExperts 不开
    // down层: 1) 不开 2) 开、非down 3)开、down、非DYNAMIC_DEQUANT
    if (!param.enableSwiGLUQuantForSharedExperts && (!param.enableSwigluQuant \
        || (param.quantType != LINEAR_W8A8_DYNAMIC_DEQUANT && param.quantType != LINEAR_W4A8_DYNAMIC_DEQUANT))) {
        CHECK_OPERATION_STATUS_RETURN(AddElewiseQuant(opGraph, param, tensorMap));
        if (param.enableFlashComm) {
            CHECK_OPERATION_STATUS_RETURN(AddAllGather(opGraph, param, tensorMap));
        }
    }
    if (param.enableCVOverlap) {
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateRecordWithoutNodeId(
            opGraph, atb_speed::EventAction::PUSH, atb_speed::common::VECTOR_CONTROL));
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateWaitWithoutNodeId(
            opGraph, atb_speed::EventAction::PUSH, atb_speed::common::CUBE_CONTROL));
    }
    CHECK_OPERATION_STATUS_RETURN(AddLinear(opGraph, param, tensorMap));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        uint32_t inputIdx = GetTensorIdx(tensorMap, "in_input");
        uint32_t weightIdx = GetTensorIdx(tensorMap, "in_weight");
        uint32_t biasIdx = GetTensorIdx(tensorMap, "in_bias");
        outTensorDescs.at(0).format = inTensorDescs.at(inputIdx).format;
        outTensorDescs.at(0).dtype = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
        outTensorDescs.at(0).shape = inTensorDescs.at(inputIdx).shape;
        auto outDimSize = outTensorDescs.at(inputIdx).shape.dimNum;
        CHECK_TENSORDESC_DIMNUM_VALID(outDimSize);
        int nDim = param.transposeType == TransposeType::TRANSPOSE ? 0 : 1;

        if (param.enableFlashComm) {
            uint32_t fakeAgShapeIdx = GetTensorIdx(tensorMap, "fake_ag_shape");
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(fakeAgShapeIdx).shape.dims[0];
        }
        if (param.quantType == LINEAR_W8A8_SC_DEQUANT || param.quantType == LINEAR_W8A8_SC_QUANT ||
            param.quantType == LINEAR_W16A16_SC) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(biasIdx).shape.dims[0];
        } else if (param.quantType == W4A16) {
            if (param.transposeType == TransposeType::TRANSPOSE) {
                outTensorDescs.at(0).shape.dims[outDimSize - 1] = \
                    inTensorDescs.at(weightIdx).shape.dims[0];  // 0: n维shape
            } else {
                outTensorDescs.at(0).shape.dims[outDimSize - 1] = \
                    CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[1], 2);  // 1, 2: 最后一维shape * 2
            }
        } else if (param.quantType == LINEAR_W4A8_DYNAMIC_DEQUANT || param.quantType == LINEAR_W4A8_DYNAMIC_QUANT) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = \
                CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[1], 8);  // 8: [m, k] @ [k, n//8] -> [m, n]
        } else if (inTensorDescs.at(weightIdx).shape.dimNum == 3) { // 3: dimNum
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(weightIdx).shape.dims[nDim + 1];
        } else if (param.enEin && inTensorDescs.at(weightIdx).shape.dimNum == 4) { // 4: dimNum
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = param.transposeType == TransposeType::TRANSPOSE ? \
                inTensorDescs.at(weightIdx).shape.dims[2] : // 2: dimNum
                    inTensorDescs.at(weightIdx).shape.dims[1] * inTensorDescs.at(weightIdx).shape.dims[3]; // 3: dimNum
        } else {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(weightIdx).shape.dims[nDim];
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> ConstructLinearWithLoraTensorMap(
    const FusionLinearParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto linearInTensorCandidates = GetLinearInTensorCandidates();
    auto linearIntermediateTensorCandidates = GetLinearIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(linearInTensorCandidates, "default", inTensorList);

    // 添加Lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(linearInTensorCandidates, "lora_with_mask", inTensorList);
            AddTensorToList(linearIntermediateTensorCandidates, "lora_with_mask", intermediateTensorList);
        }
        AddTensorToList(linearInTensorCandidates, "lora", inTensorList);
        AddTensorToList(linearIntermediateTensorCandidates, "lora", intermediateTensorList);
    }

    // 添加FlashComm特性的Tensor
    if (param.enableFlashComm) {
        AddTensorToList(linearInTensorCandidates, "flash_comm", inTensorList);
        AddTensorToList(linearIntermediateTensorCandidates, "flashComm", intermediateTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddImMask(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mulNode;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mulParam, &mulNode.operation));
    mulNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_im_mask"});
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_im_mask_out")};
    opGraph.nodes.push_back(mulNode);
    return atb::NO_ERROR;
}

int64_t AddLoraA(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap, bool enableFlashComm)
{
    // 添加Lora A
    atb::Node loraALinearNode;
    if (param.loraEnableGMM) {
        AclNNGroupedMatmulParam aclnnParam;
        aclnnParam.transposeB = true;
        loraALinearNode.operation = new atb_speed::common::GroupedMatmulOperation("loraALinearNode", aclnnParam);
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateFusionLinear(param, &loraALinearNode.operation));
    }
    if (param.useImMask) {
        loraALinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_im_mask_out", "in_lora_a"});
    } else if (enableFlashComm) {
        loraALinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_allgather_out", "in_lora_a"});
    } else {
        loraALinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_lora_a"});
    }
    if (param.loraEnableGMM) {
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    } else {
        // Lora权重暂不支持量化，以下Index仅为占位符
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_descale"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_compress_idx"));
    }
    loraALinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_lora_a_out")};
    opGraph.nodes.push_back(loraALinearNode);
    return atb::NO_ERROR;
}

int64_t AddLoraB(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    // 添加Lora B
    atb::Node loraBLinearNode;
    if (param.loraEnableGMM) {
        AclNNGroupedMatmulParam aclnnParam;
        aclnnParam.transposeB = false;
        loraBLinearNode.operation = new atb_speed::common::GroupedMatmulOperation("loraBLinearNode", aclnnParam);
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateFusionLinear(param, &loraBLinearNode.operation));
    }
    loraBLinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_lora_a_out", "in_lora_b"});
    if (param.loraEnableGMM) {
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    } else {
        // Lora权重暂不支持量化，以下Index仅为占位符
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_descale"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_compress_idx"));
    }
    loraBLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_lora_b_out")};
    opGraph.nodes.push_back(loraBLinearNode);
    return atb::NO_ERROR;
}

atb::Status CreateFusionLinearWithLora(const FusionLinearParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    std::map<std::string, uint32_t> tensorMap = ConstructLinearWithLoraTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    opGraph.name = "LinearWithLora";

    if (param.enableFlashComm) {
        CHECK_OPERATION_STATUS_RETURN(AddAllGather(opGraph, param, tensorMap));
    }

    // 添加Base模型的Linear
    atb::Node baseLinearNode;
    atb_speed::common::FusionLinearParam baseLinearParam = param;
    baseLinearParam.supportLora = false;
    baseLinearParam.loraEnableGMM = false;
    baseLinearParam.enableFlashComm = false;
    CHECK_OPERATION_STATUS_RETURN(CreateFusionLinear(baseLinearParam, &baseLinearNode.operation));
    baseLinearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        param.enableFlashComm ? "intermediate_allgather_out" : "in_input",
        "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"
    });
    baseLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_base_linear_out")};
    opGraph.nodes.push_back(baseLinearNode);

    atb_speed::common::FusionLinearParam loraLinearParam;
    loraLinearParam.isBF16 = param.isBF16;
    loraLinearParam.hasBias = false;
    loraLinearParam.transposeType = TRANSPOSE;
    loraLinearParam.loraEnableGMM = param.loraEnableGMM;
    loraLinearParam.useImMask = param.useImMask;
    if (param.useImMask) {
        CHECK_OPERATION_STATUS_RETURN(AddImMask(opGraph, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(AddLoraA(opGraph, loraLinearParam, tensorMap, param.enableFlashComm));
    loraLinearParam.transposeType = NOT_TRANSPOSE;
    CHECK_OPERATION_STATUS_RETURN(AddLoraB(opGraph, loraLinearParam, tensorMap));

    // 合并Base模型的Linear输出和Lora Linear的输出
    atb::Node addNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
    addNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_base_linear_out", "intermediate_lora_b_out"});
    addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(addNode);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation)
{
    if (param.supportLora) {
        return CreateFusionLinearWithLora(param, operation);
    } else {
        return CreateFusionLinear(param, operation);
    }
}
} // namespace common
} // namespace atb_speed