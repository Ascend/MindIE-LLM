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
#include "operations/aclnn/ops/dequant_swiglu_quant_operation.h"
#include "operations/aclrt/ops/aclrt_cmo_async.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/mlp/mlp.h"

namespace atb_speed {
namespace common {

std::map<std::string, atb::SVector<std::string>> GetMlpInTensorCandidates()
{
    std::map<std::string, atb::SVector<std::string>> mlpInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
            "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
            "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
            "in_weight_down", "in_scale_down", "in_offset_down", "in_descale_down", "in_bias_down",
            "in_compress_idx_down"}
        },
        {"add_norm", {"in_residual_add"}},
        {"add_rmsnorm_quant", {"in_mlp_scale_fill", "in_mlp_offset_fill"}},
        {"lora", {
            "in_seq_len_cum_sum", "in_lora_a_0", "in_lora_b_0", "in_lora_a_1", "in_lora_b_1",
            "in_down_lora_a", "in_down_lora_b"}
        },
        {"reduce_quant", {
            "in_reduce_quant_scale", "in_reduce_quant_offset",
            "in_gather_quant_scale", "in_gather_quant_offset"}},
        {"lora_with_mask", {"in_im_mask"}},
        {"flash_comm", {
            "send_counts", "sdispls", "send_count", "recv_counts", "rdispls", "recv_count",
            "fake_rs_shape", "fake_ag_shape"}
        },
    };
    return mlpInTensorCandidates;
}

std::map<std::string, atb::SVector<std::string>> GetMlpOutTensorCandidates()
{
    std::map<std::string, atb::SVector<std::string>> mlpOutTensorCandidates = {
        {"default", {"out_linear"}},
        {"add_norm", {"out_add"}},
    };
    return mlpOutTensorCandidates;
}

template <typename NormParamType>
atb::SVector<std::string> ConstructMlpInTensorList(const MlpParam<NormParamType> &param)
{
    auto mlpInTensorCandidates = GetMlpInTensorCandidates();

    atb::SVector<std::string> inTensorList = {};

    // 添加默认的Tensor
    AddTensorToList(mlpInTensorCandidates, "default", inTensorList);

    // 添加add norm特性的Tensor、添加AddRmsNormQuant特性的Tensor
    if (param.enableAddNorm) {
        AddTensorToList(mlpInTensorCandidates, "add_rmsnorm_quant", inTensorList);
        AddTensorToList(mlpInTensorCandidates, "add_norm", inTensorList);
    }

    // 添加lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(mlpInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(mlpInTensorCandidates, "lora", inTensorList);
    }

    // 添加lccl reduce int8特性的Tensor
    if (param.downLinearTensorParallelInfo.quantType != \
        atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        AddTensorToList(mlpInTensorCandidates, "reduce_quant", inTensorList);
    }

    // Add Flashcomm1.0 Tensor
    if (param.enableFlashComm) {
        AddTensorToList(mlpInTensorCandidates, "flash_comm", inTensorList);
    }
    return inTensorList;
}

template <typename NormParamType>
atb::SVector<std::string> ConstructMlpOutTensorList(const MlpParam<NormParamType> &param)
{
    auto mlpOutTensorCandidates = GetMlpOutTensorCandidates();

    atb::SVector<std::string> outTensorList = {};

    // 添加outTensor
    AddTensorToList(mlpOutTensorCandidates, "default", outTensorList);
    if (param.enableAddNorm) {
        AddTensorToList(mlpOutTensorCandidates, "add_norm", outTensorList);
    }

    return outTensorList;
}

template <typename NormParamType>
void SetGateUpNormLinearParam(atb_speed::common::NormLinearParam<NormParamType> &gateUpNormLinearParam,
    const MlpParam<NormParamType> &param, bool isAntiOutlier)
{
    gateUpNormLinearParam.isAntiOutlier = isAntiOutlier;
    gateUpNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[GATE_LINEAR_INDEX], param.enableNormQuantOp,
        param.layerLinearDescs[GATE_LINEAR_INDEX]);
    gateUpNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    gateUpNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
    gateUpNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    gateUpNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    gateUpNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    gateUpNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[GATE_LINEAR_INDEX];
    gateUpNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    gateUpNormLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    gateUpNormLinearParam.skipNorm = param.skipNorm;
    gateUpNormLinearParam.normHasBias = param.normHasBias;
    gateUpNormLinearParam.enableAddNorm = param.enableAddNorm;
    gateUpNormLinearParam.normParamType = param.normParamType;
    gateUpNormLinearParam.normQuantParamType = param.normQuantParamType;
    gateUpNormLinearParam.fusionLinearParam.enableFlashComm = param.enableFlashComm;
    gateUpNormLinearParam.fusionLinearParam.flashCommParallelInfo.worldSize =
        param.downLinearTensorParallelInfo.worldSize;
    gateUpNormLinearParam.fusionLinearParam.flashCommParallelInfo.rank =
        param.downLinearTensorParallelInfo.rank;
    gateUpNormLinearParam.fusionLinearParam.flashCommParallelInfo.backend =
        param.downLinearTensorParallelInfo.backend;
}

template <typename NormParamType>
atb::Status AddMlpNormLinearGateUp(const MlpParam<NormParamType> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* normLinearGateUpOp = nullptr;
    atb_speed::common::NormLinearParam<NormParamType> gateUpNormLinearParam;
    SetGateUpNormLinearParam(gateUpNormLinearParam, param, isAntiOutlier);
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(gateUpNormLinearParam, &normLinearGateUpOp));

    atb::SVector<std::string> gateUpInTensorNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
        "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
    };
    if (param.enableAddNorm) {
        gateUpInTensorNames.push_back("in_mlp_scale_fill");
        gateUpInTensorNames.push_back("in_mlp_offset_fill");
        gateUpInTensorNames.push_back("in_residual_add");
    }
    if (param.supportLora) {
        if (param.useImMask) {
            gateUpInTensorNames.push_back("in_im_mask");
        }
        gateUpInTensorNames.push_back("in_seq_len_cum_sum");
        gateUpInTensorNames.push_back("in_lora_a_0");
        gateUpInTensorNames.push_back("in_lora_b_0");
    }
    if (param.enableFlashComm) {
        gateUpInTensorNames.push_back("send_counts");
        gateUpInTensorNames.push_back("sdispls");
        gateUpInTensorNames.push_back("send_count");
        gateUpInTensorNames.push_back("recv_counts");
        gateUpInTensorNames.push_back("rdispls");
        gateUpInTensorNames.push_back("recv_count");
        gateUpInTensorNames.push_back("fake_ag_shape");
    }
    atb::SVector<std::string> gateUpOutTensorNames = {};
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        gateUpOutTensorNames = {"intermediate_gate_up"} ;
    } else if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        gateUpOutTensorNames = {"intermediate_gate"};
    } else {
        gateUpOutTensorNames = {"intermediate_up"};
    }
    if (param.enableAddNorm) {
        gateUpOutTensorNames.push_back("out_add");
    }

    graphBuilder->AddOperation(normLinearGateUpOp, gateUpInTensorNames, gateUpOutTensorNames);
    return atb::NO_ERROR;
}

template <typename NormParamType>
void SetUpNormLinearParam(atb_speed::common::NormLinearParam<NormParamType> &upNormLinearParam,
    const MlpParam<NormParamType> &param, bool isAntiOutlier)
{
    upNormLinearParam.isAntiOutlier = isAntiOutlier;
    upNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[UP_LINEAR_INDEX], param.enableNormQuantOp,
        param.layerLinearDescs[UP_LINEAR_INDEX]);
    upNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    upNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
    upNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    upNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    upNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    upNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[UP_LINEAR_INDEX];
    upNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    upNormLinearParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    upNormLinearParam.skipNorm = param.skipNorm;
    upNormLinearParam.normHasBias = param.normHasBias;
    upNormLinearParam.normParamType = param.normParamType;
    upNormLinearParam.normQuantParamType = param.normQuantParamType;
    upNormLinearParam.fusionLinearParam.enableFlashComm = param.enableFlashComm;
    upNormLinearParam.fusionLinearParam.flashCommParallelInfo.worldSize =
        param.downLinearTensorParallelInfo.worldSize;
    upNormLinearParam.fusionLinearParam.flashCommParallelInfo.rank =
        param.downLinearTensorParallelInfo.rank;
    upNormLinearParam.fusionLinearParam.flashCommParallelInfo.backend =
        param.downLinearTensorParallelInfo.backend;
}

template <typename NormParamType>
atb::Status AddMlpNormLinearUp(const MlpParam<NormParamType> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* normLinearUpOp = nullptr;
    atb_speed::common::NormLinearParam<NormParamType> upNormLinearParam;
    SetUpNormLinearParam(upNormLinearParam, param, isAntiOutlier);
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(upNormLinearParam, &normLinearUpOp));

    atb::SVector<std::string> upInTensorNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
        "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
    };
    if (param.supportLora) {
        if (param.useImMask) {
            upInTensorNames.push_back("in_im_mask");
        }
        upInTensorNames.push_back("in_seq_len_cum_sum");
        upInTensorNames.push_back("in_lora_a_1");
        upInTensorNames.push_back("in_lora_b_1");
    }
    if (param.enableFlashComm) {
        upInTensorNames.push_back("send_counts");
        upInTensorNames.push_back("sdispls");
        upInTensorNames.push_back("send_count");
        upInTensorNames.push_back("recv_counts");
        upInTensorNames.push_back("rdispls");
        upInTensorNames.push_back("recv_count");
        upInTensorNames.push_back("fake_ag_shape");
    }

    graphBuilder->AddOperation(normLinearUpOp, upInTensorNames, {"intermediate_up"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddPrefetchWeight(const MlpParam<NormParamType> &param,
    atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* computeRecordNodeOp = nullptr;
    atb::SVector<std::string> computeRecordNodeInTensorIds = {};
    atb::SVector<std::string> computeRecordNodeOutTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        computeRecordNodeOp,
        atb_speed::EventAction::PUSH,
        atb_speed::common::CMO_COMPUTE));
    graphBuilder->AddOperation(computeRecordNodeOp, computeRecordNodeInTensorIds, computeRecordNodeOutTensorIds);

    atb::Operation* commWaitNodeOp = nullptr;
    atb::SVector<std::string> commWaitNodeInTensorIds = {};
    atb::SVector<std::string> commWaitNodeOutTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        commWaitNodeOp,
        atb_speed::EventAction::POP,
        atb_speed::common::CMO_COMPUTE));
    atb::SetExecuteStreamId(commWaitNodeOp, 1);
    graphBuilder->AddOperation(commWaitNodeOp, commWaitNodeInTensorIds, commWaitNodeOutTensorIds);

    atb::Operation* cmoNodeOp = new atb_speed::common::AclrtCmoAsyncOperation(
        "AclrtCmoAsync", param.preFetchWeightSize);
    atb::SVector<std::string> cmoNodeInTensorIds = {"in_weight_down"};
    atb::SVector<std::string> cmoNodeOutTensorIds  = {};
    atb::SetExecuteStreamId(cmoNodeOp, 1);
    graphBuilder->AddOperation(cmoNodeOp, cmoNodeInTensorIds, cmoNodeOutTensorIds);
    return atb::NO_ERROR;
}

atb::Status AddMlpSplit(atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* splitOp = nullptr;
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = -1; // [batchSize, seqLen, 2 * hiddenSize]
    splitParam.splitNum = 2;  // 进行二等分
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitParam, &splitOp));

    graphBuilder->AddOperation(splitOp, {"intermediate_gate_up"}, {"intermediate_gate", "intermediate_up"});
    return atb::NO_ERROR;
}

atb::Status AddMlpSwiGLUConcat(atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* concatOp = nullptr;
    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = -1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(concatParam, &concatOp));

    graphBuilder->AddOperation(concatOp, {"intermediate_gate", "intermediate_up"}, {"intermediate_gate_up"});
    return atb::NO_ERROR;
}

atb::Status AddMlpMul(atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* mulOp = nullptr;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(elewiseParam, &mulOp));

    graphBuilder->AddOperation(
        mulOp, {"intermediate_activation_out", "intermediate_up"}, {"intermediate_activation_out"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpActivation(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpSplit(graphBuilder));
    }

    atb::Operation* activationOp = nullptr;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.activationParam, &activationOp));
    graphBuilder->AddOperation(
        activationOp,
        {param.mlpPackType == MlpPackType::UP_WEIGHT_ONLY ? "intermediate_up" : "intermediate_gate"},
        {"intermediate_activation_out"});

    if (param.mlpPackType != MlpPackType::UP_WEIGHT_ONLY) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpMul(graphBuilder));
    }
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpEdgeActivation(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpSplit(graphBuilder));
    }

    atb::Operation* sigmoidOp = nullptr;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SIGMOID;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(activationParam, &sigmoidOp));
    graphBuilder->AddOperation(sigmoidOp, {"intermediate_gate"}, {"intermediate_activation_out"});

    atb::Operation* sigmoidMulOp = nullptr;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(elewiseParam, &sigmoidMulOp));
    graphBuilder->AddOperation(
        sigmoidMulOp, {"intermediate_gate", "intermediate_activation_out"}, {"intermediate_activation_out"});

    if (param.mlpPackType != MlpPackType::UP_WEIGHT_ONLY) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpMul(graphBuilder));
    }

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpSwiGLUActivation(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpSwiGLUConcat(graphBuilder));
    }

    atb::Operation* activationOp = nullptr;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.activationParam, &activationOp));
    graphBuilder->AddOperation(activationOp, {"intermediate_gate_up"}, {"intermediate_activation_out"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
void SetDownLinearParallelParam(const MlpParam<NormParamType> &param,
    atb_speed::common::LinearParallelParam &downLinearParallelParam)
{
    if (param.enableFlashComm) {
        downLinearParallelParam.parallelType = atb_speed::common::REDUCE_SCATTER;
    } else {
        downLinearParallelParam.parallelType = atb_speed::common::ROW_PARALLEL;
    }
    downLinearParallelParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.downQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.downQuantType,
        param.layerLinearQuantType[DOWN_LINEAR_INDEX], false,
        param.layerLinearDescs[DOWN_LINEAR_INDEX]);
    bool downIsQuant = IsLinearDescQuant(param, DOWN_LINEAR_INDEX);
    if (param.enableSwigluQuant && downIsQuant) {
        downLinearParallelParam.fusionLinearParam.enableSwigluQuant = param.enableSwigluQuant;
        if (downLinearParallelParam.fusionLinearParam.quantType == \
                atb_speed::common::LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT) {
            downLinearParallelParam.fusionLinearParam.quantType = \
                atb_speed::common::LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT;
        } else if (downLinearParallelParam.fusionLinearParam.quantType == \
                atb_speed::common::LinearQuantType::LINEAR_W8A8_QUANT) {
            downLinearParallelParam.fusionLinearParam.quantType = \
                atb_speed::common::LinearQuantType::LINEAR_W8A8_DEQUANT;
        } else if (downLinearParallelParam.fusionLinearParam.quantType == \
                atb_speed::common::LinearQuantType::LINEAR_W4A8_DYNAMIC_QUANT) {
            downLinearParallelParam.fusionLinearParam.quantType = \
                atb_speed::common::LinearQuantType::LINEAR_W4A8_DYNAMIC_DEQUANT;
        }
    }
    downLinearParallelParam.biasAfterSync = param.downLinearTensorParallelInfo.worldSize > 1 && \
        downLinearParallelParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT && \
        param.downHasBias;
    downLinearParallelParam.fusionLinearParam.hasBias = param.downHasBias && !downLinearParallelParam.biasAfterSync;
    downLinearParallelParam.fusionLinearParam.isBF16 = param.isBF16;
    downLinearParallelParam.fusionLinearParam.supportLora = param.supportLora;
    downLinearParallelParam.fusionLinearParam.useImMask = param.useImMask;
    downLinearParallelParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    downLinearParallelParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[DOWN_LINEAR_INDEX];
    downLinearParallelParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    downLinearParallelParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    downLinearParallelParam.tensorParallelInfo = param.downLinearTensorParallelInfo;
    downLinearParallelParam.supportLcoc = param.supportLcoc;
    downLinearParallelParam.enableMC2 = param.enableMC2;
}

template <typename NormParamType>
atb::Status AddMlpLinearDown(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* linearDownOp = nullptr;
    atb_speed::common::LinearParallelParam downLinearParallelParam;
    SetDownLinearParallelParam(param, downLinearParallelParam);
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(downLinearParallelParam, &linearDownOp));

    atb::SVector<std::string> downInTensorNames = {
        "intermediate_activation_out",
        "in_weight_down", "in_scale_down", "in_offset_down", "in_descale_down", "in_bias_down",
        "in_compress_idx_down"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            downInTensorNames.push_back("in_im_mask");
        }
        downInTensorNames.push_back("in_seq_len_cum_sum");
        downInTensorNames.push_back("in_down_lora_a");
        downInTensorNames.push_back("in_down_lora_b");
    }
    if (param.downLinearTensorParallelInfo.quantType != atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        downInTensorNames.push_back("in_reduce_quant_scale");
        downInTensorNames.push_back("in_reduce_quant_offset");
        downInTensorNames.push_back("in_gather_quant_scale");
        downInTensorNames.push_back("in_gather_quant_offset");
    }
    if (param.enableSwigluQuant && downLinearParallelParam.fusionLinearParam.quantType == \
        atb_speed::common::LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT) {
            downInTensorNames.push_back("intermediate_swiglu_dynamic_scale");
    }
    if (downLinearParallelParam.parallelType == atb_speed::common::REDUCE_SCATTER) {
        downInTensorNames.push_back("send_counts");
        downInTensorNames.push_back("sdispls");
        downInTensorNames.push_back("recv_count");
        downInTensorNames.push_back("fake_rs_shape");
    }
    graphBuilder->AddOperation(linearDownOp, downInTensorNames, {"out_linear"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddDequantSwigluQuantNode(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    // 这里注意,不要根据layerLinearQuantType获取类型, 要根据layerLinearDescs获取
    LinearQuantType downQuantType = GetLinearQuantType(
        param.downQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
            param.packQuantType : param.downQuantType,
        param.layerLinearQuantType[DOWN_LINEAR_INDEX],
        false,
        param.layerLinearDescs[DOWN_LINEAR_INDEX]);

    AclNNDequantSwigluQuantParam aclnnParam;
    aclnnParam.activateLeft = true;
    aclnnParam.quantMode = "static";
    atb::SVector<std::string> inTensorNames = { "intermediate_gate_up" };  // 0: x
    FusionLinearParam linearParam;
    linearParam.matmulBackend = param.matmulBackend;
    linearParam.quantType = downQuantType;
    bool gateUpIsQuant = IsLinearDescQuant(param, GATE_LINEAR_INDEX);
    if (gateUpIsQuant && UseQuantBatchMatmul(linearParam)) {
        inTensorNames.push_back("in_descale_0");  // 1: weightScaleOptional, fp32
        // 2: activationScaleOptional, 这个参数传null
        inTensorNames.push_back("in_bias_0");  // 3: biasOptional, int32
    }
    inTensorNames.push_back("in_scale_down");  // 4: quantScaleOptional
    inTensorNames.push_back("in_offset_down");  // 5: quantOffsetOptional
    if (downQuantType == atb_speed::common::LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT) {
        inTensorNames = { "intermediate_gate_up" };  // 0: x
        aclnnParam.quantMode = "dynamic";  // dynamic, 只传x
    }
    atb::SVector<std::string> outTensorNames = {"intermediate_activation_out", "intermediate_swiglu_dynamic_scale"};
    aclnnParam.inTensorsNum = static_cast<int>(inTensorNames.size());
    atb::Operation* dequantSwigluQuantOp = new atb_speed::common::DequantSwigluQuantOperation(
        "aclNNDequantSwigluQuantNode", aclnnParam
    );
    graphBuilder->AddOperation(dequantSwigluQuantOp, inTensorNames, outTensorNames);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status Mlp(const MlpParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphOpBuilder* graphOpBuilder = nullptr;
    CHECK_OPERATION_STATUS_RETURN(CreateGraphOpBuilder(&graphOpBuilder));
    atb::Status res = CreateMlp(param, graphOpBuilder, operation, false);
    if (DestroyGraphOpBuilder(graphOpBuilder) != atb::NO_ERROR) {
        ATB_SPEED_LOG_WARN("Destroy graph builder failed. This may leads to memory leak, please check");
    }
    return res;
}

template <typename NormParamType>
atb::Status MlpSwiGLU(const MlpParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphOpBuilder* graphOpBuilder = nullptr;
    CHECK_OPERATION_STATUS_RETURN(CreateGraphOpBuilder(&graphOpBuilder));
    atb::Status res = CreateMlp(param, graphOpBuilder, operation, true);
    if (DestroyGraphOpBuilder(graphOpBuilder) != atb::NO_ERROR) {
        ATB_SPEED_LOG_WARN("Destroy graph builder failed. This may leads to memory leak, please check");
    }
    return res;
}

template <typename NormParamType>
atb::Status CheckMlpParam(const MlpParam<NormParamType> &param)
{
    if (param.layerLinearDescs.size() != 0 && \
        CheckParamVectorSize(param.layerLinearDescs, DOWN_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearDescs is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (param.layerLinearQuantType.size() != 0 && \
        CheckParamVectorSize(param.layerLinearQuantType, DOWN_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearQuantType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (CheckParamVectorSize(param.layerLinearTransposeType, DOWN_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearTransposeType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status CreateMlp(
    const MlpParam<NormParamType> &param,
    atb::GraphOpBuilder* &graphOpBuilder,
    atb::Operation **operation, bool isSwiGLU)
{
    bool isAntiOutlier = CheckAntiOutlier(param.packQuantType);
    isAntiOutlier = isAntiOutlier || param.isAntiOutlier;
    CHECK_OPERATION_STATUS_RETURN(CheckMlpParam(param));

    std::string graphName = isSwiGLU ? "MlpSwiGLU" : "Mlp";
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        graphName += "GateUpWeightPack";
    } else if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        graphName += "GateUpWeightNoPack";
    } else {
        graphName += "UpWeightOnly";
    }

    atb::InferShapeFunc inferShapeFunc = [param](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableAddNorm) { outTensorDescs.at(1) = inTensorDescs.at(0); }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(graphOpBuilder->Init(
        graphName, inferShapeFunc, ConstructMlpInTensorList(param), ConstructMlpOutTensorList(param)
    ));

    // Gate Up
    CHECK_OPERATION_STATUS_RETURN(AddMlpNormLinearGateUp(param, isAntiOutlier, graphOpBuilder));
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpNormLinearUp(param, isAntiOutlier, graphOpBuilder));
    }
    // prefetch weight
    if (param.preFetchWeightSize > 0) {
        CHECK_OPERATION_STATUS_RETURN(AddPrefetchWeight(param, graphOpBuilder));
    }
    // Activation
    if (param.isEdgeHardware) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpEdgeActivation(param, graphOpBuilder));
    } else if (isSwiGLU) {
        bool downIsQuant = IsLinearDescQuant(param, DOWN_LINEAR_INDEX);
        if (param.enableSwigluQuant && downIsQuant) {
            CHECK_OPERATION_STATUS_RETURN(AddDequantSwigluQuantNode(param, graphOpBuilder));
        } else {
            CHECK_OPERATION_STATUS_RETURN(AddMlpSwiGLUActivation(param, graphOpBuilder));
        }
    } else {
        CHECK_OPERATION_STATUS_RETURN(AddMlpActivation(param, graphOpBuilder));
    }
    // Down
    CHECK_OPERATION_STATUS_RETURN(AddMlpLinearDown(param, graphOpBuilder));

    *operation = graphOpBuilder->Build();
    return atb::NO_ERROR;
}

MlpPackType GetMlpPackType(
    const int &packQuantType, bool upWeightOnly, const std::vector<int> &linearDescs)
{
    if (upWeightOnly) {
        return atb_speed::common::UP_WEIGHT_ONLY;
    }
    std::vector<int> gateUpLinearIndex = {GATE_LINEAR_INDEX, UP_LINEAR_INDEX};
    bool isPack = CheckPack(packQuantType, linearDescs, gateUpLinearIndex);
    if (isPack) {
        return atb_speed::common::GATE_UP_WEIGHT_PACK;
    } else {
        return atb_speed::common::GATE_UP_WEIGHT_NO_PACK;
    }
}

template <typename NormParamType>
bool IsLinearDescQuant(const MlpParam<NormParamType> &param, const uint64_t index)
{
    return param.layerLinearDescs[index] != common::LinearDesc::INVALID_DESC && \
        param.layerLinearDescs[index] != common::LinearDesc::FLOAT16_DESC && \
        param.layerLinearDescs[index] != common::LinearDesc::BFLOAT16_DESC && \
        param.layerLinearDescs[index] != common::LinearDesc::W4A16_DESC && \
        param.layerLinearDescs[index] != common::LinearDesc::W8A16_DESC;
}

template bool IsLinearDescQuant(const MlpParam<atb::infer::RmsNormParam> &param, const uint64_t index);

template bool IsLinearDescQuant(const MlpParam<atb::infer::LayerNormParam> &param, const uint64_t index);

template void SetDownLinearParallelParam(const MlpParam<atb::infer::RmsNormParam> &param,
    atb_speed::common::LinearParallelParam &downLinearParallelParam);

template void SetDownLinearParallelParam(const MlpParam<atb::infer::LayerNormParam> &param,
    atb_speed::common::LinearParallelParam &downLinearParallelParam);

template atb::Status CheckMlpParam(const MlpParam<atb::infer::RmsNormParam> &param);

template atb::Status CheckMlpParam(const MlpParam<atb::infer::LayerNormParam> &param);

template atb::SVector<std::string> ConstructMlpInTensorList(const MlpParam<atb::infer::RmsNormParam> &param);

template atb::SVector<std::string> ConstructMlpInTensorList(const MlpParam<atb::infer::LayerNormParam> &param);

template atb::SVector<std::string> ConstructMlpOutTensorList(const MlpParam<atb::infer::RmsNormParam> &param);

template atb::SVector<std::string> ConstructMlpOutTensorList(const MlpParam<atb::infer::LayerNormParam> &param);

template void SetGateUpNormLinearParam(
    atb_speed::common::NormLinearParam<atb::infer::RmsNormParam> &gateUpNormLinearParam,
    const MlpParam<atb::infer::RmsNormParam> &param, bool isAntiOutlier);

template void SetGateUpNormLinearParam(
    atb_speed::common::NormLinearParam<atb::infer::LayerNormParam> &gateUpNormLinearParam,
    const MlpParam<atb::infer::LayerNormParam> &param, bool isAntiOutlier);

template atb::Status AddMlpNormLinearGateUp(const MlpParam<atb::infer::RmsNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpNormLinearGateUp(const MlpParam<atb::infer::LayerNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpLinearDown(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpLinearDown(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddPrefetchWeight(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddPrefetchWeight(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpNormLinearUp(const MlpParam<atb::infer::RmsNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpNormLinearUp(const MlpParam<atb::infer::LayerNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpActivation(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpActivation(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpEdgeActivation(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpSwiGLUActivation(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpSwiGLUActivation(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddDequantSwigluQuantNode(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddDequantSwigluQuantNode(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status Mlp(const MlpParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template atb::Status Mlp(const MlpParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

template atb::Status MlpSwiGLU(const MlpParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template atb::Status MlpSwiGLU(const MlpParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

template atb::Status CreateMlp(
    const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphOpBuilder, atb::Operation **operation, bool isSwiGLU);

template atb::Status CreateMlp(
    const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphOpBuilder, atb::Operation **operation, bool isSwiGLU);

} // namespace common
} // namespace atb_speed