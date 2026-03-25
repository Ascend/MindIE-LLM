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
#include <cctype>
#include <atb/types.h>
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "models/base/model/decoder_model.h"
#include "models/base/layer/decoder_layer.h"
#include "models/base/param/model_param.h"
#include "operations/aclnn/ops/split_with_size_operation.h"
#include "operations/fusion/infer_shape_functions.h"

namespace atb_speed {
namespace base {

HcclComm DecoderModel::gHcommInfo = nullptr;

DecoderModel::DecoderModel(const std::string &param) : Model("DecoderModel", param)
{
    this->param.FromString(param);
    this->modelName_ += this->param.isPrefill ? "_Prefill" : "_Decoder";
    this->inTensorCandidates = {
        {"default", {
            "input_ids", "positional_ids", "cosine_table", "sine_table", "attention_mask",
            "block_tables", "slots", "kv_cache_idx", "token_offset", "place_holder", "seq_len", "logits_indices"}
        },
        {"token_off_set", {"logits_offset_tensor"}},
        {"compress_head_alibi", {"wins_global", "in_ra_seqlens"}},
        {"compress_head_rope_common", {"wins_global", "in_reshape_seqlen"}},
        {"compress_head_rope_per_layer", {
            "ra_block_tables", "ra_slots", "in_ra_seqlens", "pffset_index", "razor_offset"}},
        {"q_len", {"q_len"}},
        {"lora_common", {"seq_len_cum_sum"}},
        {"lora_per_layer", {
            "qkv_lora_a_0", "qkv_lora_b_0", "qkv_lora_a_1", "qkv_lora_b_1",
            "qkv_lora_a_2", "qkv_lora_b_2", "qkv_dense_lora_a", "qkv_dense_lora_b",
            "mlp_lora_a_0", "mlp_lora_b_0", "mlp_lora_a_1", "mlp_lora_b_1",
            "mlp_down_lora_a", "mlp_down_lora_b"}},
        {"attn_dp", {
            "in_final_hidden_state", "in_shard_effective_token_indices", "in_token_index_with_padding",
            "in_skip_padding_token_indices"}},
        {"flash_comm", {
            "send_counts", "sdispls", "send_count", "recv_counts", "rdispls", "recv_count",
            "fake_rs_shape", "fake_ag_shape"}},
        {"reuse_embed_table", {"cosine_embed_table", "sine_embed_table"}},
    };
    this->SetLayerwiseDisaggregatedParam();
    if (this->param.skipWordEmbedding) {
        this->inTensorCandidates["default"].at(0) = "input_embedding";
    }
    
    this->internalTensorCandidates = {
        {"default", {"hidden_states"}},
        {"rope", {"cosine_embedding", "sine_embedding"}},
        {"attn_dp", {"attn_dp_last_layer"}},
        {"input_add_norm", {"last_layer_mlp_out"}}
    };

    if (this->param.enableFlashComm) {
        std::vector<std::string> hiddenStatesRanks;
        hiddenStatesRanks.resize(this->param.worldSize);
        for (int i = 0; i < this->param.worldSize; i++) {
            hiddenStatesRanks[i] = "hidden_states_rank" + std::to_string(i);
        }
        this->internalTensorCandidates["hidden_states_ranks"] = hiddenStatesRanks;
        this->internalTensorCandidates["Allgather"] = {"hidden_states_allgather"};
    }

    this->outTensorCandidates = {
        {"default", {"logits"}},
        {"output_embed_table", {"cosine_embed_table", "sine_embed_table"}},
    };
}

DecoderModel::~DecoderModel() {}

void DecoderModel::SetLayerwiseDisaggregatedParam()
{
    if (!this->param.layerwiseDisaggregated) {
        return;
    }
    if (this->param.layerwiseMode == LWD_CLOUD_MIDDLE || \
        this->param.layerwiseMode == LWD_EDGE_LAST) {
            this->weightCountWordEmbedding = 0;
        }
    if (this->param.layerwiseMode == LWD_EDGE_FIRST || \
        this->param.layerwiseMode == LWD_CLOUD_MIDDLE) {
        this->weightCountFinalNorm = 0;
        this->weightCountLmHead = 0;
    }
}


void DecoderModel::ConstructInTensorMap()
{
    this->inTensorMap.clear();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "default", this->inTensorMap);

    // 添加头压缩特性的Tensor
    if (this->param.enableCompressHead) {
        if (param.positionEmbeddingType == PositionEmbeddingType::ALIBI) {
            atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "compress_head_alibi", this->inTensorMap);
        } else if (param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
            atb_speed::common::AssignTensorIdx(
                this->inTensorCandidates, "compress_head_rope_common", this->inTensorMap);
            uint32_t currentTensorIdx = this->inTensorMap.size();
            for (uint32_t i = 0; i < this->param.numHiddenLayers; ++i) {
                for (std::string raInputName : this->inTensorCandidates.at("compress_head_rope_per_layer")) {
                    this->inTensorMap["layer_" + std::to_string(i) + "_" + raInputName] = currentTensorIdx;
                    currentTensorIdx++;
                }
            }
        }
    }
    // 添加omniattention特性的Tensor
    if (this->param.enableOmniAttention) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "compress_head_rope_common", this->inTensorMap);
        uint32_t currentTensorIdx = this->inTensorMap.size();
        for (uint32_t i = 0; i < this->param.numHiddenLayers; ++i) {
            for (std::string raInputName : this->inTensorCandidates.at("compress_head_rope_per_layer")) {
                this->inTensorMap["layer_" + std::to_string(i) + "_" + raInputName] = currentTensorIdx;
                currentTensorIdx++;
            }
        }
    }
    // 添加并行解码特性或SplitFuse的Tensor
    if (this->param.enableSpeculate || this->param.enableSplitFuse) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "q_len", this->inTensorMap);
    }

    // 添加lora特性的Tensor
    if (this->param.enableLora) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "lora_common", this->inTensorMap);
        uint32_t currentTensorIdx = this->inTensorMap.size();
        for (uint32_t i = 0; i < this->param.numHiddenLayers; i++) {
            for (std::string loraWeightName : this->inTensorCandidates.at("lora_per_layer")) {
                this->inTensorMap["layer_" + std::to_string(i) + loraWeightName] = currentTensorIdx;
                currentTensorIdx++;
            }
        }
    }

    // Append in-tensors for data parallelism of Attention
    if (this->param.hasAttnDp) {
        atb_speed::common::AssignTensorIdx(
            this->inTensorCandidates, "attn_dp", this->inTensorMap);
    }

    // 添加flashcomm1.0特性的Tensor
    if (this->param.enableFlashComm) {
        atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "flash_comm", this->inTensorMap);
    }
}

void DecoderModel::ConstructInternalTensorMap()
{
    this->internalTensorMap.clear();
    // 添加默认的Tensor
    if (!this->param.skipWordEmbedding) {
        atb_speed::common::AssignTensorIdx(
            this->internalTensorCandidates, "default", this->internalTensorMap);
    }

    // 添加rope的Tensor
    if (this->param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
        atb_speed::common::AssignTensorIdx(
            this->internalTensorCandidates, "rope", this->internalTensorMap);
    }

    // Append internal-tensors for data parallelism of Attention
    if (this->param.hasAttnDp && this->param.hasMlpTp) {
        atb_speed::common::AssignTensorIdx(
            this->internalTensorCandidates, "attn_dp", this->internalTensorMap);
    }
    // 添加add rmsnorm融合特性的中间tensor
    if (this->param.enableInterLayerAddNorm) {
        atb_speed::common::AssignTensorIdx(
            this->internalTensorCandidates, "input_add_norm", this->internalTensorMap);
    }
    if (this->param.enableFlashComm) {
        atb_speed::common::AssignTensorIdx(this->internalTensorCandidates,
            "hidden_states_ranks", this->internalTensorMap);
        atb_speed::common::AssignTensorIdx(this->internalTensorCandidates, "Allgather", this->internalTensorMap);
    }
}

void DecoderModel::ConstructOutTensorMap()
{
    this->outTensorMap.clear();
    // 添加默认的Tensor
    atb_speed::common::AssignTensorIdx(
        this->outTensorCandidates, "default", this->outTensorMap);
}

void DecoderModel::PrintTensorMapInfo(std::map<std::string, uint32_t> &tensorMap) const
{
    std::stringstream ss;
    ss << "TensorMap Info: ";
    for (auto tensor = tensorMap.cbegin(); tensor != tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG(ss.str());
}

uint32_t DecoderModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t DecoderModel::GetOutputNum() const { return graph_.outTensors.size(); }

uint32_t DecoderModel::CalcWeightTensorSize()
{
    if (this->param.enableKvQuant) {
        this->weightCountPerLayer += 8;  // 8: kv cache int8 多8个inTensor
    }
    if (this->param.enableFA3) {
        this->weightCountPerLayer += 8; // 8: FA3 多8个inTensorensor
    }
    if (this->param.enableReduceQuant) {
        this->weightCountPerLayer += 8;  // 8: lccl reduce int8 多8个inTensor
    }
    if (this->param.enableInterLayerAddNorm || this->param.enableIntraLayerAddNorm) {
        this->weightCountPerLayer += 4;  // 4: addRmsNormQuant 多4个inTensor
    }
    if (this->param.normType == LAYER_NORM) {
        this->weightCountFinalNorm = 2;  // 2: LayerNorm 权重数量
    }
    if (this->param.useQKNorm) {
        this->weightCountPerLayer += 2;  // 2: useQKNorm 多2个inTensor
    }
    const uint64_t weightTensorSize =
        this->weightCountWordEmbedding +
        CheckIntMulOverFlow(this->weightCountPerLayer, this->param.numHiddenLayers) +
        this->weightCountFinalNorm + this->weightCountLmHead;
    return weightTensorSize;
}

void DecoderModel::DuplicateTensorMapForDap(
    std::map<std::string, uint32_t> &tensorMap, std::vector<atb::Tensor> &targetTensors)
{
    std::map<uint32_t, uint32_t> tensorIndexMap = {};
    tensorIndexMap = CopyMapWithSuffix(tensorMap);
    targetTensors.resize(tensorMap.size());
    std::stringstream ss;
    ss << "Dap preceder to successor mapping info: ";
    for (auto pair = tensorIndexMap.cbegin(); pair != tensorIndexMap.cend(); ++pair) {
        this->precederToSuccessorTensorMap[&targetTensors.at(pair->first)] = &targetTensors.at(pair->second);
        ss << "tensor src index: " << pair->first << ", tensor dst index: " << pair->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG(ss.str());
}

std::map<uint32_t, uint32_t> DecoderModel::CopyMapWithSuffix(std::map<std::string, uint32_t>& tensorMap) const
{
    std::map<std::string, uint32_t> tmpTensorMap = {};
    std::map<uint32_t, uint32_t> tensorIndexMap = {};
    std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();
    uint32_t tensorMapSize = tensorMap.size();
    for (auto pair = tensorMap.cbegin(); pair != tensorMap.cend(); ++pair) {
        tmpTensorMap[pair->first + suffix] = pair->second + tensorMapSize;
        tensorIndexMap[pair->second] = pair->second + tensorMapSize;
    }
    for (auto pair = tmpTensorMap.cbegin(); pair != tmpTensorMap.cend(); ++pair) {
        tensorMap[pair->first] = pair->second;
    }
    return tensorIndexMap;
}

void DecoderModel::ReplaceDapTensors(std::vector<atb::Tensor *>& tensors)
{
    for (uint32_t i = 0; i < tensors.size(); i++) {
        auto it = this->precederToSuccessorTensorMap.find(tensors[i]);
        if (it != this->precederToSuccessorTensorMap.end()) {
            tensors[i] = it->second;
        }
    }
}

atb::Status DecoderModel::InferShape(
    const std::vector<atb::TensorDesc> &inTensorDescs,
    std::vector<atb::TensorDesc> &outTensorDescs
)
{
    const uint64_t RESULT_DIM_2 = 2;
    ATB_SPEED_LOG_DEBUG("Enter DecoderModel InferShape");
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }
    uint32_t logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "logits_indices");
    std::string inputKey = this->param.skipWordEmbedding ? "input_embedding" : "input_ids";
    uint32_t inputIdsIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, inputKey);
    CHECK_TENSORDESC_DIMNUM_VALID(graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);
    CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(inputIdsIdx).shape.dimNum);
    uint32_t dim = this->param.lmHeadTransposeType == atb_speed::common::TransposeType::NOT_TRANSPOSE ? 1 : 0;
    const int64_t vocabSizePerRank = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[dim];
    int64_t seqLenAxis = this->param.isUnpadInputs ? 0 : 1;  // 2, 3: Axis
    if (!this->param.enableGreedyPostProcessing && !this->param.layerwiseDisaggregated) {
        // unpadInputs: [batchSize, seqLen, vocabSize] padInputs: [seqLen, vocabSisze]
        outTensorDescs.at(0).dtype = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.dtype;
        outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
        outTensorDescs.at(0).shape.dimNum = this->param.isUnpadInputs ? 2 : 3;  // 2, 3: dimNum
        CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(inputIdsIdx).shape.dims[0];
        if (this->param.isPrefill || this->param.enablePrefixCache || this->param.hasAttnDp) {
            outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
        } else {
            outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(inputIdsIdx).shape.dims[seqLenAxis];
        }
        outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = this->param.isLmHeadParallel
            ? CheckIntMulOverFlow(vocabSizePerRank, this->param.hasPp ? this->param.tpWorldSize : this->param.worldSize)
            : vocabSizePerRank;
    } else if (!this->param.enableGreedyPostProcessing && this->param.layerwiseDisaggregated) {
        outTensorDescs.at(0).dtype =  this->param.isBF16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
        outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
        outTensorDescs.at(0).shape.dimNum = this->param.isUnpadInputs ? 2 : 3;  // 2, 3: dimNum
        CHECK_TENSORDESC_DIMNUM_VALID(outTensorDescs.at(0).shape.dimNum);
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(inputIdsIdx).shape.dims[0];
        bool isLayerWiseNonOutputLayer = this->param.layerwiseDisaggregated && \
                (this->param.layerwiseMode == LWD_EDGE_FIRST || this->param.layerwiseMode == LWD_CLOUD_MIDDLE);
        if (isLayerWiseNonOutputLayer) {
            outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(0).shape.dims[0];
        } else if (this->param.isPrefill || this->param.enablePrefixCache || this->param.hasAttnDp) {
            outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
        } else {
            outTensorDescs.at(0).shape.dims[seqLenAxis] = inTensorDescs.at(inputIdsIdx).shape.dims[seqLenAxis];
        }
        if (isLayerWiseNonOutputLayer) {
            outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = this->param.hiddenSize;
        } else {
            outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] = this->param.isLmHeadParallel
            ? CheckIntMulOverFlow(vocabSizePerRank, this->param.hasPp ? this->param.tpWorldSize : this->param.worldSize)
            : vocabSizePerRank;
        }
    } else {
        outTensorDescs.at(0).dtype =  aclDataType::ACL_INT64;
        outTensorDescs.at(0).format = graph_.weightTensors.at(0).desc.format;
        outTensorDescs.at(0).shape.dimNum = RESULT_DIM_2; // 二维 [batch_size,1]
        outTensorDescs.at(0).shape.dims[0] =
            inTensorDescs.at(10).shape.dims[0]; // num 10 on behalf of seq_len, dims[0] is batch_size
        outTensorDescs.at(0).shape.dims[1] = 1;
    }

    if (this->param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();
        logitsIndicesIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "logits_indices" + suffix);
        inputIdsIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, inputKey + suffix);
        CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(logitsIndicesIdx).shape.dimNum);
        CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(inputIdsIdx).shape.dimNum);
        outTensorDescs.at(1) = outTensorDescs.at(0);
        if (this->param.isPrefill || this->param.enablePrefixCache || this->param.hasAttnDp) {
            outTensorDescs.at(1).shape.dims[seqLenAxis] = inTensorDescs.at(logitsIndicesIdx).shape.dims[0];
        } else {
            outTensorDescs.at(1).shape.dims[seqLenAxis] = inTensorDescs.at(inputIdsIdx).shape.dims[seqLenAxis];
        }
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }

    if (this->param.layerwiseDisaggregated && this->param.outputEmbedTable) {
        uint32_t idx = 1;
        uint32_t positionIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids");
        outTensorDescs.at(idx).dtype = this->param.isBF16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
        outTensorDescs.at(idx).format = graph_.weightTensors.at(0).desc.format;
        outTensorDescs.at(idx).shape.dimNum = RESULT_DIM_2;
        outTensorDescs.at(idx).shape.dims[0] = inTensorDescs.at(positionIdx).shape.dims[0];
        outTensorDescs.at(idx).shape.dims[1] = this->param.hiddenSizePerAttentionHead;
        outTensorDescs.at(idx + 1) = outTensorDescs.at(idx);
    }
    return atb::NO_ERROR;
}

int64_t DecoderModel::BuildGraph()
{
    // 准备inTensor
    this->ConstructInTensorMap();
    this->PrintTensorMapInfo(this->inTensorMap);
    this->graph_.inTensors.resize(this->inTensorMap.size());
    if (this->param.enableDap) {
        this->DuplicateTensorMapForDap(this->inTensorMap, this->graph_.inTensors);
    }
    ATB_SPEED_LOG_DEBUG("graph_.inTensors " << this->graph_.inTensors.size());

    // 准备internalTensor
    this->ConstructInternalTensorMap();
    this->PrintTensorMapInfo(this->internalTensorMap);
    this->graph_.internalTensors.resize(this->internalTensorMap.size());
    if (this->param.enableDap) {
        this->DuplicateTensorMapForDap(this->internalTensorMap, this->graph_.internalTensors);
    }
    ATB_SPEED_LOG_DEBUG("graph_.internalTensors " << this->graph_.internalTensors.size());

    // 准备outTensor
    this->ConstructOutTensorMap();
    this->PrintTensorMapInfo(this->outTensorMap);
    this->graph_.outTensors.resize(this->outTensorMap.size());
    if (this->param.enableDap) {
        this->DuplicateTensorMapForDap(this->outTensorMap, this->graph_.outTensors);
    }
    ATB_SPEED_LOG_DEBUG("graph_.outTensors " << this->graph_.outTensors.size());

    // 准备weightTensor
    graph_.weightTensors.resize(this->CalcWeightTensorSize());
    ATB_SPEED_LOG_DEBUG("graph_.weightTensors " << this->graph_.weightTensors.size());

    // 准备kv cache
    graph_.kCacheTensors.resize(this->param.numHiddenLayers);
    graph_.vCacheTensors.resize(this->param.numHiddenLayers);

    GetSingleton<common::DapManager>().SetRole(common::DapRole::UNDEFINED_ROLE);
    GetSingleton<atb_speed::common::CommOpCounter>().Reset();
    auto ret = this->AddOperationToGraph();
    ATB_SPEED_LOG_DEBUG(GetSingleton<ExternalCommManager>().PrintCommInfo());
    return ret;
}

atb::Status DecoderModel::AddOperationToGraph()
{
    std::stringstream ss;
    atb::Operation *op = nullptr;

    // AddNodesBeforeLayer
    // PRECEDER Events
    if (this->param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddNodesBeforeLayer());

    // SUCCESSOR Events
    if (this->param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        atb_speed::Model::Node computeWaitNode;
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
            op, atb_speed::EventAction::POP, common::COMPUTE_EVENT));
        computeWaitNode.inTensors = {};
        computeWaitNode.outTensors = {};
        computeWaitNode.operation.reset(op);
        CHECK_OPERATION_STATUS_RETURN(SetNodeStreamId(computeWaitNode, 1));
        graph_.nodes.push_back(computeWaitNode);
        ss.str("");
        ss << "[Events] [SUCCESSOR] [POP] [WAIT] [COMPUTE] will be pushed to the graph later";
        ATB_SPEED_LOG_DEBUG(ss.str());

        uint32_t nodeCount = graph_.nodes.size();
        CHECK_OPERATION_STATUS_RETURN(this->AddNodesBeforeLayer());
        for (uint32_t index = nodeCount; index < graph_.nodes.size(); index++) {
            CHECK_OPERATION_STATUS_RETURN(SetNodeStreamId(graph_.nodes.at(index), 1));
            ReplaceDapTensors(graph_.nodes.at(index).inTensors);
            ReplaceDapTensors(graph_.nodes.at(index).outTensors);
        }
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }

    // AddLayer
    CHECK_OPERATION_STATUS_RETURN(this->AddLayer());

    // AddNodesAfterLayer
    // PRECEDER Events
    if (!this->param.layerwiseDisaggregated or this->param.layerwiseMode == LWD_EDGE_LAST) {
        CHECK_OPERATION_STATUS_RETURN(this->AddNodesAfterLayer());
    } else {
        ATB_SPEED_LOG_DEBUG("skip add nodes after layer");
    }
    if (param.enableDap) {
        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::EventManager::GetInstance().WaitEvent(op, atb_speed::EventAction::PUSH, common::COMM_EVENT));
        atb_speed::Model::Node commWaitNode;
        commWaitNode.inTensors = {};
        commWaitNode.outTensors = {};
        commWaitNode.operation.reset(op);
        graph_.nodes.push_back(commWaitNode);
        ATB_SPEED_LOG_DEBUG("[Events] [PRECEDER] [PUSH] [WAIT] [COMM]");

        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
            op, atb_speed::EventAction::PUSH, common::COMPUTE_EVENT));
        atb_speed::Model::Node computeRecordNode;
        computeRecordNode.inTensors = {};
        computeRecordNode.outTensors = {};
        computeRecordNode.operation.reset(op);
        graph_.nodes.push_back(computeRecordNode);
        ATB_SPEED_LOG_DEBUG("[Events] [PRECEDER] [PUSH] [RECORD] [COMPUTE]");

        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::EventManager::GetInstance().WaitEvent(op, atb_speed::EventAction::POP, common::END_EVENT));
        atb_speed::Model::Node endWaitNode;
        endWaitNode.inTensors = {};
        endWaitNode.outTensors = {};
        endWaitNode.operation.reset(op);
        graph_.nodes.push_back(endWaitNode);
        ATB_SPEED_LOG_DEBUG("[Events] [PRECEDER] [POP] [WAIT] [END]");
    }

    // SUCCESSOR Events
    if (this->param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        uint32_t nodeCount = graph_.nodes.size();
        CHECK_OPERATION_STATUS_RETURN(this->AddNodesAfterLayer());
        for (uint32_t index = nodeCount; index < graph_.nodes.size(); index++) {
            CHECK_OPERATION_STATUS_RETURN(SetNodeStreamId(graph_.nodes.at(index), 1));
            ReplaceDapTensors(graph_.nodes.at(index).inTensors);
            ReplaceDapTensors(graph_.nodes.at(index).outTensors);
        }

        CHECK_OPERATION_STATUS_RETURN(
            atb_speed::EventManager::GetInstance().RecordEvent(op, atb_speed::EventAction::PUSH, common::END_EVENT));
        atb_speed::Model::Node endRecordNode;
        endRecordNode.inTensors = {};
        endRecordNode.outTensors = {};
        endRecordNode.operation.reset(op);
        CHECK_OPERATION_STATUS_RETURN(SetNodeStreamId(endRecordNode, 1));
        graph_.nodes.push_back(endRecordNode);
        ATB_SPEED_LOG_DEBUG("[Events] [SUCCESSOR] [PUSH] [RECORD] [END]");
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }

    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddNodesBeforeLayer()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    if (this->param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
        CHECK_OPERATION_STATUS_RETURN(this->AddPositionalEmbedding());
    }
    if (this->param.enableFlashComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddSplitHiddenStates());
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddNodesAfterLayer()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFinalNorm());
    if (this->param.enableFlashComm) {
        CHECK_OPERATION_STATUS_RETURN(this->AddAllGather());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddLmhead());
    if (param.preFetchWeightSize > 0) {
        CHECK_OPERATION_STATUS_RETURN(AddCmoSync());
    }
    return atb::NO_ERROR;
}

void DecoderModel::SetWordEmbeddingParam(atb_speed::common::WordEmbeddingParam &wordEmbeddingParam)
{
    wordEmbeddingParam.unpadInputs = this->param.isUnpadInputs;
    if (this->param.isEmbeddingParallel && this->param.hasPp) {
        wordEmbeddingParam.tensorParallelInfo = {
            this->param.tpRank, this->param.tpWorldSize, this->param.backend, this->param.tpRankTableFile, \
            nullptr, this->param.tpDomain
        };
    } else if (this->param.isEmbeddingParallel && this->param.hasAttnDp) {
        wordEmbeddingParam.tensorParallelInfo = {
            this->param.attnTpRank, this->param.attnTpSize, this->param.backend, this->param.attnTpRankTableFile,
            nullptr, this->param.attnTpDomain
        };
    } else if (this->param.isEmbeddingParallel) {
        wordEmbeddingParam.tensorParallelInfo = {
            this->param.rank, this->param.worldSize, this->param.backend, this->param.rankTableFile
        };
        if (this->param.mapping.isInitialized_) {
            atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::WORD_EMBED_TP);
            parallelInfo.InitCommDomain(
                wordEmbeddingParam.tensorParallelInfo.hcommInfo,
                wordEmbeddingParam.tensorParallelInfo.commDomain);
        }
    };
}

atb::Status DecoderModel::AddWordEmbedding()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node wordEmbeddingNode;
    atb_speed::common::WordEmbeddingParam wordEmbeddingParam;
    this->SetWordEmbeddingParam(wordEmbeddingParam);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::WordEmbedding(wordEmbeddingParam, &op));
    wordEmbeddingNode.operation.reset(op);
    wordEmbeddingNode.inTensors = {
        &graph_.weightTensors.at(0),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_ids"))
    };
    wordEmbeddingNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"))
    };
    graph_.nodes.push_back(wordEmbeddingNode);
    ATB_SPEED_LOG_DEBUG("[+] base wordEmbeddingNode");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddPositionalEmbedding()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node positionalEmbeddingGatherNode;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::PositionalEmbeddingGather(&op));
    positionalEmbeddingGatherNode.operation.reset(op);
    positionalEmbeddingGatherNode.inTensors = {
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "positional_ids")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "cosine_table")),
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sine_table")),
    };
    positionalEmbeddingGatherNode.outTensors = {
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding")),
        &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"))
    };
    graph_.nodes.push_back(positionalEmbeddingGatherNode);
    ATB_SPEED_LOG_DEBUG("[+] base positionalEmbeddingGatherNode");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddSplitHiddenStates()
{
    atb::Operation *op = nullptr;
    if (this->param.enableFlashComm) {
        atb_speed::Model::Node splitHiddenStatesNode;
        atb_speed::common::AclNNSplitWithSizeParam aclnnSplitWithSizeParam;
        aclnnSplitWithSizeParam.num = this->param.worldSize;
        op = new atb_speed::common::SplitWithSizeOperation("splitHiddenStatesNode", aclnnSplitWithSizeParam);
        splitHiddenStatesNode.operation.reset(op);

        if (this->param.skipWordEmbedding) {
            splitHiddenStatesNode.inTensors = {
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding"))
            };
        } else {
            splitHiddenStatesNode.inTensors = {
                &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"))
            };
        }
        for (int i = 0; i < this->param.worldSize; i++) {
            splitHiddenStatesNode.outTensors.emplace_back(&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
                this->internalTensorMap, "hidden_states_rank" + std::to_string(i))));
        }

        ATB_SPEED_LOG_DEBUG("[+] splitHiddenStatesNode");
        graph_.nodes.push_back(splitHiddenStatesNode);
    }

    return atb::NO_ERROR;
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    if (this->param.normType == RMS_NORM) {
        DecoderLayer<atb::infer::RmsNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    } else {
        DecoderLayer<atb::infer::LayerNormParam> decoderLayer(layerParam);
        CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddSend()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node sendNode;
    atb::infer::SendParam sendParam;
    sendParam.rank = this->param.rank;
    sendParam.rankSize = this->param.ppGroupSize * this->param.tpWorldSize;
    sendParam.rankRoot = 0;
    sendParam.destRank = this->param.nextPpRank;
    sendParam.rankTableFile = this->param.rankTableFile;
    sendParam.commDomain = "sendRecvDomain";
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(sendParam, &op));
    sendNode.operation.reset(op);
    atb::Tensor *firstInTensor =
        this->param.skipWordEmbedding
            ? &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding"))
            : &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"));
    sendNode.inTensors = {firstInTensor};
    sendNode.outTensors = {};
    graph_.nodes.push_back(sendNode);

    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddRecv()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node receiveNode;
    atb::infer::RecvParam recvParam;
    recvParam.rank = this->param.rank;
    recvParam.rankSize = this->param.ppGroupSize * this->param.tpWorldSize;
    recvParam.rankRoot = 0;
    recvParam.srcRank = this->param.prevPpRank;
    recvParam.rankTableFile = this->param.rankTableFile;
    recvParam.commDomain = "sendRecvDomain";
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(recvParam, &op));
    receiveNode.operation.reset(op);
    atb::Tensor *firstInTensor =
        this->param.skipWordEmbedding
            ? &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding"))
            : &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"));
    receiveNode.inTensors = {firstInTensor};
    receiveNode.outTensors = {firstInTensor};
    graph_.nodes.push_back(receiveNode);

    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddLayer()
{
    uint32_t numHiddenLayers = param.enableDap ? param.numHiddenLayers * 2 : param.numHiddenLayers;
    for (uint32_t layerId = 0; layerId < numHiddenLayers; ++layerId) {
        if (param.enableDap && (layerId % 2 == 1)) {  // 2: even layer
            GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        }
        uint32_t trueLayerId = param.enableDap ? layerId / 2 : layerId;

        uint32_t nodeCount = graph_.nodes.size();
        this->AddSingleLayer(trueLayerId);
        RecordEventBeforePrefixCacheSave();
        for (uint32_t index = nodeCount; index < graph_.nodes.size(); index++) {
            if (GetSingleton<common::DapManager>().GetRole() == common::DapRole::SUCCESSOR) {
                CHECK_OPERATION_STATUS_RETURN(SetNodeStreamId(graph_.nodes.at(index), 1));
                ReplaceDapTensors(graph_.nodes.at(index).inTensors);
                ReplaceDapTensors(graph_.nodes.at(index).outTensors);
            }
        }

        if (param.enableDap) {
            GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
        }
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::RecordEventBeforePrefixCacheSave()
{
    if (param.memPoolType == atb_speed::base::MemPoolType::ASYNC_WRITE && param.isPrefill) {
        atb::Operation *op = nullptr;
        atb_speed::Model::Node recordSaveNode;
        CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
            op, atb_speed::EventAction::PUSH, param.memPoolEventPipeKey));
        recordSaveNode.inTensors = {};
        recordSaveNode.outTensors = {};
        recordSaveNode.operation.reset(op);
        graph_.nodes.push_back(recordSaveNode);
        ATB_SPEED_LOG_DEBUG("[Events] [PUSH] [RECORD] will be pushed to the graph later");
    }
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddSingleLayer(uint32_t layerId)
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node layerNode;
    CHECK_OPERATION_STATUS_RETURN(this->CreateLayerOperation(&op, layerId));

    layerNode.operation.reset(op);
    layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    layerNode.inTensorReshapeFuncs.resize(layerNode.operation->GetInputNum());
    SetLayerNodeInput(layerNode, layerId);

    bool isLayerWiseOutputLayer = this->param.layerwiseDisaggregated && this->param.isInternalLayer && \
            layerId == this->param.numHiddenLayers - 1;
    if (isLayerWiseOutputLayer) {
        ATB_SPEED_LOG_DEBUG("layerwise output layer");
        layerNode.outTensors = {&graph_.outTensors.at(0)};
    } else if (this->param.hasAttnDp && this->param.hasMlpTp) {
        layerNode.outTensors = {
            layerNode.inTensors.at(weightCountPerLayer),
            &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap, "attn_dp_last_layer"))
        };
    } else {
        layerNode.outTensors = {layerNode.inTensors.at(weightCountPerLayer)}; // 输出原地写在输入上
    }
    if (this->param.enableInterLayerAddNorm && (layerId != (param.numHiddenLayers - 1))) {
        layerNode.outTensors.push_back(
            &graph_.internalTensors.at(
                atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_layer_mlp_out")
            )
        );
    }
    graph_.nodes.push_back(layerNode);
    ATB_SPEED_LOG_DEBUG("[+] add base layerNode num" << layerId);
    return atb::NO_ERROR;
}

void DecoderModel::SetLayerParallelismParam(LayerParam &layerParam)
{
    layerParam.backend = this->param.backend;
    if (this->param.hasPp) {
        layerParam.tensorParallelInfo = {
            this->param.tpRank, this->param.tpWorldSize, this->param.backend, this->param.tpRankTableFile, \
            nullptr, this->param.tpDomain};
    } else {
        layerParam.tensorParallelInfo = {
            this->param.rank, this->param.worldSize, this->param.backend, this->param.rankTableFile, gHcommInfo};
    }
    layerParam.hasAttnTp = this->param.hasAttnTp;
    layerParam.attnTpRank = this->param.attnTpRank;
    layerParam.attnTpSize = this->param.attnTpSize;
    layerParam.attnTpDomain = this->param.attnTpDomain;
    layerParam.attnTpRankTableFile = this->param.rankTableFile;
    layerParam.hasAttnDp = this->param.hasAttnDp;
    layerParam.attnDpRank = this->param.attnDpRank;
    layerParam.attnDpSize = this->param.attnDpSize;
    layerParam.attnDpDomain = this->param.attnDpDomain;
    layerParam.attnDpRankTableFile = this->param.rankTableFile;
    layerParam.hasMlpTp = this->param.hasMlpTp;
    layerParam.mlpTpRank = this->param.mlpTpRank;
    layerParam.mlpTpSize = this->param.mlpTpSize;
    layerParam.mlpTpDomain = this->param.mlpTpDomain;
    layerParam.mlpTpRankTableFile = this->param.rankTableFile;
    layerParam.enableSwigluQuant = this->param.enableSwigluQuant;
    layerParam.mapping = this->param.mapping;
}

void DecoderModel::SetLayerParam(LayerParam &layerParam, uint32_t layerId)
{
    layerParam.layerId = layerId;
    layerParam.numHiddenLayers = this->param.numHiddenLayers;
    layerParam.isFA = this->param.isFA;
    layerParam.isUnpadInputs = this->param.isUnpadInputs;
    layerParam.isPrefill = this->param.isPrefill;
    layerParam.isBF16 = this->param.isBF16;
    layerParam.isEdgeHardware = this->param.isEdgeHardware;
    layerParam.enableSwiGLU = this->param.enableSwiGLU;
    layerParam.enableLcoc = this->param.enableLcoc;
    layerParam.enableMC2 = this->param.enableMC2;
    layerParam.enableSpeculate = this->param.enableSpeculate;
    layerParam.enableCompressHead = this->param.enableCompressHead;
    layerParam.enableOmniAttention = this->param.enableOmniAttention;
    layerParam.useQKNorm = this->param.useQKNorm;
    if (layerParam.enableOmniAttention) {
        layerParam.isomnicompressed = this->param.patternMask[layerId];
        this->param.isomnicompressed = this->param.patternMask[layerId];
    }
    layerParam.enableSplitFuse = this->param.enableSplitFuse;
    layerParam.enableLora = this->param.enableLora;
    layerParam.preFetchWeightSize = this->param.preFetchWeightSize;
    layerParam.loraEnableGMM = this->param.loraEnableGMM;
    layerParam.enableKvQuant = this->param.enableKvQuant;
    layerParam.enableFA3 = this->param.enableFA3;
    layerParam.kvQuantHasOffset = this->param.kvQuantHasOffset;
    layerParam.enableReduceQuant = this->param.enableReduceQuant;
    layerParam.enableInterLayerAddNorm = this->param.enableInterLayerAddNorm;
    layerParam.enableIntraLayerAddNorm = this->param.enableIntraLayerAddNorm;
    layerParam.enablePrefixCache = this->param.enablePrefixCache;
    layerParam.attnBackend = this->param.attnBackend;
    layerParam.matmulBackend = this->param.matmulBackend;
    layerParam.ropeBackend = this->param.ropeBackend;
    layerParam.positionEmbeddingType = this->param.positionEmbeddingType;
    layerParam.normEps = this->param.normEps;
    layerParam.normType = this->param.normType;
    layerParam.quantGroupSize = this->param.quantGroupSize;
    layerParam.numAttentionHeadsPerRank = this->param.numAttentionHeadsPerRank;
    layerParam.hiddenSizePerAttentionHead = this->param.hiddenSizePerAttentionHead;
    layerParam.numKeyValueHeadsPerRank = this->param.numKeyValueHeadsPerRank;
    layerParam.enableFlashComm = this->param.enableFlashComm;
    layerParam.enableModelConfuscation = this->param.enableModelConfuscation;
    layerParam.modelConfuscationFd = this->param.modelConfuscationFd;
    if (layerId != 0) { layerParam.enableModelConfuscation = false; }
    if (!this->param.packQuantType.empty()) {
        layerParam.packQuantType = this->param.packQuantType[layerId];
    }
    if (!this->param.linearQuantType.empty()) {
        layerParam.linearQuantType = this->param.linearQuantType[layerId];
    }
    layerParam.linearTransposeType = this->param.linearTransposeType[layerId];
    if (!this->param.linearHasBias.empty()) {
        layerParam.linearHasBias = this->param.linearHasBias[layerId];
    }
    if (!this->param.linearDescs.empty()) {
        layerParam.linearDescs = this->param.linearDescs[layerId];
    }
    if (!this->param.isAntiOutlier.empty()) {
        layerParam.isAntiOutlier = this->param.isAntiOutlier[layerId];
    }
    layerParam.weightQuantType = this->param.weightQuantType;
    SetLayerParallelismParam(layerParam);
}

void DecoderModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    uint32_t inTensorId = 0;
    this->SetLayerNodeDefaultInput(layerNode, layerId, inTensorId);
    this->SetLayerNodeOptionalInput(layerNode, layerId, inTensorId);
}

void DecoderModel::SetLayerNodeOptionalInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    if (this->param.enableCompressHead) {
        this->SetLayerNodeRaInput(layerNode, layerId, inTensorId);
    }
    if (this->param.enableOmniAttention) {
        this->SetLayerNodeOmniInput(layerNode, layerId, inTensorId);
    }
    if (this->param.enableSpeculate || this->param.enableSplitFuse) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len"));
    }
    if (this->param.enableLora) {
        this->SetLayerNodeLoraInput(layerNode, layerId, inTensorId);
    }
    if (this->param.hasAttnDp) {
        this->SetLayerNodeAttnDpInput(layerNode, inTensorId);
    }
    if (param.enableInterLayerAddNorm && layerId != 0) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "last_layer_mlp_out"));
    }
    if (this->param.enableFlashComm) {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "send_counts"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sdispls"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "send_count"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "recv_counts"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "rdispls"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "recv_count"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "fake_rs_shape"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "fake_ag_shape"));
    }
    ATB_SPEED_LOG_DEBUG("LayerNode Optional Input set success, inputs num: " << inTensorId);
}

void DecoderModel::SetLayerNodeDefaultInput(
    atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    for (uint32_t weightTensorId = 0; weightTensorId < this->weightCountPerLayer; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = &graph_.weightTensors.at(
            CheckIntMulOverFlow(layerId, this->weightCountPerLayer) + weightTensorId + this->weightCountWordEmbedding);
    }
    if (this->param.enableFlashComm) {
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap,
            "hidden_states_rank" + std::to_string(this->param.rank)));
    } else {
        layerNode.inTensors.at(inTensorId++) = this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"));
    }
    if (this->param.positionEmbeddingType == atb_speed::base::PositionEmbeddingType::ROPE) {
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "cosine_embedding"));
        layerNode.inTensors.at(inTensorId++) = &graph_.internalTensors.at(
            atb_speed::common::GetTensorIdx(this->internalTensorMap, "sine_embedding"));
    } else {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "place_holder"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, "place_holder"));
    }
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "attention_mask"));
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset"));
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "kv_cache_idx"));
    if (this->param.enableCompressHead && this->param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "ra_block_tables"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "ra_slots"));
    }  else if (this->param.enableOmniAttention) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "ra_block_tables"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "ra_slots"));
    } else {
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "block_tables"));
        layerNode.inTensors.at(inTensorId++) = \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "slots"));
    }
    ATB_SPEED_LOG_DEBUG("LayerNode Default Input set success, inputs num: " << inTensorId);
}

void DecoderModel::SetLayerNodeRaInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    if (param.positionEmbeddingType == PositionEmbeddingType::ALIBI) {
        for (std::string raInputName: this->inTensorCandidates.at("compress_head_alibi")) {
            layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
                atb_speed::common::GetTensorIdx(this->inTensorMap, raInputName));
        }
    } else if (param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "wins_global"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "in_ra_seqlens"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "pffset_index"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "razor_offset"));
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "in_reshape_seqlen"));
    }
}

void DecoderModel::SetLayerNodeOmniInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "wins_global"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "in_ra_seqlens"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "pffset_index"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "layer_" + std::to_string(layerId) + "_" + "razor_offset"));
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(atb_speed::common::GetTensorIdx(
        this->inTensorMap, "in_reshape_seqlen"));
}

void DecoderModel::SetLayerNodeLoraInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId)
{
    layerNode.inTensors.at(inTensorId++) = \
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len_cum_sum"));
    for (std::string loraWeightName : this->inTensorCandidates.at("lora_per_layer")) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(
                this->inTensorMap, "layer_" + std::to_string(layerId) + loraWeightName)
        );
    }
}

void DecoderModel::SetLayerNodeAttnDpInput(atb_speed::Model::Node &layerNode, uint32_t &inTensorId)
{
    for (std::string attnDpInputName : this->inTensorCandidates.at("attn_dp")) {
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(
            atb_speed::common::GetTensorIdx(this->inTensorMap, attnDpInputName)
        );
    }
    ATB_SPEED_LOG_DEBUG("decoder model has pushed up atten_dp intensors");
}

void DecoderModel::SetFinalNormParam(atb::infer::RmsNormParam &normParam)
{
    normParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    normParam.normParam.epsilon = this->param.normEps;
}

void DecoderModel::SetFinalNormParam(atb::infer::LayerNormParam &normParam)
{
    int32_t beginParamsAxis = this->param.isFA ? 2 : 1;
    normParam.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    normParam.normParam.epsilon = this->param.normEps;
    normParam.normParam.beginNormAxis = beginParamsAxis;
    normParam.normParam.beginParamsAxis = 1;
}

atb::Status DecoderModel::AddFinalNorm()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node finalNormNode;
    if (this->param.normType == NormType::RMS_NORM) {
        atb::infer::RmsNormParam finalNormParam;
        this->SetFinalNormParam(finalNormParam);
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    } else {
        atb::infer::LayerNormParam finalNormParam;
        this->SetFinalNormParam(finalNormParam);
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(finalNormParam, &op));
    }
    finalNormNode.operation.reset(op);
    const uint32_t finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - this->weightCountFinalNorm - this->weightCountLmHead;
    if (this->param.hasAttnDp && this->param.hasMlpTp) {
        finalNormNode.inTensors = {
            this->param.skipWordEmbedding ? \
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
                &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
                    this->internalTensorMap, "attn_dp_last_layer")),
            &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    } else if (this->param.enableFlashComm) {
        finalNormNode.inTensors = { &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
            "hidden_states_rank" + std::to_string(this->param.rank))),
            &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    } else {
        finalNormNode.inTensors = {
        this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")),
        &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    }
    if (this->param.normType == NormType::LAYER_NORM) {
        finalNormNode.inTensors.push_back(&graph_.weightTensors.at(finalLayerNormWeightTensorId + 1));
    }
    finalNormNode.outTensors = {finalNormNode.inTensors.at(0)};  // 输出原地写在输入上
    graph_.nodes.push_back(finalNormNode);
    ATB_SPEED_LOG_DEBUG("[+] base finalNormNode");
    return atb::NO_ERROR;
}

atb::Status DecoderModel::AddAllGather()
{
    atb::Operation *op = nullptr;
    if (this->param.enableFlashComm) {
        atb_speed::Model::Node allGatherVNode;
        atb::infer::AllGatherVParam allGatherVParam;
        allGatherVParam.rank = this->param.rank;
        allGatherVParam.rankSize = this->param.worldSize;
        allGatherVParam.backend = this->param.backend;
        if (this->param.mapping.isInitialized_) {
            atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
            parallelInfo.InitCommDomain(allGatherVParam.hcclComm, allGatherVParam.commDomain);
        }
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherVParam, &op));
        allGatherVNode.operation.reset(op);
        allGatherVNode.inTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
            this->internalTensorMap, "hidden_states_rank" + std::to_string(param.rank)))};
        allGatherVNode.inTensors.emplace_back(
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "recv_count")));
        allGatherVNode.inTensors.emplace_back(
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "send_counts")));
        allGatherVNode.inTensors.emplace_back(
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "sdispls")));
        allGatherVNode.inTensors.emplace_back(
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "fake_ag_shape")));
        allGatherVNode.outTensors = {&graph_.internalTensors.at(atb_speed::common::GetTensorIdx(
            this->internalTensorMap, "hidden_states_allgather"))};
        ATB_SPEED_LOG_DEBUG("[+] allGatherVNode");
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::AddDapEventsBeforeComm(this->graph_));
        graph_.nodes.push_back(allGatherVNode);
        CHECK_OPERATION_STATUS_RETURN(atb_speed::common::AddDapEventsAfterComm(this->graph_));
    }
    return atb::NO_ERROR;
}

void DecoderModel::SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam)
{
    lmHeadParam.unpadInputs = this->param.isUnpadInputs;
    lmHeadParam.gatherAhead = this->param.isPrefill || this->param.enablePrefixCache || this->param.hasAttnDp;
    lmHeadParam.hiddenSizePerAttentionHead = this->param.hiddenSizePerAttentionHead;
    lmHeadParam.linearParallelParam.fusionLinearParam.isBF16 = this->param.isBF16;
    lmHeadParam.linearParallelParam.fusionLinearParam.transposeType = this->param.lmHeadTransposeType;
    lmHeadParam.linearParallelParam.fusionLinearParam.matmulBackend = param.matmulBackend;
    lmHeadParam.linearParallelParam.unpadInputs = !this->param.isFA;
    lmHeadParam.linearParallelParam.enableMC2 = this->param.enableMC2;
    lmHeadParam.linearParallelParam.isArgmaxlogits = this->param.enableGreedyPostProcessing;
    lmHeadParam.linearParallelParam.worldSize = this->param.worldSize;
    if (this->param.isLmHeadParallel && this->param.hasPp) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo = {
            this->param.tpRank, this->param.tpWorldSize, this->param.backend, this->param.tpRankTableFile, \
            gHcommInfo, this->param.tpDomain};
    } else if (this->param.isLmHeadParallel && this->param.hasAttnDp && this->param.hasMlpTp) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo = {
            this->param.mlpTpRank, this->param.mlpTpSize, this->param.backend, this->param.mlpTpRankTableFile,
            gHcommInfo, this->param.mlpTpDomain};
    } else if (this->param.isLmHeadParallel) {
        lmHeadParam.linearParallelParam.parallelType = atb_speed::common::COLUMN_PARALLEL;
        lmHeadParam.linearParallelParam.tensorParallelInfo = {
            this->param.rank, this->param.worldSize, this->param.backend, this->param.rankTableFile, gHcommInfo};

        if (this->param.mapping.isInitialized_) {
            atb_speed::common::ParallelInfo parallelInfo = param.mapping.Get(base::LM_HEAD_TP);
            parallelInfo.InitCommDomain(
                lmHeadParam.linearParallelParam.tensorParallelInfo.hcommInfo,
                lmHeadParam.linearParallelParam.tensorParallelInfo.commDomain);
        }
    }
}

atb::Status DecoderModel::AddLmhead()
{
    atb::Operation *op = nullptr;

    atb_speed::Model::Node lmHeadNode;
    atb_speed::common::LmHeadParam lmHeadParam;
    this->SetLmHeadParam(lmHeadParam);
    CHECK_OPERATION_STATUS_RETURN(LmHead(lmHeadParam, &op));
    lmHeadNode.operation.reset(op);
    const uint64_t finalLinearWeightTensorId = graph_.weightTensors.size() - this->weightCountLmHead;
    uint32_t placeHolderIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "place_holder");

    if (this->param.hasAttnDp && this->param.hasMlpTp) {
            lmHeadNode.inTensors = {
            this->param.skipWordEmbedding ? \
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
                &graph_.internalTensors.at(
                    atb_speed::common::GetTensorIdx(this->internalTensorMap, "attn_dp_last_layer"))
            };
    } else if (this->param.enableFlashComm) {
        lmHeadNode.inTensors = { &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap,
            "hidden_states_allgather"))};
    } else {
        lmHeadNode.inTensors = {
            this->param.skipWordEmbedding ? \
                &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
                &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states")),
        };
    }
    // shape: [vocabSizePerRank, hiddenSize]
    lmHeadNode.inTensors.push_back(&graph_.weightTensors.at(finalLinearWeightTensorId));
    // LmHead未接入量化，量化权重使用placeholder代替
    lmHeadNode.inTensors.push_back(&graph_.inTensors.at(placeHolderIdx));
    lmHeadNode.inTensors.push_back(&graph_.inTensors.at(placeHolderIdx));
    lmHeadNode.inTensors.push_back(&graph_.inTensors.at(placeHolderIdx));
    lmHeadNode.inTensors.push_back(&graph_.inTensors.at(placeHolderIdx));
    lmHeadNode.inTensors.push_back(&graph_.inTensors.at(placeHolderIdx));
    lmHeadNode.inTensors.push_back(&graph_.inTensors.at(
        atb_speed::common::GetTensorIdx(this->inTensorMap, "logits_indices")));
    if (this->param.enableGreedyPostProcessing) {
        lmHeadNode.inTensors.emplace_back(&graph_.inTensors.at(atb_speed::common::GetTensorIdx(
            this->inTensorMap, "logits_offset_tensor")));
    } else {
        lmHeadNode.inTensors.emplace_back(&graph_.inTensors.at(placeHolderIdx));
    }
    // shpae: FA: [batchSize, seqLen, vocabSize] PA: [seqLen, vocabSize]
    lmHeadNode.outTensors = {&graph_.outTensors.at(atb_speed::common::GetTensorIdx(this->outTensorMap, "logits"))};
    if (this->param.enableFlashComm) {
        lmHeadNode.inTensorReshapeFuncs.resize(lmHeadNode.inTensors.size());
        lmHeadNode.inTensorReshapeFuncs.at(0) = &atb_speed::common::SqueezeBatchAndSeq;
    }
    graph_.nodes.push_back(lmHeadNode);
    ATB_SPEED_LOG_DEBUG("[+] base lmHeadNode");
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

void DecoderModel::ParseDapParam(nlohmann::json &paramJson)
{
    this->seqLenForDap.Parse("seqLen", paramJson);
    this->tokenOffsetForDap.Parse("tokenOffset", paramJson);
    this->qLenForDap.Parse("qLen", paramJson);
    if (param.enableFlashComm) {
        this->sendCounts.Parse("sendCounts", paramJson);
        this->sdispls.Parse("sdispls", paramJson);
        this->sendCount.Parse("sendCount", paramJson);
        this->recvCounts.Parse("recvCounts", paramJson);
        this->rdispls.Parse("rdispls", paramJson);
        this->recvCount.Parse("recvCount", paramJson);
    }
}

void DecoderModel::ParseParallelParam(nlohmann::json &paramJson)
{
    if (paramJson.contains("seqLenCp")) {
        this->seqLenCp.Parse("seqLenCp", paramJson);
    }
    if (paramJson.contains("seqLenSp")) {
        this->seqLenSp.Parse("seqLenSp", paramJson);
    }
    if (paramJson.contains("isNeedMask")) {
        this->isNeedMask.Parse("isNeedMask", paramJson);
    }
}

void DecoderModel::ParseFlashCommParam(nlohmann::json &paramJson)
{
    if (param.enableFlashComm) {
        this->sendCounts.Parse("sendCounts", paramJson);
        this->sdispls.Parse("sdispls", paramJson);
        this->sendCount.Parse("sendCount", paramJson);
        this->recvCounts.Parse("recvCounts", paramJson);
        this->rdispls.Parse("rdispls", paramJson);
        this->recvCount.Parse("recvCount", paramJson);
    }
}

atb::Status DecoderModel::ParseParam(const std::string &paramString)
{
    CHECK_PARAM_LT(paramString.size(), MAX_PARAM_STRING_LENGTH);
    nlohmann::json paramJson = StringToJson(paramString);

    // Dap use dynamicParam to store all params instead of using attribute to store single param
    if (param.enableDap) {
        ParseDapParam(paramJson);
        return atb::NO_ERROR;
    }

    this->tokenOffset.clear();
    for (auto item : paramJson["tokenOffset"]) {
        this->tokenOffset.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("token offset value: " << item);
    }

    if (param.enableOmniAttention) {
        this->seqLenTmp.clear();
        for (auto item : paramJson["seqLen"]) {
            this->seqLenTmp.push_back(item.get<int>());
            ATB_SPEED_LOG_DEBUG("seqLen value: " << item);
        }
        this->seqLen.clear();
        ExpandVectorToN(this->seqLenTmp, this->seqLen, param.numHiddenLayers);
    } else {
        this->seqLen.clear();
        for (auto item : paramJson["seqLen"]) {
            this->seqLen.push_back(item.get<int>());
            ATB_SPEED_LOG_DEBUG("seqLen value: " << item);
        }
    }

    this->qLen.clear();
    for (auto item : paramJson["qLen"]) {
        this->qLen.push_back(item.get<int>());
        ATB_SPEED_LOG_DEBUG("qLen value: " << item);
    }

    ParseParallelParam(paramJson);

    ParseFlashCommParam(paramJson);
    if (param.enablePrefixCache) {
        this->ringCurSeqlen.Parse("ringCurSeqlen", paramJson);
        this->ringCacheSeqlen.Parse("ringCacheSeqlen", paramJson);
        if (paramJson.contains("kvCachelen")) {
            this->kvCachelen.Parse("kvCachelen", paramJson);
        }
    }
    return atb::NO_ERROR;
}

template <typename T>
void DecoderModel::BindDapHostTensor(DynamicParam<std::vector<T>>& dynamicParam, std::string tensorName)
{
    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, tensorName);
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = dynamicParam.Get().data();
    }
    if (param.enableDap) {
        GetSingleton<common::DapManager>().SetRole(common::DapRole::SUCCESSOR);
        std::string suffix = GetSingleton<common::DapManager>().GetSuccessorSuffix();
        tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, tensorName + suffix);
        if (tensorIdx != UINT32_MAX) {
            graph_.inTensors.at(tensorIdx).hostData = dynamicParam.Get().data();
        }
        GetSingleton<common::DapManager>().SetRole(common::DapRole::PRECEDER);
    }
}

template void DecoderModel::BindDapHostTensor<int>(DynamicParam<std::vector<int>>& dynamicParam,
    std::string tensorName);

template void DecoderModel::BindDapHostTensor<int64_t>(DynamicParam<std::vector<int64_t>>& dynamicParam,
    std::string tensorName);

atb::Status DecoderModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        // 仅需在graph的intensor中bind一次
        return atb::NO_ERROR;
    }
    if (param.enableFlashComm) {
        BindDapHostTensor(this->sendCounts, "send_counts");
        BindDapHostTensor(this->sdispls, "sdispls");
        BindDapHostTensor(this->sendCount, "send_count");
        BindDapHostTensor(this->recvCounts, "recv_counts");
        BindDapHostTensor(this->rdispls, "rdispls");
        BindDapHostTensor(this->recvCount, "recv_count");
    }
    if (param.enableDap) {
        BindDapHostTensor(this->seqLenForDap, "seq_len");
        BindDapHostTensor(this->tokenOffsetForDap, "token_offset");
        BindDapHostTensor(this->qLenForDap, "q_len");
        return atb::NO_ERROR;
    }

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->tokenOffset.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->seqLen.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->qLen.data();
    }
    if (param.enablePrefixCache) {
        tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "ring_cur_seqlen");
        if (tensorIdx != UINT32_MAX) {
            graph_.inTensors.at(tensorIdx).hostData = this->ringCurSeqlen.Get().data();
        }
        tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "ring_cache_seqlen");
        if (tensorIdx != UINT32_MAX) {
            graph_.inTensors.at(tensorIdx).hostData = this->ringCacheSeqlen.Get().data();
        }
    }
    ATB_SPEED_LOG_DEBUG("BindParamHostTensor end");
    return atb::NO_ERROR;
}

/**
 * generate the seq length for each layer of each batch, the result should look like
 * [layer, batch] = compressed_head ？ min(384, seq_len) :seq_len
 */
atb::Status DecoderModel::ExpandVectorToN(const std::vector<int>& input, std::vector<int>& output, uint32_t layernums)
{
    // the number of active batches
    size_t batchSize = input.size();
    // Check if input vector is empty or N is 0
    if (layernums == 0 || batchSize == 0) {
        throw std::invalid_argument("Input vector cannot be empty");
    }
    for (size_t layer = 0; layer < layernums; layer++) {
        for (size_t batch = 0; batch < batchSize; batch++) {
            bool isCompressed = param.patternMask[layer] == 1;
            int batchSeqlen = input[batch];
            const int omniLimitSeqLen = 384;
            isCompressed ? output.push_back(std::min(batchSeqlen, omniLimitSeqLen)) :
            output.push_back(batchSeqlen);
        }
    }

    return atb::NO_ERROR;
}

} // namespace base
} // namespace atb_speed