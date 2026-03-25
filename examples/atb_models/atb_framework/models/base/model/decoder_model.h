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

#ifndef ATB_SPEED_MODELS_BASE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_BASE_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "atb_speed/base/external_comm_manager.h"
#include "models/base/param/model_param.h"
#include "models/base/param/dynamic_param.h"
#include "models/base/layer/decoder_layer.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/utils.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/tensor_util.h"

namespace atb_speed {
namespace base {

/// Base class for large language models, inherited from the `Model` class.
class DecoderModel : public Model {
public:
    explicit DecoderModel(const std::string &param);
    ~DecoderModel() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

protected:
    /// Select input tensors from all the available input tensor candidates based on the provided parameters `param`.
    /// Automatically assign an index to each tensor.
    /// The order of the input tensors should align with the sequence passed from the Python side.
    virtual void ConstructInTensorMap();
    /// Select internal tensors from all the available internal tensor candidates
    /// based on the provided parameters `param`. Automatically assign an index to each tensor.
    virtual void ConstructInternalTensorMap();
    /// Select output tensors from all the available output tensor candidates
    /// based on the provided parameters `param`. Automatically assign an index to each tensor.
    virtual void ConstructOutTensorMap();
    /// Return the total number of weight tensors. It can differ depending on various features.
    virtual uint32_t CalcWeightTensorSize();
    virtual void SetLayerwiseDisaggregatedParam();
    atb::Status ParseParam(const std::string &paramString) override;
    virtual void ParseDapParam(nlohmann::json &paramJson);
    virtual void ParseFlashCommParam(nlohmann::json &paramJson);
    virtual void ParseParallelParam(nlohmann::json &paramJson);
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    template <typename T>
    void BindDapHostTensor(DynamicParam<std::vector<T>>& dynamicParam, std::string tensorName);
    /// Update the `wordEmbeddingParam` using the values from `param`.
    /// \param wordEmbeddingParam an `WordEmbeddingParam` object that needs to be updated
    virtual void SetWordEmbeddingParam(atb_speed::common::WordEmbeddingParam &wordEmbeddingParam);
    /// Update the `layerParam` using the values from `param`.
    /// \param layerParam an `LayerParam` object that needs to be updated
    /// \param layerId the index of the current layer
    virtual void SetLayerParam(LayerParam &layerParam, uint32_t layerId);
    /// Update the `layerParam` using the values from `param`.
    /// \param layerParam an `LayerParam` object that needs to be updated
    virtual void SetLayerParallelismParam(LayerParam &layerParam);
    /// Update the `normParam` using the values from `param`.
    /// \param normParam an `RmsNormParam` object that needs to be updated
    virtual void SetFinalNormParam(atb::infer::RmsNormParam &normParam);
    /// Update the `normParam` using the values from `param`.
    /// \param normParam an `LayerNormParam` object that needs to be updated
    virtual void SetFinalNormParam(atb::infer::LayerNormParam &normParam);
    /// Update the `lmHeadParam` using the values from `param`.
    /// \param lmHeadParam an `LmHeadParam` object that needs to be updated
    virtual void SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam);
    /// The main entrance to set `layerNode`'s input tensors.
    /// It will call `SetLayerNodeDefaultInput` and `SetLayerNodeOptionalInput`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param layerId the index of the current layer
    virtual void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId);
    /// Set `layerNode`'s default input tensors based on the values from `param`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param layerId the index of the current layer
    /// \param inTensorId the starting input tensor IDs
    virtual void SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    /// Set `layerNode`'s optional input tensors based on the values from `param`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param layerId the index of the current layer
    /// \param inTensorId the starting input tensor IDs
    virtual void SetLayerNodeOptionalInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    /// Set `layerNode`'s input tensors for razor attention based on the values from `param`.
    /// It will be called by `SetLayerNodeOptionalInput`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param layerId the index of the current layer
    /// \param inTensorId the starting input tensor IDs
    virtual void SetLayerNodeRaInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    /// Set `layerNode`'s input tensors for omni attention based on the values from `param`.
    /// It will be called by `SetLayerNodeOptionalInput`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param layerId the index of the current layer
    /// \param inTensorId the starting input tensor IDs
    virtual void SetLayerNodeOmniInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    /// Set `layerNode`'s input tensors for multi lora based on the values from `param`.
    /// It will be called by `SetLayerNodeOptionalInput`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param layerId the index of the current layer
    /// \param inTensorId the starting input tensor IDs
    virtual void SetLayerNodeLoraInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    /// Set `layerNode`'s input tensors for data parallalism in the attention module based on the values from `param`.
    /// It will be called by `SetLayerNodeOptionalInput`.
    /// \param layerNode an `atb_speed::Model::Node` object that needs to be updated
    /// \param inTensorId the starting input tensor IDs
    virtual void SetLayerNodeAttnDpInput(atb_speed::Model::Node &layerNode, uint32_t &inTensorId);
    /// Create an `LayerParam` object, call `SetLayerParam` and
    /// call `DecoderLayer`'s `buildGraph` function to create an operation.
    /// \param op the address of a pointer to a default operation
    /// \param layerId the index of the current layer
    /// \return A flag indicates whether the operation was successfully created.
    virtual atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId);
    /// Add a word embedding node to the graph to convert token ids to embedding.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddWordEmbedding();
    /// Add a positional embedding node to the graph to convert positional ids to embedding.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddPositionalEmbedding();
    /// Add all layer nodes to the graph.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddLayer();
    /// Add a layer node to the graph.
    /// Overriding this function is advised in order to easily adapt the model to DAP.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddSingleLayer(uint32_t layerId);
    /// Add a normalization node to the graph.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddFinalNorm();
    /// Add a lmhead node to the graph.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddLmhead();
    /// Add a cmo synchronization node to the graph to synchronize streams.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddCmoSync();
    /// Add a send node to the graph to send hidden states to the next pipeline parallalism stage.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddSend();
    /// Add a receive node to the graph to receive hidden states to the next pipeline parallalism stage.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddRecv();
    /// Add all the operations before layer node.
    /// Overriding this function is advised in order to easily adapt the model to DAP.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddNodesBeforeLayer();
    /// Add all the operations after layer node.
    /// Overriding this function is advised in order to easily adapt the model to DAP.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddNodesAfterLayer();
    /// The primary entry point for adding all operations to the graph in sequential order.
    /// It is recommended to override `AddNodesBeforeLayer`, `AddNodesAfterLayer`, and `AddSingleLayer`
    /// instead of this funciton.
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddOperationToGraph();
    /// add a split hiddenstate node for flashcomm
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddSplitHiddenStates();
    /// add a allgather hiddenstate node for flashcomm
    /// \return A flag indicates whether the operation was successfully added to the graph.
    virtual atb::Status AddAllGather();
    /// for flashcomm 1.0
    /// Data sent for every card
    DynamicParam<std::vector<int64_t>> sendCounts;
    /// It signifies that the current rank commences sent data from rank i
    /// at an offset of n with respect to the input's initiation point
    DynamicParam<std::vector<int64_t>> sdispls;
    /// Data sent by this card
    DynamicParam<std::vector<int64_t>> sendCount;
    /// Data received for every card
    DynamicParam<std::vector<int64_t>> recvCounts;
    /// It signifies that the current rank commences receiving data from rank i
    /// at an offset of n with respect to the input's initiation point
    DynamicParam<std::vector<int64_t>> rdispls;
    /// Data sent by this card
    DynamicParam<std::vector<int64_t>> recvCount;

    int64_t BuildGraph() override;
    /// omni-attention
    atb::Status ExpandVectorToN(const std::vector<int>& input, std::vector<int>& output, uint32_t N);

    /// Parameters that will change during each forward pass.
    /// The position ID of the current token in the inference sequence.
    /// Number of elements should be equal to batch size.
    std::vector<int> tokenOffset = {};
    /// The total number of input and output tokens.
    /// In the prefill phase, each elements equals to the length of the prompt.
    /// For flash attention, each element is set to 1 in the decode phase.
    /// For paged attention, each element is set to the number of input tokens plus output tokens in the decode phase.
    /// Number of elements should be equal to batch size.
    std::vector<int> seqLen = {};
    // for omni_attention
    std::vector<int> seqLenTmp = {};
    /// Number of input tokens for the current forward pass.
    /// Number of elements should be equal to batch size.
    std::vector<int> qLen = {};
    // For Attn Inner SP
    DynamicParam<std::vector<int>> seqLenSp;
    // For Attn Inner SP and mtp = 1
    DynamicParam<std::vector<int>> isNeedMask;
    // For Attn CP
    DynamicParam<std::vector<int>> seqLenCp;
    // For prefixcache ring_cur + ring_cache
    DynamicParam<std::vector<int>> ringCurSeqlen = {};
    DynamicParam<std::vector<int>> ringCacheSeqlen = {};
    DynamicParam<std::vector<int>> kvCachelen = {};
    
    // For Dap
    DynamicParam<std::vector<int>> seqLenForDap;
    DynamicParam<std::vector<int>> tokenOffsetForDap;
    DynamicParam<std::vector<int>> qLenForDap;

    /// Specifies all potential input tensors, where the key represents the feature name,
    /// and the value corresponds to the input tensor name.
    std::map<std::string, std::vector<std::string>> inTensorCandidates = {};
    /// Specifies all potential internal tensors, where the key represents the feature name,
    /// and the value corresponds to the internal tensor name.
    std::map<std::string, std::vector<std::string>> internalTensorCandidates = {};
    /// Specifies all potential output tensors, where the key represents the feature name,
    /// and the value corresponds to the output tensor name.
    std::map<std::string, std::vector<std::string>> outTensorCandidates = {};
    /// Defines all the required input tensors for the current graph, with the key representing the input tensor name
    /// and the value corresponding to the tensor index.
    std::map<std::string, uint32_t> inTensorMap = {};
    /// Defines all the required internal tensors for the current graph,
    /// with the key representing the input tensor name
    /// and the value corresponding to the tensor index.
    std::map<std::string, uint32_t> internalTensorMap = {};
    /// Defines all the required output tensors for the current graph, with the key representing the input tensor name
    /// and the value corresponding to the tensor index.
    std::map<std::string, uint32_t> outTensorMap = {};
    /// Number of weights per layer
    uint32_t weightCountPerLayer = 50;
    /// Number of weights for the word embedding node
    uint32_t weightCountWordEmbedding = 1;
    /// Number of weights for the normalization node
    uint32_t weightCountFinalNorm = 1;
    /// Number of weights for the lmhead node
    uint32_t weightCountLmHead = 1;
    // Pointer of hccl communication domain for mc2
    static HcclComm gHcommInfo;

    /// Model parameters
    ModelParam param;

private:
    void PrintTensorMapInfo(std::map<std::string, uint32_t> &tensorMap) const;
    void DuplicateTensorMapForDap(
        std::map<std::string, uint32_t> &tensorMap, std::vector<atb::Tensor> &targetTensors);
    std::map<uint32_t, uint32_t> CopyMapWithSuffix(std::map<std::string, uint32_t>& tensorMap) const;
    std::map<atb::Tensor *, atb::Tensor *> precederToSuccessorTensorMap = {};
    void ReplaceDapTensors(std::vector<atb::Tensor *>& tensors);
    atb::Status WaitEventAfterPrefixCacheLoad();
    atb::Status RecordEventBeforePrefixCacheSave();
};

}  // namespace base
}  // namespace atb_speed
#endif
