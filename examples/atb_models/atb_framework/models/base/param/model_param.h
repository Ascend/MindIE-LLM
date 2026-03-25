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

#ifndef ATB_SPEED_BASE_MODEL_PARAM_H
#define ATB_SPEED_BASE_MODEL_PARAM_H
#include <vector>
#include <nlohmann/json.hpp>
#include "models/base/param/param.h"
#include "models/base/param/param_utils.h"

const int LWD_EDGE_FIRST = 0;
const int LWD_CLOUD_MIDDLE = 1;
const int LWD_EDGE_LAST = 2;

namespace atb_speed {
namespace base {

enum MemPoolType {
    DISABLED = 0,
    SYNC_WRITE = 1,
    ASYNC_WRITE = 2
};

/// Parse string to `nlohmann::json`
/// \param param parameters in JSON string format passed from the Python side
/// \return parameters in `nlohmann::json` format
nlohmann::json StringToJson(const std::string &param);

/// Parameters for the base model, inherited from the `Param` class.
///
/// In addition to the parameters defined in the Param class,
/// this class introduces additional parameters specific to the base `DecoderModel` class.
/// Models can inherit from this class and define further parameters tailored to their specific requirements.
class ModelParam : public Param {
public:
    ModelParam() {};
    ~ModelParam() override {};
    /// Parse the input JSON string to a `ModelParam` object, validate its contents, and print parsed values.
    /// \param param parameters in JSON string format passed from the Python side
    void FromString(const std::string &param);
    void PrintParam() override;
    void CheckParam() override;

    /// When `skipWordEmbedding` is true, input embedding is provided and the word embedding module is skipped;
    /// otherwise, input token ids are used.
    bool skipWordEmbedding = false;
    // When `isEmbeddingParallel` is true, the embedding weights are partitioned along the hiddenSize dimension;
    /// otherwise, the weights are not partitioned.
    bool isEmbeddingParallel = false;
    /// When `isLmHeadParallel` is true, the LmHead weights are partitioned; otherwise, the weight is not partitioned.
    bool isLmHeadParallel = true;
    /// A flag indicating whether the second matrix in the matmul operation within the LmHead module is transposed.
    int lmHeadTransposeType = -1;
    /// A flag indicating whether post processing greedy search
    bool enableGreedyPostProcessing = false;
    ///  Whether to enable data async parallelism (DAP) for the model.
    bool enableDap = false;
    /// Number of hidden layers
    uint32_t numHiddenLayers = 0;
    // Model Parallelism
    int rank = 0;
    int worldSize = 1;
    int localWorldSize = 1;
    bool hasPp = false;
    int ppGroupSize = 0;
    bool firstPpRank = true;
    bool lastPpRank = true;
    int prevPpRank = 0;
    int nextPpRank = 0;
    int tpRank = 0;
    int tpWorldSize = 0;

    /// The following variables will only be used when layerwiseDisaggregated is enabled.
    bool layerwiseDisaggregated = false;
    int32_t layerwiseMode = -1;
    int32_t hiddenSize = 0;
    int32_t startLayerId = 0;
    int32_t endLayerId = 64;
    bool isInternalLayer = false;
    bool reuseEmbedTable = false;
    bool outputEmbedTable = false;

    MemPoolType memPoolType = MemPoolType::DISABLED;
    // When `mempool_type` is ASYNC_WRITE, a event pipeKey is set for async layer-wise prefix cache transfer.
    std::string memPoolEventPipeKey = "default";
    std::string backend = "hccl";
    std::string tpDomain = "";
    std::string rankTableFile = "";
    std::string tpRankTableFile = "";
    /// Indicates the pack type and the quantization type of the QKV linear and Gate UP linear for each layer.
    /// The number of inner vector corresponds to `numHiddenLayers`.
    /// Each inner vector contains two integer: the first one represents the pack and the quantization type
    /// of the qkv linear and the second one represents the pack and the quantization type of the gate up linear.
    /// The pack types the quantization types are defined in the `PackQuantType` enumerator.
    std::vector<std::vector<int>> packQuantType = {};
    /// Specifies the quantization type for each linear in every layer.
    /// The number of inner vector corresponds to `numHiddenLayers`.
    /// Each inner vector contains seven interger, representing the quantization types of the following layers:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    /// The quantization types are defined in the `LinearType` enumerator.
    std::vector<std::vector<int>> linearQuantType = {};
    /// Defines the transpose type of the second matrix in the matmul operation for each linear in every layer.
    /// The number of inner vector corresponds to `numHiddenLayers`.
    /// Each inner vector contains seven interger, representing the quantization types of the following layers:
    /// q linear, k linear, v linear, dense linear, gate linear, up linear, and down linear.
    /// The transpose types are defined in the `TransposeType` enumerator.
    std::vector<std::vector<int>> linearTransposeType = {};
    /// Specifies whether linear module has bias
    /// The number of inner vector corresponds to `numHiddenLayers`.
    /// Each inner vector contains four boolean value, indicating whether the following linear module has bias:
    /// qkv linear, dense linear, gateup linear and down linear.
    std::vector<std::vector<bool>> linearHasBias = {};
    std::vector<std::vector<int>> linearDescs = {};
    std::vector<std::vector<bool>> isAntiOutlier = {};

protected:
    /// Convert an `nlohmann::json` object to a `ModelParam` object.
    /// \param paramJson An `nlohmann::json` object holds all the required parameters.
    virtual void ParseParam(const nlohmann::json &paramJson);
    /// Converts normalization-related parameters from the `nlohmann::json` object
    /// into attributes of the `ModelParam` object.
    /// This function is called by `ParseParam`.
    /// \param paramJson An `nlohmann::json` object holds all the required parameters.
    void ParseNormParam(const nlohmann::json &paramJson);
    /// Converts attention-related parameters from the `nlohmann::json` object
    /// into attributes of the `ModelParam` object.
    /// This function is called by `ParseParam`.
    /// \param paramJson An `nlohmann::json` object holds all the required parameters.
    virtual void ParseAttentionParam(const nlohmann::json &paramJson);
    /// Converts mlp-related parameters from the `nlohmann::json` object
    /// into attributes of the `ModelParam` object.
    /// This function is called by `ParseParam`.
    /// \param paramJson An `nlohmann::json` object holds all the required parameters.
    void ParseMlpParam(const nlohmann::json &paramJson);
    /// Converts matmul-related parameters from the `nlohmann::json` object into attributes of the `ModelParam` object.
    /// This function is called by `ParseParam`.
    /// \param paramJson An `nlohmann::json` object holds all the required parameters.
    virtual void ParseMatmulParam(const nlohmann::json &paramJson);
    /// Converts parallelism-related related parameters from the `nlohmann::json` object
    /// into attributes of the `ModelParam` object.
    /// This function is called by `ParseParam`.
    /// \param paramJson An `nlohmann::json` object holds all the required parameters.
    virtual void ParseTensorParallelParam(const nlohmann::json &paramJson);
    virtual void ParseParallelismParam(const nlohmann::json &paramJson);
    virtual void ParseLayerwiseDisaggregatedParm(const nlohmann::json &paramJson);
};

} // namespace base
} // namespace atb_speed
#endif