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

#ifndef ASCEND_SPEED_INFERENCE_COMMON_NORM_LINEAR_H
#define ASCEND_SPEED_INFERENCE_COMMON_NORM_LINEAR_H

#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {

/// Parameters for the normalization and linear module
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
template <typename NormParamType>
struct NormLinearParam {
    /// A flag indicating whether anti-outlier is enabled
    bool isAntiOutlier = false;
    /// A flag indicating whether normalization is skipped
    bool skipNorm = false;
    /// A flag indicating whether normalization has bias
    bool normHasBias = false;
    /// A flag indicating whether to use the AddNorm fusion operation
    bool enableAddNorm = false;
    /// Normalization parameters for float operation
    NormParamType normParamType;
    /// Normlization parameters for quantization operation
    NormParamType normQuantParamType;
    /// Parameters for the FusionLinear module
    atb_speed::common::FusionLinearParam fusionLinearParam;
    /// A flag indicating whether to use pmcc obfuscation
    bool enableModelConfuscation = false;
    /// A handle used by pmcc model obfuscation
    int32_t modelConfuscationFd = 0;
    /// Hidden size per rank
    int32_t hiddenSizePerRank = 0;
    /// Parallel info, now only used by pmcc obfuscation in multirank scenario
    atb_speed::common::TensorParallelInfo modelObfuscationParallelInfo;
};

/// Get `LinearQuantType` by the quantization type of the linear modules and the position of the linear module
/// \param packQuantType The quantization type of the packed linear modules. Refer to `PackQuantType`
/// in the `operations/utils.h`.
/// \param linearType The type of one linear module. Refer to `LinearType` in the `operations/utils.h`.
/// \param hasNorm A flag indicating whether the linear module includes a preceding normalization module
LinearQuantType GetLinearQuantType(
    const int &packQuantType = PackQuantType::PACK_QUANT_UNDEFINED,
    const int &linearType = LinearType::INVALID,
    bool hasNorm = false,
    const int &linearDesc = LinearDesc::INVALID_DESC);

/// The function construct an operation that combines a normalization module with a `FusionLinear` module.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param param Parameters for the normalization and linear module
/// \param operation the address of a pointer to a default operation
/// \return A flag that indicates whether operation has been successfully created.
///
/// Operation's inputs:
/// Name               | Dtype            | Shape | Description |
/// -------------------|------------------|-------|----------|
/// in_input           | float16/bfloat16 | [m,k] |          |
/// in_norm_weight     | float16/bfloat16 | [k]   |          |
/// in_norm_bias       | float16/bfloat16 | [k]   | Used when `param.normHasBias` is true |
/// in_norm_new_weight | float16/bfloat16 | [k]   | Used when `param.isAntiOutlier` is true |
/// in_norm_new_bias   | float16          | [1]   | Used when `param.normHasBias` and `param.isAntiOutlier` is true |
/// in_linear_weight   |                  |       | The same specifications as the FusionLinear module |
/// in_scale           |                  |       | The same specifications as the FusionLinear module |
/// in_offset          |                  |       | The same specifications as the FusionLinear module |
/// in_descale         |                  |       | The same specifications as the FusionLinear module |
/// in_bias            |                  |       | The same specifications as the FusionLinear module |
/// in_compress_idx    |                  |       | The same specifications as the FusionLinear module |
/// in_residual_input  | float16/bfloat16 | [m,k] | Used when `enableAddNorm` is true |
/// in_seq_len_cum_sum |                  |       | The same specifications as the FusionLinear module |
/// in_linear_lora_a   |                  |       | The same specifications as the FusionLinear module |
/// in_linear_lora_b   |                  |       | The same specifications as the FusionLinear module |
/// in_im_mask         |                  |       | The same specifications as the FusionLinear module |
///
/// Operations's outputs:
/// Name       | Dtype            | Shape | Description |
/// -----------|------------------|-------|----------|
/// out_linear | float16/bfloat16 | [m,n] | Output tensor of the linear module |
/// out_add    | float16/bfloat16 | [m,k] | Output tensor of the residual add. Exist when `enableAddNorm` is true. |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_NORM_WEIGHT,
///     IN_NORM_BIAS,
///     IN_LINEAR_WEIGHT,
///     IN_PLACEHOLDER,
///     OUT,
/// };
///
/// atb::Node normLinearNode;
/// atb_speed::common::NormLinearParam<atb::infer::RmsNormParam> normLinearParam;
/// // Modify normLinearParam's attribute if needed.
/// NormLinear(normLinearParam, &normLinearNode.operation);
/// normLinearNode.inTensorIds = {IN_INPUT, IN_NORM_WEIGHT, IN_NORM_BIAS, IN_PLACEHOLDER, IN_PLACEHOLDER,
/// IN_LINEAR_WEIGHT, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER, IN_PLACEHOLDER};
/// normLinearNode.outTensorIds = {OUT};
/// atb::GraphParam opGraph;
/// opGraph.nodes.push_back(normLinearNode);
/// \endcode
template <typename NormParamType>
atb::Status NormLinear(const NormLinearParam<NormParamType> &param, atb::Operation **operation);

/// This function add a normalization node into the graph.
///
/// \tparam NormParamType Types of the normalization parameters. Avaliable options are `atb::infer::RmsNormParam`
/// and `atb::infer::LayerNormParam`.
/// \param opGraph the graph to be constructed
/// \param param Parameters for the normalization and linear module
/// \param tensorMap A map contains all the required tensors for the node, with the key representing
/// the input tensor name and the value corresponding to the tensor index. This map is used to identify the node's input
/// and output tensors base on tensor names.
/// \return A flag that indicates whether operation has been successfully added to the graph.
///
/// Example:
/// \code
/// atb_speed::common::NormLinearParam<atb::infer::RmsNormParam> normParam;
/// // Modify normParam's attribute if needed.
/// std::map<std::string, uint32_t> normTensorMap = std::vector<std::string> targetNames = {
///    {"in_input", 0}, {"in_norm_weight", 1}, {"in_norm_bias", 2}, {"in_norm_new_weight", 3}, {"in_norm_new_bias", 4},
///    {"in_scale", 5}, {"in_offset", 6}, {"intermediate_norm", 7}, {"out_add", 8}, {"in_residual_input", 9}
///};
/// atb::GraphParam graph;
/// atb_speed::common::InsertNorm(graph, normParam, normTensorMap);
/// \endcode
template <typename NormParamType>
int64_t InsertNorm(atb::GraphParam &opGraph, const NormLinearParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap);

} // namespace common
} // namespace atb_speed

#endif
