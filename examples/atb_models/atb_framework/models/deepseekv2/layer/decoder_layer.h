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

#ifndef ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_LAYER_H
#define ATB_SPEED_MODELS_DEEPSEEK_V2_DECODER_LAYER_H

#include <vector>
#include <atb/comm.h>
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "atb_speed/base/external_comm_manager.h"
#include "atb_speed/utils/operation_util.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseekV2 {
class DecoderLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    bool enableFusedRouting = true;
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = false;
    bool isDenseLayer = false;
    bool isLastLayer = false;
    bool isDynamicEp = false;
    bool hasP2DWeight = false;
    bool enableCVOverlap = false; /// A flag indicating whether the model use cube and vector parallel
    bool enableExtraOprojTp = false;
    bool enableATBGateMatmul = false;
    bool enableMlaPrefetch = false;
    int maskStartIdx = 0;
    int layerId = 0;
    int numHiddenLayers = 0;
    int firstKDenseReplace = 1;
    int numOfSharedExperts = 2;       // 2:Defaulting the number of shared experts to 2
    int rank = 0;
    int worldSize = 1;
    // quant 参数
    int mlpNormQuantType = atb::infer::QUANT_UNDEFINED;
    bool isAntiOutlier = false;
    // Grouped topk参数
    int numOfGroups = 1;
    int scaledTopk = -1; /// 非deepseek模型默认不启用scaledTopk特性
    bool enableInitRoutingCutoff = false;  /// A flag indicating whether to use scaled topk option
    float routedScalingFactor = 1;
    bool enableFusedTopk = false;
    // MLA参数
    int qLoraRank = 1536;
    int kvLoraRank = 512;
    int headNum = 128;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    float softmaxScale = 0;
    bool enableMlaPreprocess = false;
    bool isNzCache = false;
    bool enablePrefixCache = false;
    /// The following variables will only be used when layerwiseDisaggregated is enabled.
    bool isCloudLastLayer = false;
    // 混合并行数据流
    int attnStreamNum = 1;
    int ffnStreamNum = 1;
    int lmheadStreamNum = 1;
    bool attnAllreduce = false;
    bool attnReduceScatter = false;
    bool attnAllGather = false;
    bool ffnAllreduce = false;
    bool ffnReduceScatter = false;
    bool ffnAllGather = false;
    bool hasAttnComm = false;
    bool hasFfnComm = false;
    bool enableExpertCumSumOutput = false;
    bool enableDenseTp = false;
    bool hasDenseTp = false;
    bool enableTopkOutput = false;
    std::string routingMethod = "deviceLimited";
    std::string processLogits = "scaling";
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::vector<int> attnLinearQuantType = {};
    std::vector<int> attnLinearTransposeType = {};
    atb::SVector<int32_t> topkGroups = {1}; // num of selected groups
    int moePackQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    // h3p
    bool enableQkvdownDp = false;
    bool enableSharedExpertDp = false;
    bool enableGatingDp = false;
    bool enableSharedExpertOverlap = false;

    bool enableLoadBalance = false;
    bool maskfree = true;

    bool enableAllToAllMC2 = false;
    HcclComm hcclComm = nullptr;
    bool enableGatherPreNorm = false;
    bool enableEPWB = false;
    uint32_t numOfRedundantExpert = 0;
    int64_t numDanglingSharedExperts = 0;

    HcclComm dispatchAndCombineHcclComm;
    std::string dispatchAndCombinecommDomain = "";

    bool enableInfNan = true;
    bool enableOutLcocTp = false;
    bool enablePreprocessLcocTp = false;
    bool enableLcocAll2All = false;
    bool mixSharedRouting = false;
    bool enableFusedMLA = false;
};

/// The index of the GATEUP linear within the mlp
const uint64_t MLP_GATEUP_LINEAR_INDEX = 0;
/// The index of the down linear within the mlp
const uint64_t MLP_DOWN_LINEAR_INDEX = 2;
/// The index of the GATEUP linear within the moe
const uint64_t MOE_GATEUP_LINEAR_INDEX = 1;
/// The index of the down linear within the moe
const uint64_t MOE_DOWN_LINEAR_INDEX = 3;

atb::Status DecoderLayer(DecoderLayerParam &param, atb::Operation **operation);

class DecoderLayer {
public:
    explicit DecoderLayer();
    ~DecoderLayer();

private:
    int32_t layerId_ = 0;
};

}  // namespace deepseekV2
}  // namespace atb_speed
#endif
