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
#include "models/base/param/model_param.h"

namespace atb_speed {
namespace base {

nlohmann::json StringToJson(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    return paramJson;
}

void ModelParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = StringToJson(param);
    ParseParam(paramJson);
    CheckParam();
    PrintParam();
}

void ModelParam::PrintParam()
{
    Param::PrintParam();
    ATB_SPEED_LOG_DEBUG("Model Param: isEmbeddingParallel: " << this->isEmbeddingParallel
                  << ", isLmHeadParallel: " << this->isLmHeadParallel
                  << ", lmHeadTransposeType: " << this->lmHeadTransposeType
                  << ", numHiddenLayers: " << this->numHiddenLayers
                  << ", rank: " << this->rank
                  << ", worldSize: " << this->worldSize
                  << ", backend: " << this->backend
                  << ", rankTableFile: " << this->rankTableFile
                  << ", enableDap: " << this->enableDap
                  << ", enableFlashComm: " << this->enableFlashComm);
}

void ModelParam::ParseParam(const nlohmann::json &paramJson)
{
    this->isUnpadInputs = FetchJsonParam<bool>(paramJson, "isUnpadInputs");
    this->isPrefill = FetchJsonParam<bool>(paramJson, "isPrefill");
    this->isBF16 = FetchJsonParam<bool>(paramJson, "isBF16");
    this->normEps = FetchJsonParam<float>(paramJson, "normEps");
    this->normType = FetchJsonParam<NormType>(paramJson, "normType");
    if (paramJson.contains("isEdgeHardware")) {
        this->isEdgeHardware = FetchJsonParam<bool>(paramJson, "isEdgeHardware");
    }
    this->numHiddenLayers = CheckNumHiddenLayersValid(FetchJsonParam<uint32_t>(paramJson, "numHiddenLayers"));
    if (paramJson.contains("skipWordEmbedding")) {
        this->skipWordEmbedding = FetchJsonParam<bool>(paramJson, "skipWordEmbedding");
    }
    if (paramJson.contains("positionEmbeddingType")) {
        this->positionEmbeddingType = FetchJsonParam<PositionEmbeddingType>(paramJson, "positionEmbeddingType");
    }
    if (paramJson.contains("enablePrefixCache")) {
        this->enablePrefixCache = FetchJsonParam<bool>(paramJson, "enablePrefixCache");
    }
    if (paramJson.contains("weightQuantType")) {
        this->weightQuantType = FetchJsonParam<std::string>(paramJson, "weightQuantType");
    }
    if (paramJson.contains("enableGreedySearchOpt")) {
        this->enableGreedyPostProcessing = paramJson["enableGreedySearchOpt"].get<bool>();
    }

    ParseLayerwiseDisaggregatedParm(paramJson);
    if (paramJson.contains("enableDap")) { this->enableDap =  FetchJsonParam<bool>(paramJson, "enableDap"); }
    if (paramJson.contains("enableCVOverlap")) {
        this->enableCVOverlap = FetchJsonParam<bool>(paramJson, "enableCVOverlap");
    }
    if (paramJson.contains("enableFlashComm")) {
        this->enableFlashComm = paramJson["enableFlashComm"].get<bool>();
    }
    if (paramJson.contains("memPoolType")) {
        this->memPoolType = FetchJsonParam<MemPoolType>(paramJson, "memPoolType");
    }
    if (paramJson.contains("pipeKey")) {
        this->memPoolEventPipeKey = FetchJsonParam<std::string>(paramJson, "pipeKey");
    }
    ParseNormParam(paramJson);
    ParseAttentionParam(paramJson);
    ParseMlpParam(paramJson);
    ParseMatmulParam(paramJson);
    ParseTensorParallelParam(paramJson);
    ParseParallelismParam(paramJson);
}

void ModelParam::ParseLayerwiseDisaggregatedParm(const nlohmann::json &paramJson)
{
    if (paramJson.contains("layerwiseDisaggregated")) {
        this->layerwiseDisaggregated = FetchJsonParam<bool>(paramJson, "layerwiseDisaggregated");
    }
    if (this->layerwiseDisaggregated) {
        if (paramJson.contains("layerwiseMode")) {
            this->layerwiseMode = FetchJsonParam<int32_t>(paramJson, "layerwiseMode");
        }
        if (paramJson.contains("hiddenSize")) {
            this->hiddenSize = FetchJsonParam<int32_t>(paramJson, "hiddenSize");
        }
        if (paramJson.contains("startLayerId")) {
            this->startLayerId = FetchJsonParam<int32_t>(paramJson, "startLayerId");
        }
        if (paramJson.contains("endLayerId")) {
            this->endLayerId = FetchJsonParam<int32_t>(paramJson, "endLayerId");
        }
        if (paramJson.contains("reuseEmbedTable")) {
            this->reuseEmbedTable = FetchJsonParam<bool>(paramJson, "reuseEmbedTable");
        }
        if (paramJson.contains("outputEmbedTable")) {
            this->outputEmbedTable = FetchJsonParam<bool>(paramJson, "outputEmbedTable");
        }
        if (this->layerwiseMode == LWD_CLOUD_MIDDLE || this->layerwiseMode == LWD_EDGE_LAST) {
            this->skipWordEmbedding = true;
        }
        this->isInternalLayer = this->layerwiseDisaggregated
            && (this->layerwiseMode == LWD_EDGE_FIRST || this->layerwiseMode == LWD_CLOUD_MIDDLE);
    }
}

void ModelParam::ParseNormParam(const nlohmann::json &paramJson)
{
    if (paramJson.contains("enableIntraLayerAddNorm")) {
        this->enableIntraLayerAddNorm = FetchJsonParam<bool>(paramJson, "enableIntraLayerAddNorm");
    }
    if (paramJson.contains("enableInterLayerAddNorm")) {
        this->enableInterLayerAddNorm = FetchJsonParam<bool>(paramJson, "enableInterLayerAddNorm");
    }
    if (paramJson.contains("isAntiOutlier")) {
        for (auto item : paramJson["isAntiOutlier"]) {
            this->isAntiOutlier.push_back(FetchJsonParam<std::vector<bool>>(item, "isAntiOutlier", true));
        }
        CheckLinearParamsSufficient(this->isAntiOutlier, this->numHiddenLayers, 2);  // 2: two norm in one layer
    }
}

void ModelParam::ParseAttentionParam(const nlohmann::json &paramJson)
{
    this->isFA = FetchJsonParam<bool>(paramJson, "isFA");
    this->numAttentionHeadsPerRank = FetchJsonParam<uint32_t>(paramJson, "numAttentionHeadsPerRank");
    this->hiddenSizePerAttentionHead = FetchJsonParam<uint32_t>(paramJson, "hiddenSizePerAttentionHead");
    this->numKeyValueHeadsPerRank = FetchJsonParam<uint32_t>(paramJson, "numKeyValueHeadsPerRank");
    if (paramJson.contains("enableKvQuant")) {
        this->enableKvQuant = FetchJsonParam<bool>(paramJson, "enableKvQuant");
    }
    if (paramJson.contains("enableFA3")) { this->enableFA3 = FetchJsonParam<bool>(paramJson, "enableFA3"); }
    if (paramJson.contains("attnBackend")) {
        this->attnBackend = FetchJsonParam<atb_speed::common::OpBackend>(paramJson, "attnBackend");
    }
    if (paramJson.contains("enableSpeculate")) {
        this->enableSpeculate = FetchJsonParam<bool>(paramJson, "enableSpeculate");
    }
    if (paramJson.contains("enableSplitFuse")) {
        this->enableSplitFuse = FetchJsonParam<bool>(paramJson, "enableSplitFuse");
    }
    if (paramJson.contains("enableCompressHead")) {
        this->enableCompressHead = FetchJsonParam<bool>(paramJson, "enableCompressHead");
    }
    if (paramJson.contains("enableOmniAttention")) {
        this->enableOmniAttention = FetchJsonParam<bool>(paramJson, "enableOmniAttention");
        if (this->enableOmniAttention) {
            for (auto item : paramJson["pattern_mask"]) {
                this->patternMask.push_back(item.get<bool>());
            }
        }
    }
    if (paramJson.contains("enableRopeQuantKvcache")) {
        this->enableRopeQuantKvcache = paramJson["enableRopeQuantKvcache"].get<bool>();
    }
    if (paramJson.contains("useQKNorm")) {
        this->useQKNorm = paramJson["useQKNorm"].get<bool>();
    }
    if (paramJson.contains("enableModelConfuscation")) {
        this->enableModelConfuscation = paramJson["enableModelConfuscation"].get<bool>();
    }
    if (paramJson.contains("modelConfuscationFd")) {
        this->modelConfuscationFd = paramJson["modelConfuscationFd"].get<int32_t>();
    }
    if (paramJson.contains("ropeBackend")) {
        this->ropeBackend = FetchJsonParam<atb_speed::common::OpBackend>(paramJson, "ropeBackend");
    }
}

void ModelParam::ParseMlpParam(const nlohmann::json &paramJson)
{
    if (paramJson.contains("enableSwiGLU")) {
        this->enableSwiGLU = FetchJsonParam<bool>(paramJson, "enableSwiGLU");
    }
    if (paramJson.contains("enableSwigluQuant")) {
        this->enableSwigluQuant = FetchJsonParam<bool>(paramJson, "enableSwigluQuant");
    }
}

void ModelParam::ParseMatmulParam(const nlohmann::json &paramJson)
{
    this->lmHeadTransposeType = FetchJsonParam<int>(paramJson, "lmHeadTransposeType");
    if (paramJson.contains("packQuantType")) {
        for (auto item : paramJson["packQuantType"]) {
            this->packQuantType.push_back(FetchJsonParam<std::vector<int>>(item, "packQuantType", true));
        }
        CheckPackQuantParamsSufficient(this->packQuantType, this->numHiddenLayers);
    }
    if (paramJson.contains("linearQuantType")) {
        for (auto item : paramJson["linearQuantType"]) {
            this->linearQuantType.push_back(FetchJsonParam<std::vector<int>>(item, "linearQuantType", true));
        }
        CheckLinearPackParamsSufficient(this->linearQuantType, this->numHiddenLayers);
    }
    if (paramJson.contains("linearTransposeType")) {
        for (auto item : paramJson["linearTransposeType"]) {
            this->linearTransposeType.push_back(FetchJsonParam<std::vector<int>>(item, "linearTransposeType", true));
        }
        CheckLinearPackParamsSufficient(this->linearTransposeType, this->numHiddenLayers);
    }
    if (paramJson.contains("linearHasBias")) {
        for (auto item : paramJson["linearHasBias"]) {
            this->linearHasBias.push_back(FetchJsonParam<std::vector<bool>>(item, "linearHasBias", true));
        }
        CheckLinearHasBiasSufficient(this->linearHasBias, this->numHiddenLayers);
    }
    if (paramJson.contains("linearDescs")) {
        for (auto item : paramJson["linearDescs"]) {
            this->linearDescs.push_back(FetchJsonParam<std::vector<int>>(item, "linearDescs", true));
        }
        CheckLinearPackParamsSufficient(this->linearDescs, this->numHiddenLayers);
    }
    if (paramJson.contains("enableReduceQuant")) {
        this->enableReduceQuant = FetchJsonParam<bool>(paramJson, "enableReduceQuant");
    }
    if (paramJson.contains("enableLora")) {
        this->enableLora = FetchJsonParam<bool>(paramJson, "enableLora");
    }
    if (paramJson.contains("preFetchWeightSize")) {
        this->preFetchWeightSize = FetchJsonParam<size_t>(paramJson, "preFetchWeightSize");
    }
    if (paramJson.contains("loraEnableGMM")) {
        this->loraEnableGMM = FetchJsonParam<bool>(paramJson, "loraEnableGMM");
    }
    if (paramJson.contains("quantGroupSize")) {
        this->quantGroupSize = FetchJsonParam<uint32_t>(paramJson, "quantGroupSize");
    }
    if (paramJson.contains("matmulBackend")) {
        this->matmulBackend = FetchJsonParam<atb_speed::common::OpBackend>(paramJson, "matmulBackend");
    }
}

void ModelParam::ParseTensorParallelParam(const nlohmann::json &paramJson)
{
    if (paramJson.contains("isEmbeddingParallel")) {
        this->isEmbeddingParallel = FetchJsonParam<bool>(paramJson, "isEmbeddingParallel");
    }
    if (paramJson.contains("isLmHeadParallel")) {
        this->isLmHeadParallel = FetchJsonParam<bool>(paramJson, "isLmHeadParallel");
    }
    this->backend = FetchJsonParam<std::string>(paramJson, "backend");
    if (paramJson.contains("mapping")) {
        this->mapping.ParseParam(paramJson["mapping"]);
        // prepare communication group
        this->mapping.InitGlobalCommDomain(this->backend);
    }
    this->rank = FetchJsonParam<int>(paramJson, "rank");
    this->worldSize = FetchJsonParam<int>(paramJson, "worldSize");
    this->worldSize = CheckPositive(this->worldSize);
    if (paramJson.contains("rankTableFile")) {
        this->rankTableFile = FetchJsonParam<std::string>(paramJson, "rankTableFile");
    }
    if (paramJson.contains("tpRankTableFile")) {
        tpRankTableFile = paramJson["tpRankTableFile"].get<std::string>();
    }
    if (paramJson.contains("hasPp")) { this->hasPp = paramJson["hasPp"].get<bool>(); }
    if (paramJson.contains("ppGroupSize")) { this->ppGroupSize = paramJson["ppGroupSize"].get<int>(); }
    if (paramJson.contains("firstPpRank")) { this->firstPpRank = paramJson["firstPpRank"].get<bool>(); }
    if (paramJson.contains("lastPpRank")) { this->lastPpRank = paramJson["lastPpRank"].get<bool>(); }
    if (paramJson.contains("prevPpRank")) { this->prevPpRank = paramJson["prevPpRank"].get<int>(); }
    if (paramJson.contains("nextPpRank")) { this->nextPpRank = paramJson["nextPpRank"].get<int>(); }
    if (paramJson.contains("tpRank")) { this->tpRank = paramJson["tpRank"].get<int>(); }
    if (paramJson.contains("tpWorldSize")) { this->tpWorldSize = paramJson["tpWorldSize"].get<int>(); }
    if (paramJson.contains("tpDomain")) { this->tpDomain = paramJson["tpDomain"].get<std::string>(); }
}


void ModelParam::ParseParallelismParam(const nlohmann::json &paramJson)
{
    if (paramJson.contains("hasAttnTp")) {
        this->hasAttnTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnTp");
    }
    if (paramJson.contains("attnTpRank")) {
        this->attnTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnTpRank");
    }
    if (paramJson.contains("attnTpSize")) {
        this->attnTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnTpSize"));
    }
    if (paramJson.contains("attnTpDomain")) {
        this->attnTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnTpDomain");
    }
    if (paramJson.contains("hasAttnDp")) {
        this->hasAttnDp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasAttnDp");
    }
    if (paramJson.contains("attnDpRank")) {
        this->attnDpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "attnDpRank");
    }
    if (paramJson.contains("attnDpSize")) {
        this->attnDpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "attnDpSize"));
    }
    if (paramJson.contains("attnDpDomain")) {
        this->attnDpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "attnDpDomain");
    }
    if (paramJson.contains("hasMlpTp")) {
        this->hasMlpTp = atb_speed::base::FetchJsonParam<bool>(paramJson, "hasMlpTp");
    }
    if (paramJson.contains("mlpTpRank")) {
        this->mlpTpRank = atb_speed::base::FetchJsonParam<int>(paramJson, "mlpTpRank");
    }
    if (paramJson.contains("mlpTpSize")) {
        this->mlpTpSize = CheckPositive(atb_speed::base::FetchJsonParam<int>(paramJson, "mlpTpSize"));
    }
    if (paramJson.contains("mlpTpDomain")) {
        this->mlpTpDomain = atb_speed::base::FetchJsonParam<std::string>(paramJson, "mlpTpDomain");
    }
    if (paramJson.contains("enableMC2")) {
        this->enableMC2 = paramJson["enableMC2"].get<bool>();
    }
    if (paramJson.contains("enableLcoc")) {
        this->enableLcoc = FetchJsonParam<bool>(paramJson, "enableLcoc");
    }
}

void ModelParam::CheckParam()
{
    if (this->hasPp && this->tpRank >= this->tpWorldSize) {
        throw std::runtime_error("tpWorldSize must be greater than tpRank, please check.");
    }
    if (this->rank >= this->worldSize) {
        throw std::runtime_error("worldSize must be greater than rank, please check.");
    }
    if (this->positionEmbeddingType != ROPE && this->positionEmbeddingType != ALIBI && \
        this->positionEmbeddingType != ABSOLUTE) {
        throw std::runtime_error("positionEmbeddingType is an enumeration variable with possible values: ROPE = 0, "
            "ALIBI = 1 or ABSOLUTE = 2, please check.");
    }
    if (this->normType != RMS_NORM && this->normType != LAYER_NORM) {
        throw std::runtime_error("normType is an enumeration variable with possible values: RMS_NORM = 0 or "
            "LAYER_NORM = 1, please check.");
    }
    if (this->attnBackend != atb_speed::common::ATB && this->attnBackend != atb_speed::common::ACLNN) {
        throw std::runtime_error("attnBackend is an enumeration variable with possible values: ACLNN = 0 or "
        "ATB = 1, please check.");
    }
    if (this->lmHeadTransposeType != atb_speed::common::TRANSPOSE_INVALID && this->lmHeadTransposeType != \
        atb_speed::common::NOT_TRANSPOSE && this->lmHeadTransposeType != atb_speed::common::TRANSPOSE) {
        throw std::runtime_error("lmHeadTransposeType is an enumeration variable with possible values: "
        "TRANSPOSE_INVALID = -1, NOT_TRANSPOSE = 0 or TRANSPOSE = 1, please check.");
    }
    auto packType = atb_speed::common::ConvertQuantTypeToPackType(this->weightQuantType);
    if (packType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED && !this->weightQuantType.empty()) {
        throw std::runtime_error(
            "weightQuantType should be float, w8a8, w8a8s, w8a8sc, w8a8_dynamic, w8a16, w4a16 or an empty string.");
    }
    // hasAttnDp 与addNorm不兼容
    if ((this->enableIntraLayerAddNorm || this->enableInterLayerAddNorm) && this->hasAttnDp) {
        throw std::runtime_error("'enableIntraLayerAddNorm or enableInterLayerAddNorm' and "
            "'hasAttnDp' are incompatible, do not enable them at the same time, please check.");
    }
    // enableDap 与 OmniAttention/enableCompressHead 不兼容
    if (this->enableDap && (this->enableOmniAttention || this->enableCompressHead)) {
        throw std::runtime_error("'DAP' and 'enableOmniAttention/enableCompressHead' are incompatible, "
            "do not enable them at the same time, please check.");
    }
    if (!this->layerwiseDisaggregated && (this->reuseEmbedTable || this->outputEmbedTable)) {
        throw std::runtime_error("'reuseEmbedTable/outputEmbedTable' can only be set to True "
            "when layerwiseDisaggregated is enabled.");
    }
    if (this->layerwiseDisaggregated && (this->reuseEmbedTable && this->outputEmbedTable)) {
        throw std::runtime_error("'reuseEmbedTable' and 'outputEmbedTable' are incompatible,"
            " do not enable them at the same time, please check.");
    }
}
} // namespace base
} // namespace atb_speed