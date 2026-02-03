/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
 
#include <algorithm>
#include <map>
#include <set>
#include <cctype>
#include "safe_io.h"
#include "safe_path.h"
#include "check_utils.h"
#include "common_util.h"
#include "param_checker.h"
#include "file_utils.h"
#include "base_config_manager.h"
#include "log.h"
#include "safe_io.h"

using mindie_llm::Json;
using namespace nlohmann::literals;

namespace mindie_llm {
const uint32_t MAX_INPUT_LEN = 1024 * 1024 * 4; // 4M
const int32_t NPU_MEM_SZIE_LIMIT = -2;
const size_t PYTHON_VERSION_LEN = 4;

std::vector<std::string> g_checkEngineNameVec;
static std::vector<ParamSpec> g_modelParentParamsConstraint = {
    {"maxSeqLen", "uint32_t", true}, {"maxInputTokenLen", "uint32_t", true}, {"truncation", "bool", false},
    {"maxLoras", "uint32_t", false}, {"maxLoraRank", "uint32_t", false}};

static std::vector<ParamSpec> g_modelParamsConstraint = {
    {"cpuMemSize", "uint32_t", true},       {"modelWeightPath", "string", true}, {"modelName", "string", true},
    {"npuMemSize", "int32_t", true},        {"worldSize", "uint32_t", true},     {"backendType", "string", true},
    {"modelInstanceType", "string", false}, {"pluginParams", "string", false},   {"trustRemoteCode", "bool", true}};

static std::set<std::string> g_modelInstanceType = {"Standard", "StandardMock", "TargetModel", "AssistantModel"};

void ModelDeployConfigManager::CheckTemplateConfig(const std::string &templateType, const uint32_t modelConfigNum)
{
    if (templateType == "Standard" && modelConfigNum != 1U) {
        MINDIE_LLM_LOG_ERROR("There should be one modelParam for Standard templateType.");
        initFlag = false;
    }
}

void GetJsonModelConfig(struct ModelDeployConfig &modelConfig)
{
    std::string modelConfigPath = modelConfig.modelWeightPath + "/config.json";
    Json configJsonData;
    Result r = LoadJson(modelConfigPath, configJsonData);
    if (!r.IsOk()) {
        MINDIE_LLM_LOG_ERROR(r.message());
        modelConfig.modelWeightPath.clear();
        return;
    }
    if (configJsonData.contains("torch_dtype") && configJsonData["torch_dtype"].is_string()) {
        modelConfig.torchDtype = configJsonData["torch_dtype"];
    }
    if (configJsonData.contains("model_type") && configJsonData["model_type"].is_string()) {
        modelConfig.modelType = configJsonData["model_type"];
    }
    if (configJsonData.contains("vocab_size") &&
        ParamChecker::IsWithinRange("uint32_t", configJsonData["vocab_size"])) {
        modelConfig.vocabSize = configJsonData["vocab_size"];
    } else if (configJsonData.contains("padded_vocab_size") &&
               ParamChecker::IsWithinRange("uint32_t", configJsonData["padded_vocab_size"])) {
        modelConfig.vocabSize = configJsonData["padded_vocab_size"];
    }

    std::string modelGenerationConfigPath = modelConfig.modelWeightPath + "/generation_config.json";

    Json generationConfigJsonData;
    r = LoadJson(modelGenerationConfigPath, generationConfigJsonData);
    if (!r.IsOk()) {
        MINDIE_LLM_LOG_WARN(r.message());
        return;
    }
    if (generationConfigJsonData.contains("top_k") &&
        ParamChecker::IsWithinRange("int32_t", generationConfigJsonData["top_k"])) {
        modelConfig.maxTopK = generationConfigJsonData["top_k"];
    }
}

static void CheckIfParallelInfoIsValid(const std::map<std::string, std::string> &modelConfig)
{
    auto checkParallelValue = [&](const std::string& parallelInfo) {
        int parallelSize = 1;
        auto it = modelConfig.find(parallelInfo);
        if (it != modelConfig.end()) {
            try {
                parallelSize = std::stoi(it->second);
                MINDIE_LLM_LOG_INFO("CheckIfParallelInfoIsValid: " <<
                    parallelInfo << " is " << parallelSize);
            } catch (const std::invalid_argument& e) {
                MINDIE_LLM_LOG_ERROR("Invalid value for " << parallelInfo <<
                    ", which is " << it->second);
                throw;
            } catch (const std::out_of_range& e) {
                MINDIE_LLM_LOG_ERROR("Value for " << parallelInfo <<
                    " out of range, got " << it->second);
                throw;
            } catch (const std::exception& e) {
                MINDIE_LLM_LOG_ERROR("Invalid " << parallelInfo);
                throw;
            }
        }

        if (parallelSize <= 0) {
            MINDIE_LLM_LOG_ERROR(parallelInfo << " cannot be smaller than 1.");
            throw std::runtime_error(parallelInfo + " cannot be smaller than 1.");
        }
    };
    checkParallelValue("dp");
    checkParallelValue("sp");
    checkParallelValue("cp");
}

static uint32_t GammaUpdate(ModelDeployConfig modelParam, uint32_t speculationGamma)
{
    // 查找 plugin_params 字段
    auto it = modelParam.modelConfig.find("plugin_params");
    if (it != modelParam.modelConfig.end()) {
        std::string pluginParams = it->second;
        try {
            // 解析 JSON 字符串
            nlohmann::json pluginConfig = nlohmann::json::parse(pluginParams, CheckJsonDepthCallbackNoLogger);
            // 检查 plugin_type 字段是否为 "mtp"
            if (!pluginConfig.contains("plugin_type") ||
            std::string(pluginConfig["plugin_type"]).find("mtp") == std::string::npos) {
                return speculationGamma;
            }
            // 检查是否存在 num_speculative_tokens 字段
            if (pluginConfig.contains("num_speculative_tokens")) {
                // 根据 num_speculative_tokens 的值计算slot预留值
                uint32_t gammaTmp = pluginConfig["num_speculative_tokens"].get<uint32_t>();
                if (gammaTmp > 0) {
                    return std::max(gammaTmp, 2 * gammaTmp - 2);
                } else {
                    return pluginConfig["num_speculative_tokens"].get<uint32_t>();
                }
            }
        } catch (const nlohmann::json::parse_error &e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        }
    }
    // 如果未找到相关字段或解析失败，返回入参 speculationGamma
    return speculationGamma;
}

static bool InitLoraConfig(const Json &loraJsonData, std::vector<LoraConfig> &loraModules)
{
    LoraConfig tmpconfig;
    for (auto singleLoraJson : loraJsonData) {
        tmpconfig.loraName = GetStringParamValue(singleLoraJson, "name");
        tmpconfig.loraPath = GetStringParamValue(singleLoraJson, "path");
        tmpconfig.baseModel = GetStringParamValue(singleLoraJson, "baseModelName");
        loraModules.push_back(tmpconfig);
    }
    return true;
}

static bool InitLoraConfig(std::string modelName, std::string modelWeightPath, std::vector<LoraConfig> &loraModules)
{
    Json loraJsonData;
    if (!ParamChecker::CheckAndGetLoraJsonFile(modelWeightPath, loraJsonData)) {
        return false;
    }

    for (auto &singleLoraJson : loraJsonData.items()) {
        LoraConfig tmpconfig;
        tmpconfig.loraName = singleLoraJson.key();
        tmpconfig.loraPath = singleLoraJson.value();
        tmpconfig.baseModel = modelName;
        loraModules.push_back(tmpconfig);
    }

    return true;
}

void ModelDeployConfigManager::InitModelConfigImpl(const Json &modelJsonData, uint32_t speculationGamma,
                                                   const uint32_t maxSeqLength, bool truncation)
{
    std::vector<std::string> modelConfigList = {"modelInstanceType", "modelName",      "modelWeightPath",
                                                "worldSize",         "cpuMemSize",     "npuMemSize",
                                                "backendType",       "trustRemoteCode"};
    for (const Json &singleModelData : modelJsonData["ModelConfig"]) {
        ModelDeployConfig modelParam{};
        Json singleModelDataTmp = singleModelData;
        for (Json::iterator it = singleModelDataTmp.begin(); it != singleModelDataTmp.end(); ++it) {
            if (std::find(modelConfigList.begin(), modelConfigList.end(), it.key()) != modelConfigList.end()) {
                continue;
            }
            if (it.value().is_string()) {
                modelParam.modelConfig[it.key()] = it.value();
            } else if (it.value().is_number() || it.value().is_boolean()) {
                int32_t valueTemp = static_cast<int32_t>(it.value());
                modelParam.modelConfig[it.key()] = std::to_string(valueTemp);
            }
        }
        if (singleModelData.contains("plugin_params") && !singleModelData["plugin_params"].is_string()) {
            Json pluginParams = singleModelData["plugin_params"];
            std::string escapedPluginParams = pluginParams.dump();
            modelParam.modelConfig["plugin_params"] = escapedPluginParams;
        }
        if (singleModelData.contains("models")) {
            Json models = singleModelData["models"];
            std::string escapedModelsParams = models.dump(-1, ' ', false, Json::error_handler_t::replace);
            modelParam.modelConfig["models"] = escapedModelsParams;
        }
        CheckIfParallelInfoIsValid(modelParam.modelConfig);
        speculationGamma = GammaUpdate(modelParam, speculationGamma);
        modelParam.modelName = singleModelData["modelName"];
        modelParam.modelInstanceType =
            ParamChecker::GetStringParamValue(singleModelData, "modelInstanceType", "Standard");
        if (g_modelInstanceType.count(modelParam.modelInstanceType) == 0) {
            std::string supportedTypes {};
            for (const auto &type : g_modelInstanceType) {
                supportedTypes += (supportedTypes.empty() ? "" : ",") + type;
            }
            std::cout << "Model instance type is invalid. supported values: [" << supportedTypes << "]" << std::endl;
            initFlag = initFlag && false;
        }

        modelParam.cpuMemSize = singleModelData["cpuMemSize"];
        modelParam.trustRemoteCode = singleModelData["trustRemoteCode"];
        modelParam.maxSeqLen = maxSeqLength;
        modelParam.speculationGamma = speculationGamma;
        modelParam.modelWeightPath = singleModelData["modelWeightPath"];
        modelParam.npuMemSize = singleModelData["npuMemSize"];
        modelParam.backendType = singleModelData["backendType"];
        modelParam.maxInputTokenLen = modelJsonData["maxInputTokenLen"];
        modelParam.truncation = truncation;
        modelParam.worldSize = singleModelData["worldSize"];
        modelParam.maxLoras = modelDeployConfig_.maxLoras;
        modelParam.maxLoraRank = modelDeployConfig_.maxLoraRank;

        GetJsonModelConfig(modelParam);
        MINDIE_LLM_LOG_INFO("ModelDeployConfig::vocabSize=" << modelParam.vocabSize);
        MINDIE_LLM_LOG_INFO("ModelDeployConfig::maxTopK=" << modelParam.maxTopK);
        modelParamVec_.push_back(modelParam);
    }

    InitLoraConfigImpl(modelJsonData);
}

static void InitLoraConfigFromJson(const Json &modelJsonData, std::vector<ModelDeployConfig> &modelParamVec)
{
    std::map<std::string, std::vector<std::string>> loraBaseCorrespondence; // key: base model, value: lora names
    std::map<std::string, std::string> loraNamePaths;                       // key: lora_names, value: paths
    std::set<std::string> uniqueLoraNames;
    std::string loraName;
    std::string loraPath;
    std::string loraBaseModel;

    // get valid base model name
    for (ModelDeployConfig &singleModelParam : modelParamVec) {
        std::string modelName = singleModelParam.modelName;
        std::vector<std::string> tmpVector;
        loraBaseCorrespondence.insert(std::make_pair(modelName, tmpVector));
    }

    for (const Json &singleLoraData : modelJsonData["LoraModules"]) {
        loraName = singleLoraData.at("name");
        loraPath = singleLoraData.at("path");
        loraBaseModel = singleLoraData.at("baseModelName");
        if (loraBaseCorrespondence.find(loraBaseModel) == loraBaseCorrespondence.end()) {
            MINDIE_LLM_LOG_WARN("`baseModelName` does not exist and corresponding lora adpater is ignored. "
                      << "Please verify that `baseModelName` "
                      << "is defined in $BackendConfig.ModelDeployConfig.ModelConfig.modelName, "
                      << "which is loaded from $BackendConfig.ModelDeployConfig.LoraModules.baseModelName "
                      << "in ${MIES_INSTALL_PATH}/conf/config.json.");
        } else {
            loraBaseCorrespondence.at(loraBaseModel).push_back(loraName);
        }

        if (uniqueLoraNames.find(loraName) != uniqueLoraNames.end()) {
            MINDIE_LLM_LOG_WARN("The `name` in `LoraModules` is duplicated and `path` is set to the first one. "
                      << "Please check $BackendConfig.ModelDeployConfig.LoraModules "
                      << "in ${MIES_INSTALL_PATH}/conf/config.json.");
        } else {
            uniqueLoraNames.insert(loraName);
            loraNamePaths.insert(std::make_pair(loraName, loraPath));
        }
    }

    for (ModelDeployConfig &singleModelParam : modelParamVec) {
        std::string modelName = singleModelParam.modelName;
        std::vector<std::string> loraNames = loraBaseCorrespondence.at(modelName);
        if (loraNames.size() <= 0) {
            continue;
        }
        singleModelParam.useLora = true;
        for (auto &tmpName : loraNames) {
            loraPath = loraNamePaths.at(tmpName);
            singleModelParam.loraModules.insert(std::make_pair(tmpName, loraPath));
        }
    }
}

static void InitLoraConfigFromFile(std::vector<ModelDeployConfig> &modelParamVec)
{
    std::set<std::string> uniqueLoraNames;
    std::string loraName;
    std::string loraPath;
    std::string weightPath;
    std::string filepath;
    Json loraJson;
    for (ModelDeployConfig &singleModelParam : modelParamVec) {
        weightPath = singleModelParam.modelWeightPath;
        filepath = weightPath + "/lora_adapter.json";
        if (!FileUtils::CheckFileExists(filepath)) {
            continue;
        }
        if (!ParamChecker::GetJsonData(filepath, weightPath, loraJson)) {
            continue;
        }

        for (Json::iterator it = loraJson.begin(); it != loraJson.end(); ++it) {
            // service 初始化时已check json 是否key value全为string
            if (uniqueLoraNames.find(it.key()) != uniqueLoraNames.end()) {
                MINDIE_LLM_LOG_WARN("The `name` in `LoraModules` is duplicated and `path` is set to the first one. "
                          << "Please check lora_adapter.json under `modelWeightPath`.");
            } else {
                uniqueLoraNames.insert(it.key());
                singleModelParam.loraModules.insert(std::make_pair(it.key(), it.value()));
            }
        }
        if (singleModelParam.loraModules.size() > 0) {
            singleModelParam.useLora = true;
        }
    }
}

void ModelDeployConfigManager::InitLoraConfigImpl(const Json &modelJsonData)
{
    if (modelJsonData.contains("LoraModules")) {
        InitLoraConfigFromJson(modelJsonData, modelParamVec_); // 优先从config.json读取
    } else {
        InitLoraConfigFromFile(modelParamVec_); // 旧的lora_adapter.json 方式
    }
}

void ModelDeployConfigManager::InitModelConfig(const Json &modelJsonData, uint32_t speculationGamma,
                                               const uint32_t maxSeqLength, const bool truncation)
{
    ModelDeployConfigManager::InitModelConfigImpl(modelJsonData, speculationGamma, maxSeqLength, truncation);

    if (modelJsonData.contains("LoraModules")) {
        if (!InitLoraConfig(modelJsonData["LoraModules"], loraModules_)) {
            initFlag &= false;
        }
    } else {
        std::map<std::string, std::string> modelNameWeights;
        for (const Json &singleModelData : modelJsonData["ModelConfig"]) {
            std::string modelName;
            std::string modelWeightPath;
            modelName = singleModelData["modelName"];
            modelWeightPath = singleModelData["modelWeightPath"];
            modelNameWeights.insert(std::make_pair(modelName, modelWeightPath));
        }
        for (auto it = modelNameWeights.begin(); it != modelNameWeights.end(); it++) {
            if (!InitLoraConfig(it->first, it->second, loraModules_)) {
                initFlag &= false;
            }
        }
    }
}

bool ModelDeployConfigManager::InitFromJson()
{
    Json backendJsonData;
    if (!CheckSystemConfig(jsonPath_, backendJsonData, "BackendConfig")) {
        initFlag = false;
        MINDIE_LLM_LOG_ERROR("Failed to parse the json data of backendJsonData.");
        return false;
    }
    Json modelJsonData = backendJsonData["ModelDeployConfig"];
    if (!ParamChecker::CheckJsonParamType(modelJsonData, g_modelParentParamsConstraint)) {
        return false;
    }
    if (!modelJsonData["ModelConfig"].is_array()) {
        MINDIE_LLM_LOG_ERROR("The type of ModelConfig should be array.");
        return false;
    }

    for (Json &singleModelData : modelJsonData["ModelConfig"]) {
        if (!ParamChecker::CheckJsonParamType(singleModelData, g_modelParamsConstraint)) {
            return false;
        }
    }
    modelDeployConfig_.maxSeqLen = backendJsonData["ModelDeployConfig"]["maxSeqLen"];
    modelDeployConfig_.truncation =
        ParamChecker::GetBoolParamValue(backendJsonData["ModelDeployConfig"], "truncation", false);

    // 校验并读取maxLoras和maxLoraRank,负数则取0(负数为异常值，一般不会为负)
    const auto& modelDeployConfig = backendJsonData["ModelDeployConfig"];
    if (modelDeployConfig.contains("maxLoras") && modelDeployConfig["maxLoras"].is_number()) {
        MINDIE_LLM_LOG_INFO("maxLoras :" << modelDeployConfig["maxLoras"].get<int32_t>());
        modelDeployConfig_.maxLoras = static_cast<uint32_t>(std::max(modelDeployConfig["maxLoras"].get<int32_t>(), 0));
    } else {
        modelDeployConfig_.maxLoras = 0;
        MINDIE_LLM_LOG_WARN("maxLoras not found or invalid in ModelDeployConfig, using default 0");
    }
    if (modelDeployConfig.contains("maxLoraRank") && modelDeployConfig["maxLoraRank"].is_number()) {
        MINDIE_LLM_LOG_INFO("maxLoraRank:" << modelDeployConfig["maxLoraRank"].get<int32_t>());
        modelDeployConfig_.maxLoraRank = static_cast<uint32_t>(
          std::max(modelDeployConfig["maxLoraRank"].get<int32_t>(), 0));
    } else {
        modelDeployConfig_.maxLoraRank = 0;
        MINDIE_LLM_LOG_WARN("maxLoraRank not found or invalid in ModelDeployConfig, using default 0");
    }

    uint32_t speculationGamma = ParamChecker::GetIntegerParamDefaultValue(modelJsonData, "speculationGamma", 0);
    if (!backendJsonData["ModelDeployConfig"]["ModelConfig"].is_array()) {
        MINDIE_LLM_LOG_ERROR("The type of ModelConfig or npuDeviceIds should be array.");
        initFlag = false;
    }
    CheckTemplateConfig(backendJsonData["ScheduleConfig"]["templateType"],
                        backendJsonData["ModelDeployConfig"]["ModelConfig"].size());
    InitModelConfig(backendJsonData["ModelDeployConfig"], speculationGamma,
        backendJsonData["ModelDeployConfig"]["maxSeqLen"], modelDeployConfig_.truncation);
    return initFlag;
}

std::vector<ModelDeployConfig> &ModelDeployConfigManager::GetParam() { return modelParamVec_; }

std::vector<LoraConfig> &ModelDeployConfigManager::GetLoraConfig() { return loraModules_; }

static bool CheckModelNameLength(std::string &name, std::string paramType)
{
    std::regex regexPattern("^[a-zA-Z0-9_.-]{1,256}$");
    if (!std::regex_match(name, regexPattern) || !isalnum(name[0]) || !isalnum(name[name.size() - 1])) {
        MINDIE_LLM_LOG_ERROR("The value of " << paramType << " must meet the following rules: "
                  << "The string length is [1, 256] and consists of a match of the type "
                  << "[a-zA-Z0-9_.-]. The first and last characters must be characters or digits.");
        return false;
    }
    return true;
}

bool ModelDeployConfigManager::CheckParam()
{
    std::set<std::string> modelNames;
    for (auto &modelParam : modelParamVec_) {
        if (modelParam.modelName.empty()) {
            MINDIE_LLM_LOG_ERROR("Error: modelName is not set in modelParam");
            initFlag = false;
        }
        std::string modelName = modelParam.modelName;
        initFlag = CheckModelNameLength(modelName, "modelName");
        if (modelParam.maxSeqLen == 0) {
            MINDIE_LLM_LOG_ERROR("The value of modelParam.maxSeqLen can not be 0.");
            initFlag = false;
        }
        if (modelParam.maxInputTokenLen == 0U || modelParam.maxInputTokenLen > MAX_INPUT_LEN) {
            MINDIE_LLM_LOG_ERROR("The value of modelParam.maxInputTokenLen should be in [1, 4194304].");
            initFlag = false;
        }
        if (modelParam.worldSize == 0) {
            MINDIE_LLM_LOG_ERROR("The value of modelParam.worldSize can not be 0.");
            initFlag = false;
        }
        if (modelParam.backendType != "atb" && modelParam.backendType != "ms" && modelParam.backendType != "torch") {
            MINDIE_LLM_LOG_ERROR("Backend type only supports atb, ms, torch.");
            initFlag = false;
        }
        if (modelParam.npuMemSize == 0 || modelParam.npuMemSize <= NPU_MEM_SZIE_LIMIT) {
            MINDIE_LLM_LOG_ERROR("The value of 'npuMemSize' in ModelDeployConfig in config.json is " <<
                modelParam.npuMemSize << " it should be within (0, 4294967295] or -1" <<
                "please set 'npuMemSize' in config.json within (0, 4294967295] or -1",
                LLM_MANAGER_CONFIG_FAILED);
            initFlag = false;
        }
        SafePath modelWeightPath(modelParam.modelWeightPath, PathType::DIR, "r", PERM_750);
        Result r = modelWeightPath.Check(modelParam.modelWeightPath);
        if (!r.IsOk()) {
            MINDIE_LLM_LOG_ERROR(r.message());
            initFlag = false;
        }
        if (initFlag != false) {
            modelNames.insert(modelName);
        }
    }

    for (auto &loraParam : loraModules_) {
        std::string loraName = loraParam.loraName;
        initFlag = CheckModelNameLength(loraName, "loraName");
        SafePath modelWeightPath(loraParam.loraPath, PathType::DIR, "r", PERM_750);
        Result r = modelWeightPath.Check(loraParam.loraPath);
        if (!r.IsOk()) {
            MINDIE_LLM_LOG_ERROR(r.message());
            initFlag = false;
        }
        if (modelNames.find(loraParam.baseModel) == modelNames.end()) {
            MINDIE_LLM_LOG_ERROR("Lora base model not found. Please check. Lora name is " << loraName);
            initFlag = false;
        }
    }
    return initFlag;
}

void ModelDeployConfigManager::SetMaxPositionEmbeddings(uint32_t maxPositionEmbeddings)
{
    modelParamVec_[0].maxPositionEmbeddings = maxPositionEmbeddings;
}
} // namespace mindie_llm
