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

#include "common_util.h"
#include "param_checker.h"
#include "base_config_manager.h"

using Json = nlohmann::json;
using namespace nlohmann::literals;

namespace mindie_llm {

constexpr int MAX_FILE_LIST_SIZE = 3;
static std::vector<ParamSpec> g_backendConfigConstraint = {
    {"backendName", "string", true},
    {"tokenizerProcessNumber", "uint32_t", true},
    {"multiNodesInferEnabled", "bool", false},
    {"multiNodesInferPort", "int32_t", false},
    {"interNodeTLSEnabled", "bool", false},
    {"interNodeTlsCaPath", "string", false},
    {"interNodeTlsCert", "string", false},
    {"interNodeTlsPk", "string", false},
    {"modelInstanceNumber", "uint32_t", true},
    {"npuDeviceIds", "array", true},
    {"ScheduleConfig", "object", true},
    {"interNodeTlsCaFiles", "array", false},
    {"interNodeTlsCrlFiles", "array", false},
    {"interNodeTlsCrlPath", "string", false},
    {"kvPoolConfig", "object", false},
    {"layerwiseDisaggregated", "object", false},
};

void BackendConfigManager::InitKvPoolConfigFromJson(Json &backendConfigData)
{
    std::string backend{};
    std::string configPath{};
    if (backendConfigData.contains("kvPoolConfig")) {
        Json& kvPoolConfig = backendConfigData["kvPoolConfig"];
        if (kvPoolConfig.contains("backend")) {
            backend = kvPoolConfig["backend"];
        }
        if (kvPoolConfig.contains("configPath")) {
            configPath = kvPoolConfig["configPath"];
        }
    }
    backendConfig_.kvPoolConfig.backend = backend;
    backendConfig_.kvPoolConfig.configPath = configPath;
}

bool BackendConfigManager::InitTlsConfigFromJson(Json &backendConfigData)
{
    backendConfig_.interNodeTlsCert = backendConfigData["interNodeTlsCert"];
    backendConfig_.interNodeTlsPk = backendConfigData["interNodeTlsPk"];
    backendConfig_.interNodeTlsCaPath = backendConfigData["interNodeTlsCaPath"];
    backendConfig_.interNodeTlsCrlPath = backendConfigData["interNodeTlsCrlPath"];

    auto ret = ParamChecker::CheckJsonArray(backendConfigData["interNodeTlsCaFiles"], "string", "");
    if (!ret) {
        std::cout << "interNodeTlsCaFiles init error" << std::endl;
        return false;
    } else {
        for (auto &caFile : backendConfigData["interNodeTlsCaFiles"]) {
            backendConfig_.interNodeTlsCaFiles += static_cast<std::string>(caFile) + ",";
            backendConfig_.interNodeTlsCaFilesVec.emplace_back(caFile);
        }
    }
    if (backendConfig_.interNodeTlsCaFilesVec.size() > MAX_FILE_LIST_SIZE ||
        backendConfig_.interNodeTlsCaFilesVec.empty()) {
        std::cout << "interNodeTlsCaFiles size is invalid" << std::endl;
        return false;
    }

    ret = ParamChecker::CheckJsonArray(backendConfigData["interNodeTlsCrlFiles"], "string", "");
    if (!ret) {
        std::cout << "interNodeTlsCrlFiles init error" << std::endl;
        return false;
    } else {
        for (auto &crlFile : backendConfigData["interNodeTlsCrlFiles"]) {
            backendConfig_.interNodeTlsCrlFiles += static_cast<std::string>(crlFile) + ",";
            backendConfig_.interNodeTlsCrlFilesVec.emplace_back(crlFile);
        }
    }

    if (backendConfig_.interNodeTlsCrlFilesVec.size() > MAX_FILE_LIST_SIZE) {
        std::cout << "interNodeTlsCrlFiles size is invalid" << std::endl;
        return false;
    }
    return true;
}

bool BackendConfigManager::CheckInterTlsParam()
{
    std::string homePath{};
    if (!GetHomePath(homePath).IsOk()) {
        std::cout << "Failed to get home path" << std::endl;
        return false;
    }
    homePath += "/";
    bool checkRes = true;
    // check cert
    std::string tlsCertPath = homePath + backendConfig_.interNodeTlsCert;
    CHECK_CONFIG_VALIDATION(checkRes,
                            ParamChecker::CheckPath(tlsCertPath, homePath, "backendConfig_.interNodeTlsCert"));
    std::string tlsPkPath = homePath + backendConfig_.interNodeTlsPk;
    CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckPath(tlsPkPath, homePath, "backendConfig_.interNodeTlsPk"));
    // check ca
    std::string interNodeTlsCaPath = homePath + backendConfig_.interNodeTlsCaPath;
    CHECK_CONFIG_VALIDATION(
        checkRes, ParamChecker::CheckPath(interNodeTlsCaPath, homePath, "backendConfig_.interNodeTlsCaPath", false));
    for (const std::string &interNodeCaFile : backendConfig_.interNodeTlsCaFilesVec) {
        std::string interNodetlsCaFilePath = interNodeTlsCaPath + interNodeCaFile;
        CHECK_CONFIG_VALIDATION(
            checkRes, ParamChecker::CheckPath(interNodetlsCaFilePath, homePath, "backendConfig_.interNodeTlsCaFiles"));
    }
    // check crl, crl can be empty
    std::string interNodeTlsCrlPath = homePath + backendConfig_.interNodeTlsCrlPath;
    if (!backendConfig_.interNodeTlsCrlPath.empty()) {
        CHECK_CONFIG_VALIDATION(
            checkRes, ParamChecker::CheckPath(interNodeTlsCrlPath, homePath, "backendConfig_.interNodeTlsCrlPath"));
        for (const std::string &interNodeCrlFile : backendConfig_.interNodeTlsCrlFilesVec) {
            std::string interNodetlsCrlFilePath = interNodeTlsCrlPath + interNodeCrlFile;
            CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckPath(interNodetlsCrlFilePath, homePath,
                                                                      "backendConfig_.interNodeTlsCrlFiles"));
        }
    }
    return checkRes;
}

void BackendConfigManager::InitLwdConfigFromJson(Json &backendConfigData)
{
    if (backendConfigData.contains("layerwiseDisaggregated")) {
        Json& lwdConfig = backendConfigData["layerwiseDisaggregated"];
        if (lwdConfig.contains("layerwiseDisaggregatedMultiNodesInferEnabled")) {
            backendConfig_.lwdMultiNodesEnable = lwdConfig["layerwiseDisaggregatedMultiNodesInferEnabled"];
            backendConfig_.lwdMultiNodesCtrlPort = lwdConfig["layerwiseDisaggregatedMultiNodesCtrlPort"];
        }
    }
}

bool BackendConfigManager::InitFromJson()
{
    Json backendConfigData;
    if (!CheckSystemConfig(jsonPath_, backendConfigData, "BackendConfig")) {
        return false;
    }
    if (!ParamChecker::CheckJsonParamType(backendConfigData, g_backendConfigConstraint)) {
        return false;
    }
    size_t npuSetNum = backendConfigData["npuDeviceIds"].size();
    for (size_t i = 0; i < npuSetNum; i++) {
        if (!ParamChecker::CheckJsonArray(backendConfigData["npuDeviceIds"][i], "integer", "size_t")) {
            return false;
        }
    }
    if (npuSetNum != backendConfigData["modelInstanceNumber"]) {
        std::cout << "The size of npuDeviceIds does not equal to modelInstanceNumber" << std::endl;
        return false;
    }
    auto singleConfig = backendConfigData["ModelDeployConfig"]["ModelConfig"][0];
    for (auto npuDeviceId : backendConfigData["npuDeviceIds"]) {
        if (npuDeviceId.size() != singleConfig["worldSize"]) {
            std::cout << "The size of npuDeviceIds (subset) does not equal to worldSize" << std::endl;
            return false;
        }
    }
    backendConfig_.worldSize = singleConfig["worldSize"];
    backendConfig_.backendName = backendConfigData["backendName"];
    backendConfig_.modelInstanceNumber = backendConfigData["modelInstanceNumber"];
    for (size_t i = 0; i < backendConfig_.modelInstanceNumber; i++) {
        std::set<size_t> emptyset{};
        for (auto num : backendConfigData["npuDeviceIds"][i]) {
            emptyset.insert(static_cast<size_t>(num));
        }
        backendConfig_.npuDeviceIds.push_back(emptyset);
    }
    backendConfig_.tokenizerProcessNumber = backendConfigData["tokenizerProcessNumber"];
    backendConfig_.multiNodesInferEnabled = backendConfigData["multiNodesInferEnabled"];
    if (backendConfigData.contains("multiNodesInferPort")) {
        backendConfig_.multiNodesInferPort = backendConfigData["multiNodesInferPort"];
    }
    backendConfig_.interNodeTLSEnabled = backendConfigData["interNodeTLSEnabled"];
    InitKvPoolConfigFromJson(backendConfigData);
    if (backendConfig_.interNodeTLSEnabled) {
        try {
            if (!InitTlsConfigFromJson(backendConfigData)) {
                std::cout << "Failed to init tls cfg" << std::endl;
                return false;
            }
        } catch (const nlohmann::json::type_error &e) {
            std::cout << "Failed to init tls cfg. [BackendConfigManager::InitFromJson] " << e.what() << std::endl;
            return false;
        }
    }

    InitLwdConfigFromJson(backendConfigData);

    return true;
}

bool BackendConfigManager::CheckParam()
{
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckEngineName(backendConfig_.backendName));
    CHECK_CONFIG_VALIDATION(initFlag,
                            ParamChecker::CheckMaxMinValue<uint32_t>(backendConfig_.modelInstanceNumber, 10U, 1U,
                                                                     "backendConfig.modelInstanceNumber"));
    CHECK_CONFIG_VALIDATION(initFlag,
                            ParamChecker::CheckMaxMinValue<uint32_t>(backendConfig_.tokenizerProcessNumber, 32U, 1U,
                                                                     "backendConfig.tokenizerProcessNumber"));
    CHECK_CONFIG_VALIDATION(initFlag,
                            ParamChecker::CheckMaxMinValue<int32_t>(backendConfig_.multiNodesInferPort, 65535U, 1024U,
                                                                    "backendConfig.multiNodesInferPort"));
    for (auto npuDeviceId : backendConfig_.npuDeviceIds) {
        if (npuDeviceId.size() != backendConfig_.worldSize) {
            std::cout << "npuDeviceID does not allow repetitive element" << std::endl;
            initFlag = false;
        }
    }
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckKvPoolBackend(backendConfig_.kvPoolConfig.backend));
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckKvPoolConfigPath(backendConfig_.kvPoolConfig.configPath));
    if (backendConfig_.lwdMultiNodesEnable) {
        CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<int32_t>(backendConfig_.lwdMultiNodesCtrlPort,
            65535U, 1024U, "backendConfig.layerwiseDisaggregatedMultiNodesCtrlPort"));
    }
    return initFlag;
}

bool BackendConfigManager::CheckBackendInterTlsParam()
{
    if (!backendConfig_.interNodeTLSEnabled) {
        return true;
    }
    if (!CheckInterTlsParam()) {
        std::cout << "Backend inter tls config is invalid" << std::endl;
        return false;
    }
    return true;
}

const struct BackendConfig &BackendConfigManager::GetParam() { return backendConfig_; }

void BackendConfigManager::UpdateMultiNodesInfer(const RanktableParam &ranktableParam)
{
    for (auto &npuDeviceId : backendConfig_.npuDeviceIds) {
        backendConfig_.worldSize = ranktableParam.worldSize;
        npuDeviceId.clear();
        for (auto &ele : ranktableParam.local.device) {
            try {
                npuDeviceId.insert(static_cast<size_t>(std::stoi(ele.deviceId)));
            } catch (const std::invalid_argument &e) {
                initFlag = false;
                std::cout << "Invalid device_id " << ele.deviceId << " in ranktable file" << std::endl;
                return;
            } catch (const std::out_of_range &e) {
                initFlag = false;
                std::cout << "Invalid device_id " << ele.deviceId << " in ranktable file" << std::endl;
                return;
            } catch (...) {
                initFlag = false;
                std::cout << "Invalid device_id " << ele.deviceId << " in ranktable file" << std::endl;
                return;
            }
        }
    }
    std::cout << "Update worldSize and npuDeviceIds of backend config successfully for Multi Nodes Inference."
              << std::endl;
}
} // namespace mindie_llm