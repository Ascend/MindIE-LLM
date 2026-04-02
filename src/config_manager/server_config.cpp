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
#include "file_utils.h"
#include "env_util.h"
#include "log.h"
#include "base_config_manager.h"
#include "check_utils.h"
#include "safe_io.h"

using Json = nlohmann::json;
using namespace nlohmann::literals;

namespace mindie_llm {
static std::vector<ParamSpec> g_serverParamsConstraint = {
    // { name, type, compulsory }
    {"ipAddress", "string", true},
    {"managementIpAddress", "string", false},
    {"port", "int32_t", true},
    {"managementPort", "int32_t", false},
    {"maxLinkNum", "uint32_t", true},
    {"httpsEnabled", "bool", true},
    {"tlsCert", "string", false},
    {"tlsCrlPath", "string", false},
    {"tlsCrlFiles", "array", false},
    {"tlsCaPath", "string", false},
    {"tlsCaFile", "array", false},
    {"tlsPk", "string", false},
    {"managementTlsCert", "string", false},
    {"managementTlsCrlPath", "string", false},
    {"managementTlsCrlFiles", "array", false},
    {"managementTlsCaFile", "array", false},
    {"managementTlsPk", "string", false},
    {"metricsTlsCert", "string", false},
    {"metricsTlsCrlPath", "string", false},
    {"metricsTlsCrlFiles", "array", false},
    {"metricsTlsCaFile", "array", false},
    {"metricsTlsPk", "string", false},
    {"fullTextEnabled", "bool", false},
    {"npuUsageThreshold", "uint32_t", false},
    {"inferMode", "string", true},
    {"allowAllZeroIpListening", "bool", true},
    {"openAiSupport", "string", false},
    {"metricsPort", "int32_t", true},
    {"tokenTimeout", "int32_t", true},
    {"e2eTimeout", "int32_t", true},
    {"maxRequestLength", "uint32_t", false},
    {"distDPServerEnabled", "bool", false},
    {"maxJsonDepth", "uint32_t", false},
    {"layerwiseDisaggregated", "bool", false},
    {"layerwiseDisaggregatedRoleType", "string", false},
    {"layerwiseDisaggregatedMasterIpAddress", "string", false},
    {"layerwiseDisaggregatedSlaveIpAddress", "array", false},
    {"layerwiseDisaggregatedDataPort", "int32_t", false},
    {"layerwiseDisaggregatedCrtlPort", "array", false}};

void ServerConfigManager::InitHttpsConfigFromJson(Json &serveJsonData, bool loadManagementSSL)
{
    if (serveJsonData.contains("tlsCaPath")) {
        serverConfig_.tlsCaPath = serveJsonData["tlsCaPath"];
    }

    if (serveJsonData.contains("tlsCaFile")) {
        auto ret = ParamChecker::CheckJsonArray(serveJsonData["tlsCaFile"], "string", "");
        if (!ret) {
            std::cout << "tlsCaFile init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["tlsCaFile"]) {
                serverConfig_.tlsCaFile.insert(static_cast<std::string>(ele));
            }
        }
    }
    InitHttpsBusinessConfigFromJson(serveJsonData);
    if (loadManagementSSL) {
        InitHttpsManagementConfigFromJson(serveJsonData);
        InitHttpsMetricsConfigFromJson(serveJsonData);
    }
}

void ServerConfigManager::InitDMIHttpsConfigFromJson(Json &serveJsonData)
{
    if (serveJsonData.contains("interCommTlsCert")) {
        serverConfig_.interCommTlsCert = serveJsonData["interCommTlsCert"];
    }
    if (serveJsonData.contains("interCommTlsCaPath")) {
        serverConfig_.interCommTlsCaPath = serveJsonData["interCommTlsCaPath"];
    }
    if (serveJsonData.contains("interCommTlsCaFiles")) {
        auto ret = ParamChecker::CheckJsonArray(serveJsonData["interCommTlsCaFiles"], "string", "");
        if (!ret) {
            std::cout << "interCommTlsCaFiles init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["interCommTlsCaFiles"]) {
                serverConfig_.interCommTlsCaFiles.push_back(static_cast<std::string>(ele));
            }
        }
    }
    if (serveJsonData.contains("interCommPk")) {
        serverConfig_.interCommPk = serveJsonData["interCommPk"];
    }
    if (serveJsonData.contains("interCommTlsCrlPath")) {
        serverConfig_.interCommTlsCrlPath = serveJsonData["interCommTlsCrlPath"];
    }
    if (serveJsonData.contains("interCommTlsCrlFiles")) {
        auto ret = ParamChecker::CheckJsonArray(serveJsonData["interCommTlsCrlFiles"], "string", "");
        if (!ret) {
            std::cout << "interCommTlsCrlFiles init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["interCommTlsCrlFiles"]) {
                serverConfig_.interCommTlsCrlFiles.push_back(static_cast<std::string>(ele));
            }
        }
    }
}

void ServerConfigManager::InitHttpsBusinessConfigFromJson(Json &serveJsonData)
{
    if (serveJsonData.contains("tlsCert")) {
        serverConfig_.tlsCert = serveJsonData["tlsCert"];
    }
    if (serveJsonData.contains("tlsCrlPath")) {
        serverConfig_.tlsCrlPath = serveJsonData["tlsCrlPath"];
    }
    if (serveJsonData.contains("tlsCrlFiles")) {
        if (!ParamChecker::CheckJsonArray(serveJsonData["tlsCrlFiles"], "string", "")) {
            std::cout << "tlsCrlFiles init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["tlsCrlFiles"]) {
                serverConfig_.tlsCrlFiles.insert(static_cast<std::string>(ele));
            }
        }
    }

    if (serveJsonData.contains("tlsCaFile")) {
        if (!ParamChecker::CheckJsonArray(serveJsonData["tlsCaFile"], "string", "")) {
            std::cout << "tlsCaFile init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["tlsCaFile"]) {
                serverConfig_.tlsCaFile.insert(static_cast<std::string>(ele));
            }
        }
    }

    if (serveJsonData.contains("tlsPk")) {
        serverConfig_.tlsPk = serveJsonData["tlsPk"];
    }
}

void ServerConfigManager::InitHttpsManagementConfigFromJson(Json &serveJsonData)
{
    if (serveJsonData.contains("managementTlsCert")) {
        serverConfig_.managementTlsCert = serveJsonData["managementTlsCert"];
    }
    if (serveJsonData.contains("managementTlsCrlPath")) {
        serverConfig_.managementTlsCrlPath = serveJsonData["managementTlsCrlPath"];
    }
    if (serveJsonData.contains("managementTlsCrlFiles")) {
        if (!ParamChecker::CheckJsonArray(serveJsonData["managementTlsCrlFiles"], "string", "")) {
            std::cout << "managementTlsCrlFiles init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["managementTlsCrlFiles"]) {
                serverConfig_.managementTlsCrlFiles.insert(static_cast<std::string>(ele));
            }
        }
    }

    if (serveJsonData.contains("managementTlsCaFile")) {
        if (!ParamChecker::CheckJsonArray(serveJsonData["managementTlsCaFile"], "string", "")) {
            std::cout << "managementTlsCaFile init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["managementTlsCaFile"]) {
                serverConfig_.managementTlsCaFile.insert(static_cast<std::string>(ele));
            }
        }
    }
    if (serveJsonData.contains("managementTlsPk")) {
        serverConfig_.managementTlsPk = serveJsonData["managementTlsPk"];
    }
}

void ServerConfigManager::InitLayerwiseDisaggregatedConfigFromJson(Json &serveJsonData)
{
    if (serveJsonData.contains("layerwiseDisaggregated")) {
        serverConfig_.layerwiseDisaggregated = serveJsonData["layerwiseDisaggregated"];
    }

    if (!serverConfig_.layerwiseDisaggregated) {
        return;
    }

    if (serveJsonData.contains("layerwiseDisaggregatedRoleType")) {
        serverConfig_.layerwiseDisaggregatedRoleType = serveJsonData["layerwiseDisaggregatedRoleType"];
    }

    if (serveJsonData.contains("layerwiseDisaggregatedMasterIpAddress")) {
        serverConfig_.layerwiseDisaggregatedMasterIpAddress = serveJsonData["layerwiseDisaggregatedMasterIpAddress"];
    }

    if (serveJsonData.contains("layerwiseDisaggregatedDataPort")) {
        serverConfig_.layerwiseDisaggregatedDataPort = serveJsonData["layerwiseDisaggregatedDataPort"];
    }

    if (serveJsonData.contains("layerwiseDisaggregatedSlaveIpAddress")) {
        auto ret = ParamChecker::CheckJsonArray(serveJsonData["layerwiseDisaggregatedSlaveIpAddress"], "string", "");
        if (!ret) {
            MINDIE_LLM_LOG_ERROR("layerwiseDisaggregatedSlaveIpAddress format is incorrect, "
                         "it should be of type string-JsonArray." << std::endl);
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["layerwiseDisaggregatedSlaveIpAddress"]) {
                serverConfig_.layerwiseDisaggregatedSlaveIpAddress.push_back(static_cast<std::string>(ele));
            }
        }
    }

    if (serveJsonData.contains("layerwiseDisaggregatedCrtlPort")) {
        auto ret = ParamChecker::CheckJsonArray(serveJsonData["layerwiseDisaggregatedCrtlPort"], "integer", "int32_t");
        if (!ret) {
            MINDIE_LLM_LOG_ERROR("layerwiseDisaggregatedCrtlPort format is incorrect, "
                         "it should be of type integer-JsonArray." << std::endl);
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["layerwiseDisaggregatedCrtlPort"]) {
                serverConfig_.layerwiseDisaggregatedCrtlPort.push_back(static_cast<int32_t>(ele));
            }
        }
    }
}

void ServerConfigManager::InitHttpsMetricsConfigFromJson(Json &serveJsonData)
{
    if (serveJsonData.contains("metricsTlsCert")) {
        serverConfig_.metricsTlsCert = serveJsonData["metricsTlsCert"];
    }
    if (serveJsonData.contains("metricsTlsCrlPath")) {
        serverConfig_.metricsTlsCrlPath = serveJsonData["metricsTlsCrlPath"];
    }
    if (serveJsonData.contains("metricsTlsCrlFiles")) {
        if (!ParamChecker::CheckJsonArray(serveJsonData["metricsTlsCrlFiles"], "string", "")) {
            std::cout << "metricsTlsCrlFiles init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["metricsTlsCrlFiles"]) {
                serverConfig_.metricsTlsCrlFiles.insert(static_cast<std::string>(ele));
            }
        }
    }

    if (serveJsonData.contains("metricsTlsCaFile")) {
        if (!ParamChecker::CheckJsonArray(serveJsonData["metricsTlsCaFile"], "string", "")) {
            std::cout << "metricsTlsCaFile init error" << std::endl;
            jsonDecodeSuccess_ = false;
            return;
        } else {
            for (auto &ele : serveJsonData["metricsTlsCaFile"]) {
                serverConfig_.metricsTlsCaFile.insert(static_cast<std::string>(ele));
            }
        }
    }
    if (serveJsonData.contains("metricsTlsPk")) {
        serverConfig_.metricsTlsPk = serveJsonData["metricsTlsPk"];
    }
}

bool ServerConfigManager::InitFromJson()
{
    Json serverParamsJsonData;
    if (!CheckSystemConfig(jsonPath_, serverParamsJsonData, "ServerConfig")) {
        return false;
    }
    if (!ParamChecker::CheckJsonParamType(serverParamsJsonData, g_serverParamsConstraint)) {
        return false;
    }
    serverConfig_.managementIpAddress = GetManagementIPAddress(serverParamsJsonData);
    serverConfig_.managementPort = (serverParamsJsonData.contains("managementPort"))
                                       ? serverParamsJsonData["managementPort"]
                                       : serverParamsJsonData["port"];
    serverConfig_.metricsPort = serverParamsJsonData["metricsPort"];
    serverConfig_.allowAllZeroIpListening = serverParamsJsonData["allowAllZeroIpListening"];
    serverConfig_.ipAddress = GetIPAddress(serverParamsJsonData);
    serverConfig_.port = serverParamsJsonData["port"];
    serverConfig_.maxLinkNum = serverParamsJsonData["maxLinkNum"];
    serverConfig_.httpsEnabled = serverParamsJsonData["httpsEnabled"];
    serverConfig_.tokenTimeout = serverParamsJsonData["tokenTimeout"];
    serverConfig_.e2eTimeout = serverParamsJsonData["e2eTimeout"];
    serverConfig_.fullTextEnabled = ParamChecker::GetBoolParamValue(serverParamsJsonData, "fullTextEnabled", false);
    serverConfig_.distDPServerEnabled = serverParamsJsonData["distDPServerEnabled"];
    
    LoadOptionalParameters(serverParamsJsonData);
    InitLayerwiseDisaggregatedConfigFromJson(serverParamsJsonData);
    return initFlag;
}

void ServerConfigManager::LoadOptionalParameters(Json& serverParamsJsonData)
{
    if (serverParamsJsonData.contains("maxRequestLength")) {
        serverConfig_.maxRequestLength = serverParamsJsonData["maxRequestLength"];
        CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<uint32_t>(serverConfig_.maxRequestLength, 100U,
                                                                                  1U, "serverConfig.maxRequestLength"));
    }
    if (serverParamsJsonData.contains("maxJsonDepth")) {
        serverConfig_.maxJsonDepth = serverParamsJsonData["maxJsonDepth"];
        CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<uint32_t>(serverConfig_.maxJsonDepth,
            JSON_DEPTH_LIMIT_MAX, JSON_DEPTH_LIMIT_MIN, "serverConfig.maxJsonDepth"));
        SetJsonDepthLimit(static_cast<int>(serverConfig_.maxJsonDepth));
    }
    if (serverParamsJsonData.contains("HealthCheckConfig") &&
        serverParamsJsonData["HealthCheckConfig"].contains("npuUsageThreshold")) {
        serverConfig_.npuUsageThreshold = serverParamsJsonData["HealthCheckConfig"]["npuUsageThreshold"];
        CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<uint32_t>(serverConfig_.npuUsageThreshold,
            100U, 0U, "serverConfig.HealthCheckConfig.npuUsageThreshold"));
    }
    if (serverParamsJsonData.contains("inferMode")) {
        serverConfig_.inferMode = serverParamsJsonData["inferMode"];
    }
    if (serverParamsJsonData.contains("interCommTLSEnabled")) {
        serverConfig_.interCommTLSEnabled = serverParamsJsonData["interCommTLSEnabled"];
    }
    if (serverParamsJsonData.contains("interCommPort")) {
        serverConfig_.interCommPort = serverParamsJsonData["interCommPort"];
        CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<int32_t>(serverConfig_.interCommPort, 65535U,
                                                                                  1024U, "serverConfig.interCommPort"));
    }
    if (serverConfig_.interCommTLSEnabled) {
        InitDMIHttpsConfigFromJson(serverParamsJsonData);
    }
    if (serverConfig_.httpsEnabled) {
        bool checkManagement = true;
        if (serverConfig_.managementIpAddress == serverConfig_.ipAddress &&
            serverConfig_.managementPort == serverConfig_.port &&
            serverConfig_.managementPort == serverConfig_.metricsPort) {
            checkManagement = false;
        }
        try {
            InitHttpsConfigFromJson(serverParamsJsonData, checkManagement);
        } catch (const nlohmann::json::type_error &e) {
            std::cout << "JSON type error: " << "[ServerConfigManager::InitFromJson] " << e.what() << std::endl;
            initFlag = false;
            return;
        }
    }
    serverConfig_.openAiSupportedvLLM = !(serverParamsJsonData.contains("openAiSupport") &&
        std::string(serverParamsJsonData["openAiSupport"]) != "vllm");
}

bool ServerConfigManager::CheckParam()
{
    // log路径的参数校验移至inferengine日志初始化处
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckInferMode(serverConfig_.inferMode));
    CHECK_CONFIG_VALIDATION(initFlag,
                            CheckIp(serverConfig_.ipAddress, "ipAddress", serverConfig_.allowAllZeroIpListening));
    CHECK_CONFIG_VALIDATION(initFlag, CheckIp(serverConfig_.managementIpAddress, "managementIpAddress",
                                              serverConfig_.allowAllZeroIpListening));
    CHECK_CONFIG_VALIDATION(
        initFlag, ParamChecker::CheckMaxMinValue<int32_t>(serverConfig_.port, 65535U, 1024U, "serverConfig.port"));
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<int32_t>(serverConfig_.managementPort, 65535U,
                                                                              1024U, "serverConfig.managementPort"));
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<int32_t>(serverConfig_.metricsPort, 65535U, 1024U,
                                                                              "serverConfig.metricsPort"));
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<uint32_t>(serverConfig_.maxLinkNum, 4096U, 1U,
                                                                               "serverConfig.maxLinkNum"));
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<int32_t>(serverConfig_.tokenTimeout, 3600U, 1U,
                                                                              "serverConfig.tokenTimeout"));
    CHECK_CONFIG_VALIDATION(initFlag, ParamChecker::CheckMaxMinValue<int32_t>(
                                          serverConfig_.e2eTimeout, 65535U, 1U,
                                          "serverConfig.e2eTimeout")); // 65535: Max end-to-end inference timeout
    if (serverConfig_.httpsEnabled) {
        bool checkManagement = true;
        if (serverConfig_.managementIpAddress == serverConfig_.ipAddress &&
            serverConfig_.managementPort == serverConfig_.port &&
            serverConfig_.managementPort == serverConfig_.metricsPort) {
            checkManagement = false;
        }
        CHECK_CONFIG_VALIDATION(initFlag, CheckHttpsConfig(checkManagement));
    }

    if (serverConfig_.layerwiseDisaggregated) {
        CHECK_CONFIG_VALIDATION(initFlag, CheckLayerwiseDisaggregatedConfig());
    }
    
    return initFlag;
}

bool ServerConfigManager::CheckHttpsConfig(bool loadManagementSSL)
{
    // log路径的参数校验移至inferengine日志初始化处
    bool checkRes = true;
    std::string homePath;
    if (!GetHomePath(homePath).IsOk()) {
        std::cout << "Failed to get home path." << std::endl;
        return false;
    }
    homePath += "/";

    CHECK_CONFIG_VALIDATION(checkRes, CheckBusinessHttpsParam());
    if (loadManagementSSL) {
        CHECK_CONFIG_VALIDATION(checkRes, CheckManagementHttpsParam());
        CHECK_CONFIG_VALIDATION(checkRes, CheckMetricsHttpsParam(homePath));
    }

    return checkRes;
}

bool ServerConfigManager::CheckBusinessHttpsParam()
{
    bool checkRes = true;
    std::string homePath;
    if (!GetHomePath(homePath).IsOk()) {
        std::cout << "Failed to get home path." << std::endl;
        return false;
    }
    homePath += "/";

    std::string tlsCertPath = homePath + serverConfig_.tlsCert;
    CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckPath(tlsCertPath, homePath, "serverConfig_.tlsCert"));

    // 吊销证书可为空
    uint32_t maxFileNumber = 3;
    if (!serverConfig_.tlsCrlPath.empty() && !serverConfig_.tlsCrlFiles.empty()) {
        if (serverConfig_.tlsCrlFiles.size() > maxFileNumber) {
            std::cout << "ERR: serverConfig_.tlsCrlFiles size must in [0, 3] ." << std::endl;
            checkRes &= false;
        }
        std::string tlsCrlPath = homePath + serverConfig_.tlsCrlPath;
        checkRes &= ParamChecker::CheckPath(tlsCrlPath, homePath, "serverConfig_.tlsCrlPath", false);
        for (const std::string &crlFilePath : serverConfig_.tlsCrlFiles) {
            std::string tlsCrlFilePath = tlsCrlPath + crlFilePath;
            if (!FileUtils::CheckFileExists(tlsCrlFilePath) || FileUtils::CheckDirectoryExists(tlsCrlFilePath)) {
                std::cout << "ERR: serverConfig_.tlsCrlFiles file not exit ." << std::endl;
                checkRes &= false;
                continue;
            }
            checkRes &= ParamChecker::CheckPath(tlsCrlFilePath, homePath, "serverConfig_.tlsCrlFiles");
        }
    }

    if (serverConfig_.tlsCaFile.empty() || serverConfig_.tlsCaFile.size() > maxFileNumber) {
        std::cout << "ERR: serverConfig_.tlsCaFile size must in [1, 3] ." << std::endl;
        checkRes = false;
    }
    std::string tlsCaPath = homePath + serverConfig_.tlsCaPath;
    CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckPath(tlsCaPath, homePath, "serverConfig_.tlsCaPath", false));
    for (const std::string &caFilePath : serverConfig_.tlsCaFile) {
        std::string tlsCaFilePath = tlsCaPath + caFilePath;
        CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckPath(tlsCaFilePath, homePath, "serverConfig_.tlsCaFile"));
    }

    std::string tlsPkPath = homePath + serverConfig_.tlsPk;
    CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckPath(tlsPkPath, homePath, "serverConfig_.tlsPk"));
    return checkRes;
}

bool ServerConfigManager::CheckManagementHttpsParam()
{
    bool checkRes = true;
    std::string homePath;
    if (!GetHomePath(homePath).IsOk()) {
        std::cout << "Failed to get home path." << std::endl;
        return false;
    }
    homePath += "/";

    std::string managementTlsCert = homePath + serverConfig_.managementTlsCert;
    CHECK_CONFIG_VALIDATION(checkRes,
                            ParamChecker::CheckPath(managementTlsCert, homePath, "ServerConfig.managementTlsCert"));

    // 吊销证书可为空
    uint32_t maxFileNumber = 3;
    if (!serverConfig_.managementTlsCrlPath.empty() && !serverConfig_.managementTlsCrlFiles.empty()) {
        if (serverConfig_.managementTlsCrlFiles.size() > maxFileNumber) {
            std::cout << "ERR: serverConfig_.managementTlsCrlFiles size must in [0, 3] ." << std::endl;
            checkRes &= false;
        }
        std::string managementTlsCrlPath = homePath + serverConfig_.managementTlsCrlPath;
        checkRes = ParamChecker::CheckPath(managementTlsCrlPath, homePath, "ServerConfig.managementTlsCrlPath", false) && checkRes;
        for (const std::string &crlFilePath : serverConfig_.managementTlsCrlFiles) {
            std::string tlsCrlFilePath = managementTlsCrlPath + crlFilePath;
            if (!FileUtils::CheckFileExists(tlsCrlFilePath) || FileUtils::CheckDirectoryExists(tlsCrlFilePath)) {
                std::cout << "ERR: serverConfig_.managementTlsCrlFiles file not exit ." << std::endl;
                checkRes &= false;
                continue;
            }
            checkRes &= ParamChecker::CheckPath(tlsCrlFilePath, homePath, "ServerConfig.managementTlsCrlFiles");
        }
    }

    if (serverConfig_.managementTlsCaFile.empty() || serverConfig_.managementTlsCaFile.size() > maxFileNumber) {
        std::cout << "ERR: serverConfig_.managementTlsCaFile size must in [1, 3]." << std::endl;
        checkRes = false;
    }

    std::string managementTlsCaPath = homePath + serverConfig_.tlsCaPath;
    CHECK_CONFIG_VALIDATION(checkRes,
                            ParamChecker::CheckPath(managementTlsCaPath, homePath, "ServerConfig.tlsCaPath", false));
    for (const std::string &managementCaFilePath : serverConfig_.managementTlsCaFile) {
        std::string managementTlsCaFilePath = managementTlsCaPath + managementCaFilePath;
        if (!FileUtils::CheckFileExists(managementTlsCaFilePath) ||
            FileUtils::CheckDirectoryExists(managementTlsCaFilePath)) {
            std::cout << "ERR: serverConfig_.managementTlsCaFile file not exit ." << std::endl;
            checkRes &= false;
            continue;
        }
        CHECK_CONFIG_VALIDATION(
            checkRes, ParamChecker::CheckPath(managementTlsCaFilePath, homePath, "ServerConfig.managementTlsCaFile"));
    }

    std::string managementTlsPkPath = homePath + serverConfig_.managementTlsPk;
    CHECK_CONFIG_VALIDATION(checkRes,
                            ParamChecker::CheckPath(managementTlsPkPath, homePath, "ServerConfig.managementTlsPk"));
    return checkRes;
}

bool ServerConfigManager::CheckLayerwiseDisaggregatedConfig()
{
    bool checkRes = true;
    if (serverConfig_.layerwiseDisaggregated) {
        if (serverConfig_.layerwiseDisaggregatedRoleType != "master" &&
            serverConfig_.layerwiseDisaggregatedRoleType != "slave") {
            checkRes = false;
        }
        CHECK_CONFIG_VALIDATION(checkRes,
                                CheckIp(serverConfig_.layerwiseDisaggregatedMasterIpAddress,
                                "layerwiseDisaggregatedMasterIpAddress", false));
        
        for (auto &slaveIp: serverConfig_.layerwiseDisaggregatedSlaveIpAddress) {
            CHECK_CONFIG_VALIDATION(checkRes,
                                    CheckIp(slaveIp, "layerwiseDisaggregatedSlaveIpAddress", false));
        }

        CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckMaxMinValue<int32_t>(
            serverConfig_.layerwiseDisaggregatedDataPort, 65535U, 1024U,
            "serverConfig.layerwiseDisaggregatedDataPort"));

        for (auto &crtlPort: serverConfig_.layerwiseDisaggregatedCrtlPort) {
            CHECK_CONFIG_VALIDATION(checkRes, ParamChecker::CheckMaxMinValue<int32_t>(
            crtlPort, 65535U, 1024U, "serverConfig.layerwiseDisaggregatedCrtlPort"));
        }
    }
    return checkRes;
}

bool ServerConfigManager::CheckMetricsHttpsParam(std::string& homePath)
{
    bool checkRes = true;

    std::string metricsTlsCert = homePath + serverConfig_.metricsTlsCert;
    CHECK_CONFIG_VALIDATION(checkRes,
                            ParamChecker::CheckPath(metricsTlsCert, homePath, "ServerConfig.metricsTlsCert"));

    // 吊销证书可为空
    uint32_t maxFileNumber = 3;
    if (!serverConfig_.metricsTlsCrlPath.empty() && !serverConfig_.metricsTlsCrlFiles.empty()) {
        if (serverConfig_.metricsTlsCrlFiles.size() > maxFileNumber) {
            std::cout << "ERR: serverConfig_.metricsTlsCrlFiles size must in [0, 3] ." << std::endl;
            checkRes = checkRes && false;
        }
        std::string metricsTlsCrlPath = homePath + serverConfig_.metricsTlsCrlPath;
        checkRes = ParamChecker::CheckPath(
            metricsTlsCrlPath, homePath, "ServerConfig.metricsTlsCrlPath", false) && checkRes;
        for (const std::string &crlFilePath : serverConfig_.metricsTlsCrlFiles) {
            std::string tlsCrlFilePath = metricsTlsCrlPath + crlFilePath;
            if (!FileUtils::CheckFileExists(tlsCrlFilePath) || FileUtils::CheckDirectoryExists(tlsCrlFilePath)) {
                std::cout << "ERR: serverConfig_.metricsTlsCrlFiles file not exit ." << std::endl;
                checkRes = checkRes && false;
                continue;
            }
            checkRes = checkRes && ParamChecker::CheckPath(tlsCrlFilePath, homePath, "ServerConfig.metricsTlsCrlFiles");
        }
    }

    if (serverConfig_.metricsTlsCaFile.empty() || serverConfig_.metricsTlsCaFile.size() > maxFileNumber) {
        std::cout << "ERR: serverConfig_.metricsTlsCaFile size must in [1, 3]." << std::endl;
        checkRes = false;
    }

    std::string metricsTlsCaPath = homePath + serverConfig_.tlsCaPath;
    CHECK_CONFIG_VALIDATION(checkRes,
        ParamChecker::CheckPath(metricsTlsCaPath, homePath, "ServerConfig.tlsCaPath", false));
    for (const std::string &metricsCaFilePath : serverConfig_.metricsTlsCaFile) {
        std::string metricsTlsCaFilePath = metricsTlsCaPath + metricsCaFilePath;
        if (!FileUtils::CheckFileExists(metricsTlsCaFilePath) ||
            FileUtils::CheckDirectoryExists(metricsTlsCaFilePath)) {
            std::cout << "ERR: serverConfig_.metricsTlsCaFile file not exit ." << std::endl;
            checkRes = checkRes && false;
            continue;
        }
        CHECK_CONFIG_VALIDATION(
            checkRes, ParamChecker::CheckPath(metricsTlsCaFilePath, homePath, "ServerConfig.metricsTlsCaFile"));
    }

    std::string metricsTlsPkPath = homePath + serverConfig_.metricsTlsPk;
    CHECK_CONFIG_VALIDATION(checkRes,
                            ParamChecker::CheckPath(metricsTlsPkPath, homePath, "ServerConfig.metricsTlsPk"));
    return checkRes;
}

bool ServerConfigManager::GetDecodeStatus() const  { return jsonDecodeSuccess_; }

std::string ServerConfigManager::GetIPAddress(Json &serveJsonData)
{
    auto ip = EnvUtil::GetInstance().Get("MIES_CONTAINER_IP");
    if (!ip.empty()) {
        return ip;
    } else {
        return serveJsonData["ipAddress"];
    }
}

std::string ServerConfigManager::GetManagementIPAddress(Json &serveJsonData)
{
    auto ip = EnvUtil::GetInstance().Get("MIES_CONTAINER_MANAGEMENT_IP");
    if (!ip.empty()) {
        return ip;
    } else if (serveJsonData.contains("managementIpAddress")) {
        return serveJsonData["managementIpAddress"];
    } else {
        return serveJsonData["ipAddress"];
    }
}

void ServerConfigManager::UpdateConfig()
{
    Json serverParamsJsonData;
    bool openAiSupportedvLLMLatest = true;

    if (CheckSystemConfig(jsonPath_, serverParamsJsonData, "ServerConfig")) {
        if (serverParamsJsonData.contains("openAiSupport") &&
            std::string(serverParamsJsonData["openAiSupport"]) != "vllm") {
            openAiSupportedvLLMLatest = false;
        } else {
            openAiSupportedvLLMLatest = true;
        }

        if (serverConfig_.openAiSupportedvLLM != openAiSupportedvLLMLatest) {
            std::cout << "ServerConfigManager::UpdateConfig openAiSupportedvLLM changed, from " << serverConfig_.openAiSupportedvLLM << " to " << openAiSupportedvLLMLatest << std::endl;
            serverConfig_.openAiSupportedvLLM = openAiSupportedvLLMLatest;
        }
    }
}

const struct ServerConfig &ServerConfigManager::GetParam() { return serverConfig_; }

void ServerConfigManager::SetPluginEnabled(bool enabled)
{
    serverConfig_.pluginEnabled = enabled;
}

void ServerConfigManager::SetDeepseekEnabled(bool enabled)
{
    serverConfig_.deepseekEnabled = enabled;
}

void ServerConfigManager::SetTokenTimeout(uint64_t tokenTimeout)
{
    std::cout << "ServerConfigManager::SetTokenTimeout tokenTimeout changed, from " <<
    serverConfig_.tokenTimeout << " to " << tokenTimeout << std::endl;

    serverConfig_.tokenTimeout = tokenTimeout;
}

void ServerConfigManager::SetE2eTimeout(uint64_t e2eTimeout)
{
    std::cout << "ServerConfigManager::SetE2eTimeout e2eTimeout changed, from " <<
    serverConfig_.e2eTimeout << " to " << e2eTimeout << std::endl;
    
    serverConfig_.e2eTimeout = e2eTimeout;
}

} // namespace mindie_llm
