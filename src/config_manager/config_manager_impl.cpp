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
 
#include <thread>
#include "cmath"
#include "common_util.h"
#include "base_config_manager.h"
#include "env_util.h"
#include "config_manager.h"
#include "log.h"
#include "config_manager_impl.h"

namespace mindie_llm {

static std::string g_configPath = "";
static bool g_initialized = false;

ConfigManager::ConfigManager(const std::string &jsonPath) : impl_(std::make_unique<Impl>(jsonPath))
{
    if (!CheckAllParam()) {
        throw std::runtime_error("[ConfigManager] Checking Config Failed");
    }
}
const BackendConfig &ConfigManager::GetBackendConfig() const { return impl_->GetBackendConfig(); }
const LogConfig &ConfigManager::GetLogConfig() const { return impl_->GetLogConfig(); }
const ServerConfig &ConfigManager::GetServerConfig() const { return impl_->GetServerConfig(); }
const std::vector<ModelDeployConfig> &ConfigManager::GetModelDeployConfig() const
{
    return impl_->GetModelDeployConfig();
}
const std::vector<LoraConfig> &ConfigManager::GetLoraConfig() const { return impl_->GetLoraConfig(); }
const ScheduleConfig &ConfigManager::GetScheduleConfig() const { return impl_->GetScheduleConfig(); }
const RanktableParam &ConfigManager::GetRanktableParam() const { return impl_->GetRanktableParam(); }

std::string ConfigManager::GetConfigJsonStr() { return impl_->GetConfigJsonStr(); };

bool ConfigManager::CheckAllParam() { return impl_->CheckAllParam(); }
bool ConfigManager::CheckAndInitLogParam() { return impl_->CheckAndInitLogParam(); }

bool ConfigManager::CreateInstance(std::string jsonPath)
{
    if (g_initialized) {
        return true;
    }

    if (jsonPath.empty() && !GetConfigPath(jsonPath).IsOk()) {
        std::cout << "ConfigManager: Get config path failed." << std::endl;
        return false;
    }
    if (!CanonicalPath(jsonPath)) {
        std::cout << "ConfigManager: Invalid config path." << std::endl;
        return false;
    }
    std::cout << "ConfigManager: Load Config from " << jsonPath << "." << std::endl;
    g_configPath = jsonPath;
    g_initialized = true;
    try {
        GetInstance();
    } catch (const nlohmann::json::exception &e) {
        std::cout << "JSON error in ConfigManager: " << e.what() << std::endl;
        return false;
    } catch (const std::runtime_error &e) {
        std::cout << "ConfigManager init exception: " << e.what() << std::endl;
        return false;
    } catch (const std::exception &e) {
        std::cout << "ConfigManager Init exception: " << e.what() << std::endl;
        return false;
    }
    return true;
}

ConfigManager &ConfigManager::GetInstance()
{
    if (!g_initialized) {
        throw std::runtime_error("ConfigManager is not initialized, call CreateInstance(jsonPath) first");
    }

    static ConfigManager instance(g_configPath);
    return instance;
}

ConfigManager::~ConfigManager() = default;

bool ConfigManager::IsMultiNodeInfer() const { return impl_->IsMultiNodeInfer(); }

bool ConfigManager::IslayerwiseDisaggregated() const
{
    return impl_->IslayerwiseDisaggregated();
}

bool ConfigManager::IsLwdMultiNodesEnable() const
{
    return impl_->IsLwdMultiNodesEnable();
}

std::string ConfigManager::GetLwdRoleType() const
{
    return impl_->GetLwdRoleType();
}

void ConfigManager::SetMaxPositionEmbeddings(unsigned int maxPositionEmbeddings)
{
    impl_->SetMaxPositionEmbeddings(maxPositionEmbeddings);
}

void ConfigManager::SetTokenTimeout(uint64_t tokenTimeout)
{
    impl_->SetTokenTimeout(tokenTimeout);
}

void ConfigManager::SetE2eTimeout(uint64_t e2eTimeout)
{
    impl_->SetE2eTimeout(e2eTimeout);
}

ConfigManager::Impl::Impl(const std::string &jsonPath)
{
    BaseConfig::CheckSystemConfig(jsonPath, configJson_, "");
    scheduleConfig_ = std::make_shared<ScheduleConfigManager>(jsonPath);
    modelDeployConfig_ = std::make_shared<ModelDeployConfigManager>(jsonPath);
    logConfig_ = std::make_shared<LogConfigManager>(jsonPath);
    serverConfig_ = std::make_shared<ServerConfigManager>(jsonPath);
    backendConfig_ = std::make_shared<BackendConfigManager>(jsonPath);
    if (!serverConfig_->InitFromJson()) {
        throw std::runtime_error("Failed to initialize ServerConfig from JSON.");
    }
    if (!backendConfig_->InitFromJson()) {
        throw std::runtime_error("Failed to initialize BackendConfig from JSON.");
    }
    if (!scheduleConfig_->InitFromJson()) {
        throw std::runtime_error("Failed to initialize ScheduleConfig from JSON.");
    }
    if (!modelDeployConfig_->InitFromJson()) {
        throw std::runtime_error("Failed to initialize ModelDeployConfig from JSON.");
    }

    // 新增边云多机场景, 边侧目前只有一台机器, 云侧才是真正的多机
    bool isLwdMultiNodesSlave = IsLwdMultiNodesEnable() && GetLwdRoleType() == "slave";
    if (IsMultiNodeInfer() || scheduleConfig_->GetParam().distributedEnable || isLwdMultiNodesSlave) {
        ranktableConfig_ = std::make_shared<RanktableConfigManager>();
        if (!ranktableConfig_->InitFromJson()) {
            throw std::runtime_error("init rank table config error.");
        }
        backendConfig_->UpdateMultiNodesInfer(ranktableConfig_->GetParam());
    }

    if (!configJson_.contains("EnableDynamicAdjustTimeoutConfig")) {
        std::cout << "The configName EnableDynamicAdjustTimeoutConfig is not found, "
                  << "using default value false. If you need to customize this "
                  << "configuration item, please add it to the config file."
                  << std::endl;
    }

    // 执行配置交互
    ExecuteConfigInteractions();
    jsonPath_ = jsonPath;
    initialized_ = true;

    task_ = std::thread([this]() { this->UpdateConfig(); });

    std::cout << "[ConfigManager::InitConfigManager] Successfully init config manager" << std::endl;
}

bool ConfigManager::Impl::InitConfigManager()
{
    if (!serverConfig_->InitFromJson() || !backendConfig_->InitFromJson() || !scheduleConfig_->InitFromJson() ||
        !modelDeployConfig_->InitFromJson()) {
        return false;
    }
    if (IsMultiNodeInfer() || scheduleConfig_->GetParam().distributedEnable) {
        ranktableConfig_ = std::make_shared<RanktableConfigManager>();
        if (!ranktableConfig_->InitFromJson()) {
            return false;
        }
        backendConfig_->UpdateMultiNodesInfer(ranktableConfig_->GetParam());
    }
    initialized_ = true;

    std::cout << "[ConfigManager::InitConfigManager] Successfully init config manager" << std::endl;
    return true;
}

bool ConfigManager::Impl::IsMultiNodeInfer() const
{
    // 判断多机只能通过该属性判断，禁止通过 RANK_TABLE_FILE 或 MIES_CONTAINER_IP 判断
    return backendConfig_->GetParam().multiNodesInferEnabled;
}

bool ConfigManager::Impl::IsLwdMultiNodesEnable() const
{
    return backendConfig_->GetParam().lwdMultiNodesEnable;
}

bool ConfigManager::Impl::IslayerwiseDisaggregated() const
{
    return serverConfig_->GetParam().layerwiseDisaggregated;
}

std::string ConfigManager::Impl::GetLwdRoleType() const
{
    return serverConfig_->GetParam().layerwiseDisaggregatedRoleType;
}

bool ConfigManager::Impl::PreCheck() const
{
    if (!serverConfig_->GetDecodeStatus()) {
        return false;
    }
#ifndef UT_ENABLED
    for (const auto &item : modelDeployConfig_->GetParam()) {
        if (item.modelWeightPath.empty()) {
            return false;
        }
    }
#endif
    return true;
}

bool ConfigManager::Impl::CheckAndInitLogParam() { return logConfig_->CheckParam(); }

static bool CheckScheduleConfigParam(const ScheduleConfig &scheduleConfigParam)
{
    if (scheduleConfigParam.templateType != "Standard") {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with templateType " <<
            scheduleConfigParam.templateType << std::endl);
        return false;
    }

    if (scheduleConfigParam.supportSelectBatch) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with supportSelectBatch." << std::endl);
        return false;
    }

    if (scheduleConfigParam.bufferResponseEnabled) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with bufferResponseEnabled." << std::endl);
        return false;
    }

    return true;
}

static bool LwdCheckMultiNodesParam(const ServerConfig &serverConfigParam, const BackendConfig &backendConfigParam,
    uint32_t dpNum)
{
    bool multiNodes = backendConfigParam.lwdMultiNodesEnable;
    uint32_t slaveIpNum = serverConfigParam.layerwiseDisaggregatedSlaveIpAddress.size();
    if (multiNodes && slaveIpNum != 2) { // 多机目前仅支持2机
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated multi nodes only support slaveIpAddress size is 2 ." << std::endl);
        return false;
    }

    if (multiNodes && (dpNum < 1 || dpNum > 2)) { // 多机目前仅支持dp=1和dp=2
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated multi nodes only support dp size is 1 or 2." << std::endl);
        return false;
    }

    return true;
}

bool ConfigManager::Impl::CheckLayerwiseDisaggregatedParam()
{
    // cross validate layerwiseDisaggregated
    const auto& backendConfigParam = backendConfig_->GetParam();
    const auto& serverConfigParam = serverConfig_->GetParam();
    const auto& scheduleConfigParam = scheduleConfig_->GetParam();
    const auto& modelDeployConfigParamVec = modelDeployConfig_->GetParam();

    if (backendConfigParam.multiNodesInferEnabled) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with multiNodesInferEnabled."<<std::endl);
        return false;
    }

    if (serverConfigParam.distDPServerEnabled) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with distDPServerEnabled." << std::endl);
        return false;
    }

    if (modelDeployConfigParamVec.size() == 0) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated modelDeployConfigParamVec is empty." << std::endl);
        return false;
    }

    const auto& modelDeployConfigParam = modelDeployConfigParamVec[0];
    const auto& modelConfigParam = modelDeployConfigParam.modelConfig;

    if (modelDeployConfigParam.backendType!= "atb") {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with backendType "
                              <<modelDeployConfigParam.backendType<< std::endl);
        return false;
    }

    auto itrFindPluginType = modelConfigParam.find("plugin_type");
    bool checkConflicit = (itrFindPluginType!= modelConfigParam.end() && itrFindPluginType->second == "splitfuse");
    if (checkConflicit) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with plugin_type: splitfuse." << std::endl);
        return false;
    }

    if (!CheckScheduleConfigParam(scheduleConfigParam)) {
        return false;
    }

    auto itrFindDp = modelConfigParam.find("dp");
    bool isFindDp = itrFindDp != modelConfigParam.end();
    uint32_t dpNum = isFindDp ? std::stol(itrFindDp->second) : 0;
    if (!LwdCheckMultiNodesParam(serverConfigParam, backendConfigParam, dpNum)) {
        return false;
    }

    bool singleNode = !backendConfigParam.lwdMultiNodesEnable;
    bool singleNodeInsNotSupportDpNum = isFindDp && (dpNum != 1);
    if (singleNode && singleNodeInsNotSupportDpNum) {
        MINDIE_LLM_LOG_ERROR("layerwiseDisaggregated not incompatible with dp "<< itrFindDp->second << std::endl);
        return false;
    }

    return true;
}

bool ConfigManager::Impl::CheckAllParam()
{
    if (!PreCheck()) {
        return false;
    }
    bool result = true;
    // single validate
    CHECK_CONFIG_VALIDATION(result, scheduleConfig_->CheckParam());
    CHECK_CONFIG_VALIDATION(result, modelDeployConfig_->CheckParam());
    CHECK_CONFIG_VALIDATION(result, serverConfig_->CheckParam());
    CHECK_CONFIG_VALIDATION(result, backendConfig_->CheckParam());
    if (IsMultiNodeInfer() && ranktableConfig_ != nullptr) {
        CHECK_CONFIG_VALIDATION(result, ranktableConfig_->CheckParam());
        CHECK_CONFIG_VALIDATION(result, backendConfig_->CheckBackendInterTlsParam());
    }

    if (serverConfig_->GetParam().port == backendConfig_->GetParam().multiNodesInferPort) {
        std::cout << "Endpoint port cannot be equal to multiNodesInferPort." << std::endl;
        result = false;
    }

    // cross validate
    auto maxPrefillTokens = scheduleConfig_->GetParam().maxPrefillTokens;
    auto item = modelDeployConfig_->GetParam();
    for (auto &modelPam : item) {
        if (scheduleConfig_->GetParam().maxIterTimes == 0) {
            std::cout << "The value of ScheduleParam.maxIterTimes can not be 0." << std::endl;
            result = false;
        }
        if ((scheduleConfig_->GetParam().maxPreemptCount != 0) && (modelPam.cpuMemSize == 0)) {
            std::cout << "The value of modelParam.cpuMemSize can not be 0 when maxPreemptCount is not 0." << std::endl;
            result = false;
        }
        if (maxPrefillTokens < modelPam.maxInputTokenLen && scheduleConfig_->GetParam().templateType != "Mix") {
            std::cout << "The value of maxPrefillTokens should not be less than maxInputTokenLen." << std::endl;
            result = false;
        }
    }

    if (this->IslayerwiseDisaggregated()) {
        result = this->CheckLayerwiseDisaggregatedParam();
    }

    hasChecked_ = true;
    return result;
}

std::string ConfigManager::Impl::GetConfigJsonStr()
{
    if (!hasChecked_) {
        std::cout << "Config not be checked, failed to get config json" << std::endl;
        return "{}";
    }
    return configJson_.dump();
}
void ConfigManager::Impl::UpdateConfig()
{
    struct stat statBuf{};
    std::string filePath{jsonPath_};
    constexpr int CHECK_INTERVAL_SECONDS = 10;
   
    auto ret = stat(filePath.c_str(), &statBuf);
    while (ret != 0 && running_) {
        std::this_thread::sleep_for(std::chrono::seconds(CHECK_INTERVAL_SECONDS));
        ret = stat(filePath.c_str(), &statBuf);
    }
    
    auto lastModified = statBuf.st_mtim;
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(CHECK_INTERVAL_SECONDS));
        ret = stat(filePath.c_str(), &statBuf);
        if (ret == 0) {
            auto currModified = statBuf.st_mtim;
            if (currModified.tv_sec != lastModified.tv_sec || currModified.tv_nsec != lastModified.tv_nsec) {
                serverConfig_->UpdateConfig();
                lastModified = currModified;
            }
        }
    }
}

void ConfigManager::Impl::ExecuteConfigInteractions()
{
    try {
        // 获取模型部署配置
        auto modelDeployConfigs = modelDeployConfig_->GetParam();
        
        // 执行插件状态检查并更新pluginEnabled字段
        ConfigInteraction::UpdatePluginEnabledStatus(modelDeployConfigs, *serverConfig_);
        ConfigInteraction::UpdateDeepseekEnabledStatus(modelDeployConfigs, *serverConfig_);
        
        std::cout << "[ConfigManager::ExecuteConfigInteractions] Configuration interactions completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ConfigManager::ExecuteConfigInteractions] Failed: " << e.what() << std::endl;
    }
}

const ServerConfig &GetServerConfig()
{
    return ConfigManager::GetInstance().GetServerConfig();
}

const std::vector<ModelDeployConfig> &GetModelDeployConfig()
{
    return ConfigManager::GetInstance().GetModelDeployConfig();
}

const BackendConfig &GetBackendConfig()
{
    return ConfigManager::GetInstance().GetBackendConfig();
}

const ScheduleConfig &GetScheduleConfig()
{
    return ConfigManager::GetInstance().GetScheduleConfig();
}

const std::vector<LoraConfig> &GetLoraConfig()
{
    return ConfigManager::GetInstance().GetLoraConfig();
}

const RanktableParam &GetRanktableParam()
{
    return ConfigManager::GetInstance().GetRanktableParam();
}

} // namespace mindie_llm