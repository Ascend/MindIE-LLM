/*
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
#include "config_interaction.h"
#include "base_config_manager.h"

#include <iostream>

#include <algorithm>

namespace mindie_llm {

// 不支持的插件类型
const std::vector<std::string> ConfigInteraction::UNSUPPORTED_PLUGINS = {
    "mtp",
    "la",
    "memory_decoding"
};

bool ConfigInteraction::UpdatePluginEnabledStatus(const std::vector<ModelDeployConfig> &modelDeployConfigs,
    ServerConfigManager &serverConfigManager)
{
    // 获取当前pluginEnabled状态
    bool originalPluginEnabled = serverConfigManager.GetParam().pluginEnabled;
    bool newPluginEnabled = false;

    if (modelDeployConfigs.empty()) {
        newPluginEnabled = false;
    } else {
        bool pluginDetected = CheckPluginEnabled(modelDeployConfigs);
        newPluginEnabled = pluginDetected;
    }
    
    // 只有当pluginEnabled状态发生变化时才更新配置
    if (originalPluginEnabled != newPluginEnabled) {
        serverConfigManager.SetPluginEnabled(newPluginEnabled);
    }
    return true;
}

bool ConfigInteraction::UpdateDeepseekEnabledStatus(const std::vector<ModelDeployConfig> &modelDeployConfigs,
    ServerConfigManager &serverConfigManager)
{
    // 获取当前deepseekEnabled状态
    bool originalDeepseekEnabled = serverConfigManager.GetParam().deepseekEnabled;
    bool newDeepseekEnabled = false;

    if (modelDeployConfigs.empty()) {
        newDeepseekEnabled = false;
    } else {
        // 综合检查：插件参数 或 模型名/类型（deepseek 前缀）
        bool deepseekDetected = CheckModelTypeDeepseek(modelDeployConfigs);
        newDeepseekEnabled = deepseekDetected;
    }
    
    // 只有当deepseekEnabled状态发生变化时才更新配置
    if (originalDeepseekEnabled != newDeepseekEnabled) {
        serverConfigManager.SetDeepseekEnabled(newDeepseekEnabled);
    }
    return true;
}

bool ConfigInteraction::CheckPluginEnabled(const std::vector<ModelDeployConfig> &modelDeployConfigs)
{
    if (modelDeployConfigs.empty()) {
        return false;
    }
    
    for (size_t i = 0; i < modelDeployConfigs.size(); ++i) {
        const auto& model = modelDeployConfigs[i];
        
        auto it = model.modelConfig.find("plugin_params");
        if (it != model.modelConfig.end()) {
            const std::string &pluginParams = it->second;
            if (HasPluginEnabled(pluginParams)) {
                return true;
            }
        }
    }
    return false;
}

bool ConfigInteraction::CheckModelTypeDeepseek(const std::vector<ModelDeployConfig> &modelDeployConfigs)
{
    for (size_t i = 0; i < modelDeployConfigs.size(); ++i) {
        const auto &model = modelDeployConfigs[i];
        std::string modelType = model.modelType;
        if (modelType.empty()) {
            continue;
        }
        
        std::string lowerModelType = modelType;
        std::transform(lowerModelType.begin(), lowerModelType.end(), lowerModelType.begin(), ::tolower);
        
        if (lowerModelType.rfind("deepseek", 0) == 0) {
            return true;
        }
    }
    return false;
}

bool ConfigInteraction::HasPluginEnabled(const std::string &pluginParams)
{
    if (pluginParams.empty()) {
        return false;
    }

    std::string pattern = GetCombinedRegexPattern();
    
    std::regex pluginRegex(pattern);
    bool result = std::regex_search(pluginParams, pluginRegex);
    return result;
}

std::string ConfigInteraction::GetCombinedRegexPattern()
{
    static std::string combinedPattern;
    if (combinedPattern.empty()) {
        std::string pattern = R"("plugin_type"\s*:\s*"[^"]*(?:)";
        for (size_t i = 0; i < UNSUPPORTED_PLUGINS.size(); ++i) {
            if (i > 0) {
                pattern += "|";
            }
            pattern += UNSUPPORTED_PLUGINS[i];
        }
        pattern += ")";
        combinedPattern = pattern;
    }
    return combinedPattern;
}

} // namespace mindie_llm
