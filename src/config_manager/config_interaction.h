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

#ifndef CONFIG_INTERACTION_H
#define CONFIG_INTERACTION_H

#include <string>
#include <vector>
#include <regex>

#include "config_info.h"
#include "base_config_manager.h"

namespace mindie_llm {

/**
 * 简单的配置交互类
 * 直接使用ConfigManager的配置内容，实现配置间的数据传递
 */
class ConfigInteraction {
public:
    ConfigInteraction() = default;
    ~ConfigInteraction() = default;

    /**
     * 仅检查插件参数是否启用（不修改serverConfig）
     */
    static bool CheckPluginEnabled(const std::vector<ModelDeployConfig> &modelDeployConfigs);

    /**
     * 检查各模型的config.json中model_type是否包含deepseek前缀
     */
    static bool CheckModelTypeDeepseek(const std::vector<ModelDeployConfig> &modelDeployConfigs);

    /**
     * 检查插件状态并更新pluginEnabled字段
     * @param modelDeployConfigs 模型部署配置列表
     * @param serverConfigManager 服务器配置管理器（将被修改）
     * @return 是否成功执行
     */
    static bool UpdatePluginEnabledStatus(const std::vector<ModelDeployConfig> &modelDeployConfigs,
        ServerConfigManager &serverConfigManager);

    static bool UpdateDeepseekEnabledStatus(const std::vector<ModelDeployConfig> &modelDeployConfigs,
        ServerConfigManager &serverConfigManager);

private:
    /**
     * 检查插件参数中是否包含指定插件
     * @param pluginParams 插件参数字符串
     * @return 是否包含插件
     */
    static bool HasPluginEnabled(const std::string &pluginParams);

    // 不支持的插件类型
    static const std::vector<std::string> UNSUPPORTED_PLUGINS;
    
    // 组合正则表达式模式
    static std::string GetCombinedRegexPattern();
};

} // namespace mindie_llm

#endif // CONFIG_INTERACTION_H
