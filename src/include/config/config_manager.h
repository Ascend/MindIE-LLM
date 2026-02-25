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
 
#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <memory>
#include <mutex>
#include "config_info.h"
#include "log_config.h"

#include "nlohmann/json.hpp"

namespace mindie_llm {

class ConfigManager {
public:
    ConfigManager() = delete;
    ConfigManager(const ConfigManager &) = delete;
    ConfigManager &operator=(const ConfigManager &) = delete;

    static bool CreateInstance(std::string jsonPath = "");
    static ConfigManager &GetInstance();

    [[nodiscard]] const BackendConfig &GetBackendConfig() const;
    [[nodiscard]] const LogConfig &GetLogConfig() const;
    [[nodiscard]] const ServerConfig &GetServerConfig() const;
    [[nodiscard]] const std::vector<ModelDeployConfig> &GetModelDeployConfig() const;
    [[nodiscard]] const std::vector<LoraConfig> &GetLoraConfig() const;
    [[nodiscard]] const ScheduleConfig &GetScheduleConfig() const;
    [[nodiscard]] const RanktableParam &GetRanktableParam() const;

    [[nodiscard]] bool CheckAllParam();
    bool CheckAndInitLogParam();

    void SetBlockNum(uint32_t cpuBlockNum, uint32_t npuBlockNum);
    void SetMaxPositionEmbeddings(uint32_t maxPositionEmbeddings);

    // dump超时参数动态调整相关接口
    void SetTokenTimeout(uint64_t tokenTimeout);
    void SetE2eTimeout(uint64_t e2eTimeout);

    std::string GetConfigJsonStr();
    bool IsMultiNodeInfer() const;
    bool IslayerwiseDisaggregated() const;
    bool IsLwdMultiNodesEnable() const;
    std::string GetLwdRoleType() const;

private:
    explicit ConfigManager(const std::string &jsonPath);
    ~ConfigManager();            // 无需手动管理，智能指针会自动释放
    class Impl;                  // 前向声明实现类
    std::unique_ptr<Impl> impl_; // PImpl 指针
};
} // namespace mindie_llm
#endif