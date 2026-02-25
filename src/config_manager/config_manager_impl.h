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

#pragma once
#include <thread>
#include "cmath"
#include "common_util.h"
#include "base_config_manager.h"
#include "env_util.h"
#include "config_manager.h"
#include "config_interaction.h"

namespace mindie_llm {

const ServerConfig &GetServerConfig();
const std::vector<ModelDeployConfig> &GetModelDeployConfig();
const BackendConfig &GetBackendConfig();
const ScheduleConfig &GetScheduleConfig();
const std::vector<LoraConfig> &GetLoraConfig();
const RanktableParam &GetRanktableParam();

class ConfigManager::Impl {
public:
    Impl() = delete;

    explicit Impl(const std::string &jsonPath);

    Impl(const Impl &) = delete;

    Impl operator=(const Impl &) = delete;

    ~Impl()
    {
        running_ = false;
        if (task_.joinable()) {
            task_.join();
        }
    }

    [[nodiscard]] const BackendConfig &GetBackendConfig() const
    {
        return std::atomic_load(&backendConfig_)->GetParam();
    }

    [[nodiscard]] const LogConfig &GetLogConfig() const { return std::atomic_load(&logConfig_)->GetParam(); }

    [[nodiscard]] const ServerConfig &GetServerConfig() const { return std::atomic_load(&serverConfig_)->GetParam(); }

    [[nodiscard]] const std::vector<ModelDeployConfig> &GetModelDeployConfig() const
    {
        return std::atomic_load(&modelDeployConfig_)->GetParam();
    }

    [[nodiscard]] const std::vector<LoraConfig> &GetLoraConfig() const
    {
        return std::atomic_load(&modelDeployConfig_)->GetLoraConfig();
    }

    [[nodiscard]] const ScheduleConfig &GetScheduleConfig() const
    {
        return std::atomic_load(&scheduleConfig_)->GetParam();
    }

    [[nodiscard]] const RanktableParam &GetRanktableParam() const
    {
        return std::atomic_load(&ranktableConfig_)->GetParam();
    }

    [[nodiscard]] bool CheckAllParam();

    bool CheckAndInitLogParam();

    static bool CreateInstance(std::string jsonPath = "");

    static ConfigManager::Impl &GetInstance();

    void SetBlockNum(uint32_t cpuBlockNum, uint32_t npuBlockNum)
    {
        scheduleConfig_->SetBlockNum(cpuBlockNum, npuBlockNum);
    }

    void SetMaxPositionEmbeddings(uint32_t maxPositionEmbeddings)
    {
        if (maxPositionEmbeddings > modelDeployConfig_->GetParam()[0].maxPositionEmbeddings) {
            modelDeployConfig_->SetMaxPositionEmbeddings(maxPositionEmbeddings);
        }
    }

    // dump超时参数动态调整相关接口
    void SetTokenTimeout(uint64_t tokenTimeout)
    {
        serverConfig_->SetTokenTimeout(tokenTimeout);
    }

    void SetE2eTimeout(uint64_t e2eTimeout)
    {
        serverConfig_->SetE2eTimeout(e2eTimeout);
    }

    std::string GetConfigJsonStr();

    // 是否多机推理
    bool IsMultiNodeInfer() const;

    bool IslayerwiseDisaggregated() const;
    bool IsLwdMultiNodesEnable() const;
    std::string GetLwdRoleType() const;

    void UpdateConfig();

    // 配置交互方法
    void ExecuteConfigInteractions();

private:
    [[nodiscard]] bool PreCheck() const;

    [[nodiscard]] bool CheckLayerwiseDisaggregatedParam();
    bool InitConfigManager();
    bool initialized_ = false;
    std::shared_ptr<ScheduleConfigManager> scheduleConfig_{nullptr};
    std::shared_ptr<LogConfigManager> logConfig_{nullptr};
    std::shared_ptr<ServerConfigManager> serverConfig_{nullptr};
    std::shared_ptr<BackendConfigManager> backendConfig_{nullptr};
    std::shared_ptr<ModelDeployConfigManager> modelDeployConfig_{nullptr};
    std::shared_ptr<RanktableConfigManager> ranktableConfig_;
    bool hasChecked_{false};
    std::string jsonPath_;
    nlohmann::json configJson_{};
    std::atomic<bool> running_{true};
    std::thread task_;
};
} // namespace mindie_llm