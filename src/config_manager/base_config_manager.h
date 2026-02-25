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

#ifndef BASE_CONFIG_H
#define BASE_CONFIG_H
#include <map>
#include <fstream>
#include "log_config.h"
#include "config_info.h"
#include "nlohmann/json.hpp"
#include "param_checker.h"

using Json = nlohmann::json;

namespace mindie_llm {
class BaseConfig {
public:
    explicit BaseConfig(std::string jsonPath) : jsonPath_(std::move(jsonPath)) {}

    virtual ~BaseConfig() = default;

    virtual bool InitFromJson() = 0;

    virtual bool CheckParam() = 0;

    friend class ParamChecker;

    static bool CheckSystemConfig(const std::string &jsonPath, Json &inputJsonData, std::string paramType);

protected:
    std::string jsonPath_;
};

class ScheduleConfigManager : public BaseConfig {
public:
    explicit ScheduleConfigManager(std::string jsonPath) : BaseConfig(std::move(jsonPath)){};

    ~ScheduleConfigManager() override = default;

    bool InitFromJson() override;

    bool CheckParam() override;

    void SetMaxPreemptCount(uint32_t value);

    const struct ScheduleConfig &GetParam();

    void SetBlockNum(uint32_t cpuBlockNum, uint32_t npuBlockNum);

    friend class ParamChecker;

private:
    struct ScheduleConfig scheduleConfig_;

    bool LoadLwdConfig(Json &scheduleJsonData);

    bool LoadBasicScheduleConfig(Json &scheduleJsonData);

    void LoadPolicyConfig(Json &scheduleJsonData);

    void LoadSplitFuseConfig(Json &scheduleJsonData);

    void LoadChunkedPrefillConfig(Json &scheduleJsonData);

    void LoadPrefixCacheConfig(Json &scheduleJsonData) const;

    void LoadMiscConfig(Json &scheduleJsonData);

    void LoadDynamicBatchConfig(Json &scheduleJsonData);

    void CheckSLOParam(bool &checkRes);
};

class LogConfigManager : public BaseConfig {
public:
    explicit LogConfigManager(std::string jsonPath) : BaseConfig(std::move(jsonPath)){};

    ~LogConfigManager() override = default;

    bool InitFromJson() override { return true; };

    bool CheckParam() override { return true; };

    const LogConfig &GetParam() { return logConfig_; };

private:
    LogConfig logConfig_;
    bool initFlag = true;
};

class ServerConfigManager : public BaseConfig {
public:
    explicit ServerConfigManager(std::string jsonPath) : BaseConfig(std::move(jsonPath)){};

    ~ServerConfigManager() override = default;

    bool InitFromJson() override;

    bool CheckParam() override;

    const struct ServerConfig &GetParam();
    
    void SetPluginEnabled(bool enabled);

    void SetDeepseekEnabled(bool enabled);

    // dump超时参数动态调整相关接口
    void SetTokenTimeout(uint64_t tokenTimeout);
    
    void SetE2eTimeout(uint64_t e2eTimeout);

    bool GetDecodeStatus() const;

    void UpdateConfig();

private:
    bool jsonDecodeSuccess_ = true;

    bool CheckHttpsConfig(bool loadManagementSSL);

    bool CheckBusinessHttpsParam();

    bool CheckManagementHttpsParam();

    bool CheckLayerwiseDisaggregatedConfig();

    bool CheckMetricsHttpsParam(std::string& homePath);

    void InitHttpsBusinessConfigFromJson(Json &serveJsonData);

    void InitDMIHttpsConfigFromJson(Json &serveJsonData);

    void InitHttpsManagementConfigFromJson(Json &serveJsonData);

    void InitHttpsMetricsConfigFromJson(Json &serveJsonData);

    void InitHttpsConfigFromJson(Json &serveJsonData, bool loadManagementSSL);

    void InitGrpcTLSConfigFromJson(Json &serveJsonData);

    void InitLayerwiseDisaggregatedConfigFromJson(Json &serveJsonData);

    void LoadOptionalParameters(Json& serverParamsJsonData);
    
    std::string GetIPAddress(Json &serveJsonData);

    std::string GetManagementIPAddress(Json &serveJsonData);

    struct ServerConfig serverConfig_;

    bool initFlag = true;
};

class BackendConfigManager : public BaseConfig {
public:
    explicit BackendConfigManager(std::string jsonPath) : BaseConfig(std::move(jsonPath)){};

    ~BackendConfigManager() override = default;

    bool InitFromJson() override;

    bool CheckParam() override;

    bool CheckBackendInterTlsParam();

    const struct BackendConfig &GetParam();

    void UpdateMultiNodesInfer(const RanktableParam &ranktableParam);

private:
    bool InitTlsConfigFromJson(Json &backendConfigData);

    void InitKvPoolConfigFromJson(Json &backendConfigData);

    void InitLwdConfigFromJson(Json &backendConfigData);

    bool CheckInterTlsParam();

    struct BackendConfig backendConfig_;

    bool initFlag = true;
};

class ModelDeployConfigManager : public BaseConfig {
public:
    explicit ModelDeployConfigManager(std::string jsonPath) : BaseConfig(std::move(jsonPath)){};

    ~ModelDeployConfigManager() override = default;

    bool InitFromJson() override;

    bool CheckParam() override;

    void CheckTemplateConfig(const std::string &templateType, const uint32_t modelConfigNum);

    std::vector<ModelDeployConfig> &GetParam();
    std::vector<LoraConfig> &GetLoraConfig();

    void SetMaxPositionEmbeddings(uint32_t maxPositionEmbeddings);
    
    void InitModelConfigImpl(const Json &modelJsonData, uint32_t speculationGamma,
        const uint32_t maxSeqLength, bool truncation);
    
    void InitLoraConfigImpl(const Json &modelJsonData); // lora 初始化

    void InitModelConfig(const Json &modelJsonData, uint32_t speculationGamma, const uint32_t maxSeqLength,
        const bool truncation);

    friend class ParamChecker;

private:
    void InitGrpcTLSConfigFromJson(Json &serveJsonData);
    bool initFlag = true;
    std::vector<ModelDeployConfig> modelParamVec_;
    std::vector<LoraConfig> loraModules_;
    struct ModelDeployConfig modelDeployConfig_;
};

class RanktableConfigManager {
public:
    explicit RanktableConfigManager();

    ~RanktableConfigManager() = default;

    static std::string GetContainerIPAddress();

    static std::string GetHostIPAddress();

    bool InitFromJson();

    bool CheckDeviceId(const std::string &deviceIdStr) const;

    bool CheckDeviceIp(const std::string &deviceIpStr) const;

    bool CheckRankId(const std::string &rankIdStr) const;

    bool CheckParam();

    const struct RanktableParam &GetParam();

private:
    bool ReadRanktableData(uint32_t &serverCount, nlohmann::json &serverListData);

    uint32_t FillServerEle(const std::string &containerIP, const std::string &hostIP, Json &serverEleData,
                           struct ServerEle &serverEle);

    std::string ranktablePath_ = "";

    struct RanktableParam ranktableParam_;

    bool initFlag = true;
};
} // namespace mindie_llm
#endif
