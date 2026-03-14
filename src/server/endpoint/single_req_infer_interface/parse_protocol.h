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

#ifndef OCK_ENDPOINT_JSON_PARSE_H
#define OCK_ENDPOINT_JSON_PARSE_H

#include <string>
#include <utility>
#include "log.h"
#include "config_manager.h"
#include "endpoint_def.h"
#include "http_metrics.h"
#include "health_checker/health_checker.h"

namespace mindie_llm {
#define GET_PARA_DAFAULT(expr, target, inputValue, defValue) \
    if ((expr)) {                                            \
        target = inputValue;                                 \
    } else {                                                 \
        target = defValue;                                   \
    }

struct OptionalItemResult { // return strcuture for CheckOptionalItemType
    bool isPresent;   // whether the key is present in JSON object and it's not null
    bool isCorrectType; // whether its value corresponds to given type
};

class JsonParse {
public:
    // get response encode
    static void EncodeTritonModel(ModelDeployConfig &modelParam, std::string &jsonStr);
    static void EncodeTritonModelConfig(ModelDeployConfig &modelParam, std::string &jsonStr);
    static void EncodeTritonEngine(const ScheduleConfig &scheduleParam, std::string &jsonStr);
    static void EncodeSlotCount(const ScheduleConfig &scheduleParam, uint64_t freeSlot,
        uint64_t availableTokensLen, std::string &jsonStr);
    static void EncodeAbnormalNodeInfo(const std::map<std::string, NodeHealthStatus> &slavesStatus,
        std::string &jsonStr);
    static void EncodeOpenAiModels(const std::vector<ModelDeployConfig> &modelParam,
                                   const std::vector<LoraConfig> &loraParam, int64_t time,
                                   std::string &jsonStr);
    static void EncodeOpenAiModel(ModelDeployConfig &modelParam, int64_t time, std::string &jsonStr);
    static void EncodeOpenAiModel(LoraConfig &loraParam, int64_t time, std::string &jsonStr);
    static void EncodeHealthStatus(const ServiceStatus &status,
        const std::vector<ErrorItem> &errorList, std::string &jsonStr);
    static void EncodeCmdResult(const Status &status, mindie_llm::RecoverCommandInfo &info, std::string &jsonStr);

    static void HandleGetInfo(const ScheduleConfig &scheduleParam,
        const std::vector<ModelDeployConfig> &modelParam, std::string &infoStr);
    static uint32_t GetInferTypeFromJsonStr(const std::string &jsonStr, uint16_t &inferType);
    static uint32_t DecodeGeneralTGIStreamMode(const std::string &jsonStr, bool &streamMode, std::string &error);
    static uint32_t DecodeFaultRecoveryCmd(const std::string &jsonStr, FaultRecoveryCmd &cmdType, std::string &cmdStr);
    static const std::string &GetDataType(InferDataType type) noexcept;
    static const std::string &GetJsonDataType(nlohmann::ordered_json::value_t type) noexcept;
    static bool JsonContainItemWithType(const nlohmann::ordered_json &jsonObj, const std::string &key,
        nlohmann::ordered_json::value_t type, std::string &error) noexcept;
    static OptionalItemResult CheckOptionalItemType(const nlohmann::ordered_json &jsonObj, const std::string &key,
        nlohmann::ordered_json::value_t type, std::string &error) noexcept;
    static nlohmann::ordered_json PrometheusFormat(std::string name, std::string value,
        std::string managementIpAddress, std::string managementPort) noexcept;
    static bool JsonHttpMetrics(HttpMetrics &httpMetricsInstance,
        std::map<std::string, uint64_t> &batchSchedulerMetrics, std::string &jsonStr) noexcept;
    static TokenizerContents ParseContentFromJson(const std::string& jsonStr);

    static bool CheckPDRoleReqJson(const nlohmann::ordered_json &jsonObj);
    static bool CheckPDIPInfo(const nlohmann::ordered_json &jsonBody);
    static bool CheckPDDeviceIPInfo(const nlohmann::ordered_json &jsonBody);
    static bool CheckIDAndHostIP(const nlohmann::ordered_json &jsonBody);
    static bool GetContextJsonBody(const ReqCtxPtr &ctx, nlohmann::ordered_json &body);
    static bool CheckPDRoleV2ReqJson(const nlohmann::ordered_json &jsonBody);
    static bool CheckPDV2IPInfo(const nlohmann::ordered_json &jsonBody);
    static bool CheckPDV2DeviceIPInfo(const nlohmann::ordered_json &jsonBody);
    static bool CheckHostIP(const nlohmann::ordered_json &jsonBody);
};
} // namespace mindie_llm
#endif // OCK_ENDPOINT_JSON_PARSE_H
