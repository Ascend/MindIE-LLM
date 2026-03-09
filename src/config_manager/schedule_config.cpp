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

#include "base_config_manager.h"
#include "common_util.h"
#include "env_util.h"
#include "log.h"

using Json = nlohmann::json;
using namespace nlohmann::literals;

namespace mindie_llm {
const uint32_t MAX_INPUT_LEN = 1024 * 1024 * 4; // 4M
static std::vector<ParamSpec> g_scheduleParamsConstraint = {{"templateName", "string", true},
                                                            {"templateType", "string", true},
                                                            {"cacheBlockSize", "uint32_t", true},
                                                            {"decodePolicyType", "uint32_t", true},
                                                            {"decodeTimeMsPerReq", "uint32_t", true},
                                                            {"maxBatchSize", "uint32_t", true},
                                                            {"maxPreemptCount", "uint32_t", true},
                                                            {"maxPrefillTokens", "uint32_t", true},
                                                            {"maxIterTimes", "uint32_t", true},
                                                            {"maxPrefillBatchSize", "uint32_t", true},
                                                            {"maxQueueDelayMicroseconds", "uint32_t", true},
                                                            {"prefillPolicyType", "uint32_t", true},
                                                            {"prefillTimeMsPerReq", "uint32_t", true},
                                                            {"supportSelectBatch", "bool", true},
                                                            {"stageSelectPolicy", "uint32_t", false},
                                                            {"dynamicBatchSizeEnable", "bool", false},
                                                            {"enablePrefixCache", "bool", false},
                                                            {"enableSplit", "bool", false},
                                                            {"enablChunkedPrefill", "bool", false},
                                                            {"policyType", "uint32_t", false},
                                                            {"splitType", "bool", false},
                                                            {"splitStartType", "bool", false},
                                                            {"splitChunkTokens", "uint32_t", false},
                                                            {"splitStartBatchSize", "uint32_t", false},
                                                            {"prefillExpectedTime", "uint32_t", false},
                                                            {"decodeExpectedTime", "uint32_t", false},
                                                            {"bufferResponseEnabled", "bool", false},
                                                            {"distributedEnable", "bool", false},
                                                            {"maxFirstTokenWaitTime", "uint32_t", false},
                                                            {"layerwiseDisaggregated", "object", false}};

static std::vector<ParamSpec> g_scheduleLwdParamsConstraint = {{"nextPHeadPrior", "bool", false}};

bool CheckSystemJson(Json &backendJsonData, const std::string &jsonPath,
                     std::vector<ParamSpec> &scheduleParamsConstraint)
{
    if (!CheckSystemConfig(jsonPath, backendJsonData, "BackendConfig")) {
        return false;
    }

    if (!ParamChecker::CheckJsonParamType(backendJsonData["ScheduleConfig"], scheduleParamsConstraint)) {
        return false;
    }
    return true;
}

bool ScheduleConfigManager::LoadBasicScheduleConfig(Json &scheduleJsonData)
{
    scheduleConfig_.templateType = scheduleJsonData["templateType"];
    scheduleConfig_.templateName = scheduleJsonData["templateName"];
    scheduleConfig_.cacheBlockSize = scheduleJsonData["cacheBlockSize"];
    scheduleConfig_.decodePolicyType = scheduleJsonData["decodePolicyType"];
    scheduleConfig_.decodeTimeMsPerReq = scheduleJsonData["decodeTimeMsPerReq"];
    scheduleConfig_.maxBatchSize = scheduleJsonData["maxBatchSize"];
    scheduleConfig_.maxPreemptCount = scheduleJsonData["maxPreemptCount"];
    scheduleConfig_.maxPrefillTokens = scheduleJsonData["maxPrefillTokens"];
    scheduleConfig_.maxIterTimes = scheduleJsonData["maxIterTimes"];
    scheduleConfig_.maxPrefillBatchSize = (scheduleJsonData["maxPrefillBatchSize"] > 0)
                                            ? scheduleJsonData["maxPrefillBatchSize"]
                                            : scheduleJsonData["maxBatchSize"];
    scheduleConfig_.maxQueueDelayMicroseconds = scheduleJsonData["maxQueueDelayMicroseconds"];
    scheduleConfig_.prefillPolicyType = scheduleJsonData["prefillPolicyType"];
    scheduleConfig_.prefillTimeMsPerReq = scheduleJsonData["prefillTimeMsPerReq"];
    scheduleConfig_.supportSelectBatch = scheduleJsonData["supportSelectBatch"];
    if (scheduleJsonData.contains("maxFirstTokenWaitTime")) {
        scheduleConfig_.maxFirstTokenWaitTime = scheduleJsonData["maxFirstTokenWaitTime"];
    }
    return true;
}

bool ScheduleConfigManager::LoadLwdConfig(Json &scheduleJsonData)
{
    if (!scheduleJsonData.contains("layerwiseDisaggregated")) {
        return true;
    }

    Json lwdData = scheduleJsonData["layerwiseDisaggregated"];
    if (!ParamChecker::CheckJsonParamType(lwdData, g_scheduleLwdParamsConstraint)) {
        return false;
    }
    
    if (lwdData.contains("nextPHeadPrior")) {
        scheduleConfig_.lwdNextPHeadPrior = lwdData["nextPHeadPrior"];
    }
    return true;
}

bool ScheduleConfigManager::InitFromJson()
{
    Json backendJsonData;
    if (!CheckSystemJson(backendJsonData, jsonPath_, g_scheduleParamsConstraint)) {
        return false;
    }

    Json scheduleJsonData = backendJsonData["ScheduleConfig"];

    if (!LoadBasicScheduleConfig(scheduleJsonData) || !LoadLwdConfig(scheduleJsonData)) {
        return false;
    }

    LoadPolicyConfig(scheduleJsonData);
    LoadSplitFuseConfig(scheduleJsonData);
    LoadChunkedPrefillConfig(scheduleJsonData);
    LoadPrefixCacheConfig(scheduleJsonData);
    LoadMiscConfig(scheduleJsonData);
    LoadDynamicBatchConfig(scheduleJsonData);

    return true;
}

void ScheduleConfigManager::LoadPolicyConfig(Json &scheduleJsonData)
{
    if (scheduleJsonData.contains("policyType")) {
        scheduleConfig_.policyType = scheduleJsonData["policyType"];
    }
}

void ScheduleConfigManager::LoadSplitFuseConfig(Json &scheduleJsonData)
{
    if (scheduleJsonData.contains("enableSplit")) {
        MINDIE_LLM_LOG_WARN("To enable the splitfuse, you only need to configure the 'plugin_params'."
                            << " 'enableSplit' parameter no longer needs to be configured,"
                            << " and any value set for it will not take effect.");
    }

    if (scheduleJsonData.contains("splitType")) {
        scheduleConfig_.splitType = scheduleJsonData["splitType"];
    }

    if (scheduleJsonData.contains("splitStartType")) {
        scheduleConfig_.splitStartType = scheduleJsonData["splitStartType"];
    }

    if (scheduleJsonData.contains("splitChunkTokens")) {
        scheduleConfig_.splitChunkTokens = scheduleJsonData["splitChunkTokens"];
    }

    if (scheduleJsonData.contains("splitStartBatchSize")) {
        scheduleConfig_.splitStartBatchSize = scheduleJsonData["splitStartBatchSize"];
    }
}

void ScheduleConfigManager::LoadChunkedPrefillConfig(Json &scheduleJsonData)
{
    if (scheduleJsonData.contains("prefillChunkSize")) {
        scheduleConfig_.prefillChunkSize = scheduleJsonData["prefillChunkSize"];
    }

    if (scheduleJsonData.contains("maxNumPartialPrefills")) {
        scheduleConfig_.maxNumPartialPrefills = scheduleJsonData["maxNumPartialPrefills"];
    }

    if (scheduleJsonData.contains("maxLongPartialPrefills")) {
        scheduleConfig_.maxLongPartialPrefills = scheduleJsonData["maxLongPartialPrefills"];
    }

    if (scheduleJsonData.contains("longPrefillTokenThreshold")) {
        scheduleConfig_.longPrefillTokenThreshold = scheduleJsonData["longPrefillTokenThreshold"];
    }
}

void ScheduleConfigManager::LoadPrefixCacheConfig(Json &scheduleJsonData) const
{
    if (scheduleJsonData.contains("enablePrefixCache")) {
        MINDIE_LLM_LOG_WARN("To enable the prefixcache, you only need to configure the 'plugin_params'."
                            << " 'enablePrefixCache' parameter no longer needs to be configured,"
                            << " and any value set for it will not take effect.");
    }
}

void ScheduleConfigManager::LoadMiscConfig(Json &scheduleJsonData)
{
    if (scheduleJsonData.contains("bufferResponseEnabled")) {
        scheduleConfig_.bufferResponseEnabled = scheduleJsonData["bufferResponseEnabled"];
    }
    if (scheduleJsonData.contains("decodeExpectedTime")) {
        scheduleConfig_.decodeExpectedTime = scheduleJsonData["decodeExpectedTime"];
    }
    if (scheduleJsonData.contains("prefillExpectedTime")) {
        scheduleConfig_.prefillExpectedTime = scheduleJsonData["prefillExpectedTime"];
    }
    if (scheduleJsonData.contains("distributedEnable")) {
        scheduleConfig_.distributedEnable = scheduleJsonData["distributedEnable"];
    }
}

void ScheduleConfigManager::LoadDynamicBatchConfig(Json &scheduleJsonData)
{
    if (scheduleJsonData.contains("stageSelectPolicy")) {
        scheduleConfig_.stageSelectPolicy = scheduleJsonData["stageSelectPolicy"];
    }
    if (scheduleConfig_.supportSelectBatch) {
        scheduleConfig_.stageSelectPolicy = 1;
    }
    if (scheduleJsonData.contains("dynamicBatchSizeEnable")) {
        scheduleConfig_.dynamicBatchSizeEnable = scheduleJsonData["dynamicBatchSizeEnable"];
    }
}

void ScheduleConfigManager::CheckSLOParam(bool &checkRes)
{
    const size_t MAX_SELECT_POLICY = 2U;
    const size_t MAX_EXPECT_TIME = 10000U;
    const size_t MIN_EXPECT_TIME = 1U;
    Json backendJsonData;
    if (!CheckSystemJson(backendJsonData, jsonPath_, g_scheduleParamsConstraint)) {
        return;
    }
    Json scheduleJsonData = backendJsonData["ScheduleConfig"];
    if (scheduleJsonData.contains("stageSelectPolicy")) {
        checkRes =
            checkRes && ParamChecker::CheckMaxMinValue<uint16_t>(scheduleConfig_.stageSelectPolicy, MAX_SELECT_POLICY,
                                                                 0U, "scheduleParam.stageSelectPolicy");
    }
    if (scheduleJsonData.contains("prefillExpectedTime")) {
        checkRes =
            checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.prefillExpectedTime, MAX_EXPECT_TIME,
                                                                 MIN_EXPECT_TIME, "scheduleParam.prefillExpectedTime");
    }
    if (scheduleJsonData.contains("decodeExpectedTime")) {
        checkRes =
            checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.decodeExpectedTime, MAX_EXPECT_TIME,
                                                                 MIN_EXPECT_TIME, "scheduleParam.decodeExpectedTime");
    }
}

bool ScheduleConfigManager::CheckParam()
{
    // 对所有参数进行校验，即使中途失败仍继续执行
    bool checkRes = true;
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.cacheBlockSize, 128U, 1U,
                                                                    "scheduleParam.cacheBlockSize");
    checkRes =
        checkRes && ParamChecker::CheckPolicyValue(scheduleConfig_.decodePolicyType, "scheduleParam.decodePolicyType");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.decodeTimeMsPerReq, 1000U, 0U,
                                                                    "scheduleParam.decodeTimeMsPerReq");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxBatchSize, 819200U, 1U,
                                                                    "scheduleParam.maxBatchSize");
    if (scheduleConfig_.maxPreemptCount > scheduleConfig_.maxBatchSize) {
        MINDIE_LLM_LOG_ERROR("The maxPreemptCount cannot be set greater than maxBatchsize"
                             << ", please set 'maxPreemptCount' to be not greater than 'maxBatchsize' in config.json.");
        checkRes = checkRes && false;
    }
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxPrefillTokens, MAX_INPUT_LEN, 1U,
                                                                    "scheduleParam.maxPrefillTokens");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxPrefillBatchSize,
                                                                    scheduleConfig_.maxBatchSize, 1U,
                                                                    "scheduleParam.maxPrefillBatchSize");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxQueueDelayMicroseconds, 1000000U,
                                                                    500U, "scheduleParam.maxQueueDelayMicroseconds");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxFirstTokenWaitTime, 3600000U,
                                                                    0U, "scheduleParam.maxFirstTokenWaitTime");
    checkRes = checkRes &&
               ParamChecker::CheckPolicyValue(scheduleConfig_.prefillPolicyType, "scheduleParam.prefillPolicyType");
    checkRes = checkRes && ParamChecker::CheckMixPolicyValue(scheduleConfig_.policyType, "scheduleParam.policyType");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.prefillTimeMsPerReq, 1000U, 0U,
                                                                    "scheduleParam.prefillTimeMsPerReq");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxNumPartialPrefills, 64U, 1U,
                                                                    "scheduleParam.maxNumPartialPrefills");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.maxLongPartialPrefills,
                                                                    scheduleConfig_.maxNumPartialPrefills, 1U,
                                                                    "scheduleParam.maxLongPartialPrefills");
    checkRes = checkRes && ParamChecker::CheckMaxMinValue<uint32_t>(scheduleConfig_.longPrefillTokenThreshold, 8192U,
                                                                    1U, "scheduleParam.longPrefillTokenThreshold");
    if (scheduleConfig_.templateType != "Standard" && scheduleConfig_.templateType != "Mix") {
        MINDIE_LLM_LOG_ERROR("The templateType must be Standard or Mix, but is " << scheduleConfig_.templateType);
        checkRes = checkRes && false;
    }
    if (scheduleConfig_.templateName != "Standard_LLM") {
        MINDIE_LLM_LOG_ERROR("The templateName must be Standard_LLM, but is " << scheduleConfig_.templateName);
        checkRes = checkRes && false;
    }
    CheckSLOParam(checkRes);
    return checkRes;
}

void ScheduleConfigManager::SetMaxPreemptCount(uint32_t value) { scheduleConfig_.maxPreemptCount = value; }

const struct ScheduleConfig &ScheduleConfigManager::GetParam() { return scheduleConfig_; }
} // namespace mindie_llm
