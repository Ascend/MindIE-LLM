/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "endpoint.h"
#include "config_manager.h"
#include "config_manager_impl.h"
#include "config_dynamic_handler.h"
#include "grpc_wrapper.h"
#include "http_wrapper.h"
#include "infer_instances.h"
#include "infer_tokenizer.h"
#include "log.h"
#include "prometheus_metrics.h"
#include "random_generator.h"

using namespace mindie_llm;

int EndPoint::Start(std::unordered_map<std::string, std::string> args)
{
    std::string configFilePath = args["configFilePath"];
    mExpertParallel = (args.find("expertParallel") != args.end() && args["expertParallel"] == "true");
    std::lock_guard<std::mutex> guard(mMutex);
    if (mEngineStarted && mServerStarted && mTokenizerStarted) {
        return 0;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    auto random = RandomGenerator::GetInstance();
    if (random == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                   GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, ABNORMAL_TRANSMISSION_ERROR),
                   "Failed to create random generator");
        ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "fail");
        return -1;
    }

    if (!GetInferInstance()->InitFromEndpointCall(configFilePath).IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "Failed to init infer model instance");
        ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "fail");
        return -1;
    }

    if (StartDynamicConfigHandler()) {
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Start dynamic config handler success.");
    }

    Log::Flush();
    mEngineStarted = true;

    if (PrometheusMetrics::GetInstance() == nullptr) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
                  "Failed to init prometheusMetrics! Please check in the mindie server log!");
    }

    if (StartEndpoint() == -1) {
        return -1;
    }
    PyGILState_Release(gstate);
    return 0;
}

bool EndPoint::StartDynamicConfigHandler() const
{
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    // 注册动态配置调整相关回调
    auto &dynamicConfigHandler = mindie_llm::DynamicConfigHandler::GetInstance();
    dynamicConfigHandler.RegisterCallBackFunction<mindie_llm::ConfigManager>(
        "EnableDynamicAdjustTimeoutConfig",
        &configManager,
        &mindie_llm::ConfigManager::SetTokenTimeout,
        3600); // 最大值 3600s
    dynamicConfigHandler.RegisterCallBackFunction<mindie_llm::ConfigManager>(
        "EnableDynamicAdjustTimeoutConfig",
        &configManager,
        &mindie_llm::ConfigManager::SetE2eTimeout,
        65535); // 最大值 65535s
    dynamicConfigHandler.Start();
    return true;
}

int EndPoint::StartEndpoint()
{
    if (ConfigManager::GetInstance().IslayerwiseDisaggregated() &&
        GetServerConfig().layerwiseDisaggregatedRoleType == "slave") {
        std::cout << "LayerwiseDisaggregated infer slave instance need not init TokenizerProcessPool and HttpWrapper"
            << std::endl;
    } else if (GetServerConfig().distDPServerEnabled || !ConfigManager::GetInstance().IsMultiNodeInfer() ||
        !GetRanktableParam().IsSlave()) {
        if (GetServerConfig().inferMode == INFER_MODE_DMI) {
            if (GrpcWrapper::GetInstance().Start() != 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                    "Failed to start grpc wrapper in DMI scene.");
                ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "fail");
                return -1;
            }
        }
        // init tokenizer
        if (!TokenizerProcessPool::GetInstance().InitTokenizerPool()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                "Init process tokenizer pool failed.");
            ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "fail");
            Log::Flush();
            return -1;
        }
        mTokenizerStarted = true;

        if (StartHealthChecker() != 0) {
            return -1;
        }

        // init http
        if (!HttpWrapper::Instance().Start()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                "Failed to start http wrapper");
            ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "fail");
            return -1;
        }
        mServerStarted = true;
        ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "success");
    } else {
        std::cout << "Multi Nodes infer slave instance need not init TokenizerProcessPool and HttpWrapper"
            << std::endl;
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT,
            "Multi Nodes infer slave instance need not init TokenizerProcessPool and HttpWrapper.");
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Start endpoint success");
    return 0;
}

int EndPoint::StartHealthChecker()
{
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "healthchecker init and start");
    if (!HealthChecker::GetInstance().Start()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
            "Failed to start healtchecker wrapper");
        ULOG_AUDIT("system", MINDIE_SERVER, "Start mindie server", "fail");
        return -1;
    }
    mHealthcheckerStarted = true;
    return 0;
}

HealthChecker& EndPoint::GetHealthcheckerInstance() const
{
    return HealthChecker::GetInstance();
}

void EndPoint::Stop()
{
    std::lock_guard<std::mutex> guard(mMutex);
    if (mEngineStarted) {
        auto backendConfig = GetBackendConfig();
        ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Finalize engine " + backendConfig.backendName);
        if (!GetInferInstance()->Finalize().IsOk()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                       GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, CHECK_ERROR),
                       "Failed to finalize backend " + backendConfig.backendName);
        }
        mEngineStarted = false;
    }

    if (mServerStarted) {
        HttpWrapper::Instance().Stop();
        GrpcWrapper::GetInstance().Stop();
        mServerStarted = false;
    }
    if (mHealthcheckerStarted) {
        HealthChecker::GetInstance().Stop();
        mHealthcheckerStarted = false;
    }
    if (mTokenizerStarted) {
        mTokenizerStarted = false;
    }
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Stop endpoint success");
}
