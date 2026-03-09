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
#include <thread>
#include <string>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "common_util.h"
#include "param_checker.h"
#include "config_dynamic_handler.h"

namespace mindie_llm {

DynamicConfigHandler::~DynamicConfigHandler()
{
    Stop();
}

DynamicConfigHandler& DynamicConfigHandler::GetInstance()
{
    static DynamicConfigHandler instance;
    return instance;
}

void DynamicConfigHandler::Start() const
{
    std::thread t([]() {
        while (GetInstance().isRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            std::lock_guard<std::mutex> locker(GetInstance().vectorMutex);
            for (auto &configTuple: GetInstance().callBackFunctions) {
                if (GetInstance().isTriggered(configTuple.first)) {
                    configTuple.second();
                }
            }
        }
    });
    t.detach();
}

void DynamicConfigHandler::Stop() const
{
    GetInstance().isRunning = false;
}

std::vector<std::string> DynamicConfigHandler::splitString(const std::string& s, const char delimiter) const
{
    std::vector<std::string> ans;
    if (s.empty()) {
        return ans;
    }

    for (std::size_t i = 0; i < s.length();) {
        std::size_t pos = s.find(delimiter, i);
        if (pos != std::string::npos) {
            if (pos == i) { // 跳过多个连续的分隔符
                i = pos + 1;
                continue;
            } else {
                std::string strTemp = s.substr(i, pos - i);
                ans.push_back(strTemp);
                i = pos + 1;
            }
        } else {
            std::string strTemp = s.substr(i, s.length() - i);
            ans.push_back(strTemp);
            break;
        }
    }
    return ans;
}

std::string DynamicConfigHandler::getConfigFilePath() const
{
    const char *miesInstallPath = std::getenv("MINDIE_LLM_HOME_PATH");
    if (miesInstallPath != nullptr) {
        return std::string(miesInstallPath) + "/conf/config.json";
    }
    return "../conf/config.json";
}

bool DynamicConfigHandler::CheckSystemConfig(const std::string &jsonPath, nlohmann::json &inputJsonData,
    std::string paramType) const
{
    std::string homePath;
    if (!GetHomePath(homePath).IsOk()) {
        std::cout << "Failed to get home path." << std::endl;
        return false;
    }
    std::string systemConfigPath = homePath + "/conf/config.json";
    std::string baseDir = "/";
    if (systemConfigPath.compare(jsonPath) == 0) {
        baseDir = homePath;
    }
    return ParamChecker::ReadJsonFile(jsonPath, baseDir, inputJsonData, paramType);
}

bool DynamicConfigHandler::isTriggered(const std::string pathExpression) const
{
    try {
        std::string configFilePath = getConfigFilePath();
        nlohmann::json configJson;
        CheckSystemConfig(configFilePath, configJson, "");

        std::vector<std::string> pathVec = splitString(pathExpression);
        for (std::string& path : pathVec) {
            if (path == "EnableDynamicAdjustTimeoutConfig" && !configJson.contains(path)) {
                return false;
            } else {
                configJson = configJson.at(path);
            }
        }
        return static_cast<bool>(configJson);
    } catch (nlohmann::json::parse_error& e) {
        std::cout << "DynamicConfigHandler parse_error: " << e.what() << std::endl;
        return false;
    } catch (std::exception& e) {
        std::cout << "DynamicConfigHandler exception: " << e.what() << std::endl;
        return false;
    }
    return true;
}

} // namespace mindie_llm