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
#include <cstring>
#include <cstdlib>
#include <string>
#include "log.h"
#include "env_util.h"

namespace mindie_llm {
constexpr size_t MAX_ENV_LENGTH = 256; // 环境变量长度最大为256个字符

EnvUtil::EnvUtil()
{
    vars["MINDIE_LLM_HOME_PATH"] = GetEnvByName("MINDIE_LLM_HOME_PATH");
    vars["MIES_CONFIG_JSON_PATH"] = GetEnvByName("MIES_CONFIG_JSON_PATH");
    vars["MIES_MEMORY_DETECTOR_MODE"] = GetEnvByName("MIES_MEMORY_DETECTOR_MODE");
    vars["MIES_PROFILER_MODE"] = GetEnvByName("MIES_PROFILER_MODE");
    vars["MIES_SERVICE_MONITOR_MODE"] = GetEnvByName("MIES_SERVICE_MONITOR_MODE");
    vars["RANK_TABLE_FILE"] = GetEnvByName("RANK_TABLE_FILE");
    vars["MIES_CONTAINER_IP"] = GetEnvByName("MIES_CONTAINER_IP");
    vars["MIES_CONTAINER_MANAGEMENT_IP"] = GetEnvByName("MIES_CONTAINER_MANAGEMENT_IP");
    vars["HOST_IP"] = GetEnvByName("HOST_IP");
    vars["MINDIE_LOG_LEVEL"] = GetEnvByName("MINDIE_LOG_LEVEL");
    vars["MINDIE_LOG_TO_FILE"] = GetEnvByName("MINDIE_LOG_TO_FILE");
    vars["MINDIE_LOG_TO_STDOUT"] = GetEnvByName("MINDIE_LOG_TO_STDOUT");
    vars["MINDIE_LOG_PATH"] = GetEnvByName("MINDIE_LOG_PATH");
    vars["MINDIE_LOG_VERBOSE"] = GetEnvByName("MINDIE_LOG_VERBOSE");
    vars["MINDIE_LOG_ROTATE"] = GetEnvByName("MINDIE_LOG_ROTATE");
    vars["HOME"] = GetEnvByName("HOME");
    vars["DYNAMIC_AVERAGE_WINDOW_SIZE"] = GetEnvByName("DYNAMIC_AVERAGE_WINDOW_SIZE");
    vars["MINDIE_LOG_TO_STDOUT"] = GetEnvByName("MINDIE_LOG_TO_STDOUT");
    vars["MINDIE_LOG_TO_FILE"] = GetEnvByName("MINDIE_LOG_TO_FILE");
    vars["MINDIE_CHECK_INPUTFILES_PERMISSION"] = GetEnvByName("MINDIE_CHECK_INPUTFILES_PERMISSION");
    vars["TOKENIZER_ENCODE_TIMEOUT"] = GetEnvByName("TOKENIZER_ENCODE_TIMEOUT");
    vars["MINDIE_LLM_BENCHMARK_ENABLE"] = GetEnvByName("MINDIE_LLM_BENCHMARK_ENABLE");
}

const std::string& EnvUtil::Get(const std::string& name) const
{
    auto it = vars.find(name);
    if (it != vars.end()) {
        return it->second;
    }
    static const std::string empty = "";
    return empty;
}

std::string EnvUtil::GetEnvByName(const std::string& name) const
{
    char* item = std::getenv(name.c_str());
    if (item == nullptr) {
        return "";
    }

    size_t itemLength = strlen(item);
    if (itemLength > MAX_ENV_LENGTH) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_SERVER_REQUEST,
            CHECK_WARNING), "Value length is too long");
        return "";
    }
    return item;
}

int32_t EnvUtil::GetInt(const std::string& name, int32_t defaultValue) const
{
    const std::string& value = Get(name);
    if (value.empty()) {
        return defaultValue;
    }

    // 字符串转 int32_t，解析失败时返回默认值
    try {
        return std::stoi(value);
    } catch (...) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_SERVER_REQUEST,
            CHECK_WARNING), "Failed to convert " << name << " to integer");
        return defaultValue;
    }
}

void EnvUtil::SetEnvVar(const std::string& name, const std::string& value)
{
    vars[name] = value;
}

void EnvUtil::ClearEnvVar(const std::string& name)
{
    vars.erase(name);
}
} // namespace mindie_llm
