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
 
#ifndef MINDIE_LLM_LOG_H
#define MINDIE_LLM_LOG_H

#include <iostream>
#include <sstream>
#include <mutex>
#include <unordered_map>
#include "log_config.h"
#include "log_level_dynamic_handler.h"
#include "logger_def.h"
#include "common_util.h"

namespace mindie_llm {

using LogLevel = spdlog::level::level_enum;

const std::map<std::string, std::string> ERROR_CODE_MAPPING = {
    // BACKEND
    {"BACKEND_CONFIG_VAL_FAILED", "MIE05E020000"},
    {"BACKEND_INIT_FAILED", "MIE05E020001"},
    // LLM_MANAGER
    {"LLM_MANAGER_CONFIG_FAILED", "MIE05E030000"},
    {"LLM_MANAGER_INIT_FAILED", "MIE05E030001"},
    // TEXT_GENERATOR
    {"TEXT_GENERATOR_PLUGIN_NAME_INVALID", "MIE05E010000"},
    {"TEXT_GENERATOR_FEAT_COMPAT_INVALID", "MIE05E010001"},
    {"TEXT_GENERATOR_REQ_ID_INVALID", "MIE05E010002"},
    {"TEXT_GENERATOR_TEMP_ZERO_DIV_ERR", "MIE05E010003"},
    {"TEXT_GENERATOR_REQ_PENALTY_ZERO_DIV_ERR", "MIE05E010004"},
    {"TEXT_GENERATOR_ZERO_ITER_ERR", "MIE05E010005"},
    {"TEXT_GENERATOR_ZERO_TIME_ERR", "MIE05E010006"},
    {"TEXT_GENERATOR_REQ_ID_UNUSED", "MIE05E010007"},
    {"TEXT_GENERATOR_GENERATOR_BACKEND_INVALID", "MIE05E010008"},
    {"TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH", "MIE05E010009"},
    // reserved code: "MIE05E01000[A-F]"
    {"TEXT_GENERATOR_MISSING_PREFILL_OR_INVALID_DECODE_REQ", "MIE05E010010"},
    {"TEXT_GENERATOR_MAX_BLOCK_SIZE_INVALID", "MIE05E010011"},
    {"TEXT_GENERATOR_EOS_TOKEN_ID_TYPE_INVALID", "MIE05E010012"},
    // ATB_MODELS
    {"ATB_MODELS_PARAM_OUT_OF_RANGE", "MIE05E000000"},
    {"ATB_MODELS_MODEL_PARAM_JSON_INVALID", "MIE05E000001"},
    {"ATB_MODELS_EXECUTION_FAILURE", "MIE05E000002"},
    {"ATB_MODELS_PARAM_INVALID", "MIE05E000003"},
    {"ATB_MODELS_INTERNAL_ERROR", "MIE05E000004"},
    {"ATB_MODELS_OUT_OF_MEMORY", "MIE05E000005"},
};

class Log {
public:
    explicit Log(const std::shared_ptr<LogConfig> logConfig);

    static std::shared_ptr<Log> GetInstance(LoggerType loggerType);

    static void CreateAllLoggers();

    static void CreateInstance(LoggerType loggerType);

    static const std::shared_ptr<LogConfig> GetLogConfig(LoggerType loggerType);

    static void LogMessage(LoggerType loggerType, LogLevel level, const std::string& message);

    static void Flush();

    static void GetErrorCode(std::ostringstream& oss, const std::string& args);

    static std::string GetLevelStr(const LogLevel level);

    static std::string GetUserName();

    static void SetAllLogLevel(LogLevel level);

    static void SetLogLevel(LoggerType loggerType, LogLevel level);

    ~Log() = default;

private:
    void SetFileEventHandle(spdlog::file_event_handlers &handlers) const;
    int Initialize(LoggerType loggerType);
    bool ShouldPrintToStdout(LoggerType loggerType);
    std::string GetLogPattern(LoggerType loggerType);
    static std::string GetLoggerFormat(LoggerType loggerType);
    static void SanitizeMessage(std::string& msg);
private:
    static std::once_flag atbLogInitFlag;
    static std::unordered_map<LoggerType, std::shared_ptr<Log>> loggerMap;
    std::shared_ptr<spdlog::logger> innerLogger_;
    std::shared_ptr<LogConfig> logConfig_;
};

}  // namespace mindie_llm

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (auto& el : vec) {
        os << el << ',';
    }
    return os;
}

#ifndef LOG_FILENAME
#define LOG_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#ifdef FUZZ_TEST
#define MINDIE_LLM_LOG(level, msg, loggerType, ...)
#else
#define MINDIE_LLM_LOG(level, msg, loggerType, ...) \
    do { \
        if (mindie_llm::Log::GetInstance(loggerType) != nullptr && \
            mindie_llm::Log::GetLogConfig(loggerType)->logLevel_ <= (level)) { \
            std::ostringstream oss; \
            MINDIE_LLM_FORMAT_LOG(oss, level, msg, ##__VA_ARGS__); \
            mindie_llm::Log::LogMessage(loggerType, level, oss.str()); \
        } \
    } while (0)
#endif

#define MINDIE_LLM_FORMAT_LOG(oss, level, msg, ...)                                                    \
    do {                                                                                               \
        oss << mindie_llm::Log::GetLevelStr(level);                                                    \
        if (mindie_llm::Log::GetLogConfig(mindie_llm::LoggerType::MINDIE_LLM)->logVerbose_) {          \
            oss << "[" << LOG_FILENAME << ":" << __LINE__ << "] ";}                                    \
        mindie_llm::Log::GetErrorCode(oss, #__VA_ARGS__);                                              \
        oss << msg;                                                                                    \
    } while (0)

// MINDIE_LLM_LOG宏函数调用示例：MINDIE_LLM_LOG_DEBUG(msg, errorcode) errorcode为可选传参
// 其他日志级别同理

#define MINDIE_LLM_LOG_WARN(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::warn, msg, mindie_llm::LoggerType::MINDIE_LLM, __VA_ARGS__)
#define MINDIE_LLM_LOG_WARN_REQUEST(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::warn, msg, mindie_llm::LoggerType::MINDIE_LLM_REQUEST, __VA_ARGS__)
#define MINDIE_LLM_LOG_WARN_TOKEN(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::warn, msg, mindie_llm::LoggerType::MINDIE_LLM_TOKEN, __VA_ARGS__)

#define MINDIE_LLM_LOG_ERROR(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::err, msg, mindie_llm::LoggerType::MINDIE_LLM, __VA_ARGS__)

#define MINDIE_LLM_LOG_FATAL(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::critical, msg, mindie_llm::LoggerType::MINDIE_LLM, __VA_ARGS__)

#define MINDIE_LLM_LOG_DEBUG(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::debug, msg, mindie_llm::LoggerType::MINDIE_LLM, __VA_ARGS__)
#define MINDIE_LLM_LOG_DEBUG_REQUEST(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::debug, msg, mindie_llm::LoggerType::MINDIE_LLM_REQUEST, __VA_ARGS__)
#define MINDIE_LLM_LOG_DEBUG_TOKEN(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::debug, msg, mindie_llm::LoggerType::MINDIE_LLM_TOKEN, __VA_ARGS__)

#define MINDIE_LLM_LOG_INFO(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::info, msg, mindie_llm::LoggerType::MINDIE_LLM, __VA_ARGS__)
#define MINDIE_LLM_LOG_INFO_REQUEST(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::info, msg, mindie_llm::LoggerType::MINDIE_LLM_REQUEST, __VA_ARGS__)
#define MINDIE_LLM_LOG_INFO_TOKEN(msg, ...) \
    MINDIE_LLM_LOG(spdlog::level::info, msg, mindie_llm::LoggerType::MINDIE_LLM_TOKEN, __VA_ARGS__)

#define ATB_SPEED_LOG(level, msg, ...)                                                                 \
    do {                                                                                               \
        if (mindie_llm::Log::GetInstance(mindie_llm::LoggerType::ATB) != nullptr &&                    \
            mindie_llm::Log::GetLogConfig(mindie_llm::LoggerType::ATB)->logLevel_ <= (level)) {        \
            std::ostringstream oss;                                                                    \
            ATB_SPEED_FORMAT_LOG(oss, level, msg, ##__VA_ARGS__);                                      \
            mindie_llm::Log::LogMessage(mindie_llm::LoggerType::ATB, level, oss.str());                \
        }                                                                                              \
    } while (0)

#define ATB_SPEED_FORMAT_LOG(oss, level, msg, ...)                                                     \
    do {                                                                                               \
        oss << mindie_llm::Log::GetLevelStr(level);                                                    \
        if (mindie_llm::Log::GetLogConfig(mindie_llm::LoggerType::ATB)->logVerbose_) {                 \
            oss << "[" << LOG_FILENAME << ":" << __LINE__ << "] ";}                                    \
        mindie_llm::Log::GetErrorCode(oss, #__VA_ARGS__);                                              \
        oss << msg;                                                                                    \
    } while (0)

#define ATB_SPEED_LOG_DEBUG(msg, ...) ATB_SPEED_LOG(spdlog::level::debug, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_INFO(msg, ...) ATB_SPEED_LOG(spdlog::level::info, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_WARN(msg, ...) ATB_SPEED_LOG(spdlog::level::warn, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_ERROR(msg, ...) ATB_SPEED_LOG(spdlog::level::err, msg, __VA_ARGS__)

#define ATB_SPEED_LOG_FATAL(msg, ...) ATB_SPEED_LOG(spdlog::level::critical, msg, __VA_ARGS__)

#ifdef UT_ENABLED
#define ULOG_LOG(level, msg, errCode, submoduleName)
#else
#define ULOG_LOG(level, msg, errCode, submoduleName)                                                   \
    do {                                                                                               \
        if (mindie_llm::Log::GetInstance(mindie_llm::LoggerType::DEBUG) != nullptr &&                  \
            mindie_llm::Log::GetLogConfig(mindie_llm::LoggerType::DEBUG)->logLevel_ <= (level)) {      \
            std::ostringstream oss;                                                                    \
            ULOG_FORMAT_LOG(oss, level, msg, errCode, submoduleName);                                  \
            mindie_llm::Log::LogMessage(mindie_llm::LoggerType::DEBUG, level, oss.str());              \
        }                                                                                              \
    } while (0)

#define ULOG_FORMAT_LOG(oss, level, msg, errCode, submoduleName)                                       \
    do {                                                                                               \
        if (mindie_llm::Log::GetLogConfig(mindie_llm::LoggerType::DEBUG)->logVerbose_) {               \
            oss << mindie_llm::Log::GetLevelStr(level) << "[" << LOG_FILENAME << ":" << __LINE__ <<    \
                "] " << (errCode) << "[" << (submoduleName) << "] ";                                        \
        } else {                                                                                       \
            oss << errCode;                                                                            \
        }                                                                                              \
        oss << msg;                                                                                    \
    } while (0)
#endif

#ifdef UT_ENABLED
#define ULOG_DEBUG(submoduleName, msg)
#define ULOG_INFO(submoduleName, msg)
#define ULOG_WARN(submoduleName, errCode, msg)
#define ULOG_ERROR(submoduleName, errCode, msg)
#define ULOG_CRITICAL(submoduleName, errCode, msg)
#else
#define ULOG_DEBUG(submoduleName, msg) ULOG_LOG(spdlog::level::debug, msg, "", submoduleName)
#define ULOG_INFO(submoduleName, msg) ULOG_LOG(spdlog::level::info, msg, "", submoduleName)
#define ULOG_WARN(submoduleName, errCode, msg) ULOG_LOG(spdlog::level::warn, msg, errCode, submoduleName)
#define ULOG_ERROR(submoduleName, errCode, msg) ULOG_LOG(spdlog::level::err, msg, errCode, submoduleName)
#define ULOG_CRITICAL(submoduleName, errCode, msg) ULOG_LOG(spdlog::level::critical, msg, errCode, submoduleName)
#endif

#ifdef UT_ENABLED
#define ULOG_AUDIT(userID, moduleName, operation, ret)
#else
#define ULOG_AUDIT(userID, moduleName, operation, ret)                                                     \
    do {                                                                                                   \
        if (mindie_llm::Log::GetInstance(mindie_llm::LoggerType::SECURITY) != nullptr) {                   \
            std::ostringstream oss;                                                                        \
            oss << "[" << mindie_llm::Log::GetUserName() << "] [" << (userID) << "] [" <<                  \
                moduleName << "] [" << operation << "] [" << ret << "]";                                   \
            mindie_llm::Log::LogMessage(mindie_llm::LoggerType::SECURITY, spdlog::level::info, oss.str()); \
        }                                                                                                  \
    } while (0)
#endif
#endif
