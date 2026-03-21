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
#include "log.h"

#include <sys/utsname.h>
#include <cstdlib>
#include <regex>
#include <stdexcept>

#include "spdlog.h"

#include "common_util.h"
#include "log_utils.h"

namespace mindie_llm {

constexpr size_t MAX_MSG_SIZE = 2048;

std::unordered_map<LoggerType, std::shared_ptr<Log>> Log::loggerMap;
std::once_flag Log::atbLogInitFlag{};

const std::unordered_set<std::string> SpecialChar = {"\n",  "\r",  "\u007f", "\b",  "\f",  "\t",  "\v", "\u000b", "%08",
                                                     "%09", "%0a", "%0b",    "%0c", "%0d", "%7f", "//", "\\",     "&"};

Log::Log(const std::shared_ptr<LogConfig> logConfig)
    : logConfig_(logConfig ? std::make_shared<LogConfig>(*logConfig) : nullptr)
{
}

std::shared_ptr<Log> Log::GetInstance(LoggerType loggerType)
{
    // ATB日志不能统一创建，需要在第一次调用ATB日志打印时进行创建
    if (loggerType == LoggerType::ATB) {
        std::call_once(atbLogInitFlag, [] {
            CreateInstance(LoggerType::ATB);
            mindie_llm::LogLevelDynamicHandler::Init(5000); // 每5秒检查动态日志配置
        });
    }
    auto logger = loggerMap.find(loggerType);
    if (logger != loggerMap.end()) {
        return logger->second;
    }
    return nullptr;
}

void Log::CreateAllLoggers()
{
    for (int i = 0; i < static_cast<int>(LoggerType::MAX_LOGGER_TYPE); i++) {
        LoggerType loggerType = static_cast<LoggerType>(i);
        if (loggerType == LoggerType::ATB) {
            continue;
        }
        CreateInstance(loggerType);
    }
}

void Log::CreateInstance(LoggerType loggerType)
{
    auto logger = loggerMap.find(loggerType);
    if (logger != loggerMap.end()) {
        return;
    }

    std::string name = LogUtils::GetLoggerNameStr(loggerType);
    std::shared_ptr<LogConfig> logConfig = std::make_shared<LogConfig>();
    if (!logConfig || logConfig->Init(loggerType) != LOG_OK) {
        throw std::runtime_error(name + " failed to init the logConfig.");
    }
    if (logConfig->ValidateSettings() != LOG_OK) {
        throw std::runtime_error(name + " log params validation failed.");
    }
    std::shared_ptr<Log> targetLogger = std::make_shared<Log>(logConfig);
    if (targetLogger == nullptr) {
        throw std::runtime_error(name + " failed to create logger.");
    }

    if (targetLogger->Initialize(loggerType) != LOG_OK) {
        throw std::runtime_error(name + " failed to initialize inner logger.");
    }
    loggerMap[loggerType] = targetLogger;
}

const std::shared_ptr<LogConfig> Log::GetLogConfig(LoggerType loggerType)
{
    std::shared_ptr<Log> logger = GetInstance(loggerType);
    if (logger != nullptr) {
        return logger->logConfig_;
    }
    throw std::runtime_error(LogUtils::GetLoggerNameStr(loggerType) + " logger is null, failed to get LogConfig.");
}

void Log::LogMessage(LoggerType loggerType, LogLevel level, const std::string &message)
{
    std::shared_ptr<Log> logger = GetInstance(loggerType);
    if (logger == nullptr || logger->innerLogger_ == nullptr) {
        throw std::runtime_error(LogUtils::GetLoggerNameStr(loggerType) + " logger is null.");
    }
    if (message.empty() || level < 0 || level > MAX_LOG_LEVEL_LIMIT) {
        throw std::runtime_error(LogUtils::GetLoggerNameStr(loggerType) + " invalid log params.");
    }

    // 长度检查和截断处理
    std::string truncatedMsg = message;
    if (message.size() > MAX_MSG_SIZE) {
        truncatedMsg = message.substr(0, MAX_MSG_SIZE);
        truncatedMsg += "...[TRUNCATED]";
    }
    SanitizeMessage(truncatedMsg);
    logger->innerLogger_->log(level, "{}", truncatedMsg.c_str());
    if (loggerType == LoggerType::SECURITY || loggerType == LoggerType::DEBUG) {
        logger->innerLogger_->flush();
    }
}

void Log::SetFileEventHandle(spdlog::file_event_handlers &handlers) const
{
    handlers.after_open = [](spdlog::filename_t filename, const std::FILE *fstream) {
        std::string baseDir = "/";
        std::string errMsg;
        std::string regularPath;
        if (!FileUtils::RegularFilePath(filename, baseDir, errMsg, true, regularPath)) {
            std::cerr << "Regular file failed by " << errMsg << std::endl;
            throw std::runtime_error("LLM Regular log file path failed");
        }
        chmod(regularPath.c_str(), MAX_OPEN_LOG_FILE_PERM);
        (void)fstream;
    };
    handlers.before_close = [](spdlog::filename_t filename, const std::FILE *fstream) {
        std::string baseDir = "/";
        std::string errMsg;
        std::string regularPath;
        if (!FileUtils::RegularFilePath(filename, baseDir, errMsg, true, regularPath)) {
            std::cerr << "Regular file failed by " << errMsg << std::endl;
            throw std::runtime_error("LLM Regular log file path failed");
        }
        chmod(regularPath.c_str(), MAX_CLOSE_LOG_FILE_PERM);
        (void)fstream;
    };
}

bool Log::ShouldPrintToStdout(LoggerType loggerType) { return loggerType != LoggerType::MINDIE_LLM_TOKEN; }

int Log::Initialize(LoggerType loggerType)
{
    std::vector<spdlog::sink_ptr> sinks;
    try {
        if (logConfig_->logToStdOut_ && ShouldPrintToStdout(loggerType)) {
            auto stdoutSink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
            stdoutSink->set_pattern(GetLogPattern(loggerType));
            sinks.push_back(stdoutSink);
        }
        if (logConfig_->logToFile_) {
            spdlog::file_event_handlers handlers;
            SetFileEventHandle(handlers);
            auto fileSink = std::make_shared<GenericRotationFileSink>(
                logConfig_->logFilePath_,
                logConfig_->logFileSize_,
                logConfig_->logFileCount_ - 1,
                handlers,
                logConfig_->baseDir_
            );
            sinks.push_back(fileSink);
            spdlog::flush_every(std::chrono::seconds(1));
        }
        if (loggerType == LoggerType::ATB) {
            // ATB日志使用同步日志器 spdlog::logger
            innerLogger_ =
                std::make_shared<spdlog::logger>(LogUtils::GetLoggerNameStr(loggerType), sinks.begin(), sinks.end());
        } else {
            innerLogger_ = std::make_shared<spdlog::async_logger>(LogUtils::GetLoggerNameStr(loggerType),
                sinks.begin(), sinks.end(), spdlog::thread_pool(),
                spdlog::async_overflow_policy::block);
        }
        innerLogger_->set_level(static_cast<spdlog::level::level_enum>(logConfig_->logLevel_));
        innerLogger_->set_pattern("%Y-%m-%d %H:%M:%S.%f %t %v");
        innerLogger_->set_pattern(GetLogPattern(loggerType));
        innerLogger_->flush_on(spdlog::level::err);
    } catch (const spdlog::spdlog_ex &e) {
        std::stringstream errMsg;
        errMsg << "Failed to create inner logger: " << e.what();
        throw std::runtime_error(errMsg.str());
    }
    return LOG_OK;
}

std::string Log::GetLogPattern(LoggerType loggerType)
{
    std::string moduleName = LogUtils::GetModuleName(loggerType);
    std::string pattern = "[%Y-%m-%d %H:%M:%S.%f] %v";
    if (loggerType <= LoggerType::ATB) {
        if (logConfig_->logVerbose_) {
            pattern = "[%Y-%m-%d %H:%M:%S.%f] [%P] [%t] [" + moduleName + "] %v";
        }
    } else if (loggerType == LoggerType::SECURITY || loggerType == LoggerType::DEBUG) {
        pattern = "[%Y-%m-%d %H:%M:%S.%f] %v";
        if (logConfig_->logVerbose_) {
            pattern = "[%Y-%m-%d %H:%M:%S.%f] [%P] [%t] [" + moduleName + "] %v";
        }
        if (loggerType == LoggerType::SECURITY) {
            pattern = "[%Y-%m-%d %H:%M:%S.%f] [%P] %v";
        }
    }
    return pattern;
}

void Log::Flush()
{
    for (const auto &pair : loggerMap) {
        const auto &logger = pair.second;
        if (logger && logger->innerLogger_) {
            logger->innerLogger_->flush();
        }
    }
}


void Log::GetErrorCode(std::ostringstream &oss, const std::string &args)
{
    std::ostringstream errorcode;
    errorcode << args;
    std::string errorcodeStr = errorcode.str();
    if (errorcodeStr.size() > 0) {
        auto it = ERROR_CODE_MAPPING.find(errorcodeStr);
        if (it != ERROR_CODE_MAPPING.end()) {
            oss << "[" << it->second << "] ";
        } else {
            std::cout << "ErrorCode not found in errorCodeMap!" << std::endl;
        }
    }
}

std::string Log::GetLevelStr(const LogLevel level)
{
    switch (level) {
        case LogLevel::critical: return "[CRITICAL] ";
        case LogLevel::err: return "[ERROR] ";
        case LogLevel::warn: return "[WARN] ";
        case LogLevel::info: return "[INFO] ";
        case LogLevel::debug: return "[DEBUG] ";
        default: return "[] ";
    }
}

std::string Log::GetUserName()
{
    struct utsname buf;
    if (uname(&buf) != 0) {
        // utsname.nodename will never be null ptr in design, so we don't need to check it
        *buf.nodename = '\0';
    }
    return buf.nodename;
}

void Log::SanitizeMessage(std::string &msg)
{
    for (const auto &bad : SpecialChar) {
        size_t pos = 0;
        while ((pos = msg.find(bad, pos)) != std::string::npos) {
            msg.replace(pos, bad.size(), "_");
            pos += 1;
        }
    }
}

void Log::SetAllLogLevel(LogLevel level)
{
    for (int i = 0; i < static_cast<int>(LoggerType::MAX_LOGGER_TYPE); i++) {
        LoggerType loggerType = static_cast<LoggerType>(i);
        SetLogLevel(loggerType, level);
    }
}

void Log::SetLogLevel(LoggerType loggerType, LogLevel level)
{
    auto logger = loggerMap.find(loggerType);
    if (logger == loggerMap.end()) {
        return;
    }
    GetLogConfig(loggerType)->logLevel_ = level;
    std::shared_ptr<spdlog::logger> innerLog = logger->second->innerLogger_;
    if (innerLog) {
        innerLog->set_level(level);
        for (auto& sink : innerLog->sinks()) {
            sink->set_level(level);
        }
        innerLog->info("Log level changed to: {}", GetLevelStr(level));
    }
}
} // namespace mindie_llm
