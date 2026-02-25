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

#ifndef SAFE_LOG_H
#define SAFE_LOG_H

#include <string>
#include <sstream>
#include <fstream>
#include <array>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <functional>

#include "string_utils.h"
#include "safe_envvar.h"

namespace mindie_llm {
// AUDIT: mindie_llm::LogLine mandatory logging.
enum class LogSeverity: uint8_t { AUDIT = 0, DEBUG, INFO, WARN, ERROR, CRITICAL, __COUNT__ };
// Logs of each type go into separate files.
enum class LogType: uint8_t { GENERAL = 0, REQUEST, TOKEN, __COUNT__ };

static const std::string ALL_COMPONENT = "__all__";
// Component enumeration object supporting independent control, with all component logs falling into same file by PID
enum class LogComponent: uint8_t { LLM = 0, LLMMODELS, SERVER, __COUNT__ };
inline const std::string& ComponentToString(LogComponent c)
{
    static const std::array<std::string, static_cast<uint8_t>(LogComponent::__COUNT__)> compNames = {
        "llm", "llmmodels", "server"
    };
    return compNames[static_cast<uint8_t>(c)];
}
// Buffer-pushing struct for async writer thread consumption
struct MsgPkg {
    LogComponent component;
    LogType type;
    std::string msg;
};
// Each component maintains its own configuration
struct ComponentConfig {
    std::atomic<LogSeverity> minLevel;
    std::atomic<bool> toStdout;
    std::atomic<bool> toFile;
    std::atomic<bool> verbose;
};
// Fetch rotation file info per type from buffer
struct LogSink {
    std::string filePath;
    std::string basePath;
    std::ofstream ofs;
    size_t curSize;
};

bool String2LogLevel(const std::string& level, LogSeverity& out);

// ================= LogManager =================

class LogManager {
public:
    static LogManager& GetInstance();

    bool IsPrintLog(LogComponent comp, LogSeverity level);
    void Push(LogComponent comp, LogType type, std::string&& formattedLog);
    ComponentConfig& GetComponentConfig(LogComponent comp);

public:
    template<class T, class Parser, class Setter>
    void LoadByComponentByEnv(const char* envKey, const std::string& defaultVal,
        const std::unordered_set<std::string>& validValues, Parser parser, Setter setter)
    {
        std::string val;
        Result r = EnvVar::GetInstance().Get(envKey, defaultVal, val);
        if (!r.IsOk()) {
            throw std::runtime_error(r.message());
        }
        LoadByComponentByString<T>(val, validValues, parser, setter);
    }

    template<class T, class Parser, class Setter>
    void LoadByComponentByString(const std::string& val, const std::unordered_set<std::string>& validValues,
        Parser parser, Setter setter)
    {
        auto kv = ParseKeyValueString(val, validValues, ALL_COMPONENT, ';', ':');
        for (size_t i = 0; i < componentCfgs_.size(); ++i) {
            auto& cfg = componentCfgs_[i];
            const std::string& componentName = ComponentToString(static_cast<LogComponent>(i));
            if (kv.count(ALL_COMPONENT)) {
                const T value = parser(kv.at(ALL_COMPONENT));
                setter(cfg, value);
            }
            if (kv.count(componentName)) {
                const T value = parser(kv.at(componentName));
                setter(cfg, value);
            }
        }
    }

private:
    LogManager();
    ~LogManager();

    void Init();
    void LoadComponentConfigs();
    bool IsAnyComponentToFile() const;
    void GetLogRotate();
    void GetLogDirs();
    void OpenLogFiles();
    void RenewLogFilePath(LogType type);

    void Writer();
    void FlushLoop();
    void RotateLogs(LogType type);
    void Stop();

private:
    std::atomic<bool> isRunning_{false};
    std::array<ComponentConfig, static_cast<size_t>(LogComponent::__COUNT__)> componentCfgs_;
    std::string logDir_;
    uint32_t logFileSize_;
    uint32_t logFileNum_;
    using BufferArray = std::array<std::vector<MsgPkg>, static_cast<size_t>(LogType::__COUNT__)>;
    BufferArray buffers_;
    std::array<std::mutex, static_cast<size_t>(LogType::__COUNT__)> bufferMutex_;
    std::array<LogSink, static_cast<size_t>(LogType::__COUNT__)> sinks_;
    std::thread flushThread_;
};

// ================= Logger =================

class Logger {
public:
    Logger() = default;
    explicit Logger(LogComponent comp, LogSeverity level);

    bool ShouldLog() const;

    template<typename T>
    Logger& operator<<(const T& v)
    {
        stream_ << v;
        return *this;
    }

    void AssembleAndPush(LogType type, const char* file, size_t line);
    void Reset();

private:
    LogComponent component_;
    LogSeverity level_;
    std::ostringstream stream_;
};

// ================= LogLine =================

class LogLine {
public:
    LogLine(LogComponent comp, LogSeverity level, const char* file, size_t line);
    ~LogLine();

    LogLine(const LogLine&) = delete;
    LogLine& operator=(const LogLine&) = delete;
    LogLine(LogLine&&) = delete;
    LogLine& operator=(LogLine&&) = delete;

    template<typename T>
    LogLine& operator<<(const T& v)
    {
        if (enabled_) {
            logger_ << v;
        }
        return *this;
    }

    LogLine& SetType(LogType t)
    {
        type_ = t;
        return *this;
    }

private:
    Logger& logger_;
    bool enabled_{false};
    LogType type_{LogType::GENERAL};
    const char* file_;
    size_t line_;
};

// ================= GetThreadLogger =================

Logger& GetThreadLogger(LogComponent comp, LogSeverity level);

} // namespace mindie_llm

// ================= Macro =================
// llm log interface
#define LOG_AUDIT_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::AUDIT, __FILE__, __LINE__)
#define LOG_DEBUG_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::DEBUG, __FILE__, __LINE__)
#define LOG_INFO_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::INFO, __FILE__, __LINE__)
#define LOG_WARN_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::WARN, __FILE__, __LINE__)
#define LOG_ERROR_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::ERROR, __FILE__, __LINE__)
#define LOG_CRITICAL_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::CRITICAL, __FILE__, __LINE__)
// model log interface
#define LOG_AUDIT_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::AUDIT, __FILE__, __LINE__)
#define LOG_DEBUG_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::DEBUG, __FILE__, __LINE__)
#define LOG_INFO_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::INFO, __FILE__, __LINE__)
#define LOG_WARN_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::WARN, __FILE__, __LINE__)
#define LOG_ERROR_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::ERROR, __FILE__, __LINE__)
#define LOG_CRITICAL_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::CRITICAL, __FILE__, __LINE__)
// server log interface
#define LOG_AUDIT_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::AUDIT, __FILE__, __LINE__)
#define LOG_DEBUG_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::DEBUG, __FILE__, __LINE__)
#define LOG_INFO_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::INFO, __FILE__, __LINE__)
#define LOG_WARN_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::WARN, __FILE__, __LINE__)
#define LOG_ERROR_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::ERROR, __FILE__, __LINE__)
#define LOG_CRITICAL_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::CRITICAL, __FILE__, __LINE__)

#endif
