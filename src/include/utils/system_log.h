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
#include "safe_io.h"

namespace mindie_llm {

static const std::map<std::string, std::string> ERROR_CODE = {
    // BACKEND
    {"BACKEND_CONFIG_VAL_FAILED", "[MIE05E020000] "},
    {"BACKEND_INIT_FAILED", "[MIE05E020001] "},
    // LLM_MANAGER
    {"LLM_MANAGER_CONFIG_FAILED", "[MIE05E030000] "},
    {"LLM_MANAGER_INIT_FAILED", "[MIE05E030001] "},
    // TEXT_GENERATOR
    {"TEXT_GENERATOR_PLUGIN_NAME_INVALID", "[MIE05E010000] "},
    {"TEXT_GENERATOR_FEAT_COMPAT_INVALID", "[MIE05E010001] "},
    {"TEXT_GENERATOR_REQ_ID_INVALID", "[MIE05E010002] "},
    {"TEXT_GENERATOR_TEMP_ZERO_DIV_ERR", "[MIE05E010003] "},
    {"TEXT_GENERATOR_REQ_PENALTY_ZERO_DIV_ERR", "[MIE05E010004] "},
    {"TEXT_GENERATOR_ZERO_ITER_ERR", "[MIE05E010005] "},
    {"TEXT_GENERATOR_ZERO_TIME_ERR", "[MIE05E010006] "},
    {"TEXT_GENERATOR_REQ_ID_UNUSED", "[MIE05E010007] "},
    {"TEXT_GENERATOR_GENERATOR_BACKEND_INVALID", "[MIE05E010008] "},
    {"TEXT_GENERATOR_LOGITS_SHAPE_MISMATCH", "[MIE05E010009] "},
    // reserved code: "MIE05E01000[A-F]"
    {"TEXT_GENERATOR_MISSING_PREFILL_OR_INVALID_DECODE_REQ", "[MIE05E010010] "},
    {"TEXT_GENERATOR_MAX_BLOCK_SIZE_INVALID", "[MIE05E010011] "},
    {"TEXT_GENERATOR_EOS_TOKEN_ID_TYPE_INVALID", "[MIE05E010012] "},
    // ATB_MODELS
    {"ATB_MODELS_PARAM_OUT_OF_RANGE", "[MIE05E000000] "},
    {"ATB_MODELS_MODEL_PARAM_JSON_INVALID", "[MIE05E000001] "},
    {"ATB_MODELS_EXECUTION_FAILURE", "[MIE05E000002] "},
};

static const std::string ALL_COMPONENT = "__all__";

// AUDIT: mindie_llm::LogLine mandatory logging.
enum class LogSeverity: uint8_t { DEBUG, INFO, WARN, ERROR, CRITICAL, AUDIT, __COUNT__ };
const std::array<std::string, static_cast<uint8_t>(LogSeverity::__COUNT__) - 1>& GetLogSeverityNameArray();
const std::unordered_set<std::string>& GetAllLogSeverity();
bool String2LogSeverity(const std::string& level, LogSeverity& out);

// Logs of each type go into separate files.
enum class LogType: uint8_t { GENERAL = 0, REQUEST, TOKEN, TOKENIZER, __COUNT__ };
const std::array<std::string, static_cast<uint8_t>(LogType::__COUNT__)>& GetLogTypeNameArray();
bool String2LogType(const std::string& s, LogType& out);

// Component enumeration object supporting independent control, with all component logs falling into same file by PID
enum class LogComponent: uint8_t { LLM = 0, LLMMODELS, SERVER, __COUNT__ };
const std::array<std::string, static_cast<uint8_t>(LogComponent::__COUNT__)>& GetComponentNameArray();
const std::string& Component2String(LogComponent c);
bool String2Component(const std::string& s, LogComponent& out);

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

// ================= LogManager =================

class LogManager {
public:
    static LogManager& GetInstance();

    bool IsPrintLog(LogComponent comp, LogSeverity level);
    void Push(LogComponent comp, LogType type, std::string&& formattedLog);
    ComponentConfig& GetComponentConfig(LogComponent comp);
    void RestartAfterFork();

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
            const std::string& componentName = Component2String(static_cast<LogComponent>(i));
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
    void StartAsyncThread();
    void LoadComponentConfigs();
    bool IsAnyComponentToFile() const;
    void GetLogRotate();
    void GetLogDirs();
    void OpenLogFiles();
    void CreateLogFilePath(LogType type);

    void Writer();
    void FlushLoop();
    uint32_t GetLogFileSizeCutOff(LogType type) const;
    uint32_t GetLogFileNumCutOff(LogType type) const;
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

    void AssembleAndPush(LogType type, const char* file, size_t line, std::string& stack);
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
    std::string BuildStackTrace();

private:
    Logger& logger_;
    bool enabled_{false};
    LogType type_{LogType::GENERAL};
    const char* file_;
    size_t line_;
    std::string stack_;
};

// ================= GetThreadLogger =================

Logger& GetThreadLogger(LogComponent comp, LogSeverity level);

// ================= DynamicLogManager =================

struct DynamicLogConfig {
    std::string logSeverity;
    int validHours{2};
    std::string validTimeStamp;
};

struct DynamicLogDiff {
    bool logSeverityChanged{false};
    bool validHoursChanged{false};
    bool validTimeStampChanged{false};
};

class DynamicLogManager {
public:
    static DynamicLogManager& GetInstance();

private:
    DynamicLogManager();
    ~DynamicLogManager();

    void Init();
    void Stop();
    void GetDefaultLogSeverity();
    void Monitor();
    void GetAndSetLogConfig();
    std::string GetConfigPath() const;
    DynamicLogConfig LoadLogConfig(const std::string& configPath);
    std::string GetLogSeverity(const Json& logConfig) const;
    int GetTimeInterval(const Json& logConfig, int lastHours) const;
    std::string GetTimeStamp(const Json& logConfig, const std::string& lastTs) const;
    bool IsValidTimeFormat(const std::string& timeStr) const;
    bool ParseTime(const std::string& s, std::time_t& out) const;
    bool IsGreaterThanNow(const std::string& timeStr) const;

    DynamicLogDiff DiffConfig(const DynamicLogConfig& current, const DynamicLogConfig& last);
    void ResetToDefaultLogSeverity();
    void ApplyLogSeverity(const std::string& severity);
    void UpdateValidTimeStamp(DynamicLogConfig& cfg);
    bool IsWithinValidRange(const DynamicLogConfig& cfg) const;

private:
    const std::string keyLogConfig = "LogConfig";
    const std::string keyLogSeverity = "dynamicLogLevel";
    const std::string keyTimeInterval = "dynamicLogLevelValidHours";
    const std::string keyTimeStamp = "dynamicLogLevelValidTime";

private:
    std::atomic<bool> isRunning_{false};
    std::thread monitorThread_;
    std::mutex mtx_;
    static constexpr uint8_t monitorInterval_{5};
    static constexpr int defaultHours_{2};

    std::string defaultLogSeverity_{"info"};
    std::string lastLogSeverity_;
    int lastValidHours_{defaultHours_};
    std::string lastValidTimeStamp_;
};

// ================= InitSystemLog =================

void InitSystemLog();

} // namespace mindie_llm

// ================= Macro =================
// llm log interface
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
#define LOG_AUDIT_LLM \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLM, mindie_llm::LogSeverity::AUDIT, __FILE__, __LINE__)
// model log interface
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
#define LOG_AUDIT_MODEL \
    mindie_llm::LogLine(mindie_llm::LogComponent::LLMMODELS, mindie_llm::LogSeverity::AUDIT, __FILE__, __LINE__)
// server log interface
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
#define LOG_AUDIT_SERVER \
    mindie_llm::LogLine(mindie_llm::LogComponent::SERVER, mindie_llm::LogSeverity::AUDIT, __FILE__, __LINE__)

#endif
