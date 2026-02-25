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

#include "system_log.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/prctl.h>
#include <algorithm>

#include "safe_envvar.h"
#include "safe_path.h"
#include "string_utils.h"

namespace mindie_llm {

const std::string LLM = "llm";
constexpr int DECIMAL_BASE = 10;
constexpr size_t BUFFER_SIZE_32 = 32;

static const std::unordered_map<std::string, LogSeverity> str2LogLevelMap = {
    {"debug", LogSeverity::DEBUG},
    {"info",  LogSeverity::INFO},
    {"warn",  LogSeverity::WARN},
    {"error", LogSeverity::ERROR},
    {"critical", LogSeverity::CRITICAL}
};

static const std::unordered_map<LogSeverity, std::string> logLevel2StrMap = {
    {LogSeverity::AUDIT, "AUDIT"},
    {LogSeverity::DEBUG, "DEBUG"},
    {LogSeverity::INFO, "INFO"},
    {LogSeverity::WARN, "WARN"},
    {LogSeverity::ERROR, "ERROR"},
    {LogSeverity::CRITICAL, "CRITICAL"}
};

static const std::unordered_map<LogType, std::string> logType2StrMap = {
    {LogType::GENERAL, "xxxmindie-llm"},
    {LogType::REQUEST, "xxxmindie-llm-request"},
    {LogType::TOKEN, "xxxmindie-llm-token"}
};

enum class TimestampFormat { READABLE, TIGHT };

// ================= Log utils =================

bool String2LogLevel(const std::string& in, LogSeverity& out)
{
    std::string key = in;
    ToLower(key);
    auto it = str2LogLevelMap.find(key);
    if (it == str2LogLevelMap.end()) {
        return false;
    }
    out = it->second;
    return true;
}

void AppendCurTimestamp(std::string& out, TimestampFormat format)
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto sec = time_point_cast<seconds>(now);
    auto ms  = duration_cast<milliseconds>(now - sec).count();
    std::time_t t = system_clock::to_time_t(sec);
    std::tm tbuf;
    localtime_r(&t, &tbuf);
    char buf[BUFFER_SIZE_32];
    size_t length = 0;
    if (format == TimestampFormat::READABLE) {
        buf[length++] = '[';
        length += std::strftime(buf + length, sizeof(buf) - length, "%Y-%m-%d %H:%M:%S", &tbuf);
        buf[length++] = '.';
    } else if (format == TimestampFormat::TIGHT) {
        length = std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tbuf);
    }
    buf[length++] = '0' + (ms / (DECIMAL_BASE * DECIMAL_BASE));
    buf[length++] = '0' + (ms / DECIMAL_BASE % DECIMAL_BASE);
    buf[length++] = '0' + (ms % DECIMAL_BASE);
    if (format == TimestampFormat::READABLE) {
        buf[length++] = ']';
    }
    buf[length] = '\0';
    out.append(buf, length);
}

std::string GetTightTimestamp()
{
    constexpr size_t tightTimestampLength = 17; // YYYYMMDDHHMMSSmmm
    std::string ts;
    ts.reserve(tightTimestampLength);
    AppendCurTimestamp(ts, TimestampFormat::TIGHT);
    return ts;
}

inline void AppendComponent(std::string& out, const std::string& comp)
{
    out.append(" [");
    out.append(comp);
    out.push_back(']');
}

inline void AppendInt(std::string& out, uint64_t v, int width = 0)
{
    char buf[BUFFER_SIZE_32];
    char* p = buf + sizeof(buf);
    do {
        *--p = '0' + (v % DECIMAL_BASE);
        v /= DECIMAL_BASE;
    } while (v);
    int length = buf + sizeof(buf) - p;
    for (; length < width; ++length) {
        out.push_back('0');
    }
    out.append(p, buf + sizeof(buf));
}

inline void AppendPid(std::string& out)
{
    out.append(" [");
    AppendInt(out, ::getpid());
    out.push_back(']');
}

inline void AppendTid(std::string& out)
{
    out.append(" [");
    AppendInt(out, static_cast<uint64_t>(::syscall(SYS_gettid)));
    out.push_back(']');
}

inline std::string LogLevelToString(LogSeverity level)
{
    auto it = logLevel2StrMap.find(level);
    if (it != logLevel2StrMap.end()) {
        return it->second;
    }
    return "INFO";
}

inline void AppendLevel(std::string& out, LogSeverity level)
{
    out.append(" [");
    out.append(LogLevelToString(level));
    out.push_back(']');
}

inline void FilterAndAppend(std::string& out, const char* input, size_t length)
{
    constexpr unsigned char kAsciiControlMin = 0x00;
    constexpr unsigned char kAsciiControlMax = 0x1F;
    constexpr unsigned char kAsciiDelete = 0x7F;
    constexpr unsigned char kLineFeed = '\n';
    constexpr unsigned char kCarriageReturn = '\r';
    const char* cursor = input;
    const char* end = input + length;
    while (cursor < end) {
        unsigned char ch = static_cast<unsigned char>(*cursor++);
        const bool isControlChar = (ch >= kAsciiControlMin && ch <= kAsciiControlMax) || (ch == kAsciiDelete);
        const bool isLineBreak = (ch == kLineFeed) || (ch == kCarriageReturn);
        if (isControlChar || isLineBreak) {
            out.push_back('_');
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
}

void ParseRotateArgs(const std::string& argsStr, uint32_t& outLogFileSize, uint32_t& outLogFileNum)
{
    std::unordered_map<std::string, std::string> rotateArgs = ParseArgs(argsStr);
    if (rotateArgs.find("-fs") != rotateArgs.end()) {
        Result r = Str2Int(rotateArgs["-fs"], "logFileSize", outLogFileSize);
        if (!r.IsOk()) {
            throw std::runtime_error(r.message());
        }
        outLogFileSize *= SIZE_1MB;
        if (outLogFileSize < SIZE_1MB || outLogFileSize > SIZE_500MB) {
            throw std::runtime_error("Log file size must be between 1 MB and 500 MB, got: " +
                                     std::to_string(outLogFileSize) + " bytes (" +
                                     std::to_string(outLogFileSize / SIZE_1MB) + " MB)");
        }
    }
    if (rotateArgs.find("-r") != rotateArgs.end()) {
        Result r = Str2Int(rotateArgs["-r"], "logFileNum", outLogFileNum);
        if (!r.IsOk()) {
            throw std::runtime_error(r.message());
        }
        constexpr size_t fileNumLimit1 = 1;
        constexpr size_t fileNumLimit64 = 64;
        if (outLogFileNum < fileNumLimit1 || outLogFileNum > fileNumLimit64) {
            throw std::runtime_error("Log file count must be between " + std::to_string(fileNumLimit1) + " and " +
                                     std::to_string(fileNumLimit64) + ", got: " + std::to_string(outLogFileNum));
        }
    }
}

inline void AppendFileLine(std::string& out, const char* file, int line)
{
    out.append(" [");
    out.append(GetBasename(file));
    out.push_back(':');
    AppendInt(out, line);
    out.append("] ");
}

static std::string MakeRotateName(const std::string& base, int idx)
{
    // idx = 1 → .01.log
    char buf[BUFFER_SIZE_32];
    std::snprintf(buf, sizeof(buf), ".%02d.log", idx);
    return base + buf;
}

// ================= LogManager =================

LogManager& LogManager::GetInstance()
{
    static LogManager inst;
    inst.Init();
    return inst;
}

LogManager::LogManager() = default;

LogManager::~LogManager()
{
    Stop();
}

void LogManager::Stop()
{
    if (!isRunning_) {
        return;
    }
    isRunning_ = false;
    if (flushThread_.joinable()) {
        flushThread_.join();
    }
}

void LogManager::Init()
{
    LoadComponentConfigs();
    if (IsAnyComponentToFile()) {
        GetLogRotate();
        GetLogDirs();
        OpenLogFiles();
    }
    isRunning_ = true;
    flushThread_ = std::thread(&LogManager::FlushLoop, this);
    pthread_setname_np(flushThread_.native_handle(), "LogFlushThread");
}

void LogManager::LoadComponentConfigs()
{
    LoadByComponentByEnv<LogSeverity>(MINDIE_LOG_LEVEL, DEFAULT_MINDIE_LOG_LEVEL,
        {"debug", "info", "warn", "error", "critical"},
        [](const std::string& s) {
            LogSeverity lvl;
            if (!String2LogLevel(s, lvl)) {
                return LogSeverity::INFO;
            }
            return lvl;
        },
        [](ComponentConfig& c, LogSeverity v) {
            c.minLevel = v;
        }
    );
    LoadByComponentByEnv<bool>(MINDIE_LOG_TO_STDOUT, DEFAULT_MINDIE_LOG_TO_STDOUT, {"true", "false", "1", "0"},
        [](const std::string& s) {
            return s == "1" || s == "true";
        },
        [](ComponentConfig& c, bool v) {
            c.toStdout = v;
        }
    );
    LoadByComponentByEnv<bool>(MINDIE_LOG_TO_FILE, DEFAULT_MINDIE_LOG_TO_FILE, {"true", "false", "1", "0"},
        [](const std::string& s) {
            return s == "1" || s == "true";
        },
        [](ComponentConfig& c, bool v) {
            c.toFile = v;
        }
    );
    LoadByComponentByEnv<bool>(MINDIE_LOG_VERBOSE, DEFAULT_MINDIE_LOG_VERBOSE, {"true", "false", "1", "0"},
        [](const std::string& s) {
            return s == "1" || s == "true";
        },
        [](ComponentConfig& c, bool v) {
            c.verbose = v;
        }
    );
}

bool LogManager::IsAnyComponentToFile() const
{
    for (const auto& c : componentCfgs_) {
        if (c.toFile) {
            return true;
        }
    }
    return false;
}

ComponentConfig& LogManager::GetComponentConfig(LogComponent comp)
{
    return componentCfgs_[static_cast<size_t>(comp)];
}

bool LogManager::IsPrintLog(LogComponent comp, LogSeverity level)
{
    if (level == LogSeverity::AUDIT) {
        return isRunning_;
    }
    auto& cfg = GetComponentConfig(comp);
    return isRunning_ && level >= cfg.minLevel && (cfg.toStdout || cfg.toFile);
}

void LogManager::GetLogRotate()
{
    std::string rotateVal;
    Result r = EnvVar::GetInstance().Get(MINDIE_LOG_ROTATE, DEFAULT_MINDIE_LOG_ROTATE, rotateVal);
    if (!r.IsOk()) {
        throw std::runtime_error(r.message());
    }
    std::unordered_map<std::string, std::string> rotateValMap = ParseKeyValueString(
        rotateVal, {}, ALL_COMPONENT, ';', ':');
    if (rotateValMap.count(LLM)) {
        ParseRotateArgs(rotateValMap[LLM], logFileSize_, logFileNum_);
    }
    if (rotateValMap.count(ALL_COMPONENT)) {
        ParseRotateArgs(rotateValMap[ALL_COMPONENT], logFileSize_, logFileNum_);
    }
}

void LogManager::GetLogDirs()
{
    Result r = EnvVar::GetInstance().Get(MINDIE_LOG_PATH, DEFAULT_MINDIE_LOG_PATH, logDir_);
    if (!r.IsOk()) {
        throw std::runtime_error(r.message());
    }
    auto logDirMap = ParseKeyValueString(logDir_, {}, ALL_COMPONENT, ';', ':');
    std::string base;
    if (logDirMap.count(LLM)) {
        base = logDirMap[LLM];
    }
    if (logDirMap.count(ALL_COMPONENT)) {
        base = logDirMap[ALL_COMPONENT];
    }
    logDir_ = base + "/debug/";
    r = MakeDirs(logDir_);
    if (!r.IsOk()) {
        throw std::runtime_error(r.message());
    }
}

void LogManager::OpenLogFiles()
{
    for (size_t i = 0; i < static_cast<size_t>(LogType::__COUNT__); ++i) {
        LogType type = static_cast<LogType>(i);
        RenewLogFilePath(type);
        auto& sink = sinks_[i];
        sink.ofs.open(sink.filePath, std::ios::app);
        if (sink.ofs.is_open()) {
            std::error_code ec;
            auto sz = fs::file_size(sink.filePath, ec);
            sink.curSize = ec ? 0 : static_cast<size_t>(sz);
        }
    }
}

void LogManager::RenewLogFilePath(LogType type)
{
    const size_t idx = static_cast<size_t>(type);
    auto& sink = sinks_[idx];
    sink.basePath = logDir_ + Join(
        std::vector<std::string>{logType2StrMap.at(type), std::to_string(getpid()), GetTightTimestamp()}, "_");
    SafePath logBasePath(sink.basePath, PathType::FILE, "a+", PERM_440);
    Result r = logBasePath.Check(sink.basePath, false);
    if (!r.IsOk()) {
        throw std::runtime_error(r.message());
    }
    sink.filePath = sink.basePath + ".log";
}

void LogManager::Push(LogComponent comp, LogType type, std::string&& msg)
{
    if (!isRunning_) {
        return;
    }
    const size_t idx = static_cast<size_t>(type);
    if (idx >= buffers_.size()) {
        return;
    }
    std::lock_guard<std::mutex> lock(bufferMutex_[idx]);
    buffers_[idx].push_back(MsgPkg{comp, type, std::move(msg)});
}

void LogManager::FlushLoop()
{
    while (isRunning_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        Writer();
    }
    Writer();
}

void LogManager::Writer()
{
    BufferArray local;
    for (size_t i = 0; i < local.size(); ++i) {
        std::lock_guard<std::mutex> lock(bufferMutex_[i]);
        local[i].swap(buffers_[i]);
    }
    for (size_t i = 0; i < local.size(); ++i) {
        auto& msgs = local[i];
        if (msgs.empty()) {
            continue;
        }
        auto& sink = sinks_[i];
        for (auto& m : msgs) {
            const auto& cfg = componentCfgs_[static_cast<size_t>(m.component)];
            if (cfg.toStdout) {
                struct iovec iov[2] = {
                    { const_cast<char*>(m.msg.data()), m.msg.size() },
                    { const_cast<char*>("\n"), 1 }
                };
                ssize_t ret = ::writev(STDOUT_FILENO, iov, 2);
                if (ret == -1) {
                    perror("writev failed for system log.");
                }
            }
            if (cfg.toFile && sink.ofs.is_open()) {
                if (sink.curSize + m.msg.size() + 1 >= logFileSize_) {
                    RotateLogs(static_cast<LogType>(i));
                }
                sink.ofs << m.msg << '\n';
                sink.curSize += m.msg.size() + 1;
            }
        }
        std::cout.flush();
        if (sink.ofs.is_open()) {
            sink.ofs.flush();
        }
    }
}

void LogManager::RotateLogs(LogType type)
{
    // RotateLogs only called in flush thread. RotateLogs <- Writer <- FlushLoop
    const size_t idx = static_cast<size_t>(type);
    auto& sink = sinks_[idx];

    sink.ofs.close();
    ChangePermission(sink.filePath, PERM_440);

    const std::string& base = sink.basePath;
    std::error_code ec;
    fs::remove(MakeRotateName(base, logFileNum_), ec);
    ec.clear();
    for (int i = static_cast<int>(logFileNum_) - 1; i >= 1; --i) {
        fs::rename(MakeRotateName(base, i), MakeRotateName(base, i + 1), ec);
        ec.clear();
    }
    fs::rename(base + ".log", MakeRotateName(base, 1), ec);
    ec.clear();
    sink.filePath = base + ".log";
    sink.ofs.open(sink.filePath, std::ios::app);
    std::error_code ec2;
    auto sz = fs::file_size(sink.filePath, ec2);
    sink.curSize = ec2 ? 0 : static_cast<size_t>(sz);
}

// ================= Logger =================

Logger::Logger(LogComponent comp, LogSeverity level)
    : component_(comp), level_(level) {}

bool Logger::ShouldLog() const
{
    return LogManager::GetInstance().IsPrintLog(component_, level_);
}

void Logger::AssembleAndPush(LogType type, const char* file, size_t line)
{
    if (stream_.tellp() == std::streampos(0)) {
        return;
    }
    constexpr size_t maxLogMsgLength = 2048UL;
    constexpr size_t prefixMaxLength = 256;
    std::string msg = stream_.str();
    const size_t length = std::min(msg.size(), maxLogMsgLength);
    std::string out;
    out.reserve(length + prefixMaxLength);
    AppendCurTimestamp(out, TimestampFormat::READABLE);
    auto& cfg = LogManager::GetInstance().GetComponentConfig(component_);
    if (cfg.verbose) {
        AppendPid(out);
        AppendTid(out);
        AppendComponent(out, ComponentToString(component_));
    }
    AppendLevel(out, level_);
    AppendFileLine(out, file, line);
    FilterAndAppend(out, msg.data(), length);
    LogManager::GetInstance().Push(component_, type, std::move(out));
}

void Logger::Reset()
{
    stream_.str("");
    stream_.clear();
}

// ================= LogLine =================
LogLine::LogLine(LogComponent comp, LogSeverity level, const char* file, size_t line)
    : logger_(GetThreadLogger(comp, level)), enabled_(false), file_(file), line_(line)
{
    logger_.Reset();
    enabled_ = logger_.ShouldLog();
}

LogLine::~LogLine()
{
    if (!enabled_) {
        return;
    }
    logger_.AssembleAndPush(type_, file_, line_);
    logger_.Reset();
}

// ================= thread_local Logger =================

Logger& GetThreadLogger(LogComponent comp, LogSeverity level)
{
    static thread_local std::array<std::array<Logger, static_cast<uint8_t>(LogSeverity::__COUNT__)>,
        static_cast<uint8_t>(LogComponent::__COUNT__)> loggers = [] {
        std::array<std::array<Logger, static_cast<uint8_t>(LogSeverity::__COUNT__)>,
            static_cast<uint8_t>(LogComponent::__COUNT__)
        > arr{};
        for (uint8_t c = 0; c < static_cast<uint8_t>(LogComponent::__COUNT__); ++c) {
            for (uint8_t l = 0; l < static_cast<uint8_t>(LogSeverity::__COUNT__); ++l) {
                arr[c][l] = Logger(
                    static_cast<LogComponent>(c), static_cast<LogSeverity>(l)
                );
            }
        }
        return arr;
    }();
    return loggers[static_cast<uint8_t>(comp)][static_cast<uint8_t>(level)];
}

} // namespace mindie_llm
