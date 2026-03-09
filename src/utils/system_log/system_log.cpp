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
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>

#include <Python.h>

#include "safe_envvar.h"
#include "safe_path.h"
#include "string_utils.h"

namespace mindie_llm {

const std::string LLM = "llm";
static constexpr size_t DECIMAL_BASE = 10;
static constexpr size_t BUFFER_SIZE_32 = 32;
static constexpr size_t BUFFER_SIZE_256 = 256;
static constexpr size_t BUFFER_SIZE_512 = 512;
static constexpr size_t BUFFER_SIZE_2048 = 2048;

const std::array<std::string, static_cast<uint8_t>(LogSeverity::__COUNT__) - 1>& GetLogSeverityNameArray()
{
    // Note: levelNames are in the same order as LogSeverity.
    static const std::array<std::string, static_cast<uint8_t>(LogSeverity::__COUNT__) - 1> levelNames = {
        "debug", "info", "warn", "error", "critical"
    };
    return levelNames;
}

const std::unordered_set<std::string>& GetAllLogSeverity()
{
    static const std::unordered_set<std::string> values = [] {
        std::unordered_set<std::string> s;
        for (const auto& name : GetLogSeverityNameArray()) {
            s.insert(name);
        }
        return s;
    }();
    return values;
}

const std::array<std::string, static_cast<uint8_t>(LogType::__COUNT__)>& GetLogTypeNameArray()
{
    // Note: typeNames are in the same order as LogType.
    static const std::array<std::string, static_cast<uint8_t>(LogType::__COUNT__)> typeNames = {
        "general", "request", "token", "tokenizer"
    };
    return typeNames;
}

bool String2LogType(const std::string& s, LogType& out)
{
    const auto& typeNames = GetLogTypeNameArray();
    for (uint8_t i = 0; i < typeNames.size(); ++i) {
        if (typeNames[i] == s) {
            out = static_cast<LogType>(i);
            return true;
        }
    }
    return false;
}

static const std::unordered_map<LogType, std::string> logType2StrMap = {
    {LogType::GENERAL, "mindie-llm"},
    {LogType::REQUEST, "mindie-llm-request"},
    {LogType::TOKEN, "mindie-llm-token"},
    {LogType::TOKENIZER, "mindie-llm-tokenizer"}
};

const std::array<std::string, static_cast<uint8_t>(LogComponent::__COUNT__)>& GetComponentNameArray()
{
    // Note: compNames are in the same order as LogComponent.
    static const std::array<std::string, static_cast<uint8_t>(LogComponent::__COUNT__)> compNames = {
        "llm", "llmmodels", "server"
    };
    return compNames;
}

const std::string& Component2String(LogComponent c)
{
    const auto& compNames = GetComponentNameArray();
    return compNames[static_cast<uint8_t>(c)];
}

bool String2Component(const std::string& s, LogComponent& out)
{
    const auto& compNames = GetComponentNameArray();
    for (uint8_t i = 0; i < compNames.size(); ++i) {
        if (compNames[i] == s) {
            out = static_cast<LogComponent>(i);
            return true;
        }
    }
    return false;
}

enum class TimestampFormat { READABLE, TIGHT };

// ================= Log utils =================

bool String2LogSeverity(const std::string& in, LogSeverity& out)
{
    std::string key = in;
    ToLower(key);
    const auto& names = GetLogSeverityNameArray();
    for (uint8_t i = 0; i < names.size(); ++i) {
        if (names[i] == key) {
            out = static_cast<LogSeverity>(i);
            return true;
        }
    }
    return false;
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
    static constexpr size_t tightTimestampLength = 17; // YYYYMMDDHHMMSSmmm
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

inline std::string LogSeverity2String(LogSeverity level)
{
    static const std::unordered_map<LogSeverity, std::string> logSeverity2StrMap = {
        {LogSeverity::DEBUG, "DEBUG"},
        {LogSeverity::INFO, "INFO"},
        {LogSeverity::WARN, "WARN"},
        {LogSeverity::ERROR, "ERROR"},
        {LogSeverity::CRITICAL, "CRITICAL"},
        {LogSeverity::AUDIT, "AUDIT"}
    };
    auto it = logSeverity2StrMap.find(level);
    if (it != logSeverity2StrMap.end()) {
        return it->second;
    }
    return "INFO";
}

inline void AppendLevel(std::string& out, LogSeverity level)
{
    out.append(" [");
    out.append(LogSeverity2String(level));
    out.push_back(']');
}

inline void FilterAndAppend(std::string& out, const char* input, size_t length)
{
    static constexpr unsigned char kAsciiControlMax = 0x1F;
    static constexpr unsigned char kAsciiDelete = 0x7F;
    static constexpr unsigned char kLineFeed = '\n';
    static constexpr unsigned char kCarriageReturn = '\r';

    const char* cursor = input;
    const char* end = input + length;
    while (cursor < end) {
        unsigned char ch = static_cast<unsigned char>(*cursor++);
        const bool isControlChar = (ch <= kAsciiControlMax) || (ch == kAsciiDelete);
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
        static constexpr size_t fileNumLimit1 = 1;
        static constexpr size_t fileNumLimit64 = 64;
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

static std::string GetStackTrace(size_t skip)
{
    void* buffer[BUFFER_SIZE_32];
    int nptrs = ::backtrace(buffer, BUFFER_SIZE_32) - 3;
    if (nptrs <= 0 || nptrs <= static_cast<int>(skip)) {
        return "";
    }

    std::ostringstream oss;
    oss << "\nStack trace:\n";
    for (int i = skip; i < nptrs; ++i) {
        Dl_info info{};
        if (!dladdr(buffer[i], &info)) {
            oss << "#" << (i - skip) << " ??\n";
            continue;
        }
        std::string function = "??";
        if (info.dli_sname) {
            int status = 0;
            std::unique_ptr<char, void(*)(void*)> demangled(
                abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status),
                std::free
            );
            if (status == 0 && demangled) {
                function = demangled.get();
            } else {
                function = info.dli_sname;
            }
        }
        oss << "#" << (i - skip) << " " << function << "\n";
    }
    return oss.str();
}

static std::string GetPythonStackTrace()
{
    std::string result;
    PyGILState_STATE gil = PyGILState_Ensure();

    PyObject* traceback = PyImport_ImportModule("traceback");
    if (!traceback) {
        PyGILState_Release(gil);
        return result;
    }

    PyObject* formatFunc = PyObject_GetAttrString(traceback, "format_stack");
    if (!formatFunc) {
        Py_DECREF(traceback);
        PyGILState_Release(gil);
        return result;
    }

    PyObject* stackList = PyObject_CallObject(formatFunc, nullptr);
    if (!stackList) {
        Py_DECREF(formatFunc);
        Py_DECREF(traceback);
        PyGILState_Release(gil);
        return result;
    }

    PyObject* sep = PyUnicode_FromString("");
    if (!sep) {
        Py_DECREF(stackList);
        Py_DECREF(formatFunc);
        Py_DECREF(traceback);
        PyGILState_Release(gil);
        return result;
    }

    PyObject* joined = PyUnicode_Join(sep, stackList);
    if (joined) {
        result = PyUnicode_AsUTF8(joined);
        Py_DECREF(joined);
    }

    Py_DECREF(sep);
    Py_DECREF(stackList);
    Py_DECREF(formatFunc);
    Py_DECREF(traceback);
    PyGILState_Release(gil);
    return result;
}

// ================= LogManager =================

LogManager& LogManager::GetInstance()
{
    static LogManager inst;
    return inst;
}

LogManager::LogManager()
{
    Init();
}

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

void LogManager::StartAsyncThread()
{
    if (isRunning_) {
        return;
    }
    isRunning_ = true;
    flushThread_ = std::thread(&LogManager::FlushLoop, this);
    pthread_setname_np(flushThread_.native_handle(), "LogFlushThread");
}

void LogManager::RestartAfterFork()
{
    if (flushThread_.joinable()) {
        flushThread_.detach();
    }
    isRunning_ = false;
    StartAsyncThread();
}

void LogManager::LoadComponentConfigs()
{
    LoadByComponentByEnv<LogSeverity>(MINDIE_LOG_LEVEL, DEFAULT_MINDIE_LOG_LEVEL, GetAllLogSeverity(),
        [](const std::string& s) {
            LogSeverity lvl;
            if (!String2LogSeverity(s, lvl)) {
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
    if (level != LogSeverity::AUDIT) {
        auto& cfg = GetComponentConfig(comp);
        return isRunning_ && level >= cfg.minLevel && (cfg.toStdout || cfg.toFile);
    } else {
        return isRunning_;
    }
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
        CreateLogFilePath(type);
        auto& sink = sinks_[i];
        sink.ofs.open(sink.filePath, std::ios::app);
        if (sink.ofs.is_open()) {
            std::error_code ec;
            auto sz = fs::file_size(sink.filePath, ec);
            sink.curSize = ec ? 0 : static_cast<size_t>(sz);
        }
    }
}

void LogManager::CreateLogFilePath(LogType type)
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
            if (cfg.toStdout && (static_cast<LogType>(i) != LogType::TOKEN)) {
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
                if (sink.curSize + m.msg.size() + 1 >= GetLogFileSizeCutOff(static_cast<LogType>(i))) {
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

uint32_t LogManager::GetLogFileSizeCutOff(LogType type) const
{
    if (type != LogType::TOKEN) {
        return logFileSize_;
    } else {
        return SIZE_1MB;
    }
}

uint32_t LogManager::GetLogFileNumCutOff(LogType type) const
{
    if (type != LogType::TOKEN) {
        return logFileNum_;
    } else {
        uint32_t maxFileNumForTokenType = 2;
        return maxFileNumForTokenType;
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
    fs::remove(MakeRotateName(base, GetLogFileNumCutOff(type)), ec);
    ec.clear();
    for (int i = static_cast<int>(GetLogFileNumCutOff(type)) - 1; i >= 1; --i) {
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

void Logger::AssembleAndPush(LogType type, const char* file, size_t line, std::string& stack)
{
    if (stream_.tellp() == std::streampos(0)) {
        return;
    }
    std::string msg = stream_.str();
    const size_t length = std::min(msg.size(), BUFFER_SIZE_2048);
    std::string out;
    out.reserve(length + BUFFER_SIZE_256);
    AppendCurTimestamp(out, TimestampFormat::READABLE);
    auto& cfg = LogManager::GetInstance().GetComponentConfig(component_);
    if (cfg.verbose) {
        AppendPid(out);
        AppendTid(out);
        AppendComponent(out, Component2String(component_));
    }
    AppendLevel(out, level_);
    AppendFileLine(out, file, line);
    FilterAndAppend(out, msg.data(), length);
    if (!stack.empty()) {
        out.append(stack);
    }
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
    if (enabled_ && (level == LogSeverity::ERROR || level == LogSeverity::CRITICAL)) {
        stack_ = BuildStackTrace();
    }
}

LogLine::~LogLine()
{
    if (!enabled_) {
        return;
    }
    logger_.AssembleAndPush(type_, file_, line_, stack_);
    logger_.Reset();
}

std::string LogLine::BuildStackTrace()
{
    static constexpr size_t kSkip = 3;
    std::string result;

    if (Py_IsInitialized()) {
        PyGILState_STATE gil = PyGILState_Ensure();
        PyFrameObject* frame = PyEval_GetFrame();
        if (frame != nullptr) {
            result += GetPythonStackTrace();
        }
        PyGILState_Release(gil);
    }

    result += GetStackTrace(kSkip);
    return result;
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

// ================= DynamicLogManager =================

DynamicLogManager &DynamicLogManager::GetInstance()
{
    static DynamicLogManager inst;
    return inst;
}

DynamicLogManager::DynamicLogManager()
{
    Init();
}

void DynamicLogManager::Init()
{
    GetDefaultLogSeverity();
    isRunning_ = true;
    monitorThread_ = std::thread(&DynamicLogManager::Monitor, this);
    pthread_setname_np(monitorThread_.native_handle(), "DynamicLogMonitorThread");
}

DynamicLogManager::~DynamicLogManager()
{
    Stop();
}

void DynamicLogManager::Stop()
{
    if (!isRunning_) {
        return;
    }
    isRunning_ = false;
    if (monitorThread_.joinable()) {
        monitorThread_.join();
    }
}

void DynamicLogManager::GetDefaultLogSeverity()
{
    EnvVar::GetInstance().Get(MINDIE_LOG_LEVEL, DEFAULT_MINDIE_LOG_LEVEL, defaultLogSeverity_);
}

void DynamicLogManager::Monitor()
{
    while (isRunning_) {
        try {
            GetAndSetLogConfig();
        } catch (const std::exception& e) {
            std::cout << "DynamicLogManager exception: " << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(monitorInterval_));
    }
}

void DynamicLogManager::GetAndSetLogConfig()
{
    std::lock_guard<std::mutex> guard(mtx_);
    const std::string configPath = GetConfigPath();
    DynamicLogConfig newCfg = LoadLogConfig(configPath);
    if (newCfg.logSeverity.empty() && !lastLogSeverity_.empty()) {
        ResetToDefaultLogSeverity();
        return;
    }
    DynamicLogConfig lastCfg { lastLogSeverity_, lastValidHours_, lastValidTimeStamp_ };
    auto diff = DiffConfig(newCfg, lastCfg);
    
    UpdateValidTimeStamp(newCfg);
    if (!IsWithinValidRange(newCfg)) {
        ResetToDefaultLogSeverity();
        return;
    }

    if (!diff.logSeverityChanged && !diff.validHoursChanged) {
        return;
    }
    ApplyLogSeverity(newCfg.logSeverity);
    lastLogSeverity_     = newCfg.logSeverity;
    lastValidHours_      = newCfg.validHours;
    lastValidTimeStamp_  = newCfg.validTimeStamp;
}

std::string DynamicLogManager::GetConfigPath() const
{
    std::string configPath;
    Result r = EnvVar::GetInstance().Get(MINDIE_LLM_HOME_PATH, GetDefaultMindIELLMHomePath(), configPath);
    if (!r.IsOk()) {
        throw std::runtime_error(r.message());
    }
    configPath += "/conf/config.json";
    return configPath;
}

DynamicLogConfig DynamicLogManager::LoadLogConfig(const std::string& configPath)
{
    Json configJsonData;
    Result r = LoadJson(configPath, configJsonData);
    if (!r.IsOk()) {
        throw std::runtime_error(r.message());
    }
    const Json& cfgJson = configJsonData.value(keyLogConfig, Json::object());
    std::string logSeverity = GetLogSeverity(cfgJson);
    int timeInterval = GetTimeInterval(cfgJson, lastValidHours_);
    std::string timeStamp = GetTimeStamp(cfgJson, lastValidTimeStamp_);
    return {logSeverity, timeInterval, timeStamp};
}

std::string DynamicLogManager::GetLogSeverity(const Json& logConfig) const
{
    if (!logConfig.contains(keyLogSeverity) || !logConfig[keyLogSeverity].is_string()) {
        return "";
    }
    return logConfig[keyLogSeverity].get<std::string>();
}

int DynamicLogManager::GetTimeInterval(const Json& logConfig, int lastHours) const
{
    static constexpr int minValidHours = 1;
    static constexpr int maxValidHours = 168;  // A week (7 * 24)
    if (!logConfig.contains(keyTimeInterval) || !logConfig[keyTimeInterval].is_number_integer()) {
        return lastHours;
    }
    int hours = logConfig[keyTimeInterval].get<int>();
    if (hours < minValidHours || hours > maxValidHours) {
        return lastHours;
    }
    return hours;
}

std::string DynamicLogManager::GetTimeStamp(const Json& logConfig, const std::string& lastTs) const
{
    if (!logConfig.contains(keyTimeStamp) || !logConfig[keyTimeStamp].is_string()) {
        return lastTs;
    }
    const std::string ts = logConfig[keyTimeStamp].get<std::string>();
    if (!IsValidTimeFormat(ts)) {
        return lastTs;
    }
    if (!ts.empty() && IsGreaterThanNow(ts)) {
        return "";
    }
    return ts;
}

DynamicLogDiff DynamicLogManager::DiffConfig(const DynamicLogConfig& current, const DynamicLogConfig& last)
{
    return {
        current.logSeverity != last.logSeverity,
        current.validHours != last.validHours,
        current.validTimeStamp != last.validTimeStamp
    };
}

bool DynamicLogManager::IsValidTimeFormat(const std::string& timeStr) const
{
    static constexpr size_t strLen = 19;
    static constexpr int maxHour = 23;
    static constexpr int maxMinute = 59;
    static constexpr int maxSecond = 59;
    static constexpr int maxMonth = 11;
    static constexpr int maxDay = 31;
    if (timeStr.empty()) {
        return true;
    }
    if (timeStr.length() != strLen) { // "YYYY-MM-DD HH:MM:SS"
        return false;
    }
    std::tm tm = {};
    std::istringstream iss(timeStr);
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (iss.fail()) {
        return false;
    }
    if (tm.tm_hour < 0 || tm.tm_hour > maxHour || tm.tm_min  < 0 || tm.tm_min  > maxMinute ||
        tm.tm_sec  < 0 || tm.tm_sec  > maxSecond || tm.tm_mon  < 0 || tm.tm_mon  > maxMonth ||
        tm.tm_mday < 1 || tm.tm_mday > maxDay) {
        return false;
    }
    std::tm tm_copy = tm;
    std::time_t t = std::mktime(&tm_copy);
    if (t == -1) {
        return false;
    }
    return tm.tm_year == tm_copy.tm_year && tm.tm_mon  == tm_copy.tm_mon  && tm.tm_mday == tm_copy.tm_mday &&
           tm.tm_hour == tm_copy.tm_hour && tm.tm_min  == tm_copy.tm_min  && tm.tm_sec  == tm_copy.tm_sec;
}

bool DynamicLogManager::ParseTime(const std::string& s, std::time_t& out) const
{
    std::tm tm {};
    tm.tm_isdst = -1;
    std::istringstream iss(s);
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (iss.fail()) {
        return false;
    }
    out = std::mktime(&tm);
    return out != -1;
}

bool DynamicLogManager::IsGreaterThanNow(const std::string& timeStr) const
{
    std::time_t t {};
    if (!ParseTime(timeStr, t)) {
        return false;
    }
    return t > std::time(nullptr);
}

void DynamicLogManager::ResetToDefaultLogSeverity()
{
    lastLogSeverity_.clear();
    lastValidHours_ = defaultHours_;
    lastValidTimeStamp_.clear();
    ApplyLogSeverity(defaultLogSeverity_);
}

void DynamicLogManager::ApplyLogSeverity(const std::string& severity)
{
    LogManager::GetInstance().LoadByComponentByString<LogSeverity>(severity, GetAllLogSeverity(),
        [](const std::string& s) {
            LogSeverity lvl;
            if (!String2LogSeverity(s, lvl)) {
                return LogSeverity::INFO;
            }
            return lvl;
        },
        [](ComponentConfig& c, LogSeverity v) {
            c.minLevel = v;
        }
    );
}

void DynamicLogManager::UpdateValidTimeStamp(DynamicLogConfig& cfg)
{
    if (cfg.logSeverity.empty() || !cfg.validTimeStamp.empty()) {
        return;
    }
    std::time_t now = std::time(nullptr);
    std::tm tm {};
    localtime_r(&now, &tm);
    char buf[BUFFER_SIZE_32] = {};
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    cfg.validTimeStamp = buf;
}

bool DynamicLogManager::IsWithinValidRange(const DynamicLogConfig& cfg) const
{
    std::time_t start {};
    if (!ParseTime(cfg.validTimeStamp, start)) {
        return false;
    }
    const std::time_t end = start + cfg.validHours * 3600;
    const std::time_t now = std::time(nullptr);
    return now >= start && now < end;
}

// ================= InitSystemLog =================

void InitSystemLog()
{
    LogManager::GetInstance();
    DynamicLogManager::GetInstance();
}

} // namespace mindie_llm
