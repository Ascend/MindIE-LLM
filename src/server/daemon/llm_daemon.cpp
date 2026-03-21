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

#include <iostream>
#include <atomic>
#include <sys/wait.h>
#include <thread>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <csignal>
#include <cstdio>
#include <unistd.h>
#include <dirent.h>
#include <pybind11/embed.h>
#include "file_utils.h"
#include "common_util.h"
#include "log.h"
#include "log_config.h"
#include "log_level_dynamic_handler.h"
#include "config_manager.h"
#include "endpoint.h"
#include "msServiceProfiler/Tracer.h"
#include "system_log.h"

using namespace mindie_llm;
static std::mutex g_exitMtx;
static std::condition_variable g_exitCv;
static bool g_processExit = false;
static bool g_expertParallel = false;
static constexpr int EP_STOP_WAIT_TIME = 5000;
static constexpr auto COMMAND_ARGS_KEY_CONFIG_PATH = "configFilePath";
static constexpr auto COMMAND_ARGS_KEY_EXPERT_PARALLEL = "expertParallel";
static constexpr int COMMAND_ARGS_MAX_NUM = 256;
static pid_t g_mainPid = 0;
std::atomic<bool> g_isKillingAll(false);

namespace {
const std::unordered_map<std::string, std::string> COMMAND_ARGS_MAP = {{"--config-file", "configFilePath"},
                                                                       {"--expert-parallel", "expertParallel"}};
}

namespace mindie_llm {

pid_t GetPGid(pid_t pid)
{
    // Helper: Get the PGID of a process
    std::ostringstream path;
    path << "/proc/" << pid << "/stat";
    std::ifstream statFile(path.str().c_str());
    if (!statFile.is_open()) {
        return -1;
    }

    // /proc/[pid]/stat format:
    // pid (comm) state ppid pgid ...
    int pidField = 0;
    int ppidField = 0;
    int pgidField = 0;
    char state = 0;
    std::string comm;

    statFile >> pidField >> comm >> state >> ppidField >> pgidField;
    return pgidField;
}

std::vector<pid_t> GetPidsInGroup(pid_t pgid)
{
    // Get all PIDs belonging to a process group
    std::vector<pid_t> pids;

    DIR* dir = opendir("/proc");
    if (!dir) {
        perror("opendir /proc failed");
        return pids;
    }

    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        // Skip non-numeric directories
        char* endptr = nullptr;
        long pidLong = strtol(entry->d_name, &endptr, 10);
        if (*endptr != '\0' || pidLong <= 0) {
            continue;
        }

        pid_t pid = static_cast<pid_t>(pidLong);
        pid_t procPGID = GetPGid(pid);
        if (procPGID == pgid) {
            pids.push_back(pid);
        }
    }
    closedir(dir);
    return pids;
}

bool IsProcessAlive(pid_t pid)
{
    return (kill(pid, 0) == 0 || errno == EPERM);
}

void ReapZombies()
{
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        std::cout << "Reaped child " << pid << std::endl;
    }
}

void WaitForSubProcessExit(const std::vector<pid_t>& pids, int timeoutSec)
{
    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Reap any exited children
        ReapZombies();

        bool anyAlive = false;
        for (pid_t pid : pids) {
            if (pid != g_mainPid && IsProcessAlive(pid)) {
                anyAlive = true;
                break;
            }
        }

        if (!anyAlive) {
            std::cerr << "All subprocesses exited gracefully." << std::endl;
            return;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start
        ).count();
        if (elapsed >= timeoutSec) {
            std::cerr << "Timeout reached, some subprocesses still alive. Force to kill them." << std::endl;
            return;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cerr << "Waiting for subprocess exits for " << (elapsed + 1) << "/" <<
            timeoutSec << " seconds." << std::endl;
    }
}

void KillProcessGroup()
{
    bool expected = false;
    if (!g_isKillingAll.compare_exchange_strong(expected, true)) {
        return;  // Is killing all processes
    }

    std::cerr << "Daemon is killing, please wait about 15 seconds..." << std::endl;

    ULOG_AUDIT("system", MINDIE_SERVER, "stop endpoint", "success");
    ULOG_AUDIT("system", MINDIE_SERVER, "stop mindie server", "success");
    Log::Flush();

    pid_t pgid = getpgrp();
    std::vector<pid_t> pids = GetPidsInGroup(pgid);

    for (const auto& pid: pids) {
        if (pid != g_mainPid) {
            kill(pid, SIGTERM);
        }
    }

    ULOG_AUDIT("system", MINDIE_SERVER, "stop the parent process of endpoint", "success");
    kill(getppid(), SIGCHLD);

    // Wait a bit for graceful shutdown
    WaitForSubProcessExit(pids, 10);

    // Force kill remaining
    for (const auto& pid: pids) {
        if (pid != g_mainPid) {
            kill(pid, SIGKILL);
        }
    }

    ReapZombies();
    abort();
}

void SignalInterruptHandler(int sig)
{
    if (g_isKillingAll) {
        return;
    }

    ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, EXIT_SUBPROCESS_WARNING),
              "Received exit signal[" << sig << "]");

    int status = 0;
    pid_t pid = 0;
    while ((pid = waitpid(0, &status, WNOHANG)) > 0) {
        ULOG_INFO(SUBMODLE_NAME_DAEMON, "Daemon wait pid with " << pid << ", status " << status);
    }

    {
        std::unique_lock<std::mutex> lock(g_exitMtx);
        g_processExit = true;
    }
    HealthManager::UpdateHealth(false);
    g_exitCv.notify_all();
    ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, EXIT_SUBPROCESS_WARNING),
        "Successfully handled one of SIGSEGV、SIGABRT、SIGINT、SIGTERM, now killing process group");
    std::this_thread::sleep_for(std::chrono::milliseconds(EP_STOP_WAIT_TIME)); // wait for Endpoint.Stop
    KillProcessGroup();
}

void SignalChldHandler(int sig)
{
    int status = 0;
    pid_t pid = 0;
    bool exitFlag = false;
    while ((pid = waitpid(0, &status, WNOHANG)) > 0) {
        unsigned int ustatus = static_cast<unsigned int>(status);
        if (WIFEXITED(ustatus)) {
            // Exited normally
            exitFlag = false;
            int exitCode = WEXITSTATUS(ustatus);
            if (exitCode != 0) {
                exitFlag = true;
            }
        } else if (WIFSIGNALED(ustatus)) {
            // Terminated by signal
            exitFlag = true;
            int signalNum = WTERMSIG(ustatus);
            ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, SYSTEM_INVOKING_ERROR),
                "Process " << pid << " was terminated by signal " << signalNum << " (" << strsignal(signalNum) << ")");
        } else if (WIFSTOPPED(ustatus)) {
            // Stopped by signal
            exitFlag = true;
            int stopSignal = WSTOPSIG(ustatus);
            ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, SYSTEM_INVOKING_ERROR),
                "Process " << pid << " was stopped by signal " << stopSignal << " (" << strsignal(stopSignal) << ")");
        } else {
            // Other unknown
            exitFlag = true;
            ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, SYSTEM_INVOKING_ERROR),
                "Process " << pid << " terminated with unknown status " << status);
        }
    }
    if (exitFlag) {
        {
            std::unique_lock<std::mutex> lock(g_exitMtx);
            g_processExit = true;
        }
        HealthManager::UpdateHealth(false);
        g_exitCv.notify_all();

        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, EXIT_SUBPROCESS_WARNING),
            "Successfully handled SIGCHLD, now killing process group");
        KillProcessGroup();
    }
}

void SignalPipeHandler(int sig)
{
    ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, EXIT_SUBPROCESS_WARNING),
        "Received exit signal[" << sig << "]");
}

void RegisterSignal(void)
{
    sighandler_t oldSegvHandler = signal(SIGSEGV, SignalInterruptHandler); // segmentation fault
    if (oldSegvHandler == SIG_ERR) {
        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
            "Register SIGSEGV handler failed");
    }
    sighandler_t oldAbrtHandler = signal(SIGABRT, SignalInterruptHandler); // abort()
    if (oldAbrtHandler == SIG_ERR) {
        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
            "Register SIGABRT handler failed");
    }

    sighandler_t oldIntHandler = signal(SIGINT, SignalInterruptHandler);
    if (oldIntHandler == SIG_ERR) {
        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
            "Register SIGINT handler failed");
    }

    sighandler_t oldTermHandler = signal(SIGTERM, SignalInterruptHandler);
    if (oldTermHandler == SIG_ERR) {
        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
            "Register SIGTERM handler failed");
    }

    sighandler_t oldChildHandler = signal(SIGCHLD, SignalChldHandler);
    if (oldChildHandler == SIG_ERR) {
        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
            "Register SIGCHLD handler failed");
    }

    sighandler_t oldPipeHandler = signal(SIGPIPE, SignalPipeHandler);
    if (oldPipeHandler == SIG_ERR) {
        ULOG_WARN(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(WARNING, SUBMODLE_FEATURE_INIT, CHECK_WARNING),
            "Register SIGPIPE handler failed");
    }
}

void RunEP(std::unordered_map<std::string, std::string> commandLineArgsMap)
{
    PyEval_SaveThread();
    pthread_setname_np(pthread_self(), "RunEP");
    std::string fileNamePrefix = "mindie-server";
    EndPoint ep;

    if (ep.Start(commandLineArgsMap) != 0) {
        ULOG_AUDIT("system", MINDIE_SERVER, "start endpoint", "fail");
        ULOG_AUDIT("system", MINDIE_SERVER, "start mindie server", "fail");
        mindie_llm::Log::Flush();
        ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
            "Failed to init endpoint! Please check the service log or console output.");
        killpg(getpgrp(), SIGKILL);
    } else {
        ULOG_AUDIT("system", MINDIE_SERVER, "start endpoint", "success");
        std::cout << "Daemon start success!" << std::endl;
        ULOG_INFO(SUBMODLE_NAME_DAEMON, "Daemon start success!");
        HealthManager::UpdateHealth(true);
        while (!g_processExit) {
            std::unique_lock<std::mutex> lock(g_exitMtx);
            g_exitCv.wait(lock, []() { return g_processExit; });
            ep.GetHealthcheckerInstance().EnqueueErrorMessage(
                GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, SUBPROCESS_ERROR),
                SUBMODLE_NAME_DAEMON);
            if (g_processExit && g_expertParallel) {
                ULOG_INFO(SUBMODLE_NAME_DAEMON, "Update Status By ErrorItem");
                g_processExit = false;
            }
        }
        KillProcessGroup();
    }
}

bool ParseCommandLineArgs(int &argc, char **argv, std::unordered_map<std::string, std::string> &commandLineArgsMap)
{
    if (argv == nullptr || argc <= 0) {
        ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "Invalid command-line arguments.");
        return false;
    }
    if (argc > COMMAND_ARGS_MAX_NUM) {
        ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "The number of command-line arguments exceeds the limit.");
        return false;
    }
    size_t i = 1; // 从第一个参数开始解析
    size_t interval = 2;
    while (i < static_cast<size_t>(argc)) {
        const std::string configKey(argv[i]);
        auto it = COMMAND_ARGS_MAP.find(configKey);
        if (it != COMMAND_ARGS_MAP.end()) {
            if (i + 1 >= static_cast<size_t>(argc)) {
                ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                           "Missing value for key: " << configKey);
                return false;
            }
            commandLineArgsMap[it->second] = argv[i + 1];
            i += interval;
        } else {
            std::string expectedKeys = std::accumulate(COMMAND_ARGS_MAP.begin(), COMMAND_ARGS_MAP.end(),
                std::string{}, [](const std::string& totalKeys, const auto& pair) {
                    return totalKeys + (totalKeys.empty() ? "" : ", ") + pair.first;
                });
            ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                       "Unrecognized command-line argument, expect [" << expectedKeys << "]");
            return false;
        }
    }
    return true;
}
} // namespace mindie_llm

int main(int argc, char *argv[])
{
    Py_Initialize();
    InitSystemLog();
    static_assert(std::atomic<bool>::is_always_lock_free, "Bool type should be lock-free.");
    g_mainPid = getpid();
    std::cerr << "g_mainPid = " << g_mainPid << std::endl;
    spdlog::init_thread_pool(LOGGER_QUEUE_SIZE, LOGGER_THREAD_NUM);
    mindie_llm::Log::CreateAllLoggers();
    constexpr int kDynamicLogCheckIntervalMs = 5000;
    LogLevelDynamicHandler::Init(kDynamicLogCheckIntervalMs, true); // 每5秒检查动态日志配置
    std::unordered_map<std::string, std::string> commandLineArgsMap;
    if (setpgrp() == -1) {
        ULOG_ERROR(SUBMODLE_NAME_DAEMON, GenerateDaemonErrCode(ERROR, SUBMODLE_FEATURE_INIT, INIT_ERROR),
                   "Failed to set process group. errno is " << errno);
        return -1;
    }
    if (!ParseCommandLineArgs(argc, argv, commandLineArgsMap)) {
        return -1;
    }
    auto it = commandLineArgsMap.find(COMMAND_ARGS_KEY_EXPERT_PARALLEL);
    if (it != commandLineArgsMap.end() && it->second == "true") {
        g_expertParallel = true;
    }
    RegisterSignal(); // register signal
    PROF(msServiceProfiler::TraceContext::addResAttribute("service.name", "mindie.service"));
    std::thread businessThread(RunEP, commandLineArgsMap);
    businessThread.join();
    return 0;
}