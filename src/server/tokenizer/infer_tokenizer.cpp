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
#include "infer_tokenizer.h"
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <unistd.h>
#include <csignal>
#include <regex>
#include <nlohmann/json.hpp>
#include "config_manager.h"
#include "env_util.h"
#include "log.h"
#include "file_utils.h"
#include "memory_utils.h"
#include "config_manager_impl.h"
#include "safe_io.h"

using json = nlohmann::json;

namespace mindie_llm {
std::string g_tokenizerSharedMemName = "/llm_tokenizer_shared_memory_";
static constexpr uint32_t MEM_PAGE_SIZE = 4096U;

static constexpr time_t INIT_WAIT_TIME = 120;   // 启动等待时长
static constexpr time_t DECODE_WAIT_TIME = 60;  // Decode等待时长
static constexpr time_t WAIT_TIME = 60;         // Encode等待时长
static constexpr time_t MIN_WAIT_TIME = 5;
static constexpr time_t MAX_WAIT_TIME = 300;

static std::string GetModelConfigString()
{
    constexpr auto kBackendConfig = "BackendConfig";
    constexpr auto kModelDeployConfig = "ModelDeployConfig";
    constexpr auto kModelConfig = "ModelConfig";
    constexpr auto kModels = "models";

    std::string modelsString;
    try {
        auto configJson = json::parse(ConfigManager::GetInstance().GetConfigJsonStr(), CheckJsonDepthCallbackNoLogger);
        if (configJson.contains(kBackendConfig) &&
            configJson[kBackendConfig].contains(kModelDeployConfig) &&
            configJson[kBackendConfig][kModelDeployConfig].contains(kModelConfig) &&
            configJson[kBackendConfig][kModelDeployConfig][kModelConfig].is_array() &&
            !configJson[kBackendConfig][kModelDeployConfig][kModelConfig].empty() &&
            configJson[kBackendConfig][kModelDeployConfig][kModelConfig][0].contains(kModels)) {
            modelsString =
                configJson[kBackendConfig][kModelDeployConfig][kModelConfig][0][kModels].dump();
        } else {
            modelsString = "";
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("[InferTokenizer] GetModelConfigString failed : ") + e.what());
    }
    return modelsString;
}

static uint32_t GetMaxTextLength()
{
    auto serverConfig = mindie_llm::ConfigManager::GetInstance().GetServerConfig();
    uint32_t maxTextLength = serverConfig.maxRequestLength * 1024 * 1024;
    return maxTextLength;
}

// 公共函数：处理token数量限制和转换
void InferTokenizer::ProcessTokenIds(pybind11::list &originalTokenIds, std::vector<int64_t> &tokenIds) const
{
    size_t responseSize = std::min(pybind11::len(originalTokenIds), GetMaxTextLength() / sizeof(int64_t));
    if (responseSize < pybind11::len(originalTokenIds)) {
        // ULOG has been disabled in subprocess
    }
    for (size_t i = 0; i < responseSize; i++) {
        tokenIds.emplace_back(originalTokenIds[i].cast<int64_t>());
    }
}

void InferTokenizer::EncodeToken(std::string &prompt, std::vector<int64_t> &tokenIds)
{
    try {
        auto inputText = prompt.substr(0, GetMaxTextLength());
        if (inputText.length() < prompt.length()) {
            // ULOG has been disabled in subprocess
        }
        auto modelDeployParam = GetModelDeployConfig();
        if (modelDeployParam.empty()) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, INIT_ERROR),
                       "modelDeployParam is empty, please provide model deployment parameter in conf/config.json");
            return;
        }
        pybind11::dict kwargs;

        kwargs["truncation"] = modelDeployParam[0].truncation;
        kwargs["max_length"] = std::min(modelDeployParam[0].maxInputTokenLen, modelDeployParam[0].maxSeqLen - 1);
        pybind11::list originalTokenIds = autoTokenizer->attr("encode")(inputText, kwargs);
        ProcessTokenIds(originalTokenIds, tokenIds);
    } catch (const std::exception &e) {
        // ULOG has been disabled in subprocess
    } catch (...) {
        // ULOG has been disabled in subprocess
    }
}

 std::string InferTokenizer::ExtractCoreErrorMessage(const std::string& originalError)
{
    std::regex coreErrorRegex(R"(Original error: (.*?)\n\nAt:)");
    std::smatch match;
    if (std::regex_search(originalError, match, coreErrorRegex) && match.size() > 1) {
        return match.str(1);
    }
        
    return "Token encoding failed: invalid chat template syntax 33333";
}

void InferTokenizer::EncodeChatToken(std::string &prompt, std::optional<bool> enableThinking,
    std::optional<std::string> chatTemplate, std::vector<int64_t> &tokenIds)
{
    try {
        auto inputText = prompt.substr(0, GetMaxTextLength());
        if (inputText.length() < prompt.length()) {
            // ULOG has been disabled in subprocess
        }
        
        auto modelDeployParam = GetModelDeployConfig();
        if (modelDeployParam.empty()) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, INIT_ERROR),
                       "modelDeployParam is empty, please provide model deployment parameter in conf/config.json");
            return;
        }
        pybind11::dict kwargs;
        kwargs["truncation"] = modelDeployParam[0].truncation;
        kwargs["max_length"] = std::min(modelDeployParam[0].maxInputTokenLen, modelDeployParam[0].maxSeqLen - 1);
        if (enableThinking.has_value()) {
            kwargs["enable_thinking"] = enableThinking.value();
        }
        if (chatTemplate.has_value()) {
            kwargs["chat_template"] = chatTemplate.value();
        }

        pybind11::list originalTokenIds = autoTokenizer->attr("encode_chat")(inputText, kwargs);
        ProcessTokenIds(originalTokenIds, tokenIds);
    } catch (const std::exception &e) {
        std::string coreError = ExtractCoreErrorMessage(e.what());
        throw std::runtime_error("[Tokenizer] encode failed. " + coreError);
    } catch (...) {
        // ULOG has been disabled in subprocess
    }
}

void InferTokenizer::DecodeToken(std::vector<int64_t> &tokenIds, std::string &outputText, SharedMemoryHeader *header)
{
    try {
        pybind11::list pythonList;
        for (auto &tokenId : tokenIds) {
            pythonList.append(tokenId);
        }
        pybind11::dict kwargs;
        kwargs["use_tool_call"] = header->useToolsCall;
        kwargs["skip_special_tokens"] = header->skipSpecialTokens;
        kwargs["is_chat_req"] = header->isChatReq;
        if (header->reqEnableReasoning.has_value()) {
            kwargs["req_enable_thinking"] = header->reqEnableReasoning.value();
        }
        kwargs["reasoning_tokens"] = header->reasoningTokens;
        if (header->toolCallsJsonSize > 0 &&
            header->toolCallsJsonSize < TOOL_CALLS_JSON_MAX_SIZE) {
            std::string toolCallsJson(header->toolCallsJson, header->toolCallsJsonSize);
            kwargs["tool_calls_json"] = toolCallsJson;
        }
        pybind11::object text = autoTokenizer->attr("decode")(pythonList, kwargs);

        outputText = text.cast<std::string>().substr(0, GetMaxTextLength());
    } catch (...) {
        // ULOG has been disabled in subprocess
    }
}

void InferTokenizer::DecodeOneToken(std::vector<int64_t> &tokenIds, std::string &outputText, SharedMemoryHeader *header)
{
    try {
        pybind11::list pythonList;
        for (auto &tokenId : tokenIds) {
            pythonList.append(tokenId);
        }

        pybind11::dict kwargs;
        kwargs["prev_decode_index"] = header->prevDecodeIndex;
        kwargs["curr_decode_index"] = header->currentDecodeIndex;
        kwargs["use_tool_call"] = header->useToolsCall;
        kwargs["skip_special_tokens"] = header->skipSpecialTokens;
        kwargs["current_tool_name_sent"] = header->isCurrentToolNameSent;
        kwargs["current_tool_arguments_sent"] = header->isCurrentArgumentSent;
        kwargs["current_tool_id"] = header->currentToolId;
        kwargs["is_chat_req"] = header->isChatReq;
        if (header->reqEnableReasoning.has_value()) {
            kwargs["req_enable_thinking"] = header->reqEnableReasoning.value();
        }
        kwargs["reasoning_tokens"] = header->reasoningTokens;
        if (header->toolCallsJsonSize > 0 &&
            header->toolCallsJsonSize < TOOL_CALLS_JSON_MAX_SIZE) {
            std::string toolCallsJson(header->toolCallsJson, header->toolCallsJsonSize);
            kwargs["tool_calls_json"] = toolCallsJson;
        }
        kwargs["req_end_flag"] = header->requestEndFlag;
        pybind11::object text = autoTokenizer->attr("decode_one")(pythonList, kwargs);

        outputText = text.cast<std::string>().substr(0, GetMaxTextLength());
    } catch (...) {
        // ULOG has been disabled in subprocess
    }
}

std::string InferTokenizer::MaskPathsInString(const std::string &input) const
{
    // 匹配形如 /root/mindie/log.log 的路径
    std::regex pathPattern(R"(/(?:[\w\-.]+/)*[\w\-.]+)");

    std::string result;
    std::sregex_iterator begin(input.begin(), input.end(), pathPattern);
    std::sregex_iterator end;
    std::size_t lastPos = 0;

    for (auto it = begin; it != end; ++it) {
        std::smatch match = *it;
        result.append(input, lastPos, match.position() - lastPos);

        std::string masked = FileUtils::GetSafeRelativePath(match.str());
        result.append(masked);
        std::size_t matchPos = static_cast<std::size_t>(match.position());
        std::size_t matchLength = static_cast<std::size_t>(match.length());

        lastPos = matchPos + matchLength;
    }

    result.append(input, lastPos, std::string::npos);
    return result;
}

bool InferTokenizer::DownloadUrl(std::string &prompt, uint64_t reqId, std::string &msg)
{
    try {
        autoTokenizer->attr("download_url")(prompt, reqId);
        return true;
    } catch (const std::exception &e) {
        std::ostringstream oss;
        oss << "[InferTokenizer::DownloadUrl] Download fail: " << MaskPathsInString(e.what());
        std::string str = oss.str();
        size_t pos = str.find('\n');
        if (pos != std::string::npos) {
            msg = str.substr(0, pos);
        } else {
            msg = str;
        }
        return false;
    } catch (...) {
        // ULOG has been disabled in subprocess
        msg = "[InferTokenizer::DownloadUrl] Get unknown error";
        return false;
    }
}

void InferTokenizer::DeleteMultimodalCache(uint64_t reqId)
{
    try {
        autoTokenizer->attr("delete_multimodal_cache")(reqId);
    } catch (const std::exception &e) {
        // ULOG has been disabled in subprocess
    } catch (...) {
        // ULOG has been disabled in subprocess
    }
}

bool ShareTokenMemory::SharedMemorySizeCheck(const uint32_t &pendingMemoryAllocationSize)
{
    const std::string path = "/dev/shm";

    // check path exists or not
    if (!FileUtils::CheckDirectoryExists(path)) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            CHECK_ERROR), "Shared memory directory not exists.");
        return false;
    }

    // check path is a link or not
    if (FileUtils::IsSymlink(path)) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            CHECK_ERROR), "Shared memory path is symlink.");
        return false;
    }

    struct statfs buf;
    // get filesystem information by statfs function
    if (statfs(path.c_str(), &buf) == -1) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            CHECK_ERROR), "Failed to get shared memory file system information.");
        return false;
    }

    // total size of the shared memory filesystem
    uint64_t totalSize = static_cast<uint64_t>(buf.f_bsize) * buf.f_blocks;

    // available size of the shared memory filesystem
    uint64_t availSize = static_cast<uint64_t>(buf.f_bsize) * buf.f_bavail;

    if (availSize < pendingMemoryAllocationSize) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            CHECK_ERROR), "Shared memory available is not enough on the filesystem.");
        return false;
    }

    // transfer MB
    double totalSizeMb = static_cast<double>(totalSize) / (1024 * 1024);
    double availSizeMb = static_cast<double>(availSize) / (1024 * 1024);

    ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Total shared memory size is " << totalSizeMb <<
        "MB, and available shared memory size is " << availSizeMb << "MB.");
    ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Shared memory size check success.");

    return true;
}

bool ShareTokenMemory::SharedMemoryNameChecker(const std::string &name)
{
    if (name.empty() || name.length() > SHARED_MEMORY_MAX_NAME_LEN) {
        return false;
    }

    std::regex regex("^\\/[^\\/]*$");
    bool match = std::regex_match(name, regex);
    return match;
}

int ShareTokenMemory::Create(const std::string &name)
{
    this->mName = name;
    this->mCurSize = GetMaxTextLength() + sizeof(SharedMemoryHeader);
    if (!SharedMemorySizeCheck(this->mCurSize)) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Available shared memory size is not enough.");
        return -1;
    }
    if (!SharedMemoryNameChecker(name)) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Shared memory name is error. name is " << name);
        return -1;
    }
    mFd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    if (mFd < 0) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to open shared memory " << errno);
        return -1;
    }
    auto ret = shm_unlink(name.c_str());
    if (ret != 0) {
        ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(WARNING, SUBMODLE_FEATURE_TOKENIZER, CHECK_WARNING),
            "Failed to unlink shared memory " << name);
    }
    if (ftruncate(mFd, this->mCurSize) == -1) {
        close(mFd);
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to alloc shared memory size. Current size is " << this->mCurSize);
        return -1;
    }

    mMapBuf = (uint8_t *)mmap(nullptr, this->mCurSize, PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0);
    if (mMapBuf == MAP_FAILED) {
        close(mFd);
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Failed to mmap shared memory");
        return -1;
    }
    // 访问内存物理页
    for (auto pos = 0U; pos < mCurSize; pos += MEM_PAGE_SIZE) {
        mMapBuf[pos] = '\0';
    }

    ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Create share memory " << name << " success.");
    return 0;
}

uint8_t *ShareTokenMemory::GetBuf()
{
    return mMapBuf;
}

ShareTokenMemory::~ShareTokenMemory()
{
    if (mMapBuf != nullptr) {
        munmap(mMapBuf, mCurSize);
        mMapBuf = nullptr;
    }
    if (mFd >= 0) {
        close(mFd);
    }
}

bool TokenizerProcessPool::InitTokenizerPool()
{
    int32_t parentPid = getpid();
    auto modelDeployParam = GetModelDeployConfig();
    if (modelDeployParam.empty()) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            INIT_ERROR), "modelDeployParam is empty, please provide model deployment parameter in conf/config.json");
        return false;
    }
    modelWeightPath_ = modelDeployParam[0].modelWeightPath;
    backendType_ = modelDeployParam[0].backendType;
    trustRemoteCode_ = modelDeployParam[0].trustRemoteCode;
    tokenizerNumber_ = GetBackendConfig().tokenizerProcessNumber;
    return InitSharedMemory(parentPid) && InitProcesses();
}

bool TokenizerProcessPool::InitProcesses()
{
    std::vector<pid_t> pids;
    if (!CreateChildProcesses(pids)) {
        return false;
    }

    uint32_t idx = 0;
    for (pid_t pid : pids) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) != 0) {
            ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(WARNING, SUBMODLE_FEATURE_TOKENIZER,
                WATTING_SUBPROCESS_WARNING), "Failed to wait tokenizer sub process start, pid " << pid);
            continue;
        } else {
            availablePid.push_back(pid);
            // map sub process with share memory
            pidMemoryMap.insert({ pid, idx++ });
            ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Create and wait process success, sub pid " << std::to_string(pid));
        }
    }

    for (size_t del = sharedMemory_.size(); del > idx; del--) {
        sharedMemory_.erase(del - 1);
        ULOG_DEBUG(SUBMODLE_NAME_TOKENIZER, "Free share memory idx " << (del - 1));
    }

    for (auto const &kv : sharedMemory_) {
        SharedMemoryHeader *header = reinterpret_cast<SharedMemoryHeader *>(kv.second->GetBuf());
        if (header == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                ABNORMAL_TRANSMISSION_ERROR), "Cast buffer to header failed.");
            return false;
        }

        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += INIT_WAIT_TIME;
        int ret = sem_timedwait(&header->sems.subInitialized, &ts);
        if (ret == -1 || errno == ETIMEDOUT || header->magic == MAGIC_HEAD_FAILED) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                INIT_ERROR), "Timeout, Failed to init tokenizer process after " << static_cast<uint32_t>(ts.tv_sec) <<
                " seconds, step=" << static_cast<uint32_t>(header->sems.step));
            return false;
        }
        ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Success to init tokenizer process after " <<
            static_cast<uint32_t>(ts.tv_sec) << " seconds, step=" << static_cast<uint32_t>(header->sems.step));
    }

    ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Finished to init tokenizer sub process size " << availablePid.size());
    return !availablePid.empty();
}

bool TokenizerProcessPool::CreateChildProcesses(std::vector<pid_t> &pids)
{
    for (auto &shmPair : sharedMemory_) {
        sleep(0);
        PyOS_BeforeFork();
        pid_t pid = fork();
        if (pid == 0) {
            // Reset signal dispositions to default so child won't run parent's handlers
            signal(SIGTERM, SIG_DFL);
            signal(SIGINT,  SIG_DFL);
            signal(SIGCHLD, SIG_DFL);
            signal(SIGSEGV, SIG_DFL);
            signal(SIGABRT, SIG_DFL);
            signal(SIGPIPE, SIG_DFL);

            PyOS_AfterFork_Child();
            PyGILState_STATE gstate = PyGILState_Ensure();
            ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Begin to process with pid " << getpid());
            if (!ProcessWorker(shmPair.second)) {
                ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ABNORMAL_TRANSMISSION_ERROR), "Tokenizer pool process worker failed.");
                PyGILState_Release(gstate);
                return false;
            }
            PyGILState_Release(gstate);
            return true;
        }

        if (pid > 0) {
            PyOS_AfterFork_Parent();
            pids.push_back(pid);
            ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Started tokenizer sub process with pid " << pid);
        } else {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    CHECK_ERROR), "Failed to create tokenizer sub process.");
            return false;
        }
    }
    return true;
}

bool TokenizerProcessPool::InitSharedMemory(int32_t parentPid)
{
    for (uint32_t i = 0; i < tokenizerNumber_; i++) {
        std::shared_ptr<ShareTokenMemory> memory = std::make_shared<ShareTokenMemory>();
        auto memoryName = g_tokenizerSharedMemName + std::to_string(parentPid) + "_" + std::to_string(i);
        int ret = memory->Create(memoryName);
        if (ret != 0) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                ABNORMAL_TRANSMISSION_ERROR), "Failed to create tokenizer share memory, token index is "
                << i << ", status value is " << ret);
            return false;
        }

        // init sem first and then set magic
        auto header = reinterpret_cast<SharedMemoryHeader *>(memory->GetBuf());
        if (header == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                ABNORMAL_TRANSMISSION_ERROR), "Cast buffer to header failed.");
            return false;
        }

        header->sems.state = detail::E_SEM_STATE_FREE;
        header->sems.step = detail::E_SEM_STEP_INIT;
        sem_init(&header->sems.produce, 1, 0);
        sem_init(&header->sems.consume, 1, 0);
        sem_init(&header->sems.subInitialized, 1, 0);
        header->magic = MAGIC_HEAD_BEGIN;

        sharedMemory_.insert({ i, memory });
    }
    ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Init share memory done with size " << sharedMemory_.size());
    return !sharedMemory_.empty();
}

pid_t TokenizerProcessPool::GetAvailablePid()
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() -> bool { return !availablePid.empty(); });

    pid_t out = availablePid.front();
    availablePid.erase(availablePid.begin());
    return out;
}

void TokenizerProcessPool::ReturnPid(pid_t pid)
{
    std::unique_lock<std::mutex> lock(mutex);
    availablePid.push_back(pid);
    cv.notify_all();
}

// 公共函数：获取PID和对应的共享内存
Status TokenizerProcessPool::GetPidAndMemory(pid_t &pid, std::shared_ptr<ShareTokenMemory> &memory,
    SharedMemoryHeader *&header)
{
    constexpr int twice = 2;
    for (int i = 0; i < twice; i++) {
        Status status = GetPidAndMemoryOnce(pid, memory, header);
        if (!status.IsOk()) {
            return status;
        }

        detail::SharedTokenSemState state = header->sems.state;
        switch (state) {
            case detail::E_SEM_STATE_FREE: {
                return Status(Error::Code::OK);
            }
            case detail::E_SEM_STATE_IN_USE: {
                ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ABNORMAL_TRANSMISSION_ERROR), "Tokenizer is busy: " << pid);
                ReturnPid(pid);
                continue; // try to get another pid
            }
            case detail::E_SEM_STATE_PRE_FREE: {
                auto ret = sem_trywait(&header->sems.produce);
                ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ABNORMAL_TRANSMISSION_ERROR), "Try to clear produce semaphore: " << pid << ", return " << ret);
                return Status(Error::Code::OK);
            }
            default: {
                ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ABNORMAL_TRANSMISSION_ERROR), "Tokenizer state is unknown: " << pid << ", state=" << state);
                ReturnPid(pid);
                continue; // try to get another pid
            }
        }
    }
    ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
        ABNORMAL_TRANSMISSION_ERROR), "Cannot find available tokenizer.");
    return Status(Error::Code::ERROR, "Cannot find available tokenizer.");
}

Status TokenizerProcessPool::GetPidAndMemoryOnce(pid_t &pid, std::shared_ptr<ShareTokenMemory> &memory,
    SharedMemoryHeader *&header)
{
    pid = GetAvailablePid();
    auto idxIter = pidMemoryMap.find(pid);
    if (idxIter == pidMemoryMap.end()) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Cannot find pid memory index " << std::to_string(pid));
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Cannot find pid memory index " + std::to_string(pid));
    }

    auto memoryIter = sharedMemory_.find(idxIter->second);
    if (memoryIter == sharedMemory_.end()) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Cannot find mem, idx = "<< std::to_string(idxIter->second));
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Cannot find share memory");
    }

    memory = memoryIter->second;
    header = reinterpret_cast<SharedMemoryHeader *>(memory->GetBuf());
    if (header == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Cast buffer to header failed.");
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Cast buffer to header failed.");
    }

    return Status(Error::Code::OK, "Get pid and memory success");
}

time_t TokenizerProcessPool::GetEncodeTimeout() const
{
    const std::string timeoutInfo = EnvUtil::GetInstance().Get("TOKENIZER_ENCODE_TIMEOUT");
    time_t waitTime = WAIT_TIME;
    time_t minWaitTime = MIN_WAIT_TIME;
    time_t maxWaitTime = MAX_WAIT_TIME;
    if (!timeoutInfo.empty()) {
        try {
            waitTime = static_cast<time_t>(std::stoi(timeoutInfo));
            if (waitTime < minWaitTime) {
                ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(WARNING, SUBMODLE_FEATURE_TOKENIZER,
                    CHECK_WARNING), "TOKENIZER_ENCODE_TIMEOUT must be in [5, 300], use min value: 5");
                waitTime = minWaitTime;
            } else if (waitTime > maxWaitTime) {
                ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(WARNING, SUBMODLE_FEATURE_TOKENIZER,
                    CHECK_WARNING), "TOKENIZER_ENCODE_TIMEOUT must be in [5, 300], use max value: 300");
                waitTime = maxWaitTime;
            }
        } catch (const std::exception& e) {
            ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(WARNING, SUBMODLE_FEATURE_TOKENIZER,
                CHECK_WARNING), "Invalid TOKENIZER_ENCODE_TIMEOUT, use default value: 60");
            waitTime = WAIT_TIME;
            return waitTime;
        }
    }
    return waitTime;
}

Status TokenizerProcessPool::Encode(const std::string &prompt, std::vector<int64_t> &tokenIds, HeadFlag flag,
    uint64_t &timestamp, std::optional<bool> enableThinking, std::optional<std::string> chatTemplate)
{
    uint32_t maxTextLength = GetMaxTextLength();
    uint32_t maxTokenLength = maxTextLength / sizeof(int64_t);
    if (prompt.length() > maxTextLength) {
        return Status(Error::Code::ERROR, "Invalid input prompt length " + std::to_string(prompt.length()));
    }

    pid_t pid;
    std::shared_ptr<ShareTokenMemory> memory;
    SharedMemoryHeader *header;
    auto status = GetPidAndMemory(pid, memory, header);
    if (!status.IsOk()) {
        return status;
    }

    if (enableThinking != std::nullopt && enableThinking.has_value()) {
        header->enableThinking = enableThinking;
    }
    if (chatTemplate != std::nullopt && chatTemplate.has_value()) {
        size_t copy_size = std::min(chatTemplate.value().size(),
            sizeof(header->chatTemplate) - 1);
        chatTemplate.value().copy(header->chatTemplate, copy_size, 0);
        header->chatTemplate[copy_size] = '\0';
    }

    header->flag = flag;
    header->size = prompt.length();
    header->timestamp = timestamp;
    header->isSuccess = false;

    std::copy(prompt.begin(), prompt.end(), header->buffer);

    // notify sub process
    header->sems.state = detail::E_SEM_STATE_IN_USE;
    sem_post(&header->sems.consume);

    // wait sub process done
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += GetEncodeTimeout();
    int ret = sem_timedwait(&header->sems.produce, &ts);
    if (ret == -1) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Tokenizer encode wait sub process timeout. errno is " << errno);
        if (memset_s(header->buffer, maxTextLength, 0, maxTextLength) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        }
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Tokenizer encode wait sub process timeout.");
    }
    header->sems.state = detail::E_SEM_STATE_FREE;
    if (!header->isSuccess) {
        std::string errMsg = header->errMsg;
        if (memset_s(header->errMsg, SHARE_ERROR_INFO_SIZE, '\0', SHARE_ERROR_INFO_SIZE) != EOK ||
            memset_s(header->buffer, sizeof(char) * prompt.length(), '\0', sizeof(char) * prompt.length()) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        }
        ReturnPid(pid);
        return Status(Error::Code::ERROR, errMsg);
    }

    auto tokenIdSize = header->size;
    if (tokenIdSize > maxTokenLength) {
        if (memset_s(header->buffer, maxTextLength, 0, maxTextLength) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        }
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Invalid output token length " + std::to_string(tokenIdSize));
    }
    std::string response = std::string(header->chatTemplate);
    if (!response.empty()) {
        header->chatTemplate[0] = '\0';
        ReturnPid(pid);
        return Status(Error::Code::ERROR, response);
    }
    // read buffer
    tokenIds.clear();
    int64_t *outBuf = reinterpret_cast<int64_t *>(header->buffer);
    if (outBuf == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Encode cast buffer to buffer int64 failed.");
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Encode cast buffer to int64 failed.");
    }
    for (uint64_t i = 0; i < tokenIdSize; i++) {
        tokenIds.push_back(outBuf[i]);
    }
    if (memset_s(header->buffer, sizeof(int64_t) * tokenIdSize, 0, sizeof(int64_t) * tokenIdSize) != EOK) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Encode memset_s failed.");
    }
    // reset status
    header->enableThinking = std::nullopt;
    header->chatTemplate[0] = '\0';
    header->reqEnableReasoning = std::nullopt;
    ReturnPid(pid);
    ULOG_DEBUG(SUBMODLE_NAME_TOKENIZER, "Encode prompt returns " << tokenIds.size() << " tokens.");
    return Status(Error::Code::OK);
}

Status TokenizerProcessPool::Decode(std::vector<int64_t> &tokenIds, std::string &output,
    const uint64_t &timestamp, bool useToolsCall, const bool &skipSpecialTokens,
    const DetokenizeExtraInfo &detokenizeStatus)
{
    return DoDecode(tokenIds, output, DECODE_FLAG, 0, 0, timestamp,
        useToolsCall, skipSpecialTokens, true, detokenizeStatus);
}

Status TokenizerProcessPool::TikToken(const std::string &prompt, int &numTokenId,
                                      std::vector<std::string> &tokens, bool doDecode)
{
    std::vector<int64_t> tokenIds;
    uint64_t timestamp = 0;
    auto status = Encode(prompt, tokenIds, ENCODE_FLAG, timestamp);
    if (!status.IsOk()) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ENCODE_DECODE_ERROR), "Cannot encode prompt in tiktoken");
        return Status(Error::Code::ERROR, "[TokenizerPool] Cannot encode prompt in tiktoken");
    }
    numTokenId = static_cast<int>(tokenIds.size());
    if (doDecode) {
        std::vector<int64_t> postTokenId;
        std::string postSingleText;
        uint32_t prevDecodeIndex = 0;
        uint32_t currentDecodeIndex = 0;
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            postTokenId.push_back(tokenIds[i]);
            auto decodeStatus = DecodeOne(postTokenId, postSingleText, prevDecodeIndex, currentDecodeIndex, 0);
            if (!decodeStatus.IsOk()) {
                ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    ENCODE_DECODE_ERROR), "Cannot decode one in tiktoken.");
                return Status(Error::Code::ERROR, "[TokenizerPool] Cannot decode one in tiktoken");
            }
            if (!postSingleText.empty()) {
                try {
                    json j = json::parse(postSingleText, CheckJsonDepthCallbackNoLogger);
                    if (j.contains("content") && j["content"].is_string()) {
                        std::string parsedContent = j["content"];
                        tokens.push_back(parsedContent);
                    } else {
                        tokens.push_back(postSingleText);
                    }
                } catch(const json::parse_error& e) {
                    ULOG_WARN(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                    JSON_PARSE_ERROR), "Parse json error in tiktoken.");
                    tokens.push_back(postSingleText);
                }
                currentDecodeIndex = postTokenId.size();
            }
        }
    }
    return Status(Error::Code::OK);
}

Status TokenizerProcessPool::DecodeOne(std::vector<int64_t> &tokenIds, std::string &output, uint32_t prevDecodeIndex,
    uint32_t currentDecodeIndex, const uint64_t &timestamp, const bool &useToolsCall, const bool &skipSpecialTokens,
    const bool requestEndFlag, const DetokenizeExtraInfo &detokenizeStatus)
{
    return DoDecode(tokenIds, output, DECODE_ONE_FLAG, prevDecodeIndex, currentDecodeIndex, timestamp,
        useToolsCall, skipSpecialTokens, requestEndFlag, detokenizeStatus);
}

Status TokenizerProcessPool::FillToolCallsJson(SharedMemoryHeader *&header,
    const DetokenizeExtraInfo &detokenizeStatus) const
{
    // 默认清空
    header->toolCallsJsonSize = 0;
    header->toolCallsJson[0] = '\0';

    if (!detokenizeStatus.toolCallsJson.has_value()) {
        return Status(Error::Code::OK);
    }

    const std::string &jsonStr = detokenizeStatus.toolCallsJson.value();
    if (jsonStr.empty()) {
        return Status(Error::Code::OK);
    }

    const size_t maxCopyLen = static_cast<size_t>(TOOL_CALLS_JSON_MAX_SIZE - 1); // 预留 '\0'
    const size_t copyLen = std::min(jsonStr.size(), maxCopyLen);

    // 如果copyLen不够容纳整个json，打error
    if (jsonStr.size() > maxCopyLen) {
        ULOG_WARN(SUBMODLE_NAME_TOKENIZER,
            GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, ABNORMAL_TRANSMISSION_ERROR),
            "toolCallsJson too long, will be ignored. Copylen: " << copyLen);
        return Status(Error::Code::OK);
    }

    if (memcpy_s(header->toolCallsJson, TOOL_CALLS_JSON_MAX_SIZE, jsonStr.data(), copyLen) != EOK) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER,
            GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER, ABNORMAL_TRANSMISSION_ERROR),
            "Copy toolCallsJson failed");
        return Status(Error::Code::ERROR, "Copy toolCallsJson failed.");
    }

    header->toolCallsJson[copyLen] = '\0';
    header->toolCallsJsonSize = static_cast<uint32_t>(copyLen);
    return Status(Error::Code::OK); // OK
}

Status TokenizerProcessPool::DoDecode(std::vector<int64_t> &tokenIds, std::string &output, HeadFlag flag,
    uint32_t prevDecodeIndex, uint32_t currentDecodeIndex, const uint64_t &timestamp, bool useToolsCall,
    const bool &skipSpecialTokens, const bool requestEndFlag, const DetokenizeExtraInfo &detokenizeStatus)
{
    uint32_t maxTextLength = GetMaxTextLength();
    uint32_t maxTokenLength = maxTextLength / sizeof(int64_t);
    if (tokenIds.size() > maxTokenLength) {
        return Status(Error::Code::ERROR, "Invalid input token length " + std::to_string(tokenIds.size()));
    }

    pid_t pid;
    std::shared_ptr<ShareTokenMemory> memory;
    SharedMemoryHeader *header;
    auto status = GetPidAndMemory(pid, memory, header);
    if (!status.IsOk()) {
        return status;
    }

    header->flag = flag;
    header->prevDecodeIndex = prevDecodeIndex;
    header->currentDecodeIndex = currentDecodeIndex;
    header->size = tokenIds.size();
    header->timestamp = timestamp;
    header->skipSpecialTokens = skipSpecialTokens;
    header->useToolsCall = useToolsCall;
    header->isSuccess = false;
    header->isCurrentToolNameSent = detokenizeStatus.isCurrentToolNameSent.value_or(false);
    header->isCurrentArgumentSent = detokenizeStatus.isCurrentArgumentSent.value_or(false);
    header->currentToolId = detokenizeStatus.currentToolId.value_or(-1);
    header->isChatReq = detokenizeStatus.isChatReq.value_or(false);
    header->reqEnableReasoning = detokenizeStatus.reqEnableReasoning;
    header->reasoningTokens = detokenizeStatus.reasoningTokens;
    header->requestEndFlag = requestEndFlag;
    status = FillToolCallsJson(header, detokenizeStatus);
    if (!status.IsOk()) {
        ReturnPid(pid);
        return status;
    }

    int64_t *tmpBuffer = reinterpret_cast<int64_t *>(header->buffer);
    if (tmpBuffer == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "DoDecode cast buffer to dataBuff int64 failed.");
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "DoDecode cast buffer to dataBuff int64 failed.");
    }
    std::copy(tokenIds.begin(), tokenIds.end(), tmpBuffer);

    // notify sub process
    header->sems.state = detail::E_SEM_STATE_IN_USE;
    sem_post(&header->sems.consume);

    // wait sub process done
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += DECODE_WAIT_TIME;
    int ret = sem_timedwait(&header->sems.produce, &ts);
    if (ret == -1) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Tokenizer decode wait sub process timeout. errno is " << errno);
        if (memset_s(header->buffer, maxTextLength, '\0', maxTextLength) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed.");
        }
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Tokenizer decode wait sub process timeout.");
    }
    header->sems.state = detail::E_SEM_STATE_FREE;
    if (!header->isSuccess) {
        std::string errMsg = header->errMsg;
        if (memset_s(header->errMsg, SHARE_ERROR_INFO_SIZE, '\0', SHARE_ERROR_INFO_SIZE) != EOK ||
            memset_s(header->buffer, sizeof(int64_t) * tokenIds.size(), 0, sizeof(int64_t) * tokenIds.size()) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        }
        ReturnPid(pid);
        return Status(Error::Code::ERROR, errMsg);
    }

    auto textSize = header->size;
    if (textSize > maxTextLength) {
        if (memset_s(header->buffer, maxTextLength, '\0', maxTextLength) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        }
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "Invalid output prompt length " + std::to_string(textSize));
    }

    char *tempCharBuffer = reinterpret_cast<char *>(header->buffer);
    if (tempCharBuffer == nullptr) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "DoDecode cast buffer to dataBuff char failed.");
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "DoDecode cast buffer to dataBuff char failed.");
    }
    output = std::string(tempCharBuffer, textSize);
    if (memset_s(header->buffer, sizeof(char) * textSize, '\0', sizeof(char) * textSize) != EOK) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            SYSTEM_INVOKING_ERROR), "The memset_s failed");
        ReturnPid(pid);
        return Status(Error::Code::ERROR, "DoDecode memset_s failed.");
    }
    ReturnPid(pid);
    return Status(Error::Code::OK);
}

// 公共函数：验证Token ID数量并转换为vector
bool TokenizerProcessPool::ValidateAndConvertTokenIds(SharedMemoryHeader &header, int64_t *dataBuff,
    std::vector<int64_t> &tokenIds)
{
    auto tokenIdSize = header.size;
    if (tokenIdSize > GetMaxTextLength() / sizeof(int64_t)) {
        return false;
    }
    tokenIds.clear();
    for (uint64_t i = 0; i < tokenIdSize; i++) {
        tokenIds.push_back(dataBuff[i]);
    }
    return true;
}

bool TokenizerProcessPool::InitSubProcessMemory(const std::shared_ptr<ShareTokenMemory> &curMemory)
{
    uint32_t sleepTotal = 0;
    constexpr int maxSleep = 60;
    auto header = reinterpret_cast<SharedMemoryHeader *>(curMemory->GetBuf());
    if (header == nullptr) {
        return false;
    }

    header->sems.step = detail::E_SEM_STEP_WAIT_HEAD;
    while (header->magic != MAGIC_HEAD_BEGIN) {
        sleep(1);
        if (++sleepTotal > maxSleep) {
            header->sems.step = detail::E_SEM_STEP_WAIT_HEAD_FAIL;
            return false;
        }
    }
    header->sems.step = detail::E_SEM_STEP_WAIT_HEAD_SUCC;
    return true;
}

bool TokenizerProcessPool::InitSubProcessTokenizer(const std::shared_ptr<ShareTokenMemory> &curMemory,
    std::shared_ptr<InferTokenizer> &tokenizer)
{
    auto header = reinterpret_cast<SharedMemoryHeader *>(curMemory->GetBuf());
    if (header == nullptr) {
        return false;
    }
    try {
        header->sems.step = detail::E_SEM_STEP_BIND_START;
        pybind11::module module = pybind11::module_::import("mindie_llm.tokenizer");
        if (!pybind11::hasattr(module, "IbisTokenizer")) {
            header->sems.step = detail::E_SEM_STEP_BIND_NO_IBIS;
            return false;
        }

        header->sems.step = detail::E_SEM_STEP_BIND_IBIS;
        pybind11::object autoTokenizerClass = module.attr("IbisTokenizer");
        pybind11::object autoTokenizer = autoTokenizerClass(
            modelWeightPath_, backendType_, trustRemoteCode_, GetModelConfigString()
        );
        if (!pybind11::hasattr(autoTokenizer, "encode") || !pybind11::hasattr(autoTokenizer, "decode") ||
            !pybind11::hasattr(autoTokenizer, "encode_chat")) {
            header->sems.step = detail::E_SEM_STEP_BIND_NO_FUNC;
            return false;
        }

        header->sems.step = detail::E_SEM_STEP_BIND_CHECK_OK;
        tokenizer = std::make_shared<InferTokenizer>(std::make_shared<pybind11::object>(autoTokenizer));
        header->sems.step = detail::E_SEM_STEP_BIND_SUCC;
    } catch (const std::exception &e) {
        header->sems.step = detail::E_SEM_STEP_BIND_FAIL;
        return false;
    } catch (...) {
        header->sems.step = detail::E_SEM_STEP_BIND_FAIL;
        return false;
    }
    return true;
}

bool TokenizerProcessPool::InitWorkerResource(const std::shared_ptr<ShareTokenMemory> &curMemory,
    std::shared_ptr<InferTokenizer> &tokenizer)
{
    // init cur shared memory
    bool init = InitSubProcessMemory(curMemory);
    if (!init) {
        return false;
    }

    init = InitSubProcessTokenizer(curMemory, tokenizer);
    if (!init) {
        auto header = reinterpret_cast<SharedMemoryHeader *>(curMemory->GetBuf());
        if (header == nullptr) {
            return false;
        }
        header->magic = MAGIC_HEAD_FAILED;
        return false;
    }
    return true;
}

bool TokenizerProcessPool::ProcessEncode(SharedMemoryHeader *header, const std::shared_ptr<InferTokenizer> &tokenizer,
    HeadFlag flag)
{
    char *tmpStrBuffer = reinterpret_cast<char *>(header->buffer);
    if (tmpStrBuffer == nullptr) {
        return false;
    }
    auto textSize = header->size;
    if (textSize > GetMaxTextLength()) {
        return false;
    }
    std::string prompt(tmpStrBuffer, textSize);
    uint64_t timestamp = header->timestamp;

    std::vector<int64_t> tokenIds;
    std::optional<std::string> chatTemplate = std::nullopt;
    std::string errMsg;
    auto ret = tokenizer->DownloadUrl(prompt, timestamp, errMsg);
    if (!ret) {
        auto res = strncpy_s(header->errMsg, SHARE_ERROR_INFO_SIZE, errMsg.c_str(), errMsg.length());
        if (res != EOK) {
            // ULOG has been disabled in subprocess
        }
        return false;
    }
    if (flag == ENCODE_CHAT_FLAG) {
        if (header->chatTemplate != nullptr && strlen(header->chatTemplate) > 0) {
            chatTemplate = std::string(header->chatTemplate);
            header->chatTemplate[0] = '\0';
        } else {
            chatTemplate = std::nullopt;
        }
        try {
            tokenizer->EncodeChatToken(prompt, header->enableThinking, chatTemplate, tokenIds);
        } catch (const std::exception &e) {
        strncpy_s(header->chatTemplate, sizeof(header->chatTemplate),
            e.what() ? e.what() : "Unknown exception while tokenize",
            sizeof(header->chatTemplate) - 1);
        header->chatTemplate[sizeof(header->chatTemplate) - 1] = '\0';
        } catch (...) {
            // ULOG has been disabled in subprocess;
        }
    } else {
        tokenizer->EncodeToken(prompt, tokenIds);
    }

    if (tokenIds.size() > GetMaxTextLength() / sizeof(int64_t)) {
        return false;
    }
    header->size = tokenIds.size();
    int64_t *tmpBuffer = reinterpret_cast<int64_t *>(header->buffer);
    if (tmpBuffer == nullptr) {
        return false;
    }
    std::copy(tokenIds.begin(), tokenIds.end(), tmpBuffer);
    return true;
}

bool TokenizerProcessPool::ProcessDecode(SharedMemoryHeader *header, const std::shared_ptr<InferTokenizer> &tokenizer)
{
    int64_t *dataBuff = reinterpret_cast<int64_t *>(header->buffer);
    if (dataBuff == nullptr) {
        return false;
    }
    uint64_t timestamp = header->timestamp;

    std::vector<int64_t> tokenIds;
    if (!ValidateAndConvertTokenIds(*header, dataBuff, tokenIds)) {
        return false;
    }

    std::string outputText;
    tokenizer->DecodeToken(tokenIds, outputText, header);

    if (outputText.length() > GetMaxTextLength()) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Invalid text length " + std::to_string(outputText.length()));
        return false;
    }
    header->size = outputText.length();

    std::copy(outputText.begin(), outputText.end(), header->buffer);
    // delete image here
    tokenizer->DeleteMultimodalCache(timestamp);
    return true;
}

bool TokenizerProcessPool::ProcessDecodeOne(SharedMemoryHeader *header, const std::shared_ptr<InferTokenizer>
 &tokenizer)
{
    int64_t *dataBuff = reinterpret_cast<int64_t *>(header->buffer);
    if (dataBuff == nullptr) {
        return false;
    }

    std::vector<int64_t> tokenIds;
    if (!ValidateAndConvertTokenIds(*header, dataBuff, tokenIds)) {
        return false;
    }

    std::string outputText;
    tokenizer->DecodeOneToken(tokenIds, outputText, header);

    if (outputText.length() > GetMaxTextLength()) {
        return false;
    }
    header->size = outputText.length();
    std::copy(outputText.begin(), outputText.end(), header->buffer);
    uint64_t timestamp = header->timestamp;
    if (header->requestEndFlag) {
        tokenizer->DeleteMultimodalCache(timestamp);
    }
    return true;
}

void TokenizerProcessPool::RemoveMultimodalCache(const uint64_t &timestamp)
{
    pid_t pid;
    std::shared_ptr<ShareTokenMemory> memory;
    SharedMemoryHeader *header;
    auto status = GetPidAndMemory(pid, memory, header);
    if (!status.IsOk()) {
        return;
    }

    header->flag = STOP_FLAG;
    header->timestamp = timestamp;
    header->isSuccess = false;
    ULOG_INFO(SUBMODLE_NAME_TOKENIZER, "Removing multimodal cache, timestamp: " << timestamp);

    // notify sub process
    header->sems.state = detail::E_SEM_STATE_IN_USE;
    sem_post(&header->sems.consume);

    // wait sub process done
    uint32_t maxTextLength = GetMaxTextLength();
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 60U;
    int ret = sem_timedwait(&header->sems.produce, &ts);
    if (ret == -1) {
        ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
            ABNORMAL_TRANSMISSION_ERROR), "Tokenizer removecache wait sub process timeout. errno is " << errno);
        if (memset_s(header->buffer, maxTextLength, '\0', maxTextLength) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed.");
        }
        ReturnPid(pid);
        return;
    }
    header->sems.state = detail::E_SEM_STATE_FREE;
    if (!header->isSuccess) {
        std::string errMsg = header->errMsg;
        if (memset_s(header->errMsg, SHARE_ERROR_INFO_SIZE, '\0', SHARE_ERROR_INFO_SIZE) != EOK ||
            memset_s(header->buffer, maxTextLength, '\0', maxTextLength) != EOK) {
            ULOG_ERROR(SUBMODLE_NAME_TOKENIZER, GenerateTokenizerErrCode(ERROR, SUBMODLE_FEATURE_TOKENIZER,
                SYSTEM_INVOKING_ERROR), "The memset_s failed");
        }
        ReturnPid(pid);
        return;
    }
    ReturnPid(pid);
}

bool TokenizerProcessPool::ProcessStop(SharedMemoryHeader &header, const std::shared_ptr<InferTokenizer> &tokenizer)
{
    uint64_t timestamp = header.timestamp;
    tokenizer->DeleteMultimodalCache(timestamp);
    return true;
}

bool TokenizerProcessPool::ProcessWorker(std::shared_ptr<ShareTokenMemory> shm)
{
    // init cur shared memory and tokenizer
    std::shared_ptr<ShareTokenMemory> curMemory{ std::move(shm) };
    std::shared_ptr<InferTokenizer> tokenizer;

    auto header = reinterpret_cast<SharedMemoryHeader *>(curMemory->GetBuf());
    if (header == nullptr) {
        return false;
    }
    header->sems.step = detail::E_SEM_STEP_START;

    sharedMemory_.clear();
    if (!InitWorkerResource(curMemory, tokenizer)) {
        return false;
    }

    header->sems.step = detail::E_SEM_STEP_START_SUCC;
    sem_post(&header->sems.subInitialized);
    while (header->magic == MAGIC_HEAD_BEGIN) {
        header->sems.step = detail::E_SEM_STEP_RUN;
        sem_wait(&header->sems.consume);
        bool succeeded = true;
        if (header->flag == ENCODE_FLAG || header->flag == ENCODE_CHAT_FLAG) {
            succeeded = ProcessEncode(header, tokenizer, header->flag);
        } else if (header->flag == DECODE_FLAG) {
            succeeded = ProcessDecode(header, tokenizer);
        } else if (header->flag == DECODE_ONE_FLAG) {
            succeeded = ProcessDecodeOne(header, tokenizer);
        } else if (header->flag == STOP_FLAG) {
            succeeded = ProcessStop(*header, tokenizer);
        } else {
            // ULOG has been disabled in subprocess
        }
        header->isSuccess = succeeded;
        header->sems.state = detail::E_SEM_STATE_PRE_FREE;
        sem_post(&header->sems.produce);
    }
    
    killpg(getpgrp(), SIGTERM);

    return true;
}
} // namespace mindie_llm