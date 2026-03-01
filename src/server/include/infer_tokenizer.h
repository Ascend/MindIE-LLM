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

#ifndef IBIS_INFER_TOKENIZER_H
#define IBIS_INFER_TOKENIZER_H

#include <condition_variable>
#include <chrono>
#include <mutex>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <queue>
#include <semaphore.h>
#include <string>
#include <utility>
#include <sys/wait.h>
#include <sys/types.h>
#include <optional>
#include "status.h"
#include "memory_utils.h"
namespace mindie_llm {
constexpr int SHARE_ERROR_INFO_SIZE = 1024;
constexpr uint32_t MAGIC_HEAD_BEGIN = 0xFEDCBA;
constexpr uint32_t MAGIC_HEAD_FAILED = 0xFA11ED;
constexpr uint32_t SHARED_MEMORY_MAX_NAME_LEN = 255;
constexpr uint32_t TOOL_CALLS_JSON_MAX_SIZE = 32768;

enum HeadFlag : uint32_t {
    ENCODE_FLAG = 1,
    DECODE_FLAG = 2,
    DECODE_ONE_FLAG = 3,
    ENCODE_CHAT_FLAG = 4,
    STOP_FLAG = 5,
};

struct DetokenizeExtraInfo {
    std::optional<bool> isCurrentToolNameSent;
    std::optional<bool> isCurrentArgumentSent;
    std::optional<int64_t> currentToolId;
    std::optional<bool> isChatReq;
    std::optional<bool> reqEnableReasoning;
    std::optional<std::string> toolCallsJson;
    std::optional<int64_t> reasoningTokens = -1;
};

namespace detail {
    enum SharedTokenSemState : uint32_t {
        E_SEM_STATE_FREE = 0,
        E_SEM_STATE_IN_USE = 1,
        E_SEM_STATE_PRE_FREE = 2,
    };
    enum SharedTokenSemStep : uint32_t {
        E_SEM_STEP_INIT = 0,
        E_SEM_STEP_START,
        E_SEM_STEP_WAIT_HEAD,
        E_SEM_STEP_WAIT_HEAD_FAIL,
        E_SEM_STEP_WAIT_HEAD_SUCC,

        E_SEM_STEP_BIND_START,
        E_SEM_STEP_BIND_NO_IBIS,
        E_SEM_STEP_BIND_IBIS,
        E_SEM_STEP_BIND_NO_FUNC,
        E_SEM_STEP_BIND_CHECK_OK,
        E_SEM_STEP_BIND_FAIL,
        E_SEM_STEP_BIND_SUCC,

        E_SEM_STEP_START_SUCC,
        E_SEM_STEP_RUN = 100,
    };
    struct SharedTokenSemaphore {
        sem_t produce;
        sem_t consume;
        sem_t subInitialized;
        SharedTokenSemState state;
        SharedTokenSemStep step;
    };
}

struct SharedMemoryHeader {
    uint32_t magic = 0;
    HeadFlag flag;
    bool isSuccess = true;
    char errMsg[SHARE_ERROR_INFO_SIZE];
    detail::SharedTokenSemaphore sems;
    uint32_t prevDecodeIndex = 0;
    uint32_t currentDecodeIndex = 0;
    uint64_t size = 0;
    uint64_t timestamp = 0;
    bool requestEndFlag = false;
    bool skipSpecialTokens = false;
    uint32_t useToolsCall = 0;
    std::optional<bool> enableThinking = true;
    bool isCurrentToolNameSent = false;
    bool isCurrentArgumentSent = false;
    int64_t currentToolId = -1;
    bool isChatReq = false;
    std::optional<bool> reqEnableReasoning;
    std::optional<int64_t> reasoningTokens = -1;
    uint32_t toolCallsJsonSize = 0;
    char toolCallsJson[TOOL_CALLS_JSON_MAX_SIZE] = {0};
    uint8_t buffer[0];
};

#pragma GCC visibility push(default)
class InferTokenizer {
public:
    explicit InferTokenizer(std::shared_ptr<pybind11::object> tokenizer) : autoTokenizer(std::move(tokenizer)) {}

public:
    void EncodeToken(std::string &prompt, std::vector<int64_t> &tokenIds);
    void EncodeChatToken(std::string &prompt, std::optional<bool> enableThinking, std::vector<int64_t> &tokenIds);
    void DecodeToken(std::vector<int64_t> &tokenIds, std::string &outputText, SharedMemoryHeader *header);
    void DecodeOneToken(std::vector<int64_t> &tokenIds, std::string &outputText, SharedMemoryHeader *header);
    bool DownloadUrl(std::string &prompt, uint64_t reqId, std::string &msg);
    void DeleteMultimodalCache(uint64_t reqId);
    std::string MaskPathsInString(const std::string &input) const;

private:
    void ProcessTokenIds(pybind11::list &originalTokenIds, std::vector<int64_t> &tokenIds) const;
    std::shared_ptr<pybind11::object> autoTokenizer{};
};
#pragma GCC visibility pop

class ShareTokenMemory {
public:
    ShareTokenMemory() = default;
    ShareTokenMemory& operator=(const ShareTokenMemory& stm)=delete;
    ShareTokenMemory(const ShareTokenMemory& stm)=delete;
    ~ShareTokenMemory();
    int Create(const std::string &name);
    uint8_t *GetBuf();
    static bool SharedMemorySizeCheck(const uint32_t &pendingMemoryAllocationSize);
    static bool SharedMemoryNameChecker(const std::string &name);

private:
    int mFd{};
    std::string mName{};
    uint32_t mCurSize{};

    // magic(4B) + encode/decode flag(4B) + SharedTokenSemaphore + size(8B) + data
    uint8_t *mMapBuf = nullptr;
};


class TokenizerProcessPool {
public:
    static TokenizerProcessPool &GetInstance() noexcept
    {
        static TokenizerProcessPool pool;
        return pool;
    }

    bool InitTokenizerPool();
    Status Encode(const std::string &prompt, std::vector<int64_t> &tokenIds, HeadFlag flag, uint64_t &timestamp,
        std::optional<bool> enableThinking = std::nullopt);
    Status Decode(std::vector<int64_t> &tokenIds, std::string &output,
        const uint64_t &timestamp, bool useToolsCall = false, const bool &skipSpecialTokens = true,
        const DetokenizeExtraInfo &detokenizeStatus = DetokenizeExtraInfo{});
    Status DecodeOne(std::vector<int64_t> &tokenIds, std::string &output, uint32_t prevDecodeIndex,
                        uint32_t currentDecodeIndex, const uint64_t &timestamp, const bool &useToolsCall = false,
                        const bool &skipSpecialTokens = true, const bool requestEndFlag = false,
                        const DetokenizeExtraInfo &detokenizeStatus = DetokenizeExtraInfo{});
    Status TikToken(const std::string &prompt, int &numTokenId, std::vector<std::string> &tokens, bool doDecode);
    void RemoveMultimodalCache(const uint64_t &timestamp);

private:
    explicit TokenizerProcessPool() = default;
    ~TokenizerProcessPool() = default;

    bool InitProcesses();
    bool CreateChildProcesses(std::vector<pid_t> &pids);
    bool InitSharedMemory(int32_t parentPid);
    bool ProcessWorker(std::shared_ptr<ShareTokenMemory> shm);
    bool InitWorkerResource(const std::shared_ptr<ShareTokenMemory> &curMemory,
        std::shared_ptr<InferTokenizer> &tokenizer);
    pid_t GetAvailablePid();
    void ReturnPid(pid_t pid);
    bool InitSubProcessMemory(const std::shared_ptr<ShareTokenMemory> &curMemory);
    bool InitSubProcessTokenizer(const std::shared_ptr<ShareTokenMemory> &curMemory,
        std::shared_ptr<InferTokenizer> &tokenizer);
    Status DoDecode(std::vector<int64_t> &tokenIds, std::string &output, HeadFlag flag, uint32_t prevDecodeIndex,
        uint32_t currentDecodeIndex, const uint64_t &timestamp, bool useToolsCall, const bool &skipSpecialTokens,
        const bool requestEndFlag, const DetokenizeExtraInfo &detokenizeStatus = DetokenizeExtraInfo{});
    static bool ProcessEncode(SharedMemoryHeader *header, const std::shared_ptr<InferTokenizer> &tokenizer,
     HeadFlag flag);
    static bool ProcessDecode(SharedMemoryHeader *header, const std::shared_ptr<InferTokenizer> &tokenizer);
    static bool ProcessDecodeOne(SharedMemoryHeader *header, const std::shared_ptr<InferTokenizer> &tokenizer);
    static bool ProcessStop(SharedMemoryHeader &header, const std::shared_ptr<InferTokenizer> &tokenizer);
    Status FillToolCallsJson(SharedMemoryHeader *&header, const DetokenizeExtraInfo &detokenizeStatus) const;
    time_t GetEncodeTimeout() const;
    Status GetPidAndMemory(pid_t &pid, std::shared_ptr<ShareTokenMemory> &memory, SharedMemoryHeader *&header);
    Status GetPidAndMemoryOnce(pid_t &pid, std::shared_ptr<ShareTokenMemory> &memory, SharedMemoryHeader *&header);
    static bool ValidateAndConvertTokenIds(SharedMemoryHeader &header, int64_t *dataBuff, std::vector<int64_t> &tokenIds);

    // for getting a process in multi-thread safely
    std::vector<pid_t> availablePid{};
    std::mutex mutex{};
    std::condition_variable cv{};

    // for communication between processes
    std::map<uint32_t, std::shared_ptr<ShareTokenMemory>> sharedMemory_{};
    std::map<pid_t, uint32_t> pidMemoryMap{};

    std::string modelWeightPath_{};
    std::string backendType_{};
    bool trustRemoteCode_ = false;
    uint32_t tokenizerNumber_{};
};
} // namespace mindie_llm

#endif // IBIS_INFER_TOKENIZER_H
