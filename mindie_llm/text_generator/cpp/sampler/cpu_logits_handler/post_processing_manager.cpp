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
 
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <absl/types/span.h>
#include "check_utils.h"
#include "log.h"
#include "log_level_dynamic_handler.h"
#include "post_processing_manager.h"

namespace {

const std::uint64_t MAX_HOST_TO_DEVICE_SIZE = 10000000000;
const int MAX_BATCH_SIZE = 100000;
const int MAX_NUM_LOGPROBS = 20;
const int DEFAULT_TOKEN_ID = -1;
const float DEFAULT_LOGPROB = -9999.0;

enum class ErrorType {
    DTYPE_ERROR,
    MALLOC_SIZE_ERROR,
    MALLOC_ERROR,
    MEMCPY_ERROR,
    OVERFLOW_ERROR,
    UNKNOWN_ERROR,
    SUCCESS,
};

union VPtrUInt64 {
    void* vPtrVal;
    uint64_t uInt64Val;

    explicit VPtrUInt64(uint64_t uInt64Val) : uInt64Val(uInt64Val) {}
};

template <typename T>
T* GetPyArray(py::array_t<T> &pyArray, const int &dim0Size, const int &dim1Size = 0, const T &defVal = 0)
{
    if (dim0Size > MAX_BATCH_SIZE) {
        throw std::invalid_argument("The batch size is greater than the MAX_BATCH_SIZE.");
    }
    if (dim1Size == 0) {
        auto emptyVectorSPtr = std::make_shared<std::vector<T>>(dim0Size, defVal);
        pyArray = py::array_t<T>(emptyVectorSPtr->size(), emptyVectorSPtr->data());
    } else {
        auto emptyVectorSPtr = std::make_shared<std::vector<T>>(dim0Size * dim1Size, defVal);
        pyArray = py::array_t<T>({dim0Size, dim1Size}, emptyVectorSPtr->data());
    }
    py::buffer_info arrBuf = pyArray.request();
    T *arrPtr = static_cast<T *>(arrBuf.ptr);
    return arrPtr;
}

ErrorType GetMemSizeByDtype(const unsigned int &uBatchScoreSize, const std::string &dtype, size_t &h2dSize)
{
    switch (mindie_llm::cpu_logits_handler::GetDtype(dtype)) {
        case mindie_llm::cpu_logits_handler::Dtype::FLOAT16:
            h2dSize = mindie_llm::CheckIntMulOverFlow(uBatchScoreSize, sizeof(int16_t));
            break;

        case mindie_llm::cpu_logits_handler::Dtype::BFLOAT16:
            h2dSize = mindie_llm::CheckIntMulOverFlow(uBatchScoreSize, sizeof(int16_t));
            break;

        case mindie_llm::cpu_logits_handler::Dtype::FLOAT32:
            h2dSize = mindie_llm::CheckIntMulOverFlow(uBatchScoreSize, sizeof(float));
            break;

        default:
            return ErrorType::DTYPE_ERROR;
    }
    return ErrorType::SUCCESS;
}

ErrorType ScoreCpy(const unsigned int &uBatchScoreSize, const std::string &dtype,
                   void *&hostAddrScore, void* const &deviceAddrScore)
{
    size_t h2dSize = 0;
    ErrorType h2dSizeError = GetMemSizeByDtype(uBatchScoreSize, dtype, h2dSize);
    if (h2dSizeError != ErrorType::SUCCESS) { return h2dSizeError; }
    if (h2dSize == 0 || h2dSize > MAX_HOST_TO_DEVICE_SIZE) { return ErrorType::MALLOC_SIZE_ERROR; }
    hostAddrScore = malloc(h2dSize);
    if (hostAddrScore == nullptr) { return ErrorType::MALLOC_ERROR; }
    aclError ret = aclrtMemcpy(hostAddrScore, h2dSize, deviceAddrScore, h2dSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) { return ErrorType::MEMCPY_ERROR; }
    return ErrorType::SUCCESS;
}

ErrorType IndexCpy(const unsigned int &uBatchScoreSize, void *&hostAddrIndex, void* const &deviceAddrIndex)
{
    size_t h2dSize = mindie_llm::CheckIntMulOverFlow(uBatchScoreSize, sizeof(int64_t));
    if (h2dSize == 0 || h2dSize > MAX_HOST_TO_DEVICE_SIZE) { return ErrorType::MALLOC_SIZE_ERROR; }
    hostAddrIndex = malloc(h2dSize);
    if (hostAddrIndex == nullptr) { return ErrorType::MALLOC_ERROR; }
    aclError ret = aclrtMemcpy(hostAddrIndex, h2dSize, deviceAddrIndex, h2dSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) { return ErrorType::MEMCPY_ERROR; }
    return ErrorType::SUCCESS;
}

ErrorType ScoreIndexCpy(int batchSize, int scoreSize, bool speedMode, std::string dtype, void *&hostAddrScore,
                        void *&hostAddrIndex, void* const &deviceAddrScore, void* const &deviceAddrIndex)
{
    try {
        int batchScoreSize = mindie_llm::CheckIntMulOverFlow(batchSize, scoreSize);
        if (batchScoreSize < 0) { return ErrorType::MALLOC_SIZE_ERROR; }
        unsigned int uBatchScoreSize = static_cast<unsigned int>(batchScoreSize);
        ErrorType scoreCpyError = ScoreCpy(uBatchScoreSize, dtype, hostAddrScore, deviceAddrScore);
        if (scoreCpyError != ErrorType::SUCCESS) { return scoreCpyError; }

        if (!speedMode) {
            ErrorType indexCpyError = IndexCpy(uBatchScoreSize, hostAddrIndex, deviceAddrIndex);
            if (indexCpyError != ErrorType::SUCCESS) { return indexCpyError; }
        }
    } catch (const std::overflow_error &e) {
        return ErrorType::OVERFLOW_ERROR;
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_ERROR("Unknown error occurs in ScoreIndexCpy: " << e.what());
        return ErrorType::UNKNOWN_ERROR;
    }

    return ErrorType::SUCCESS;
}

void RunFunc(void *arg)
{
    mindie_llm::cpu_logits_handler::PostProcessing *task =
    static_cast<mindie_llm::cpu_logits_handler::PostProcessing *>(arg);
    if (task == nullptr) {
        MINDIE_LLM_LOG_ERROR("Task is nullptr in RunFunc.");
        throw std::runtime_error("Task is nullptr in RunFunc.");
    }
    task->Run();
}

class MemoryGuard {
public:
    explicit MemoryGuard(void *memoryPtr) : memoryPtr(memoryPtr) {};

    ~MemoryGuard() {
        if (memoryPtr != nullptr) {
            free(memoryPtr);
            memoryPtr = nullptr;
        }
    }

    void *memoryPtr;
};

}  // namespace

void PostProcessingManager::Init()
{
    spdlog::init_thread_pool(mindie_llm::LOGGER_QUEUE_SIZE, mindie_llm::LOGGER_THREAD_NUM);
    mindie_llm::Log::CreateInstance(mindie_llm::LoggerType::MINDIE_LLM);
    mindie_llm::LogLevelDynamicHandler::Init(5000); // 每5秒检查动态日志配置
    MINDIE_LLM_LOG_INFO("Get post processing manager");
    m_pool = new mindie_llm::cpu_logits_handler::ThreadPool(threadNum);
    m_pool->Init();

    if (aclrtSetDevice(deviceId) != ACL_ERROR_NONE) {
        MINDIE_LLM_LOG_ERROR("Open device failed. device id = " << deviceId);
    }
}

PostProcessingManager::~PostProcessingManager()
{
    MINDIE_LLM_LOG_INFO("Destroy post processing manager");
    if (m_pool != nullptr) {
        delete m_pool;
        m_pool = nullptr;
    }
    aclError ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        MINDIE_LLM_LOG_ERROR("Device resetting failed with error number " << ret);
    }
    aclFinalize();
}

std::pair<py::array_t<int>, py::array_t<float>> PostProcessingManager::NextTokenChooser(py::array_t<int> requestIds,
    uint64_t scoresAddr, uint64_t indexAddr, int batchSize, int scoreSize, int maxLogprobs, std::string dtype,
    bool speedMode, bool useApproxIn)
{
    if (threadNum < 1) { throw std::invalid_argument("The number of tasks is less than 1."); }
    int batchThread = batchSize / threadNum;
    int mod = batchSize % threadNum;
    auto task = std::make_unique<mindie_llm::cpu_logits_handler::PostProcessing[]>(threadNum);
    if (maxLogprobs > MAX_NUM_LOGPROBS) {
        throw std::invalid_argument("The maxLogprobs exceeds the safe maxNumLogprobs.");
    }

    int *rPtr = GetPyArray<int>(result, batchSize, maxLogprobs + 1, DEFAULT_TOKEN_ID);
    if (rPtr == nullptr) {
        throw std::invalid_argument("The rPtr casted from result is nullptr.");
    }
    float *logprobsPtr = GetPyArray<float>(logprobs, batchSize, maxLogprobs + 1, DEFAULT_LOGPROB);
    if (logprobsPtr == nullptr) {
        throw std::invalid_argument("The logprobsPtr casted from logprobs is nullptr.");
    }
    py::buffer_info requestIdsBuf = requestIds.request();
    int *requestIdsPtr = static_cast<int *>(requestIdsBuf.ptr);
    if (requestIdsPtr == nullptr) {
        throw std::invalid_argument("The requestIdsPtr casted from requestIds is nullptr.");
    }
    
    int requestIdsSize = requestIdsBuf.shape[0];
    UpdateRandomValue(batchSize, requestIdsPtr, requestIdsSize);

    VPtrUInt64 deviceScore(scoresAddr);
    VPtrUInt64 deviceIndex(indexAddr);

    MemoryGuard hostAddrScoreGuard = MemoryGuard(nullptr);
    MemoryGuard hostAddrIndexGuard = MemoryGuard(nullptr);
    ErrorType cpyError = ScoreIndexCpy(batchSize, scoreSize, speedMode, dtype, hostAddrScoreGuard.memoryPtr,
                                       hostAddrIndexGuard.memoryPtr, deviceScore.vPtrVal, deviceIndex.vPtrVal);
    if (cpyError != ErrorType::SUCCESS) { throw std::runtime_error("sampling error: score copying failed."); }

    try {
        uint16_t *scores16Ptr = (uint16_t *)(hostAddrScoreGuard.memoryPtr);
        float *scores32Ptr = (float *)(hostAddrScoreGuard.memoryPtr);
        uint64_t *indexPtr = (uint64_t *)(hostAddrIndexGuard.memoryPtr);

        int size = 0;
        int sizeOffset = 0;
        for (int i = 0; i < threadNum; i++) {
            int perbatchThread = batchThread;
            if (mod > 0) {
                perbatchThread++;
                mod--;
            }

            sizeOffset = mindie_llm::CheckIntMulOverFlow(scoreSize, size);
            absl::Span<int> requestIdsSpan = absl::MakeSpan(requestIdsPtr + size, perbatchThread);
            task[i].Init(&dictConf, requestIdsSpan, scores16Ptr + sizeOffset, scores32Ptr + sizeOffset,
                         indexPtr + sizeOffset, scoreSize, rPtr + size * (maxLogprobs + 1),
                         logprobsPtr + size * (maxLogprobs + 1), perbatchThread, maxLogprobs, dtype,
                         speedMode, useApproxIn);
            void *arg = static_cast<mindie_llm::cpu_logits_handler::PostProcessing *>(&task[i]);
            m_pool->AddTask(mindie_llm::cpu_logits_handler::Task(RunFunc, arg));
            size += perbatchThread;
        }
        m_pool->Join();
    } catch (const std::overflow_error &e) {
        MINDIE_LLM_LOG_ERROR("Overflow occurs in NextTokenChooser.");
        throw;
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_ERROR("Error occurs in NextTokenChooser: " << e.what());
        throw;
    }
    return std::pair<py::array_t<int>, py::array_t<float>>(result, logprobs);
}

void PostProcessingManager::SetBatchConfigs(py::array_t<int> requestIds, py::array_t<int> topK, py::array_t<float> topP,
    py::array_t<bool> sample, py::array_t<int> numLogprobs, py::array_t<unsigned long long> seed,
    std::string sampleMethod)
{
    auto size = requestIds.size();
    if ((size != topK.size()) || (size != topP.size()) || (size != sample.size()) || (size != seed.size())) {
        throw std::invalid_argument("The size of params requestIds, topK, topP, sample and seed not equal.");
    }
    if (size != numLogprobs.size()) {
        throw std::invalid_argument("The size of numLogprobs is not equal to requestIds.");
    }
    py::buffer_info topKBuff = topK.request();
    int *topKPtr = static_cast<int *>(topKBuff.ptr);
    py::buffer_info topPBuff = topP.request();
    float *topPPtr = static_cast<float *>(topPBuff.ptr);
    py::buffer_info sampleBuff = sample.request();
    bool *samplePtr = static_cast<bool *>(sampleBuff.ptr);
    py::buffer_info logprobsBuff = numLogprobs.request();
    int *logprobsPtr = static_cast<int *>(logprobsBuff.ptr);
    if (logprobsPtr == nullptr) {
        throw std::invalid_argument("The logprobsPtr casted from numLogprobs is nullptr.");
    }
    py::buffer_info seedBuff = seed.request();
    unsigned long long *seedPtr = static_cast<unsigned long long *>(seedBuff.ptr);
    py::buffer_info requestIdsBuff = requestIds.request();
    int* requestIdsPtr = static_cast<int *>(requestIdsBuff.ptr);

    for (long int i = 0; i < size; i++) {
        int key = *(requestIdsPtr + i);
        auto it = dictConf.find(key);
        if (it != dictConf.end()) {
            it->second = mindie_llm::cpu_logits_handler::Configure(topKPtr[i],
            topPPtr[i], samplePtr[i], logprobsPtr[i], seedPtr[i], sampleMethod);
        } else {
            dictConf.emplace(key, mindie_llm::cpu_logits_handler::Configure(topKPtr[i],
            topPPtr[i], samplePtr[i], logprobsPtr[i], seedPtr[i], sampleMethod));
        }
    }
}

void PostProcessingManager::DeleteConf(py::array_t<int> requestIdsD)
{
    auto size = requestIdsD.size();
    py::buffer_info requestIdsDBuff = requestIdsD.request();
    int* requestIdsDPtr = static_cast<int *>(requestIdsDBuff.ptr);

    for (long int i = 0; i < size; i++) {
        dictConf.erase(requestIdsDPtr[i]);
        MINDIE_LLM_LOG_DEBUG("Delete conf for " << requestIdsDPtr[i]);
    }
}

void PostProcessingManager::UpdateRandomValue(int batchSize, int *requestIdsPtr, int requestIdsSize)
{
    std::unordered_set<int> closedReqIds;
    int loopSize = std::min(batchSize, requestIdsSize);
    for (int i = 0; i < loopSize; i++) {
        if (closedReqIds.find(requestIdsPtr[i]) == closedReqIds.end()) {
            closedReqIds.insert(requestIdsPtr[i]);
            auto it = dictConf.find(requestIdsPtr[i]);
            if (it != dictConf.end()) {
                it->second.UpdateRandomValue();
            }
        }
    }
}
