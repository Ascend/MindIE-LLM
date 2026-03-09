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

#include "ipc_communicator.h"

#include <experimental/filesystem>
#include <sys/stat.h>

#include "system_log.h"
#include "msServiceProfiler/msServiceProfiler.h"

using model_execute_data::ExecuteRequest;
using model_execute_data::ExecuteResponse;
using model_execute_data::ExecuteType_IsValid;
namespace fs = std::experimental::filesystem;
constexpr mode_t FULL_PERMISSION_MASK = 0777;
constexpr mode_t REQUIRED_PERMISSION = 0600;
namespace mindie_llm {
bool SerializeExecuteMessage(ExecuteRequest &request, std::string &buf)
{
    const size_t msgSize = request.ByteSizeLong();
    try {
        buf.resize(msgSize + sizeof(uint32_t));
        if (!request.SerializeToArray(buf.data(), msgSize)) {
            LOG_ERROR_LLM << "Fail to serialize protobuf message, current execute_type of request is "
                                 << request.execute_type();
            return false;
        }
    } catch (const std::exception &e) {
        LOG_ERROR_LLM << "Fail to alloc buffer, buffer length " << msgSize;
        return false;
    }
    return true;
}

IPCCommunicator::IPCCommunicator(std::string prefixName, uint32_t workerNum)
    : requestSharedMemory_(IPCSharedMemoryType::REQUEST, prefixName + "_request", workerNum),
      responseSharedMemory_(IPCSharedMemoryType::RESPONSE, prefixName + "_response", workerNum), workerNum_(workerNum)
{
}

IPCSharedMemory::IPCSharedMemory(IPCSharedMemoryType iPCSharedMemoryType, std::string prefix, uint32_t semNum)
    : sharedMemoryType(iPCSharedMemoryType), sharedMemoryName(prefix), semProduceVec(semNum, nullptr),
      semConsumeVec(semNum, nullptr)
{
    for (uint32_t i = 0; i < semNum; ++i) {
        std::string suffix = "_" + std::to_string(i);
        semProduceNameVec.push_back(prefix + "_produce" + suffix);
        semConsumeNameVec.push_back(prefix + "_consume" + suffix);
    }
}

bool IPCCommunicator::InitSemaphores(IPCSharedMemory &iPCSharedMemory) const
{
    for (uint32_t i = 0; i < iPCSharedMemory.semProduceVec.size(); i++) {
        if (sem_init(iPCSharedMemory.semProduceVec.at(i), 1, 1) != 0) {
            LOG_ERROR_LLM << "Failed to initialize produce semaphore at index " << i;
            return false;
        }
        if (sem_init(iPCSharedMemory.semConsumeVec.at(i), 1, 0) != 0) {
            LOG_ERROR_LLM << "Failed to initialize consume semaphore at index " << i;
            return false;
        }
    }
    return true;
}

bool IPCCommunicator::WriteMessage(const char *message, uint32_t length)
{
    if (!requestSharedMemory_.sharedMemory->Write(0, reinterpret_cast<const char *>(&length), sizeof(uint32_t))) {
        LOG_ERROR_LLM << "Failed to write sizeof message: " << message;
        return false;
    }
    if (!requestSharedMemory_.sharedMemory->Write(sizeof(uint32_t), message, length)) {
        LOG_ERROR_LLM << "Failed to write: " << message;
        return false;
    }
    return true;
}

bool IPCCommunicator::CreateSharedMemory(IPCSharedMemory &iPCSharedMemory, const size_t sharedMemorySize) const
{
    iPCSharedMemory.sharedMemory = std::make_unique<SharedMemory>();
    if (!iPCSharedMemory.sharedMemory->Create(iPCSharedMemory.sharedMemoryName, sharedMemorySize)) {
        LOG_ERROR_LLM << "Failed to create shared memory.";
        return false;
    }
    return true;
}

bool IPCCommunicator::CheckSemaphoreOwnerAndPermission(const std::string &semName) const
{
    fs::path semPath = fs::path("/dev/shm") / ("sem." + semName.substr(1));
    struct stat semStat;
    if (stat(semPath.c_str(), &semStat) != 0) {
        LOG_ERROR_LLM << "Failed to stat semaphore file: " << semPath;
        return false;
    }
    uid_t currentUid = getuid();
    if (semStat.st_uid != currentUid) {
        LOG_ERROR_LLM << "Semaphore " << semName << " owned by uid " << semStat.st_uid << ", but current uid is "
                                        << currentUid;
        return false;
    }
    if ((semStat.st_mode & FULL_PERMISSION_MASK) != REQUIRED_PERMISSION) {
        LOG_ERROR_LLM << "Semaphore " << semName << " permission expected 0600";
        return false;
    }
    return true;
}

void IPCCommunicator::CreateSemaphores(IPCSharedMemory &iPCSharedMemory) const
{
    const mode_t kSemPerms = REQUIRED_PERMISSION; // owner can read/write, others have no access
    for (uint32_t i = 0; i < iPCSharedMemory.semProduceVec.size(); ++i) {
        const std::string& semProduceName = iPCSharedMemory.semProduceNameVec.at(i);
        const std::string& semConsumeName = iPCSharedMemory.semConsumeNameVec.at(i);

        sem_t *semProduce = sem_open(semProduceName.c_str(), O_CREAT, kSemPerms, 0);
        if (semProduce == SEM_FAILED || !CheckSemaphoreOwnerAndPermission(semProduceName.c_str())) {
            LOG_ERROR_LLM << "semaphore create fail, name:" << semProduceName;
            sem_close(semProduce);
            sem_unlink(semProduceName.c_str());
            return;
        }
        sem_t *semConsume = sem_open(semConsumeName.c_str(), O_CREAT, kSemPerms, 0);
        if (semConsume == SEM_FAILED || !CheckSemaphoreOwnerAndPermission(semConsumeName.c_str())) {
            LOG_ERROR_LLM << "semaphore create fail, name:" << semConsumeName;
            sem_close(semConsume);
            sem_unlink(semConsumeName.c_str());
            return;
        }
        iPCSharedMemory.semProduceVec.at(i) = semProduce;
        iPCSharedMemory.semConsumeVec.at(i) = semConsume;
    }
}

bool IPCCommunicator::SetupChannel(const ShmSizeConfig &shmSizeConfig)
{
    if (!CreateSharedMemory(requestSharedMemory_, shmSizeConfig.requestShmSize) ||
        !CreateSharedMemory(responseSharedMemory_, shmSizeConfig.responseShmSize)) {
        LOG_ERROR_LLM << "Failed to create shared memory.";
        return false;
    }
    requestShmSize_ = shmSizeConfig.requestShmSize;
    responseShmSize_ = shmSizeConfig.responseShmSize;
    CreateSemaphores(requestSharedMemory_);
    CreateSemaphores(responseSharedMemory_);

    if (!InitSemaphores(requestSharedMemory_) || !InitSemaphores(responseSharedMemory_)) {
        LOG_ERROR_LLM << "Failed to initialize semaphores.";
        return false;
    }
    return true;
}

bool IPCCommunicator::StartHandleResponseThread()
{
    if (responseHandler_ == nullptr) {
        LOG_ERROR_LLM << "No response handler registered.";
        return false;
    }
    if (handleResponseThread_ && handleResponseThread_->joinable()) {
        LOG_ERROR_LLM << "Handle response thread is already running.";
        return false;
    }
    recvChannelActive_ = true;
    handleResponseThread_ = std::make_unique<std::thread>(&IPCCommunicator::HandleRcvMsg, this);
    return true;
}

bool IPCCommunicator::RegisterResponseHandler(ResponseHandler handler)
{
    if (responseHandler_ != nullptr) {
        LOG_ERROR_LLM << "A response handler is already registered.";
        return false;
    }
    responseHandler_ = handler;
    return true;
}

// Wait (P) on every semaphore in the list, decrementing each by 1
void IPCCommunicator::WaitOnAllSemaphores(std::vector<sem_t *> &semaphoreList) const
{
    for (size_t i = 0; i < semaphoreList.size(); ++i) {
        SemP(*semaphoreList[i], 1);
    }
}

// Signal (V) every semaphore in the list, incrementing each by 1
void IPCCommunicator::SignalAllSemaphores(std::vector<sem_t *> &semaphoreList) const
{
    for (size_t i = 0; i < semaphoreList.size(); ++i) {
        SemV(*semaphoreList[i], 1);
    }
}

bool IPCCommunicator::SendMessageViaSM(ExecuteRequest &request)
{
    std::string buf;
    int profExecType = request.execute_type();
    auto spanSerialize =
        PROF(INFO, Domain("Executor").SpanStart("SerializeRequests").Attr("execute_type", profExecType));
    const size_t msgSize = request.ByteSizeLong();
    const size_t maxRequestBufSize = requestShmSize_ - sizeof(uint32_t); // 8MB - 4 bytes for size
    if (msgSize > maxRequestBufSize) {
        LOG_ERROR_LLM << "The message size cannot be greater than " << maxRequestBufSize;
        return false;
    }
    if (!SerializeExecuteMessage(request, buf)) {
        LOG_ERROR_LLM << "Failed to serialize execute message.";
        PROF(spanSerialize.SpanEnd());
        return false;
    }
    PROF(spanSerialize.SpanEnd());

    // Wait until the write slot is available (produce semaphore == 1),
    // then decrement it to acquire the slot for writing.
    WaitOnAllSemaphores(requestSharedMemory_.semProduceVec);
    if (!WriteMessage(buf.data(), buf.size() - sizeof(uint32_t))) {
        // If writing fails, release the write slot by incrementing the produce semaphore,
        // allowing future retries.
        SignalAllSemaphores(requestSharedMemory_.semProduceVec);
        LOG_ERROR_LLM << "Failed to broadcast execute message.";
        return false;
    }
    // Signal that the message is ready to be read by incrementing the consume semaphore.
    SignalAllSemaphores(requestSharedMemory_.semConsumeVec);

    return true;
}

bool IPCCommunicator::ParseResponse(ExecuteResponse &executeResponse, char *sharedBuf) const
{
    uint32_t messageSize = *reinterpret_cast<uint32_t *>(sharedBuf);
    auto spanDeserialize = PROF(INFO, Domain("Executor").SpanStart("deserializeResponses"));
    if (!executeResponse.ParseFromArray(sharedBuf + sizeof(uint32_t), messageSize)) {
        LOG_ERROR_LLM << "Failed to deserialize buffer.";
        PROF(spanDeserialize.SpanEnd());
        return false;
    }
    PROF(spanDeserialize.SpanEnd());
    if (executeResponse.status() != 0) {
        LOG_ERROR_LLM << "Receive wrong status: " << executeResponse.status();
        return false;
    }
    if (!ExecuteType_IsValid(executeResponse.msg_type())) {
        LOG_ERROR_LLM << "Receive message type: " << executeResponse.msg_type();
        return false;
    }
    return true;
}

bool IPCCommunicator::ReceiveInitResponses(std::vector<ExecuteResponse> &responses)
{
    // Wait until all consume semaphores reach 1,
    // then decrement each of them to mark the response buffer as being read.
    WaitOnAllSemaphores(responseSharedMemory_.semConsumeVec);
    for (size_t i = 0; i < workerNum_; ++i) {
        ExecuteResponse response;
        if (!ParseResponse(response, responseSharedMemory_.sharedMemory->GetBuf() + i * MODEL_INIT_RESP_SIZE)) {
            LOG_ERROR_LLM << "Failed to parse init response at index: " << i;
            // Release buffer anyway so the producer isn't stuck: increment all produce semaphores by 1.
            SignalAllSemaphores(responseSharedMemory_.semProduceVec);
            return false;
        }
        responses.push_back(response);
    }
    // Signal that the response buffer is free and can be reused: increment all produce semaphores by 1.
    SignalAllSemaphores(responseSharedMemory_.semProduceVec);

    return true;
}

bool IPCCommunicator::ReceiveRecoverCommandResponses(std::vector<ExecuteResponse> &responses)
{
    // Wait until all consume semaphores reach 1,
    // then decrement each of them to mark the response buffer as being read.
    WaitOnAllSemaphores(responseSharedMemory_.semConsumeVec);
    for (size_t i = 0; i < workerNum_; ++i) {
        ExecuteResponse response;
        if (!ParseResponse(response, responseSharedMemory_.sharedMemory->GetBuf() + i * RECOVER_COMMAND_RESP_SIZE)) {
            LOG_ERROR_LLM << "Failed to parse recover command response at index: " << i;
            // Release buffer anyway so the producer isn't stuck: increment all produce semaphores by 1.
            SignalAllSemaphores(responseSharedMemory_.semProduceVec);
            return false;
        }
        responses.push_back(response);
    }
    // Signal that the response buffer is free and can be reused: increment all produce semaphores by 1.
    SignalAllSemaphores(responseSharedMemory_.semProduceVec);
    return true;
}

bool IPCCommunicator::ReceiveResponse(ExecuteResponse &response)
{
    // Wait until all consume semaphores reach 1,
    // then decrement each of them to mark the response buffer as being read.
    WaitOnAllSemaphores(responseSharedMemory_.semConsumeVec);
    bool parseResponseResult = true;
    parseResponseResult = ParseResponse(response, responseSharedMemory_.sharedMemory->GetBuf());
    // Signal that the response buffer is free and can be reused: increment all produce semaphores by 1.
    SignalAllSemaphores(responseSharedMemory_.semProduceVec);

    return parseResponseResult;
}

bool IPCCommunicator::HandleRcvMsg()
{
    pthread_setname_np(pthread_self(), "HandleRcvMsg");
    while (recvChannelActive_) {
        ExecuteResponse response;
        if (!ReceiveResponse(response)) {
            LOG_ERROR_LLM << "Failed to receive response.";
            continue;
        }
        responseHandler_(response);
    }
    LOG_WARN_LLM << "Terminating HandleRcvMsg";
    return true;
}

void IPCCommunicator::CleanUp()
{
    StopHandleResponseThread();
    CloseSemaphores(requestSharedMemory_);
    CloseSemaphores(responseSharedMemory_);
    UnlinkSemaphores(requestSharedMemory_);
    UnlinkSemaphores(responseSharedMemory_);
}

void IPCCommunicator::CloseSemaphores(IPCSharedMemory &iPCSharedMemory) const
{
    for (uint32_t i = 0; i < iPCSharedMemory.semProduceVec.size(); i++) {
        if (sem_close(iPCSharedMemory.semProduceVec.at(i)) != 0) {
            LOG_ERROR_LLM << "Failed to close produce semaphore at index " << i;
        }
        if (sem_close(iPCSharedMemory.semConsumeVec.at(i)) != 0) {
            LOG_ERROR_LLM << "Failed to close consume semaphore at index " << i;
        }
    }
}

void IPCCommunicator::UnlinkSemaphores(IPCSharedMemory &iPCSharedMemory) const
{
    for (uint32_t i = 0; i < iPCSharedMemory.semProduceVec.size(); ++i) {
        if (sem_unlink(iPCSharedMemory.semProduceNameVec.at(i).c_str()) != 0) {
            LOG_ERROR_LLM << "Failed to unlink produce semaphore " << i;
        }
        if (sem_unlink(iPCSharedMemory.semConsumeNameVec.at(i).c_str()) != 0) {
            LOG_ERROR_LLM << "Failed to unlink consume semaphore " << i;
        }
    }
}

void IPCCommunicator::StopHandleResponseThread()
{
    recvChannelActive_ = false;
    // Set semaphore to wake up the thread so it can exit.
    SignalAllSemaphores(responseSharedMemory_.semConsumeVec);
    if (handleResponseThread_ && handleResponseThread_->joinable()) {
        handleResponseThread_->join();
    }
}

} // namespace mindie_llm
