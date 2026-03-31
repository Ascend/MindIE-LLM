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

#ifndef IPC_COMMUNICATOR_H
#define IPC_COMMUNICATOR_H

#include <memory>
#include <condition_variable>

#include "executor/executor_interface.h"
#include "memory_utils.h"
#include "shared_memory.h"
#include "model_execute_data.pb.h"

namespace mindie_llm {

enum class IPCSharedMemoryType { REQUEST, RESPONSE };
struct IPCSharedMemory {
    IPCSharedMemoryType sharedMemoryType;
    std::string sharedMemoryName;
    std::unique_ptr<SharedMemory> sharedMemory;
    std::vector<std::string> semProduceNameVec;
    std::vector<sem_t *> semProduceVec;
    std::vector<std::string> semConsumeNameVec;
    std::vector<sem_t *> semConsumeVec;

    IPCSharedMemory() = default;

    IPCSharedMemory(IPCSharedMemoryType iPCSharedMemoryType, std::string prefix, uint32_t semNum);
};

class IPCCommunicator {
public:
    IPCCommunicator(std::string prefixName, const SemaphoreConfig &semConfig);
    ~IPCCommunicator() = default;

    bool SetupChannel(const ShmSizeConfig &shmSizeConfig);

    bool StartHandleResponseThread();

    bool SendMessageViaSM(ExecuteRequest &request);

    bool RegisterResponseHandler(ResponseHandler handler);

    bool ReceiveInitResponses(std::vector<ExecuteResponse> &responses);

    bool ReceiveRecoverCommandResponses(std::vector<ExecuteResponse> &responses);

    bool ReceiveResponse(ExecuteResponse &response);

    void CleanUp();

private:
    bool InitSemaphores(IPCSharedMemory &iPCSharedMemory) const;
    bool WriteMessage(const char *message, uint32_t length);
    void WaitOnAllSemaphores(std::vector<sem_t *> &semaphoreList) const;
    void SignalAllSemaphores(std::vector<sem_t *> &semaphoreList) const;
    bool ParseResponse(ExecuteResponse &executeResponse, char *sharedBuf) const;
    bool HandleRcvMsg();

    bool CheckSemaphoreOwnerAndPermission(const std::string &semName) const;
    bool CreateSharedMemory(IPCSharedMemory &iPCSharedMemory, const size_t sharedMemorySize) const;
    void CreateSemaphores(IPCSharedMemory &iPCSharedMemory) const;
    void CloseSemaphores(IPCSharedMemory &iPCSharedMemory) const;
    void UnlinkSemaphores(IPCSharedMemory &iPCSharedMemory) const;
    void StopHandleResponseThread();

    IPCSharedMemory requestSharedMemory_;
    IPCSharedMemory responseSharedMemory_;
    uint32_t responseWorkerNum_;

    bool recvChannelActive_ = false;
    std::unique_ptr<std::thread> handleResponseThread_ = nullptr;
    ResponseHandler responseHandler_ = nullptr;
    size_t requestShmSize_ = DEFAULT_SHARED_MEMORY_SIZE;
    size_t responseShmSize_ = DEFAULT_SHARED_MEMORY_SIZE;
};

bool SerializeExecuteMessage(ExecuteRequest &request, std::string &buf);
} // namespace mindie_llm

#endif
