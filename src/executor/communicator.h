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

#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#include <atomic>
#include <thread>

#include "grpc_communicator.h"
#include "ipc_communicator.h"

namespace mindie_llm {
using SlaveModelInitReqHandler = std::function<bool(std::map<std::string, std::string> &)>;

// Communiator has two ipc communicators for intra node commmunication through shared memeory, and a grpc communicator
// for cross node communication.
class Communicator {
public:
    Communicator(std::unordered_map<std::string, std::string> &config, bool isMultiNodesInfer, int dpRankIdx,
                 int remoteDPRankIdx, bool intraNodeTP);

    ~Communicator();

    bool InitIPCCommunicators(const std::string &sharedMemPrefix, uint32_t localWorldSize);

    bool InitGRPCCommunicator(std::unordered_map<std::string, std::string> &config,
                              ResponseHandler responseFromSlaveHandler, uint32_t grpcCommunicatorNum);

    void RegisterModelInitReqHandler(SlaveModelInitReqHandler handler);

    bool SendModelInitRequestAndReceive(ExecuteRequest &request, std::vector<ExecuteResponse> &responses);

    bool SendSharedSyncRequestAndReceive(ExecuteRequest &request, std::vector<ExecuteResponse> &responses);

    bool SendRecoverCommandRequestAndReceive(ExecuteRequest &request, std::vector<ExecuteResponse> &responses);

    bool LaunchIPCHandleResponseThreads(ResponseHandler handler);

    bool SendAsyncRequest(ExecuteRequest &request);

    bool SendAsyncRequestToRemote(ExecuteRequest &request);

    bool GRPCGetSyncResponse(ExecuteResponse &response);

    bool SendAsyncReponseToRemote(ExecuteResponse &response);

    void CleanUp();

private:
    bool LwdGRPCCommunicatorInit(std::unordered_map<std::string, std::string> &config,
                                uint32_t grpcCommunicatorNum);
    std::unique_ptr<IPCCommunicator> InitSingleIPCCommunicator(const std::string &sharedMemName,
                                                              uint32_t localWorldSize,
                                                              const ShmSizeConfig &shmSizeConfig) const;

    bool RegisterAndStartIPCHandler(std::shared_ptr<IPCCommunicator> ipcCommunicator, ResponseHandler handler) const;

    bool SlaveNodeGRPCRequestHandler(ExecuteRequest &request);

    bool SlaveNodeIPCResponseHandler(ExecuteResponse &response);

    bool SendAsyncRequestToLocal(ExecuteRequest &request);

    bool isMultiNodesInfer_;
    bool layerwiseDisaggregated_{false};
    bool isLwdMultiNodesInfer_{false};
    MasterSlaveRole msRole_;
    uint32_t slaveNum_{0};
    int dpRankIdx_;             // The rank index of the current executor in the data parallel group.
    int remoteDPRankIdx_;       // The rank index of the remote executor in the data parallel group.
    bool intraNodeTP_;          // Whether the current executor is in intra-node tensor parallel mode.
    std::string remoteSlaveIP_; // The IP address of the corresponding slave node, only valid for master node.
    SlaveModelInitReqHandler slaveModelInitReqHandler_;

    std::shared_ptr<GRPCCommunicator> grpcCommunicator_;
    std::shared_ptr<IPCCommunicator> ipcCommunicatorExecute_;
    // 非线程安全不可重入 pdlink、loraload、loraunload共用
    std::shared_ptr<IPCCommunicator> ipcCommunicatorSharedSync_;
    std::shared_ptr<IPCCommunicator> ipcCommunicatorKVTransfer_;
    std::shared_ptr<IPCCommunicator> ipcCommunicatorExecuteError_;

    void HandleExecuteErrorResponse(ResponseHandler handler) const;
    std::atomic<bool> executeErrorRecvActive_{false};
    std::unique_ptr<std::thread> handleExecuteErrorThread_{nullptr};
};

} // namespace mindie_llm
#endif