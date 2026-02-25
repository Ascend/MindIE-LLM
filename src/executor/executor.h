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
 
#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <thread>
#include "executor/executor_interface.h"
#include "communicator.h"
#include "data_type.h"

namespace mindie_llm {
struct ModelLaunchConfig {
    std::string deployType{};
    std::string executorType{};
    uint32_t npuNumPerNode{1};
    uint32_t globalWorldSize{1};
    std::vector<std::string> npuDeviceIds{};
    std::string modelInstanceType{};
    bool isMultiNodesInfer{false};
    std::vector<std::string> globalRankIds{};
    bool isMasterNode{true};
    std::string localIP{};
    std::vector<std::string> slaveIPs{};
    uint32_t npuNumPerDP{1};
    uint32_t ipcCommunicatorNum{1};
    uint32_t dp{1};
    bool layerwiseDisaggregated{false};
    std::string layerwiseDisaggregatedRoleType{};
    bool lwdMultiNodesEnable{false};
    bool isLwdMultiNodesMaster{false};
};

class Executor : public IExecutor {
public:
    Executor() : IExecutor() {}
    ~Executor() override = default;

    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    bool AsyncExecuteModel(ExecuteModelRequestPtr &modelRequest,
                           ExecuteModelResponseHandler executeModelResponseHandler = nullptr) override;

    bool AsyncTGCleanup(TGCleanupRequestPtr &TGCleanupRequest) override;

    bool AsyncEOSCleanup(TGCleanupRequestPtr &TGCleanupRequest) override;

    bool ExecutorParseConfigAndInitGRPC(std::map<std::string, std::string> &configFromManager, bool isMultiNodesInfer,
                                        size_t rankIdx) override;

    bool MasterAndSlaveModelInit(const std::map<std::string, std::string> &pdInfo) override;

    bool ExecutorInstanceInit(std::map<std::string, std::string> &configFromManager, bool isMultiNodesInfer,
                              size_t rankIdx = 0) override;

    bool SetupPDLink(PDLinkRequest &pdLinkRequest) override;

    bool ExecuteKVTransfer(PullKVRequestPtr &pullKVRequest,
                           PullKVResponseHandler pullKVResponseHandler = nullptr) override;

    bool ExecutorInstanceFinalize() override;

    uint32_t GetCpuBlockNum() const override;

    uint32_t GetNpuBlockNum() const override;

    uint32_t GetMaxPositionEmbeddings() const override;

    PDLinkResponse GetPDLinkResponse() const override;

    bool ExecutLoraRequest(LoraOperationRequest &loraOperationRequest) override;
    void ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) override;

    LoraOperationResponse GetLoraOperationResponse() const override;

private:
    bool InitWorkerProcesses(const ModelLaunchConfig &modelConfig, const std::string &sharedMemPrefix);
    bool AsyncResponseHandler(ExecuteResponse &response);
    bool ParseFromModelConfig(std::unordered_map<std::string, std::string> &config,
                              ModelLaunchConfig &modelLaunchConfig, bool isMultiNodesInfer) const;
    void LayerwiseParseFromModelConfig(std::unordered_map<std::string, std::string> &config,
                                       ModelLaunchConfig &modelLaunchConfig) const;
    bool InitIPCAndLaunchModel();
    bool InitModelExecution(std::unordered_map<std::string, std::string> &config);
    bool ExecutorModelInitAndSync();
    bool MasterSendPDInfoToSlave(const std::map<std::string, std::string> &pdInfo);
    bool SlaveSendInitResponseToMaster();
    bool MasterHandleSlaveInitResponse(ExecuteResponse &response) const;
    void RegisterExecuteModelResponseHandler(ExecuteModelResponseHandler handler);
    void RegisterPullKVResponseHandler(PullKVResponseHandler handler);
    bool HandleInitResult(std::vector<ExecuteResponse> &responses) const;
    void HandleExecuteModelResponse(ExecuteResponse &modelExecuteResponse);
    bool HandleRecoverCommandResult(RecoverCommandInfo &commandInfo, std::vector<ExecuteResponse> &responses) const;
    bool AggregatePDLinkResponses(const std::vector<ExecuteResponse> &responseVec,
                                  ExecuteResponse &aggregatedResponse) const;
    bool HandlePDLinkResponse(ExecuteResponse &executeResponse);
    void HandleKVTransferResponse(ExecuteResponse &executeResponse);
    std::vector<std::string> BuildConnectorCommand(const ModelLaunchConfig &modelConfig, const std::string &sharedMemPrefix,
        uint32_t rankInDP) const;
    bool ExecuteCommand(const std::vector<std::string> &command);
    static void ConsumePipe(FILE *pipe);
    void JoinPipeThreads();
    int GetRemoteDPRankIdx(ModelLaunchConfig &modelConfig, int rankIdx, bool intraNodeTP) const;
    uint32_t GetGRPCCommunicatorNum(ModelLaunchConfig &modelConfig, bool intraNodeTP) const;
    bool HandleLoraOperationResponse(ExecuteResponse &executeResponse);

    inline static std::atomic<uint64_t> ipcInitCounter_{0};
    bool isMultiNodesInfer_{false};
    bool isGRPCInit_{false};
    size_t dpRankIdx_{0};
    std::unordered_map<std::string, std::string> configFromManager_;
    ModelLaunchConfig modelLaunchConfig_;
    std::unique_ptr<Communicator> communicator_{nullptr};
    ExecuteModelResponseHandler executeModelResponseHandler_{nullptr};
    PullKVResponseHandler pullKVResponseHandler_{nullptr};
    std::map<RequestId, Role> requestId2Role_;
    PDLinkResponse pdLinkResponse_;
    std::vector<std::thread> pipeThreads_;
    LoraOperationResponse loraOperationResponse_;
};
} // namespace mindie_llm
#endif