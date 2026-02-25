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
#include <future>
#include <algorithm>
#include <functional>
#include "string_utils.h"
#include "log.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "config_manager.h"
#include "communicator.h"

namespace mindie_llm {

Communicator::Communicator(std::unordered_map<std::string, std::string> &config, bool isMultiNodesInfer, int dpRankIdx,
                           int remoteDPRankIdx, bool intraNodeTP)
    : isMultiNodesInfer_(isMultiNodesInfer), dpRankIdx_(dpRankIdx), remoteDPRankIdx_(remoteDPRankIdx),
      intraNodeTP_(intraNodeTP)
{
    std::string lwdRoletype = "";
    auto lwdIt = config.find("layerwiseDisaggregated");
    if (lwdIt != config.end() && lwdIt->second == "true") {
        layerwiseDisaggregated_ = true;
        lwdRoletype = config.at("layerwiseDisaggregatedRoleType");
    }

    auto lwdMultiIt = config.find("layerwiseDisaggregatedMultiNodesInferEnabled");
    if (lwdMultiIt != config.end() && lwdMultiIt->second == "true") {
        isLwdMultiNodesInfer_ = true;
    }

    msRole_ = MasterSlaveRole::MASTER; // Default to master role for single node inference.
    if ((isMultiNodesInfer && config.at("isMaster") == "0") or (lwdRoletype == "slave")) {
        msRole_ = MasterSlaveRole::SLAVE;
    }

    if (isMultiNodesInfer_ && msRole_ == MasterSlaveRole::MASTER) {
        std::vector<std::string> slaveIPs;
        mindie_llm::Split(config.at("slaveIPs"), ",", slaveIPs);
        size_t slaveCount = slaveIPs.size();
        size_t dpNumPerNode = intraNodeTP ? 1 : std::stoul(config.at("dp")) / (slaveCount + 1);
        if (intraNodeTP || static_cast<std::size_t>(dpRankIdx_) < dpNumPerNode) {
            remoteSlaveIP_ = ""; // The first segment of DP ranks in Master node does not have remote DP rank.
        } else {
            // Calculate the corresponding slave IP.
            remoteSlaveIP_ = slaveIPs.at(static_cast<std::size_t>(dpRankIdx_) / dpNumPerNode - 1);
        }
    }

    if (isLwdMultiNodesInfer_ && msRole_ == MasterSlaveRole::MASTER && std::stoul(config.at("dp")) > 1) {
        std::vector<std::string> slaveIPs;
        mindie_llm::Split(config.at("slaveIPs"), ",", slaveIPs);
        remoteSlaveIP_ = slaveIPs.at(static_cast<std::size_t>(dpRankIdx_));
    }

    PROF(INFO, AddMetaInfo("msRole", static_cast<int>(msRole_)));
}

bool Communicator::InitIPCCommunicators(const std::string &sharedMemPrefix, uint32_t localWorldSize)
{
    ShmSizeConfig executeShmConfig{SHARED_MEMORY_256MB, DEFAULT_SHARED_MEMORY_SIZE};
    ipcCommunicatorExecute_ = InitSingleIPCCommunicator(sharedMemPrefix + "_execute", localWorldSize, executeShmConfig);
    if (ipcCommunicatorExecute_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize IPC Communicator for Execute channel.");
        return false;
    }

    ShmSizeConfig sharedSyncShmConfig{DEFAULT_SHARED_MEMORY_SIZE, DEFAULT_SHARED_MEMORY_SIZE};
    ipcCommunicatorSharedSync_ =
        InitSingleIPCCommunicator(sharedMemPrefix + "_shared_sync_link", localWorldSize, sharedSyncShmConfig);
    if (ipcCommunicatorSharedSync_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize IPC Communicator for Shared Link channel.");
        return false;
    }

    ShmSizeConfig kvTransferShmConfig{SHARED_MEMORY_256MB, DEFAULT_SHARED_MEMORY_SIZE};
    ipcCommunicatorKVTransfer_ =
        InitSingleIPCCommunicator(sharedMemPrefix + "_transfer", localWorldSize, kvTransferShmConfig);
    if (ipcCommunicatorKVTransfer_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize IPC Communicator for KV Transfer channel.");
        return false;
    }

    return true;
}

bool Communicator::LwdGRPCCommunicatorInit(std::unordered_map<std::string, std::string> &config,
    uint32_t grpcCommunicatorNum)
{
    auto itrFindDp = config.find("dp");
    uint32_t dpNum = std::stol(itrFindDp->second);
    if (dpNum == 0) {
        MINDIE_LLM_LOG_ERROR("Lwd The value of dp_num must be greater than 0.");
        return false;
    }
    if (dpNum == grpcCommunicatorNum) { /* dp(excutor)数量与grpcCommunicatorNum一致, 由excutor保证调用次数 */
        if (!grpcCommunicator_->Init(grpcCommunicatorNum)) {
            MINDIE_LLM_LOG_ERROR("Lwd Failed to initialize GRPC Communicator.");
            return false;
        }

        return true;
    }

    /* 1个dp起多个GRPC的情况 */
    auto itrFindSlaveIPs = config.find("slaveIPs");
    std::string slaveIPsStr = std::string(itrFindSlaveIPs->second);
    uint32_t slaveNum = std::count(slaveIPsStr.begin(), slaveIPsStr.end(), ',') + 1;
    uint32_t multiGrpcNumPerExcutor = slaveNum / dpNum;
    std::vector<std::future<bool>> futures;
    for (uint32_t i = 0; i < multiGrpcNumPerExcutor; i++) {
        futures.push_back(std::async(std::launch::async, [&, i, grpcCommunicatorNum]() {
            if (!grpcCommunicator_->Init(grpcCommunicatorNum)) {
                MINDIE_LLM_LOG_ERROR("Lwd Failed to initialize GRPC Communicator:" << i << ".");
                return false;
            }

            return true;
        }));
    }

    // 检查所有的future结果都为true
    for (auto &fut : futures) {
        if (!fut.get()) {
            MINDIE_LLM_LOG_ERROR("Lwd Failed to initialize GRPC Communicator, one of Communicator failed.");
            return false;
        }
    }

    return true;
}

bool Communicator::InitGRPCCommunicator(std::unordered_map<std::string, std::string> &config,
                                        ResponseHandler responseFromSlaveHandler, uint32_t grpcCommunicatorNum)
{
    grpcCommunicator_ = GRPCCommunicator::GetInstance(config);

    if (msRole_ == MasterSlaveRole::MASTER) {
        if (!grpcCommunicator_->RegisterResponseHandler(responseFromSlaveHandler, dpRankIdx_)) {
            MINDIE_LLM_LOG_ERROR("Failed to register response handler for master node.");
            return false;
        }
    } else if (msRole_ == MasterSlaveRole::SLAVE) {
        RequestHandler requestFromMasterHandler =
            std::bind(&Communicator::SlaveNodeGRPCRequestHandler, this, std::placeholders::_1);
        if (!grpcCommunicator_->RegisterRequestHandler(requestFromMasterHandler, dpRankIdx_)) {
            MINDIE_LLM_LOG_ERROR("Failed to register request handler for slave node.");
            return false;
        }
    }

    auto itrFindLwdMultiNodesEn = config.find("lwd_multi_nodes_enable");
    if (itrFindLwdMultiNodesEn != config.end() && itrFindLwdMultiNodesEn->second == "true") {
        return LwdGRPCCommunicatorInit(config, grpcCommunicatorNum);
    }

    if (!grpcCommunicator_->Init(grpcCommunicatorNum)) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize GRPC Communicator.");
        return false;
    }
    return true;
}

void Communicator::RegisterModelInitReqHandler(SlaveModelInitReqHandler handler)
{
    slaveModelInitReqHandler_ = handler;
}

bool Communicator::SendModelInitRequestAndReceive(ExecuteRequest &request, std::vector<ExecuteResponse> &responses)
{
    if (!ipcCommunicatorExecute_->SendMessageViaSM(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send MODEL_INIT request to local executors.");
        return false;
    }
    // Wait until the responses are received.
    if (!ipcCommunicatorExecute_->ReceiveInitResponses(responses)) {
        MINDIE_LLM_LOG_ERROR("Failed to receive MODEL_INIT responses from local executors.");
        return false;
    }
    return true;
}

bool Communicator::SendSharedSyncRequestAndReceive(ExecuteRequest &request, std::vector<ExecuteResponse> &responses)
{
    // Send the sync request to local workers if ipcCommunicatoSharedSyncLink_ is initialized.
    if (ipcCommunicatorSharedSync_ != nullptr) {
        if (!ipcCommunicatorSharedSync_->SendMessageViaSM(request)) {
            MINDIE_LLM_LOG_ERROR("Failed to send a sync request to local workers.");
            return false;
        }
    }
    // Send the sync request to remote slave node if grpcCommunicator_ is initialized.
    if (grpcCommunicator_ != nullptr) {
        if (!grpcCommunicator_->SendRequest(request, dpRankIdx_, remoteDPRankIdx_, remoteSlaveIP_)) {
            MINDIE_LLM_LOG_ERROR("Failed to send a sync request to remote slave node.");
            return false;
        }
    }
    // Wait until the response is received from local workers if ipcCommunicatorSharedSync_ is initialized.
    if (ipcCommunicatorSharedSync_ != nullptr) {
        ExecuteResponse ipcResponse;
        if (!ipcCommunicatorSharedSync_->ReceiveResponse(ipcResponse)) {
            MINDIE_LLM_LOG_ERROR("Failed to receive a sync response from local workers.");
            return false;
        }
        responses.emplace_back(std::move(ipcResponse));
    }
    // Wait until the response is received from remote slave node if grpcCommunicator_ is initialized.
    if (grpcCommunicator_ != nullptr) {
        ExecuteResponse grpcResponse;
        if (!grpcCommunicator_->GetSyncResponse(grpcResponse, dpRankIdx_)) {
            MINDIE_LLM_LOG_ERROR("Failed to receive a sync response from remote slave node.");
            return false;
        }
        responses.emplace_back(std::move(grpcResponse));
    }
    return true;
}

bool Communicator::SendRecoverCommandRequestAndReceive(ExecuteRequest &request, std::vector<ExecuteResponse> &responses)
{
    if (ipcCommunicatorSharedSync_ != nullptr) {
        if (!ipcCommunicatorSharedSync_->SendMessageViaSM(request)) {
            MINDIE_LLM_LOG_ERROR("Failed to send a sync recover command request to local workers.");
            return false;
        }
    }

    // Wait until the responses are received.
    if (ipcCommunicatorSharedSync_ != nullptr) {
        if (!ipcCommunicatorSharedSync_->ReceiveRecoverCommandResponses(responses)) {
            MINDIE_LLM_LOG_ERROR("Failed to receive a sync recover command responses from local executors.");
            return false;
        }
    }
    return true;
}

bool Communicator::SendAsyncReponseToRemote(ExecuteResponse &response)
{
    if (grpcCommunicator_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("grpcCommunicator_ is null, cannot send response to master.");
        return false;
    }
    if (!grpcCommunicator_->SendResponse(response, dpRankIdx_, remoteDPRankIdx_)) {
        MINDIE_LLM_LOG_ERROR("Slave Node: failed to send response to remote master node.");
        return false;
    }
    return true;
}

bool Communicator::LaunchIPCHandleResponseThreads(ResponseHandler handler)
{
    ResponseHandler responseHandler = nullptr;
    if ((isMultiNodesInfer_ || layerwiseDisaggregated_) && msRole_ == MasterSlaveRole::SLAVE) {
        responseHandler = std::bind(&Communicator::SlaveNodeIPCResponseHandler, this, std::placeholders::_1);
    } else {
        responseHandler = handler;
    }

    if (!RegisterAndStartIPCHandler(ipcCommunicatorExecute_, responseHandler)) {
        MINDIE_LLM_LOG_ERROR("Failed to register and start handler for Execute channel.");
        return false;
    }
    if (!RegisterAndStartIPCHandler(ipcCommunicatorKVTransfer_, responseHandler)) {
        MINDIE_LLM_LOG_ERROR("Failed to register and start handler for KV Transfer channel.");
        return false;
    }
    if (msRole_ == MasterSlaveRole::SLAVE) {
        // Only slave node needs to asynchronously send sync response to master node.
        if (!RegisterAndStartIPCHandler(ipcCommunicatorSharedSync_, responseHandler)) {
            MINDIE_LLM_LOG_ERROR("Failed to register and start handler for PD Link channel.");
            return false;
        }
    }
    return true;
}

bool Communicator::RegisterAndStartIPCHandler(std::shared_ptr<IPCCommunicator> ipcCommunicator,
                                              ResponseHandler handler) const
{
    if (!ipcCommunicator->RegisterResponseHandler(handler)) {
        MINDIE_LLM_LOG_ERROR("Failed to register response handler for IPC Communicator.");
        return false;
    }
    if (!ipcCommunicator->StartHandleResponseThread()) {
        MINDIE_LLM_LOG_ERROR("Failed to start handle response thread for IPC Communicator.");
        return false;
    }
    return true;
}

bool Communicator::SlaveNodeGRPCRequestHandler(ExecuteRequest &request)
{
    if (request.execute_type() == REMOTE_MODEL_INIT) {
        std::map<std::string, std::string> pdInfo;
        auto &initRequest = request.remote_model_init_request();
        for (const auto &pair : initRequest.pd_info()) {
            pdInfo[pair.first] = pair.second;
        }
        if (slaveModelInitReqHandler_ == nullptr || !slaveModelInitReqHandler_(pdInfo)) {
            MINDIE_LLM_LOG_ERROR("Slave Node: failed to handle model init request from master node.");
            return false;
        }
    } else {
        if (!SendAsyncRequestToLocal(request)) {
            MINDIE_LLM_LOG_ERROR("Slave Node: failed to send asynchronous request to local workers.");
            return false;
        }
    }
    return true;
}

bool Communicator::SlaveNodeIPCResponseHandler(ExecuteResponse &response)
{
    // for edge-cloud, slave(cloud) node doesnt need to handle response
    if (layerwiseDisaggregated_) {
        return true;
    }
    // Skip sending to remote master if intra-node TP is enabled and response type is not PD_LINK.
    if (intraNodeTP_ && response.msg_type() != PD_LINK) {
        return true; // Intra-node TP does not send responses to remote master nodes.
    }
    return SendAsyncReponseToRemote(response);
}

bool Communicator::GRPCGetSyncResponse(ExecuteResponse &response)
{
    if (grpcCommunicator_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("grpcCommunicator_ is null, cannot get sync response.");
        return false;
    }
    return grpcCommunicator_->GetSyncResponse(response, dpRankIdx_);
}

std::unique_ptr<IPCCommunicator> Communicator::InitSingleIPCCommunicator(const std::string &sharedMemName,
                                                                         uint32_t localWorldSize,
                                                                         const ShmSizeConfig &shmSizeConfig) const
{
    std::unique_ptr<IPCCommunicator> ipcCommunicator = std::make_unique<IPCCommunicator>(sharedMemName, localWorldSize);
    if (!ipcCommunicator->SetupChannel(shmSizeConfig)) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize Execute channel.");
        return nullptr;
    }
    return ipcCommunicator;
}

bool Communicator::SendAsyncRequest(ExecuteRequest &request)
{
    if (isMultiNodesInfer_ && msRole_ == MasterSlaveRole::SLAVE) {
        MINDIE_LLM_LOG_ERROR("Slave nodes cannot call SendAsyncRequest themselves.");
        return false;
    }

    if (ipcCommunicatorExecute_ != nullptr) {
        auto spanLocal = PROF(INFO, Domain("Executor").SpanStart("SendRequestToLocal"));
        if (!SendAsyncRequestToLocal(request)) {
            MINDIE_LLM_LOG_ERROR("Failed to send asynchronous request to local workers.");
            PROF(spanLocal.SpanEnd());
            return false;
        }
        PROF(spanLocal.SpanEnd());
    }

    if (grpcCommunicator_ != nullptr) {
        auto spanRemote = PROF(INFO, Domain("Executor").SpanStart("SendRequestToRemote"));
        if (request.execute_type() == MODEL_INFER && remoteDPRankIdx_ != 0) {
            // MODEL_INFER requests are sent only once to the first DP rank of each target remote node.
            PROF(spanRemote.SpanEnd());
            return true;
        }
        if (!SendAsyncRequestToRemote(request)) {
            MINDIE_LLM_LOG_ERROR("Failed to send asynchronous request to remote workers.");
            PROF(spanRemote.SpanEnd());
            return false;
        }
        PROF(spanRemote.SpanEnd());
    }
    return true;
}

bool Communicator::SendAsyncRequestToLocal(ExecuteRequest &request)
{
    IPCCommunicator *ipcCommunicator = nullptr;
    if (request.execute_type() == MODEL_INFER || request.execute_type() == TEXT_GENERATOR_CLEANUP ||
        request.execute_type() == MODEL_FINALIZE || request.execute_type() == EOS_CLEANUP) {
        ipcCommunicator = ipcCommunicatorExecute_.get();
    } else if (request.execute_type() == KV_TRANSFER) {
        ipcCommunicator = ipcCommunicatorKVTransfer_.get();
    } else if (request.execute_type() == PD_LINK) {
        ipcCommunicator = ipcCommunicatorSharedSync_.get();
    } else {
        MINDIE_LLM_LOG_ERROR("Unsupported execute type for asynchronous request: " << request.execute_type());
        return false;
    }

    if (!ipcCommunicator->SendMessageViaSM(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send asynchronous request to local workers.");
        return false;
    }
    return true;
}

bool Communicator::SendAsyncRequestToRemote(ExecuteRequest &request)
{
    auto &configManager = mindie_llm::ConfigManager::GetInstance();
    if (configManager.IslayerwiseDisaggregated() && request.has_execute_model_request()) {
        model_execute_data::ExecuteModelRequest *modelReq = request.mutable_execute_model_request();
        for (int i = 0; i < modelReq->seq_group_metadata_list_size(); ++i) {
            model_execute_data::SequenceGroupMetadata *meta = modelReq->mutable_seq_group_metadata_list(i);
            if (!meta->has_prompt_token_ids()) {
                continue;
            }
            const std::string &raw = meta->prompt_token_ids();
            size_t num = raw.size() / sizeof(TokenId);
            const TokenId fill_value = 100L;
            std::vector<TokenId> tmp(num, fill_value);
            meta->set_prompt_token_ids(tmp.data(), tmp.size() * sizeof(TokenId));
        }
    }
    if (!grpcCommunicator_->SendRequest(request, dpRankIdx_, remoteDPRankIdx_, remoteSlaveIP_)) {
        MINDIE_LLM_LOG_ERROR("Failed to send request from DP " << dpRankIdx_ << " to remote DP " << remoteDPRankIdx_);
        return false;
    }
    return true;
}

void Communicator::CleanUp()
{
    if (ipcCommunicatorExecute_) {
        ipcCommunicatorExecute_->CleanUp();
        ipcCommunicatorExecute_.reset();
    }
    if (ipcCommunicatorSharedSync_) {
        ipcCommunicatorSharedSync_->CleanUp();
        ipcCommunicatorSharedSync_.reset();
    }
    if (ipcCommunicatorKVTransfer_) {
        ipcCommunicatorKVTransfer_->CleanUp();
        ipcCommunicatorKVTransfer_.reset();
    }
    grpcCommunicator_.reset();
}

Communicator::~Communicator() {}

} // namespace mindie_llm