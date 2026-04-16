/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "executor.h"

#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <vector>

#include "common_util.h"
#include "log.h"
#include "math_utils.h"
#include "msServiceProfiler/msServiceProfiler.h"
#include "string_utils.h"

namespace mindie_llm {
static std::set<std::string> requiredModelConfigKeys = {"globalWorldSize", "globalRankIds",      "model_instance_type",
                                                        "world_size",      "npu_device_ids",     "deploy_type",
                                                        "executor_type",   "asyncBatchscheduler"};

void Executor::LayerwiseParseFromModelConfig(std::unordered_map<std::string, std::string> &config,
                                             ModelLaunchConfig &modelLaunchConfig) const {
    if ((config.count("layerwiseDisaggregated") != 0) && (config.at("layerwiseDisaggregated") == "true")) {
        modelLaunchConfig.layerwiseDisaggregated = true;
        if (config.count("layerwiseDisaggregatedRoleType") != 0) {
            modelLaunchConfig.layerwiseDisaggregatedRoleType = config.at("layerwiseDisaggregatedRoleType");
        }

        auto lwdMultiEnIt = config.find("layerwiseDisaggregatedMultiNodesInferEnabled");
        if (lwdMultiEnIt != config.end() && lwdMultiEnIt->second == "true") {
            modelLaunchConfig.lwdMultiNodesEnable = true;

            auto lwdMultiMstIt = config.find("layerwiseDisaggregatedMultiNodesMaster");
            if (lwdMultiMstIt != config.end() && lwdMultiMstIt->second == "true") {
                modelLaunchConfig.isLwdMultiNodesMaster = true;
            }
        }
    }
}

bool Executor::ParseFromModelConfig(std::unordered_map<std::string, std::string> &config,
                                    ModelLaunchConfig &modelLaunchConfig, bool isMultiNodesInfer) const {
    for (auto &key : requiredModelConfigKeys) {
        if (config.find(key) == config.end()) {
            MINDIE_LLM_LOG_ERROR("Invalid model config without key " << key);
            return false;
        }
    }
    LayerwiseParseFromModelConfig(config, modelLaunchConfig);
    bool lwdCloudMultiNodesInfer =
        modelLaunchConfig.lwdMultiNodesEnable && modelLaunchConfig.layerwiseDisaggregatedRoleType == "slave";
    modelLaunchConfig.globalRankIds.clear();
    modelLaunchConfig.npuDeviceIds.clear();
    modelLaunchConfig.slaveIPs.clear();
    mindie_llm::Split(config.at("npu_device_ids"), ",", modelLaunchConfig.npuDeviceIds);
    if (isMultiNodesInfer || lwdCloudMultiNodesInfer) {
        mindie_llm::Split(config.at("globalRankIds"), ",", modelLaunchConfig.globalRankIds);
        mindie_llm::Split(config.at("slaveIPs"), ",", modelLaunchConfig.slaveIPs);
    }
    if (modelLaunchConfig.layerwiseDisaggregatedRoleType == "master") {
        mindie_llm::Split(config.at("slaveIPs"), ",", modelLaunchConfig.slaveIPs);
    }

    modelLaunchConfig.deployType = config.at("deploy_type");
    modelLaunchConfig.executorType = config.at("executor_type");
    modelLaunchConfig.npuNumPerNode = std::stoul(config.at("world_size"));
    if (modelLaunchConfig.npuNumPerNode < 1) {
        MINDIE_LLM_LOG_ERROR("Invalid world size in model config, localWorldSize: " << modelLaunchConfig.npuNumPerNode);
        return false;
    }
    modelLaunchConfig.globalWorldSize = std::stoul(config.at("globalWorldSize"));
    modelLaunchConfig.modelInstanceType = config.at("model_instance_type");
    modelLaunchConfig.isMultiNodesInfer = isMultiNodesInfer;
    modelLaunchConfig.isMasterNode = (config.at("isMaster") == "1");
    modelLaunchConfig.localIP = config.at("localIP");
    uint32_t tp = (config.count("tp") > 0) ? std::stoul(config.at("tp")) : 1;
    uint32_t cp = (config.count("cp") > 0) ? std::stoul(config.at("cp")) : 1;
    uint32_t sp = (config.count("sp") > 0) ? std::stoul(config.at("sp")) : 1;
    modelLaunchConfig.scp = cp * sp;
    if (tp > std::numeric_limits<uint32_t>::max() / cp) {
        MINDIE_LLM_LOG_ERROR(
            "ParseFromModelConfig failed: tp * cp is out of range uint32_t, it "
            "should be worldSize/dp.");
        return false;
    }
    modelLaunchConfig.npuNumPerDP = (config.count("tp") > 0) ? tp * cp : modelLaunchConfig.npuNumPerNode;
    modelLaunchConfig.dp = (config.count("dp") > 0) ? std::stoul(config.at("dp")) : 1;
    // Calculate the number of IPC communicators needed: ceil(npuNumPerNode /
    // npuNumPerDP)
    modelLaunchConfig.ipcCommunicatorNum = CeilDiv(modelLaunchConfig.npuNumPerNode, modelLaunchConfig.npuNumPerDP);
    if (modelLaunchConfig.deployType != "INTER_PROCESS") {
        MINDIE_LLM_LOG_ERROR("Supported deploy_type list should be [INTER_PROCESS], rather than "
                             << modelLaunchConfig.deployType << ", please check model config");
        return false;
    }
    return true;
}

bool Executor::ExecutorInstanceInit(std::map<std::string, std::string> &configFromManager, bool isMultiNodesInfer,
                                    size_t rankIdx) {
    if (!ExecutorParseConfigAndInitGRPC(configFromManager, isMultiNodesInfer, rankIdx)) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize Executor with GRPC.");
        return false;
    }
    if (!ExecutorModelInitAndSync()) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize Executor model and sync.");
        return false;
    }
    return true;
}

bool Executor::ExecutorParseConfigAndInitGRPC(std::map<std::string, std::string> &configFromManager,
                                              bool isMultiNodesInfer, size_t rankIdx) {
    isMultiNodesInfer_ = isMultiNodesInfer;
    dpRankIdx_ = rankIdx;
    configFromManager_ =
        std::unordered_map<std::string, std::string>(configFromManager.begin(), configFromManager.end());
    if (!ParseFromModelConfig(configFromManager_, modelLaunchConfig_, isMultiNodesInfer_)) {
        MINDIE_LLM_LOG_ERROR("Failed to parse from invalid model config.");
        return false;
    }

    // Multi-node rank usage:
    // - 1 master + N slaves: Master uses 1/(N+1) ranks for IPC, rest for gRPC
    // - Slaves always use all ranks for IPC + gRPC
    bool intraNodeTP = (isMultiNodesInfer_ && modelLaunchConfig_.npuNumPerDP > modelLaunchConfig_.npuNumPerNode);
    modelLaunchConfig_.intraNodeTP = intraNodeTP;
    int remoteDPRankIdx = GetRemoteDPRankIdx(modelLaunchConfig_, rankIdx, intraNodeTP);
    communicator_ =
        std::make_unique<Communicator>(configFromManager_, isMultiNodesInfer_, rankIdx, remoteDPRankIdx, intraNodeTP);
    communicator_->RegisterModelInitReqHandler(
        std::bind(&Executor::MasterAndSlaveModelInit, this, std::placeholders::_1));
    MINDIE_LLM_LOG_INFO("Executor instance init with rankIdx " << rankIdx << ", remoteDPRankIdx " << remoteDPRankIdx
                                                               << ", isMultiNodesInfer " << isMultiNodesInfer_);

    bool layerwiseDisaggregated = false;
    if ((configFromManager_.count("layerwiseDisaggregated") != 0) &&
        (configFromManager_.at("layerwiseDisaggregated") == "true")) {
        layerwiseDisaggregated = true;
    }
    // Initialize GRPC communicator if needed.
    bool isSlave = (isMultiNodesInfer_ && !modelLaunchConfig_.isMasterNode);
    if (isSlave || rankIdx >= modelLaunchConfig_.ipcCommunicatorNum || intraNodeTP || layerwiseDisaggregated) {
        uint32_t grpcCommunicatorNum = GetGRPCCommunicatorNum(modelLaunchConfig_, intraNodeTP);
        ResponseHandler asyncResponseHandler = std::bind(&Executor::AsyncResponseHandler, this, std::placeholders::_1);
        if (!communicator_->InitGRPCCommunicator(configFromManager_, asyncResponseHandler, grpcCommunicatorNum)) {
            MINDIE_LLM_LOG_ERROR(
                "Failed to initialize GRPC communicator for multi-nodes "
                "inference.");
            return false;
        }
        isGRPCInit_ = true;
    }
    return true;
}

bool Executor::LwdMasterAndSlaveSync() {
    MINDIE_LLM_LOG_INFO("[layerwiseDisaggregated|executor] " << "Start to synchronize model initialization between "
                                                                "Master and Slave nodes.");
    // Master node receives init response from slave
    if (modelLaunchConfig_.layerwiseDisaggregatedRoleType == "master" && modelLaunchConfig_.dp > 1) {
        if (!modelLaunchConfig_.lwdMultiNodesEnable && dpRankIdx_ > 0) {
            return true;  // 单机多dp场景,只有rank0处理model_init
        }
        ExecuteResponse rawSlaveResponse;
        communicator_->GRPCGetSyncResponse(rawSlaveResponse);
        if (!MasterHandleSlaveInitResponse(rawSlaveResponse)) {
            MINDIE_LLM_LOG_ERROR("Failed to handle slave initialization response.");
            return false;
        }
    } else if (modelLaunchConfig_.layerwiseDisaggregatedRoleType == "master" && modelLaunchConfig_.dp == 1) {
        uint32_t slaveCount = modelLaunchConfig_.slaveIPs.size();
        for (uint32_t i = 0; i < slaveCount; i++) {
            ExecuteResponse rawSlaveResponse;
            communicator_->GRPCGetSyncResponse(rawSlaveResponse);
            if (!MasterHandleSlaveInitResponse(rawSlaveResponse)) {
                MINDIE_LLM_LOG_ERROR("Failed to handle slave initialization response.");
                return false;
            }
        }
    } else if (!SlaveSendInitResponseToMaster()) {  // Slave node sends init
                                                    // response to master
        MINDIE_LLM_LOG_ERROR("Failed to send initialization response to master node.");
        return false;
    }
    MINDIE_LLM_LOG_INFO(
        "Successfully synchronize model initialization between Master and"
        " Slave nodes in layerwise disaggregated scenario.");
    return true;
}

bool Executor::ExecutorModelInitAndSync() {
    // Initialize IPC communicator and launch model if needed.
    if (dpRankIdx_ < modelLaunchConfig_.ipcCommunicatorNum) {
        if (!InitIPCAndLaunchModel()) {
            MINDIE_LLM_LOG_ERROR("Failed to initialize Executor with IPC and launch model.");
            return false;
        }
    }
    if (modelLaunchConfig_.layerwiseDisaggregated && isGRPCInit_) {
        return LwdMasterAndSlaveSync();
    } else if (isMultiNodesInfer_ && isGRPCInit_) {
        // 以下是集中式场景下，Master和Slave节点之间的同步逻辑
        if (modelLaunchConfig_.isMasterNode) {  // Master node receives init
                                                // response from slave
            // For intraNodeTP scenario(where one GRPC message is broadcasted to
            // multiple slaves), master needs to receive responses from all
            // slaves and aggregate the results. Otherwise, one executor
            // instance corresponds to one slave and only one response is
            // expected.
            size_t numExpectedResponses = modelLaunchConfig_.intraNodeTP ? modelLaunchConfig_.slaveIPs.size() : 1;
            for (int i = 0; i < numExpectedResponses; ++i) {
                ExecuteResponse rawSlaveResponse;
                communicator_->GRPCGetSyncResponse(rawSlaveResponse);
                if (!MasterHandleSlaveInitResponse(rawSlaveResponse)) {
                    MINDIE_LLM_LOG_ERROR("Failed to handle slave initialization response.");
                    return false;
                }
            }
        } else {  // Slave node sends init response to master
            if (!SlaveSendInitResponseToMaster()) {
                MINDIE_LLM_LOG_ERROR("Failed to send initialization response to master node.");
                return false;
            }
        }
    }
    MINDIE_LLM_LOG_INFO("Successfully initialize Executor with IPC and lanuch model!");
    return true;
}

bool Executor::InitIPCAndLaunchModel() {
    // Set the shared memory name and semaphore name prefix to pid.
    uint64_t ipcInitId = ipcInitCounter_.fetch_add(1, std::memory_order_relaxed);
    std::string sharedMemPrefix = "/" + std::to_string(getpid()) + "_" + std::to_string(ipcInitId);

    uint32_t workerNum = std::min(modelLaunchConfig_.npuNumPerDP, modelLaunchConfig_.npuNumPerNode);
    if (!communicator_->InitIPCCommunicators(sharedMemPrefix, workerNum)) {
        MINDIE_LLM_LOG_ERROR("Failed to initialize communicator.");
        return false;
    }
    // Initialize Python worker processes.
    if (!InitWorkerProcesses(modelLaunchConfig_, sharedMemPrefix)) {
        MINDIE_LLM_LOG_ERROR("Failed to launch Python worker processes.");
        return false;
    }
    // Build and send the init request.
    if (!InitModelExecution(configFromManager_)) {
        MINDIE_LLM_LOG_ERROR("Failed to send initialization request and handle response.");
        return false;
    }
    return true;
}

bool Executor::InitModelExecution(std::unordered_map<std::string, std::string> &config) {
    ExecuteRequest request;
    request.set_execute_type(MODEL_INIT);
    for (const auto &[key, value] : config) {
        (*request.mutable_config())[key] = value;
    }
    std::vector<ExecuteResponse> initResponses;
    if (!communicator_->SendModelInitRequestAndReceive(request, initResponses)) {
        MINDIE_LLM_LOG_ERROR("Failed to send initialization request to worker.");
        return false;
    }
    if (!HandleInitResult(initResponses)) {
        MINDIE_LLM_LOG_ERROR("Invalid initialization response format.");
        return false;
    }
    ResponseHandler asyncResponseHandler = std::bind(&Executor::AsyncResponseHandler, this, std::placeholders::_1);
    if (!communicator_->LaunchIPCHandleResponseThreads(asyncResponseHandler)) {
        MINDIE_LLM_LOG_ERROR("Failed to launch IPC handle response threads.");
        return false;
    }
    return true;
}

bool Executor::MasterAndSlaveModelInit(const std::map<std::string, std::string> &pdInfo) {
    configFromManager_.insert(pdInfo.begin(), pdInfo.end());

    // If the pdRole is not PnD, we need to send a request to the remote slave
    // node.
    if (isMultiNodesInfer_ && modelLaunchConfig_.isMasterNode && isGRPCInit_) {
        if (!MasterSendPDInfoToSlave(pdInfo)) {
            MINDIE_LLM_LOG_ERROR("Failed to send PD role to remote slave node.");
            return false;
        }
    }
    return ExecutorModelInitAndSync();
}

bool Executor::MasterSendPDInfoToSlave(const std::map<std::string, std::string> &pdInfo) {
    ExecuteRequest request;
    request.set_execute_type(REMOTE_MODEL_INIT);
    auto *initSlaveModelReq = request.mutable_remote_model_init_request();
    for (const auto &[key, value] : pdInfo) {
        (*initSlaveModelReq->mutable_pd_info())[key] = value;
    }
    if (!communicator_->SendAsyncRequestToRemote(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send config to remote slave node.");
        return false;
    }
    return true;
}

bool Executor::SlaveSendInitResponseToMaster() {
    ExecuteResponse response;
    response.set_msg_type(REMOTE_MODEL_INIT);

    auto *slaveResponse = response.mutable_remote_model_init_results();
    slaveResponse->set_cpu_block_num(IExecutor::kvCacheOverview_.cpuBlockNum);
    slaveResponse->set_max_position_embeddings(IExecutor::kvCacheOverview_.maxPositionEmbeddings);
    for (const auto &desc : IExecutor::kvCacheOverview_.kvCacheDescs) {
        auto *protoDesc = slaveResponse->add_kv_cache_descs();
        protoDesc->set_npu_block_num(static_cast<int32_t>(desc.npuBlockNum));
        protoDesc->set_block_size(static_cast<int32_t>(desc.blockSize));
        protoDesc->set_compression_ratio(static_cast<uint32_t>(desc.compressionRatio));
        protoDesc->set_cache_type(desc.cacheType);
    }
    communicator_->SendAsyncReponseToRemote(response);
    return true;
}

bool Executor::MasterHandleSlaveInitResponse(ExecuteResponse &response) const {
    if (response.msg_type() != REMOTE_MODEL_INIT || !response.has_remote_model_init_results()) {
        MINDIE_LLM_LOG_ERROR("Invalid model init info response from slave node.");
        return false;
    }
    auto &slaveInfo = response.remote_model_init_results();
    if (slaveInfo.kv_cache_descs_size() == 0) {
        MINDIE_LLM_LOG_ERROR(
            "Invalid model init info response from slave node: missing "
            "kv_cache_descs.");
        return false;
    }
    const uint32_t npuBlockNum = static_cast<uint32_t>(slaveInfo.kv_cache_descs(0).npu_block_num());
    if (modelLaunchConfig_.layerwiseDisaggregated && modelLaunchConfig_.scp > 1) {
        uint32_t lwdCloudNpuBlockNum = npuBlockNum;
        kvCacheOverview_.lwdCloudNpuBlockNum = std::min(kvCacheOverview_.lwdCloudNpuBlockNum, lwdCloudNpuBlockNum);
    } else {
        IExecutor::kvCacheOverview_.UpdateIfSmaller(slaveInfo.cpu_block_num(), npuBlockNum,
                                                    slaveInfo.max_position_embeddings());
        if (slaveInfo.kv_cache_descs_size() > 0) {
            std::vector<KVCacheOverview::KVCacheDesc> descs = ParseProtoKvCacheDescs(response);
            if (!IExecutor::kvCacheOverview_.UpdateKvCacheDescsIfEmptyOrEqual(descs)) {
                MINDIE_LLM_LOG_WARN(
                    "KV cache descs mismatch between master and slave; keep "
                    "existing master descs.");
            }
        }
    }
    MINDIE_LLM_LOG_INFO(
        "[Executor::MasterHandleSlaveInitResponse]: Updated KV cache overview "
        "from slave: CPU blocks = "
        << IExecutor::kvCacheOverview_.cpuBlockNum << ", NPU blocks = " << IExecutor::kvCacheOverview_.npuBlockNum
        << ", MaxPosEmb = " << IExecutor::kvCacheOverview_.maxPositionEmbeddings);
    return true;
}

std::vector<KVCacheOverview::KVCacheDesc> Executor::ParseProtoKvCacheDescs(const ExecuteResponse &response) const {
    std::vector<KVCacheOverview::KVCacheDesc> descs;
    const ::google::protobuf::RepeatedPtrField<model_execute_data::KVCacheDesc> *protoDescs = nullptr;

    if (response.has_remote_model_init_results() && response.remote_model_init_results().kv_cache_descs_size() > 0) {
        protoDescs = &response.remote_model_init_results().kv_cache_descs();
    } else if (response.has_init_results() && response.init_results().kv_cache_descs_size() > 0) {
        protoDescs = &response.init_results().kv_cache_descs();
    }

    if (protoDescs != nullptr) {
        descs.reserve(static_cast<size_t>(protoDescs->size()));
        for (const auto &protoDesc : *protoDescs) {
            KVCacheOverview::KVCacheDesc d;
            d.npuBlockNum = static_cast<uint32_t>(protoDesc.npu_block_num());
            d.blockSize = static_cast<uint32_t>(protoDesc.block_size());
            d.compressionRatio = static_cast<uint32_t>(protoDesc.compression_ratio());
            d.cacheType = protoDesc.cache_type();
            descs.push_back(d);
        }
    }
    return descs;
}

bool Executor::AsyncExecuteModel(ExecuteModelRequestPtr &modelRequest,
                                 std::function<void(ModelBatchResultSPtr)> executeModelResponseHandler) {
    if (modelRequest == nullptr) {
        MINDIE_LLM_LOG_ERROR("Inference model request is null.");
        return false;
    }

    ExecuteRequest request;
    request.set_execute_type(MODEL_INFER);
    request.mutable_execute_model_request()->CopyFrom(*modelRequest);
    RegisterExecuteModelResponseHandler(executeModelResponseHandler);
    MINDIE_LLM_LOG_DEBUG_REQUEST("Ready to execute inference requests.");

    if (!communicator_->SendAsyncRequest(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send execute message to local workers.");
        return false;
    }
    return true;
}

bool Executor::AsyncTGCleanup(TGCleanupRequestPtr &TGCleanupRequest) {
    ExecuteRequest request;
    request.set_execute_type(TEXT_GENERATOR_CLEANUP);
    request.mutable_text_generator_cleanup_request()->CopyFrom(*TGCleanupRequest);

    MINDIE_LLM_LOG_DEBUG_REQUEST("Ready to execute clear cache requests.");
    if (!communicator_->SendAsyncRequest(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send clear cache message to local workers.");
        return false;
    }
    return true;
}

bool Executor::AsyncEOSCleanup(TGCleanupRequestPtr &TGCleanupRequest) {
    ExecuteRequest request;
    request.set_execute_type(EOS_CLEANUP);
    request.mutable_text_generator_cleanup_request()->CopyFrom(*TGCleanupRequest);

    MINDIE_LLM_LOG_DEBUG("[layerwiseDisaggregated|executor] " << "Ready to execute clear cache requests.");
    if (!communicator_->SendAsyncRequest(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send clear cache message to local workers.");
        return false;
    }
    return true;
}

// SetupPDLink是同步的
bool Executor::SetupPDLink(PDLinkRequest &pdLinkRequest) {
    ExecuteRequest request;
    request.set_execute_type(PD_LINK);
    request.mutable_pd_link_request()->CopyFrom(pdLinkRequest);

    // No need to Wait for PDLink response.
    if (!communicator_->SendSharedSyncRequest(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send PDLink request to worker.");
        return false;
    }
    return true;
}

bool Executor::QueryPDLinkStatus(PDLinkStatusRequest &pdLinkStatusRequest) {
    ExecuteRequest request;
    request.set_execute_type(model_execute_data::PD_LINK_STATUS_QUERY);
    request.mutable_pd_link_status_request()->CopyFrom(pdLinkStatusRequest);

    // Wait for PDLink response from local and/or remote workers.
    std::vector<ExecuteResponse> pdLinkQueryResponses;
    if (!communicator_->SendSharedSyncRequestAndReceive(request, pdLinkQueryResponses)) {
        MINDIE_LLM_LOG_ERROR("Failed to send PDLinkStatus request to worker.");
        return false;
    }

    // Aggregate PDLink Status responses from Master and Slaves if needed.
    ExecuteResponse aggregatedResponse;
    if (!AggregatePDLinkStatusResponses(pdLinkQueryResponses, aggregatedResponse)) {
        MINDIE_LLM_LOG_ERROR("Failed to aggregate PDLinkStatus responses.");
        return false;
    }
    if (!HandlePDLinkStatusResponse(aggregatedResponse)) {
        MINDIE_LLM_LOG_ERROR("Failed to handle a PDLinkStatus response.");
        return false;
    }
    return true;
}

bool Executor::ExecuteKVTransfer(PullKVRequestPtr &pullKVRequest, PullKVResponseHandler pullKVResponseHandler) {
    if (pullKVRequest == nullptr) {
        MINDIE_LLM_LOG_ERROR("Pull KV cache request is null.");
        return false;
    }

    ExecuteRequest request;
    request.set_execute_type(KV_TRANSFER);
    request.mutable_pull_kv_request()->CopyFrom(*pullKVRequest);

    RegisterPullKVResponseHandler(pullKVResponseHandler);

    MINDIE_LLM_LOG_DEBUG_REQUEST("Ready to execute instance transfer request.");
    if (!communicator_->SendAsyncRequest(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send transfer message to another local worker.");
        return false;
    }
    return true;
}

bool Executor::ExecutorInstanceFinalize() {
    ExecuteRequest request;
    request.set_execute_type(MODEL_FINALIZE);

    if (!communicator_->SendAsyncRequest(request)) {
        MINDIE_LLM_LOG_ERROR("Failed to send finialize message.");
        return false;
    }
    for (pid_t pid : childPids_) {
        if (pid <= 0) continue;

        if (kill(pid, SIGTERM) == 0) {
            MINDIE_LLM_LOG_INFO("Sent SIGTERM to child process " << pid);
        } else {
            MINDIE_LLM_LOG_ERROR("Failed to send SIGTERM to " << pid);
            continue;
        }

        constexpr int maxWaitMs = 500;
        constexpr int checkIntervalMs = 50;
        constexpr int checkIntervalUs = checkIntervalMs * 1000;  // 50ms
        constexpr int maxIterations = maxWaitMs / checkIntervalMs;
        for (int i = 0; i < maxIterations; ++i) {
            if (waitpid(pid, nullptr, WNOHANG) > 0) {
                break;
            }
            usleep(checkIntervalUs);
        }
    }

    // 清空 PID 列表
    childPids_.clear();
    JoinPipeThreads();  // Ensure all pipe threads are joined before finalizing
    communicator_->CleanUp();
    MINDIE_LLM_LOG_DEBUG("Executor finalized and resources cleaned up.");
    return true;
}

bool Executor::HandleThinkingConfig(std::vector<ExecuteResponse> &responses) {
    if (responses.size() == 0) {
        return true;
    }
    const auto &initResults = responses[0].init_results().init_result_map();
    if (initResults.count("earlyStoppingIds") == 0 || initResults.count("startThinkingId") == 0 ||
        initResults.count("stopThinkingId") == 0) {
        return true;
    }
    try {
        thinkingConfig_.startThinkingId = std::stol(initResults.at("startThinkingId"));
        thinkingConfig_.stopThinkingId = std::stol(initResults.at("stopThinkingId"));
        SplitTokensToVec(initResults.at("earlyStoppingIds"), ',', thinkingConfig_.earlyStoppingIds);
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_ERROR("Invalid init result format for startThinkingId: "
                             << initResults.at("startThinkingId")
                             << ", stopThinkingId: " << initResults.at("stopThinkingId")
                             << ", earlyStoppingIds: " << initResults.at("earlyStoppingIds"));
        return false;
    }
    return true;
}

bool Executor::HandleInitResult(std::vector<ExecuteResponse> &responses) {
    if (!HandleThinkingConfig(responses)) {
        return false;
    }
    for (size_t i = 0; i < responses.size(); ++i) {
        const auto &initResults = responses[i].init_results().init_result_map();
        if (modelLaunchConfig_.layerwiseDisaggregated) {
            auto itrResultStatus = initResults.find("status");
            if (itrResultStatus != initResults.end() && itrResultStatus->second == "error") {
                MINDIE_LLM_LOG_ERROR("Init result error: Required fields missing in response.");
                return false;
            }
        }
        if (initResults.count("cpuBlockNum") == 0 || initResults.count("maxPositionEmbeddings") == 0) {
            MINDIE_LLM_LOG_ERROR("Init result error: Required fields missing in response.");
            return false;
        }
        try {
            // npuBlockNum is no longer carried in init_result_map; use
            // kv_cache_descs instead.
            if (responses[i].init_results().kv_cache_descs_size() == 0) {
                MINDIE_LLM_LOG_ERROR(
                    "Init result error: kv_cache_descs is missing in "
                    "response.");
                return false;
            }
            const uint32_t npuBlockNum =
                static_cast<uint32_t>(responses[i].init_results().kv_cache_descs(0).npu_block_num());
            IExecutor::kvCacheOverview_.UpdateIfSmaller(std::stoul(initResults.at("cpuBlockNum")), npuBlockNum,
                                                        std::stoul(initResults.at("maxPositionEmbeddings")));

            std::vector<KVCacheOverview::KVCacheDesc> descs = ParseProtoKvCacheDescs(responses[i]);
            if (!IExecutor::kvCacheOverview_.UpdateKvCacheDescsIfEmptyOrEqual(descs)) {
                MINDIE_LLM_LOG_WARN(
                    "kv_cache_descs mismatch across init responses; keep "
                    "existing descs.");
            }
        } catch (const std::exception &e) {
            const auto itCpu = initResults.find("cpuBlockNum");
            const auto itMaxPos = initResults.find("maxPositionEmbeddings");
            const int kvSize = responses[i].init_results().kv_cache_descs_size();

            MINDIE_LLM_LOG_ERROR("Invalid init result format: cpuBlockNum="
                                 << (itCpu == initResults.end() ? "<missing>" : itCpu->second)
                                 << ", maxPositionEmbeddings="
                                 << (itMaxPos == initResults.end() ? "<missing>" : itMaxPos->second)
                                 << ", kv_cache_descs_size=" << kvSize << ", exception=" << e.what());
            return false;
        }
    }
    MINDIE_LLM_LOG_INFO(
        "[Executor::HandleInitResult]: Initialized KV cache overview: CPU "
        "blocks = "
        << IExecutor::kvCacheOverview_.cpuBlockNum << ", NPU blocks = " << IExecutor::kvCacheOverview_.npuBlockNum
        << ", MaxPosEmb = " << IExecutor::kvCacheOverview_.maxPositionEmbeddings);
    return true;
}

bool Executor::HandleRecoverCommandResult(RecoverCommandInfo &commandInfo,
                                          std::vector<ExecuteResponse> &responses) const {
    if (responses.empty()) {
        MINDIE_LLM_LOG_ERROR("Recover command result error: empty responses.");
        return false;
    }

    for (size_t i = 0; i < responses.size(); ++i) {
        const auto &reocverCommandResponse = responses[i].recover_command_response();
        mindie_llm::NPUExecutionResult result;
        result.npuDeviceId = reocverCommandResponse.npu_device_id();
        result.commandResult = reocverCommandResponse.command_result();
        result.errorMsg = reocverCommandResponse.error_msg();
        commandInfo.results.PushBack(result);
    }
    return true;
}

void Executor::HandleExecuteModelResponse(ExecuteResponse &modelExecuteResponse) {
    if (executeModelResponseHandler_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("No response handler for ExecuteModelResponse.");
        return;
    }
    if (!modelExecuteResponse.has_execute_model_response()) {
        // TBC_此处如果本身是正常的请求，但是返回异常了，会导致asyncBatchNum_变量不满足导致schedule阻塞
        // 现在此处现在陪跑也会走到这里，陪跑本身不设置asyncBatchNum_
        ExecuteModelResponse resp;
        ModelBatchResultSPtr modelBatchResultSPtr = std::make_shared<ExecuteModelResponse>(resp);
        executeModelResponseHandler_(modelBatchResultSPtr);
        return;
    }
    ModelBatchResultSPtr modelBatchResultSPtr =
        std::make_shared<ExecuteModelResponse>(modelExecuteResponse.execute_model_response());
    executeModelResponseHandler_(modelBatchResultSPtr);
}

bool Executor::AggregatePDLinkStatusResponses(const std::vector<ExecuteResponse> &responseVec,
                                              ExecuteResponse &aggregatedResponse) const {
    aggregatedResponse.set_msg_type(PD_LINK_STATUS_QUERY);
    auto *aggregatedPDLinkStatus = aggregatedResponse.mutable_pd_link_status_response();

    for (const auto &singleResponse : responseVec) {
        if (singleResponse.msg_type() != PD_LINK_STATUS_QUERY || !singleResponse.has_pd_link_status_response()) {
            MINDIE_LLM_LOG_ERROR(
                "AggregatePDLinkStatusResponses: invalid response type or "
                "missing"
                << " PDLinkStatusResponse field.");
            aggregatedResponse.Clear();
            return false;
        }

        const auto &failedLinkInfoItems = singleResponse.pd_link_status_response().failed_link_info();
        for (const auto &failedLinkInfo : failedLinkInfoItems) {
            auto *newFailedLinkInfo = aggregatedPDLinkStatus->add_failed_link_info();
            newFailedLinkInfo->set_cluster_id(failedLinkInfo.cluster_id());
            newFailedLinkInfo->set_pd_error_code(failedLinkInfo.pd_error_code());
        }
        const auto &successLinkInfoItems = singleResponse.pd_link_status_response().success_link_info();
        for (const auto &successLinkInfo : successLinkInfoItems) {
            aggregatedPDLinkStatus->add_success_link_info(successLinkInfo);
        }
        const auto &runningLinkInfoItems = singleResponse.pd_link_status_response().running_link_info();
        for (const auto &runningLinkInfo : runningLinkInfoItems) {
            aggregatedPDLinkStatus->add_running_link_info(runningLinkInfo);
        }
        const auto &waitingLinkInfoItems = singleResponse.pd_link_status_response().waiting_link_info();
        for (const auto &waitingLinkInfo : waitingLinkInfoItems) {
            aggregatedPDLinkStatus->add_waiting_link_info(waitingLinkInfo);
        }
    }
    return true;
}

bool Executor::HandlePDLinkStatusResponse(ExecuteResponse &executeResponse) {
    if (!executeResponse.has_pd_link_status_response()) {
        MINDIE_LLM_LOG_ERROR("Invalid response: missing PDLinkStatusResponse field.");
        return false;
    }
    pdLinkStatusResponse_ = executeResponse.pd_link_status_response();
    return true;
}

void Executor::HandleKVTransferResponse(ExecuteResponse &executeResponse) {
    if (pullKVResponseHandler_ == nullptr) {
        MINDIE_LLM_LOG_ERROR("No response handler for TransferModelResponse.");
        return;
    }
    if (!executeResponse.has_pull_kv_response()) {
        MINDIE_LLM_LOG_ERROR("Invalid response: missing TransferModelResponse field.");
        return;
    }
    PullKVResponseSPtr pullKVResponse = std::make_shared<PullKVResponse>(executeResponse.pull_kv_response());
    pullKVResponseHandler_(pullKVResponse);
}

bool Executor::AsyncResponseHandler(ExecuteResponse &response) {
    auto executeType = response.msg_type();
    if (executeType == MODEL_INFER || executeType == EXECUTE_ERROR) {
        // EXECUTE_ERROR: inference error with err_msg, same handling as
        // MODEL_INFER
        HandleExecuteModelResponse(response);
    } else if (executeType == KV_TRANSFER) {  // Handle KV cache transfer message.
        HandleKVTransferResponse(response);
    } else {
        MINDIE_LLM_LOG_ERROR("Receive wrong message type: " << executeType
                                                            << ". You can ignore this warning if you are "
                                                               "shutting down engine.");
        return false;
    }
    return true;
}

template <typename HandlerType>
void RegisterHandler(HandlerType &memberHandler, HandlerType handler) {
    // If the handler is already registered, do not overwrite it.
    if (memberHandler == nullptr && handler != nullptr) {
        memberHandler = handler;
    }
}

void Executor::RegisterExecuteModelResponseHandler(ExecuteModelResponseHandler handler) {
    RegisterHandler(executeModelResponseHandler_, handler);
}

void Executor::RegisterPullKVResponseHandler(PullKVResponseHandler handler) {
    RegisterHandler(pullKVResponseHandler_, handler);
}

uint32_t Executor::GetCpuBlockNum() const {
    if (IExecutor::kvCacheOverview_.cpuBlockNum == 0xFFFFFFFF) {
        MINDIE_LLM_LOG_ERROR("CPU block number is not initialized.");
        return 0;
    }
    return IExecutor::kvCacheOverview_.cpuBlockNum;
}

uint32_t Executor::GetNpuBlockNum() const {
    if (IExecutor::kvCacheOverview_.npuBlockNum == 0xFFFFFFFF) {
        MINDIE_LLM_LOG_ERROR("NPU block number is not initialized.");
        return 0;
    }
    return IExecutor::kvCacheOverview_.npuBlockNum;
}

uint32_t Executor::GetLwdCloudNpuBlockNum() const {
    if (IExecutor::kvCacheOverview_.lwdCloudNpuBlockNum == 0xFFFFFFFF) {
        MINDIE_LLM_LOG_ERROR("Cloud NPU block number is not initialized.");
        return 0;
    }
    return IExecutor::kvCacheOverview_.lwdCloudNpuBlockNum;
}

ThinkingConfig Executor::GetThinkingConfig() const { return thinkingConfig_; }

uint32_t Executor::GetMaxPositionEmbeddings() const {
    if (IExecutor::kvCacheOverview_.maxPositionEmbeddings == 0xFFFFFFFF) {
        MINDIE_LLM_LOG_ERROR("Max position embeddings is not initialized.");
        return 0;
    }
    return IExecutor::kvCacheOverview_.maxPositionEmbeddings;
}

PDLinkStatusResponse Executor::GetPDLinkStatusResponse() const { return pdLinkStatusResponse_; };

std::vector<std::string> Executor::BuildConnectorCommand(const ModelLaunchConfig &modelConfig,
                                                         const std::string &sharedMemPrefix, uint32_t rankInDP) const {
    uint32_t rankInNode = rankInDP + dpRankIdx_ * modelConfig.npuNumPerDP;
    uint32_t globalRankId;
    // 云侧为真多机, 边侧为单机
    bool lwdCloudMultiNodesInfer =
        modelConfig.lwdMultiNodesEnable && modelConfig.layerwiseDisaggregatedRoleType == "slave";
    if ((modelConfig.isMultiNodesInfer || lwdCloudMultiNodesInfer) &&
        (rankInNode >= modelConfig.globalRankIds.size() ||
         !StrToUint32(globalRankId, modelConfig.globalRankIds[rankInNode]))) {
        MINDIE_LLM_LOG_ERROR(
            "Error: Failed to BuildConnectorCommand: could not get "
            "globalRankId.");
        return std::vector<std::string>{};
    }
    if (rankInNode >= modelConfig.npuDeviceIds.size()) {
        MINDIE_LLM_LOG_ERROR("Error: Failed to BuildConnectorCommand: rankInNode out of range.");
        return std::vector<std::string>{};
    }
    uint32_t globalRank = (modelConfig.isMultiNodesInfer || lwdCloudMultiNodesInfer) ? globalRankId : rankInNode;
    std::vector<std::string> command = {"mindie_llm_backend",
                                        "--local_rank",
                                        std::to_string(rankInNode),
                                        "--local_world_size",
                                        std::to_string(modelConfig.npuNumPerNode),
                                        "--npu_num_per_dp",
                                        std::to_string(modelConfig.npuNumPerDP),
                                        "--npu_device_id",
                                        modelConfig.npuDeviceIds[rankInNode],
                                        "--parent_pid",
                                        std::to_string(getpid()),
                                        "--shm_name_prefix",
                                        sharedMemPrefix};

    if (modelConfig.layerwiseDisaggregated) {
        command.push_back("--layerwise_disaggregated_role_type");
        command.push_back(modelConfig.layerwiseDisaggregatedRoleType);
        command.push_back("--layerwise_disaggregated");
        command.push_back("true");
    }

    if (modelConfig.isMultiNodesInfer || lwdCloudMultiNodesInfer) {
        command.push_back("--global_rank");
        command.push_back(std::to_string(globalRank));
        command.push_back("--global_world_size");
        command.push_back(std::to_string(modelConfig.globalWorldSize));
    }

    return command;
}

void Executor::JoinPipeThreads() {
    for (auto &thread : pipeThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    pipeThreads_.clear();
}

void Executor::ConsumePipe(FILE *pipe) {
    pthread_setname_np(pthread_self(), "ConsumePipe");
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::cout << buffer;
    }
    pclose(pipe);
}

bool Executor::ExecuteCommand(const std::vector<std::string> &command) {
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        MINDIE_LLM_LOG_ERROR("Error: Failed to create pipe for backend process");
        return false;
    }

    pid_t pid = fork();
    if (pid == -1) {
        MINDIE_LLM_LOG_ERROR("Error: Failed to fork for backend process");
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }

    if (pid == 0) {
        // Reset signal dispositions to default so child won't run parent's
        // handlers
        signal(SIGTERM, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        signal(SIGCHLD, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        signal(SIGPIPE, SIG_DFL);

        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        std::vector<char *> argv;
        for (auto &arg : command) {
            argv.push_back(strdup(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp("mindie_llm_backend", argv.data());
        perror("execvp mindie_llm_backend failed");
        for (auto ptr : argv) {
            std::free(ptr);
        }
        argv.clear();

        return false;
    } else {
        if (pid > 0) {
            childPids_.push_back(pid);
        }
        close(pipefd[1]);

        FILE *pipe = fdopen(pipefd[0], "r");
        if (!pipe) {
            MINDIE_LLM_LOG_ERROR("Error: Failed to fdopen pipe for backend process");
            close(pipefd[0]);
            return false;
        }

        pipeThreads_.emplace_back(&Executor::ConsumePipe, pipe);

        return true;
    }
}

void Executor::ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) {
    // init npu command request
    ExecuteRequest request;
    if (commandInfo.command == "CMD_PAUSE_ENGINE") {
        request.set_execute_type(PAUSE_COMMAND_EXEC);
    } else if (commandInfo.command == "CMD_PAUSE_ENGINE_ROCE") {
        request.set_execute_type(PAUSE_COMMAND_EXEC_ROCE);
    } else if (commandInfo.command == "CMD_CLEAR_TRANSER") {
        request.set_execute_type(CLEAR_COMMAND_EXEC);
    } else if (commandInfo.command == "CMD_REINIT_NPU") {
        request.set_execute_type(RECOVER_COMMAND_EXEC);
    } else if (commandInfo.command == "CMD_START_ENGINE") {
        request.set_execute_type(START_COMMAND_EXEC);
    }
    request.mutable_recover_command_request()->set_command(commandInfo.command);

    std::vector<ExecuteResponse> recoverCommandResponses;
    if (!communicator_->SendRecoverCommandRequestAndReceive(request, recoverCommandResponses)) {
        MINDIE_LLM_LOG_ERROR("Failed to send recover command request to worker.");
    }

    // wait until all recover commands responses are received
    HandleRecoverCommandResult(commandInfo, recoverCommandResponses);
}

bool Executor::InitWorkerProcesses(const ModelLaunchConfig &modelConfig, const std::string &sharedMemPrefix) {
    // workerNum is npuNumPerDP, except in TP16 case where it's npuNumPerNode
    uint32_t workerNum = std::min(modelConfig.npuNumPerDP, modelConfig.npuNumPerNode);
    for (uint32_t rankInDP = 0; rankInDP < workerNum; ++rankInDP) {
        std::vector<std::string> command = BuildConnectorCommand(modelConfig, sharedMemPrefix, rankInDP);
        std::ostringstream cmdStream;
        for (size_t i = 0; i < command.size(); ++i) {
            if (i > 0) {
                cmdStream << " ";
            }
            cmdStream << command[i];
        }
        MINDIE_LLM_LOG_INFO("Execute command: " << cmdStream.str());

        if (!ExecuteCommand(command)) {
            return false;
        }
    }
    return true;
}

int Executor::GetRemoteDPRankIdx(ModelLaunchConfig &modelConfig, int rankIdx, bool intraNodeTP) const {
    if (modelConfig.layerwiseDisaggregated) {
        int remotedpRankId = 0;  // 其实就是所在slaveIp数组的下标, 边云的matser节点中没有意义
        if (modelConfig.lwdMultiNodesEnable && modelConfig.layerwiseDisaggregatedRoleType == "slave" &&
            modelConfig.dp > 1) {
            // 当前这样只能适配双机, 更多机这里适配不了, 要使用别的变量来判断
            remotedpRankId = modelConfig.isLwdMultiNodesMaster ? 0 : 1;
        }
        return remotedpRankId;
    }

    // Single node inference does not have remote DP rank.
    if (!modelConfig.isMultiNodesInfer) {
        return -1;
    }
    // For Intra-node TP case, both Master and Slave nodes only have DP Rank 0.
    if (intraNodeTP) {
        return 0;
    }

    int slaveCount = static_cast<int>(modelConfig.slaveIPs.size());
    int dpNumPerNode = static_cast<int>(modelConfig.dp) / (slaveCount + 1);
    if (dpNumPerNode < 1) {
        MINDIE_LLM_LOG_ERROR("Invalid DP number per node: " << dpNumPerNode);
        return -1;
    }

    if (modelConfig.isMasterNode) {
        // 1 master + N slaves: Master uses 1/(N+1) ranks for IPC, rest for gRPC
        return (rankIdx < dpNumPerNode) ? -1 : (rankIdx % dpNumPerNode);
    } else {
        // Slaves always use all ranks for IPC + gRPC
        auto it = std::find(modelConfig.slaveIPs.begin(), modelConfig.slaveIPs.end(), modelConfig.localIP);
        if (it == modelConfig.slaveIPs.end()) {
            MINDIE_LLM_LOG_ERROR("Failed to find local IP " << modelConfig.localIP << " in slave IPs.");
            return -1;
        }
        int slaveIdx = std::distance(modelConfig.slaveIPs.begin(), it);
        return (slaveIdx + 1) * dpNumPerNode + rankIdx;  // Connect to the corresponding DP rank in Master node.
    }
}

uint32_t Executor::GetGRPCCommunicatorNum(ModelLaunchConfig &modelConfig, bool intraNodeTP) const {
    uint32_t slaveCount = modelConfig.slaveIPs.size();
    if (modelConfig.layerwiseDisaggregated) {
        // 边侧起slaveIpNum个GRPC(比如双机起2个, 单机起1个), 云侧都是起一个
        return modelConfig.layerwiseDisaggregatedRoleType == "master" ? slaveCount : 1;
    }

    if (intraNodeTP) {
        return 1;
    }
    uint32_t nodeCount = slaveCount + 1;

    if (modelConfig.isMasterNode) {
        // For Master, it uses dp/nodeCount*slaveCount communicators to connect
        // to all Slaves.
        return modelConfig.dp / nodeCount * slaveCount;
    } else {
        // For Slave, it uses dp/nodeCount communicators to connect to Master.
        return modelConfig.dp / nodeCount;
    }
}

LoraOperationResponse Executor::GetLoraOperationResponse() const { return loraOperationResponse_; };

bool Executor::HandleLoraOperationResponse(ExecuteResponse &executeResponse) {
    if (!executeResponse.has_lora_operation_response()) {
        MINDIE_LLM_LOG_ERROR("Invalid response: missing LoraOperationResponse field.");
        return false;
    }
    loraOperationResponse_ = executeResponse.lora_operation_response();
    return true;
}

bool Executor::ExecutLoraRequest(LoraOperationRequest &loraOperationRequest) {
    ExecuteRequest request;
    request.set_execute_type(LORA_OPERATION);
    request.mutable_lora_operation_request()->CopyFrom(loraOperationRequest);
    std::vector<ExecuteResponse> loraOperationResponses;
    if (!communicator_->SendSharedSyncRequestAndReceive(request, loraOperationResponses)) {
        MINDIE_LLM_LOG_ERROR("Failed to send LoadLoraOperation request to worker.");
        return false;
    }
    if (loraOperationResponses.size() != 1) {
        MINDIE_LLM_LOG_ERROR("Invalid LoadLoraOperation response size: " << loraOperationResponses.size());
        return false;
    }
    if (!HandleLoraOperationResponse(loraOperationResponses[0])) {
        MINDIE_LLM_LOG_ERROR("Failed to handle LoadLoraOperation response.");
        return false;
    }
    MINDIE_LLM_LOG_DEBUG("Successfully set LoadLoraOperation.");
    return true;
}

IExecutorSPtr CreateExecutor() { return std::make_shared<Executor>(); }

}  // namespace mindie_llm
