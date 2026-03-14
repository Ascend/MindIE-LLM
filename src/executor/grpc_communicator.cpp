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
#include "grpc_communicator.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/server_builder.h>
#include <experimental/filesystem>

using grpc::Server;
using grpc::ServerBuilder;

namespace fs = std::experimental::filesystem;
namespace mindie_llm {
bool ReadFileToString(const fs::path &filePath, std::string &outContent)
{
    std::string path = filePath.string();
    if (!CanonicalPath(path)) {
        MINDIE_LLM_LOG_ERROR("Invalid Path: " + path);
        return false;
    }
    std::ifstream file(path);
    if (!file) {
        MINDIE_LLM_LOG_ERROR("Cannot open file: " + path);
        return false;
    }
    std::stringstream buf;
    buf << file.rdbuf();
    outContent = buf.str();
    return true;
}

// SERVER上允许最大链接线程数
constexpr auto MAX_CONTACT_THREAD_NUM = 100;
std::shared_ptr<GRPCCommunicator>
GRPCCommunicator::GetInstance(const std::unordered_map<std::string, std::string> &modelConfig)
{
    static std::shared_ptr<GRPCCommunicator> instance = std::make_shared<GRPCCommunicator>(modelConfig);
    return instance;
}

GRPCCommunicator::GRPCCommunicator(const std::unordered_map<std::string, std::string> &modelConfig)
{
    isMaster_ = modelConfig.at("isMaster") == "1";
    std::vector<std::string> slaveIPs;
    mindie_llm::Split(modelConfig.at("slaveIPs"), ",", slaveIPs);
    slaveCount_ = slaveIPs.size();
    masterIP_ = modelConfig.at("masterIP");
    multiNodesInferPort_ = modelConfig.at("multiNodesInferPort");
    slaveIp_ = modelConfig.at("localIP");

    // 读取证书及安全相关配置
    std::string homePath;
    GetHomePath(homePath);
    auto it = modelConfig.find("interNodeTLSEnabled");
    interNodeTLSEnabled_ = (it != modelConfig.end() && it->second == "1");
    if (interNodeTLSEnabled_) {
        interNodeTlsCaPath_ = fs::path(homePath) / modelConfig.at("interNodeTlsCaPath");
        mindie_llm::Split(modelConfig.at("interNodeTlsCaFiles"), ",", interNodeTlsCaFiles_);
        interNodeTlsCert_ = fs::path(homePath) / modelConfig.at("interNodeTlsCert");
        interNodeTlsPk_ = fs::path(homePath) / modelConfig.at("interNodeTlsPk");
        interNodeTlsCrlPath_ = fs::path(homePath) / modelConfig.at("interNodeTlsCrlPath");
        mindie_llm::Split(modelConfig.at("interNodeTlsCrlFiles"), ",", interNodeTlsCrlFiles_);
        if (!LoadCertificates()) {
            MINDIE_LLM_LOG_ERROR("Failed to load TLS certificates. Shutting down.");
            throw std::runtime_error("Failed to load TLS certificates");
        }
    }
}

void GRPCCommunicator::StopServer()
{
    if (server_) {
        server_->Shutdown();
    }
    if (masterWorkerThread_.joinable()) {
        masterWorkerThread_.join();
    }
    MINDIE_LLM_LOG_INFO("gRPC server shutdown complete");
}

void GRPCCommunicator::StopClient()
{
    // 1. 关闭流式连接
    if (slaveStream_) {
        slaveStream_->WritesDone();
        grpc::Status status = slaveStream_->Finish();
        if (!status.ok()) {
            MINDIE_LLM_LOG_ERROR("Stream shutdown error: " + status.error_message());
        }
    }

    // 2. 停止工作线程
    if (slaveWorkerThread_.joinable()) {
        slaveWorkerThread_.join();
    }

    // 3. 释放资源
    slaveStream_.reset();
    stub_.reset();
    channel_.reset();

    MINDIE_LLM_LOG_INFO("gRPC connection shutdown complete");
}

GRPCCommunicator::~GRPCCommunicator()
{
    MINDIE_LLM_LOG_INFO("GRPCCommunicator Starting destruction");
    if (isMaster_) {
        StopServer();
    } else {
        StopClient();
    }

    MINDIE_LLM_LOG_INFO("GRPCCommunicator destruction completed");
}

bool GRPCCommunicator::Init(int initCount)
{
    // 确保仅在最后一次调用Init()时初始化，此时所有NPU节点启动信息已收集，Slave可向Master注册。
    int oldCallInitCount = callInitCount_.fetch_add(1, std::memory_order_acq_rel);
    if (oldCallInitCount == initCount - 1) {
        MINDIE_LLM_LOG_INFO("Start to init GRPCCommunicator");
        if (isMaster_) {
            // On Master node, the number of response handler threads equals to the number of remote DP ranks
            return InitMaster(initCount);
        } else {
            return InitSlave();
        }
    } else {
        if (isMaster_) { // Master节点需要等待所有Slave节点连接成功后才返回
            WaitForAllSlavesConnected();
        }
    }
    return true; // 如果不是最后一次调用，直接返回true
}

// 需要保证master节点收到所有slave节点的注册信息后才初始化完成
bool GRPCCommunicator::InitMaster(int respHandlerThreadCount)
{
    MINDIE_LLM_LOG_INFO("GRPCCommunicator: Start to init as Master");
    service_ = std::make_shared<MasterServiceImpl>(this, respHandlerThreadCount);
    masterWorkerThread_ = std::thread([this]() {
        std::string localAddr = FormatGrpcAddress(masterIP_, multiNodesInferPort_);
        pthread_setname_np(pthread_self(), "GRPCServer");
        ServerBuilder builder;
        builder.AddChannelArgument(GRPC_ARG_MAX_CONCURRENT_STREAMS, maxConcurrentStreams);
        builder.SetMaxReceiveMessageSize(grpcSendReceiveBufSize);
        builder.SetMaxSendMessageSize(grpcSendReceiveBufSize);
        std::shared_ptr<grpc::ServerCredentials> creds;
        if (interNodeTLSEnabled_) {
            std::vector<grpc::experimental::IdentityKeyCertPair> identityKeyCertPairList = {
                {tlsCertPrivateKey_, tlsCert_}};
            std::shared_ptr<grpc::experimental::CertificateProviderInterface> certificateProvider =
                std::make_shared<grpc::experimental::StaticDataCertificateProvider>(caCert_, identityKeyCertPairList);
            grpc::experimental::TlsServerCredentialsOptions tlsServerOpts(certificateProvider);
            // 配置认证方式为双向认证
            tlsServerOpts.set_cert_request_type(GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);

            if (!interNodeTlsCrlPath_.empty() && !interNodeTlsCrlFiles_.empty()) {
                std::vector<std::string> crlContentVec;
                for (const auto& crlFile : interNodeTlsCrlFiles_) {
                    fs::path crlPath = fs::path(interNodeTlsCrlPath_) / crlFile;
                    std::string crlContent;
                    ReadFileToString(crlPath, crlContent);
                    crlContentVec.emplace_back(crlContent);
                }
                if (!crlContentVec.empty()) {
                    auto crlProviderSpan = grpc_core::experimental::CreateStaticCrlProvider(crlContentVec);
                    auto crlProvider = crlProviderSpan.value_or(nullptr);
                    if (crlProvider == nullptr) {
                        MINDIE_LLM_LOG_ERROR("Failed to create crl provider");
                        return false;
                    }
                    tlsServerOpts.set_crl_provider(crlProvider);
                }
            }
            tlsServerOpts.watch_root_certs();
            tlsServerOpts.watch_identity_key_cert_pairs();
            creds = grpc::experimental::TlsServerCredentials(tlsServerOpts);
        } else {
            creds = grpc::InsecureServerCredentials();
        }

        builder.AddListeningPort(localAddr, creds);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        if (!server_) {
            MINDIE_LLM_LOG_ERROR("Failed to start gRPC server on port " + multiNodesInferPort_);
            return false;
        }
        MINDIE_LLM_LOG_INFO("gRPC server started on port " + multiNodesInferPort_ + " with " +
                            (interNodeTLSEnabled_ ? "TLS" : "no encryption"));

        server_->Wait();
        return true;
    });

    // 插入用于处理同步请求的blocking queue，dpRankIdx为responseHandlers_中的key
    std::shared_ptr<MasterServiceImpl> masterService = std::static_pointer_cast<MasterServiceImpl>(service_);
    for (int dpRankIdx : responseHandlers_.KeySet()) {
        masterService->DPRankIdxToSyncResp().Insert(dpRankIdx, std::make_shared<ExecRespBlockingQueue>());
    }

    // 阻塞线程，需要等待所有slave节点连接成功后继续运行
    MINDIE_LLM_LOG_INFO("GRPCCommunicator: wait slave connecting...");
    WaitForAllSlavesConnected();
    MINDIE_LLM_LOG_INFO("GRPCCommunicator: All " + std::to_string(slaveCount_) + " slaves connected");
    return true;
}

bool GRPCCommunicator::InitSlave()
{
    MINDIE_LLM_LOG_INFO("GRPCCommunicator: Start to init as Slave (IP=" + slaveIp_ + ")");
    int retryCount = 0;
    int sleepInterval = 1;
    int maxRetries = 120; // 重试120次,持续2分钟,等待master端口启动成功（假设每次重试间隔1秒）
    bool connected = false;
    while (retryCount++ < maxRetries) {
        try {
            MINDIE_LLM_LOG_INFO("GRPCCommunicator: attempting connection to server at IP = " + masterIP_ + ", port = " + multiNodesInferPort_);
            grpc::ChannelArguments channelArgs;
            channelArgs.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, maxConcurrentStreams);
            channelArgs.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, grpcSendReceiveBufSize);
            channelArgs.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, grpcSendReceiveBufSize);
            std::shared_ptr<grpc::ChannelCredentials> creds;
            if (interNodeTLSEnabled_) {
                std::vector<grpc::experimental::IdentityKeyCertPair> identityKeyCertPairList = {
                    {tlsCertPrivateKey_, tlsCert_}};
                std::shared_ptr<grpc::experimental::CertificateProviderInterface> certificateProvider =
                    std::make_shared<grpc::experimental::StaticDataCertificateProvider>(caCert_, identityKeyCertPairList);
                auto tlsChannelOpts = std::make_unique<grpc::experimental::TlsChannelCredentialsOptions>();
                tlsChannelOpts->set_certificate_provider(certificateProvider);

                // --- CRL 证书吊销列表 ---
                if (!interNodeTlsCrlPath_.empty() && !interNodeTlsCrlFiles_.empty()) {
                    std::vector<std::string> crlContentVec;
                    for (const auto& crlFile : interNodeTlsCrlFiles_) {
                        fs::path crlPath = fs::path(interNodeTlsCrlPath_) / crlFile;
                        std::string crlContent;
                        ReadFileToString(crlPath, crlContent);
                        crlContentVec.emplace_back(crlContent);
                    }
                    if (!crlContentVec.empty()) {
                        auto crlProviderSpan = grpc_core::experimental::CreateStaticCrlProvider(crlContentVec);
                        auto crlProvider = crlProviderSpan.value_or(nullptr);
                        if (crlProvider == nullptr) {
                            MINDIE_LLM_LOG_ERROR("Failed to create crl provider");
                            return false;
                        }
                        tlsChannelOpts->set_crl_provider(crlProvider);
                    }
                }
                tlsChannelOpts->watch_root_certs();
                tlsChannelOpts->watch_identity_key_cert_pairs();
                creds = grpc::experimental::TlsCredentials(*tlsChannelOpts);
            } else {
                creds = grpc::InsecureChannelCredentials();
            }
            channel_ =
                grpc::CreateCustomChannel(FormatGrpcAddress(masterIP_, multiNodesInferPort_), creds, channelArgs);
            stub_ = MasterService::NewStub(channel_);
            context_ = std::make_unique<grpc::ClientContext>();
            slaveStream_ = stub_->RegisterAndCommunicate(context_.get());
            if (!slaveStream_) {
                // 连接失败，等待1秒后重试
                MINDIE_LLM_LOG_WARN("Failed to establish bidirectional stream to master. "
                                    << "The master may not be ready yet. Retrying in 1 second...");
                std::this_thread::sleep_for(std::chrono::seconds(sleepInterval));
                continue;
            }
            MINDIE_LLM_LOG_INFO("Successfully connected to master and obtained slaveStream");

            // 发送注册信息,如果注册失败则等待1秒后重试
            if (SendRegistration()) {
                MINDIE_LLM_LOG_INFO("Registration succeeded");
                connected = true;
                break;
            } else {
                slaveStream_->WritesDone();
                slaveStream_->Finish();
                MINDIE_LLM_LOG_WARN("Send registration message to master failed. Retrying in 1 second...");
                std::this_thread::sleep_for(std::chrono::seconds(sleepInterval));
            }
        } catch (const std::exception &e) {
            MINDIE_LLM_LOG_ERROR("gRPC CreateChannel Error: " + std::string(e.what()));
        } catch (...) {
            MINDIE_LLM_LOG_ERROR("gRPC CreateChannel failed with unknown exception");
        }
    }
    if (!connected) {
        MINDIE_LLM_LOG_ERROR("Failed to establish connection to master after maximum retries");
        return false;
    }
    // 启动工作线程处理任务
    StartWorkerThread();
    return true;
}

void GRPCCommunicator::WaitForAllSlavesConnected()
{
    std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    cv_.wait(lock, [this] { return AllSlavesConnected(); });
}

bool GRPCCommunicator::SendRegistration()
{
    SlaveToMasterMsg msg;
    auto *reg = msg.mutable_register_request();
    reg->set_slave_ip(slaveIp_);
    MINDIE_LLM_LOG_INFO("Sent registration to master: slave_ip=" + slaveIp_);
    if (slaveStream_->Write(msg)) {
        return true;
    } else {
        return false;
    }
}

void GRPCCommunicator::StartWorkerThread()
{
    slaveWorkerThread_ = std::thread([this] {
        pthread_setname_np(pthread_self(), "GRPCSlave");
        MasterToSlaveMsg task;
        try {
            while (slaveStream_->Read(&task)) {
                int targetDPRank = task.target_dp_rank();
                ExecuteRequest request = task.execute_request();
                HandleRequestFromMaster(request, targetDPRank);
            }
            MINDIE_LLM_LOG_INFO("gRPC Slave Worker Thread: stream closed by server");
        } catch (const std::exception &e) {
            MINDIE_LLM_LOG_ERROR("gRPC Slave Worker Thread Exception: " + std::string(e.what()));
        } catch (...) {
            MINDIE_LLM_LOG_ERROR("gRPC Slave Worker Thread unknown exception");
        }
    });
}

template <typename StreamType, typename MsgType>
bool GRPCCommunicator::SafeWriteMsgToStream(StreamType stream, const MsgType &msg)
{
    if (!stream) {
        MINDIE_LLM_LOG_ERROR("SafeWriteMsgToStream: stream is null (cannot write message)");
        return false;
    }
    std::lock_guard<std::mutex> lock(streamWriteMutex_); // 确保线程安全
    if (!stream->Write(msg)) {
        MINDIE_LLM_LOG_ERROR("SafeWriteMsgToStream: failed to write message to stream");
        return false;
    }
    return true;
}

bool GRPCCommunicator::SendRequest(ExecuteRequest &request, int sourceDPRank, int targetDPRank,
                                   const std::string &slaveIp)
{
    if (sourceDPRank < 0 || targetDPRank < 0) {
        MINDIE_LLM_LOG_ERROR("SendRequest: sourceDPRank and targetDPRank must be non-negative integers.");
        return false;
    }
    MasterToSlaveMsg msg;
    msg.set_source_dp_rank(sourceDPRank);
    msg.set_target_dp_rank(targetDPRank);
    *msg.mutable_execute_request() = request;

    if (slaveIp.empty()) { // 广播
        for (std::optional<SlaveStreamPtr> stream : slaveIpToStream_.Values()) {
            if (!SafeWriteMsgToStream(stream.value_or(nullptr), msg)) {
                return false;
            }
        }
    } else { // 单播
        std::optional<SlaveStreamPtr> stream = slaveIpToStream_.Get(slaveIp);
        if (!SafeWriteMsgToStream(stream.value_or(nullptr), msg)) {
            return false;
        }
    }
    return true;
}

bool GRPCCommunicator::GetSyncResponse(ExecuteResponse &response, int sourceDPRank)
{
    std::shared_ptr<MasterServiceImpl> masterService = std::static_pointer_cast<MasterServiceImpl>(service_);
    return masterService->Take(sourceDPRank, response);
}

bool GRPCCommunicator::SendResponse(ExecuteResponse &response, int sourceDPRank, int targetDPRank)
{
    if (sourceDPRank < 0 || targetDPRank < 0) {
        MINDIE_LLM_LOG_ERROR("SendResponse: sourceDPRank and targetDPRank must be non-negative integers.");
        return false;
    }
    SlaveToMasterMsg msg;
    msg.set_source_dp_rank(sourceDPRank);
    msg.set_target_dp_rank(targetDPRank);
    *msg.mutable_execute_response() = response;

    if (!SafeWriteMsgToStream(slaveStream_.get(), msg)) {
        MINDIE_LLM_LOG_ERROR("SendResponse: failed to write response to slave stream.");
        return false;
    }
    return true;
}

template <typename HandlerType>
bool RegisterHandler(ConcurrentMap<int, HandlerType> &handlers, int dpRankIdx, HandlerType handler)
{
    if (handler == nullptr) {
        MINDIE_LLM_LOG_ERROR("GRPC RegisterHandler: handler is null.");
        return false;
    }
    if (handlers.Count(dpRankIdx) > 0) {
        MINDIE_LLM_LOG_ERROR("GRPC RegisterHandler: handler for dpRankIdx " << dpRankIdx << " is already registered.");
        return false;
    }
    handlers.Insert(dpRankIdx, handler);
    return true;
}

bool GRPCCommunicator::RegisterRequestHandler(RequestHandler handler, int dpRankIdx)
{
    return RegisterHandler(requestHandlers_, dpRankIdx, handler);
}

bool GRPCCommunicator::RegisterResponseHandler(ResponseHandler handler, int dpRankIdx)
{
    return RegisterHandler(responseHandlers_, dpRankIdx, handler);
}

// 以下函数会并发调用，需要保证线程安全
bool GRPCCommunicator::HandleResponseFromSlave(ExecuteResponse &response, int targetDPRank)
{
    // ResponseHandler 对应于 ModelExecOutputHandler::Entry4Executor() 函数指针，并且是线程安全的
    std::optional<ResponseHandler> optHandler = responseHandlers_.Get(targetDPRank);
    if (!optHandler.has_value()) {
        MINDIE_LLM_LOG_ERROR("HandleResponseFromSlave: response handler for targetDPRank "
                             << targetDPRank << " is not set or does not exist.");
        return false;
    }
    try {
        // 调用对应的处理函数
        optHandler.value()(response);
    } catch (const std::exception &e) {
        MINDIE_LLM_LOG_ERROR("HandleResponseFromSlave: exception occurred while handling response: " +
                             std::string(e.what()));
        return false;
    } catch (...) {
        MINDIE_LLM_LOG_ERROR("HandleResponseFromSlave: unknown exception occurred while handling response.");
        return false;
    }
    return true;
}

void GRPCCommunicator::HandleRequestFromMaster(ExecuteRequest &request, int targetDPRank)
{
    if (request.execute_type() == MODEL_INFER ||
        request.execute_type() == RECOVER_COMMAND_EXEC ||
        request.execute_type() == START_COMMAND_EXEC ||
        request.execute_type() == PAUSE_COMMAND_EXEC ||
        request.execute_type() == CLEAR_COMMAND_EXEC) {
        // MODEL_INFER request will be handled by all DP ranks in the slave node
        std::vector<RequestHandler> handlers = requestHandlers_.Values();
        for (const auto &handler : handlers) {
            handler(request);
        }
    } else {
        std::optional<RequestHandler> optHandler = requestHandlers_.Get(targetDPRank);
        if (!optHandler.has_value()) {
            MINDIE_LLM_LOG_ERROR("GRPCCommunicator: request handler for targetDPRank "
                                 << targetDPRank << " is not set or does not exist.");
            return;
        }
        optHandler.value()(request); // 调用对应的处理函数
    }
}

bool GRPCCommunicator::AllSlavesConnected() { return slaveIpToStream_.Size() >= slaveCount_; }
void GRPCCommunicator::NotifyAll() { cv_.notify_all(); }

ConcurrentMap<std::string, SlaveStreamPtr> &GRPCCommunicator::SlaveIpToStream() { return slaveIpToStream_; }

MasterServiceImpl::MasterServiceImpl(GRPCCommunicator *comm, int respHandlerThreadCount) : gRPCCommunicator_(comm)
{
    respHandlerThreads_.reserve(respHandlerThreadCount);
    for (int i = 0; i < respHandlerThreadCount; ++i) {
        respHandlerThreads_.emplace_back([this] { RespHandlerLoop(); });
        pthread_setname_np(respHandlerThreads_.back().native_handle(), "GRPCResponseHandler");
    }
}

MasterServiceImpl::~MasterServiceImpl() { StopRespHandlerThreads(); }

void MasterServiceImpl::RespHandlerLoop()
{
    while (respHandlerThreadActive_.load(std::memory_order_relaxed)) {
        // 从阻塞队列中获取响应任务并处理，处理函数需保证线程安全
        std::shared_ptr<SlaveResponseTask> task = pendingRespFromSlaveQueue_.pull();
        gRPCCommunicator_->HandleResponseFromSlave(*task->response, task->targetDPRank);
    }
}

void MasterServiceImpl::StopRespHandlerThreads()
{
    respHandlerThreadActive_.store(false, std::memory_order_relaxed);
    pendingRespFromSlaveQueue_.close();
    for (auto &thread : respHandlerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    respHandlerThreads_.clear();
}

grpc::Status MasterServiceImpl::RegisterAndCommunicate(ServerContext *context, SlaveStreamPtr stream)
{
    SlaveToMasterMsg client_msg;
    while (stream->Read(&client_msg)) {
        if (client_msg.has_register_request()) {
            auto &register_request = client_msg.register_request();
            gRPCCommunicator_->SlaveIpToStream().Insert(register_request.slave_ip(), stream);
            (void)context;
            if (gRPCCommunicator_->AllSlavesConnected()) {
                gRPCCommunicator_->NotifyAll();
            }
        } else if (client_msg.has_execute_response()) {
            int targetDPRank = client_msg.target_dp_rank();
            ExecuteResponse executeResponse = client_msg.execute_response();
            // If the response type is REMOTE_MODEL_INIT, PD_LINK, LORA_OPERATION, RECOVER_COMMAND_EXEC, push to queue
            if (executeResponse.msg_type() == REMOTE_MODEL_INIT || executeResponse.msg_type() == PD_LINK ||
                executeResponse.msg_type() == LORA_OPERATION ||
                executeResponse.msg_type() == RECOVER_COMMAND_EXEC ||
                executeResponse.msg_type() == START_COMMAND_EXEC ||
                executeResponse.msg_type() == PAUSE_COMMAND_EXEC ||
                executeResponse.msg_type() == CLEAR_COMMAND_EXEC) {
                dpRankIdxToSyncResp_.Get(targetDPRank)
                    .value()
                    ->push(std::make_shared<ExecuteResponse>(std::move(executeResponse)));
            } else {
                // Enqueue the asynchronous response for parallel processing by response handler threads.
                std::shared_ptr<SlaveResponseTask> respTask = std::make_shared<SlaveResponseTask>();
                respTask->targetDPRank = targetDPRank;
                respTask->response = std::make_shared<ExecuteResponse>(std::move(executeResponse));
                pendingRespFromSlaveQueue_.push(std::move(respTask));
            }
        }
    }
    return grpc::Status::OK;
}

bool MasterServiceImpl::Take(int targetDPRank, ExecuteResponse &response)
{
    auto blockingQueueOpt = dpRankIdxToSyncResp_.Get(targetDPRank);
    if (!blockingQueueOpt.has_value()) {
        MINDIE_LLM_LOG_ERROR("No blocking queue found for targetDPRank " << targetDPRank);
        return false;
    }
    // This call will block until a response is available
    response = *blockingQueueOpt.value()->pull();
    return true;
}

ConcurrentMap<int, std::shared_ptr<ExecRespBlockingQueue>> &MasterServiceImpl::DPRankIdxToSyncResp()
{
    return dpRankIdxToSyncResp_;
}

bool GRPCCommunicator::LoadCertificates()
{
    MINDIE_LLM_LOG_INFO("Loading TLS certificates for mutual authentication...");
    std::string homePath;
    GetHomePath(homePath);
    // 1. 加载所有 CA 证书
    caCert_.clear();
    for (const auto &caFile : interNodeTlsCaFiles_) {
        fs::path caPath = fs::path(interNodeTlsCaPath_) / caFile;
        ReadFileToString(caPath, caCert_);
        caCert_ += "\n";
        MINDIE_LLM_LOG_INFO("Loaded CA certificate: " + caPath.string());
    }

    // 2. 加载本端(服务端，客户端)证书
    fs::path certPath = interNodeTlsCert_;
    ReadFileToString(certPath, tlsCert_);
    MINDIE_LLM_LOG_INFO("Loaded server/client certificate: " + certPath.string());
  
    // 2. 读取本端证书的私钥
    fs::path keyPath = fs::path(interNodeTlsPk_);
    std::string keyContent;
    ReadFileToString(keyPath, keyContent);
 
    tlsCertPrivateKey_.assign(keyContent.data(), keyContent.size());

    return true;
}

} // namespace mindie_llm
