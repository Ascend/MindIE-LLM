/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <memory>
#include <regex>
#include <cstring> // for memset
#include <sys/socket.h> // for socket, bind, listen, close
#include <netinet/in.h> // for sockaddr_in, htons
#include <arpa/inet.h> // for inet_pton
#include <unistd.h> // for close
#include <cerrno> // for errno
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/server.h>
#include <grpc/grpc_crl_provider.h>
#include <src/core/lib/security/credentials/tls/grpc_tls_credentials_options.h>

#include "log.h"
#include "endpoint_def.h"
#include "param_checker.h"
#include "file_utils.h"
#include "memory_utils.h"
#include "config_manager.h"
#include "common_util.h"
#include "infer_instances.h"
#include "grpc_tls_function_expansion.h"
#include "check_utils.h"
#include "basic_types.h"
#include "config_manager_impl.h"
#include "grpc_communication_mng.h"

static const int ERR_MSG_BUF_LEN = 256;
static const int MAX_MESSAGE_LENGTH = 16 * 1024 * 1024;

namespace mindie_llm {
        using grpc::Server;
        using grpc::ServerBuilder;

        // SERVER上允许最大链接线程数
        constexpr auto MAX_CONTACT_THREAD_NUM = 2000;
        // http2上允许的并发传入流的最大数量
        constexpr auto MAX_CONCURRENT_STREAMS = 2000;

        struct BioDeleter {
            void operator()(BIO* bio) const
            {
                if (bio != nullptr) {
                    // 获取bio中的数据和长度
                    char* data = nullptr;
                    auto dataLen = BIO_get_mem_data(bio, &data);
                    if (dataLen > 0) {
                        OPENSSL_cleanse(data, dataLen);
                    }
                    BIO_vfree(bio);
                }
            }
        };

        struct PkeyDeleter {
            void operator()(EVP_PKEY* pkey) const
            {
                if (pkey != nullptr) {
                    EVP_PKEY_free(pkey);
                }
            }
        };

        using BioManager = std::unique_ptr<BIO, BioDeleter>;
        using PkeyManager = std::unique_ptr<EVP_PKEY, PkeyDeleter>;

        GrpcCommunicationMng::~GrpcCommunicationMng()
        {
            try {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Enter destructor ~GrpcCommunicationMng()");
                StopServerThread();
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Finish destructor ~GrpcCommunicationMng()");
            } catch (const std::exception &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST,
                    REMOVE_ERROR), "Exception in destructor ~GrpcCommunicationMng(). " << e.what());
            } catch (...) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SERVER_REQUEST,
                    REMOVE_ERROR), "Unknown exception in destructor ~GrpcCommunicationMng()");
            }
        }

        bool GrpcCommunicationMng::RegisterDecodeRequestHandler(DecodeRequestHandler decodeRequestHandler)
        {
            if (isRunning_ == true) {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Server is already running");
                return false;
            }
            decodeRequestHandler_ = std::move(decodeRequestHandler);
            return true;
        }

        bool GrpcCommunicationMng::RegisterKvReleaseHandler(KVReleaseHandler kvReleaseHandler)
        {
            if (isRunning_ == true) {
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Server is running");
                return false;
            }
            kvReleaseHandler_ = std::move(kvReleaseHandler);
            return true;
        }

        bool GrpcCommunicationMng::RunServer()
        {
            ServerBuilder builder;

            // 配置最大线程数
            grpc::ResourceQuota quota;
            quota.SetMaxThreads(MAX_CONTACT_THREAD_NUM);
            builder.SetResourceQuota(quota);
            // 设置最大传输长度为16M
            builder.SetMaxReceiveMessageSize(MAX_MESSAGE_LENGTH);
            // 设置最大并发流数量
            builder.AddChannelArgument(GRPC_ARG_MAX_CONCURRENT_STREAMS, MAX_CONCURRENT_STREAMS);
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "ServerAddr is  " << localAddr_);
            if (useTls_) {
                ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Server run with ssl.");
                // get certificate_provider
                std::shared_ptr<grpc::experimental::CertificateProviderInterface> certificateProvider = nullptr;
                if (!GetCertificateProvider(certificateProvider)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to get certificateProvider");
                    return false;
                }
                grpc::experimental::TlsServerCredentialsOptions tlsServerOpts(certificateProvider);
                // 配置认证方式为双向认证
                tlsServerOpts.set_cert_request_type(GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
                if (!SetTlsOps(tlsServerOpts)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to set tls opt");
                    return false;
                }
                builder.AddListeningPort(localAddr_, grpc::experimental::TlsServerCredentials(tlsServerOpts));
            } else {
                builder.AddListeningPort(localAddr_, grpc::InsecureServerCredentials());
            }
            return this->RegisterAndStartService(localAddr_, builder);
        }

        bool GrpcCommunicationMng::RegisterAndStartService(const std::string &serverAddr, ServerBuilder &builder)
        {
            // decode请求服务注册
            DecodeRequestReceiver decodeService{serverAddr};
            if (!decodeService.RegisterMsgHandler(decodeRequestHandler_)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to register decode request handler");
                return false;
            }
            builder.RegisterService(&decodeService);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "RegisterService decoder " << serverAddr);

            // kv释放服务注册
            KvReleaseReceiver kvService{serverAddr};
            if (!kvService.RegisterMsgHandler(kvReleaseHandler_)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to register kv release handler");
                return false;
            }
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "RegisterService kvService " << serverAddr);
            builder.RegisterService(&kvService);

            std::unique_ptr<Server> server(builder.BuildAndStart());
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Bind port " << serverAddr << " success");
            isRunning_.store(true);
            if (sem_wait(&terminateSem_) != 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    SYSTEM_INVOKING_ERROR), "Failed to wait terminate semaphore");
                return false;
            }
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Shutting down gRPC server...");
            return true;
        }

        bool GrpcCommunicationMng::IsValidNetPort(const std::string& port) const
        {
            std::regex pattern("^([0-9]|[1-9]\\d{1,3}|[1-5]\\d{4}|6[0-5]{2}[0-3][0-5])$");
            if (!std::regex_match(port, pattern)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The port is invalid");
                return false;
            }
            return true;
        }

        bool GrpcCommunicationMng::IsValidNetAddr(const std::string& ip) const
        {
            std::string netAddr = ip;
            if (ip.find(IP_PORT_DELIMITER) != std::string::npos) {
                netAddr = SplitString(ip, ';')[0];
            }

            auto &serverConfig = GetServerConfig();
            if (!CheckIp(netAddr, "grpc ip", serverConfig.allowAllZeroIpListening)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The ip is invalid");
                return false;
            }
            return true;
        }

        bool GrpcCommunicationMng::IsValidServerAddr(const std::string& ip, const std::string& port) const
        {
            bool isIPv6 = IsIPv6(ip);
            bool isIPv4 = IsIPv4(ip);
            if (!isIPv4 && !isIPv6) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Invalid IP address format");
                return false;
            }
            uint16_t portValue = 0;
            try {
                portValue = std::stoi(port);
            } catch (...) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Invalid port");
                return false;
            }
            
            // 创建并配置socket
            int sockfd;
            if (!CreateAndConfigureSocket(sockfd, isIPv6)) {
                return false;
            }

            bool bindResult = false;
            if (isIPv6) {
                bindResult = BindIPv6Address(sockfd, ip, portValue);
            } else {
                bindResult = BindIPv4Address(sockfd, ip, portValue);
            }

            close(sockfd);
            return bindResult;
        }

        bool GrpcCommunicationMng::CreateAndConfigureSocket(int& sockfd, bool isIPv6) const
        {
            if (isIPv6) {
                sockfd = socket(AF_INET6, SOCK_STREAM, 0);
            } else {
                sockfd = socket(AF_INET, SOCK_STREAM, 0);
            }
            
            if (sockfd == -1) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to create socket");
                return false;
            }

            int enable = 1;
            if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) == -1) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to setsockopt");
                close(sockfd);
                return false;
            }
            
            return true;
        }

        bool GrpcCommunicationMng::BindIPv4Address(int sockfd, const std::string& ipPart, uint16_t port) const
        {
            struct sockaddr_in sockAddrIn;
            if (memset_s(&sockAddrIn, sizeof(sockAddrIn), 0, sizeof(sockAddrIn)) != EOK) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to memset");
                return false;
            }
            
            sockAddrIn.sin_family = AF_INET;
            sockAddrIn.sin_port = htons(port);

            if (inet_pton(AF_INET, ipPart.c_str(), &sockAddrIn.sin_addr) <= 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to inet_pton for IPv4");
                return false;
            }

            if (bind(sockfd, reinterpret_cast<struct sockaddr *>(&sockAddrIn), sizeof(sockAddrIn)) == -1) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to bind IPv4 server addr");
                return false;
            }
            
            return true;
        }

        bool GrpcCommunicationMng::BindIPv6Address(int sockfd, const std::string& ipPart, uint16_t port) const
        {
            struct sockaddr_in6 sockAddrIn6;
            if (memset_s(&sockAddrIn6, sizeof(sockAddrIn6), 0, sizeof(sockAddrIn6)) != EOK) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to memset");
                return false;
            }
            
            sockAddrIn6.sin6_family = AF_INET6;
            sockAddrIn6.sin6_port = htons(port);

            if (inet_pton(AF_INET6, ipPart.c_str(), &sockAddrIn6.sin6_addr) <= 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to inet_pton for IPv6");
                return false;
            }
            if (bind(sockfd, reinterpret_cast<struct sockaddr *>(&sockAddrIn6), sizeof(sockAddrIn6)) == -1) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to bind IPv6 server addr");
                return false;
            }
            
            return true;
        }

        bool GrpcCommunicationMng::GetClientTlsOpts(
            std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions>& tlsChannelOpts)
        {
            try {
                tlsChannelOpts = std::move(std::make_unique<grpc::experimental::TlsChannelCredentialsOptions>());
            } catch (const std::bad_alloc &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to alloc mem.");
                return false;
            } catch (...) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to make unique ptr.");
                return false;
            }
            // get certificate_provider
            std::shared_ptr<grpc::experimental::CertificateProviderInterface> certificateProvider = nullptr;
            if (!GetCertificateProvider(certificateProvider)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get certificateProvider");
                return false;
            }
            // 设置证书
            tlsChannelOpts->set_certificate_provider(certificateProvider);
            if (!SetTlsOps(*tlsChannelOpts)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to set tls opt");
                return false;
            }
            return true;
        }

        bool GrpcCommunicationMng::CreateDecodeRequestSender(const std::string& decodeRequestReceiveNode)
        {
            std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> tlsChannelOpts = nullptr;
            std::unique_ptr<DecodeRequestSender> decodeRequestSender = nullptr;
            if (useTls_) {
                if (!GetClientTlsOpts(tlsChannelOpts)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to init client tls opt");
                    return false;
                }
            }
            try {
                decodeRequestSender = std::make_unique<DecodeRequestSender>(decodeRequestReceiveNode, localAddr_,
                    useTls_, std::move(tlsChannelOpts));
            } catch (const std::bad_alloc &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to alloc mem.");
                return false;
            } catch (...) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to make unique ptr.");
                return false;
            }
            if (decodeRequestSender == nullptr || !decodeRequestSender->Init()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to init decode sender");
                return false;
            }
            decodeRequestSenders_[decodeRequestReceiveNode] = std::move(decodeRequestSender);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Init decode sender success");
            return true;
        }

        bool GrpcCommunicationMng::CreateKvReleaseSender(const std::string& kvReleaseReceiveNode)
        {
            std::unique_ptr<grpc::experimental::TlsChannelCredentialsOptions> tlsChannelOpts = nullptr;
            std::unique_ptr<KvReleaseSender> kvReleaseSender = nullptr;
            if (useTls_) {
                if (!GetClientTlsOpts(tlsChannelOpts)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        SECURITY_ERROR), "Failed to init client tls opt");
                    return false;
                }
            }
            try {
                kvReleaseSender = std::make_unique<KvReleaseSender>(kvReleaseReceiveNode, localAddr_,
                    useTls_, std::move(tlsChannelOpts));
            } catch (const std::bad_alloc &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to alloc mem.");
                return false;
            } catch (...) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                    GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE, ABNORMAL_TRANSMISSION_ERROR),
                    "Failed to make unique ptr.");
                return false;
            }
            if (kvReleaseSender == nullptr || !kvReleaseSender->Init()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to init kv sender");
                return false;
            }
            kvReleaseSenders_[kvReleaseReceiveNode] = std::move(kvReleaseSender);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Init kv sender success");
            return true;
        }

        void GrpcCommunicationMng::StopServerThread()
        {
            if (terminateSemInitialized_ && sem_post(&terminateSem_) != 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to post terminate semaphore");
            }
            if (serverThread_.joinable()) {
                serverThread_.join();
            }
            isRunning_.store(false);
            if (terminateSemInitialized_ && sem_destroy(&terminateSem_) != 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to destroy terminate semaphore");
            }
        }

        bool GrpcCommunicationMng::Init(bool useTls, const std::string& localIp, const std::string& port)
        {
            if (sem_init(&terminateSem_, 0, 0) != 0) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to init terminate semaphore");
                return false;
            }
            terminateSemInitialized_ = true;
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "useTls is " << useTls << ", localIp is " << localIp
                << ", port is " << port);
            if (!IsValidNetAddr(localIp) || !IsValidNetPort(port)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The netaddr or port is invalid");
                return false;
            }
            useTls_ = useTls;
            port_ = port;
            localAddr_ = FormatGrpcAddress(localIp, port);
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "localAddr_ is " << localAddr_);
            if (useTls_) {
                if (!SetEnvForSecurity()) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to config cert tools");
                    return false;
                }
                if (!GetSecFilePath() || !GrpcTlsFunctionExpansion::CheckTlsOption(caPath_, certPath_, crlPath_)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to check sec file");
                    return false;
                }
            }

            if (!IsValidServerAddr(localIp, port)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The localAddr_ is invalid to bind");
                return false;
            }
            serverThread_ = std::thread([this]() { this->RunServer(); });
            pthread_setname_np(serverThread_.native_handle(), "GRPCRunServer");
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Init GrpcCommunicationMng success");
            return true;
        }

        bool GrpcCommunicationMng::SendDecodeRequest(prefillAndDecodeCommunication::DecodeParameters& decodeParameters,
            const std::string& decodeNodeIp, const std::string& reqId, std::string& errMsg)
        {
            std::string decodeNodeAddr = decodeNodeIp;
            FillIpAddress(decodeNodeAddr);
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Send decode request to node " << decodeNodeAddr);
            if (!IsValidNetAddr(decodeNodeIp)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), (errMsg = "Decode node addr is invalid"));
                return false;
            }
            std::shared_ptr<DecodeRequestSender> currSender;
            {
                std::lock_guard <std::mutex> lock(this->decodeReqSenderMutex_);
                auto iter = decodeRequestSenders_.find(decodeNodeAddr);
                if (iter == decodeRequestSenders_.end()) {
                    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Failed to Find decodeRequestSenders_ for " << decodeNodeAddr);
                    if (!CreateDecodeRequestSender(decodeNodeAddr)) {
                        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                            ABNORMAL_TRANSMISSION_ERROR), (errMsg = "Failed to create decode request sender"));
                        return false;
                    }
                    ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "Create decode request sender success.");
                    currSender = decodeRequestSenders_.find(decodeNodeAddr)->second;
                } else {
                    currSender = iter->second;
                }
            }

            if (currSender == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), (errMsg = "Failed to find DecodeRequestSender"));
                return false;
            }
            if (!currSender->SendDecodeRequestMsg(decodeParameters, reqId, errMsg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to send decode request" << errMsg);
                return false;
            }
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT, "[GrpcCommunicationMng] Finish sending decode request requestId:"
                                                  << reqId << " to node " << decodeNodeAddr);
            return true;
        }

        bool GrpcCommunicationMng::SendKvReleaseMsg(const prefillAndDecodeCommunication::RequestId& requestId,
            const std::string& prefillNodeIp)
        {
            // release 请求有端口
            std::string prefillNodeAddr = prefillNodeIp;
            FillIpAddress(prefillNodeAddr);
            ULOG_DEBUG(SUBMODLE_NAME_ENDPOINT,
                       "[GrpcCommunicationMng::SendKvReleaseMsg] send kv release request requestId: "
                           << requestId.reqid() << " to P node " << prefillNodeAddr);

            if (!IsValidNetAddr(prefillNodeIp)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Prefill node addr is invalid");
                return false;
            }

            std::shared_ptr<KvReleaseSender> currSender;
            {
                std::lock_guard <std::mutex> lock(this->kvReleaseSenderMutex_);
                auto iter = kvReleaseSenders_.find(prefillNodeAddr);
                if (iter == kvReleaseSenders_.end()) {
                    if (!CreateKvReleaseSender(prefillNodeAddr)) {
                        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                            ABNORMAL_TRANSMISSION_ERROR), "Failed to create kv release sender");
                        return false;
                    }
                    currSender = kvReleaseSenders_.find(prefillNodeAddr)->second;
                } else {
                    currSender = iter->second;
                }
            }

            if (currSender == nullptr) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to find KvReleaseSender");
                return false;
            }
            if (!currSender->SendKvReleaseMsg(requestId)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to send kv release");
                return false;
            }
            return true;
        }

        GrpcCommunicationMng& GrpcCommunicationMng::GetInstance()
        {
            static GrpcCommunicationMng instance;
            return instance;
        }

        bool GrpcCommunicationMng::SetTlsOps(grpc::experimental::TlsCredentialsOptions& tlsOpts)
        {
            if (!crlPath_.empty()) {
                std::vector<std::string> crlContentVec;
                for (const auto& crlPath : crlPath_) {
                    std::string crlContent;
                    if (!GetFileContent(crlPath, crlContent)) {
                        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                            CHECK_ERROR), "Failed to get crl content");
                        return false;
                    }
                    crlContentVec.push_back(crlContent);
                }
                auto crlProviderSpan = grpc_core::experimental::CreateStaticCrlProvider(crlContentVec);
                auto crlProvider = crlProviderSpan.value_or(nullptr);
                if (crlProvider == nullptr) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to get crl provider");
                    return false;
                }
                tlsOpts.set_crl_provider(crlProvider);
            }
            if (caPath_.empty()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Ca path is empty");
                return false;
            }
            std::string caPathContent;
            for (const auto& caPath : caPath_) {
                std::string caContent;
                if (!GetFileContent(caPath, caContent)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Ca path path content");
                    return false;
                }
                if (!caPathContent.empty()) {
                    caPathContent += "\n";
                }
                caPathContent += caContent;
            }
            tlsOpts.set_root_cert_name(caPathContent);
            tlsOpts.set_identity_cert_name(certPath_);
            tlsOpts.watch_root_certs();
            tlsOpts.watch_identity_key_cert_pairs();
            return true;
        }

        bool GrpcCommunicationMng::GetCertificateProvider(
            std::shared_ptr<grpc::experimental::CertificateProviderInterface>& certificateProvider)
        {
            // 服务端秘钥和证书
            std::string serverCertContent = "";
            if (!GetFileContent(certPath_, serverCertContent)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get cert file content");
                return false;
            }
            SensitiveInfoManager serverKeyContent{
                    nullptr, 0, MAX_PRIVATE_KEY_CONTENT_BYTE_LEN, MIN_PRIVATE_KEY_CONTENT_BYTE_LEN};
            if (!GetKeyContent(serverKeyContent)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get key file content");
                return false;
            }
            if (!serverKeyContent.IsValid()) {
                serverKeyContent.Clear();
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get key content");
                return false;
            }
            std::vector<grpc::experimental::IdentityKeyCertPair> identityKeyCertPairList{
                    {serverKeyContent.GetSensitiveInfoContent(), serverCertContent}};
            serverKeyContent.Clear();
            // get ca
            std::string serverCaContent;
            if (caPath_.empty()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                           CHECK_ERROR), "Failed to get ca file content");
                std::fill_n(identityKeyCertPairList[0].private_key.begin(),
                    identityKeyCertPairList[0].private_key.size(), '\0');
                return false;
            }
            for (const auto& caPath : caPath_) {
                if (!GetFileContent(caPath, serverCaContent)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                           CHECK_ERROR), "Failed to get ca file content");
                    std::fill_n(identityKeyCertPairList[0].private_key.begin(),
                        identityKeyCertPairList[0].private_key.size(), '\0');
                    return false;
                }
            }
            certificateProvider = std::make_shared<grpc::experimental::StaticDataCertificateProvider>(
                    serverCaContent, identityKeyCertPairList);
            std::fill_n(identityKeyCertPairList[0].private_key.begin(),
                identityKeyCertPairList[0].private_key.size(), '\0');
            return true;
        }

        bool GrpcCommunicationMng::GetFileContent(const std::string &filePath, std::string &content)
        {
            // 需要支持配置文件配置绝对路径，在此处做校验，绝对路径直接使用，相对路径转化为绝对路径
            std::string realFilePath = "";
            std::string errMsg = "";
            if (!FileUtils::GetRealFilePath(filePath, realFilePath, errMsg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get real file because " << errMsg);
                return false;
            }
            if (!FileUtils::IsFileValid(filePath, errMsg, true, FileUtils::FILE_MODE_600, false)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The file is invalid because " << errMsg);
                return false;
            }
            std::ifstream file(realFilePath);
            if (!file.is_open()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to open file");
                return false;
            }
            // 使用 stringstream 读取整个文件内容到 string
            std::stringstream buffer;
            buffer << file.rdbuf();
            file.close();
            content = buffer.str();
            buffer.str("");
            return true;
        }
        bool GrpcCommunicationMng::GetKeyContent(SensitiveInfoManager &keyContent)
        {
            // 获取私钥
            std::string privateKeyContent;
            if (!GetFileContent(certKeyPath_, privateKeyContent)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get private key content");

                return false;
            }
           
            if (!keyContent.CopySensitiveInfo(privateKeyContent.data(), privateKeyContent.size())) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    ABNORMAL_TRANSMISSION_ERROR), "Failed to get bioOut with key because " << strerror(errno));
                if (privateKeyContent.size() > 0) {
                    OPENSSL_cleanse(privateKeyContent.data(), privateKeyContent.size());
                }
                privateKeyContent = nullptr;
                return false;
            }
            if (privateKeyContent.size() > 0) {
                OPENSSL_cleanse(privateKeyContent.data(), privateKeyContent.size());
            }
            privateKeyContent = nullptr;
            return true;
        }

        bool GrpcCommunicationMng::SetEnvForSecurity()
        {
            std::string workDir = "";
            std::string errMsg = "";
            std::string regularPath = "";
            if (!FileUtils::GetInstallPath(workDir, errMsg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get install path because " << errMsg);
                return false;
            }
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Enter SetEnvForSecurity");
            std::string libPath = workDir + "lib";
            if (!FileUtils::RegularFilePath(libPath, workDir, errMsg, regularPath)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Get lib path failed");
                return false;
            }

            int ret = setenv("EP_OPENSSL_PATH", regularPath.c_str(), 1);
            if (ret) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Set ep openssl libPath failed, error :" << strerror(errno));
                return false;
            }
            return true;
        }

        bool GrpcCommunicationMng::GetSecFilePath()
        {
            if (!GetServerConfig().interCommTLSEnabled) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "The tls enable is invalid");
                return false;
            }
            std::string errMsg = "";
            auto interCommTlsCaFiles = GetServerConfig().interCommTlsCaFiles;
            if (interCommTlsCaFiles.empty()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Tls ca file is empty");
                return false;
            }
            for (const std::string &interCommTlsCaFile : interCommTlsCaFiles) {
                std::string interCommTlsCaFilePath =
                    GetServerConfig().interCommTlsCaPath + interCommTlsCaFile;
                std::string caPath = "";
                if (!FileUtils::GetRealFilePath(interCommTlsCaFilePath, caPath, errMsg)) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                        CHECK_ERROR), "Failed to get ca path because " << errMsg);

                    return false;
                }
                caPath_.push_back(static_cast<std::string>(caPath));
            }
            if (!FileUtils::GetRealFilePath(
                GetServerConfig().interCommTlsCert, certPath_, errMsg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get token path because " << errMsg);
                return false;
            }
            if (!FileUtils::GetRealFilePath(
                GetServerConfig().interCommPk, certKeyPath_, errMsg)) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                    CHECK_ERROR), "Failed to get cert key path because " << errMsg);
                return false;
            }
            if (!GetServerConfig().interCommTlsCrlPath.empty()) {
                auto interCommTlsCrlFiles = GetServerConfig().interCommTlsCrlFiles;
                for (const std::string &interCommTlsCrlFile : interCommTlsCrlFiles) {
                    std::string interCommTlsCrlFilePath =
                        GetServerConfig().interCommTlsCrlPath + interCommTlsCrlFile;
                    std::string crlPath = "";
                    if (!FileUtils::GetRealFilePath(interCommTlsCrlFilePath, crlPath, errMsg)) {
                        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                            CHECK_ERROR), "Failed to get ca path because " << errMsg);
                        return false;
                    }
                    crlPath_.push_back(static_cast<std::string>(crlPath));
                }
            }
            return true;
        }

        void GrpcCommunicationMng::FillIpAddress(std::string &ipAddress) const
        {
            auto &serverConfig = GetServerConfig();
            if (ipAddress.empty()) {
                ipAddress.append(serverConfig.ipAddress);
            }
            if (ipAddress.find(IP_PORT_DELIMITER) == std::string::npos) {
                ipAddress = FormatGrpcAddress(ipAddress, std::to_string(serverConfig.interCommPort));
            } else {
                if (ipAddress.find(':') != std::string::npos) {
                    size_t pos = ipAddress.find(IP_PORT_DELIMITER);
                    std::string ip = ipAddress.substr(0, pos);
                    std::string port = ipAddress.substr(pos + 1);
                    ipAddress = FormatGrpcAddress(ip, port);
                } else {
                    std::replace(ipAddress.begin(), ipAddress.end(), ';', ':');
                }
            }
        }
} // namespace mindie_llm
