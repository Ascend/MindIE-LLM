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
#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <cstring>
#include <thread>
#include <chrono>
#include <regex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <atomic>
#include "memory_utils.h"
#include "config_manager.h"
#include "endpoint_def.h"
#include "http_handler.h"
#include "httplib.h"
#include "https_server_helper.h"
#include "http_ssl_secret.h"
#include "log.h"
#include "random_generator.h"
#include "threadpool_monitor.h"
#include "msServiceProfiler/Tracer.h"
#include "basic_types.h"
#include "config_manager_impl.h"
#include "http_server.h"

namespace mindie_llm {
static std::shared_ptr<HttpSsl> g_businessHttpSsl = std::make_shared<HttpSsl>();
static std::shared_ptr<HttpSsl> g_managementHttpSsl = std::make_shared<HttpSsl>();
static std::shared_ptr<HttpSsl> g_metricsHttpSsl = std::make_shared<HttpSsl>();
static std::shared_ptr<HttpSslSecret> g_httpSslSecret = std::make_shared<HttpSslSecret>();
static std::map<SSLCertCategory, HttpsServerHelper*> g_mServerInstances;
static std::thread g_businessServerThread;
static std::thread g_managementServerThread;
static std::thread g_metricsServerThread;
const uint32_t MANAGEMENT_WORK_THREAD_NUM = 100;
const uint32_t QUEUE_REQUESTS_NUM_MULTI = 2;
const uint32_t PAYLOAD_MAX_LENGTH = 1024 * 1024 * 512;
const uint32_t READ_TIMEOUT = 60;
const uint32_t WRITE_TIMEOUT = 600;

static uint64_t GetCurrentTimeInNanoseconds()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return static_cast<uint64_t>(nanoseconds.count());
}

static uint64_t GetStartTimeFromResponseHeader(const std::string& timeString)
{
    try {
        return std::stoull(timeString);
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, ERROR,
            "start time from response header transform exception: " << e.what());
        return GetCurrentTimeInNanoseconds();
    }
}

void AddTracerData(const httplib::Request &request, httplib::Response &response)
{
    std::string traceParent = "";
    std::string traceB3 = "";
    if (request.has_header("traceparent")) {
        traceParent = request.get_header_value("traceparent");
    }
    if (request.has_header("b3")) {
        traceB3 = request.get_header_value("b3");
    } else if (request.has_header("X-B3-TraceId") &&
            request.has_header("X-B3-SpanId") &&
            request.has_header("X-B3-Sampled")) {
        std::string traceId = request.get_header_value("X-B3-TraceId");
        std::string spanId = request.get_header_value("X-B3-SpanId");
        std::string sampled = request.get_header_value("X-B3-Sampled");
        traceB3 = traceId + "-" + spanId + "-" + sampled;
    }
    auto attachIndex = PROF(msServiceProfiler::TraceContext::GetTraceCtx().ExtractAndAttach(traceParent, traceB3));
    auto span = PROF(std::make_unique<msServiceProfiler::Span>(
        msServiceProfiler::Tracer::StartSpanAsActive("server.Request", "LLM", false)
    ));

    PROF(span->SetAttribute("server.path", request.path.c_str()));
    PROF(span->SetAttribute("server.method", request.method.c_str()));
    PROF(span->SetAttribute("server.net.peer.ip", request.remote_addr.c_str()));
    PROF(span->SetAttribute("server.net.host.ip", request.local_addr.c_str()));
    PROF(span->SetAttribute("server.net.peer.port", std::to_string(request.remote_port).c_str()));
    PROF(span->SetAttribute("server.net.host.port", std::to_string(request.local_port).c_str()));

    if (response.has_header("RequsetUUID")) {
        PROF(span->SetAttribute("server.RequsetUUID", response.get_header_value("RequsetUUID").c_str()));
    }

    if (response.has_header("startTime")) {
        uint64_t startTime = GetStartTimeFromResponseHeader(response.get_header_value("startTime"));
        PROF(span->Activate(startTime));
        response.headers.erase("startTime");
    }

    PROF(span->SetAttribute("server.response.status", std::to_string(response.status).c_str()));
    if (response.status == httplib::StatusCode::OK_200) {
        PROF(span->SetStatus(true, ""));
    } else {
        PROF(span->SetStatus(false, response.body.c_str()));
    }

    PROF(span->End());
    PROF(msServiceProfiler::TraceContext::GetTraceCtx().Unattach(attachIndex));
    PROF(span = nullptr);
}

void HttpServerInitCallback(HttpsServerHelper &server)
{
    server.set_pre_routing_handler([&server](const httplib::Request &req, httplib::Response &res) {
        std::string requestUser = req.remote_addr + ":" + std::to_string(req.remote_port);
        std::string requestId = GenerateHTTPRequestUUID();
        res.set_header("RequestUUID", requestId);

        // 可观测插桩
        auto startTime = GetCurrentTimeInNanoseconds();
        res.set_header("startTime", std::to_string(startTime));
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_post_routing_handler([&server](const httplib::Request &req, httplib::Response &res) {
        // 可观测插桩
        AddTracerData(req, res);

        auto reqId = res.get_header_value("RequestUUID");
        // stream response will keep transferring data chunks after httplib::Response reach post_routing_handler
        if (res.get_header_value("Transfer-Encoding") != "chunked") {
            server.RemoveMonitorRequest(reqId);
        }
        std::string requestUser = req.local_addr + ":" + std::to_string(req.local_port);
        std::string statusCode = "status code: " + std::to_string(res.status);
        std::string path = req.path;
        std::string value = "/stopService";
        if (path == value && res.status == httplib::StatusCode::OK_200) {
            ULOG_AUDIT(requestUser, MINDIE_SERVER, "stop mindie server", "success");
            killpg(getpgrp(), SIGTERM);
        }
    });
    server.set_exception_handler([&server](const auto &req, auto &res, std::exception_ptr ep) {
        std::string msg = "Handle request error, request from " + req.remote_addr + ":"
            + std::to_string(req.remote_port) + ", method=" + req.method + ", uri is " + req.path;
        ULOG_WARN("endpoint", "[MIE04E020205]", msg);
        try {
            auto reqId = res.get_header_value("RequestUUID");
            server.RemoveMonitorRequest(reqId);
            std::rethrow_exception(std::move(ep));
        } catch (const std::exception &e) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                UNKNOWN_ERROR), "Internal exception: " << e.what());
        } catch (...) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                UNKNOWN_ERROR), "Internal exception: Unknown Exception");
        }
        res.set_content("Server internal error", "application/json");
        res.status = httplib::StatusCode::InternalServerError_500;

        // 可观测插桩
        AddTracerData(req, res);
    });
}

void HttpServerInitHttpLibConfig(HttpsServerHelper &server, uint32_t workThreadNum, uint32_t maxQueueRequestsNum)
{
    // 最大连接数==>server工作线程, 超出线程处理能力后在等待队列（默认队列和工作线程数相等） 中等待工作线程处理
    server.new_task_queue = [&, workThreadNum, maxQueueRequestsNum]() -> ThreadPoolMonitor* {
        // 队列长度是并发数两倍，有效防止尖峰并发
        uint32_t maxQueuedRequests = maxQueueRequestsNum;
        uint32_t numWorkThread = workThreadNum;
        // 默认线程池实现，初始化工作线程数 numWorkThread 和 可选参数(请求等待队列) maxQueuedRequests
        auto threadPool = new (std::nothrow) ThreadPoolMonitor(numWorkThread, maxQueuedRequests);
        if (threadPool == nullptr) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SPLITWISE,
                CHECK_ERROR), "Failed to create thread pool.");
            server.threadpool_ = nullptr;
            return nullptr;
        }
        server.threadpool_ = threadPool;
        return threadPool;
    };
    // 设置最大请求体长度
    server.set_payload_max_length(GetMaxInputLen());
    server.set_error_handler([](const httplib::Request &req, httplib::Response &res) {
        (void)req;
        if (res.status == httplib::StatusCode::PayloadTooLarge_413) {
            res.set_content("Request body too large. Maximum allowed size is 512MB.", "application/json");
        }
    });
    // 设置等待时间
    server.set_read_timeout(READ_TIMEOUT, 0);
    server.set_write_timeout(WRITE_TIMEOUT, 0);
    const uint64_t IDLE_TIMEOUT_SEC = 60;
    server.set_idle_interval(IDLE_TIMEOUT_SEC, 0);
}

static bool IsPortUsed(int32_t port)
{
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            ABNORMAL_TRANSMISSION_ERROR), "Socket creation failed.");
        return true;
    }

    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);

    int bindResult = bind(sock, reinterpret_cast<struct sockaddr*>(&address), sizeof(address));
    if (bindResult != 0) {
        // 端口已被占用或其它错误
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Socket bind failed, bind_result is " << bindResult << ". " << strerror(errno));
        close(sock);
        return true;
    }
    // 端口未被占用
    close(sock);
    return false;
}

uint32_t HttpServerInitOfAllPlane(SSLCertCategory sslCertCategory, HttpsServerHelper &server)
{
    const ServerConfig &serverConfig = GetServerConfig();
    if (sslCertCategory == BUSINESS_CERT) {
        g_businessServerThread = std::thread([&serverConfig, &server]() {
            try {
                bool ret = server.listen(serverConfig.ipAddress, serverConfig.port);
                if (!ret) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                        SUBMODLE_FEATURE_SINGLE_INFERENCE, INIT_ERROR),
                        "Failed to listen " << serverConfig.ipAddress << ":" << serverConfig.port <<
                        ", please check if business ip and port are legal.");
                }
            } catch (const std::exception &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                    SUBMODLE_FEATURE_SINGLE_INFERENCE, INIT_ERROR),
                    "Init business http server failed! Port is " << serverConfig.port);
            }
        });
    } else if (sslCertCategory == MANAGEMENT_CERT) {
        g_managementServerThread = std::thread([&serverConfig, &server]() {
            try {
                bool ret = server.listen(serverConfig.managementIpAddress, serverConfig.managementPort);
                if (!ret) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                        SUBMODLE_FEATURE_SINGLE_INFERENCE, INIT_ERROR),
                        "Failed to listen " << serverConfig.managementIpAddress << ":" << serverConfig.managementPort <<
                        ", please check if management ip and port are legal.");
                }
            } catch (const std::exception &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    INIT_ERROR), "Init management http server failed! Port is " << serverConfig.managementPort);
            }
        });
    } else if (sslCertCategory == METRICS_CERT) {
        g_metricsServerThread = std::thread([&serverConfig, &server]() {
            try {
                bool ret = server.listen(serverConfig.managementIpAddress, serverConfig.metricsPort);
                if (!ret) {
                    ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR,
                        SUBMODLE_FEATURE_SINGLE_INFERENCE, INIT_ERROR),
                        "Failed to listen " << serverConfig.managementIpAddress << ":" << serverConfig.metricsPort <<
                        ", please check if management ip and metrics port are legal.");
                }
            } catch (const std::exception &e) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    INIT_ERROR), "Init metrics http server failed! Port is " << serverConfig.metricsPort);
            }
        });
    } else {
        return 1U;
    }
    server.wait_until_ready();
    if (server.is_running()) {
        return 0U;
    }
    return 1U;
}

uint32_t CreateHttpServerPoint(SSLCertCategory sslCertCategory, ServerGroupType serverGroupType)
{
    const ServerConfig &serverConfig = GetServerConfig();
    std::atomic_bool isHttpsOk = {true};
    auto server = new (std::nothrow)
        HttpsServerHelper(serverConfig.httpsEnabled, [&isHttpsOk, sslCertCategory](auto &sslCtx) {
            SSL_CTX *mSslCtx = &sslCtx;
            if (sslCertCategory == BUSINESS_CERT && g_businessHttpSsl->Start(mSslCtx, sslCertCategory) != EP_OK) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    INIT_ERROR), "Business Http ssl Initialize failed");
                isHttpsOk.store(false);
                return false;
            } else if (sslCertCategory == MANAGEMENT_CERT &&
                g_managementHttpSsl->Start(mSslCtx, MANAGEMENT_CERT) != EP_OK) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    INIT_ERROR), "Management Http ssl Initialize failed");
                isHttpsOk.store(false);
                return false;
            } else if (sslCertCategory == METRICS_CERT &&
                g_metricsHttpSsl->Start(mSslCtx, METRICS_CERT) != EP_OK) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    INIT_ERROR), "Metrics Http ssl Initialize failed");
                isHttpsOk.store(false);
                return false;
            }
            return true;
        }, serverConfig.maxLinkNum);
    if (server == nullptr) {
        return 1U;
    }
    if (!isHttpsOk.load()) {
        delete server;
        return 1U;
    }
    HttpServerInitCallback(*server);
    if (serverGroupType == ALL_SAME && sslCertCategory == BUSINESS_CERT) {
        HttpHandler::BusinessInitialize(*server);
        HttpHandler::ManagementInitialize(*server);
        HttpHandler::InitializeMetricsResource(*server);
    } else if (serverGroupType == BUSINESS_MANAGEMENT_SAME && sslCertCategory == BUSINESS_CERT) {
        HttpHandler::BusinessInitialize(*server);
        HttpHandler::ManagementInitialize(*server);
    } else if (serverGroupType == BUSINESS_MANAGEMENT_SAME && sslCertCategory == METRICS_CERT) {
        HttpHandler::InitializeMetricsResource(*server);
    } else if (serverGroupType == MANAGEMENT_METRICS_SAME && sslCertCategory == BUSINESS_CERT) {
        HttpHandler::BusinessInitialize(*server);
    } else if (serverGroupType == MANAGEMENT_METRICS_SAME && sslCertCategory == MANAGEMENT_CERT) {
        HttpHandler::ManagementInitialize(*server);
        HttpHandler::InitializeMetricsResource(*server);
    } else if (serverGroupType == ALL_DIFFERENT && sslCertCategory == BUSINESS_CERT) {
        HttpHandler::BusinessInitialize(*server);
    } else if (serverGroupType == ALL_DIFFERENT && sslCertCategory == MANAGEMENT_CERT) {
        HttpHandler::ManagementInitialize(*server);
    } else if (serverGroupType == ALL_DIFFERENT && sslCertCategory == METRICS_CERT) {
        HttpHandler::InitializeMetricsResource(*server);
    } else {
        delete server;
        return 1U;
    }
    HttpServerInitHttpLibConfig(*server, serverConfig.maxLinkNum, serverConfig.maxLinkNum * QUEUE_REQUESTS_NUM_MULTI);
    g_mServerInstances[sslCertCategory] = server;
    return 0U;
}

uint32_t HttpServer::HttpServerInit()
{
    const ServerConfig &serverConfig = GetServerConfig();
    if (!CheckIp(serverConfig.managementIpAddress, "managementIpAddress", serverConfig.allowAllZeroIpListening)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Management address is invalid");
        return 1U;
    }
    if (!CheckIp(serverConfig.ipAddress, "ipAddress", serverConfig.allowAllZeroIpListening)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "ipAddress is invalid");
        return 1U;
    }

    if (IsPortUsed(serverConfig.managementPort) || IsPortUsed(serverConfig.port) ||
        IsPortUsed(serverConfig.metricsPort)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Management or Business or Metrics Http server listen port is occupied");
        return 1U;
    }
    if (serverConfig.httpsEnabled) {
        g_httpSslSecret->Start();
    }
    uint32_t res = 0U;
    // 单ip单port单httpserver实例、其他为多httpserver实例
    if (serverConfig.ipAddress == serverConfig.managementIpAddress &&
        serverConfig.port == serverConfig.managementPort &&
        serverConfig.port == serverConfig.metricsPort) {
        res = CreateHttpServerPoint(BUSINESS_CERT, ALL_SAME);
    } else if (serverConfig.ipAddress == serverConfig.managementIpAddress &&
        serverConfig.port == serverConfig.managementPort &&
        serverConfig.port != serverConfig.metricsPort) {
        res |= CreateHttpServerPoint(BUSINESS_CERT, BUSINESS_MANAGEMENT_SAME);
        res |= CreateHttpServerPoint(METRICS_CERT, BUSINESS_MANAGEMENT_SAME);
    } else if (serverConfig.port != serverConfig.managementPort &&
        serverConfig.managementPort == serverConfig.metricsPort) {
        res |= CreateHttpServerPoint(BUSINESS_CERT, MANAGEMENT_METRICS_SAME);
        res |= CreateHttpServerPoint(MANAGEMENT_CERT, MANAGEMENT_METRICS_SAME);
    } else if (serverConfig.port != serverConfig.managementPort &&
        serverConfig.managementPort != serverConfig.metricsPort &&
        serverConfig.port != serverConfig.metricsPort) {
        res |= CreateHttpServerPoint(BUSINESS_CERT, ALL_DIFFERENT);
        res |= CreateHttpServerPoint(MANAGEMENT_CERT, ALL_DIFFERENT);
        res |= CreateHttpServerPoint(METRICS_CERT, ALL_DIFFERENT);
    } else if (serverConfig.ipAddress != serverConfig.managementIpAddress &&
        serverConfig.managementPort == serverConfig.metricsPort) {
        res |= CreateHttpServerPoint(BUSINESS_CERT, BUSINESS_MANAGEMENT_SAME);
        res |= CreateHttpServerPoint(METRICS_CERT, BUSINESS_MANAGEMENT_SAME);
    } else if (serverConfig.ipAddress != serverConfig.managementIpAddress &&
        serverConfig.managementPort != serverConfig.metricsPort) {
        res |= CreateHttpServerPoint(BUSINESS_CERT, ALL_DIFFERENT);
        res |= CreateHttpServerPoint(MANAGEMENT_CERT, ALL_DIFFERENT);
        res |= CreateHttpServerPoint(METRICS_CERT, ALL_DIFFERENT);
    } else {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Management or Business or Metrics port is invalid.");
        res = 1U;
    }
    if (res != 0U) {
        HttpServerDeInit();
        return res;
    }
    // 启动server
    for (const auto &server : g_mServerInstances) {
        if (server.second != nullptr) {
            res |= HttpServerInitOfAllPlane(server.first, *server.second);
        }
    }
    if (res != 0U) {
        HttpServerDeInit();
        return res;
    }
    return res;
}

uint32_t HttpServer::HttpServerDeInit()
{
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Https server stop start.");
    g_httpSslSecret->Stop();
    for (const auto &it : g_mServerInstances) {
        it.second->stop();
    }
    if (g_businessServerThread.joinable()) {
        g_businessServerThread.join();
    }
    if (g_managementServerThread.joinable()) {
        g_managementServerThread.join();
    }
    if (g_metricsServerThread.joinable()) {
        g_metricsServerThread.join();
    }
    
    for (auto &it : g_mServerInstances) {
        delete it.second;
        it.second = nullptr;
    }
    g_mServerInstances.clear();
    ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Https server stop sucess.");
    return 0;
}
} // namespace mindie_llm
