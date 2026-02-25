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

#ifndef GRPC_COMMUNICATOR_H
#define GRPC_COMMUNICATOR_H

#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <openssl/pem.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <boost/thread/sync_queue.hpp>
#include "executor/executor_interface.h"
#include "model_execute_data.pb.h"
#include "model_execute_data.grpc.pb.h"
#include "concurrent_map.h"
#include "log.h"
#include "common_util.h"
#include "string_utils.h"

namespace mindie_llm {

using grpc::ServerReaderWriter;
using grpc::ServerContext;
using model_execute_data::MasterToSlaveMsg;
using model_execute_data::SlaveToMasterMsg;
using model_execute_data::MasterService;
using SlaveStreamPtr = ServerReaderWriter<MasterToSlaveMsg, SlaveToMasterMsg> *;
using ExecRespBlockingQueue = boost::sync_queue<std::shared_ptr<ExecuteResponse>>;

class GRPCCommunicator {
public:
    static std::shared_ptr<GRPCCommunicator>
    GetInstance(const std::unordered_map<std::string, std::string> &modelConfig);

    // 删除拷贝构造和赋值操作
    GRPCCommunicator(const GRPCCommunicator &) = delete;
    GRPCCommunicator &operator=(const GRPCCommunicator &) = delete;

    explicit GRPCCommunicator(const std::unordered_map<std::string, std::string> &modelConfig);

    ~GRPCCommunicator();

    bool Init(int initCount);

    bool SendRequest(ExecuteRequest &request, int sourceDPRank, int targetDPRank,
                     const std::string &slaveIp = ""); // Only for master node

    // Blocks until a synchronous response is received (master node only)
    bool GetSyncResponse(ExecuteResponse &response, int sourceDPRank);

    bool SendResponse(ExecuteResponse &response, int sourceDPRank, int targetDPRank); // Only for slave node

    bool RegisterRequestHandler(RequestHandler handler, int dpRankIdx); // Only for slave node

    bool RegisterResponseHandler(ResponseHandler handler, int dpRankIdx); // Only for master node

    void HandleRequestFromMaster(ExecuteRequest &request, int targetDPRank); // Only for slave node

    bool HandleResponseFromSlave(ExecuteResponse &response, int targetDPRank); // Only for master node

    bool AllSlavesConnected();

    void NotifyAll();

    void StopServer();

    void StopClient();

    ConcurrentMap<std::string, SlaveStreamPtr> &SlaveIpToStream();

private:
    static constexpr int grpcSendReceiveBufSize = 256 * 1024 * 1024; // 256MB, 和共享内存大小对齐
    static constexpr int maxConcurrentStreams = 128;                   // 最大并发流数

    bool InitMaster(int respHandlerThreadCount);

    bool InitSlave();

    void WaitForAllSlavesConnected();

    void StartWorkerThread();

    bool SendRegistration();

    bool LoadCertificates();

    template <typename StreamType, typename MsgType> bool SafeWriteMsgToStream(StreamType stream, const MsgType &msg);

    bool interNodeTLSEnabled_;
    std::string interNodeTlsCaPath_;
    std::vector<std::string> interNodeTlsCaFiles_;
    std::string interNodeTlsCert_;
    std::string interNodeTlsPk_;
    std::string interNodeTlsCrlPath_;
    std::vector<std::string> interNodeTlsCrlFiles_;

    // 缓存读取后的证书内容
    std::string caCert_;
    std::string tlsCert_;
    std::string tlsCertPrivateKey_;

    bool isMaster_ = false;
    std::string masterIP_;
    // communication port ex: "1120"
    std::string multiNodesInferPort_;

    // The Init() function is called muiltiple times, and only the last call will initialize the GRPC communicator.
    std::atomic<int> callInitCount_{0};
    // Concurrency for SendRequest() and SendResponse()
    std::mutex streamWriteMutex_;

    // Valid only for master node
    ConcurrentMap<std::string, SlaveStreamPtr> slaveIpToStream_;
    uint32_t slaveCount_{0};
    std::condition_variable cv_;
    std::thread masterWorkerThread_;
    ConcurrentMap<int, ResponseHandler> responseHandlers_;
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<MasterService::Service> service_;

    // Valid only for slave node
    std::string slaveIp_;
    std::unique_ptr<
        ::grpc::ClientReaderWriter<::model_execute_data::SlaveToMasterMsg, ::model_execute_data::MasterToSlaveMsg>>
        slaveStream_;
    std::unique_ptr<MasterService::Stub> stub_;
    std::shared_ptr<grpc::Channel> channel_;
    std::thread slaveWorkerThread_;
    std::unique_ptr<grpc::ClientContext> context_;
    ConcurrentMap<int, RequestHandler> requestHandlers_;
};

class MasterServiceImpl final : public MasterService::Service {
public:
    explicit MasterServiceImpl(GRPCCommunicator *comm, int respHandlerThreadCount);
    ~MasterServiceImpl() override;
    grpc::Status RegisterAndCommunicate(ServerContext *context, SlaveStreamPtr stream) override;
    bool Take(int targetDPRank, ExecuteResponse &response); // Get response from the blocking queue, master node only
    ConcurrentMap<int, std::shared_ptr<ExecRespBlockingQueue>> &DPRankIdxToSyncResp();

private:
    GRPCCommunicator *gRPCCommunicator_;
    ConcurrentMap<int, std::shared_ptr<ExecRespBlockingQueue>> dpRankIdxToSyncResp_; // Only for master node

    struct SlaveResponseTask {
        int targetDPRank;
        std::shared_ptr<ExecuteResponse> response;
    };
    // Blocking queue for async response handling
    boost::sync_queue<std::shared_ptr<SlaveResponseTask>> pendingRespFromSlaveQueue_;
    std::vector<std::thread> respHandlerThreads_;
    std::atomic<bool> respHandlerThreadActive_{true};
    void RespHandlerLoop();
    void StopRespHandlerThreads();
};

} // namespace mindie_llm
#endif