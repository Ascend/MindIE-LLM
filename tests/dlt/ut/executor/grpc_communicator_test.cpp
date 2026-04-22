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

#include "grpc_communicator.h"

#include <gtest/gtest.h>

#include <mockcpp/mockcpp.hpp>

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))
#define private public
namespace mindie_llm {

// Helper to build a minimal config
std::unordered_map<std::string, std::string> MakeConfig(bool isMaster, const std::string &masterIP = "0.0.0.0",
                                                        const std::string &slaveIPs = "1.1.1.1",
                                                        const std::string &port = "4242",
                                                        const std::string &localIP = "3.3.3.3") {
    return {
        {"isMaster", isMaster ? "1" : "0"}, {"masterIP", masterIP}, {"slaveIPs", slaveIPs},
        {"multiNodesInferPort", port},      {"localIP", localIP},
    };
}

class GRPCCommunicatorTest : public ::testing::Test {
   protected:
    void SetUp() override {}
    void TearDown() override {
        // Reset the global mock object to clear any registered mocks.
        MOCKCPP_NS::GlobalMockObject::reset();
    }
};

TEST_F(GRPCCommunicatorTest, GetInstance_ReturnsSameSingletonPerProcess) {
    auto config = MakeConfig(true, "0.0.0.0", "1.1.1.1,2.2.2.2", "5555", "9.9.9.9");
    auto instance1 = GRPCCommunicator::GetInstance(config);
    auto instance2 = GRPCCommunicator::GetInstance(config);
    EXPECT_EQ(instance1.get(), instance2.get());
}

TEST_F(GRPCCommunicatorTest, SendRequest_InvalidRanks_FailsEarly) {
    auto cfg = MakeConfig(true);
    GRPCCommunicator comm(cfg);

    ExecuteRequest req;
    req.set_execute_type(model_execute_data::MODEL_INFER);

    EXPECT_FALSE(comm.SendRequest(req, -1, 0));  // invalid source
    EXPECT_FALSE(comm.SendRequest(req, 0, -1));  // invalid target
}

TEST_F(GRPCCommunicatorTest, SendRequest_EmptyRegistry_BroadcastSucceeds_NoStreams) {
    auto cfg = MakeConfig(true);
    GRPCCommunicator comm(cfg);

    ExecuteRequest req;
    req.set_execute_type(model_execute_data::MODEL_INFER);

    // With no registered streams, the broadcast loop does nothing and returns true.
    EXPECT_TRUE(comm.SendRequest(req, 0, 0));
}

TEST_F(GRPCCommunicatorTest, SendRequest_BroadcastWithNullStream_FailsOnSafeWrite) {
    auto cfg = MakeConfig(true);
    GRPCCommunicator comm(cfg);

    // Insert a null stream -> SafeWriteMsgToStream returns false.
    comm.SlaveIpToStream().Insert("1.1.1.1", static_cast<SlaveStreamPtr>(nullptr));

    ExecuteRequest req;
    req.set_execute_type(model_execute_data::MODEL_INFER);
    EXPECT_FALSE(comm.SendRequest(req, 0, 0));
}

TEST_F(GRPCCommunicatorTest, SendRequest_UnicastWithNullStream_FailsOnSafeWrite) {
    auto cfg = MakeConfig(true);
    GRPCCommunicator comm(cfg);

    comm.SlaveIpToStream().Insert("2.2.2.2", static_cast<SlaveStreamPtr>(nullptr));

    ExecuteRequest req;
    req.set_execute_type(model_execute_data::MODEL_INFER);
    EXPECT_FALSE(comm.SendRequest(req, 0, 1, "2.2.2.2"));
}

TEST_F(GRPCCommunicatorTest, SendResponse_InvalidRanks_FailsEarly) {
    auto cfg = MakeConfig(false);  // slave
    GRPCCommunicator comm(cfg);

    ExecuteResponse resp;
    resp.set_msg_type(model_execute_data::MODEL_INFER);
    EXPECT_FALSE(comm.SendResponse(resp, -1, 0));
    EXPECT_FALSE(comm.SendResponse(resp, 0, -1));
}

TEST_F(GRPCCommunicatorTest, SendResponse_NullClientStream_FailsOnSafeWrite) {
    auto cfg = MakeConfig(false);  // slave
    GRPCCommunicator comm(cfg);

    // slaveStream_ is null by default -> SafeWriteMsgToStream sees null and fails.
    ExecuteResponse resp;
    resp.set_msg_type(model_execute_data::MODEL_INFER);
    EXPECT_FALSE(comm.SendResponse(resp, 0, 0));
}

TEST_F(GRPCCommunicatorTest, RegisterRequestHandler_Null_Duplicate_Success) {
    auto cfg = MakeConfig(false);  // slave side uses request handlers
    GRPCCommunicator comm(cfg);

    // Null handler
    EXPECT_FALSE(comm.RegisterRequestHandler(nullptr, /*dpRankIdx=*/1));

    // Valid once
    bool called = false;
    RequestHandler requestHandler = [&](ExecuteRequest &) { called = true; };
    EXPECT_TRUE(comm.RegisterRequestHandler(requestHandler, 1));

    // Duplicate for same rank
    EXPECT_FALSE(comm.RegisterRequestHandler(requestHandler, 1));
}

TEST_F(GRPCCommunicatorTest, RegisterResponseHandler_And_HandleResponseFromSlave) {
    auto cfg = MakeConfig(true);  // master side uses response handlers
    GRPCCommunicator comm(cfg);

    ExecuteResponse resp;
    resp.set_msg_type(model_execute_data::MODEL_INFER);

    // No handler present
    EXPECT_FALSE(comm.HandleResponseFromSlave(resp, /*targetDPRank=*/7));

    // Register then handle
    bool handled = false;
    ResponseHandler responseHandler = [&](ExecuteResponse &) { handled = true; };
    EXPECT_TRUE(comm.RegisterResponseHandler(responseHandler, 7));
    EXPECT_TRUE(comm.HandleResponseFromSlave(resp, 7));
    EXPECT_TRUE(handled);

    // Duplicate register fails
    EXPECT_FALSE(comm.RegisterResponseHandler(responseHandler, 7));
}

TEST_F(GRPCCommunicatorTest, AllSlavesConnected_ReflectsRegisteredStreams) {
    // Two slaves in config
    auto cfg = MakeConfig(true, "0.0.0.0", "1.1.1.1,2.2.2.2");
    GRPCCommunicator comm(cfg);

    // Initially false
    EXPECT_FALSE(comm.AllSlavesConnected());

    // Insert one
    comm.SlaveIpToStream().Insert("1.1.1.1", static_cast<SlaveStreamPtr>(nullptr));
    EXPECT_FALSE(comm.AllSlavesConnected());

    // Insert second
    comm.SlaveIpToStream().Insert("2.2.2.2", static_cast<SlaveStreamPtr>(nullptr));
    EXPECT_TRUE(comm.AllSlavesConnected());

    // NotifyAll is a no-op for coverage (no waiter here)
    comm.NotifyAll();
}

TEST_F(GRPCCommunicatorTest, MasterServiceImpl_Take_NoQueue_Fails) {
    auto cfg = MakeConfig(true);
    GRPCCommunicator comm(cfg);
    int respHandlerThreadCount = 0;
    MasterServiceImpl svc(&comm, respHandlerThreadCount);

    ExecuteResponse resp;
    EXPECT_FALSE(svc.Take(/*targetDPRank=*/3, resp));  // no queue present
}

TEST_F(GRPCCommunicatorTest, MasterServiceImpl_Take_WithQueue_SucceedsAndMovesValue) {
    auto cfg = MakeConfig(true);
    GRPCCommunicator comm(cfg);
    int respHandlerThreadCount = 0;
    MasterServiceImpl masterServiceImpl(&comm, respHandlerThreadCount);

    const int rank = 5;
    // Prepare queue and push one response
    auto &map = masterServiceImpl.DPRankIdxToSyncResp();
    auto blockingQueue = std::make_shared<ExecRespBlockingQueue>();
    map.Insert(rank, blockingQueue);

    ExecuteResponse inResp;
    inResp.set_msg_type(model_execute_data::REMOTE_MODEL_INIT);
    blockingQueue->push(std::make_shared<ExecuteResponse>(inResp));

    ExecuteResponse outResp;
    EXPECT_TRUE(masterServiceImpl.Take(rank, outResp));
    EXPECT_EQ(outResp.msg_type(), model_execute_data::REMOTE_MODEL_INIT);
}

TEST_F(GRPCCommunicatorTest, StopClientAndServer_NoResources_NoCrash) {
    {
        auto cfgMaster = MakeConfig(true);
        GRPCCommunicator master(cfgMaster);
        master.StopServer();  // server_ not set; masterWorkerThread_ not joinable
    }
    {
        auto cfgSlave = MakeConfig(false);
        GRPCCommunicator slave(cfgSlave);
        slave.StopClient();  // slaveStream_ null; worker thread not joinable
    }

    // Exercise destructor paths (RAII)
    { GRPCCommunicator tmpMaster(MakeConfig(true)); }
    { GRPCCommunicator tmpSlave(MakeConfig(false)); }
}
}  // namespace mindie_llm
