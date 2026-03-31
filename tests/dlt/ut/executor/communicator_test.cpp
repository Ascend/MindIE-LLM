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
 
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "communicator.h"

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

namespace mindie_llm {

class CommunicatorTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void InitCommunicator(bool isMultiNodesInfer = false, bool isMaster = false)
    {
        config_["isMaster"] = (isMaster ? "1" : "0");
        communicator_ = std::make_shared<Communicator>(config_, isMultiNodesInfer, 1, 0, false);
    }

    void TearDown() override
    {
        // Reset the global mock object to clear any registered mocks.
        MOCKCPP_NS::GlobalMockObject::reset();
    }

    std::unordered_map<std::string, std::string> config_ = {{"slaveCount", "2"},
                                                            {"masterIP", "0.0.0.0"},
                                                            {"slaveIPs", "1.1.1.1"},
                                                            {"dp", "2"},
                                                            {"multiNodesInferPort", "42"},
                                                            {"localIP", "3.3.3.3"}};
    std::shared_ptr<Communicator> communicator_;
};

TEST_F(CommunicatorTest, InitIPCCommunicators_SucceedsWhenBothChannelsSetup)
{
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::CleanUp, void (*)()).stubs();

    InitCommunicator(false, false);
    bool ok = communicator_->InitIPCCommunicators("test_shm_prefix", /*localWorldSize=*/4);
    EXPECT_TRUE(ok);
}

TEST_F(CommunicatorTest, InitIPCCommunicators_FailsWhenExecuteChannelFails)
{
    // Mock SetupChannel to return false.
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(false));

    InitCommunicator(false, false);
    bool ok = communicator_->InitIPCCommunicators("test_shm_prefix", /*localWorldSize=*/4);
    EXPECT_FALSE(ok);
}

TEST_F(CommunicatorTest, InitGRPCCommunicator_MasterNode_Succeeds)
{
    // Stub out GRPCCommunicator::Init() to return true.
    MOCKER_CPP(&GRPCCommunicator::Init, bool (*)()).stubs().will(returnValue(true));

    InitCommunicator(true, true);

    bool ok = communicator_->InitGRPCCommunicator(
        config_, [&](ExecuteResponse &r) {}, 1);
    EXPECT_TRUE(ok);
}

TEST_F(CommunicatorTest, InitGRPCCommunicator_SlaveNode_Success)
{
    MOCKER_CPP(&GRPCCommunicator::Init, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&GRPCCommunicator::RegisterRequestHandler, bool (*)(RequestHandler)).stubs().will(returnValue(true));

    InitCommunicator(true, false);

    bool ok = communicator_->InitGRPCCommunicator(
        config_, [&](ExecuteResponse &r) {}, 1);
    EXPECT_TRUE(ok);
}

TEST_F(CommunicatorTest, SendSyncRequest_Succeeds)
{
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(true));
    // Stub out SendMessageViaSM() and ReceiveResponse() to both return true.
    MOCKER_CPP(&IPCCommunicator::SendMessageViaSM, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::ReceiveInitResponses, bool (*)(std::vector<ExecuteResponse> &))
        .stubs()
        .will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::ReceiveResponse, bool (*)(ExecuteResponse &)).stubs().will(returnValue(true));

    InitCommunicator(true, false);
    bool init_ok = communicator_->InitIPCCommunicators("prefix", /*localWorldSize=*/2);
    ASSERT_TRUE(init_ok);

    ExecuteRequest request1;
    request1.set_execute_type(MODEL_INIT);
    ExecuteResponse response1;
    std::vector<ExecuteResponse> responses;
    responses.push_back(response1);

    bool modelInitSuccess = communicator_->SendModelInitRequestAndReceive(request1, responses);
    EXPECT_TRUE(modelInitSuccess);

    ExecuteRequest request2;
    request2.set_execute_type(PD_LINK);

    bool pdLinkSuccess = communicator_->SendSharedSyncRequest(request2);
    EXPECT_TRUE(pdLinkSuccess);

    ExecuteRequest request3;
    request3.set_execute_type(PD_LINK_STATUS_QUERY);
    std::vector<ExecuteResponse> responses2;

    bool pdLinkQuerySuccess = communicator_->SendSharedSyncRequestAndReceive(request3, responses2);
    EXPECT_TRUE(pdLinkQuerySuccess);
}

TEST_F(CommunicatorTest, LaunchIPCHandleResponseThreads_MasterNode_RegisterAndStartSucceed)
{
    // Stub both RegisterResponseHandler() and StartHandleResponseThread() to return true.
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::RegisterResponseHandler, bool (*)(ResponseHandler)).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::StartHandleResponseThread, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::CleanUp, void (*)()).stubs();

    InitCommunicator(true, true);
    ASSERT_TRUE(communicator_->InitIPCCommunicators("prefix", 2));

    bool ok = communicator_->LaunchIPCHandleResponseThreads([&](ExecuteResponse &r) {});
    EXPECT_TRUE(ok);

    communicator_->CleanUp();
}

TEST_F(CommunicatorTest, LaunchIPCHandleResponseThreads_SlaveNode_UsesInternalHandler)
{
    // Stub both RegisterResponseHandler() and StartHandleResponseThread() to return true.
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::RegisterResponseHandler, bool (*)(ResponseHandler)).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::StartHandleResponseThread, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::CleanUp, void (*)()).stubs();

    InitCommunicator(true, false);
    ASSERT_TRUE(communicator_->InitIPCCommunicators("prefix", 2));

    bool ok = communicator_->LaunchIPCHandleResponseThreads([&](ExecuteResponse &r) { /*should not be called*/ });
    EXPECT_TRUE(ok);

    communicator_->CleanUp();
}

TEST_F(CommunicatorTest, SendAsyncRequestToLocal_ModelInfer_Succeeds)
{
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(true));
    MOCKER_CPP(&IPCCommunicator::SendMessageViaSM, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));
    MOCKER_CPP(&GRPCCommunicator::SendRequest, bool (*)(ExecuteRequest &)).stubs().will(returnValue(true));

    InitCommunicator(true, true);
    ASSERT_TRUE(communicator_->InitIPCCommunicators("prefix", 2));

    ExecuteRequest request;
    request.set_execute_type(MODEL_INFER);
    bool ok = communicator_->SendAsyncRequest(request);
    EXPECT_TRUE(ok);
}

TEST_F(CommunicatorTest, SendAsyncRequestToLocal_UnsupportedType_Fails)
{
    MOCKER_CPP(&IPCCommunicator::SetupChannel, bool (*)()).stubs().will(returnValue(true));
    InitCommunicator(true, true);
    ASSERT_TRUE(communicator_->InitIPCCommunicators("prefix", 2));

    ExecuteRequest request;
    request.set_execute_type(MODEL_INIT); // Unsupported type for async request
    bool ok = communicator_->SendAsyncRequest(request);
    EXPECT_FALSE(ok);
}

} // namespace mindie_llm
