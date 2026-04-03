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
#include <thread>
#include "ipc_communicator.h"
#include "shared_memory.h"

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

namespace mindie_llm {
class IPCCommunicatorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Mock the SharedMemory::SharedMemorySizeCheck because "/dev/shm" can be symlink for some systems
        MOCKER_CPP(&SharedMemorySizeCheck, bool (*)(const uint32_t &)).stubs().will(returnValue(true));
        MOCKER_CPP(&SharedMemory::Create, bool (*)(const std::string &, uint32_t)).stubs().will(returnValue(true));

        prefix_ = "/gtest_ipc_" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
        readerCount_ = 2;
        iPCCommunicator_ = std::make_unique<IPCCommunicator>(prefix_, readerCount_);
    }
    void TearDown() override
    {
        // If SetupChannel() was called, call CleanUp() to avoid leaving semaphores open.
        if (didSetup_) {
            iPCCommunicator_->CleanUp();
        }
        // Reset the global mock object to clear any registered mocks.
        MOCKCPP_NS::GlobalMockObject::reset();
    }

    std::shared_ptr<IPCCommunicator> iPCCommunicator_;
    std::string prefix_;
    uint32_t readerCount_;
    bool didSetup_ = false;
};

// Test SerializeExecuteMessage on a small, valid ExecuteRequest.
TEST_F(IPCCommunicatorTest, SerializeExecuteMessageReturnsTrue)
{
    std::string buf;
    ExecuteRequest request;
    ASSERT_TRUE(SerializeExecuteMessage(request, buf));

    // The returned buffer should be exactly (4 bytes for size + msgSize)
    size_t msgSize = request.ByteSizeLong();
    EXPECT_EQ(buf.size(), msgSize + sizeof(uint32_t));
}

// Test that SetupChannel() succeeds and returns true.
TEST_F(IPCCommunicatorTest, SetupChannelReturnsTrue)
{
    ShmSizeConfig shmConfig{DEFAULT_SHARED_MEMORY_SIZE, DEFAULT_SHARED_MEMORY_SIZE};
    bool ok = iPCCommunicator_->SetupChannel(shmConfig);
    EXPECT_TRUE(ok);
    didSetup_ = ok;
}

// Test RegisterResponseHandler: first time succeeds, second time fails.
TEST_F(IPCCommunicatorTest, RegisterResponseHandler_DuplicateFails)
{
    ShmSizeConfig shmConfig{DEFAULT_SHARED_MEMORY_SIZE, DEFAULT_SHARED_MEMORY_SIZE};
    ASSERT_TRUE(iPCCommunicator_->SetupChannel(shmConfig));
    didSetup_ = true;

    // First registration should succeed.
    bool first = iPCCommunicator_->RegisterResponseHandler([](ExecuteResponse &) {});
    EXPECT_TRUE(first);

    // Second registration should fail (already have a handler).
    bool second = iPCCommunicator_->RegisterResponseHandler([](ExecuteResponse &) {});
    EXPECT_FALSE(second);
}

// Test StartHandleResponseThread() without registering a handler: should return false.
TEST_F(IPCCommunicatorTest, StartHandleResponseThread_NoHandlerFails)
{
    ShmSizeConfig shmConfig{DEFAULT_SHARED_MEMORY_SIZE, DEFAULT_SHARED_MEMORY_SIZE};
    ASSERT_TRUE(iPCCommunicator_->SetupChannel(shmConfig));
    didSetup_ = true;

    // StartHandleResponseThread without a registered handler should fail
    EXPECT_FALSE(iPCCommunicator_->StartHandleResponseThread());
}

// Test StartHandleResponseThread() after registering a handler: should return true.
TEST_F(IPCCommunicatorTest, StartHandleResponseThread_AfterRegisterSucceeds)
{
    MOCKER_CPP(&IPCCommunicator::ParseResponse, bool (*)(ExecuteResponse &, char *)).stubs().will(returnValue(true));
    
    ShmSizeConfig shmConfig{DEFAULT_SHARED_MEMORY_SIZE, DEFAULT_SHARED_MEMORY_SIZE};
    ASSERT_TRUE(iPCCommunicator_->SetupChannel(shmConfig));
    didSetup_ = true;

    bool reg = iPCCommunicator_->RegisterResponseHandler([](ExecuteResponse &) {});
    ASSERT_TRUE(reg);

    bool started = iPCCommunicator_->StartHandleResponseThread();
    EXPECT_TRUE(started);

    // Let the thread spin for a short duration (it will be blocked on ReceiveSingleResponse).
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

// Test SendMessageViaSM() only for return‐value.
TEST_F(IPCCommunicatorTest, SendMessageViaSM_ReturnsTrueAfterSetup)
{
    MOCKER_CPP(&IPCCommunicator::WriteMessage, bool (*)(const char *, uint32_t)).stubs().will(returnValue(true));

    ShmSizeConfig shmConfig{DEFAULT_SHARED_MEMORY_SIZE, DEFAULT_SHARED_MEMORY_SIZE};
    ASSERT_TRUE(iPCCommunicator_->SetupChannel(shmConfig));
    didSetup_ = true;

    ExecuteRequest request;
    EXPECT_TRUE(iPCCommunicator_->SendMessageViaSM(request));
}

} // namespace mindie_llm