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
#include <acl/acl.h>
#include <vector>
#include <stdexcept>
#include <atb/atb_infer.h>

#include "atb_speed/base/event_manager.h"

namespace atb_speed {
// 单例测试，验证返回的实例地址是否相同
TEST(EventManager, SingletonTest)
{
    EventManager& instance1 = EventManager::GetInstance();
    EventManager& instance2 = EventManager::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

// 单线程，测试 RecordEvent 接口（使用 PUSH 动作），返回一个 op
TEST(EventManager, RecordEventPushTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe1";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, ACL_SUCCESS);

    atb::Operation* op = nullptr;

    auto status = instance.RecordEvent(op, EventAction::PUSH, pipeKey);
    EXPECT_EQ(status, atb::NO_ERROR);
    EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 1);
    EXPECT_NE(op, nullptr);
}

// 单线程，测试 RecordEvent 接口（使用 PUSH 和 POP 动作），返回两个 op
TEST(EventManager, RecordEventPushAndPopTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe2";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, ACL_SUCCESS);

    atb::Operation* op1 = nullptr;
    atb::Operation* op2 = nullptr;

    auto statusPush = instance.RecordEvent(op1, EventAction::PUSH, pipeKey);
    EXPECT_EQ(statusPush, atb::NO_ERROR);

    auto statusPop = instance.RecordEvent(op2, EventAction::POP, pipeKey);
    EXPECT_EQ(statusPop, atb::NO_ERROR);
    EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 2);
    EXPECT_NE(op1, nullptr);
    EXPECT_NE(op2, nullptr);
}

// 单线程，测试 WaitEvent 接口（使用 PUSH 动作），返回一个 op
TEST(EventManager, WaitEventPushTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe3";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, ACL_SUCCESS);

    atb::Operation* op = nullptr;

    auto status = instance.WaitEvent(op, EventAction::PUSH, pipeKey);
    EXPECT_EQ(status, atb::NO_ERROR);
    EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 1);
    EXPECT_NE(op, nullptr);
}

// 单线程，测试 WaitEvent 接口（使用 PUSH 和 POP 动作），返回两个 op
TEST(EventManager, WaitEventPushAndPopTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe4";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, ACL_SUCCESS);

    atb::Operation* op1 = nullptr;
    atb::Operation* op2 = nullptr;

    auto statusPush = instance.WaitEvent(op1, EventAction::PUSH, pipeKey);
    EXPECT_EQ(statusPush, atb::NO_ERROR);

    auto statusPop = instance.WaitEvent(op2, EventAction::POP, pipeKey);
    EXPECT_EQ(statusPop, atb::NO_ERROR);
    EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 2);
    EXPECT_NE(op1, nullptr);
    EXPECT_NE(op2, nullptr);
}

// 单线程，测试 RecordEvent 和 WaitEvent 的 PUSH 和 POP 组合，返回两个 op
TEST(EventManager, RecordEventPushAndWaitEventPopTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe5";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, ACL_SUCCESS);

    atb::Operation* op1 = nullptr;
    atb::Operation* op2 = nullptr;

    auto statusPush = instance.RecordEvent(op1, EventAction::PUSH, pipeKey);
    EXPECT_EQ(statusPush, atb::NO_ERROR);

    auto statusPop = instance.WaitEvent(op2, EventAction::POP, pipeKey);
    EXPECT_EQ(statusPop, atb::NO_ERROR);
    EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 2);
    EXPECT_NE(op1, nullptr);
    EXPECT_NE(op2, nullptr);
}

// 单线程，测试 WaitEvent 和 RecordEvent 的 PUSH 和 POP 组合，返回两个 op
TEST(EventManager, WaitEventPushAndRecordEventPopTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe6";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    ASSERT_EQ(ret, ACL_SUCCESS);

    atb::Operation* op1 = nullptr;
    atb::Operation* op2 = nullptr;

    auto statusPush = instance.WaitEvent(op1, EventAction::PUSH, pipeKey);
    EXPECT_EQ(statusPush, atb::NO_ERROR);

    auto statusPop = instance.RecordEvent(op2, EventAction::POP, pipeKey);
    EXPECT_EQ(statusPop, atb::NO_ERROR);
    EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 2);
    EXPECT_NE(op1, nullptr);
    EXPECT_NE(op2, nullptr);
}

// 双线程，测试 RecordEvent 和 WaitEvent 的 PUSH 和 POP 组合，返回两个 op
TEST(EventManager, TwoThreadsRecordEventPushAndWaitEventPopTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe7";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    std::atomic<bool> isRecordEventDone(false);
    std::atomic<bool> isWaitEventDone(false);

    int32_t deviceId1 = 0;
    int32_t deviceId2 = 0;
    aclrtStream stream1;
    aclrtStream stream2;

    std::thread recordThread([&]() {
        aclError ret1 = aclrtSetDevice(deviceId1);
        ASSERT_EQ(ret1, ACL_SUCCESS);

        aclError ret2 = aclrtCreateStream(&stream1);
        ASSERT_EQ(ret2, ACL_SUCCESS);

        atb::Operation* op1 = nullptr;

        auto statusPush = instance.RecordEvent(op1, EventAction::PUSH, pipeKey);
        EXPECT_EQ(statusPush, atb::NO_ERROR);
        EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 1);
        EXPECT_NE(op1, nullptr);
        isRecordEventDone.store(true);

        aclError ret3 = aclrtDestroyStream(stream1);
        ASSERT_EQ(ret3, ACL_SUCCESS);
    });

    std::thread waitThread([&]() {
        while (!isRecordEventDone.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        aclError ret4 = aclrtSetDevice(deviceId2);
        ASSERT_EQ(ret4, ACL_SUCCESS);

        aclError ret5 = aclrtCreateStream(&stream2);
        ASSERT_EQ(ret5, ACL_SUCCESS);

        atb::Operation* op2 = nullptr;

        auto statusPop = instance.WaitEvent(op2, EventAction::POP, pipeKey);
        EXPECT_EQ(statusPop, atb::NO_ERROR);
        EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 1);
        EXPECT_NE(op2, nullptr);
        isWaitEventDone.store(true);
        
        aclError ret7 = aclrtSynchronizeDevice();
        ASSERT_EQ(ret7, ACL_SUCCESS);

        aclError ret6 = aclrtDestroyStream(stream2);
        ASSERT_EQ(ret6, ACL_SUCCESS);
    });

    recordThread.join();
    waitThread.join();

    EXPECT_TRUE(isRecordEventDone.load());
    EXPECT_TRUE(isWaitEventDone.load());

    aclError ret8 = aclrtResetDevice(deviceId1);
    ASSERT_EQ(ret8, ACL_SUCCESS);

    aclError ret9 = aclrtResetDevice(deviceId2);
    ASSERT_EQ(ret9, ACL_SUCCESS);
}

// 双线程，测试 WaitEvent 和 RecordEvent 的 PUSH 和 POP 组合，返回两个 op
TEST(EventManager, TwoThreadsWaitEventPushAndRecordEventPopTest)
{
    g_eventOperationsOfModel.clear();
    std::string pipeKey = "pipe8";
    size_t eventOpInitialSize = g_eventOperationsOfModel.size();
    EventManager& instance = EventManager::GetInstance();

    std::atomic<bool> isRecordEventDone(false);
    std::atomic<bool> isWaitEventDone(false);

    int32_t deviceId1 = 0;
    int32_t deviceId2 = 0;
    aclrtStream stream1;
    aclrtStream stream2;

    std::thread waitThread([&]() {
        aclError ret1 = aclrtSetDevice(deviceId1);
        ASSERT_EQ(ret1, ACL_SUCCESS);

        aclError ret2 = aclrtCreateStream(&stream1);
        ASSERT_EQ(ret2, ACL_SUCCESS);

        atb::Operation* op1 = nullptr;

        auto statusPush = instance.WaitEvent(op1, EventAction::PUSH, pipeKey);
        EXPECT_EQ(statusPush, atb::NO_ERROR);
        EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 1);
        EXPECT_NE(op1, nullptr);
        isWaitEventDone.store(true);

        aclError ret7 = aclrtSynchronizeDevice();
        ASSERT_EQ(ret7, ACL_SUCCESS);

        aclError ret3 = aclrtDestroyStream(stream1);
        ASSERT_EQ(ret3, ACL_SUCCESS);
    });

    std::thread recordThread([&]() {
        while (!isWaitEventDone.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        aclError ret4 = aclrtSetDevice(deviceId2);
        ASSERT_EQ(ret4, ACL_SUCCESS);

        aclError ret5 = aclrtCreateStream(&stream2);
        ASSERT_EQ(ret5, ACL_SUCCESS);

        atb::Operation* op2 = nullptr;

        auto statusPop = instance.RecordEvent(op2, EventAction::POP, pipeKey);
        EXPECT_EQ(statusPop, atb::NO_ERROR);
        EXPECT_EQ(g_eventOperationsOfModel.size(), eventOpInitialSize + 1);
        EXPECT_NE(op2, nullptr);
        isRecordEventDone.store(true);

        aclError ret6 = aclrtDestroyStream(stream2);
        ASSERT_EQ(ret6, ACL_SUCCESS);
    });

    waitThread.join();
    recordThread.join();

    EXPECT_TRUE(isRecordEventDone.load());
    EXPECT_TRUE(isWaitEventDone.load());

    aclError ret8 = aclrtResetDevice(deviceId1);
    ASSERT_EQ(ret8, ACL_SUCCESS);

    aclError ret9 = aclrtResetDevice(deviceId2);
    ASSERT_EQ(ret9, ACL_SUCCESS);
}

// 测试 SetWaitOperationTimeout 接口
TEST(EventManager, SetWaitOperationTimeoutTest)
{
    EventManager& instance = EventManager::GetInstance();
    EXPECT_NO_THROW(instance.SetWaitOperationTimeout(60));
    EXPECT_NO_THROW(instance.SetWaitOperationTimeout(300));
    EXPECT_NO_THROW(instance.SetWaitOperationTimeout(1800));
}

// 测试传入非法 EventAction 时，RecordEvent 返回错误码 EventManagerStatus::EM_INVALID_ACTION
TEST(EventManager, RecordEventInvalidActionTest)
{
    std::string pipeKey = "pipe10";
    EventManager& instance = EventManager::GetInstance();
    atb::Operation* op = nullptr;

    auto status = instance.RecordEvent(op, static_cast<EventAction>(999), pipeKey);
    EXPECT_EQ(status, EventManagerStatus::EM_INVALID_ACTION);
    EXPECT_EQ(op, nullptr);
}

// 测试CheckPipeKey函数
TEST(EventManager, CheckPipeKeyTest)
{
    EventManager& manager = EventManager::GetInstance();
    const std::string pipeKey = "test_pipe_1";
    aclrtStream subStream;
    std::vector<aclrtEvent> queue;
    manager.eventsForExternal_[pipeKey] = std::make_tuple(0, queue, subStream);
    EXPECT_EQ(manager.CheckPipeKey(pipeKey), EM_SUCCESS);
    aclrtEvent event = nullptr;
    manager.CreateAndPushEvent(event, pipeKey);
    EXPECT_EQ(manager.CheckPipeKey(pipeKey), EM_SUCCESS);
    EXPECT_TRUE(manager.eventsForExternal_.count(pipeKey) > 0);
    
    // 验证数据结构是否正确初始化
    auto& tuple = manager.eventsForExternal_[pipeKey];
    const std::string pipeKey2 = "test_pipe_2";
    EXPECT_EQ(manager.CheckPipeKey(pipeKey2), EM_INVALID_ACTION);
}

// 测试RecordEvent函数
TEST(EventManager, RecordEventTest) {
    EventManager& manager = EventManager::GetInstance();
    const std::string pipeKey = "record_normal";
    aclrtEvent event = nullptr;
    manager.CreateAndPushEvent(event, pipeKey);
    EXPECT_EQ(manager.RecordEvent(pipeKey), EM_SUCCESS);
}

// 测试RecordEvent函数
TEST(EventManager, WaitEventTest) {
    EventManager& manager = EventManager::GetInstance();
    const std::string pipeKey = "wait_normal";
    aclrtEvent event = nullptr;
    manager.CreateAndPushEvent(event, pipeKey);
    EXPECT_EQ(manager.RecordEvent(pipeKey), EM_SUCCESS);
    EXPECT_EQ(manager.WaitEvent(pipeKey), EM_SUCCESS);
}

} // namespace atb_speed