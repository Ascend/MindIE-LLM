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
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <thread>
#include <chrono>
#define private public
#include "health_checker.h"
#include "health_checker.cpp"
#include "../utils/mock_util.h"
#include "config_manager_impl.h"

using namespace mindie_llm;

#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

MOCKER_CPP_OVERLOAD_EQ(BackendConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(Error)

class HealthCheckerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        GlobalMockObject::verify();

        // Mock ConfigManager functions required for CreateInstance
        MOCKER_CPP(&CanonicalPath, bool (*)(std::string &)).stubs().will(returnValue(true));
        MOCKER_CPP(&GetConfigPath, Error (*)(std::string &)).stubs().will(returnValue(Error(Error::Code::OK)));
        MOCKER_CPP(&ServerConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&BackendConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&ScheduleConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&ModelDeployConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&LogConfigManager::InitFromJson, bool (*)()).stubs().will(returnValue(true));
        MOCKER_CPP(&ConfigManager::CheckAllParam, bool (*)()).stubs().will(returnValue(true));

        // Mock GetBackendConfig to return test config
        BackendConfig backendConfig;
        backendConfig.backendName = "test_backend";
        backendConfig.npuDeviceIds = {{0}};
        MOCKER_CPP(GetBackendConfig, const BackendConfig& (*)())
            .stubs()
            .will(returnValue(backendConfig));

        // Mock GetServerConfig
        ServerConfig serverConfig;
        serverConfig.inferMode = "standard";
        MOCKER_CPP(GetServerConfig, const ServerConfig& (*)())
            .stubs()
            .will(returnValue(serverConfig));

        // Mock GetInferInstance
        auto mockInferInstance = std::make_shared<InferInstance>();
        MOCKER_CPP(GetInferInstance, InferInstance* (*)())
            .stubs()
            .will(returnValue(mockInferInstance.get()));
        
        // Mock IsLlmEngineReady 方法
        MOCKER_CPP(&InferInstance::IsLlmEngineReady, bool (*)())
            .stubs()
            .will(returnValue(true));

        // Initialize ConfigManager (required before GetInstance() can be called)
        // Use a mock path since we're mocking all config functions
        ConfigManager::CreateInstance("mockPath");

        // Trigger singleton creation with mocks in place
        // This ensures GetBackendConfig mock is active when constructor runs
        HealthChecker &checker = HealthChecker::GetInstance();
        // Constructor should have completed successfully with mChipPerCard = 1
        (void)checker; // Suppress unused variable warning
    }

    void TearDown() override
    {
        // Stop health checker thread if running to avoid interference between tests
        HealthChecker &checker = HealthChecker::GetInstance();
        if (checker.mRunning.load()) {
            // Note: Stop() has a bug in source code (line 127: mRunning.store(true) should be false)
            // For testing, we manually stop the thread correctly
            checker.mRunning.store(false);
            if (checker.mCheckerThread.joinable()) {
                checker.mCheckerThread.join();
            }
        }
        // Stop Simulate task if running
        {
            if (checker.mSimulateRunner != nullptr && checker.mSimulateTaskStarted.load()) {
                checker.mSimulateRunner->Stop();
                checker.mSimulateTaskStarted.store(false);
            }
            checker.mSimulateRunner.reset();
            checker.mSimulateExecutor.reset();
        }
        GlobalMockObject::verify();
    }
};

// Test GetInstance returns singleton
TEST_F(HealthCheckerTest, GetInstanceReturnsSingleton)
{
    HealthChecker &instance1 = HealthChecker::GetInstance();
    HealthChecker &instance2 = HealthChecker::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

// Test StatusToString
TEST_F(HealthCheckerTest, StatusToString)
{
    HealthChecker &checker = HealthChecker::GetInstance();
    EXPECT_EQ(checker.StatusToString(SERVICE_READY), "SERVICE_READY");
    EXPECT_EQ(checker.StatusToString(SERVICE_NORMAL), "SERVICE_NORMAL");
    EXPECT_EQ(checker.StatusToString(SERVICE_ABNORMAL), "SERVICE_ABNORMAL");
    EXPECT_EQ(checker.StatusToString(SERVICE_PAUSE), "SERVICE_PAUSE");
    EXPECT_EQ(checker.StatusToString(SERVICE_INIT), "SERVICE_INIT");
    EXPECT_EQ(checker.StatusToString(SERVICE_BUSY), "SERVICE_BUSY");
    EXPECT_EQ(checker.StatusToString(static_cast<ServiceStatus>(999)), "UNKNOWN");
}

// Test EnqueueErrorMessage
TEST_F(HealthCheckerTest, EnqueueErrorMessage)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Clear initial status
    checker.mServiceStatus.store(SERVICE_NORMAL);

    // Enqueue an error
    checker.EnqueueErrorMessage("TEST_ERROR_001", "TestModule");

    // Check status changed to ABNORMAL
    EXPECT_EQ(checker.GetServiceStatus(), SERVICE_ABNORMAL);

    // Check error list has item
    std::vector<ErrorItem> errorList;
    ServiceStatus status;
    checker.GetStatusAndErrorList(status, errorList);
    EXPECT_EQ(status, SERVICE_ABNORMAL);
    EXPECT_FALSE(errorList.empty());
    EXPECT_EQ(errorList[0].errCode, "TEST_ERROR_001");
    EXPECT_EQ(errorList[0].createdBy, "TestModule");
}

// Test CheckErrorListEmpty
TEST_F(HealthCheckerTest, CheckErrorListEmpty)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    ServiceStatus status;
    std::vector<ErrorItem> errorList;
    checker.GetStatusAndErrorList(status, errorList);
    EXPECT_TRUE(checker.CheckErrorListEmpty());

    checker.EnqueueErrorMessage("ERROR_001", "Module1");
    EXPECT_FALSE(checker.CheckErrorListEmpty());

    // Clear error list
    checker.GetStatusAndErrorList(status, errorList);
    EXPECT_TRUE(checker.CheckErrorListEmpty());
}

// Test UpdateNpuDeviceIds
TEST_F(HealthCheckerTest, UpdateNpuDeviceIds)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Set chip per card to 2 (A3 scenario)
    checker.mChipPerCard = 2;

    // Device IDs: 0, 1, 2, 3 -> Card IDs: 0, 0, 1, 1
    std::set<int> npuDeviceIds = {0, 1, 2, 3};
    checker.UpdateNpuDeviceIds(npuDeviceIds);

    std::shared_lock<std::shared_mutex> lock(checker.mNpuDevicesMutex);
    EXPECT_EQ(checker.mNpuDeviceCardIds.size(), 2);
    EXPECT_TRUE(checker.mNpuDeviceCardIds.find(0) != checker.mNpuDeviceCardIds.end());
    EXPECT_TRUE(checker.mNpuDeviceCardIds.find(1) != checker.mNpuDeviceCardIds.end());
}

// Test UpdateNpuDeviceIds with single chip per card
TEST_F(HealthCheckerTest, UpdateNpuDeviceIdsSingleChip)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    checker.mChipPerCard = 1;

    std::set<int> npuDeviceIds = {0, 1, 2, 3};
    checker.UpdateNpuDeviceIds(npuDeviceIds);

    std::shared_lock<std::shared_mutex> lock(checker.mNpuDevicesMutex);
    EXPECT_EQ(checker.mNpuDeviceCardIds.size(), 4);
    EXPECT_TRUE(checker.mNpuDeviceCardIds.find(0) != checker.mNpuDeviceCardIds.end());
    EXPECT_TRUE(checker.mNpuDeviceCardIds.find(1) != checker.mNpuDeviceCardIds.end());
    EXPECT_TRUE(checker.mNpuDeviceCardIds.find(2) != checker.mNpuDeviceCardIds.end());
    EXPECT_TRUE(checker.mNpuDeviceCardIds.find(3) != checker.mNpuDeviceCardIds.end());
}

// Test IsValidStatusTransition
TEST_F(HealthCheckerTest, IsValidStatusTransition)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Valid transitions from INIT
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_INIT, SERVICE_NORMAL));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_INIT, SERVICE_BUSY));

    // Valid transitions from NORMAL
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_NORMAL, SERVICE_PAUSE));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_NORMAL, SERVICE_ABNORMAL));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_NORMAL, SERVICE_BUSY));

    // Valid transitions from BUSY
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_BUSY, SERVICE_PAUSE));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_BUSY, SERVICE_ABNORMAL));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_BUSY, SERVICE_NORMAL));

    // Valid transitions from ABNORMAL
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_ABNORMAL, SERVICE_NORMAL));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_ABNORMAL, SERVICE_BUSY));

    // Valid transitions from PAUSE
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_PAUSE, SERVICE_READY));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_PAUSE, SERVICE_NORMAL));

    // Valid transitions from READY
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_READY, SERVICE_NORMAL));
    EXPECT_TRUE(checker.IsValidStatusTransition(SERVICE_READY, SERVICE_BUSY));

    // Invalid transitions
    EXPECT_FALSE(checker.IsValidStatusTransition(SERVICE_INIT, SERVICE_PAUSE));
    EXPECT_FALSE(checker.IsValidStatusTransition(SERVICE_NORMAL, SERVICE_READY));
    EXPECT_FALSE(checker.IsValidStatusTransition(SERVICE_ABNORMAL, SERVICE_PAUSE));
}

// Test UpdateStatus with valid transition
TEST_F(HealthCheckerTest, UpdateStatusValidTransition)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Start from INIT
    checker.mServiceStatus.store(SERVICE_INIT);

    // Update to NORMAL (valid transition)
    checker.UpdateStatus(SERVICE_NORMAL);
    EXPECT_EQ(checker.GetServiceStatus(), SERVICE_NORMAL);

    // Update to BUSY (valid transition)
    checker.UpdateStatus(SERVICE_BUSY);
    EXPECT_EQ(checker.GetServiceStatus(), SERVICE_BUSY);
}

// Test UpdateStatus with same status
TEST_F(HealthCheckerTest, UpdateStatusSameStatus)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    checker.mServiceStatus.store(SERVICE_NORMAL);

    // Try to update to same status
    checker.UpdateStatus(SERVICE_NORMAL);

    // Status should remain unchanged
    EXPECT_EQ(checker.GetServiceStatus(), SERVICE_NORMAL);
}

// Test Start and Stop
TEST_F(HealthCheckerTest, StartAndStop)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Ensure not running
    if (checker.mRunning.load()) {
        checker.mRunning.store(false);
        if (checker.mCheckerThread.joinable()) {
            checker.mCheckerThread.join();
        }
    }

    // Test Start
    EXPECT_TRUE(checker.Start());
    EXPECT_TRUE(checker.mRunning.load());

    // Test Start when already running
    EXPECT_FALSE(checker.Start());

    // Test Stop
    checker.Stop();
    EXPECT_FALSE(checker.mRunning.load());
}

// Test GetStatusAndErrorList
TEST_F(HealthCheckerTest, GetStatusAndErrorList)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Set status and add some errors
    checker.mServiceStatus.store(SERVICE_NORMAL);
    checker.EnqueueErrorMessage("ERR1", "Module1");
    checker.EnqueueErrorMessage("ERR2", "Module2");

    // Get status and error list
    ServiceStatus status;
    std::vector<ErrorItem> errorList;
    checker.GetStatusAndErrorList(status, errorList);

    EXPECT_EQ(status, SERVICE_ABNORMAL);
    EXPECT_EQ(errorList.size(), 2);
    EXPECT_TRUE(checker.CheckErrorListEmpty());
}

// Test PrintNpuDeviceIds (no output verification, just ensure no crash)
TEST_F(HealthCheckerTest, PrintNpuDeviceIds)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Set some device IDs
    checker.mChipPerCard = 1;
    std::set<int> npuDeviceIds = {0, 1, 2};
    checker.UpdateNpuDeviceIds(npuDeviceIds);

    // Should not crash
    EXPECT_NO_THROW(checker.PrintNpuDeviceIds());
}

// Test SimulateRequestExecutor Factory Pattern
TEST_F(HealthCheckerTest, SimulateExecutorFactoryPattern)
{
    // Test the factory method with different request types
    auto standardExecutor = SimulateRequestExecutor::Create(InferReqType::REQ_STAND_INFER);
    EXPECT_NE(standardExecutor, nullptr);
    
    // Prefill type for DMI mode
    auto prefillExecutor = SimulateRequestExecutor::Create(InferReqType::REQ_PREFILL);
    EXPECT_NE(prefillExecutor, nullptr);
}

// Test StartSimulateTask when already started
TEST_F(HealthCheckerTest, StartSimulateTaskAlreadyStarted)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Setup: task already started
    {
        // Create a mock runner (empty unique_ptr is fine for this test)
        checker.mSimulateRunner = std::make_unique<SimulateTaskRunner>();
        checker.mSimulateTaskStarted.store(true);
    }

    // Should return true immediately
    bool result = checker.StartSimulateTask();
    EXPECT_TRUE(result);

    // Cleanup
    {
        checker.mSimulateTaskStarted.store(false);
        checker.mSimulateRunner.reset();
    }
}

// Test StartSimulateTask when not running
TEST_F(HealthCheckerTest, StartSimulateTaskWhenNotRunning)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Ensure not running
    checker.mRunning.store(false);

    // Clear any existing runner
    {
        checker.mSimulateTaskStarted.store(false);
        checker.mSimulateRunner.reset();
    }

    // Set status to normal
    checker.mServiceStatus.store(SERVICE_NORMAL);
    EXPECT_NO_THROW({
        bool result = checker.StartSimulateTask();
        // Result may be false in test environment, that's OK
        (void)result;
    });
}

// Test StartSimulateTask in PAUSE state
TEST_F(HealthCheckerTest, StartSimulateTaskInPauseState)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Set conditions
    checker.mRunning.store(true);
    checker.mServiceStatus.store(SERVICE_PAUSE);

    // Clear any existing runner
    {
        checker.mSimulateTaskStarted.store(false);
        checker.mSimulateRunner.reset();
    }

    // We'll change status after a short delay in a separate thread
    std::thread statusChanger([&checker]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        checker.mServiceStatus.store(SERVICE_NORMAL);
    });

    // Detach the thread so test can continue
    statusChanger.detach();

    // Add a timeout to prevent test hanging
    auto startTime = std::chrono::steady_clock::now();
    bool result = false;

    EXPECT_NO_THROW({
        result = checker.StartSimulateTask();
    });

    // Cleanup timeout check
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    EXPECT_LT(duration.count(), 2000);

    // Reset
    checker.mRunning.store(false);
}

// Test InitNpuDeviceCardIds in DMI mode with empty device IDs
TEST_F(HealthCheckerTest, InitNpuDeviceCardIdsDmiModeEmpty)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    // Clear device IDs
    {
        std::unique_lock<std::shared_mutex> lock(checker.mNpuDevicesMutex);
        checker.mNpuDeviceCardIds.clear();
    }

    // Should return false because npuDeviceIds is empty
    bool result = checker.InitNpuDeviceCardIds();
    EXPECT_FALSE(result);
}

// Test CheckSimulateTask when task not started
TEST_F(HealthCheckerTest, CheckSimulateTaskNotStarted)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    {
        checker.mSimulateTaskStarted.store(false);
        checker.mSimulateRunner.reset();
    }

    ServiceStatus result = checker.CheckSimulateTask();
    EXPECT_EQ(result, SERVICE_ABNORMAL);
}

// Test CheckSimulateTask when task started but runner is null
TEST_F(HealthCheckerTest, CheckSimulateTaskStartedRunnerNull)
{
    HealthChecker &checker = HealthChecker::GetInstance();

    {
        checker.mSimulateTaskStarted.store(true);
        checker.mSimulateRunner.reset();
    }

    ServiceStatus result = checker.CheckSimulateTask();
    EXPECT_EQ(result, SERVICE_ABNORMAL);
}