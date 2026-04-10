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
#include <vector>

#define private public
#include "simulate_task_runner.h"

using namespace mindie_llm;

/// @brief Mock 执行器，用于测试
class MockSimulateExecutor : public ISimulateExecutor {
public:
    MockSimulateExecutor() = default;
    ~MockSimulateExecutor() override = default;

    SimulateResult RunSimulateOnce() override
    {
        runCount_++;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs_));
        return result_;
    }

    void SetResult(SimulateResult::Status status, const std::string& message)
    {
        result_.status = status;
        result_.message = message;
    }

    void SetSleepMs(uint32_t ms) { sleepMs_ = ms; }
    uint32_t GetRunCount() const { return runCount_; }

private:
    SimulateResult result_{SimulateResult::Status::SUCCESS, "mock success"};
    uint32_t sleepMs_{10};
    std::atomic<uint32_t> runCount_{0};
};

class SimulateTaskRunnerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        GlobalMockObject::verify();
        executor_ = std::make_shared<MockSimulateExecutor>();
        // 与 HealthChecker::UpdateNpuDeviceIds 一致：逻辑设备 0、1，chipPerCard=1 -> (卡, chip)
        npuDeviceCardIds_ = {{0, 0}, {1, 0}};
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
    }

    std::shared_ptr<MockSimulateExecutor> executor_;
    std::vector<std::pair<int, int>> npuDeviceCardIds_;
};

// ==================== SimulateResult 测试 ====================

TEST_F(SimulateTaskRunnerTest, SimulateResultStatus)
{
    SimulateResult result;
    result.status = SimulateResult::Status::SUCCESS;
    EXPECT_EQ(result.status, SimulateResult::Status::SUCCESS);

    result.status = SimulateResult::Status::BUSY;
    EXPECT_EQ(result.status, SimulateResult::Status::BUSY);

    result.status = SimulateResult::Status::TIMEOUT;
    EXPECT_EQ(result.status, SimulateResult::Status::TIMEOUT);

    result.status = SimulateResult::Status::ERROR;
    EXPECT_EQ(result.status, SimulateResult::Status::ERROR);
}

// ==================== SimulateHealthStatus 测试 ====================

TEST_F(SimulateTaskRunnerTest, SimulateHealthStatusDefaults)
{
    SimulateHealthStatus status;
    EXPECT_EQ(status.lastStatus, SimulateResult::Status::SUCCESS);
    EXPECT_EQ(status.lastMessage, "not started");
    EXPECT_EQ(status.successCount, 0);
    EXPECT_EQ(status.failureCount, 0);
    EXPECT_FALSE(status.isRunning);
}

TEST_F(SimulateTaskRunnerTest, SimulateHealthStatusIsHealthy)
{
    SimulateHealthStatus status;
    status.lastStatus = SimulateResult::Status::SUCCESS;
    EXPECT_TRUE(status.IsHealthy());

    status.lastStatus = SimulateResult::Status::BUSY;
    EXPECT_FALSE(status.IsHealthy());

    status.lastStatus = SimulateResult::Status::TIMEOUT;
    EXPECT_FALSE(status.IsHealthy());

    status.lastStatus = SimulateResult::Status::ERROR;
    EXPECT_FALSE(status.IsHealthy());
}

TEST_F(SimulateTaskRunnerTest, SimulateHealthStatusGetSecondsSinceLastUpdate)
{
    SimulateHealthStatus status;
    status.lastUpdateTime = std::chrono::steady_clock::now();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    uint64_t seconds = status.GetSecondsSinceLastUpdate();
    // 100ms 应该小于 1 秒
    EXPECT_EQ(seconds, 0);
}

// ==================== SimulateTaskRunner 构造和初始化测试 ====================

TEST_F(SimulateTaskRunnerTest, ConstructorCreatesInvalidInstance)
{
    SimulateTaskRunner runner;
    EXPECT_FALSE(runner.IsValid());
    EXPECT_FALSE(runner.IsRunning());
    EXPECT_FALSE(runner.IsPaused());
}

TEST_F(SimulateTaskRunnerTest, InitWithValidParams)
{
    SimulateTaskRunner runner;
    bool result = runner.Init(executor_, npuDeviceCardIds_, 10);
    
    EXPECT_TRUE(result);
    EXPECT_TRUE(runner.IsValid());
}

TEST_F(SimulateTaskRunnerTest, InitWithNullExecutor)
{
    SimulateTaskRunner runner;
    bool result = runner.Init(nullptr, npuDeviceCardIds_, 10);
    
    EXPECT_FALSE(result);
    EXPECT_FALSE(runner.IsValid());
}

TEST_F(SimulateTaskRunnerTest, InitWithEmptyNpuIds)
{
    SimulateTaskRunner runner;
    std::vector<std::pair<int, int>> emptyIds;
    bool result = runner.Init(executor_, emptyIds, 10);
    
    EXPECT_FALSE(result);
    EXPECT_FALSE(runner.IsValid());
}

TEST_F(SimulateTaskRunnerTest, InitTwiceReturnsTrue)
{
    SimulateTaskRunner runner;
    EXPECT_TRUE(runner.Init(executor_, npuDeviceCardIds_, 10));
    
    // 再次初始化应该返回 true（已初始化）
    EXPECT_TRUE(runner.Init(executor_, npuDeviceCardIds_, 10));
    EXPECT_TRUE(runner.IsValid());
}

// ==================== Start/Stop 测试 ====================

TEST_F(SimulateTaskRunnerTest, StartWithoutInitFails)
{
    SimulateTaskRunner runner;
    runner.Start(1);
    
    // 未初始化，Start 应该失败
    EXPECT_FALSE(runner.IsRunning());
}

TEST_F(SimulateTaskRunnerTest, StartAndStopBasic)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    runner.Start(1);
    EXPECT_TRUE(runner.IsRunning());
    
    // 等待一小段时间确保线程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    runner.Stop();
    EXPECT_FALSE(runner.IsRunning());
}

TEST_F(SimulateTaskRunnerTest, StartTwiceIgnoresSecondCall)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    runner.Start(1);
    EXPECT_TRUE(runner.IsRunning());
    
    // 再次 Start 应该被忽略
    runner.Start(2);
    EXPECT_TRUE(runner.IsRunning());
    
    runner.Stop();
}

TEST_F(SimulateTaskRunnerTest, StopWhenNotRunningDoesNothing)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    // 未启动时 Stop 不应该有问题
    runner.Stop();
    EXPECT_FALSE(runner.IsRunning());
}

// ==================== Pause/Resume 测试 ====================

TEST_F(SimulateTaskRunnerTest, PauseWhenNotRunningFails)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    runner.Pause();
    EXPECT_FALSE(runner.IsPaused());
}

TEST_F(SimulateTaskRunnerTest, ResumeWhenNotRunningFails)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    runner.Resume();
    EXPECT_FALSE(runner.IsPaused());
}

TEST_F(SimulateTaskRunnerTest, PauseAndResume)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    runner.Start(1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    runner.Pause();
    EXPECT_TRUE(runner.IsPaused());
    
    runner.Resume();
    EXPECT_FALSE(runner.IsPaused());
    
    runner.Stop();
}

TEST_F(SimulateTaskRunnerTest, PauseTwiceIgnoresSecondCall)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    runner.Start(1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    runner.Pause();
    EXPECT_TRUE(runner.IsPaused());
    
    runner.Pause();  // 重复 Pause
    EXPECT_TRUE(runner.IsPaused());
    
    runner.Stop();
}

TEST_F(SimulateTaskRunnerTest, ResumeWhenNotPausedDoesNothing)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    runner.Start(1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    EXPECT_FALSE(runner.IsPaused());
    runner.Resume();  // 未暂停时 Resume
    EXPECT_FALSE(runner.IsPaused());
    
    runner.Stop();
}

// ==================== GetHealthStatus 测试 ====================

TEST_F(SimulateTaskRunnerTest, GetHealthStatusInitial)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_EQ(status.lastMessage, "not started");
    EXPECT_FALSE(status.isRunning);
}

TEST_F(SimulateTaskRunnerTest, GetHealthStatusAfterStart)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    runner.Start(1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_TRUE(status.isRunning);
    
    runner.Stop();
}

TEST_F(SimulateTaskRunnerTest, GetHealthStatusAfterStop)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    runner.Start(1);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    runner.Stop();
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_FALSE(status.isRunning);
    EXPECT_EQ(status.lastMessage, "task stopped");
}

// ==================== GetNpuUtilization 测试 ====================

TEST_F(SimulateTaskRunnerTest, GetNpuUtilizationInitial)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    // 初始值应该是 -1（未检测）
    EXPECT_EQ(runner.GetNpuUtilization(), -1);
}

// ==================== UpdateHealthStatus 内部方法测试 ====================

TEST_F(SimulateTaskRunnerTest, UpdateHealthStatusSuccess)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    SimulateResult result{SimulateResult::Status::SUCCESS, "test success"};
    runner.UpdateHealthStatus(result);
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_EQ(status.lastStatus, SimulateResult::Status::SUCCESS);
    EXPECT_EQ(status.lastMessage, "test success");
    EXPECT_EQ(status.successCount, 1);
    EXPECT_EQ(status.failureCount, 0);
}

TEST_F(SimulateTaskRunnerTest, UpdateHealthStatusBusy)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    SimulateResult result{SimulateResult::Status::BUSY, "device busy"};
    runner.UpdateHealthStatus(result);
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_EQ(status.lastStatus, SimulateResult::Status::BUSY);
    EXPECT_EQ(status.successCount, 1);  // BUSY 也计入成功
    EXPECT_EQ(status.failureCount, 0);
}

TEST_F(SimulateTaskRunnerTest, UpdateHealthStatusTimeout)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    SimulateResult result{SimulateResult::Status::TIMEOUT, "timeout"};
    runner.UpdateHealthStatus(result);
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_EQ(status.lastStatus, SimulateResult::Status::TIMEOUT);
    EXPECT_EQ(status.successCount, 0);
    EXPECT_EQ(status.failureCount, 1);  // TIMEOUT 计入失败
}

TEST_F(SimulateTaskRunnerTest, UpdateHealthStatusError)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    SimulateResult result{SimulateResult::Status::ERROR, "error occurred"};
    runner.UpdateHealthStatus(result);
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_EQ(status.lastStatus, SimulateResult::Status::ERROR);
    EXPECT_EQ(status.successCount, 0);
    EXPECT_EQ(status.failureCount, 1);  // ERROR 计入失败
}

TEST_F(SimulateTaskRunnerTest, UpdateHealthStatusMultipleTimes)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    runner.UpdateHealthStatus({SimulateResult::Status::SUCCESS, "1"});
    runner.UpdateHealthStatus({SimulateResult::Status::SUCCESS, "2"});
    runner.UpdateHealthStatus({SimulateResult::Status::ERROR, "3"});
    runner.UpdateHealthStatus({SimulateResult::Status::BUSY, "4"});
    
    SimulateHealthStatus status = runner.GetHealthStatus();
    EXPECT_EQ(status.successCount, 3);  // 2 SUCCESS + 1 BUSY
    EXPECT_EQ(status.failureCount, 1);  // 1 ERROR
    EXPECT_EQ(status.lastMessage, "4");
}

// ==================== 集成测试 ====================

TEST_F(SimulateTaskRunnerTest, ExecutorRunsMultipleTimes)
{
    SimulateTaskRunner runner;
    runner.Init(executor_, npuDeviceCardIds_, 10);
    
    executor_->SetSleepMs(10);
    runner.Start(1);  // 间隔 1 秒
    
    // 等待足够时间让执行器运行几次
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    runner.Stop();
    
    // 执行器至少应该运行了 1 次
    EXPECT_GE(executor_->GetRunCount(), 1);
}

TEST_F(SimulateTaskRunnerTest, DestructorStopsRunningTask)
{
    auto executor = std::make_shared<MockSimulateExecutor>();
    
    {
        SimulateTaskRunner runner;
        runner.Init(executor, npuDeviceCardIds_, 10);
        runner.Start(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // 作用域结束，析构函数应该自动 Stop
    }
    
    // 不应该崩溃
    SUCCEED();
}

