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
#include <vector>
#include <string>
#include "lora_manager.h"
#include "executor/executor_interface.h"

using namespace mindie_llm;

// ------------------------ Mock Executor ------------------------ //

class MockModelExecutor : public IExecutor {
public:
    void ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) override {}
    bool ExecutorInstanceInit(std::map<std::string, std::string> &config, bool isMultiNodesInfer,
                              size_t dpIdx = 0) override
    {
        return true;
    }
    bool AsyncExecuteModel(ExecuteModelRequestPtr &modelExecRequest,
                           std::function<void(ModelBatchResultSPtr)> callback = nullptr) override
    {
        return true; // Always return true
    }
    bool AsyncTGCleanup(TGCleanupRequestPtr &TGCleanupRequest) override
    {
        return true; // Always return true
    }
    bool ExecutorParseConfigAndInitGRPC(std::map<std::string, std::string> &configFromManager, bool isMultiNodesInfer,
                                        size_t rankIdx) override
    {
        return true;
    }
    bool MasterAndSlaveModelInit(const std::map<std::string, std::string> &pdInfo) override { return true; }
    bool SetupPDLink(model_execute_data::PDLinkRequest &pdLinkRequest) override { return true; }
    bool ExecuteKVTransfer(PullKVRequestPtr &pullKVRequest,
                           std::function<void(PullKVResponseSPtr)> callback = nullptr) override
    {
        return true;
    }
    bool ExecutorInstanceFinalize() override { return true; }
    uint32_t GetCpuBlockNum() const override { return 1; }
    uint32_t GetNpuBlockNum() const override { return 1; }
    uint32_t GetLwdCloudNpuBlockNum() const override { return 1; }
    uint32_t GetMaxPositionEmbeddings() const override { return 4096; }
    ThinkingConfig GetThinkingConfig() const override
    {
        ThinkingConfig conf;
        return conf;
    }
    model_execute_data::PDLinkResponse GetPDLinkResponse() const override
    {
        return model_execute_data::PDLinkResponse();
    }
    bool ExecutLoraRequest(LoraOperationRequest &loraOperationRequest) override
    {
        return true;
    }

    bool AsyncEOSCleanup(TGCleanupRequestPtr &TGCleanupRequest) override
    {
        return true;
    }

    model_execute_data::LoraOperationResponse GetLoraOperationResponse() const override
    {
        return model_execute_data::LoraOperationResponse();
    }
};

class LoraManagerTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        maxLoras_ = 2;
        mockExecutor_ = std::make_shared<MockModelExecutor>();
        loraManager_ = std::make_shared<LoraManager>(mockExecutor_, maxLoras_);
    }

    void TearDown() override
    {
        loraManager_.reset();
        mockExecutor_.reset();
    }

    uint32_t maxLoras_;
    std::shared_ptr<MockModelExecutor> mockExecutor_;
    std::shared_ptr<LoraManager> loraManager_;
};

TEST_F(LoraManagerTest, TestGetInstanceSuccess)
{
    std::vector<IExecutorSPtr> executors(1, mockExecutor_);
    LoraManager::Initialize(executors, maxLoras_);

    LlmLoraPtr instance = LoraManager::GetInstance(0);
    EXPECT_NE(instance, nullptr);
}

TEST_F(LoraManagerTest, TestLoadSuccess)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    Status status = loraManager_->Load(loraInfo);
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_NE(status.StatusMsg().find("Success"), std::string::npos);
}

TEST_F(LoraManagerTest, TestLoadDuplicated)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    Status status = loraManager_->Load(loraInfo);
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_NE(status.StatusMsg().find("has already been added"), std::string::npos);
}

TEST_F(LoraManagerTest, TestInitLoadedLoras)
{
    std::map<std::string, std::string> testLoraModules = {
        {"test_lora1", "/path/to/test_lora"},
        {"test_lora2", "/path/to/test_lora"}
    };
    ModelParam modelParam;
    modelParam.loraModules = testLoraModules;
    modelParam.modelName = "master_model";
    std::vector<ModelParam> modelParamVec = {modelParam};
    loraManager_->InitLoadedLoras(modelParamVec);
    EXPECT_EQ(loraManager_->loaded_.Size(), 2);
}

TEST_F(LoraManagerTest, TestLoadInvalidPath)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "", "master_model");
    Status status = loraManager_->Load(loraInfo);
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_NE(status.StatusMsg().find("No adapter found"), std::string::npos);
}

TEST_F(LoraManagerTest, TestLoadSlotsFull)
{
    LoraParamSPtr lora1 = std::make_shared<LoraParam>("test_lora1", "/path/to/test_lora1", "master_model");
    LoraParamSPtr lora2 = std::make_shared<LoraParam>("test_lora2", "/path/to/test_lora2", "master_model");
    loraManager_->Load(lora1);
    loraManager_->Load(lora2);
    LoraParamSPtr lora3 = std::make_shared<LoraParam>("test_lora3", "/path/to/test_lora3", "master_model");
    Status status = loraManager_->Load(lora3);
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_NE(status.StatusMsg().find("none are currently unloading"), std::string::npos);
}


TEST_F(LoraManagerTest, TestStartToUnloadSuccess)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    Status result = loraManager_->StartToUnload("test_lora");
    EXPECT_EQ(result.StatusCode(), Error::Code::OK);
    EXPECT_NE(result.StatusMsg().find("removed successfully"), std::string::npos);
}

TEST_F(LoraManagerTest, TestStartToUnloadNotFound)
{
    Status status = loraManager_->StartToUnload("nonexistent_lora");
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_NE(status.StatusMsg().find("cannot be found"), std::string::npos);
}

TEST_F(LoraManagerTest, TestStartToUnloadAlreadyUnloading)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    loraManager_->StartToUnload("test_lora");
    Status status = loraManager_->StartToUnload("test_lora");
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_NE(status.StatusMsg().find("cannot be found"), std::string::npos);
}

TEST_F(LoraManagerTest, TestGetLoadedLorasSuccess)
{
    LoraParamSPtr lora1 = std::make_shared<LoraParam>("test_lora1", "/path/to/test_lora1", "master_model");
    LoraParamSPtr lora2 = std::make_shared<LoraParam>("test_lora2", "/path/to/test_lora2", "master_model");
    loraManager_->Load(lora1);
    loraManager_->Load(lora2);
    std::vector<LoraParamSPtr> loadedLoras;
    Status status = loraManager_->GetLoadedLoras(loadedLoras);
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_EQ(loraManager_->loaded_.Size(), 2);
    EXPECT_EQ(loadedLoras.size(), 2);
}

TEST_F(LoraManagerTest, TestGetLoadedLorasEmpty)
{
    std::vector<LoraParamSPtr> loadedLoras;
    Status status = loraManager_->GetLoadedLoras(loadedLoras);
    EXPECT_EQ(status.StatusCode(), Error::Code::OK);
    EXPECT_EQ(loadedLoras.size(), 0);
}

TEST_F(LoraManagerTest, TestTryUnLoadWaitingSuccess)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    loraManager_->StartToUnload("test_lora");
    loraManager_->TryUnLoadWaiting();
    std::vector<LoraParamSPtr> loadedLoras;
    loraManager_->GetLoadedLoras(loadedLoras);
    EXPECT_EQ(loadedLoras.size(), 0);
    EXPECT_EQ(loraManager_->loaded_.Size(), 0);
    EXPECT_EQ(loraManager_->wait2Unloaded_.Size(), 0);
}

TEST_F(LoraManagerTest, TestTryUnLoadWaitingReferenceExists)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    loraManager_->IncLoraRef("test_lora");
    loraManager_->StartToUnload("test_lora");
    loraManager_->TryUnLoadWaiting();
    EXPECT_EQ(loraManager_->loaded_.Size(), 1);
    EXPECT_EQ(loraManager_->wait2Unloaded_.Size(), 1);
}

TEST_F(LoraManagerTest, TestValidateLoraIdSuccess)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    bool isValid = loraManager_->ValidateLoraId("test_lora");
    EXPECT_TRUE(isValid);
}

TEST_F(LoraManagerTest, TestValidateLoraIdFailed)
{
    bool isValid = loraManager_->ValidateLoraId("nonexistent_lora");
    EXPECT_FALSE(isValid);
}

TEST_F(LoraManagerTest, TestIncDecLoraRefSuccess)
{
    LoraParamSPtr loraInfo = std::make_shared<LoraParam>("test_lora", "/path/to/test_lora", "master_model");
    loraManager_->Load(loraInfo);
    loraManager_->IncLoraRef("test_lora");
    EXPECT_EQ(loraManager_->loraIdRef_.Get("test_lora").value(), 1);
    loraManager_->DecLoraRef("test_lora");
    EXPECT_EQ(loraManager_->loraIdRef_.Get("test_lora").value(), 0);
}

TEST_F(LoraManagerTest, TestIncDecLoraRefInvalidId)
{
    loraManager_->IncLoraRef("nonexistent_lora");
    EXPECT_EQ(loraManager_->loraIdRef_.Count("nonexistent_lora"), 0);
    loraManager_->DecLoraRef("nonexistent_lora");
    EXPECT_EQ(loraManager_->loraIdRef_.Count("nonexistent_lora"), 0);
}
