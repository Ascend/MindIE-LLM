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
#include "lora/loraops_mixin.h"
#include "lora_manager.h"

using namespace mindie_llm;

LoraParamSPtr CreateTestLoraParam(const std::string& name = "test_lora",
                                  const std::string& path = "/path/to/test_lora",
                                  const std::string& master = "master_model")
{
    auto param = std::make_shared<LoraParam>(name, path, master);
    return param;
}

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

class LoraOpsMixinTest : public testing::Test {
protected:
    LoraOpsMixin mixin;

    void SetUp() override
    {
        maxLoras_ = 2;
        mockExecutor_ = std::make_shared<MockModelExecutor>();
        std::vector<IExecutorSPtr> executors(1, mockExecutor_);
        LoraManager::Initialize(executors, maxLoras_);
    }

    void TearDown() override
    {
        mockExecutor_.reset();
    }
    uint32_t maxLoras_;
    std::shared_ptr<MockModelExecutor> mockExecutor_;
};

TEST_F(LoraOpsMixinTest, InitStaticLoras)
{
    std::map<std::string, std::string> testLoraModules = {
        {"test_lora", "/path/to/test_lora"}
    };
    ModelParam modelParam;
    modelParam.loraModules = testLoraModules;
    modelParam.modelName = "master_model";
    std::vector<ModelParam> modelParamVec = {modelParam};
    size_t dpSize = 1;

    mixin.InitStaticLoras(modelParamVec, dpSize);
    EXPECT_EQ(LoraManager::GetInstance(0)->loaded_.Size(), 1);
}

TEST_F(LoraOpsMixinTest, TestLoraLoadInvalidLoraInfoCount)
{
    std::vector<LoraParamSPtr> loraInfo;
    Status status = mixin.LoraLoad(loraInfo, 1);

    EXPECT_EQ(status.StatusCode(), Error::Code::ERROR);
    EXPECT_NE(status.StatusMsg().find("invalid"), std::string::npos);
}

TEST_F(LoraOpsMixinTest, TestLoraLoadSuccess)
{
    std::vector<mindie_llm::LoraParamSPtr> loraInfo;
    loraInfo.push_back(CreateTestLoraParam());
    size_t dpSize = 1;

    Status status = mixin.LoraLoad(loraInfo, dpSize);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(LoraOpsMixinTest, TestLoraUnLoadInvalidLoraInfoCount)
{
    std::vector<LoraParamSPtr> loraInfo;
    Status status = mixin.LoraUnLoad(loraInfo, 1);
    EXPECT_EQ(status.StatusCode(), Error::Code::ERROR);
    EXPECT_NE(status.StatusMsg().find("invalid"), std::string::npos);
}

TEST_F(LoraOpsMixinTest, TestLoraUnLoadSuccess)
{
    std::vector<LoraParamSPtr> loraInfo;
    loraInfo.push_back(CreateTestLoraParam("unload_test"));
    size_t dpSize = 1;

    Status status = mixin.LoraUnLoad(loraInfo, dpSize);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(LoraOpsMixinTest, TestLoraGetLoaded)
{
    std::vector<LoraParamSPtr> loraInfo;
    size_t dpSize = 1;

    Status status = mixin.LoraGetLoaded(loraInfo, dpSize);
    EXPECT_TRUE(status.IsOk());
}