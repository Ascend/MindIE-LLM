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
#include "mockcpp/mockcpp.hpp"
#include "param_checker.h"
#include "config_manager.h"
#include "base_config_manager.h"
#include "dt_tools.h"

using namespace mindie_llm;
using OrderedJson = nlohmann::ordered_json;

bool MockGetVocabJsonData(const std::string &configFile, std::string &baseDir, Json &jsonData)
{
    jsonData["torch_dtype"] = "mockTorchType";
    jsonData["vocab_size"] = 1;
    return true;
}

bool MockGetPaddedVocabJsonData(const std::string &configFile, std::string &baseDir, Json &jsonData)
{
    jsonData["torch_dtype"] = "mockTorchType";
    jsonData["padded_vocab_size"] = 1;
    return true;
}

namespace mindie_llm {
class ModelDeployConfigManagerTest : public testing::Test {
protected:
    void SetUp() { jsonPath = GetCwdDirectory() + "/conf/config.json"; }
    void TearDown() { GlobalMockObject::verify(); }

    std::unique_ptr<ModelDeployConfigManager> configManager;
    std::string jsonPath;
};

extern "C++" {
void GetJsonModelConfig(struct ModelDeployConfig &modelConfig);
}

TEST_F(ModelDeployConfigManagerTest, TestGetJsonModelConfig)
{
    ModelDeployConfig modelConfig;
    GetJsonModelConfig(modelConfig);
    modelConfig.modelWeightPath = GetCwdDirectory() + "/conf";
    GetJsonModelConfig(modelConfig);
    MOCKER(ParamChecker::IsWithinRange).stubs().will(returnValue(true));
    MOCKER(ParamChecker::GetJsonData).stubs().will(invoke(MockGetVocabJsonData));
    GetJsonModelConfig(modelConfig);
    modelConfig.modelWeightPath = GetCwdDirectory() + "/mockconf";
    GetJsonModelConfig(modelConfig);
}

TEST_F(ModelDeployConfigManagerTest, TestGetPaddedVocabJsonModelConfig)
{
    ModelDeployConfig modelConfig;
    modelConfig.modelWeightPath = GetCwdDirectory() + "/conf";
    MOCKER(ParamChecker::GetJsonData).stubs().will(invoke(MockGetPaddedVocabJsonData));
    GetJsonModelConfig(modelConfig);
}

TEST_F(ModelDeployConfigManagerTest, TestInitFromJson)
{
    configManager = std::make_unique<ModelDeployConfigManager>(jsonPath);
    EXPECT_TRUE(configManager->InitFromJson());
    MOCKER(ParamChecker::CheckJsonParamType).stubs().will(returnValue(false));
    EXPECT_FALSE(configManager->InitFromJson());
    MOCKER(BaseConfig::CheckSystemConfig).stubs().will(returnValue(false));
    EXPECT_FALSE(configManager->InitFromJson());
}

TEST_F(ModelDeployConfigManagerTest, TestInitFromJsonWithErrorModelConfig)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newModelDeployConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {"BackendConfig", {{"ModelDeployConfig", {{"ModelConfig", 1}}}}}};
    MOCKER(ParamChecker::CheckJsonParamType).stubs().will(returnValue(true));
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    configManager = std::make_unique<ModelDeployConfigManager>(newJsonPath);
    EXPECT_FALSE(configManager->InitFromJson());
}

TEST_F(ModelDeployConfigManagerTest, TestCheckParamWithWrongParam)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newModelDeployConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {
            "BackendConfig", {
                {"ModelDeployConfig", {
                    {"maxSeqLen", 0},
                    {"maxInputTokenLen", 4194305},
                    {"truncation", -1},
                    {"ModelConfig", OrderedJson::array({
                        {
                            {"modelInstanceType", "StandardMock"},
                            {"modelName", "llama_65b"},
                            {"modelWeightPath", "../conf"},
                            {"worldSize", 0},
                            {"cpuMemSize", 5},
                            {"npuMemSize", -1},
                            {"backendType", "invalid"},
                            {"trustRemoteCode", false}
                        }
                    })}
                }}
            }
        }
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    MOCKER(ParamChecker::CheckJsonParamType).stubs().will(returnValue(true));
    configManager = std::make_unique<ModelDeployConfigManager>(newJsonPath);
    EXPECT_TRUE(configManager->InitFromJson());
    EXPECT_FALSE(configManager->CheckParam());
}

TEST_F(ModelDeployConfigManagerTest, CheckTemplateConfig)
{
    configManager = std::make_unique<ModelDeployConfigManager>(jsonPath);
    EXPECT_TRUE(configManager->CheckParam());
    configManager->CheckTemplateConfig("Standard", 2);
    EXPECT_FALSE(configManager->CheckParam());
}

} // namespace mindie_llm