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

#include <libgen.h>
#include <sys/stat.h>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "config_manager.h"
#include "file_utils.h"
#include "common_util.h"
#include "dt_tools.h"
#include "base_config_manager.h"

using OrderedJson = nlohmann::ordered_json;
using Json = nlohmann::json;

using namespace mindie_llm;

namespace mindie_llm {
class ServerConfigManagerTest : public testing::Test {
protected:
    void SetUp()
    {
        jsonPath = GetCwdDirectory() + "/conf/config.json";
        jsonPathError = GetCwdDirectory() + "/conf/config_er.json";
    }
    void TearDown()
    {
    }

    std::string jsonPath;
    std::string jsonPathError;
};

void ModifyConfigJson(const std::string &inputFile, const std::string &outputFile)
{
    mode_t fileMode = 0640;
    chmod(outputFile.c_str(), fileMode);
    std::ifstream file(inputFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open source config json file." << std::endl;
        return;
    }

    OrderedJson data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse json file." << e.what() << std::endl;
        return;
    }
    file.close();

    data["ServerConfig"]["httpsEnabled"] = true;
    data["ServerConfig"]["interCommTLSEnabled"] = true;

    std::ofstream out_file(outputFile);
    if (!out_file.is_open()) {
        std::cerr << "Failed to open modified config json file." << std::endl;
        return;
    }

    size_t indent = 4;
    try {
        out_file << data.dump(indent);
    } catch (const std::exception& e) {
        std::cerr << "Failed to write json file." << e.what() << std::endl;
        return;
    }
    out_file.close();
}

TEST_F(ServerConfigManagerTest, testParamFuncsSuccess)
{
    ServerConfigManager serverConfigManager(jsonPath);
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    EXPECT_TRUE(serverConfigManager.CheckParam());
    auto serverConfig = serverConfigManager.GetParam();
    int32_t expectedPort = 1025;
    EXPECT_EQ(serverConfig.port, expectedPort);
    EXPECT_TRUE(serverConfigManager.GetDecodeStatus());
}

TEST_F(ServerConfigManagerTest, testParamFuncsFail)
{
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = {
        {"ServerConfig", {{"httpsEnabled", true}}}
    };
    UpdateConfigJson(jsonPathError, newJsonPath, updates);
    ServerConfigManager serverConfigManager(newJsonPath);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    EXPECT_FALSE(serverConfigManager.CheckParam());
    auto serverConfig = serverConfigManager.GetParam();
    int32_t expectedPort = 3006;
    EXPECT_FALSE(serverConfig.port == expectedPort);
    GlobalMockObject::verify();
}

TEST_F(ServerConfigManagerTest, InitFromJson)
{
    const char* envVal = std::getenv("MINDIE_LLM_HOME_PATH");
    if (envVal != nullptr) {
        std::cout << "MINDIE_LLM_HOME_PATH is: " << envVal << std::endl;
    }
    std::string homePath;
    GetHomePath(homePath);
    std::string inputFile = GetCwdDirectory() + "/conf/config.json";
    std::string outputFile = GetCwdDirectory() + "/conf/modifiedConfig.json";
    ModifyConfigJson(inputFile, outputFile);
    ServerConfigManager serverConfigManager(outputFile);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    MOCKER(ParamChecker::CheckJsonArray).stubs().will(returnValue(false));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    GlobalMockObject::verify();
}

TEST_F(ServerConfigManagerTest, InitHttpsConfigFromJson_HandlesEmptyJson)
{
    Json emptyJson = Json::object();
    ServerConfigManager manager(jsonPath);
    manager.InitHttpsConfigFromJson(emptyJson, false);
    
    EXPECT_TRUE(manager.serverConfig_.tlsCaPath.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsCaFile.empty());
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, SetsTlsCaPath)
{
    ServerConfigManager manager(jsonPath);
    Json json = {{"tlsCaPath", "/etc/certs/ca.pem"}};
    manager.InitHttpsConfigFromJson(json, false);
    
    EXPECT_EQ(manager.serverConfig_.tlsCaPath, "/etc/certs/ca.pem");
    EXPECT_TRUE(manager.serverConfig_.tlsCaFile.empty());
}

TEST_F(ServerConfigManagerTest, AddsTlsCaFiles)
{
    ServerConfigManager manager(jsonPath);
    Json json = {{"tlsCaFile", {"file1.pem", "file2.pem"}}};
    manager.InitHttpsConfigFromJson(json, false);
    
    std::set<std::string> expected = {"file1.pem", "file2.pem"};
    EXPECT_EQ(manager.serverConfig_.tlsCaFile, expected);
}

TEST_F(ServerConfigManagerTest, InitHttpsConfigFromJson_HandlesInvalidTlsCaFileType)
{
    ServerConfigManager manager(jsonPath);
    Json json = {{"tlsCaFile", {1, 2, 3}}};
    manager.InitHttpsConfigFromJson(json, false);
    
    EXPECT_TRUE(manager.serverConfig_.tlsCaFile.empty());
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, SetsBothTlsConfigs)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"tlsCaPath", "/etc/certs/ca.pem"},
        {"tlsCaFile", {"file1.pem", "file2.pem"}}
    };
    manager.InitHttpsConfigFromJson(json, false);
    
    EXPECT_EQ(manager.serverConfig_.tlsCaPath, "/etc/certs/ca.pem");
    std::set<std::string> expectedFiles = {"file1.pem", "file2.pem"};
    EXPECT_EQ(manager.serverConfig_.tlsCaFile, expectedFiles);
}

TEST_F(ServerConfigManagerTest, InitHttpsBusinessConfigFromJson_HandlesEmptyJson)
{
    ServerConfigManager manager(jsonPath);
    Json json = Json::object();
    manager.InitHttpsBusinessConfigFromJson(json);
    
    EXPECT_TRUE(manager.serverConfig_.tlsCert.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsCrlPath.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsCrlFiles.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsCaFile.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsPk.empty());
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsBusinessConfigFromJson_SetsAllValidFields)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"tlsCert", "server.crt"},
        {"tlsCrlPath", "/crl/path"},
        {"tlsCrlFiles", {"crl1.pem", "crl2.pem"}},
        {"tlsCaFile", {"ca1.pem", "ca2.pem"}},
        {"tlsPk", "private.key"},
    };
    
    manager.InitHttpsBusinessConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.tlsCert, "server.crt");
    EXPECT_EQ(manager.serverConfig_.tlsCrlPath, "/crl/path");
    EXPECT_EQ(manager.serverConfig_.tlsCrlFiles,
              (std::set<std::string>{"crl1.pem", "crl2.pem"}));
    EXPECT_EQ(manager.serverConfig_.tlsCaFile,
              (std::set<std::string>{"ca1.pem", "ca2.pem"}));
    EXPECT_EQ(manager.serverConfig_.tlsPk, "private.key");
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, HandlesInvalidTlsCrlFilesType)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"tlsCrlFiles", {1, 2, 3}}, // 整数数组，应为字符串
        {"tlsPk", "should-not-be-set.key"}
    };
    
    manager.InitHttpsBusinessConfigFromJson(json);
    
    EXPECT_TRUE(manager.serverConfig_.tlsCrlFiles.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsPk.empty()); // 后续字段不应处理
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsBusinessConfigFromJson_HandlesInvalidTlsCaFileType)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"tlsCaFile", true}, // 布尔值，应为数组
    };
    
    manager.InitHttpsBusinessConfigFromJson(json);
    
    EXPECT_TRUE(manager.serverConfig_.tlsCaFile.empty());
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsBusinessConfigFromJson_SetsPartialFields)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"tlsCert", "partial.crt"},
        {"tlsPk", "partial.key"}
    };
    
    manager.InitHttpsBusinessConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.tlsCert, "partial.crt");
    EXPECT_EQ(manager.serverConfig_.tlsPk, "partial.key");
    EXPECT_TRUE(manager.serverConfig_.tlsCrlPath.empty());
    EXPECT_TRUE(manager.serverConfig_.tlsCrlFiles.empty());
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsBusinessConfigFromJson_OrderHandlingWhenCrlFilesInvalid)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"tlsCert", "should-be-set.crt"},
        {"tlsCrlFiles", {1, 2}}, // 无效类型
        {"tlsPk", "should-not-be-set.key"}
    };
    
    manager.InitHttpsBusinessConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.tlsCert, "should-be-set.crt"); // 在错误前设置
    EXPECT_TRUE(manager.serverConfig_.tlsPk.empty()); // 在错误后未设置
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsManagementConfigFromJson_HandlesEmptyJson)
{
    ServerConfigManager manager(jsonPath);
    Json json = Json::object();
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_TRUE(manager.serverConfig_.managementTlsCert.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsCrlPath.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsCrlFiles.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsCaFile.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsPk.empty());
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsManagementConfigFromJson_SetsAllValidFields)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"managementTlsCert", "mgmt.crt"},
        {"managementTlsCrlPath", "/mgmt/crl/path"},
        {"managementTlsCrlFiles", {"mgmt_crl1.pem", "mgmt_crl2.pem"}},
        {"managementTlsCaFile", {"mgmt_ca1.pem", "mgmt_ca2.pem"}},
        {"managementTlsPk", "mgmt_private.key"},
    };
    
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.managementTlsCert, "mgmt.crt");
    EXPECT_EQ(manager.serverConfig_.managementTlsCrlPath, "/mgmt/crl/path");
    EXPECT_EQ(manager.serverConfig_.managementTlsCrlFiles,
              (std::set<std::string>{"mgmt_crl1.pem", "mgmt_crl2.pem"}));
    EXPECT_EQ(manager.serverConfig_.managementTlsCaFile,
              (std::set<std::string>{"mgmt_ca1.pem", "mgmt_ca2.pem"}));
    EXPECT_EQ(manager.serverConfig_.managementTlsPk, "mgmt_private.key");
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, HandlesInvalidManagementTlsCrlFilesType)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"managementTlsCrlFiles", {1, 2, 3}}, // 整数数组，应为字符串
        {"managementTlsPk", "should-not-be-set.key"}
    };
    
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_TRUE(manager.serverConfig_.managementTlsCrlFiles.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsPk.empty());
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, HandlesInvalidManagementTlsCaFileType)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"managementTlsCaFile", true}, // 布尔值，应为数组
    };
    
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_TRUE(manager.serverConfig_.managementTlsCaFile.empty());
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsManagementConfigFromJson_SetsPartialFields)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"managementTlsCert", "partial_mgmt.crt"},
        {"managementTlsPk", "partial_mgmt.key"}
    };
    
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.managementTlsCert, "partial_mgmt.crt");
    EXPECT_EQ(manager.serverConfig_.managementTlsPk, "partial_mgmt.key");
    EXPECT_TRUE(manager.serverConfig_.managementTlsCrlPath.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsCrlFiles.empty());
    EXPECT_TRUE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, InitHttpsManagementConfigFromJson_OrderHandlingWhenCrlFilesInvalid)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"managementTlsCert", "should-be-set.crt"},
        {"managementTlsCrlFiles", {1, 2}}, // 无效类型
        {"managementTlsPk", "should-not-be-set.key"}
    };
    
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.managementTlsCert, "should-be-set.crt"); // 在错误前设置
    EXPECT_TRUE(manager.serverConfig_.managementTlsPk.empty()); // 在错误后未设置
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, OrderHandlingWhenCaFileInvalid)
{
    ServerConfigManager manager(jsonPath);
    Json json = {
        {"managementTlsCert", "should-be-set.crt"},
        {"managementTlsCrlPath", "/crl/path"},
        {"managementTlsCaFile", 123}, // 无效类型
        {"managementTlsPk", "should-not-be-set.key"}
    };
    
    manager.InitHttpsManagementConfigFromJson(json);
    
    EXPECT_EQ(manager.serverConfig_.managementTlsCert, "should-be-set.crt");
    EXPECT_EQ(manager.serverConfig_.managementTlsCrlPath, "/crl/path");
    EXPECT_TRUE(manager.serverConfig_.managementTlsCaFile.empty());
    EXPECT_TRUE(manager.serverConfig_.managementTlsPk.empty()); // 后续字段未设置
    EXPECT_FALSE(manager.jsonDecodeSuccess_);
}

TEST_F(ServerConfigManagerTest, CheckHttpsConfig)
{
    std::string inputFile = GetCwdDirectory() + "/conf/config.json";
    std::string outputFile = GetCwdDirectory() + "/conf/modifiedConfig.json";
    ModifyConfigJson(inputFile, outputFile);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    MOCKER(ParamChecker::CheckPath).stubs().will(returnValue(true));
    MOCKER(FileUtils::CheckFileExists).stubs().will(returnValue(true));
    MOCKER(FileUtils::CheckDirectoryExists).stubs().will(returnValue(false));
    ServerConfigManager serverConfigManager(outputFile);
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    EXPECT_TRUE(serverConfigManager.CheckHttpsConfig(true));
}


TEST_F(ServerConfigManagerTest, TestIpAddressParamFuncsSucc)
{
    std::string jsonPath = GetCwdDirectory() + "/conf/config.json";
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig1.json";
    std::unordered_map<std::string, OrderedJson> updates = { \
        {"ServerConfig", \
        {{"layerwiseDisaggregated",true},\
        {"layerwiseDisaggregatedRoleType","slave"},\
        {"layerwiseDisaggregatedDataPort",10001},\
        {"layerwiseDisaggregatedCrtlPort",{10001,10002}},\
        {"layerwiseDisaggregatedMasterIpAddress", "76.76.26.197"}, \
        {"layerwiseDisaggregatedSlaveIpAddress", {"76.76.26.198"}}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    ServerConfigManager serverConfigManager(newJsonPath);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    EXPECT_TRUE(serverConfigManager.CheckParam());
    auto serverConfig = serverConfigManager.GetParam();
    std::string edgeIpAddress = "76.76.26.197";
    std::vector<std::string> cloudIpAddress = {"76.76.26.198"};
    EXPECT_TRUE(serverConfig.layerwiseDisaggregatedMasterIpAddress == edgeIpAddress);
    EXPECT_TRUE(serverConfig.layerwiseDisaggregatedSlaveIpAddress[0] == cloudIpAddress[0]);
    GlobalMockObject::verify();
}

TEST_F(ServerConfigManagerTest, TestIpAddressParamFuncsFail)
{
    std::string jsonPath = GetCwdDirectory() + "/conf/config.json";
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = { \
        {"ServerConfig", \
        {{"layerwiseDisaggregated",true},\
        {"layerwiseDisaggregatedRoleType","slave"},\
        {"layerwiseDisaggregatedDataPort",102},\
        {"layerwiseDisaggregatedCrtlPort",{102,103}},\
        {"layerwiseDisaggregatedMasterIpAddress", "76.76.26.197.1"}, \
        {"layerwiseDisaggregatedSlaveIpAddress", {"76.76.26.198.2"}}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    ServerConfigManager serverConfigManager(newJsonPath);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    auto serverConfig = serverConfigManager.GetParam();
    if (serverConfig.layerwiseDisaggregated) {
        EXPECT_FALSE(serverConfigManager.CheckParam());
    }
   
    GlobalMockObject::verify();
}


TEST_F(ServerConfigManagerTest, TestLayerwiseDisaggregatedFuncsSucc)
{
    std::string jsonPath = GetCwdDirectory() + "/conf/config.json";
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = { \
        {"ServerConfig", \
        {{"layerwiseDisaggregated",true},\
        {"layerwiseDisaggregatedRoleType","slave"},\
        {"layerwiseDisaggregatedDataPort",10001},\
        {"layerwiseDisaggregatedCrtlPort",{10001,10002}},\
        {"layerwiseDisaggregatedMasterIpAddress", "76.76.26.197"}, \
        {"layerwiseDisaggregatedSlaveIpAddress", {"76.76.26.198"}}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    ServerConfigManager serverConfigManager(newJsonPath);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    auto serverConfig = serverConfigManager.GetParam();
    EXPECT_TRUE(serverConfig.layerwiseDisaggregated == true);
    EXPECT_TRUE(serverConfig.layerwiseDisaggregatedRoleType == "slave");
    EXPECT_TRUE(serverConfigManager.CheckParam());
    GlobalMockObject::verify();
}

TEST_F(ServerConfigManagerTest, TestLayerwiseDisaggregatedFuncsFail)
{
    std::string jsonPath = GetCwdDirectory() + "/conf/config.json";
    std::string newJsonPath = GetCwdDirectory() + "/conf/newConfig.json";
    std::unordered_map<std::string, OrderedJson> updates = { \
        {"ServerConfig", \
        {{"layerwiseDisaggregated",true},\
        {"layerwiseDisaggregatedRoleType","test"},\
        {"layerwiseDisaggregatedDataPort",10001},\
        {"layerwiseDisaggregatedCrtlPort",{10001,10002}},\
        {"layerwiseDisaggregatedMasterIpAddress", "76.76.26.197"}, \
        {"layerwiseDisaggregatedSlaveIpAddress", {"76.76.26.198"}}}}
    };
    UpdateConfigJson(jsonPath, newJsonPath, updates);
    ServerConfigManager serverConfigManager(newJsonPath);
    MOCKER(static_cast<bool(*)(const std::string&, std::string&, bool, mode_t, bool, uint64_t)>(FileUtils::IsFileValid))
    .stubs().will(returnValue(true));
    MOCKER(static_cast<bool(*)(const std::string&, const std::string&, std::string&, bool, std::string&)>(FileUtils::RegularFilePath))
    .stubs().will(returnValue(true));
    EXPECT_TRUE(serverConfigManager.InitFromJson());
    auto serverConfig = serverConfigManager.GetParam();
    EXPECT_TRUE(serverConfig.layerwiseDisaggregated == true);
    EXPECT_TRUE(serverConfig.layerwiseDisaggregatedRoleType == "test");
    EXPECT_FALSE(serverConfigManager.CheckParam());
    GlobalMockObject::verify();
}
} // namespace mindie_llm
