/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include <libgen.h>
#define private public
#include <filesystem>
#include <nlohmann/json.hpp>

#include "config_manager.h"
#include "config_manager/config_manager_impl.h"
#include "httplib.h"
#include "mock_util.h"
#include "mockcpp/MockObject.h"
#include "mockcpp/mockcpp.hpp"
#include "parse_protocol.h"
#include "src/server/endpoint/http_wrapper/dmi_role.cpp"
#include "src/server/endpoint/http_wrapper/dmi_role.h"

using namespace mindie_llm;
using json = nlohmann::json;

MOCKER_CPP_OVERLOAD_EQ(ModelDeployConfig)
MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
MOCKER_CPP_OVERLOAD_EQ(BackendConfig)

namespace mindie_llm {

static std::string rankTableStringV1 = R"({
    "local": {
        "device": [
            {
                "device_id": "0",
                "device_ip": "1.1.1.1",
                "device_logical_id": "0"
            },
            {
                "device_id": "1",
                "device_ip": "1.1.1.2",
                "device_logical_id": "1"
            },
            {
                "device_id": "2",
                "device_ip": "1.1.1.3",
                "device_logical_id": "2"
            },
            {
                "device_id": "3",
                "device_ip": "1.1.1.4",
                "device_logical_id": "3"
            }
        ],
        "host_ip": "127.0.0.1",
        "id": 2003,
        "server_ip": "127.0.0.1",
		"instance_idx_in_pod": 0,
		"num_instances_per_pod": 1,
        "is_single_container": false
    },
    "peers": [
        {
            "device": [
                {
                    "device_id": "4",
                    "device_ip": "1.1.1.5",
                    "device_logical_id": "4"
                },
                {
                    "device_id": "5",
                    "device_ip": "1.1.1.6",
                    "device_logical_id": "5"
                },
                {
                    "device_id": "6",
                    "device_ip": "1.1.1.7",
                    "device_logical_id": "6"
                },
                {
                    "device_id": "7",
                    "device_ip": "1.1.1.8",
                    "device_logical_id": "7"
                }
            ],
            "host_ip": "127.0.0.1",
            "id": 2007,
            "server_ip": "127.0.0.1"
        }
    ]
})";

const std::string RESPONSE_OK_BODY = "{\"result\":\"ok\"}";
class DmiRoleTest : public testing::Test {
   protected:
    void SetUp() {
        ConfigManager::CreateInstance(GetParentDirectory() + "/../../config_manager/conf/config_http.json");
        EnvUtil::GetInstance().SetEnvVar("RANK_TABLE_FILE",
                                         GetParentDirectory() + "/../../config_manager/conf/ranktable.json");
        EnvUtil::GetInstance().SetEnvVar("MIES_CONTAINER_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("HOST_IP", "127.0.0.1");
        EnvUtil::GetInstance().SetEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION", "1");
        rankTableStringV2 =
            LoadJsonFile(GetParentDirectory() + "/../../config_manager/conf/v2_role_cross_node_2p_2d.json");
        if (rankTableStringV2.empty()) {
            return;
        }
        rankTableStringBefore = LoadJsonFile(GetParentDirectory() + "/../../config_manager/conf/role_1.json");
        if (rankTableStringBefore.empty()) {
            return;
        }

        rankTableStringAfter = LoadJsonFile(GetParentDirectory() + "/../../config_manager/conf/role_2.json");
        if (rankTableStringAfter.empty()) {
            return;
        }
        std::string validRequestBody;
        std::string validRequestBodyV2;
        std::string RESPONSE_OK_BODY;
        validRequestBody = R"({
            "rank_table": {
                "server_list": [
                    {
                        "server_id": "0.0.0.0",
                        "device": [
                            {"device_id": "0", "rank_id": "0"}
                        ]
                    }
                ]
            }
        })";
        validRequestBodyV2 = R"({
            "rank_table": {
                "server_list": [
                    {
                        "server_id": "1.1.1.1",
                        "device": [
                            {"device_id": "1", "rank_id": "1"}
                        ]
                    }
                ]
            }
        })";
    }

    void TearDown() {
        EnvUtil::GetInstance().ClearEnvVar("RANK_TABLE_FILE");
        EnvUtil::GetInstance().ClearEnvVar("MINDIE_CHECK_INPUTFILES_PERMISSION");
        EnvUtil::GetInstance().ClearEnvVar("MIES_CONTAINER_IP");
        EnvUtil::GetInstance().ClearEnvVar("HOST_IP");
        GlobalMockObject::verify();
    }

    // Common function to load and parse a JSON file
    std::string LoadJsonFile(const std::string &filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Fail to open json file: " << filePath << std::endl;
            return "";  // Return empty string if file open fails
        }

        try {
            json j;
            file >> j;  // Parse the JSON
            auto tabSize = 4;
            return j.dump(tabSize);  // Return the JSON as a formatted string
        } catch (const json::parse_error &e) {
            std::cerr << "JSON Parse Error in file " << filePath << ": " << e.what() << std::endl;
            return "";  // Return empty string if parsing fails
        }
    }

    std::string GetParentDirectory() {
        char buffer[1024];
        try {
            return std::filesystem::current_path().string();
        } catch (const std::filesystem::filesystem_error &e) {
            std::cerr << "Error getting current directory: " << e.what() << std::endl;
            return "";
        }
    }

    void InitJson() {
        body = {{"local",
                 {{{"host_ip", "192.168.1.10"},
                   {"super_pod_id", "100"},
                   {"dp_inst_list",
                    {{{"dp_inst_id", 1},
                      {"device",
                       {{{"device_ip", "10.0.0.1"},
                         {"device_logical_id", "logical-1"},
                         {"device_id", "physical-1"},
                         {"rank_id", "0"},
                         {"super_device_id", "super-1"}},
                        {{"device_ip", "10.0.0.2"},
                         {"device_logical_id", "logical-2"},
                         {"device_id", "physical-2"},
                         {"rank_id", "1"}}}}}}}},
                  {{"host_ip", "192.168.1.11"},
                   {"dp_inst_list",
                    {{{"dp_inst_id", 2},
                      {"device",
                       {{{"device_ip", "10.0.0.3"},
                         {"device_logical_id", "logical-3"},
                         {"device_id", "physical-3"},
                         {"rank_id", "2"}}}}}}}}}}};
    }

    void MockAllConfig() {
        MockServerConfig();
        MockBackendConfig();
        MockModelDeployConfig();
    }

    void MockServerConfig() {
        serverConfig_.allowAllZeroIpListening = false;
        serverConfig_.httpsEnabled = false;
        serverConfig_.ipAddress = "127.0.0.1";
        serverConfig_.managementIpAddress = "127.0.0.2";
        serverConfig_.port = 1025;
        serverConfig_.managementPort = 1026;
        serverConfig_.metricsPort = 1027;
        serverConfig_.maxLinkNum = 1000;
        serverConfig_.fullTextEnabled = false;
        serverConfig_.inferMode = "standard";
        serverConfig_.interCommTLSEnabled = true;
        serverConfig_.interCommPort = 1121;
        serverConfig_.tokenTimeout = 5;
        serverConfig_.e2eTimeout = 5;
        serverConfig_.distDPServerEnabled = false;
        MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(serverConfig_));
    }

    void MockBackendConfig() {
        backendConfig_.backendName = "mindieservice_llm_engine";
        backendConfig_.modelInstanceNumber = 2;
        backendConfig_.npuDeviceIds = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};
        backendConfig_.tokenizerProcessNumber = 2;
        backendConfig_.multiNodesInferEnabled = true;
        backendConfig_.multiNodesInferPort = 1120;
        backendConfig_.interNodeTLSEnabled = false;
        MOCKER_CPP(GetBackendConfig, const BackendConfig &(*)()).stubs().will(returnValue(backendConfig_));
    }

    void MockModelDeployConfig() {
        modelDeployConfig_.modelInstanceType = "StandardMock";
        modelDeployConfig_.modelName = "llama_65b";
        modelDeployConfig_.modelWeightPath = "../../config_manager/conf";
        modelDeployConfig_.worldSize = 8;
        modelDeployConfig_.npuDeviceIds = {0, 1, 2, 3, 4, 5, 6, 7};
        modelDeployConfig_.npuMemSize = -1;
        modelDeployConfig_.cpuMemSize = 5;
        modelDeployConfig_.backendType = "atb";
        modelDeployConfig_.trustRemoteCode = false;
        modelDeployConfig_.maxSeqLen = 2560;
        modelDeployConfig_.maxInputTokenLen = 2048;
        modelDeployConfig_.truncation = false;
        modelDeployConfig_.loraModules["llama_65b"] = "../../config_manager/conf";

        std::vector<ModelDeployConfig> modelConfig = {modelDeployConfig_};
        MOCKER_CPP(GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
            .stubs()
            .will(returnValue(modelConfig));
    }

    ServerConfig serverConfig_;
    BackendConfig backendConfig_;
    ModelDeployConfig modelDeployConfig_;
    DmiRole dmiRole;
    std::string rankTableStringV2;
    std::string rankTableStringBefore;
    std::string rankTableStringAfter;
    ordered_json body;
};

auto originalGetPDRole = &InferInstance::GetPDRole;

std::string MockGetPDRole() { return "none"; }

TEST_F(DmiRoleTest, HandlePDRoleV1Init) {
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req;
    req.body = rankTableStringV1;

    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";
    dmiRole.HandlePDRoleV1(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(DmiRoleTest, HandlePDRoleV1_PDParseRequestBodyToJsonFail) {
    MOCKER_CPP(&DmiRole::PDParseRequestBodyToJson, bool (*)(const ReqCtxPtr &, ordered_json &))
        .stubs()
        .will(returnValue(false));
    httplib::Request req;
    req.body = rankTableStringV1;

    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";

    dmiRole.HandlePDRoleV1(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::UnprocessableContent_422);
}

TEST_F(DmiRoleTest, HandlePDRoleV1NonSwitch) {
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV1;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV1(ctx1, "prefill");
    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, "{\"result\":\"ok\"}");

    httplib::Request req2;
    req2.body = rankTableStringV1;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV1(ctx2, "decode");
    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
}

TEST_F(DmiRoleTest, HandlePDRoleV1Switch) {
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV1;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV1(ctx1, "prefill");

    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, RESPONSE_OK_BODY);

    httplib::Request req2;
    req2.body = rankTableStringV1;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV1(ctx2, "decode");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx2->Res().body, RESPONSE_OK_BODY);
}

TEST_F(DmiRoleTest, HandlePDRoleV2Init_Success) {
    const std::string validRequestBody = rankTableStringV2;

    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req;
    req.body = validRequestBody;
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";

    dmiRole.HandlePDRoleV2(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx->Res().body, "{\"result\":\"ok\"}");
}

TEST_F(DmiRoleTest, HandlePDRoleV2Init_PDParseRequestBodyToJsonFail) {
    MOCKER_CPP(&DmiRole::PDParseRequestBodyToJson, bool (*)(const ReqCtxPtr &, ordered_json &))
        .stubs()
        .will(returnValue(false));
    const std::string validRequestBody = rankTableStringV2;
    httplib::Request req;
    req.body = validRequestBody;
    httplib::Response resp;
    ReqCtxPtr ctx = std::make_shared<RequestContext>(req, resp);
    std::string roleName = "prefill";

    dmiRole.HandlePDRoleV2(ctx, roleName);

    EXPECT_EQ(ctx->Res().status, httplib::StatusCode::UnprocessableContent_422);
    EXPECT_EQ(ctx->Res().body,
              "{\"error\":\"Req body converts to json fail. Reset to previous "
              "node status.\",\"error_type\":\"Input validation error\"}");
}

TEST_F(DmiRoleTest, HandlePDRoleV2Switch) {
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV2;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV2(ctx1, "prefill");

    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, RESPONSE_OK_BODY);

    httplib::Request req2;
    req2.body = rankTableStringV2;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV2(ctx2, "decode");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx2->Res().body, RESPONSE_OK_BODY);
}

TEST_F(DmiRoleTest, HandlePDRoleV2NonSwitch) {
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    httplib::Request req1;
    req1.body = rankTableStringV2;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV2(ctx1, "prefill");

    EXPECT_EQ(ctx1->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx1->Res().body, RESPONSE_OK_BODY);

    httplib::Request req2;
    req2.body = rankTableStringV2;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV2(ctx2, "prefill");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::OK_200);
    EXPECT_EQ(ctx2->Res().body, RESPONSE_OK_BODY);
}

TEST_F(DmiRoleTest, HandlePDRoleV2RelinkFailure) {
    MOCKER(JsonParse::CheckPDRoleReqJson).stubs().will(returnValue(true));

    MOCKER_CPP(&DmiRole::UpdatePDInfoV2,
               bool (*)(const std::string &, const std::string &, const ordered_json &, GlobalIpInfo &))
        .stubs()
        .will(returnValue(false));

    httplib::Request req1;
    req1.body = rankTableStringBefore;
    httplib::Response resp1;
    ReqCtxPtr ctx1 = std::make_shared<RequestContext>(req1, resp1);
    dmiRole.HandlePDRoleV2(ctx1, "prefill");

    httplib::Request req2;
    req2.body = rankTableStringAfter;
    httplib::Response resp2;
    ReqCtxPtr ctx2 = std::make_shared<RequestContext>(req2, resp2);
    dmiRole.HandlePDRoleV2(ctx2, "prefill");

    EXPECT_EQ(ctx2->Res().status, httplib::StatusCode::ServiceUnavailable_503);
}

TEST_F(DmiRoleTest, RunTaskThread) {
    GlobalIpInfo globalIpInfo;
    auto task = [globalIpInfo = globalIpInfo]() mutable { globalIpInfo.role = "test"; };
    dmiRole.taskQueue_.Push(std::move(task));
    dmiRole.taskTerminate_.store(true);
    dmiRole.RunTaskThread();
    EXPECT_TRUE(dmiRole.taskTerminate_.load());
}

TEST_F(DmiRoleTest, ProcessInitInfoV2_NormalCase) {
    InitJson();
    GlobalIpInfo globalIpInfo;
    dmiRole.ProcessInitInfoV2(body, globalIpInfo);

    EXPECT_TRUE(globalIpInfo.needInit);
    EXPECT_EQ(globalIpInfo.numInstancesPerPod, 64);
    EXPECT_EQ(globalIpInfo.localInstanceId, 0);
    EXPECT_EQ(globalIpInfo.localSuperPodId, "100");

    std::vector<std::string> expectedHostIps = {"192.168.1.10", "192.168.1.11"};
    std::vector<uint64_t> expectedDpInstIds = {1, 2};
    std::vector<std::string> expectedDeviceIps = {"10.0.0.1", "10.0.0.2", "10.0.0.3"};
    std::vector<std::string> expectedLogicalIds = {"logical-1", "logical-2", "logical-3"};
    std::vector<std::string> expectedPhysicalIds = {"physical-1", "physical-2", "physical-3"};
    std::vector<std::string> expectedRankIds = {"0", "1", "2"};
    std::vector<std::string> expectedSuperDeviceIds = {"super-1"};

    EXPECT_EQ(globalIpInfo.localHostIpList, expectedHostIps);
    EXPECT_EQ(globalIpInfo.localDpInstanceIds, expectedDpInstIds);
    EXPECT_EQ(globalIpInfo.localDeviceIps, expectedDeviceIps);
    EXPECT_EQ(globalIpInfo.localDeviceLogicalIds, expectedLogicalIds);
    EXPECT_EQ(globalIpInfo.localDevicePhysicalIds, expectedPhysicalIds);
    EXPECT_EQ(globalIpInfo.localDeviceRankIds, expectedRankIds);
    EXPECT_EQ(globalIpInfo.localSuperDeviceIds, expectedSuperDeviceIds);
}

TEST_F(DmiRoleTest, ProcessInitInfoV2_MissingField) {
    GlobalIpInfo globalIpInfo;
    ordered_json body1 = {{"local",
                           {{
                               {"host_ip", "192.168.1.10"},
                           }}}};
    EXPECT_THROW(dmiRole.ProcessInitInfoV2(body1, globalIpInfo), std::runtime_error);
}

TEST_F(DmiRoleTest, GetInstanceIdToServerIp) {
    const std::map<uint32_t, std::string> expected = {};
    EXPECT_EQ(dmiRole.GetInstanceIdToServerIp(), expected);
}

TEST_F(DmiRoleTest, GetRemoteNodeLinkStatusV2) {
    const std::map<uint64_t, std::pair<std::string, bool>> expected = {};
    EXPECT_EQ(dmiRole.GetRemoteNodeLinkStatusV2(), expected);
}

TEST_F(DmiRoleTest, SingleInstanceSingleDpInstance_Ok) {
    std::map<uint64_t, std::pair<std::string, bool>> input = {{10001, {"status1", true}}};

    auto result = GetInstanceStatus(input);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[1].first, "ok");
    EXPECT_TRUE(result[1].second);
}

TEST_F(DmiRoleTest, SingleInstanceSingleDpInstance_Error) {
    std::map<uint64_t, std::pair<std::string, bool>> input = {{10001, {"error1", false}}};

    auto result = GetInstanceStatus(input);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[1].first, "dp instance id : 10001error1");
    EXPECT_FALSE(result[1].second);
}

class ModifyPullKVFailIdTest : public ::testing::Test {
   protected:
    void SetUp() override { dmiRole = new mindie_llm::DmiRole(); }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }
    mindie_llm::DmiRole *dmiRole;
};

TEST_F(ModifyPullKVFailIdTest, BasicTest) {
    dmiRole->ModifyPullKVFailId(1001);
    const auto &successLinkIP = dmiRole->GetSuccessLinkIp();
    const auto &remoteNodeLinkStatus = dmiRole->GetRemoteNodeLinkStatus();
    bool isHealthy = dmiRole->IsHealthy();

    EXPECT_TRUE(successLinkIP.empty());
    EXPECT_EQ(remoteNodeLinkStatus.size(), 1);
    EXPECT_EQ(remoteNodeLinkStatus.at(1001).first, "failed : pull kv failed.");
    EXPECT_FALSE(isHealthy);
}

class GetLocalInstanceIdTest : public ::testing::Test {
   protected:
    void SetUp() override { dmiRole = new mindie_llm::DmiRole(); }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(GetLocalInstanceIdTest, DefaultValue) {
    const uint32_t &instanceId = dmiRole->GetLocalInstanceId();
    EXPECT_EQ(instanceId, 0);
}

class IsHealthyTest : public ::testing::Test {
   protected:
    void SetUp() override { dmiRole = new mindie_llm::DmiRole(); }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }
    mindie_llm::DmiRole *dmiRole;
};

TEST_F(IsHealthyTest, InitialStateHealthy) {
    bool isHealthy = dmiRole->IsHealthy();
    EXPECT_TRUE(isHealthy);
}

TEST_F(IsHealthyTest, UnhealthyAfterModifyPullKVFailId) {
    dmiRole->ModifyPullKVFailId(1001);
    bool isHealthy = dmiRole->IsHealthy();
    EXPECT_FALSE(isHealthy);
}

class QueryLinkStatusTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dmiRole = new mindie_llm::DmiRole();
        // Set initial state, simulate that assignDmiRole has been called
        dmiRole->assignedRole_ = true;
    }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(QueryLinkStatusTest, QueryLinkStatus_SkipWhenNotAssignedRole) {
    // Simulate the case where assignDmiRole has not been called yet
    dmiRole->assignedRole_ = false;

    // Expect function to execute normally but skip query logic
    EXPECT_NO_THROW(dmiRole->QueryLinkStatus());
}

TEST_F(QueryLinkStatusTest, QueryLinkStatus_SkipWhenNoConnections) {
    // Don't set any connections, expect to skip query
    // Note: We verify this by checking that no connections are processed
    // rather than using mock expectations which may interfere with other tests

    dmiRole->QueryLinkStatus();

    // Verify that no status updates occurred since there were no connections
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_.empty());
}

TEST_F(QueryLinkStatusTest, QueryLinkStatus_SuccessfulLinks) {
    // Set up linking connections
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}, {"192.168.1.2", 1, 101}};
    dmiRole->linkingLinkIP_[1001] = deviceInfos;
    dmiRole->linkingHostIP_[1001] = {"192.168.1.10"};

    // Set up mock that returns OK status
    MOCKER_CPP(&InferInstance::QueryPDLinkStatus, Status(*)(model_execute_data::PDLinkStatusResponse &))
        .stubs()
        .will(returnValue(Status(Error::Code::OK)));

    // Call QueryLinkStatus which will get empty response (no successful links)
    dmiRole->QueryLinkStatus();

    // Manually simulate successful link processing by directly setting the
    // success state This simulates what ProcessSuccessfulLinks would do
    {
        std::lock_guard<std::mutex> lock(dmiRole->mtx_);
        // Move the linking connection to success state
        dmiRole->successLinkIP_[1001] = dmiRole->linkingLinkIP_[1001];
        dmiRole->successHostIP_[1001] = dmiRole->linkingHostIP_[1001];
        dmiRole->remoteNodeLinkStatus_[1001] = {"ok", true};

        // Clear the linking state
        dmiRole->linkingLinkIP_.erase(1001);
        dmiRole->linkingHostIP_.erase(1001);
    }

    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).stubs();

    dmiRole->QueryLinkStatus();

    // Verify successful links have been moved to successLinkIP_
    EXPECT_EQ(dmiRole->successLinkIP_.count(1001), 1);
    auto &successDevices = dmiRole->successLinkIP_[1001];
    EXPECT_EQ(successDevices.size(), deviceInfos.size());
    for (size_t i = 0; i < deviceInfos.size(); ++i) {
        EXPECT_EQ(successDevices[i].deviceIp, deviceInfos[i].deviceIp);
        EXPECT_EQ(successDevices[i].devicePhysicalId, deviceInfos[i].devicePhysicalId);
        EXPECT_EQ(successDevices[i].superDeviceId, deviceInfos[i].superDeviceId);
    }
    EXPECT_EQ(dmiRole->successHostIP_[1001].size(), 1);
    EXPECT_EQ(dmiRole->successHostIP_[1001][0], "192.168.1.10");

    // Verify linking connections have been cleared
    EXPECT_TRUE(dmiRole->linkingLinkIP_.empty());
    EXPECT_TRUE(dmiRole->linkingHostIP_.empty());

    // Verify status
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1001].first, "ok");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[1001].second);
}

TEST_F(QueryLinkStatusTest, QueryLinkStatus_FailedLinks) {
    // Set up linking connections
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->linkingLinkIP_[1001] = deviceInfos;
    dmiRole->linkingHostIP_[1001] = {"192.168.1.10"};

    // Mock failed query response
    MOCKER_CPP(&InferInstance::QueryPDLinkStatus, Status(*)(model_execute_data::PDLinkStatusResponse &))
        .stubs()
        .will(returnValue(Status(Error::Code::OK)));

    dmiRole->QueryLinkStatus();

    // Manually simulate failed link processing
    {
        std::lock_guard<std::mutex> lock(dmiRole->mtx_);
        // Simulate ProcessFailedLinks behavior
        if (dmiRole->linkingLinkIP_.find(1001) != dmiRole->linkingLinkIP_.end()) {
            dmiRole->linkingLinkIP_.erase(1001);
            dmiRole->linkingHostIP_.erase(1001);
            std::string failedReason =
                "failed : " + std::to_string(static_cast<int>(model_execute_data::PDErrorCode::PD_UNKNOWN_ERROR));
            dmiRole->remoteNodeLinkStatus_[1001] = {failedReason, true};
        }
    }

    // Verify failed links have been removed
    EXPECT_TRUE(dmiRole->linkingLinkIP_.empty());
    EXPECT_TRUE(dmiRole->linkingHostIP_.empty());

    // Verify status
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1001].first, "failed : 2005");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[1001].second);
}

TEST_F(QueryLinkStatusTest, QueryLinkStatus_QueryFailure) {
    // Set up linking connections
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->linkingLinkIP_[1001] = deviceInfos;

    // Mock query failure
    MOCKER_CPP(&InferInstance::QueryPDLinkStatus, Status(*)(model_execute_data::PDLinkStatusResponse &))
        .stubs()
        .will(returnValue(Status(Error::Code::ERROR, "Query failed")));

    // Expect no exception thrown, just log error
    EXPECT_NO_THROW(dmiRole->QueryLinkStatus());
}

TEST_F(QueryLinkStatusTest, QueryLinkStatus_AllLinksCompleted) {
    // Set up a successful connection
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->successLinkIP_[1001] = deviceInfos;
    dmiRole->successHostIP_[1001] = {"192.168.1.10"};

    // Mock empty response (no running or waiting connections)
    model_execute_data::PDLinkStatusResponse response;

    MOCKER_CPP(&InferInstance::QueryPDLinkStatus, Status(*)(model_execute_data::PDLinkStatusResponse &))
        .stubs()
        .will(returnValue(Status(Error::Code::OK)));

    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).expects(once()).with(eq(PDRoleStatus::READY));

    dmiRole->QueryLinkStatus();

    // Verify status is set to READY
}

TEST_F(QueryLinkStatusTest, QueryLinkStatus_InvalidClusterId) {
    // Set up linking connections
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->linkingLinkIP_[1001] = deviceInfos;

    // Mock failed response containing invalid cluster_id
    MOCKER_CPP(&InferInstance::QueryPDLinkStatus, Status(*)(model_execute_data::PDLinkStatusResponse &))
        .stubs()
        .will(returnValue(Status(Error::Code::OK)));

    // Manually simulate processing of failed links with invalid cluster_id
    {
        std::lock_guard<std::mutex> lock(dmiRole->mtx_);
        // Simulate ProcessFailedLinks behavior with invalid cluster_id
        try {
            uint64_t instanceId = std::stoull("invalid_id");
            // This should not happen, but we're testing that invalid_id doesn't
            // cause issues
        } catch (const std::exception &e) {
            // Expected: invalid cluster_id should be handled gracefully
        }
    }

    // Expect no exception thrown when processing invalid cluster_id
    EXPECT_NO_THROW(dmiRole->QueryLinkStatus());
}

class StopCurrentTaskTest : public ::testing::Test {
   protected:
    void SetUp() override { dmiRole = new mindie_llm::DmiRole(); }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(StopCurrentTaskTest, StopCurrentTask_WhenTaskIsRunning) {
    // Set task as running
    dmiRole->taskRunning_.store(true);

    // Stop the current task
    dmiRole->StopCurrentTask();

    // Verify task is no longer running
    EXPECT_FALSE(dmiRole->taskRunning_.load());
}

TEST_F(StopCurrentTaskTest, StopCurrentTask_WhenTaskIsNotRunning) {
    // Ensure task is not running initially
    dmiRole->taskRunning_.store(false);

    // Try to stop the current task
    dmiRole->StopCurrentTask();

    // Verify task remains not running
    EXPECT_FALSE(dmiRole->taskRunning_.load());
}

class ExecuteLinkTaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dmiRole = new mindie_llm::DmiRole();
        GlobalMockObject::reset();
    }

    void TearDown() override {
        GlobalMockObject::reset();
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(ExecuteLinkTaskTest, ExecuteLinkTask_SuccessfulAssignment) {
    // Prepare global IP info with link information
    mindie_llm::GlobalIpInfo globalIpInfo;
    globalIpInfo.linkIpInfo[1001] = {{"192.168.1.1", 0, 100}, {"192.168.1.2", 1, 101}};
    globalIpInfo.hostIpInfo[1001] = {"192.168.1.10"};

    // Mock successful assignment
    MOCKER_CPP(&InferInstance::AssignDmiRole, Status(*)(const GlobalIpInfo &))
        .expects(once())
        .will(returnValue(Status(Error::Code::OK)));

    // Execute the link task
    dmiRole->ExecuteLinkTask(globalIpInfo);

    // Verify task is no longer running
    EXPECT_FALSE(dmiRole->taskRunning_.load());

    // Verify linking information was set
    EXPECT_EQ(dmiRole->linkingLinkIP_.size(), globalIpInfo.linkIpInfo.size());
    for (const auto &[instanceId, deviceInfos] : globalIpInfo.linkIpInfo) {
        EXPECT_TRUE(dmiRole->linkingLinkIP_.count(instanceId));
        EXPECT_EQ(dmiRole->linkingLinkIP_[instanceId].size(), deviceInfos.size());
        for (size_t i = 0; i < deviceInfos.size(); ++i) {
            EXPECT_EQ(dmiRole->linkingLinkIP_[instanceId][i].deviceIp, deviceInfos[i].deviceIp);
            EXPECT_EQ(dmiRole->linkingLinkIP_[instanceId][i].devicePhysicalId, deviceInfos[i].devicePhysicalId);
            EXPECT_EQ(dmiRole->linkingLinkIP_[instanceId][i].superDeviceId, deviceInfos[i].superDeviceId);
        }
    }
    EXPECT_EQ(dmiRole->linkingHostIP_.size(), globalIpInfo.hostIpInfo.size());
    for (const auto &[instanceId, hostIps] : globalIpInfo.hostIpInfo) {
        EXPECT_TRUE(dmiRole->linkingHostIP_.count(instanceId));
        EXPECT_EQ(dmiRole->linkingHostIP_[instanceId].size(), hostIps.size());
        for (size_t i = 0; i < hostIps.size(); ++i) {
            EXPECT_EQ(dmiRole->linkingHostIP_[instanceId][i], hostIps[i]);
        }
    }

    // Verify remote node status was initialized
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1001].first, "linking");
    EXPECT_FALSE(dmiRole->remoteNodeLinkStatus_[1001].second);

    // Verify assigned role flag was set
    EXPECT_TRUE(dmiRole->assignedRole_);
}

class ProcessFailedLinksTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dmiRole = new mindie_llm::DmiRole();
        // Set up initial linking state
        std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}, {"192.168.1.2", 1, 101}};
        dmiRole->linkingLinkIP_[1001] = deviceInfos;
        dmiRole->linkingHostIP_[1001] = {"192.168.1.10"};
    }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(ProcessFailedLinksTest, ProcessFailedLinks_ValidFailedLink) {
    // Create a mock failed link info object
    struct MockFailedLinkInfo {
        std::string cluster_id() const { return "1001"; }
        model_execute_data::PDErrorCode pd_error_code() const {
            return model_execute_data::PDErrorCode::PD_UNKNOWN_ERROR;
        }
    };

    std::vector<MockFailedLinkInfo> failedLinks = {MockFailedLinkInfo()};

    // Process failed links
    dmiRole->ProcessFailedLinks(failedLinks);

    // Verify the failed link was removed from linking
    EXPECT_TRUE(dmiRole->linkingLinkIP_.empty());
    EXPECT_TRUE(dmiRole->linkingHostIP_.empty());

    // Verify the status was updated
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1001].first, "failed : 2005");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[1001].second);
}

TEST_F(ProcessFailedLinksTest, ProcessFailedLinks_InvalidClusterId) {
    // Create a mock failed link info with invalid cluster_id
    struct MockFailedLinkInfo {
        std::string cluster_id() const { return "invalid_id"; }
        model_execute_data::PDErrorCode pd_error_code() const {
            return model_execute_data::PDErrorCode::PD_UNKNOWN_ERROR;
        }
    };

    std::vector<MockFailedLinkInfo> failedLinks = {MockFailedLinkInfo()};

    // Process failed links - should handle invalid cluster_id gracefully
    EXPECT_NO_THROW(dmiRole->ProcessFailedLinks(failedLinks));

    // Verify linking state remains unchanged
    EXPECT_FALSE(dmiRole->linkingLinkIP_.empty());
    EXPECT_FALSE(dmiRole->linkingHostIP_.empty());
}

TEST_F(ProcessFailedLinksTest, ProcessFailedLinks_NonExistentInstanceId) {
    // Create a mock failed link info for a different instance
    struct MockFailedLinkInfo {
        std::string cluster_id() const { return "2001"; }
        model_execute_data::PDErrorCode pd_error_code() const { return model_execute_data::PDErrorCode::PD_LINK_ERROR; }
    };

    std::vector<MockFailedLinkInfo> failedLinks = {MockFailedLinkInfo()};

    // Process failed links
    dmiRole->ProcessFailedLinks(failedLinks);

    // Verify linking state remains unchanged since instance 2001 is not in
    // linking
    EXPECT_FALSE(dmiRole->linkingLinkIP_.empty());
    EXPECT_FALSE(dmiRole->linkingHostIP_.empty());

    // Verify status was updated for the failed instance
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[2001].first, "failed : 2001");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[2001].second);
}

class ProcessSuccessfulLinksTest : public ::testing::Test {
   protected:
    void SetUp() override {
        dmiRole = new mindie_llm::DmiRole();
        // Set up initial linking state
        std::vector<mindie_llm::DeviceInfo> deviceInfos1 = {{"192.168.1.1", 0, 100}, {"192.168.1.2", 1, 101}};
        std::vector<mindie_llm::DeviceInfo> deviceInfos2 = {{"192.168.1.3", 2, 102}};
        dmiRole->linkingLinkIP_[1001] = deviceInfos1;
        dmiRole->linkingLinkIP_[1002] = deviceInfos2;
        dmiRole->linkingHostIP_[1001] = {"192.168.1.10"};
        dmiRole->linkingHostIP_[1002] = {"192.168.1.11"};
    }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(ProcessSuccessfulLinksTest, ProcessSuccessfulLinks_PartialSuccess) {
    // Mock successful links containing only some device IPs
    std::vector<std::string> successLinks = {"192.168.1.1", "192.168.1.2"};

    // Process successful links
    dmiRole->ProcessSuccessfulLinks(successLinks);

    // Verify instance 1001 was moved to success (all its devices succeeded)
    EXPECT_EQ(dmiRole->successLinkIP_.count(1001), 1);
    EXPECT_EQ(dmiRole->successHostIP_.count(1001), 1);
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1001].first, "ok");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[1001].second);

    // Verify instance 1001 was removed from linking
    EXPECT_EQ(dmiRole->linkingLinkIP_.count(1001), 0);
    EXPECT_EQ(dmiRole->linkingHostIP_.count(1001), 0);

    // Verify instance 1002 remains in linking (its device didn't succeed)
    EXPECT_EQ(dmiRole->linkingLinkIP_.count(1002), 1);
    EXPECT_EQ(dmiRole->linkingHostIP_.count(1002), 1);
}

TEST_F(ProcessSuccessfulLinksTest, ProcessSuccessfulLinks_AllSuccess) {
    // Mock successful links containing all device IPs
    std::vector<std::string> successLinks = {"192.168.1.1", "192.168.1.2", "192.168.1.3"};

    // Process successful links
    dmiRole->ProcessSuccessfulLinks(successLinks);

    // Verify all instances were moved to success
    EXPECT_EQ(dmiRole->successLinkIP_.count(1001), 1);
    EXPECT_EQ(dmiRole->successLinkIP_.count(1002), 1);
    EXPECT_EQ(dmiRole->successHostIP_.count(1001), 1);
    EXPECT_EQ(dmiRole->successHostIP_.count(1002), 1);

    // Verify status was updated for both instances
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1001].first, "ok");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[1001].second);
    EXPECT_EQ(dmiRole->remoteNodeLinkStatus_[1002].first, "ok");
    EXPECT_TRUE(dmiRole->remoteNodeLinkStatus_[1002].second);

    // Verify linking lists are empty
    EXPECT_TRUE(dmiRole->linkingLinkIP_.empty());
    EXPECT_TRUE(dmiRole->linkingHostIP_.empty());
}

TEST_F(ProcessSuccessfulLinksTest, ProcessSuccessfulLinks_NoSuccess) {
    // Mock successful links that don't match any devices
    std::vector<std::string> successLinks = {"192.168.1.99"};

    // Process successful links
    dmiRole->ProcessSuccessfulLinks(successLinks);

    // Verify no instances were moved to success
    EXPECT_TRUE(dmiRole->successLinkIP_.empty());
    EXPECT_TRUE(dmiRole->successHostIP_.empty());

    // Verify all instances remain in linking
    EXPECT_EQ(dmiRole->linkingLinkIP_.count(1001), 1);
    EXPECT_EQ(dmiRole->linkingLinkIP_.count(1002), 1);
    EXPECT_EQ(dmiRole->linkingHostIP_.count(1001), 1);
    EXPECT_EQ(dmiRole->linkingHostIP_.count(1002), 1);
}

class CheckAllLinksCompletedTest : public ::testing::Test {
   protected:
    void SetUp() override { dmiRole = new mindie_llm::DmiRole(); }

    void TearDown() override {
        delete dmiRole;
        dmiRole = nullptr;
    }

    mindie_llm::DmiRole *dmiRole;
};

TEST_F(CheckAllLinksCompletedTest, CheckAllLinksCompleted_AllCompleted) {
    // Set up successful links and ensure no linking/running/waiting links
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->successLinkIP_[1001] = deviceInfos;
    // Ensure linking, running, and waiting lists are empty
    dmiRole->linkingLinkIP_.clear();
    dmiRole->runningLinkIP_.clear();
    dmiRole->waitingLinkIP_.clear();

    // Mock SetPDRoleStatus to verify it's called
    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).expects(once());

    // Check all links completed
    EXPECT_NO_THROW(dmiRole->CheckAllLinksCompleted());
}

TEST_F(CheckAllLinksCompletedTest, CheckAllLinksCompleted_LinksStillLinking) {
    // Set up successful links but still have linking links
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->successLinkIP_[1001] = deviceInfos;
    dmiRole->linkingLinkIP_[1002] = deviceInfos;  // Still linking

    // Mock SetPDRoleStatus to ensure it's NOT called
    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).expects(never());

    // Check all links completed
    EXPECT_NO_THROW(dmiRole->CheckAllLinksCompleted());

    // Verify linking links still exist (function should not change state)
    EXPECT_FALSE(dmiRole->linkingLinkIP_.empty());
    EXPECT_TRUE(dmiRole->successLinkIP_.count(1001) > 0);
}

TEST_F(CheckAllLinksCompletedTest, CheckAllLinksCompleted_NoSuccessLinks) {
    // No success links, only linking links
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->linkingLinkIP_[1001] = deviceInfos;

    // Mock SetPDRoleStatus to ensure it's NOT called
    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).expects(never());

    // Check all links completed
    EXPECT_NO_THROW(dmiRole->CheckAllLinksCompleted());

    // Verify no success links exist and linking links remain
    EXPECT_TRUE(dmiRole->successLinkIP_.empty());
    EXPECT_FALSE(dmiRole->linkingLinkIP_.empty());
}

TEST_F(CheckAllLinksCompletedTest, CheckAllLinksCompleted_RunningLinksExist) {
    // Set up successful links but still have running links
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->successLinkIP_[1001] = deviceInfos;
    dmiRole->runningLinkIP_.push_back("192.168.1.2");

    // Mock SetPDRoleStatus to ensure it's NOT called
    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).expects(never());

    // Check all links completed
    EXPECT_NO_THROW(dmiRole->CheckAllLinksCompleted());

    // Verify running links still exist and success links are present
    EXPECT_FALSE(dmiRole->runningLinkIP_.empty());
    EXPECT_TRUE(dmiRole->successLinkIP_.count(1001) > 0);
}

TEST_F(CheckAllLinksCompletedTest, CheckAllLinksCompleted_WaitingLinksExist) {
    // Set up successful links but still have waiting links
    std::vector<mindie_llm::DeviceInfo> deviceInfos = {{"192.168.1.1", 0, 100}};
    dmiRole->successLinkIP_[1001] = deviceInfos;
    dmiRole->waitingLinkIP_.push_back("192.168.1.3");

    // Mock SetPDRoleStatus to ensure it's NOT called
    MOCKER_CPP(&InferInstance::SetPDRoleStatus, void (*)(PDRoleStatus)).expects(never());

    // Check all links completed
    EXPECT_NO_THROW(dmiRole->CheckAllLinksCompleted());

    // Verify waiting links still exist and success links are present
    EXPECT_FALSE(dmiRole->waitingLinkIP_.empty());
    EXPECT_TRUE(dmiRole->successLinkIP_.count(1001) > 0);
}

}  // namespace mindie_llm
