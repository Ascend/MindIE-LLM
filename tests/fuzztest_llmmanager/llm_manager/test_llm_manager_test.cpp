/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
 
#include <string>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <thread>
#include <iostream>
#include <memory>
#include "file_system.h"
#include "log.h"
#include "secodeFuzz.h"
#include "infer_request.h"
#include "pd_role.h"
#include "infer_request_inner.h"
#include "common_util.h"
#include "config_manager.h"
#include "error.h"
#include "file_utils.h"
#include "ibis_scheduler.h"
#include "infer_request_impl.h"
#include "llm_infer_engine.h"
#include "llm_infer_model.h"
#include "llm_infer_model_instance.h"
#include "llm_infer_response.h"
#include "mindie_llm/file_system.h"
#include "mindie_llm/log.h"
#include "model_backend.h"
#include "param_checker.h"
#include "secodeFuzz.h"
#include "status.h"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <thread>
namespace mindie_llm {
const int BACKEND_REQUEST_BOUND = 10000;
const int VOCAB_SIZE_BOUND = 1000;
const int REQUEST_SIZE_BOUND = 1000;
const int LOGEVEL_NUM = 5;
const int MAX_LOGFILE_SIZE = 20;
const int MAX_MODELINSTANCE_NUMBER = 11;
const int MAX_TOKENIZERPROCEENUMBER = 32;
const int MIN_MULTINODEINFERPORT = 1024;
const int MAX_MULTINODEINFERPORT = 65535;
const int MAX_MAXSEQLEN = 65535;
const int MAX_INPUTTOKENLEN = 65535;
const int MAX_WORLD_SIZE = 65535;
const int MAX_CPU_MEM_SIZE = 65535;
const int MAX_NPU_MEM_SIZE = 65535;

const int MAX_CACHEBLOCKSIZE = 129;
const int MAX_BATCHSIZE = 5001;
const int MAX_PREFILL_TOKENS = 512001;
const int MAX_PREFILLTIMESMSPERREQ = 1001;
const int MAX_PREFILLPOLICYTYPE = 3;
const int MAXQUEUEDELAYMICROSECONDS = 1000001;
const int DEFAULTCACHEBLOCKSIZE = 128;
const int DEFAULTMAXPREFILLBATCHSIZE = 50;
const int DEFAULTMAXPREFILLTOKENS = 8192;
const int DEFAULTPREFILLTIMEMSPERREQ = 150;
const int DEFAULTMAXBATCHSIZE = 200;
const int DEFAULTMAXITERTIMES = 512;
const int DEFAULTSPLITSTARTBATCHSIZE = 16;
const int DEFAULTQUEUEDELAYMICROSECONDS = 5000;
constexpr int JSONDATA_DUMP = 4;
enum class PDRole {
    UNKNOWN = 0,
    PREFILL = 1,
    DECODE = 2,
};
using IsFileValidStringFunc = bool (*)(const std::string &filePath, std::string &errMsg,
                                       const FileValidationParams &params);

void ConstructFuzzSchedulerConfig(Json &ScheduleJsonData, uint32_t &fuzzIndex)
{
    ScheduleJsonData["templateType"] = "Standard";
    ScheduleJsonData["templateName"] = "Standard_LLM";
    ScheduleJsonData["cacheBlockSize"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTCACHEBLOCKSIZE, 0, MAX_CACHEBLOCKSIZE);
    ScheduleJsonData["maxPrefillBatchSize"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXPREFILLBATCHSIZE, 1, MAX_BATCHSIZE);
    ScheduleJsonData["maxPrefillTokens"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXPREFILLTOKENS, 1, MAX_PREFILL_TOKENS);
    ScheduleJsonData["prefillTimeMsPerReq"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTPREFILLTIMEMSPERREQ, 1, MAX_PREFILLTIMESMSPERREQ);
    ScheduleJsonData["prefillPolicyType"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, MAX_PREFILLPOLICYTYPE);
    ScheduleJsonData["decodeTimeMsPerReq"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXPREFILLBATCHSIZE, 1, MAX_PREFILLTIMESMSPERREQ);
    ScheduleJsonData["decodePolicyType"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, MAX_PREFILLPOLICYTYPE);
    ScheduleJsonData["maxBatchSize"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXBATCHSIZE, 1, MAX_BATCHSIZE);
    ScheduleJsonData["maxIterTimes"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXITERTIMES, 1, MAX_MAXSEQLEN);
    ScheduleJsonData["maxPreemptCount"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_BATCHSIZE);
    ScheduleJsonData["supportSelectBatch"] = true;
    ScheduleJsonData["maxQueueDelayMicroseconds"] = *(int *)DT_SetGetNumberRange(
        &g_Element[fuzzIndex++], DEFAULTQUEUEDELAYMICROSECONDS, 1, MAXQUEUEDELAYMICROSECONDS);
    ScheduleJsonData["policyType"] = 0;
    ScheduleJsonData["splitType"] = false;
    ScheduleJsonData["splitStartType"] = false;
    ScheduleJsonData["splitChunkTokens"] = DEFAULTMAXITERTIMES;
    ScheduleJsonData["splitStartBatchSize"] = DEFAULTSPLITSTARTBATCHSIZE;
    ScheduleJsonData["enablePrefixCache"] = false;
    ScheduleJsonData["enableSplit"] = false;
    ScheduleJsonData["dpScheduling"] = false;
    ScheduleJsonData["prefillExpectedTime"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXPREFILLBATCHSIZE, 1, MAX_PREFILLTIMESMSPERREQ);
    ;
    ScheduleJsonData["decodeExpectedTime"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], DEFAULTMAXPREFILLBATCHSIZE, 1, MAX_PREFILLTIMESMSPERREQ);
    ;
    ScheduleJsonData["bufferResponseEnabled"] = false;
    ScheduleJsonData["asyncInferEnable"] = false;
    ScheduleJsonData["distributedEnable"] = false;
}

void ConstructBackendFuzzConfig(Json &jsonData, uint32_t &fuzzIndex)
{
    jsonData["Version"] = "1.0.0";
    std::vector<std::string> logLevelEnumTable = {"warning", "Error", "Warning", "Info", "Verbose"};
    jsonData["LogConfig"]["logLevel"] =
        logLevelEnumTable[*(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 0, LOGEVEL_NUM - 1)];
    jsonData["LogConfig"]["logFileSize"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_LOGFILE_SIZE);
    jsonData["LogConfig"]["logFileNum"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_LOGFILE_SIZE);
    jsonData["LogConfig"]["logPath"] = "logs/mindservice.log";
    jsonData["BackendConfig"]["backendName"] = "mindieservice_llm_engine";
    jsonData["BackendConfig"]["modelInstanceNumber"] = 1;
    jsonData["BackendConfig"]["npuDeviceIds"] = {{0}};
    jsonData["BackendConfig"]["tokenizerProcessNumber"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_TOKENIZERPROCEENUMBER);
    jsonData["BackendConfig"]["multiNodesInferEnabled"] = true;
    jsonData["BackendConfig"]["multiNodesInferPort"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1024, // 1024 is the minimum port number
                                     MIN_MULTINODEINFERPORT, MAX_MULTINODEINFERPORT);
    jsonData["BackendConfig"]["interNodeTLSEnabled"] = true;
    jsonData["BackendConfig"]["interNodeTlsCaPath"] = "security/grpc/ca/";
    jsonData["BackendConfig"]["interNodeTlsCaFiles"] = {"ca.pem"};
    jsonData["BackendConfig"]["interNodeTlsCert"] = "security/grpc/certs/server.pem";
    jsonData["BackendConfig"]["interNodeTlsPk"] = "security/grpc/keys/server.key.pem";
    jsonData["BackendConfig"]["interNodeTlsCrlPath"] = "security/grpc/certs/";
    jsonData["BackendConfig"]["interNodeTlsCrlFiles"] = {"server_crl.pem"};
}
// 构造config.json
void ConstructFuzzConfig(uint32_t &fuzzIndex, std::string &configPath)
{
    std::ofstream file(configPath);
    if (!file.is_open()) {
        return;
    }
    Json jsonData;
    ConstructBackendFuzzConfig(jsonData, fuzzIndex);
    Json ModelDeployConfigData;
    ModelDeployConfigData["maxSeqLen"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_MAXSEQLEN);
    ModelDeployConfigData["maxInputTokenLen"] =
        *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_INPUTTOKENLEN);
    ModelDeployConfigData["truncation"] = 0;
    Json modelJsonData;
    modelJsonData["modelInstanceType"] = "StandardMock";
    modelJsonData["modelName"] = "llama_65b";
    std::string homePath;
    mindie_llm::GetLlmPath(homePath);
    modelJsonData["modelWeightPath"] = homePath + "/mock/Qwen2.5-7B-Instruct";
    modelJsonData["worldSize"] = 1;
    modelJsonData["cpuMemSize"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, MAX_CPU_MEM_SIZE);
    modelJsonData["npuMemSize"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], -1, -1, MAX_NPU_MEM_SIZE);
    modelJsonData["backendType"] = "ms";
    modelJsonData["trustRemoteCode"] = false;
    modelJsonData["vocabsize"] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 0, MAX_MAXSEQLEN);
    ModelDeployConfigData["ModelConfig"] = std::vector<Json>{modelJsonData};
    Json LoraModulesJsonData;
    LoraModulesJsonData["name"] = "test";
    LoraModulesJsonData["path"] = "test";
    LoraModulesJsonData["baseModelName"] = "test";
    ModelDeployConfigData["LoraModules"] = std::vector<Json>{LoraModulesJsonData};
    jsonData["BackendConfig"]["ModelDeployConfig"] = ModelDeployConfigData;
    Json ScheduleJsonData;
    ConstructFuzzSchedulerConfig(ScheduleJsonData, fuzzIndex);
    jsonData["BackendConfig"]["ScheduleConfig"] = ScheduleJsonData;

    // 输出JSON
    file << jsonData;
    file.close();
}


/**
 * 请求回调
 * @param response
 */
void ResponseCallback(std::shared_ptr<SimpleLLMInference::InferenceResponse> &response)
{
    auto reqId = response->response.reqId;
    if (response->IsEOS()) {
        std::cout << "ReqId:" << reqId << " Finished" << std::endl;
    }
}

void SendRequest(SimpleLLMInference::InferenceEngine &engine, uint32_t &fuzzIndex)
{
    uint64_t requestId = *(u64 *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, REQUEST_SIZE_BOUND);
    std::shared_ptr<InferRequest> request = std::make_shared<InferRequest>(InferRequestId(requestId));
    std::string testName = "INPUT_IDS";
    mindie_llm::InferDataType dataType = mindie_llm::InferDataType::TYPE_INT64;
    std::vector<int64_t> shape({1, 1});
    auto testTensor = std::make_shared<mindie_llm::InferTensor>(testName, dataType, shape);
    int64_t *data = (int64_t *)malloc(1 * sizeof(int64_t));
    if (data == nullptr) {
        return;
    }
    data[0] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, VOCAB_SIZE_BOUND);
    testTensor->SetBuffer(data, sizeof(int64_t), true);

    std::string topKName = "TOP_K";
    dataType = mindie_llm::InferDataType::TYPE_INT32;
    std::vector<int64_t> shapeTopK({1, 1});
    auto topKTensor = std::make_shared<mindie_llm::InferTensor>(topKName, dataType, shapeTopK);
    int32_t *dataTopK = (int32_t *)malloc(1 * sizeof(int32_t));
    if (dataTopK == nullptr) {
        return;
    }
    dataTopK[0] = 0;
    topKTensor->SetBuffer(dataTopK, sizeof(int32_t), true);
    request->AddTensor(testName, testTensor);
    request->AddTensor(topKName, topKTensor);
    request->SetRecompute(false);
    std::cout << "SendRequest" << std::endl;
    engine.ForwardRequest(request);
    std::this_thread::sleep_for(std::chrono::milliseconds(4L));
    auto ControlRequest = std::make_pair(request->GetRequestId(), mindie_llm::OperationV2::STOP);
    engine.ForwardControlRequest(ControlRequest);
    std::this_thread::sleep_for(std::chrono::milliseconds(4L));
}

int MockInstanceInit(llm::backend::ModelBackend *modelBackend, const std::map<std::string, std::string> &modelConfig,
                     std::map<std::string, std::string> &initResults, bool isMultiNodesInfer)
{
    initResults["cpuBlockNum"] = "1";
    initResults["maxPositionEmbeddings"] = "1";
    initResults["status"] = "ok";
    MINDIE_LLM_LOG_INFO("MockInstanceInit success");
    return 0;
}

TEST(LlmManagerDTFuzz, Exists)
{
    std::srand(time(NULL));
    MINDIE_LLM_LOG_DEBUG("begin====================");
    std::string fuzzName = "LlmManagerDTFuzz";
    std::string homePath;
    mindie_llm::GetLlmPath(homePath);
    std::string configPath = homePath + "/conf/config.json";
    std::vector<std::string> sData = mindie_llm::Split(configPath, ',');
    MOCKER(IsFileValidStringFunc(FileUtils::IsFileValid)).stubs().will(returnValue(true));
    MOCKER_CPP(&llm::backend::ModelBackend::InstanceInit,
               int (*)(llm::backend::ModelBackend *modelBackend, const std::map<std::string, std::string> &,
                       std::map<std::string, std::string> &, bool))
        .stubs()
        .will(invoke(&MockInstanceInit));
    MOCKER_CPP(&llm::backend::ModelBackend::InstanceFinalize, int (*)(llm::backend::ModelBackend *modelBackend))
        .stubs()
        .will(returnValue(int(0)));
    IBISSCHEDULER_Error *nullptrError = nullptr;
    MOCKER(IBISSCHEDULER_Init).stubs().will(returnValue(nullptrError));
    MOCKER(IBISSCHEDULER_Enqueue).stubs().will(returnValue(nullptrError));
    MOCKER(IBISSCHEDULER_Finalize).stubs().will(returnValue(nullptrError));

    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        LlmInferEngine inferEngine;
        bool ret = inferEngine.Create(configPath);
        if (!ret) {
            std::cout << "engine create fail" << std::endl;
        }
        std::set<size_t> npuDeviceIds = {1};
        ret = inferEngine.Init(0, npuDeviceIds);
        if (!ret) {
            std::cout << "engine init fail" << std::endl;
        }
        uint32_t maxPositionEmbeddings = inferEngine.GetMaxPositionEmbeddings();
        std::cout << "maxPositionEmbeddings is " << std::to_string(maxPositionEmbeddings) << std::endl;
        Forward(inferEngine, fuzzIndex);
        inferEngine.Finalize();
    }
    auto &configManager = mindie_llm::ConfigManager::GetInstance(configPath);
    configManager.Release();
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, EngineInit)
{
    std::srand(time(NULL));
    MINDIE_LLM_LOG_DEBUG("begin====================");
    std::string fuzzName = "LlmManagerDTFuzz";
    std::string homePath;
    mindie_llm::GetLlmPath(homePath);
    std::string configPath = homePath + "/conf/confignew.json";
    ;
    MOCKER(IsFileValidStringFunc(FileUtils::IsFileValid)).stubs().will(returnValue(true));
    MOCKER(mindie_llm::ParamChecker::CheckPath).stubs().will(returnValue(true));
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        ConstructFuzzConfig(fuzzIndex, configPath);
        LlmInferEngine inferEngine;
        bool ret = inferEngine.Create(configPath);
        if (!ret) {
            std::cout << "engine create fail" << std::endl;
        }
        std::set<size_t> npuDeviceIds = {1};
        try {
            auto ret = inferEngine.Init(0, npuDeviceIds);
            if (!ret) {
                std::cout << "engine init failed:  " << std::endl;
                continue;
            }

            std::cout << "engine init success:  " << std::endl;
            inferEngine.Finalize();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        }
        auto &configManager = mindie_llm::ConfigManager::GetInstance(configPath);
        configManager.Release();
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, CommonUtilError)
{
    std::srand(time(NULL));
    std::string fuzzName = "CommonUtilError";
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        auto error = Error(Error::Code::OK);
        auto code = error.ErrorCode();
        auto ret = error.IsOk();

        auto str = error.ToString();
        auto msg = error.Message();
        error = Error(Error::Code::ERROR);
        str = error.ToString();
        error = Error(Error::Code::INVALID_ARG);
        str = error.ToString();
        error = Error(Error::Code::NOT_FOUND);
        str = error.ToString();

        auto status1 = Status(Error::Code::OK);
        auto retStatus1 = status1.IsOk();
        error = Error(Error::Code::OK);
        auto status2 = Status(error);
        auto statusCode = status2.StatusCode();
        status2 = Status(Error::Code::OK, "test");
        auto statusMsg = status2.StatusMsg();

    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, InitInferEngineForDistributed)
{
    std::srand(time(NULL));
    MINDIE_LLM_LOG_DEBUG("begin====================");
    std::string fuzzName = "LlmManagerDTFuzz:InitInferEngine";
    std::string homePath;
    mindie_llm::GetLlmPath(homePath);
    std::string configPath = homePath + "/conf/confignew.json";

    MOCKER(IsFileValidStringFunc(FileUtils::IsFileValid)).stubs().will(returnValue(true));
    MOCKER(mindie_llm::ParamChecker::CheckPath).stubs().will(returnValue(true));
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        ConstructFuzzConfig(fuzzIndex, configPath);
        // 设置配置
        std::ifstream inputFile(configPath);
        if (!inputFile.is_open()) {
            return;
        }
        Json jsonData;
        inputFile >> jsonData;
        inputFile.close();
        auto &schedulerConfigs = jsonData["BackendConfig"]["ScheduleConfig"];
        schedulerConfigs["distributedEnable"] = true;
        std::ofstream output_file(configPath);
        if (!output_file.is_open()) {
            return;
        }
        output_file << jsonData.dump(JSONDATA_DUMP);
        output_file.close();
        LlmInferEngine inferEngine;
        bool ret = inferEngine.Create(configPath);
        if (!ret) {
            std::cout << "engine create fail" << std::endl;
        }
        std::set<size_t> npuDeviceIds = {1};
        std::map<std::string, std::string> extendInfo;
        extendInfo["local_rank_ids"] = "1,2,3,4";
        try {
            auto ret = inferEngine.Init(0, npuDeviceIds, extendInfo);
            if (!ret) {
                std::cout << "engine init failed:  " << std::endl;
                continue;
            }
            std::cout << "engine init success:  " << std::endl;
            inferEngine.Finalize();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        }

        // 回退配置
        std::ifstream inputRecovFile(configPath);
        if (!inputRecovFile.is_open()) {
            return;
        }
        Json jsonRecovData;
        inputRecovFile >> jsonRecovData;
        inputRecovFile.close();
        auto &schedulerRecovConfigs = jsonData["BackendConfig"]["ScheduleConfig"];
        schedulerRecovConfigs.erase("distributedEnable");
        std::ofstream outputRecovfile(configPath);
        if (!outputRecovfile.is_open()) {
            return;
        }
        outputRecovfile << jsonData.dump(JSONDATA_DUMP);
        outputRecovfile.close();
    }
    auto &configManager = mindie_llm::ConfigManager::GetInstance(configPath);
    configManager.Release();
    DT_FUZZ_END()
    SUCCEED();
}

int MockInstanceTransfer(llm::backend::ModelBackend *modelBackend,
                         const std::vector<LLM_ENGINE_BACKEND_Request *> &requests,
                         const std::map<std::string, std::string> &config,
                         std::vector<LLM_ENGINE_BACKEND_Response *> &responses)
{
    auto *resp = reinterpret_cast<mindie_llm::LlmInferResponse *>(responses[0]);
    std::string name = "FAILED_LINK_IP_ATTR";
    mindie_llm::InferDataType dataType = mindie_llm::InferDataType::TYPE_UINT64;
    std::vector<int64_t> dataShape = {1U, 5U};
    auto linkIpAttr = std::make_shared<mindie_llm::InferTensor>(name, dataType, dataShape);
    EXPECT_EQ(linkIpAttr->Allocate(5U * sizeof(uint64_t)), true);
    resp->AddOutput(linkIpAttr);
    return 0;
}

TEST(LlmManagerDTFuzz1, InitInferEngineForPd)
{
    std::srand(time(NULL));
    MINDIE_LLM_LOG_DEBUG("begin====================");
    std::string fuzzName = "LlmManagerDTFuzz:InitInferEngineForPd";
    std::string homePath;
    mindie_llm::GetLlmPath(homePath);
    std::string configPath = homePath + "/conf/confignew.json";
    ;
    MOCKER(IsFileValidStringFunc(FileUtils::IsFileValid)).stubs().will(returnValue(true));
    MOCKER(mindie_llm::ParamChecker::CheckPath).stubs().will(returnValue(true));
    DT_FUZZ_START(0, 1, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        ConstructFuzzConfig(fuzzIndex, configPath);
        // 设置配置
        std::ifstream inputFile(configPath);
        if (!inputFile.is_open()) {
            return;
        }
        Json jsonData;
        inputFile >> jsonData;
        inputFile.close();
        auto &modelConfigs = jsonData["BackendConfig"]["ModelDeployConfig"]["ModelConfig"];
        modelConfigs[0]["is_dmi_infer"] = 1;
        std::ofstream output_file(configPath);
        if (!output_file.is_open()) {
            return;
        }
        output_file << jsonData.dump(JSONDATA_DUMP);
        output_file.close();

        // 初始化PNode
        std::map<std::string, std::string> prefillIpInfo;
        prefillIpInfo["infer_mode"] = "dmi";
        prefillIpInfo["role"] = "prefill";
        LlmInferEngine inferEnginePnode;
        bool ret = inferEnginePnode.Create(configPath, prefillIpInfo);
        if (!ret) {
            std::cout << "engine create fail" << std::endl;
        }
        // 初始化DNode
        LlmInferEngine inferEngineDnode;
        std::map<std::string, std::string> decodeIpInfo;
        decodeIpInfo["infer_mode"] = "dmi";
        decodeIpInfo["role"] = "decode";
        ret = inferEngineDnode.Create(configPath, decodeIpInfo);
        std::set<size_t> npuDeviceIds = {1};
        MOCKER_CPP(&llm::backend::ModelBackend::InstanceInit,
                   int (*)(llm::backend::ModelBackend *modelBackend, const std::map<std::string, std::string> &,
                           std::map<std::string, std::string> &, bool))
            .stubs()
            .will(invoke(&MockInstanceInit));
        MOCKER_CPP(&llm::backend::ModelBackend::InstanceFinalize, int (*)(llm::backend::ModelBackend *modelBackend))
            .stubs()
            .will(returnValue(int(0)));
        MOCKER_CPP(&llm::backend::ModelBackend::InstanceTransfer,
                   int (*)(llm::backend::ModelBackend *modelBackend, const std::vector<LLM_ENGINE_BACKEND_Request *>,
                           const std::map<std::string, std::string> &, std::vector<LLM_ENGINE_BACKEND_Response *> &))
            .stubs()
            .will((invoke(&MockInstanceTransfer)));
        IBISSCHEDULER_Error *nullptrError = nullptr;
        MOCKER(IBISSCHEDULER_Init).stubs().will(returnValue(nullptrError));
        MOCKER(IBISSCHEDULER_Enqueue).stubs().will(returnValue(nullptrError));
        MOCKER(mindie_llm::ParamChecker::CheckPath).stubs().will(returnValue(true));
        try {
            auto ret = inferEnginePnode.Init(0, npuDeviceIds);
            if (!ret) {
                std::cout << "engine init failed:  " << std::endl;
                continue;
            }
            ret = inferEngineDnode.Init(0, npuDeviceIds);
            if (!ret) {
                std::cout << "engine init failed:  " << std::endl;
                continue;
            }
            // 测试InitModelForMultiPd接口
            std::map<std::string, std::string> ipInfo;
            auto status = inferEnginePnode.llmManager_->InitModelForMultiPd(ipInfo, 0);

            status = inferEngineDnode.llmManager_->InitModelForMultiPd(ipInfo, 0);

            // 测试UpdateEngineInfo接口
            mindie_llm::InferRequestId runtimeReqIdDecode(0);
            auto runtimeReqDecode = std::make_shared<mindie_llm::InferRequest>(runtimeReqIdDecode);
            std::string name = "ATTRIBUTES";
            mindie_llm::InferDataType dataType = mindie_llm::InferDataType::TYPE_INT64;
            std::vector<int64_t> dataShape = {1U, 5U};
            auto attrTensorDecode = std::make_shared<mindie_llm::InferTensor>(name, dataType, dataShape);
            attrTensorDecode->Allocate(5U * sizeof(int64_t));
            auto bufferdecode = static_cast<int64_t *>(attrTensorDecode->GetData());
            int64_t switchInt = 1;
            bufferdecode[0] = static_cast<int64_t>(PDRole::DECODE);
            bufferdecode[1] = switchInt;
            status = runtimeReqDecode->AddTensor("ATTRIBUTES", attrTensorDecode);
            ret = inferEngineDnode.llmManager_->UpdateEngineInfo(runtimeReqDecode, false);

            mindie_llm::InferRequestId runtimeReqIdPrefill(1);
            auto runtimeReqPrefill = std::make_shared<mindie_llm::InferRequest>(runtimeReqIdPrefill);
            auto attrTensorPrefill = std::make_shared<mindie_llm::InferTensor>(name, dataType, dataShape);
            attrTensorPrefill->Allocate(5U * sizeof(int64_t));
            auto bufferprefill = static_cast<int64_t *>(attrTensorPrefill->GetData());
            bufferprefill[0] = static_cast<int64_t>(PDRole::PREFILL);
            bufferprefill[1] = switchInt;
            status = runtimeReqPrefill->AddTensor("ATTRIBUTES", attrTensorPrefill);
            ret = inferEnginePnode.llmManager_->UpdateEngineInfo(runtimeReqPrefill, false);
            std::cout << "engine init success:  " << std::endl;
            inferEnginePnode.Finalize();
            inferEngineDnode.Finalize();
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
        }
        auto &configManager = mindie_llm::ConfigManager::GetInstance(configPath);
        configManager.Release();
        // 回退配置
        std::ifstream inputRecovFile(configPath);
        if (!inputRecovFile.is_open()) {
            return;
        }
        Json jsonRecovData;
        inputRecovFile >> jsonRecovData;
        inputRecovFile.close();
        auto &modelRecovConfigs = jsonRecovData["BackendConfig"]["ModelDeployConfig"]["ModelConfig"];
        modelRecovConfigs[0].erase("is_dmi_infer");
        std::ofstream outputRecovfile(configPath);
        if (!outputRecovfile.is_open()) {
            return;
        }
        outputRecovfile << jsonRecovData.dump(JSONDATA_DUMP);
        outputRecovfile.close();
    }
    auto &configManager = mindie_llm::ConfigManager::GetInstance(configPath);
    configManager.Release();
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, GetRequestInput)
{
    std::srand(time(NULL));
    std::string fuzzName = "GetRequestInput";
    int inputsIndex = 2;
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        InferRequestId requestId(1);
        InferRequest inferRequest(requestId);

        auto tensor1 = std::make_shared<InferTensor>("input1", InferDataType::TYPE_INT32, std::vector<int64_t>{1, 2});
        auto tensor2 = std::make_shared<InferTensor>("input2", InferDataType::TYPE_INT64, std::vector<int64_t>{3, 4});
        int32_t *data1 = (int32_t *)malloc(1 * sizeof(int32_t));
        if (data1 == nullptr) {
            return;
        }
        data1[0] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, VOCAB_SIZE_BOUND);

        int64_t *data2 = (int64_t *)malloc(1 * sizeof(int64_t));
        if (data2 == nullptr) {
            return;
        }
        data2[0] = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, VOCAB_SIZE_BOUND);
        tensor1->SetBuffer(data1, sizeof(data1) * sizeof(int32_t), true);
        tensor2->SetBuffer(data2, sizeof(data2) * sizeof(int32_t), true);
        TensorPtr tensorPtr1 = tensor1;
        TensorPtr tensorPtr2 = tensor2;

        inferRequest.AddTensor("input1", tensorPtr1);
        inferRequest.AddTensor("input2", tensorPtr2);

        const struct LLM_ENGINE_BACKEND_InputTensor *input = nullptr;
        LLM_ENGINE_BACKEND_GetRequestInput(reinterpret_cast<const LLM_ENGINE_BACKEND_Request *>(&inferRequest), 1,
                                           &input);

        LLM_ENGINE_BACKEND_GetRequestInput(nullptr, 0, &input);

        LLM_ENGINE_BACKEND_GetRequestInput(reinterpret_cast<const LLM_ENGINE_BACKEND_Request *>(&inferRequest), 0,
                                           nullptr);

        LLM_ENGINE_BACKEND_GetRequestInput(reinterpret_cast<const LLM_ENGINE_BACKEND_Request *>(&inferRequest),
                                           inputsIndex, &input);
        uint32_t count = 0;
        LLM_ENGINE_BACKEND_RequestInputCount(reinterpret_cast<const LLM_ENGINE_BACKEND_Request *>(&inferRequest),
                                             &count);
        LLM_ENGINE_BACKEND_RequestInputCount(nullptr, &count);
        LLM_ENGINE_BACKEND_RequestInputCount(reinterpret_cast<const LLM_ENGINE_BACKEND_Request *>(&inferRequest),
                                             nullptr);

        const char *inputName = nullptr;
        BACKEND_DATA_TYPE dataType;
        const int64_t *shape = nullptr;
        uint32_t dimsCount = 0;

        LLM_ENGINE_BACKEND_GetInputAttribute(reinterpret_cast<const LLM_ENGINE_BACKEND_InputTensor *>(tensor1.get()),
                                             &inputName, &dataType, &shape, &dimsCount);

        const void *buffer = nullptr;
        uint64_t bufferByteSize = 0;
        BACKEND_MEMORY_TYPE memoryType = BACKEND_MEMORY_CPU;
        int64_t memoryTypeId = 0;
        int result =
            LLM_ENGINE_BACKEND_GetInputBuffer(reinterpret_cast<const LLM_ENGINE_BACKEND_InputTensor *>(tensor1.get()),
                                              &buffer, &bufferByteSize, &memoryType, &memoryTypeId);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, ResponseAddOutput)
{
    std::srand(time(NULL));
    std::string fuzzName = "ResponseAddOutput";
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        mindie_llm::InferRequestId testReqId(0);
        auto inferResponse = std::make_shared<LlmInferResponse>(testReqId);
        const char *outputName = "test";
        BACKEND_DATA_TYPE dataType = DATA_TYPE_INT32;
        int64_t shape[2] = {2, 3};
        uint32_t dimsCount = 2;
        struct LLM_ENGINE_BACKEND_OutputTensor *outputTensor = nullptr;
        LLM_ENGINE_BACKEND_ResponseAddOutput(nullptr, outputName, dataType, shape, dimsCount, &outputTensor);
        LLM_ENGINE_BACKEND_ResponseAddOutput(reinterpret_cast<LLM_ENGINE_BACKEND_Response *>(inferResponse.get()),
                                             outputName, dataType, shape, dimsCount, &outputTensor);
        inferResponse->GetRequestId();
        inferResponse->IsEnd();
        inferResponse->GetFlags();
        inferResponse->GetIterTimes();
        inferResponse->GetTransferFlag();
        void *buffer = nullptr;
        uint64_t bufferSize = 0;
        BACKEND_MEMORY_TYPE memoryType = BACKEND_MEMORY_CPU;
        int64_t memoryTypeId = *(int *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 1, 1, BACKEND_REQUEST_BOUND);
        ;
        auto tensor =
            std::make_shared<mindie_llm::InferTensor>("input1", InferDataType::TYPE_INT32, std::vector<int64_t>{1, 2});
        LLM_ENGINE_BACKEND_GetOutputBuffer(reinterpret_cast<LLM_ENGINE_BACKEND_OutputTensor *>(tensor.get()), &buffer,
                                           &bufferSize, &memoryType, &memoryTypeId);
        LLM_ENGINE_BACKEND_GetOutputBuffer(nullptr, &buffer, &bufferSize, &memoryType, &memoryTypeId);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, IbisApiResponse)
{
    std::srand(time(NULL));
    std::string fuzzName = "IbisApiResponse";
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        auto callback = [this](std::shared_ptr<mindie_llm::InferResponse> &response) {
            std::cout << "SendResponseCallback" << std::endl;
        };
        uint64_t requestId = *(u64 *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, REQUEST_SIZE_BOUND);
        std::shared_ptr<InferRequest> request = std::make_shared<InferRequest>(InferRequestId(requestId));
        auto request1 = reinterpret_cast<IBISSCHEDULER_Request *>(request.get());
        auto requestInner = request->GetRequestInner();
        requestInner->SetSendResponseCallback(callback);
        IBISSCHEDULER_Response *response = nullptr;
        IBISSCHEDULER_ResponseCreate(request1, &response);
        IBISSCHEDULER_ResponseSetIterTimes(response, 0);
        IBISSCHEDULER_ResponseSetTransferFlag(response, 0);
        std::vector<int64_t> blocks = {1, 2, 3, 4};
        IBISSCHEDULER_ResponseSetBlockTable(response, blocks);
        std::vector<uint64_t> dpInstanceIds = {1, 2};
        IBISSCHEDULER_ResponseSetDpInstanceIds(response, dpInstanceIds);
        const char *outputName = "test";
        BACKEND_DATA_TYPE dataType = DATA_TYPE_INT32;
        int64_t shape[2] = {2, 3};
        uint32_t dimsCount = 2;
        struct LLM_ENGINE_BACKEND_OutputTensor *outputTensor = nullptr;
        int result = LLM_ENGINE_BACKEND_ResponseAddOutput(reinterpret_cast<LLM_ENGINE_BACKEND_Response *>(response),
                                                          outputName, dataType, shape, dimsCount, &outputTensor);
        const int64_t *shapeOutput = nullptr;
        const void *buffer = nullptr;
        uint64_t dimCount = 0;
        IBISSCHEDULER_ResponseGetOutputByName(response, "test", &shapeOutput, &dimCount, &buffer);

        std::vector<uint64_t> metrics = {1, 2};
        IBISSCHEDULER_ResponseSend(response, 0, metrics);

        IBISSCHEDULER_Request *requestnull = nullptr;
        IBISSCHEDULER_Response *responsenull = nullptr;
        IBISSCHEDULER_ResponseCreate(requestnull, &responsenull);
        IBISSCHEDULER_ResponseSetIterTimes(responsenull, 0);
        IBISSCHEDULER_ResponseSetTransferFlag(responsenull, 0);
        IBISSCHEDULER_ResponseSetBlockTable(responsenull, blocks);
        IBISSCHEDULER_ResponseRelease(responsenull);
        IBISSCHEDULER_ResponseSend(responsenull, 0, metrics);
        const int64_t *shapeNew = nullptr;
        IBISSCHEDULER_ResponseGetOutputByName(responsenull, "test", &shapeNew, &dimCount, &buffer);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, ModelInstanceProperties)
{
    std::srand(time(NULL));
    std::string fuzzName = "ModelInstanceProperties";
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        IBISSCHEDULER_ModelInstanceProperties(nullptr, nullptr);
        IBISSCHEDULER_ModelProperties(nullptr, nullptr);
        const char *modelInstanceType;
        std::shared_ptr<LlmInferModel> model;
        std::shared_ptr<LlmInferModelInstance> instance;
        std::vector<std::shared_ptr<LlmInferModelInstance>> instancesVec;
        auto tmpBackend = std::make_shared<llm::backend::ModelBackend>();
        instance = std::make_shared<LlmInferModelInstance>(tmpBackend, "test");
        auto modelInstance = reinterpret_cast<IBISSCHEDULER_ModelInstance *>(instance.get());
        instancesVec.push_back(instance);
        model = std::make_shared<LlmInferModel>("test", instancesVec);
        IBISSCHEDULER_ModelInstanceProperties(modelInstance, &modelInstanceType);
        const char *nameTemp;
        auto modelTemp = reinterpret_cast<IBISSCHEDULER_Model *>(model.get());
        IBISSCHEDULER_ModelProperties(modelTemp, &nameTemp);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, InferRequestForPd)
{
    std::srand(time(NULL));
    std::string fuzzName = "InferRequestForPd";
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        uint32_t fuzzIndex = 0;
        std::vector<int64_t> blocks;
        IBISSCHEDULER_Request *nullrequest = nullptr;
        IBISSCHEDULER_RequestGetPSrcBlockTable(nullrequest, &blocks);
        uint64_t requestId = *(u64 *)DT_SetGetNumberRange(&g_Element[fuzzIndex++], 0, 1, REQUEST_SIZE_BOUND);
        std::shared_ptr<InferRequest> request = std::make_shared<InferRequest>(InferRequestId(requestId));
        std::vector<int64_t> blocksTemp = {1, 2};
        request->SetSrcBlockTable(blocksTemp);
        request->SetReqType(mindie_llm::InferReqType::REQ_STAND_INFER);
        auto request1 = reinterpret_cast<IBISSCHEDULER_Request *>(request.get());
        IBISSCHEDULER_RequestGetPSrcBlockTable(request1, &blocks);
        std::vector<uint64_t> dpInstanceIdsTemp = {1, 2, 3};
        request->SetDpInstanceIds(dpInstanceIdsTemp);
        std::vector<uint64_t> dpInstanceIds;
        IBISSCHEDULER_Error *error = IBISSCHEDULER_RequestGetPDpInstanceIds(request1, &dpInstanceIds);
    }
    DT_FUZZ_END()
    SUCCEED();
}

TEST(LlmManagerDTFuzz1, DeleteRequest)
{
    std::srand(time(NULL));
    std::string fuzzName = "DeleteRequest";
    DT_FUZZ_START(0, 100, const_cast<char *>(fuzzName.c_str()), 0)
    {
        IBISSCHEDULER_Request *nullrequest = nullptr;
        IBISSCHEDULER_RequestDelete(nullrequest);
    }
    DT_FUZZ_END()
    SUCCEED();
}
} // namespace mindie_llm