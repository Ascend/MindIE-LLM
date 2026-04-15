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

#include "http_rest_resource.h"
#define private public
#include "base64_util.h"
#include "basic_types.h"
#include "config_manager.h"
#include "config_manager_impl.h"
#include "endpoint_def.h"
#include "executor.h"
#include "infer_instances.h"
#include "infer_tokenizer.h"
#include "llm_engine.h"
#include "llm_manager_impl.h"
#include "llm_manager_v2.h"
#include "mock_util.h"
#include "response.h"
#include "single_llm_pnd_req_handler.h"
#include "single_req_infer_interface_base.h"
#include "single_req_openai_infer_interface.h"

namespace mindie_llm {
extern uint32_t g_vocabSizeConfig;
extern uint32_t g_maxPositionEmbeddings;
extern uint32_t g_maxSeqLen;
extern uint32_t g_maxInputTokenLen;
extern uint32_t g_maxTopKConfig;
#define MOCKER_CPP(api, TT) (MOCKCPP_NS::mockAPI((#api), (reinterpret_cast<TT>(api))))

MOCKER_CPP_OVERLOAD_EQ(ServerConfig)
static Status MockEncodeSuccess(TokenizerProcessPool *pool, const std::string &prompt, std::vector<int64_t> &tokenIds,
                                HeadFlag flag, uint64_t &timestamp) {
    tokenIds = {1, 2, 3};
    return Status(Error::Code::OK, "Success");
}
static void EmptyMock(const std::string &) {}

static bool MockCheckModelNameCustom(const std::string &modelName) { return true; }

static bool MockAssignIncludeStopStrSuccess(const OrderedJson &jsonObj, RequestSPtr param, std::string &error) {
    return true;
}

static bool MockAssignMaxTokensSuccess(const OrderedJson &jsonObj, InferParamSPtr param, std::string &error) {
    return true;
}

class OpenAiInferTestF : public ::testing::Test {
   protected:
    std::map<std::string, std::string> modelConfig;
    std::map<std::string, std::string> ipInfo;
    SchedulerConfig config;
    Role pdRole;
    std::shared_ptr<InferInstance> inferInstance;
    void SetUp() override {
        LiveInferContext::GetInstance(0)->reqId2SeqGroupMap_.clear();
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.clear();
        LiveInferContext::GetInstance(0)->seqId2RootSeqGroupMap_.clear();
        MOCKER_CPP(&TokenizerProcessPool::Encode, Status(*)(TokenizerProcessPool *, const std::string &,
                                                            std::vector<int64_t> &, HeadFlag, uint64_t &, const bool))
            .stubs()
            .will(invoke(&MockEncodeSuccess));

        MOCKER_CPP(&SingleLLMPnDReqHandler::ProcessNonStreamModeRequest, void (*)(const std::string &))
            .stubs()
            .will(invoke(&EmptyMock));

        MOCKER_CPP(&SingleReqOpenAiInferInterface::CheckModelName, bool (*)(const std::string &))
            .stubs()
            .will(invoke(&MockCheckModelNameCustom));

        MOCKER_CPP(&AssignIncludeStopStrInOutput, bool (*)(const OrderedJson &, RequestSPtr, std::string &))
            .stubs()
            .will(invoke(&MockAssignIncludeStopStrSuccess));

        MOCKER_CPP(&AssignBeamSearch, bool (*)(const OrderedJson &, RequestSPtr, std::string &))
            .stubs()
            .will(invoke(&MockAssignIncludeStopStrSuccess));

        MOCKER_CPP(&AssignBestOf, bool (*)(const OrderedJson &, RequestSPtr, std::string &))
            .stubs()
            .will(invoke(&MockAssignIncludeStopStrSuccess));

        MOCKER_CPP(&AssignN, bool (*)(const OrderedJson &, RequestSPtr, std::string &))
            .stubs()
            .will(invoke(&MockAssignIncludeStopStrSuccess));

        MOCKER_CPP(&AssignMaxTokens, bool (*)(const OrderedJson &, InferParamSPtr, std::string &))
            .stubs()
            .will(invoke(&MockAssignMaxTokensSuccess));
        modelConfig["configPath"] = "";
        modelConfig["npuDeviceIds"] = "0";
        modelConfig["inferMode"] = "standard";
        ipInfo = {{"infer_mode", "standard"}};

        config.cacheBlockSize = 128;
        config.npuBlockNum = 1024;
        config.cpuBlockNum = 0;
        config.policyType = 0;
        config.maxSeqLen = 1024;
        config.maxIterTimes = 512;
        config.dpSize = 1;
        config.enablePrefixCache = false;
        config.enableSplit = false;
        config.prefillPolicyType = 0;
        config.decodePolicyType = 0;
        pdRole = Role::PnD;

        g_vocabSizeConfig = 1024;
        g_maxPositionEmbeddings = 1024;
        g_maxSeqLen = 1024;
        g_maxInputTokenLen = 1024;
        g_maxTopKConfig = 1024;
        inferInstance = GetInferInstance();
        inferInstance->started_ = false;  // make sure not go through InferInstance::Forward
    }
};

TEST_F(OpenAiInferTestF, ShouldSeqGroupMatchParametersInRequest) {
    std::vector<ModelDeployConfig> mockDeployConfig{ModelDeployConfig{.modelName = "mockModel"}};
    MOCKER_CPP(GetModelDeployConfig, const std::vector<ModelDeployConfig> &(*)())
        .stubs()
        .will(returnValue(mockDeployConfig));
    ServerConfig mockServerConfig;
    MOCKER_CPP(GetServerConfig, const ServerConfig &(*)()).stubs().will(returnValue(mockServerConfig));
    httplib::Request request;
    httplib::Response response;
    request.method = "mockMethod";
    request.path = "mockPath";
    request.version = "mockVersion";
    request.body = R"({"model": "llama_65b", "stream": false,
        "messages": [{"role": "user", "content": "mock test"}], "timeout": 1, "stop": ["stop1", "stop2"],
        "priority": 1, "presence_penalty": -1.1, "frequency_penalty": -1.2,
        "skip_special_tokens": false, "max_tokens": 1, "temperature": 1.5, "top_k": 10,
        "stop_token_ids": [1, 2, 3], "top_p": 0.5, "ignore_eos": true, "repetition_penalty": 1.1,
        "seed": 1, "watermark": false, "batch_size": 1, "max_new_tokens": 1
    })";

    std::shared_ptr<RequestContext> context = std::make_shared<RequestContext>(request, response);
    std::shared_ptr<SingleLLMPnDReqHandler> singleLLMPnDReqHandler = std::make_shared<SingleLLMPnDReqHandler>(context);
    bool isReCompute = false;
    auto infer = std::make_shared<SingleReqOpenAiInferInterface>(singleLLMPnDReqHandler, isReCompute);
    // create request and process
    infer->Process();

    std::shared_ptr<LlmManagerV2> llmManager =
        std::make_shared<LlmManagerV2>("", nullptr, nullptr, nullptr, nullptr, nullptr, ipInfo);
    inferInstance->llmManagers_.push_back(llmManager);

    inferInstance->started_ = true;
    // imitate InferInstance::Forward
    // prepare for LlmManagerImpl::ProcessRequests
    std::vector<IExecutorSPtr> executors;
    executors.push_back(std::make_shared<Executor>());

    inferInstance->llmManagers_[0]->impl_->llmEnginePtr_ = MakeLlmEngine(config, executors, nullptr, pdRole);
    auto llmInferRequest = infer->request_;
    inferInstance->llmManagers_[0]->AddRequest(llmInferRequest);

    // add request to llmEngine
    inferInstance->llmManagers_[0]->impl_->ProcessRequests();

    bool ignoreEos = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->ignoreEos_.value();
    EXPECT_EQ(ignoreEos, true);

    std::string stop = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->stopStrings;
    std::string trueStop = Base64Util::Encode("[\"stop1\",\"stop2\"]");
    EXPECT_EQ(stop, trueStop);

    std::vector<TokenId> stopTokens =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->stopTokenIds;
    std::vector<TokenId> trueStopTokens = {1, 2, 3};
    EXPECT_EQ(stopTokens, trueStopTokens);

    float trueTemperature = 1.5f;
    float temperature =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->temperature.value();
    EXPECT_EQ(temperature, trueTemperature);

    int32_t trueTopK = 10;
    int32_t topK = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->topK.value();
    EXPECT_EQ(topK, trueTopK);

    float trueTopP = 0.5f;
    float topP = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->topP.value();
    EXPECT_EQ(topP, trueTopP);

    int64_t seed = LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->seed.value();
    EXPECT_EQ(seed, 1);

    float repetitionPenalty =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->repetitionPenalty.value();
    EXPECT_EQ(repetitionPenalty, 1.1f);

    float presencePenalty =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->presencePenalty.value();
    EXPECT_EQ(presencePenalty, -1.1f);
    float frequencyPenalty =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->sampling->frequencyPenalty.value();
    EXPECT_EQ(frequencyPenalty, -1.2f);

    bool skipSpecialTokens =
        LiveInferContext::GetInstance(0)->seqId2SeqGroupMap_.begin()->second->skipSpecialTokens_.value();
    EXPECT_EQ(skipSpecialTokens, false);
}

}  // namespace mindie_llm
