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

#ifndef GLOBAL_PARAM_H
#define GLOBAL_PARAM_H
#include <iostream>
#include <cstring>
#include <vector>
#include <set>
#include <string>
#include <cstdint>
#include <memory>
#include <map>

namespace mindie_llm {

const std::string INFER_MODE_STANDARD = "standard";
const std::string INFER_MODE_DMI = "dmi";

constexpr uint32_t JSON_DEPTH_LIMIT_MIN = 10U;
constexpr uint32_t JSON_DEPTH_LIMIT_MAX = 100U;

enum class WorkFlowType : uint32_t {
    STANDARD_INFERENCE = 0,
    SPECULATIVE_INFERENCE = 1,
};

struct KvPoolConfig {
    std::string backend;
    std::string configPath;
};

struct ServerConfig {
    // 以下为endpoint相关内容
    std::string ipAddress = "127.0.0.1";
    std::string managementIpAddress = "127.0.0.2";
    int32_t port = 1025;
    int32_t managementPort = 1026;
    int32_t metricsPort = 1027;
    bool allowAllZeroIpListening = false;
    uint32_t maxLinkNum = 1000U;
    uint32_t npuUsageThreshold = 0U;
    bool httpsEnabled = true;
    bool fullTextEnabled = false;
    bool pluginEnabled = false;
    bool deepseekEnabled = false;
    std::string tlsCert;
    std::string tlsCrlPath;
    std::set<std::string> tlsCrlFiles;
    std::string tlsCaPath;
    std::set<std::string> tlsCaFile;
    std::string tlsPk;
    uint64_t tokenTimeout = 600;
    uint64_t e2eTimeout = 600;
    uint32_t maxRequestLength = 40U;
    // pd分离
    int32_t interCommPort = 1211;
    bool interCommTLSEnabled = true;
    std::string inferMode = INFER_MODE_STANDARD;
    std::string interCommTlsCaPath;
    std::vector<std::string> interCommTlsCaFiles;
    std::string interCommTlsCert;
    std::string interCommPk;
    std::string interCommTlsCrlPath;
    std::vector<std::string> interCommTlsCrlFiles;
    std::string managementTlsCert;
    std::string managementTlsCrlPath;
    std::set<std::string> managementTlsCrlFiles;
    std::set<std::string> managementTlsCaFile;
    std::string managementTlsPk;
    std::string metricsTlsCert;
    std::string metricsTlsCrlPath;
    std::set<std::string> metricsTlsCrlFiles;
    std::set<std::string> metricsTlsCaFile;
    std::string metricsTlsPk;
    bool openAiSupportedvLLM{true};
    bool distDPServerEnabled{false};
    uint32_t maxJsonDepth = JSON_DEPTH_LIMIT_MIN;

    bool layerwiseDisaggregated{false};
    std::string layerwiseDisaggregatedRoleType;
    std::string layerwiseDisaggregatedMasterIpAddress;
    std::vector<std::string> layerwiseDisaggregatedSlaveIpAddress;
    int32_t layerwiseDisaggregatedDataPort;
    std::vector<int32_t> layerwiseDisaggregatedCrtlPort;
};

struct ModelParam {
    uint32_t modelInstanceNumber{};
    std::string modelName;
    std::string modelWeightPath;
    uint32_t worldSize = 0;
    std::set<size_t> npuDeviceIds;
    int32_t npuMemSize;
    uint32_t cpuMemSize;
    std::string modelInstanceType = "Standard";
    std::string backendType = "atb";
    bool trustRemoteCode;
    uint32_t maxSeqLen;
    uint32_t maxInputTokenLen;
    uint32_t maxPositionEmbeddings = 0;
    std::string torchDtype;
    uint32_t vocabSize = 0;
    int32_t maxTopK = 1000;
    uint32_t inputDatatype = 9;
    uint32_t outputDatatype = 9;
    uint32_t speculationGamma = 0;
    int32_t maxPositionEmbeddingsModel = -1;
    int32_t maxSequenceLength = -1;
    int32_t maxSeqLenModel = -1;
    int32_t bosTokenId = -1;
    std::string eosTokenId = "";
    int32_t padTokenId = -1;
    bool loadTokenizer = true;
    bool useLora = false;
    uint32_t numThreads = 8;
    std::string pluginParams;
    std::map<std::string, std::string> modelConfig;
    std::map<std::string, std::string> loraModules;
    uint32_t maxLoras = 0;
    uint32_t maxLoraRank = 0;
};

struct EngineConfig {
    // FixMe: tmp field to pass compile
    std::string configPath;
    std::string inferMode;
    // model config
    int modelInstanceNumber;
    std::vector<ModelParam> modelDeployParam;
    std::vector<std::set<size_t>> npuDeviceIds;
    // schedule config
    std::string templateType;
    std::string templateName;
    // kvcache
    uint32_t cacheBlockSize{};
    uint32_t cpuBlockNum = 500U;
    uint32_t npuBlockNum = 500U;
    // prefill
    uint32_t maxPrefillBatchSize{};
    uint32_t maxPrefillTokens{};
    uint32_t prefillTimeMsPerReq{};
    uint32_t prefillPolicyType{};
    uint32_t minPrefillBatchSize = 0U;
    uint32_t maxFirstTokenWaitTime = 2500;
    // decode
    uint32_t decodeTimeMsPerReq{};
    uint32_t decodePolicyType{};
    // batch common
    uint32_t maxBatchSize{};
    uint32_t maxPreemptCount{};
    bool supportSelectBatch{};
    bool supportEarlyFinish = true;
    uint32_t maxQueueDelayMicroseconds{};
    // policy
    uint32_t policyType = 0U;
    uint32_t maxIterTimes;
    bool dpScheduling = false;
    bool activateAsyncInference = false;
    bool distributedEnable = false;

    // slo
    uint16_t stageSelectPolicy = 0U;
    bool dynamicBatchSizeEnable = false;

    // rank
    bool isMaster = false;
    uint32_t globalWorldSize = 0;
    std::string masterIP;
    std::string localIP;
    std::vector<std::string> slaveIPs;
    std::vector<std::string> globalRankIds;

    // backendconfig
    std::string backendName;
    uint32_t tokenizerProcessNumber = 8;
    std::string deployType = "INTER_PROCESS";
    std::string executorType = "LLM_EXECUTOR_PYTHON";
    std::string backendBinPath = "/bin/";

    // 以下为跨机通信相关内容
    bool multiNodesInferEnabled = false;
    bool interNodeTLSEnabled = true;
    int32_t multiNodesInferPort = 1120;
    std::string interNodeTlsCaPath;
    std::string interNodeTlsCaFiles;
    std::string interNodeTlsCert;
    std::string interNodeTlsPk;
    std::string interNodeTlsCrlPath;
    std::string interNodeTlsCrlFiles;

    // mix
    bool enableSplit = false;
    bool splitType;
    bool splitStartType;
    uint32_t splitChunkTokens;
    uint32_t splitStartBatchSize;

    // chunked prefill
    bool enableChunkedPrefill{false};
    size_t prefillChunkSize;
    size_t maxNumPartialPrefills;
    size_t longPrefillTokenThreshold;
    size_t maxLongPartialPrefills;

    // prefix cache
    bool enablePrefixCache = false;

    // kv cache pool
    struct KvPoolConfig kvPoolConfig;

    // buffer response
    bool bufferResponseEnabled = false;
    uint32_t prefillExpectedTime;
    uint32_t decodeExpectedTime;

    // edge-cloud
    bool layerwiseDisaggregated = false;
    bool lwdMultiNodesEnable = false;
    bool isLwdMultiNodesMaster = false;
};

struct BackendConfig {
    std::string backendName;
    uint32_t modelInstanceNumber{};
    std::vector<std::set<size_t>> npuDeviceIds;
    uint32_t tokenizerProcessNumber = 8;
    std::string deployType = "INTER_PROCESS";
    std::string executorType = "LLM_EXECUTOR_PYTHON";
    std::string backendBinPath = "/bin/";
    // 以下为跨机通信相关内容
    bool multiNodesInferEnabled = false;
    bool interNodeTLSEnabled = true;
    int32_t multiNodesInferPort = 1120;
    std::string interNodeTlsCaPath;
    std::string interNodeTlsCaFiles;
    std::string interNodeTlsCert;
    std::string interNodeTlsPk;
    std::string interNodeTlsCrlPath;
    std::string interNodeTlsCrlFiles;
    std::vector<std::string> interNodeTlsCaFilesVec;
    std::vector<std::string> interNodeTlsCrlFilesVec;
    uint32_t worldSize = 0;
    struct KvPoolConfig kvPoolConfig;
    // 分布式边云协同多机
    bool lwdMultiNodesEnable = false;
    int32_t lwdMultiNodesCtrlPort = 10003;
};

struct ModelDeployConfig {
    uint32_t modelInstanceNumber{};
    std::string modelName;
    std::string modelWeightPath;
    uint32_t worldSize = 0;
    std::set<size_t> npuDeviceIds;
    int32_t npuMemSize = 0;
    uint32_t cpuMemSize = 0;
    std::string modelInstanceType = "Standard";
    std::string backendType = "atb";
    bool trustRemoteCode = false;
    uint32_t maxSeqLen = 0;
    uint32_t maxInputTokenLen = 0;
    uint32_t maxPositionEmbeddings = 0;
    std::string torchDtype;
    std::string modelType;
    uint32_t vocabSize = 0;
    int32_t maxTopK = 1000;
    uint32_t inputDatatype = 9;
    uint32_t outputDatatype = 9;
    uint32_t speculationGamma = 0;
    std::string modelCutPolicy = "standard";
    struct ModelCutConfig {
        uint32_t tp = 0;
        uint32_t pp = 0;
        uint32_t dp = 0;
        uint32_t ep = 0;
    } modelCutConfig;
    int32_t maxPositionEmbeddingsModel = -1;
    int32_t maxSeqLenModel = -1;
    int32_t bosTokenId = -1;
    int32_t maxSequenceLength = -1;
    std::string eosTokenId = "";
    int32_t padTokenId = -1;
    int32_t truncation = 0;
    bool loadTokenizer = true;
    bool useLora = false;
    uint32_t numThreads = 8;
    std::string pluginParams;
    std::map<std::string, std::string> modelConfig;
    std::map<std::string, std::string> loraModules;
    uint32_t maxLoras = 0;
    uint32_t maxLoraRank = 0;
};

struct LoraConfig {
    std::string loraName;
    std::string loraPath;
    std::string baseModel;
};

struct DeviceEle {
    std::string deviceId;
    std::string deviceIp;
    std::string rankId;
};

struct ServerEle {
    std::string serverId;
    std::string containerIp;
    std::vector<DeviceEle> device;
};

struct RanktableParam {
    uint32_t serverCount = 0;
    std::vector<ServerEle> serverList;
    struct ServerEle local;
    struct ServerEle master;
    std::vector<ServerEle> slaves;

    // 默认为slave类型实例
    bool isMaster = false;
    uint32_t worldSize = 0;
    uint32_t globalWorldSize = 0;

    [[nodiscard]] bool IsSlave() const { return !isMaster; }
};

// config.json配置项中的 ScheduleConfig
struct ScheduleConfig {
    std::string templateType;
    std::string templateName;
    // kvcache
    uint32_t cacheBlockSize{};
    uint32_t cpuBlockNum = 500U;
    uint32_t npuBlockNum = 500U;
    // prefill
    uint32_t maxPrefillBatchSize{};
    uint32_t maxPrefillTokens{};
    uint32_t prefillTimeMsPerReq{};
    uint32_t prefillPolicyType{};
    uint32_t minPrefillBatchSize = 0U;
    uint32_t maxFirstTokenWaitTime = 2500U;
    // decode
    uint32_t decodeTimeMsPerReq{};
    uint32_t decodePolicyType{};
    // batch common
    uint32_t maxBatchSize{};
    uint32_t maxPreemptCount{};
    bool supportSelectBatch{};
    uint32_t maxQueueDelayMicroseconds{};
    // policy
    uint32_t policyType = 0U;
    uint32_t maxIterTimes = 0U;
    bool dpScheduling = false;
    // mix
    bool enableSplit = false;
    bool splitType = false;
    bool splitStartType = false;
    uint32_t splitChunkTokens = 512U;
    uint32_t splitStartBatchSize = 16U;

    // chunked prefill
    bool enableChunkedPrefill = false;
    size_t prefillChunkSize = 0;
    size_t maxNumPartialPrefills = 64;
    size_t longPrefillTokenThreshold = 1024U;
    size_t maxLongPartialPrefills = 8;

    // prefix cache
    bool enablePrefixCache = false;
    // buffer response
    bool bufferResponseEnabled = false;
    uint32_t prefillExpectedTime = 1500U;
    uint32_t decodeExpectedTime = 50U;
    bool activateAsyncInference = false;
    bool distributedEnable = false;

    // slo
    uint16_t stageSelectPolicy = 0;
    bool dynamicBatchSizeEnable = false;
    
    // layerwiseDisaggregated
    bool lwdNextPHeadPrior = false;
};

struct SchedulerConfig {
    uint64_t instanceId;
    // policy config
    uint32_t mlfqQueueNum;
    uint32_t mlfqMinQuantumMs;
    uint32_t mlfqStarveLimitMs;
    uint32_t prefillPolicyType;
    uint32_t decodePolicyType;
    uint32_t policyType = 0;
    uint32_t batchPnum = 1;
    bool dpScheduling;

    // batch config
    uint32_t worldSize = 0;
    uint32_t maxPreemptCount;
    bool supportSelectBatch;
    uint16_t stageSelectPolicy = 0;
    bool dynamicBatchSizeEnable = false;
    uint32_t prefillTimeMsPerReq;
    uint32_t decodeTimeMsPerReq;
    uint32_t maxPrefillBatchSize;
    uint32_t maxPrefillTokens;
    uint32_t minPrefillBatchSize;
    uint32_t maxFirstTokenWaitTime = 2500U;
    uint32_t prefillWaitingTimeout;
    float lowQPSForWaitBatch = 0.0;
    float waitingCompromiseRatio;
    uint32_t maxBatchSize;
    uint32_t maxQueueDelayMicroseconds;

    bool activateAsyncInference;
    bool distributedEnable{false};

    // model config
    std::vector<std::set<size_t>> npuDeviceIds;
    /* 当前和maxModelLen/maxNumBatchedTokens 保持一致 */
    uint32_t maxSeqLen;
    uint32_t maxInputTokenLen;
    std::string eosTokenId;
    uint32_t maxIterTimes;
    uint32_t cpuBlockNum;
    uint32_t npuBlockNum;
    uint32_t lwdCloudNpuBlockNum;
    uint32_t speculationGamma;
    std::vector<std::string> globalRankIds{};
    uint32_t globalWorldSize = 0;
    uint32_t maxLoras = 0;
    uint32_t maxLoraRank = 0;

    // parallel policy
    uint32_t dpSize{1};
    uint32_t spSize{1};
    uint32_t tpSize{1};
    uint32_t cpSize{1};
    
    // store rankid for logging
    int dpRankId_{0};
    
    // blockManageConfig
    uint32_t cacheBlockSize;

    // New requirement: multiple KV cache block managers (e.g. different block sizes / compression ratios).
    // If empty, fallback to legacy single block manager using `cacheBlockSize`.
    struct KVCacheDesc {
        uint32_t npuBlockNum{0};
        uint32_t blockSize{0};
        uint32_t compressionRatio{1};
        int32_t cacheType{0};
    };
    std::vector<KVCacheDesc> kvCacheDescs{};

    // workflow
    std::string modelName;
    std::string templateType;
    std::string templateName;
    uint32_t pipelineNumber;

    // log param
    std::string logHomePath;
    std::string logLevel;
    uint32_t logFileSize;
    uint32_t logFileNum;

    // mix
    bool enableSplit;
    bool splitType;
    bool splitStartType;
    uint32_t splitChunkTokens;
    uint32_t splitStartBatchSize;

    // chunked prefill（外部参数缺少参数校验）
    bool enableChunkedPrefill{false};
    size_t prefillChunkSize;          // 固定切分长度的块大小
    size_t maxNumPartialPrefills;     // batch中可以被并行做partial prefill的最大请求数
    size_t longPrefillTokenThreshold; // 判定为长prefill的未计算token数阈值
    size_t maxLongPartialPrefills;    // batch中允许容纳的长prefill个数

    // prefix cache
    bool enablePrefixCache;

    // kv pool config
    bool enableKvPool = false;
    struct KvPoolConfig kvPoolConfig;

    // buffer response
    bool bufferResponseEnabled;
    uint32_t prefillExpectedTime;
    uint32_t decodeExpectedTime;

    // 边云修改
    uint32_t maxDispatchBatchNum = 1;
    bool layerwiseDisaggregated = false;

    bool isMultiNodeInfer{false}; // 集中式

    bool ChooseV2BlockManager() const;
};

using SchedulerConfigSPtr = std::shared_ptr<SchedulerConfig>;
} // namespace mindie_llm
#endif
