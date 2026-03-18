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

#ifndef IMODEL_WORKER_H
#define IMODEL_WORKER_H

#include <algorithm>
#include <stdexcept>
#include <memory>
#include <map>
#include <vector>
#include <functional>
#include <mutex>
#include "basic_types.h"
#include "sequence_group.h"
#include "sequence_group_meta_data.h"
#include "model_execute_data.pb.h"
#include "data_type.h"

namespace mindie_llm {

using model_execute_data::ExecuteRequest;
using model_execute_data::PDLinkRequest;
using model_execute_data::TGCleanupRequest;
using model_execute_data::ExecuteResponse;
using model_execute_data::ExecuteModelResponse;
using model_execute_data::PDLinkResponse;
using model_execute_data::PullKVResponse;
using model_execute_data::ExecuteType;
using model_execute_data::MODEL_INIT;
using model_execute_data::MODEL_INFER;
using model_execute_data::MODEL_FINALIZE;
using model_execute_data::REMOTE_MODEL_INIT;
using model_execute_data::PD_LINK;
using model_execute_data::KV_TRANSFER;
using model_execute_data::TEXT_GENERATOR_CLEANUP;
using model_execute_data::EOS_CLEANUP;
using model_execute_data::LoraOperationRequest;
using model_execute_data::LoraOperationResponse;
using model_execute_data::LORA_OPERATION;
using model_execute_data::RECOVER_COMMAND_EXEC;
using model_execute_data::PAUSE_COMMAND_EXEC;
using model_execute_data::PAUSE_COMMAND_EXEC_ROCE;
using model_execute_data::CLEAR_COMMAND_EXEC;
using model_execute_data::START_COMMAND_EXEC;
using model_execute_data::EXECUTE_ERROR;

// request
using ExecuteModelRequestPtr = std::unique_ptr<model_execute_data::ExecuteModelRequest>;
using PullKVRequestPtr = std::unique_ptr<model_execute_data::PullKVRequest>;
using TGCleanupRequestPtr = std::unique_ptr<model_execute_data::TGCleanupRequest>;

// request handler
using RequestHandler = std::function<void(ExecuteRequest &)>;

// response
using ModelBatchResultSPtr = std::shared_ptr<model_execute_data::ExecuteModelResponse>;
using ModelBatchResultPtr = std::unique_ptr<model_execute_data::ExecuteModelResponse>;
using PDLinkResponseSPtr = std::shared_ptr<model_execute_data::PDLinkResponse>;
using PullKVResponseSPtr = std::shared_ptr<model_execute_data::PullKVResponse>;

// response handler
using ResponseHandler = std::function<void(ExecuteResponse &)>;
using ExecuteModelResponseHandler = std::function<void(ModelBatchResultSPtr)>;
using PullKVResponseHandler = std::function<void(PullKVResponseSPtr)>;

enum class MasterSlaveRole { MASTER, SLAVE };

struct KVCacheOverview {
    struct KVCacheDesc {
        uint32_t npuBlockNum{0};
        uint32_t blockSize{0};
        uint32_t compressionRatio{1};
        int32_t cacheType{0};

        bool operator==(const KVCacheDesc &other) const
        {
            return npuBlockNum == other.npuBlockNum && blockSize == other.blockSize &&
                   compressionRatio == other.compressionRatio && cacheType == other.cacheType;
        }
    };

    uint32_t cpuBlockNum{0xFFFFFFFF};
    uint32_t npuBlockNum{0xFFFFFFFF};
    uint32_t maxPositionEmbeddings{0xFFFFFFFF};
    uint32_t lwdCloudNpuBlockNum{0xFFFFFFFF};
    std::vector<KVCacheDesc> kvCacheDescs{};
    mutable std::mutex updateValueMutex; // Internal mutex to support thread-safe updates

    void UpdateIfSmaller(uint32_t newCpuBlockNum, uint32_t newNpuBlockNum, uint32_t newMaxPositionEmbeddings)
    {
        std::lock_guard<std::mutex> lock(updateValueMutex);
        cpuBlockNum = std::min(cpuBlockNum, newCpuBlockNum);
        npuBlockNum = std::min(npuBlockNum, newNpuBlockNum);
        maxPositionEmbeddings = std::min(maxPositionEmbeddings, newMaxPositionEmbeddings);
    }

    bool UpdateKvCacheDescsIfEmptyOrEqual(const std::vector<KVCacheDesc> &newDescs)
    {
        std::lock_guard<std::mutex> lock(updateValueMutex);
        if (newDescs.empty()) {
            return true;
        }
        if (kvCacheDescs.empty()) {
            kvCacheDescs = newDescs;
            return true;
        }
        if (kvCacheDescs.size() != newDescs.size()) {
            return false;
        }

        for (size_t i = 0; i < kvCacheDescs.size(); ++i) {
            if (kvCacheDescs[i].blockSize != newDescs[i].blockSize ||
                kvCacheDescs[i].compressionRatio != newDescs[i].compressionRatio ||
                kvCacheDescs[i].cacheType != newDescs[i].cacheType) {
                return false;
            }
            kvCacheDescs[i].npuBlockNum = std::min(kvCacheDescs[i].npuBlockNum, newDescs[i].npuBlockNum);
        }
        return true;
    }
};

/**
 * executor is an agent sending model initialization, execution, kv transfer messages to backend model (each NPU has a
 * SPMD process to handle model forward calculation)
 */
class IExecutor {
public:
    IExecutor() = default;
    virtual ~IExecutor() = default;

    IExecutor(const IExecutor &) = delete;
    IExecutor &operator=(const IExecutor &) = delete;

    // 做模型初始化，从llm_manager调用。 （先做模型初始化，再做engine的初始化）
    virtual bool ExecutorInstanceInit(std::map<std::string, std::string> &config, bool isMultiNodesInfer,
                                      size_t dpIdx = 0) = 0;

    // if successfully broadcast/send the request to backend model, return true, otherwise return false
    virtual bool AsyncExecuteModel(ExecuteModelRequestPtr &modelExecRequest,
                                   ExecuteModelResponseHandler callback = nullptr) = 0;

    virtual bool AsyncTGCleanup(TGCleanupRequestPtr &TGCleanupRequest) = 0;

    virtual bool AsyncEOSCleanup(TGCleanupRequestPtr &TGCleanupRequest) = 0;

    virtual bool ExecutorParseConfigAndInitGRPC(std::map<std::string, std::string> &configFromManager,
                                                bool isMultiNodesInfer, size_t rankIdx) = 0;

    virtual bool MasterAndSlaveModelInit(const std::map<std::string, std::string> &pdInfo) = 0;

    virtual bool SetupPDLink(model_execute_data::PDLinkRequest &pdLinkRequest) = 0;

    virtual bool ExecuteKVTransfer(PullKVRequestPtr &pullKVRequest, PullKVResponseHandler callback = nullptr) = 0;

    virtual bool ExecutorInstanceFinalize() = 0;

    virtual uint32_t GetCpuBlockNum() const = 0;

    virtual uint32_t GetNpuBlockNum() const = 0;

    virtual uint32_t GetLwdCloudNpuBlockNum() const = 0;

    virtual uint32_t GetMaxPositionEmbeddings() const = 0;

    virtual model_execute_data::PDLinkResponse GetPDLinkResponse() const = 0;

    // Static member to hold the KV cache overview, shared across all executor instances
    inline static KVCacheOverview kvCacheOverview_;

    virtual bool ExecutLoraRequest(LoraOperationRequest &loraOperationRequest) = 0;

    virtual void ExecuteRecoverCommand(RecoverCommandInfo &commandInfo) = 0;
    
    virtual model_execute_data::LoraOperationResponse GetLoraOperationResponse() const = 0;
};

using IExecutorSPtr = std::shared_ptr<IExecutor>;

IExecutorSPtr CreateExecutor();

} // namespace mindie_llm

#endif