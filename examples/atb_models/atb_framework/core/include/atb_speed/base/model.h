/*
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

#ifndef ATB_SPEED_BASE_MODEL_H
#define ATB_SPEED_BASE_MODEL_H
#include <acl/acl.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <atomic>
#include <set>
#include <nlohmann/json.hpp>
#include "atb_speed/utils/operation_util.h"
#include "atb_speed/utils/check_util.h"
#include "atb_speed/utils/model_task_executor.h"
#include "atb_speed/base/event_manager.h"


namespace atb_speed {
class Model {
public:
    using ReshapeFunc = std::function<void(const atb::Dims &oldDims, atb::Dims &newDims)>;
    using GetWorkspaceFunc = std::function<void*(uint64_t bufferSize, uint32_t bufferKey)>;
    using CreateTensorFromTensorDescFunc = std::function<atb::Tensor(const atb::TensorDesc &tensorDesc)>;
    using Task = std::function<int()>;
    using RunTaskFunc = std::function<void(const std::string &taskName, Task task)>;
    enum class TensorType {
        INTERMEDIATE_TENSOR = 0,
        NOT_INTERMEDIATE_TENSOR,
    };

    struct Node {
        std::shared_ptr<atb::Operation> operation;
        std::vector<atb::Tensor *> inTensors;
        std::vector<atb::Tensor *> outTensors;
        atb::VariantPack variantPack;
        // std::vector<torch::Tensor> torchTensors;
        std::vector<ReshapeFunc> inTensorReshapeFuncs;
        atb::SVector<TensorType> inTensorTypes;
        atb::SVector<TensorType> outTensorTypes;
        uint32_t streamId = 0;
        uint64_t workspaceSize = 0;
        void *workspace = nullptr;
    };

    struct Graph {
        std::vector<atb::Tensor> weightTensors;
        std::vector<atb::Tensor> kCacheTensors;
        std::vector<atb::Tensor> vCacheTensors;
        std::vector<atb::Tensor> inTensors;
        std::vector<atb::Tensor> outTensors;
        std::vector<atb::Tensor> internalTensors;
        std::vector<Node> nodes;
        std::map<uint64_t, std::set<atb::Tensor *>> maxNodeIdTensorMap;
        void Init();
        std::string ToString() const;

    private:
        void InitTensorType();
        bool IsInternalTensor(const atb::Tensor *tensor);
        void InitTensorMaxNodeMap();
    };

    Model(const std::string &modelName, const std::string &param);
    virtual ~Model();
    int64_t Init(GetWorkspaceFunc getWorkSpaceFunc, CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
            RunTaskFunc runTaskFunc = nullptr);
    void ClearInternalTensors();

    virtual uint32_t GetInputNum() const = 0;
    virtual uint32_t GetOutputNum() const = 0;
    virtual atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                   std::vector<atb::TensorDesc> &outTensorDescs) = 0;

    int64_t SetWeight(const std::vector<atb::Tensor> &weightTensors);
    int64_t SetWeightFormat(const uint64_t weightId);
    int64_t SetKVCache(const std::vector<atb::Tensor> &kCacheTensors, const std::vector<atb::Tensor> &vCacheTensors);
    atb::Status SkipEvent(bool isSkipEvent);
    atb::Status SetNodeStreamId(Node& node, uint32_t streamId) const;
    atb::Status Execute(atb::Context *context, std::vector<atb::Tensor> &inTensors,
        std::vector<atb::Tensor> &outTensors, const std::string &param);

    int64_t UpdateWeightsPtr(void *newWeightsPtr, int64_t oldWeightIds);
    void ResetExecutionStatus();

protected:
    virtual int64_t BuildGraph() = 0;
    virtual atb::Status ParseParam(const std::string &param);
    virtual atb::Status BindParamHostTensor(uint32_t nodeId);
    virtual void BuildNodeVariantPack(int nodeId);

protected:
    bool IsTensorDescEqual(const atb::TensorDesc &tensorDesc, const atb::Tensor &atbTensor) const;
    void ExecuteNodeView(int nodeId);
    void CheckPreviousStepError();
    void CheckSetupError(atb::Status st, int nodeId, const std::string &opName);
    atb::Status ExecuteNode(int nodeId);
    void ThreadProcessTask();
    atb::Status ExecutePlanSync(int nodeId, bool doExecuteNormal = true);
    void ExecutePlanAsync(int nodeId);
    atb::Status PreExecutePlanSync(int nodeId);
    void PushPreTask(int nodeId);
    void PushTask(int nodeId);
    int PopTask();
    void WaitAsyncPlanExecuteFinish();
    atb::Tensor MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId,
        const atb::TensorDesc &tensorDesc);
    void FreeInternalTensor(const atb::Tensor *tensorDeviceData, int nodeId = 0);
    void GetModelTensorNameList(nlohmann::json &modelJson,
        std::map<atb::Tensor *, std::string> &tensorNameMap);
    void GetNodeTopoInfo(nlohmann::json &nodeJson, const Node &opNode,
        const std::map<atb::Tensor *, std::string> tensorNameMap) const;
    std::string GetModelTopoInfo();
    void BuildNodeOutTensorImpl(
        int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs);

protected:
    GetWorkspaceFunc getWorkSpaceFunc_;
    CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc_;
    RunTaskFunc runTaskFunc_ = nullptr;
    std::string modelName_;
    std::string param_;
    Graph graph_;

    uint64_t executeCount_ = 0;
    atb::Context *context_;

    bool isUsePlanExecuteAsync_ = false;
    bool isUsePlanPreExecuteAsync_ = false;
    bool isSkipEvent_ = false;
    std::queue<int> taskQueue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread taskProcessThread_;
    std::atomic_bool allTaskFinish_;
    int32_t currentDevId_ = 0;
    static std::map<uint32_t, std::vector<std::pair<atb::Tensor, bool>>> internalTensors_;
    std::map<uint32_t, std::vector<atb::Tensor*>> nodeOutTensors_;
    std::vector<std::pair<atb::Operation*, atb::common::EventParam>> eventOps_;
};
// Max length of param string
const size_t MAX_PARAM_STRING_LENGTH = 200000;
// Max value of tokenOffset, seqLen and qLen
const int MAX_PARAM_VALUE = 600000;
// Max value of vocab_size
const int64_t MAX_VOCAB_SIZE = 10000000;

#define CHECK_THROW(condition, message) \
    do { \
        if (condition) { \
            std::stringstream ss; \
            ss << message << std::endl; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)

} // namespace atb_speed
#endif