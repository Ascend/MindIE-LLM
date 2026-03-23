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
#include <atomic>
#include <nlohmann/json.hpp>
#include <acl/acl.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <map>
#include <deque>

#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/speed_probe.h"
#include "atb_speed/base/model.h"

namespace atb_speed {
static std::atomic<atb::Status> g_executeStatus(atb::NO_ERROR);
static std::atomic<atb::Status> g_preExecuteStatus(atb::NO_ERROR);

std::map<uint32_t, std::vector<std::pair<atb::Tensor, bool>>> Model::internalTensors_ = {};

static bool IsTensorDimsEqual(const atb::Dims &left, const atb::Dims &other)
{
    if (left.dimNum != other.dimNum) {
        return false;
    }

    for (uint64_t i = 0; i < left.dimNum; ++i) {
        if (left.dims[i] != other.dims[i]) {
            return false;
        }
    }

    return true;
}

std::string Model::Graph::ToString() const
{
    std::stringstream ss;
    for (size_t i = 0; i < weightTensors.size(); ++i) {
        ss << "weightTensors[" << i << "]:" << &weightTensors.at(i) << " "
           << TensorUtil::TensorToString(weightTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << &inTensors.at(i) << " " << TensorUtil::TensorToString(inTensors.at(i))
           << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << &outTensors.at(i) << " " << TensorUtil::TensorToString(outTensors.at(i))
           << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << &internalTensors.at(i) << " "
           << TensorUtil::TensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;

    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] operation:" << node.operation.get() << ", operationName:" << node.operation->GetName()
           << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "] inTensor:" << tensorIt << " " << TensorUtil::TensorToString(*tensorIt)
               << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "] outTensor:" << tensorIt << " " << TensorUtil::TensorToString(*tensorIt)
               << std::endl;
        }
    }
    return ss.str();
}

atb::Status Model::ParseParam(const std::string &param)
{
    (void)param;
    return atb::NO_ERROR;
}

atb::Status Model::BindParamHostTensor(uint32_t nodeId)
{
    (void)nodeId;
    return atb::NO_ERROR;
}

void Model::Graph::Init()
{
    for (size_t i = 0; i < nodes.size(); i++) {
        auto &node = nodes.at(i);
        node.variantPack.inTensors.reserve(node.inTensors.size());
        node.variantPack.inTensors.resize(node.inTensors.size());
        node.variantPack.outTensors.reserve(node.outTensors.size());
        node.variantPack.outTensors.resize(node.outTensors.size());
    }
    InitTensorType();
    InitTensorMaxNodeMap();
}

void Model::Graph::InitTensorType()
{
    for (auto &node : nodes) {
        node.inTensorTypes.reserve(node.inTensors.size());
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.reserve(node.outTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) =
                IsInternalTensor(node.inTensors.at(i)) ?
                    Model::TensorType::INTERMEDIATE_TENSOR : Model::TensorType::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) =
                IsInternalTensor(node.outTensors.at(i)) ?
                    Model::TensorType::INTERMEDIATE_TENSOR : Model::TensorType::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool Model::Graph::IsInternalTensor(const atb::Tensor *tensor)
{
    for (auto &internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }

    return false;
}

void Model::Graph::InitTensorMaxNodeMap()
{
    std::map<atb::Tensor *, uint64_t> tensorMaxNodeIdMap;
    maxNodeIdTensorMap.clear();

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        atb::Tensor &internalTensor = internalTensors[i];
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
            auto &node = nodes.at(nodeId);
            for (auto inTensorIt : node.inTensors) {
                if (&internalTensor == inTensorIt) {
                    maxNodeId = nodeId;
                    dependNodeCount++;
                }
            }
        }
        tensorMaxNodeIdMap[&internalTensor] = maxNodeId;
        if (dependNodeCount == 0) {
            ATB_SPEED_LOG_WARN("Runner graph internal tensor[" << i << "] dependNodeCount is 0");
        }
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

Model::Model(const std::string &modelName, const std::string &param) : modelName_(modelName), param_(param)
{
    currentDevId_ = 0;
    int ret = aclrtGetDevice(&currentDevId_);
    if (ret != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("aclrtGetDevice failed, error: " << ret);
    }

    CHECK_THROW(param_.empty(), "Model init failed, param is empty, please check.");
    CHECK_THROW(param_.size() > MAX_PARAM_STRING_LENGTH, "Model init failed, param is too long, please check.");
}

Model::~Model() {}

int64_t Model::Init(GetWorkspaceFunc getWorkSpaceFunc, CreateTensorFromTensorDescFunc createTensorFromTensorDescFunc,
    RunTaskFunc runTaskFunc)
{
    // ATB_OPERATION_EXECUTE_ASYNC: whether to enable operator execute pipeline
    // 0 - disable, 1 - enable level-2 pipeline (default), 2 - enable level-3 pipeline
    const char *envStr = std::getenv("ATB_OPERATION_EXECUTE_ASYNC");
    isUsePlanExecuteAsync_ = (envStr && (std::string(envStr) == "2" || std::string(envStr) == "1"));
    isUsePlanPreExecuteAsync_ = (envStr && std::string(envStr) == "2");
    if (isUsePlanExecuteAsync_ && !runTaskFunc) {
        std::thread thread = std::thread(std::bind(&Model::ThreadProcessTask, this));
        taskProcessThread_ = std::move(thread);
    }

    ATB_SPEED_LOG_DEBUG(modelName_ << " new, isTaskQueueEnable:" << (runTaskFunc != nullptr)
                   << ", isUsePlanExecuteAsync:" << isUsePlanExecuteAsync_ << ", currentDevId:" << currentDevId_);

    getWorkSpaceFunc_ = getWorkSpaceFunc;
    createTensorFromTensorDescFunc_ = createTensorFromTensorDescFunc;
    runTaskFunc_ = runTaskFunc;

    int64_t atbStatus = BuildGraph();
    CHECK_THROW(atbStatus != atb::NO_ERROR,
        "Build model graph failed. enable log: export ASCEND_SLOG_PRINT_TO_STDOUT=1, "
        "export ASCEND_GLOBAL_LOG_LEVEL=3 to find "
        "the first error. For more details, see the MindIE official document.");
    eventOps_.clear();
    for (auto& eventOp : g_eventOperationsOfModel) {
        eventOps_.push_back(eventOp);
    }
    g_eventOperationsOfModel.clear();
    graph_.Init();
    ATB_SPEED_LOG_DEBUG(modelName_ << " init graph:\n" << graph_.ToString());
    return atbStatus;
}

atb::Status Model::SkipEvent(bool isSkipEvent)
{
    atb::Status rt = atb::NO_ERROR;
    if (isSkipEvent_ != isSkipEvent) {
        isSkipEvent_ = isSkipEvent;
        atb::common::EventParam eventOpParam;
        eventOpParam.operatorType = atb::common::EventParam::OperatorType::UNDEFINED;
        for (auto& eventOp : eventOps_) {
            if (!isSkipEvent_) {
                rt = atb::UpdateOperationParam(eventOp.first, eventOp.second);
            } else {
                rt = atb::UpdateOperationParam(eventOp.first, eventOpParam);
            }
            if (rt != atb::NO_ERROR) {
                ATB_SPEED_LOG_ERROR(modelName_
                                    << "skip event failed. Enable log: export ASCEND_SLOG_PRINT_TO_STDOUT=1, "
                                       "export ASCEND_GLOBAL_LOG_LEVEL=3 to find the first error. "
                                       "For more details, see the MindIE official document.");
                return rt;
            }
        }
    }
    return rt;
}

atb::Status Model::SetNodeStreamId(Node& node, uint32_t streamId) const
{
    node.streamId = streamId;
    auto rt = atb::SetExecuteStreamId(node.operation.get(), streamId);
    CHECK_THROW(rt != atb::NO_ERROR, "atb::SetExecuteStreamId fail: " << rt);
    return rt;
}

int64_t Model::SetWeight(const std::vector<atb::Tensor> &weightTensors)
{
    if (graph_.weightTensors.size() != weightTensors.size()) {
        ATB_SPEED_LOG_ERROR(modelName_ << " weightTensors.size:" << weightTensors.size() << " != "
                       << " graph.weightTensors.size:" << graph_.weightTensors.size());
        return atb::ERROR_INVALID_IN_TENSOR_NUM;
    }

    graph_.weightTensors = weightTensors;
    return atb::NO_ERROR;
}

int64_t Model::SetKVCache(const std::vector<atb::Tensor> &kCacheTensors, const std::vector<atb::Tensor> &vCacheTensors)
{
    if (graph_.kCacheTensors.size() != kCacheTensors.size()) {
        ATB_SPEED_LOG_ERROR(modelName_ << " kCacheTensors.size:" << kCacheTensors.size() << " != "
                       << " graph.kCacheTensors.size:" << graph_.kCacheTensors.size());
        return atb::ERROR_INVALID_IN_TENSOR_NUM;
    }

    if (graph_.vCacheTensors.size() != vCacheTensors.size()) {
        ATB_SPEED_LOG_ERROR(modelName_ << " vCacheTensors.size:" << vCacheTensors.size() << " != "
                       << " graph.vCacheTensors.size:" << graph_.vCacheTensors.size());
        return atb::ERROR_INVALID_IN_TENSOR_NUM;
    }

    graph_.kCacheTensors = kCacheTensors;
    graph_.vCacheTensors = vCacheTensors;
    return atb::NO_ERROR;
}

atb::Status Model::Execute(atb::Context *context, std::vector<atb::Tensor> &inTensors,
    std::vector<atb::Tensor> &outTensors, const std::string &param)
{
    if (graph_.inTensors.size() != inTensors.size() || graph_.outTensors.size() != outTensors.size()) {
        ATB_SPEED_LOG_ERROR(modelName_ << " graph.inTensors.size:" << graph_.inTensors.size()
                       << ", inTensors.size:" << inTensors.size()
                       << ", graph.outTensors.size:" << graph_.outTensors.size()
                       << ", outTensors.size:" << outTensors.size());
        return atb::ERROR_INVALID_GRAPH;
    }

    ParseParam(param);

    for (auto& i : nodeOutTensors_) {
        i.second.clear();
    }
    allTaskFinish_ = false;
    context_ = context;
    graph_.inTensors = inTensors;
    graph_.outTensors = outTensors;
    ATB_SPEED_LOG_DEBUG(modelName_ << " execute start, executeCount:" << executeCount_ << ", graph:\n"
                  << graph_.ToString());

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); ++nodeId) {
        BuildNodeVariantPack(nodeId);
        BindParamHostTensor(nodeId);
        atb::Status st = ExecuteNode(nodeId);
        if (st != atb::NO_ERROR) {
            return st;
        }
    }

    // The purpose of this code is to dump the model topo for msit.
    if (atb_speed::SpeedProbe::IsReportModelTopoInfo(modelName_)) {
        std::string modelTopo = GetModelTopoInfo();
        atb_speed::SpeedProbe::ReportModelTopoInfo(modelName_, modelTopo);
    }

    WaitAsyncPlanExecuteFinish();

    return atb::NO_ERROR;
}

int64_t Model::UpdateWeightsPtr(void *newWeightsPtr, int64_t oldWeightIds)
{
    if (static_cast<uint64_t>(oldWeightIds) >= graph_.weightTensors.size()) {
        ATB_SPEED_LOG_ERROR(modelName_ << ", replace target weight idx is exceeds the graph weights size, idx: "
                                       << oldWeightIds << ", graph weights size: " << graph_.weightTensors.size());
        return atb::ERROR_INVALID_PARAM;
    }
    graph_.weightTensors.at(oldWeightIds).deviceData = newWeightsPtr;
    return atb::NO_ERROR;
}

int64_t Model::SetWeightFormat(const uint64_t weightId)
{
    if (weightId >= graph_.weightTensors.size()) {
        ATB_SPEED_LOG_ERROR(modelName_ << ", replace source weight idx is exceeds the weights size, idx: "
                                       << weightId << ", weights size: "
                                       << graph_.weightTensors.size());
        return atb::ERROR_INVALID_PARAM;
    }
    graph_.weightTensors.at(weightId).desc.format = ACL_FORMAT_FRACTAL_NZ;
    return atb::NO_ERROR;
}

void Model::BuildNodeOutTensorImpl(
    int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs)
{
    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.reserve(node.operation->GetOutputNum());
    outTensorDescs.resize(node.operation->GetOutputNum());
    atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);
    if (st != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR(modelName_ << " nodes[" << nodeId << "] "
            << " infer shape fail, error code: " << st);
    }
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(modelName_ << " nodes[" << nodeId << "] outTensorDescs[" << i
                      << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i)));
    }

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        CHECK_THROW(!node.outTensors.at(i),
            modelName_ << " nodes[" << nodeId << "] "
                       << "outTensor " << i << "is NULL");
        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == Model::TensorType::INTERMEDIATE_TENSOR) {
            node.variantPack.outTensors.at(i)
                = MallocInternalTensor(node.outTensors.at(i), nodeId, i, outTensorDescs.at(i));
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        }
        if (!TensorUtil::TensorDescEqual(node.variantPack.outTensors.at(i).desc, outTensorDescs.at(i))) {
            ATB_SPEED_LOG_DEBUG(modelName_ << "  nodes[" << nodeId << "] new outTensorDescs[" << i
                           << "]:" << TensorUtil::TensorDescToString(outTensorDescs.at(i))
                           << ", node.variantPack.outTensors.at[" << i
                           << "].desc:" << TensorUtil::TensorDescToString(node.variantPack.outTensors.at(i).desc));
        }
    }
}

void Model::BuildNodeVariantPack(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);

    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.reserve(node.variantPack.inTensors.size());
    inTensorDescs.resize(node.variantPack.inTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        CHECK_THROW(!node.inTensors.at(i),
            modelName_ << " nodes[" << nodeId << "] "
                       << "inTensor " << i << "is NULL");
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        inTensorDescs.at(i) = node.inTensors.at(i)->desc;
        ATB_SPEED_LOG_DEBUG(modelName_ << " nodes[" << nodeId << "] inTensors[" << i
                      << "]:" << TensorUtil::TensorToString(node.variantPack.inTensors.at(i)));
    }

    BuildNodeOutTensorImpl(nodeId, node, inTensorDescs);

    auto it = graph_.maxNodeIdTensorMap.find(nodeId);
    if (it != graph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            FreeInternalTensor(tensorIt, nodeId);
        }
    }
}

void Model::CheckPreviousStepError()
{
    // The OOM status might be captured by the TaskQueue in PTA.
    // Consequently, when an HCCL OOM occurs, this code block might not be executed.
    if (g_preExecuteStatus == atb::ERROR_OUT_OF_DEVICE_MEMORY || g_executeStatus == atb::ERROR_OUT_OF_DEVICE_MEMORY) {
        std::stringstream ss;
        std::string stepName = (g_preExecuteStatus == atb::ERROR_OUT_OF_DEVICE_MEMORY) ? "PreExecute" : "Execute";
        ss << "Device out of memory during a previous asynchronous step (" << stepName << "). "
           << "Error code: "
           << mindie_llm::ERROR_CODE_MAPPING.at("ATB_MODELS_OUT_OF_MEMORY")
           << ". Enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find "
           << "the first error. For more details, see the MindIE official document."
           << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    } else if (g_preExecuteStatus != atb::NO_ERROR || g_executeStatus != atb::NO_ERROR) {
        std::stringstream ss;
        ss << "Execute fail, enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, "
              "export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find the first error. "
              "For more details, see the MindIE official document."
           << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str(), ATB_MODELS_EXECUTION_FAILURE);
        throw std::runtime_error(ss.str());
    }
}

void Model::CheckSetupError(atb::Status st, int nodeId, const std::string &opName)
{
    // When an OOM error occurs during Setup, it indicates an LCCL memory allocation failure.
    if (st == atb::ERROR_OUT_OF_DEVICE_MEMORY) {
        std::stringstream ss;
        ss << "Device out of memory during Setup for operation " << opName
           << " at node index " << nodeId << ". "
           << "Error code: "
           << mindie_llm::ERROR_CODE_MAPPING.at("ATB_MODELS_OUT_OF_MEMORY")
           << ". Enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find "
           << "the first error. For more details, see the MindIE official document."
           << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        throw std::runtime_error(ss.str());
    }
    if (st != atb::NO_ERROR) {
        std::stringstream ss;
        ss << "Setup fail, enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, "
              "export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find the first error. "
              "For more details, see the MindIE official document."
           << std::endl;
        throw std::runtime_error(ss.str());
    }
}

atb::Status Model::ExecuteNode(int nodeId)
{
    ExecuteNodeView(nodeId);
    auto &node = graph_.nodes.at(nodeId);
    CheckPreviousStepError();

    atb::Status st = node.operation->Setup(node.variantPack, node.workspaceSize, context_);
    CheckSetupError(st, nodeId, node.operation->GetName());

    ATB_SPEED_LOG_DEBUG(modelName_ << " get node[" << nodeId << "] workspace size:" << node.workspaceSize);
    if (node.workspaceSize > 0) {
        node.workspace = getWorkSpaceFunc_(node.workspaceSize, node.streamId);
    }
    if (isUsePlanExecuteAsync_) {
        ExecutePlanAsync(nodeId);
    } else {
        st = ExecutePlanSync(nodeId);
    }
    return st;
}

void Model::ThreadProcessTask()
{
    ATB_SPEED_LOG_DEBUG(modelName_ << " thread process operations start");
    int ret = aclrtSetDevice(currentDevId_);
    if (ret != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("AsdRtDeviceSetCurrent fail, error:" << ret);
    }

    size_t processTaskCount = 0;
    while (true) {
        int nodeId = PopTask();
        if (nodeId == -1) {
            ATB_SPEED_LOG_DEBUG(modelName_ << "placeholder task for sync communicate operation");
        } else {
            atb::Status st = ExecutePlanSync(nodeId, !isUsePlanPreExecuteAsync_);
            if (st != atb::NO_ERROR) {
                allTaskFinish_ = true;
                processTaskCount = 0;
                return;
            }
        }

        processTaskCount++;
        if (processTaskCount == graph_.nodes.size()) {
            ATB_SPEED_LOG_DEBUG(modelName_ << " thread process all operations");
            processTaskCount = 0;
            allTaskFinish_ = true;
        }
    }
}

void Model::PushPreTask(int nodeId)
{
    auto task = [this, nodeId] {
        atb::Status st = PreExecutePlanSync(nodeId);
        if (st != atb::NO_ERROR) {
            return;
        }
        // 推送任务给下一级流水
        if (runTaskFunc_) {
            runTaskFunc_(modelName_ + graph_.nodes[nodeId].operation->GetName(), [this, nodeId]() {
                int st = ExecutePlanSync(nodeId, false);
                return st;
            });
        } else {
            PushTask(nodeId);
        }

        if (size_t(nodeId + 1) == graph_.nodes.size() && runTaskFunc_) {
            // 所有任务已经触发，标记结束
            allTaskFinish_ = true;
        }
    };
    ModelTaskExecutor::Instance().PushTask(currentDevId_, task);
}

void Model::PushTask(int nodeId)
{
    std::unique_lock<std::mutex> lock(mutex_);
    taskQueue_.push(nodeId);
    lock.unlock();
    cond_.notify_one();
}

int Model::PopTask()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (taskQueue_.empty()) {
        cond_.wait(lock);
    }
    int nodeId = taskQueue_.front();
    taskQueue_.pop();
    return nodeId;
}

atb::Status Model::ExecutePlanSync(int nodeId, bool doExecuteNormal)
{
    auto &node = graph_.nodes.at(nodeId);
    auto oldType = context_->GetExecuteType();

    atb::ExecuteType targetExecuteType = doExecuteNormal ? atb::EXECUTE_NORMAL : atb::EXECUTE_LAUNCH;
    std::string executeStage = doExecuteNormal ? "normal" : "launch";

    atb::VariantPack &variantPack = node.variantPack;

    ATB_SPEED_LOG_DEBUG(modelName_ << "execute node[" << nodeId << "] start");
    context_->SetExecuteType(targetExecuteType);
    atb::Status st = node.operation->Execute(variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    context_->SetExecuteType(oldType);

    // This code block is executed in a sub-thread.
    // Do not throw exceptions here; instead, propagate errors by setting the global status.
    if (st == atb::ERROR_OUT_OF_DEVICE_MEMORY) {
        std::stringstream ss;
        ss << "Device out of memory during Execute (" << executeStage << ") for operation " << node.operation->GetName()
           << " at node index " << nodeId << ". "
           << "Error code: "
           << mindie_llm::ERROR_CODE_MAPPING.at("ATB_MODELS_OUT_OF_MEMORY")
           << ". Enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find "
           << "the first error. For more details, see the MindIE official document."
           << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        g_executeStatus = st;
    } else if (st != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("Execute node[" << nodeId << "] fail, error code: " << st);
        g_executeStatus = st;
    }
    return st;
}

atb::Status Model::PreExecutePlanSync(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    auto oldType = context_->GetExecuteType();
    atb::VariantPack &variantPack = node.variantPack;
    context_->SetExecuteType(atb::EXECUTE_PRELAUNCH);
    ATB_SPEED_LOG_DEBUG(modelName_ << "pre execute node[" << nodeId << "] start");
    atb::Status st = node.operation->Execute(variantPack, (uint8_t*)(node.workspace), node.workspaceSize, context_);
    context_->SetExecuteType(oldType);

    // This code block is executed in a sub-thread.
    // Do not throw exceptions here; instead, propagate errors by setting the global status.
    if (st == atb::ERROR_OUT_OF_DEVICE_MEMORY) {
        std::stringstream ss;
        ss << "Device out of memory during Execute (prelaunch) for operation " << node.operation->GetName()
           << " at node index " << nodeId << ". "
           << "Error code: "
           << mindie_llm::ERROR_CODE_MAPPING.at("ATB_MODELS_OUT_OF_MEMORY")
           << ". Enable log: export ASCEND_GLOBAL_LOG_LEVEL=3, export ASCEND_SLOG_PRINT_TO_STDOUT=1 to find "
           << "the first error. For more details, see the MindIE official document."
           << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str());
        g_preExecuteStatus = st;
    } else if (st != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("pre execute node[" << nodeId << "] fail, error code: " << st);
        g_preExecuteStatus = st;
    }
    return st;
}

void Model::ExecutePlanAsync(int nodeId)
{
    if (executeCount_ == 0) {
        ExecutePlanSync(nodeId);
        executeCount_++;
        // put a placeholder task for communicate operation
        PushTask(-1);
        return;
    }
    if (isUsePlanPreExecuteAsync_) {
        PushPreTask(nodeId);
    } else if (runTaskFunc_) {
        runTaskFunc_(modelName_ + std::to_string(nodeId), [=]() {
            int st = ExecutePlanSync(nodeId);
            return st;
        });
    } else {
        PushTask(nodeId);
    }
}

void Model::WaitAsyncPlanExecuteFinish()
{
    if (!isUsePlanExecuteAsync_) {
        return;
    }

    if (!isUsePlanPreExecuteAsync_ && runTaskFunc_) {
        return;
    }

    while (!allTaskFinish_) {
        ;
    }
    return;
}

void Model::ExecuteNodeView(int nodeId)
{
    auto &node = graph_.nodes.at(nodeId);
    if (node.inTensorReshapeFuncs.size() > 0) {
        for (size_t i = 0; i < node.inTensorReshapeFuncs.size(); i++) {
            if (node.inTensorReshapeFuncs.at(i)) {
                node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape,
                                                node.variantPack.inTensors.at(i).desc.shape);
            }
        }
    }
}

bool Model::IsTensorDescEqual(const atb::TensorDesc &tensorDesc, const atb::Tensor &atbTensor) const
{
    return atbTensor.desc.dtype == tensorDesc.dtype && atbTensor.desc.format == tensorDesc.format &&
           IsTensorDimsEqual(atbTensor.desc.shape, tensorDesc.shape);
}

void Model::ClearInternalTensors()
{
    for (auto& i : internalTensors_) {
        i.second.clear();
    }
}

atb::Tensor Model::MallocInternalTensor(atb::Tensor* outTensor, size_t nodeId, size_t outTensorId,
    const atb::TensorDesc &tensorDesc)
{
    auto key = graph_.nodes.at(nodeId).streamId;
    if (nodeOutTensors_.count(key) == 0) {
        std::vector<atb::Tensor*> emptyOuts;
        nodeOutTensors_[key] = emptyOuts;
    }
    if (internalTensors_.count(key) == 0) {
        std::vector<std::pair<atb::Tensor, bool>>  emptyInte;
        internalTensors_[key] = emptyInte;
    }
    if (GetSingleton<Config>().IsLayerInternalTensorReuse()) {
        std::vector<atb::Tensor*>::iterator iter =
            std::find(nodeOutTensors_[key].begin(), nodeOutTensors_[key].end(), outTensor);
        if (iter != nodeOutTensors_[key].end()) {
            ATB_SPEED_LOG_DEBUG(modelName_ << " nodeId: " << nodeId << ", out tensor id: "
                << outTensorId << " write inplace");
            return **iter;
        }
        for (auto &it : internalTensors_[key]) {
            if (it.second) { // Tensor被使用中，不能被分配其他Op
                continue;
            }

            if (IsTensorDescEqual(tensorDesc, it.first)) {
                it.second = true;
                ATB_SPEED_LOG_DEBUG(modelName_ << " use old internal tensor");
                return it.first;
            }
        }
    }

    ATB_SPEED_LOG_DEBUG(modelName_ << " create internal tensor, node["
                                   << nodeId << "], outTensor[" << outTensorId << "]");
    atb::Tensor newTensor = createTensorFromTensorDescFunc_(tensorDesc);
    internalTensors_[key].push_back(std::make_pair(newTensor, true));
    nodeOutTensors_[key].push_back(outTensor);
    return newTensor;
}

void Model::FreeInternalTensor(const atb::Tensor *tensorDeviceData, int nodeId)
{
    auto key = graph_.nodes[nodeId].streamId;
    if (GetSingleton<Config>().IsLayerInternalTensorReuse()) {
        for (auto &it : internalTensors_[key]) {
            if (it.first.deviceData == tensorDeviceData->deviceData) {
                it.second = false; // Tensor被释放，可以被后来者使用
                ATB_SPEED_LOG_DEBUG(modelName_ << " free internal tensor");
                break;
            }
        }
    }
}

void Model::GetModelTensorNameList(nlohmann::json &modelJson, std::map<atb::Tensor *, std::string> &tensorNameMap)
{
    std::string tensorName;
    for (size_t i = 0; i < graph_.weightTensors.size(); i++) {
        tensorName = modelName_ + "_weight_" + std::to_string(i);
        modelJson["weightTensors"].emplace_back(tensorName);
        atb::Tensor &weightTensor = graph_.weightTensors[i];
        tensorNameMap[&weightTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.inTensors.size(); i++) {
        tensorName = modelName_ + "_input_" + std::to_string(i);
        modelJson["inTensors"].emplace_back(tensorName);
        atb::Tensor &inTensor = graph_.inTensors[i];
        tensorNameMap[&inTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.outTensors.size(); i++) {
        tensorName = modelName_ + "_output_" + std::to_string(i);
        modelJson["outTensors"].emplace_back(tensorName);
        atb::Tensor &outTensor = graph_.outTensors[i];
        tensorNameMap[&outTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.internalTensors.size(); i++) {
        tensorName = modelName_ + "_internal_" + std::to_string(i);
        modelJson["internalTensors"].emplace_back(tensorName);
        atb::Tensor &internalTensor = graph_.internalTensors[i];
        tensorNameMap[&internalTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.kCacheTensors.size(); i++) {
        tensorName = modelName_ + "_kCache_" + std::to_string(i);
        modelJson["kCacheTensors"].emplace_back(tensorName);
        atb::Tensor &kCacheTensor = graph_.kCacheTensors[i];
        tensorNameMap[&kCacheTensor] = tensorName;
    }

    for (size_t i = 0; i < graph_.vCacheTensors.size(); i++) {
        tensorName = modelName_ + "_vCache_" + std::to_string(i);
        modelJson["vCacheTensors"].emplace_back(tensorName);
        atb::Tensor &vCacheTensor = graph_.vCacheTensors[i];
        tensorNameMap[&vCacheTensor] = tensorName;
    }
}

void Model::GetNodeTopoInfo(nlohmann::json &nodeJson, const Node &opNode,
    const std::map<atb::Tensor *, std::string> tensorNameMap) const
{
    nodeJson["opName"] = opNode.operation->GetName();

    for (auto inTensor : opNode.inTensors) {
        auto it = tensorNameMap.find(inTensor);
        if (it != tensorNameMap.end()) {
            nodeJson["inTensors"].emplace_back(it->second);
        }
    }

    for (auto outTensor : opNode.outTensors) {
        auto it = tensorNameMap.find(outTensor);
        if (it != tensorNameMap.end()) {
            nodeJson["outTensors"].emplace_back(it->second);
        }
    }
}

std::string Model::GetModelTopoInfo()
{
    nlohmann::json modelJson;
    modelJson["modelName"] = modelName_;

    std::map<atb::Tensor *, std::string> tensorNameMap;
    GetModelTensorNameList(modelJson, tensorNameMap);

    for (size_t nodeId = 0; nodeId < graph_.nodes.size(); nodeId++) {
        const auto &opNode = graph_.nodes.at(nodeId);
        nlohmann::json nodeJson;
        GetNodeTopoInfo(nodeJson, opNode, tensorNameMap);
        modelJson["nodes"].emplace_back(nodeJson);
    }
    return modelJson.dump();
}

void Model::ResetExecutionStatus()
{
    g_executeStatus = atb::NO_ERROR;
    g_preExecuteStatus = atb::NO_ERROR;
}
} // namespace atb_speed
