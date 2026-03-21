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
#include "lora_manager.h"
#include <sstream>
#include "basic_types.h"
#include "log.h"

namespace mindie_llm {
std::once_flag LoraManager::initFlag_;
std::vector<std::shared_ptr<LoraManager>> LoraManager::instances_;

std::unordered_map<model_execute_data::LoraOperationStatus, LoraStatus> loraOpStatusToLoraStatus = {
    {model_execute_data::LoraOperationStatus::LORA_CMD_SUCCESS, LoraStatus::LOAD_SUCCESS},
    {model_execute_data::LoraOperationStatus::SLOTS_FULL, LoraStatus::SLOTS_FULL},
    {model_execute_data::LoraOperationStatus::DUPLICATED_LORA_ID, LoraStatus::DUPLICATED_LORA_ID},
    {model_execute_data::LoraOperationStatus::INVALID_LORA_ID, LoraStatus::INVALID_LORA_ID},
    {model_execute_data::LoraOperationStatus::INVALID_LORA_PATH, LoraStatus::INVALID_LORA_PATH},
    {model_execute_data::LoraOperationStatus::INVALID_LORA_RANK, LoraStatus::INVALID_LORA_RANK},
    {model_execute_data::LoraOperationStatus::UNSUPPORT_CMD, LoraStatus::UNSUPPORT_CMD}
};

LoraOperationRequest BuildLoraOperationRequest(const LoraParamSPtr LoraOperationRequestData, const model_execute_data::LoraOperationType loraType)
{
    LoraOperationRequest loraOperationRequest;
    loraOperationRequest.set_master_model(LoraOperationRequestData->masterModel);
    loraOperationRequest.set_lora_name(LoraOperationRequestData->loraName);
    loraOperationRequest.set_lora_path(LoraOperationRequestData->loraPath);
    loraOperationRequest.set_lora_op_type(loraType);

    return loraOperationRequest;
}

model_execute_data::LoraOperationStatus ParseResponse(const LoraOperationResponse &response)
{
    model_execute_data::LoraOperationStatus loraResult = response.lora_op_status();
    return loraResult;
}

std::string GetLoraMessage(LoraStatus lora_result, const std::string &loraName, const std::string &loraPath)
{
    std::stringstream ss;
    switch (lora_result) {
        case LoraStatus::LOAD_SUCCESS:
            ss << "Success: LoRA adapter '" << loraName << "' added successfully.";
            break;
        case LoraStatus::DUPLICATED_LORA_ID:
            ss << "The LoRA adapter '" << loraName << "' has already been added.";
            break;
        case LoraStatus::UNLOADING:
            ss << "The LoRA adapter '" << loraName << "' is waiting to unload.";
            break;
        case LoraStatus::INVALID_LORA_ID:
            ss << "Call to load LoRA method failed: The LoRA adapter '" << loraName << "' is invalid.";
            break;
        case LoraStatus::INVALID_LORA_PATH:
            ss << "Call to load LoRA method failed: Loading LoRA '" << loraName << "' failed: "
                          "No adapter found for '" << loraPath << "'.";
            break;
        case LoraStatus::INVALID_LORA_RANK:
            ss << "Call to load LoRA method failed: LoRA rank is greater than max_lora_rank.";
            break;
        case LoraStatus::SLOTS_FULL:
            ss << "Call to load LoRA method failed:"
                          "The number of LoRA adapters exceeds 'max_loras', and none are currently unloading.";
            break;
        case LoraStatus::SLOTS_FULL_WITH_UNLOADING:
            ss << "Call to load LoRA method failed: "
                          "The number of LoRA adapters exceeds 'max_loras', some adapters are currently being unloaded.";
            break;
        case LoraStatus::UNLOAD_SUCCESS:
            ss << "Success: LoRA adapter '" << loraName << "' removed successfully.";
            break;
        case LoraStatus::LORA_NOT_FOUND:
            ss << "The LoRA adapter '" << loraName << "' cannot be found.";
            break;
        case LoraStatus::UNSUPPORT_CMD:
            ss << "Call to load LoRA method failed: The LoRA command only supports Python graph, "
                          "please check the model graph type.";
            break;
        default:
            ss << "Unknown LoRA status.";
            break;
    }
    return ss.str();
}

void LoraManager::InitLoadedLoras(const std::vector<ModelParam> &modelParamVec)
{
    for (const auto& singleModelParam : modelParamVec) {
        std::string masterModel = singleModelParam.modelName;
        for (const auto& it : singleModelParam.loraModules) {
            LoraParamSPtr loraParam = std::make_shared<LoraParam>(LoraParam{it.first, it.second, masterModel});
            loaded_.Insert(loraParam->loraName, loraParam);
        }
    }
    return;
}

void LoraManager::Initialize(std::vector<IExecutorSPtr> executors, uint32_t maxLoras)
{
    std::call_once(initFlag_, [&] {
        instances_.resize(executors.size());
        for (size_t i = 0; i < executors.size(); ++i) {
            instances_[i] = std::make_shared<LoraManager>(executors.at(i), maxLoras);
        }
    });
}

LlmLoraPtr LoraManager::GetInstance(size_t localDPRank)
{
    if (!instances_.at(localDPRank)) {
        MINDIE_LLM_LOG_ERROR("[LoraManager::GetInstance] LoraManager not initialized");
        return nullptr;
    }
    return instances_.at(localDPRank);
}

LoraManager::LoraManager(IExecutorSPtr executor, uint32_t maxLoras) : executor_(executor), maxLoras_(maxLoras) {}

LoraStatus LoraManager::GetLoraStatus(const LoraParamSPtr loraInfo, bool &loraIsInvalid)
{
    // 校验加载时传入的lora信息
    std::string loraName = loraInfo->loraName;
    std::string loraPath = loraInfo->loraPath;
    LoraStatus ret;

    // lora path校验
    if (loraPath == "") {
        ret = LoraStatus::INVALID_LORA_PATH;
        loraIsInvalid = false;
    }
    // lora name校验
    if (wait2Unloaded_.Count(loraName) != 0) {
        ret = LoraStatus::UNLOADING;
        loraIsInvalid = false;
    } else if (loaded_.Count(loraName) != 0) {
        ret = LoraStatus::DUPLICATED_LORA_ID;
        loraIsInvalid = false;
    } else if (loaded_.Size() == maxLoras_) {
        if (wait2Unloaded_.Size() != 0) {
            ret = LoraStatus::SLOTS_FULL_WITH_UNLOADING;
        } else {
            ret = LoraStatus::SLOTS_FULL;
        }
        loraIsInvalid = false;
    }
    return ret;
}

Status LoraManager::Load(const LoraParamSPtr loraInfo)
{
    bool loraIsInvalid = true;
    std::string loraMessage;
    LoraStatus loraStatus = this->GetLoraStatus(loraInfo, loraIsInvalid);
    if (!loraIsInvalid) {
        loraMessage = GetLoraMessage(loraStatus, loraInfo->loraName, loraInfo->loraPath);
        MINDIE_LLM_LOG_INFO("[LoraManager::Load] " << loraMessage);
        return Status(Error::Code::OK, loraMessage);
    }

    // 通过Execute下放
    LoraOperationRequest lorarequest = BuildLoraOperationRequest(loraInfo, model_execute_data::LoraOperationType::LOAD);
    if (!executor_->ExecutLoraRequest(lorarequest)) {
        MINDIE_LLM_LOG_ERROR("[LoraManager::Load] Failed to execute load LoRA request.");
        return Status(Error::Code::ERROR, "Failed to execute load LoRA request.");
    }
    LoraOperationResponse loraOperationResponse = executor_->GetLoraOperationResponse();
    model_execute_data::LoraOperationStatus loraOpStatus = ParseResponse(loraOperationResponse);

    loraStatus = loraOpStatusToLoraStatus[loraOpStatus];
    if (loraStatus == LoraStatus::LOAD_SUCCESS) {
        loaded_.Insert(loraInfo->loraName, loraInfo);
    }
    loraMessage = GetLoraMessage(loraStatus, loraInfo->loraName, loraInfo->loraPath);
    MINDIE_LLM_LOG_INFO("[LoraManager::Load] " << loraMessage);
    return Status(Error::Code::OK, loraMessage);
}

Status LoraManager::StartToUnload(const std::string &loraName)
{
    std::string loraMessage;
    // lora name校验
    std::string loraPath = "";
    if (loaded_.Count(loraName) == 0 || wait2Unloaded_.Count(loraName) != 0) {
        loraMessage = GetLoraMessage(LoraStatus::LORA_NOT_FOUND, loraName, loraPath);
        MINDIE_LLM_LOG_INFO("[LoraManager::StartToUnload] Failed to unload LoRA: LoRA has not been loaded or is waiting to unload.");
        return Status(Error::Code::OK, loraMessage);
    }
    LoraParamSPtr wait2unloadloraparam;
    auto opt = loaded_.Get(loraName);
    wait2unloadloraparam = opt.value();

    wait2Unloaded_.Insert(loraName, wait2unloadloraparam);
    loraMessage = GetLoraMessage(LoraStatus::UNLOAD_SUCCESS, loraName, loraPath);
    MINDIE_LLM_LOG_INFO("[LoraManager::StartToUnload] Start to unload LoRA.");
    return Status(Error::Code::OK, loraMessage);
}

Status LoraManager::GetLoadedLoras(std::vector<LoraParamSPtr> &loraInfo)
{
    // 可用的是加载的-等待卸载的
    std::vector<LoraParamSPtr> available;
    std::vector<LoraParamSPtr> loadList = loaded_.Values();
    for (const auto& singleParam :  loadList) {
        if (wait2Unloaded_.Count(singleParam->loraName) == 0) {
            available.push_back(singleParam);
        }
    }
    loraInfo = available;
    return Status(Error::Code::OK, "Success Query.");
}

// engine调用卸载Lora
void LoraManager::TryUnLoadWaiting()
{
    std::vector<LoraParamSPtr> wait2UnloadedList = wait2Unloaded_.Values();
    for (const auto &unloadLoraParam : wait2UnloadedList) {
        if (loraIdRef_.Count(unloadLoraParam->loraName) == 0 || loraIdRef_.Get(unloadLoraParam->loraName) == 0) {
            LoraOperationRequest lorarequest = BuildLoraOperationRequest(unloadLoraParam,
                                                                         model_execute_data::LoraOperationType::UNLOAD);
            bool succ = executor_->ExecutLoraRequest(lorarequest);
            if (!succ) {
                MINDIE_LLM_LOG_ERROR("Call ExecuteUnloadLora failed.");
                throw std::runtime_error("The unload execution failed.Check logs.");
            }
            // 卸载成功删除lora
            wait2Unloaded_.Erase(unloadLoraParam->loraName);
            loaded_.Erase(unloadLoraParam->loraName);
            MINDIE_LLM_LOG_INFO("[LoraManager::UnLoad] " << unloadLoraParam->loraName <<
                                " is successfully unload");
        }
    }
}

// sequence使用，如果loraId不可用则将sequence的loraId设置为None
bool LoraManager::ValidateLoraId(const std::optional<std::string> &loraId)
{
    if (loraId.has_value()) {
        std::string loraIdValue = loraId.value();
        if (wait2Unloaded_.Count(loraIdValue) == 0 && loaded_.Count(loraIdValue) != 0) {
            return true;
        }
    }
    return false;
}

void LoraManager::IncLoraRef(const std::optional<std::string> &loraId)
{
    if (loraId.has_value()) {
        std::string loraIdValue = loraId.value();
        if (loraIdValue != "" && loraIdValue != "None" && loaded_.Count(loraIdValue) != 0) {
            loraIdRef_.IncValue(loraIdValue);
        }
    }
}

void LoraManager::DecLoraRef(const std::optional<std::string> &loraId)
{
    if (loraId.has_value()) {
        std::string loraIdValue = loraId.value();
        if (loraIdValue != "" && loraIdValue != "None") {
            if (loraIdRef_.Count(loraIdValue) == 0 || !loraIdRef_.Get(loraIdValue).has_value() ||
                loraIdRef_.Get(loraIdValue).value() == 0) {
                MINDIE_LLM_LOG_WARN("The LoraId(" << loraIdValue << ") does not exist");
            } else {
                loraIdRef_.DecValue(loraIdValue);
            }
        }
    }
}
} // namespace mindie_llm
