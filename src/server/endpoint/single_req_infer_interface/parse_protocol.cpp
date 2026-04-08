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
#include "parse_protocol.h"
#include <cmath>
#include <climits>
#include <codecvt>
#include "endpoint_def.h"
#include "nlohmann/json.hpp"
#include "http_wrapper.h"
#include "random_generator.h"
#include "common_util.h"
#include "check_utils.h"
#include "config_manager_impl.h"
#include "safe_io.h"

#ifdef UT_ENABLED
#define LOCAL_API
#else
#define LOCAL_API static
#endif

using OrderedJson = nlohmann::ordered_json;
using Json = nlohmann::json;

namespace mindie_llm {
struct DoubleSearchMapping {
    std::unordered_map<InferDataType, std::string> typeToString;
    std::unordered_map<std::string, InferDataType> stringToType;
    explicit DoubleSearchMapping(std::unordered_map<InferDataType, std::string> input) noexcept
        : typeToString{ std::move(input) }
    {
        for (auto &it : typeToString) {
            stringToType.emplace(it.second, it.first);
        }
    }
};

static const DoubleSearchMapping DATA_TYPE_MAPPING{ std::unordered_map<InferDataType, std::string>{
    { InferDataType::TYPE_BOOL, "BOOL" },
    { InferDataType::TYPE_UINT8, "UINT8" },
    { InferDataType::TYPE_UINT16, "UINT16" },
    { InferDataType::TYPE_UINT32, "UINT32" },
    { InferDataType::TYPE_UINT64, "UINT64" },
    { InferDataType::TYPE_INT8, "INT8" },
    { InferDataType::TYPE_INT16, "INT16" },
    { InferDataType::TYPE_INT32, "INT32" },
    { InferDataType::TYPE_INT64, "INT64" },
    { InferDataType::TYPE_FP16, "FP16" },
    { InferDataType::TYPE_FP32, "FP32" },
    { InferDataType::TYPE_FP64, "FP64" },
    { InferDataType::TYPE_STRING, "STRING" },
    { InferDataType::TYPE_BF16, "BF16" } } };

static const std::unordered_map<OrderedJson::value_t, std::string> JSON_TYPE_MAPPING = {
    { OrderedJson::value_t::null, "null" },
    { OrderedJson::value_t::object, "object" },
    { OrderedJson::value_t::array, "array" },
    { OrderedJson::value_t::string, "string" },
    { OrderedJson::value_t::boolean, "boolean" },
    { OrderedJson::value_t::number_integer, "integer" },
    { OrderedJson::value_t::number_unsigned, "unsigned" },
    { OrderedJson::value_t::number_float, "float" },
    { OrderedJson::value_t::binary, "binary" } };

static const uint16_t VERSION_LENGTH = 256;

std::string CheckVersionInfo(std::string &version)
{
    std::string baseVersion = "1.0.0";
    if (version.empty() || version.length() > VERSION_LENGTH) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Version length must be in (0, " << VERSION_LENGTH << "], but got " << version.length());
        return baseVersion;
    }
    std::regex versionPattern(R"(^[A-Za-z0-9\-\._]{1,256}$)");
    if (!std::regex_match(version, versionPattern)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "The input string does not meet the expected format. The string should:\n" <<
          "- Be between 1 and 256 characters long.\n" <<
          "- Only contain alphanumeric characters (letters and digits), hyphens (-), periods (.), " <<
          "and underscores (_).\n" <<
          "- Not begin or end with a hyphen (-), period (.), or underscore (_).\n" <<
          "But got " << version << ". Program will use default version " << baseVersion);
        return baseVersion;
    }
    return version;
}

LOCAL_API bool ValidateJsonType(const OrderedJson &jsonObj, const std::string &key, OrderedJson::value_t type,
                                std::string &error) noexcept
{
    if (type == OrderedJson::value_t::number_float && jsonObj[key].is_number()) {
        return true;
    }

    if (type == OrderedJson::value_t::number_integer && jsonObj[key].is_number_integer()) {
        return true;
    }

    if (jsonObj[key].type() != type) {
        error = std::string(key).append(" must be ").append(JsonParse::GetJsonDataType(type)).append(" type.");
        return false;
    }

    return true;
}

LOCAL_API bool ValidateHostIP(const OrderedJson &jsonBody, std::string &error) noexcept
{
    bool isHostIpContained = jsonBody.contains("host_ip");
    if (isHostIpContained) {
        if (jsonBody["host_ip"].is_null() || !jsonBody["host_ip"].is_string()) {
            error = "parse host_ip info in PD Role Request fail.";
            return false;
        }
        if (!CheckIp(jsonBody["host_ip"], "host_ip", false)) {
            error = "host_ip is invalid for IPV4 in PD Role Request.";
            return false;
        }
    }
    return true;
}

LOCAL_API bool ValidateDeviceField(const OrderedJson &jsonBody, std::string &error) noexcept
{
    if (!jsonBody.contains("device") || jsonBody["device"].is_null() || !jsonBody["device"].is_array()) {
        error = "Parse device info in PD Role Request fail.";
        return false;
    }
    if (!JsonParse::CheckPDDeviceIPInfo(jsonBody["device"])) {
        return false;
    }
    return true;
}

LOCAL_API uint32_t IsVllmFormat(const OrderedJson &jsonData)
{
    if (!jsonData.contains("prompt") || jsonData["prompt"].is_null()) {
        return EP_PARSE_NO_PARAM_ERR;
    }

    return EP_OK;
}

LOCAL_API uint32_t IsTgiFormat(const OrderedJson &jsonData)
{
    if (!jsonData.contains("inputs") || jsonData["inputs"].is_null()) {
        return EP_ERROR;
    }
    return EP_OK;
}

LOCAL_API uint32_t GetInferTypeFromJson(const OrderedJson &jsonData, uint16_t &inferType)
{
    try {
        if (IsVllmFormat(jsonData) == EP_OK) {
            inferType = MSG_TYPE_VLLM;
            return EP_OK;
        }
        if (IsTgiFormat(jsonData) == EP_OK) {
            inferType = MSG_TYPE_TGI;
            return EP_OK;
        }
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "MSGType not support");
        return EP_INVALID_PARAM;
    } catch (...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            CHECK_ERROR), "Get MSGType ERROR");
        return EP_PARSE_JSON_ERR;
    }
}

LOCAL_API std::string FaultRecoveryCmdToString(FaultRecoveryCmd cmdType)
{
    switch (cmdType) {
        case FaultRecoveryCmd::CMD_PAUSE_ENGINE:
            return "CMD_PAUSE_ENGINE";
        case FaultRecoveryCmd::CMD_REINIT_NPU:
            return "CMD_REINIT_NPU";
        case FaultRecoveryCmd::CMD_START_ENGINE:
            return "CMD_START_ENGINE";
        case FaultRecoveryCmd::CMD_PAUSE_ENGINE_ROCE:
            return "CMD_PAUSE_ENGINE_ROCE";
        default:
            return "CMD_UNKNOWN";
    }
}

LOCAL_API uint32_t GetFaultRecoveryCmdType(const OrderedJson &jsonData, FaultRecoveryCmd &cmdType, std::string &cmdStr)
{
    if (!jsonData.contains("cmd") || jsonData["cmd"].is_null()) {
        return EP_PARSE_NO_PARAM_ERR;
    }
    // convert int to FaultRecoveryCmd and check if the int is valid
    try {
        cmdType = static_cast<FaultRecoveryCmd>(jsonData["cmd"].get<int>());
        cmdStr = FaultRecoveryCmdToString(cmdType);
    } catch (std::exception &exception) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_FAULT_CONTROL,
            CHECK_ERROR), "Get FaultRecoveryCmdType type failed.");
        return EP_PARSE_JSON_ERR;
    }
    return EP_OK;
}

uint32_t JsonParse::GetInferTypeFromJsonStr(const std::string &jsonStr, uint16_t &inferType)
{
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> convertor;
        std::wstring wstr = convertor.from_bytes(jsonStr);
        auto wjsonStr = convertor.to_bytes(wstr);
        if (!OrderedJson::accept(wjsonStr)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Decode request string to json error.");
            return EP_PARSE_JSON_ERR;
        }
        OrderedJson jsonData = OrderedJson::parse(wjsonStr, CheckOrderedJsonDepthCallback);
        auto code = GetInferTypeFromJson(jsonData, inferType);
        if (code != EP_OK) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "Get infer type failed.");
            return code;
        }
        return EP_OK;
    } catch (std::exception &exception) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), exception.what());
        return EP_PARSE_JSON_ERR;
    }
}

uint32_t JsonParse::DecodeFaultRecoveryCmd(const std::string &jsonStr, FaultRecoveryCmd &cmdType, std::string &cmdStr)
{
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> convertor;
        std::wstring wstr = convertor.from_bytes(jsonStr);
        auto wjsonStr = convertor.to_bytes(wstr);
        if (!OrderedJson::accept(wjsonStr)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_FAULT_CONTROL,
                JSON_PARSE_ERROR), "Decode FaultRecoveryCmd string to json error.");
            return EP_PARSE_JSON_ERR;
        }
        OrderedJson jsonData = OrderedJson::parse(wjsonStr, CheckOrderedJsonDepthCallback);
        auto code = GetFaultRecoveryCmdType(jsonData, cmdType, cmdStr);
        if (code != EP_OK) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_FAULT_CONTROL,
                CHECK_ERROR), "Get FaultRecoveryCmdType type failed.");
            return code;
        }
        return EP_OK;
    } catch (std::exception &exception) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_FAULT_CONTROL,
            JSON_PARSE_ERROR), exception.what());
        return EP_PARSE_JSON_ERR;
    }
}

uint32_t JsonParse::DecodeGeneralTGIStreamMode(const std::string &jsonStr, bool &streamMode, std::string &error)
{
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> convertor;
        std::wstring wstr = convertor.from_bytes(jsonStr);
        auto wjsonStr = convertor.to_bytes(wstr);
        if (!OrderedJson::accept(wjsonStr)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Failed to parse context to json body");
            error = "Failed to parse context to json body";
            return EP_PARSE_JSON_ERR;
        }
        OrderedJson jsonData = OrderedJson::parse(wjsonStr, CheckOrderedJsonDepthCallback);
        if (!jsonData.contains("stream") || jsonData["stream"].is_null()) {
            streamMode = false;
            return EP_OK;
        }
        if (!jsonData["stream"].is_boolean()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Stream must be boolean type.");
            error = "Stream must be boolean type.";
            return EP_PARSE_JSON_ERR;
        }
        streamMode = jsonData["stream"];
        return EP_OK;
    } catch (std::exception &exception) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), exception.what());
        error = "Failed to parse context to json body";
        return EP_PARSE_JSON_ERR;
    }
}

void JsonParse::HandleGetInfo(const ScheduleConfig &scheduleParam,
                              const std::vector<ModelDeployConfig> &modelParam, std::string &infoStr)
{
#ifdef MINDIE_VERSION
    std::string version = MINDIE_VERSION;
    version = CheckVersionInfo(version);
#else
    std::string version = "1.0.0";
#endif
    OrderedJson jsonData;
    jsonData["docker_label"] = nullptr;
    jsonData["max_batch_total_tokens"] = scheduleParam.maxPrefillTokens;
    jsonData["max_best_of"] = 1;
    jsonData["max_concurrent_requests"] = scheduleParam.maxBatchSize;
    jsonData["max_stop_sequences"] = nullptr;
    jsonData["max_waiting_tokens"] = nullptr;
    jsonData["sha"] = nullptr;
    jsonData["validation_workers"] = nullptr;
    jsonData["version"] = version;
    jsonData["waiting_served_ratio"] = nullptr;

    auto i = 0;
    for (auto& mParam : modelParam) {
        jsonData["models"][i]["model_device_type"] = "npu";
        jsonData["models"][i]["model_dtype"] = mParam.torchDtype;
        jsonData["models"][i]["model_id"] = mParam.modelName;
        jsonData["models"][i]["model_pipeline_tag"] = "text-generation";
        jsonData["models"][i]["model_sha"] = nullptr ;
        jsonData["models"][i]["max_total_tokens"] = mParam.maxSeqLen;
        jsonData["max_input_length"] = mParam.maxInputTokenLen;
        ++i;
    }

    infoStr = jsonData.dump();
}

void JsonParse::EncodeTritonModel(ModelDeployConfig &modelParam, std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["name"] = modelParam.modelName;
    jsonData["platform"] = "MindIE Server";
    jsonData["inputs"][0]["name"] = "input0";
    jsonData["inputs"][0]["shape"] = {-1};
    jsonData["inputs"][0]["datatype"] = "UINT32";
    jsonData["outputs"][0]["name"] = "output0";
    jsonData["outputs"][0]["shape"] = {-1};
    jsonData["outputs"][0]["datatype"] = "UINT32";

    jsonStr = jsonData.dump();
}

LOCAL_API std::string GetLastPartOfPath(std::string &path)
{
    if (path.back() == '/' || path.back() == '\\') {
        path.pop_back();
    }
    size_t lastSlashPos = path.find_last_of("/\\");
    if (lastSlashPos == std::string::npos) {
        return path;
    }

    return path.substr(lastSlashPos + 1);
}

void JsonParse::EncodeTritonModelConfig(ModelDeployConfig &modelParam, std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["model_name"] = modelParam.modelName;
    jsonData["input_datatype"] = GetDataType(static_cast<InferDataType>(modelParam.inputDatatype));
    jsonData["output_datatype"] = GetDataType(static_cast<InferDataType>(modelParam.outputDatatype));
    jsonData["max_seq_len"] = modelParam.maxSeqLen;
    if (modelParam.npuMemSize == 0) {
        jsonData["npu_mem_size"] = -1;
    } else {
        jsonData["npu_mem_size"] = modelParam.npuMemSize;
    }
    jsonData["cpu_mem_size"] = modelParam.cpuMemSize;
    jsonData["world_size"] = modelParam.worldSize;
    jsonData["model_weight_path"] = GetLastPartOfPath(modelParam.modelWeightPath);
    jsonData["model_instance_type"] = modelParam.modelInstanceType;

    jsonStr = jsonData.dump();
}

void JsonParse::EncodeTritonEngine(const ScheduleConfig &scheduleParam, std::string &jsonStr)
{
#ifdef MINDIE_VERSION
    std::string version = MINDIE_VERSION;
    version = CheckVersionInfo(version);
#else
    std::string version = "1.0.0";
#endif
    OrderedJson jsonData;
    jsonData["name"] = "MindIE Server";
    jsonData["version"] = version;
    jsonData["extensions"]["max_iter_times"] = scheduleParam.maxIterTimes;
    jsonData["extensions"]["prefill_policy_type"] = scheduleParam.prefillPolicyType;
    jsonData["extensions"]["decode_policy_type"] = scheduleParam.decodePolicyType;
    jsonData["extensions"]["max_prefill_batch_size"] = scheduleParam.maxPrefillBatchSize;
    jsonData["extensions"]["max_prefill_tokens"] = scheduleParam.maxPrefillTokens;
    jsonStr = jsonData.dump();
}

void JsonParse::EncodeSlotCount(const ScheduleConfig &scheduleParam, uint64_t freeSlot, uint64_t availableTokensLen,
    std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["total_slots"] = scheduleParam.maxBatchSize;
    jsonData["free_slots"] = freeSlot;
    jsonData["available_tokens_length"] = availableTokensLen;
    jsonStr = jsonData.dump();
}

void JsonParse::EncodeOpenAiModel(ModelDeployConfig &modelParam, int64_t time, std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["id"] = modelParam.modelName;
    jsonData["object"] = "model";
    jsonData["created"] = time;
    jsonData["owned_by"] = "MindIE Server";

    jsonStr = jsonData.dump();
}

void JsonParse::EncodeOpenAiModel(LoraConfig &loraParam, int64_t time, std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["id"] = loraParam.loraName;
    jsonData["object"] = "model";
    jsonData["created"] = time;
    jsonData["owned_by"] = "MindIE Server";
    jsonData["parent"] = loraParam.baseModel;

    jsonStr = jsonData.dump();
}


void JsonParse::EncodeAbnormalNodeInfo(const std::map<std::string, NodeHealthStatus>& slavesStatus,
    std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["message"] = "Abnormal node detected";
    jsonData["no_contact_node"] = OrderedJson::array();
    for (auto& slave : slavesStatus) {
        if (slave.second == NodeHealthStatus::ABNORMAL) {
            jsonData["no_contact_node"].emplace_back("node(" + slave.first + ") is abnormal.");
        }
    }
    jsonStr = jsonData.dump();
}

void JsonParse::EncodeOpenAiModels(const std::vector<ModelDeployConfig> &modelParam,
    const std::vector<LoraConfig> &loraParam, int64_t time, std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["object"] = "list";
    auto i = 0;
    for (auto& singleModelParams : modelParam) {
        jsonData["data"][i]["id"] = singleModelParams.modelName;
        jsonData["data"][i]["object"] = "model";
        jsonData["data"][i]["created"] = time;
        jsonData["data"][i]["owned_by"] = "MindIE Server";
        ++i;
    }

    for (auto& singleLoraParam : loraParam) {
        jsonData["data"][i]["id"] = singleLoraParam.loraName;
        jsonData["data"][i]["object"] = "model";
        jsonData["data"][i]["created"] = time;
        jsonData["data"][i]["owned_by"] = "MindIE Server";
        jsonData["data"][i]["parent"] = singleLoraParam.baseModel;
        ++i;
    }

    jsonStr = jsonData.dump();
}

void JsonParse::EncodeHealthStatus(const ServiceStatus &status,
    const std::vector<ErrorItem> &errorList, std::string &jsonStr)
{
    OrderedJson jsonData;
    auto &serverConfig = mindie_llm::ConfigManager::GetInstance().GetServerConfig();
    std::string managementIpAddress = serverConfig.managementIpAddress;
    std::string managementPort = std::to_string(serverConfig.managementPort);
    jsonData["status"] = status;
    jsonData["errors"] = OrderedJson::array();
    for (const auto &error : errorList) {
        OrderedJson errorJson;
        errorJson["timestamp"] = error.timestamp;
        errorJson["errCode"] = error.errCode;
        errorJson["createdBy"] = error.createdBy;
        jsonData["errors"].emplace_back(errorJson);
    }
    jsonStr = jsonData.dump();
}

void JsonParse::EncodeCmdResult(const Status &status, mindie_llm::RecoverCommandInfo &info, std::string &jsonStr)
{
    OrderedJson jsonData;
    jsonData["status"] = status.IsOk();
    jsonData["message"] = status.StatusMsg();
    jsonData["reason"] = OrderedJson::array();
    info.results.ForEach([&jsonData](const mindie_llm::NPUExecutionResult &r) {
        OrderedJson reasonJson;
        reasonJson["device_id"] = r.npuDeviceId;
        reasonJson["result"] = r.commandResult == 0;
        reasonJson["message"] = r.errorMsg;
        jsonData["reason"].emplace_back(reasonJson);
        },
        info.results.Size());
    jsonStr = jsonData.dump();
}

const std::string &JsonParse::GetDataType(InferDataType type) noexcept
{
    static const std::string unknownName = "UNKNOWN_TYPE";
    auto pos = DATA_TYPE_MAPPING.typeToString.find(type);
    if (pos != DATA_TYPE_MAPPING.typeToString.end()) {
        return pos->second;
    }

    return unknownName;
}

const std::string &JsonParse::GetJsonDataType(nlohmann::ordered_json::value_t type) noexcept
{
    static const std::string unknownName = "invalid_type";
    auto pos = JSON_TYPE_MAPPING.find(type);
    if (pos == JSON_TYPE_MAPPING.end()) {
        return unknownName;
    }

    return pos->second;
}

bool JsonParse::JsonContainItemWithType(const OrderedJson &jsonObj, const std::string &key, OrderedJson::value_t type,
    std::string &error) noexcept
{
    if (!jsonObj.contains(key)) {
        error = std::string("Not found ").append(key).append(".");
        return false;
    }

    if (jsonObj[key].is_null()) {
        error = std::string(key).append(" must not be null.");
        return false;
    }

    return ValidateJsonType(jsonObj, key, type, error);
}

OptionalItemResult JsonParse::CheckOptionalItemType(const OrderedJson &jsonObj, const std::string &key,
    OrderedJson::value_t type, std::string &error) noexcept
{
    if (!jsonObj.contains(key) || jsonObj[key].is_null()) {
        return OptionalItemResult{
            .isPresent = false,
            .isCorrectType = true
        };
    }

    return OptionalItemResult{
        .isPresent = true,
        .isCorrectType = JsonContainItemWithType(jsonObj, key, type, error)
    };
}

bool JsonParse::CheckPDDeviceIPInfo(const OrderedJson &jsonBody)
{
    for (auto &deviceInfo : jsonBody) {
        if (!deviceInfo.contains("device_logical_id") || deviceInfo["device_logical_id"].is_null()
            || !deviceInfo["device_logical_id"].is_string()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Parse device_logical_id info in PD Role Request fail.");
            return false;
        }
        if (!IsNumber(deviceInfo["device_logical_id"])) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "The device_logical_id info in PD Role Request should be number.");
            return false;
        }
        if (!deviceInfo.contains("device_id") || deviceInfo["device_id"].is_null()
        || !deviceInfo["device_id"].is_string()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Parse device_logical_id info in PD Role Request fail.");
            return false;
        }
        if (!IsNumber(deviceInfo["device_id"])) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "The device_logical_id info in PD Role Request should be number.");
            return false;
        }
        if (!deviceInfo.contains("device_ip") || deviceInfo["device_ip"].is_null()
            || !deviceInfo["device_ip"].is_string()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Parse device_ip info in PD Role Request fail.");
            return false;
        }
        if (!CheckIp(deviceInfo["device_ip"], "device_ip", false)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                CHECK_ERROR), "The device_ip is invalid for IPV4 in PD Role Request.");
            return false;
        }

        if (deviceInfo.contains("rank_id")) { // rank_id is optional field
            if (deviceInfo["rank_id"].is_null() || !deviceInfo["rank_id"].is_string()) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    JSON_PARSE_ERROR), "Parse rank_id info in PD Role Request fail.");
                return false;
            }
            if (!IsNumber(deviceInfo["rank_id"])) {
                ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                    CHECK_ERROR), "The rank_id info in PD Role Request should be number.");
                return false;
            }
        }
    }
    return true;
}

bool JsonParse::CheckIDAndHostIP(const OrderedJson &jsonBody)
{
    bool isIdContained = jsonBody.contains("id");
    bool isHostIpContained = jsonBody.contains("host_ip");
    if (!((isIdContained && isHostIpContained) || (!isIdContained && !isHostIpContained))) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Both id and host_ip must either exist together or not exist.");
        return false;
    }
    if (isIdContained) {
        if (jsonBody["id"].is_null() || !jsonBody["id"].is_number_unsigned()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Parse id info in PD Role Request fail.");
            return false;
        }
    }
    if (isHostIpContained) {
        std::string error;
        if (!ValidateHostIP(jsonBody, error)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), error);
            return false;
        }
    }
    return true;
}

bool JsonParse::CheckPDIPInfo(const OrderedJson &jsonBody)
{
    unsigned int minPPercentage = 0;
    unsigned int maxPPercentage = 100;
    if (jsonBody.contains("p_percentage")) {
        if (jsonBody["p_percentage"].is_null() || !jsonBody["p_percentage"].is_number_unsigned()) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "parse p_percentage info in PD Role Request fail.");
            return false;
        } else if (jsonBody["p_percentage"] < minPPercentage || jsonBody["p_percentage"] > maxPPercentage) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR),
                "parse p_percentage info in PD Role Request fail, the value must be an integer in [0, 100].");
            return false;
        }
    }
    if (!jsonBody.contains("server_ip") || jsonBody["server_ip"].is_null()
        || !jsonBody["server_ip"].is_string() || !CheckIp(jsonBody["server_ip"], "server_ip", false)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse server_ip info in PD Role Request fail.");
        return false;
    }

    if (!JsonParse::CheckIDAndHostIP(jsonBody)) {
        return false;
    }

    std::string error;
    if (!ValidateDeviceField(jsonBody, error)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), error);
        return false;
    }
    return true;
}

bool JsonParse::CheckPDRoleReqJson(const OrderedJson &jsonBody)
{
    if (!jsonBody.contains("local") || jsonBody["local"].is_null() || !jsonBody["local"].is_object()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse local info in PD Role Request fail.");
        return false;
    }
    if (!JsonParse::CheckPDIPInfo(jsonBody["local"])) {
        return false;
    }
    if (!jsonBody.contains("peers") || jsonBody["peers"].is_null() || !jsonBody["peers"].is_array()) {
        return true;
    }
    for (auto &peersInfo : jsonBody["peers"]) {
        if (!JsonParse::CheckPDIPInfo(peersInfo)) {
            return false;
        }
    }
    return true;
}

OrderedJson JsonParse::PrometheusFormat(std::string name, std::string value, std::string managementIpAddress,
    std::string managementPort) noexcept
{
    OrderedJson jsonObj;
    OrderedJson MetricsJsonObj;
    jsonObj["metric"] = OrderedJson::array();

    MetricsJsonObj["__name__"] = name;
    MetricsJsonObj["job"] = "node";
    std::stringstream ss;
    ss << managementIpAddress << ":" << managementPort;
    MetricsJsonObj["instance"] = ss.str();

    jsonObj["metric"].emplace_back(MetricsJsonObj);
    jsonObj["value"] = value;

    return jsonObj;
}

bool JsonParse::JsonHttpMetrics(HttpMetrics &httpMetricsInstance,
    std::map<std::string, uint64_t> &batchSchedulerMetrics, std::string &jsonStr) noexcept
{
    auto &serverConfig = GetServerConfig();
    std::string managementIpAddress = serverConfig.managementIpAddress;
    std::string managementPort = std::to_string(serverConfig.managementPort);

    try {
        OrderedJson jsonObj;
        jsonObj["resultType"] = "vector";
        jsonObj["result"] = OrderedJson::array();

        auto TTFTValue = std::to_string(httpMetricsInstance.DynamicAverageTTFT());
        auto TTFTValuePrometheusJson = PrometheusFormat("TTFT", TTFTValue, managementIpAddress, managementPort);
        jsonObj["result"].emplace_back(TTFTValuePrometheusJson);

        auto TBTValue = std::to_string(httpMetricsInstance.DynamicAverageTBT());
        auto TBTValuePrometheusJson = PrometheusFormat("TBT", TBTValue, managementIpAddress, managementPort);
        jsonObj["result"].emplace_back(TBTValuePrometheusJson);

        std::string waitingInferRequestNum = "0";
        if (batchSchedulerMetrics.find("waitingInferRequestNum") != batchSchedulerMetrics.end()) {
            waitingInferRequestNum = std::to_string(batchSchedulerMetrics["waitingInferRequestNum"]);
        }
        auto waitingInferRequestNumJson = PrometheusFormat("waitingInferRequestNum", waitingInferRequestNum,
            managementIpAddress, managementPort);
        jsonObj["result"].emplace_back(waitingInferRequestNumJson);

        std::string processingInferRequestNum = "0";
        if (batchSchedulerMetrics.find("processingInferRequestNum") != batchSchedulerMetrics.end()) {
            processingInferRequestNum = std::to_string(batchSchedulerMetrics["processingInferRequestNum"]);
        }
        auto processingInferRequestNumJson = PrometheusFormat("processingInferRequestNum", processingInferRequestNum,
            managementIpAddress, managementPort);
        jsonObj["result"].emplace_back(processingInferRequestNumJson);

        std::string remainBlocks = "0";
        if (batchSchedulerMetrics.find("remainBlocks")!= batchSchedulerMetrics.end()) {
            remainBlocks = std::to_string(batchSchedulerMetrics["remainBlocks"]);
        }
        auto remainBlocksJson = PrometheusFormat("remainBlocks", remainBlocks, managementIpAddress, managementPort);
        jsonObj["result"].emplace_back(remainBlocksJson);

        jsonStr = jsonObj.dump();
        return true;
    } catch (const std::exception &e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to get http metrics.");
        return false;
    }
}

bool JsonParse::GetContextJsonBody(const ReqCtxPtr &ctx, OrderedJson &body)
{
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> convertor;
        auto converted = convertor.to_bytes(convertor.from_bytes(ctx->MsgBody()));
        if (!OrderedJson::accept(converted)) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                JSON_PARSE_ERROR), "Convert string to json object exception, cbId is " << ctx->CallbackId());
            return false;
        }
        body = OrderedJson::parse(converted, CheckOrderedJsonDepthCallback);
    } catch(...) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Convert string to json object exception, cbId is " << ctx->CallbackId());
        return false;
    }
    return true;
}

bool JsonParse::CheckHostIP(const OrderedJson &jsonBody)
{
    std::string error;
    if (!ValidateHostIP(jsonBody, error)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), error);
        return false;
    }
    return true;
}

bool JsonParse::CheckPDV2DeviceIPInfo(const OrderedJson &jsonBody)
{
    if (!jsonBody.contains("dp_inst_id") ||
        jsonBody["dp_inst_id"].is_null() ||
        !jsonBody["dp_inst_id"].is_number_unsigned()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse dp_inst_id info in PD Role Request fail.");
        return false;
    }

    std::string error;
    if (!ValidateDeviceField(jsonBody, error)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), error);
        return false;
    }
    return true;
}

bool JsonParse::CheckPDV2IPInfo(const OrderedJson &jsonBody)
{
    if (!jsonBody.contains("server_ip") || jsonBody["server_ip"].is_null()
        || !jsonBody["server_ip"].is_string() || !CheckIp(jsonBody["server_ip"], "server_ip", false)) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse server_ip info in PD Role Request fail.");
        return false;
    }

    if (!JsonParse::CheckHostIP(jsonBody)) {
        return false;
    }

    if (!jsonBody.contains("dp_inst_list") ||
        jsonBody["dp_inst_list"].is_null() ||
        !jsonBody["dp_inst_list"].is_array()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse dp_inst_list info in PD Role Request fail.");
        return false;
    }

    for (auto &dpInstInfo : jsonBody["dp_inst_list"]) {
        if (!JsonParse::CheckPDV2DeviceIPInfo(dpInstInfo)) {
            return false;
        }
    }
    return true;
}

bool JsonParse::CheckPDRoleV2ReqJson(const OrderedJson &jsonBody)
{
    if (!jsonBody.contains("local") || jsonBody["local"].is_null() || !jsonBody["local"].is_array()) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse local info in PD Role Request fail.");
        return false;
    }
    if (!jsonBody.contains("peers") || jsonBody["peers"].is_null() || !jsonBody["peers"].is_array()) {
        return true;
    }
    for (const auto &node : jsonBody["local"]) {
        if (!JsonParse::CheckPDV2IPInfo(node)) {
            return false;
        }
    }

    for (const auto &peerInfo : jsonBody["peers"]) {
        for (const auto &node : peerInfo) {
            if (!JsonParse::CheckPDV2IPInfo(node)) {
                return false;
            }
        }
    }
    return true;
}

void from_json(const OrderedJson& j, DetokenizeExtraInfo& info)
{
    if (j.contains("current_tool_name_sent") && j["current_tool_name_sent"].is_boolean()) {
        info.isCurrentToolNameSent = j["current_tool_name_sent"].get<bool>();
    }
    if (j.contains("current_tool_arguments_sent") && j["current_tool_arguments_sent"].is_boolean()) {
        info.isCurrentArgumentSent = j["current_tool_arguments_sent"].get<bool>();
    }
    if (j.contains("current_tool_id") && j["current_tool_id"].is_number_unsigned()) {
        info.currentToolId = j["current_tool_id"].get<int64_t>();
    }
    if (j.contains("reasoning_tokens") && j["reasoning_tokens"].is_number_unsigned()) {
        info.reasoningTokens = j["reasoning_tokens"].get<int64_t>();
    }
}

void from_json(const OrderedJson& jsonBody, TokenizerContents& contents)
{
    if (jsonBody.contains("content") && jsonBody["content"].is_string()) {
        contents.content = jsonBody["content"].get<std::string>();
    }
    if (jsonBody.contains("origintext") && jsonBody["origintext"].is_string()) {
        contents.content = jsonBody["origintext"].get<std::string>();
    }
    if (jsonBody.contains("reasoning_content") && jsonBody["reasoning_content"].is_string()) {
        contents.reasoningContent = jsonBody["reasoning_content"].get<std::string>();
    }
    if (jsonBody.contains("tool_calls") && (jsonBody["tool_calls"].is_object() || jsonBody["tool_calls"].is_array())) {
        contents.toolCalls = jsonBody["tool_calls"].get<nlohmann::ordered_json>();
    }
    if (jsonBody.contains("update_index") && jsonBody["update_index"].is_boolean()) {
        contents.needUpdateIndex = jsonBody["update_index"].get<bool>();
    }
    if (jsonBody.contains("metadata") && jsonBody["metadata"].is_object()) {
        try {
            DetokenizeExtraInfo info;
            from_json(jsonBody["metadata"], info);
            contents.detokenizeStatus = info;
        } catch (const Json::parse_error& e) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
                LOCAL_INVOKING_ERROR), "Parse json object from detokenize_status failed: " << e.what());
        } catch (const std::exception& e) {
            ULOG_ERROR(SUBMODLE_NAME_ENDPOINT,
                GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE, LOCAL_INVOKING_ERROR),
                "Error during deserialization: " << e.what());
        }
    }
}

TokenizerContents JsonParse::ParseContentFromJson(const std::string& jsonStr)
{
    try {
        OrderedJson jsonBody = OrderedJson::parse(jsonStr, CheckOrderedJsonDepthCallback);
        TokenizerContents contents;
        from_json(jsonBody, contents);
        return contents;
    } catch (const Json::parse_error& e) {
        ULOG_WARN(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(WARNING, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Parse json object from tokenizer failed: " << e.what());
        return TokenizerContents{jsonStr};
    } catch (const std::exception& e) {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Error during deserialization: " << e.what());
        return TokenizerContents{};
    }
}
} // namespace mindie_llm