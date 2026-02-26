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

#include <limits>
#include "env_util.h"
#include "file_utils.h"
#include "base_config_manager.h"
#include "log.h"

using Json = nlohmann::json;
using namespace nlohmann::literals;
namespace mindie_llm {

static const uint16_t PORT_MIN = 0;
static const uint16_t PORT_MAX = 65535;
constexpr uint32_t FCFS = 0;
constexpr uint32_t STATE = 1U;
constexpr uint32_t PRIORITY = 2U;
constexpr uint32_t MLFQ = 3U;
constexpr uint32_t MAX_IPV4_LENGTH = 32;

bool ParamChecker::ReadJsonFile(const std::string &jsonPath, std::string &baseDir, Json &inputJsonData,
                                std::string configType)
{
    bool checkFlag = true;
    const std::string isCheck = EnvUtil::GetInstance().Get("MINDIE_CHECK_INPUTFILES_PERMISSION");
    if (isCheck == "0") {
        checkFlag = false;
    }
    std::string errMsg{};
    std::string regularPath;
    if (!FileUtils::RegularFilePath(jsonPath, baseDir, errMsg, regularPath) ||
        !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_640, checkFlag)) {
        std::cout << errMsg << std::endl;
        return false;
    }
    std::ifstream file(regularPath);
    if (!file.is_open()) {
        std::cout << "Error: Open json file failed" << std::endl;
        return false;
    }

    Json jsonData;
    try {
        file >> jsonData;
        file.close();
    } catch (const std::exception &e) {
        file.close();
        std::cout << "Json file is invalid. Please check json format! " << std::endl;
        return false;
    }

    try {
        if (configType.empty()) {
            inputJsonData = jsonData;
        } else {
            inputJsonData = jsonData.at(configType);
        }
    } catch (const Json::exception &e) {
        std::cout << configType << ": " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool ParamChecker::IsWithinRange(std::string integerType, Json jsonValue)
{
    std::string value = jsonValue.dump();
    // The jsonValue maybe string or number. '"' will be added if it's string. Strip it before convert it to number
    if (value.size() > 0 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2); // 1 and 2 is used to remove '"' before and end of string
    }
    try {
        if (integerType == "int32_t") {
            long long num = std::stoll(value);
            return num >= std::numeric_limits<int32_t>::min() && num <= std::numeric_limits<int32_t>::max();
        } else if (integerType == "uint32_t") {
            if (value.find('-') != std::string::npos) { // Check if it's negative
                std::cerr << "Negative value is invalid for uint32_t.\n";
                return false;
            }
            unsigned long long num = std::stoull(value);
            return num <= std::numeric_limits<uint32_t>::max();
        } else if (integerType == "size_t") {
            if (value.find('-') != std::string::npos) { // Check if it's negative
                std::cerr << "Negative value is invalid for size_t.\n";
                return false;
            }
            unsigned long long num = std::stoull(value);
            return num <= std::numeric_limits<size_t>::max();
        } else {
            std::cerr << "Unsupported integer type: " << integerType << "\n";
            return false;
        }
    } catch (const std::invalid_argument &) {
        std::cerr << "Invalid argument: Unable to convert to " << integerType << ".\n";
        return false;
    } catch (const std::out_of_range &) {
        std::cerr << "Out of range: Value is too large for " << integerType << ".\n";
        return false;
    }
}

bool ParamChecker::CheckJsonArray(Json jsonData, const std::string &eleType, const std::string &integerType)
{
    for (auto &ele : jsonData) {
        if (eleType == "string" && !ele.is_string()) {
            std::cout << "Type of element in json array does not match " << eleType << std::endl;
            return false;
        } else if (eleType == "integer") {
            if (!ele.is_number_integer()) {
                std::cout << "Type of element in json array does not match " << eleType << std::endl;
                return false;
            }
            if (!IsWithinRange(integerType, ele)) {
                std::cout << "The value of element in json array for " << integerType << " is invalid." << std::endl;
                return false;
            }
        } else if (eleType == "bool" && !ele.is_boolean()) {
            std::cout << "Type of element in json array does not match " << eleType << std::endl;
            return false;
        }
    }
    return true;
}

bool ParamChecker::GetJsonData(const std::string &configFile, std::string &baseDir, Json &jsonData,
                               const bool &skipPermissionCheck)
{
    try {
        bool checkFlag = true;
        const std::string isCheck = EnvUtil::GetInstance().Get("MINDIE_CHECK_INPUTFILES_PERMISSION");
        checkFlag = (skipPermissionCheck || isCheck == "0") ? false : checkFlag;
        std::string errMsg;
        std::string regularPath;
        if (!FileUtils::RegularFilePath(configFile, baseDir, errMsg, regularPath) ||
            !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_750, checkFlag)) {
            std::cout << errMsg << std::endl;
            return false;
        }
        std::ifstream file(configFile);
        if (!file.is_open()) {
            std::cerr << "ERR: Failed to open model config.json file! " << std::endl;
            return false;
        }
        file >> jsonData;
        file.close();
        return true;
    } catch (const std::exception &e) {
        std::cout << "ERR: model config.json file is invalid. Please check it! " << std::endl;
        return false;
    }
}

bool ParamChecker::CheckAndGetLoraJsonFile(std::string &baseDir, nlohmann::json &loraJsonData)
{
    Json tmpLoraJson;
    std::string loraName;
    std::string loraPath;

    std::string jsonPath = baseDir + "/lora_adapter.json";
    if (!FileUtils::CheckFileExists(jsonPath)) {
        return true;
    }

    std::string errMsg{};
    std::string regularPath;
    if (!FileUtils::RegularFilePath(jsonPath, baseDir, errMsg, regularPath) || !FileUtils::IsFileValid(regularPath, errMsg)) {
        std::cout << errMsg << std::endl;
        return false;
    }

    if (!GetJsonData(regularPath, baseDir, tmpLoraJson)) {
        std::cout << "ERR: Read lora_adapter.json file in " << baseDir << " failed, Please check it!" << std::endl;
        return false;
    }

    for (auto &it : tmpLoraJson.items()) {
        loraName = it.key();
        loraPath = it.value();
    }

    loraJsonData = std::move(tmpLoraJson);
    return true;
}

uint32_t ParamChecker::GetIntegerParamDefaultValue(nlohmann::json jsonData, const std::string &configName,
                                                   uint32_t defaultVal)
{
    uint32_t targetParam = defaultVal;
    if (jsonData.contains(configName)) {
        if (jsonData[configName].is_number_integer()) {
            if (ParamChecker::IsWithinRange("uint32_t", jsonData[configName])) {
                targetParam = jsonData[configName];
            } else {
                std::cout << "The value of configName " << configName << " for  uint32_t is invalid, use default value."
                          << std::endl;
            }
        } else {
            std::cout << "The type of configName " << configName << " should be integer, but is "
                      << jsonData[configName].type_name() << ", use default value." << std::endl;
        }
    } else {
        std::cout << "The configName " << configName << " is not found, use default value." << std::endl;
    }
    return targetParam;
}

std::string ParamChecker::GetStringParamValue(nlohmann::json jsonData, const std::string &configName,
                                              std::string defaultVal)
{
    std::string targetParam = defaultVal;
    if (jsonData.contains(configName) && jsonData[configName].is_string()) {
        targetParam = jsonData[configName];
    }
    return targetParam;
}

bool ParamChecker::GetBoolParamValue(nlohmann::json jsonData, const std::string &configName, bool defaultVal)
{
    bool targetParam = defaultVal;
    if (jsonData.contains(configName) && jsonData[configName].is_boolean()) {
        targetParam = jsonData[configName];
    }
    return targetParam;
}

bool ParamChecker::CheckNpuRange(Json jsonValue)
{
    try {
        for (const auto &setArray : jsonValue) {
            if (!setArray.is_array()) {
                std::cout << "The type of param in each npuDeviceIds set should be array, but is "
                          << setArray.type_name() << std::endl;
                return false;
            }
            for (auto &npuId : setArray) {
                if (!IsWithinRange("size_t", npuId)) {
                    std::cout << "The values of npu_ids should not be less than 0, but got " << npuId << std::endl;
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception &e) {
        std::cout << "npu_ids in json is invalid. Please check it!" << std::endl;
        return false;
    }
}

bool ParamChecker::IsArrayValid(const std::string &configName, Json jsonValue)
{
    if (!jsonValue.is_array()) {
        std::cout << "The type of param " << configName << " should be array, but is " << jsonValue.type_name()
                  << std::endl;
        return false;
    }
    if (configName == "npuDeviceIds") {
        return CheckNpuRange(jsonValue);
    }
    return true;
}

bool ParamChecker::CheckJsonParamType(Json &jsonData, std::vector<ParamSpec> &paramSpecs)
{
    for (auto &paramSpec : paramSpecs) {
        Json param;
        if (jsonData.contains(paramSpec.name)) {
            param = jsonData.at(paramSpec.name); // 存在就取
        } else {
            if (paramSpec.compulsory) {
                std::cout << "[ParamChecker::CheckJsonParamType] " << paramSpec.name << ": missing compulsory field"
                          << std::endl;
                return false;
            } else {
                continue; // 可选参数，不存在也没事
            }
        }

        if (paramSpec.Type == "string" && !param.is_string()) {
            std::cout << "The type of param " << paramSpec.name << " should be string, but got " << param.type_name()
                      << std::endl;
            return false;
        } else if (paramSpec.Type == "int32_t" || paramSpec.Type == "uint32_t" || paramSpec.Type == "size_t") {
            if (!param.is_number_integer()) {
                std::cout << "The type of param " << paramSpec.name << " should be " << paramSpec.Type << ", but got "
                          << param.type_name() << std::endl;
                return false;
            }
            if (!IsWithinRange(paramSpec.Type, param)) {
                std::cout << "The value of param " << paramSpec.name << " for " << paramSpec.Type << " is invalid."
                          << std::endl;
                return false;
            }
        } else if (paramSpec.Type == "array") {
            if (!IsArrayValid(paramSpec.name, param)) {
                return false;
            }
        } else if (paramSpec.Type == "bool" && !param.is_boolean()) {
            std::cout << "The type of param " << paramSpec.name << " should be bool, but got " << param.type_name()
                      << std::endl;
            return false;
        } else if (paramSpec.Type == "object" && !param.is_object()) {
            std::cout << "The type of param " << paramSpec.name << " should be object, but got " << param.type_name()
                      << std::endl;
            return false;
        }
    }
    return true;
}

bool ParamChecker::CheckPath(const std::string &path, std::string &baseDir, const std::string &inputName, bool flag,
                             uint64_t maxFileSize)
{
    // 默认校验文件，否则校验目录
    std::regex reg(".{1,4096}");
    if (!std::regex_match(path, reg)) {
        std::cout << "The " << inputName << " path is too long." << std::endl;
        return false;
    }

    if (flag) {
        bool checkFlag = true;
        const std::string isCheck = EnvUtil::GetInstance().Get("MINDIE_CHECK_INPUTFILES_PERMISSION");
        if (isCheck == "0") {
            checkFlag = false;
        }
        std::string errMsg;
        std::string regularPath;
        if (!FileUtils::RegularFilePath(path, baseDir, errMsg, regularPath) ||
            !FileUtils::IsFileValid(regularPath, errMsg, true, FileUtils::FILE_MODE_640, checkFlag, maxFileSize)) {
            std::cout << "The " << inputName << " path is invalid by: " << errMsg << std::endl;
            return false;
        }
    } else {
        std::string errMsg;
        std::string regularPath;
        if (!FileUtils::CheckDirectoryExists(path) || !FileUtils::RegularFilePath(path, baseDir, errMsg, regularPath)) {
            std::cout << "The " << inputName << " path is not a dir by: " << errMsg << std::endl;
            return false;
        }
    }
    return true;
}

bool ParamChecker::CheckPolicyValue(uint32_t inputValue, const std::string &inputName)
{
    if (inputValue != FCFS && inputValue != STATE && inputValue != PRIORITY && inputValue != MLFQ) {
        std::cout << inputName << " [" << std::to_string(inputValue)
                  << "] is outside the expected schedule policy range: 0, 1, 2, 3";
        return false;
    }
    return true;
}

bool ParamChecker::CheckMixPolicyValue(uint32_t inputValue, const std::string &inputName)
{
    if (inputValue != 0 && inputValue != 4U && inputValue != 5U && inputValue != 6U && inputValue != 7U) {
        MINDIE_LLM_LOG_ERROR("The " << inputName << " [" << inputValue << "] is outside the expected range: 0 or 4~7.");
        return false;
    }
    return true;
}

bool ParamChecker::CheckEngineName(const std::string &engineName)
{
    if (engineName.size() > 50U) {
        std::cout << "The length of backendName exceeds 50." << std::endl;
        return false;
    }
    std::regex pattern("^[a-z]+(_[a-z]+)*$");
    if (!std::regex_match(engineName, pattern)) {
        std::cout << "The pattern of backendName is invalid." << std::endl;
        return false;
    }
    return true;
}

bool ParamChecker::CheckKvPoolBackend(const std::string &kvPoolBackend)
{
    if (kvPoolBackend != "" && kvPoolBackend != "unifiedcache" && kvPoolBackend != "mooncake" &&
        kvPoolBackend != "memcache") {
        std::cout << "Unknow kv pool backend. And only [`unifiedcache`, `mooncake`, `memcache`, ``] is available!"
                  << std::endl;
        return false;
    }
    return true;
}

bool ParamChecker::CheckKvPoolConfigPath(const std::string &kvPoolConfigPath)
{
    if (kvPoolConfigPath.size() > 500U) {
        std::cout << "The length of kvPool Config Path exceeds 500." << std::endl;
        return false;
    }
    return true;
}

bool ParamChecker::CheckInferMode(const std::string &inferMode)
{
    if (inferMode != INFER_MODE_STANDARD && inferMode != INFER_MODE_DMI) {
        std::cout << "The inferMode should be standard or dmi" << std::endl;
        return false;
    }
    return true;
}

} // namespace mindie_llm