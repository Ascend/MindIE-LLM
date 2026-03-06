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

#ifndef PARAM_CHECKER_H
#define PARAM_CHECKER_H
#include <regex>
#include <iostream>
#include <string>
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace mindie_llm {
struct ParamSpec {
    std::string name;
    std::string Type;
    bool compulsory;
};

#define CHECK_CONFIG_VALIDATION(checkRes, expr)                                                                        \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            (checkRes) = false;                                                                                        \
        }                                                                                                              \
    } while (0)

class ParamChecker {
public:
    /*
    @brief: 从json文件中读取configType对象到inputJsonData中
    */
    static bool ReadJsonFile(const std::string &jsonPath, std::string &baseDir, nlohmann::json &inputJsonData,
                             std::string configType);

    /*
    @brief: 校验jsonData（array）中的数据类型是否为eleType（可选string、integer、bool）
    */
    static bool CheckJsonArray(nlohmann::json jsonData, const std::string &eleType, const std::string &integerType);

    /*
    @brief: 从json文件中读取数据到jsonData中
    */
    static bool GetJsonData(const std::string &configFile, std::string &baseDir, nlohmann::json &jsonData,
                            const bool &skipPermissionCheck = false);

    /*
    @brief: 检查各个文件夹下的lora_adapter.json中内容是否符合预期，比较特殊，单独使用函数。
    */
    static bool CheckAndGetLoraJsonFile(std::string &baseDir, nlohmann::json &loraJsonData);

    // ToDo: jsonData 换成引用
    /*
    @brief: 获取jsonData中的configName名称的值，若不存在，则以默认值代替
    */
    static uint32_t GetIntegerParamDefaultValue(nlohmann::json jsonData, const std::string &configName,
                                                uint32_t defaultVal);
    
    static int32_t GetTruncationParamDefaultValue(nlohmann::json jsonData, const std::string &configName,
                                                uint32_t defaultVal);

    static std::string GetStringParamValue(nlohmann::json jsonData, const std::string &configName,
                                           std::string defaultVal);

    static bool GetBoolParamValue(nlohmann::json jsonData, const std::string &configName, bool defaultVal);

    /*
    @brief: 校验输入inputValue的值是否在最大最小值区间内
    */
    template <typename Integer>
    static bool CheckMaxMinValue(Integer inputValue, Integer maxValue, Integer minValue, const std::string &inputName)
    {
        if (inputValue > maxValue) {
            std::cout << inputName << " [" << std::to_string(inputValue) << "] is out of bound ["
                      << std::to_string(maxValue) << "]" << std::endl;
            return false;
        } else if (inputValue < minValue) {
            std::cout << inputName << " [" << std::to_string(inputValue) << "] is less than ["
                      << std::to_string(minValue) << "]" << std::endl;
            return false;
        }
        return true;
    }

    /*
    @brief: 校验jsonValue的值是否满足其类型integerType的取值范围
    */
    static bool IsWithinRange(std::string integerType, nlohmann::json jsonValue);
    // ToDo: jsonData 换成引用
    /*
    @brief: 校验array里的元素范围
    */
    static bool CheckNpuRange(nlohmann::json jsonValue);

    /*
    @brief: 校验array是否有效
    */
    static bool IsArrayValid(const std::string &configName, nlohmann::json jsonValue);

    /*
    @brief: 校验jsonData里ParamSpec描述的参数类型是否都正确
    */
    static bool CheckJsonParamType(nlohmann::json &jsonData, std::vector<ParamSpec> &paramSpecs);

    /*
    @brief: 校验路径path是否在系统中存在
    */
    static bool CheckPath(const std::string &path, std::string &baseDir, const std::string &inputName, bool flag = true,
                          uint64_t maxFileSize = 10 * 1024 * 1024);

    static bool CheckPolicyValue(uint32_t inputValue, const std::string &inputName);

    static bool CheckMixPolicyValue(uint32_t inputValue, const std::string &inputName);

    static bool CheckEngineName(const std::string &engineName);

    static bool CheckKvPoolBackend(const std::string &kvPoolBackend);

    static bool CheckKvPoolConfigPath(const std::string &kvPoolConfigPath);

    /*
    @brief: 校验PD 相关配置有效性
    */
    static bool CheckInferMode(const std::string &inferMode);
};
} // namespace mindie_llm
#endif