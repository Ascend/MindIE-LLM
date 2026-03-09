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

#include "common_util.h"
#include "base_config_manager.h"

using Json = nlohmann::json;
using namespace nlohmann::literals;

namespace mindie_llm {
bool BaseConfig::CheckSystemConfig(const std::string &jsonPath, Json &inputJsonData, std::string paramType)
{
    std::string homePath;
    if (!GetHomePath(homePath).IsOk()) {
        return false;
    }
    std::string systemConfigPath = homePath + "/conf/config.json";
    std::string baseDir = "/";
    if (systemConfigPath.compare(jsonPath) == 0) {
        baseDir = homePath;
    }
    return ParamChecker::ReadJsonFile(jsonPath, baseDir, inputJsonData, paramType);
}
} // namespace mindie_llm
