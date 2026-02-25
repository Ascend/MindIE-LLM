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
#include "safe_io.h"

#include <fstream>

#include "safe_path.h"
#include "log.h"

namespace mindie_llm {

static constexpr int JSON_STR_DEP_MAX = 10; // 限制嵌套层次10层

Result LoadJson(const std::string& path, Json& json)
{
    std::string checkedPath;
    SafePath inFile(path, PathType::FILE, "r", PERM_640, SIZE_500MB, ".json");
    Result r = inFile.Check(checkedPath);
    if (!r.IsOk()) {
        return r;
    }
    std::ifstream file(checkedPath);
    if (!file.is_open()) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to open file: " + checkedPath);
    }
    json = nlohmann::json::parse(file, CheckJsonDepthCallbackNoLogger);
    return Result::OK();
}

bool CheckJsonDepth(int depth, Json::parse_event_t ev)
{
    switch (ev) {
        case Json::parse_event_t::object_start:
        case Json::parse_event_t::array_start:
            return depth <= JSON_STR_DEP_MAX;
        default:
            return true;
    }
}

bool CheckJsonDepthWithLogger(int depth, Json::parse_event_t ev, std::function<void(void)> logger)
{
    if (!CheckJsonDepth(depth, ev)) {
        if (logger) {
            logger();
        }
        return false;
    }
    return true;
}

bool CheckJsonDepthCallbackNoLogger(int depth, Json::parse_event_t ev, Json& obj)
{
    return CheckJsonDepthWithLogger(depth, ev, [depth, &obj]() {
        std::cerr << "Failed to parse json: depth is " << depth <<  ", object is " << sizeof(obj);
    });
}

bool CheckJsonDepthCallback(int depth, Json::parse_event_t ev, Json& obj)
{
    return CheckJsonDepthWithLogger(depth, ev, [depth, &obj]() {
        MINDIE_LLM_LOG_ERROR("Failed to parse json: depth is " << depth <<  ", object is " << sizeof(obj));
    });
}

bool CheckJsonDepthCallbackUlog(int depth, Json::parse_event_t ev, Json& obj)
{
    return CheckJsonDepthWithLogger(depth, ev, [depth, &obj]() {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to parse json: depth is " << depth <<  ", object is " << sizeof(obj));
    });
}

bool CheckOrderedJsonDepthCallback(int depth, OrderedJson::parse_event_t ev, OrderedJson& obj)
{
    return CheckJsonDepthWithLogger(depth, ev, [depth, &obj]() {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to parse json: depth is " << depth <<  ", object is " << sizeof(obj));
    });
}

} // namespace mindie_llm
