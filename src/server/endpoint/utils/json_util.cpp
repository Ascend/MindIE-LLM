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

#include "json_util.h"
#include "log.h"

namespace mindie_llm {

bool CheckJsonDepthCallback(int depth, Json::parse_event_t ev, [[maybe_unused]] Json& obj)
{
    return CheckJsonDepthWithLogger(depth, ev, [depth]() {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to parse json: depth is " << depth <<  ", limit is " << GetJsonDepthLimit());
    });
}

bool CheckOrderedJsonDepthCallback(int depth, OrderedJson::parse_event_t ev, [[maybe_unused]] OrderedJson& obj)
{
    return CheckJsonDepthWithLogger(depth, ev, [depth]() {
        ULOG_ERROR(SUBMODLE_NAME_ENDPOINT, GenerateEndpointErrCode(ERROR, SUBMODLE_FEATURE_SINGLE_INFERENCE,
            JSON_PARSE_ERROR), "Failed to parse json: depth is " << depth <<  ", limit is " << GetJsonDepthLimit());
    });
}

} // namespace mindie_llm
