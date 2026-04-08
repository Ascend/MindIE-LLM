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

#ifndef SAFE_IO_H
#define SAFE_IO_H

#include <functional>
#include "nlohmann/json.hpp"

#include "safe_result.h"

namespace mindie_llm {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

Result LoadJson(const std::string& path, Json& json);

void SetJsonDepthLimit(int depth);
int GetJsonDepthLimit();

bool CheckJsonDepth(int depth, Json::parse_event_t ev);
bool CheckJsonDepthWithLogger(int depth, Json::parse_event_t ev, std::function<void(void)> logger);
bool CheckJsonDepthCallbackNoLogger(int depth, Json::parse_event_t ev, Json& obj);
bool CheckJsonDepthCallback(int depth, Json::parse_event_t ev, Json& obj);
bool CheckJsonDepthCallbackUlog(int depth, Json::parse_event_t ev, Json& obj);
bool CheckOrderedJsonDepthCallback(int depth, OrderedJson::parse_event_t ev, OrderedJson& obj);

} // namespace mindie_llm

#endif
