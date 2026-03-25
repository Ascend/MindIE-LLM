/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_UTILS_EVENT_H
#define ATB_SPEED_UTILS_EVENT_H

#include <torch/torch.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include "atb_speed/base/event_manager.h"

namespace atb_speed {
class Event : public torch::CustomClassHolder {
public:
    static void Record(const std::string& pipeKey);
    static void Wait(const std::string& pipeKey);
};
} // namespace atb_speed
#endif