/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "pid_manage.h"

namespace mindie_llm {
// 获取单例实例
PidManager& PidManager::Instance() {
    static PidManager instance;
    return instance;
}

// 添加 PID（若已存在则忽略）
void PidManager::AddIgnorePid(pid_t pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    ignore_pids_.insert(pid);
}

// 判断 PID 是否在列表中
bool PidManager::IsIgnorePid(pid_t pid) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ignore_pids_.find(pid) != ignore_pids_.end();
}

}  // namespace mindie_llm
