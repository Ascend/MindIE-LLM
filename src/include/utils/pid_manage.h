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

#ifndef MINDIE_PID_MANAGE_H
#define MINDIE_PID_MANAGE_H

#include <sys/types.h>

#include <mutex>
#include <unordered_set>

namespace mindie_llm {
class PidManager {
   public:
    // 获取单例实例
    static PidManager& Instance();

    // 添加 PID（若已存在则忽略）
    void AddIgnorePid(pid_t pid);
    // 判断 PID 是否在列表中
    bool IsIgnorePid(pid_t pid) const;

   private:
    // 私有构造函数和析构函数
    PidManager() = default;
    ~PidManager() = default;
    // 删除拷贝构造和赋值操作，确保单例
    PidManager(const PidManager&) = delete;
    PidManager& operator=(const PidManager&) = delete;

    // 存储 PID 的无序集合（自动去重）
    std::unordered_set<pid_t> ignore_pids_;
    // 互斥锁，保护共享数据
    mutable std::mutex mutex_;
};
}  // namespace mindie_llm

#endif  // MINDIE_PID_MANAGE_H
