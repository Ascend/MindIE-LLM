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

#ifndef MINDIE_LLM_POST_PROCESSING_PROFILER_H
#define MINDIE_LLM_POST_PROCESSING_PROFILER_H

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <stack>
#include <thread>
#include <mutex>
#include "system_log.h"

namespace PostProcessingProfiler {
class TimeCost {
public:
    TimeCost(std::string nameStr, std::string pidStr, std::string tidStr) noexcept
        : name(nameStr), pid(pidStr), tid(tidStr)
    {
        start_ = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds d = std::chrono::duration_cast<std::chrono::microseconds>(start_.time_since_epoch());
        start = static_cast<unsigned long>(d.count());
    }

    ~TimeCost() = default;

    void ElapsedUS()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds d = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        duration = static_cast<unsigned long>(d.count());
    }

    unsigned long start = 0;
    unsigned long duration = 0;
    std::string name;
    std::string pid;
    std::string tid;

private:
    std::chrono::high_resolution_clock::time_point start_;
};

class Profiler {
public:
    Profiler(const Profiler &) = delete;

    Profiler &operator = (const Profiler &) = delete;

    ~Profiler(){};

    static Profiler &GetInstance()
    {
        static Profiler instance;
        return instance;
    }

    void TimerStart(std::string name, std::string pid, std::string tid)
    {
        if (activate) {
            std::thread::id threadId = std::this_thread::get_id();
            TimeCost tc{ name, pid, tid };
            timeStack[threadId].push(tc);
        }
    }

    void TimerEnd()
    {
        if (activate) {
            std::thread::id threadId = std::this_thread::get_id();
            auto tstack = timeStack[threadId];
            if (tstack.empty()) {
                LOG_WARN_LLM << "Inside timer, try to timerEnd without timerStart!";
                return;
            }

            auto tc = tstack.top();
            tstack.pop();
            tc.ElapsedUS();
            lk.lock();
            timeVec.push_back(tc);
            lk.unlock();
        }
    }

    std::vector<TimeCost> ExportResult()
    {
        for (auto &kv : timeStack) {
            if (!kv.second.empty()) {
                LOG_WARN_LLM << "There are unclosed timer, with name: " << kv.second.top().name;
            }
        }
        return timeVec;
    }

    void SetActivate(bool isActivate)
    {
        activate = isActivate;
    }

    void Elapsed(TimeCost tc)
    {
        tc.ElapsedUS();
        lk.lock();
        timeVec.push_back(tc);
        lk.unlock();
    }

private:
    Profiler(){};
    std::unordered_map<std::thread::id, std::stack<TimeCost>> timeStack;
    std::vector<TimeCost> timeVec;
    std::mutex lk;
    bool activate = false;
};
} // PostProcessingProfiler

#define PROFILER Profiler::GetInstance()

#endif // MINDIE_LLM_POST_PROCESSING_PROFILER_H
