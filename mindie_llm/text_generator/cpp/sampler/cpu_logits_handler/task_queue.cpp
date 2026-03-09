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
 
#include "task_queue.h"
#include "system_log.h"

void mindie_llm::cpu_logits_handler::TaskQueue::Init()
{
    if (pthread_mutex_init(&m_mutex, nullptr) != 0) {
        throw std::runtime_error("Failed to initialize the task queue of sampling.");
    }
}

mindie_llm::cpu_logits_handler::TaskQueue::~TaskQueue()
{
    if (pthread_mutex_destroy(&m_mutex) != 0) {
        LOG_ERROR_LLM << "Failed to destroy the task queue of sampling.";
    }
}

void mindie_llm::cpu_logits_handler::TaskQueue::AddTask(const Task &task)
{
    if (pthread_mutex_lock(&m_mutex) != 0) {
        LOG_ERROR_LLM << "Failed to lock task when adding it into task queue.";
    } else {
        m_queue.push(task);
        if (pthread_mutex_unlock(&m_mutex) != 0) {
            LOG_ERROR_LLM << "Failed to unlock task when adding it into task queue.";
        }
    }
}

void mindie_llm::cpu_logits_handler::TaskQueue::AddTask(
    Callback func,
    void *arg
    )
{
    if (pthread_mutex_lock(&m_mutex) != 0) {
        LOG_ERROR_LLM << "Failed to lock task when adding it into task queue.";
    } else {
        mindie_llm::cpu_logits_handler::Task task;
        task.function = func;
        task.arg = arg;
        m_queue.push(task);
        if (pthread_mutex_unlock(&m_mutex) != 0) {
            LOG_ERROR_LLM << "Failed to unlock task when adding it into task queue.";
        }
    }
}

mindie_llm::cpu_logits_handler::Task mindie_llm::cpu_logits_handler::TaskQueue::TakeTask()
{
    mindie_llm::cpu_logits_handler::Task t;
    if (pthread_mutex_lock(&m_mutex) != 0) {
        LOG_ERROR_LLM << "Failed to lock task when taking it out from task queue.";
    } else {
        if (m_queue.size() > 0) {
            t = m_queue.front();
            m_queue.pop();
        }
        if (pthread_mutex_unlock(&m_mutex) != 0) {
            LOG_ERROR_LLM << "Failed to unlock task when taking it out from task queue.";
        }
    }
    return t;
}

