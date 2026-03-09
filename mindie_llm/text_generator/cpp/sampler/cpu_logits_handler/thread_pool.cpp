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
 
#include "thread_pool.h"

#include <iostream>
#include <stdexcept>

#include "securec.h"
#include "system_log.h"

const int MAX_NUM_THREADS = 256;

using namespace std;

void mindie_llm::cpu_logits_handler::ThreadPool::Init()
{
    m_taskQ = new TaskQueue;
    m_taskQ->Init();
    do {
        if (m_threadNum < 1 || m_threadNum > MAX_NUM_THREADS) {
            LOG_ERROR_LLM << "The threadNum exceeds the limit of 1 to " << MAX_NUM_THREADS;
            throw runtime_error("The threadNum exceeds the limit.");
        }
        m_threadIDs = new pthread_t[m_threadNum];
        int ret = memset_s(m_threadIDs, sizeof(pthread_t) * m_threadNum, 0, sizeof(pthread_t) * m_threadNum);
        if (ret != 0) {
            LOG_ERROR_LLM << "memset_s thread_t[] failed";
            throw runtime_error("memset_s thread_t[] failed.");
        }

        if (pthread_mutex_init(&m_lock, nullptr) != 0 || pthread_mutex_init(&m_taskLock, nullptr) != 0 ||
            pthread_cond_init(&m_notEmpty, nullptr) != 0 || pthread_cond_init(&m_empty, nullptr) != 0) {
            LOG_ERROR_LLM << "Init mutext or condition failed";
            throw runtime_error("Init mutext or condition failed.");
        }

        for (int i = 0; i < m_threadNum; ++i) {
            ret = pthread_create(&m_threadIDs[i], nullptr, Worker, this);
            if (ret != 0) {
                LOG_ERROR_LLM << "pthread_create failed, error code: " << ret;
                throw runtime_error("pthread_create failed with error " + std::to_string(ret));
            }
        }
    } while (0);
}

mindie_llm::cpu_logits_handler::ThreadPool::~ThreadPool()
{
    m_shutdown = 1;

    pthread_cond_broadcast(&m_notEmpty);
    if (m_threadIDs) {
        for (int i = 0; i < m_threadNum; ++i) {
            if (m_threadIDs[i] != 0) {
                pthread_join(m_threadIDs[i], nullptr);
                }
            }
    }

    if (m_taskQ) {
        delete m_taskQ;
        m_taskQ = nullptr;
    }
    if (m_threadIDs) {
        delete[] m_threadIDs;
        m_threadIDs = nullptr;
    }
    pthread_mutex_destroy(&m_lock);
    pthread_cond_destroy(&m_notEmpty);
}

void mindie_llm::cpu_logits_handler::ThreadPool::AddTask(Task task)
{
    if (m_shutdown) {
        return;
    }
    pthread_mutex_lock(&m_taskLock);
    m_taskNum++;
    pthread_mutex_unlock(&m_taskLock);
    m_taskQ->AddTask(task);
    pthread_cond_signal(&m_notEmpty);
}

void *mindie_llm::cpu_logits_handler::ThreadPool::Worker(void *arg)
{
    mindie_llm::cpu_logits_handler::ThreadPool *pool = \
    static_cast<mindie_llm::cpu_logits_handler::ThreadPool *>(arg);
    while (true) {
        pthread_mutex_lock(&pool->m_lock);

        while (pool->m_taskQ->TaskNumber() == 0 && !pool->m_shutdown) {
            pthread_cond_wait(&pool->m_notEmpty, &pool->m_lock);
        }

        if (pool->m_shutdown) {
            pthread_mutex_unlock(&pool->m_lock);
            pool->ThreadExit();
            break;
        }

        Task task = pool->m_taskQ->TakeTask();
        pthread_mutex_unlock(&pool->m_lock);
        task.function(task.arg);
        task.arg = nullptr;

        pthread_mutex_lock(&pool->m_taskLock);
        pool->m_taskNum--;
        if (pool->m_taskNum == 0) {
            pthread_cond_signal(&pool->m_empty);
        }
        pthread_mutex_unlock(&pool->m_taskLock);
    }

    return nullptr;
}

void mindie_llm::cpu_logits_handler::ThreadPool::ThreadExit()
{
    pthread_t tid = pthread_self();
    for (int i = 0; i < m_threadNum; ++i) {
        if (m_threadIDs[i] == tid) {
            m_threadIDs[i] = 0;
            break;
        }
    }
}

void mindie_llm::cpu_logits_handler::ThreadPool::Join()
{
    pthread_mutex_lock(&m_taskLock);
    while (m_taskNum != 0) {
        pthread_cond_wait(&m_empty, &m_taskLock);
    }
    pthread_mutex_unlock(&m_taskLock);
}
