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

#include "memory_utils.h"

#include <algorithm>
#include <climits>
#include <ctime>
#include <fcntl.h> /* For O_* constants */
#include <regex>
#include <sys/mman.h>
#include <thread>

#include "file_utils.h"
#include "system_log.h"

namespace mindie_llm {

void SemP(sem_t &sem, short int num)
{
    for (short int i = 0; i < num; i++) {
        if (sem_wait(&sem) != 0) {
            LOG_ERROR_LLM << "Execute sem_wait failed iteration=" << i << "/" << num;
        }
    }
}

void SemV(sem_t &sem, short int num)
{
    for (short int i = 0; i < num; i++) {
        if (sem_post(&sem) != 0) {
            LOG_ERROR_LLM << "Execute sem_post failed iteration=" << i << "/" << num;
        }
    }
}

int MemcpyAfterCheck(char *dest, uint64_t destBufLen, uint64_t destCurPos, const char *src, size_t count)
{
    if (dest == nullptr) {
        LOG_ERROR_LLM << "The 'dest' cannot be nullptr";
        return -1;
    }
    if (src == nullptr) {
        LOG_ERROR_LLM << "The 'src' cannot be nullptr";
        return -1;
    }
    if (destBufLen < destCurPos) {
        LOG_ERROR_LLM << "The 'destBufLen' cannnot be smaller than 'destCurPos'";
        return -1;
    }

    return memcpy_s(dest + destCurPos, destBufLen - destCurPos, src, count);
}

bool CheckMemorySize(const char *pointer, uint64_t byteSize, const char *start, const char *end)
{
    if (pointer == nullptr || start == nullptr || end == nullptr) {
        LOG_ERROR_LLM << "The 'pointer' or 'start' or 'end' cannot be nullptr";
        return false;
    }
    if (pointer > end || pointer < start) {
        LOG_ERROR_LLM << "The 'pointer' should be in the range of [start, end]";
        return false;
    }
    if (pointer + byteSize < start || pointer + byteSize > end) {
        LOG_ERROR_LLM << "The 'pointer + byteSize' should be in the range of [start, end]";
        return false;
    }
    return true;
}

bool IsValidNonNegativeInteger(const std::string &input)
{
    static std::string maxInt32Str = "2147483647"; // int32 max value
    if (input.size() == 0 || input.size() > maxInt32Str.size() ||
        (input.size() == maxInt32Str.size() && input > maxInt32Str)) {
        LOG_ERROR_LLM << "Slave worker initialization result " << input << " is invalid";
        return false;
    }
    // 正则表达式模式
    std::regex pattern("^(0|[1-9][0-9]*)$");

    // 使用 regex_match 来检查整个字符串是否匹配
    return std::regex_match(input, pattern);
}
} // namespace mindie_llm
