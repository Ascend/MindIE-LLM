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

#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <cstdint>
#include <string>

namespace mindie_llm {
const size_t LLM_SHARED_MEMORY_MAX_NAME_LEN = 255;
// Default size of single shared memory buffer is 32MB, which is sufficient for most use cases.
const size_t DEFAULT_SHARED_MEMORY_SIZE = 1024 * 1024 * 32;
// Error message shared memory only contains err string
const size_t ERROR_SHARED_MEMORY_SIZE = 1024;
 // For prefixcache in long sequence generation, one batch's shared memory may exceed 8MB.
const size_t SHARED_MEMORY_256MB = 1024 * 1024 * 256;
const size_t TOTAL_SHARED_MEMORY_PER_DP = 2 * SHARED_MEMORY_256MB + 4 * DEFAULT_SHARED_MEMORY_SIZE;
// This is set to 0.5MB. Since a single machine can host up to 16 NPUs, the total maximum memory required is 0.5MB * 16
// = 8MB, which aligns with DEFAULT_SHARED_MEMORY_SIZE.
const size_t MODEL_INIT_RESP_SIZE = 1024 * 512;
const size_t RECOVER_COMMAND_RESP_SIZE = 1024 * 512;
const size_t EXECUTE_RESP_SLOT_SIZE = 1024 * 512;
struct ShmSizeConfig {
    size_t requestShmSize;
    size_t responseShmSize;
};

struct SemaphoreConfig {
    uint32_t requestSemNum;
    uint32_t responseSemNum;
};

using FileDesc = int;
bool SharedMemorySizeCheck(const uint64_t &pendingMemoryAllocationSize);
class SharedMemory {
public:
    SharedMemory() = default;
    ~SharedMemory();
    bool Create(const std::string &name, uint32_t size);
    bool Write(uint32_t dstOffset, const char *src, uint32_t size) const;
    static bool SharedMemoryNameChecker(const std::string &name);
    bool SharedMemoryUIDAndPermissionChecker(FileDesc mFd);
    char *GetBuf();
    char *GetBufEnd() const;
    int GetFd() const;

private:
    FileDesc mFd_{0};
    std::string mName;
    uint32_t mCurSize{0};
    char *mMapBuf = nullptr;
    bool valid{false};
};
} // namespace mindie_llm

#endif
