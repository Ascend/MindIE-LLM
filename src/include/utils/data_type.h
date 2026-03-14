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

#ifndef MINDIE_LLM_COMMON_H
#define MINDIE_LLM_COMMON_H
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>
#include "utils/concurrent_deque.h"

namespace mindie_llm {

enum class MemType {
    HOST_MEM = 0,
};
/// The enum class of data type of the tensor.
enum class InferDataType {
    TYPE_INVALID = 0,

    TYPE_BOOL = 1,

    TYPE_UINT8 = 2,

    TYPE_UINT16 = 3,
 
    TYPE_UINT32 = 4,

    TYPE_UINT64 = 5,

    TYPE_INT8 = 6,

    TYPE_INT16 = 7,

    TYPE_INT32 = 8,

    TYPE_INT64 = 9,

    TYPE_FP16 = 10,

    TYPE_FP32 = 11,

    TYPE_FP64 = 12,

    TYPE_STRING = 13,

    TYPE_BF16 = 14,
    TYPE_BUTT,
};

/// The byte size of the data type.
const std::unordered_map<InferDataType, size_t> BYTE_SIZE_MAP = {
    {InferDataType::TYPE_INVALID, 0},
    {InferDataType::TYPE_BOOL, sizeof(bool)},
    {InferDataType::TYPE_UINT8, sizeof(uint8_t)},
    {InferDataType::TYPE_UINT16, sizeof(uint16_t)},
    {InferDataType::TYPE_UINT32, sizeof(uint32_t)},
    {InferDataType::TYPE_UINT64, sizeof(uint64_t)},
    {InferDataType::TYPE_INT8, sizeof(int8_t)},
    {InferDataType::TYPE_INT16, sizeof(int16_t)},
    {InferDataType::TYPE_INT32, sizeof(int32_t)},
    {InferDataType::TYPE_INT64, sizeof(int64_t)},
    {InferDataType::TYPE_FP16, sizeof(int16_t)},    // float16 类型不一定支持
    {InferDataType::TYPE_FP32, sizeof(float)},
    {InferDataType::TYPE_FP64, sizeof(double)},
    {InferDataType::TYPE_STRING, 0},                // 长度不确定
    {InferDataType::TYPE_BF16, sizeof(int16_t)},    // bfloat16 类型不一定支持
};

/// This function can get the byte size of the data type.
size_t GetTypeByteSize(InferDataType dataType);

/// The type of the infer request.
enum class InferReqType {
    REQ_STAND_INFER = 0,
    REQ_PREFILL = 1,
    REQ_DECODE = 2,
    REQ_FLEX_LOCAL = 3 // 配比微调Flex节点本地请求
};

struct FlexSwitchInfo {
    uint32_t flexPrefillPercentage;
};

enum FaultRecoveryCmd: int32_t {
    CMD_UNKNOWN = -1,
    CMD_PAUSE_ENGINE = 0,
    CMD_REINIT_NPU = 1,
    CMD_START_ENGINE = 2
};

struct NPUExecutionResult {
    int32_t npuDeviceId = -1;           // 设备 ID
    int32_t commandResult = -1;         // 1: success, 0: failure, -1: not executed
    std::string errorMsg;              // 仅当 result == 0 时有效
};

struct RecoverCommandInfo {
    std::string command;
    ConcurrentDeque<NPUExecutionResult> results;
    explicit RecoverCommandInfo(std::string command) : command(command) {}
};
} // namespace mindie_llm

#endif // MINDIE_LLM_COMMON_H