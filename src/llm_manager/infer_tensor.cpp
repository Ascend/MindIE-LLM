/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "llm_manager/infer_tensor.h"

#include <cstring>

#include "memory_utils.h"
#include "system_log.h"
#include "check_utils.h"
namespace mindie_llm {

constexpr uint32_t MAX_INPUTS_NUM = 4 * 1024 * 1024;
constexpr uint32_t MAX_BYTE_ALLOWED = MAX_INPUTS_NUM * sizeof(int64_t);
constexpr uint32_t MAX_DIMCOUNT = 10000;
InferTensor::InferTensor(std::string name, InferDataType dataType, std::vector<int64_t> dataShape)
{
    if (!CheckStringInputLength(name, MAX_STRING_LENGTH)) {
        LOG_ERROR_LLM << "The Input name of inferTensor: " << name << "is too long.";
        return;
    }
    if (dataShape.size() > MAX_DIMCOUNT) {
        LOG_ERROR_LLM << "The Input dataShape of inferTensor: " << name << "is too long";
        return;
    }
    this->name = name;
    this->dataType = dataType;
    this->dataShape = dataShape;
}

const std::vector<int64_t> &InferTensor::GetShape() const
{
    return dataShape;
}

size_t InferTensor::GetSize() const
{
    return byteSize;
}

MemType InferTensor::GetMemType() const
{
    return MemType::HOST_MEM;
}

InferDataType InferTensor::GetDataType() const
{
    return dataType;
}

const std::string &InferTensor::GetName() const
{
    return name;
}

void* InferTensor::GetData() const
{
    return data;
}

bool InferTensor::Truncate(const size_t truncLen) // for tensor of INPUT_IDS.
{
    if (dataShape.size() == 0) {
        LOG_ERROR_LLM << "Truncate: dataShape is empty.";
        return false;
    }
    if (truncLen > MAX_INPUTS_NUM) {
        LOG_ERROR_LLM << "Truncate: truncLen is too large.";
        return false;
    }
    if (data == nullptr) {
        return false;
    }
    const size_t truncByteSize = truncLen * GetTypeByteSize(dataType);
    if (truncByteSize > byteSize) {
        return true;
    }

    void* truncatedData = malloc(truncByteSize);
    if (truncatedData == nullptr) {
        return false;
    }
    auto ret = memmove_s(truncatedData, truncByteSize, data, truncByteSize);
    if (ret > 0) {
        free(truncatedData);
        return false;
    }
    free(data);
    data = truncatedData;

    byteSize = truncByteSize;
    if (dataShape.size() > 1) {
        dataShape[1] = truncLen;
    } else {
        dataShape[0] = truncLen;
    }
    LOG_INFO_LLM << "Input truncation success: truncated length =" << truncLen;
    return true;
}

bool InferTensor::Allocate(size_t size)
{
    if (size > 0 && size <= MAX_BYTE_ALLOWED) {
        data = malloc(size);
        if (data == nullptr) {
            return false;
        }
        if (memset_s(data, size, 0, size) != EOK) {
            free(data);
            return false;
        }
        byteSize = size;
        needRelease = true;
        return true;
    }
    return false;
}

void InferTensor::SetBuffer(const void *buffer, size_t tensorbyteSize, bool tensorNeedRelease)
{
    if (buffer == nullptr) {
        LOG_ERROR_LLM << "SetBuffer fail: buffer is nullptr";
        return;
    }
    if (tensorbyteSize > MAX_BYTE_ALLOWED) {
        LOG_ERROR_LLM << "SetBuffer fail: tensorbyteSize is too large";
        return;
    }
    data = const_cast<void *>(buffer);
    byteSize = tensorbyteSize;
    this->needRelease = tensorNeedRelease;
}

void InferTensor::SetRelease(bool releaseFlag)
{
    this->needRelease = releaseFlag;
}
void InferTensor::Release()
{
    if (data != nullptr && needRelease) {
        free(data);
        data = nullptr;
    }
}

InferTensor::~InferTensor()
{
    Release();
}

size_t InferTensor::GetTypeByteSize(InferDataType inferDataType)
{
    auto iter = BYTE_SIZE_MAP.find(inferDataType);
    if (iter == BYTE_SIZE_MAP.end()) {
        return 0;
    }
    return iter->second;
}

} // namespace mindie_llm
