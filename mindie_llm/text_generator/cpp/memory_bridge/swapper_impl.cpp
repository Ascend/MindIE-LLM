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

#include <vector>
#include <mutex>
#include "acl/acl.h"
#include "system_log.h"
#include "swapper.h"

namespace hmm {

    union UnUInt64VPtr {
        uint64_t uInt64Val;
        void* vPtrVal;

        explicit UnUInt64VPtr(const uint64_t& value) : uInt64Val(value) {}
    };

    class SwapperImpl : public Swapper {
    public:
        SwapperImpl() = default;
        ~SwapperImpl() noexcept override = default;

        // params: cpuRowBytes, npuRowBytes, miniBlockBytes, numLayers * 2
        void H2dSwap(
            const uint64_t &cpuBlock,
            const uint64_t &npuBlock,
            const std::vector<uint64_t> &params) const override
        {
            if (params.size() != 4U) {
                LOG_ERROR_LLM << "params.size() is not 4.";
                throw std::invalid_argument("Invalid params size, expected 4 elements.");
            }
            UnUInt64VPtr cBlock(cpuBlock);
            UnUInt64VPtr nBlock(npuBlock);
            aclrtMemcpy2d(nBlock.vPtrVal, params[1U], cBlock.vPtrVal, params[0U], params[2U], params[3U],
                          ACL_MEMCPY_HOST_TO_DEVICE);
        }

        // params: cpuRowBytes, npuRowBytes, miniBlockBytes, numLayers * 2
        void D2hSwap(
            const uint64_t &cpuBlock,
            const uint64_t &npuBlock,
            const std::vector<uint64_t> &params) const override
        {
            if (params.size() != 4U) {
                LOG_ERROR_LLM << "params.size() is not 4.";
                throw std::invalid_argument("Invalid params size, expected 4 elements.");
            }
            UnUInt64VPtr cBlock(cpuBlock);
            UnUInt64VPtr nBlock(npuBlock);
            aclrtMemcpy2d(cBlock.vPtrVal, params[0U], nBlock.vPtrVal, params[1U], params[2U], params[3U],
                          ACL_MEMCPY_DEVICE_TO_HOST);
        }
    };

    std::shared_ptr<Swapper> &Swapper::Create()
    {
        static std::shared_ptr<Swapper> swapperInstance = nullptr;
        if (swapperInstance == nullptr) {
            swapperInstance.reset(new (std::nothrow) SwapperImpl());
        }
        return swapperInstance;
    }

} // namespace hmm