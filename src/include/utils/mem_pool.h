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

#ifndef MEM_POOL_H
#define MEM_POOL_H

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>

#include "log.h"

namespace py = pybind11;

namespace mindie_llm {

#pragma GCC visibility push(default)
class MemPool {
   public:
    explicit MemPool(std::shared_ptr<py::object> instance) : impl_(std::move(instance)) {
        enable_batch_lookup_ = py::hasattr(*impl_, "batch_exist");
    }

    std::vector<bool> LookUp(const std::vector<std::string> &keys) {
        PyGILState_STATE state = PyGILState_Ensure();
        std::vector<bool> res(keys.size(), false);
        if (keys.empty()) {
            return res;
        }

        if (this->enable_batch_lookup_) {
            auto tmp = impl_->attr("batch_exist")(keys).cast<std::vector<bool>>();
            res.assign(tmp.begin(), tmp.end());
        } else {
            size_t i = 0;
            bool single_res = true;
            while (i < keys.size() && single_res) {
                single_res = impl_->attr("exists")(keys[i]).cast<bool>();
                res[i] = single_res;
                i++;
            }
        }
        PyGILState_Release(state);
        return res;
    }

   private:
    std::shared_ptr<py::object> impl_{};
    bool enable_batch_lookup_ = false;
};

using MemPoolSPtr = std::shared_ptr<MemPool>;
#pragma GCC visibility pop
}  // namespace mindie_llm
#endif
