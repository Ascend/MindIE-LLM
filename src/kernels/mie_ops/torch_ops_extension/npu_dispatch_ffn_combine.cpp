/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace mie_ops {
using namespace at_npu::native;

// step2, 为NPU设备实现前向接口
at::Tensor& npu_dispatch_ffn_combine_npu(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out
) {
    char *group_ep_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD_V1(aclnnDispatchFFNCombine,
                    x,
                    weight1,
                    weight2,
                    expert_idx,
                    scale1,
                    scale2,
                    probs,
                    group_ep_ptr,
                    max_output_size,
                    out);
    return out;
}

// step3, 为META设备实现前向接口
at::Tensor& npu_dispatch_ffn_combine_meta(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::Tensor& probs,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out
) {
    return out;
}

}  // namespace mie_ops

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(mie_ops, PrivateUse1, m) {
    m.impl("npu_dispatch_ffn_combine", &mie_ops::npu_dispatch_ffn_combine_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(mie_ops, Meta, m) {
    m.impl("npu_dispatch_ffn_combine", &mie_ops::npu_dispatch_ffn_combine_meta);
}
