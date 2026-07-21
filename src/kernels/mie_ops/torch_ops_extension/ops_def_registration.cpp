/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册add_custom和npu_selected_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(mie_ops, m) {
    m.def(
        "npu_mla_process(Tensor input, Tensor gamma0, Tensor beta0, Tensor wdqkv, Tensor descale0, Tensor gamma1, "
        "Tensor beta1, Tensor wuq, Tensor descale1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor "
        "kv_cache, Tensor kv_cache_rope, Tensor slotmapping, *, Tensor? quant_scale0=None, Tensor? quant_offset0=None, "
        "Tensor? bias0=None, Tensor? quant_scale1=None, Tensor? quant_offset1=None, Tensor? bias1=None, Tensor? "
        "ctkv_scale=None, Tensor? q_nope_scale=None, str? cache_mode_opt=None, str? quant_mode_opt=None) -> (Tensor, "
        "Tensor, Tensor, Tensor)");
    m.def(
        "npu_dispatch_gmm_combine_decode(Tensor x, Tensor expert_ids, Tensor[] gmm1_permuted_weight, Tensor[] "
        "gmm1_permuted_weight_scale, Tensor[] gmm2_weight, Tensor[] gmm2_weight_scale, Tensor expert_scales, Tensor? "
        "expert_smooth_scales=None, Tensor? x_active_mask=None, str group_ep='', int ep_rank_size=0, int ep_rank_id=0, "
        "int moe_expert_num=0, int shared_expert_num=1, int shared_expert_rank_num=0, int quant_mode=0, int "
        "global_bs=0) -> (Tensor output, Tensor expert_token_nums)");
    m.def(
        "npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, str layout_query='BSND', str "
        "layout_key='PA_BSND', int selected_count=2048, int sparse_mode=3) -> Tensor");
    m.def(
        "npu_dispatch_ffn_combine("
        "Tensor x, "
        "Tensor[] weight1, "
        "Tensor[] weight2, "
        "Tensor expert_idx, "
        "Tensor[] scale1, "
        "Tensor[] scale2, "
        "Tensor probs, "
        "str group, "
        "int max_output_size, "
        "Tensor! out"
        ") -> Tensor");
    m.def(
        "apply_top_k_top_p_custom("
        "Tensor logits, "
        "Tensor? p, "
        "Tensor? k, "
        "Tensor! out"
        ") -> Tensor");
}
