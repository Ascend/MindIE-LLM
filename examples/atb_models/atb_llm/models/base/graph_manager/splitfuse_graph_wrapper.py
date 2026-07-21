# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.models.base.graph_manager.graph_wrapper import ATBGraphWrapper
from atb_llm.models.base.graph_manager.compatible_matrix import FeatureType
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM
from atb_llm.utils.mempool_utils import generate_mem_pool_event_key


class SplitFuseGraphWrapper(ATBGraphWrapper):
    "ATBGraphWrapper class for prefixcache and splitfuse"

    pipe_key: str = generate_mem_pool_event_key(only_save_kv=False)

    def __init__(self):
        super().__init__()
        self.feature_name = FeatureType.SPLITFUSE
        self.feature_params = {"enableSplitFuse": True, "isPrefill": True, "pipekey": self.pipe_key}

    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        pa_enable = False if context.inference_mode is None else context.inference_mode.enable_prefill_pa
        q_lens = '"qLen"' in runtime_params
        is_prefill = kwargs.get("is_prefill", False)
        if q_lens and is_prefill and pa_enable:
            return True
        return False
