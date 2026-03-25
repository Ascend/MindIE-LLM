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
from atb_llm.runner.model_runner import generate_mem_pool_event_key
from mindie_llm.text_generator.plugins.plugin_manager import MemPoolType


class MemPoolGraphWrapper(ATBGraphWrapper):
    pipe_key: str = generate_mem_pool_event_key(only_save_kv=True)

    def __init__(self):
        super().__init__()
        self.feature_name = FeatureType.MOONCAKE
        self.feature_params = {"memPoolType": int(MemPoolType.ASYNC_WRITE), "pipeKey": self.pipe_key}

    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        mempool_type = kwargs.get("mempool_type", MemPoolType.DISABLED)
        return mempool_type == MemPoolType.ASYNC_WRITE
