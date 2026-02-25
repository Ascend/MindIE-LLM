# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
from typing import List, Dict

import torch

from atb_llm.models.base.graph_manager.graph_wrapper import ATBGraphWrapper
from atb_llm.models.base.graph_manager.compatible_matrix import FeatureType
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM, LayerWiseAttr, LwdLayerStatus, DistributedType


def get_layerwise_prefill_graph(config, attr: LayerWiseAttr) -> ATBGraphWrapper:
    if attr is not None and attr.split_type == DistributedType.CLOUD:
        return LayerwiseCloudPrefillGraphWrapper(config, attr)
    return LayerwiseEdgePrefillGraphWrapper(attr)


class LayerwisePrefillGraphWrapper(ATBGraphWrapper):
    def __init__(self, attr: LayerWiseAttr):
        super().__init__()
        self.feature_name = FeatureType.LAYERWISE_PREFILL
        self.feature_params = {"layerwiseDisaggregated": True, "enableLora": False}
        self.attr = attr

    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        is_prefill = kwargs.get("is_prefill", False)
        return is_prefill
    
    def add_feature_params(self, params: Dict):
        self.feature_params.update(params)


class LayerwiseEdgePrefillGraphWrapper(LayerwisePrefillGraphWrapper):
    def __init__(self, attr: LayerWiseAttr):
        super().__init__(attr)

    def set_param(self, model_type: str, params: Dict):
        head_params = params['head']
        self.head_graph = torch.classes.ModelTorch.ModelTorch(model_type)
        self.head_graph.set_param(json.dumps({**head_params, **self.feature_params}))
        
        tail_params = params['tail']
        self.tail_graph = torch.classes.ModelTorch.ModelTorch(model_type)
        self.tail_graph.set_param(json.dumps({**tail_params, **self.feature_params}))
        
    def set_weight(self, weights):
        head_weights = weights['head']
        self.head_graph.set_weight(head_weights)
        
        tail_weights = weights['tail']
        self.tail_graph.set_weight(tail_weights)
    
    def set_kv_cache(self, k_caches, v_caches):
        head_k_caches = k_caches['head']
        head_v_caches = v_caches['head']
        self.head_graph.set_kv_cache(head_k_caches, head_v_caches)
        
        tail_k_caches = k_caches['tail']
        tail_v_caches = v_caches['tail']
        self.tail_graph.set_kv_cache(tail_k_caches, tail_v_caches)
    
    def execute(self, inputs, runtime_params, **kwargs):
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage", None)
        if exe_stage is None: # warmup
            acl_model_out = self.head_graph.execute(inputs, runtime_params)
            inputs[0] = acl_model_out[0]
            acl_model_out = self.tail_graph.execute(inputs, runtime_params)
            return acl_model_out
        if exe_stage.start_exec_layer == 0: # first layer
            acl_model_out = self.head_graph.execute(inputs, runtime_params)
            return acl_model_out
        if exe_stage.end_exec_layer == 1: # last layer
            acl_model_out = self.tail_graph.execute(inputs, runtime_params)
            return acl_model_out
        return None
    

class LayerwiseCloudPrefillGraphWrapper(LayerwisePrefillGraphWrapper):
    def __init__(self, config, attr: LayerWiseAttr):
        super().__init__(attr)
        self.config = config
        self.long_seq = hasattr(config, 'rope_scaling') and config.rope_scaling.rope_type == "yarn" \
            or config.rope_scaling.type == "yarn"

        self.sine_embed_tbl = None
        self.cosine_embed_tbl = None

    def set_param(self, model_type: str, params: Dict):
        layers_num = self.attr.num_hidden_layers - self.attr.edge_start_layer_count - self.attr.edge_end_layer_count
        self.graph_list = [torch.classes.ModelTorch.ModelTorch(model_type) for _ in range(layers_num)]
        params_list = params['layers']
        for graph, layer_params in zip(self.graph_list, params_list):
            graph.set_param(json.dumps({**layer_params, **self.feature_params}))

    def set_weight(self, weights):
        weights_list = weights['layers']
        for graph, layer_weights in zip(self.graph_list, weights_list):
            graph.set_weight(layer_weights)
    
    def set_kv_cache(self, k_caches, v_caches):
        k_caches_list, v_caches_list = k_caches['layers'], v_caches['layers']
        for graph, layer_k_caches, layer_v_caches in \
            zip(self.graph_list, k_caches_list, v_caches_list):
            graph.set_kv_cache(layer_k_caches, layer_v_caches)
    
    def execute(self, inputs, runtime_params, **kwargs):
        exe_stage = kwargs.get("layerwise_disaggregated_exe_stage", None)

        if exe_stage is None: # warmup
            acl_model_out = self.graph_list[0].execute(inputs, runtime_params)
            if self.long_seq:
                inputs.append(acl_model_out[1])
                inputs.append(acl_model_out[2])
            for graph in self.graph_list[1:]:
                inputs[0] = acl_model_out[0]
                acl_model_out = graph.execute(inputs, runtime_params)
            return acl_model_out

        acl_model_out = self.graph_list[exe_stage.start_exec_layer].execute(inputs, runtime_params)
        if self.long_seq and exe_stage.start_exec_layer == 0:
            inputs.extend(acl_model_out[1:])
        for graph in self.graph_list[exe_stage.start_exec_layer + 1: exe_stage.end_exec_layer]:
            inputs[0] = acl_model_out[0]
            acl_model_out = graph.execute(inputs, runtime_params)
        return acl_model_out

