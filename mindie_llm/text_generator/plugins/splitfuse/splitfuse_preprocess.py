# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from ...utils.model_input import ModelInput
from ...utils.input_metadata import InputMetadata
from ....utils.decorators.time_decorator import timer
from ....utils.env import ENV
from ....modeling.backend_type import BackendType


class SplitFusePreprocess:
    def __init__(self, infer_context, model_wrapper, kvcache_settings):
        self.infer_context = infer_context
        self.model_wrapper = model_wrapper
        self.kvcache_settings = kvcache_settings
        if ENV.framework_backend == BackendType.MS:
            self.model_wrapper.device = None
        self.device = self.model_wrapper.device
        self.is_300i = False
        if ENV.framework_backend == BackendType.ATB:
            self.is_300i = self.model_wrapper.model_runner.soc_info.is_300i()
        self.async_inference = self.infer_context.context_params.async_infer

    def make_attention_mask(
        self,
        model_inputs: ModelInput,
        input_metadata: InputMetadata,
        q_lens,
        hit_mask=None
    ):
        req_mask = None
        if ENV.framework_backend == BackendType.MS:
            return req_mask
        if input_metadata.is_prefill:
            kv_device = self.model_wrapper.device
            if self.is_300i:
                batch_size = len(q_lens)
                kv_dtype = self.kvcache_settings.dtype
                atten_mask = self.model_wrapper.model_runner.attn_mask.get_attn_mask(model_inputs.max_seq_len,
                                                                                     kv_dtype, kv_device)
                if model_inputs.max_seq_len > 1 and atten_mask[0][1] > 0:
                    atten_mask = atten_mask * -10000.0
                req_mask_list = []
                for i in range(batch_size):
                    start = model_inputs.context_length[i] - q_lens[i]
                    end = model_inputs.context_length[i]
                    if self.async_inference and (hit_mask is None):
                        message = ("Inference requires 'hit_mask' to be provided, but got None")
                        raise ValueError(message)
                    if self.async_inference and hit_mask[i] and not input_metadata.batch_is_prefill[i]:
                        start += 1
                        end += 1
                    req_mask_list.append(atten_mask[start:end])
                import torch
                req_mask = torch.cat(req_mask_list, 0)
            else:
                req_mask = self.model_wrapper.model_runner.attn_mask.get_splitfuse_mask(kv_device)
        return req_mask

    @timer.track_time('preprocess')
    def splitfuse_preprocess(self, input_metadata: InputMetadata, warmup=False, hit_mask=None):

        cache_ids = self.infer_context.get_batch_context_handles(input_metadata)

        model_inputs, sampling_metadata, q_len, trace_ids = \
            self.infer_context.splitfuse_concatenate(input_metadata, cache_ids, warmup=warmup, hit_mask=hit_mask)
        attention_mask = self.make_attention_mask(model_inputs, input_metadata, q_len, hit_mask=hit_mask)
        res = (model_inputs, cache_ids, sampling_metadata, q_len, attention_mask, trace_ids)
        return res
