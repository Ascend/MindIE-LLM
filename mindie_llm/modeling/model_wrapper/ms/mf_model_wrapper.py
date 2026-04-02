# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import Enum
from typing import Iterable

import mindspore as ms
import numpy as np
from mindformers import ModelRunner

from ..model_info import ModelInfo
from ..wrapper import ModelWrapper
from ....utils.log.logging import logger


class SwapDirection(int, Enum):
    CPU_TO_NPU = 0
    NPU_TO_CPU = 1
    RECOMPUTE = 2  # mf不处理


class MFModelWrapper(ModelWrapper):
    def __init__(self, rank: int, world_size: int, model_id: str, cpu_mem: int, npu_mem: int, block_size: int,
                 npu_device_id: int, **kwargs):
        plugin_params = kwargs.get('plugin_params')
        self.model_runner = ModelRunner(model_path=model_id, npu_mem_size=npu_mem, cpu_mem_size=cpu_mem,
                                        block_size=block_size, rank_id=rank, world_size=world_size,
                                        npu_device_ids=[npu_device_id], plugin_params=plugin_params)
        self.config = self.model_runner.model_config
        self.tokenizer = self.model_runner.tokenizer
        self.rank = rank
        self.config_dict = self.config.__dict__
        self.model_info = ModelInfo(None,
                                    self.model_runner.dtype,
                                    ms.Tensor(0, dtype=self.model_runner.dtype).itemsize,
                                    self.model_runner.num_layers,
                                    self.model_runner.num_kv_heads,
                                    self.model_runner.head_size)
        self.max_position_embeddings = self.model_runner.model_config.max_position_embedding
        self.dp_size = 1
        self.sp_size = 1
        self.cp_size = 1
        self.is_multimodal = False

    def forward(self, model_inputs, key_cache=None, value_cache=None, **kwargs):
        valid_length_each_example = model_inputs.context_length
        input_ids, slots = self.get_model_input_ids(model_inputs)
        # new param  spec_mask q_seq_lens(mindspore代码适配， 要求必须为 q_seq_lens)
        old_name = 'q_lens'
        ms_name = 'q_seq_lens'
        if old_name in kwargs:
            kwargs[ms_name] = kwargs[old_name]
            del kwargs[old_name]
        try:
            logits = self.model_runner.forward(
                input_ids=input_ids,
                valid_length_each_example=valid_length_each_example,
                block_tables=model_inputs.block_tables,
                slot_mapping=slots,
                prefill=model_inputs.is_prefill,
                position_ids=model_inputs.position_ids,
                adapter_ids=model_inputs.adapter_ids,
                prefill_head_indices=model_inputs.prefill_head_indices,
                key_cache=key_cache,
                value_cache=value_cache,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in forward: {e}")
            raise e
        return logits

    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        if hasattr(self.model_runner, "generate_position_ids"):
            logger.debug('Using custom position ids')
            try:
                position_ids = self.model_runner.generate_position_ids(input_ids)
            except Exception as e:
                logger.error(f"Error in generate_position_ids: {e}")
                raise e
            return position_ids
        if self.rank == 0:
            logger.debug('Using default position ids')
        return range(len(input_ids))

    def get_model_input_ids(self, model_inputs):
        if hasattr(self.model_runner, 'use_legacy') and not self.model_runner.use_legacy:
            return model_inputs.input_ids.reshape(-1), model_inputs.slots.reshape(-1)
        if model_inputs.is_prefill:
            return model_inputs.input_ids.reshape(1, -1), model_inputs.slots.reshape(-1)

        return model_inputs.input_ids.reshape(-1, 1), model_inputs.slots

    def swap_cache(self, swap_decision):
        # src_block和dst_block小于0对于mf非法，需要过滤
        valid_idx = (swap_decision[:, 1] >= 0) & (swap_decision[:, 2] >= 0)
        valid_swap_decision = swap_decision[valid_idx]

        for single_swap_decision in valid_swap_decision:
            if single_swap_decision[0] == SwapDirection.CPU_TO_NPU or \
                    single_swap_decision[0] == SwapDirection.NPU_TO_CPU:
                try:
                    self.model_runner.swap(np.expand_dims(single_swap_decision[1:], axis=0),
                                           bool(single_swap_decision[0]))
                except Exception as e:
                    logger.error(f"Error in swap_cache: {e}")
                    raise e