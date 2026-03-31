# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_npu

from ..samplers.sampler import Sampler
from ..utils.config import SamplerConfig
from ..utils.model_input import ModelInput
from ..utils.sampling_output import SamplingOutput
from ..utils.sampling_metadata import SamplingMetadata, SamplingData, SamplingParam
from ...modeling.model_wrapper import get_model_wrapper
from ...utils.decorators.time_decorator import timer
from ...utils.log.error_code import ErrorCode
from ...utils.log.logging import logger
from ...utils.tensor import op
from ...utils.validation import parse_config, ParseType, MODEL_CONFIG_KEY_TYPE
from .recovery_utils import check_and_recover_uce_in_cache

MAX_WORLD_SIZE = 1048576
MAX_KEY_LENGTH = 256


class GeneratorBackend:
    """The base class for a generator backend.

    This class provides basic implementation of forward inference and sampling. All methods here can be overridden by
    subclasses, so it should not be instantiated directly.

    Args:
        model_config: A dictionary containing the model configuration as detailed in
            `mindie_llm.text_generator.utils.config.ModelConfig`.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        self.model_name = parse_config(model_config, 'model_name', required=False, parse_type=ParseType.TO_STR)
        backend_type = parse_config(model_config, 'backend_type', required=True)
        num_threads = parse_config(model_config, 'num_threads', parse_type=ParseType.TO_INT, default_value=8)
        self.npu_device_id = parse_config(model_config, 'npu_device_id', required=True, parse_type=ParseType.TO_INT)
        local_rank = parse_config(model_config, 'local_rank', required=True, parse_type=ParseType.TO_INT)
        self.rank = parse_config(model_config, 'rank', required=True, parse_type=ParseType.TO_INT)
        self.world_size = parse_config(model_config, 'world_size', required=True, parse_type=ParseType.TO_INT)
        self.trust_remote_code = parse_config(model_config, 'trust_remote_code', required=True,
                                              parse_type=ParseType.TO_BOOL, default_value=False)
        self.distributed_enable = parse_config(model_config, 'distributed_enable', required=False, 
                                                parse_type=ParseType.TO_BOOL, default_value=False)
        self.max_batch_size = parse_config(model_config, 'max_batch_size', required=False, 
                                                parse_type=ParseType.TO_INT, default_value=0)
        self.local_super_device_id = parse_config(model_config, 'local_super_device_id', required=False, 
                                                  parse_type=ParseType.TO_STR, default_value=None)
        self.block_size = parse_config(model_config, 'block_size', required=False,
                                       parse_type=ParseType.TO_INT, default_value=128)
        self.splitfuse_enabled = parse_config(model_config, 'splitfuse_enabled', required=False,
                                              parse_type=ParseType.TO_BOOL, default_value=False)
        self.kv_pool_backend = parse_config(model_config, 'kv_pool_backend', required=False, 
                                        parse_type=ParseType.TO_STR, default_value='')
        self.kv_pool_config_path = parse_config(model_config, 'kv_pool_config_path', required=False, 
                                        parse_type=ParseType.TO_STR, default_value='')

        if self.world_size < 1 or self.world_size > MAX_WORLD_SIZE:
            raise ValueError("World size should be in the range of 1 to 1048576.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("Rank should be in the range of 0 to world_size - 1.")
        dp = parse_config(model_config, 'dp', required=False, parse_type=ParseType.TO_INT, default_value=-1)
        self.dp = dp
        tp = parse_config(model_config, 'tp', required=False, parse_type=ParseType.TO_INT, default_value=-1)
        attn_inner_sp = parse_config(model_config, 'sp', required=False, parse_type=ParseType.TO_INT, default_value=-1)
        cp = parse_config(model_config, 'cp', required=False, parse_type=ParseType.TO_INT, default_value=-1)
        moe_tp = parse_config(model_config, 'moe_tp', required=False, parse_type=ParseType.TO_INT, default_value=-1)
        moe_ep = parse_config(model_config, 'moe_ep', required=False, parse_type=ParseType.TO_INT, default_value=-1)
        soc_version = parse_config(model_config, 'soc_version', required=False, parse_type=ParseType.TO_INT,
                                   default_value=-1)
        num_lccl_comm_shards = parse_config(
            model_config, 'num_lccl_comm_shards', parse_type=ParseType.TO_INT, default_value=1)
        lccl_comm_shard_id = parse_config(
            model_config, 'lccl_comm_shard_id', parse_type=ParseType.TO_INT, default_value=0)
        if local_rank < 0 or local_rank >= self.world_size:
            raise ValueError("Local rank should be in the range of 0 to world_size - 1.")
        max_loras = parse_config(model_config, 'max_loras', required=False, parse_type=ParseType.TO_INT, default_value=0)
        max_lora_rank = parse_config(model_config, 'max_lora_rank', required=False, parse_type=ParseType.TO_INT, default_value=0)
        self.__parse_config_key(model_config)
        lwd_next_p_head_prior = parse_config(model_config, 'lwdNextPHeadPrior', 
                                         parse_type=ParseType.TO_BOOL, default_value=False)

        sampler_config = SamplerConfig(
            backend_type=backend_type,
            npu_id=self.npu_device_id,
            num_threads=num_threads,
            rank=self.rank,
            splitfuse_enabled=self.splitfuse_enabled
        )
        self.sampler = Sampler(sampler_config)

        model_config["rank"] = self.rank
        model_config["world_size"] = self.world_size
        model_config['npu_device_id'] = self.npu_device_id
        model_config['local_rank'] = local_rank
        model_config['dp'] = dp
        model_config['tp'] = tp
        model_config['attn_inner_sp'] = attn_inner_sp
        model_config['sp'] = attn_inner_sp
        model_config['cp'] = cp
        model_config['moe_tp'] = moe_tp
        model_config['moe_ep'] = moe_ep
        model_config['soc_version'] = soc_version
        model_config['distributed_enable'] = self.distributed_enable
        model_config['max_batch_size'] = self.max_batch_size
        model_config['num_lccl_comm_shards'] = num_lccl_comm_shards
        model_config['lccl_comm_shard_id'] = lccl_comm_shard_id
        model_config['max_loras'] = max_loras
        model_config['max_lora_rank'] = max_lora_rank
        model_config['sampler_config'] = sampler_config
        model_config['lwdNextPHeadPrior'] = lwd_next_p_head_prior

        self.backend_type = backend_type
        self.model_wrapper = get_model_wrapper(model_config, backend_type)
        self.config = self.model_wrapper.config
        self.config_dict = self.model_wrapper.config_dict
        self.update_config(model_config)
        self.model_info = self.model_wrapper.model_info
        self.num_speculative_tokens = model_config.get('num_speculative_tokens', 0)
        self.llm_config = None
        self.enable_dap = False
        self.obfuscation_func = None
        self.device = None
        self.cache_pool = None

        # Thread-safe mechanism for detecting FORCE STOP exception
        self.force_stop_exception_occurred = threading.Event()
        self.is_fault_device = False

        self.max_position_embeddings = self.model_wrapper.max_position_embeddings

    @staticmethod
    def repeat_sample_param(param_tensor, tokens_num_per_batch):
        if param_tensor is None:
            return None
        result = []
        for tensor_tmp, n in zip(param_tensor, tokens_num_per_batch):
            repeat_tensor = tensor_tmp.repeat(n, 1)
            result.append(repeat_tensor)
        out_tensor = op.cat(result, dim=0)
        return out_tensor

    @staticmethod
    def __parse_config_key(model_config):
        for key, value in MODEL_CONFIG_KEY_TYPE.items():
            if key in model_config:
                model_config[key] = parse_config(model_config, key, parse_type=value)

    def configure_sampler(self, sampling_metadata):
        self.sampler.configure(sampling_metadata)

    def init_sampler(self, eos_token_id):
        self.sampler.initialize(self.device, eos_token_id)

    def set_device(self):
        pass

    def notify_force_stop_exception(self):
        '''
        Notify that a FORCE STOP exception has occurred in the inference thread.
        This method should be called from the inference thread when catching FORCE STOP exceptions.
        '''
        self.force_stop_exception_occurred.set()
        logger.info(f"FORCE STOP exception detected and notified for device {self.npu_device_id}")
    
    def execute_recover_command(self, command: str) -> dict:
        '''
        Execute recover related command.
        Args:
            command (str): recover command, including "CMD_PAUSE_ENGINE".
        Returns:
            dict: {"command_result": int, "error_msg": str, "npu_device_id": int}.
                  command_result: 0 for success, 1 for failure.
        '''
        error_msg = ""
        command_result = 1
        try:
            if command == "CMD_PAUSE_ENGINE":
                command_result, error_msg = self._execute_cmd_pause_engine()
            elif command == "CMD_REINIT_NPU":
                self._execute_cmd_reinit_npu()
                command_result = 0
        except Exception as e:
            error_msg = f"Execute recover command {command} failed, exception msg: {e}"
            logger.error(error_msg, ErrorCode.TEXT_GENERATOR_INTERNAL_ERROR)
            error_msg = str(e)
        return {"command_result": command_result, "error_msg": error_msg, "npu_device_id": self.npu_device_id}

    def build_inputs(self, conversations: List[List[Dict[str, str]]], **kwargs) -> List[List[int]]:
        return [self.model_wrapper.make_context(conversation, **kwargs) for conversation in conversations]

    def clear_cache(self, sequence_ids: Iterable) -> int:
        self.sampler.clear_cache(np.asarray(sequence_ids))
        return 1

    def update_config(self, kwargs):
        for key, value in kwargs.items():
            if key in self.config_dict.keys():
                setattr(self.config, key, value)

    @timer.track_time('forward')
    def forward(self, model_inputs: ModelInput, **kwargs) -> Any:
        """Call the `forward` method of the model wrapper, which should return a Tensor of corresponding backend."""
        result = self.model_wrapper.forward(model_inputs, **kwargs)
        return result

    @timer.track_time('sample')
    def sample(
            self,
            logits: Any,
            sampling_metadata: Optional[Union[SamplingMetadata, SamplingData]] = None,
            sampling_param: Optional[SamplingParam] = None,
            **kwargs
    ) -> Union[SamplingOutput, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Call the sampler of mindie-llm.

        This method samples from the input logits based on the post-processing parameters and selects the token ids. The
        type of this argument determines whether to use the `SamplingMetadata` or the deprecated parameter combination
        `SamplingData` and `SamplingParam`. The deprecated parameter combination does not support the `best_of` and
        `logprobs` functions.

        Args:
            logits: A Tensor of corresponding backend. It is the output obtained from the model's forward propagation.
            sampling_metadata: It can be an instance of `SamplingMetadata` or `SamplingData`. `SamplingMetadata`
                contains all sampling parameters like penalty, temperature, top_k, top_p, input and output token ids,
                etc. `SamplingData` only contains the request ids, prefilling flag, input token ids and output token
                ids.
            sampling_param: A deprecated argument used to store sampling parameters, which can only used with
                `SamplingData`.

        Returns:
            Union[SamplingOutput, Tuple[np.ndarray, Optional[np.ndarray]]]: An instance of `SamplingOutput` will be
                returned when `SamplingMetadata` object is passed. Otherwise, it will return a tuple of token ids and
                logprobs.
        """
        sampling_data = None
        if isinstance(sampling_metadata, SamplingData):
            sampling_data = sampling_metadata
        elif 'sampling_data' in kwargs:
            sampling_data = kwargs.get('sampling_data')
        if sampling_data is not None:  # Enter deprecated branch
            sampling_metadata = SamplingMetadata.from_deprecated(sampling_data, sampling_param)
            output = self.sampler(logits, sampling_metadata)
            output = (output.token_ids, output.logprobs)
        else:
            output = self.sampler(logits, sampling_metadata)
        return output

    def _execute_cmd_pause_engine(self):
        self.force_stop_exception_occurred.clear()
        wait_exception_time = 10.0
        time.sleep(wait_exception_time)
        if torch_npu.npu.stop_device(self.npu_device_id) != 0:
            error_msg = "Stop device failed"
            command_result = 1
        else:
            uce_command_result, uce_error_msg = self._handle_uce_error()
            if uce_command_result == 1:
                command_result = uce_command_result
                error_msg = uce_error_msg
            elif uce_command_result == 2:
                command_result = 0
                error_msg = ""
            elif not self._wait_for_force_stop_exception():
                command_result = 1
                error_msg = "Timeout waiting for FORCE STOP exception"
            else:
                command_result = 0
                error_msg = ""
        return command_result, error_msg
    
    def _execute_cmd_reinit_npu(self):
        '''Reinitialize NPU. Subclasses must override with backend-specific logic.'''
        raise NotImplementedError("Subclasses must implement _execute_cmd_reinit_npu")

    def _wait_for_force_stop_exception(self):
        if not self.is_fault_device:
            timeout = 60.0
            exception_detected = self.force_stop_exception_occurred.wait(timeout=timeout)
            if exception_detected:
                logger.info(
                    f"FORCE STOP exception detected for device {self.npu_device_id}, "
                    "stop_device execution successful"
                )
                return True
            else:
                logger.warning(
                    f"Timeout waiting for FORCE STOP exception for device {self.npu_device_id} "
                    f"after {timeout} seconds"
                )
                return False
        else:
            return True

    def _handle_uce_error(self):
        '''Check and recover UCE error in kvcache. Returns (command_result, error_msg).'''
        command_result = 0
        error_msg = ""
        res = torch.npu.check_uce_in_memory(self.npu_device_id)
        if res == 2 or res == 3:
            logger.info(f"Encountered HBM UCE error, check_uce_in_memory result: {res}")
            command_result = 2
            if not self._check_and_recover_uce_in_kvcache():
                logger.warning(f"HBM UCE address not in any kvcache, should trigger reschedule")
                command_result = 1
                error_msg = "HBM uce address not overlap kvcache address, should trigger reschedule"
        elif res == 1:
            logger.warning(f"Encountered HBM UCE error, but unknown UCE address, should trigger reschedule")
            command_result = 1
            error_msg = "HBM uce address unknown, should trigger reschedule"
        return command_result, error_msg

    def _check_and_recover_uce_in_kvcache(self):
        '''Check and recover UCE error in kvcache. Returns True if recovered, False otherwise.'''
        uce_addr_list = torch_npu.npu._get_uce_addr()
        logger.info(f"UCE address list: {uce_addr_list}")
        if len(uce_addr_list) == 0:
            return False
        for addr_entry in uce_addr_list:
            uce_addr = addr_entry["ptr"]
            addr_size = addr_entry["size"]
            uce_addr_start = uce_addr
            uce_addr_end = uce_addr_start + addr_size

            for n in range(self.cache_pool.kvcache_settings.num_layers):
                k_cache = self.cache_pool.npu_cache[n][0]
                if check_and_recover_uce_in_cache(uce_addr_start, uce_addr_end, k_cache, n, "kcache"):
                    return True
                v_cache = self.cache_pool.npu_cache[n][1]
                if check_and_recover_uce_in_cache(uce_addr_start, uce_addr_end, v_cache, n, "vcache"):
                    return True
        return False