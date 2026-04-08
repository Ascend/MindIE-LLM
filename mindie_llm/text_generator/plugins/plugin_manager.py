# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

import importlib
import queue
import threading
import time
import copy
from dataclasses import fields
from enum import IntEnum
from typing import Iterable, Optional, Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from mindie_llm.text_generator.utils import (
    GenerationOutput,
    InputMetadata,
    ModelInputWrapper,
    ModelOutputWrapper,
    SamplingOutput,
    NpuMemoryWatcher
)
from mindie_llm.text_generator.plugins.plugin_utils import PluginDataParam
from mindie_llm.text_generator.utils.input_metadata import SIMULATE_SEQUENCE_ID
from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.utils.status import CoreThread
from mindie_llm.utils.decorators.time_decorator import timer
from mindie_llm.utils.env import ENV
from mindie_llm.utils.log import logger, HandlerType
from mindie_llm.utils.prof.profiler import span_start, span_end, span_req, span_attr, count_block
from mindie_llm.utils.log.error_code import ErrorCodeException, convert_exception_to_error_code, is_force_stop_exception

if TYPE_CHECKING:
    from mindie_llm.text_generator.utils import (
        KVCacheSettings,
        TGInferContextStore,
        OutputFilter
    )
    from mindie_llm.text_generator.utils.separate_deployment_engine import DmiModeNodeRole
    from mindie_llm.text_generator.adapter.generator_backend import GeneratorBackend

LAUNCH_DONE_TIMEOUT = 20 * 60     # unit: second
MEM_DETECT_INTERVAL = 1000        # unit: second


class MemPoolType(IntEnum):
    DISABLED = 0
    SYNC_WRITE = 1
    ASYNC_WRITE = 2


class PluginManager:
    def __init__(
        self,
        generator_backend: GeneratorBackend,
        kvcache_settings: KVCacheSettings,
        infer_context: TGInferContextStore,
        output_filter: OutputFilter,
        is_mix_model: bool,
        plugin_list: list[str],
        model_role: DmiModeNodeRole | str,
        watcher: NpuMemoryWatcher,
        **kwargs
    ):
        self.generator_backend = generator_backend
        self.model_wrapper = self.generator_backend.model_wrapper
        self.sampler = self.generator_backend.sampler
        self.kvcache_settings = kvcache_settings
        self.infer_context = infer_context
        self.output_filter = output_filter
        self.is_mix_model = is_mix_model
        self.rank = self.generator_backend.rank
        self.watcher = watcher
        if is_mix_model:
            self.mix_preprocess = None
        self.plugin_data_param = PluginDataParam()

        self.plugin_list = plugin_list
        self.async_inference = self.infer_context.context_params.async_infer
        self.max_generated_tokens = self.infer_context.context_params.max_generated_tokens
        kwargs.update({'model_role': model_role})
        self.kwargs = kwargs

        if self.async_inference:
            self.input_queue = queue.Queue()
            self.output_queue = queue.Queue()
            self.output_queue.put(ModelOutputWrapper.make_empty())
            self.forward_thread = CoreThread(target=self.forward_loop, daemon=True, name="async_forward")
            self.forward_thread.start()
            self.execution_stream = torch.npu.current_stream()
        self.last_sequence_ids = None
        self.previous_batch_is_prefill = False
        self.is_inference_pause = False
        self.mem_det_trigger_counter = 0
        self.error_code_collected_in_async = None
        self.mempool_type = MemPoolType.DISABLED
        self.warmup_is_end = True
        # 结构化输出管理器 (延迟初始化)
        self._structured_output_manager: Optional[Any] = None
        self._structured_output_enabled = kwargs.get('enable_structured_output', True)

    @staticmethod
    def unsqueeze_sampling_output(sampling_output: SamplingOutput):
        if ENV.model_runner_exp and ENV.async_inference:
            sampling_output.token_ids = torch.unsqueeze(sampling_output.token_ids, 1)
            sampling_output.logprobs = torch.unsqueeze(sampling_output.logprobs, 1)
            sampling_output.top_token_ids = torch.unsqueeze(sampling_output.top_token_ids, 1)
            sampling_output.top_logprobs = torch.unsqueeze(sampling_output.top_logprobs, 1)
        else:
            sampling_output.token_ids = np.expand_dims(sampling_output.token_ids, 1)
            sampling_output.logprobs = np.expand_dims(sampling_output.logprobs, 1)
            sampling_output.top_token_ids = np.expand_dims(sampling_output.top_token_ids, 1)
            sampling_output.top_logprobs = np.expand_dims(sampling_output.top_logprobs, 1)

    @staticmethod
    def filter_splitfuse_token_ids(input_metadata: InputMetadata, sampling_output: SamplingOutput):
        # splitfuse非最后一块prefill的token_ids都需要置为-1
        if input_metadata.batch_is_prefill is not None and input_metadata.batch_last_prompt is not None:
            batch_is_prefill = input_metadata.batch_is_prefill
            batch_last_prompt = input_metadata.batch_last_prompt
            if sampling_output.repeating_indices is not None:
                batch_is_prefill = batch_is_prefill[sampling_output.repeating_indices]
                batch_last_prompt = batch_last_prompt[sampling_output.repeating_indices]
            sampling_output.token_ids[batch_is_prefill & ~batch_last_prompt] = -1

    @staticmethod
    def _to_host(data_instance: Any):
        if data_instance is None:
            return None
        new_instance = copy.deepcopy(data_instance)
        for field in fields(new_instance):
            field_value = getattr(new_instance, field.name)
            if isinstance(field_value, torch.Tensor):
                if field_value.dtype == torch.bfloat16:
                    host_array = field_value.cpu().float().numpy()
                else:
                    host_array = field_value.cpu().numpy()
                setattr(new_instance, field.name, host_array)
        return new_instance

    def initialize(self):
        if self.is_mix_model:
            from .splitfuse.splitfuse_plugin import SplitfusePlugin
            self.mix_preprocess = SplitfusePlugin(self.model_wrapper, self.kvcache_settings, self.infer_context)
        for plugin in self.plugin_list:
            cls_name = ''.join([word.capitalize() for word in plugin.split('_')]) + 'Plugin'
            plugin_path = f"mindie_llm.text_generator.plugins.{plugin}.{plugin}_plugin"
            plugin_module = importlib.import_module(plugin_path)
            plugin_cls = getattr(plugin_module, f"{cls_name}")
            plugin_tmp = plugin_cls(
                self.generator_backend,
                self.kvcache_settings,
                self.infer_context,
                self.plugin_data_param,
                **self.kwargs,
            )
            setattr(self, plugin, plugin_tmp)
        if "prefix_cache" in self.plugin_list:
            self.mempool_type = self.prefix_cache.mempool_type

        # 初始化结构化输出管理器
        self._init_structured_output_manager()

    def wait_put_finish(self, input_metadata):
        if "prefix_cache" in self.plugin_list and input_metadata.is_prefill:
            logger.info("Waiting save to finished")
            start_t, timeout_t = time.time(), self.prefix_cache.save_timeout
            if self.prefix_cache.save_event.wait(timeout=timeout_t):
                logger.info(f"Save finished in {(time.time() - start_t)*1000:.1f} ms")
            else:
                logger.error(f"[TIMEOUT] Save unfinished after {timeout_t} seconds. Exit")

    def mem_det_trigger_counter_acc(self):
        if self.mem_det_trigger_counter < MEM_DETECT_INTERVAL:
            self.mem_det_trigger_counter = self.mem_det_trigger_counter + 1
        else:
            self.mem_det_trigger_counter = 0

    @timer.track_time_async('generate_token')
    def generate_token(self, input_metadata: InputMetadata, warmup=False) -> GenerationOutput:
        try:
            prof = span_start("preprocess")
            cache_ids, model_inputs, sampling_metadata, trace_ids = self.preprocess(input_metadata, warmup=warmup)
            if not self.is_mix_model:
                self.plugin_data_param.q_len = None
                self.plugin_data_param.mask = None
            model_inputs, qlen, mask = self.model_inputs_update_manager(
                model_inputs, input_metadata, sampling_metadata, cache_ids)
            self.plugin_data_param.q_len = qlen if qlen is not None else self.plugin_data_param.q_len
            self.plugin_data_param.mask = mask if mask is not None else self.plugin_data_param.mask
            if not warmup and "prefix_cache" in self.plugin_list and \
                self.prefix_cache.mempool_type == MemPoolType.ASYNC_WRITE:
                self.prefix_cache.async_put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'After preprocess', 
                                       trigger_count=self.mem_det_trigger_counter)

            prof = span_start("forward", True)
            span_req(prof, trace_ids)
            span_attr(prof, "blocks", count_block(input_metadata.block_tables))
            if hasattr(self.model_wrapper, "mapping"):
                span_attr(prof, "dp_rank", str(self.model_wrapper.mapping.attn_dp.rank))
                
            if ENV.framework_backend == BackendType.ATB:
                forward_extra_kwargs = {}
                if warmup:
                    forward_extra_kwargs["warmup_is_end"] = False
                else:
                    self.model_wrapper.model_runner.clear_internal_tensors()
                if (self.plugin_list and "mtp" not in self.plugin_list) or self.is_mix_model:
                    result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                            attn_mask=self.plugin_data_param.mask,
                                                            **forward_extra_kwargs)  # q_len spec_mask
                # old graph forward
                else:
                    result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                            spec_mask=self.plugin_data_param.mask,
                                                            sub_model_inputs=self.plugin_data_param.mtp_model_inputs,
                                                            hidden_states=self.plugin_data_param.hidden_states,
                                                            **forward_extra_kwargs)
            else:
                result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                        spec_mask=self.plugin_data_param.mask)  # q_len spec_mask
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result
            span_end(prof, True)
            if warmup:
                torch.npu.synchronize()
                torch.npu.empty_cache()
            self.watcher.watch_npu_mem(self.rank, f'After forward', trigger_count=self.mem_det_trigger_counter)

            prof = span_start("sample")
            draft_filtered_logits = self.sample_preprocess_manager(logits, result, sampling_metadata, input_metadata)
            sampling_output = self.generator_backend.sample(draft_filtered_logits, sampling_metadata)
            if ENV.framework_backend == BackendType.ATB and not warmup:
                self.model_wrapper.model_runner.clear_internal_tensors()
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'After sample', trigger_count=self.mem_det_trigger_counter)
            logger.info("sample end", extra={"handler_ids": HandlerType.TOKEN})
            prof = span_start("postprocess")
            if self.mempool_type == MemPoolType.SYNC_WRITE:
                self.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
            elif not warmup and self.mempool_type == MemPoolType.ASYNC_WRITE:
                self.wait_put_finish(input_metadata)
            generation_output = self.postprocess(
                cache_ids, input_metadata, result, sampling_metadata, sampling_output)
            generation_output.trace_ids = trace_ids
            generation_output.simulator_ids = input_metadata.simulator_ids
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'After postprocess', trigger_count=self.mem_det_trigger_counter)
            self.mem_det_trigger_counter_acc()
            return generation_output

        except Exception as e:
            if self.is_inference_pause:
                logger.info(f"Mocking response due to inference pause for trace_ids={trace_ids}.")
                # Check for FORCE STOP exception and notify generator_backend if it's GeneratorTorch
                if is_force_stop_exception(e):
                    logger.info(f"FORCE STOP exception detected in plugin_manager.generate_token: {e}")
                    self.generator_backend.notify_force_stop_exception()
                return GenerationOutput.make_empty()
            logger.exception(
                f"Error encountered in generate_token (trace_ids={trace_ids}). "
                f"trigger recovery or terminate inference thread. Error: {e}"
            )
            raise e

    def generate_token_async(self, input_metadata: InputMetadata, warmup=False) -> GenerationOutput:
        with self.generator_backend.get_new_stream():
            prof = span_start("preprocess")
            hit_mask = np.isin(input_metadata.all_sequence_ids, self.last_sequence_ids)
            cache_ids, model_input, sampling_metadata, trace_ids = self.preprocess(
                input_metadata, warmup=warmup, hit_mask=hit_mask
            )
            self.infer_context.last_sampling_metadata.clear()  # Do not use sampling cache under async inference.
            model_input, _, _ = self.model_inputs_update_manager(
                model_input, input_metadata, sampling_metadata, cache_ids, hit_mask=hit_mask)
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'In asyn inference mode, after preprocess', 
                                        trigger_count=self.mem_det_trigger_counter)

            prof = span_start("prepare_model_inputs")
            if not ENV.framework_backend == BackendType.ATB:
                raise RuntimeError('The current backend type does not support async inference.')

            if (self.plugin_list and "mtp" not in self.plugin_list) or self.is_mix_model:
                model_input, model_kwargs = self.generator_backend.prepare_model_inputs(
                    model_input,
                    q_lens=self.plugin_data_param.q_len,
                    attn_mask=self.plugin_data_param.mask,
                )
            else:
                model_input, model_kwargs = self.generator_backend.prepare_model_inputs(
                    model_input,
                    q_lens=self.plugin_data_param.q_len,
                    spec_mask=self.plugin_data_param.mask,
                    sub_model_inputs=self.plugin_data_param.mtp_model_inputs,
                    hidden_states=self.plugin_data_param.hidden_states
                )
            
            self.warmup_is_end = True
            if warmup:
                if model_kwargs is None:
                    model_kwargs = {}
                model_kwargs["warmup_is_end"] = False
                self.warmup_is_end = False

            if self.generator_backend.dp > 1:
                cur_dp_rank_id_per_token_mask = model_input.dp_rank_ids == self.generator_backend.mapping.attn_dp.rank
                current_dp_sequence_ids = input_metadata.all_sequence_ids[cur_dp_rank_id_per_token_mask]
                current_dp_batch_is_prefill = (
                    input_metadata.batch_is_prefill[cur_dp_rank_id_per_token_mask]
                    if input_metadata.batch_is_prefill is not None else None
                )
                current_dp_token_num_per_seq = (
                    self._get_token_num_per_seq(input_metadata)[cur_dp_rank_id_per_token_mask]
                    if self.is_mix_model else None
                )
            else:
                current_dp_sequence_ids = input_metadata.all_sequence_ids
                current_dp_batch_is_prefill = input_metadata.batch_is_prefill
                current_dp_token_num_per_seq = (
                    self._get_token_num_per_seq(input_metadata)
                    if self.is_mix_model else None
                )

            filling_masks = self._prepare_masks_for_filling(
                model_input,
                current_dp_sequence_ids,
                input_metadata,
                current_dp_batch_is_prefill,
                current_dp_token_num_per_seq)

            postprocess_done = threading.Event()
            model_input_wrapper = ModelInputWrapper(
                cache_ids, input_metadata, model_input, model_kwargs, sampling_metadata,
                trace_ids, current_dp_sequence_ids, postprocess_done, filling_masks)
            span_end(prof)

            prof = span_start('get_from_output_queue')
            model_output_wrapper = self.output_queue.get(timeout=900)
            span_end(prof)

            is_mock = model_output_wrapper.is_mock
            # Move '_fill_in_model_result' into 'forward_loop' to reduce inter-token latency.
            # This requires 'Sampler' to perform on-device post-processing.
            if not ENV.model_runner_exp:
                prof = span_start("fill_in_model_result") 
                if not is_mock and model_output_wrapper.model_output:	 
                    self._fill_in_model_result(input_metadata, model_input_wrapper, model_output_wrapper, 
                                                filling_masks, cache_ids)
                span_end(prof)
            else:
                model_input_wrapper.model_inputs.input_ids.record_stream(self.execution_stream)
                model_input_wrapper.model_inputs.position_ids.record_stream(self.execution_stream)
                model_input_wrapper.model_inputs.forward_context.record_stream(self.execution_stream)
                if model_input_wrapper.model_inputs.forward_context.sub_forward_context is not None:
                    model_input_wrapper.model_inputs.forward_context.sub_forward_context.record_stream(
                        self.execution_stream)

            prof = span_start("synchronize_processing_stream")
            self.generator_backend.synchronize()
            span_end(prof)

            if not warmup and "prefix_cache" in self.plugin_list and \
                self.prefix_cache.mempool_type == MemPoolType.ASYNC_WRITE:
                self.prefix_cache.async_put_prefix_kvcache_to_mempool(input_metadata, cache_ids)

            prof = span_start("put_into_input_queue")
            self.input_queue.put(model_input_wrapper)
            span_end(prof)

            if not input_metadata.is_prefill and (ENV.model_runner_exp or not self.previous_batch_is_prefill):
                prof = span_start("wait_to_postprocess")
                if model_output_wrapper.launch_done is not None:
                    if not self.is_inference_pause:
                        if not model_output_wrapper.launch_done.wait(timeout=LAUNCH_DONE_TIMEOUT):
                            logger.warning("Timeout waiting for launch_done signal.")
                    else:  # branch for quick recovery
                        if not model_output_wrapper.launch_done.wait(timeout=1):
                            is_mock = True
                span_end(prof)
            self.previous_batch_is_prefill = input_metadata.is_prefill

            # Maintain backward compatibility with the previous implementation
            sampling_output = model_output_wrapper.sampling_output
            if ENV.model_runner_exp:
                if model_output_wrapper.execution_done is not None:
                    model_output_wrapper.execution_done.synchronize()
                sampling_output = self._to_host(model_output_wrapper.sampling_output)
            self.watcher.watch_npu_mem(self.rank, f'In asyn inference mode, before postprocess',
                                       trigger_count=self.mem_det_trigger_counter)
            prof = span_start("postprocess")
            if not is_mock and model_output_wrapper.cache_ids is not None and \
                not model_output_wrapper.input_metadata.is_dummy_batch:
                if model_output_wrapper.model_output.hidden_states is None:
                    model_result = model_output_wrapper.model_output.logits
                else:
                    model_result = (model_output_wrapper.model_output.logits,
                                    model_output_wrapper.model_output.hidden_states)
                generation_output = self.postprocess(
                    model_output_wrapper.cache_ids,
                    model_output_wrapper.input_metadata,
                    model_result,
                    model_output_wrapper.sampling_metadata,
                    sampling_output,
                )
                generation_output.trace_ids = model_output_wrapper.trace_ids
            else:
                generation_output = GenerationOutput.make_empty()
            postprocess_done.set()
            generation_output.fill_dummy(input_metadata, self.max_generated_tokens)
            self.last_sequence_ids = input_metadata.all_sequence_ids
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'In asyn inference mode, after postprocess', 
                                       trigger_count=self.mem_det_trigger_counter)
            self.mem_det_trigger_counter_acc()

        # 将forward loop中捕获到的故障码异常向上层抛出
        if self.error_code_collected_in_async is not None:
            message = (f'Detect {self.error_code_collected_in_async.name} fault happened in forward loop, '
                       f'error code: {self.error_code_collected_in_async.value}.')
            error_code = self.error_code_collected_in_async
            self.error_code_collected_in_async = None
            logger.error(message)
            raise ErrorCodeException(error_code)

        return generation_output

    @timer.track_time('preprocess')
    def preprocess(self, input_metadata, warmup=False, hit_mask=None):
        cache_ids = self.infer_context.get_batch_context_handles(input_metadata)
        if self.is_mix_model:
            (
                model_inputs,
                cache_ids,
                sampling_metadata,
                q_len,
                attention_mask,
                trace_ids,
            ) = self.mix_preprocess.splitfuse_preprocess.splitfuse_preprocess(
                input_metadata,
                warmup=warmup,
                hit_mask=hit_mask,
            )
            self.plugin_data_param.q_len = q_len
            self.plugin_data_param.mask = attention_mask
        else:
            (
                model_inputs,
                sampling_metadata,
                trace_ids
            ) = self.infer_context.compose_model_inputs(
                input_metadata,
                cache_ids,
                warmup=warmup,
                hit_mask=hit_mask
            )

        if not self.async_inference and self._structured_output_manager is not None:
            response_format_array = (
                input_metadata.batch_response_format
                if input_metadata.is_prefill
                else self.infer_context.get_response_format(cache_ids)
            )
            self._structured_output_manager.build_and_assign_structured_guided_bitmask(
                input_metadata, sampling_metadata, cache_ids, response_format_array
            )

        if sampling_metadata is not None and ENV.model_runner_exp and not sampling_metadata.is_prefill:
            for plugin in self.plugin_list:
                plugin_instance = getattr(self, plugin, None)
                method = getattr(plugin_instance, 'compose_model_inputs_exp', None)
                if method is not None:
                    sampling_metadata = method(sampling_metadata)
        res = (cache_ids, model_inputs, sampling_metadata, trace_ids)
        return res

    @timer.track_time('stop')
    def postprocess(self, cache_ids, input_metadata, result, sampling_metadata, sampling_output):
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        if ENV.framework_backend == BackendType.ATB:
            from atb_llm.utils.initial import NPUSocInfo
            from atb_llm.utils.env import ENV as atb_env
            soc_info = NPUSocInfo()
            if atb_env.enable_greedy_search_opt and not soc_info.need_nz:
                logits = logits.squeeze(1)
                sampling_output.token_ids = logits.cpu().numpy()

        is_structured_accepted = sampling_output.is_structured_accepted
        if not self.async_inference:
            self.plugin_verify_manager(sampling_output, cache_ids, result)
            if self._structured_output_manager is not None:
                is_structured_accepted = (
                    self._structured_output_manager.compute_structured_output_accepted(
                        cache_ids=cache_ids,
                        token_ids=sampling_output.token_ids,
                    )
                )
            else:
                is_structured_accepted = None
        # 兜底：无结构化输出时（_compute_... 返回 None 或异步路径本批次无结构化请求），使用全 True 数组
        if is_structured_accepted is None:
            batch_size = len(cache_ids) if cache_ids is not None else 0
            is_structured_accepted = np.ones(batch_size, dtype=bool)

        finish_reason, filtered_indices, truncation_indices = (
            self.output_filter.filter_finished_sequences(
                cache_ids, input_metadata, sampling_output, is_structured_accepted))

        # If best_of sampling or beam search is open, get new cache ids
        if sampling_metadata is not None:
            best_of_sampling = sampling_metadata.best_of_array is not None and sampling_metadata.is_prefill
            has_beam_search = sampling_metadata.use_beam_search_array is not None
            if best_of_sampling or has_beam_search:
                cache_ids = self.infer_context.fork_context(sampling_output)

        la_cache_input = (result, sampling_metadata)
        self.plugin_cache_update_manager(cache_ids, sampling_output, la_cache_input, input_metadata.is_prefill)

        metadata = (input_metadata, sampling_metadata)
        finished_cache_ids, finished_sequence_ids = self.infer_context.update_context(
            cache_ids, filtered_indices, metadata, sampling_output)
        has_sampling = sampling_metadata is not None
        sequence_ids_to_clear = np.array([], dtype=np.int64)
        # 不清理dummy batch 的 cache id
        if not input_metadata.is_dummy_batch:
            sequence_ids_to_clear = self.infer_context.clear_finished_context(finished_sequence_ids, finished_cache_ids)
            if has_sampling and sequence_ids_to_clear.size != 0:
                self.sampler.clear_cache(sequence_ids_to_clear)
            self.plugin_cache_clear_manager(cache_ids, finish_reason)
        self.infer_context.clear_aborted_context()
        token_indices = self.infer_context.get_output_len_count(cache_ids)

        if has_sampling:
            sequence_ids = sampling_output.sequence_ids
            parent_sequence_ids = sampling_output.parent_sequence_ids
        else:
            sequence_ids = input_metadata.all_sequence_ids
            parent_sequence_ids = input_metadata.all_sequence_ids

        self.filter_splitfuse_token_ids(input_metadata, sampling_output)

        generation_output = GenerationOutput(
            sequence_ids=sequence_ids,
            parent_sequence_ids=parent_sequence_ids,
            group_indices=sampling_output.group_indices,
            token_ids=sampling_output.token_ids,
            logprobs=sampling_output.logprobs,
            top_token_ids=sampling_output.top_token_ids,
            top_logprobs=sampling_output.top_logprobs,
            num_new_tokens=sampling_output.num_new_tokens,
            num_top_tokens=sampling_output.num_top_tokens,
            cumulative_logprobs=sampling_output.cumulative_logprobs,
            finish_reason=finish_reason,
            truncation_indices=truncation_indices,
            current_token_indices=token_indices
        )
        if self.async_inference and sequence_ids_to_clear.size != 0:
            generation_output.remove(sequence_ids_to_clear)
        return generation_output

    def model_inputs_update_manager(self, model_inputs, input_metadata, sampling_metadata, cache_ids, **kwargs):
        if not self.is_mix_model:
            self.plugin_data_param.q_len = None
            self.plugin_data_param.mask = None
        q_len = None
        spec_mask = None
        input_len_mask = (q_len, spec_mask)
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'model_inputs_update', None)
            model_inputs, input_len_mask = method(
                model_inputs, input_metadata, sampling_metadata, cache_ids, input_len_mask, **kwargs)
        (q_len, spec_mask) = input_len_mask
        # 需要确保虚推 context_length 至少为 1，否则会导致模型内部维度计算错误 (coreDim = 0)
        if input_metadata.all_sequence_ids is not None and not input_metadata.is_prefill:
            has_simulate = any(sid == SIMULATE_SEQUENCE_ID for sid in input_metadata.all_sequence_ids)
            if has_simulate and model_inputs.context_length[0] == 0:
                model_inputs.context_length[0] = 1
        self.plugin_data_param.q_len = q_len if q_len is not None else self.plugin_data_param.q_len
        self.plugin_data_param.mask = spec_mask if spec_mask is not None else self.plugin_data_param.mask
        return model_inputs, q_len, spec_mask

    def sample_preprocess_manager(self, logits, result, sampling_metadata, input_metadata):
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'sample_preprocess', None)
            logits = method(logits, result, sampling_metadata, input_metadata)
        return logits

    def plugin_verify_manager(self, sampling_output, cache_ids, result):
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            if ENV.model_runner_exp and ENV.async_inference:
                method = getattr(plugin_instance, 'plugin_verify_exp', None)
            else:
                method = getattr(plugin_instance, 'plugin_verify', None)
            method(sampling_output, cache_ids, result)
        if len(sampling_output.token_ids.shape) != 2:
            self.unsqueeze_sampling_output(sampling_output)

    def plugin_cache_update_manager(self, cache_ids, sampling_output, la_cache_input, is_prefill):
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'plugin_cache_update', None)
            method(cache_ids, sampling_output, la_cache_input, is_prefill=is_prefill)

    def plugin_cache_clear_manager(self, cache_ids, finish_reason):
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'plugin_cache_clear', None)
            method(cache_ids, finish_reason)

    def put_prefix_kvcache_to_mempool(self, input_metadata, cache_ids):
        if "prefix_cache" in self.plugin_list:
            plugin_instance = getattr(self, "prefix_cache", None)
            method = getattr(plugin_instance, 'put_prefix_kvcache_to_mempool', None)
            method(input_metadata, cache_ids)

    def forward_loop(self):
        self.generator_backend.set_device()
        launch_done = None
        model_output_wrapper = None
        while True:
            prof = span_start("get_from_input_queue")
            model_input_wrapper: ModelInputWrapper = self.input_queue.get()
            span_end(prof)

            if self._structured_output_manager is not None:
                input_metadata_for_batch = model_input_wrapper.input_metadata
                response_format_array = (
                    input_metadata_for_batch.batch_response_format
                    if input_metadata_for_batch.is_prefill
                    else self.infer_context.get_response_format(model_input_wrapper.cache_ids)
                )
                self._structured_output_manager.build_and_assign_structured_guided_bitmask(
                    input_metadata_for_batch,
                    model_input_wrapper.sampling_metadata,
                    model_input_wrapper.cache_ids,
                    response_format_array,
                )

            # Maintain backward compatibility with the previous implementation
            if ENV.model_runner_exp:
                prof = span_start("fill_in_model_result")
                if model_output_wrapper is not None:
                    self._fill_in_model_result_exp(model_input_wrapper, model_output_wrapper)
                span_end(prof)
            try:
                prof = span_start("forward")
                span_req(prof, model_input_wrapper.trace_ids)
                span_attr(prof, "async", True)
                if ENV.framework_backend == BackendType.ATB:
                    self.model_wrapper.model_runner.clear_internal_tensors()
                model_output = self.generator_backend.forward_from_model_inputs(
                    model_input_wrapper.model_inputs, **model_input_wrapper.model_kwargs)
                if launch_done is not None:
                    launch_done.set()
                span_end(prof)

                prof = span_start("sample")
                draft_filtered_logits = self.sample_preprocess_manager(
                    model_output.logits,
                    model_output.original_result,
                    model_input_wrapper.sampling_metadata,
                    model_input_wrapper.input_metadata
                )
                sampling_output = self.generator_backend.sample(
                    draft_filtered_logits, model_input_wrapper.sampling_metadata)
                if self._structured_output_manager is not None:
                    async_is_structured_accepted = (
                        self._structured_output_manager.compute_structured_output_accepted(
                            cache_ids=model_input_wrapper.cache_ids,
                            token_ids=sampling_output.token_ids,
                        )
                    )
                else:
                    async_is_structured_accepted = None

                if ENV.framework_backend == BackendType.ATB:
                    self.model_wrapper.model_runner.clear_internal_tensors()
                span_end(prof)
                logger.info("sample end", extra={"handler_ids": HandlerType.TOKEN})

                if not self.is_inference_pause:
                    model_input_wrapper.postprocess_done.wait()

                prof = span_start("verify")
                self.plugin_verify_manager(
                    sampling_output, model_input_wrapper.cache_ids, model_output.original_result)
                span_end(prof)

                if self.mempool_type == MemPoolType.SYNC_WRITE:
                    prof = span_start("put_prefix_kvcache_to_mempool")
                    if (
                        model_input_wrapper.cache_ids is not None
                        and not model_input_wrapper.input_metadata.is_dummy_batch
                    ):
                        self.put_prefix_kvcache_to_mempool(
                            model_input_wrapper.input_metadata, model_input_wrapper.cache_ids)
                    span_end(prof)
                elif self.warmup_is_end and self.mempool_type == MemPoolType.ASYNC_WRITE:
                    self.wait_put_finish(model_input_wrapper.input_metadata)

                launch_done = threading.Event()
                model_output_wrapper = ModelOutputWrapper(
                    cache_ids=model_input_wrapper.cache_ids,
                    input_metadata=model_input_wrapper.input_metadata,
                    model_output=model_output,
                    sampling_metadata=model_input_wrapper.sampling_metadata,
                    sampling_output=sampling_output,
                    trace_ids=model_input_wrapper.trace_ids,
                    current_dp_sequence_ids=model_input_wrapper.current_dp_sequence_ids,
                    launch_done=launch_done,
                )
                sampling_output.is_structured_accepted = async_is_structured_accepted
            except Exception as e:
                trace_ids = getattr(model_input_wrapper, 'trace_ids', 'unknown')

                # Check for FORCE STOP exception and notify generator_backend if it's GeneratorTorch
                if is_force_stop_exception(e):
                    logger.info(f"FORCE STOP exception detected in plugin_manager.forward_loop: {e}")
                    self.generator_backend.notify_force_stop_exception()

                error_code = convert_exception_to_error_code(str(e))

                # Handle PyTorch OOM(Only supports Torch 2.6+ native exception)
                # If torch version is 2.1 or lower, please check exception message directly.
                if hasattr(torch, "OutOfMemoryError") and isinstance(e, torch.OutOfMemoryError):
                    error_msg = (
                            "Device out of memory (OOM) reported by PyTorch, but it can possibly triggered by HCCL. "
                            "Enable logs: export ASCEND_SLOG_PRINT_TO_STDOUT=1, "
                            "export ASCEND_GLOBAL_LOG_LEVEL=3 to check if there's HCCL error messages"
                        )
                    logger.error(error_msg)
                    error_code = ErrorCode.TEXT_GENERATOR_OUT_OF_MEMORY

                if error_code is not None:
                    self.error_code_collected_in_async = error_code

                if self.is_inference_pause or self.error_code_collected_in_async is not None:
                    logger.info(f"Mocking response due to inference pause for trace_ids={trace_ids}.")
                    if self.error_code_collected_in_async and launch_done is not None:
                        launch_done.set()
                    model_output_wrapper = ModelOutputWrapper(
                        cache_ids=model_input_wrapper.cache_ids,
                        input_metadata=model_input_wrapper.input_metadata,
                        model_output=None,
                        sampling_metadata=model_input_wrapper.sampling_metadata,
                        sampling_output=None,
                        trace_ids=model_input_wrapper.trace_ids,
                        current_dp_sequence_ids=model_input_wrapper.current_dp_sequence_ids,
                        launch_done=None,
                        is_mock=True
                    )
                    self.output_queue.put(model_output_wrapper)
                    continue
                logger.exception(
                    f"Unrecoverable error in forward loop (trace_ids={trace_ids}). "
                    f"Terminating inference thread. Error: {e}"
                )
                raise e
            
            if ENV.model_runner_exp:
                execution_done = torch.npu.Event()
                execution_done.record(torch.npu.current_stream())
                model_output_wrapper.execution_done = execution_done
            self.output_queue.put(model_output_wrapper)

    def _fill_in_model_result_exp(self, model_input_wrapper, model_output_wrapper):
        filling_masks = model_input_wrapper.filling_masks
        model_inputs = model_input_wrapper.model_inputs
        method = None
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'fill_in_model_result_exp', None)
            if method is not None:
                break
        if method is not None:
            method(model_input_wrapper.input_metadata, model_inputs, model_input_wrapper.model_kwargs,
                model_output_wrapper, filling_masks, model_input_wrapper.cache_ids)
        if method is None:
            sampling_output = model_output_wrapper.sampling_output
            hit_sequence_ids_mask = filling_masks.get('hit_sequence_ids_mask')
            if hit_sequence_ids_mask is not None:
                hit_indices_tensor = filling_masks.get('hit_indices_tensor')
                true_token_ids = sampling_output.token_ids.index_select(dim=0, index=hit_indices_tensor).flatten()
                update_indices = filling_masks.get('update_indices')
                ones_int32 = filling_masks.get('ones_int32')
                ones_int64 = filling_masks.get('ones_int64')
                if len(update_indices) > 0:
                    model_inputs.input_ids.scatter_(0, update_indices, true_token_ids)
                    model_inputs.position_ids.scatter_add_(0, update_indices, ones_int64)
                    model_inputs.input_lengths.scatter_add_(0, update_indices, ones_int32)
                model_inputs.context_length[hit_sequence_ids_mask] += 1
                model_inputs.max_seq_len = max(model_inputs.context_length)
                model_inputs.forward_context.attn_metadata.max_seq_len = model_inputs.max_seq_len
                actual_seq_lengths_query = torch.ones_like(model_inputs.input_lengths, dtype=torch.int32)
                actual_seq_lengths_query = F.pad(
                    torch.cumsum(actual_seq_lengths_query, dim=0, dtype=torch.int32), (0, 0), value=0)
                seq_lens_q = actual_seq_lengths_query[1:] - actual_seq_lengths_query[:-1]
                model_inputs.forward_context.attn_metadata.actual_seq_lengths_query = actual_seq_lengths_query
                model_inputs.forward_context.attn_metadata.actual_seq_lengths_kv = model_inputs.input_lengths
                model_inputs.forward_context.attn_metadata.seq_lens = model_inputs.input_lengths

    def _init_structured_output_manager(self) -> None:
        if not self._structured_output_enabled:
            return
        
        try:
            from .structured_output import (
                StructuredOutputManager,
                StructuredOutputConfig,
                GuidedDecodingBackendType,
            )
            
            # 获取 tokenizer 和 vocab_size
            tokenizer = self.generator_backend.tokenizer
            vocab_size = None
            
            if tokenizer is not None:
                if hasattr(tokenizer, '__len__'):
                    vocab_size = len(tokenizer)
                elif hasattr(tokenizer, 'vocab_size'):
                    vocab_size = tokenizer.vocab_size
            
            if tokenizer is None or vocab_size is None:
                logger.warning("Cannot initialize structured output manager: tokenizer or vocab_size not available")
                self._structured_output_enabled = False
                return
            
            # 配置
            backend_type = self.kwargs.get('guided_decoding_backend', 'xgrammar')
            config = StructuredOutputConfig(
                backend=GuidedDecodingBackendType(backend_type),
            )
            
            # 创建管理器
            self._structured_output_manager = StructuredOutputManager(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                config=config,
            )
            self.infer_context.set_structured_output_manager(self._structured_output_manager)

        except ImportError as e:
            logger.warning(f"Failed to import structured output module: {e}")
            self._structured_output_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize structured output manager: {e}")
            self._structured_output_enabled = False

    def _prepare_masks_for_filling(
        self,
        model_inputs,
        current_dp_sequence_ids,
        input_metadata,
        current_dp_batch_is_prefill=None,
        current_dp_token_num_per_seq=None
    ):
        if input_metadata.batch_is_prefill is None and input_metadata.is_prefill:
            # Under forced preemption, prefill batch must not be hit.
            return {}
        current_all_sequence_ids = input_metadata.all_sequence_ids
        method = None
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'prepare_masks_for_filling', None)
            if method is not None:
                break
        if method is not None:
            masks = method(model_inputs, current_dp_sequence_ids, current_all_sequence_ids, self.last_sequence_ids)
        else:
            masks = {}
            hit_sequence_ids_mask = np.isin(current_dp_sequence_ids, self.last_sequence_ids)
            if current_dp_batch_is_prefill is not None:
                hit_sequence_ids_mask[current_dp_batch_is_prefill] = False
            elif input_metadata.is_prefill:
                hit_sequence_ids_mask[:] = False
            if hit_sequence_ids_mask.any():
                hit_sequence_ids = current_dp_sequence_ids[hit_sequence_ids_mask]
                if self.is_mix_model:
                    token_num_per_seq = current_dp_token_num_per_seq
                    if token_num_per_seq is None:
                        token_num_per_seq = self._get_token_num_per_seq(input_metadata)
                    repeating_indices = np.repeat(np.arange(len(token_num_per_seq)), token_num_per_seq)
                    hit_mask_per_token = hit_sequence_ids_mask[repeating_indices]
                    masks['hit_mask_per_token'] = self.generator_backend.to_tensor(hit_mask_per_token)
                hit_indices = np.where(hit_sequence_ids[:, None] == self.last_sequence_ids[None, :])[1]
                masks['hit_sequence_ids_mask'] = hit_sequence_ids_mask
                hit_sequence_ids_mask_tensor = self.generator_backend.to_tensor(hit_sequence_ids_mask)
                masks['hit_sequence_ids_mask_tensor'] = hit_sequence_ids_mask_tensor
                masks['hit_indices'] = hit_indices
                masks['hit_indices_tensor'] = self.generator_backend.to_tensor(hit_indices)
                if ENV.model_runner_exp:
                    update_indices = hit_sequence_ids_mask_tensor.nonzero(as_tuple=True)[0]
                    ones_int32 = torch.ones((len(update_indices),), device='npu', dtype=torch.int32)
                    ones_int64 = torch.ones((len(update_indices),), device='npu', dtype=torch.int64)
                    masks['update_indices'] = update_indices
                    masks['ones_int32'] = ones_int32
                    masks['ones_int64'] = ones_int64
        return masks

    def _fill_in_model_result(self, input_metadata, model_input_wrapper, model_output_wrapper,
                              filling_masks, cache_ids):
        method = None
        model_inputs = model_input_wrapper.model_inputs
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, 'fill_in_model_result', None)
            if method is not None:
                break
        if method is not None:
            method(input_metadata, model_inputs, model_input_wrapper.model_kwargs, model_output_wrapper,
                   filling_masks, cache_ids)
        else:
            sampling_output = model_output_wrapper.sampling_output
            hit_sequence_ids_mask = filling_masks.get('hit_sequence_ids_mask')
            if hit_sequence_ids_mask is not None:
                hit_indices = filling_masks.get('hit_indices')
                hit_sequence_ids_mask_tensor = filling_masks.get('hit_sequence_ids_mask_tensor')
                true_token_ids = sampling_output.token_ids[hit_indices].reshape(-1).astype(np.int64)
                hit_mask_per_token = filling_masks.get('hit_mask_per_token')
                if hit_mask_per_token is not None:
                    model_inputs.input_ids[hit_mask_per_token] = self.generator_backend.to_tensor(true_token_ids)
                    model_inputs.position_ids[hit_mask_per_token] += 1
                else:
                    model_inputs.input_ids[hit_sequence_ids_mask_tensor] = \
                        self.generator_backend.to_tensor(true_token_ids)
                    model_inputs.position_ids[hit_sequence_ids_mask_tensor] += 1
                if not self.generator_backend.mapping.has_attn_cp():
                    model_inputs.input_lengths[hit_sequence_ids_mask_tensor] += 1
                    model_inputs.context_length[hit_sequence_ids_mask] += 1
                    model_inputs.max_seq_len = max(model_inputs.context_length)

    def _get_token_num_per_seq(self, input_metadata: InputMetadata):
        batch_seq_len = input_metadata.split_end_position - input_metadata.split_start_position
        # computed_blocks为None时prefixcache无命中，batch_seq_len即为q_len
        if input_metadata.computed_blocks is None:
            token_num_per_seq = batch_seq_len
        # computed_blocks非None时prefixcache有命中，需从batch_seq_len中减去命中token
        else:
            token_num_per_seq = np.where(
                input_metadata.batch_is_prefill & (input_metadata.split_start_position == 0),
                batch_seq_len - self.generator_backend.block_size * input_metadata.computed_blocks,
                batch_seq_len
            ).astype(np.int64)
            cache_len_equal_mask = input_metadata.batch_is_prefill & (token_num_per_seq == 0)
            token_num_per_seq[cache_len_equal_mask] = self.generator_backend.block_size
        return token_num_per_seq