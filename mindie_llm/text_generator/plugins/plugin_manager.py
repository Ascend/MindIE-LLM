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
import copy
from dataclasses import fields
from typing import Iterable, Optional, Any, TYPE_CHECKING

import numpy as np
import torch

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
from mindie_llm.utils.log.error_code import ErrorCodeException, convert_exception_to_error_code

if TYPE_CHECKING:
    from mindie_llm.text_generator.utils import (
        KVCacheSettings,
        TGInferContextStore,
        OutputFilter
    )
    from mindie_llm.text_generator.utils.separate_deployment_engine import DmiModeNodeRole
    from mindie_llm.text_generator.adapter.generator_backend import GeneratorBackend

LAUNCH_DONE_TIMEOUT = 1
MEM_DETECT_INTERVAL = 1000


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
        self.last_sequence_ids = None
        self.previous_batch_is_prefill = False
        self.is_inference_pause = False
        self.mem_det_trigger_counter = 0
        self.error_code_collected_in_async = None
        # 结构化输出管理器 (延迟初始化)
        self._structured_output_manager: Optional[Any] = None
        self._structured_output_enabled = kwargs.get('enable_structured_output', True)

    @staticmethod
    def unsqueeze_sampling_output(sampling_output: SamplingOutput):
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
                host_array = field_value.cpu().numpy()
                setattr(new_instance, field.name, host_array)
        return new_instance

    @staticmethod
    def _fill_in_model_result_exp(model_input_wrapper, model_output_wrapper):
        filling_masks = model_input_wrapper.filling_masks
        model_inputs = model_input_wrapper.model_inputs
        method = None
        if method is None:
            # NOTE: Add MTP and other plugin-based capabilites later
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
        
    def clear_cache(
        self,
        sequence_ids: Iterable[int],
        cache_ids: Optional[Iterable[int]] = None,
        has_sampling: bool = True
    ):
        if cache_ids is None:
            self.infer_context.clear_context_by_seq_ids(sequence_ids)
        else:
            self.infer_context.clear_finished_context(sequence_ids, cache_ids)
        if has_sampling:
            self.sampler.clear_cache(sequence_ids)

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
        
        # 初始化结构化输出管理器
        self._init_structured_output_manager()

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
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'After preprocess', 
                                       trigger_count=self.mem_det_trigger_counter)

            prof = span_start("forward", True)
            span_req(prof, trace_ids)
            span_attr(prof, "blocks", count_block(input_metadata.block_tables))
            if hasattr(self.model_wrapper, "mapping"):
                span_attr(prof, "dp_rank", str(self.model_wrapper.mapping.attn_dp.rank))
                
            if ENV.framework_backend == BackendType.ATB:
                self.model_wrapper.model_runner.clear_internal_tensors()
                if (self.plugin_list and "mtp" not in self.plugin_list) or self.is_mix_model:
                    result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                            attn_mask=self.plugin_data_param.mask)  # q_len spec_mask
                # old graph forward
                else:
                    result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                            spec_mask=self.plugin_data_param.mask,
                                                            sub_model_inputs=self.plugin_data_param.mtp_model_inputs,
                                                            hidden_states=self.plugin_data_param.hidden_states)
            else:
                result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                        spec_mask=self.plugin_data_param.mask)  # q_len spec_mask
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result
            span_end(prof, True)
            self.watcher.watch_npu_mem(self.rank, f'After forward', trigger_count=self.mem_det_trigger_counter)

            prof = span_start("sample")
            draft_filtered_logits = self.sample_preprocess_manager(logits, result, sampling_metadata, input_metadata)
            sampling_output = self.generator_backend.sample(draft_filtered_logits, sampling_metadata)
            if ENV.framework_backend == BackendType.ATB:
                self.model_wrapper.model_runner.clear_internal_tensors()
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'After sample', trigger_count=self.mem_det_trigger_counter)
            logger.info("sample end", extra={"handler_ids": HandlerType.TOKEN})
            prof = span_start("postprocess")
            self.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
            generation_output = self.postprocess(
                cache_ids, input_metadata, result, sampling_metadata, sampling_output)
            generation_output.trace_ids = trace_ids
            generation_output.simulator_ids = input_metadata.simulator_ids
            span_end(prof)
            self.watcher.watch_npu_mem(self.rank, f'After postprocess', trigger_count=self.mem_det_trigger_counter)
            self.mem_det_trigger_counter_acc()
            return generation_output

        except Exception as e:
            error_code = convert_exception_to_error_code(str(e))
            if isinstance(e, RuntimeError) and error_code is not None:
                message = (
                    f'{error_code.name} fault happened in generate_token, error code: {error_code.value}.'
                )
                logger.error(message)
                raise ErrorCodeException(error_code) from e
            if self.is_inference_pause:
                logger.info(f"Mocking response due to inference pause for trace_ids={trace_ids}.")
                return GenerationOutput.make_empty()
            logger.exception(
                f"Unrecoverable error in generate_token (trace_ids={trace_ids}). "
                f"Terminating inference thread. Error: {e}"
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

            if self.generator_backend.dp > 1:
                cur_dp_rank_id_per_token_mask = model_input.dp_rank_ids == self.generator_backend.mapping.attn_dp.rank
                current_dp_sequence_ids = input_metadata.all_sequence_ids[cur_dp_rank_id_per_token_mask]
            else:
                current_dp_sequence_ids = input_metadata.all_sequence_ids

            filling_masks = self._prepare_masks_for_filling(
                model_input,
                current_dp_sequence_ids,
                input_metadata)

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
            if ENV.model_runner_exp is False:
                prof = span_start("fill_in_model_result") 
                if not is_mock and model_output_wrapper.model_output:	 
                    self._fill_in_model_result(input_metadata, model_input_wrapper, model_output_wrapper, 
                                                filling_masks, cache_ids)
                span_end(prof)

            prof = span_start("synchronize_processing_stream")
            self.generator_backend.synchronize()
            span_end(prof)

            prof = span_start("put_into_input_queue")
            self.input_queue.put(model_input_wrapper)
            span_end(prof)

            if not input_metadata.is_prefill and not self.previous_batch_is_prefill:
                prof = span_start("wait_to_postprocess")
                if model_output_wrapper.launch_done is not None and \
                not model_output_wrapper.launch_done.wait(timeout=LAUNCH_DONE_TIMEOUT):
                    if not self.is_inference_pause:
                        logger.warning("Timeout waitting for launch_done signal.")
                    else:
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
        
        # 生成结构化输出 bitmask (直接调用 StructuredOutputManager)
        if self._structured_output_manager is not None and sampling_metadata is not None:
            response_format_array = input_metadata.batch_response_format
            all_sequence_ids = sampling_metadata.all_sequence_ids
            
            if response_format_array is not None and all_sequence_ids is not None:
                num_with_constraint = sum(1 for rf in response_format_array if rf is not None)
                if num_with_constraint > 0:
                    bitmask = self._structured_output_manager.process_batch_for_generation(
                        sequence_ids=list(all_sequence_ids),
                        response_format_array=response_format_array,
                    )
                    if bitmask is not None:
                        sampling_metadata.guided_bitmask = bitmask
        
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

        if not self.async_inference:
            self.plugin_verify_manager(sampling_output, cache_ids, result)
        
        # 更新结构化输出 FSM 状态（直接调用 StructuredOutputManager）
        if self._structured_output_manager is not None and sampling_metadata is not None:
            all_sequence_ids = sampling_metadata.all_sequence_ids
            token_ids = sampling_output.token_ids
            if all_sequence_ids is not None and token_ids is not None:
                self._structured_output_manager.update_states_after_sampling(
                    sequence_ids=list(all_sequence_ids),
                    token_ids=token_ids,
                )

        finish_reason, filtered_indices, truncation_indices = (
            self.output_filter.filter_finished_sequences(cache_ids, input_metadata, sampling_output))

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
            if has_sampling and finished_sequence_ids.size != 0:
                self.sampler.clear_cache(finished_sequence_ids)
                # 清理结构化输出 FSM 状态（直接调用 StructuredOutputManager）
                if self._structured_output_manager is not None:
                    self._structured_output_manager.clear_finished_requests(finished_sequence_ids)
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
                sampling_output = self.generator_backend.sample(draft_filtered_logits,
                                                                model_input_wrapper.sampling_metadata)
                if ENV.framework_backend == BackendType.ATB:
                    self.model_wrapper.model_runner.clear_internal_tensors()
                span_end(prof)
                logger.info("sample end", extra={"handler_ids": HandlerType.TOKEN})
                if not self.is_inference_pause:
                    model_input_wrapper.postprocess_done.wait()

                if ENV.model_runner_exp:
                # NOTE: Add MTP and other plugin-based capabilites later
                    if len(sampling_output.token_ids.shape) != 2:
                        sampling_output.token_ids = torch.unsqueeze(sampling_output.token_ids, 1)
                        sampling_output.logprobs = np.expand_dims(sampling_output.logprobs, 1)
                        sampling_output.top_token_ids = np.expand_dims(sampling_output.top_token_ids, 1)
                        sampling_output.top_logprobs = np.expand_dims(sampling_output.top_logprobs, 1)
                else:
                    prof = span_start("verify")	 
                    self.plugin_verify_manager(
                        sampling_output, model_input_wrapper.cache_ids, model_output.original_result)
                    span_end(prof)

                prof = span_start("put_prefix_kvcache_to_mempool")
                if model_input_wrapper.cache_ids is not None and not model_input_wrapper.input_metadata.is_dummy_batch:
                    self.put_prefix_kvcache_to_mempool(
                        model_input_wrapper.input_metadata, 
                        model_input_wrapper.cache_ids
                    )
                span_end(prof)

                launch_done = threading.Event()
                model_output_wrapper = ModelOutputWrapper(
                    cache_ids=model_input_wrapper.cache_ids,
                    input_metadata=model_input_wrapper.input_metadata,
                    model_output=model_output,
                    sampling_metadata=model_input_wrapper.sampling_metadata,
                    sampling_output=sampling_output,
                    trace_ids=model_input_wrapper.trace_ids,
                    current_dp_sequence_ids=model_input_wrapper.current_dp_sequence_ids,
                    launch_done=launch_done
                )
            except Exception as e:
                trace_ids = getattr(model_input_wrapper, 'trace_ids', 'unknown')

                error_code = convert_exception_to_error_code(str(e))
                if isinstance(e, RuntimeError) and error_code is not None:
                    self.error_code_collected_in_async = error_code

                if self.is_inference_pause or self.error_code_collected_in_async is not None:
                    logger.info(f"Mocking response due to inference pause for trace_ids={trace_ids}.")
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
            
        except ImportError as e:
            logger.warning(f"Failed to import structured output module: {e}")
            self._structured_output_enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize structured output manager: {e}")
            self._structured_output_enabled = False

    def _prepare_masks_for_filling(self, model_inputs, current_dp_sequence_ids, input_metadata):
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
            if input_metadata.batch_is_prefill is not None:
                hit_sequence_ids_mask[input_metadata.batch_is_prefill] = False
            if hit_sequence_ids_mask.any():
                hit_sequence_ids = current_dp_sequence_ids[hit_sequence_ids_mask]
                if self.is_mix_model:
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