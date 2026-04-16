# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import queue
import threading
import copy
import time
from enum import IntEnum
import numpy as np
import torch

from mindie_llm.text_generator.plugins.plugin_manager import PluginManager
from mindie_llm.text_generator.utils import (
    GenerationOutput,
    InputMetadata,
    ModelInputWrapper,
    ModelOutputWrapper,
    SamplingOutput,
    NpuMemoryWatcher,
)
from mindie_llm.utils.decorators.time_decorator import timer
from mindie_llm.utils.log import logger
from mindie_llm.utils.prof.profiler import (
    span_start,
    span_end,
    span_req,
    span_attr,
    count_block,
)

VERIFY = "verify"


class RoleType(IntEnum):
    EDGE = 0
    CLOUD = 1


class PluginManagerLwd(PluginManager):
    def __init__(
        self,
        generator_backend,
        kvcache_settings,
        infer_context,
        output_filter,
        is_mix_model,
        plugin_list,
        model_role,
        watcher: NpuMemoryWatcher,
        **kwargs,
    ):
        super().__init__(
            generator_backend,
            kvcache_settings,
            infer_context,
            output_filter,
            is_mix_model,
            plugin_list,
            model_role,
            watcher,
            **kwargs,
        )

        role_type_str = kwargs.get("layerwise_disaggregated_role_type", "")
        self.role_type = RoleType.CLOUD if role_type_str == "slave" else RoleType.EDGE
        self.prefill_total_seq_len = 0
        self.cached_p_model_input_wrapper_queue = queue.Queue()
        self.cached_p_model_input_wrapper = None
        self.cached_d_model_input_wrapper = None

        if self.async_inference:
            self.return_queue = queue.Queue()

        self.clean_sequence_ids = None
        self.place_holder = np.array([0.0])
        self.sampling_out_tmp = SamplingOutput(
            sequence_ids=self.place_holder,
            parent_sequence_ids=self.place_holder,
            group_indices=[(0, 0)],
            repeating_indices=self.place_holder,
            token_ids=self.place_holder,
            logprobs=self.place_holder,
            top_token_ids=self.place_holder,
            top_logprobs=self.place_holder,
            cumulative_logprobs=self.place_holder,
            num_new_tokens=self.place_holder,
        )

        if self.infer_context.spcp_parallel_info.cp_size > 1:
            from .prefix_cache.prefix_cache_plugin import PrefixCachePlugin

            self.prefix_cache_plugin = PrefixCachePlugin(
                self.generator_backend,
                self.kvcache_settings,
                self.infer_context,
                self.plugin_data_param,
                **self.kwargs,
            )

    @staticmethod
    def lwd_sampling_output(input_metadata: InputMetadata):
        batch_size = input_metadata.batch_size
        sampling_output = SamplingOutput(
            sequence_ids=input_metadata.all_sequence_ids if input_metadata is not None else None,
            parent_sequence_ids=input_metadata.all_sequence_ids,
            group_indices=[(i, i + 1) for i in range(batch_size)],
            repeating_indices=np.arange(batch_size),
            token_ids=np.arange(batch_size),
            logprobs=np.full((batch_size,), -9999.0, dtype=np.float32),
            top_token_ids=np.zeros((batch_size, 0), dtype=np.int64),  # shape=[batchsize, 0/1]
            top_logprobs=np.zeros((batch_size, 0), dtype=np.float32),  # shape=[batchsize, 0/1]
            num_new_tokens=np.ones(batch_size, dtype=np.int64),  # [1,1,1,1] batchsize len
            num_top_tokens=np.ones(batch_size, dtype=np.int64),  # top_token_ids 0/1, len: batchsize
            cumulative_logprobs=np.arange(batch_size),
            seeds=None,
        )
        return sampling_output

    @staticmethod
    def should_skip_return_tokens(input_metadata):
        if not input_metadata.layerwise_disaggregated_exe_stage:
            return False
        is_p_first = (
            input_metadata.is_prefill and input_metadata.layerwise_disaggregated_exe_stage.start_exec_layer == 0
        )

        is_d_last = (
            not input_metadata.is_prefill and input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer == 1
        )

        is_not_p_chunk_last = (
            input_metadata.is_prefill
            and input_metadata.layerwise_disaggregated_exe_stage.end_exec_layer == 1
            and not input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token
        )
        return is_p_first or is_d_last or is_not_p_chunk_last

    def set_clean_sequence_ids(self, sequence_ids):
        self.clean_sequence_ids = sequence_ids
        # 多清一个0, 因为后面postprocess可能会用到0的cache来计算
        self.infer_context.clear_finished_context(np.array([0]), np.array([0]))

    def prepare_metadata_for_longseq_chunk_cp(self, input_metadata: InputMetadata) -> InputMetadata:
        lwd_exe_stage = input_metadata.layerwise_disaggregated_exe_stage
        cp_size = self.infer_context.spcp_parallel_info.cp_size
        sp_size = self.infer_context.spcp_parallel_info.sp_size
        scp_size = self.infer_context.spcp_parallel_info.scp_size
        block_size = self.generator_backend.block_size

        start_idx = lwd_exe_stage.long_seq_start_idx
        end_idx = lwd_exe_stage.long_seq_end_idx
        start_total_idx = lwd_exe_stage.long_seq_start_idx * cp_size
        end_total_idx = lwd_exe_stage.long_seq_end_idx * cp_size

        if not lwd_exe_stage.is_last_chunk:
            input_metadata.sp_tokens[:] = end_total_idx // scp_size
            input_metadata.input_ids = input_metadata.input_ids[:end_total_idx]
            input_metadata.total_seq_num = end_total_idx
            input_metadata.batch_seq_len = np.array([end_total_idx], dtype=np.int64)

            block_table_end_idx = (end_idx // (sp_size * block_size)) + (end_idx % (sp_size * block_size) != 0)
            input_metadata.batch_block_tables = input_metadata.batch_block_tables[:, :, :block_table_end_idx]

        if start_total_idx != 0:
            prefix_blocks = (start_idx + block_size - 1) // block_size

            remote_computed_blocks = np.array([[prefix_blocks // sp_size]], dtype=np.int32)
            remote_computed_blocks = np.repeat(remote_computed_blocks, scp_size, axis=1)
            input_metadata.remote_computed_blocks = np.array(remote_computed_blocks, dtype=np.int32)

        return input_metadata

    def model_inputs_update_manager_longseq_chunk_cp(
        self, model_inputs, input_metadata, sampling_metadata, cache_ids, **kwargs
    ):
        if not self.is_mix_model:
            self.plugin_data_param.q_len = None
            self.plugin_data_param.mask = None
        q_len = None
        spec_mask = None
        input_len_mask = (q_len, spec_mask)
        for plugin in self.plugin_list:
            plugin_instance = getattr(self, plugin, None)
            method = getattr(plugin_instance, "model_inputs_update", None)
            model_inputs, input_len_mask = method(
                model_inputs,
                input_metadata,
                sampling_metadata,
                cache_ids,
                input_len_mask,
                **kwargs,
            )
        model_inputs, input_len_mask = self.prefix_cache_plugin.model_inputs_update(
            model_inputs,
            input_metadata,
            sampling_metadata,
            cache_ids,
            input_len_mask,
            **kwargs,
        )
        (q_len, spec_mask) = input_len_mask
        self.plugin_data_param.q_len = q_len if q_len is not None else self.plugin_data_param.q_len
        self.plugin_data_param.mask = spec_mask if spec_mask is not None else self.plugin_data_param.mask
        return model_inputs, q_len, spec_mask

    @timer.track_time_async("generate_token")
    def generate_token(self, input_metadata: InputMetadata, warmup=False):
        prof = span_start("preprocess")
        cache_ids, model_inputs, sampling_metadata, trace_ids = self.preprocess(input_metadata, warmup=warmup)
        if not self.is_mix_model:
            self.plugin_data_param.q_len = None
            self.plugin_data_param.mask = None
        model_inputs, qlen, mask = self.model_inputs_update_manager(
            model_inputs, input_metadata, sampling_metadata, cache_ids
        )
        self.plugin_data_param.q_len = qlen if qlen is not None else self.plugin_data_param.q_len
        self.plugin_data_param.mask = mask if mask is not None else self.plugin_data_param.mask
        model_inputs.layerwise_disaggregated_exe_stage = input_metadata.layerwise_disaggregated_exe_stage
        span_end(prof)
        self.watcher.watch_npu_mem(self.rank, "After preprocess", trigger_count=self.mem_det_trigger_counter)

        prof = span_start("forward", True)
        span_req(prof, trace_ids)
        span_attr(prof, "blocks", count_block(input_metadata.block_tables))
        if hasattr(self.model_wrapper, "mapping"):
            span_attr(prof, "dp_rank", str(self.model_wrapper.mapping.attn_dp.rank))
        if input_metadata.layerwise_disaggregated_exe_stage:
            self.prefill_total_seq_len = (
                input_metadata.layerwise_disaggregated_exe_stage.prefill_total_seq_len
                if input_metadata.is_prefill
                else self.prefill_total_seq_len
            )
        cloud_cut_instance = self.model_wrapper.model_runner.time_counter

        if not warmup:
            self.model_wrapper.model_runner.clear_internal_tensors()
        if (self.plugin_list and "mtp" not in self.plugin_list) or self.is_mix_model:
            result = self.generator_backend.forward(
                model_inputs,
                q_lens=self.plugin_data_param.q_len,
                attn_mask=self.plugin_data_param.mask,
            )  # q_len spec_mask
        # old graph forward
        else:
            result = self.generator_backend.forward(
                model_inputs,
                q_lens=self.plugin_data_param.q_len,
                spec_mask=self.plugin_data_param.mask,
                sub_model_inputs=self.plugin_data_param.mtp_model_inputs,
                hidden_states=self.plugin_data_param.hidden_states,
            )

        cloud_cut_instance.set_decode_end_time(input_metadata.is_prefill, time.time())
        if (
            input_metadata.layerwise_disaggregated_exe_stage
            and not input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token
        ):
            return None

        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        span_end(prof, True)
        if warmup:
            torch.npu.synchronize()
            torch.npu.empty_cache()
        self.watcher.watch_npu_mem(self.rank, "After forward", trigger_count=self.mem_det_trigger_counter)

        prof = span_start("sample")
        if self.role_type == RoleType.EDGE:
            draft_filtered_logits = self.sample_preprocess_manager(logits, result, sampling_metadata, input_metadata)
            sampling_output = self.generator_backend.sample(draft_filtered_logits, sampling_metadata)
        else:
            sampling_output = self.lwd_sampling_output(input_metadata)

        if not warmup:
            self.model_wrapper.model_runner.clear_internal_tensors()
        self.watcher.watch_npu_mem(self.rank, "After sample", trigger_count=self.mem_det_trigger_counter)

        prof = span_start("postprocess")
        self.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
        generation_output = self.postprocess(cache_ids, input_metadata, result, sampling_metadata, sampling_output)
        generation_output.trace_ids = trace_ids
        generation_output.simulator_ids = input_metadata.simulator_ids
        span_end(prof)
        self.watcher.watch_npu_mem(self.rank, "After postprocess", trigger_count=self.mem_det_trigger_counter)
        self.mem_det_trigger_counter_acc()
        return generation_output

    def generate_token_async_edge(self, input_metadata: InputMetadata, warmup=False) -> GenerationOutput:
        with self.generator_backend.get_new_stream():
            cache_ids = None
            filling_masks = None
            postprocess_done = threading.Event()
            if warmup or input_metadata.layerwise_disaggregated_exe_stage.start_exec_layer == 0:
                prof = span_start("preprocess")
                if (
                    input_metadata.layerwise_disaggregated_exe_stage.is_long_seq
                    and not input_metadata.layerwise_disaggregated_exe_stage.request_dp_empty
                ):
                    if self.infer_context.spcp_parallel_info.cp_size > 1:
                        input_metadata = self.prepare_metadata_for_longseq_chunk_cp(input_metadata)
                hit_mask = np.isin(input_metadata.all_sequence_ids, self.last_sequence_ids)
                cache_ids, model_input, sampling_metadata, trace_ids = self.preprocess(
                    input_metadata, warmup=warmup, hit_mask=hit_mask
                )
                self.infer_context.last_sampling_metadata.clear()  # Do not use sampling cache under async inference.

                if (
                    input_metadata.layerwise_disaggregated_exe_stage.is_long_seq
                    and not input_metadata.layerwise_disaggregated_exe_stage.request_dp_empty
                ):
                    if self.infer_context.spcp_parallel_info.cp_size > 1:
                        model_input, _, _ = self.model_inputs_update_manager_longseq_chunk_cp(
                            model_input,
                            input_metadata,
                            sampling_metadata,
                            cache_ids,
                            hit_mask=hit_mask,
                        )
                else:
                    model_input, _, _ = self.model_inputs_update_manager(
                        model_input,
                        input_metadata,
                        sampling_metadata,
                        cache_ids,
                        hit_mask=hit_mask,
                    )
                span_end(prof)
                self.watcher.watch_npu_mem(
                    self.rank,
                    "In asyn inference mode, after preprocess",
                    trigger_count=self.mem_det_trigger_counter,
                )
                prof = span_start("prepare_model_inputs")

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
                        hidden_states=self.plugin_data_param.hidden_states,
                    )

                if self.generator_backend.dp > 1:
                    map_attn_dp_rank = self.generator_backend.mapping.attn_dp.rank
                    cur_dp_rank_id_per_token_mask = model_input.dp_rank_ids == map_attn_dp_rank
                    current_dp_sequence_ids = input_metadata.all_sequence_ids[cur_dp_rank_id_per_token_mask]
                else:
                    current_dp_sequence_ids = input_metadata.all_sequence_ids

                filling_masks = self._prepare_masks_for_filling(model_input, current_dp_sequence_ids, input_metadata)

                model_input_wrapper = ModelInputWrapper(
                    cache_ids,
                    input_metadata,
                    model_input,
                    model_kwargs,
                    sampling_metadata,
                    trace_ids,
                    current_dp_sequence_ids,
                    postprocess_done,
                    warmup,
                )
                if not warmup:
                    if input_metadata.is_prefill:
                        self.cached_p_model_input_wrapper_queue.put(model_input_wrapper)
                    else:
                        self.cached_d_model_input_wrapper = model_input_wrapper
                span_end(prof)
            else:
                prof = span_start("preprocess_and_prepare_model_inputs")
                if input_metadata.is_prefill:
                    if self.cached_p_model_input_wrapper is None:
                        self.cached_p_model_input_wrapper = self.cached_p_model_input_wrapper_queue.get(timeout=900)

                    model_input_wrapper = copy.copy(self.cached_p_model_input_wrapper)

                    self.cached_p_model_input_wrapper = None
                else:
                    model_input_wrapper = copy.copy(self.cached_d_model_input_wrapper)
                model_input_wrapper.input_metadata = input_metadata
                span_end(prof)
            logger.info(
                f"[layerwiseDisaggregated] get_from_output_queue before is_prefill {input_metadata.is_prefill} "
                f"input_metadata.all_sequence_ids {input_metadata.all_sequence_ids}"
            )
            prof = span_start("get_from_output_queue")
            model_output_wrapper = self.output_queue.get(timeout=900)
            span_end(prof)
            logger.info(f"[layerwiseDisaggregated] get_from_output_queue after is_prefill {input_metadata.is_prefill}")

            # Only D-begin tasks require fill_in; fill according to the filling_masks.
            if warmup or (
                not input_metadata.is_prefill and input_metadata.layerwise_disaggregated_exe_stage.start_exec_layer == 0
            ):
                prof = span_start("fill_in_model_result")
                self._fill_in_model_result(
                    input_metadata,
                    model_input_wrapper,
                    model_output_wrapper,
                    filling_masks,
                    cache_ids,
                )
                self.generator_backend.synchronize()
                span_end(prof)

            prof = span_start("put_into_input_queue")
            self.input_queue.put(model_input_wrapper)
            span_end(prof)

            prof = span_start("wait_to_postprocess")
            if model_output_wrapper.launch_done is not None:
                model_output_wrapper.launch_done.wait()
            span_end(prof)

            if warmup or (
                model_output_wrapper.input_metadata
                and model_output_wrapper.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token
            ):
                self.watcher.watch_npu_mem(
                    self.rank,
                    "In asyn inference mode, before postprocess",
                    trigger_count=self.mem_det_trigger_counter,
                )
                prof = span_start("postprocess")
                if (
                    model_output_wrapper.cache_ids is not None
                    and not model_output_wrapper.input_metadata.is_dummy_batch
                ):
                    if model_output_wrapper.model_output.hidden_states is None:
                        model_result = model_output_wrapper.model_output.logits
                    else:
                        model_result = (
                            model_output_wrapper.model_output.logits,
                            model_output_wrapper.model_output.hidden_states,
                        )
                    self.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
                    if self.clean_sequence_ids and model_output_wrapper.input_metadata.is_prefill:
                        # For P tasks that are recomputed, clear their cache_id and do not perform postprocess.
                        all_sequence_ids_dict = {}
                        for i, x in enumerate(model_output_wrapper.input_metadata.all_sequence_ids):
                            all_sequence_ids_dict[x] = i

                        clean_positions = [all_sequence_ids_dict.get(x, -1) for x in self.clean_sequence_ids]
                        for idx in clean_positions:
                            if idx != -1:
                                logger.info(
                                    f"[layerwiseDisaggregated]recompute seq: {self.clean_sequence_ids} "
                                    f"idx:{idx}"
                                    f" out_seq_ids {model_output_wrapper.input_metadata.all_sequence_ids}"
                                    f" cache_ids:{model_output_wrapper.cache_ids} "
                                    f" clear model_output_wrapper cache."
                                )
                                model_output_wrapper.cache_ids[idx] = 0
                    generation_output_tmp = self.postprocess(
                        model_output_wrapper.cache_ids,
                        model_output_wrapper.input_metadata,
                        model_result,
                        model_output_wrapper.sampling_metadata,
                        model_output_wrapper.sampling_output,
                    )
                    self.return_queue.put(generation_output_tmp)
                    logger.info(
                        f"[layerwiseDisaggregated]last request is_prefill "
                        f"{model_output_wrapper.input_metadata.is_prefill} "
                        f"all_sequence_ids {model_output_wrapper.input_metadata.all_sequence_ids}"
                    )
                span_end(prof)
                self.watcher.watch_npu_mem(
                    self.rank,
                    "In asyn inference mode, after postprocess",
                    trigger_count=self.mem_det_trigger_counter,
                )
            postprocess_done.set()

            # If it is a P-first or D-last, do not return tokens.
            if self.should_skip_return_tokens(input_metadata):
                self.last_sequence_ids = input_metadata.all_sequence_ids
                return None

            generation_output = None
            prof = span_start("generation_output")
            while not self.return_queue.empty():
                generation_output_tmp = self.return_queue.get()
                if self.clean_sequence_ids:
                    # For P/D that undergo re-computation, clear its token_ids and do not return any results.
                    out_sequence_ids_dict = {x: i for i, x in enumerate(generation_output_tmp.sequence_ids)}
                    clean_positions = [out_sequence_ids_dict.get(x, -1) for x in self.clean_sequence_ids]
                    for idx in clean_positions:
                        # After triggering recomputation, calculation will continue using cache_id 0.
                        # All generated tokens are untrustworthy and need to be discarded.
                        if idx != -1:
                            logger.info(
                                f"[layerwiseDisaggregated]recompute seq: {self.clean_sequence_ids} idx:{idx} "
                                f"out_seq_ids {generation_output_tmp.sequence_ids} "
                                f"clear generation_output token."
                            )
                            generation_output_tmp.token_ids[idx] = -1
                            generation_output_tmp.top_token_ids[idx] = -1
                            generation_output_tmp.finish_reason[idx] = 0

                if self.return_queue.empty():
                    self.clean_sequence_ids = None

                if generation_output:
                    generation_output.concatenate_output(generation_output_tmp, self.max_generated_tokens)
                else:
                    generation_output = generation_output_tmp
                logger.info("[layerwiseDisaggregated]concatenate last generation_output")

            if generation_output is None:
                generation_output = GenerationOutput.make_empty()
            generation_output.fill_dummy(input_metadata, self.max_generated_tokens)
            logger.info(
                f"[layerwiseDisaggregated] curr request is_prefill {input_metadata.is_prefill} "
                f"input_metadata.all_sequence_ids {input_metadata.all_sequence_ids}"
            )
            self.last_sequence_ids = input_metadata.all_sequence_ids
            span_end(prof)
            self.watcher.watch_npu_mem(
                self.rank,
                "In asyn inference mode, after generation_output",
                trigger_count=self.mem_det_trigger_counter,
            )
            self.mem_det_trigger_counter_acc()
        return generation_output

    def generate_token_async_cloud(self, input_metadata: InputMetadata, warmup=False) -> GenerationOutput:
        with self.generator_backend.get_new_stream():
            cache_ids = None
            postprocess_done = threading.Event()
            if warmup or input_metadata.layerwise_disaggregated_exe_stage.start_exec_layer == 0:
                prof = span_start("preprocess")
                if (
                    input_metadata.layerwise_disaggregated_exe_stage.is_long_seq
                    and not input_metadata.layerwise_disaggregated_exe_stage.request_dp_empty
                ):
                    if self.infer_context.spcp_parallel_info.cp_size > 1:
                        input_metadata = self.prepare_metadata_for_longseq_chunk_cp(input_metadata)

                hit_mask = np.isin(input_metadata.all_sequence_ids, self.last_sequence_ids)
                cache_ids, model_input, sampling_metadata, trace_ids = self.preprocess(
                    input_metadata, warmup=warmup, hit_mask=hit_mask
                )
                self.infer_context.last_sampling_metadata.clear()  # Do not use sampling cache under async inference.
                if (
                    input_metadata.layerwise_disaggregated_exe_stage.is_long_seq
                    and not input_metadata.layerwise_disaggregated_exe_stage.request_dp_empty
                ):
                    if self.infer_context.spcp_parallel_info.cp_size > 1:
                        model_input, _, _ = self.model_inputs_update_manager_longseq_chunk_cp(
                            model_input,
                            input_metadata,
                            sampling_metadata,
                            cache_ids,
                            hit_mask=hit_mask,
                        )
                else:
                    model_input, _, _ = self.model_inputs_update_manager(
                        model_input,
                        input_metadata,
                        sampling_metadata,
                        cache_ids,
                        hit_mask=hit_mask,
                    )
                span_end(prof)

                prof = span_start("prepare_model_inputs")

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
                        hidden_states=self.plugin_data_param.hidden_states,
                    )

                if self.generator_backend.dp > 1:
                    mapp_attn_dp_rank = self.generator_backend.mapping.attn_dp.rank
                    cur_dp_rank_id_per_token_mask = model_input.dp_rank_ids == mapp_attn_dp_rank
                    current_dp_sequence_ids = input_metadata.all_sequence_ids[cur_dp_rank_id_per_token_mask]
                else:
                    current_dp_sequence_ids = input_metadata.all_sequence_ids

                filling_masks = self._prepare_masks_for_filling(model_input, current_dp_sequence_ids, input_metadata)

                model_input_wrapper = ModelInputWrapper(
                    cache_ids,
                    input_metadata,
                    model_input,
                    model_kwargs,
                    sampling_metadata,
                    trace_ids,
                    current_dp_sequence_ids,
                    postprocess_done,
                )
                if input_metadata.is_prefill:
                    self.cached_p_model_input_wrapper = model_input_wrapper
                span_end(prof)
            else:
                prof = span_start("preprocess_and_prepare_model_inputs")
                if input_metadata.is_prefill:
                    model_input_wrapper = copy.copy(self.cached_p_model_input_wrapper)
                else:
                    logger.info(
                        f"[layerwiseDisaggregated] Cloud has decode layer wrong: "
                        f"{input_metadata.layerwise_disaggregated_exe_stage.start_exec_layer}"
                    )
                model_input_wrapper.input_metadata = input_metadata
                span_end(prof)
            logger.info(
                f"[layerwiseDisaggregated] get_from_output_queue before is_prefill {input_metadata.is_prefill} "
                f"input_metadata.all_sequence_ids {input_metadata.all_sequence_ids}"
            )
            prof = span_start("get_from_output_queue")
            model_output_wrapper = self.output_queue.get(timeout=900)
            span_end(prof)
            logger.info(f"[layerwiseDisaggregated] get_from_output_queue after is_prefill {input_metadata.is_prefill}")

            is_decode_end_of_generate_token = (
                not input_metadata.is_prefill
                and model_output_wrapper.input_metadata
                and model_output_wrapper.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token
            )

            if warmup or is_decode_end_of_generate_token:
                prof = span_start("fill_in_model_result")
                self._fill_in_model_result(
                    input_metadata,
                    model_input_wrapper,
                    model_output_wrapper,
                    filling_masks,
                    cache_ids,
                )
                self.generator_backend.synchronize()
                span_end(prof)

            prof = span_start("put_into_input_queue")
            self.input_queue.put(model_input_wrapper)
            span_end(prof)

            prof = span_start("wait_to_postprocess")
            if model_output_wrapper.launch_done is not None:
                model_output_wrapper.launch_done.wait()
            span_end(prof)

            if not warmup and (
                model_output_wrapper.input_metadata
                and not model_output_wrapper.input_metadata.layerwise_disaggregated_exe_stage.end_of_generate_token
            ):
                self.last_sequence_ids = input_metadata.all_sequence_ids
                postprocess_done.set()
                return None

            self.watcher.watch_npu_mem(
                self.rank,
                "In asyn inference mode, before postprocess",
                trigger_count=self.mem_det_trigger_counter,
            )
            prof = span_start("postprocess")
            if model_output_wrapper.cache_ids is not None and not model_output_wrapper.input_metadata.is_dummy_batch:
                if model_output_wrapper.model_output.hidden_states is None:
                    model_result = model_output_wrapper.model_output.logits
                else:
                    model_result = (
                        model_output_wrapper.model_output.logits,
                        model_output_wrapper.model_output.hidden_states,
                    )
                self.put_prefix_kvcache_to_mempool(input_metadata, cache_ids)
                generation_output = self.postprocess(
                    model_output_wrapper.cache_ids,
                    model_output_wrapper.input_metadata,
                    model_result,
                    model_output_wrapper.sampling_metadata,
                    model_output_wrapper.sampling_output,
                )
                generation_output.trace_ids = model_output_wrapper.trace_ids
            else:
                generation_output = GenerationOutput.make_empty()
            postprocess_done.set()
            generation_output.fill_dummy(input_metadata, self.max_generated_tokens)
            self.last_sequence_ids = input_metadata.all_sequence_ids
            span_end(prof)
            self.watcher.watch_npu_mem(
                self.rank,
                "In asyn inference mode, after postprocess",
                trigger_count=self.mem_det_trigger_counter,
            )
            self.mem_det_trigger_counter_acc()
        return generation_output

    def prepare_inputs_for_longseq_chunk(self, model_input_wrapper: ModelInputWrapper):
        block_size = self.generator_backend.block_size
        lwd_exe_stage = model_input_wrapper.input_metadata.layerwise_disaggregated_exe_stage
        block_table_end_idx = (lwd_exe_stage.long_seq_end_idx // block_size) + (
            lwd_exe_stage.long_seq_end_idx % block_size != 0
        )
        model_input_wrapper.model_inputs.input_lengths[0] = lwd_exe_stage.long_seq_end_idx
        model_input_wrapper.model_inputs.max_seq_len = lwd_exe_stage.long_seq_end_idx
        model_input_wrapper.model_inputs.lm_head_indices[0] = (
            lwd_exe_stage.long_seq_end_idx - lwd_exe_stage.long_seq_start_idx - 1
        )
        if lwd_exe_stage.long_seq_start_idx != 0:
            q_lens = np.array(
                [lwd_exe_stage.long_seq_end_idx - lwd_exe_stage.long_seq_start_idx],
                dtype=np.int32,
            ).tolist()
            model_input_wrapper.model_kwargs.update({"q_lens": q_lens})
            attn_mask = self.generator_backend.model_wrapper.model_runner.attn_mask.get_splitfuse_mask(
                self.generator_backend.model_wrapper.device
            )
            model_input_wrapper.model_kwargs.update({"attn_mask": attn_mask})
        model_input_wrapper.model_inputs.input_ids = model_input_wrapper.model_inputs.input_ids[
            lwd_exe_stage.long_seq_start_idx : lwd_exe_stage.long_seq_end_idx
        ]
        model_input_wrapper.model_inputs.position_ids = model_input_wrapper.model_inputs.position_ids[
            lwd_exe_stage.long_seq_start_idx : lwd_exe_stage.long_seq_end_idx
        ]
        model_input_wrapper.model_inputs.block_tables = model_input_wrapper.model_inputs.block_tables[
            :, 0:block_table_end_idx
        ]
        model_input_wrapper.model_inputs.slots = model_input_wrapper.model_inputs.slots[
            lwd_exe_stage.long_seq_start_idx : lwd_exe_stage.long_seq_end_idx
        ]

    def generate_token_async(self, input_metadata: InputMetadata, warmup=False) -> GenerationOutput:
        if self.role_type == RoleType.CLOUD:
            return self.generate_token_async_cloud(input_metadata, warmup)

        return self.generate_token_async_edge(input_metadata, warmup)

    def forward_loop(self):
        self.generator_backend.set_device()
        launch_done = None
        while True:
            prof = span_start("get_from_input_queue")
            model_input_wrapper = self.input_queue.get()
            span_end(prof)
            layerwise_disaggregated_exe_stage = model_input_wrapper.input_metadata.layerwise_disaggregated_exe_stage
            if layerwise_disaggregated_exe_stage:
                self.prefill_total_seq_len = (
                    layerwise_disaggregated_exe_stage.prefill_total_seq_len
                    if model_input_wrapper.input_metadata.is_prefill
                    else self.prefill_total_seq_len
                )
            cloud_cut_instance = self.model_wrapper.model_runner.time_counter
            prof = span_start("forward", True)
            span_req(prof, model_input_wrapper.trace_ids)
            span_attr(prof, "async", True)

            self.model_wrapper.model_runner.clear_internal_tensors()
            exe_stage_kwargs = {
                "layerwise_disaggregated_exe_stage": model_input_wrapper.input_metadata.layerwise_disaggregated_exe_stage
            }
            model_input_wrapper.model_kwargs.update(exe_stage_kwargs)

            if layerwise_disaggregated_exe_stage and layerwise_disaggregated_exe_stage.is_long_seq:
                if not layerwise_disaggregated_exe_stage.request_dp_empty:
                    if self.infer_context.spcp_parallel_info.cp_size == 1:
                        self.prepare_inputs_for_longseq_chunk(model_input_wrapper)
                model_output = self.generator_backend.forward_from_model_inputs(
                    model_input_wrapper.model_inputs, **model_input_wrapper.model_kwargs
                )
                model_input_wrapper.model_kwargs.update({"q_lens": None})
                model_input_wrapper.model_kwargs.pop("attn_mask", None)
            else:
                model_output = self.generator_backend.forward_from_model_inputs(
                    model_input_wrapper.model_inputs, **model_input_wrapper.model_kwargs
                )

            if launch_done is not None:
                launch_done.set()
            span_end(prof, True)

            if not layerwise_disaggregated_exe_stage or layerwise_disaggregated_exe_stage.end_of_generate_token:
                prof = span_start("sample")
                if self.role_type == RoleType.CLOUD:
                    sampling_output = self.lwd_sampling_output(model_input_wrapper.input_metadata)
                    model_output.logits[0][0].cpu()
                else:
                    draft_filtered_logits = self.sample_preprocess_manager(
                        model_output.logits,
                        model_output.original_result,
                        model_input_wrapper.sampling_metadata,
                        model_input_wrapper.input_metadata,
                    )
                    sampling_output = self.generator_backend.sample(
                        draft_filtered_logits, model_input_wrapper.sampling_metadata
                    )

                    self.model_wrapper.model_runner.clear_internal_tensors()
                span_end(prof)

                model_input_wrapper.postprocess_done.wait()
                prof = span_start(VERIFY)
                self.plugin_verify_manager(
                    sampling_output,
                    model_input_wrapper.cache_ids,
                    model_output.original_result,
                )
                span_end(prof)
            else:
                # Synchronize hidden state  CPU-NPU.
                model_output.logits[0][0].cpu()
                model_input_wrapper.postprocess_done.wait()
                sampling_output = self.sampling_out_tmp

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
            cloud_cut_instance.set_decode_end_time(model_input_wrapper.input_metadata.is_prefill, time.time())
            self.output_queue.put(model_output_wrapper)
