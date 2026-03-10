# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Iterable, Optional
import os
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F

from mindie_llm.runtime.utils.helpers.env import ENV
from mindie_llm.runtime.utils.cpu.affinity import bind_cpus
from mindie_llm.runtime.utils.npu.device_utils import get_npu_hbm_info
from mindie_llm.runtime.models import get_router_ins
from mindie_llm.runtime.model_runner.forward_context import create_forward_context, set_forward_context, \
    get_forward_context, AttentionMetadata, BatchDescriptor
from mindie_llm.runtime.utils.npu.device_utils import get_npu_node_info
from mindie_llm.runtime.utils.torch_utils import set_default_torch_dtype
from mindie_llm.runtime.utils.loader.default_model_loader import DefaultModelLoader
from mindie_llm.runtime.utils.distributed import get_parallel_info_manager, init_distributed
from mindie_llm.runtime.utils.distributed.parallel_info_manager import ParallelType
from mindie_llm.runtime.utils.distributed.utils import set_device
from mindie_llm.runtime.layers.attention import get_global_attn_dict, clear_global_attn_dict, flush_global_attn_dict
from mindie_llm.runtime.compilation.aclgraph_backend import set_aclgraph_capturing_enabled, AclGraphBackend, \
    set_global_graph_memory_pool
from mindie_llm.runtime.config.mindie_llm_config import MindIELLMConfig, SpeculativeConfig
from mindie_llm.runtime.config.load_config import LoadConfig
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.runtime.layers.attention.sparse_attention_layer import SFA
from .spec_worker import auto_speculative_method_router, speculative_worker_selector
# Allow tensor initialization and casting with internal format(e.g., NZ)
torch.npu.config.allow_internal_format = True


@auto_speculative_method_router(selector_fn=speculative_worker_selector)
class ModelRunner:

    input_metadata: dict = None

    def __init__(self,
                 model_name_or_path: str,
                 rank: int,
                 world_size: int,
                 npu_id: Optional[int] = None,
                 local_rank: Optional[int] = None,
                 load_tokenizer: bool = True,
                 max_position_embeddings: Optional[int] = None,
                 tokenizer_path: Optional[str] = None,
                 llm_config_path: str = None,
                 models_dict: dict = None,
                 **kwargs):
        # parse configs
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.world_size = world_size
        self.inference_mode = kwargs.get("inference_mode")
        self.plugin_params = kwargs.get("plugin_params", None)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.num_speculative_tokens = kwargs.get('num_speculative_tokens', 0)
        self.distributed_enable = kwargs.get("distributed_enable", False)
        self.max_batch_size = kwargs.get("max_batch_size", -1)
        self.model_role = kwargs.get("model_role", "standard")
        self.max_seq_len = kwargs.get("max_seq_len", -1)
        self.block_size = kwargs.get("block_size", -1)
        self.device = set_device(self.rank, self.npu_id)
        self.soc_info = get_npu_node_info()

        # bin cpus to the NUMA
        if ENV.bind_cpu:
            try:
                bind_cpus(ratio=1.0)
            except RuntimeError as e:
                print_log(rank, logger.info, e)
            except ValueError as e:
                print_log(rank, logger.info, e)
            except Exception as _:
                print_log(rank, logger.info, 'Skip binding cpu.')

        load_config_dict = {
            "model_name_or_path": model_name_or_path,
            "is_flash_causal_lm": True,
            "load_tokenizer": load_tokenizer,
            "max_position_embeddings": max_position_embeddings,
            "revision": None,
            "tokenizer_path": tokenizer_path,
            "trust_remote_code": self.trust_remote_code,
            "enable_atb_torch": False,
            "enable_edge": False,
            "llm_config_path": llm_config_path,
            "models_dict": models_dict,
            "sub_model_path": f'part{self.local_rank}-of-{self.world_size}'
        }
        load_config = LoadConfig.from_dict(load_config_dict)
        router_ins = get_router_ins(load_config)

        #NOTE: to be refactored to enum values for better extensibility across model types
        self.is_draft_model = kwargs.get("is_draft_model", False)
        self.model_cls = router_ins.draft_cls if self.is_draft_model else router_ins.model_cls

        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.config_dict = router_ins.config_dict
        self.llm_config = router_ins.llm_config

        self.model = None

        if hasattr(self.config, "max_position_embeddings"):
            self.max_position_embeddings = self.config.max_position_embeddings
        else:
            self.max_position_embeddings = 2048
        self.dtype = self.config.torch_dtype

        self.enable_nz = self.llm_config.llm.kv_cache_options.enable_nz

        print_log(rank, logger.info, f'model_runner.dtype: {self.dtype}', need_filter=True)

        # NOTE: we will discuss lora later.

        # init distributed
        init_distributed(self.rank, self.world_size, self.local_rank, llm_config=self.llm_config, server_config=kwargs)
        self.mapping = get_parallel_info_manager()
        self.cp_size = self.mapping.attn_cp.group_size
        self.process_group = self.mapping.world.process_group  # NOTE: depreciated

        self.mindie_llm_config = MindIELLMConfig(
            self.model_name_or_path,
            self.config,
            self.llm_config,
            router_ins.generation_config,
            speculative_config=SpeculativeConfig(self.num_speculative_tokens)
        )
        quant_config = getattr(self.mindie_llm_config, 'quant_config', None)
        if quant_config and getattr(quant_config, 'kv_quant_type', None) is not None:
            self.kv_cache_dtype = torch.int8
        else:
            self.kv_cache_dtype = self.dtype
        self.mask = None
        
        self.head_size = None
        self.num_heads = None
        self.num_kv_heads = None
        self.num_layers = None
        self.k_head_size = None
        self.v_head_size = None
        self.index_head_dim = None
        self.num_index_heads = None

        self.kvcache_quant_layers = []
        self.adapter_manager = None
        self.lora_adapter = None

        # All attention layers in model
        self.attn_layers = None

        # False: eager mode, True: AclGraph mode; 
        # prefill node in pd disaggregated mode do not need acl graph currently.
        self.enable_acl_graph = not self.model_role == "prefill"
        
        # we store tensors here, decide not to use auto padding now.
        if self.enable_acl_graph:
            max_graph_batch_size = self.max_batch_size * \
                (self.num_speculative_tokens + 1) if self.max_batch_size > 0 else 128 
            self.graph_batch_sizes = [1, 2, 4] + list(range(8, max_graph_batch_size + 8, 8))
            max_num_token = self.graph_batch_sizes[-1]
            self.input_ids = torch.zeros(max_num_token, dtype=torch.int32, device=self.device)
            self.position_ids = torch.zeros(max_num_token, dtype=torch.int64, device=self.device)
            self.seq_lens = torch.zeros(max_num_token, dtype=torch.int32, device=self.device)

            max_block_num = (self.max_seq_len + self.block_size - 1) // self.block_size
            self.block_tables = torch.zeros((max_num_token, max_block_num), dtype=torch.int32, device=self.device)
            self.slot_mapping = -torch.ones(max_num_token, dtype=torch.int32, device=self.device)
            self.lmhead_indices = torch.zeros(max_num_token, dtype=torch.int64, device=self.device)

            self.actual_seq_lengths_kv = torch.zeros(max_num_token, dtype=torch.int32, device=self.device)
            self.actual_seq_lengths_query = torch.zeros(max_num_token, dtype=torch.int32, device=self.device)

            self.mc2_mask = torch.zeros(max_num_token, dtype=torch.bool, device=self.device)
            # MTP
            self.hidden_states_mtp = torch.zeros(
                (max_num_token, self.mindie_llm_config.hf_config.hidden_size), 
                dtype=self.mindie_llm_config.hf_config.torch_dtype, 
                device=self.device
            )
            logger.info(f"AclGraph enabled. Graph batch sizes contains {self.graph_batch_sizes}.")

        self.mtp_count = 0

    def load_weights(self, **kwargs):
        if "OMP_NUM_THREADS" not in os.environ and self.world_size > 1:
            os.environ["OMP_NUM_THREADS"] = "1"

        # load model
        if self.max_seq_len > self.mindie_llm_config.hf_config.max_position_embeddings:
            _msg = f"`max_seq_len` cannot be larger than `max_position_embeddings`\
                or `max_position_embeddings`*`scaling_factor` when scaling."
            logger.error(_msg)
            raise ValueError(_msg)
        with set_default_torch_dtype(self.config.torch_dtype):
            with self.device:
                self.model = self.model_cls(self.mindie_llm_config)
        print_log(self.rank, logger.info, f"initialize model cls done")
        DefaultModelLoader().load_weights(self.model, self.model_name_or_path)
        print_log(self.rank, logger.info, f"load weight done")

        self.mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).npu()
        if self.world_size > 1:
            torch_dist.barrier()
        self.head_size = self.mindie_llm_config.hf_config.head_dim
        self.num_heads = self.mindie_llm_config.hf_config.get_num_attention_heads_per_rank()
        self.num_kv_heads = self.mindie_llm_config.hf_config.get_num_kv_heads_per_rank()
        self.num_layers = self.mindie_llm_config.hf_config.num_hidden_layers
        self.index_head_dim = getattr(self.mindie_llm_config.hf_config, "index_head_dim", None)
        self.num_index_heads = getattr(self.mindie_llm_config.hf_config, "index_n_heads", None)

        self.attn_layers = get_global_attn_dict().copy()
        clear_global_attn_dict()

        # not equal k v length for mla
        if hasattr(self.model, 'kv_lora_rank') and hasattr(self.model, 'qk_rope_head_dim'):   # deepseekv2/v3/r1
            self.num_kv_heads = 1
            if self.index_head_dim is not None:
                self.num_index_heads = 1
            self.k_head_size = self.model.kv_lora_rank
            self.v_head_size = self.model.qk_rope_head_dim
        else:
            self.k_head_size = self.head_size
            self.v_head_size = self.head_size
        
        print_log(self.rank, logger.info, f'model:\n {self.model}')

        if self.enable_acl_graph:
            # NOTE: `model_runner.py` and `model_runner_exp.py` share the same `AclGraphBackend` class.
            # The calculation of capture sizes has been moved into `AclGraphBackend`.
            self.model = AclGraphBackend(self.model, self.graph_batch_sizes[-1])
            logger.info(f"AclGraph enabled. Graph batch sizes contains {self.model.capture_sizes}.")

    def warm_up_and_compile(self, **kwargs):
        if not self.enable_acl_graph:
            return

        max_memory = get_npu_hbm_info().get_hbm_capacity()
        current_memory = int(max_memory * get_npu_hbm_info().get_hbm_usage()) / (1024 ** 3)
        logger.info(f"Before capturing, {current_memory=}")

        set_aclgraph_capturing_enabled(True)
        set_global_graph_memory_pool(None)
        self.model.graphs.clear()
        for num_tokens in tqdm(
            list(reversed(self.graph_batch_sizes)),
            desc="Capturing acl graph",
            disable=self.rank
        ):
            self._dummy_run(num_tokens)
        set_aclgraph_capturing_enabled(False)

        current_memory = int(max_memory * get_npu_hbm_info().get_hbm_usage()) / (1024 ** 3)
        logger.info(f"After capturing, {current_memory=}")

        output_buffer_hbm = 0
        for k in self.model.output_buffer:
            v = self.model.output_buffer[k]
            if torch.is_tensor(v):
                tmp = v.numel() * v.element_size() / (1024 ** 2)
                output_buffer_hbm += tmp
        logger.info(f"After capturing, {output_buffer_hbm=} MB")
    
    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        """Generate position ids."""
        position_ids = self.input_builder.generate_position_ids(input_ids)
        return position_ids

    def forward(self, **kwargs):
        # NOTE: This will be removed after get_rope is ready
        flush_global_attn_dict(self.attn_layers)
        kv_cache = kwargs.get("kv_cache", None)
        if self.is_draft_model:
            kv_cache = kv_cache[-1:]
        # When the address of kv_cache changes, we will recapture graphs.
        if id(next(iter(self.attn_layers.values())).key_cache) != id(kv_cache[0][0]):
            bind_kv_cache(kv_cache, self.attn_layers)
            self.warm_up_and_compile(**kwargs)

        input_ids, position_ids, input_metadata = self._prepare_inputs(**kwargs)
        is_prefill = kwargs.get("is_prefill", True)
        if not is_prefill:
            ModelRunner.input_metadata = input_metadata
            if self.is_draft_model:
                self.mtp_count += 1
    
        attn_metadata_dict = build_layerwise_attn_metadata(input_metadata, self.attn_layers)
        forward_context = create_forward_context(input_metadata)
        forward_context.attn_metadata_dict = attn_metadata_dict
        # NOTE: this flag will be update to FlashCommMetaData()
        forward_context.batch_descriptor = BatchDescriptor(forward_context.batch_descriptor.num_tokens,
            get_parallel_info_manager().get(ParallelType.ATTN_DP).is_enabled())
        set_forward_context(forward_context)
        hidden_states = self.model(input_ids, position_ids)
        hidden_states = maybe_gather_and_unpad_for_flashcomm(hidden_states)
        # (lzp) will be removed after PDmix support dp in and out.
        if not self.distributed_enable:
            hidden_states = maybe_pad_and_gather_cross_dp_and_unpad(hidden_states)
        if not self.is_draft_model and self.cp_size > 1:
            hidden_states_cp = maybe_allgather_cp(hidden_states)
            logits = self.model.compute_logits(hidden_states_cp)
        else:
            logits = self.model.compute_logits(hidden_states)
        if self.num_speculative_tokens > 0:
            return logits, hidden_states
        return logits

    def clear_internal_tensors(self):
        # NOTE: need to delete after TG is refactored
        pass   

    def _prepare_inputs(self, **kwargs) -> dict:
        is_prefill = kwargs.get("is_prefill", True)
        try:
            input_ids = kwargs["input_ids"]
            position_ids = kwargs["position_ids"].to(torch.int64)
        except KeyError as e:
            raise ValueError("Inputs must contains `input_ids` and `position_ids`.") from e

        is_prefill_or_no_mtp = is_prefill or self.num_speculative_tokens == 0
        is_mtp_0 = self.is_draft_model and self.mtp_count % self.num_speculative_tokens == 0
        is_mtp_0_or_main = not self.is_draft_model or is_mtp_0

        # NOTE: I recommend to put the following things to Attention and LMHead module's `build_input_metadata` methods.
        mask = self.mask
        slot_mapping = kwargs.get("slots", None).to(torch.int32)
        if is_prefill_or_no_mtp or is_mtp_0_or_main:
            seq_lens = kwargs.get("input_lengths", None).to(torch.int32)
            seq_lens_list = seq_lens.cpu().tolist()
            block_tables = kwargs.get("block_tables", None)

            lm_head_indices = kwargs.get("lm_head_indices", None)
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                            dtype=torch.int64, device=input_ids.device)
            lm_head_indices = lm_head_indices.to(torch.int64)
        else:
            seq_lens = ModelRunner.input_metadata["seq_lens"]
            seq_lens_list = ModelRunner.input_metadata["seq_lens_list"]
            block_tables = ModelRunner.input_metadata["block_tables"]
            lm_head_indices = ModelRunner.input_metadata["lm_head_indices"]

        if ModelRunner.input_metadata is None or is_prefill_or_no_mtp or is_mtp_0:
            q_lens = kwargs.get("q_lens", None)
            actual_seq_lengths_kv = None
            actual_seq_lengths_query = None
            # (NOTE): move attn related params to attnmetadata build
            use_actual_lengths_model = ["DeepseekV3ForCausalLM", "DeepseekV3MTP"]
            if self.model_cls.__name__ in use_actual_lengths_model:
                if is_prefill:
                    actual_seq_lengths_kv = seq_lens
                else:
                    actual_seq_lengths_kv = torch.tensor(q_lens, dtype=torch.int32).npu() \
                        if self.num_speculative_tokens \
                            else torch.tensor([1] * block_tables.shape[0], dtype=torch.int32).npu()
                actual_seq_lengths_query = torch.cumsum(actual_seq_lengths_kv, dim=0, dtype=torch.int32).npu()

            num_tokens_across_dp_cpu = get_num_tokens_across_dp_npu(input_ids.shape[0])
        else:
            q_lens = ModelRunner.input_metadata["q_lens"]
            actual_seq_lengths_kv = ModelRunner.input_metadata["actual_seq_lengths_kv"]
            actual_seq_lengths_query = ModelRunner.input_metadata["actual_seq_lengths_query"]
            num_tokens_across_dp_cpu = ModelRunner.input_metadata["num_tokens_across_dp_cpu"]
        
        # MTP
        hidden_states_mtp = kwargs.get("last_hidden_states", None)

        input_metadata = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "is_prefill": is_prefill,
            "block_tables": block_tables,
            "attn_mask": mask,
            "slot_mapping": slot_mapping,
            "seq_lens": seq_lens,
            "seq_lens_list": seq_lens_list,
            "q_lens": q_lens,
            "lm_head_indices": lm_head_indices,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "num_tokens_across_dp_cpu": num_tokens_across_dp_cpu,
            "last_hidden_states": hidden_states_mtp, # MTP
        }
        if self.enable_acl_graph and not is_prefill:
            return self._prepare_graph_inputs(input_metadata)
        return input_ids, position_ids, input_metadata
    
    def _prepare_graph_inputs(self, input_metadata):
        input_ids = input_metadata["input_ids"]
        position_ids = input_metadata["position_ids"]
        seq_lens = input_metadata["seq_lens"]
        seq_lens_list = input_metadata["seq_lens_list"]
        slot_mapping = input_metadata["slot_mapping"]
        block_tables = input_metadata["block_tables"]

        actual_seq_lengths_kv = input_metadata["actual_seq_lengths_kv"]
        actual_seq_lengths_query = input_metadata["actual_seq_lengths_query"]

        mc2_mask = ModelRunner.input_metadata["mc2_mask"] if ModelRunner.input_metadata is not None else None
        
        is_mtp_0 = self.is_draft_model and self.mtp_count % self.num_speculative_tokens == 0
        is_mtp_0_or_main = not self.is_draft_model or is_mtp_0

        num_actual_tokens = input_ids.shape[0]
        num_tokens = self.model.get_padded_graph_size(
            input_metadata['num_tokens_across_dp_cpu'].max().item())
            
        if num_tokens > self.graph_batch_sizes[-1]:
            input_metadata["num_tokens"] = num_tokens
            logger.info(f"Current batch size {num_tokens} is larger than {self.graph_batch_sizes[-1]},"
                        " using eager mode.")
            return input_ids, position_ids, input_metadata

        num_reqs = num_actual_tokens // (self.num_speculative_tokens + 1)
        self.input_ids[:num_actual_tokens].copy_(input_ids[:num_actual_tokens])
        self.position_ids[:num_actual_tokens].copy_(position_ids[:num_actual_tokens])
        self.slot_mapping[:num_actual_tokens].copy_(slot_mapping[:num_actual_tokens])

        input_ids = self.input_ids[:num_tokens]
        position_ids = self.position_ids[:num_tokens]
        slot_mapping = self.slot_mapping[:num_tokens]

        if self.num_speculative_tokens == 0 or is_mtp_0_or_main:
            self.seq_lens[:num_reqs].copy_(seq_lens[:num_reqs])
            max_len = seq_lens.max().item()
            max_seq_pages = (max_len + self.block_size - 1) // self.block_size
            seq_lens = self.seq_lens[:num_tokens]

            self.block_tables[:num_reqs, :block_tables.shape[-1]].copy_(block_tables)
            self.block_tables[:num_reqs, max_seq_pages:].fill_(0)
            self.block_tables[num_reqs:, :].fill_(0)
            block_tables = self.block_tables[:num_tokens, :]

            if actual_seq_lengths_kv is not None:
                actual_len, _ = \
                    get_speculative_reqs_padding_length(
                        num_tokens=num_tokens,
                        num_actual_tokens=self.num_speculative_tokens + 1
                    )
                reqs_padding_length = actual_len - actual_seq_lengths_kv.shape[0]
                seq_lens_pad = torch.tensor([0] * reqs_padding_length, dtype=torch.int32).npu()
                seq_lens = torch.cat([seq_lens, seq_lens_pad])
                self.seq_lens[:actual_len].copy_(seq_lens[:actual_len])
                seq_lens = self.seq_lens[:actual_len]

                block_tables = self.block_tables[:actual_len, :]

            seq_lens_list = seq_lens.cpu().tolist()

        tp_size = self.mapping.attn_tp.group_size
        tp_rank = self.mapping.attn_tp.rank
        num_padded_tokens = num_tokens + (tp_size - num_tokens % tp_size) % tp_size
        unit_size = num_padded_tokens // tp_size
    
        if ModelRunner.input_metadata is None or self.num_speculative_tokens == 0 or is_mtp_0:
            if actual_seq_lengths_kv is not None:
                actual_len, last_req_tokens = \
                    get_speculative_reqs_padding_length(
                        num_tokens=num_tokens,
                        num_actual_tokens=self.num_speculative_tokens + 1
                    )

                reqs_padding_length = actual_len - actual_seq_lengths_kv.shape[0]
                actual_seq_lengths_kv_pad = \
                    torch.tensor([self.num_speculative_tokens + 1] * reqs_padding_length, dtype=torch.int32).npu()
                actual_seq_lengths_kv = torch.cat([actual_seq_lengths_kv, actual_seq_lengths_kv_pad])
                if last_req_tokens > 0:
                    actual_seq_lengths_kv[-1] = last_req_tokens
                actual_seq_lengths_query = torch.cumsum(actual_seq_lengths_kv, dim=0, dtype=torch.int32).npu()

                self.actual_seq_lengths_kv[:actual_len].copy_(actual_seq_lengths_kv[:actual_len])
                self.actual_seq_lengths_query[:actual_len].copy_(actual_seq_lengths_query[:actual_len])
                actual_seq_lengths_kv = self.actual_seq_lengths_kv[:actual_len]
                actual_seq_lengths_query = self.actual_seq_lengths_query[:actual_len]
            
            all_mask = [1] * num_actual_tokens + [0] * (num_padded_tokens - num_actual_tokens)
            mc2_mask = torch.tensor(all_mask[unit_size * tp_rank:unit_size * (tp_rank + 1)], dtype=torch.bool).npu()
            self.mc2_mask[:unit_size].copy_(mc2_mask)
            mc2_mask = self.mc2_mask[:unit_size]
            
        elif not self.is_draft_model:
            if actual_seq_lengths_kv is not None:
                actual_len = len(actual_seq_lengths_kv)
                self.actual_seq_lengths_kv[:actual_len].copy_(actual_seq_lengths_kv[:actual_len])
                self.actual_seq_lengths_query[:actual_len].copy_(actual_seq_lengths_query[:actual_len])
                actual_seq_lengths_kv = self.actual_seq_lengths_kv[:actual_len]
                actual_seq_lengths_query = self.actual_seq_lengths_query[:actual_len]
            
            self.mc2_mask[:unit_size].copy_(mc2_mask)
            mc2_mask = self.mc2_mask[:unit_size]
        
        # MTP
        hidden_states_mtp = input_metadata["last_hidden_states"]
        if hidden_states_mtp is not None:
            self.hidden_states_mtp[:num_actual_tokens, :].copy_(
                hidden_states_mtp[:num_actual_tokens, :])
            hidden_states_mtp = self.hidden_states_mtp[:num_tokens, :]

        input_metadata["input_ids"] = input_ids
        input_metadata["position_ids"] = position_ids
        input_metadata["seq_lens"] = seq_lens
        input_metadata["seq_lens_list"] = seq_lens_list
        input_metadata["block_tables"] = block_tables
        input_metadata["slot_mapping"] = slot_mapping

        input_metadata["num_tokens"] = num_tokens
        input_metadata["num_actual_tokens"] = num_actual_tokens

        input_metadata["actual_seq_lengths_kv"] = actual_seq_lengths_kv
        input_metadata["actual_seq_lengths_query"] = actual_seq_lengths_query

        input_metadata["mc2_mask"] = mc2_mask
        # MTP
        input_metadata["last_hidden_states"] = hidden_states_mtp
        return input_ids, position_ids, input_metadata

    def _dummy_run(self, num_tokens):
        if num_tokens > self.graph_batch_sizes[-1]:
            raise ValueError("Dummy run failed for capture batch size is larger than max input_len.")
        input_ids, position_ids, input_metadata = self._generate_dummy_inputs(num_tokens)

        attn_metadata_dict = build_layerwise_attn_metadata(input_metadata, self.attn_layers)
        forward_context = create_forward_context(input_metadata=input_metadata,
                                                 capturing=True)
        forward_context.attn_metadata_dict = attn_metadata_dict
        forward_context.batch_descriptor = BatchDescriptor(num_tokens,
            get_parallel_info_manager().get(ParallelType.ATTN_DP).is_enabled())
        set_forward_context(forward_context)
        _ = self.model(input_ids, position_ids)
    
    def _generate_dummy_inputs(self, num_tokens):
        is_prefill = False
        input_ids = self.input_ids[:num_tokens]
        position_ids = self.position_ids[:num_tokens]
        mask = self.mask
        slot_mapping = self.slot_mapping[:num_tokens]
        lm_head_indices = self.lmhead_indices[:num_tokens]
        num_tokens_across_dp_cpu = get_num_tokens_across_dp_npu(input_ids.shape[0])
        
        reqs_padding_length, _ = get_speculative_reqs_padding_length(num_tokens=num_tokens,
                                                                    num_actual_tokens=self.num_speculative_tokens + 1)

        seq_lens = self.seq_lens[:reqs_padding_length]
        block_tables = self.block_tables[:reqs_padding_length, :]
        # DSV32
        actual_seq_lengths_kv = self.actual_seq_lengths_kv[:reqs_padding_length]
        actual_seq_lengths_query = self.actual_seq_lengths_query[:reqs_padding_length]
        # MTP
        hidden_states_mtp = self.hidden_states_mtp[:num_tokens, :]

        input_metadata = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "is_prefill": is_prefill,
            "attn_mask": mask,
            "slot_mapping": slot_mapping,
            "q_lens": seq_lens, 
            "seq_lens": seq_lens,
            "seq_lens_list": seq_lens.cpu().tolist(),
            "block_tables": block_tables,
            "lm_head_indices": lm_head_indices,
            "num_tokens": num_tokens,
            "num_actual_tokens": num_tokens,
            "num_tokens_across_dp_cpu": num_tokens_across_dp_cpu,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "last_hidden_states": hidden_states_mtp if self.is_draft_model else None
        }

        return input_ids, position_ids, input_metadata


def bind_kv_cache(kv_caches, attns):
    # the location of this function will be adjusted in the future
    for i, prefix in enumerate(attns):
        attn_layer = attns[prefix]
        attn_layer.key_cache = kv_caches[i][0]
        attn_layer.value_cache = kv_caches[i][1]
        if isinstance(attn_layer, SFA):
            attn_layer.index_cache = kv_caches[i][2]


def build_layerwise_attn_metadata(input_metadata, attns):
    common_attn_metadata = AttentionMetadata.from_dict(input_metadata)
    attn_metadata_dict = {}
    for _, prefix in enumerate(attns):
        attn_layer = attns[prefix]
        attn_backend = attn_layer.get_attn_backend()
        builder = attn_backend.get_builder_cls()()
        attn_metadata_dict[prefix] = builder.build(common_attn_metadata, input_metadata)
    return attn_metadata_dict


def get_num_tokens_across_dp_cpu(num_token_cur_dp):
    dp_para_info = get_parallel_info_manager().get(ParallelType.ATTN_DP)
    num_token_tensor = torch.tensor([
    num_token_cur_dp if i == dp_para_info.rank else 0 for i in range(dp_para_info.group_size)
    ], dtype=torch.int32, device="cpu")
    if dp_para_info.is_enabled():
        torch_dist.all_reduce(num_token_tensor, group=dp_para_info.cpu_process_group)
    return num_token_tensor


def get_num_tokens_across_dp_npu(num_token_cur_dp):
    dp_para_info = get_parallel_info_manager().get(ParallelType.ATTN_DP)
    num_token_tensor = torch.tensor([
    num_token_cur_dp if i == dp_para_info.rank else 0 for i in range(dp_para_info.group_size)
    ], dtype=torch.int32).npu()
    if dp_para_info.is_enabled():
        torch_dist.all_reduce(num_token_tensor, group=dp_para_info.process_group)
    return num_token_tensor.cpu()


def get_speculative_reqs_padding_length(num_tokens, num_actual_tokens):
    reqs_padding_length = num_tokens // num_actual_tokens
    last_req_tokens = num_tokens % num_actual_tokens
    if last_req_tokens > 0:
        reqs_padding_length += 1
    return reqs_padding_length, last_req_tokens


def maybe_gather_and_unpad_for_flashcomm(hidden_states):
    forward_context = get_forward_context()
    if not forward_context.batch_descriptor.is_flash_comm_enabled:
        return hidden_states

    from mindie_llm.runtime.layers.linear.linear_op import maybe_all_gather_and_maybe_unpad
    hidden_states = maybe_all_gather_and_maybe_unpad(
        hidden_states,
        get_parallel_info_manager().get(ParallelType.ATTN_TP)
    )
    return hidden_states


def maybe_allgather_cp(hidden_states):
    cp = get_parallel_info_manager().get(ParallelType.ATTN_CP)
    if not cp.is_enabled():
        return hidden_states

    group_size = cp.group_size
    hidden_states_out = torch.zeros_like(hidden_states).repeat(group_size, *(1,) * (hidden_states.dim() - 1))
    torch.distributed.all_gather_into_tensor(hidden_states_out, hidden_states,
                                    group=cp.process_group)
    return hidden_states_out


def maybe_pad_cross_dp(hidden_states):
    # maybe pad hidden_states cross dp for all_gather
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return hidden_states
    
    num_tokens_across_dp_cpu = forward_context.num_tokens_across_dp_cpu
    pad_size = max(num_tokens_across_dp_cpu) - hidden_states.shape[0]
        
    if pad_size > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))

    return hidden_states


def maybe_unpad_cross_dp(hidden_states):
    # unpad hidden_states after all_gather
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return hidden_states
    
    dp = get_parallel_info_manager().get(ParallelType.ATTN_DP)
    dp_world_size = dp.group_size
    num_tokens_across_dp_cpu = forward_context.num_tokens_across_dp_cpu
    result = torch.empty(
        (num_tokens_across_dp_cpu.sum(), *hidden_states.shape[1:]),
        device=hidden_states.device,
        dtype=hidden_states.dtype)
        
    hidden_states = hidden_states.view(dp.group_size, -1, *hidden_states.shape[1:])
    offset = 0
    for idx in range(dp_world_size):
        num_tokens_dp = num_tokens_across_dp_cpu[idx]
        result[offset:offset + num_tokens_dp] = hidden_states[idx, :num_tokens_dp]
        offset += num_tokens_dp
    hidden_states = result
    return hidden_states


def maybe_pad_and_gather_cross_dp_and_unpad(hidden_states):
    # NOTE: Temporary support for DP for PDmix, will be removed after server support dp in and out.  
    dp = get_parallel_info_manager().get(ParallelType.ATTN_DP)
    if not dp.is_enabled():
        return hidden_states
    
    hidden_states = maybe_pad_cross_dp(hidden_states)
    
    # create output tensor
    gather_list = [torch.empty_like(hidden_states) for _ in range(dp.group_size)]

    # do all_gather
    torch.distributed.all_gather(gather_list, hidden_states, group=dp.process_group)
    output_parallel_ = torch.cat(gather_list, dim=0)
    hidden_states = maybe_unpad_cross_dp(output_parallel_)

    return hidden_states