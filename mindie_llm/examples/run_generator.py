# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import argparse
import json
import os
import time
from typing import Optional

import numpy as np

from mindie_llm.examples.scheduler import Scheduler
from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.text_generator.generator import Generator
from mindie_llm.text_generator.utils.generation_metadata import GenerationParams
from mindie_llm.text_generator.utils.input_metadata import SAMPLING_DTYPE
from mindie_llm.text_generator.utils.request import Request
from mindie_llm.utils import file_utils
from mindie_llm.utils.env import ENV
from mindie_llm.utils.log.logging import logger, print_log
from mindie_llm.utils.tensor import backend
from mindie_llm.utils.validation import (UPPER_SAFE_BLOCK_SIZE, LOWER_SAFE_BLOCK_SIZE, UPPER_SAFE_NPU_MEM,
                                         LOWER_SAFE_NPU_MEM, UPPER_SAFE_BATCH_SIZE, LOWER_SAFE_BATCH_SIZE,
                                         UPPER_SAFE_SEQUENCE_LENGTH, LOWER_SAFE_SEQUENCE_LENGTH, ValidationError,
                                         OutOfBoundsError, validate_list, validate_string)

DEFAULT_INPUT_TEXTS = ["What's deep learning?", "How do you feel?", "What's deep learning?", "How do you feel?"]


class GeneratorRunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.async_inference = kwargs.get('async_inference', False)
        self.model_path = kwargs.get('model_path')
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.input_text = kwargs.get('input_text')
        self.max_position_embeddings = kwargs.get('max_position_embeddings')
        self.max_input_length = kwargs.get('max_input_length')
        self.max_prefill_tokens = kwargs.get('max_prefill_tokens')
        self.max_output_length = kwargs.get('max_output_length')
        self.max_seq_len = self.max_position_embeddings if self.max_position_embeddings else \
            self.max_input_length + self.max_output_length
        self.max_batch_size = kwargs.get('max_batch_size')
        self.max_prefill_batch_size = kwargs.get('max_prefill_batch_size', 50)
        self.npu_mem = kwargs.get('npu_mem', -1)
        self.chat_template = kwargs.get('chat_template')

        self.best_of = kwargs.get('best_of', 1)
        self.eos_token_id = kwargs.get('eos_token_id')
        self.ignore_eos = kwargs.get('ignore_eos', False)
        self.include_stop_str_in_output = kwargs.get('include_stop_str_in_output', False)
        self.length_penalty = kwargs.get('length_penalty', 1.0)
        self.logprobs = kwargs.get('logprobs')
        self.max_new_tokens = kwargs.get('max_new_tokens')
        self.n = kwargs.get('param_n', 1)
        self.request_ignore_eos = kwargs.get('request_ignore_eos', False)
        self.seed = kwargs.get('seed', 0)
        self.skip_special_tokens = kwargs.get('skip_special_tokens', True)
        self.stop_strings = kwargs.get('stop_strings')
        self.stop_token_ids = kwargs.get('stop_token_ids')
        self.use_beam_search = kwargs.get('use_beam_search', False)

        self.block_size = kwargs.get('block_size')

        self.split_chunk_tokens = kwargs.get('split_chunk_tokens')

        npu_id = kwargs.get('npu_id')
        self.npu_id = npu_id if npu_id is not None else self.local_rank

        backend_type = kwargs.get('backend_type', 'atb')
        self.backend_type = BackendType.MS if backend_type and backend_type.lower() == BackendType.MS \
            else BackendType.ATB
        self.load_tokenizer = kwargs.get('load_tokenizer')
        self.plugin_params = kwargs.get('plugin_params')

        self.model_role = kwargs.get('model_role', 'standard')
        self.local_model_instance_id = kwargs.get('local_model_instance_id', 0)
        self.local_device_ip = kwargs.get('local_device_ip', None)
        self.remote_model_instance_ids = kwargs.get('remote_model_instance_ids', None)
        self.remote_device_ips = kwargs.get('remote_device_ips', '')
        self.speculation_gamma = kwargs.get('speculation_gamma', None)
        self.num_speculative_tokens = kwargs.get('num_speculative_tokens', 0)
        self.local_physical_device_id = kwargs.get('local_physical_device_id', 0)
        self.local_logic_device_id = kwargs.get('local_logic_device_id', 0)
        self.local_host_ip = kwargs.get('local_host_ip', None)
        self.local_super_device_id = kwargs.get('local_super_device_id', None)
        self.local_super_pod_id = kwargs.get('local_super_pod_id', None)
        self.kv_trans_timeout = kwargs.get('kv_trans_timeout', 1)
        model_config = {
            'async_inference': self.async_inference,
            'backend_type': self.backend_type,
            'block_size': 128,
            'cpu_mem': 20,
            'ignore_eos': self.ignore_eos,
            'load_tokenizer': self.load_tokenizer,
            'local_rank': self.local_rank,
            'max_input_len': self.max_input_length,
            'max_iter_times': self.max_output_length,
            'max_batch_size': self.max_batch_size,
            'max_prefill_batch_size': self.max_prefill_batch_size,
            'max_prefill_tokens': self.max_prefill_tokens,
            'max_seq_len': self.max_seq_len,
            'model_id': self.model_path,
            'trust_remote_code': self.trust_remote_code,
            'npu_device_id': self.npu_id,
            'npu_mem': self.npu_mem,
            'num_threads': 8,
            'plugin_params': self.plugin_params,
            'rank': self.rank,
            'world_size': self.world_size,
            'model_role': self.model_role,
            'role': self.model_role,
            'local_instance_id': self.local_model_instance_id,
            'local_device_ip': self.local_device_ip,
            'remote_model_instance_ids': self.remote_model_instance_ids,
            'remote_device_ips': self.remote_device_ips,
            'local_super_device_id': self.local_super_device_id,
            'local_super_pod_id': self.local_super_pod_id,
            'local_physical_device_id': self.local_physical_device_id,
            'local_logic_device_id': self.local_logic_device_id,
            'local_host_ip': self.local_host_ip,
            'speculation_gamma': self.speculation_gamma,
            'dp': kwargs.get('dp', -1),
            'tp': kwargs.get('tp', -1),
            'moe_ep': kwargs.get('moe_ep', -1),
            'moe_tp': kwargs.get('moe_tp', -1),
            'num_speculative_tokens': self.num_speculative_tokens,
            'kv_trans_timeout': self.kv_trans_timeout
        }
        if self.eos_token_id is not None:
            model_config['eos_token_id'] = self.eos_token_id

        self.generator = Generator(model_config)
        if self.load_tokenizer:
            self.tokenizer = self.generator.model_wrapper.tokenizer
            if self.chat_template:
                self.tokenizer.chat_template = self._load_chat_template(self.chat_template)
        self.is_mix_model = self.generator.is_mix_model
        self.max_memory = 0
        if self.backend_type == BackendType.ATB:
            from atb_llm.utils.cpu_binding import NpuHbmInfo

            self.soc_info = self.generator.model_wrapper.soc_info
            self.device = self.generator.model_wrapper.device

            self.max_memory = NpuHbmInfo.get_hbm_capacity(self.local_rank, self.world_size, self.soc_info.need_nz)
            self.init_memory = int(
                self.max_memory * NpuHbmInfo.get_hbm_usage(self.local_rank, self.world_size, self.soc_info.need_nz))
            print_log(self.rank, logger.info, f'hbm_capacity(GB): {self.max_memory / (1024 ** 3)}, '
                                              f'init_memory(GB): {self.init_memory / (1024 ** 3)}')
        else:  # mindformer
            pass

        self.warm_up_memory = -1

    def __repr__(self):
        return (
            "GeneratorRunner("
            + f"max_position_embeddings={self.max_position_embeddings}, "
            + f"max_input_length={self.max_input_length}, "
            + f"max_output_length={self.max_output_length}, "
            + f"max_prefill_tokens={self.max_prefill_tokens}, "
            + f"max_batch_size={self.max_batch_size}, "
            + f"block_size={self.block_size}, "
            + f"model_config={self.generator.model_wrapper.model_info}, "
            + f"max_memory={self.max_memory}, "
        )

    @staticmethod
    def _load_chat_template(chat_template):
        if os.path.exists(chat_template):
            with file_utils.safe_open(chat_template, 'r') as f:
                chat_template_content = f.read()
        else:
            chat_template_content = chat_template
        return chat_template_content

    def infer(self, inputs, batch_size, sampling_params, is_chat_model=False):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        batch_input_ids, adapter_ids = self.build_model_inputs(inputs, is_chat_model)

        max_new_tokens = self.max_new_tokens if self.max_new_tokens else self.max_output_length
        generation_params = GenerationParams(
            best_of=self.best_of,
            ignore_eos=self.request_ignore_eos,
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            logprobs=self.logprobs,
            max_new_tokens=max_new_tokens,
            n=self.n,
            seed=self.seed,
            skip_special_tokens=self.skip_special_tokens,
            stop_strings=self.stop_strings,
            stop_token_ids=self.stop_token_ids,
            use_beam_search=self.use_beam_search
        )

        scheduler = Scheduler(self.max_batch_size, self.max_prefill_tokens, self.generator,
                              self.load_tokenizer, self.is_mix_model, self.split_chunk_tokens,
                              speculation_gamma=self.speculation_gamma)

        req_list = []
        if len(batch_input_ids) == 1:
            generation_params.adapter_id = adapter_ids[0]
            req_list = [Request.request_from_token(batch_input_ids[0], sampling_params, generation_params,
                        req_id=i, seq_id=scheduler.get_sequence_id()) for i in range(batch_size)]
        else:
            for idx, input_ids in enumerate(batch_input_ids):
                generation_params.adapter_id = adapter_ids[idx]
                req_list.append(Request.request_from_token(
                    input_ids, sampling_params, generation_params, req_id=idx, seq_id=scheduler.get_sequence_id()))

        print_log(self.rank, logger.debug, f'req_list[0].input_ids: {req_list[0].input_ids}')

        e2e_start = time.time()

        generate_text_list, token_num_list, cumulative_logprobs, logprobs_list, top_logprobs_list = (
            scheduler.generate(req_list))

        e2e_end = time.time()
        e2e_time = e2e_end - e2e_start
        inference_result = (generate_text_list, token_num_list, cumulative_logprobs,
                            logprobs_list, top_logprobs_list, e2e_time)
        return inference_result

    def build_model_inputs(self, inputs, is_chat_model):
        input_texts, input_ids, input_conversations, input_adapters = [], [], [], []
        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], str):
                input_texts = inputs
            elif isinstance(inputs[0], backend.Tensor):
                input_ids = inputs
                print_log(self.rank, logger.info, "input is mindspore tensor or torch tensor")
            elif isinstance(inputs[0], list) and inputs[0]:
                if isinstance(inputs[0][0], int):
                    input_ids = inputs
                elif isinstance(inputs[0][0], dict):
                    input_conversations = inputs
            elif isinstance(inputs[0], dict):
                input_texts = [item.get("prompt") for item in inputs]
                input_adapters = [item.get("adapter") for item in inputs]
        if not (input_texts or input_ids or input_conversations):
            raise ValueError(f"The inputs of `PARunner.infer` must be as List[str], List[torch.Tensor], List[List[int]]"
                             f" or List[List[Dict]]. Now the inputs ({inputs}) is not acceptable or is empty.")
        if is_chat_model:
            if input_conversations:
                input_ids = self.generator.build_inputs(input_conversations)
            elif input_texts:
                input_conversations = [[{"role": "user", "content": t}] for t in input_texts]
                input_ids = self.generator.build_inputs(input_conversations)
            else:
                print_log(self.rank, logger.warning, "Neither conversations nor input_texts exist, "
                                                     "'chat' parameter is not effective.")
        elif input_texts:
            input_ids = [self.tokenizer([text], return_tensors="pt")["input_ids"].flatten() for text in input_texts]
        if not input_adapters:
            input_adapters = [None] * len(input_ids)
        return input_ids, input_adapters


def validate_sequence_length(param_key, param_value):
    if param_value > UPPER_SAFE_SEQUENCE_LENGTH or param_value < LOWER_SAFE_SEQUENCE_LENGTH:
        raise OutOfBoundsError(param_key, 'SAFE_SEQUENCE_LENGTH',
                               f'[{LOWER_SAFE_SEQUENCE_LENGTH}, {UPPER_SAFE_SEQUENCE_LENGTH}]')


def validate_args(args_):
    if args_.chat_template is not None:
        validate_string('chat_template', args_.chat_template)
    if isinstance(args_.eos_token_id, list):
        validate_list('eos_token_ids', args_.eos_token_ids)
    if args_.block_size > UPPER_SAFE_BLOCK_SIZE or args_.block_size < LOWER_SAFE_BLOCK_SIZE:
        raise OutOfBoundsError('block_size', 'SAFE_BLOCK_SIZE',
                               f'[{LOWER_SAFE_BLOCK_SIZE}, {UPPER_SAFE_BLOCK_SIZE}]')
    if args_.max_batch_size > UPPER_SAFE_BATCH_SIZE or args_.max_batch_size < LOWER_SAFE_BATCH_SIZE:
        raise OutOfBoundsError('max_batch_size', 'SAFE_BATCH_SIZE',
                               f'[{LOWER_SAFE_BATCH_SIZE}, {UPPER_SAFE_BATCH_SIZE}]')
    validate_sequence_length('max_input_length', args_.max_input_length)
    validate_sequence_length('max_output_length', args_.max_output_length)
    validate_sequence_length('max_position_embeddings', args_.max_position_embeddings)
    validate_sequence_length('max_prefill_tokens', args_.max_prefill_tokens)
    validate_sequence_length('split_chunk_tokens', args_.split_chunk_tokens)
    if args_.npu_mem != -1 and (args_.npu_mem > UPPER_SAFE_NPU_MEM or args_.npu_mem < LOWER_SAFE_NPU_MEM):
        raise OutOfBoundsError('npu_mem', 'SAFE_NPU_MEM',
                               f'-1|[{LOWER_SAFE_NPU_MEM}, {UPPER_SAFE_NPU_MEM}]')
    if args_.best_of > 1 and not args_.do_post_process:
        raise ValidationError('best_of', 'If you want to enable `best_of` function, please set `--do_post_process`.')
    if args_.max_new_tokens is not None:
        validate_sequence_length('max_new_tokens', args_.max_new_tokens)
    if args_.param_n > 1 and not args_.do_post_process:
        raise ValidationError('n', 'If you want to enable `n` function, please set `--do_post_process`.')
    if args_.param_n > args_.best_of:
        raise ValidationError('n', 'The parameter `n` must <= `best_of`.')
    if args_.stop_strings is not None:
        validate_list('stop_strings', args_.stop_strings)
    if args_.stop_token_ids is not None:
        validate_list('stop_token_ids', args_.stop_token_ids)
    if args_.use_beam_search and not args_.do_post_process:
        raise ValidationError('use_beam_search',
                              'If you want to enable `use_beam_search` function, please set `--do_post_process`.')


def parse_bool(bool_str):
    bool_str = bool_str.lower()
    if bool_str == 'true':
        return True
    elif bool_str == 'false':
        return False
    else:
        raise ValueError('Failed to parse a bool variable, please check if there is a bool arg given a special string '
                         'which is neither True nor False.')


def parse_json(json_str):
    return json.loads(json_str)


def parse_ids(list_str):
    return [int(item) for item in list_str.split(',')]


def parse_arguments():
    store_true = 'store_true'

    parser = argparse.ArgumentParser()

    # 权重路径与输入
    parser.add_argument('--model_path',
                        help='model and tokenizer path',
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b',
                        )
    parser.add_argument(
        '--input_texts',
        type=str,
        nargs='+',
        default=DEFAULT_INPUT_TEXTS)
    parser.add_argument(
        '--input_ids',
        type=parse_ids,
        nargs='+',
        default=None)
    parser.add_argument(
        '--input_file',
        type=str,
        help='CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--trust_remote_code', action=store_true, default=False)

    # 全局策略配置
    parser.add_argument('--backend_type', type=str, choices=['atb', 'ms'], default='atb')
    parser.add_argument('--load_tokenizer', type=parse_bool, help='whether to load the tokenizer', default=True)
    parser.add_argument('--plugin_params', type=str, help='json input', default='{'
                                                                                '\"plugin_type\": \"\", '
                                                                                '\"decoding_length\": 8, '
                                                                                '\"level\": 4, '
                                                                                '\"window\": 5, '
                                                                                '\"guess_set_size\": 5'
                                                                                '}')
    parser.add_argument(
        '--input_dict',
        help="Lora input, accepted format: "
             "'[{\"prompt\": \"prompt in text\", \"adapter\": \"adapter id defined in lora_adapter param\"}]'",
        type=parse_json,
        default=None)
    
    # 全局并行策略配置
    parser.add_argument('--dp', type=int, default=-1)
    parser.add_argument('--tp', type=int, default=-1)
    parser.add_argument('--moe_ep', type=int, default=-1)
    parser.add_argument('--moe_tp', type=int, default=-1)

    # 全局前后处理配置
    parser.add_argument('--chat_template', type=str, default=None)
    parser.add_argument('--do_post_process', action=store_true)
    parser.add_argument('--eos_token_id', type=str,
                        help='It should be a json string whose content is a int or list. If it is a list, '
                             'the element can be a int or List[int].',
                        default=None)
    parser.add_argument('--ignore_eos', action=store_true)
    parser.add_argument('--is_chat_model', action=store_true)

    # 全局模型参数
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=200)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--max_position_embeddings', type=int, default=2048)
    parser.add_argument('--max_prefill_tokens', type=int, default=4096)
    parser.add_argument('--npu_mem', type=int, default=-1)
    parser.add_argument('--speculation_gamma', type=int, default=0)
    parser.add_argument('--split_chunk_tokens', type=int, default=1)
    parser.add_argument('--num_speculative_tokens', type=int, default=0)

    # 请求后处理参数
    parser.add_argument('--repetition_penalty', type=float, default=np.nan)
    parser.add_argument('--frequency_penalty', type=float, default=np.nan)
    parser.add_argument('--presence_penalty', type=float, default=np.nan)
    parser.add_argument('--temperature', type=float, default=np.nan)
    parser.add_argument('--top_k', type=int, default=np.nan)
    parser.add_argument('--top_p', type=float, default=np.nan)
    parser.add_argument('--do_sample', type=parse_bool, default=np.nan)
    parser.add_argument('--top_logprobs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # 请求生成策略参数
    parser.add_argument('--best_of', type=int, default=1)
    parser.add_argument('--include_stop_str_in_output', action=store_true)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--logprobs', action=store_true)
    parser.add_argument('--max_new_tokens', type=Optional[int], default=None)
    parser.add_argument('--param_n', type=int, default=1)
    parser.add_argument('--request_ignore_eos', action=store_true)
    parser.add_argument('--skip_special_tokens', type=parse_bool, default=True)
    parser.add_argument('--stop_strings', type=parse_json, help='Optional[List[str]]', default=None)
    parser.add_argument('--stop_token_ids', type=parse_json, help='Optional[List[Union[int, List[int]]]]',
                        default=None)
    parser.add_argument('--use_beam_search', action=store_true)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    validate_args(args)

    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(args)
    }

    if not args.load_tokenizer and not args.input_ids:
        raise RuntimeError('The input_ids must be used when you do not load tokenizer!')

    if args.input_dict:
        infer_inputs = args.input_dict
    elif args.input_ids:
        infer_inputs = args.input_ids
    else:
        infer_inputs = args.input_texts
    if args.is_chat_model and args.input_file:
        conversations = []
        with file_utils.safe_open(args.input_file, 'r', encoding='utf-8') as file:
            for line in file_utils.safe_readlines(file):
                data_line = json.loads(line)
                conversations.append(data_line)
        infer_inputs = conversations

    generator_runner = GeneratorRunner(**input_dict)
    print_log(rank, logger.info, f'generator_runner: {generator_runner}')

    # repetition, frequency, presence, temperature, top_k, top_p, do_sample, top_logprobs
    sampling_params_ins = np.array(
        [(args.repetition_penalty, args.frequency_penalty, args.presence_penalty, args.temperature,
          args.top_k, args.top_p, args.do_sample, args.top_logprobs)], SAMPLING_DTYPE) if args.do_post_process else None
    generate_res, token_nums, cumulative_logprobs_out, logprobs_out, top_logprobs_out, _ = (
        generator_runner.infer(infer_inputs, args.max_batch_size, sampling_params_ins, args.is_chat_model))

    are_inputs_safe = infer_inputs == DEFAULT_INPUT_TEXTS
    for i, generate_res in enumerate(generate_res):
        index = int(i / args.best_of)
        choice_id = i % args.best_of
        if are_inputs_safe:
            print_log(rank, logger.info, f'Question[{index}]: {infer_inputs[index]}')
        print_log(rank, logger.info, f'Answer[{index}][{choice_id}]: {generate_res}')
        print_log(rank, logger.info, f'Generate[{index}][{choice_id}] token num: {token_nums[i]}')
        if args.best_of > 1:
            print_log(rank, logger.info, f'cumulative_logprobs[{index}][{choice_id}]: {cumulative_logprobs_out[i]}')
        if args.logprobs:
            print_log(rank, logger.info, f'Logprobs[{index}][{choice_id}]: {logprobs_out[i]}')
            print_log(rank, logger.info, f'TopLogprobs[{index}][{choice_id}]: {top_logprobs_out[i]}')
