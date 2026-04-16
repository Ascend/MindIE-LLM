# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import numpy as np

from ....utils.log.logging import logger
from ....utils.log.error_code import ErrorCode
from .la_statistics import RequestStatsList
from ....utils.tensor import backend, tensor_backend


class CacheEngine:
    def __init__(self, level, window, guess_set_size):
        self.level = level
        self.guess_set_size = guess_set_size
        self.window = window
        self.requests_idx_mapping = {}
        self.used_idx = []
        self.token_map = []
        self.past_tokens = []
        self.input_tail_tokens = []
        self.last_gen_tokens = []
        self.need_cal_kv = []
        self.input_len_list = []

    @staticmethod
    def set_token(prompt_tokens, seed):
        seed = seed % np.iinfo(np.int32).max
        np.random.seed(seed)
        return np.random.choice(prompt_tokens)

    def raise_error_req_id(self, req_id):
        if req_id not in self.requests_idx_mapping:
            message = f"req_id {req_id} not allocated! This request not prefilled before decoding!"
            logger.error(message, ErrorCode.TEXT_GENERATOR_REQ_ID_INVALID)
            raise RuntimeError(message)

    def get_guess_token(self, req_id, lst_token):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        if self.past_tokens[map_id][self.level - 2] is None:
            return None
        if lst_token not in self.token_map[map_id]:
            return None
        return self.token_map[map_id][lst_token]

    def add_guess_tokens(self, map_id, lst_token, n_grams):
        tup = tuple(n_grams)
        if lst_token not in self.token_map[map_id]:
            self.token_map[map_id][lst_token] = []
        if tup in self.token_map[map_id][lst_token]:
            self.token_map[map_id][lst_token].remove(tup)
            self.token_map[map_id][lst_token].append(tup)
        elif len(self.token_map[map_id][lst_token]) < self.guess_set_size:
            self.token_map[map_id][lst_token].append(tup)
        else:
            self.token_map[map_id][lst_token] = self.token_map[map_id][lst_token][1:] + [tup]
        return

    def allocate_map_id(self, req_id):
        if req_id not in self.requests_idx_mapping:
            available_idx = np.flatnonzero(np.array(self.used_idx) == 0)
            if available_idx.shape[0] == 0:
                self.used_idx.append(1)
                self.token_map.append({})
                self.past_tokens.append(None)
                self.input_tail_tokens.append([])
                self.last_gen_tokens.append([])
                self.need_cal_kv.append(False)
                self.input_len_list.append(0)
                append_idx = len(self.used_idx) - 1
                self.requests_idx_mapping[req_id] = append_idx
            else:
                unused_idx = available_idx[0]
                self.used_idx[unused_idx] = 1
                self.requests_idx_mapping[req_id] = unused_idx
        map_id = self.requests_idx_mapping[req_id]
        return map_id

    def del_req_cache(self, req_id):
        if req_id not in self.requests_idx_mapping:
            return
        map_id = self.requests_idx_mapping[req_id]
        self.used_idx[map_id] = 0
        self.token_map[map_id] = {}
        self.past_tokens[map_id] = None
        self.input_tail_tokens[map_id] = []
        self.last_gen_tokens[map_id] = []
        self.need_cal_kv[map_id] = False
        self.input_len_list[map_id] = 0
        del self.requests_idx_mapping[req_id]
        return

    def initiate_past_tokens_with_prompt(self, map_id, prompt_tokens, seed):
        self.past_tokens[map_id] = [
            [self.set_token(prompt_tokens, seed) for _ in range(self.level + self.window - 3)]
        ] + [None for _ in range(self.level - 2)]
        return

    def fill_token_map_with_tails(self, map_id):
        tail_len = len(self.input_tail_tokens[map_id])
        start = 0
        while tail_len >= self.level:
            self.add_guess_tokens(
                map_id,
                self.input_tail_tokens[map_id][start],
                self.input_tail_tokens[map_id][start + 1 : start + self.level],
            )
            start += 1
            tail_len -= 1

        self.input_tail_tokens[map_id] = self.input_tail_tokens[map_id][-tail_len:]
        return

    def fill_input_tail_tokens(self, map_id, tail_tokens, num_new_tokens):
        tail_tokens = tail_tokens[:num_new_tokens]
        tail_tokens = tail_tokens.tolist()
        self.last_gen_tokens[map_id] = tail_tokens
        self.input_tail_tokens[map_id].extend(tail_tokens)
        self.fill_token_map_with_tails(map_id)
        return

    def fill_pool_with_prompt(self, req_id, seed):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        prompt_tokens = self.input_tail_tokens[map_id]
        self.initiate_past_tokens_with_prompt(map_id, prompt_tokens, seed)
        self.fill_token_map_with_tails(map_id)
        return

    def save_tail_tokens(self, req_id, prompt_tokens):
        map_id = self.allocate_map_id(req_id)
        self.input_tail_tokens[map_id].extend(prompt_tokens)
        return

    def get_past_tokens(self, req_id):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        return self.past_tokens[map_id]

    def get_past_tokens_len(self, req_id):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        if not self.past_tokens[map_id]:
            return 0
        past_tokens_len = 0
        for i in range(len(self.past_tokens[map_id])):
            if self.past_tokens[map_id][i] is None:
                break
            past_tokens_len += len(self.past_tokens[map_id][i])
        return past_tokens_len

    def update_token_map(self, map_id, past_tokens, lst_token, new_results):
        for column_idx in range(self.window):
            if column_idx == 0:
                head = lst_token
            else:
                head = past_tokens[0][column_idx - 1]
            column = []
            for line in range(1, self.level - 1):
                column.append(past_tokens[line][column_idx])
            column.append(new_results[column_idx])
            self.add_guess_tokens(map_id, head, column)
        return

    def update_past_tokens(self, map_id, lst_token, new_results):
        past_tokens = self.past_tokens[map_id]
        if past_tokens[self.level - 2] is not None:
            self.update_token_map(map_id, past_tokens, lst_token, new_results)
            past_tokens[0] = past_tokens[1][1:]
            for line in range(1, self.level - 2):
                past_tokens[line] = past_tokens[line + 1][:]

            past_tokens[self.level - 2] = new_results
            return

        if past_tokens[1] is None:
            past_tokens[0] = past_tokens[0][1:]
            past_tokens[1] = new_results
            return

        line = 0
        while past_tokens[line] is not None:
            past_tokens[line] = past_tokens[line][1:]
            line += 1
        past_tokens[line] = new_results[1:]
        return

    def update_new_results(self, req_id, lst_token, new_results):
        map_id = self.requests_idx_mapping[req_id]
        self.update_past_tokens(map_id, lst_token, new_results)
        return

    def append_new_generated_pool(self, req_id, new_tokens, num_new_tokens):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        self.fill_input_tail_tokens(map_id, new_tokens, num_new_tokens)

    def get_gen_tokens(self, req_id):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        if self.need_cal_kv[map_id]:
            last_gen = self.last_gen_tokens[map_id]
        else:
            last_gen = self.last_gen_tokens[map_id][-1:]
        self.need_cal_kv[map_id] = False
        return last_gen

    def set_need_cal_kv(self, req_id):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        self.need_cal_kv[map_id] = True

    def set_input_len(self, req_id, input_len):
        self.raise_error_req_id(req_id)
        map_id = self.requests_idx_mapping[req_id]
        self.input_len_list[map_id] = input_len

    def get_input_lens(self, req_ids):
        input_len_list = []
        for req_id in req_ids:
            self.raise_error_req_id(req_id)
            map_id = self.requests_idx_mapping[req_id]
            input_len_list.append(self.input_len_list[map_id])
        return input_len_list


DEFAULT_MASK_START_SIZE = 200


def make_triangle_mask(mask, device):
    mask_cond = tensor_backend.tensor(np.arange(tensor_backend.shape(mask, -1)), device=device)
    return tensor_backend.masked_fill(mask, mask_cond < (mask_cond + 1).view(tensor_backend.shape(mask, -1), 1), 0)


def make_triangle_mask_numpy(mask):
    mask_cond = np.arange(mask.shape[-1])
    mask_cond = mask_cond < (mask_cond + 1).reshape(mask.shape[-1], 1)
    mask = np.where(mask_cond, 0, mask)
    return mask


def make_diagonal_mask(mask):
    return tensor_backend.fill_diagonal(mask, fill_value=0.0)


def make_single_mask(level, window, line, guess_set_size, device):
    dtype = backend.half
    cur_window = level + window - 2 - line
    guess_len = level - 1
    if line > 1:
        cur_window += 1
    tgt_len = cur_window * line + guess_len * guess_set_size
    min_num = float(np.finfo(np.half).min)
    mask = tensor_backend.full((tgt_len, tgt_len), min_num, dtype=dtype, device=device)
    for guess_index in range(guess_set_size):
        start_guess = guess_len * guess_index + 1
        mask[start_guess : start_guess + guess_len, start_guess : start_guess + guess_len] = make_triangle_mask(
            mask[
                start_guess : start_guess + guess_len,
                start_guess : start_guess + guess_len,
            ],
            device,
        )

    guess_total_len = guess_len * guess_set_size
    for line_index in range(line):
        start = guess_total_len + 1
        stop = guess_total_len + cur_window
        start_past = guess_total_len + cur_window * line_index + 1
        mask[start_past : start_past + cur_window - 1, start:stop] = make_triangle_mask(
            mask[start_past : start_past + cur_window - 1, start:stop], device
        )

    for line_index in range(1, line):
        start = guess_total_len + cur_window
        stop = guess_total_len + cur_window * (1 + line_index)
        mask[tgt_len - cur_window * line_index :, start:stop] = make_diagonal_mask(
            mask[tgt_len - cur_window * line_index :, start:stop]
        )

    mask[:, 0] *= 0
    return mask


def make_default_mask(level, window, guess_set_size, device):
    default_mask = []

    guess_num = 0
    for line in range(level - 1):
        if line == level - 2:
            guess_num = guess_set_size
        single_mask = make_single_mask(level, window, line + 1, guess_num, device)
        single_mask_numpy = tensor_backend.numpy(tensor_backend.cpu(single_mask[:, :]))
        default_mask.append(single_mask_numpy)

    dtype = backend.half
    min_num = float(np.finfo(np.float16).min)
    mask = tensor_backend.full(
        (DEFAULT_MASK_START_SIZE, DEFAULT_MASK_START_SIZE),
        min_num,
        dtype=dtype,
        device=device,
    )
    mask_numpy = tensor_backend.numpy(tensor_backend.cpu(mask[:, :]))
    mask_numpy = make_triangle_mask_numpy(mask_numpy)
    default_mask.append(mask_numpy)

    return default_mask


MAX_LEVEL_VALUE = 16
MAX_WINDOW_VALUE = 16
MAX_GUESS_SET_SIZE = 16
MIN_LEVEL_VALUE = 3
MIN_WINDOW_VALUE = 1
MIN_GUESS_SET_SIZE = 1


class DecodingPolicy:
    def __init__(self, kwargs, infer_context, model_wrapper, log_enable, block_size, eos_token_id):
        self.level = kwargs.get("level", 4)
        self.window = kwargs.get("window", 5)
        self.guess_set_size = kwargs.get("guess_set_size", 5)
        if self.level > MAX_LEVEL_VALUE:
            logger.warning(f"The value of level is larger than max value {MAX_LEVEL_VALUE}, run with max value!")
            self.level = MAX_LEVEL_VALUE

        if self.window > MAX_WINDOW_VALUE:
            logger.warning(f"The value of window is larger than max value {MAX_WINDOW_VALUE}, run with max value!")
            self.window = MAX_WINDOW_VALUE

        if self.guess_set_size > MAX_GUESS_SET_SIZE:
            logger.warning(
                f"The value of guess_set_size is larger than max value {MAX_GUESS_SET_SIZE}, run with max value!"
            )
            self.guess_set_size = MAX_GUESS_SET_SIZE

        if self.level < MIN_LEVEL_VALUE:
            message = (
                f"The value of level cannot be less than 3, but now is {self.level},"
                " please check the value of level in plugin_params!"
            )
            logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_PARAM_VALUE_ERR)
            raise ValueError(message)

        if self.window < MIN_WINDOW_VALUE:
            message = (
                f"The value of window cannot be less than 1, but now is {self.window},"
                " please check the value of window in plugin_params!"
            )
            logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_PARAM_VALUE_ERR)
            raise ValueError(message)

        if self.guess_set_size < MIN_GUESS_SET_SIZE:
            message = (
                f"The value of guess_set_size cannot be less than 1, but now is {self.guess_set_size},"
                " please check the value of guess_set_size in plugin_params!"
            )
            logger.error(message, ErrorCode.TEXT_GENERATOR_PLUGIN_PARAM_VALUE_ERR)
            raise ValueError(message)

        self.la_cache = CacheEngine(self.level, self.window, self.guess_set_size)

        self.store_guess_tokens = []
        self.store_sampling_metadata = None
        self.last_tokens = None
        self.block_size = block_size
        self.infer_context = infer_context
        self.model_wrapper = model_wrapper
        self.dtype = self.model_wrapper.model_runner.dtype
        self.device = self.model_wrapper.device

        self.default_mask = make_default_mask(self.level, self.window, self.guess_set_size, self.device)
        self.eos_token_id = eos_token_id

        self.request_stats_list = None
        if log_enable:
            self.request_stats_list = RequestStatsList(self.level, self.window, self.guess_set_size)

        self.cu_seq_len = None
        self.mt = None
        self.mt_list = []
        self.position_list = None

    @staticmethod
    def get_flatten_tensor(lst_position_id, past_token, prep_guess_token):
        flatten_input_token = []
        flatten_position = []

        if prep_guess_token is not None:
            for _, guess in enumerate(prep_guess_token):
                flatten_input_token.extend(guess)
                flatten_position.extend(list(range(lst_position_id + 1, lst_position_id + 1 + len(guess))))

        for line, past_token_line in enumerate(past_token):
            if past_token_line is None:
                break
            offset = max(line, 1)
            flatten_position.extend(
                list(
                    range(
                        lst_position_id + offset,
                        lst_position_id + offset + len(past_token_line),
                    )
                )
            )
            flatten_input_token.extend(past_token_line)

        return flatten_input_token, flatten_position

    @staticmethod
    def get_last_gen_size(batch_size, last_gen_tokens):
        gen_size = 0
        for batch in range(batch_size):
            gen_size += len(last_gen_tokens[batch])
        return gen_size

    @staticmethod
    def cal_extend_len_with_guess_tokens(guess_tokens, index):
        cur_extend_len = 0
        if guess_tokens is not None and guess_tokens[index] is not None:
            guess_num = len(guess_tokens[index])
            for guess_index in range(guess_num):
                cur_extend_len += len(guess_tokens[index][guess_index])

        return cur_extend_len

    @staticmethod
    def cal_extend_len_with_past_tokens(past_tokens, index):
        cur_extend_len = 0
        if past_tokens[index] is not None:
            for _, past_token_line in enumerate(past_tokens[index]):
                if past_token_line is None:
                    break
                cur_extend_len += len(past_token_line)

        return cur_extend_len

    def count_extend_len(self, batch_size, guess_tokens, past_tokens):
        sum_extend_len = 0
        max_extend_len = 0
        for batch in range(batch_size):
            this_extend_len = 0
            this_extend_len += self.cal_extend_len_with_guess_tokens(guess_tokens, batch)
            this_extend_len += self.cal_extend_len_with_past_tokens(past_tokens, batch)
            sum_extend_len += this_extend_len
            max_extend_len = max(max_extend_len, this_extend_len)
        return sum_extend_len, max_extend_len

    def handle_input(self, raw_model_inputs, input_metadata, attention_mask):
        if input_metadata.is_prefill:
            if self.request_stats_list is not None:
                for batch in range(input_metadata.batch_size):
                    self.request_stats_list.new_request_in(
                        input_metadata.batch_seq_len[batch],
                        input_metadata.batch_request_ids[batch],
                    )

            self.cu_seq_len = np.zeros(input_metadata.batch_size + 1, dtype=np.int64)
            self.cu_seq_len[1:] = raw_model_inputs.prefill_head_indices + 1
            start = 0
            for i, length in enumerate(input_metadata.batch_seq_len):
                input_ids = input_metadata.input_ids[start : start + length]
                start += length
                self.la_cache.save_tail_tokens(input_metadata.all_sequence_ids[i], input_ids)
                self.la_cache.set_input_len(input_metadata.all_sequence_ids[i], length)
            model_inputs = raw_model_inputs
            self.position_list = [0] * input_metadata.batch_size
            q_len = None
        else:
            seed_array = input_metadata.batch_seeds
            self.position_list = raw_model_inputs.position_ids.tolist()
            input_len_list = self.la_cache.get_input_lens(input_metadata.all_sequence_ids)
            for batch, position in enumerate(self.position_list):
                self.position_list[batch] = position + 1 - input_len_list[batch]
            model_inputs, attention_mask = self.la_preprocess(raw_model_inputs, input_metadata, seed_array)
            q_len = (self.cu_seq_len[1:] - self.cu_seq_len[:-1]).tolist()
        return model_inputs, q_len, attention_mask

    def make_attention_mask(self, input_len, last_gen_len, is_prefill, past_token, prep_guess_token):
        dtype = np.float16
        cur_window = len(past_token[0]) if past_token is not None else 0
        if is_prefill:
            tgt_len = input_len + cur_window
            past_length = input_len - last_gen_len
            triangle_mask = self.default_mask[-1]
            tri_len = triangle_mask.shape[0]
            if tri_len < tgt_len:
                self.default_mask[-1] = np.array([])
                add_default_size_time = ((tgt_len - tri_len) // DEFAULT_MASK_START_SIZE) + 1
                triangle_mask = np.full(
                    (
                        tri_len + DEFAULT_MASK_START_SIZE * add_default_size_time,
                        tri_len + DEFAULT_MASK_START_SIZE * add_default_size_time,
                    ),
                    np.finfo(dtype).min,
                    dtype=dtype,
                )
                triangle_mask = make_triangle_mask_numpy(triangle_mask)
                self.default_mask[-1] = triangle_mask
            mask = triangle_mask[past_length:tgt_len, :tgt_len].copy()
            return mask

        cur_window += 1
        line = 0
        if past_token is not None:
            while line < len(past_token):
                if past_token[line] is None:
                    break
                line += 1

        num_guess = 0 if prep_guess_token is None else len(prep_guess_token)
        guess_len = self.la_cache.level - 1
        tgt_len = cur_window * line + num_guess * guess_len - 1

        past_length = input_len - last_gen_len
        mask_length = input_len + tgt_len if tgt_len >= 0 else input_len
        triangle_mask = self.default_mask[-1]
        if triangle_mask.shape[0] < mask_length:
            self.default_mask[-1] = np.array([])
            add_default_size_time = ((mask_length - triangle_mask.shape[0]) // DEFAULT_MASK_START_SIZE) + 1
            triangle_mask = np.full(
                (
                    triangle_mask.shape[0] + DEFAULT_MASK_START_SIZE * add_default_size_time,
                    triangle_mask.shape[0] + DEFAULT_MASK_START_SIZE * add_default_size_time,
                ),
                np.finfo(dtype).min,
                dtype=dtype,
            )
            triangle_mask = make_triangle_mask_numpy(triangle_mask)
            self.default_mask[-1] = triangle_mask
        mask = triangle_mask[past_length:mask_length, :mask_length].copy()
        if tgt_len >= 0:
            default_mask = self.default_mask[line - 1]
            mask[last_gen_len:, mask_length - tgt_len : mask_length] = default_mask[-tgt_len:, -tgt_len:]
        return mask

    def insert_slots(self, start, last_gen_len, extend_seq_len, block_tables):
        block_size = self.block_size
        remain = last_gen_len + extend_seq_len
        new_slots = np.empty(last_gen_len + extend_seq_len, dtype=np.int32)
        block_idx = start // block_size
        slot_start = start % block_size
        pos_now = 0

        while remain > 0:
            block_rest = block_size - slot_start
            next_length = min(block_rest, remain)
            block_num = block_tables[block_idx].item()
            slots = self.infer_context.block_table_to_slots(block_num)
            new_slots[pos_now : pos_now + next_length] = slots[slot_start : slot_start + next_length].tolist()
            slot_start += next_length
            block_idx += slot_start // block_size
            slot_start = slot_start % block_size
            remain -= next_length
            pos_now += next_length

        return new_slots

    def update_decode_model_inputs_prepare(self, model_inputs, past_tokens, prep_guess_tokens, last_gen_tokens):
        batch_size = model_inputs.context_length.shape[0]
        sum_extend_len, max_extend_len = self.count_extend_len(batch_size, prep_guess_tokens, past_tokens)
        input_size = self.get_last_gen_size(batch_size, last_gen_tokens)
        input_ids_tensor = np.zeros(sum_extend_len + input_size, dtype=np.int64)
        position_ids_tensor = np.zeros(sum_extend_len + input_size, dtype=np.int64)
        is_prefill = np.zeros(batch_size + 1, dtype=np.int32)
        max_seq_len = model_inputs.max_seq_len
        slots_tensor = np.zeros(sum_extend_len + input_size, dtype=np.int32)
        dtype = np.float16
        batch_mask = np.full(
            (sum_extend_len + input_size, max_extend_len + max_seq_len),
            np.finfo(dtype).min,
            dtype=dtype,
        )

        last_tokens = np.zeros(batch_size, dtype=np.int32)

        res = (
            batch_size,
            last_tokens,
            input_ids_tensor,
            position_ids_tensor,
            slots_tensor,
            is_prefill,
            batch_mask,
            max_seq_len,
        )

        return res

    def update_decode_model_inputs(self, model_inputs, past_tokens, prep_guess_tokens, last_gen_tokens):
        (
            batch_size,
            last_tokens,
            input_ids_tensor,
            position_ids_tensor,
            slots_tensor,
            is_prefill,
            batch_mask,
            max_seq_len,
        ) = self.update_decode_model_inputs_prepare(model_inputs, past_tokens, prep_guess_tokens, last_gen_tokens)
        total_seq_len = 0
        mask_pos = 0
        for batch in range(batch_size):
            past_token = past_tokens[batch]
            prep_guess_token = prep_guess_tokens[batch]
            lst_position_id = model_inputs.position_ids[batch].item()
            last_tokens[batch] = model_inputs.input_ids[batch].item()
            last_gen_token = last_gen_tokens[batch]
            last_gen_len = len(last_gen_token)

            flatten_input_tensor, flatten_position_tensor = self.get_flatten_tensor(
                lst_position_id, past_token, prep_guess_token
            )

            seq_len = model_inputs.context_length[batch].item()
            extend_seq_len = len(flatten_input_tensor)
            new_seq_len = seq_len + extend_seq_len

            new_slot_tensor = self.insert_slots(
                seq_len - last_gen_len,
                last_gen_len,
                extend_seq_len,
                model_inputs.block_tables[batch],
            )

            input_ids_tensor[total_seq_len : total_seq_len + last_gen_len] = last_gen_token
            gen_position_id = list(range(lst_position_id - last_gen_len + 1, lst_position_id + 1))
            position_ids_tensor[total_seq_len : total_seq_len + last_gen_len] = gen_position_id

            slots_tensor[total_seq_len : total_seq_len + last_gen_len + extend_seq_len] = new_slot_tensor
            total_seq_len += last_gen_len

            if extend_seq_len > 0:
                input_ids_tensor[total_seq_len : total_seq_len + extend_seq_len] = flatten_input_tensor
                position_ids_tensor[total_seq_len : total_seq_len + extend_seq_len] = flatten_position_tensor

            model_inputs.context_length[batch] = new_seq_len
            max_seq_len = max(max_seq_len, new_seq_len)
            total_seq_len += extend_seq_len
            is_prefill[batch + 1] = total_seq_len

            if_la_first_round = False
            if past_token is not None:
                if_la_first_round = past_token[1] is None
            attention_mask = self.make_attention_mask(
                seq_len, last_gen_len, if_la_first_round, past_token, prep_guess_token
            )
            batch_mask[mask_pos : mask_pos + attention_mask.shape[0], : attention_mask.shape[1]] = attention_mask
            mask_pos += attention_mask.shape[0]
        self.last_tokens = last_tokens
        model_inputs.input_ids = input_ids_tensor[:total_seq_len]
        model_inputs.position_ids = position_ids_tensor[:total_seq_len]
        model_inputs.slots = slots_tensor[:total_seq_len]
        model_inputs.max_seq_len = max_seq_len
        self.cu_seq_len = tensor_backend.tensor(is_prefill, device=self.device)
        batch_mask = tensor_backend.tensor(batch_mask, dtype=self.dtype, device=self.device)
        return model_inputs, batch_mask[:mask_pos, :max_seq_len]

    def la_preprocess(self, model_inputs, meta_data, seed_array: np.ndarray):
        past_tokens = []
        prep_guess_tokens = []
        last_gen_tokens = []
        for batch, _ in enumerate(meta_data.all_sequence_ids):
            req_id = meta_data.all_sequence_ids[batch]
            batch_past_token = self.la_cache.get_past_tokens(req_id)
            # seed maybe array(None, dtype=object)
            seed = seed_array.item() if seed_array.ndim == 0 else seed_array[batch]
            seed = seed if seed is not None else 1
            if batch_past_token is None:
                self.la_cache.fill_pool_with_prompt(req_id, seed)
                prep_guess_tokens.append(None)
                last_gen_tokens.append([model_inputs.input_ids[batch].item()])
                batch_past_token = self.la_cache.get_past_tokens(req_id)
            else:
                lst_token = model_inputs.input_ids[batch].item()
                prep_guess_tokens.append(self.la_cache.get_guess_token(req_id, lst_token))
                last_gen_tokens.append(self.la_cache.get_gen_tokens(req_id))
            past_tokens.append(batch_past_token)
        self.store_guess_tokens = prep_guess_tokens
        return self.update_decode_model_inputs(model_inputs, past_tokens, prep_guess_tokens, last_gen_tokens)

    def handle_eos(self, req_id):
        self.la_cache.del_req_cache(req_id)
        self.store_guess_tokens = []
        self.last_tokens = None
        return

    def get_input_ids_pad_preprocess_guess(
        self,
        batch_size,
        total_logits_nums,
        logits_num_per_batch,
        guess_tokens_num_per_batch,
    ):
        if not self.store_guess_tokens:
            return total_logits_nums, logits_num_per_batch
        for batch in range(batch_size):
            guess_tokens = self.store_guess_tokens[batch]
            if guess_tokens is None:
                continue
            guess_token_num = 0
            for guess_token in guess_tokens:
                guess_token_num += len(guess_token)
            guess_tokens_num_per_batch[batch] = guess_token_num
            total_logits_nums += guess_token_num
            logits_num_per_batch[batch] += guess_token_num

        return total_logits_nums, logits_num_per_batch

    def get_input_ids_pad_process_guess(self, index, input_ids_pad, guess_index, input_ids_len):
        if not self.store_guess_tokens:
            return True
        guess_tokens = self.store_guess_tokens[index]
        if guess_tokens is None:
            return False
        for guess_set in guess_tokens:
            for idx, _ in enumerate(guess_set):
                input_ids_pad[guess_index, input_ids_len : input_ids_len + idx + 1] = np.array(
                    list(guess_set[: idx + 1])
                )
                guess_index += 1
        return True

    def get_input_ids_pad(self, batch_size, sampling_data):
        total_logits_nums = batch_size
        logits_num_per_batch = [1] * batch_size
        guess_tokens_num_per_batch = [0] * batch_size
        input_ids_pad = None
        total_logits_nums, logits_num_per_batch = self.get_input_ids_pad_preprocess_guess(
            batch_size,
            total_logits_nums,
            logits_num_per_batch,
            guess_tokens_num_per_batch,
        )

        if sampling_data.all_token_ids is None:
            return input_ids_pad
        input_ids_len = tensor_backend.shape(sampling_data.all_token_ids, -1)

        input_ids_pad = np.zeros((total_logits_nums, input_ids_len + (self.level - 1)), dtype=np.int64)
        index = 0
        for batch in range(batch_size):
            input_ids = np.expand_dims(
                tensor_backend.numpy(tensor_backend.cpu(sampling_data.all_token_ids[batch])),
                axis=0,
            )
            input_ids_pad[index : index + logits_num_per_batch[batch], :input_ids_len] = np.repeat(
                input_ids, logits_num_per_batch[batch], axis=0
            )
            input_ids_pad[index : index + logits_num_per_batch[batch], input_ids_len:] = input_ids[0][0]
            guess_index = index + 1
            index += logits_num_per_batch[batch]
            guess_flag = self.get_input_ids_pad_process_guess(batch, input_ids_pad, guess_index, input_ids_len)
            if not guess_flag:
                continue
        input_ids_pad = tensor_backend.tensor(
            input_ids_pad, dtype=sampling_data.all_token_ids.dtype, device=self.device
        )
        return input_ids_pad

    def get_past_logits(self, model_outputs, req_id, start_pos):
        past_logits = None

        past_token = self.la_cache.get_past_tokens(req_id)
        if past_token is not None:
            past_token_out_len = len(past_token[0]) if past_token[1] is None else len(past_token[0]) + 1
            past_logits = model_outputs[start_pos - past_token_out_len : start_pos]

        return past_logits

    def la_verify_greedy_one_batch(self, verify_guess_tokens, next_guess_tokens, next_guess_indices):
        need_cal_kv = False
        if verify_guess_tokens is None:
            return [next_guess_indices[0]], need_cal_kv

        first_guess = next_guess_tokens[0]
        guess_size = self.level - 1
        hits = [next_guess_indices[0]]
        max_hit = 0

        guess_indices = next_guess_indices[1:]
        guess_results = next_guess_tokens[1:]
        for eg, guess_tokens in enumerate(verify_guess_tokens):
            egx = eg * guess_size
            correct_indices = [next_guess_indices[0]] + guess_indices[egx : egx + guess_size]
            correct = [first_guess] + guess_results[egx : egx + guess_size].tolist()
            my_guess = guess_tokens
            gg = 0
            while gg < len(my_guess):
                if my_guess[gg] != correct[gg]:
                    break
                gg += 1
            if gg > max_hit:
                max_hit = gg
                hits = correct_indices[: max_hit + 1]
                if eg > 0:
                    need_cal_kv = True
        return hits, need_cal_kv

    def truncate_token_ids(self, cache_ids, metadata, sampling_output, next_tokens_indices):
        output_token_len = self.infer_context.get_output_len_count(cache_ids)[sampling_output.repeating_indices]
        output_space_left = metadata.batch_max_output_lens[sampling_output.repeating_indices] - output_token_len
        for idx, token_indices in enumerate(next_tokens_indices):
            num_new_tokens = len(token_indices)
            seq_token_ids = sampling_output.token_ids[token_indices]
            for i, token_id in enumerate(seq_token_ids):
                if isinstance(self.eos_token_id, int):
                    if token_id == self.eos_token_id:
                        num_new_tokens = i + 1
                        break
                elif isinstance(self.eos_token_id, list) and self.eos_token_id:
                    if token_id in self.eos_token_id:
                        num_new_tokens = i + 1
                        break

            if output_space_left[idx] < num_new_tokens:
                num_new_tokens = output_space_left[idx]
            next_tokens_indices[idx] = token_indices[:num_new_tokens]

    def la_update(self, metadata, sampling_output, inp_tokens_batch):
        for idx, seq_token_ids in enumerate(sampling_output.token_ids):
            verify_guess_tokens = self.store_guess_tokens[idx] if not metadata.is_prefill else None
            seq_id = sampling_output.sequence_ids[idx]
            guess_size = self.level - 1

            if not metadata.is_prefill:
                lst_token = self.last_tokens[idx]
                if inp_tokens_batch:
                    self.la_cache.update_new_results(seq_id, lst_token, inp_tokens_batch[idx])
                self.la_cache.append_new_generated_pool(seq_id, seq_token_ids, sampling_output.num_new_tokens[idx])
            if self.request_stats_list is not None:
                total_guess_len = 0
                if verify_guess_tokens is not None:
                    total_guess_len = len(verify_guess_tokens) * guess_size
                self.request_stats_list.update_request_stats(
                    len(seq_token_ids) - 1, len(seq_token_ids), total_guess_len, seq_id
                )
            if metadata.is_prefill:
                self.la_cache.save_tail_tokens(seq_id, seq_token_ids)
