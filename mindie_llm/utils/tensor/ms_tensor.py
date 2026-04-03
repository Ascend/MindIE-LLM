#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from functools import wraps

import numpy as np

from mindie_llm.utils.tensor.llm_tensor import LLMBackend


def delete_param(param):
    """定义装饰器，删除某个参数"""
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            if param in kwargs:
                del kwargs[param]
            return func(*args, **kwargs)

        return inner_wrapper

    return wrapper


def rename_param(old_name, new_name):
    """定义装饰器，将old_name参数重命名为新参数new_name"""
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            if old_name in kwargs and kwargs[old_name] is not None:
                kwargs[new_name] = kwargs[old_name]
                del kwargs[old_name]
            return func(*args, **kwargs)
        return inner_wrapper

    return wrapper


class MindSporeBackend(LLMBackend):
    def __init__(self):
        import mindspore
        self._backend = mindspore

    def get_backend(self):
        return self._backend

    def get_op(self):
        import mindspore.ops as op
        return op

    def get_mint(self):
        from mindspore import mint
        return mint

    def get_npu(self):
        return self.get_hal()

    def get_hal(self):
        return self._backend.hal

    @delete_param('device')
    def ones(self, *args, **kwargs):
        return self.get_op().ones(*args, **kwargs)

    def equal(self, *args, **kwargs):
        return self.get_op().Equal()(*args, **kwargs).all()

    def repeat(self, value, size):
        return self.get_op().Tile()(value, size)

    @rename_param(old_name='dim', new_name='axis')
    def softmax(self, *args, **kwargs):
        return self.get_op().softmax(*args, **kwargs)

    def shape(self, value, dim=None):
        return value.shape[dim]

    @rename_param(old_name='dim', new_name='axis')
    def cumsum(self, *args, **kwargs):
        return self.get_op().cumsum(*args, **kwargs)

    def gather(self, input_params, index, dim):
        return self.get_mint().gather(input_params, dim, index)

    def numpy(self, value):
        return value.asnumpy()

    def where(self, condition, input_param=None, other=None):
        if input_param is None and other is None:
            return np.where(self.numpy(condition))
        return self.get_mint().where(condition, input_param, other)

    @delete_param('device')
    def full(self, *args, **kwargs):
        return self.get_op().full(*args, **kwargs)

    @delete_param('device')
    def tensor(self, *args, **kwargs):
        return self._backend.tensor(*args, **kwargs)

    @delete_param('device')
    def zeros(self, *args, **kwargs):
        return self.get_op().zeros(*args, **kwargs)

    def fill_diagonal(self, mask, fill_value):
        return mask.fill_diagonal(fill_value)

    def to(self, value, device):
        return value

    def get_device(self, value):
        res = None
        return res

    def cpu(self, value):
        return value

    def scatter(self, input_params, axis, index, src):
        return self.get_op().scatter(input_params, axis, index, src)

    def masked_fill(self, input_params, mask, value):
        return input_params.masked_fill(mask, value)

    def pad(self, input_tensor, pad, mode, value):
        return self._backend.ops.pad(input_tensor, pad, mode, value)

    @rename_param(old_name='dim', new_name='axis')
    def cat(self, *args, **kwargs):
        return self.get_op().cat(*args, **kwargs)
