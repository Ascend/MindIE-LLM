# 仅用于 CPU 单测在无 PyTorch 环境下的最小桩；不用于生产推理。
from __future__ import annotations

import numpy as np


class OutOfMemoryError(RuntimeError):
    pass


def _np_dtype(dtype):
    if dtype is None:
        return np.float64
    if dtype is long:
        return np.int64
    if dtype is int32:
        return np.int32
    if dtype is int64:
        return np.int64
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return dtype
    return np.float64


class Tensor:
    __slots__ = ("_np",)

    def __init__(self, data, dtype=None, device=None):
        if isinstance(data, Tensor):
            self._np = data._np.astype(_np_dtype(dtype), copy=True)
        else:
            self._np = np.asarray(data, dtype=_np_dtype(dtype))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._np

    def flatten(self):
        return Tensor(self._np.reshape(-1))

    def index_select(self, dim, index):
        idx = index._np if isinstance(index, Tensor) else np.asarray(index, dtype=np.int64)
        return Tensor(np.take(self._np, idx, axis=dim))

    def scatter_(self, dim, index, src):
        ind = index._np if isinstance(index, Tensor) else np.asarray(index, dtype=np.int64)
        s = src._np if isinstance(src, Tensor) else np.asarray(src)
        if self._np.ndim == 1 and dim == 0:
            self._np[ind] = s
        else:
            np.put(self._np, ind, s.flat)
        return self

    def __len__(self):
        return int(self._np.size)

    def __int__(self):
        return int(self._np.reshape(-1)[0])

    def __getitem__(self, key):
        return self._np[key]


def tensor(data, dtype=None, device=None, **kwargs):
    return Tensor(data, dtype=dtype, device=device)


def zeros(shape, dtype=None, **kwargs):
    return Tensor(np.zeros(shape, dtype=_np_dtype(dtype)))


def ones(shape, dtype=None, device=None, **kwargs):
    return Tensor(np.ones(shape, dtype=_np_dtype(dtype)))


def unsqueeze(x, dim):
    if isinstance(x, Tensor):
        return Tensor(np.expand_dims(x._np, axis=dim))
    return Tensor(np.expand_dims(np.asarray(x), axis=dim))


def from_numpy(arr):
    return Tensor(np.asarray(arr))


long = np.dtype(np.int64)
int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)


class _NpuModule:
    def current_stream(self):
        return object()

    class Event:
        def record(self, _stream):
            pass


npu = _NpuModule()
