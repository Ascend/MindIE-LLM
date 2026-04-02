# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
import numpy as np

from mindie_llm.modeling.backend_type import BackendType
from mindie_llm.utils.env import ENV
from mindie_llm.utils.tensor import _set_tensor_backend


class TestTensorBackend(unittest.TestCase):

    def use_operations(self):
        device = 'npu'
        tensor_backend = _set_tensor_backend()
        tensor = tensor_backend.tensor(np.array([[0]] * 16, dtype=np.int32))
        tensor = tensor_backend.to(tensor, device)
        tensor_backend.cpu(tensor)
        tensor_backend.cumsum(tensor, 0)
        tensor_backend.equal(tensor, tensor)
        tensor_backend.fill_diagonal(tensor, 1.0)
        tensor_backend.full((16,), 0)
        tensor_backend.gather(tensor, tensor, 0)
        tensor_backend.get_device(tensor)
        tensor_backend.masked_fill(
            tensor,
            tensor_backend.to(tensor_backend.tensor(np.array([[1]] * 16, dtype=np.bool_)), device),
            0
        )
        tensor_backend.numpy(tensor)
        tensor_backend.ones((16,))
        tensor_backend.repeat(tensor, (2, 2))
        tensor_backend.scatter(tensor, 0, tensor, tensor)
        tensor_backend.shape(tensor, 0)
        tensor_backend.softmax(
            tensor_backend.to(tensor_backend.tensor(np.array([[1]] * 16, dtype=np.float16)), device),
            0
        )
        tensor_backend.where(tensor_backend.to(tensor_backend.tensor(np.array([[1]] * 16, dtype=np.float32)), device))
        tensor_backend.zeros((16,))
        
        tensor_list = [tensor_backend.tensor(np.array([[1, 2, 3]])), tensor_backend.tensor(np.array([[4, 5, 6]]))]
        golden_cat_tensor = tensor_backend.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        test_cat_tensor = tensor_backend.cat(tensor_list, dim=0)
        self.assertTrue(tensor_backend.equal(test_cat_tensor, golden_cat_tensor))

    def test_torch(self):
        ENV.framework_backend = BackendType.ATB
        self.use_operations()

    def test_mindspore(self):
        ENV.framework_backend = BackendType.MS
        self.use_operations()


if __name__ == '__main__':
    unittest.main()