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
from unittest.mock import patch
from unittest.mock import MagicMock

import mindspore

from mindie_llm.text_generator.adapter.generator_ms import GeneratorMS


class TestGeneartorMS(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    @patch("mindie_llm.text_generator.adapter.generator_backend.get_model_wrapper")
    def setUp(self, model_wrapper):
        a = {
            "backend_type": "ms",
            "num_thread": 1,
            "npu_mem": 1,
            "cpu_mem": 1,
            "block_size": 1,
            "npu_device_id": 1,
            "local_rank": 1,
            "rank": 1,
            "world_size": 8,
            "trust_remote_code": 1,
            "dp": 1,
            "tp": 1,
            "moe_tp": 1,
            "plugin_name": "abc",
        }
        model_wrapper.return_value = MagicMock()
        self.generator = GeneratorMS(a)
        self.generator.npu_mem_size = 1

    def test_init(self):
        class CacheManager:
            def __init__(self):
                self.sepd_worker = None
                self.block_shape = (1, 2, 3)
                self.num_npu_blocks = 1
                self.num_layers = 0
                self.dtype = mindspore.float16

            def set_property(self):
                self.sepd_worker = MagicMock()
                self.sepd_worker.addrs = []

            def set_num_layers(self):
                self.num_layers = 1

        self.generator.forward("")
        cache_manager = CacheManager()
        self.generator.update_cache_policy(cache_manager)
        cache_manager.set_property()
        self.generator.update_cache_policy(cache_manager)
        self.generator.alloc_kv_cache(cache_manager)

    @patch('mindie_llm.text_generator.adapter.generator_ms.GeneratorMS.forward', return_value=None)
    @patch('mindspore.hal.synchronize', return_value=None)
    def test_warm_up(self, _1, _2):
        class Data:
            def __init__(self):
                self.size = 0
        data = Data()
        self.generator.clear_kv_cache()
        self.generator.update_cache_after_switch_pd_role()
        self.generator.to_tensor(data)

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()