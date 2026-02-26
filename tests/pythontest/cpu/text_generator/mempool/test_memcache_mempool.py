#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
import unittest
from unittest import mock
import torch
from mindie_llm.text_generator.mempool.base import MemPool


class TestUnifiedCache(unittest.TestCase):
    @mock.patch.dict(sys.modules, {"memcache_hybrid": mock.MagicMock()})
    @mock.patch("memcache_hybrid.DistributedObjectStore")
    def setUp(self, MockDistributedObjectStore):
        self.mock_store = mock.MagicMock()
        MockDistributedObjectStore.return_value = self.mock_store
        self.mock_store.init.return_value = 0
        self.config_path = "test.conf"
        self.mempool = MemPool.create_pool("memcache", self.config_path, role="worker")

    def test_exists(self):
        self.mock_store.is_exist.reset_mock()
        self.mock_store.is_exist.return_value = 1
        self.assertTrue(self.mempool.exists("abc"))
        self.mock_store.is_exist.return_value = 0
        self.assertFalse(self.mempool.exists("def"))

    def test_exists_fail(self):
        for bad in [123, ["k"], {"k": 1}, None]:
            with self.subTest(bad=bad):
                self.mock_store.is_exist.reset_mock()
                self.assertFalse(self.mempool.exists(bad))
                self.mempool.store.is_exist.assert_not_called()
    
    def test_batch_exist(self):
        self.mock_store.batch_is_exist.reset_mock()
        keys = ['abc', 'edf']
        expect_ret = [1, 0]
        self.mock_store.batch_is_exist.return_value = expect_ret
        self.assertEqual(self.mempool.batch_exist(keys), expect_ret)

    def test_batch_exist_fail(self):
        for bad in [123, "k"]:
            with self.subTest(bad=bad):
                self.mock_store.batch_is_exist.reset_mock()
                self.assertEqual(self.mempool.batch_exist(bad), [False])
                self.mempool.store.batch_is_exist.assert_not_called()

    def test_put_success(self):
        single_tensor1 = torch.rand(32, 128, 16)
        single_tensor2 = torch.rand(4, 128, 16)
        key_cases = [
            "k_single_str",
            ["k1_list", "k2_list"] 
        ]
        tensor_cases = [
            [single_tensor1, single_tensor2],
            [[single_tensor1, single_tensor1], [single_tensor2, single_tensor2]]
        ]
        with self.subTest("single key"):
            except_res = [True]
            self.mock_store.batch_put_from_layers.return_value = [0]
            ret = self.mempool.put(key_cases[0], tensor_cases[0])
            self.assertEqual(ret, except_res)
        with self.subTest("multi keys"):
            except_res = [True] * len(key_cases[1])
            self.mock_store.batch_put_from_layers.return_value = [0] * len(key_cases[1])
            ret = self.mempool.put(key_cases[1], tensor_cases[1])
            self.assertEqual(ret, except_res)
        
    def test_put_fail(self):
        single_tensor = torch.rand(2, 3)
        with self.subTest("len mismatch"):
            self.assertEqual(self.mempool.put(["k1", "k2"], [single_tensor]), [False])
        with self.subTest("bad key type"):
            self.assertEqual(self.mempool.put(123, single_tensor), [False])
        with self.subTest("backend error"):
            self.mock_store.batch_put_from_layers.return_value = (-1,)  # -1: put failed
            self.assertEqual(self.mempool.put("bad", single_tensor), [False])
        with self.subTest("put exception"):
            self.mock_store.batch_put_from_layers.side_effect = Exception("mock error")
            self.assertEqual(self.mempool.put("k1", single_tensor), [False])

    def test_get_success(self):
        single_tensor1 = torch.rand(32, 128, 16)
        single_tensor2 = torch.rand(4, 128, 16)
        key_cases = [
            "k_single_str",
            ["k1_list", "k2_list"] 
        ]
        tensor_cases = [
            [single_tensor1, single_tensor2],
            [[single_tensor1, single_tensor1], [single_tensor2, single_tensor2]]
        ]
        with self.subTest("single key"):
            except_res = [True]
            self.mock_store.batch_get_into_layers.return_value = [0]
            ret = self.mempool.get(key_cases[0], tensor_cases[0])
            self.assertEqual(ret, except_res)
        with self.subTest("multi keys"):
            except_res = [True] * len(key_cases[1])
            self.mock_store.batch_get_into_layers.return_value = [0] * len(key_cases[1])
            ret = self.mempool.get(key_cases[1], tensor_cases[1])
            self.assertEqual(ret, except_res)

    def test_get_fail(self):
        single_tensor = torch.empty(2, 3)
        with self.subTest("len mismatch"):
            self.assertEqual(self.mempool.get(["k1", "k2"], [single_tensor]), [False])
        with self.subTest("bad key type"):
            self.assertEqual(self.mempool.get(123, single_tensor), [False])
        with self.subTest("backend error"):
            self.mock_store.is_exist.return_value = 1
            self.mock_store.batch_get_into_layers.return_value = (-1,)  # -1: get failed
            self.assertEqual(self.mempool.get("bad", single_tensor), [False])
        with self.subTest("put exception"):
            self.mock_store.batch_get_into_layers.side_effect = Exception("mock error")
            self.assertEqual(self.mempool.get("k1", single_tensor), [False])
