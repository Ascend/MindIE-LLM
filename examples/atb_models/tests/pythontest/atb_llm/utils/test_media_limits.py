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

import os
import unittest
from unittest.mock import patch

from atb_llm.utils.media_limits import (
    DEFAULT_MAX_AUDIO_FILE_SIZE_MB,
    DEFAULT_MAX_IMAGE_FILE_SIZE_MB,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_VIDEO_FILE_SIZE_MB,
    _get_env_int,
    get_max_audio_file_size_bytes,
    get_max_image_file_size_bytes,
    get_max_image_pixels,
    get_max_video_file_size_bytes,
)


class TestMediaLimits(unittest.TestCase):
    def test_get_env_int_returns_default_for_missing_and_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_get_env_int("TEST_ENV", 7), 7)

        with patch.dict(os.environ, {"TEST_ENV": ""}, clear=True):
            self.assertEqual(_get_env_int("TEST_ENV", 7), 7)

    def test_get_env_int_returns_default_for_invalid_and_non_positive_values(self):
        with patch.dict(os.environ, {"TEST_ENV": "abc"}, clear=True):
            self.assertEqual(_get_env_int("TEST_ENV", 9), 9)

        with patch.dict(os.environ, {"TEST_ENV": "0"}, clear=True):
            self.assertEqual(_get_env_int("TEST_ENV", 9), 9)

        with patch.dict(os.environ, {"TEST_ENV": "-3"}, clear=True):
            self.assertEqual(_get_env_int("TEST_ENV", 9), 9)

    def test_get_env_int_returns_positive_integer_value(self):
        with patch.dict(os.environ, {"TEST_ENV": "12"}, clear=True):
            self.assertEqual(_get_env_int("TEST_ENV", 5), 12)

    def test_getters_return_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_max_image_file_size_bytes(), DEFAULT_MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024)
            self.assertEqual(get_max_audio_file_size_bytes(), DEFAULT_MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024)
            self.assertEqual(get_max_video_file_size_bytes(), DEFAULT_MAX_VIDEO_FILE_SIZE_MB * 1024 * 1024)
            self.assertEqual(get_max_image_pixels(), DEFAULT_MAX_IMAGE_PIXELS)

    def test_getters_return_values_from_environment(self):
        env = {
            "MINDIE_LLM_MAX_IMAGE_FILE_SIZE_MB": "21",
            "MINDIE_LLM_MAX_AUDIO_FILE_SIZE_MB": "22",
            "MINDIE_LLM_MAX_VIDEO_FILE_SIZE_MB": "23",
            "MINDIE_LLM_MAX_IMAGE_PIXELS": "24000000",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_max_image_file_size_bytes(), 21 * 1024 * 1024)
            self.assertEqual(get_max_audio_file_size_bytes(), 22 * 1024 * 1024)
            self.assertEqual(get_max_video_file_size_bytes(), 23 * 1024 * 1024)
            self.assertEqual(get_max_image_pixels(), 24000000)


if __name__ == "__main__":
    unittest.main()
