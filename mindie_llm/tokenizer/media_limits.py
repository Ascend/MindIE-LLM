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


DEFAULT_MAX_IMAGE_FILE_SIZE_MB = 20
DEFAULT_MAX_AUDIO_FILE_SIZE_MB = 20
DEFAULT_MAX_VIDEO_FILE_SIZE_MB = 512
DEFAULT_MAX_TOTAL_MEDIA_SIZE_MB = 1000
DEFAULT_MAX_IMAGE_PIXELS = 10000 * 10000


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def get_max_image_file_size_mb() -> int:
    return _get_env_int("MINDIE_LLM_MAX_IMAGE_FILE_SIZE_MB", DEFAULT_MAX_IMAGE_FILE_SIZE_MB)


def get_max_audio_file_size_mb() -> int:
    return _get_env_int("MINDIE_LLM_MAX_AUDIO_FILE_SIZE_MB", DEFAULT_MAX_AUDIO_FILE_SIZE_MB)


def get_max_video_file_size_mb() -> int:
    return _get_env_int("MINDIE_LLM_MAX_VIDEO_FILE_SIZE_MB", DEFAULT_MAX_VIDEO_FILE_SIZE_MB)


def get_max_total_media_size_mb() -> int:
    return _get_env_int("MINDIE_LLM_MAX_TOTAL_MEDIA_SIZE_MB", DEFAULT_MAX_TOTAL_MEDIA_SIZE_MB)


def get_max_image_pixels() -> int:
    return _get_env_int("MINDIE_LLM_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS)


def get_max_image_file_size_bytes() -> int:
    return get_max_image_file_size_mb() * 1024 * 1024


def get_max_audio_file_size_bytes() -> int:
    return get_max_audio_file_size_mb() * 1024 * 1024


def get_max_video_file_size_bytes() -> int:
    return get_max_video_file_size_mb() * 1024 * 1024


def get_max_total_media_size_bytes() -> int:
    return get_max_total_media_size_mb() * 1024 * 1024
