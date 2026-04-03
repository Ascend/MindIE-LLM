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

import base64
import binascii
import os
import shutil
import time
from io import BytesIO
from multiprocessing import shared_memory
import re
import psutil
import requests
from PIL import Image
from urllib3.util import parse_url

from .tokenizer_log import logger
from . import file_utils
from .media_limits import get_max_image_file_size_bytes, get_max_image_file_size_mb


_CHUNK_SIZE = 1024 * 1024
_TOKENIZER_ENCODE_TIMEOUT = "TOKENIZER_ENCODE_TIMEOUT"
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 30
_MAX_TOKENIZER_NUMBER = 32

_STANDARD_S3_PATTERNS = [
    re.compile(r'^[a-z0-9.\-]+\.s3[.-][a-z0-9\-]+\.amazonaws\.com$'),
    re.compile(r'^s3[.-][a-z0-9\-]+\.amazonaws\.com$'),
    re.compile(r'^[a-z0-9.\-]+\.s3\.amazonaws\.com$'),
    re.compile(r'^s3\.amazonaws\.com$'),
    re.compile(r'^[a-z0-9.\-]+\.s3-website[.-][a-z0-9\-]+\.amazonaws\.com$'),
]
_S3_COMPATIBLE_PATTERNS = [
    re.compile(r'^[a-z0-9.\-]+\.([a-z0-9\-]+\.)?digitaloceanspaces\.com$'),
    re.compile(r'^[a-z0-9.\-]+\.minio([-.][a-z0-9\-]+)?\.[a-z0-9\-]+$'),
    re.compile(r'^[a-z0-9.\-]+\.([a-z0-9\-]+\.)?(storage|objectstorage|s3)\.[a-z0-9\-]+\.[a-z0-9\-]+$')
]
_S3_SIGNATURE_PATTERNS = frozenset([
    'x-amz-algorithm',
    'x-amz-credential',
    'x-amz-date',
    'x-amz-expires',
    'x-amz-signedheaders',
    'x-amz-signature',
    'awsaccesskeyid',
    'signature',
    'x-amz-security-token'
])
_S3_DOMAINS = frozenset(['s3.amazonaws.com', 's3.us-east-1.amazonaws.com'])
_S3_KEYWORDS = frozenset(['amazonaws', 's3', 'storage', 'objectstorage', 'spaces', 'minio'])


def fetch_media_url(image_url, input_type: str, ext: str, limit_params: tuple,
                    media_type_dict: dict[str, list[str]]):
    if ext.lower() not in media_type_dict.get(input_type):
        raise ValueError(f"The media type is {input_type}, url must end with one of {media_type_dict.get(input_type)}.")
    size_limit_dict, total_start_time = limit_params
    size_limit = size_limit_dict.get(input_type, 0)
    if size_limit <= 0:
        raise ValueError(f'Invalid size limit for input type: {input_type}.')

    current_memory = psutil.virtual_memory()
    if current_memory.available < size_limit * _MAX_TOKENIZER_NUMBER * 2:
        raise ValueError("Insufficient system memory for download.")

    download_timeout = os.getenv(_TOKENIZER_ENCODE_TIMEOUT, "60")
    try:
        download_timeout = float(download_timeout)
    except ValueError:
        download_timeout = 60.0

    media_content = bytearray()
    total_size = 0

    try:
        time_params = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
        with requests.get(image_url, stream=True, timeout=time_params, verify=True, allow_redirects=False) as response:
            response.raise_for_status()
            elapsed_time = time.time() - total_start_time
            if elapsed_time > download_timeout:
                media_content = bytearray()
                raise ValueError(f"Download timed out during initial response after {download_timeout} seconds.")

            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                elapsed_time = time.time() - total_start_time
                if elapsed_time > download_timeout:
                    media_content = bytearray()
                    raise ValueError(f"Download timed out after {download_timeout} seconds.")
                if chunk:
                    total_size += len(chunk)
                    current_memory = psutil.virtual_memory()
                    if current_memory.available < size_limit * _MAX_TOKENIZER_NUMBER * 2:
                        media_content = bytearray()
                        raise ValueError("Insufficient system memory for download.")
                    if total_size > size_limit:
                        media_content = bytearray()
                        raise ValueError(f'The size of {input_type} exceeds the limit '
                                         f'of {size_limit / (1024 * 1024):.2f} MB.')
                    media_content.extend(chunk)
            return bytes(media_content), total_size
    except requests.exceptions.ConnectTimeout as e:
        media_content = bytearray()
        raise RuntimeError(f"Connection timed out after {_CONNECT_TIMEOUT} seconds.") from e
    except requests.exceptions.ReadTimeout as e:
        media_content = bytearray()
        raise RuntimeError(f"Read timed out after {_READ_TIMEOUT} seconds.") from e
    except requests.RequestException as e:
        media_content = bytearray()
        raise RuntimeError("Download error") from e


def save_image(image_byte_data, image_save_path):
    single_image_limit = get_max_image_file_size_bytes()
    if len(image_byte_data) > single_image_limit:
        raise ValueError(f'The size of image cannot exceed {get_max_image_file_size_mb()}MB')
    try:
        # Image.open will check whether the binary content is a valid picture content.
        # verify() can check the integrity of content.
        # Both are used here to validate content.
        with Image.open(BytesIO(image_byte_data)) as img:
            img.verify()
        with file_utils.safe_open(image_save_path, mode='wb') as f:
            f.write(image_byte_data)
    except IOError as e:
        raise RuntimeError("Invalid image content, check the input image") from e
    except Exception as e:
        raise RuntimeError("Error when saving img") from e


def decode_base64_content(url: str):
    try:
        decoded_bytes = base64.b64decode(url, validate=True)
        return decoded_bytes
    except binascii.Error as e:
        raise ValueError("Invalid base64 url") from e


def copy_media(ori_path, save_dir, ext):
    save_dir = file_utils.standardize_path(save_dir)
    file_utils.check_path_permission(save_dir)
    file_count = len(os.listdir(save_dir))
    new_filename = f"{file_count + 1}{ext}"
    save_path = os.path.join(save_dir, new_filename)
    try:
        shutil.copy(ori_path, save_path)
    except FileNotFoundError as file_not_found_error:
        raise IOError("Media not found to copy to the cache dir.") from file_not_found_error
    except Exception as e:
        raise IOError("Error when copy media to the cache dir.") from e


def save_media(content, cache_dir, ext):
    cache_dir = file_utils.standardize_path(cache_dir)
    file_utils.check_path_permission(cache_dir)
    file_count = len(os.listdir(cache_dir))
    new_filename = f"{file_count + 1}{ext}"
    save_path = os.path.join(cache_dir, new_filename)
    try:
        with file_utils.safe_open(save_path, mode='wb', permission_mode=0o640) as fd:
            fd.write(content)
    except FileNotFoundError as file_not_found_error:
        raise IOError("Error when save media, file not found.") from file_not_found_error
    except Exception as e:
        raise IOError("Error when save media.") from e


def create_cache_dir(dir_path):
    cache_dir_paths = [
        dir_path,
        os.path.join(dir_path, "image"),
        os.path.join(dir_path, "video"),
        os.path.join(dir_path, "audio")
    ]
    shm_save_path = os.path.join(dir_path, "shm_name.txt")

    try:
        for single_dir in cache_dir_paths:
            if os.path.exists(single_dir):
                single_dir = file_utils.standardize_path(single_dir)
                file_utils.check_path_permission(single_dir)
            else:
                os.makedirs(single_dir, exist_ok=True)
                os.chmod(single_dir, 0o750)
        with file_utils.safe_open(shm_save_path, mode='wb', permission_mode=0o640):
            pass
    except FileNotFoundError as file_not_found_error:
        raise IOError("Error when create cache dir, file not found.") from file_not_found_error
    except Exception as e:
        raise IOError("Error when create cache dir.") from e


def release_shared_memory(file_path):
    if not file_utils.is_path_exists(file_path):
        return

    file_path = file_utils.standardize_path(file_path)
    file_utils.check_path_permission(file_path, mode=0o640)
    with file_utils.safe_open(file_path, mode='r', permission_mode=0o640) as f:
        shm_names = [line.strip() for line in file_utils.safe_readlines(f)]
        for name in shm_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except ValueError as value_error:
                logger.info(f"Share memory may have been released. {value_error}")
            except Exception as e:
                logger.info(f"Share memory may have been released. {e}")


def remove_cache_dir(dir_path):
    if not file_utils.is_path_exists(dir_path):
        return

    if os.path.exists(dir_path):
        dir_path = file_utils.standardize_path(dir_path)
        file_utils.check_path_permission(dir_path)
    else:
        os.makedirs(dir_path, exist_ok=True)
        os.chmod(dir_path, 0o640)
    shm_save_path = os.path.join(dir_path, "shm_name.txt") 
    release_shared_memory(shm_save_path)

    try:
        dir_path = file_utils.standardize_path(dir_path)
        file_utils.check_path_permission(dir_path)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            raise ValueError("Cache path is not a directory.")
    except ValueError as value_error:
        raise IOError("Remove cache dir error.") from value_error
    except Exception as e:
        raise IOError("Remove cache dir error.") from e


def clear_meida_cache(dir_path: str):
    if not file_utils.is_path_exists(dir_path):
        return
    try:
        dir_path = file_utils.standardize_path(dir_path)
        file_utils.check_path_permission(dir_path)
        for media_type in ['image', 'video', 'audio']:
            media_dir = os.path.join(dir_path, media_type)
            if not file_utils.is_path_exists(media_dir):
                continue
            media_dir = file_utils.standardize_path(media_dir)
            file_utils.check_path_permission(media_dir)
            if os.path.isdir(media_dir):
                shutil.rmtree(media_dir)
    except OSError as os_error:
        raise IOError("Clear cache media failed.") from os_error
    except Exception as e:
        raise IOError("Clear cache media failed.") from e


def is_s3_compatible_url(url):
    """
    Check if a URL is S3-compatible, including standard S3 URLs and pre-signed URLs.
    This function identifies S3-compatible URLs by checking:
    1. Standard S3 URL patterns (virtual-hosted and path-style)
    2. S3 signature parameters in query string for pre-signed URLs
    3. S3-like domain patterns or bucket structure
    Args:
        url (str): The URL to check
    Returns:
        bool: True if the URL is S3-compatible, False otherwise
    """
    try:
        parsed_url = parse_url(url)
        scheme = parsed_url.scheme or ''
        netloc = parsed_url.netloc or ''
        query = parsed_url.query or ''
        path = parsed_url.path or ''
        netloc_lower = netloc.lower()
        query_lower = query.lower()
        path_count = sum(1 for p in path.strip('/').split('/') if p)
        if not scheme or not netloc:
            return False
        # 1. Check for standard S3 URL patterns using regex
        if any(pattern.match(netloc_lower) for pattern in _STANDARD_S3_PATTERNS):
            logger.info(f"Standard S3 URL detected: {url}")
            return True
        # 2. Check for S3 signature parameters in query string (pre-signed URLs)
        if any(pattern in query_lower for pattern in _S3_SIGNATURE_PATTERNS):
            logger.info(f"S3 pre-signed URL detected: {url}")
            return True
        # 3. Check for S3-compatible services (MinIO, DigitalOcean Spaces, etc.)
        if any(pattern.match(netloc_lower) for pattern in _S3_COMPATIBLE_PATTERNS):
            logger.info(f"S3-compatible service URL detected: {url}")
            return True
        # 4. Check for bucket-like path structure in path-style URLs
        if netloc_lower in _S3_DOMAINS and path_count >= 2:
            logger.warning(f"Path-style S3 URL detected: {url}")
            return True
        # 5. Fallback check for S3-like keywords in domain
        if (any(keyword in netloc_lower for keyword in _S3_KEYWORDS) and path_count >= 2):
            logger.info(f"S3-like domain detected: {url}")
            return True
        logger.info("This URL is not S3-compatible.")
        return False
    except Exception as e:
        logger.warning(f"Invalid URL: {url}, error: {e}")
        return False


def extract_s3file_extension(url):
    """
    Extract the file extension from a URL including the dot.
    Args:
        url (str): The URL to extract file extension from
    Returns:
        str: The extracted file extension including the dot in lowercase, or empty string if none
    """
    try:
        parsed_url = parse_url(url)
        path = parsed_url.path or ''
        if not path:
            return ''
        filename = path.rstrip('/').split('/')[-1]
        if not filename:
            return ''
        last_dot_index = filename.rfind('.')
        if last_dot_index > 0 and last_dot_index < len(filename) - 1:
            file_extension = filename[last_dot_index:].lower()
            return file_extension
        return ''
    except Exception as e:
        logger.warning(f"Invalid URL: {url}, error: {e}")
        return ''


def extract_extension_from_url(ext):
    """
    Determine whether the input ext represents a regular file path or an S3 URL, and extract the file extension.
    """
    if not isinstance(ext, str) or not ext:
        return ''
    if is_s3_compatible_url(ext):
        extension = extract_s3file_extension(ext)
        if extension:
            logger.info(f"Extracted extension from S3 URL: {extension}")
            return extension
        else:
            return "Failed to extract file extension"
    else:
        _, extension = os.path.splitext(ext)
        return extension
