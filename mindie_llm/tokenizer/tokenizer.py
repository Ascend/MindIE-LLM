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

from typing import List
import json
import os
import signal
import multiprocessing
import threading
import time
import re
import traceback
import numpy as np
from urllib3.util import parse_url
from urllib3.exceptions import LocationParseError

from mindie_llm.modeling.model_wrapper import get_tokenizer_wrapper
from . import io_utils
from . import file_utils


logger = file_utils.get_tokenizer_logger()

_ALLOWED_MEDIA_DOMAINS_ENV = "ALLOWED_MEDIA_DOMAINS_ENV"
_ALLOWED_LOCAL_MEDIA_PATH = "/data/multimodal_inputs/"
once_flag = threading.Event()

_DURATION = 30  # check undeleted cache dir pre 30 seconds
_DELET_DURATION = 2 ** 16  # delete dirs in the cache that are older than 2**16 seconds, which is equal to e2eTimeout
_SINGLE_IMAGE_LIMIT = 20 * 1024 * 1024  # 20 MB
_SINGLE_AUDIO_LIMIT = 20 * 1024 * 1024  # 20 MB
_SINGLE_VIDEO_LIMIT = 512 * 1024 * 1024  # 512 MB
_MEDIA_SIZE_LIMIT = 1000 * 1024 * 1024 # 1 GB
_URL_LENGTH_LIMIT = 4096

_TEXT_KEY = "text"
_CONTENT_NAME_KEY = "content"
_ROLE_KEY = "role"
_IMAGE_KEY = "image_url"
_VIDEO_KEY = "video_url"
_AUDIO_KEY = "audio_url"
_INPUT_AUDIO_KEY = "input_audio"

_MEDIA_TYPE = {
    _IMAGE_KEY: ['.jpg', '.jpeg', '.png'],
    _VIDEO_KEY: ['.mp4', '.avi', '.wmv'],
    _AUDIO_KEY: ['.mp3', '.wav', '.flac'],
    _INPUT_AUDIO_KEY: ['.mp3', '.wav', '.flac'],
}

_SIZE_LIMITS = {
    _IMAGE_KEY: _SINGLE_IMAGE_LIMIT,
    _AUDIO_KEY: _SINGLE_AUDIO_LIMIT,
    _VIDEO_KEY: _SINGLE_VIDEO_LIMIT,
    _INPUT_AUDIO_KEY: _SINGLE_AUDIO_LIMIT,
}

_MIME_TYPE2EXT = {
    "jpeg": ".jpg",
    "png": ".png",
    "mp4": ".mp4",
    "x-msvideo": ".avi",
    "x-ms-wmv": ".wmv",
    "mpeg": ".mp3",
    "x-wav": ".wav",
    "flac": ".flac"
}


class IbisTokenizer:
    def __init__(self, path: str, bakend_type: str, trust_remote_code: bool, models_dict_str: str):
        try:
            parent_pid_cache_prefix = 'cache_' + str(os.getppid())
            self.cache_prefix = str(os.getpid()) + '_'
            self.cache_path = self._get_cache_base_path(parent_pid_cache_prefix)
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path, exist_ok=True)
                os.chmod(self.cache_path, 0o750)
                self.check_path(self.cache_path)
                self.release_thread = multiprocessing.Process(target=self.release_cache, args=(_DURATION,), daemon=True)
                self.release_thread.start()

            wrapper = get_tokenizer_wrapper(
                path, bakend_type, trust_remote_code=trust_remote_code, models_dict=models_dict_str)
            self.tokenizer = wrapper.tokenizer
            self.input_builder = wrapper.input_builder
            self.tokenize = wrapper.tokenize
            self.wrapper_encode = wrapper.encode
            self.wrapper_decode = wrapper.decode
        except BaseException as e:
            logger.exception("IbisTokenizer init failed!!!")
            raise RuntimeError("IbisTokenizer init failed!!!") from e
        self.media_cache_dirs = None
        self.timestamp = None

    def __del__(self):
        cache_path = getattr(self, 'cache_path', None)
        if cache_path is None:
            return
        dir_path = file_utils.standardize_path(cache_path)
        file_utils.check_path_permission(dir_path)
        all_request = os.listdir(dir_path)
        for request in all_request:
            if request.startswith(self.cache_prefix):
                self.delete_multimodal_cache(int(request), self.cache_prefix)

    @staticmethod
    def is_mm(prompt):
        try:
            prompt_obj = json.loads(prompt)
        except ValueError as value_error:
            # the prompt is not multimodal format.
            return False
        except Exception as e:
            # adapt to the old input format.
            return False
        if isinstance(prompt_obj, list):
            for single_msg in prompt_obj:
                if not isinstance(single_msg, dict):
                    logger.error(f"The input type of '{type(single_msg)}' is invalid, it should be a List[Dict].")
                    raise ValueError(f"The input type of '{type(single_msg)}' is invalid, it should be a List[Dict].")
                if _ROLE_KEY not in single_msg or isinstance(single_msg.get(_CONTENT_NAME_KEY), list):
                    return True
        return False

    @staticmethod
    def check_path(path):
        if os.path.exists(path):
            path = file_utils.standardize_path(path)
            file_utils.check_path_permission(path)
        else:
            raise ValueError(f"'{path}' path not exist.")

    @staticmethod
    def _process_url_path(media_url, ext, input_type, cache_dir, total_start_time):
        try:
            parsed_url = parse_url(media_url)
            scheme = parsed_url.scheme if parsed_url.scheme else ""
            if scheme not in ("http", "https"):
                logger.error("Invalid HTTP URL.")
                raise ValueError("Invalid HTTP URL.")
        except LocationParseError as e:
            logger.error(f"Failed to parse URL: {media_url}")
            raise ValueError(f"Invalid URL: {media_url}") from e
        except Exception as e:
            logger.error(f"Failed to parse URL: {media_url}")
            raise ValueError(f"Invalid URL: {media_url}") from e

        limit_params = (_SIZE_LIMITS, total_start_time)
        media_content, media_size = io_utils.fetch_media_url(media_url, input_type, ext, limit_params, _MEDIA_TYPE)
        if input_type == _IMAGE_KEY:
            image_count = len(os.listdir(cache_dir))
            image_save_path = os.path.join(cache_dir, f"{image_count + 1}.jpg")
            io_utils.save_image(media_content, image_save_path)
        else:
            io_utils.save_media(media_content, cache_dir, ext)
        return media_size

    @staticmethod
    def _process_local_path(media_url, ext, input_type, cache_dir):
        if not os.path.exists(media_url):
            logger.error("Can not find the input media file.")
            raise FileNotFoundError("Can not find the input media file.")

        path_pattern = re.compile(
            r"^(\/(?:[\w\-\.]+\/)*[\w\-\.]*\/?)?$|^(?:[\w\-\.]+\/)*[\w\-\.]*\/?$"
        )
        if not path_pattern.fullmatch(media_url):
            msg = (f'The media url contains dangerous characters, '
               f'only allow a-zA-Z0-9_-.')
            logger.error(msg)
            raise ValueError(msg)
        
        file_utils.check_path_permission(media_url, mode=0o640)
        media_size = os.path.getsize(media_url)
        if media_size > _SIZE_LIMITS.get(input_type):
            logger.error(f'The size of {input_type} cannot exceed '
                         f'{_SIZE_LIMITS.get(input_type) / (1024 * 1024)} MB')
            raise ValueError(f'The size of {input_type} cannot exceed '
                             f'{_SIZE_LIMITS.get(input_type) / (1024 * 1024)} MB')
        io_utils.copy_media(media_url, cache_dir, ext)
        return media_size

    @staticmethod
    def _get_cache_base_path(child_dir_name):
        dir_path = os.getenv("LOCAL_CACHE_DIR", None)
        if dir_path is None:
            dir_path = os.path.expanduser("~/mindie/cache")
            os.makedirs(dir_path, exist_ok=True)
            os.chmod(dir_path, 0o750)
        else:
            dir_path = file_utils.standardize_path(dir_path)
            file_utils.check_path_permission(dir_path)
            if not os.access(dir_path, os.R_OK | os.W_OK):
                logger.error(f'Check {dir_path} failed: the current user does not have read and write permissions. ' +
                            'Please update the permissions or export LOCAL_CACHE_DIR to a correct directory path.')
                raise ValueError(
                    f'Check {dir_path} failed: the current user does not have read and write permissions. ' +
                     'Please update the permissions or export LOCAL_CACHE_DIR to a correct directory path.')
        dir_path = os.path.join(dir_path, child_dir_name)
        file_utils.check_path_length_lt(dir_path)
        return dir_path

    @staticmethod
    def _extract_domain(url: str) -> str:
        """
        Extract normalized domain (hostname) from a URL.
        """
        try:
            parsed = parse_url(url)
            if not parsed.scheme or not parsed.host:
                raise ValueError(f"Invalid URL: {url}")
            return parsed.host.lower()
        except LocationParseError as e:
            logger.error(f"Invalid URL: {url}")
            raise ValueError(f"Invalid URL: {url}") from e
        except Exception as e:
            logger.error(f"Invalid URL: {url}")
            raise ValueError(f"Invalid URL: {url}") from e

    @staticmethod
    def _load_allowed_media_domains() -> set[str]:
        """
        Load allowed media domains from environment variable.

        Example:
        ALLOWED_MEDIA_DOMAINS_ENV="upload.xxxmedia.org, cxxx.xxx.com"

        """
        allowed_media_domains_env = os.getenv(_ALLOWED_MEDIA_DOMAINS_ENV)
        if not allowed_media_domains_env:
            return set()

        return {d.strip().lower() for d in allowed_media_domains_env.split(",") if d.strip()}

    @staticmethod
    def _check_domain_allowed(media_url: str, allowed_domains: set[str]) -> None:
        """
        Validate whether media_url's domain is allowed.
        """
        domain = IbisTokenizer._extract_domain(media_url)
        if domain not in allowed_domains:
            raise ValueError(
                f"Domain '{domain}' is not in allowed domain list"
            )

    def delete_multimodal_cache(self, timestamp: int, cache_prefix=None):
        dir_path = self.cache_path
        if cache_prefix is not None:
            dir_path = os.path.join(dir_path, cache_prefix + f'{timestamp}')
            if os.path.exists(dir_path):
                dir_path = file_utils.standardize_path(dir_path)
                file_utils.check_path_permission(dir_path)
                io_utils.remove_cache_dir(dir_path)
            return

        dir_path = file_utils.standardize_path(dir_path)
        file_utils.check_path_permission(dir_path)
        child_dirs = os.listdir(dir_path)
        for elem_dir in child_dirs:
            if elem_dir.endswith(str(timestamp)):
                release_path = os.path.join(dir_path, elem_dir)
                io_utils.remove_cache_dir(release_path)
                return

    def release_cache(self, duration):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        signal.signal(signal.SIGABRT, signal.SIG_DFL)

        dir_path = self.cache_path
        dir_path = file_utils.standardize_path(dir_path)
        file_utils.check_path_permission(dir_path)
        while True:
            time.sleep(duration)
            all_request = os.listdir(dir_path)
            for request in all_request:
                split_text = request.split('_')
                if len(split_text) != 2 or len(split_text[0]) < 1 or len(split_text[1]) < 1:
                    continue
                req_time = int(split_text[1])
                if time.time() - (req_time / 1_000_000_000) > _DELET_DURATION:
                    self.delete_multimodal_cache(req_time, split_text[0] + '_')

    def download_url(self, prompt: str, timestamp: int):
        def download_elems(elem_list):
            media_size = 0
            total_start_time = time.time()
            for elem in elem_list:
                media_size += self._download(elem, total_start_time)
                if media_size > _MEDIA_SIZE_LIMIT:
                    err_str = f'The total media input cannot exceed {_MEDIA_SIZE_LIMIT / (1024 * 1024)} MB.'
                    logger.error(err_str)
                    raise ValueError(err_str)

        if not self.is_mm(prompt):
            return
        self.timestamp = timestamp
        self._creat_cache_dir(timestamp)
        prompt_obj = json.loads(prompt)

        try:
            if _CONTENT_NAME_KEY not in prompt_obj[0]:
                download_elems(prompt_obj)
                return
            for single in prompt_obj:
                if isinstance(single.get(_CONTENT_NAME_KEY), list):
                    download_elems(single.get(_CONTENT_NAME_KEY))
        except ValueError as value_error:
            logger.error("Download url failed: %s", value_error)
            self.delete_multimodal_cache(timestamp, self.cache_prefix)
            raise value_error
        except Exception as e:
            logger.error("Download url failed: %s", e)
            self.delete_multimodal_cache(timestamp, self.cache_prefix)
            raise e

    def encode(self, prompt):
        try:
            if self.is_mm(prompt):
                mm_inputs = self.process_mm_inputs(prompt)
                token_list = self.tokenize(mm_inputs)
                self._clear_media_cache(self.timestamp)
                if isinstance(token_list, np.ndarray):
                    token_list = token_list.tolist()
                else:
                    token_list = token_list.numpy().tolist()
            else:
                token_list = self.wrapper_encode([prompt], return_tensors="np")

            for elem in token_list:
                if isinstance(elem, list):
                    logger.error("[Tokenizer encode]\t>>> tokenizer encode result is not 1 Dimension")
                    return elem
            return token_list
        except ValueError as value_error:
            self.delete_multimodal_cache(self.timestamp, self.cache_prefix)
            logger.error(f'IbisTokenizer encode error: {value_error}')
            raise RuntimeError(f'IbisTokenizer encode error: {value_error}') from value_error
        except Exception as e:
            self.delete_multimodal_cache(self.timestamp, self.cache_prefix)
            logger.error(f'IbisTokenizer encode error: {e}')
            raise RuntimeError(f'IbisTokenizer encode error: {e}') from e

    def encode_chat(self, prompt, chat_template_kwargs: dict):
        try:
            if self.is_mm(prompt):
                inputs = self.process_mm_inputs(prompt)
                token_list = self.input_builder.make_context(0, inputs, chat_template_kwargs=chat_template_kwargs)
                self._clear_media_cache(self.timestamp)
            else:
                inputs = json.loads(prompt)
                token_list = self.wrapper_encode(inputs, chat_template_kwargs=chat_template_kwargs, is_chatting=True)

            return token_list
        except ValueError as value_error:
            self.delete_multimodal_cache(self.timestamp, self.cache_prefix)
            logger.warning("[Tokenizer]\t>>> Exception:%s", value_error)
            raise RuntimeError(f"[Tokenizer] encode failed. Original error: {value_error}") from value_error
        except Exception as e:
            self.delete_multimodal_cache(self.timestamp, self.cache_prefix)
            logger.warning("[Tokenizer]\t>>> Exception:%s", e)
            raise RuntimeError(f"[Tokenizer] encode failed. Original error: {e}") from e

    def decode(self, all_token_ids: List[int], kwargs: dict = None):
        """
        decode all token ids
        Args:
            all_token_ids (list): The list of tokens to decode.
            kwargs (dict): additional parameters

        Returns:
            JSON object
        """
        output_content = dict()
        use_tool_call = kwargs.get("use_tool_call", False)
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        is_chat_req = kwargs.get("is_chat_req", False)
        tool_calls_json = kwargs.get("tool_calls_json", None)
        try:
            tools = json.loads(tool_calls_json) if tool_calls_json is not None else None
        except Exception as e:
            logger.debug('Decode tokens failed to load tools json of request: %s', e)
            tools = None

        is_stream = False
        if len(all_token_ids) > 0 and all_token_ids[-1] == -1:
            all_token_ids = all_token_ids[:-1]
        if len(all_token_ids) <= 0:
            return json.dumps(output_content)

        input_kwargs = dict()
        meta_data = dict()
        req_enable_reasoning_label = "req_enable_thinking"
        try:
            if req_enable_reasoning_label in kwargs:
                meta_data.update({req_enable_reasoning_label: kwargs[req_enable_reasoning_label]})
            meta_data.update({"reasoning_tokens": kwargs.get("reasoning_tokens", -1), "tools": tools})
            input_kwargs.update({"metadata": meta_data})
            output_content = self.wrapper_decode(
                all_token_ids, skip_special_tokens, use_tool_call, is_chat_req, is_stream, **input_kwargs)
        except ValueError as e:
            strace = traceback.format_exc()
            logger.error(f'Decode tokens failed and the reason is ValueError: {e}')
            logger.error(strace)
        except Exception as e:
            strace = traceback.format_exc()
            logger.error(f'Decode tokens failed and the reason: {e}')
            logger.error(strace)

        return json.dumps(output_content)

    def decode_one(self, all_token_ids: List[int], kwargs: dict = None):
        """
        decode incremental token ids
        Args:
            all_token_ids (list): The list of tokens to decode.
            kwargs (dict): additional parameters

        Returns:
            JSON object
        """
        output_content = dict()
        use_tool_call = kwargs.get("use_tool_call", False)
        skip_special_tokens = kwargs.get("skip_special_tokens", False)
        is_chat_req = kwargs.get("is_chat_req", False)
        pre_index = kwargs.get("prev_decode_index", -1)
        current_index = kwargs.get("curr_decode_index", -1)
        tool_calls_json = kwargs.get("tool_calls_json", None)
        try:
            tools = json.loads(tool_calls_json) if tool_calls_json is not None else None
        except Exception as e:
            logger.debug('Decode tokens failed to load tools json of request %s', e)
            tools = None

        is_stream = True
        if pre_index < 0 or pre_index > len(all_token_ids):
            logger.error("Parameter pre_index %d is invalid.", pre_index)
            raise ValueError("pre_index is invalid.")
        if current_index < 0 or current_index > len(all_token_ids):
            logger.error("Parameter current_index %d is invalid.", current_index)
            raise ValueError("Parameter current_index is invalid.")
        if all_token_ids[-1] == -1:
            return json.dumps(output_content)

        input_kwargs = dict()
        meta_data = dict()
        req_enable_reasoning_label = "req_enable_thinking"
        try:
            if req_enable_reasoning_label in kwargs:
                meta_data.update({req_enable_reasoning_label: kwargs[req_enable_reasoning_label]})

            meta_data.update({"current_tool_name_sent": kwargs.get("current_tool_name_sent", False),
                              "current_tool_arguments_sent": kwargs.get("current_tool_arguments_sent", False),
                              "current_tool_id": kwargs.get("current_tool_id", -1),
                              "reasoning_tokens": kwargs.get("reasoning_tokens", -1),
                              "tools": tools
                              })
            input_kwargs.update({"prev_decode_index": pre_index, "curr_decode_index": current_index})
            input_kwargs.update({"metadata": meta_data})

            output_content = self.wrapper_decode(
                all_token_ids, skip_special_tokens, use_tool_call, is_chat_req, is_stream, **input_kwargs)
        except ValueError as e:
            strace = traceback.format_exc()
            logger.error(f'Decode one token failed and the reason is ValueError: {e}')
            logger.error(strace)
        except Exception as e:
            strace = traceback.format_exc()
            logger.error(f'Decode one token failed and the reason: {e}')
            logger.error(strace)

        return json.dumps(output_content)

    def process_mm_inputs(self, prompt):
        logger.info("Process multimodal inputs.")

        if isinstance(prompt, str):
            prompt_obj = json.loads(prompt)
        else:
            prompt_obj = prompt

        images = sorted(os.listdir(self.media_cache_dirs.get(_IMAGE_KEY)))
        videos = sorted(os.listdir(self.media_cache_dirs.get(_VIDEO_KEY)))
        audios = sorted(os.listdir(self.media_cache_dirs.get(_AUDIO_KEY)))
        medias = {
            _IMAGE_KEY: images,
            _VIDEO_KEY: videos,
            _AUDIO_KEY: audios
        }
        media_idx = {
            _IMAGE_KEY: 0,
            _VIDEO_KEY: 0,
            _AUDIO_KEY: 0
        }

        mm_inputs = []
        if _ROLE_KEY not in prompt_obj[0]:
            media_idx, mm_inputs = self._process_single_input(prompt_obj, medias, media_idx)
        else:
            for i, message in enumerate(prompt_obj):
                if _CONTENT_NAME_KEY in message:
                    media_idx, single_content = self._process_single_input(message[_CONTENT_NAME_KEY], medias,
                                                                           media_idx)
                    prompt_obj[i][_CONTENT_NAME_KEY] = single_content
            mm_inputs = prompt_obj

        return mm_inputs

    def finalize(self):
        del self.tokenizer
        return 0

    def _creat_cache_dir(self, timestamp):
        dir_path = os.path.join(self.cache_path, self.cache_prefix + f"{timestamp}")

        io_utils.create_cache_dir(dir_path)
        self.media_cache_dirs = {
            _IMAGE_KEY: os.path.join(dir_path, "image"),
            _VIDEO_KEY: os.path.join(dir_path, "video"),
            _AUDIO_KEY: os.path.join(dir_path, "audio"),
        }

    def _download(self, info, total_start_time):
        media_size = 0
        input_type = info['type']
        if input_type == 'text':
            return media_size
        elif input_type not in _MEDIA_TYPE:
            logger.error(f"Input type {input_type} does not match the mm type.")
            raise ValueError(f"Input type {input_type} does not match the mm type.")

        media_url = ""
        if isinstance(info[input_type], str):
            media_url: str = info[input_type]
        elif isinstance(info[input_type], dict):
            if input_type != _INPUT_AUDIO_KEY:
                media_url: str = info[input_type].get("url")
                if input_type == _IMAGE_KEY and info[input_type].get("detail", "auto") != "auto":
                    logger.warning(f"{_IMAGE_KEY}.detail is currently not support now, it will be ignored.")
        else:
            logger.error(f"Input of {input_type} should be str or dict.")
            raise ValueError(f"Input of {input_type} should be str or dict.")

        ext = io_utils.extract_extension_from_url(media_url)
        cache_dir = self.media_cache_dirs.get(input_type)

        if media_url.startswith("http://") or media_url.startswith("https://"):  # http or https
            if len(media_url) > _URL_LENGTH_LIMIT:
                logger.error(f"The length of media_url should be less than {_URL_LENGTH_LIMIT}, "
                             f"but got {len(media_url)}.")
                raise ValueError(f"The length of media_url should be less than {_URL_LENGTH_LIMIT}, "
                                 f"but got {len(media_url)}.")
     
            allowed_domains = self._load_allowed_media_domains()
            if allowed_domains:
                try:
                    self._check_domain_allowed(media_url, allowed_domains)
                except ValueError as e:
                    logger.error("Domain whitelist validation failed: %s", e)
                    raise ValueError(
                        f"The media URL domain is not allowed: {e}"
                    ) from e
            else:
                if not once_flag.is_set():
                    with threading.Lock():
                        if not once_flag.is_set():
                            logger.warning(
                                "ALLOWED_MEDIA_DOMAIN_ENV is not set. "
                                "Domain whitelist check is disabled."
                            )
                            once_flag.set()
            
            media_size = self._process_url_path(media_url, ext, input_type, cache_dir, total_start_time)
        elif ext.lower() in _MEDIA_TYPE.get(input_type):  # local path
            if media_url.startswith("file://"):
                _, media_url = media_url.split("file://", 1)

            allowed_real_base = os.path.realpath(_ALLOWED_LOCAL_MEDIA_PATH)
            media_url = file_utils.standardize_path(media_url)

            # Ensure the file is inside the allowlist directory 
            if not media_url.startswith(allowed_real_base + os.sep): 
                logger.error(f"Your input local file path is not allowed!"
                  f"please ensure your multimedia files are placed under {_ALLOWED_LOCAL_MEDIA_PATH}") 
                raise ValueError(f"Your input local file path is not allowed!"
                  f"please ensure your multimedia files are placed under {_ALLOWED_LOCAL_MEDIA_PATH}")
                  
            media_size = self._process_local_path(media_url, ext, input_type, cache_dir)
        else:
            media_size = self._process_base64(info, media_url, input_type)

        return media_size

    def _process_base64(self, info, media_url, input_type):

        if media_url.startswith("data:"):  # {"xxx_url": "data:{MIME};base64,{base64_encoded_data}"}
            pattern = r"data:(image|video|audio)/" \
                      r"((?:jpeg|png|mp4|x-msvideo|x-ms-wmv|mpeg|x-wav|flac));base64,([A-Za-z0-9+/=]+)"
            match = re.search(pattern, media_url)
            cache_dir = self.media_cache_dirs.get(input_type)
            if match:
                mime_type = match.group(2)
                base64_data = match.group(3)
                ext = _MIME_TYPE2EXT.get(mime_type)
            else:
                logger.error("Expected format: data:<mime_type>/<subtype>;base64,<base64_data>")
                raise ValueError("Expected format: data:<mime_type>/<subtype>;base64,<base64_data>")
        elif input_type == _INPUT_AUDIO_KEY:  # {"data": {base64_encoded_data}, "format": {ext}}
            base64_data = info[input_type].get("data", None)
            ext = "." + info[input_type].get("format", "")
            cache_dir = self.media_cache_dirs.get(_AUDIO_KEY)
            if not base64_data or ext.lower() not in _MEDIA_TYPE.get(input_type):
                logger.error("'input_audio.data' should be base64 encoded data, "
                             "and the 'input_audio.format' should be in ['mp3', 'wav', 'flac'].")
                raise ValueError("'input_audio.data' should be base64 encoded data, "
                                 "and the 'input_audio.format' should be in ['mp3', 'wav', 'flac'].")
            input_type = _AUDIO_KEY
        elif input_type == _IMAGE_KEY:  # {"image_url": {base64_encoded_data}}
            base64_data = media_url
            ext = ".jpg"
            cache_dir = self.media_cache_dirs.get(input_type)
        else:
            logger.error(f"Input type {input_type} does not match the mm type.")
            raise ValueError(f"Input type {input_type} does not match the mm type.")

        data_content = io_utils.decode_base64_content(base64_data)
        media_size = len(data_content)
        if media_size > _SIZE_LIMITS.get(input_type):
            logger.error(f'The size of {input_type} cannot exceed '
                         f'{_SIZE_LIMITS.get(input_type) / (1024 * 1024)} MB')
            raise ValueError(f'The size of {input_type} cannot exceed '
                            f'{_SIZE_LIMITS.get(input_type) / (1024 * 1024)} MB')
        cache_dir = file_utils.standardize_path(cache_dir)
        file_utils.check_path_permission(cache_dir)
        data_count = len(os.listdir(cache_dir))
        if input_type == _IMAGE_KEY:
            data_save_path = os.path.join(cache_dir, f"{data_count + 1}{ext}")
            io_utils.save_image(data_content, data_save_path)
        else:
            io_utils.save_media(data_content, cache_dir, ext)
        return media_size

    def _process_single_input(self, prompt_obj, medias, media_idx):
        if isinstance(prompt_obj, str):
            return media_idx, [{"text": prompt_obj}]

        media_keys = {
            _IMAGE_KEY: "image",
            _VIDEO_KEY: "video",
            _AUDIO_KEY: "audio"
        }
        mm_inputs = []
        for elem in prompt_obj:
            input_type = _AUDIO_KEY if elem['type'] == _INPUT_AUDIO_KEY else elem['type']
            path_idx = media_idx.get(input_type)
            media_list = medias.get(input_type)
            if input_type == _TEXT_KEY:
                mm_inputs.append({_TEXT_KEY: elem[_TEXT_KEY]})
            elif input_type in media_keys.keys():
                if path_idx > len(media_list):
                    logger.error(f"Requested {path_idx} {input_type} \
                        but only downloaded {len(media_list)}")
                    raise ValueError(f"Requested {path_idx} {input_type} \
                        but only downloaded {len(media_list)}")
                media_path = os.path.join(self.media_cache_dirs.get(input_type),
                    media_list[path_idx])
                IbisTokenizer.check_path(media_path)
                mm_inputs.append({media_keys[input_type]: media_path})
                media_idx[input_type] += 1
            else:
                logger.error(f"Input type: {input_type} does not match the mm type.")
                raise ValueError(f"Input type: {input_type} does not match the mm type.")
        return media_idx, mm_inputs

    def _clear_media_cache(self, timestamp: int):
        dir_path = os.path.join(self.cache_path, self.cache_prefix + f"{timestamp}")
        io_utils.clear_meida_cache(dir_path)