# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import json
from typing import List, Any, Tuple
from pathlib import Path

import torch
import transformers
import safetensors

from mindie_llm.utils.log.logging import logger
from mindie_llm.runtime.utils.helpers.safety.file import safe_open
from mindie_llm.utils.file_utils import standardize_path, check_path_permission


class WeightsFileHandler:
    def __init__(self, model_path: str, extension: str):
        self._handlers = {}
        self._filenames = self._get_weight_filenames(model_path, extension)
        self._routing = self._load_weight_file_routing()

    @property
    def extension(self) -> str:
        return ".safetensors"
    
    @classmethod
    def _get_weight_filenames(cls,
                              model_weight_path: str, extension: str,
                              index_file_name: str = transformers.utils.SAFE_WEIGHTS_INDEX_NAME) -> List[Path]:
        """Get the local files"""
        if Path(model_weight_path).exists() and Path(model_weight_path).is_dir():
            local_files = list(Path(model_weight_path).glob(f"*{extension}"))
            if not local_files:
                raise FileNotFoundError(
                    f"No local weights found with extension {extension};"
                    f"Only safetensor format is supported. Please refer to model's README for more details."
                )

            # Filter file names by index file
            index_file_path = os.path.join(model_weight_path, index_file_name)
            if not os.path.isfile(index_file_path):
                return [str(file) for file in local_files]

            with safe_open(index_file_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            file_names_from_index_file = set(weight_map.values())

            filtered_file_path = []
            for file in local_files:
                if file.name in file_names_from_index_file:
                    filtered_file_path.append(str(file))
                else:
                    logger.info(f"{str(file)} is filtered by index_file {index_file_path}.")
            return filtered_file_path

        raise FileNotFoundError("The input model id is not exists or not a directory")
    
    def release_file_handler(self) -> None:
        """Release all file handlers"""
        if self._handlers:
            del self._handlers
            self._handlers = {}


    def get_tensor(self, tensor_name: str) -> Any:
        """Get tensor by full name."""
        filename, tensor_name = self._get_filename(tensor_name)
        f = self._get_handler(filename)
        tensor = f.get_tensor(tensor_name)
        return tensor

    def _get_handler(self, filename: str) -> Any:
        """Get file handler by filename."""
        if filename not in self._handlers:
            # Note: manually call the release_file_handler method after use.
            f = safetensors.safe_open(filename, framework="pytorch")
            self._handlers[filename] = f
            return f
        return self._handlers[filename]
  
    def _get_filename(self, tensor_name: str) -> Tuple[str, str]:
        """Get file name for tensor name."""
        filename = self._routing.get(tensor_name)
        if filename is None:
            raise ValueError(f"Weight file was not found for tensor named with {tensor_name}.")
        return str(filename), tensor_name

    def _load_weight_file_routing(self) -> dict:
        """Build routing of weight files."""
        routing = {}
        for filename in self._filenames:
            filename = standardize_path(str(filename), check_link=False)
            check_path_permission(filename)
            with safetensors.safe_open(filename, framework="pt") as f:
                for k in f.keys():
                    if k in routing:
                        raise ValueError("Weight was found in multiple files.")
                    routing[k] = filename
        return routing
