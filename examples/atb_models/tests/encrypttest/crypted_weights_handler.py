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

import importlib
from typing import Any

import safetensors

from mindie_llm.runtime.utils.loader.weight_utils import WeightsFileHandler
from mindie_llm.utils.log.logging import logger


class CryptedWeightsFileHandler(WeightsFileHandler):
    def __init__(self, model_path: str, extension: str, quantize: str = None):
        super().__init__(model_path, extension)

        self._sf_metadata = {}
        self._decrypt_ins = None
        self._model_path = model_path

        if self.encrypt_enable:
            try:
                decrypt_script = importlib.import_module("tests.encrypttest.custom_crypt")
                decrypt_cls = getattr(decrypt_script, "CustomDecrypt")
                self._decrypt_ins = decrypt_cls()

                for filename in self._filenames:
                    with safetensors.torch.safe_open(filename, framework="pt") as f:
                        self._sf_metadata.update(f.metadata())

            except Exception as e:
                logger.warning(f"Failed to initialize decryptor: {e}, proceeding without decryption")

    @property
    def encrypt_enable(self) -> bool:
        return self._model_path.endswith("crypt") or self._model_path.endswith("crypt/")

    def get_tensor(self, tensor_name: str) -> Any:
        tensor = super().get_tensor(tensor_name)

        if self.encrypt_enable and self._decrypt_ins:
            try:
                tensor = self._decrypt_ins.decrypt(tensor)
                if tensor_name in self._sf_metadata:
                    module_name, attribute_name = self._sf_metadata[tensor_name].split(".")
                    module = importlib.import_module(module_name)
                    dtype_ = getattr(module, attribute_name)
                    tensor = tensor.to(dtype_)
            except Exception as e:
                logger.warning(f"Failed to decrypt tensor '{tensor_name}': {e}, using original weight")

        return tensor
