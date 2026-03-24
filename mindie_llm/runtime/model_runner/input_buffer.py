# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Dict, List
import torch
from mindie_llm.runtime.utils.singleton import Singleton
from mindie_llm.utils.log.logging import logger


class InputBuffer(Singleton):
    """Singleton buffer for storing input tensors.
    
    This class provides a centralized storage for input tensors used during
    model forward passes, particularly for graph mode execution.
    """
    
    def __init__(self) -> None:
        """Initialize the input buffer."""
        if hasattr(self, "_initialized"):
            return

        self._buffers: Dict[str, torch.Tensor] = {}
        self._initialized = True
    
    def get(self, key: str) -> torch.Tensor:
        """Get a tensor from the buffer.
        
        Args:
            key: Key identifying the tensor.
            
        Returns:
            The tensor associated with the key.
            
        Raises:
            KeyError: If the key is not found.
        """
        return self._buffers[key]

    def register(self, key: str, tensor: torch.Tensor) -> None:
        """Register a tensor in the buffer.
        
        Args:
            key: Key to identify the tensor.
            tensor: Tensor to register.
            
        Raises:
            KeyError: If the key is already registered.
        """
        # When enable MTP, both main and draft models share the same input buffer.
        # Therefore, we need to skip the registration if the key already exists.
        if key in self._buffers:
            logger.warning(f"Buffer '{key}' already registered, skip to update it.")
            return

        self._buffers[key] = tensor

    def _record_input_buffer_addresses(self) -> List[int]:
        """Record memory addresses of all tensors in the buffer.
        
        Returns:
            List of memory addresses (data pointers) of tensors.
        """
        addresses = []
        for value in self._buffers.values():
            if torch.is_tensor(value):
                addresses.append(value.data_ptr())
        return addresses


input_buffer = InputBuffer()
