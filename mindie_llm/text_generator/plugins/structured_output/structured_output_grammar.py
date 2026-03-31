# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from mindie_llm.utils.log.logging import logger


class StructuredOutputType(str, Enum):
    JSON_OBJECT = "json_object"  # 通用 JSON 对象约束
    JSON_SCHEMA = "json_schema"  # 用户指定 JSON Schema 约束


@dataclass
class StructuredOutputRequest:
    output_type: StructuredOutputType  # 结构化输出类型
    grammar_spec: str  # JSON Schema 字符串
    
    # 编译后的 grammar（延迟填充）
    grammar: Optional['StructuredOutputGrammar'] = field(default=None, repr=False)
    
    @classmethod
    def from_response_format(
        cls,
        response_format: str
    ) -> Optional['StructuredOutputRequest']:
        import json
        
        if response_format is None:
            return None
        
        # 解析 JSON 字符串
        try:
            response_format_dict = json.loads(response_format)
        except json.JSONDecodeError:
            logger.warning(f"Invalid response_format JSON: {response_format}")
            return None
        
        format_type = response_format_dict.get("type")
        
        if format_type is None:
            logger.warning("response_format missing 'type' field")
            return None
        
        if format_type == "json_object":
            # 通用 JSON 对象约束
            return cls(
                output_type=StructuredOutputType.JSON_OBJECT,
                grammar_spec='{"type": "object"}'
            )
        elif format_type == "json_schema":
            # 用户指定 JSON Schema；name 在 json_schema 对象内，如 { "json_schema": { "name": "xxx", "schema": {} } }
            json_schema = response_format_dict.get("json_schema")
            if not isinstance(json_schema, dict):
                logger.warning(
                    "response_format for type 'json_schema' must contain a 'json_schema' object"
                )
                return None
            schema_name = json_schema.get("name")
            if not isinstance(schema_name, str) or not schema_name:
                logger.warning(
                    "response_format.json_schema must contain a non-empty 'name' field"
                )
                return None

            schema = json_schema.get("schema", json_schema)
            if isinstance(schema, dict):
                schema = json.dumps(schema)
            return cls(
                output_type=StructuredOutputType.JSON_SCHEMA,
                grammar_spec=schema
            )
        else:
            logger.warning(f"Unsupported response_format type: {format_type}. "
                          f"Supported types: {[t.value for t in StructuredOutputType]}")
            return None


class StructuredOutputGrammar(ABC):
    
    @property
    @abstractmethod
    def num_processed_tokens(self) -> int:
        """已被 FSM 合法接受的 token 数量（不含被拒绝的 token）"""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_tried_tokens(self) -> int:
        """在 replay buffer 中已尝试过的 token 数量（含被拒绝的），用作 replay buffer 游标"""
        raise NotImplementedError

    @abstractmethod
    def accept_tokens(self, state_key: int, tokens: List[int]) -> bool:
        """
        接受 token 并推进 FSM 状态
        
        Args:
            state_key: 当前运行态状态 key（用于日志）
            tokens: 要接受的 token 列表
            
        Returns:
            是否成功接受所有 token（如果某个 token 不被当前状态接受，返回 False）
        """
        raise NotImplementedError
    
    @abstractmethod
    def fill_bitmask(self, bitmask: np.ndarray, idx: int) -> None:
        """
        根据当前 FSM 状态填充允许的 token bitmask
        
        Args:
            bitmask: [batch_size, vocab_size // 32] 的 int32 数组
                     每个 bit 表示一个 token 是否允许
            idx: 当前请求在 batch 中的索引
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_terminated(self) -> bool:
        """
        检查是否到达接受状态
        
        Returns:
            是否已完成生成（FSM 到达接受状态）
        """
        raise NotImplementedError
    

class XgrammarGrammar(StructuredOutputGrammar):
    def __init__(
        self,
        matcher: Any,  # xgr.GrammarMatcher
        vocab_size: int,
        ctx: Any,  # xgr.CompiledGrammar
    ):
        """
        初始化 Grammar
        
        Args:
            matcher: xgrammar 的 GrammarMatcher 实例
            vocab_size: 词表大小
            ctx: 编译后的 Grammar 上下文
        """
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self._num_processed_tokens = 0
        self._num_tried_tokens = 0
        self._is_terminated = False
    
    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def num_tried_tokens(self) -> int:
        return self._num_tried_tokens

    def accept_tokens(self, state_key: int, tokens: List[int]) -> bool:
        """接受 token 并推进 FSM 状态"""
        if self._is_terminated:
            return True

        for token in tokens:
            # 无论 accept/reject，先推进 replay 游标，保证 num_tried_tokens 始终对齐
            # replay buffer 中的实际位置（含被 C++ 无条件存储的 rejected token）
            self._num_tried_tokens += 1
            accepted = self.matcher.accept_token(token)
            if not accepted:
                logger.warning(
                    f"[StructuredOutput] Token {token} rejected for state_key {state_key}"
                )
                logger.debug(
                    "[StructuredOutput][Accept] layer=grammar state_key=%s reject_token=%s "
                    "tried=%s accepted=%s (matcher rejected, replay cursor still advanced)",
                    state_key,
                    token,
                    self._num_tried_tokens,
                    self._num_processed_tokens,
                )
                return False

            self._num_processed_tokens += 1

            matcher_terminated = self.matcher.is_terminated()

            if matcher_terminated:
                self._is_terminated = True
                break
        
        return True
    
    def fill_bitmask(self, bitmask: np.ndarray, idx: int) -> None:
        """填充 bitmask"""
        if self._is_terminated:
            bitmask[idx, :] = -1
            return

        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        """检查是否终止"""
        return self._is_terminated