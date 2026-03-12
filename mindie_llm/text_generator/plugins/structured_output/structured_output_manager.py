# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mindie_llm.utils.log.logging import logger

from .structured_output_grammar import (
    StructuredOutputGrammar,
    StructuredOutputRequest,
    StructuredOutputType,
    XgrammarGrammar,
)


# 模块级缓存：延迟导入 xgrammar 模块（如果可用）
_xgrammar_module = None
_xgrammar_import_attempted = False


def _get_xgrammar_module():
    """
    获取 xgrammar 模块（如果可用）
    
    Returns:
        xgrammar 模块，如果不可用则返回 None
    """
    global _xgrammar_module, _xgrammar_import_attempted
    
    # 如果已经尝试过导入，直接返回缓存的结果
    if _xgrammar_import_attempted:
        return _xgrammar_module
    
    # 尝试导入
    logger.debug("Attempting to import xgrammar module")
    _xgrammar_import_attempted = True
    try:
        import xgrammar as xgr
        _xgrammar_module = xgr
        logger.debug("Successfully imported xgrammar module")
        return xgr
    except ImportError:
        _xgrammar_module = None
        logger.warning("Failed to import xgrammar module, falling back to Python implementation")
        return None


class GuidedDecodingBackendType(str, Enum):
    """约束解码后端类型"""
    XGRAMMAR = "xgrammar"

_DEFAULT_BITMASK_PREALLOC_BATCH = 64
_DEFAULT_GRAMMAR_CACHE_SIZE = 100


@dataclass
class StructuredOutputConfig:
    """结构化输出配置"""
    backend: GuidedDecodingBackendType = GuidedDecodingBackendType.XGRAMMAR
    xgrammar_any_whitespace: bool = True
    grammar_cache_size: int = _DEFAULT_GRAMMAR_CACHE_SIZE
    bitmask_prealloc_batch: int = _DEFAULT_BITMASK_PREALLOC_BATCH


class GrammarBackend:
    """
    Grammar 后端封装
    
    负责：
    1. 初始化后端库（xgrammar）
    2. 编译 JSON Schema → Grammar
    3. 创建 GrammarMatcher（每个请求独立的 FSM 状态）
    """
    
    def __init__(
        self,
        backend_type: GuidedDecodingBackendType,
        tokenizer: Any,
        vocab_size: int,
        config: StructuredOutputConfig,
    ):
        """
        初始化后端
        
        Args:
            backend_type: 后端类型
            tokenizer: HuggingFace tokenizer
            vocab_size: 词表大小
            config: 配置
        """
        self.backend_type = backend_type
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.config = config
        
        # 后端特定的对象
        self._xgr = None
        self._xgr_tokenizer_info = None
        self._xgr_compiler = None
        
        # 初始化后端
        logger.debug(
            f"[StructuredOutput|Backend|Diagnose] "
            f"GrammarBackend.__init__() called, about to call _init_backend()"
        )
        self._init_backend()
    
    @staticmethod
    def create_grammar(compiled: 'CompiledGrammar') -> StructuredOutputGrammar:
        """
        从编译后的 Grammar 创建状态追踪器
        
        Args:
            compiled: 编译后的 Grammar
            
        Returns:
            StructuredOutputGrammar 实例（带独立状态）
        """
        matcher = compiled.xgr_module.GrammarMatcher(compiled.ctx)
        result = XgrammarGrammar(
            matcher=matcher,
            vocab_size=compiled.vocab_size,
            ctx=compiled.ctx,
        )
        return result

    def compile_grammar(
        self,
        output_type: StructuredOutputType,
        grammar_spec: str,
    ) -> 'CompiledGrammar':
        """
        编译 Grammar
        
        Args:
            output_type: 请求类型（json_object / json_schema）
            grammar_spec: Grammar 规范字符串
            
        Returns:
            CompiledGrammar 对象
        """
        return self._compile_xgrammar(output_type, grammar_spec)
    
    def _init_backend(self) -> None:
        """初始化后端库"""
        self._init_xgrammar()
    
    def _init_xgrammar(self) -> None:
        """初始化 xgrammar 后端"""
        logger.debug("[StructuredOutput|Backend] Initializing xgrammar backend...")
        
        xgr = _get_xgrammar_module()
        if xgr is None:
            logger.error(
                f"[StructuredOutput|Backend] "
                f"_get_xgrammar_module() returned None, xgrammar is not installed!"
            )
            raise ImportError(
                "xgrammar is not installed. Please install it with: pip install xgrammar"
            )
        self._xgr = xgr
        logger.debug(
            f"[StructuredOutput|Backend] xgrammar module loaded (version: {getattr(xgr, '__version__', 'unknown')})"
        )
        
        # 创建 TokenizerInfo
        try:
            logger.debug("[StructuredOutput|Backend] Creating TokenizerInfo from HuggingFace tokenizer...")
            self._xgr_tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer, vocab_size=self.vocab_size
            )
            logger.debug("[StructuredOutput|Backend] TokenizerInfo created from HuggingFace")
        except Exception as e:
            logger.error(
                f"[StructuredOutput|Backend] Failed to create TokenizerInfo from HuggingFace: {e}. "
                "Manual TokenizerInfo fallback is not used to avoid inconsistent behavior."
            )
            raise RuntimeError(
                "Cannot initialize xgrammar: TokenizerInfo.from_huggingface failed. "
                "Ensure the tokenizer is compatible with xgrammar."
            ) from e
        
        # 创建 GrammarCompiler（可复用）
        logger.debug("[StructuredOutput|Backend] Creating GrammarCompiler...")
        self._xgr_compiler = xgr.GrammarCompiler(self._xgr_tokenizer_info)
        
        logger.debug(
            f"[StructuredOutput|Backend] xgrammar backend initialized: "
            f"vocab_size={self.vocab_size}"
        )
    
    def _compile_xgrammar(
        self,
        output_type: StructuredOutputType,
        grammar_spec: str,
    ) -> 'CompiledGrammar':
        """使用 xgrammar 编译"""
        xgr = self._xgr
        
        if output_type in (StructuredOutputType.JSON_SCHEMA, StructuredOutputType.JSON_OBJECT):
            # JSON Schema → CompiledGrammar
            logger.debug(
                f"[StructuredOutput|XGrammar] "
                f"Compiling JSON schema with type={output_type.value}, "
                f"any_whitespace={self.config.xgrammar_any_whitespace}"
            )
            
            ctx = self._xgr_compiler.compile_json_schema(
                grammar_spec,
                any_whitespace=self.config.xgrammar_any_whitespace
            )
            
            logger.debug("[StructuredOutput|XGrammar] JSON schema compiled to FSM")
        else:
            raise ValueError(f"Unsupported request type for xgrammar: {output_type}")
        
        result = CompiledGrammar(
            backend_type=GuidedDecodingBackendType.XGRAMMAR,
            ctx=ctx,
            vocab_size=self.vocab_size,
            xgr_module=xgr,
        )
        return result
    

@dataclass
class CompiledGrammar:
    """
    编译后的 Grammar
    
    可以被多个请求复用（每个请求创建独立的 Matcher）
    """
    backend_type: GuidedDecodingBackendType
    ctx: Any  # xgr.CompiledGrammar
    vocab_size: int
    xgr_module: Any = None  # xgrammar 模块引用


class StructuredOutputManager:
    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        config: Optional[StructuredOutputConfig] = None,
    ):
        """
        初始化管理器
        
        Args:
            tokenizer: HuggingFace tokenizer
            vocab_size: 词表大小
            config: 配置
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.config = config or StructuredOutputConfig()
        
        # 后端（延迟初始化）
        self._backend: Optional[GrammarBackend] = None
        
        # Grammar 缓存：short_key → (grammar_spec, CompiledGrammar)，value 存 spec 用于碰撞校验
        self._grammar_cache: Dict[str, Tuple[str, CompiledGrammar]] = {}
        
        # 请求状态：request_id → StructuredOutputGrammar
        self._request_grammars: Dict[int, StructuredOutputGrammar] = {}
        
        # 预分配的 bitmask 缓冲区
        self._bitmask_buffer: Optional[np.ndarray] = None
        self._init_bitmask_buffer()
        
        logger.debug(
            f"StructuredOutputManager initialized: "
            f"backend={self.config.backend.value}, "
            f"vocab_size={vocab_size}"
        )
    
    @staticmethod
    def _get_cache_key(
        output_type: StructuredOutputType,
        grammar_spec: str,
    ) -> str:
        """生成缓存键（使用 schema 的 SHA256，避免长 key 的哈希与内存开销）"""
        h = hashlib.sha256(grammar_spec.encode()).hexdigest()
        return f"{output_type.value}:{h}"

    def grammar_init(
        self,
        request_id: int,
        structured_output_request: StructuredOutputRequest,
    ) -> Optional[StructuredOutputGrammar]:
        """
        为请求初始化 Grammar
        
        Args:
            request_id: 请求 ID
            structured_output_request: 结构化输出请求
            
        Returns:
            Grammar 实例，或 None（如果不需要约束）
        """
        if structured_output_request is None:
            return None
        try:
            logger.debug(
                f"[StructuredOutput] "
                f"request_id={request_id}, compiling schema..."
            )
            compiled = self._compile_grammar(
                structured_output_request.output_type,
                structured_output_request.grammar_spec,
            )
            logger.debug(
                f"[StructuredOutput] "
                f"request_id={request_id} Schema compiled, creating grammar matcher..."
            )
            grammar = GrammarBackend.create_grammar(compiled)
            logger.debug(
                f"[StructuredOutput] "
                f"request_id={request_id} Grammar matcher created"
            )
            self._request_grammars[request_id] = grammar
            structured_output_request.grammar = grammar
            logger.debug(
                f"[StructuredOutput] "
                f"request_id={request_id} Grammar stored in manager. "
                f"Total grammars: {len(self._request_grammars)}, "
                f"keys: {list(self._request_grammars.keys())}"
            )
            return grammar
            
        except Exception as e:
            logger.error(
                f"[StructuredOutput] "
                f"request_id={request_id} ✗ Exception: {e}"
            )
            return None
    
    def grammar_bitmask(
        self,
        request_ids: List[int],
        apply_bitmask_flags: Optional[List[bool]] = None,
    ) -> Optional[np.ndarray]:
        """
        批量生成 token bitmask
        
        Args:
            request_ids: 请求 ID 列表
            apply_bitmask_flags: 每个请求是否应用 bitmask（可选）
            
        Returns:
            bitmask 数组 [batch_size, vocab_size // 32]，或 None（如果没有需要约束的请求）
        """
        if not request_ids:
            return None
        
        batch_size = len(request_ids)
        if not any(req_id in self._request_grammars for req_id in request_ids):
            return None
        if batch_size > self._bitmask_buffer.shape[0]:
            self._bitmask_buffer = np.zeros((batch_size, self._bitmask_width), dtype=np.int32)
        bitmask = self._bitmask_buffer[:batch_size]
        bitmask.fill(self._full_mask)
        for idx, req_id in enumerate(request_ids):
            grammar = self._request_grammars.get(req_id)
            if grammar is None:
                continue
            if apply_bitmask_flags is not None and not apply_bitmask_flags[idx]:
                continue
            if grammar.is_terminated():
                continue
            try:
                grammar.fill_bitmask(bitmask, idx)
            except Exception as e:
                logger.warning(f"Failed to fill bitmask for request {req_id}: {e}")
        
        return bitmask.copy()
    
    def accept_tokens(
        self,
        request_id: int,
        tokens: List[int],
    ) -> bool:
        """
        接受 token 并更新 FSM 状态
        
        在采样后调用，推进 FSM 状态。
        
        Args:
            request_id: 请求 ID
            tokens: 采样得到的 token 列表
            
        Returns:
            是否成功接受 token
        """
        grammar = self._request_grammars.get(request_id)
        if grammar is None:
            return True  # 没有约束，直接返回成功
        
        return grammar.accept_tokens(request_id, tokens)
    
    def should_advance(self, request_id: int) -> bool:
        """
        检查是否应该推进 FSM 状态
        
        用于判断请求是否有约束且未终止
        
        Args:
            request_id: 请求 ID
            
        Returns:
            是否应该调用 accept_tokens
        """
        grammar = self._request_grammars.get(request_id)
        if grammar is None:
            logger.warning(
                f"[StructuredOutput] "
                f"should_advance(request_id={request_id}) → False (grammar is None). "
                f"Current grammars in manager: {list(self._request_grammars.keys())}"
            )
            return False
        
        return not grammar.is_terminated()
    
    def is_terminated(self, request_id: int) -> bool:
        """
        检查请求是否已完成约束生成
        
        Args:
            request_id: 请求 ID
            
        Returns:
            是否已终止
        """
        grammar = self._request_grammars.get(request_id)
        if grammar is None:
            return True  # 没有约束，视为已完成
        
        return grammar.is_terminated()
    
    def clear_requests(self, request_ids: List[int]) -> None:
        for req_id in request_ids:
            if req_id in self._request_grammars:
                del self._request_grammars[req_id]
    
    def get_request_grammar(self, request_id: int) -> Optional[StructuredOutputGrammar]:
        return self._request_grammars.get(request_id)
    
    def has_structured_output(self, request_id: int) -> bool:
        return request_id in self._request_grammars
    
    def shutdown(self) -> None:
        self._request_grammars.clear()
        self._grammar_cache.clear()
        
        logger.debug("StructuredOutputManager shutdown")
    
    def process_batch_for_generation(
        self,
        sequence_ids: List[int],
        response_format_array: List[Optional[str]],
    ) -> Optional[np.ndarray]:
        if not sequence_ids or not response_format_array:
            return None
        
        # 快速路径：检查是否有任何请求需要约束解码
        has_constraint = any(rf is not None for rf in response_format_array)
        if not has_constraint:
            return None
        
        # 1. 为新请求初始化 grammar（如果尚未初始化）
        request_ids = []
        grammar_init_count = 0
        for i, seq_id in enumerate(sequence_ids):
            request_id = int(seq_id)
            
            # 检查该请求是否需要结构化输出
            if i < len(response_format_array) and response_format_array[i] is not None:
                # 需要约束：初始化 grammar（如果尚未初始化）
                if not self.has_structured_output(request_id):
                    success = self._init_grammar_from_response_format(request_id, response_format_array[i])
                    if not success:
                        logger.warning(f"[StructuredOutput] Failed to init grammar for request {request_id}")
                    else:
                        grammar_init_count += 1
                request_ids.append(request_id)
            else:
                # 不需要约束：添加 request_id 但后续会跳过
                request_ids.append(request_id)
        # 2. 生成 bitmask（批量）
        # 注意：grammar_bitmask 内部会检查每个 request_id 是否有 grammar
        # 没有 grammar 的请求会使用 full mask（允许所有 token）
        bitmask = self.grammar_bitmask(request_ids)
        return bitmask
    
    def update_states_after_sampling(
        self,
        sequence_ids: List[int],
        token_ids: np.ndarray,
    ) -> None:
        """
        采样后更新 FSM 状态
        
        Args:
            sequence_ids: 序列 ID 列表
            token_ids: 采样得到的 token ID 数组
        """
        if not sequence_ids or token_ids is None:
            return
        
        update_count = 0
        for i, seq_id in enumerate(sequence_ids):
            request_id = int(seq_id)
            
            # 检查是否应该更新
            if self.should_advance(request_id):
                try:
                    if i < len(token_ids):
                        token = int(token_ids[i])
                        success = self.accept_tokens(request_id, [token])
                        if not success:
                            logger.warning(f"[StructuredOutput] Token {token} rejected for request {request_id}")
                        else:
                            update_count += 1
                except Exception as e:
                    logger.warning(f"[StructuredOutput] Exception updating request {request_id}: {e}")
        
    def clear_finished_requests(self, sequence_ids: np.ndarray) -> None:
        """
        清理已完成的请求
        
        Args:
            sequence_ids: 需要清理的序列 ID 数组
        """
        request_ids = [int(sid) for sid in sequence_ids]
        self.clear_requests(request_ids)

    def _init_grammar_from_response_format(
        self,
        request_id: int,
        response_format: str,
    ) -> bool:
        """
        从 response_format 字符串初始化 grammar
        
        Args:
            request_id: 请求 ID
            response_format: response_format JSON 字符串
            
        Returns:
            是否成功初始化
        """
        
        try:
            logger.debug(
                f"[StructuredOutput|Parse] "
                f"request_id={request_id}, parsing response_format..."
            )
            
            # 解析 response_format
            structured_output = StructuredOutputRequest.from_response_format(response_format)
            
            if structured_output is None:
                logger.warning(
                    f"[StructuredOutput] "
                    f"request_id={request_id} ✗ Failed to parse response_format"
                )
                return False  # 无效的 response_format
            
            
            # 初始化 grammar
            logger.debug(f"[StructuredOutput|Compile] request_id={request_id}, compiling grammar...")
            grammar = self.grammar_init(request_id, structured_output)
            
            if grammar is None:
                logger.warning(
                    f"[StructuredOutput] "
                    f"request_id={request_id} Failed to compile grammar"
                )
                return False
            
            logger.debug(
                f"[StructuredOutput|Compile] "
                f"request_id={request_id} Grammar compiled and initialized"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"[StructuredOutput] "
                f"request_id={request_id} Exception: {e}"
            )
            return False

    def _init_bitmask_buffer(self) -> None:
        """初始化 bitmask 缓冲区"""
        # bitmask 形状：[max_batch_size, vocab_size // 32]
        # 每个 int32 存储 32 个 bit
        self._bitmask_width = (self.vocab_size + 31) // 32
        self._bitmask_buffer = np.zeros(
            (self.config.bitmask_prealloc_batch, self._bitmask_width),
            dtype=np.int32
        )
        self._full_mask = -1  # 0xFFFFFFFF，允许所有 token
    
    def _ensure_backend(self) -> GrammarBackend:
        """确保后端已初始化"""
        if self._backend is None:
            logger.debug(
                f"[StructuredOutput|Backend|Diagnose] "
                f"Lazy initializing backend (backend_type={self.config.backend.value})..."
            )
            self._backend = GrammarBackend(
                backend_type=self.config.backend,
                tokenizer=self.tokenizer,
                vocab_size=self.vocab_size,
                config=self.config,
            )
            logger.debug(
                f"[StructuredOutput|Backend|Diagnose] "
                f" Backend initialized successfully"
            )
        return self._backend

    def _compile_grammar(
        self,
        output_type: StructuredOutputType,
        grammar_spec: str,
    ) -> CompiledGrammar:
        """编译 Grammar（带缓存）。缓存 value 为 (grammar_spec, compiled)，用于哈希碰撞校验。"""
        cache_key = StructuredOutputManager._get_cache_key(output_type, grammar_spec)

        if cache_key in self._grammar_cache:
            stored_spec, compiled = self._grammar_cache[cache_key]
            if stored_spec == grammar_spec:
                logger.debug(
                    f"[StructuredOutput|Compile] "
                    f"✓ Cache hit for type={output_type.value}"
                )
                return compiled
            # 哈希碰撞，按未命中处理，下方会重新编译并覆盖

        logger.debug(
            f"[StructuredOutput|Compile] "
            f"Compiling new grammar: type={output_type.value}, "
            f"spec_len={len(grammar_spec)}"
        )
        backend = self._ensure_backend()
        compiled = backend.compile_grammar(output_type, grammar_spec)
        logger.debug(
            f"[StructuredOutput|Compile] "
            f" Grammar compiled successfully"
        )

        if len(self._grammar_cache) >= self.config.grammar_cache_size:
            first_key = next(iter(self._grammar_cache))
            del self._grammar_cache[first_key]
            logger.debug(f"[StructuredOutput|Compile] Cache evicted oldest entry")
        self._grammar_cache[cache_key] = (grammar_spec, compiled)
        logger.debug(f"[StructuredOutput|Compile] Grammar cached (total: {len(self._grammar_cache)})")
        return compiled
    