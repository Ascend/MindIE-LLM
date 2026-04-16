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

BITS_PER_BITMASK_WORD = 32


def parse_bitmask_allowed_tokens(bitmask: np.ndarray, vocab_size: int) -> List[int]:
    """
    将 xgrammar 打包的 int32 bitmask 解析为「允许」的 token id 列表（位为 1 表示允许）。
    """
    allowed_tokens: List[int] = []
    if bitmask is None:
        return []
    bm = np.asarray(bitmask, dtype=np.int32, order="C")
    if bm.size == 0:
        return []
    if bm.ndim != 2:
        return []
    batch_size, bitmask_width = int(bm.shape[0]), int(bm.shape[1])
    for batch_idx in range(batch_size):
        for col_idx in range(bitmask_width):
            val = int(bm[batch_idx, col_idx])
            if val == 0:
                continue
            for bit_idx in range(BITS_PER_BITMASK_WORD):
                if val & (1 << bit_idx):
                    token_id = col_idx * BITS_PER_BITMASK_WORD + bit_idx
                    if token_id < vocab_size:
                        allowed_tokens.append(token_id)
    return sorted(set(allowed_tokens))


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
    xgrammar_any_whitespace: bool = False
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
            "[StructuredOutput|Backend|Diagnose] GrammarBackend.__init__() called, about to call _init_backend()"
        )
        self._init_backend()

    @staticmethod
    def create_grammar(compiled: "CompiledGrammar") -> StructuredOutputGrammar:
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
    ) -> "CompiledGrammar":
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
            logger.error("[StructuredOutput|Backend] _get_xgrammar_module() returned None, xgrammar is not installed!")
            raise ImportError("xgrammar is not installed. Please install it with: pip install xgrammar")
        self._xgr = xgr
        xgr_ver = xgr.__version__ if hasattr(xgr, "__version__") else "unknown"
        logger.debug(f"[StructuredOutput|Backend] xgrammar module loaded (version: {xgr_ver})")

        # 创建 TokenizerInfo
        try:
            logger.debug("[StructuredOutput|Backend] Creating TokenizerInfo from HuggingFace tokenizer...")
            self._xgr_tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
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

        logger.debug(f"[StructuredOutput|Backend] xgrammar backend initialized: vocab_size={self.vocab_size}")

    def _compile_xgrammar(
        self,
        output_type: StructuredOutputType,
        grammar_spec: str,
    ) -> "CompiledGrammar":
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
                grammar_spec, any_whitespace=self.config.xgrammar_any_whitespace
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

        # 运行态状态：state_key(当前由 context_handle 承担) → StructuredOutputGrammar
        self._request_grammars: Dict[int, StructuredOutputGrammar] = {}

        # 预分配的 bitmask 缓冲区
        self._bitmask_buffer: Optional[np.ndarray] = None
        self._init_bitmask_buffer()

        logger.debug(
            f"StructuredOutputManager initialized: backend={self.config.backend.value}, vocab_size={vocab_size}"
        )

    @staticmethod
    def has_structured_response_formats(response_format_array: Optional[List[Any]]) -> bool:
        if response_format_array is None:
            return False
        try:
            return any(response_format is not None for response_format in response_format_array)
        except TypeError:
            return response_format_array is not None

    @staticmethod
    def _get_cache_key(
        output_type: StructuredOutputType,
        grammar_spec: str,
    ) -> str:
        """生成缓存键（使用 schema 的 SHA256，避免长 key 的哈希与内存开销）"""
        h = hashlib.sha256(grammar_spec.encode()).hexdigest()
        return f"{output_type.value}:{h}"

    @staticmethod
    def _to_numpy_array(data_instance: Any):
        if data_instance is None:
            return None
        if isinstance(data_instance, np.ndarray):
            return data_instance
        if hasattr(data_instance, "asnumpy"):
            return data_instance.asnumpy()
        if hasattr(data_instance, "detach") and hasattr(data_instance, "cpu") and hasattr(data_instance, "numpy"):
            return data_instance.detach().cpu().numpy()
        if hasattr(data_instance, "cpu") and hasattr(data_instance, "numpy"):
            return data_instance.cpu().numpy()
        try:
            return np.asarray(data_instance)
        except Exception:
            return None

    @staticmethod
    def _sanitize_replay_tokens(
        replay_tokens: Optional[List[int]],
        state_key: Optional[int] = None,
        source: str = "unknown",
    ) -> List[int]:
        if not replay_tokens:
            return []
        valid_tokens = [int(token) for token in replay_tokens if int(token) >= 0]
        dropped_count = len(replay_tokens) - len(valid_tokens)
        if dropped_count > 0:
            logger.debug(
                "[StructuredOutput][ReplaySanitize] state_key=%s source=%s dropped_invalid=%s raw_count=%s",
                state_key,
                source,
                dropped_count,
                len(replay_tokens),
            )
        return valid_tokens

    @staticmethod
    def _accept_tokens_on_grammar(
        state_key: int,
        grammar: StructuredOutputGrammar,
        tokens: List[int],
    ) -> bool:
        """在已解析 grammar 时推进 FSM，供 accept_tokens / replay / decode sync 复用。"""
        tried_before = grammar.num_tried_tokens
        accepted_before = grammar.num_processed_tokens
        terminated_before = grammar.is_terminated()
        ok = grammar.accept_tokens(state_key, tokens)
        tried_after = grammar.num_tried_tokens
        accepted_after = grammar.num_processed_tokens
        logger.debug(
            "[StructuredOutput][Accept] state_key=%s tokens=%s ok=%s "
            "tried %s→%s accepted %s→%s terminated_before=%s terminated_after=%s",
            state_key,
            tokens,
            ok,
            tried_before,
            tried_after,
            accepted_before,
            accepted_after,
            terminated_before,
            grammar.is_terminated(),
        )
        return ok

    @classmethod
    def _extract_replay_tokens(cls, sampling_metadata: Any, sequence_index: int) -> List[int]:
        output_token_ids = cls._to_numpy_array(sampling_metadata.output_token_ids)
        if output_token_ids is not None and sequence_index < len(output_token_ids):
            output_tokens = np.asarray(output_token_ids[sequence_index]).reshape(-1)
            valid_output_tokens = output_tokens[output_tokens >= 0]
            if valid_output_tokens.size > 0:
                return [int(token_id) for token_id in valid_output_tokens.tolist()]

        all_token_ids = cls._to_numpy_array(sampling_metadata.all_token_ids)
        if all_token_ids is not None and sequence_index < len(all_token_ids):
            all_tokens = np.asarray(all_token_ids[sequence_index]).reshape(-1)
            valid_all_tokens = all_tokens[all_tokens >= 0]
            if valid_all_tokens.size > 0:
                return [int(token_id) for token_id in valid_all_tokens.tolist()]
        return []

    def grammar_init(
        self,
        state_key: int,
        structured_output_request: StructuredOutputRequest,
    ) -> Optional[StructuredOutputGrammar]:
        """
        为请求初始化 Grammar

        Args:
            state_key: 当前运行态状态 key
            structured_output_request: 结构化输出请求

        Returns:
            Grammar 实例，或 None（如果不需要约束）
        """
        if structured_output_request is None:
            return None
        existing_grammar = self._request_grammars.get(state_key)
        if existing_grammar is not None:
            structured_output_request.grammar = existing_grammar
            return existing_grammar
        try:
            logger.debug(f"[StructuredOutput] state_key={state_key}, compiling schema...")
            compiled = self._compile_grammar(
                structured_output_request.output_type,
                structured_output_request.grammar_spec,
            )
            logger.debug(f"[StructuredOutput] state_key={state_key} Schema compiled, creating grammar matcher...")
            grammar = GrammarBackend.create_grammar(compiled)
            logger.debug(f"[StructuredOutput] state_key={state_key} Grammar matcher created")
            self._request_grammars[state_key] = grammar
            total_grammars = len(self._request_grammars)
            grammar_keys = list(self._request_grammars.keys())
            structured_output_request.grammar = grammar
            logger.debug(
                f"[StructuredOutput] "
                f"state_key={state_key} Grammar stored in manager. "
                f"Total grammars: {total_grammars}, "
                f"keys: {grammar_keys}"
            )
            return grammar

        except Exception as e:
            logger.error(f"[StructuredOutput] state_key={state_key} ✗ Exception: {e}")
            return None

    def grammar_bitmask(
        self,
        cache_ids: List[int],
        apply_bitmask_flags: Optional[List[bool]] = None,
    ) -> Optional[np.ndarray]:
        """
        批量生成 token bitmask

        Args:
            cache_ids: 状态 key 列表
            apply_bitmask_flags: 每个请求是否应用 bitmask（可选）

        Returns:
            bitmask 数组 [batch_size, vocab_size // 32]，或 None（如果没有需要约束的请求）
        """
        if cache_ids is None or len(cache_ids) == 0:
            return None

        has_any_grammar = any(state_key in self._request_grammars for state_key in cache_ids)
        if not has_any_grammar:
            return None

        batch_size = len(cache_ids)
        if batch_size > self._bitmask_buffer.shape[0]:
            self._bitmask_buffer = np.zeros((batch_size, self._bitmask_width), dtype=np.int32)
        bitmask = self._bitmask_buffer[:batch_size]
        bitmask.fill(self._full_mask)
        for idx, state_key in enumerate(cache_ids):
            if apply_bitmask_flags is not None and not apply_bitmask_flags[idx]:
                continue
            grammar = self._request_grammars.get(state_key)
            if grammar is None or grammar.is_terminated():
                logger.debug(
                    "[StructuredOutput][Bitmask] state_key=%s skipped: grammar=%s terminated=%s "
                    "→ full_mask applied (all tokens allowed)",
                    state_key,
                    "None" if grammar is None else "exists",
                    True if grammar is None else grammar.is_terminated(),
                )
                continue
            try:
                grammar.fill_bitmask(bitmask, idx)
                allowed_tokens = parse_bitmask_allowed_tokens(bitmask[idx : idx + 1], self.vocab_size)
                logger.debug(
                    "[StructuredOutput][Bitmask] state_key=%s tried=%s accepted=%s terminated=%s "
                    "allowed_token_count=%s/%s allowed_tokens=%s",
                    state_key,
                    grammar.num_tried_tokens,
                    grammar.num_processed_tokens,
                    grammar.is_terminated(),
                    len(allowed_tokens),
                    self.vocab_size,
                    allowed_tokens,
                )
            except Exception as e:
                logger.warning(f"Failed to fill bitmask for state_key {state_key}: {e}")

        return bitmask.copy()

    def accept_tokens(
        self,
        state_key: int,
        tokens: List[int],
    ) -> bool:
        """
        接受 token 并更新 FSM 状态

        在采样后调用，推进 FSM 状态。

        Args:
            state_key: 当前运行态状态 key
            tokens: 采样得到的 token 列表

        Returns:
            是否成功接受 token
        """
        grammar = self._request_grammars.get(state_key)
        if grammar is None:
            return True  # 没有约束，直接返回成功

        return self._accept_tokens_on_grammar(state_key, grammar, tokens)

    def should_advance(self, state_key: int) -> bool:
        """
        检查是否应该推进 FSM 状态

        用于判断请求是否有约束且未终止

        Args:
            state_key: 当前运行态状态 key

        Returns:
            是否应该调用 accept_tokens
        """
        grammar = self._request_grammars.get(state_key)
        if grammar is None:
            return False
        return not grammar.is_terminated()

    def is_terminated(self, state_key: int) -> bool:
        """
        检查请求是否已完成约束生成

        Args:
            state_key: 当前运行态状态 key

        Returns:
            是否已终止
        """
        grammar = self._request_grammars.get(state_key)
        if grammar is None:
            return True  # 没有约束，视为已完成
        return grammar.is_terminated()

    def clear_requests(self, cache_ids: List[int]) -> None:
        for state_key in cache_ids:
            self._request_grammars.pop(state_key, None)

    def get_request_grammar(self, state_key: int) -> Optional[StructuredOutputGrammar]:
        return self._request_grammars.get(state_key)

    def has_structured_output(self, state_key: int) -> bool:
        return state_key in self._request_grammars

    def has_any_structured_output(self, cache_ids: List[int]) -> bool:
        return any(int(state_key) in self._request_grammars for state_key in cache_ids)

    def shutdown(self) -> None:
        self._request_grammars.clear()
        self._grammar_cache.clear()

        logger.debug("StructuredOutputManager shutdown")

    def process_batch_for_generation(
        self,
        cache_ids: List[int],
        response_format_array: List[Optional[str]],
    ) -> Optional[np.ndarray]:
        if cache_ids is None or response_format_array is None:
            return None
        if len(cache_ids) == 0 or len(response_format_array) == 0:
            return None

        has_constraint = any(rf is not None for rf in response_format_array)
        if not has_constraint:
            return None

        state_key_list = []
        grammar_init_count = 0
        for i, state_key in enumerate(cache_ids):
            normalized_state_key = int(state_key)
            if i < len(response_format_array) and response_format_array[i] is not None:
                has_grammar = normalized_state_key in self._request_grammars
                if not has_grammar:
                    success = self._init_grammar_from_response_format(normalized_state_key, response_format_array[i])
                    if not success:
                        logger.warning(f"[StructuredOutput] Failed to init grammar for state_key{normalized_state_key}")
                    else:
                        grammar_init_count += 1
                state_key_list.append(normalized_state_key)
            else:
                state_key_list.append(normalized_state_key)

        bitmask = self.grammar_bitmask(state_key_list)
        return bitmask

    def compute_structured_output_accepted(
        self,
        cache_ids: Optional[np.ndarray],
        token_ids: Any,
    ) -> Optional[np.ndarray]:
        """刷新结构化输出 FSM 状态，返回与 cache_ids 对齐的 is_accepted 数组。"""
        if cache_ids is None or token_ids is None or not self.has_any_structured_output(cache_ids):
            return None
        return self.update_states_after_sampling(
            cache_ids=cache_ids,
            token_ids=token_ids,
        )

    def build_and_assign_structured_guided_bitmask(
        self,
        input_metadata: Any,
        sampling_metadata: Any,
        cache_ids: Any,
        response_format_array: Optional[List[Optional[str]]],
    ) -> None:
        """生成结构化 guided bitmask 并写入 sampling_metadata.guided_bitmask。"""
        if sampling_metadata is None:
            return
        all_sequence_ids = getattr(sampling_metadata, "all_sequence_ids", None)
        if not self.has_structured_response_formats(response_format_array) or all_sequence_ids is None:
            return

        num_with_constraint = sum(1 for rf in response_format_array if rf is not None)
        if num_with_constraint == 0:
            return

        predicted_token_ids = input_metadata.batch_predicted_token_ids
        # Decode 须先 sync_states_for_decode 再 process_batch_for_generation，否则会先为无 grammar 的
        # sequence 初始化 grammar（初始态），导致 sync 误判「已有 grammar」而跳过回放，产生多余 '{'。
        if not input_metadata.is_prefill:
            self.sync_states_for_decode(
                cache_ids=cache_ids,
                response_format_array=response_format_array,
                predicted_token_ids=predicted_token_ids,
                sampling_metadata=sampling_metadata,
            )
        bitmask = self.process_batch_for_generation(
            cache_ids=cache_ids,
            response_format_array=response_format_array,
        )
        # PD 分离/重计算：prefill 初始化 grammar 后回放 predicted_token_ids；回放后须重算 bitmask，
        # 否则当前轮仍使用回放前初始态的约束。
        if predicted_token_ids and input_metadata.is_prefill:
            self.replay_predicted_tokens_after_init(
                cache_ids=cache_ids,
                predicted_token_ids=predicted_token_ids,
            )
            bitmask = self.process_batch_for_generation(
                cache_ids=cache_ids,
                response_format_array=response_format_array,
            )
        if bitmask is not None:
            per_seq_allowed = [
                parse_bitmask_allowed_tokens(bitmask[i : i + 1], self.vocab_size) for i in range(bitmask.shape[0])
            ]
            per_seq_counts = [len(t) for t in per_seq_allowed]
            constrained_count = sum(1 for c in per_seq_counts if c < self.vocab_size)
            logger.debug(
                "[StructuredOutput][BitmaskAssign] is_prefill=%s cache_ids=%s "
                "bitmask_shape=%s constrained_seq_count=%s allowed_counts_per_seq=%s",
                input_metadata.is_prefill,
                list(cache_ids),
                bitmask.shape,
                constrained_count,
                per_seq_counts,
            )
            sampling_metadata.guided_bitmask = bitmask
        else:
            logger.debug(
                "[StructuredOutput][BitmaskAssign] is_prefill=%s cache_ids=%s bitmask=None (no constraint applied)",
                input_metadata.is_prefill,
                list(cache_ids),
            )

    def update_states_after_sampling(
        self,
        cache_ids: List[int],
        token_ids: np.ndarray,
    ) -> np.ndarray:
        """
        采样后更新 FSM 状态

        Args:
            cache_ids: 当前批次状态 key 列表
            token_ids: 采样得到的 token ID 数组

        Returns:
            每个序列的结构接受状态数组，与 cache_ids 维度对齐
        """
        if token_ids is None:
            n = 0 if cache_ids is None else len(cache_ids)
            return np.ones(n, dtype=bool)
        if cache_ids is None or len(cache_ids) == 0:
            return np.ones(0, dtype=bool)

        flattened_token_ids = np.asarray(token_ids).reshape(-1)
        is_accepted_array = np.ones(len(cache_ids), dtype=bool)
        for i, state_key in enumerate(cache_ids):
            normalized_state_key = int(state_key)
            grammar = self._request_grammars.get(normalized_state_key)
            if grammar is None or grammar.is_terminated():
                continue
            try:
                if i < len(flattened_token_ids):
                    token = int(flattened_token_ids[i])
                    is_accepted = self._accept_tokens_on_grammar(normalized_state_key, grammar, [token])
                    is_accepted_array[i] = is_accepted
            except Exception as e:
                logger.warning(f"[StructuredOutput] Exception updating state_key {normalized_state_key}: {e}")
                is_accepted_array[i] = False
        return is_accepted_array

    def clear_finished_requests(self, cache_ids: np.ndarray) -> None:
        """
        清理已完成的请求

        Args:
            cache_ids: 需要清理的状态 key 数组
        """
        normalized_cache_ids = [int(state_key) for state_key in cache_ids]
        self.clear_requests(normalized_cache_ids)

    def replay_predicted_tokens_after_init(
        self,
        cache_ids: List[int],
        predicted_token_ids: Optional[List[Optional[List[int]]]],
    ) -> None:
        """PD分离/重计算 prefill 后，将已生成 token 回放到刚初始化的 grammar 中。"""
        if not predicted_token_ids:
            logger.debug("[StructuredOutput][PrefillReplay] skipped because predicted_token_ids is empty")
            return
        for idx, state_key in enumerate(cache_ids):
            normalized_state_key = int(state_key)
            grammar = self._request_grammars.get(normalized_state_key)
            if grammar is None:
                logger.warning(
                    "[StructuredOutput][PrefillReplay] state_key=%s has no grammar before replay, skip",
                    normalized_state_key,
                )
                continue
            if idx >= len(predicted_token_ids) or not predicted_token_ids[idx]:
                logger.debug(
                    "[StructuredOutput][PrefillReplay] state_key=%s has no predicted tokens, skip",
                    normalized_state_key,
                )
                continue
            replay_tokens = self._sanitize_replay_tokens(
                predicted_token_ids[idx],
                state_key=normalized_state_key,
                source="predicted_token_ids",
            )
            if not replay_tokens:
                logger.debug(
                    "[StructuredOutput][PrefillReplay] state_key=%s has no valid predicted tokens after sanitize, skip",
                    normalized_state_key,
                )
                continue
            if self._accept_tokens_on_grammar(normalized_state_key, grammar, replay_tokens):
                logger.debug(
                    f"[StructuredOutput][Prefill] replayed {len(replay_tokens)} predicted tokens "
                    f"for state_key={normalized_state_key}"
                )
            else:
                logger.warning(
                    f"[StructuredOutput][Prefill] failed to replay predicted tokens "
                    f"for state_key={normalized_state_key}, token_count={len(replay_tokens)}"
                )

    def sync_states_for_decode(
        self,
        cache_ids: List[int],
        response_format_array: List[Optional[str]],
        predicted_token_ids: Optional[List[Optional[List[int]]]],
        sampling_metadata: Any,
    ) -> None:
        for idx, state_key in enumerate(cache_ids):
            if idx >= len(response_format_array) or response_format_array[idx] is None:
                continue
            normalized_state_key = int(state_key)
            replay_tokens, source = self._resolve_replay_tokens(
                predicted_token_ids=predicted_token_ids,
                sequence_index=idx,
                sampling_metadata=sampling_metadata,
                state_key=normalized_state_key,
            )
            grammar = self._request_grammars.get(normalized_state_key)
            # replay_position（num_tried_tokens）：replay buffer 游标，含被 reject 的 token，
            # 用作切片下标，避免 C++ 无条件存储的 rejected token 导致的索引错位。
            # accepted_count（num_processed_tokens）：FSM 合法接受的 token 数，仅用于日志。
            replay_position = 0 if grammar is None else int(grammar.num_tried_tokens)
            accepted_count = 0 if grammar is None else int(grammar.num_processed_tokens)
            if grammar is not None:
                if not replay_tokens:
                    continue
                if replay_position == len(replay_tokens):
                    continue
                if replay_position > len(replay_tokens):
                    # predicted_token_ids 可能仅包含 P 节点预生成的 token，D 节点自身采样后
                    # update_states_after_sampling 已将 grammar 推进；此时 replay_position >
                    # len(replay_tokens) 属于正常的「grammar 比 predicted_token_ids 超前」，
                    # 不应回退 grammar 状态。
                    logger.debug(
                        "[StructuredOutput][DecodeSync] state_key=%s grammar replay_pos=%s exceeds replay_count=%s "
                        "(source=%s), keep current grammar to avoid rollback",
                        normalized_state_key,
                        replay_position,
                        len(replay_tokens),
                        source,
                    )
                    continue
                # 增量推进：从 replay_position（游标）起切片，而非从 accepted_count 起。
                # 使用 num_tried_tokens 作为下标，确保 C++ 无条件存储的 rejected token 不会
                # 阻塞其后的合法 token，也不会导致合法 token 被重复喂入 FSM。
                if 0 < replay_position < len(replay_tokens):
                    incremental_tokens = replay_tokens[replay_position:]
                    self._accept_tokens_on_grammar(normalized_state_key, grammar, incremental_tokens)
                    new_replay_pos = grammar.num_tried_tokens
                    new_accepted = grammar.num_processed_tokens
                    if new_replay_pos > replay_position:
                        status = "accepted" if new_accepted > accepted_count else "rejected"
                        logger.debug(
                            "[StructuredOutput][DecodeSync] state_key=%s cursor %s→%s %s %s→%s (%s)",
                            normalized_state_key,
                            replay_position,
                            new_replay_pos,
                            status,
                            accepted_count,
                            new_accepted,
                            source,
                        )
                        continue
                    logger.debug(
                        "[StructuredOutput][DecodeSync] state_key=%s cursor no-advance (%s) pos=%s acc=%s",
                        normalized_state_key,
                        source,
                        replay_position,
                        accepted_count,
                    )
                    continue
                logger.debug(
                    "[StructuredOutput][DecodeSync] state_key=%s pos=%s != replay_len=%s (%s), rebuild "
                    "grammar_before=(tried=%s, accepted=%s)",
                    normalized_state_key,
                    replay_position,
                    len(replay_tokens),
                    source,
                    replay_position,
                    accepted_count,
                )
                self._request_grammars.pop(normalized_state_key, None)

            # grammar 为 None（首次 Decode）或上方 rebuild 路径弹出后，从初始态重建并全量回放。
            logger.debug(
                "[StructuredOutput][DecodeSync] state_key=%s rebuild start: "
                "replay_tokens_len=%s source=%s grammar_was_none=%s",
                normalized_state_key,
                len(replay_tokens),
                source,
                normalized_state_key not in self._request_grammars,
            )
            replay_count = self._build_and_replay_structured_output_state(
                normalized_state_key, response_format_array[idx], replay_tokens
            )
            extra = ""
            if replay_count == 0 and replay_tokens:
                extra = f", input_replay_len={len(replay_tokens)}"
            grammar_after = self._request_grammars.get(normalized_state_key)
            tried_after = 0 if grammar_after is None else int(grammar_after.num_tried_tokens)
            accepted_after = 0 if grammar_after is None else int(grammar_after.num_processed_tokens)
            terminated_after = True if grammar_after is None else grammar_after.is_terminated()
            logger.debug(
                f"[StructuredOutput][Recompute] synced state_key={normalized_state_key}, "
                f"replay_accepted={replay_count}, source={source}{extra}, "
                f"grammar_after=(tried={tried_after}, accepted={accepted_after}, terminated={terminated_after})"
            )

    def _resolve_replay_tokens(
        self,
        predicted_token_ids: Optional[List[Optional[List[int]]]],
        sequence_index: int,
        sampling_metadata: Any,
        state_key: int,
    ) -> Tuple[List[int], str]:
        predicted_tokens = []
        if predicted_token_ids and sequence_index < len(predicted_token_ids):
            predicted_tokens = self._sanitize_replay_tokens(
                predicted_token_ids[sequence_index],
                state_key=state_key,
                source="predicted_token_ids",
            )

        sampling_tokens = self._extract_replay_tokens(sampling_metadata, sequence_index)

        if predicted_tokens:
            return predicted_tokens, "predicted_token_ids"
        if sampling_tokens:
            return sampling_tokens, "sampling_metadata"
        return [], "empty"

    def _build_and_replay_structured_output_state(
        self,
        state_key: int,
        response_format: str,
        replay_tokens: List[int],
    ) -> int:
        if not self._init_grammar_from_response_format(state_key, response_format):
            logger.warning(f"[StructuredOutput][Recompute] failed to init grammar for state_key={state_key}")
            return 0
        if not replay_tokens:
            return 0

        logger.debug(
            "[StructuredOutput][Replay] step=rebuild_accept state_key=%s replay_len=%s replay_tokens=%s",
            state_key,
            len(replay_tokens),
            replay_tokens,
        )
        # 优先完整重放；accept_tokens 是逐 token 推进的，失败时 FSM 已停在合法的部分状态。
        if self.accept_tokens(state_key, replay_tokens):
            return len(replay_tokens)

        # 完整重放失败：accept_tokens 逐 token 处理，在第一个非法 token 处停止，
        # 此时 grammar 已处于接受了合法前缀后的正确状态（_num_processed_tokens 已更新）。
        # 直接使用该部分状态，避免 suffix search 清空 grammar 导致状态归零。
        grammar = self._request_grammars.get(state_key)
        partial_count = 0 if grammar is None else grammar.num_processed_tokens
        if partial_count > 0:
            return partial_count

        # 所有 token 均在起始位置被拒绝：可能 replay_tokens 包含 prompt 等前缀噪声，
        # 尝试在尾部窗口内寻找可重放后缀。
        search_start = max(1, len(replay_tokens) - 512)
        for start_idx in range(search_start, len(replay_tokens)):
            self.clear_requests([state_key])
            if not self._init_grammar_from_response_format(state_key, response_format):
                return 0
            suffix_tokens = replay_tokens[start_idx:]
            if self.accept_tokens(state_key, suffix_tokens):
                logger.debug(
                    f"[StructuredOutput][Recompute] state_key={state_key} replayed from suffix start={start_idx}, "
                    f"suffix_len={len(suffix_tokens)}"
                )
                return len(suffix_tokens)
        return 0

    def _init_grammar_from_response_format(
        self,
        state_key: int,
        response_format: str,
    ) -> bool:
        """
        从 response_format 字符串初始化 grammar

        Args:
            state_key: 当前运行态状态 key
            response_format: response_format JSON 字符串

        Returns:
            是否成功初始化
        """

        try:
            logger.debug(f"[StructuredOutput|Parse] state_key={state_key}, parsing response_format...")
            # 解析 response_format
            structured_output = StructuredOutputRequest.from_response_format(response_format)

            if structured_output is None:
                logger.warning(f"[StructuredOutput] state_key={state_key} ✗ Failed to parse response_format")
                return False  # 无效的 response_format

            # 初始化 grammar
            logger.debug(f"[StructuredOutput|Compile] state_key={state_key}, compiling grammar...")
            grammar = self.grammar_init(state_key, structured_output)

            if grammar is None:
                logger.warning(f"[StructuredOutput] state_key={state_key} Failed to compile grammar")
                return False

            logger.debug(f"[StructuredOutput|Compile] state_key={state_key} Grammar compiled and initialized")
            return True

        except Exception as e:
            logger.error(f"[StructuredOutput] state_key={state_key} Exception: {e}")
            return False

    def _init_bitmask_buffer(self) -> None:
        """初始化 bitmask 缓冲区"""
        # bitmask 形状：[max_batch_size, vocab_size // 32]
        # 每个 int32 存储 32 个 bit
        self._bitmask_width = (self.vocab_size + 31) // 32
        self._bitmask_buffer = np.zeros((self.config.bitmask_prealloc_batch, self._bitmask_width), dtype=np.int32)
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
            logger.debug("[StructuredOutput|Backend|Diagnose]  Backend initialized successfully")
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
                logger.debug(f"[StructuredOutput|Compile] ✓ Cache hit for type={output_type.value}")
                return compiled
            # 哈希碰撞，按未命中处理，下方会重新编译并覆盖

        logger.debug(
            f"[StructuredOutput|Compile] Compiling new grammar: type={output_type.value}, spec_len={len(grammar_spec)}"
        )
        backend = self._ensure_backend()
        compiled = backend.compile_grammar(output_type, grammar_spec)
        logger.debug("[StructuredOutput|Compile]  Grammar compiled successfully")

        if len(self._grammar_cache) >= self.config.grammar_cache_size:
            first_key = next(iter(self._grammar_cache))
            del self._grammar_cache[first_key]
            logger.debug("[StructuredOutput|Compile] Cache evicted oldest entry")
        self._grammar_cache[cache_key] = (grammar_spec, compiled)
        logger.debug(f"[StructuredOutput|Compile] Grammar cached (total: {len(self._grammar_cache)})")
        return compiled
