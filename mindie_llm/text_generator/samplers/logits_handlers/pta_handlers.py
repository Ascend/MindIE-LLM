# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from functools import wraps
from typing import Any, Optional

import torch

from .logits_handler import LogitsHandler
from ...utils.sampling_metadata import SamplingMetadata
from ....utils.log.logging import logger, print_log

PTA_HANDLER_REGISTRY = {}


def register_class(name):
    def decorator(cls):
        PTA_HANDLER_REGISTRY[name] = cls

        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        return wrapper

    return decorator


@register_class("repetition_penalty")
class RepetitionPenaltyLogitsHandler(LogitsHandler):
    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata):
        if metadata.all_token_ids is None:
            return logits
        repetition_penalty = metadata.repetition_penalty
        vocab_size = len(logits[0])
        sequence_tokens_counts = torch.zeros(
            (len(logits), vocab_size + 1), dtype=metadata.all_token_ids.dtype, device=metadata.all_token_ids.device
        )
        all_token_ids = torch.clamp(metadata.all_token_ids, -1, vocab_size)
        sequence_tokens_counts.scatter_add_(1, all_token_ids, torch.ones_like(all_token_ids))
        sequence_tokens_mask = sequence_tokens_counts[:, :vocab_size] > 0
        try:
            logits = torch.where(sequence_tokens_mask & torch.lt(logits, 0), logits * repetition_penalty, logits).to(
                logits.dtype
            )
            logits = torch.where(sequence_tokens_mask & torch.ge(logits, 0), logits / repetition_penalty, logits).to(
                logits.dtype
            )
        except ZeroDivisionError as e:
            print_log(self.params.rank, logger.error, "repetition penalty cannot be `0`!")
            raise e
        return logits


@register_class("frequency_penalty")
class FrequencyPenaltyLogitsHandler(LogitsHandler):
    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata):
        if metadata.output_token_ids is not None:
            self.params.count_output_tokens(metadata.output_token_ids)
            logits -= metadata.frequency_penalty * self.params.output_token_counts
        return logits


@register_class("presence_penalty")
class PresencePenaltyLogitsHandler(LogitsHandler):
    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata):
        if metadata.output_token_ids is not None:
            self.params.count_output_tokens(metadata.output_token_ids)
            output_tokens_mask = self.params.output_token_counts > 0
            logits -= metadata.presence_penalty * output_tokens_mask
        return logits


@register_class("temperature")
class TemperatureLogitsHandler(LogitsHandler):
    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata):
        try:
            logits = logits / metadata.temperature
        except ZeroDivisionError as e:
            print_log(self.params.rank, logger.error, "temperature cannot be `0`!")
            raise e
        return logits


@register_class("guided_decoding")
class GuidedDecodingLogitsHandler(LogitsHandler):
    def __init__(self, params):
        super().__init__(params)
        self._apply_token_bitmask_inplace: Optional[Any] = None
        self._import_attempted = False

    def __call__(self, logits: torch.Tensor, metadata: SamplingMetadata) -> torch.Tensor:
        bitmask = metadata.guided_bitmask
        if bitmask is None:
            return logits

        _, vocab_size = logits.shape

        try:
            import_success = self._lazy_import()
            if not import_success:
                logger.warning(
                    "[GuidedDecoding] Failed to import apply_token_bitmask_inplace, returning original logits"
                )
                return logits

            self._apply_token_bitmask_inplace(logits, bitmask, vocab_size)

        except Exception as e:
            logger.warning(f"[GuidedDecoding] Failed to apply grammar bitmask: {e}")

        return logits

    def _lazy_import(self) -> bool:
        if self._import_attempted:
            return self._apply_token_bitmask_inplace is not None
        self._import_attempted = True
        try:
            from ...plugins.structured_output import apply_token_bitmask_inplace

            self._apply_token_bitmask_inplace = apply_token_bitmask_inplace
            return True
        except ImportError as e:
            logger.warning(f"Failed to import apply_token_bitmask_inplace: {e}")
            return False
