#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
from mindie_llm.utils.log.logging import logger as baselog
from mindie_llm.utils.log.logging_base import HandlerType


class TokenizerLogger:
    @staticmethod
    def debug(msg):
        baselog.debug(msg, extra={"handler_ids": HandlerType.TOKENIZER})

    @staticmethod
    def info(msg):
        baselog.info(msg, extra={"handler_ids": HandlerType.TOKENIZER})

    @staticmethod
    def warning(msg):
        baselog.warning(msg, extra={"handler_ids": HandlerType.TOKENIZER})

    @staticmethod
    def error(msg):
        baselog.error(msg, extra={"handler_ids": HandlerType.TOKENIZER})

    @staticmethod
    def exception(msg):
        baselog.critical(msg, extra={"handler_ids": HandlerType.TOKENIZER})


logger = TokenizerLogger()
logger.info(f"tokenizer-{os.getpid()} start.")
