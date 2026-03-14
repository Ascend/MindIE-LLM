# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import threading
import traceback
import os
from enum import Enum
from mindie_llm.utils.log.logging import logger
from mindie_llm.utils.log.error_code import convert_exception_to_error_code


# 核心线程异常后后会退出进程
class CoreThread(threading.Thread):
    def run(self):
        try:
            # 调用父类的 run 方法，执行目标函数
            super().run()
        except Exception as e:
            error_code = convert_exception_to_error_code(str(e))
            if isinstance(e, RuntimeError) and error_code is not None:
                message = (f'{error_code.name} fault happened and will be reported to Llm engine with'
                            f'error code: {error_code.value}.')
                logger.error(message)
            else:
                traceback.print_exc()
                logger.error(
                    f"Core thread encountered an exception: tid={self.ident}, tname={self.name}, "
                    f"exception_type={type(e).__name__}, exception_message={str(e)}"
                )
                os._exit(1)


class MindieLlmStatusCode(Enum):
    SUCCESS = 0
    TEXT_GENERATOR_PD_RETRY_LINK = 1
    TEXT_GENERATOR_PD_ALREADY_LINK = 2