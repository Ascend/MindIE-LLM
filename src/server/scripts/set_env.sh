#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd
# All rights reserved.

path="${BASH_SOURCE[0]}"
mies_path=$(cd $(dirname $path); pwd )
export MINDIE_LLM_HOME_PATH=${mies_path}/latest
export LD_LIBRARY_PATH=$(find "${MINDIE_LLM_HOME_PATH}/lib" -type d | tr '\n' ':' | sed 's/:$//'):${LD_LIBRARY_PATH}
export PYTHONPATH=${MINDIE_LLM_HOME_PATH}/bin:${PYTHONPATH}
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=120
# mindie日志
export MINDIE_LOG_LEVEL=ERROR
export MINDIE_LOG_TO_STDOUT=0
export MINDIE_LOG_TO_FILE=1
# 运行时日志
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0