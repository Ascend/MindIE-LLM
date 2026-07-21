#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

path="${BASH_SOURCE[0]}"

mindie_llm_path=$(cd $(dirname $path); pwd)
chmod u+w "${mindie_llm_path}"
rm -rf /dev/shm/* #对于共享内存小的测试场景，每次启动前都清一下
export MINDIE_LLM_HOME_PATH="${mindie_llm_path}"

export MINDIE_LLM_RECOMPUTE_THRESHOLD=0.5
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))' | grep "site-packages/torch")"
if [ -n "$PYTORCH_INSTALL_PATH" ]; then
    export LD_LIBRARY_PATH="$PYTORCH_INSTALL_PATH/lib:$PYTORCH_INSTALL_PATH/../torch.libs:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH=$(find "$MINDIE_LLM_HOME_PATH/lib" -type d | tr '\n' ':' | sed 's/:$//'):${LD_LIBRARY_PATH}
export PYTHONPATH=$MINDIE_LLM_HOME_PATH:$PYTHONPATH
export PYTHONPATH=$MINDIE_LLM_HOME_PATH/lib:$PYTHONPATH

export MINDIE_LOG_LEVEL=INFO
export MINDIE_LOG_TO_STDOUT=0
export MINDIE_LOG_TO_FILE=1
export GRPC_POLL_STRATEGY=poll

export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=120

# Plog日志
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0

if [[ -z "$1" ]]; then
    MINDIE_LLM_BACKEND=("atb" "pt" "ms")
else
    if [[ "$1" == "--backend="* ]]; then
        MINDIE_LLM_BACKEND="${1#*=}"
    else
        echo "Usage: source set_env.sh --backend=<backend>"
    fi
fi

for backend_opt in "${MINDIE_LLM_BACKEND[@]}"; do
    case "$backend_opt" in
        atb)
            ATB_SET_ENV_PATH=$MINDIE_LLM_HOME_PATH/../examples/atb_models/output/atb_models/set_env.sh
            if [ ! -f "$ATB_SET_ENV_PATH" ]; then
                ATB_SET_ENV_PATH=ATBMODELSETENV
            fi
            if [ -f "$ATB_SET_ENV_PATH" ]; then
                source $ATB_SET_ENV_PATH
            fi
            ;;
        pt)
            ;;
        ms)
            ;;
        *)
            echo "Inner Error: unknown option'$backend_opt'"
            ;;
    esac
done
