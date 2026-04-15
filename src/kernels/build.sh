#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

set -e

contains() {
    local str="$1"
    local substr="$2"
    [[ "$str" == *"$substr"* ]]
}

soc_name=$(python3 -c "\
import torch;\
import torch_npu;\
soc_name = torch.npu.get_device_properties().name;\
print(f'mie_ops_version:{soc_name}');\
" 2>/dev/null) || soc_name="unknown"

if contains "$soc_name" "mie_ops_version:Ascend910B" ]; then
    ops="ascend910b"
elif contains "$soc_name" "mie_ops_version:Ascend910_93" ]; then
    ops="ascend910_93"
else
    if [ $# -eq 1 ]; then
        ops=$1
    else
        echo "Soc version not identified and missing input parameter 'ops'. Skipping mie_ops build!"
        exit 0
    fi
fi

BASE_DIR=$(realpath "$(dirname "$0")")

if ! python3 -c "import psutil" 2>/dev/null; then
    echo "正在安装 psutil..."
    pip3 install psutil
fi
export CC=$(which gcc)
export CXX=$(which g++)

$CC --version
$CXX --version

cd $BASE_DIR/mie_ops
echo "ascendc build ops: $ops"
bash csrc/build_aclnn.sh $BASE_DIR/mie_ops $ops
./csrc/build/cann-ops-transformer-custom*.run --quiet -- --install-path=$BASE_DIR/mie_ops/opp

# 编译wheel包
cd -
python3 $BASE_DIR/setup.py build bdist_wheel --ops $ops
