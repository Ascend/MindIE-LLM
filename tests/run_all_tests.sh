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

set -ex
echo "gcc version in run all test"
g++ -v 2>&1


DT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=${DT_DIR}/..
OUTPUT_DIR=${PROJECT_DIR}/output
CACHE_DIR=${PROJECT_DIR}/build
TEST_MODE=${1:-""} # cpp / python
CODE_ROOT=$(pwd)
CPP_TEST_ENABLE=0
PYTHON_TEST_ENABLE=1

function export_mindie_llm_env()
{
    if [ $CPP_TEST_ENABLE -eq 1 ] || [ $PYTHON_TEST_ENABLE -eq 1 ]; then
        source $PROJECT_DIR/output/set_env.sh
    fi
}

# dt测试build入口
function fn_build_dt()
{
    if [ $CPP_TEST_ENABLE -eq 1 ]; then
        bash ${PROJECT_DIR}/build.sh unittest
    elif [ $PYTHON_TEST_ENABLE -eq 1 ]; then
        bash ${PROJECT_DIR}/build.sh
    fi
}

# 构建cpp代码覆盖率
function fn_build_coverage()
{
    GCOV_DIR=$OUTPUT_DIR/gcov
    GCOV_CACHE_DIR=$OUTPUT_DIR/gcov/cache
    GCOV_INFO_DIR=$OUTPUT_DIR/gcov/cov_info
    if which lcov > /dev/null 2>&1; then
        LCOV_PATH=$(which lcov)
        GENHTML_PATH=$(which genhtml)
    else
        cd ${PROJECT_DIR}/third_party/lcov
        make -j8 && make install PREFIX=${PROJECT_DIR}/third_party/lcov/output
        LCOV_PATH=${PROJECT_DIR}/third_party/lcov/output/bin/lcov
        GENHTML_PATH=${PROJECT_DIR}/third_party/lcov/output/bin/genhtml
    fi
    FIND_IGNORE_PATH=$CACHE_DIR/core/CMakeFiles/mindie_llm_static.dir/*
    if [ -d "$GCOV_DIR" ]
    then
        rm -rf $GCOV_DIR
    fi
    mkdir $GCOV_DIR
    mkdir $GCOV_CACHE_DIR
    mkdir $GCOV_INFO_DIR

    $LCOV_PATH -d $GCOV_CACHE_DIR --zerocounters >> $GCOV_DIR/log.txt

    find $CACHE_DIR -not -path "$FIND_IGNORE_PATH" -name "*.gcno" | xargs -i cp {} $GCOV_CACHE_DIR
    $LCOV_PATH -c -i -d $GCOV_CACHE_DIR -o $GCOV_INFO_DIR/init.info >> $GCOV_DIR/log.txt

    find $CACHE_DIR -name "*.gcda" | xargs -i cp {} $GCOV_CACHE_DIR
    cd $GCOV_CACHE_DIR
    find . -name "*.cpp" | xargs -i gcov {} >> $GCOV_DIR/log.txt
    cd ..
    $LCOV_PATH -c -d $GCOV_CACHE_DIR -o $GCOV_INFO_DIR/cover.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt --rc lcov_excl_br_line='MINDIE_LLM_LOG_*'
    $LCOV_PATH -a $GCOV_INFO_DIR/init.info -a $GCOV_INFO_DIR/cover.info -o $GCOV_INFO_DIR/total.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $LCOV_PATH --remove $GCOV_INFO_DIR/total.info '*/third_party/*' '*torch/*' '*c10/*' '*ATen/*' '*cpu_logits_handler/*' '*/c++/7*' '*/llm_backend/*' '*/src/*' '*tests/*' '*tools/*' '/usr/*' '*ascend-transformer-boost/*' '*python_api/*' '*/server/* */src/utils/common_util.cpp ' '*/mindie_llm/text_generator/cpp/*' '*connector/cpp/*' -o $GCOV_INFO_DIR/final.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $GENHTML_PATH --rc lcov_branch_coverage=1 -o cover_result $GCOV_INFO_DIR/final.info -o cover_result >> $GCOV_DIR/log.txt
    tail -n 4 $GCOV_DIR/log.txt
    cd $OUTPUT_DIR
    tar -czf gcov.tar.gz gcov

    coverage_info=$($LCOV_PATH --list-full-path --list $GCOV_INFO_DIR/final.info --rc lcov_branch_coverage=1)

    echo "$coverage_info" | while IFS= read -r line; do
        if [[ $line =~ ^([^[:space:]]+)[[:space:]]*\|[[:space:]]*([0-9.-]+%)[[:space:]]*[0-9]+\|[[:space:]]*([0-9.-]+%)[[:space:]]*[0-9]+\|[[:space:]]*([0-9.-]+%?)* ]]; then
            file_path="${BASH_REMATCH[1]}"
            line_coverage="${BASH_REMATCH[2]}"
            br_coverage="${BASH_REMATCH[4]}"
            if [[ "$br_coverage" == "-" ]] || [[ -z "$br_coverage" ]]; then
                br_coverage="100%"
            fi
            echo "$file_path $line_coverage $br_coverage" >> result.txt
        fi
    done
}

# python dt测试门禁入口
function fn_run_pythontest()
{
    chmod 750 $OUTPUT_DIR
    cd $OUTPUT_DIR
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export PYTHONPATH=$PROJECT_DIR/src/server/tokenizer:$PYTHONPATH
    export PYTHONPATH=$PROJECT_DIR/build/mindie_llm/connector/cpp::$PYTHONPATH
    export PYTHONPATH=$PROJECT_DIR/build/mindie_llm/text_generator/cpp/memory_bridge:$PYTHONPATH
    export PYTHONPATH=$PROJECT_DIR/build/mindie_llm/text_generator/cpp/prefix_tree:$PYTHONPATH
    export PYTHONPATH=$PROJECT_DIR/build/mindie_llm/text_generator/cpp/sampler/cpu_logits_handler:$PYTHONPATH
    export MINDIE_LOG_TO_FILE=1
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD
    export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/output/abseil-cpp/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/output/protobuf/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/output/grpc/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/output/boost/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${PROJECT_DIR}/third_party/output/libboundscheck/lib:${LD_LIBRARY_PATH}

    devices=("cpu" "npu")

    for device in "${devices[@]}"; do
        pytest ${PROJECT_DIR}/tests/pythontest/$device --cov=${PROJECT_DIR}/mindie_llm --cov-branch --cov-append \
        --cov-report xml:coverage.xml --cov-report html --continue-on-collection-errors --forked \
        --ignore=${PROJECT_DIR}/tests/pythontest/npu/llm_manager \
        --ignore=${PROJECT_DIR}/tests/pythontest/npu/text_generator/test_plugins/test_plugin_manager_edge.py \
        --ignore=${PROJECT_DIR}/tests/pythontest/npu/text_generator/separate_deployment_engine/test_generator_pd_separate.py \
        --ignore=${PROJECT_DIR}/tests/pythontest/npu/text_generator/separate_deployment_engine/test_separate_deployment_engine.py \
        --ignore=${PROJECT_DIR}/tests/pythontest/npu/test_block_copy.py \
        --ignore=${PROJECT_DIR}/tests/pythontest/cpu/runtime ;
    done

    export MINDIE_LLM_FRAMEWORK_BACKEND='ms'
    pytest ${PROJECT_DIR}/tests/pythontest/mindspore --cov=${PROJECT_DIR}/mindie_llm --cov-branch --cov-append \
    --cov-report xml:coverage.xml --cov-report html --continue-on-collection-errors --forked

    # 提取每个文件的行覆盖率
    grep -o '<class name="[^"]*" filename="[^"]*" complexity="[^"]*" line-rate="[^"]*" branch-rate="[^"]*">' coverage.xml |
    while read -r line; do
        filename=$(echo "$line" | awk -F '"' '{print $4}')
        if [[ ! "$line" =~ .*block_copy.* && ! "$line" =~ .*examples/run_generator.* && ! "$line" =~ .*examples/scheduler.* && ! "$line" =~ .*cache_manager.* && ! "$line" =~ .*utils/config.* && ! "$line" =~ .*__init__.* && ! "$line" =~ .*utils/log/logging.* && ! "$line" =~ .*mf_model_wrapper.* && ! "$line" =~ .*generator_ms.* && ! "$line" =~ .*plugin_manager_edge.* && ! "$line" =~ .*runtime.* && ! "$line" =~ .*aclgraph.* && ! "$line" =~ tokenizer/.*py ]]; then
            echo "$line" | awk -F '"' '{print "mindie_llm/" $4, $8*100 "%", $10*100 "%"}' >> result.txt
        fi
    done
}

# 显示覆盖率
function fn_show_coverage()
{
    echo "==================================="
    echo "File name    Line coverage    Branch coverage"
    cat $OUTPUT_DIR/result.txt
    echo "==================================="
}

# 覆盖率门禁
function fn_coverage_gate()
{
    grep -f modify_files.txt $OUTPUT_DIR/result.txt | while IFS=' ' read -r file_name line_cov branch_cov; do
        line_cov_percent=${line_cov%\%}
        branch_cov_percent=${branch_cov%\%}
        line_flag=$(awk -v val="$line_cov_percent" 'BEGIN{
            print(val < 80 ? 1 : 0)
        }')
        branch_flag=$(awk -v val="$branch_cov_percent" 'BEGIN{
            print(val < 30 ? 1 : 0)
        }')
        if [ "$line_flag" -eq 1 ]; then
            echo "The line coverage of $file_name is $line_cov, less than 80%, please add unit test." >> temp.txt
        fi
        if [ "$branch_flag" -eq 1 ]; then
            echo "The branch coverage of $file_name is $branch_cov, less than 30%, please add unit test." >> temp.txt
        fi
    done

    if [ -e "temp.txt" ]; then
        echo "Dt test failed, please check the following message:"
        cat temp.txt
        rm temp.txt
        exit -1
    fi
}

# 门禁默认运行全量ut
function fn_check_test_ci()
{
    CPP_TEST_ENABLE=1
    PYTHON_TEST_ENABLE=1
}

# 非门禁情况下支持通过传参控制是否运行cpp测试和python测试
function fn_check_test()
{
    if [ "$TEST_MODE" == "cpp" ]; then
        echo "Run cpp test only"
        CPP_TEST_ENABLE=1
    elif [ "$TEST_MODE" == "python" ]; then
        echo "Run python test only"
        PYTHON_TEST_ENABLE=1
    else
        echo "Run all tests"
        CPP_TEST_ENABLE=1
        PYTHON_TEST_ENABLE=1
    fi
}

function fn_main()
{
    if [ -e "modify_files.txt" ]; then
        fn_check_test_ci
    else
        fn_check_test
        if [ -e "$OUTPUT_DIR/result.txt" ]; then
            rm $OUTPUT_DIR/result.txt
        fi
    fi

    fn_build_dt

    export_mindie_llm_env
    if [ $PYTHON_TEST_ENABLE -eq 1 ]; then
        fn_run_pythontest
    fi

    if [ $CPP_TEST_ENABLE -eq 1 ]; then
        fn_build_coverage
    fi

    fn_show_coverage

    # 门禁包含修改文件列表时，启动覆盖率检测，修改文件的行覆盖率小于80%或者分支覆盖率小于30%时，门禁不予通过
    cd $PROJECT_DIR
    if [ -e "modify_files.txt" ]; then
        fn_coverage_gate
    fi
}

fn_main "$@"
