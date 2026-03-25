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
DT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=${DT_DIR}/..
OUTPUT_DIR=${PROJECT_DIR}/output/atb_models
CACHE_DIR=${PROJECT_DIR}/build
TEST_MODE=""
CPP_TEST_ENABLE=0
PYTHON_TEST_ENABLE=0
BUILD_ENABLE=1

function export_atb_models_env()
{
    source $PROJECT_DIR/output/atb_models/set_env.sh
    export MINDIE_LLM_HOME_PATH=/
}

function fn_build_dt()
{
    if [ $BUILD_ENABLE -eq 1 ]; then
        if [ $CPP_TEST_ENABLE -eq 1 ]; then
            if [ -e "$PROJECT_DIR/modify_files.txt" ]; then
                # 门禁gcc版本不支持-Wno-dangling-reference编译选项
                sed -i '/set(CMAKE_CXX_FLAGS.*-Wno-dangling-reference.*)/d' ${PROJECT_DIR}/CMakeLists.txt
            fi
            cp ${PROJECT_DIR}/../../tests/update.patch ${PROJECT_DIR}/tests/
            bash ${PROJECT_DIR}/scripts/build.sh unittest
            export LD_PRELOAD=$OUTPUT_DIR/lib/libop_stub.so:$LD_PRELOAD
        else
            bash ${PROJECT_DIR}/scripts/build.sh
        fi
    fi
}

# cpp代码测试入口
function fn_run_cpptest()
{
    echo "run $OUTPUT_DIR/bin/speed_unittest"
    $OUTPUT_DIR/bin/speed_unittest
}

# python代码测试入口
function fn_run_pythontest()
{
    cd $OUTPUT_DIR
    export PYTHONPATH=$PROJECT_DIR/../..:$PYTHONPATH
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    MINDIE_LOG_DIR_RELATIVE="${PROJECT_DIR}/../../src/utils/log"
    MINDIE_LOG_DIR=$(readlink -f "${MINDIE_LOG_DIR_RELATIVE}")
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MINDIE_LOG_DIR}/lib"
    export NUMBA_DISABLE_JIT=1 # 禁用JIT编译防止相关函数覆盖率无法被统计
    # atb_llm
    pytest ${PROJECT_DIR}/tests/pythontest/atb_llm --cov=${PROJECT_DIR}/atb_llm \
    --cov-branch --cov-report xml:coverage_atb_llm.xml --cov-report html:htmlcov_atb_llm --continue-on-collection-errors \
    --ignore=${PROJECT_DIR}/tests/pythontest/atb_llm/common_op_builders \
    --ignore=${PROJECT_DIR}/tests/pythontest/atb_llm/runner/test_runner.py \
    --ignore=${PROJECT_DIR}/tests/pythontest/atb_llm/nn \

    # 提取每个文件的行覆盖率
    grep -o '<class name="[^"]*" filename="[^"]*" complexity="[^"]*" line-rate="[^"]*" branch-rate="[^"]*">' coverage_atb_llm.xml |
    while read -r line; do
        if [[ ! "$line" =~ "nn/" &&
              ! "$line" =~ "weights" &&
              ! "$line" =~ "flash_causal_deepseekv2" &&
              ! "$line" =~ "flash_causal_deepseek" &&
              ! "$line" =~ "flash_causal_lm" &&
              ! "$line" =~ "modeling_deepseekv2" &&
              ! "$line" =~ "mapping" &&
              ! "$line" =~ "decoder_layer" &&
              ! "$line" =~ "decoder_model" &&
              ! "$line" =~ "mtp_decoder_model" &&
              ! "$line" =~ "dynamic_ep_moe" &&
              ! "$line" =~ "fused_all2all_gmm" &&
              ! "$line" =~ "flash_causal_llama" &&
              ! "$line" =~ "dist" &&
              ! "$line" =~ "model_runner" &&
              ! "$line" =~ "model_utils" &&
              ! "$line" =~ "modeling" &&
              ! "$line" =~ "modeling_qwen2" &&
              ! "$line" =~ "weight_wrapper" &&
              ! "$line" =~ "pack_type" &&
              ! "$line" =~ "hub" &&
              ! "$line" =~ "__init__" &&
              ! "$line" =~ "weights" &&
              ! "$line" =~ "quant_type" &&
              ! "$line" =~ "w16a16sc" &&
              ! "$line" =~ "base_lm_cpp" &&
              ! "$line" =~ "model_runner" &&
              ! "$line" =~ "position_rotary_embedding" &&
              ! "$line" =~ "internlm2" &&
              ! "$line" =~ "internlm3" &&
              ! "$line" =~ "eplb_expert_data_collect" &&
              ! "$line" =~ "common_op_builder" ]]; then
            echo "$line" | awk -F '"' '{print "examples/atb_models/atb_llm/" $4, $8*100 "%", $10*100 "%"}' >> result.txt
        fi
    done

    # examples
    pytest ${PROJECT_DIR}/tests/pythontest/examples --cov=${PROJECT_DIR}/examples \
    --cov-branch --cov-report xml:coverage_examples.xml --cov-report html:htmlcov_examples --continue-on-collection-errors \
    --ignore=${PROJECT_DIR}/tests/pythontest/examples/server/test_batch.py \
    --ignore=${PROJECT_DIR}/tests/pythontest/examples/convert/model_slim/get_calibration_dataset.py \
    --ignore=${PROJECT_DIR}/tests/pythontest/examples/convert/model_slim/test_sparse_compressor.py \

    # 提取每个文件的行覆盖率
    grep -o '<class name="[^"]*" filename="[^"]*" complexity="[^"]*" line-rate="[^"]*" branch-rate="[^"]*">' coverage_examples.xml |
    while read -r line; do
        if [[ ! "$line" =~ .*run_fa.* && ! "$line" =~ .*run_pa.* && ! "$line" =~ .*cache.* && ! "$line" =~ .*batch.* && ! "$line" =~ .*generate.*  && ! "$line" =~ .*sparse_compressor.* ]]; then
            echo "$line" | awk -F '"' '{print "examples/atb_models/examples/" $4, $8*100 "%", $10*100 "%"}' >> result.txt
        fi
    done

    unset NUMBA_DISABLE_JIT
}

# 构建覆盖率
function fn_build_coverage()
{
    GCOV_DIR=$OUTPUT_DIR/gcov
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
    mkdir $GCOV_INFO_DIR

    $LCOV_PATH -d $CACHE_DIR --zerocounters >> $GCOV_DIR/log.txt
    $LCOV_PATH -c -i -d $CACHE_DIR -o $GCOV_INFO_DIR/init.info >> $GCOV_DIR/log.txt

    fn_run_cpptest

    cd $GCOV_DIR
    $LCOV_PATH -c -d $CACHE_DIR -o $GCOV_INFO_DIR/cover.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt --rc lcov_excl_br_line='ATB_SPEED_LOG_*'
    $LCOV_PATH -a $GCOV_INFO_DIR/init.info -a $GCOV_INFO_DIR/cover.info -o $GCOV_INFO_DIR/total.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $LCOV_PATH --remove $GCOV_INFO_DIR/total.info '*/third_party/*' '*torch/*' '*c10/*' '*ATen/*' '*/c++/7*' '*tests/*' '*/include/utils/*' '*/utils/log/*' '*tools/*' '/usr/*' '*ascend-transformer-boost/*' '*qkv_linear_split*' '*dequant_bias_operation*' '*w8a8_operation*' '*linear*' '*model.cpp*' '*/moe/layer/decoder_layer*' '*/moe/model/decoder_model*' '*/sparse_moe*' '*/obfuscation_setup_operation.cpp' '*/obfuscation_calculate_operation.cpp' '*/deepseekv2/*' '*/glm/*' '*/mapping*' '*/fusion/moe/ep/*' '*add_rms_norm_*' '*check_util.cpp*' '*/base/layer/decoder_layer*' '*base/param/model_param*' '*base/param/param*' '*/concat_operation.cpp' '*mlp*' '*/qwen/*/moe_decoder*' '*/core/base/context_factory.cpp' '*/fusion_attention.cpp' '*/rotary_pos_emb_operation.cpp' '*/attn_v3_operation.cpp' '*/self_attention.cpp' '*/operations/aclrt/ops/aclrt_cmo_async.cpp' '*/operations/aclnn/core/acl_nn_operation.cpp' '*/len_operation.cpp' '*/repeat_operation.cpp' '*/inplacemasked_filltensor_operation.cpp' '*/decoder_model_edge.cpp' '*edge*' '*/operations/aclnn/core/acl_nn_global_cache.cpp' '*/operations/aclnn/ops/*' '*/operations/aclnn/utils/*' '*/fusion/utils.cpp' '*event_manager.cpp*' -o $GCOV_INFO_DIR/final.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
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

function fn_show_coverage()
{
    echo "================================"
    echo "File name    Line coverage    Branch coverage"
    cat $OUTPUT_DIR/result.txt
    echo "================================"
}

# 覆盖率门禁
function fn_coverage_gate()
{
    cd $PROJECT_DIR
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

# 脚本提示信息
function show_help()
{
    echo "Usage: bash tests/run_all_tests.sh [--mode [cpp|python]] [--no-build] [--help]"
    echo
    echo "Options:"
    echo "  --mode      Specify the component to test (cpp or python)"
    echo "  --no-build  Run tests without build"
    echo "  --help      Show this help message"
}

function parse_args()
{
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                TEST_MODE="$2"
                shift 2
                ;;
            --no-build)
                BUILD_ENABLE=0
                echo "Run tests without build, make sure you have run build.sh before"
                shift 1
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 门禁默认运行全量ut
function fn_check_test_ci()
{
    PYTHON_TEST_ENABLE=1
    CPP_TEST_ENABLE=1
}

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
        PYTHON_TEST_ENABLE=1
        CPP_TEST_ENABLE=1
    fi
}

function fn_main()
{
    if [ -e "$PROJECT_DIR/modify_files.txt" ]; then
        fn_check_test_ci
    else
        parse_args "$@"
        fn_check_test
        if [ -e "$OUTPUT_DIR/result.txt" ]; then
            rm $OUTPUT_DIR/result.txt
        fi
    fi

    fn_build_dt
    export_atb_models_env
    if [ $PYTHON_TEST_ENABLE -eq 1 ]; then
        fn_run_pythontest
    fi

    if [ $CPP_TEST_ENABLE -eq 1 ]; then
        fn_build_coverage
    fi

    fn_show_coverage

    # 门禁包含修改文件列表时，启动覆盖率检测，修改文件的行覆盖率小于80%或者分支覆盖率小于60%时，门禁不予通过
    if [ -e "$PROJECT_DIR/modify_files.txt" ]; then
        fn_coverage_gate
    fi
}

fn_main "$@"
