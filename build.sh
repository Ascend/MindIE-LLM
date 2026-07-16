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
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CODE_ROOT=$(cd $(dirname -- $0); pwd)
SCRIPT_DIR=$CODE_ROOT/scripts

source $SCRIPT_DIR/build_env.sh
source $SCRIPT_DIR/build_version_info.sh
source $SCRIPT_DIR/make_run_package.sh
source $SCRIPT_DIR/extract_debug_symbols.sh
source $SCRIPT_DIR/build_src.sh
source $SCRIPT_DIR/build_third_party.sh
source $SCRIPT_DIR/build_dlt.sh

function fn_build()
{
    if [ -d "$OUTPUT_DIR" ];then
        rm -rf $OUTPUT_DIR
    fi

    if [ -d "$BUILD_DIR/_deps" ];then
        rm -rf $BUILD_DIR/_deps
    fi

    if [ -d "$CODE_ROOT/llm_debug_symbols" ]; then
        rm -rf "$CODE_ROOT/llm_debug_symbols"
    fi

    mkdir -p $OUTPUT_DIR $CACHE_DIR $BUILD_DIR $MINDIE_LLM_LIB_DIR

    if [ "$CMAKE_CXX_COMPILER_LAUNCHER" == "" ] && command -v ccache &> /dev/null;then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    fi

    fn_build_version_info
    fn_build_third_party

    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "env ASCEND_HOME_PATH not exist, skip kernels compilation"
    else
        source $SCRIPT_DIR/build_kernels.sh
    fi

    fn_build_src
    if [ -z "$UT_BUILD_TARGETS" ]; then
        cp $OUTPUT_DIR/lib/libfoundation.so $MINDIE_LLM_LIB_DIR/foundation.so
        if [ "$build_type" = "release" ]; then
            fn_extract_debug_symbols $OUTPUT_DIR "$CODE_ROOT/llm_debug_symbols"
        fi
        fn_build_for_ci
    fi
    cp $SCRIPT_DIR/set_env.sh $OUTPUT_DIR
}

function fn_clean() {
    echo "Cleaning build and output directories..."

    # 删除构建目录
    if [ -d "$BUILD_DIR" ]; then
        echo "Removing build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi

    # 删除输出目录
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Removing output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
     fi

    echo "Clean completed."
}

function fn_hitest_env(){
    mkdir -p ${WORKSPACE}/opt
    cd ${WORKSPACE}/opt
    if [ "${ARCH}"x == "aarch64"x ]; then
        wget -q https://mindie.obs.cn-north-4.myhuaweicloud.com/Hitest-tool/Aurogon-compile-arm-v6.2.0-simple.zip
        unzip -q -o Aurogon-compile-arm-v6.2.0-simple.zip
        rm -rf Aurogon-compile-arm-v6.2.0-simple.zip
        cd hitest/linux_avatar_arm_64
        chmod +x *
        hitest_tool="linux_avatar_arm_64"
    else
        wget -q https://mindie.obs.cn-north-4.myhuaweicloud.com/Hitest-tool/Aurogon-compile-v6.2.0-simple.zip
        unzip -q -o Aurogon-compile-v6.2.0-simple.zip
        rm -rf Aurogon-compile-v6.2.0-simple.zip
        cd hitest/linux_avatar_x86_64
        chmod +x *
        hitest_tool="linux_avatar_x86_64"
    fi
    ls ${WORKSPACE}/opt
    HitestHome=${WORKSPACE}/opt/hitest/${hitest_tool} # hitest的根目录
    export isOverlappedCompile=0
    export HITEST_PRINT_LOG_ENABLE=0
    export LLT_EXCLUDEFILE=sequence_group.cpp
    export PlatformToken=BOARD
    export gcovmode=0
    export TimerPolicy=1         #是否开启线程定时器采集覆盖率数据, 1: 开启, 0: 不开启
    export TimeInterval=60       #线程定时器，各隔多少秒采集一次数据，60表示60秒
    export SignalPolicy=1        #是否开启发信号采集覆盖率，1: 开启, 0: 不开启
    export SignalNUM=34          #kill -34 pid 采覆盖率， kill -44 pid 重置覆盖率
    export lltwrapper_cfg=0      # 0: 普通模式，4: 无OS极简模式(单模块), 5: 无OS通用模式
    export HITEST_AGENT_INSIDE=1 # 1: 使用内嵌agent.o, 0: 不使用内嵌agent.o
    export USE_HLLT_COVERAGE=1
    export USE_HLLT_TESTCASE=0
    export simplemode=0                                 # 0: 非精简模式, 1: 精简模式
    export ncs_coverage_stub_mold=1       # 0: 非计数模式, 1: 计数模式
    export HITEST_ENABLE_SOKCET=0         # 覆盖率实时展示服务端开关：1-打开；0-关闭。默认关闭，默认端口60005
    export hitest_disable_cfg=0           # 是否需要导出CFG控制流图, 默认导出 0:导出  1:不导出
    export hitest_disable_dfg=1           # 是否需要导出污点数据, 默认不导出 0:导出  1:不导出
    export hitest_disable_ir=1            # 是否需要导出IR，进而探索函数指针调用信息，0：导出  1：不导出
    export HITEST_DISABLE_MACRO=0         # 1:禁止宏插桩，0:允许宏插桩，默认开启宏插桩
    export HITEST_REMOVE_INCLUDE_DIR=0    # 不删除编译命令中的-I和-include编译选项
    export HITEST_AGENT_SET_THREADNAME_PRCTL=1 # 使用版本的api给线程设置名称
    export HITEST_INST_HEADER_FILE=0      # 1:插桩h文件，0:不插桩h文件，默认值为0，4.9.0版本以上才有这个功能
    export HITEST_USER_ACCOUNT=a00000000
    export lltcovRootpath=/tmp/shiqiang/task/covdata          #(测试执行环境中，覆盖率数据保存的根目录)
    export HITEST_COVSTUB_ROOT_DIR=/tmp/shiqiang/daily/  #设置covstub生成路径
    export PATH=${HitestHome}:$PATH             #(添加工具包到PATH环境变量)
    export LD_LIBRARY_PATH=${HitestHome}:$LD_LIBRARY_PATH           #(添加工具包到LD_LIBRARY_PATH环境变量)
    find ${HITEST_COVSTUB_ROOT_DIR}/covstub -mindepth 1 -maxdepth 1 -type d ! -name 'hitest_log' ! -name 'LLTTEMP' -exec rm -rf {} + || echo "Find covstub"

}

function fn_main()
{
    # 检查参数中是否包含 --coverage
    if [[ "$*" =~ "--coverage" ]]; then
        echo "插桩编译"
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_HITESTWRAPPER=ON"
        fn_hitest_env
    fi
    get_version
    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="master"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ "$1" =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ "$cfg_flag" == 1 ]];then
            arg1="master"
        else
            echo "argument $1 is unknown, please type build.sh help for more information"
            exit 1
        fi
    fi

    USE_VERBOSE=OFF

    if [[ $arg1 = "dlt" ]];then
        parse_args "$@"
    fi

    UT_BUILD_TARGETS=""  # unittest 模式下精准编译的 target 列表
    ut_target_next=""    # 状态变量：等待下一个参数作为 -t 的值

    until [[ -z "$1" ]]
    do {
        # 上一个参数是 -t，当前参数是它的值
        if [ -n "$ut_target_next" ]; then
            IFS=',' read -ra ut_modules <<< "$1"
            for m in "${ut_modules[@]}"; do
                m=$(echo "$m")
                if [[ "$m" != *_ut ]] && [[ "$m" != *_it ]] && [[ "$m" != *_st ]]; then
                    m="${m}_ut"
                fi
                UT_BUILD_TARGETS="$UT_BUILD_TARGETS MindIE-LLM_${m}"
            done
            ut_target_next=""
            shift
            continue
        fi

        arg2=$1
        case "${arg2}" in
        "--use_cxx11_abi=1")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=1"
            ;;
        "--use_cxx11_abi=0")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=0"
            ;;
        "--ini=version")
            VERSION_INFO_FILE=$CODE_ROOT/../CI/config/version.ini
            ;;
        "--ini=version_item")
            VERSION_INFO_FILE=$CODE_ROOT/../CI/config/version_item.ini
            ;;
        "-t" | "--target")
            ut_target_next="true"
            ;;
        esac
        shift
    }
    done
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_INSTALL_PREFIX='$OUTPUT_DIR'"
    case "${arg1}" in
        "debug")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug -DDOMAIN_LAYERED_TEST=OFF"
            set -x
            fn_build
            fn_make_run_package
            ;;
        "master")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDOMAIN_LAYERED_TEST=OFF"
            fn_build
            ;;
        "3rd")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release -DDOMAIN_LAYERED_TEST=ON"
            fn_build_third_party
            ;;
        "release")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDOMAIN_LAYERED_TEST=OFF"
            build_type="release"
            fn_build
            fn_make_whl
            fn_make_run_package
            fn_make_debug_symbols_package
            ;;
        "clean")
            fn_clean
            ;;
        "unittest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS:-} -DCMAKE_BUILD_TYPE=Debug -DDOMAIN_LAYERED_TEST=ON"
            echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
            export COVERAGE_TYPE="unittest"
            export MINDIE_LLM_HOME_PATH="$OUTPUT_DIR"
            build_type="release"
            fn_build
            ;;
        "dlt")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug -DDOMAIN_LAYERED_TEST=ON -DENABLE_COVERAGE=$enable_coverage"
            cd $CODE_ROOT
            fn_build_third_party
            fn_dlt
            ;;
        "help")
            echo "Usage: build.sh <mode> [options]"
            echo ""
            echo "Modes:"
            echo "  3rd       Build third-party dependencies only"
            echo "  dlt       Build with DLT instrumentation"
            echo "  debug     Build debug + run package"
            echo "  release   Build release + whl + run package"
            echo "  master    Build RelWithDebInfo only"
            echo "  unittest  Build with unit test coverage"
            echo "  clean     Remove build and output directories"
            echo ""
            echo "Options:"
            echo "  --use_cxx11_abi=0|1   Select C++11 ABI version"
            echo "  --ini=version         Use CI version.ini"
            echo "  --ini=version_item    Use CI version_item.ini"
            ;;
        *)
            echo "unknown build type:${arg1}";
            ;;
    esac
}

fn_main "$@"
