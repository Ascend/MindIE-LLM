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
    cp $OUTPUT_DIR/lib/libfoundation.so $MINDIE_LLM_LIB_DIR/foundation.so
    if [ "$build_type" = "release" ]; then
        fn_extract_debug_symbols $OUTPUT_DIR "$CODE_ROOT/llm_debug_symbols"
    fi
    fn_build_for_ci
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

function fn_main()
{
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

    until [[ -z "$1" ]]
    do {
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
