#!/bin/bash
function fn_build_third_party()
{
    THIRD_BUILD_DIR=${CODE_ROOT}/build/third_party
    mkdir -p ${THIRD_BUILD_DIR}
    cd ${THIRD_BUILD_DIR}

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake ${CODE_ROOT}/third_party_entry/ $COMPILE_OPTIONS
    if [ "$USE_VERBOSE" == "ON" ];then
        cmake --build . --target all_third_party -- VERBOSE=1 -j"$thread_num"
    else
        cmake --build . --target all_third_party -- -j"$thread_num"
    fi
    cd -
}
