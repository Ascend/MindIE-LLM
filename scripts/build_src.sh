#!/bin/bash
function fn_build_src()
{
    cd $BUILD_DIR

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake .. $COMPILE_OPTIONS

    if [ -n "$UT_BUILD_TARGETS" ]; then
        echo "Building specified UT targets:$UT_BUILD_TARGETS"
        if [ "$USE_VERBOSE" == "ON" ];then
            cmake --build . --target $UT_BUILD_TARGETS -- VERBOSE=1 -j"$thread_num"
        else
            cmake --build . --target $UT_BUILD_TARGETS -- -j"$thread_num"
        fi
    elif [ "$USE_VERBOSE" == "ON" ];then
        cmake --build . --target all -- VERBOSE=1 -j"$thread_num"
    else
        cmake --build . --target all -- -j"$thread_num"
    fi
    if [ -z "$UT_BUILD_TARGETS" ]; then
        cmake --install .
    fi
    cd -
}
