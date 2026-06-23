#!/bin/bash

PACKAGE_NAME=""

function fn_get_version()
{
    PACKAGE_NAME=${MINDIE_LLM_VERSION_OVERRIDE:-1.0.0}
}

function fn_make_run_package()
{
    fn_get_version
    if [ -f /etc/openEuler-release ]; then
        LIBDIR="lib64"
    else
        LIBDIR="lib"
    fi
    mkdir -p $OUTPUT_DIR/scripts $OUTPUT_DIR/lib $RELEASE_DIR/$ARCH $OUTPUT_DIR/conf $OUTPUT_DIR/server/scripts/
    cp $CODE_ROOT/scripts/install.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/build_run_package/set_env.sh $OUTPUT_DIR
    cp $CODE_ROOT/scripts/uninstall.sh $OUTPUT_DIR/scripts
    cp $CODE_ROOT/src/server/conf/config.json $OUTPUT_DIR/conf
    cp -r $CODE_ROOT/src/server/scripts/* $OUTPUT_DIR/server/scripts
    cp $CODE_ROOT/scripts/eula.txt $OUTPUT_DIR

    protobuf_version="so.25.1.0"
    protobuf_so_list=(
        "libprotobuf.${protobuf_version}"
        "libprotobuf-lite.${protobuf_version}"
        "libprotoc.${protobuf_version}"
    )

    absl_so_list=(libabsl_*.so.2308.0.0)

    boost_version="so.1.87.0"
    boost_so_list=(
        "libboost_system.${boost_version}"
        "libboost_thread.${boost_version}"
        "libboost_chrono.${boost_version}"
    )

    zlib_so_list=("libz.so.1")
    re2_so_list=("libre2.so.11")
    cares_so_list=("libcares.so.2")
    prometheus_so_list=("libprometheus-cpp-core.so.1.3")
    grpc_version="so.37"
    grpcpp_version="so.1.60"
    grpc_so_list=(lib*."${grpcpp_version}" lib*."${grpc_version}")
    libboundscheck_so_list=("libboundscheck.so")

    copy_so() {
        local src_dir="$1"
        local dst_dir="$2"
        shift 2
        local patterns=("$@")

        mkdir -p "${dst_dir}"
        if [ ${#patterns[@]} -eq 0 ]; then
            patterns=("*.so*")
        fi

        (
            cd "${src_dir}"
            shopt -s nullglob
            for pattern in "${patterns[@]}"; do
                for file in ${pattern}; do
                    if [ -f "${file}" ] && [[ "${file}" == *.so* ]]; then
                        cp -p "${file}" "${dst_dir}/"
                    fi
                done
            done
            shopt -u nullglob
        )
    }

    copy_so "$THIRD_PARTY_OUTPUT_DIR/protobuf/$LIBDIR"       "$OUTPUT_DIR/lib/protobuf"       "${protobuf_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/abseil-cpp/$LIBDIR"     "$OUTPUT_DIR/lib/absl"           "${absl_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/boost/lib"              "$OUTPUT_DIR/lib/boost"          "${boost_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/zlib/lib"               "$OUTPUT_DIR/lib/zlib"           "${zlib_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/re2/$LIBDIR"            "$OUTPUT_DIR/lib/re2"            "${re2_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/cares/$LIBDIR"          "$OUTPUT_DIR/lib/cares"          "${cares_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/prometheus-cpp/$LIBDIR" "$OUTPUT_DIR/lib/prometheus-cpp" "${prometheus_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/grpc/lib"               "$OUTPUT_DIR/lib/grpc"           "${grpc_so_list[@]}"
    copy_so "$THIRD_PARTY_OUTPUT_DIR/libboundscheck/lib" \
        "$OUTPUT_DIR/lib/libboundscheck" "${libboundscheck_so_list[@]}"

    sed -i "s/MINDIELLMPKGARCH/${ARCH}/" $OUTPUT_DIR/install.sh
    sed -i "s!VERSION_PLACEHOLDER!${PACKAGE_NAME}!" $OUTPUT_DIR/install.sh $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_PATH_PLACEHOLDER!${LOG_PATH}!" $OUTPUT_DIR/install.sh $OUTPUT_DIR/scripts/uninstall.sh
    sed -i "s!LOG_NAME_PLACEHOLDER!${LOG_NAME}!" $OUTPUT_DIR/install.sh $OUTPUT_DIR/scripts/uninstall.sh

    sed -i "s/ATBMODELSETENV/latest\/atb_models\/set_env.sh/" $OUTPUT_DIR/set_env.sh
    sed -i 's|${mindie_llm_path}|${mindie_llm_path}/latest|g' $OUTPUT_DIR/set_env.sh

    # makeself ascend-mindie-llm.run
    TMP_VERSION=$(python3 -c 'import sys; print(sys.version_info[0], ".", sys.version_info[1])' | tr -d ' ')
    PY_MINOR_VERSION=${TMP_VERSION##*.}
    PY_VERSION="py3$PY_MINOR_VERSION"
    chmod +x $OUTPUT_DIR/*
    $THIRD_PARTY_OUTPUT_DIR/makeself/makeself.sh --header $CODE_ROOT/scripts/makeself-header.sh \
       --help-header $CODE_ROOT/scripts/help.info --gzip --complevel 4 --nomd5 --sha256 --chown --tar-format gnu \
        ${OUTPUT_DIR} $RELEASE_DIR/$ARCH/Ascend-mindie-llm_${PACKAGE_NAME}_${PY_VERSION}_linux-${ARCH}.run "Ascend-mindie-llm" ./install.sh

    mv $RELEASE_DIR/$ARCH $OUTPUT_DIR
    echo "Ascend-mindie-llm_${PACKAGE_NAME}_${PY_VERSION}_linux-${ARCH}.run is successfully generated in $OUTPUT_DIR"
}

function fn_make_debug_symbols_package() {
    fn_get_version
    mkdir -p "$OUTPUT_DIR/debug_symbols"
    debug_symbols_package_name="$OUTPUT_DIR/debug_symbols/Ascend-mindie-llm-debug-symbols_${PACKAGE_NAME}_${PY_VERSION}_linux-${ARCH}.tar.gz"
    cd "$CODE_ROOT"
    tar czpf $debug_symbols_package_name llm_debug_symbols
    echo "Build tar package for llm debug symbols: $debug_symbols_package_name"
    cd -
}

function fn_make_whl() {
    # PACKAGE_NAME=$(echo $PACKAGE_NAME | sed -E 's/([0-9]+)\.([0-9]+)\.RC([0-9]+)\.([0-9]+)/\1.\2rc\3.post\4/')
    # PACKAGE_NAME=$(echo $PACKAGE_NAME | sed -s 's!.T!.alpha!')
    fn_get_version
    echo "MindIELLMWHLVersion $PACKAGE_NAME"
    echo "make mindie-llm whl package"
    cd $CODE_ROOT
    python3 setup_mindie_llm.py --setup_cmd="bdist_wheel" --version=${PACKAGE_NAME}
    cp dist/*.whl $OUTPUT_DIR
    rm -rf dist mindie_llm.egg-info
    cd -
    if [ "$build_type" = "release" ]; then
        cd $CODE_ROOT/tools
        cp $OUTPUT_DIR/lib/llm_manager_python.so $CODE_ROOT/tools/llm_manager_python_api_demo
        python3 setup.py --setup_cmd="bdist_wheel" --version=${PACKAGE_NAME}
        cp dist/*.whl $OUTPUT_DIR
        rm -rf dist llm_manager_python_api_demo.egg-info
        cd -
    fi
}

function fn_build_for_ci()
{
    cd $OUTPUT_DIR
    mkdir -p include
    mkdir -p include/utils
    cp -r $CODE_ROOT/src/include/llm_manager ./include/

    cp $CODE_ROOT/src/include/utils/error.h ./include/utils/
    cp $CODE_ROOT/src/include/utils/status.h ./include/utils/
    cp $CODE_ROOT/src/include/utils/concurrent_deque.h ./include/utils/
}
