function(add_builder target_name component_name FILE_GLOB_PATTERN)
    set(BUILDER_DIR "${CMAKE_BINARY_DIR}/${component_name}")
    file(REMOVE_RECURSE "${BUILDER_DIR}")
    file(MAKE_DIRECTORY "${BUILDER_DIR}")

    file(WRITE "${BUILDER_DIR}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.19.0)\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/open_source_build/${component_name}.cmake\")\n"
    )
    add_custom_target(${target_name}
        COMMAND +${CMAKE_COMMAND}
            --log-level=ERROR
            -Wno-dev
            -DDEPENDENCY_JSON_FILE="${DEPENDENCY_JSON_FILE}"
            -DTHIRD_PARTY_CACHE_DIR="${THIRD_PARTY_CACHE_DIR}"
            -DTHIRD_PARTY_SRC_DIR="${THIRD_PARTY_SRC_DIR}"
            -DOPENSOURCE_COMPONENT_NAME="${component_name}"
            -DFILE_GLOB_PATTERN="${FILE_GLOB_PATTERN}"
            -DTHREAD_NUM="${thread_num}"
            ${ARGN}
            -S "${BUILDER_DIR}"
            -B "${BUILDER_DIR}/build"
        COMMENT "Building ${component_name}..."
    )
endfunction()

add_builder(zlib_pkg "zlib" "zlib-*"
    -DZLIB_OUTPUT_DIR="${ZLIB_OUTPUT_DIR}"
    -DTHIRD_PARTY_C_FLAGS="${THIRD_PARTY_C_FLAGS}"
)
add_builder(abseil_pkg "abseil-cpp" "abseil-cpp-*"
    -DABSEILCPP_OUTPUT_DIR="${ABSEILCPP_OUTPUT_DIR}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
)
add_builder(re2_pkg "re2" ""
    -DRE2_OUTPUT_DIR="${RE2_OUTPUT_DIR}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
    -DABSEILCPP_OUTPUT_DIR="${ABSEILCPP_OUTPUT_DIR}"
)
add_builder(protobuf_pkg "protobuf" "protobuf-*"
    -DPROTOBUF_OUTPUT_DIR="${PROTOBUF_OUTPUT_DIR}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
    -DZLIB_OUTPUT_DIR="${ZLIB_OUTPUT_DIR}"
    -DABSEILCPP_OUTPUT_DIR="${ABSEILCPP_OUTPUT_DIR}"
)
add_builder(openssl_pkg "openssl" "ssl/bio_ssl.c"
    -DOPENSSL_OUTPUT_DIR="${OPENSSL_OUTPUT_DIR}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
)
add_builder(cares_pkg "cares" "c-ares-*"
    -DCARES_OUTPUT_DIR="${CARES_OUTPUT_DIR}"
    -DTHIRD_PARTY_C_FLAGS="${THIRD_PARTY_C_FLAGS}"
)
add_builder(grpc_pkg "grpc" "grpc-*"
    -DGRPC_OUTPUT_DIR="${GRPC_OUTPUT_DIR}"
    -DTHIRD_PARTY_C_FLAGS="${THIRD_PARTY_C_FLAGS}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
    -DABSEILCPP_OUTPUT_DIR="${ABSEILCPP_OUTPUT_DIR}"
    -DPROTOBUF_OUTPUT_DIR="${PROTOBUF_OUTPUT_DIR}"
    -DZLIB_OUTPUT_DIR="${ZLIB_OUTPUT_DIR}"
    -DRE2_OUTPUT_DIR="${RE2_OUTPUT_DIR}"
    -DOPENSSL_OUTPUT_DIR="${OPENSSL_OUTPUT_DIR}"
    -DCARES_OUTPUT_DIR="${CARES_OUTPUT_DIR}"
)
add_builder(boost_pkg "boost" "bootstrap.sh"
    -DBOOST_OUTPUT_DIR="${BOOST_OUTPUT_DIR}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
)
add_builder(prometheus_pkg "prometheus-cpp" "CMakeLists.txt"
    -DPROMETHEUS_OUTPUT_DIR="${PROMETHEUS_OUTPUT_DIR}"
    -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
)
add_builder(libboundscheck_pkg "libboundscheck" "Makefile"
    -DLIBBOUNDSCHECK_OUTPUT_DIR="${LIBBOUNDSCHECK_OUTPUT_DIR}"
)
add_builder(tiny_pkg "copy_tiny_pkg" ""
    -DTHIRD_PARTY_OUTPUT_DIR="${THIRD_PARTY_OUTPUT_DIR}"
)

add_dependencies(re2_pkg abseil_pkg)
add_dependencies(protobuf_pkg abseil_pkg zlib_pkg)
add_dependencies(grpc_pkg protobuf_pkg re2_pkg openssl_pkg cares_pkg)

set(PKG_LIST zlib_pkg abseil_pkg re2_pkg protobuf_pkg openssl_pkg cares_pkg grpc_pkg boost_pkg prometheus_pkg
    libboundscheck_pkg tiny_pkg
)

if(NOT DEFINED ENV{BUILD_ZONE} OR NOT "$ENV{BUILD_ZONE}" STREQUAL "yellow")
    if(USE_FUZZ_TEST OR DOMAIN_LAYERED_TEST)
        add_builder(mockcpp_pkg "mockcpp" "CMakeLists.txt"
            -DMOCKCPP_OUTPUT_DIR="${MOCKCPP_OUTPUT_DIR}"
            -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
        )
        add_builder(googletest_pkg "googletest" "CMakeLists.txt"
            -DGTEST_OUTPUT_DIR="${GTEST_OUTPUT_DIR}"
            -DTHIRD_PARTY_CXX_FLAGS="${THIRD_PARTY_CXX_FLAGS}"
        )
        list(APPEND PKG_LIST mockcpp_pkg googletest_pkg)
    endif()
endif()
