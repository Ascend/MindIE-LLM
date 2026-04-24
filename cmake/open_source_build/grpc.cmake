include(${CMAKE_CURRENT_LIST_DIR}/../utils.cmake)

file(READ "${DEPENDENCY_JSON_FILE}" DEP_JSON_STRING)
download_open_source("${OPENSOURCE_COMPONENT_NAME}" "${FILE_GLOB_PATTERN}" "${THIRD_PARTY_CACHE_DIR}")

set(THREAD_NUM "${THREAD_NUM}") # clean warning
set(ABSEILCPP_OUTPUT_DIR "${ABSEILCPP_OUTPUT_DIR}") # clean warning
set(ZLIB_OUTPUT_DIR "${ZLIB_OUTPUT_DIR}") # clean warning
set(PROTOBUF_OUTPUT_DIR "${PROTOBUF_OUTPUT_DIR}") # clean warning
set(OPENSSL_OUTPUT_DIR "${OPENSSL_OUTPUT_DIR}") # clean warning
set(CARES_OUTPUT_DIR "${CARES_OUTPUT_DIR}") # clean warning
set(RE2_OUTPUT_DIR "${RE2_OUTPUT_DIR}") # clean warning
list(JOIN THIRD_PARTY_C_FLAGS " " THIRD_PARTY_C_FLAGS_STR)
list(APPEND THIRD_PARTY_CXX_FLAGS "-Wno-attributes" "-Wno-stringop-overflow" "-Wno-deprecated-declarations")
list(JOIN THIRD_PARTY_CXX_FLAGS " " THIRD_PARTY_CXX_FLAGS_STR)

if(EXISTS "${GRPC_OUTPUT_DIR}/include/grpc/grpc.h")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already built, skipping.")
    return()
endif()

set(PKG_DOWNLOAD_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
file(MAKE_DIRECTORY "${PKG_DOWNLOAD_DIR}/SOURCE")
file(GLOB GRPC_TAR "${PKG_DOWNLOAD_DIR}/*.tar.*")
execute_process(
    COMMAND tar xf ${GRPC_TAR} -C ${PKG_DOWNLOAD_DIR}/SOURCE
    WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR}
)

apply_patches("${PKG_DOWNLOAD_DIR}" "${FILE_GLOB_PATTERN}")

file(GLOB SOURCE_DIR_LIST "${PKG_DOWNLOAD_DIR}/SOURCE/${FILE_GLOB_PATTERN}")
list(GET SOURCE_DIR_LIST 0 SOURCE_DIR)
set(GRPC_BUILD_DIR "${SOURCE_DIR}/build")
file(MAKE_DIRECTORY "${GRPC_BUILD_DIR}")

# eliminate warnings
file(MAKE_DIRECTORY "${SOURCE_DIR}/third_party/opencensus-proto/src")
execute_process(
    COMMAND sed -i
        -e "s/set(CMAKE_C_FLAGS \"\\(.*\\) -Wp,-D_FORTIFY_SOURCE=2 -O2\"/set(CMAKE_C_FLAGS \"\\1\"/"
        -e "s/set(CMAKE_CXX_FLAGS \"\\(.*\\) -Wp,-D_FORTIFY_SOURCE=2 -O2\"/set(CMAKE_CXX_FLAGS \"\\1\"/"
        "${SOURCE_DIR}/CMakeLists.txt"
    RESULT_VARIABLE SED_RESULT
)

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S ${SOURCE_DIR}
        -B ${GRPC_BUILD_DIR}
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${GRPC_OUTPUT_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_FLAGS=${THIRD_PARTY_C_FLAGS_STR}
        -DCMAKE_CXX_FLAGS=${THIRD_PARTY_CXX_FLAGS_STR}

        -DCMAKE_SKIP_BUILD_RPATH=ON
        -DCMAKE_SKIP_INSTALL_RPATH=ON
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF

        -DCMAKE_PREFIX_PATH=${PROTOBUF_OUTPUT_DIR};${ABSEILCPP_OUTPUT_DIR};${ZLIB_OUTPUT_DIR};${CARES_OUTPUT_DIR}

        # gRPC switches
        -DgRPC_BUILD_TESTS=OFF
        -DgRPC_INSTALL=ON
        -DgRPC_BUILD_CODEGEN=ON
        -DgRPC_BUILD_GRPC_CPP_PLUGIN=ON
        -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=ON

        -DgRPC_PROTOBUF_PROVIDER=package
        -DProtobuf_DIR=${PROTOBUF_OUTPUT_DIR}/lib/cmake/protobuf

        -DgRPC_ABSL_PROVIDER=package
        -Dabsl_DIR=${ABSEILCPP_OUTPUT_DIR}/lib/cmake/absl

        -DgRPC_ZLIB_PROVIDER=package
        -DZLIB_ROOT=${ZLIB_OUTPUT_DIR}

        -DgRPC_CARES_PROVIDER=package
        -Dc-ares_DIR=${CARES_OUTPUT_DIR}/lib/cmake/c-ares

        -DgRPC_RE2_PROVIDER=package
        -Dre2_DIR=${RE2_OUTPUT_DIR}/lib/cmake/re2

        -DgRPC_SSL_PROVIDER=package
        -DOPENSSL_ROOT_DIR=${OPENSSL_OUTPUT_DIR}
    RESULT_VARIABLE CONFIG_RESULT
    OUTPUT_QUIET
)
if(NOT CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR "CMake configuration failed for gRPC")
endif()

execute_process(
    COMMAND env LD_LIBRARY_PATH=${GRPC_BUILD_DIR}:${GRPC_BUILD_DIR}/lib:${ABSEILCPP_OUTPUT_DIR}/lib:${PROTOBUF_OUTPUT_DIR}/lib:${ZLIB_OUTPUT_DIR}/lib:${CARES_OUTPUT_DIR}/lib:${RE2_OUTPUT_DIR}/lib:${OPENSSL_OUTPUT_DIR}/lib
    ${CMAKE_COMMAND}
    --build ${GRPC_BUILD_DIR}
    --parallel ${THREAD_NUM}
    RESULT_VARIABLE BUILD_RESULT
    OUTPUT_QUIET
)
if(NOT BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "Build failed for gRPC")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND}
            --install ${GRPC_BUILD_DIR}
            --config Release
    RESULT_VARIABLE INSTALL_RESULT
    OUTPUT_QUIET
)
if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation failed for gRPC")
endif()

file(COPY "${SOURCE_DIR}/src/core" DESTINATION "${GRPC_OUTPUT_DIR}/include/src")
message(STATUS "${OPENSOURCE_COMPONENT_NAME} has been successfully built and installed to ${GRPC_OUTPUT_DIR}")
