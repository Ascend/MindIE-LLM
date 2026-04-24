include(${CMAKE_CURRENT_LIST_DIR}/../utils.cmake)
function(_patch_protobuf_cmakelist CMAKELIST_FILE)
    if(NOT EXISTS "${CMAKELIST_FILE}")
        message(WARNING "CMakeLists.txt not found in ${SOURCE_DIR}, skip patch.")
        return()
    endif()

    file(READ "${CMAKELIST_FILE}" CMAKELIST_CONTENTS)
    string(REGEX REPLACE "-fstack-check(=[^ ]*)?" "" CMAKELIST_CONTENTS "${CMAKELIST_CONTENTS}")
    if(NOT CMAKELIST_CONTENTS MATCHES "extra_flags.cmake")
        set(INCLUDE_LINE "include(\"./build/extra_flags.cmake\")\n")
        set(CMAKELIST_CONTENTS "${INCLUDE_LINE}${CMAKELIST_CONTENTS}")
    endif()
    file(WRITE "${CMAKELIST_FILE}" "${CMAKELIST_CONTENTS}")
endfunction()

file(READ "${DEPENDENCY_JSON_FILE}" DEP_JSON_STRING)
download_open_source("${OPENSOURCE_COMPONENT_NAME}" "${FILE_GLOB_PATTERN}" "${THIRD_PARTY_CACHE_DIR}")

set(THREAD_NUM "${THREAD_NUM}") # clean warning
set(ABSEILCPP_OUTPUT_DIR "${ABSEILCPP_OUTPUT_DIR}") # clean warning
set(ZLIB_OUTPUT_DIR "${ZLIB_OUTPUT_DIR}") # clean warning
list(JOIN THIRD_PARTY_CXX_FLAGS " " THIRD_PARTY_CXX_FLAGS_STR)

if(EXISTS "${PROTOBUF_OUTPUT_DIR}/include/google/protobuf/message.h")
    message(STATUS "${OPENSOURCE_COMPONENT_NAME} already built, skipping.")
    return()
endif()

set(PKG_DOWNLOAD_DIR "${THIRD_PARTY_SRC_DIR}/${OPENSOURCE_COMPONENT_NAME}")
file(MAKE_DIRECTORY "${PKG_DOWNLOAD_DIR}/SOURCE")
file(GLOB PROTOBUF_TAR "${PKG_DOWNLOAD_DIR}/*.tar.*")
execute_process(
    COMMAND tar xf ${PROTOBUF_TAR} -C ${PKG_DOWNLOAD_DIR}/SOURCE
    WORKING_DIRECTORY ${PKG_DOWNLOAD_DIR}
)

apply_patches("${PKG_DOWNLOAD_DIR}" "${FILE_GLOB_PATTERN}")
file(GLOB SOURCE_DIR_LIST "${PKG_DOWNLOAD_DIR}/SOURCE/${FILE_GLOB_PATTERN}")
list(GET SOURCE_DIR_LIST 0 SOURCE_DIR)
set(PROTOBUF_BUILD_DIR "${SOURCE_DIR}/build")
file(MAKE_DIRECTORY "${PROTOBUF_BUILD_DIR}")
file(WRITE "${PROTOBUF_BUILD_DIR}/extra_flags.cmake" "
add_compile_options(
${THIRD_PARTY_CXX_FLAGS_STR}
-Wno-attributes
-Wno-deprecated-declarations
)
")
_patch_protobuf_cmakelist(${SOURCE_DIR}/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S ${SOURCE_DIR}
        -B ${PROTOBUF_BUILD_DIR}
        -DCMAKE_INSTALL_PREFIX=${PROTOBUF_OUTPUT_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON

        -DCMAKE_SKIP_BUILD_RPATH=ON
        -DCMAKE_SKIP_INSTALL_RPATH=ON
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF

        -DBUILD_SHARED_LIBS=ON
        -Dprotobuf_BUILD_TESTS=OFF
        -Dprotobuf_BUILD_SHARED_LIBS=ON

        -Dprotobuf_BUILD_LIBPROTOC=ON
        -Dprotobuf_BUILD_PROTOC_BINARIES=ON

        -Dprotobuf_WITH_ZLIB=ON
        -DZLIB_LIBRARY=${ZLIB_OUTPUT_DIR}/lib/libz.so
        -DZLIB_INCLUDE_DIR=${ZLIB_OUTPUT_DIR}/include

        -Dprotobuf_ABSL_PROVIDER=package
        -Dabsl_DIR=${ABSEILCPP_OUTPUT_DIR}/lib/cmake/absl
    OUTPUT_QUIET
)

execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --parallel ${THREAD_NUM}
    WORKING_DIRECTORY ${PROTOBUF_BUILD_DIR}
    OUTPUT_QUIET
)

execute_process(
    COMMAND ${CMAKE_COMMAND} --install . --config Release
    WORKING_DIRECTORY ${PROTOBUF_BUILD_DIR}
    OUTPUT_QUIET
)
message(STATUS "${OPENSOURCE_COMPONENT_NAME} has been successfully built and installed to ${PROTOBUF_OUTPUT_DIR}")
