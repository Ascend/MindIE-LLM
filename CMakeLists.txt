cmake_minimum_required(VERSION 3.19.0)
project("MindIE-LLM")
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source build is forbidden!\n"
        "Please delete created:['CMakeFiles', 'CMakeCache.txt'] and create a new dir for build, such as: \n"
        "mkdir build & cd build\n"
        "cmake ..\n"
    )
endif()

include(${PROJECT_SOURCE_DIR}/cmake/utils.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/default_config.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/compile_options.cmake)

ensure_aslr_level(2)

set(prometheus-cpp_DIR         "${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/${THIRD_PARTY_LIBDIR}/cmake/prometheus-cpp")
set(Boost_DIR                  "${THIRD_PARTY_OUTPUT_DIR}/boost/lib/cmake/Boost-1.87.0")
set(OPENSSL_ROOT_DIR           "${THIRD_PARTY_OUTPUT_DIR}/openssl")
set(gRPC_DIR                   "${THIRD_PARTY_OUTPUT_DIR}/grpc/lib/cmake/grpc")
set(absl_DIR                   "${THIRD_PARTY_OUTPUT_DIR}/abseil-cpp/${THIRD_PARTY_LIBDIR}/cmake/absl")
set(ZLIB_ROOT                  "${THIRD_PARTY_OUTPUT_DIR}/zlib")
set(c-ares_DIR                 "${THIRD_PARTY_OUTPUT_DIR}/cares/${THIRD_PARTY_LIBDIR}/cmake/c-ares")
set(utf8_range_DIR             "${THIRD_PARTY_OUTPUT_DIR}/protobuf/${THIRD_PARTY_LIBDIR}/cmake/utf8_range")
set(Protobuf_DIR               "${THIRD_PARTY_OUTPUT_DIR}/protobuf/${THIRD_PARTY_LIBDIR}/cmake/protobuf")
set(re2_DIR                    "${THIRD_PARTY_OUTPUT_DIR}/re2/${THIRD_PARTY_LIBDIR}/cmake/re2")

find_package(ZLIB REQUIRED)
find_package(prometheus-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread chrono)
find_package(absl REQUIRED CONFIG)
find_package(re2 REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)
find_package(Protobuf REQUIRED CONFIG) # protobuf和libtorch有先后关系
find_package(gRPC REQUIRED CONFIG)

find_pytorch(PYTORCH_INSTALL_PATH)
set(PYTORCH_EXTRA_LIB_SO "${PYTORCH_INSTALL_PATH}/../torch.libs")
find_package(Torch REQUIRED PATHS "${PYTORCH_INSTALL_PATH}/share/cmake/Torch")

execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${pybind11_cmake_dir}")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(Threads REQUIRED)

# 项目自身的头文件目录（不使用SYSTEM标记，保持警告）
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/include/utils
    ${PROJECT_SOURCE_DIR}/src/utils/include
    ${PROJECT_SOURCE_DIR}/src/utils/log
    ${PROJECT_SOURCE_DIR}/mindie_llm/connector/cpp/
    ${PROJECT_SOURCE_DIR}/mindie_llm/text_generator/cpp/memory_bridge
    ${PROJECT_SOURCE_DIR}/mindie_llm/text_generator/cpp/sampler/cpu_logits_handler
    ${PROJECT_SOURCE_DIR}/mindie_llm/text_generator/cpp/sampler/cpu_logits_handler/log
    ${PROJECT_SOURCE_DIR}/src/include/llm_manager_v2
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/include/dataclass
    ${PROJECT_SOURCE_DIR}/src/include/config
    ${PROJECT_SOURCE_DIR}/src/llm_manager_v2/include/impl
    ${PROJECT_SOURCE_DIR}/src/llm_manager
    ${PROJECT_SOURCE_DIR}/src/config_manager
    ${PROJECT_SOURCE_DIR}/src/include/llm_manager
    ${PROJECT_SOURCE_DIR}/src/include/block_manager
    ${PROJECT_SOURCE_DIR}/src/include/scheduler
    ${PROJECT_SOURCE_DIR}/src/block_manager
    ${PROJECT_SOURCE_DIR}/src/scheduler
    ${PROJECT_SOURCE_DIR}/src/engine
    ${PROJECT_SOURCE_DIR}/src/executor
    ${PROJECT_SOURCE_DIR}/src/include/load_balance
    ${PROJECT_SOURCE_DIR}/src/server/infer_instances
    ${PROJECT_SOURCE_DIR}/src/server/common
    ${PROJECT_SOURCE_DIR}/src/include/request_response
    ${PROJECT_SOURCE_DIR}/proto
)

# 第三方库头文件使用SYSTEM标记来抑制警告
include_directories(SYSTEM
    ${THIRD_PARTY_OUTPUT_DIR}/abseil-cpp/include
    ${THIRD_PARTY_OUTPUT_DIR}/boost/include
    ${THIRD_PARTY_OUTPUT_DIR}/cares/include
    ${THIRD_PARTY_OUTPUT_DIR}/gloo
    ${THIRD_PARTY_OUTPUT_DIR}/grpc/include
    ${THIRD_PARTY_OUTPUT_DIR}/gtest/include
    ${THIRD_PARTY_OUTPUT_DIR}/http
    ${THIRD_PARTY_OUTPUT_DIR}/libboundscheck/include
    ${THIRD_PARTY_OUTPUT_DIR}/mockcpp/include
    ${THIRD_PARTY_OUTPUT_DIR}/nlohmannJson/include
    ${THIRD_PARTY_OUTPUT_DIR}/protobuf/include
    ${THIRD_PARTY_OUTPUT_DIR}/re2/include
    ${THIRD_PARTY_OUTPUT_DIR}/spdlog/include
    ${pybind11_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/runtime/include
    ${PROJECT_SOURCE_DIR}/src/include/msServiceProfiler/include
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
    ${THIRD_PARTY_OUTPUT_DIR}/libboundscheck/lib
)

add_subdirectory(proto)
add_subdirectory(src)

if(DEFINED ENV{ASCEND_HOME_PATH})
    message("Detected that 'CANN' is installed, starting to compile the module 'text_generator'.")
    add_subdirectory(mindie_llm/text_generator/cpp)
    add_subdirectory(mindie_llm/connector/cpp)
endif()

if(USE_PYTHONTEST_TEST OR USE_FUZZ_TEST OR DOMAIN_LAYERED_TEST)
    add_subdirectory(tests)
endif()

if(USE_HITESTWRAPPER)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE hitestwrapper)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK hitestwrapper)
endif()
