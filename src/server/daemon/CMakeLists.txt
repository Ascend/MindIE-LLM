CMAKE_MINIMUM_REQUIRED(VERSION 3.18)

include_directories(
        ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/include
        ${THIRD_PARTY_OUTPUT_DIR}/http/include
        ${THIRD_PARTY_OUTPUT_DIR}/grpc/include
)

add_link_options(-pie)

set(TARGET_NAME mindieservice_daemon)

add_compile_options(-fpie)
add_executable(${TARGET_NAME} llm_daemon.cpp)

target_link_directories(${TARGET_NAME}
        PUBLIC
        ${THIRD_PARTY_OUTPUT_DIR}/openssl/lib
        ${THIRD_PARTY_OUTPUT_DIR}/spdlog/lib
        ${THIRD_PARTY_OUTPUT_DIR}/boost/${THIRD_PARTY_LIBDIR}
        ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/${THIRD_PARTY_LIBDIR}
)

find_package(pybind11 CONFIG)

target_link_libraries(${TARGET_NAME}
        PUBLIC
        ssl
        crypto
        boost_thread
        boost_chrono
        mindieservice_endpoint
        config_manager
        mindie_llm_utils
        system_log
        pybind11::embed
)

install(TARGETS ${TARGET_NAME} DESTINATION bin)
