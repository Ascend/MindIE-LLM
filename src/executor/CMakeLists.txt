file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(mindie_executor SHARED ${SOURCE_FILES})

target_include_directories(mindie_executor
    PUBLIC
    ${THIRD_PARTY_OUTPUT_DIR}/grpc/include
    ${THIRD_PARTY_OUTPUT_DIR}/openssl/include
)
target_link_libraries(mindie_executor
    system_log
    ${PROTOBUF_SO_FILES}
    ${RE2_SO_FILES}
    ${GRPC_SO_FILES}
    mindie_protobuf
    mindie_grpc
    ${Boost_LIBRARIES}
    stdc++fs
    config_manager
)

install(TARGETS mindie_executor DESTINATION lib)