set(PROTO_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/grpc_proto")
set(DMI_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/grpc_wrapper")
set(ENV{LD_LIBRARY_PATH} ${THIRD_PARTY_OUTPUT_DIR}/grpc/lib:$ENV{LD_LIBRARY_PATH})
set(ENV{PATH} ${THIRD_PARTY_OUTPUT_DIR}/grpc/bin:$ENV{PATH})
# 创建 proto 目录

find_program(GRPC_CPP_PLUGIN NAMES grpc_cpp_plugin REQUIRED)
file(MAKE_DIRECTORY "${PROTO_INSTALL_DIR}")

# 定义 proto 文件和生成的文件
set(PROTO_FILE "${DMI_PROTO_DIR}/prefillAndDecodeCommunication.proto")
set(DST_FILE_ONE "${PROTO_INSTALL_DIR}/prefillAndDecodeCommunication.pb.h")

# 检查是否需要生成文件
if(NOT EXISTS "${DST_FILE_ONE}" OR "${PROTO_FILE}" IS_NEWER_THAN "${DST_FILE_ONE}")
    message(STATUS "Generate cxx and header file to grpc_proto from .proto file...")

    # 生成 gRPC 文件
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I ${DMI_PROTO_DIR} --grpc_out=${PROTO_INSTALL_DIR}
                --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                ${PROTO_FILE}
        RESULT_VARIABLE GRPC_GENERATE_RESULT
        ERROR_VARIABLE GRPC_GENERATE_ERROR
    )

    # 检查 gRPC 生成结果
    if(NOT GRPC_GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate gRPC files: ${GRPC_GENERATE_ERROR}")
    endif()

    # 生成 C++ 文件
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I ${DMI_PROTO_DIR} --cpp_out=${PROTO_INSTALL_DIR}
                ${PROTO_FILE}
        RESULT_VARIABLE CPP_GENERATE_RESULT
        ERROR_VARIABLE CPP_GENERATE_ERROR
    )

    # 检查 C++ 生成结果
    if(NOT CPP_GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate C++ files: ${CPP_GENERATE_ERROR}")
    endif()
else()
    message(STATUS "No need to convert. ${DST_FILE_ONE} is up to date.")
endif()
file(GLOB_RECURSE EndPoint *.cpp *.h)

add_library(mindieservice_endpoint_static STATIC ${EndPoint})
add_compile_options(-fPIC)

include_directories(
        ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/include
        ${THIRD_PARTY_OUTPUT_DIR}/http/include
        ${THIRD_PARTY_OUTPUT_DIR}/grpc/include
        ${THIRD_PARTY_OUTPUT_DIR}/grpc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../config_manager
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/request_response
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc_wrapper
        ${CMAKE_CURRENT_SOURCE_DIR}/single_req_infer_interface
        ${CMAKE_CURRENT_SOURCE_DIR}/health_checker
)

if(DEFINED ENV{MINDIE_LLM_VERSION_OVERRIDE})
    set(MINDIE_VERSION $ENV{MINDIE_LLM_VERSION_OVERRIDE})
    add_definitions(-DMINDIE_VERSION="${MINDIE_VERSION}")
else()
    message(WARNING "Environment MINDIE_LLM_VERSION_OVERRIDE is not set")
endif()

# 编译endpoint的proto文件
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

add_library(pd_grpc_proto_file
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc_proto/prefillAndDecodeCommunication.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc_proto/prefillAndDecodeCommunication.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc_proto/prefillAndDecodeCommunication.grpc.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc_proto/prefillAndDecodeCommunication.grpc.pb.h
)

target_link_libraries(pd_grpc_proto_file
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF}
)

target_link_libraries(mindieservice_endpoint_static
        -Wl,--start-group
        ${Boost_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        config_manager mindie_llm_utils system_log
        mindieservice_infer_instances mindieservice_tokenizer
        -Wl,--end-group
)

link_directories(
    ${THIRD_PARTY_OUTPUT_DIR}/grpc/lib
    ${THIRD_PARTY_OUTPUT_DIR}/prometheus-cpp/lib)
add_library(mindieservice_endpoint SHARED ${EndPoint})
target_link_libraries(mindieservice_endpoint
        PUBLIC
        -Wl,--start-group
        ${Boost_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        ${PROTOBUF_SO_FILES}
        boundscheck
        prometheus-cpp::core
        prometheus-cpp::pull
        mindieservice_infer_instances
        mindie_llm_utils
        system_log
        config_manager
        mindieservice_tokenizer
        pd_grpc_proto_file
        -Wl,--end-group
)

install(TARGETS mindieservice_endpoint_static pd_grpc_proto_file ARCHIVE DESTINATION lib)
install(TARGETS mindieservice_endpoint LIBRARY DESTINATION lib)
