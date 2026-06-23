set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROTO_FILE "${PROTO_PATH}/model_execute_data.proto")
set(ENV{LD_LIBRARY_PATH} ${THIRD_PARTY_OUTPUT_DIR}/protobuf/${THIRD_PARTY_LIBDIR}:${THIRD_PARTY_OUTPUT_DIR}/abseil-cpp/${THIRD_PARTY_LIBDIR}:${THIRD_PARTY_OUTPUT_DIR}/grpc/lib:$ENV{LD_LIBRARY_PATH})
set(ENV{PATH} ${THIRD_PARTY_OUTPUT_DIR}/protobuf/bin:$ENV{PATH})
find_program(GRPC_CPP_PLUGIN NAMES grpc_cpp_plugin REQUIRED)

set(PROTO_SRCS "${PROTO_PATH}/model_execute_data.pb.cc")
set(PROTO_HDRS "${PROTO_PATH}/model_execute_data.pb.h")
if(NOT EXISTS "${PROTO_SRCS}" OR "${PROTO_FILE}" IS_NEWER_THAN "${PROTO_SRCS}")
    message(STATUS "Generating C++ code from model_execute_data.proto")
    # 生成 C++ 文件
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I ${PROTO_PATH} --cpp_out=${PROTO_PATH} --experimental_allow_proto3_optional
                ${PROTO_FILE}
        RESULT_VARIABLE CPP_GENERATE_RESULT
        ERROR_VARIABLE CPP_GENERATE_ERROR
    )

    # 检查 C++ 生成结果
    if(NOT CPP_GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate C++ files: ${CPP_GENERATE_ERROR}")
    endif()
else()
    message(STATUS "No need to convert. ${PROTO_SRCS} is up to date.")
endif()

add_library(mindie_protobuf STATIC ${PROTO_SRCS} ${PROTO_HDRS})

set_target_properties(mindie_protobuf PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

target_link_libraries(mindie_protobuf
        ${PROTOBUF_SO_FILES}
)

install(TARGETS mindie_protobuf DESTINATION lib)


set(GRPC_DST_FILE "${PROTO_PATH}/model_execute_data.grpc.pb.h")

if(NOT EXISTS "${GRPC_DST_FILE}" OR "${PROTO_FILE}" IS_NEWER_THAN "${GRPC_DST_FILE}")
    message(STATUS "Generate grpc header file to proto from model_execute_data.proto file...")

    # 生成 gRPC 文件
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I ${PROTO_PATH} --grpc_out=${PROTO_PATH}
                --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                ${PROTO_FILE}
        RESULT_VARIABLE GRPC_GENERATE_RESULT
        ERROR_VARIABLE GRPC_GENERATE_ERROR
    )

    # 检查 gRPC 生成结果
    if(NOT GRPC_GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate gRPC files: ${GRPC_GENERATE_ERROR}")
    endif()
else()
    message(STATUS "No need to convert. ${GRPC_DST_FILE} is up to date.")
endif()

add_library(mindie_grpc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_execute_data.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_execute_data.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/model_execute_data.grpc.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_execute_data.grpc.pb.h
)

target_link_libraries(mindie_grpc
        ${PROTOBUF_SO_FILES}
)
set_target_properties(mindie_grpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
install(TARGETS mindie_grpc DESTINATION lib)

set(PY_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mindie_llm/connector/common)
set(GENERATED_PY ${PY_OUT_DIR}/model_execute_data_pb2.py)
if(NOT EXISTS "${GENERATED_PY}" OR "${PROTO_FILE}" IS_NEWER_THAN "${GENERATED_PY}")
    message(STATUS "Generate grpc header file to proto from model_execute_data.proto file...")
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I ${PROTO_PATH}
                --experimental_allow_proto3_optional
                --python_out=${PY_OUT_DIR}
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
                ${PROTO_FILE}
        RESULT_VARIABLE GRPC_GENERATE_RESULT
        ERROR_VARIABLE GRPC_GENERATE_ERROR
    )
endif()
