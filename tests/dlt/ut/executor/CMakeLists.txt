build_test(executor ut
    "mindie_executor;mindie_llm_utils;pybind11::embed;${PROTOBUF_SO_FILES};${Boost_LIBRARIES};${OPENSSL_LIBRARIES}"
    "${PROJECT_SOURCE_DIR}/src/include/executor;${Protobuf_INCLUDE_DIRS}"
)
