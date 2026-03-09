configure_file(
    "${PROJECT_SOURCE_DIR}/tests/dlt/ut/utils/config.json"
    "${PROJECT_SOURCE_DIR}/build/tests/dlt/ut/conf/config.json"
    COPYONLY
)
execute_process(
    COMMAND chmod 750 "${PROJECT_SOURCE_DIR}/build/tests/dlt/ut"
)
execute_process(
    COMMAND chmod 640 "${PROJECT_SOURCE_DIR}/build/tests/dlt/ut/conf/config.json"
)
add_definitions(-DMINDIE_LLM_HOME_PATH_TEST="${PROJECT_SOURCE_DIR}/build/tests/dlt/ut") 
build_test(utils ut "system_log;mindie_llm_log;mindie_llm_utils;pybind11::embed" "${PROJECT_SOURCE_DIR}/src/utils")