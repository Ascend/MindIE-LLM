build_test(server_infer_instances ut "mindie_llm_utils;mindieservice_infer_instances;mindieservice_endpoint;" "${PROJECT_SOURCE_DIR}/src/server/include;${PROJECT_SOURCE_DIR}/src/include/config;${PROJECT_SOURCE_DIR}/src/server/infer_instances;${PROJECT_SOURCE_DIR}/src/config_manager;${PROJECT_SOURCE_DIR}/tests/dlt/ut/server/utils")
set_target_properties(MindIE-LLM_server_infer_instances_ut PROPERTIES 
  LINK_FLAGS "-Wl,--no-as-needed"
)