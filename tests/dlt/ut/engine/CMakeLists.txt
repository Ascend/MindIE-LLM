build_test(engine ut "mindie_llm_utils;mindie_engine;mindieservice_endpoint;mindie_scheduler;dl;pybind11::embed;${PROTOBUF_SO_FILES}" "${PROJECT_SOURCE_DIR}/src/include/engine;${PROJECT_SOURCE_DIR}/src/include/executor;${PROJECT_SOURCE_DIR}/src/server/endpoint/health_checker;${PROJECT_SOURCE_DIR}/src/server/infer_instances;${PROJECT_SOURCE_DIR}/src/include/config;${Protobuf_INCLUDE_DIRS}")
set_target_properties(MindIE-LLM_engine_ut PROPERTIES 
  LINK_FLAGS "-Wl,--no-as-needed"
)