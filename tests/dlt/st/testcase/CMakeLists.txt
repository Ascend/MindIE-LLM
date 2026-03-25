build_test(llm st "mindie_llm_utils;dlt_framework;mindie_block_manager;mindie_scheduler;mindie_engine;mindie_executor;sequence;\
${PROTOBUF_SO_FILES};" "${CMAKE_CURRENT_LIST_DIR}/../;${PROJECT_SOURCE_DIR}/src/include/engine;\
${PROJECT_SOURCE_DIR}/src/include/executor;\
${PROJECT_SOURCE_DIR}/src/scheduler/policy;${Protobuf_INCLUDE_DIRS}")

install(DIRECTORY
  ${CMAKE_CURRENT_LIST_DIR}/../config
  DESTINATION config
)