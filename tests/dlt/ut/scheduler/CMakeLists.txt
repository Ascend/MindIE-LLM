build_test(scheduler ut "mindie_scheduler;sequence;mindie_engine;mindieservice_endpoint;pybind11::embed" "${PROJECT_SOURCE_DIR}/src/scheduler/policy;\
${PROJECT_SOURCE_DIR}/src/scheduler/latency_predictor;\
${PROJECT_SOURCE_DIR}/src/scheduler/policy/stage_policy")
set_target_properties(MindIE-LLM_scheduler_ut PROPERTIES 
  LINK_FLAGS "-Wl,--no-as-needed"
)
