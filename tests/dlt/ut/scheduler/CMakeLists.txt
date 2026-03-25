build_test(scheduler ut "mindie_scheduler;sequence;mindie_engine;pybind11::embed" "${PROJECT_SOURCE_DIR}/src/scheduler/policy;\
${PROJECT_SOURCE_DIR}/src/scheduler/latency_predictor;\
${PROJECT_SOURCE_DIR}/src/scheduler/policy/stage_policy")