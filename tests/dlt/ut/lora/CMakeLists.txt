build_test(lora ut 
  "mindie_engine;mindieservice_endpoint"
  "${PROJECT_SOURCE_DIR}/src/include/lora"
)

set_target_properties(MindIE-LLM_lora_ut PROPERTIES 
  LINK_FLAGS "-Wl,--no-as-needed"
)