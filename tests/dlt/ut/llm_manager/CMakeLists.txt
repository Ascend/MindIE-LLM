build_test(llm_manager ut "mindie_llm_manager_v2;mindie_llm_manager;llm_manager_python;mindieservice_endpoint" "")
set_target_properties(MindIE-LLM_llm_manager_ut PROPERTIES 
  LINK_FLAGS "-Wl,--no-as-needed"
)