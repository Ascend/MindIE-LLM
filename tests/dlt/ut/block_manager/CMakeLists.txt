build_test(block_manager ut "mindie_block_manager;sequence;mindie_engine;mindieservice_endpoint;pybind11::embed" "")

set_target_properties(MindIE-LLM_block_manager_ut PROPERTIES 
  LINK_FLAGS "-Wl,--no-as-needed"
)