file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(mindie_scheduler SHARED ${SOURCE_FILES})

target_link_libraries(mindie_scheduler
        mindie_block_manager
        sequence
        mindie_llm_utils
        system_log
)

install(TARGETS mindie_scheduler DESTINATION lib)
