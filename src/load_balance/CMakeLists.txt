file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(mindie_load_balancer SHARED ${SOURCE_FILES})

target_link_libraries(mindie_load_balancer
        sequence
        mindie_llm_utils
        system_log
)

install(TARGETS mindie_load_balancer DESTINATION lib)
