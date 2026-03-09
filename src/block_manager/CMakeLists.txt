file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(mindie_block_manager SHARED ${SOURCE_FILES})

target_link_libraries(mindie_block_manager
        PUBLIC
        sequence
        system_log
)

install(TARGETS mindie_block_manager DESTINATION lib)
