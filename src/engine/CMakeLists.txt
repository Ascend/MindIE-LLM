file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(mindie_engine SHARED ${SOURCE_FILES})

target_link_libraries(mindie_engine PUBLIC
        mindie_scheduler
        mindie_load_balancer
        sequence
        mindie_llm_utils
        mindie_protobuf
        ${Boost_LIBRARIES}
)

install(TARGETS mindie_engine DESTINATION lib)