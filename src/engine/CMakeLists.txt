file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(mindie_engine SHARED ${SOURCE_FILES})

target_include_directories(mindie_engine PRIVATE
${CMAKE_SOURCE_DIR}/src/server/endpoint/health_checker
${CMAKE_SOURCE_DIR}/src/server/infer_instances
${CMAKE_SOURCE_DIR}/src/include/utils
)

target_link_libraries(mindie_engine PUBLIC
        mindie_scheduler
        mindie_load_balancer
        sequence
        mindie_llm_utils
        mindie_protobuf
        ${Boost_LIBRARIES}
)

install(TARGETS mindie_engine DESTINATION lib)