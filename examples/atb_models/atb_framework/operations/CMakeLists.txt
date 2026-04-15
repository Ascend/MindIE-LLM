file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_library(atb_speed_operations SHARED ${SOURCE_FILES})
link_directories(
    ${CMAKE_CURRENT_LIST_DIR}/../../../../src/utils/log/lib
)
set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/aclnn/ops/attn_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/aclnn/ops/attn_v3_operation.cpp
    PROPERTIES
    COMPILE_FLAGS "-Wno-error=cpp"
)
target_link_libraries(atb_speed_operations PRIVATE atb_log atb opapi stdc++fs)
install(TARGETS atb_speed_operations DESTINATION lib)
