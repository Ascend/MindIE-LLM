set(ENABLE_CPU "on")
set(ENABLE_D "on")
add_compile_definitions(ENABLE_CPU)
add_compile_definitions(ENABLE_D)

file(GLOB SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp" "log/*.cpp")

message("main source ${SOURCE_FILES}")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src/include/utils
)

add_library(cpu_logits_handler SHARED ${SOURCE_FILES})
set_target_properties(cpu_logits_handler PROPERTIES PREFIX "_")
target_link_directories(cpu_logits_handler
        PUBLIC
        $ENV{ASCEND_HOME_PATH}/runtime/lib64/stub
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src/utils/lib)

target_link_libraries(cpu_logits_handler PUBLIC
        -Wl,--start-group
        -lpthread
        -Wl,--end-group
        system_log
        mindie_llm_utils
        ascendcl
        ${Python3_LIBRARIES}
        boundscheck
)

install(TARGETS cpu_logits_handler DESTINATION lib)
