file(GLOB SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(${command_output})
set(pybind11_DIR "${command_output}")
find_package(pybind11 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/include/utils
)

add_library(memory_bridge SHARED ${SOURCE_FILES})
set_target_properties(memory_bridge PROPERTIES PREFIX "_")

target_link_directories(memory_bridge
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/utils/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src/utils/log/lib
)

target_link_libraries(memory_bridge PUBLIC system_log mindie_llm_utils ascendcl stdc++ pybind11::pybind11)

install(TARGETS memory_bridge DESTINATION lib)
