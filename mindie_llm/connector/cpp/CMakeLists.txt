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
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/include/utils
)

add_library(mindie_llm_connector SHARED ${SOURCE_FILES})
set_target_properties(mindie_llm_connector PROPERTIES PREFIX "_")

target_link_directories(mindie_llm_connector
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/utils/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/utils/log/lib
)

target_link_libraries(mindie_llm_connector PUBLIC system_log mindie_llm_utils
    mindie_protobuf ${Boost_LIBRARIES} stdc++ pybind11::pybind11)

install(TARGETS mindie_llm_connector DESTINATION lib)
