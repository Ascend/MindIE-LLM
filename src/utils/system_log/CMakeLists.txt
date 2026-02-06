cmake_minimum_required(VERSION 3.19.0)
project("system_log")
set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCE_FILES 
    "${CMAKE_CURRENT_LIST_DIR}/system_log.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../safe_path.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../safe_envvar.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../safe_result.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../string_utils.cpp"
)
set(LIB_NAME "system_log")
add_library(${LIB_NAME} SHARED ${SOURCE_FILES})
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../")
include_directories(
    ${SOURCE_DIR}/src/include/utils
    ${SOURCE_DIR}/third_party/output/nlohmannJson/include
)
target_link_libraries(${LIB_NAME} PRIVATE stdc++fs)
install(TARGETS ${LIB_NAME} DESTINATION lib)
