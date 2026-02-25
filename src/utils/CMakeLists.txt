file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
#filterout log folder
list(FILTER SOURCE_FILES EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/log/.*")
list(FILTER SOURCE_FILES EXCLUDE REGEX "${CMAKE_CURRENT_LIST_DIR}/system_log/.*")

find_library(LIBRT rt)
message(STATUS "LIBRT: ${LIBRT}")
add_library(mindie_llm_utils SHARED ${SOURCE_FILES})

target_link_libraries(mindie_llm_utils PUBLIC mindie_llm_log ${OPENSSL_LIBRARIES} ${LIBRT} ${PROTOBUF_SO_FILES} ${TORCH_LIBRARIES} stdc++fs pybind11::embed boundscheck)

set_target_properties(mindie_llm_utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

add_subdirectory(log)
add_subdirectory(system_log)
install(TARGETS mindie_llm_utils DESTINATION lib)
