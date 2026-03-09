include(utils.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-subobject-linkage")

#coverage option
option(ENABLE_COVERAGE "Enable coverage" OFF)
message(DEBUG ENABLE_COVERAGE=${ENABLE_COVERAGE})
if (ENABLE_COVERAGE)
    setup_coverage()
endif()

add_library(crash_handler STATIC
    crash_handler.cpp
)

target_link_libraries(crash_handler PUBLIC dl)
target_include_directories(crash_handler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ut)
add_subdirectory(st)

if(USE_FUZZ_TEST)
    add_subdirectory(fuzztest)
endif()