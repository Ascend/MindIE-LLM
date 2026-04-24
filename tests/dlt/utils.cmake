find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)
find_program(GCOVR_PATH gcovr)
find_program(POWERSHELL_PATH powershell.exe)
find_program(Python3 python3)

if (CMAKE_CROSSCOMPILING)
    set(GCOV_PATH "$ENV{GCOV_PATH}")
endif()

function(setup_coverage)
    if (NOT GCOV_PATH)
        message(FATAL_ERROR "gcov not found")
    else()
        message(STATUS "gcov found at ${GCOV_PATH}")
    endif()

    set(COVERAGE_COMPILER_FLAGS "--coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_COMPILER_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILER_FLAGS}" PARENT_SCOPE)
    link_libraries(gcov)
    set(COVERAGE_DIR ${CMAKE_BINARY_DIR}/coverage)

    execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/coverage
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    if(LCOV_PATH)
        message(STATUS "lcov found at ${LCOV_PATH}")
        set(HTML_PATH "${COVERAGE_DIR}/index.html")
        add_custom_target(coverage
            COMMAND ${LCOV_PATH} --rc lcov_branch_coverage=1 --capture --directory . --output-file ${COVERAGE_DIR}/${TEST_MODULE}_coverage.info >> ${COVERAGE_DIR}/${TEST_MODULE}_coverage.log
            COMMAND ${LCOV_PATH} --rc lcov_branch_coverage=1 --remove ${COVERAGE_DIR}/${TEST_MODULE}_coverage.info '*/third_party/*' '*/tests/*' '*/proto/*' -o ${COVERAGE_DIR}/${TEST_MODULE}_coverage.info >> ${COVERAGE_DIR}/${TEST_MODULE}_coverage.log
            COMMAND ${GENHTML_PATH} ${COVERAGE_DIR}/${TEST_MODULE}_coverage.info -o ${COVERAGE_DIR} --rc lcov_branch_coverage=1 >> ${COVERAGE_DIR}/${TEST_MODULE}_coverage.log
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    elseif(GCOVR_PATH)
        message(STATUS "gcovr found at ${GCOVR_PATH}")
        set(HTML_PATH "${COVERAGE_DIR}/index.html")
        add_custom_target(coverage
            COMMAND ${GCOVR_PATH} -f src --gcov-exclude '.+log.+' --html-details ${HTML_PATH}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endif()

    if (POWSHELL_PATH)
        message(STATUS "powershell found at ${POWERSHELL_PATH}")
        add_custom_command(TARGET coverage POST_BUILD
            COMMAND ${POWERSHELL_PATH} /c start build/coverage/index.html;
            COMMENT "Open ${HTML_PATH} in your browser to view the coverage report."
            WORKING_DIRECTORY ${{PROJECT_SOURCE_DIR}}
        )
    else()
        add_custom_command(TARGET coverage POST_BUILD
            COMMAND ;
            COMMENT "Open ${HTML_PATH} in your browser to view the coverage report."
        )
    endif()
endfunction()

# 添加明确的分隔符（如空格）
function(build_test module type list_libraries list_includes)
    set(TEST_BINARY ${CMAKE_PROJECT_NAME}_${module}_${type})

    file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    )

    add_executable(${TEST_BINARY} ${TEST_SOURCES})

    target_compile_options(${TEST_BINARY} PRIVATE -fno-access-control)

    target_link_options(${TEST_BINARY} PRIVATE
        -rdynamic
        -Wl,--no-as-needed
        -Wl,-rpath-link,${THIRD_PARTY_OUTPUT_DIR}/abseil-cpp/lib
        -Wl,-rpath-link,${THIRD_PARTY_OUTPUT_DIR}/grpc/lib
        -Wl,-rpath-link,${THIRD_PARTY_OUTPUT_DIR}/protobuf/lib
        -Wl,-rpath-link,${THIRD_PARTY_OUTPUT_DIR}/cares/lib
        -Wl,-rpath-link,${THIRD_PARTY_OUTPUT_DIR}/re2/lib
        -Wl,-rpath-link,${THIRD_PARTY_OUTPUT_DIR}/zlib/lib
    )

    target_include_directories(${TEST_BINARY} PRIVATE
        ${THIRD_PARTY_OUTPUT_DIR}/googletest/include
        ${THIRD_PARTY_OUTPUT_DIR}/mockcpp/include
        ${list_includes}
    )

    target_link_directories(${TEST_BINARY} PRIVATE
        ${THIRD_PARTY_OUTPUT_DIR}/gtest/lib
        ${THIRD_PARTY_OUTPUT_DIR}/mockcpp/lib
    )

    target_link_libraries(${TEST_BINARY} PRIVATE
        gtest
        gtest_main
        pthread
        mockcpp
        ${list_libraries}
    )

    add_test(NAME ${module}_${type}
        COMMAND ${TEST_BINARY}
            --gtest_color=yes
            --gtest_brief=0
            --gtest_output=xml:${CMAKE_BINARY_DIR}/dlt_info/${module}_${type}_detail.xml
            --gtest_break_on_failure
    )

    set(_LD_PATH
        "${THIRD_PARTY_OUTPUT_DIR}/grpc/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/re2/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/zlib/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/protobuf/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/abseil-cpp/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/cares/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/openssl/lib"
        "${THIRD_PARTY_OUTPUT_DIR}/boost/lib"
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/src"
        "${CMAKE_BINARY_DIR}/src/utils"
        "${CMAKE_BINARY_DIR}/lib"
    )

    string(JOIN ":" LD_PATH_STR ${_LD_PATH})
    set_tests_properties(${module}_${type} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${LD_PATH_STR}"
    )
endfunction()
