include(googletest)
enable_testing()
include(GoogleTest)


add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
add_custom_target(tests)

configure_file("${PROJECT_SOURCE_DIR}/src/include/config.h.in" "${PROJECT_SOURCE_DIR}/src/include/config.h")

function(add_gtest TEST_NAME)
  message("Adding Test: " ${TEST_NAME})
  add_executable(test_${TEST_NAME} ${PROJECT_SOURCE_DIR}/src/fin.cpp ${PROJECT_SOURCE_DIR}/src/base64.cpp ${TEST_NAME}.cpp)
  add_dependencies(tests test_${TEST_NAME})
  add_dependencies(check test_${TEST_NAME})
  target_compile_options(test_${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  target_compile_definitions(test_${TEST_NAME} PUBLIC TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
  target_link_libraries(test_${TEST_NAME} gtest_main MIOpen ${Boost_LIBRARIES} hip::host $<BUILD_INTERFACE:roc::rocblas>)
  target_include_directories(test_${TEST_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/include)
  gtest_discover_tests(test_${TEST_NAME})
endfunction()

file(GLOB TESTS *.cpp)
foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_gtest(${BASE_NAME})
endforeach()
