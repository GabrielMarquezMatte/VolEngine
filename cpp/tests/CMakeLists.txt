include(FetchContent)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

if(MSVC)
    set(VOLENGINE_TEST_WARNINGS /W4 /WX)
else()
    set(VOLENGINE_TEST_WARNINGS -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(volengine_cpp_tests
    diagnostics_test.cpp
    error_test.cpp
    fit_test.cpp
    future_test.cpp)
target_link_libraries(volengine_cpp_tests PRIVATE volengine::volengine GTest::gtest_main)
target_compile_options(volengine_cpp_tests PRIVATE ${VOLENGINE_TEST_WARNINGS})
target_compile_definitions(volengine_cpp_tests PRIVATE
    VOLENGINE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/reference/data")
# Before test discovery: discovery runs the executable, which needs the DLL beside it on Windows.
volengine_copy_native_library(volengine_cpp_tests)
gtest_discover_tests(volengine_cpp_tests)

add_test(NAME asset_names COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/asset_names.cmake)
