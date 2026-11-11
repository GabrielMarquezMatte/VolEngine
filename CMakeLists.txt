cmake_minimum_required(VERSION 3.28)
project(VolEngine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(VE_IS_TOP_LEVEL ON)
else()
    set(VE_IS_TOP_LEVEL OFF)
endif()
option(VE_BUILD_TESTS "Build the VolEngine tests" ${VE_IS_TOP_LEVEL})

if(MSVC)
    set(VE_WARNING_FLAGS /W4 /WX)
else()
    set(VE_WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)
endif()

add_library(volengine SHARED src/capi/version_api.cpp)
target_include_directories(volengine PUBLIC include)
target_compile_definitions(volengine PRIVATE VE_BUILDING)
target_compile_options(volengine PRIVATE ${VE_WARNING_FLAGS})
if(MINGW)
    target_link_options(volengine PRIVATE
        -static-libgcc -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic)
endif()

if(VE_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
    FetchContent_MakeAvailable(googletest)

    add_library(volengine_c_header_check OBJECT tests/unit/capi/c_header_check.c)
    target_link_libraries(volengine_c_header_check PRIVATE volengine)
    target_compile_options(volengine_c_header_check PRIVATE ${VE_WARNING_FLAGS})

    add_executable(volengine_tests tests/unit/capi/abi_test.cpp)
    target_link_libraries(volengine_tests PRIVATE volengine GTest::gtest_main)
    target_compile_options(volengine_tests PRIVATE ${VE_WARNING_FLAGS})
    include(GoogleTest)
    gtest_discover_tests(volengine_tests)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME style
        COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/style/check_style.py ${CMAKE_SOURCE_DIR})
endif()
