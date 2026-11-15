cmake_minimum_required(VERSION 3.28)
project(VolEngineCpp VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VOLENGINE_VERSION "" CACHE STRING
    "Release tag whose assets are downloaded (taken from `git describe --tags --exact-match` if empty)")
if(NOT VOLENGINE_VERSION)
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _git_tag
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _git_tag_result)
        if(_git_tag_result EQUAL 0)
            set(VOLENGINE_VERSION "${_git_tag}")
        endif()
    endif()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchNativeLib.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyNativeLibrary.cmake")
volengine_fetch_native_lib()

# ../include holds the C header (volengine/volengine.h) in this repository; include/ holds the C++ headers.
add_library(volengine_cpp INTERFACE)
target_include_directories(volengine_cpp INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_features(volengine_cpp INTERFACE cxx_std_23)
target_link_libraries(volengine_cpp INTERFACE volengine::native)
add_library(volengine::volengine ALIAS volengine_cpp)

option(VOLENGINE_BUILD_TESTS "Build the VolEngine C++ wrapper tests" OFF)
if(VOLENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
