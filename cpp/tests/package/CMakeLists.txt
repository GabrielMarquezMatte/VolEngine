cmake_minimum_required(VERSION 3.28)

# Consumes the wrapper the way a downstream project does, in one of two modes:
#   fetch_content  FetchContent over this repository (SOURCE_DIR instead of a URL, so no network)
#   find_package   an install prefix made by `cmake --install` (pass -DCMAKE_PREFIX_PATH=<prefix>)
# It is configured as its own project, so a config file that only works inside the build tree fails here.
project(volengine_package_consumer LANGUAGES CXX)

set(VOLENGINE_CONSUMER_MODE "find_package" CACHE STRING "fetch_content or find_package")
set(VOLENGINE_CONSUMER_VERSION "" CACHE STRING "Version find_package asks for (empty: any)")
set(VOLENGINE_REPOSITORY "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "Root of the VolEngine repository")

if(VOLENGINE_CONSUMER_MODE STREQUAL "fetch_content")
    include(FetchContent)
    FetchContent_Declare(volengine SOURCE_DIR "${VOLENGINE_REPOSITORY}" SOURCE_SUBDIR cpp)
    FetchContent_MakeAvailable(volengine)
elseif(VOLENGINE_CONSUMER_MODE STREQUAL "find_package")
    find_package(volengine ${VOLENGINE_CONSUMER_VERSION} REQUIRED)
else()
    message(FATAL_ERROR "VOLENGINE_CONSUMER_MODE must be fetch_content or find_package")
endif()

add_executable(package_consumer main.cpp)
target_link_libraries(package_consumer PRIVATE volengine::volengine)
volengine_copy_native_library(package_consumer)

enable_testing()
add_test(NAME package_consumer COMMAND package_consumer)
