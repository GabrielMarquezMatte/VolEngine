cmake_minimum_required(VERSION 3.28)
project(VolEngineCpp VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

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

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_install_default ON)
else()
    set(_install_default OFF)
endif()
option(VOLENGINE_INSTALL "Generate install rules and a find_package(volengine) config package" ${_install_default})

if(VOLENGINE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    get_target_property(_native_lib volengine_native IMPORTED_LOCATION)
    get_filename_component(VOLENGINE_INSTALLED_NATIVE_NAME "${_native_lib}" NAME)
    if(WIN32)
        set(VOLENGINE_INSTALLED_NATIVE_DIR "${CMAKE_INSTALL_BINDIR}")
        get_target_property(_native_implib volengine_native IMPORTED_IMPLIB)
        get_filename_component(VOLENGINE_INSTALLED_IMPLIB_NAME "${_native_implib}" NAME)
    else()
        set(VOLENGINE_INSTALLED_NATIVE_DIR "${CMAKE_INSTALL_LIBDIR}")
        get_target_property(VOLENGINE_INSTALLED_SONAME volengine_native IMPORTED_SONAME)
    endif()
    set(VOLENGINE_INSTALLED_IMPLIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(VOLENGINE_INSTALLED_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")

    install(DIRECTORY include/volengine/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volengine" FILES_MATCHING PATTERN "*.hpp")
    install(FILES ../include/volengine/volengine.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volengine")
    install(PROGRAMS "${_native_lib}" DESTINATION "${VOLENGINE_INSTALLED_NATIVE_DIR}")
    if(WIN32)
        install(FILES "${_native_implib}" DESTINATION "${VOLENGINE_INSTALLED_IMPLIB_DIR}")
    endif()

    set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/volengine")
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/volengineConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/volengineConfig.cmake"
        INSTALL_DESTINATION "${_config_dir}"
        PATH_VARS VOLENGINE_INSTALLED_NATIVE_DIR VOLENGINE_INSTALLED_IMPLIB_DIR VOLENGINE_INSTALLED_INCLUDE_DIR)

    # VOLENGINE_VERSION is a release tag ("v0.9.0"); the version file takes digits and dots only.
    string(REGEX REPLACE "^v" "" _package_version "${VOLENGINE_VERSION}")
    if(NOT _package_version MATCHES "^[0-9]+(\\.[0-9]+)*$")
        set(_package_version "0.0.0")
    endif()
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/volengineConfigVersion.cmake"
        VERSION "${_package_version}"
        COMPATIBILITY SameMajorVersion)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/volengineConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/volengineConfigVersion.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyNativeLibrary.cmake"
        DESTINATION "${_config_dir}")
endif()

option(VOLENGINE_BUILD_TESTS "Build the VolEngine C++ wrapper tests" OFF)
if(VOLENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
