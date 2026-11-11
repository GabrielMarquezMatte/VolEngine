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

# loglik<double> and loglik<Dual<N>> must round identically, and only the double path could be fused into
# multiply-adds.
set(VE_FLOAT_FLAGS)
if(NOT MSVC)
    set(VE_FLOAT_FLAGS -ffp-contract=off)
endif()

include(FetchContent)
FetchContent_Declare(bs_thread_pool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool
    GIT_TAG v5.0.0
    SOURCE_SUBDIR header-only-no-cmake)
FetchContent_MakeAvailable(bs_thread_pool)

add_library(volengine_core STATIC
    src/ad/special.cpp
    src/core/context.cpp
    src/fit/batch.cpp
    src/fit/cholesky.cpp
    src/fit/criteria.cpp
    src/fit/fit.cpp
    src/fit/standard_errors.cpp
    src/model/layout.cpp
    src/model/likelihood.cpp
    src/model/parameter_map.cpp
    src/model/start_values.cpp)
set_target_properties(volengine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(volengine_core PUBLIC src include)
target_include_directories(volengine_core SYSTEM PUBLIC ${bs_thread_pool_SOURCE_DIR}/include)
target_compile_options(volengine_core PRIVATE ${VE_WARNING_FLAGS} ${VE_FLOAT_FLAGS})

add_library(volengine SHARED
    src/capi/version_api.cpp
    src/capi/batch_api.cpp
    src/capi/context_api.cpp
    src/capi/fit_api.cpp
    src/capi/spec_api.cpp
    src/capi/panel_api.cpp
    src/capi/error.cpp)
target_include_directories(volengine PUBLIC include)
target_compile_definitions(volengine PRIVATE VE_BUILDING)
target_compile_options(volengine PRIVATE ${VE_WARNING_FLAGS})
target_link_libraries(volengine PRIVATE volengine_core)
if(MINGW)
    target_link_options(volengine PRIVATE
        -static-libgcc -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic)
endif()

if(VE_BUILD_TESTS)
    enable_testing()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
    FetchContent_MakeAvailable(googletest)

    add_library(volengine_c_header_check OBJECT tests/unit/capi/c_header_check.c)
    target_link_libraries(volengine_c_header_check PRIVATE volengine)
    target_compile_options(volengine_c_header_check PRIVATE ${VE_WARNING_FLAGS})

    add_executable(volengine_tests
        tests/unit/ad/dual_test.cpp
        tests/unit/ad/special_test.cpp
        tests/unit/capi/abi_test.cpp
        tests/unit/capi/batch_test.cpp
        tests/unit/capi/context_test.cpp
        tests/unit/capi/fit_accessors_test.cpp
        tests/unit/capi/panel_test.cpp
        tests/unit/capi/spec_test.cpp
        tests/unit/dist/densities_test.cpp
        tests/unit/fit/criteria_test.cpp
        tests/unit/fit/fit_test.cpp
        tests/unit/fit/standard_errors_test.cpp
        tests/unit/garch/sgarch_test.cpp
        tests/unit/mean/arma_test.cpp
        tests/unit/model/layout_test.cpp
        tests/unit/model/likelihood_test.cpp
        tests/unit/model/parameter_map_test.cpp
        tests/unit/model/start_values_test.cpp
        tests/unit/optim/lbfgs_test.cpp)
    target_include_directories(volengine_tests PRIVATE tests/unit)
    target_link_libraries(volengine_tests PRIVATE volengine_core volengine GTest::gtest_main)
    target_compile_options(volengine_tests PRIVATE ${VE_WARNING_FLAGS} ${VE_FLOAT_FLAGS})
    include(GoogleTest)
    gtest_discover_tests(volengine_tests)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME style
        COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tests/style/check_style.py ${CMAKE_SOURCE_DIR})
endif()
