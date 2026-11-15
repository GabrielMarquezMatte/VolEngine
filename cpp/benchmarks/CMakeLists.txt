include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    URL_HASH SHA256=32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981
    SYSTEM)
FetchContent_MakeAvailable(benchmark)

add_executable(volengine_cpp_benchmarks wrapper_overhead.cpp)
target_link_libraries(volengine_cpp_benchmarks PRIVATE volengine::volengine benchmark::benchmark_main)
volengine_copy_native_library(volengine_cpp_benchmarks)
