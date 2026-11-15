# Run with: cmake -P asset_names.cmake. Checks the release asset name of every platform the 6a release publishes.
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/FetchNativeLib.cmake")

set(_cases
    "win,x64,volengine-native-win-x64.dll"
    "osx,arm64,volengine-native-osx-arm64.dylib"
    "osx,x64,volengine-native-osx-x64.dylib"
    "linux,x64,volengine-native-linux-x64.so"
    "linux,arm64,volengine-native-linux-arm64.so"
    "linux-musl,x64,volengine-native-linux-musl-x64.so"
    "linux-musl,arm64,volengine-native-linux-musl-arm64.so")
foreach(_case IN LISTS _cases)
    string(REPLACE "," ";" _case "${_case}")
    list(GET _case 0 _os)
    list(GET _case 1 _arch)
    list(GET _case 2 _expected)
    volengine_asset_name("${_os}" "${_arch}" _actual)
    if(NOT _actual STREQUAL _expected)
        message(FATAL_ERROR "volengine_asset_name(${_os} ${_arch}) = '${_actual}', expected '${_expected}'")
    endif()
endforeach()
# A downloaded library is cached per release: moving a build directory from one tag to the next must not link the old one.
volengine_cache_directory("/build" "v0.9.0" _old)
volengine_cache_directory("/build" "v1.0.0" _new)
if(_old STREQUAL _new OR NOT _new STREQUAL "/build/volengine-native/v1.0.0")
    message(FATAL_ERROR "cache directories: '${_old}' and '${_new}'")
endif()
message(STATUS "asset names ok")
