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
message(STATUS "asset names ok")
