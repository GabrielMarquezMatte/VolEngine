#!/usr/bin/env bash
# Tests the C++ wrapper against a local build of the library, then checks that a downstream project can use it both
# through FetchContent and through find_package after `cmake --install`. cpp.yml runs it on every platform.
#
# Environment:
#   VOLENGINE_NATIVE_LIB  the library to link (python/src/volengine/_lib/ after build_native.py)
#   CPP_CMAKE_ARGS        extra configure arguments, for example -DCMAKE_CXX_COMPILER=g++-14
set -euo pipefail
cd "$(dirname "$0")/../.."

# A consumer asks find_package for this version; the install below is made under the matching tag.
package_version=1.1.0

: "${VOLENGINE_NATIVE_LIB:?set VOLENGINE_NATIVE_LIB to the library}"
if command -v cygpath >/dev/null; then
    VOLENGINE_NATIVE_LIB=$(cygpath -m "$VOLENGINE_NATIVE_LIB")
fi
export VOLENGINE_NATIVE_LIB

# shellcheck disable=SC2206
args=(${CPP_CMAKE_ARGS:-} -G Ninja -DCMAKE_BUILD_TYPE=Release)

cmake -S cpp -B build/cpp "${args[@]}" -DVOLENGINE_BUILD_TESTS=ON
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure

cmake -S cpp -B build/cpp-install "${args[@]}" "-DVOLENGINE_VERSION=v${package_version}"
cmake --install build/cpp-install --prefix "$PWD/build/cpp-prefix"

consumer=(cmake -S cpp/tests/package "${args[@]}")
"${consumer[@]}" -B build/consumer-fetch -DVOLENGINE_CONSUMER_MODE=fetch_content
cmake --build build/consumer-fetch
ctest --test-dir build/consumer-fetch --output-on-failure

"${consumer[@]}" -B build/consumer-find -DVOLENGINE_CONSUMER_MODE=find_package \
    "-DVOLENGINE_CONSUMER_VERSION=${package_version%.*}" "-DCMAKE_PREFIX_PATH=$PWD/build/cpp-prefix"
cmake --build build/consumer-find
ctest --test-dir build/consumer-find --output-on-failure
