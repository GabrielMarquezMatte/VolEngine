#!/usr/bin/env bash
# Writes the ABI of libvolengine.so as libabigail XML: scripts/abi_dump.sh OUT.xml
#
# The release attaches this file for its Linux x64 library (volengine-abi-linux-x64.xml); the CI job `abi` makes one
# for the current code and runs abidiff against the last release's. Both use the same toolchain (GCC 15 on
# ubuntu-24.04, scripts/install_abi_tools.sh), so the two dumps differ only where the code does. The library is built
# with -g because abidw reads the types from the debug information; --headers-dir keeps only what volengine.h declares.
#
# Environment: CC and CXX name the compiler (default gcc-15 and g++-15).
set -euo pipefail
out=$(realpath -m "${1:?usage: abi_dump.sh OUT.xml}")
cd "$(dirname "$0")/.."

cmake -S . -B build/abi -G Ninja -DCMAKE_BUILD_TYPE=Release -DVE_BUILD_TESTS=OFF \
    "-DCMAKE_C_COMPILER=${CC:-gcc-15}" "-DCMAKE_CXX_COMPILER=${CXX:-g++-15}" \
    -DCMAKE_C_FLAGS=-g -DCMAKE_CXX_FLAGS=-g
cmake --build build/abi --target volengine
mkdir -p "$(dirname "$out")"
abidw --headers-dir include/volengine --drop-private-types --out-file "$out" build/abi/libvolengine.so
grep -c 've_fit_criterion' "$out" > /dev/null
echo "wrote $out"
