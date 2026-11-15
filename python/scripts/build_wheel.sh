#!/usr/bin/env bash
# Builds one platform's native library and wheel, checks the wheel, tests the installed wheel and stages the
# release assets. wheels.yml runs it on every platform, so a platform is built the same way on every push and on a tag.
#
# Environment:
#   EXPECTED_TAG  the platform tag the wheel must get (for example manylinux_2_28_x86_64)
#   ASSET         the platform part of the asset names (for example linux-musl-arm64)
#   VERSION       release tag such as v1.0.0-rc.1; empty builds the version in the repository
#
# Results: dist/ holds the wheel, out/ holds the volengine-native-<ASSET>.* files.
set -euo pipefail
cd "$(dirname "$0")/../.."

py() { uv run --no-project --python 3.14 --with packaging python "$@"; }

rm -rf dist out build/wheel-venv
cmake_args=()
if [ -n "${VERSION:-}" ]; then
    versions=$(py python/scripts/release_version.py "$VERSION")
    wheel_version=$(echo "$versions" | sed -n 1p)
    cmake_version=$(echo "$versions" | sed -n 2p)
    echo "release ${VERSION}: wheel ${wheel_version}, CMake ${cmake_version}"
    sed -i.bak "s/^version = .*/version = \"${wheel_version}\"/" python/pyproject.toml
    rm python/pyproject.toml.bak
    cmake_args+=("-DVE_VERSION=${cmake_version}")
fi

command -v gcc g++ cmake ninja || true
# macOS ships bash 3.2, where an empty array is an unbound variable under set -u.
py python/scripts/build_native.py ${cmake_args[@]+"${cmake_args[@]}"}

lib_dir=python/src/volengine/_lib
for candidate in "$lib_dir"/libvolengine.so "$lib_dir"/libvolengine.dylib "$lib_dir"/libvolengine.dll; do
    if [ -f "$candidate" ]; then library=$candidate; fi
done
py python/scripts/check_native_deps.py "$library"
if [ "$ASSET" = win-x64 ]; then
    # The .def feeds the .lib the release job builds for MSVC users; an empty one would ship a .lib with no exports.
    exports=$(grep -c '^ *ve_' "$lib_dir/volengine.def")
    echo "volengine.def lists $exports exports"
    [ "$exports" -gt 100 ]
fi

uv build --wheel python --out-dir dist
wheel=$(ls dist/*.whl)
py python/scripts/check_wheel_tag.py dist "$EXPECTED_TAG"
if [ "$(uname)" = Linux ]; then
    uvx auditwheel show "$wheel"
fi

# The wheel is tested from site-packages, never from python/src, so a broken wheel cannot hide behind the source tree.
uv venv --python 3.14 -q build/wheel-venv
if [ -x build/wheel-venv/Scripts/python.exe ]; then venv_python=build/wheel-venv/Scripts/python.exe; else venv_python=build/wheel-venv/bin/python; fi
uv pip install -q --python "$venv_python" "$wheel" pytest mypy hatchling
unset VOLENGINE_LIB
"$venv_python" -c "import volengine._native as n; assert 'site-packages' in str(n.LIBRARY_DIRECTORY), n.LIBRARY_DIRECTORY; print(n.LIBRARY_DIRECTORY)"
"$venv_python" -m pytest python/tests -q -p no:cacheprovider

mkdir out
cp "$library" "out/volengine-native-${ASSET}.${library##*.}"
if [ "$ASSET" = win-x64 ]; then
    cp "$lib_dir/libvolengine.dll.a" "out/volengine-native-${ASSET}.dll.a"
    cp "$lib_dir/volengine.def" "out/volengine-native-${ASSET}.def"
fi
ls -l dist out
