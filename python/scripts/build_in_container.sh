#!/bin/sh
# Runs build_wheel.sh inside a manylinux or musllinux container, with the repository mounted at /work:
#   docker run --rm -v "$PWD:/work" -w /work -e EXPECTED_TAG -e ASSET -e VERSION IMAGE sh python/scripts/build_in_container.sh
set -eu

# The images carry CPython builds under /opt/python; pip gives them uv, cmake and ninja without a download script.
toolchain=$(ls -d /opt/python/cp313-*/bin | head -n 1)
"$toolchain/python" -m pip install --quiet uv cmake ninja
PATH="$toolchain:$PATH"
export PATH

# manylinux_2_28 ships GCC 14 as a toolset that its default PATH may not select.
if [ -f /opt/rh/gcc-toolset-14/enable ]; then
    . /opt/rh/gcc-toolset-14/enable
fi
command -v bash >/dev/null || apk add --no-cache bash

exec bash python/scripts/build_wheel.sh
