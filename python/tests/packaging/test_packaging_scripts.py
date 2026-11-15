"""Packaging scripts live in python/scripts and python/hatch_build.py; they are not importable packages."""

import importlib.util
import struct
from pathlib import Path
from types import ModuleType

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[2]


def load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deps = load(PYTHON_DIR / "scripts" / "check_native_deps.py")
build = load(PYTHON_DIR / "scripts" / "build_native.py")
hook = load(PYTHON_DIR / "hatch_build.py")


def test_windows_mingw_runtime_is_forbidden():
    assert deps.forbidden("Windows", ["KERNEL32.dll", "libstdc++-6.dll", "libwinpthread-1.dll"]) == [
        "libstdc++-6.dll",
        "libwinpthread-1.dll",
    ]


def test_windows_system_dlls_pass():
    assert deps.forbidden("Windows", ["KERNEL32.dll", "msvcrt.dll", "api-ms-win-crt-runtime-l1-1-0.dll"]) == []


def test_linux_shared_libstdcxx_is_forbidden():
    assert deps.forbidden("Linux", ["libstdc++.so.6", "libgcc_s.so.1", "libm.so.6", "libc.so.6"]) == [
        "libstdc++.so.6",
        "libgcc_s.so.1",
    ]


def test_macos_allows_only_system_paths():
    assert deps.forbidden("Darwin", ["/usr/lib/libc++.1.dylib", "/opt/homebrew/lib/libomp.dylib"]) == [
        "/opt/homebrew/lib/libomp.dylib"
    ]


def test_build_refuses_isa_flags():
    with pytest.raises(SystemExit, match="-march"):
        build.check_flags({"CXXFLAGS": "-O2 -march=native"})


def mach_o_with_minos(major: int, minor: int) -> bytes:
    """A minimal 64-bit Mach-O header with one LC_BUILD_VERSION command."""
    header = bytearray(32)
    struct.pack_into("<I", header, 16, 1)
    command = struct.pack("<IIIIII", 0x32, 24, 1, major << 16 | minor << 8, 0, 0)
    return bytes(header) + command


def test_glibc_floor_is_newest_symbol_version():
    assert hook.glibc_floor(b"\0GLIBC_2.17\0GLIBC_2.28\0GLIBC_2.2.5\0") == "2_28"


def test_glibc_floor_defaults_to_2_17():
    assert hook.glibc_floor(b"no versions") == "2_17"


def test_macos_floor_reads_lc_build_version():
    assert hook.macos_floor(mach_o_with_minos(11, 0)) == "11_0"
