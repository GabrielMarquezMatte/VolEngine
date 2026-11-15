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
wheel_tag = load(PYTHON_DIR / "scripts" / "check_wheel_tag.py")


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


def test_tag_equal_to_expected_is_allowed():
    assert wheel_tag.tag_allowed("win_amd64", "win_amd64")
    assert wheel_tag.tag_allowed("macosx_11_0_arm64", "macosx_11_0_arm64")


def test_glibc_floor_below_expected_is_allowed():
    assert wheel_tag.tag_allowed("manylinux_2_25_x86_64", "manylinux_2_28_x86_64")


def test_glibc_floor_above_expected_is_refused():
    assert not wheel_tag.tag_allowed("manylinux_2_34_x86_64", "manylinux_2_28_x86_64")


def test_other_architecture_or_platform_is_refused():
    assert not wheel_tag.tag_allowed("manylinux_2_17_aarch64", "manylinux_2_28_x86_64")
    assert not wheel_tag.tag_allowed("macosx_14_0_arm64", "macosx_11_0_arm64")
    assert not wheel_tag.tag_allowed("musllinux_1_2_x86_64", "manylinux_2_28_x86_64")


def test_editable_build_needs_no_bundled_library(tmp_path, monkeypatch):
    """CI and developers point VOLENGINE_LIB at a library, so _lib is empty during `uv run`'s editable build."""
    monkeypatch.setattr(hook, "LIB_DIR", tmp_path)
    build_data = {"force_include": {}}
    hook.NativeLibraryBuildHook(str(tmp_path), {}, {}, None, str(tmp_path), "wheel").initialize("editable", build_data)
    assert build_data["force_include"] == {}
    assert build_data["tag"].startswith("py3-none-")
