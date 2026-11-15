"""Fails if the native library imports a compiler runtime that a clean machine does not have.

Run after building the library:  python python/scripts/check_native_deps.py PATH
"""

import platform
import subprocess
import sys

WINDOWS_FORBIDDEN = ("libstdc++", "libgcc", "libwinpthread")
LINUX_FORBIDDEN = ("libstdc++", "libgcc_s")
MACOS_ALLOWED = ("/usr/lib/", "/System/")


def forbidden(system: str, needed: list[str]) -> list[str]:
    if system == "Windows":
        return [name for name in needed if name.lower().startswith(WINDOWS_FORBIDDEN)]
    if system == "Linux":
        return [name for name in needed if name.startswith(LINUX_FORBIDDEN)]
    return [name for name in needed if not name.startswith(MACOS_ALLOWED)]


def run(*command: str) -> list[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()


def needed(system: str, path: str) -> list[str]:
    if system == "Windows":
        return [line.split("DLL Name:")[1].strip() for line in run("objdump", "-p", path) if "DLL Name:" in line]
    if system == "Linux":
        return [
            line.split("[")[1].rstrip("]")
            for line in run("readelf", "-d", path)
            if "(NEEDED)" in line
        ]
    # otool -L: the file name, then the library's own id, then what it imports.
    return [line.split()[0] for line in run("otool", "-L", path)[2:]]


def main() -> int:
    system = platform.system()
    dependencies = needed(system, sys.argv[1])
    bad = forbidden(system, dependencies)
    if bad:
        print(f"error: {sys.argv[1]} needs {', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"ok: {', '.join(dependencies)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
