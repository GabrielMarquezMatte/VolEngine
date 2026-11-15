"""Turns a release tag (v1.0.0-rc.1) into the versions the build needs.

Run:  python python/scripts/release_version.py TAG
Prints the PEP 440 version for the wheel, then the x.y.z version for CMake.
"""

import sys

from packaging.version import Version


def parse(tag: str) -> Version:
    if not tag.startswith("v"):
        raise ValueError(f"tag {tag!r} must start with 'v'")
    return Version(tag[1:])


def python_version(tag: str) -> str:
    return str(parse(tag))


def cmake_version(tag: str) -> str:
    return parse(tag).base_version


if __name__ == "__main__":
    print(python_version(sys.argv[1]))
    print(cmake_version(sys.argv[1]))
