import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_version.py"
spec = importlib.util.spec_from_file_location("release_version", SCRIPT)
assert spec is not None and spec.loader is not None
release_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_version)


def test_plain_tag():
    assert release_version.python_version("v0.9.0") == "0.9.0"
    assert release_version.cmake_version("v0.9.0") == "0.9.0"


def test_prerelease_tag():
    assert release_version.python_version("v1.0.0-rc.1") == "1.0.0rc1"
    assert release_version.cmake_version("v1.0.0-rc.1") == "1.0.0"


def test_tag_without_v_is_rejected():
    with pytest.raises(ValueError):
        release_version.python_version("1.0.0")
