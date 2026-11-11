from pathlib import Path

import pytest

from volengine import _native


def test_missing_override_names_the_variable_and_the_path(tmp_path: Path) -> None:
    missing = str(tmp_path / "absent.dll")
    with pytest.raises(ImportError, match="VOLENGINE_LIB") as error:
        _native.locate_library(missing, tmp_path)
    assert missing in str(error.value)


def test_missing_library_names_both_places(tmp_path: Path) -> None:
    with pytest.raises(ImportError) as error:
        _native.locate_library(None, tmp_path)
    assert "VOLENGINE_LIB" in str(error.value)
    assert str(tmp_path) in str(error.value)


@pytest.mark.parametrize("name", ["libvolengine.dll", "volengine.dll", "libvolengine.so", "libvolengine.dylib"])
def test_directory_lookup_finds_each_library_name(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"")
    assert _native.locate_library(None, tmp_path) == tmp_path / name


def test_override_wins_over_directory(tmp_path: Path) -> None:
    (tmp_path / "libvolengine.dll").write_bytes(b"")
    chosen = tmp_path / "custom.bin"
    chosen.write_bytes(b"")
    assert _native.locate_library(str(chosen), tmp_path) == chosen


def test_not_a_library_file_is_an_import_error(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.dll"
    bogus.write_bytes(b"not a library")
    with pytest.raises(ImportError, match="cannot load"):
        _native.load_library({"VOLENGINE_LIB": str(bogus)}, tmp_path)
