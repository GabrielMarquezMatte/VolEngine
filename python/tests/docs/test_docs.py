"""The Python examples of the README and of the guides run, one block after another in a shared namespace."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS = ["README.md", "docs/guide/python.md"]
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def python_blocks(document: str) -> list[str]:
    return PYTHON_BLOCK.findall((ROOT / document).read_text(encoding="utf-8"))


def test_the_extractor_finds_blocks() -> None:
    assert len(python_blocks("README.md")) >= 10


@pytest.mark.parametrize("document", DOCUMENTS)
def test_python_blocks_run(document: str) -> None:
    namespace: dict[str, object] = {"__name__": f"docs:{document}"}
    blocks = python_blocks(document)
    assert blocks, f"{document} has no python block"
    for number, block in enumerate(blocks, start=1):
        try:
            exec(compile(block, f"{document} block {number}", "exec"), namespace)
        except Exception as error:
            raise AssertionError(f"{document} block {number} failed: {error!r}\n{block}") from error
