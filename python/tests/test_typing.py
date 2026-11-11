from pathlib import Path

from mypy import api

TYPING_CASES = Path(__file__).parent / "typing_cases.py"


def test_fit_overloads_infer_return_types() -> None:
    stdout, stderr, status = api.run(["--strict", "--follow-imports=silent", str(TYPING_CASES)])
    assert status == 0, stdout + stderr
