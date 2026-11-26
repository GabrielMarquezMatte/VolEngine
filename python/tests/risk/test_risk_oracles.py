"""The parametric VaR and ES against rugarch's qdist and the tail integral of ddist (fixtures/rugarch_risk.json)."""

import json
from pathlib import Path
from typing import Any

import pytest

import volengine as ve

REFERENCE = Path(__file__).resolve().parents[3] / "tests" / "reference"
FIXTURE = json.loads((REFERENCE / "fixtures" / "rugarch_risk.json").read_text(encoding="utf-8"))
CASES = FIXTURE["results"]
# Observed: at most 5e-15 relative on the quantile and 5.1e-14 on the tail mean (std, nu = 4.5, at 0.999).
TOLERANCE = 1e-12


def case_id(case: dict[str, Any]) -> str:
    return f"{case['dist']}-{case['skew']}-{case['shape']}-{case['ghlambda']}-{case['confidence']}"


def parameters(case: dict[str, Any]) -> dict[str, Any]:
    return {"dist": case["dist"], "skew": case["skew"], "shape": case["shape"], "ghlambda": case["ghlambda"]}


def test_the_fixture_covers_every_density() -> None:
    assert FIXTURE["rugarch_version"] == "1.5.6"
    assert {case["dist"] for case in CASES} == {"norm", "std", "ged", "sstd", "sged", "nig", "ghyp", "jsu"}


@pytest.mark.parametrize("case", CASES, ids=[case_id(case) for case in CASES])
def test_standard_measures_match_rugarch(case: dict[str, Any]) -> None:
    var, es = ve.risk.parametric(0.0, 1.0, confidence=case["confidence"], **parameters(case))
    assert var == pytest.approx(-case["quantile"], rel=TOLERANCE)
    assert es == pytest.approx(-case["tail_mean"], rel=TOLERANCE)


@pytest.mark.parametrize("case", CASES[::3], ids=[case_id(case) for case in CASES[::3]])
def test_location_and_scale(case: dict[str, Any]) -> None:
    var, es = ve.risk.parametric(0.001, 0.02, confidence=case["confidence"], **parameters(case))
    assert var == pytest.approx(-(0.001 + 0.02 * case["quantile"]), rel=TOLERANCE)
    assert es == pytest.approx(-(0.001 + 0.02 * case["tail_mean"]), rel=TOLERANCE)
