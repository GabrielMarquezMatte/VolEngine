"""The covariance estimators against numpy.cov and sklearn.covariance.ledoit_wolf (fixtures/sklearn_cov.json)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import volengine as ve

REFERENCE = Path(__file__).resolve().parents[3] / "tests" / "reference"
FIXTURE = json.loads((REFERENCE / "fixtures" / "sklearn_cov.json").read_text(encoding="utf-8"))
CASES = FIXTURE["results"]
TOLERANCE = 1e-12


def assert_matrix_close(actual: np.ndarray, expected: list[list[float]]) -> None:
    reference = np.array(expected)
    assert actual.shape == reference.shape
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(actual - reference)) <= TOLERANCE * scale


def test_the_fixture_has_every_case() -> None:
    assert [case["id"] for case in CASES] == ["copula3", "single", "wide", "wider"]
    assert FIXTURE["sklearn_version"] == "1.9.1"


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_sample_matches_numpy(case: dict[str, Any]) -> None:
    assert_matrix_close(ve.cov.sample(np.array(case["returns"])), case["sample"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_ledoit_wolf_matches_sklearn(case: dict[str, Any]) -> None:
    covariance, shrinkage = ve.cov.ledoit_wolf(np.array(case["returns"]))
    assert_matrix_close(covariance, case["ledoit_wolf"])
    assert shrinkage == pytest.approx(case["shrinkage"], rel=TOLERANCE, abs=1e-15)
