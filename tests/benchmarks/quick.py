"""Quick benchmark suite and A/B comparisons. Standard library only. See docs/performance/measuring.md.

    python tests/benchmarks/quick.py run --out FILE [--exe PATH]
    python tests/benchmarks/quick.py compare A.json B.json
    python tests/benchmarks/quick.py compare-profile A.json B.json
    python tests/benchmarks/quick.py selftest
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_EXE = Path("build/gcc-bench/volengine_benchmarks.exe")
QUICK_FILTER = (
    "^BM_Loglik/(s11_norm|s21_std|arma11_s11_ged|arma22_s33_std)/(double|dual)$"
    "|^BM_FitOne/(s11_norm|s21_std|arma11_s11_ged|arma22_s33_std|egarch11_std)$"
    "|^BM_TccGridSample/"
)


def medians(report: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """run_name -> (median, standard deviation) of real_time, from a Google Benchmark JSON with aggregates."""
    values: dict[str, dict[str, float]] = {}
    for entry in report["benchmarks"]:
        if entry.get("run_type") != "aggregate":
            continue
        values.setdefault(entry["run_name"], {})[entry["aggregate_name"]] = entry["real_time"]
    return {name: (kinds["median"], kinds.get("stddev", 0.0)) for name, kinds in values.items() if "median" in kinds}


def compare_rows(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str, float, float, float, bool]]:
    """(name, median A, median B, B / A, within noise) for every benchmark in both runs."""
    before = medians(a)
    after = medians(b)
    rows = []
    for name, (median_a, sd_a) in before.items():
        if name not in after:
            continue
        median_b, sd_b = after[name]
        rows.append((name, median_a, median_b, median_b / median_a, abs(median_b - median_a) <= max(sd_a, sd_b)))
    return rows


def status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts


def worst_loglik_change(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[float, str]:
    """The lowest (loglik B - loglik A) / |loglik A| over the fits finite in both, and which fit it is."""
    before = {(record["spec"], record["series"]): record["loglik"] for record in a}
    worst = (float("inf"), "")
    for record in b:
        loglik_a = before.get((record["spec"], record["series"]))
        if loglik_a is None or record["loglik"] is None:
            continue
        change = (record["loglik"] - loglik_a) / abs(loglik_a)
        worst = min(worst, (change, f"{record['spec']}/{record['series']}"))
    return worst


def profile_summary(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Time and evaluation ratios, worst relative loglik change and status counts of two grid profiles."""
    change, where = worst_loglik_change(a, b)
    return {
        "time_ratio": sum(r["seconds"] for r in b) / sum(r["seconds"] for r in a),
        "evaluation_ratio": sum(r["evaluations"] for r in b) / sum(r["evaluations"] for r in a),
        "worst_loglik_change": change,
        "worst_loglik_fit": where,
        "status_a": status_counts(a),
        "status_b": status_counts(b),
    }


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def run(out: str, exe: str) -> None:
    command = [str(Path(exe).resolve()), f"--benchmark_filter={QUICK_FILTER}", "--benchmark_repetitions=3",
               "--benchmark_report_aggregates_only=true", f"--benchmark_out={out}", "--benchmark_out_format=json"]
    subprocess.run(command, check=True)


def print_compare(a_path: str, b_path: str) -> None:
    print(f"{'benchmark':<60} {'A':>12} {'B':>12} {'B/A':>7}")
    for name, median_a, median_b, ratio, noise in compare_rows(load(a_path), load(b_path)):
        print(f"{name:<60} {median_a:>12.4g} {median_b:>12.4g} {ratio:>7.3f} {'noise' if noise else ''}")


def print_profile(a_path: str, b_path: str) -> None:
    for key, value in profile_summary(load(a_path), load(b_path)).items():
        print(f"{key}: {value}")


def selftest() -> None:
    def aggregate(name: str, kind: str, value: float) -> dict[str, Any]:
        return {"run_name": name, "run_type": "aggregate", "aggregate_name": kind, "real_time": value}

    a = {"benchmarks": [aggregate("x", "median", 10.0), aggregate("x", "stddev", 1.0), aggregate("y", "median", 4.0),
                        aggregate("y", "stddev", 0.1), aggregate("only_a", "median", 1.0)]}
    b = {"benchmarks": [aggregate("x", "median", 10.5), aggregate("x", "stddev", 0.2), aggregate("y", "median", 2.0),
                        aggregate("y", "stddev", 0.1)]}
    assert medians(a)["x"] == (10.0, 1.0)
    rows = {row[0]: row for row in compare_rows(a, b)}
    assert set(rows) == {"x", "y"}
    assert rows["x"][4] is True
    assert rows["y"][3] == 0.5
    assert rows["y"][4] is False

    def fit(spec: str, seconds: float, evaluations: int, loglik: float | None, status: str) -> dict[str, Any]:
        return {"spec": spec, "series": 0, "status": status, "seconds": seconds, "optimizer_seconds": seconds,
                "iterations": 1, "evaluations": evaluations, "loglik": loglik}

    base = [fit("s1", 2.0, 10, -100.0, "converged"), fit("s2", 2.0, 10, -50.0, "max_iter")]
    new = [fit("s1", 1.0, 5, -100.001, "converged"), fit("s2", 1.0, 5, -49.0, "converged")]
    summary = profile_summary(base, new)
    assert summary["time_ratio"] == 0.25 * 2
    assert summary["evaluation_ratio"] == 0.5
    assert abs(summary["worst_loglik_change"] - (-1e-5)) < 1e-12
    assert summary["worst_loglik_fit"] == "s1/0"
    assert summary["status_a"] == {"converged": 1, "max_iter": 1}
    assert summary["status_b"] == {"converged": 2}
    print("selftest ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--exe", default=str(DEFAULT_EXE))
    for name in ("compare", "compare-profile"):
        pair = commands.add_parser(name)
        pair.add_argument("a")
        pair.add_argument("b")
    arguments = parser.parse_args()
    actions = {
        "selftest": selftest,
        "run": lambda: run(arguments.out, arguments.exe),
        "compare": lambda: print_compare(arguments.a, arguments.b),
        "compare-profile": lambda: print_profile(arguments.a, arguments.b),
    }
    actions[arguments.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
