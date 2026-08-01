#!/usr/bin/env python3
"""Fail-closed structural verification for neural response-fibre artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_SOURCE_HASHES = {
    "src/response_fibre_nn_jet_filtration_v0_12_0.py": "d4ccca014bccb55a082ffe6c785cbf57ceb786aa2d986bf391c749e7f5d99d0c",
    "src/response_fibre_nn_dimension_matched_task_advantage_v0_13_0.py": "e6bf00c999e9f41ddc71e3f64bd3e650e6d81923ec7e6f555cdeaffae5efc90b",
    "src/response_fibre_nn_fair_baseline_pareto_v0_14_0.py": "2661735581b0a53e4cd263ec4b36618ac4248a5e68c60593d3b2069d1823f168",
    "src/response_fibre_nn_prospective_task_advantage_v0_14_1.py": "1a67ccead8835eb611b3cdcdcbd0eb8d772d7e42dc5c7621b95a5f7dc46eb8b4",
}
EXPECTED_STATUSES = {
    "v0.12.0": "NEURAL_RESPONSE_JET_FILTRATION_SUPPORTED",
    "v0.13.0": "DIMENSION_MATCHED_NEURAL_RESPONSE_EFFICIENCY_AND_TASK_ADVANTAGE_NOT_SUPPORTED",
    "v0.14.0": "FAIR_NON_GEOMETRIC_BASELINE_ARCHITECTURE_QUALIFIED_FOR_PROSPECTIVE_AUDIT",
    "v0.14.1": "PROSPECTIVE_NEURAL_RESPONSE_FIBRE_TASK_ADVANTAGE_SUPPORTED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: dict[str, bool] = {}
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = ROOT / relative
        checks[f"{relative}_exists"] = path.is_file()
        checks[f"{relative}_sha256"] = path.is_file() and sha256(path) == expected

    summary_path = ROOT / "results" / "reference_summary.json"
    checks["reference_summary_exists"] = summary_path.is_file()
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audits = summary.get("audits", {})
        for version, expected in EXPECTED_STATUSES.items():
            checks[f"{version}_status"] = (
                audits.get(version, {}).get("scientific_status") == expected
            )
        checks["quantum_release_not_modified"] = (
            summary.get("quantum_v0_9_3_release_modified") is False
        )
        checks["v0141_two_co_primary_3_of_3"] = (
            audits.get("v0.14.1", {}).get("co_primary_level_seed_counts")
            == {"R_value72": 3, "R_jet72": 3}
        )

    stdout_140 = ROOT / "results" / "v0_14_0_reference_stdout.txt"
    stdout_141 = ROOT / "results" / "v0_14_1_reference_stdout.txt"
    checks["v0140_stdout_bound"] = stdout_140.is_file() and all(
        token in stdout_140.read_text(encoding="utf-8")
        for token in (
            EXPECTED_STATUSES["v0.14.0"],
            "83e88e44fd207be14950c9005f93a0ef2bb7cc97549dd38967e24a0cce44ffc4",
            "ad71c665c30dd8e697da125c06c9e79cbf356be7ca8b59b00a7fbc288ee7908e",
        )
    )
    checks["v0141_stdout_bound"] = stdout_141.is_file() and all(
        token in stdout_141.read_text(encoding="utf-8")
        for token in (
            EXPECTED_STATUSES["v0.14.1"],
            "de264dd5be7d681e00e2e40ec3f188823d6861fc3aa4ccb5de467eb16ee68d8d",
            "acc06340c1750f8a225b539527b0bd4e493c2fa6a92bbe395fcd9bf0b64a6bbe",
        )
    )

    print(json.dumps(checks, indent=2, sort_keys=True))
    passed = bool(checks) and all(checks.values())
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
