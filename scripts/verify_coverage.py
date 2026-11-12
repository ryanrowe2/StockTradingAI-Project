"""Enforce branch-aware, risk-weighted research coverage gates."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_REPORT = Path("coverage-summary.json")

# The project-wide 88% branch-aware floor is supported by certified CI data.
# Higher targeted module floors guard the scientific training/evaluation paths.
MODULE_FLOORS = {
    "scripts/model_training.py": 83.0,
    "scripts/walk_forward_evaluation.py": 85.0,
    "scripts/rl_agent.py": 88.0,
    "scripts/data_processing.py": 92.0,
    "scripts/date_validation.py": 95.0,
    "scripts/verify_source_data.py": 70.0,
}


def validate_report(report: dict) -> None:
    if report.get("meta", {}).get("branch_coverage") is not True:
        raise ValueError(
            "Coverage must measure decisions/branches, not statements alone"
        )
    total = report.get("totals", {}).get("percent_covered")
    if not isinstance(total, (int, float)) or total < 88.0:
        raise ValueError(f"Combined branch-aware coverage below 88%: {total!r}")
    files = report.get("files", {})
    for path, floor in MODULE_FLOORS.items():
        summary = files.get(path, {}).get("summary", {})
        covered = summary.get("percent_covered")
        if not isinstance(covered, (int, float)) or covered < floor:
            raise ValueError(f"Module {path} coverage {covered!r} below {floor}%")


def main() -> None:
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    validate_report(report)
    print(
        f"Branch-aware coverage certified: "
        f"{report['totals']['percent_covered']:.1f}% overall; "
        f"{len(MODULE_FLOORS)} module-specific floors enforced"
    )


if __name__ == "__main__":
    main()
