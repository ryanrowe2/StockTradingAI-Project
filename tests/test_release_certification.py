"""Validate release gates independently from the workflow configuration."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import verify_source_data
from scripts.verify_coverage import MODULE_FLOORS, validate_report


def _coverage_fixture():
    return {
        "meta": {"branch_coverage": True},
        "totals": {"percent_covered": 95.0},
        "files": {
            module: {"summary": {"percent_covered": 100.0}} for module in MODULE_FLOORS
        },
    }


def test_branch_aware_module_gates_accept_verified_report():
    validate_report(_coverage_fixture())


def test_statement_only_report_is_not_branch_certified():
    report = _coverage_fixture()
    report["meta"]["branch_coverage"] = False
    with pytest.raises(ValueError, match="branches"):
        validate_report(report)


def test_low_global_branch_coverage_is_rejected():
    report = _coverage_fixture()
    report["totals"]["percent_covered"] = 87
    with pytest.raises(ValueError, match="below 88"):
        validate_report(report)


def test_missing_module_or_insufficient_module_coverage_rejected():
    report = _coverage_fixture()
    report["files"].pop("scripts/model_training.py")
    with pytest.raises(ValueError, match="model_training"):
        validate_report(report)
    report = _coverage_fixture()
    report["files"]["scripts/walk_forward_evaluation.py"]["summary"][
        "percent_covered"
    ] = 50.0
    with pytest.raises(ValueError, match="walk_forward"):
        validate_report(report)


def test_source_manifest_cli_emits_reviewable_data(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["source-data", "--emit"])
    verify_source_data.main()
    stdout = capsys.readouterr().out
    assert stdout.startswith("SOURCE_MANIFEST_JSON=")
    result = json.loads(stdout.split("=", 1)[1])
    assert result["row_count"] == 93612


def test_source_manifest_cli_checks_committed_digest(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["source-data"])
    verify_source_data.main()
    assert "Verified 33 bundled source files" in capsys.readouterr().out


def test_source_manifest_cli_rejects_missing_reviewed_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["source-data"])
    monkeypatch.setattr(verify_source_data, "MANIFEST", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="manifest missing"):
        verify_source_data.main()
