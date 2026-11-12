"""Guardrails for native lock construction and automated branch hygiene."""

from __future__ import annotations

import hashlib
import urllib.error

import pytest

from scripts import cleanup_merged_branches as cleanup
from scripts.generate_portable_lock import lock_from_wheels, normalized


def branch(name: str, sha: str = "a" * 40, protected: bool = False) -> dict:
    return {
        "name": name,
        "commit": {"sha": sha},
        "protected": protected,
    }


def test_normalized_wheel_package_names() -> None:
    assert normalized("Scikit_Learn") == "scikit-learn"
    with pytest.raises(ValueError, match="Invalid distribution"):
        normalized("bad package/name")


def test_lock_contains_exact_wheel_sha256(tmp_path) -> None:
    filename = tmp_path / "example_pkg-1.2.3-py3-none-any.whl"
    data = b"exact-native-distribution-bytes"
    filename.write_bytes(data)
    result = lock_from_wheels([("example-pkg", "1.2.3")], tmp_path)
    expected = hashlib.sha256(data).hexdigest()
    assert f"example-pkg==1.2.3 --hash=sha256:{expected}" in result
    assert result.endswith("\n")


def test_wheel_lock_rejects_incomplete_or_extra_package_set(tmp_path) -> None:
    with pytest.raises(ValueError, match="Wheel set mismatch"):
        lock_from_wheels([("example-pkg", "1.2.3")], tmp_path)
    (tmp_path / "unrelated-1.0-py3-none-any.whl").write_bytes(b"test")
    with pytest.raises(ValueError, match="Wheel set mismatch"):
        lock_from_wheels([("example-pkg", "1.2.3")], tmp_path)


@pytest.mark.parametrize("name", ["Milestone3", "Milestone2", "main"])
def test_historical_and_production_branches_are_never_pruned(name) -> None:
    assert cleanup.eligible(branch(name), {}) is None
    assert cleanup.eligible(branch(name, protected=True), {}) is None


def test_unrelated_branch_and_protected_fix_are_preserved() -> None:
    assert cleanup.eligible(branch("feature/research"), {}) is None
    assert (
        cleanup.eligible(
            branch("fix/example", protected=True), {"fix/example": "a" * 40}
        )
        is None
    )


def test_exact_merged_pull_request_head_is_eligible() -> None:
    assert (
        cleanup.eligible(branch("fix/example"), {"fix/example": "a" * 40})
        == "matching merged pull request"
    )


def test_moved_or_divergent_pr_head_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(
        cleanup, "request", lambda path: {"behind_by": 3, "status": "diverged"}
    )
    assert cleanup.eligible(branch("fix/example"), {"fix/example": "b" * 40}) is None


def test_true_ancestor_temporary_branch_can_be_cleaned(monkeypatch) -> None:
    monkeypatch.setattr(
        cleanup, "request", lambda path: {"behind_by": 0, "status": "ahead"}
    )
    assert (
        cleanup.eligible(branch("fix/old-merged"), {})
        == "branch history already contained in Milestone3"
    )


def test_superseded_pr_only_matches_verified_head() -> None:
    name = "fix/audit-20261111-session-bootstrap-integrity"
    sha = cleanup.SUPERSEDED[name]
    assert "superseded" in cleanup.eligible(branch(name, sha), {})


def test_transient_delete_502_rechecks_matching_sha(monkeypatch) -> None:
    events: list[str] = []

    def api(path: str, *, method: str = "GET") -> object:
        events.append(method)
        if method == "DELETE" and events.count("DELETE") == 1:
            raise urllib.error.HTTPError(path, 502, "Bad Gateway", {}, None)
        return {"object": {"sha": "a" * 40}}

    monkeypatch.setattr(cleanup, "request", api)
    monkeypatch.setattr(cleanup.time, "sleep", lambda seconds: None)
    cleanup.delete_verified_branch("fix/safe", "a" * 40)
    assert events == ["GET", "DELETE", "GET", "DELETE"]


def test_transient_delete_never_removes_moved_branch(monkeypatch) -> None:
    count = 0

    def api(path: str, *, method: str = "GET") -> object:
        nonlocal count
        count += 1
        if count == 1:
            return {"object": {"sha": "a" * 40}}
        if count == 2:
            raise urllib.error.HTTPError(path, 502, "Bad Gateway", {}, None)
        return {"object": {"sha": "b" * 40}}

    monkeypatch.setattr(cleanup, "request", api)
    monkeypatch.setattr(cleanup.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Unsafe branch-head change"):
        cleanup.delete_verified_branch("fix/moved", "a" * 40)
    assert count == 3
