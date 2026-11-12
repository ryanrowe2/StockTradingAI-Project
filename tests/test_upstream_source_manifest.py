"""Publisher origin is independently checked against immutable Git blobs."""

from __future__ import annotations

import json

import pytest

from scripts.verify_upstream_source import MANIFEST, verify_publisher_sources


def test_all_31_bundled_ticker_csvs_match_publishers_git_blobs():
    assert verify_publisher_sources() == 31


def test_publisher_verifier_rejects_modified_source_digest(tmp_path):
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata["files_git_blob_sha1"]["AAPL_2006-01-01_to_2018-01-01.csv"] = "0" * 40
    candidate = tmp_path / "publisher.json"
    candidate.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from publisher"):
        verify_publisher_sources(manifest_path=candidate)


def test_publisher_verifier_rejects_missing_ticker_or_changed_source_commit(tmp_path):
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata["files_git_blob_sha1"].pop("AAPL_2006-01-01_to_2018-01-01.csv")
    candidate = tmp_path / "publisher.json"
    candidate.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="source file set"):
        verify_publisher_sources(manifest_path=candidate)

    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata["source_commit"] = "not-the-original-commit"
    candidate.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable commit"):
        verify_publisher_sources(manifest_path=candidate)
