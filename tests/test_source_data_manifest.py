"""Source-data SHA-256 verification and full combined-file reconciliation."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.verify_source_data import MANIFEST, calculate_manifest


def test_all_bundled_price_files_match_reviewed_source_manifest():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert calculate_manifest() == expected
    assert expected["row_count"] == 93612
    assert len(expected["files_sha256"]) == 33


def _synthetic_bundle(root):
    fields = ["Date", "Name", "Open", "High", "Low", "Close", "Volume"]
    rows = []
    for number in range(31):
        ticker = f"S{number:02d}"
        row = dict(
            zip(
                fields,
                ["2017-01-03", ticker, "10", "11", "9", "10", "100"],
                strict=True,
            )
        )
        rows.append(row)
        path = root / f"{ticker}_2006-01-01_to_2018-01-01.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
    for name in (
        "all_stocks_2006-01-01_to_2018-01-01.csv",
        "all_stocks_2017-01-01_to_2018-01-01.csv",
    ):
        with (root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def test_reconciliation_rejects_corrupted_ticker_identity(tmp_path):
    _synthetic_bundle(tmp_path)
    assert calculate_manifest(tmp_path)["row_count"] == 31
    file = tmp_path / "S00_2006-01-01_to_2018-01-01.csv"
    file.write_text(
        file.read_text(encoding="utf-8").replace("S00", "WRONG"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source ticker identity"):
        calculate_manifest(tmp_path)


def test_reconciliation_rejects_modified_combined_prices(tmp_path):
    _synthetic_bundle(tmp_path)
    file = tmp_path / "all_stocks_2006-01-01_to_2018-01-01.csv"
    file.write_text(
        file.read_text(encoding="utf-8").replace(",10,11,9,10,100", ",12,13,9,10,100"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Combined source differs"):
        calculate_manifest(tmp_path)


def test_reconciliation_rejects_missing_source_files(tmp_path):
    with pytest.raises(ValueError, match="31 bundled ticker"):
        calculate_manifest(tmp_path)


def test_reconciliation_rejects_corrupted_2017_subset(tmp_path):
    _synthetic_bundle(tmp_path)
    file = tmp_path / "all_stocks_2017-01-01_to_2018-01-01.csv"
    file.write_text(
        file.read_text(encoding="utf-8").replace("S01", "WRONG"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="2017 source"):
        calculate_manifest(tmp_path)


def test_reconciliation_rejects_malformed_source_schema(tmp_path):
    _synthetic_bundle(tmp_path)
    file = tmp_path / "S00_2006-01-01_to_2018-01-01.csv"
    file.write_text("Date,Name,Open\n2017-01-03,S00,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incorrect bundled CSV schema"):
        calculate_manifest(tmp_path)
