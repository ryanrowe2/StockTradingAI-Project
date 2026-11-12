"""Deterministically reconcile bundled price files and release-internal SHA-256.

The manifest pins the bytes in this repository. It is not a publisher-signed
upstream attestation and does not assert authorization to redistribute data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "source_data_manifest.json"
FIELDS = ("Date", "Name", "Open", "High", "Low", "Close", "Volume")
COMBINED = "all_stocks_2006-01-01_to_2018-01-01.csv"
LATEST = "all_stocks_2017-01-01_to_2018-01-01.csv"


def _contents(path: Path) -> Counter[tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != set(FIELDS):
            raise ValueError(f"Incorrect bundled CSV schema: {path.name}")
        return Counter(tuple(row[name] for name in FIELDS) for row in reader)


def calculate_manifest(data_dir: Path = DATA_DIR) -> dict:
    individual = sorted(data_dir.glob("*_2006-01-01_to_2018-01-01.csv"))
    individual = [path for path in individual if path.name != COMBINED]
    if len(individual) != 31:
        raise ValueError("Expected exactly 31 bundled ticker files")
    files = [data_dir / COMBINED, data_dir / LATEST, *individual]
    if any(not path.is_file() for path in files):
        raise ValueError("A bundled historical source file is missing")
    master = _contents(files[0])
    concatenated = Counter()
    for path in individual:
        ticker = path.name.split("_", 1)[0]
        rows = _contents(path)
        if any(row[1] != ticker for row in rows):
            raise ValueError(f"Incorrect source ticker identity in {path.name}")
        concatenated.update(rows)
    if concatenated != master:
        missing = sum((concatenated - master).values())
        extra = sum((master - concatenated).values())
        raise ValueError(
            f"Combined source differs from individual ticker files: {missing} "
            f"individual-only rows, {extra} combined-only rows"
        )
    subset = Counter(row for row, count in master.items() for _ in range(count) if row[0].startswith("2017-"))
    if _contents(files[1]) != subset:
        raise ValueError("2017 source is not the corresponding complete-data subset")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files, key=lambda item: item.name)
    }
    return {
        "schema_version": 1,
        "publisher": "szrlee",
        "dataset": "stock-time-series-20050101-to-20171231",
        "integrity_scope": "release-internal SHA-256; not publisher-signed",
        "row_count": sum(master.values()),
        "ticker_file_count": len(individual),
        "files_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="Print candidate manifest for review")
    args = parser.parse_args()
    observed = calculate_manifest()
    if args.emit:
        print("SOURCE_MANIFEST_JSON=" + json.dumps(observed, sort_keys=True))
        return
    if not MANIFEST.is_file():
        raise ValueError("Source-data manifest missing; generate and review --emit output")
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if expected != observed:
        raise ValueError("Source-data SHA-256 manifest or row reconciliation has changed")
    print(f"Verified {len(observed['files_sha256'])} bundled source files and "
          f"{observed['row_count']:,} unique ticker/day source rows")


if __name__ == "__main__":
    main()
