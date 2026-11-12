"""Verify bundled price files against an immutable publisher-owned Git commit.

A local copy may have CRLF line endings while the publisher's Git objects use
LF. Canonicalizing only newlines permits a byte-for-byte content comparison
without rewriting or otherwise modifying the historical observations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "docs" / "upstream_source_manifest.json"


def verify_publisher_sources(
    data_dir: Path = DATA_DIR, manifest_path: Path = MANIFEST
) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files_git_blob_sha1", {})
    sources = sorted(data_dir.glob("*_2006-01-01_to_2018-01-01.csv"))
    names = {source.name for source in sources}
    names.discard("all_stocks_2006-01-01_to_2018-01-01.csv")
    if (
        not isinstance(expected, dict)
        or len(expected) != 31
        or names != set(expected)
        or manifest.get("source_commit") != "8da42f14ae64fca79b9b8b51208e98674fc3d4e9"
    ):
        raise ValueError("Publisher source file set or immutable commit is invalid")
    for name, upstream_blob in expected.items():
        canonical = (data_dir / name).read_bytes().replace(b"\r\n", b"\n")
        payload = b"blob " + str(len(canonical)).encode("ascii") + b"\0" + canonical
        actual = hashlib.sha1(payload).hexdigest()
        if actual != upstream_blob:
            raise ValueError(f"Bundled {name} differs from publisher's Git blob")
    return len(expected)


def main() -> None:
    count = verify_publisher_sources()
    print(
        f"Verified {count} historical ticker CSVs against immutable publisher Git blobs"
    )


if __name__ == "__main__":
    main()
