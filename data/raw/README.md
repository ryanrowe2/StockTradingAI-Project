# Historical price data

The CSV files in this directory originate from the third-party
[Stock Time Series 20050101 to 20171231 dataset](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231).

These source data files are **not covered by this project's MIT License**.
Their use and redistribution are governed by the original dataset's terms.
Refer to the dataset provider for its licensing and attribution information.

## Provenance and redistribution review — 2026-11-12

- Publisher: [szrlee / DJIA 30 Stock Time Series on Kaggle](https://www.kaggle.com/datasets/szrlee/stock-time-series-20050101-to-20171231).
- The original publisher's [GitHub project](https://github.com/szrlee/Stock-Time-Series-Analysis) names these historical CSVs, and documents the original collection workflow. Its repository has no separate dataset LICENSE file.
- All 31 bundled per-ticker CSVs were compared individually against the publisher's immutable Git commit [`8da42f14`](https://github.com/szrlee/Stock-Time-Series-Analysis/commit/8da42f14ae64fca79b9b8b51208e98674fc3d4e9). Every file is identical after converting CRLF line endings to LF; [the complete pinned upstream Git blob register](../../docs/upstream_source_manifest.json) and `python -m scripts.verify_upstream_source` make this claim testable without external network access. The combined CSV and 2017 subset are independently reconciled against these 31 files and pinned with SHA-256 hashes.
- A publicly archived index of Kaggle notebook metadata records the **same dataset slug and publisher** with the designation **"CC0: Public Domain"**: [archived metadata index](https://huggingface.co/datasets/bigcomputer/meta-kaggle-notebook-batch-1/viewer/default/train?p=25).
- [Kaggle's dataset documentation](https://www.kaggle.com/docs/datasets) identifies CC0 as Public Domain.
- **Direct publisher license verified 2026-11-12:** Kaggle's live [dataset API](https://www.kaggle.com/api/v1/datasets/view/szrlee/stock-time-series-20050101-to-20171231) returned dataset ID `8388`, current version `3`, exact dataset ref `szrlee/stock-time-series-20050101-to-20171231`, and `licenseName: "CC0: Public Domain"`. The [independent GitHub-hosted verification log](https://github.com/ryanrowe2/StockTradingAI-Project/actions/runs/43291452611) records the primary-source response. This directly corroborates the archived metadata; no login credentials or secondary license inference were required.

These files remain third-party source data. Do not relabel them as code created by this repository's author or subject them to the software's MIT copyright claim. The publisher's current designation is **CC0: Public Domain** (verified 2026-11-12). Keep the live dataset reference and verification evidence with redistributed copies so the third-party source remains identifiable.
