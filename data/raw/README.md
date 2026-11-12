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
- The live publisher-selected license on Kaggle could not be independently read through the automated review tools. The archived metadata supports CC0 classification, but does not substitute for current, direct publisher-license verification. Preserve the source and attribution details if the CSVs are redistributed.

These files remain third-party source data. Do not relabel them as code created by this repository's author or subject them to the software's MIT copyright claim. For definitive redistribution clearance, inspect the license on the live Kaggle listing and retain the publisher-selected license record. If contrary restrictions are discovered, remove or replace the bundled CSVs.
