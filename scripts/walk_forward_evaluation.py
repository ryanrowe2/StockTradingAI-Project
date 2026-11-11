"""Expanding-window out-of-sample evaluation with paired date-block intervals.

Folds use earlier dates for fitting and hyperparameter selection, followed by
disjoint forward test windows. Bootstrap blocks keep all tickers from the same
market dates together, reducing false precision from cross-ticker correlation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .data_processing import (
        DEFAULT_RAW_DATA,
        TARGET_DEFINITIONS,
        load_data,
        prepare_features,
    )
    from .date_validation import parse_daily_dates
    from .model_training import (
        BASELINE_FEATURES,
        ENHANCED_FEATURES,
        ChronologicalSplit,
        classification_metrics,
        train_and_select,
    )
else:
    from data_processing import (
        DEFAULT_RAW_DATA,
        TARGET_DEFINITIONS,
        load_data,
        prepare_features,
    )
    from date_validation import parse_daily_dates
    from model_training import (
        BASELINE_FEATURES,
        ENHANCED_FEATURES,
        ChronologicalSplit,
        classification_metrics,
        train_and_select,
    )


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "data/generated/walk_forward_metrics.json"
)


def _validate_bootstrap_configuration(
    block_sessions: int, bootstrap_samples: int
) -> None:
    if (
        not isinstance(block_sessions, int)
        or isinstance(block_sessions, bool)
        or block_sessions < 1
        or not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        raise ValueError(
            "block_sessions must be positive and bootstrap_samples at least 100"
        )


def paired_block_accuracy_intervals(
    scored: pd.DataFrame,
    *,
    block_sessions: int = 10,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Bootstrap *paired accuracy differences* over chronological market-date blocks.

    This is descriptive uncertainty conditional on the fixed research design;
    it does not supply a guarantee of profitability or future generalization.
    """
    _validate_bootstrap_configuration(block_sessions, bootstrap_samples)
    if scored.empty:
        raise ValueError("Cannot bootstrap an empty holdout")
    required = {"Date", "actual", "majority", "baseline_prob", "enhanced_prob"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Missing bootstrap columns: {sorted(missing)}")
    if (
        not scored["actual"].isin((0, 1)).all()
        or not scored["majority"].isin((0, 1)).all()
    ):
        raise ValueError("Bootstrap actual and majority labels must be binary (0 or 1)")
    try:
        probabilities = scored[["baseline_prob", "enhanced_prob"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Bootstrap probabilities must be numeric values in [0, 1]"
        ) from exc
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Bootstrap probabilities must be finite values in [0, 1]")
    dates = parse_daily_dates(scored["Date"], context="Bootstrap dates")
    days = pd.DatetimeIndex(dates.unique()).sort_values()
    block_id = days.get_indexer(dates) // block_sessions
    if len(np.unique(block_id)) < 2:
        raise ValueError("Bootstrap requires at least two chronological date blocks")
    expected = (scored["actual"] == scored["majority"]).to_numpy(dtype=int)
    baseline = (scored["actual"] == (scored["baseline_prob"] >= 0.5)).to_numpy(
        dtype=int
    )
    enhanced = (scored["actual"] == (scored["enhanced_prob"] >= 0.5)).to_numpy(
        dtype=int
    )
    blocks = (
        pd.DataFrame(
            {
                "block": block_id,
                "count": 1,
                "majority": expected,
                "baseline": baseline,
                "enhanced": enhanced,
            }
        )
        .groupby("block", sort=True)
        .sum()
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(bootstrap_samples, len(blocks)))
    samples = blocks[draws].sum(axis=1)
    denominators = samples[:, 0]
    benchmark_accuracy = samples[:, 1] / denominators
    result = {}
    for label, column in (("baseline", 2), ("enhanced", 3)):
        differences = samples[:, column] / denominators - benchmark_accuracy
        result[label] = [
            float(value) for value in np.quantile(differences, [0.025, 0.975])
        ]
    return result


def walk_forward_experiment(
    raw_data: str | Path = DEFAULT_RAW_DATA,
    *,
    tickers: tuple[str, ...] | None = None,
    target: str = "next_open_to_close",
    num_bins: int = 4,
    folds: int = 5,
    block_sessions: int = 10,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluate five possible expanding folds without reusing any test observation.

    Fold zero trains on the first 40% of eligible dates, validates on the
    following 10%, and tests on the next 10%. Each additional fold advances
    those boundaries by 10%, incorporating earlier test history into later
    training only after that history would have become observable.
    """
    if target not in TARGET_DEFINITIONS:
        raise ValueError(f"target must be one of {sorted(TARGET_DEFINITIONS)}")
    if not isinstance(folds, int) or isinstance(folds, bool) or not 1 <= folds <= 5:
        raise ValueError("folds must be an integer between 1 and 5")
    _validate_bootstrap_configuration(block_sessions, bootstrap_samples)
    data = load_data(raw_data, tickers=tickers)
    prepared = prepare_features(data, target=target, verified_datetime_dates=True)
    dates = pd.DatetimeIndex(prepared["Date"].unique()).sort_values()
    if len(dates) < 80:
        raise ValueError("Walk-forward evaluation requires at least 80 eligible dates")
    date = prepared["Date"]
    target_date = prepared["Target_Date"]
    # Check every proposed holdout before any potentially expensive fitting.
    fold_partitions = []
    for number in range(folds):
        train_end = dates[(len(dates) * (4 + number)) // 10 - 1]
        validation_end = dates[(len(dates) * (5 + number)) // 10 - 1]
        test_end = dates[(len(dates) * (6 + number)) // 10 - 1]
        train = prepared.loc[(date <= train_end) & (target_date <= train_end)].copy()
        validation = prepared.loc[
            (date > train_end)
            & (date <= validation_end)
            & (target_date <= validation_end)
        ].copy()
        test = prepared.loc[
            (date > validation_end) & (date <= test_end) & (target_date <= test_end)
        ].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"Fold {number + 1} has an empty chronological partition")
        if test["Date"].nunique() <= block_sessions:
            raise ValueError(
                f"Fold {number + 1} has too few test dates for two "
                f"{block_sessions}-session bootstrap blocks"
            )
        symbols = set(prepared["Name"])
        if any(set(part["Name"]) != symbols for part in (train, validation, test)):
            raise ValueError(f"Fold {number + 1} has incomplete ticker coverage")
        fold_partitions.append(
            (number, train_end, validation_end, test_end, train, validation, test)
        )
    fold_reports = []
    scored_folds = []
    scored_row_indices: set[int] = set()
    for (
        number,
        train_end,
        validation_end,
        test_end,
        train,
        validation,
        test,
    ) in fold_partitions:
        if scored_row_indices.intersection(test.index):
            raise RuntimeError("Walk-forward test folds must not overlap")
        scored_row_indices.update(test.index)
        split = ChronologicalSplit(train, validation, test, train_end, validation_end)
        baseline = train_and_select(
            split, BASELINE_FEATURES, num_bins, include_predictions=True
        )
        enhanced = train_and_select(
            split, ENHANCED_FEATURES, num_bins, include_predictions=True
        )
        majority = int(train["Trend"].mean() >= 0.5)
        fold_reports.append(
            {
                "fold": number + 1,
                "train_end": train_end.date().isoformat(),
                "validation_end": validation_end.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "observations": {
                    "train": len(train),
                    "validation": len(validation),
                    "test": len(test),
                },
                "majority_accuracy": float((test["Trend"] == majority).mean()),
                "baseline": baseline["test"],
                "enhanced": enhanced["test"],
                "selected_prior": {
                    "baseline": baseline["equivalent_sample_size"],
                    "enhanced": enhanced["equivalent_sample_size"],
                },
            }
        )
        scored_folds.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].to_numpy(),
                    "actual": test["Trend"].to_numpy(dtype=int),
                    "majority": majority,
                    "baseline_prob": baseline["test_probabilities"],
                    "enhanced_prob": enhanced["test_probabilities"],
                }
            )
        )
    scored = pd.concat(scored_folds, ignore_index=True)
    intervals = paired_block_accuracy_intervals(
        scored,
        block_sessions=block_sessions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "source": Path(raw_data).name,
        "target": target,
        "target_definition": TARGET_DEFINITIONS[target],
        "folds": fold_reports,
        "holdout_observations": len(scored),
        "majority_accuracy": float((scored["actual"] == scored["majority"]).mean()),
        "baseline": classification_metrics(
            scored["actual"], scored["baseline_prob"].to_numpy()
        ),
        "enhanced": classification_metrics(
            scored["actual"], scored["enhanced_prob"].to_numpy()
        ),
        "accuracy_lift_over_majority_95pct_block_intervals": intervals,
        "bootstrap": {
            "block_sessions": block_sessions,
            "samples": bootstrap_samples,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument(
        "--target", choices=tuple(TARGET_DEFINITIONS), default="next_open_to_close"
    )
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-sessions", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = walk_forward_experiment(
        args.input,
        tickers=tuple(args.tickers) if args.tickers else None,
        target=args.target,
        num_bins=args.num_bins,
        folds=args.folds,
        block_sessions=args.block_sessions,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
