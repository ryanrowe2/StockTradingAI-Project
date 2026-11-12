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
    from .date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from .market_calendar import (
        _KNOWN_EXCHANGES,
        inferred_equity_sessions,
    )
    from .session_validation import (
        _normalized_exchange_labels,
        validate_session_positions,
    )
    from .numeric_validation import binary_label_array, real_float_array
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
    from date_validation import (
        mark_verified_daily_dates,
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from market_calendar import (
        _KNOWN_EXCHANGES,
        inferred_equity_sessions,
    )
    from session_validation import (
        _normalized_exchange_labels,
        validate_session_positions,
    )
    from numeric_validation import binary_label_array, real_float_array
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

# Bounded bootstrap execution: 1,000,000 result rows ~= 32 MB for four
# float64 scores. Draw indices/selected blocks use at most 100,000 cells.
MAX_BOOTSTRAP_SAMPLES = 1_000_000
MAX_BOOTSTRAP_BATCH_CELLS = 100_000
# Cap total random block selections, not only the peak batch allocation.
MAX_BOOTSTRAP_DRAW_CELLS = 50_000_000


def _validate_bootstrap_configuration(
    block_sessions: int, bootstrap_samples: int
) -> None:
    if (
        not isinstance(block_sessions, int)
        or isinstance(block_sessions, bool)
        or block_sessions < 1
        or not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or not 100 <= bootstrap_samples <= MAX_BOOTSTRAP_SAMPLES
    ):
        raise ValueError(
            "block_sessions must be positive and bootstrap_samples must be "
            f"between 100 and {MAX_BOOTSTRAP_SAMPLES:,}"
        )


def _validate_bootstrap_seed(seed: int) -> None:
    """Reject unsupported RNG seeds before loading data or fitting models."""
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or seed < 0
    ):
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    # NumPy also enforces its own range and SeedSequence constraints.
    try:
        np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bootstrap seed must be a valid nonnegative integer") from exc


def _validate_bootstrap_workload(block_count: int, bootstrap_samples: int) -> None:
    """Reject expensive resampling before model fitting or random allocation."""
    total = block_count * bootstrap_samples
    if total > MAX_BOOTSTRAP_DRAW_CELLS:
        raise ValueError(
            "Bootstrap workload exceeds the allowed random-block selection budget "
            f"({total:,} > {MAX_BOOTSTRAP_DRAW_CELLS:,})"
        )
    if block_count > MAX_BOOTSTRAP_BATCH_CELLS:
        raise ValueError("Too many independent blocks for bounded bootstrap batches")


def _market_date_block_ids(
    frame: pd.DataFrame,
    block_sessions: int,
    *,
    verified_datetime_dates: bool = False,
) -> np.ndarray:
    """Return gap-aware, cross-ticker paired blocks for observed market dates.

    Verified historical and custom-market ordinals are sufficient where the
    modern exchange calendar cannot be inferred; known modern markets use
    scheduled sessions even if a source omits some observations.
    """
    if "Date" not in frame or frame.empty:
        raise ValueError("Nonempty market dates are required for resampling")
    if not isinstance(block_sessions, int) or isinstance(block_sessions, bool):
        raise ValueError("block_sessions must be a positive integer")
    if block_sessions < 1:
        raise ValueError("block_sessions must be a positive integer")
    require_verified_daily_date_source(
        frame, verified_datetime_dates=verified_datetime_dates
    )
    dates = parse_daily_dates(frame["Date"], context="Bootstrap dates")
    days = pd.DatetimeIndex(dates.unique()).sort_values()
    columns = ["Date"]
    if "Exchange" in frame:
        columns.append("Exchange")
    if "Session_Number" in frame:
        columns.append("Session_Number")
    metadata = frame.loc[:, columns].copy()
    metadata["Date"] = parse_daily_dates(metadata["Date"], context="Bootstrap dates")
    exchanges = _normalized_exchange_labels(metadata)
    if exchanges is not None:
        metadata["Exchange"] = exchanges
        metadata["Name"] = metadata["Exchange"]
    # Equivalent ticker exchange aliases can contribute observations on the
    # same market date. Normalize before deduplicating, while refusing
    # inconsistent session numbers for a single canonical market/date.
    if "Session_Number" in metadata:
        identities = ["Date", "Exchange"] if exchanges is not None else ["Date"]
        if metadata.groupby(identities, dropna=False)["Session_Number"].nunique().gt(1).any():
            raise ValueError("Conflicting session ordinals for the same market date")
    metadata = metadata.drop_duplicates().copy()
    mark_verified_daily_dates(metadata)
    validate_session_positions(metadata)

    known_calendars = []
    grouping = (
        metadata.groupby("Exchange", sort=False)
        if exchanges is not None
        else (("NYSE", metadata),)
    )
    # Track missing sessions within each market independently: a traded date
    # in a different exchange does not replace an absent London/NYSE session.
    gap_boundaries = np.zeros(len(days), dtype=bool)
    for exchange, group in grouping:
        group = group.sort_values("Date")
        observed = pd.DatetimeIndex(group["Date"])
        if (
            exchange in _KNOWN_EXCHANGES
            and (observed >= pd.Timestamp("1971-01-01")).all()
        ):
            schedule = inferred_equity_sessions(
                observed,
                start=observed.min(),
                end=observed.max(),
                exchange=exchange,
            )
            known_calendars.append(schedule)
            positions = schedule.get_indexer(observed)
            has_gap = np.diff(positions) > 1
        else:
            if "Session_Number" not in group:
                raise ValueError(
                    "Historical and custom exchanges require verified "
                    "Session_Number metadata"
                )
            positions = [int(value) for value in group["Session_Number"]]
            has_gap = [
                following > preceding + 1
                for preceding, following in zip(
                    positions[:-1], positions[1:], strict=True
                )
            ]
        for earlier, missing_session in zip(observed[:-1], has_gap, strict=True):
            if missing_session:
                # A missing market session must separate any shared global
                # observations after 'earlier' from the preceding block.
                # For custom exchanges, its exact missing date is unknown;
                # this conservative boundary never resamples across the gap.
                boundary = days.searchsorted(earlier, side="right")
                if boundary < len(days):
                    gap_boundaries[boundary] = True
    calendar = days
    for known_sessions in known_calendars:
        calendar = calendar.union(known_sessions).sort_values()
    timeline_positions = calendar.get_indexer(days)
    increments = np.diff(timeline_positions, prepend=timeline_positions[0])
    increments[0] = 1
    increments[gap_boundaries] = np.maximum(increments[gap_boundaries], 2)
    blocks = np.empty(len(days), dtype=np.int64)
    current_block = -1
    sessions_in_block = 0
    for index, step in enumerate(increments):
        if index == 0 or step != 1 or sessions_in_block == block_sessions:
            current_block += 1
            sessions_in_block = 0
        blocks[index] = current_block
        sessions_in_block += 1
    return blocks[days.get_indexer(dates)]


def paired_block_accuracy_intervals(
    scored: pd.DataFrame,
    *,
    block_sessions: int = 10,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    verified_datetime_dates: bool = False,
) -> dict[str, list[float]]:
    """Bootstrap *paired accuracy differences* over chronological market-date blocks.

    This is descriptive uncertainty conditional on the fixed research design;
    it does not supply a guarantee of profitability or future generalization.
    """
    _validate_bootstrap_configuration(block_sessions, bootstrap_samples)
    _validate_bootstrap_seed(seed)
    if scored.empty:
        raise ValueError("Cannot bootstrap an empty holdout")
    required = {"Date", "actual", "majority", "baseline_prob", "enhanced_prob"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Missing bootstrap columns: {sorted(missing)}")
    actual_labels = binary_label_array(
        scored["actual"].to_numpy(), context="Bootstrap actual labels"
    )
    majority_labels = binary_label_array(
        scored["majority"].to_numpy(), context="Bootstrap majority labels"
    )
    try:
        probabilities = real_float_array(
            scored[["baseline_prob", "enhanced_prob"]],
            context="Bootstrap probabilities",
        )
    except (TypeError, ValueError) as exc:
        if "complex" in str(exc):
            raise ValueError(
                "Bootstrap probabilities cannot contain complex values"
            ) from exc
        if "finite" in str(exc):
            raise ValueError(
                "Bootstrap probabilities must be finite values in [0, 1]"
            ) from exc
        raise ValueError(
            "Bootstrap probabilities must be numeric values in [0, 1]"
        ) from exc
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Bootstrap probabilities must be finite values in [0, 1]")
    block_id = _market_date_block_ids(
        scored, block_sessions, verified_datetime_dates=verified_datetime_dates
    )
    if len(np.unique(block_id)) < 2:
        raise ValueError("Bootstrap requires at least two chronological date blocks")
    expected = (actual_labels == majority_labels).astype(int)
    actual = actual_labels
    baseline = (actual == (probabilities[:, 0] >= 0.5)).astype(int)
    enhanced = (actual == (probabilities[:, 1] >= 0.5)).astype(int)
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
    _validate_bootstrap_workload(len(blocks), bootstrap_samples)
    rng = np.random.default_rng(seed)
    samples = np.empty((bootstrap_samples, blocks.shape[1]), dtype=np.float64)
    batch_size = max(1, MAX_BOOTSTRAP_BATCH_CELLS // len(blocks))
    for first in range(0, bootstrap_samples, batch_size):
        count = min(batch_size, bootstrap_samples - first)
        draws = rng.integers(0, len(blocks), size=(count, len(blocks)))
        samples[first : first + count] = blocks[draws].sum(axis=1)
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
    _validate_bootstrap_seed(seed)
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
        symbols = set(prepared["Name"])
        if any(set(part["Name"]) != symbols for part in (train, validation, test)):
            raise ValueError(f"Fold {number + 1} has incomplete ticker coverage")
        fold_partitions.append(
            (number, train_end, validation_end, test_end, train, validation, test)
        )
    # The published bootstrap is pooled across folds. Check pooled holdout
    # feasibility before fitting rather than requiring two blocks per fold.
    planned_holdout = pd.concat(
        [partition[-1] for partition in fold_partitions], ignore_index=True
    )
    planned_blocks = len(
        np.unique(_market_date_block_ids(planned_holdout, block_sessions))
    )
    if planned_blocks < 2:
        raise ValueError(
            "Walk-forward holdout has too few independent trading-session "
            "blocks for bootstrap evaluation"
        )
    _validate_bootstrap_workload(planned_blocks, bootstrap_samples)
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
                    **(
                        {"Exchange": test["Exchange"].to_numpy()}
                        if "Exchange" in test
                        else {}
                    ),
                    **(
                        {"Session_Number": test["Session_Number"].to_numpy()}
                        if "Session_Number" in test
                        else {}
                    ),
                    "actual": test["Trend"].to_numpy(dtype=int),
                    "majority": majority,
                    "baseline_prob": baseline["test_probabilities"],
                    "enhanced_prob": enhanced["test_probabilities"],
                }
            )
        )
    scored = mark_verified_daily_dates(pd.concat(scored_folds, ignore_index=True))
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
