"""Chronological, reproducible evaluation of stock-direction Bayesian networks.

All feature engineering is causal. Discretization is learned on the training
partition only, model selection uses validation data, and the final test period
is neither used for fitting nor for tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__:
    from .data_processing import (
        DEFAULT_RAW_DATA,
        MODEL_FEATURES,
        TARGET_DEFINITIONS,
        load_data,
        prepare_features,
    )
    from .date_validation import (
        parse_daily_dates,
        require_verified_daily_date_source,
    )
    from .market_calendar import (
        _KNOWN_EXCHANGES,
        inferred_equity_sessions,
        validate_exchange_open_dates,
    )
    from .session_validation import validate_session_positions
else:
    from data_processing import (
        DEFAULT_RAW_DATA,
        MODEL_FEATURES,
        TARGET_DEFINITIONS,
        load_data,
        prepare_features,
    )
    from date_validation import parse_daily_dates, require_verified_daily_date_source
    from market_calendar import (
        _KNOWN_EXCHANGES,
        inferred_equity_sessions,
        validate_exchange_open_dates,
    )
    from session_validation import validate_session_positions


BASELINE_FEATURES = ("Return", "Intraday_Return", "Daily_Range")
ENHANCED_FEATURES = MODEL_FEATURES
DEFAULT_METRICS = (
    Path(__file__).resolve().parents[1] / "data/generated/model_metrics.json"
)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ChronologicalSplit:
    """Split globally by observation date, purging cross-period target labels."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Train and validation fractions must leave a test period")
    if df.empty or "Target_Date" not in df or "Date" not in df:
        raise ValueError("A nonempty, feature-prepared DataFrame is required")
    date = parse_daily_dates(df["Date"], context="Feature dates")
    target_date = parse_daily_dates(df["Target_Date"], context="Target dates")
    if (target_date <= date).any():
        raise ValueError("Every Target_Date must follow its observation Date")
    require_verified_daily_date_source(df)
    require_verified_daily_date_source(df, column="Target_Date")
    validate_session_positions(df)
    if "Exchange" in df:
        labels = df["Exchange"].astype(str).str.strip().str.upper()
        for exchange, indices in df.groupby(labels, sort=False).indices.items():
            validate_exchange_open_dates(
                pd.DatetimeIndex(target_date.iloc[indices]),
                str(exchange),
                verified_session_ordinals="Session_Number" in df,
            )
    else:
        validate_exchange_open_dates(pd.DatetimeIndex(target_date))

    # Validate the supervised horizon itself, not only chronological ordering.
    # Custom/historical markets have no independently inferrable schedule; they
    # must provide the ordinal of the target session at feature preparation.
    if "Target_Session" in df:
        if "Session_Number" not in df:
            raise ValueError("Target_Session requires Session_Number metadata")
        source_positions = df["Session_Number"]
        target_positions = df["Target_Session"]
        for positions, label in (
            (source_positions, "Session_Number"),
            (target_positions, "Target_Session"),
        ):
            if (
                positions.isna().any()
                or not pd.api.types.is_integer_dtype(positions.dtype)
                or pd.api.types.is_bool_dtype(positions.dtype)
                or (positions < 0).any()
            ):
                raise ValueError(f"{label} must contain nonnegative integer ordinals")
        if any(
            int(target) != int(source) + 1
            for source, target in zip(source_positions, target_positions, strict=True)
        ):
            raise ValueError("Target_Date must be the next market session")
    labels = (
        df["Exchange"].astype(str).str.strip().str.upper()
        if "Exchange" in df
        else pd.Series("NYSE", index=df.index)
    )
    for exchange, indices in df.groupby(labels, sort=False).indices.items():
        source_dates = pd.DatetimeIndex(date.iloc[indices])
        destination_dates = pd.DatetimeIndex(target_date.iloc[indices])
        if exchange not in _KNOWN_EXCHANGES or source_dates.min() < pd.Timestamp(
            "1971-01-01"
        ):
            if "Target_Session" not in df:
                raise ValueError(
                    "Custom or historical market targets require verified "
                    "Target_Session metadata"
                )
            continue
        sessions = inferred_equity_sessions(
            source_dates,
            start=source_dates.min(),
            end=destination_dates.max(),
            exchange=str(exchange),
        )
        start_positions = sessions.get_indexer(source_dates)
        end_positions = sessions.get_indexer(destination_dates)
        if ((end_positions - start_positions) != 1).any():
            raise ValueError("Target_Date must be the next market session")
    dates = pd.DatetimeIndex(date.unique()).sort_values()
    if len(dates) < 15:
        raise ValueError("At least 15 distinct trading dates are required")
    first_cut = int(len(dates) * train_fraction)
    second_cut = int(len(dates) * (train_fraction + validation_fraction))
    train_end, validation_end = dates[first_cut - 1], dates[second_cut - 1]
    # A row at the split boundary could have a next-session label drawn from the
    # following period; those labels must not be available to earlier models.
    train = df.loc[(date <= train_end) & (target_date <= train_end)].copy()
    validation = df.loc[
        (date > train_end) & (date <= validation_end) & (target_date <= validation_end)
    ].copy()
    test = df.loc[date > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("All three chronological partitions must contain samples")
    if "Name" in df.columns:
        requested = set(df["Name"])
        missing = {
            label: sorted(requested - set(partition["Name"]))
            for label, partition in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        }
        if any(missing.values()):
            details = "; ".join(
                f"{label}: {', '.join(symbols)}"
                for label, symbols in missing.items()
                if symbols
            )
            raise ValueError(
                f"Missing ticker coverage in chronological partitions: {details}"
            )
    return ChronologicalSplit(train, validation, test, train_end, validation_end)


class QuantileBinner:
    """Train-fitted discretizer. No quantile is estimated from held-out rows."""

    def __init__(self, num_bins: int = 4):
        integer_count = isinstance(num_bins, Integral) and not isinstance(
            num_bins, (bool, np.bool_)
        )
        if not integer_count or not 2 <= num_bins <= 20:
            raise ValueError("num_bins must be an integer between 2 and 20")
        self.num_bins = int(num_bins)
        self.edges: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, features: tuple[str, ...]) -> "QuantileBinner":
        new_edges = {}
        for feature in features:
            values = df[feature].to_numpy(dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Feature {feature} contains missing/nonfinite values")
            quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]
            with np.errstate(over="ignore", invalid="ignore"):
                edges = np.quantile(values, quantiles)
            if not np.isfinite(edges).all():
                # Scaling avoids overflowing a subtraction between two finite
                # values of opposite sign during quantile interpolation.
                scale = np.max(np.abs(values))
                edges = np.quantile(values / scale, quantiles) * scale
            if not np.isfinite(edges).all():
                raise ValueError(f"Could not calculate finite quantiles for {feature}")
            new_edges[feature] = np.unique(edges)
        self.edges = new_edges
        return self

    def transform(self, df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for feature in features:
            if feature not in self.edges:
                raise RuntimeError(f"Binner not fitted for {feature}")
            values = df[feature].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Feature {feature} contains missing/nonfinite values")
            result[feature] = np.searchsorted(self.edges[feature], values, side="right")
        return result


def fit_network(
    df: pd.DataFrame,
    features: tuple[str, ...],
    num_bins: int,
    equivalent_sample_size: int,
) -> DiscreteBayesianNetwork:
    """Fit a simple, interpretable feature-to-next-session-direction network."""
    if (
        not isinstance(equivalent_sample_size, Real)
        or isinstance(equivalent_sample_size, (bool, np.bool_))
        or not np.isfinite(equivalent_sample_size)
        or equivalent_sample_size <= 0
    ):
        raise ValueError("BDeu equivalent_sample_size must be a finite positive number")
    network = DiscreteBayesianNetwork([(name, "Trend") for name in features])
    # Specify the full bin vocabulary even when a particular training period
    # does not observe one of its bins.
    state_names = {name: list(range(num_bins)) for name in features}
    state_names["Trend"] = [0, 1]
    network.fit(
        df[[*features, "Trend"]],
        estimator=DiscreteBayesianEstimator(
            prior_type="BDeu",
            equivalent_sample_size=equivalent_sample_size,
            state_names=state_names,
        ),
    )
    if not network.check_model():
        raise RuntimeError("Bayesian network parameters are not valid")
    return network


def predict_up_probability(
    network: DiscreteBayesianNetwork, frame: pd.DataFrame
) -> np.ndarray:
    """Vectorized posterior lookup for the fully observed feature-parent graph."""
    conditional = network.get_cpds("Trend")
    parents = conditional.variables[1:]
    positions = []
    for parent in parents:
        name_to_index = {
            state: i for i, state in enumerate(conditional.state_names[parent])
        }
        if parent not in frame:
            raise ValueError(f"Missing Bayesian-network parent feature: {parent}")
        mapped = frame[parent].map(name_to_index)
        if mapped.isna().any():
            invalid = frame.loc[mapped.isna(), parent].head(3).tolist()
            raise ValueError(f"Unknown Bayesian-network states for {parent}: {invalid}")
        positions.append(mapped.to_numpy(dtype=int))
    up_index = conditional.state_names["Trend"].index(1)
    return np.asarray(conditional.values[(up_index, *positions)], dtype=float)


def classification_metrics(
    labels: pd.Series, probabilities: np.ndarray
) -> dict[str, float]:
    supplied_labels = labels.to_numpy()
    if not np.isin(supplied_labels, (0, 1)).all():
        raise ValueError("Classification labels must be binary (0 or 1)")
    actual = supplied_labels.astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != actual.shape
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Probabilities must be finite values between 0 and 1 for every label"
        )
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
    }
    result["roc_auc"] = (
        float(roc_auc_score(actual, probabilities))
        if len(np.unique(actual)) == 2
        else None
    )
    return result


def train_and_select(
    split: ChronologicalSplit,
    features: tuple[str, ...],
    num_bins: int,
    sample_sizes: tuple[int, ...] = (5, 10, 15, 20),
    *,
    include_predictions: bool = False,
) -> dict:
    """Select prior strength on validation; refit and score on test once."""
    binner = QuantileBinner(num_bins).fit(split.train, features)
    train = binner.transform(split.train, features)
    train["Trend"] = split.train["Trend"].to_numpy(dtype=int)
    validation = binner.transform(split.validation, features)

    choices = []
    for size in sample_sizes:
        candidate = fit_network(train, features, num_bins, size)
        probability = predict_up_probability(candidate, validation)
        metrics = classification_metrics(split.validation["Trend"], probability)
        choices.append(
            (metrics["balanced_accuracy"], metrics["accuracy"], size, metrics)
        )

    # All hyperparameters are decided before looking at test performance.
    _, _, best_size, validation_metrics = max(
        choices, key=lambda item: (item[0], item[1], -item[2])
    )
    combined = pd.concat([split.train, split.validation], ignore_index=True)
    final_binner = QuantileBinner(num_bins).fit(combined, features)
    combined_discrete = final_binner.transform(combined, features)
    combined_discrete["Trend"] = combined["Trend"].to_numpy(dtype=int)
    final_network = fit_network(combined_discrete, features, num_bins, best_size)
    test = final_binner.transform(split.test, features)
    probabilities = predict_up_probability(final_network, test)
    result = {
        "features": list(features),
        "num_bins": num_bins,
        "equivalent_sample_size": best_size,
        "validation": validation_metrics,
        "test": classification_metrics(split.test["Trend"], probabilities),
    }
    if include_predictions:
        # Internal research API. The ordinary CLI result remains JSON-ready.
        result["test_probabilities"] = probabilities
    return result


def run_experiment(
    raw_data: str | Path = DEFAULT_RAW_DATA,
    *,
    tickers: tuple[str, ...] | None = None,
    num_bins: int = 4,
    target: str = "close_to_close",
) -> dict:
    data = load_data(raw_data, tickers=tickers)
    frame = prepare_features(data, target=target, verified_datetime_dates=True)
    split = chronological_split(frame)
    baseline = train_and_select(split, BASELINE_FEATURES, num_bins)
    enhanced = train_and_select(split, ENHANCED_FEATURES, num_bins)
    majority_label = int(split.train["Trend"].mean() >= 0.5)
    majority = float((split.test["Trend"] == majority_label).mean())
    return {
        "source": str(Path(raw_data).name),
        "target": target,
        "target_definition": TARGET_DEFINITIONS[target],
        "tickers": int(frame["Name"].nunique()),
        "tickers_per_partition": {
            "train": int(split.train["Name"].nunique()),
            "validation": int(split.validation["Name"].nunique()),
            "test": int(split.test["Name"].nunique()),
        },
        "observations": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "periods": {
            "train_end": split.train_end.date().isoformat(),
            "validation_end": split.validation_end.date().isoformat(),
            "test_start": pd.Timestamp(split.test["Date"].min()).date().isoformat(),
            "test_end": pd.Timestamp(split.test["Date"].max()).date().isoformat(),
        },
        "majority_class_test_accuracy": majority,
        "baseline": baseline,
        "enhanced": enhanced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_DATA)
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Repeat to select multiple symbols",
    )
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument(
        "--target", choices=tuple(TARGET_DEFINITIONS), default="close_to_close"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    result = run_experiment(
        args.input,
        tickers=tuple(args.tickers) if args.tickers else None,
        num_bins=args.num_bins,
        target=args.target,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
