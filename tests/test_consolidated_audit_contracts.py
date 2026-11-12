"""Consolidated audit: independently exercise public contract boundaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.data_processing import validate_bars
from scripts.date_validation import require_verified_daily_date_source
from scripts.hmm_integration import fit_hmm_on_indicators
from scripts.model_training import (
    ChronologicalSplit,
    _validate_direct_model_split,
    fit_network,
)
from scripts.walk_forward_evaluation import _market_date_block_ids


def _sample(date: str, target: str, name: str | None = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {"Name": [name], "Date": [date], "Target_Date": [target], "Trend": [1]}
    )


def _good_split() -> ChronologicalSplit:
    return ChronologicalSplit(
        train=_sample("2024-02-05", "2024-02-06"),
        validation=_sample("2024-02-08", "2024-02-09"),
        test=_sample("2024-02-12", "2024-02-13"),
        train_end=pd.Timestamp("2024-02-06"),
        validation_end=pd.Timestamp("2024-02-09"),
    )


def test_all_three_direct_partitions_obey_shared_market_horizon_contract():
    _validate_direct_model_split(_good_split())


def test_direct_training_rejects_later_but_nonadjacent_market_target():
    split = _good_split()
    bad = replace(split, test=_sample("2024-02-12", "2024-02-23"))
    with pytest.raises(ValueError, match="next market session"):
        _validate_direct_model_split(bad)


def test_direct_training_rejects_nyse_closed_observation():
    split = _good_split()
    bad = replace(split, test=_sample("2024-02-19", "2024-02-20"))
    with pytest.raises(ValueError, match="closed-market date"):
        _validate_direct_model_split(bad)


@pytest.mark.parametrize("name", [None, "", " aapl!", "N/A", "<NA>"])
def test_direct_training_rejects_unusable_ticker_names(name):
    split = _good_split()
    bad = replace(split, test=_sample("2024-02-12", "2024-02-13", name))
    with pytest.raises(ValueError, match="ticker identit"):
        _validate_direct_model_split(bad)


def test_direct_training_requires_stable_ticker_coverage():
    split = _good_split()
    bad = replace(split, validation=_sample("2024-02-08", "2024-02-09", "BBB"))
    with pytest.raises(ValueError, match="identical ticker coverage"):
        _validate_direct_model_split(bad)


@pytest.mark.parametrize("count", [0, 1, 21, 2.5, True, "3"])
def test_direct_bayesian_fitting_rejects_invalid_bin_count(count):
    sample = pd.DataFrame({"Return": [0, 1], "Trend": [0, 1]})
    with pytest.raises(ValueError, match="num_bins"):
        fit_network(sample, ("Return",), count, 5)


@pytest.mark.parametrize("value", [None, -1, 2, 0.5, 1.0, True, 1 + 0j])
def test_direct_bayesian_fitting_rejects_invalid_discrete_states(value):
    sample = pd.DataFrame({"Return": [0, value], "Trend": [0, 1]})
    with pytest.raises(ValueError, match="integer states"):
        fit_network(sample, ("Return",), 2, 5)


def test_direct_bayesian_fitting_rejects_duplicate_features():
    sample = pd.DataFrame({"Return": [0, 1], "Trend": [0, 1]})
    with pytest.raises(ValueError, match="distinct"):
        fit_network(sample, ("Return", "Return"), 2, 5)


def test_raw_unverified_target_date_is_rejected_by_bar_validator():
    bars = pd.DataFrame(
        {
            "Name": ["AAA", "AAA"],
            "Date": ["2024-04-01", "2024-04-02"],
            "Target_Date": ["not-a-date", "2024-04-03"],
            "Open": [10, 11],
            "High": [12, 13],
            "Low": [9, 10],
            "Close": [11, 12],
            "Volume": [100, 110],
        }
    )
    with pytest.raises(ValueError, match="Existing Target_Date"):
        validate_bars(bars)


def test_hmm_output_dates_remain_attested_after_normalizing_ticker():
    days = pd.bdate_range("2024-04-01", periods=14)
    frame = pd.DataFrame(
        {
            "Date": days.strftime("%Y-%m-%d"),
            "Name": [" aapl "] * len(days),
            "MA_20": np.linspace(-1, 1, len(days)),
            "RSI_14": np.linspace(30, 65, len(days)),
            "MACD": np.linspace(-0.3, 0.3, len(days)),
        }
    )
    _, regimes = fit_hmm_on_indicators(frame, n_components=1, random_state=42)
    assert regimes["Name"].eq("AAPL").all()
    require_verified_daily_date_source(regimes)


def test_bootstrap_accepts_independent_modern_session_ordinal_origins():
    days = pd.bdate_range("2024-04-01", periods=5)
    scored = pd.DataFrame(
        {
            "Date": list(days) * 2,
            "Name": ["AAA"] * 5 + ["BBB"] * 5,
            "Session_Number": list(range(5)) + list(range(100, 105)),
        }
    )
    block_ids = _market_date_block_ids(scored, 2, verified_datetime_dates=True)
    assert block_ids[:5].tolist() == block_ids[5:].tolist()


def test_bootstrap_rejects_duplicate_unidentified_date_only_observations():
    scored = pd.DataFrame({"Date": ["2024-04-01", "2024-04-01", "2024-04-02"]})
    with pytest.raises(ValueError, match="Date-only bootstrap"):
        _market_date_block_ids(scored, 2)
