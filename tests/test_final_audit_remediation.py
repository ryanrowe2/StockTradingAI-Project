"""Regression tests for the independent audit's seven remediation findings."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.date_validation import mark_verified_daily_dates
from scripts.data_processing import validate_bars, prepare_features
from scripts.market_calendar import _scheduled_sessions
from scripts.model_training import chronological_split
from scripts.session_validation import (
    adjacent_session_flags,
    session_positions,
    validate_session_positions,
)
from scripts.walk_forward_evaluation import paired_block_accuracy_intervals


def _daily_dates(count: int = 60) -> list[str]:
    return (
        _scheduled_sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-05-31"))[
            :count
        ]
        .strftime("%Y-%m-%d")
        .tolist()
    )


def _split_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": dates[:-1],
            "Target_Date": dates[1:],
            "Name": ["AAPL"] * (len(dates) - 1),
        }
    )


def _scores(dates) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": dates,
            "actual": [0, 1] * (len(dates) // 2),
            "majority": [0] * len(dates),
            "baseline_prob": ["0.2", "0.8"] * (len(dates) // 2),
            "enhanced_prob": ["0.8", "0.2"] * (len(dates) // 2),
        }
    )


def test_split_accepts_day_precision_and_rejects_coerced_monthly_source():
    daily = _split_frame(_daily_dates())
    assert len(chronological_split(daily).train) > 0
    monthly = _split_frame(
        pd.date_range("2019-01-01", periods=50, freq="MS").strftime("%Y-%m-%d").tolist()
    )
    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly["Target_Date"] = pd.to_datetime(monthly["Target_Date"])
    with pytest.raises(ValueError, match="unknown original precision"):
        chronological_split(monthly)


def test_split_validates_target_calendar():
    frame = _split_frame(_daily_dates())
    frame.loc[0, "Target_Date"] = "2024-01-15"
    with pytest.raises(ValueError, match="closed-market date"):
        chronological_split(frame)


def test_split_rejects_unattested_target_datetimes():
    frame = _split_frame(_daily_dates())
    frame["Target_Date"] = pd.to_datetime(frame["Target_Date"])
    with pytest.raises(ValueError, match="unknown original precision"):
        chronological_split(frame)


def test_bootstrap_rejects_month_precision_coercion():
    dates = pd.date_range("2020-01-01", periods=32, freq="MS")
    with pytest.raises(ValueError, match="unknown original precision"):
        paired_block_accuracy_intervals(_scores(dates), bootstrap_samples=100)


def test_bootstrap_rejects_exchange_holidays():
    dates = _daily_dates(32)
    dates[1] = "2024-01-15"
    with pytest.raises(ValueError, match="closed-market date"):
        paired_block_accuracy_intervals(_scores(dates), bootstrap_samples=100)


def test_bootstrap_uses_converted_numeric_string_probabilities():
    dates = _daily_dates(32)
    actual = _scores(dates)
    numeric = actual.copy()
    numeric["baseline_prob"] = numeric["baseline_prob"].astype(float)
    numeric["enhanced_prob"] = numeric["enhanced_prob"].astype(float)
    assert paired_block_accuracy_intervals(
        actual, bootstrap_samples=100, block_sessions=5
    ) == paired_block_accuracy_intervals(
        numeric, bootstrap_samples=100, block_sessions=5
    )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"Date": ["not-a-date"], "Session_Number": [-7]}),
        pd.DataFrame({"Date": ["2024-01-15"], "Session_Number": [7]}),
        pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "Session_Number": [7, 7]}),
    ],
)
def test_session_accessor_rejects_invalid_supplied_metadata(frame):
    with pytest.raises(ValueError):
        session_positions(frame)


def test_session_accessor_checks_explicit_bounds_with_supplied_ordinals():
    frame = pd.DataFrame(
        {"Date": ["2024-01-02", "2024-01-03"], "Session_Number": [7, 8]}
    )
    with pytest.raises(ValueError, match="Calendar bounds"):
        session_positions(frame, calendar_start=pd.Timestamp("2024-01-03"))


def test_custom_exchange_verified_sunday_ordinals_are_supported():
    frame = pd.DataFrame(
        {
            "Name": ["TASE"] * 3,
            "Exchange": ["XTAE"] * 3,
            "Date": ["2024-01-04", "2024-01-07", "2024-01-08"],
            "Session_Number": [400, 401, 402],
        }
    )
    validate_session_positions(frame, require_consecutive=True)
    assert session_positions(frame).tolist() == [400, 401, 402]
    without_ordinals = frame.drop(columns="Session_Number")
    with pytest.raises(ValueError):
        validate_session_positions(without_ordinals)


def test_recognized_market_rejects_sunday_despite_ordinals():
    frame = pd.DataFrame(
        {
            "Exchange": ["NYSE", "NYSE"],
            "Date": ["2024-01-05", "2024-01-07"],
            "Session_Number": [10, 11],
        }
    )
    with pytest.raises(ValueError, match="closed-market date"):
        validate_session_positions(frame)


def test_explicitly_attested_split_and_bootstrap_dates():
    dates = _daily_dates(40)
    split = _split_frame(dates)
    split["Date"] = pd.to_datetime(split["Date"])
    split["Target_Date"] = pd.to_datetime(split["Target_Date"])
    mark_verified_daily_dates(split)
    assert len(chronological_split(split).train) > 0
    scored = mark_verified_daily_dates(_scores(pd.to_datetime(dates[:30])))
    assert "baseline" in paired_block_accuracy_intervals(
        scored, bootstrap_samples=100, block_sessions=5
    )


def test_split_rejects_two_session_target_horizon():
    dates = _daily_dates()
    frame = _split_frame(dates)
    frame["Target_Date"] = dates[2:] + [dates[-1]]
    with pytest.raises(ValueError, match="next trading session"):
        chronological_split(frame)


def test_mutated_verified_target_column_cannot_inherit_original_attestation():
    dates = _daily_dates(40)
    frame = _split_frame(dates)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Target_Date"] = pd.to_datetime(frame["Target_Date"])
    mark_verified_daily_dates(frame)
    frame["Target_Date"] = pd.date_range("2019-01-01", periods=len(frame), freq="MS")
    with pytest.raises(ValueError, match="verification is stale"):
        chronological_split(frame)


def test_unknown_weekday_exchange_requires_verified_ordinals():
    frame = pd.DataFrame(
        {"Exchange": ["XTEST", "XTEST"], "Date": ["2024-01-08", "2024-01-09"]}
    )
    with pytest.raises(ValueError, match="requires verified Session_Number"):
        validate_session_positions(frame)


@pytest.mark.parametrize("exchange", ["", " ", None])
def test_bootstrap_rejects_invalid_exchange_identifiers(exchange):
    scored = _scores(_daily_dates(32))
    scored["Exchange"] = exchange
    with pytest.raises(ValueError, match="Exchange"):
        paired_block_accuracy_intervals(scored, bootstrap_samples=100)


def test_adjacent_session_helper_rejects_fractional_values():
    with pytest.raises(ValueError, match="nonnegative integers"):
        adjacent_session_flags(pd.Series([1.2, 2.2, 3.8]))


@pytest.mark.parametrize("bad", [True, -1, float("nan"), "1"])
def test_adjacent_session_helper_rejects_other_bad_values(bad):
    with pytest.raises(ValueError, match="nonnegative integers"):
        adjacent_session_flags(pd.Series([0, bad], dtype=object))


def test_bootstrap_splits_blocks_at_missing_sessions():
    dates = _daily_dates(40)[::2]
    scored = _scores(dates)
    # Twenty observations and a 30-observation maximum would create only
    # one block with the old observed-date-only grouping algorithm.
    intervals = paired_block_accuracy_intervals(
        scored, block_sessions=30, bootstrap_samples=100
    )
    assert set(intervals) == {"baseline", "enhanced"}


def test_bootstrap_valid_london_dates_not_rejected_as_us_holidays():
    # London was open on Martin Luther King Jr Day, 2024-01-15.
    dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.bdate_range("2024-01-08", periods=32)
    ]
    dates = [d for d in dates if d not in ("2024-02-09",)]
    scored = _scores(dates[:30])
    scored["Exchange"] = "LSE"
    assert "baseline" in paired_block_accuracy_intervals(
        scored, bootstrap_samples=100, block_sessions=5
    )


def test_bootstrap_custom_market_sunday_ordinals():
    dates = pd.date_range("2024-01-07", periods=32, freq="D")
    dates = [d.strftime("%Y-%m-%d") for d in dates if d.dayofweek != 5]
    scored = _scores(dates[:26])
    scored["Exchange"] = "XTAE"
    scored["Session_Number"] = list(range(len(scored)))
    assert "enhanced" in paired_block_accuracy_intervals(
        scored, bootstrap_samples=100, block_sessions=5
    )


def test_prepared_custom_targets_keep_exact_large_integer_ordinals():
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=120)]
    count = len(dates)
    frame = pd.DataFrame(
        {
            "Name": ["CUSTOM"] * count,
            "Exchange": ["XTEST"] * count,
            "Date": dates,
            "Session_Number": pd.Series(
                [2**63 + index for index in range(count)], dtype="uint64"
            ),
            "Open": [100.0 + 0.01 * index for index in range(count)],
            "Close": [100.1 + 0.01 * index for index in range(count)],
            "High": [102.0 + 0.01 * index for index in range(count)],
            "Low": [99.0 + 0.01 * index for index in range(count)],
            "Volume": [1000] * count,
        }
    )
    prepared = prepare_features(validate_bars(frame), verified_datetime_dates=True)
    expected = 2**63 + 34
    assert expected in prepared["Target_Session_Number"].tolist()
    assert len(chronological_split(prepared).test) > 0
