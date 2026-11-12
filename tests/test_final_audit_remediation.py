"""Regression tests for the independent audit's seven remediation findings."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.date_validation import mark_verified_daily_dates
from scripts.market_calendar import _scheduled_sessions
from scripts.model_training import chronological_split
from scripts.session_validation import session_positions, validate_session_positions
from scripts.walk_forward_evaluation import paired_block_accuracy_intervals


def _daily_dates(count: int = 60) -> list[str]:
    return _scheduled_sessions(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-05-31")
    )[:count].strftime("%Y-%m-%d").tolist()


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
        pd.date_range("2019-01-01", periods=50, freq="MS")
        .strftime("%Y-%m-%d")
        .tolist()
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
        pd.DataFrame(
            {"Date": ["2024-01-02", "2024-01-03"], "Session_Number": [7, 7]}
        ),
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
