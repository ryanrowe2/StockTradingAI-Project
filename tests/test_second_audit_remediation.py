"""Regression coverage for the second independent research-integrity audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.date_validation import (
    mark_verified_daily_dates,
    require_verified_daily_date_source,
)
from scripts.market_calendar import _scheduled_sessions
from scripts.model_training import chronological_split
from scripts.session_validation import adjacent_session_flags, validate_session_positions
from scripts.walk_forward_evaluation import paired_block_accuracy_intervals


def _sessions(count: int = 45) -> list[str]:
    return _scheduled_sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-05-31"))[
        :count
    ].strftime("%Y-%m-%d").tolist()


def _scored(dates: list[str], *, exchange=None, ordinals=None) -> pd.DataFrame:
    count = len(dates)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "actual": [i % 2 for i in range(count)],
            "majority": [0] * count,
            "baseline_prob": [0.3] * count,
            "enhanced_prob": [0.7] * count,
        }
    )
    if exchange is not None:
        frame["Exchange"] = exchange
    if ordinals is not None:
        frame["Session_Number"] = ordinals
    return frame


def test_next_session_date_is_required_in_direct_split():
    dates = _sessions()
    two_ahead = pd.DataFrame(
        {
            "Date": dates[:-2],
            "Target_Date": dates[2:],
            "Name": ["AAA"] * (len(dates) - 2),
        }
    )
    with pytest.raises(ValueError, match="next market session"):
        chronological_split(two_ahead)


def test_split_rejects_incorrect_target_ordinal_for_custom_market():
    dates = ["2024-01-04", "2024-01-07", "2024-01-08"] * 16
    days = pd.date_range("2024-01-01", periods=48, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame(
        {
            "Name": ["AAA"] * 47,
            "Exchange": ["CUSTOM"] * 47,
            "Date": days[:-1],
            "Target_Date": days[1:],
            "Session_Number": np.arange(47, dtype=np.int64),
            "Target_Session": np.arange(1, 48, dtype=np.int64),
        }
    )
    del dates
    edited = frame.copy()
    edited.loc[4, "Target_Session"] = 999
    with pytest.raises(ValueError, match="next market session"):
        chronological_split(edited)


def test_provenance_rejects_replaced_target_dates_but_allows_subset():
    dates = _sessions()
    frame = pd.DataFrame(
        {
            "Name": ["AAA"] * 35,
            "Date": pd.to_datetime(dates[:35]),
            "Target_Date": pd.to_datetime(dates[1:36]),
        }
    )
    mark_verified_daily_dates(frame)
    subset = frame.iloc[:20].copy()
    require_verified_daily_date_source(subset, column="Date")
    require_verified_daily_date_source(subset, column="Target_Date")
    frame["Target_Date"] = pd.date_range("2017-01-01", periods=35, freq="MS")
    with pytest.raises(ValueError, match="unknown original precision"):
        require_verified_daily_date_source(frame, column="Target_Date")


def test_custom_market_requires_ordinals_at_standalone_boundary():
    frame = pd.DataFrame(
        {
            "Exchange": ["XTEST", "XTEST"],
            "Date": ["2024-01-08", "2024-01-09"],
        }
    )
    with pytest.raises(ValueError, match="verified Session_Number"):
        validate_session_positions(frame)


def test_bootstrap_valid_london_holiday_and_custom_sunday():
    london = _scored(
        ["2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17"],
        exchange=["LSE"] * 4,
    )
    assert "baseline" in paired_block_accuracy_intervals(
        london, block_sessions=2, bootstrap_samples=100
    )
    custom = _scored(
        ["2024-01-04", "2024-01-07", "2024-01-08", "2024-01-09"],
        exchange=["XTAE"] * 4,
        ordinals=[100, 101, 102, 103],
    )
    assert "enhanced" in paired_block_accuracy_intervals(
        custom, block_sessions=2, bootstrap_samples=100
    )


@pytest.mark.parametrize("identifier", ["", " ", None, "N/A"])
def test_bootstrap_rejects_invalid_exchange_labels(identifier):
    scored = _scored(_sessions(24), exchange=["NYSE"] * 23 + [identifier])
    with pytest.raises(ValueError, match="Exchange"):
        paired_block_accuracy_intervals(scored, bootstrap_samples=100)


def test_bootstrap_rejects_unverified_custom_market():
    scored = _scored(_sessions(24), exchange=["XTAE"] * 24)
    with pytest.raises(ValueError, match="verified Session_Number"):
        paired_block_accuracy_intervals(scored, bootstrap_samples=100)


def test_bootstrap_session_blocks_respect_missing_calendar_sessions():
    # Twenty observations, each separated by a full week, cannot comprise two
    # contiguous 10-session resampling blocks.
    dates = _sessions(45)[::2]
    scored = _scored(dates)
    with pytest.raises(ValueError, match="at least two chronological date blocks"):
        paired_block_accuracy_intervals(scored, block_sessions=10, bootstrap_samples=100)


@pytest.mark.parametrize(
    "positions",
    [
        pd.Series([1.2, 2.2, 3.8]),
        pd.Series([0, -1]),
        pd.Series([True, False]),
        pd.Series([0, np.nan]),
    ],
)
def test_adjacency_rejects_invalid_unvalidated_ordinals(positions):
    with pytest.raises(ValueError, match="nonnegative integer"):
        adjacent_session_flags(positions)
