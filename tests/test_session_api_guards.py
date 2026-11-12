"""Regression tests for direct session API date provenance and calendar bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.date_validation import mark_verified_daily_dates
from scripts.market_calendar import inferred_equity_sessions
from scripts.session_validation import session_positions, validate_session_positions


@pytest.mark.parametrize("supplied_ordinals", [False, True])
def test_session_positions_rejects_unverified_coerced_months(supplied_ordinals):
    source = pd.DataFrame(
        {
            "Name": ["AAPL", "AAPL"],
            "Date": [np.datetime64("2024-04"), np.datetime64("2024-05")],
        }
    )
    if supplied_ordinals:
        source["Session_Number"] = [100, 122]
    assert pd.api.types.is_datetime64_any_dtype(source["Date"].dtype)
    with pytest.raises(ValueError, match="unknown original precision"):
        session_positions(source)


@pytest.mark.parametrize("supplied_ordinals", [False, True])
def test_validate_session_positions_rejects_unverified_coerced_months(
    supplied_ordinals,
):
    source = pd.DataFrame(
        {
            "Name": ["AAPL", "AAPL"],
            "Date": [np.datetime64("2024-04"), np.datetime64("2024-05")],
        }
    )
    if supplied_ordinals:
        source["Session_Number"] = [100, 122]
    with pytest.raises(ValueError, match="unknown original precision"):
        validate_session_positions(source)


def test_standalone_helpers_accept_attested_day_precise_dates():
    frame = pd.DataFrame(
        {
            "Name": ["AAPL", "AAPL"],
            "Date": pd.to_datetime(["2024-04-01", "2024-04-02"]),
            "Session_Number": [20, 21],
        }
    )
    with pytest.raises(ValueError, match="unknown original precision"):
        session_positions(frame)
    with pytest.raises(ValueError, match="unknown original precision"):
        validate_session_positions(frame)
    assert session_positions(frame, verified_datetime_dates=True).tolist() == [20, 21]
    validate_session_positions(frame, verified_datetime_dates=True)
    verified = mark_verified_daily_dates(frame.copy())
    assert session_positions(verified).tolist() == [20, 21]
    validate_session_positions(verified)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-04-02", None),
        (None, "2024-04-01"),
        ("2024-04-03", "2024-04-01"),
        ("2024-04-01", pd.NaT),
    ],
)
def test_session_positions_rejects_out_of_range_calendar(start, end):
    frame = pd.DataFrame(
        {
            "Name": ["AAPL", "AAPL"],
            "Date": ["2024-04-01", "2024-04-02"],
        }
    )
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="Calendar bounds"):
        session_positions(
            frame,
            calendar_start=pd.Timestamp(start) if start is not None else None,
            calendar_end=pd.Timestamp(end) if end is not None else None,
        )
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-04-02", None),
        (None, "2024-04-01"),
        ("2024-04-03", "2024-04-01"),
        (pd.NaT, None),
    ],
)
def test_direct_market_inference_rejects_out_of_range_calendar(start, end):
    dates = pd.DatetimeIndex(["2024-04-01", "2024-04-02"])
    with pytest.raises(ValueError, match="Calendar bounds"):
        inferred_equity_sessions(
            dates,
            start=pd.Timestamp(start) if start is not None else None,
            end=pd.Timestamp(end) if end is not None else None,
        )


def test_valid_bounds_preserve_expected_market_positions():
    frame = pd.DataFrame(
        {
            "Name": ["AAPL", "AAPL"],
            "Date": ["2024-04-01", "2024-04-02"],
        }
    )
    positions = session_positions(
        frame,
        calendar_start=pd.Timestamp("2024-03-27"),
        calendar_end=pd.Timestamp("2024-04-03"),
    )
    assert positions.tolist() == [2, 3]
    assert (positions >= 0).all()


def test_validated_synthetic_timestamps_do_not_accept_intraday_dates():
    frame = mark_verified_daily_dates(
        pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-04-01 09:30:00", "2024-04-01 16:00:00"]),
            }
        )
    )
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        session_positions(frame)
    with pytest.raises(ValueError, match="parseable ISO calendar dates"):
        validate_session_positions(frame)
