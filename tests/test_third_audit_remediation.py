"""Third audit: historical-session integrity and bounded gap-aware bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.date_validation import mark_verified_daily_dates
from scripts.market_calendar import exchange_open_mask
from scripts.session_validation import validate_session_positions
from scripts.walk_forward_evaluation import (
    MAX_BOOTSTRAP_SAMPLES,
    _market_date_block_ids,
    paired_block_accuracy_intervals,
)


def _scores(dates, *, exchange="NYSE", ordinals=None):
    count = len(dates)
    result = pd.DataFrame(
        {
            "Date": dates,
            "actual": [i % 2 for i in range(count)],
            "majority": [0] * count,
            "baseline_prob": [0.6] * count,
            "enhanced_prob": [0.4] * count,
            "Exchange": [exchange] * count,
        }
    )
    if ordinals is not None:
        result["Session_Number"] = ordinals
    return result


def test_historical_known_weekdays_without_ordinals_rejected():
    frame = pd.DataFrame(
        {"Date": ["1968-07-09", "1968-07-11"], "Exchange": ["NYSE", "NYSE"]}
    )
    with pytest.raises(ValueError, match="before 1971"):
        validate_session_positions(frame)


def test_historical_nyse_ordinal_backed_bootstrap():
    frame = _scores(
        ["1951-01-05", "1951-01-06", "1951-01-08", "1951-01-09"],
        ordinals=[400, 401, 402, 403],
    )
    result = paired_block_accuracy_intervals(
        frame, block_sessions=2, bootstrap_samples=100
    )
    assert set(result) == {"baseline", "enhanced"}


def test_historical_london_does_not_inherit_nyse_saturday_cutoff():
    dates = pd.DatetimeIndex(["1920-06-05", "1932-06-04"])
    mask = exchange_open_mask(dates, "LSE", verified_session_ordinals=True)
    assert mask.tolist() == [False, True]


def test_sparse_market_data_produces_multiple_blocks():
    all_sessions = pd.bdate_range("2024-01-02", periods=22)
    sparse = [d.strftime("%Y-%m-%d") for d in all_sessions[::2]]
    frame = _scores(sparse)
    blocks = _market_date_block_ids(frame, block_sessions=20)
    assert len(np.unique(blocks)) > 1


def test_large_sample_count_rejected_before_allocation():
    frame = _scores(["2024-01-02", "2024-01-03"])
    with pytest.raises(ValueError, match="between 100"):
        paired_block_accuracy_intervals(
            frame, bootstrap_samples=MAX_BOOTSTRAP_SAMPLES + 1
        )


def test_batched_bootstrap_remains_deterministic():
    from scripts.market_calendar import _scheduled_sessions

    dates = (
        _scheduled_sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-15"))[:35]
        .strftime("%Y-%m-%d")
        .tolist()
    )
    frame = _scores(dates)
    one = paired_block_accuracy_intervals(
        frame, block_sessions=5, bootstrap_samples=150, seed=21
    )
    two = paired_block_accuracy_intervals(
        frame, block_sessions=5, bootstrap_samples=150, seed=21
    )
    assert one == two


def test_verified_historical_datetime_provenance():
    frame = _scores(pd.to_datetime(["1951-01-05", "1951-01-06", "1951-01-08", "1951-01-09"]), ordinals=[400, 401, 402, 403])
    mark_verified_daily_dates(frame)
    assert len(np.unique(_market_date_block_ids(frame, 2))) == 2
