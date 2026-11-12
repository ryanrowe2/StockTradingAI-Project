"""Fourth independent audit: exact states, per-market gaps, numerical budgets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.market_calendar import _scheduled_sessions
from scripts.rl_agent import SimpleTradingEnv
from scripts.session_validation import validate_session_positions
from scripts.walk_forward_evaluation import (
    MAX_BOOTSTRAP_DRAW_CELLS,
    _market_date_block_ids,
    paired_block_accuracy_intervals,
)


def _nyse_dates(count: int = 30) -> list[str]:
    return (
        _scheduled_sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-04-30"))[
            :count
        ]
        .strftime("%Y-%m-%d")
        .tolist()
    )


def _scores(dates) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Date": dates,
            "actual": [int(i % 2) for i in range(n)],
            "majority": [0] * n,
            "baseline_prob": [0.75] * n,
            "enhanced_prob": [0.25] * n,
        }
    )


def test_trading_state_preserves_large_unsigned_integers():
    exact = 2**63 + 11
    bars = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [100.0, 101.0],
            "Close": [100.5, 101.5],
            "State": pd.Series([exact, exact + 1], dtype="uint64"),
        }
    )
    env = SimpleTradingEnv(bars, state_columns=("State",))
    assert env.get_state() == (exact, 0)
    state, _, _, _ = env.step("hold")
    assert state == (exact + 1, 0)


def test_mixed_exchange_missing_session_breaks_paired_blocks():
    london = ["2024-01-16", "2024-01-18", "2024-01-19", "2024-01-22"]
    nyse = ["2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-22"]
    frame = pd.concat(
        [
            _scores(london).assign(Exchange="LSE"),
            _scores(nyse).assign(Exchange="NYSE"),
        ],
        ignore_index=True,
    )
    blocks = _market_date_block_ids(frame, 10)
    per_date = dict(zip(frame["Date"], blocks))
    assert per_date["2024-01-16"] != per_date["2024-01-17"]
    assert per_date["2024-01-17"] == per_date["2024-01-18"]


def test_custom_market_missing_session_breaks_shared_date_blocks():
    data = pd.concat(
        [
            _scores(["2024-01-16", "2024-01-18"]).assign(
                Exchange="XTEST", Session_Number=pd.Series([100, 103], dtype="int64")
            ),
            _scores(["2024-01-16", "2024-01-17", "2024-01-18"]).assign(
                Exchange="NYSE", Session_Number=pd.Series([400, 401, 402], dtype="int64")
            ),
        ],
        ignore_index=True,
    )
    blocks = _market_date_block_ids(data, 10)
    per_date = dict(zip(data["Date"], blocks))
    assert per_date["2024-01-16"] != per_date["2024-01-17"]


def test_exchange_level_metadata_can_share_dates():
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-16", "2024-01-16", "2024-01-17"],
            "Exchange": ["LSE", "NYSE", "NYSE"],
            "Session_Number": [8, 9, 10],
        }
    )
    validate_session_positions(frame)


def test_duplicate_exchange_level_dates_still_rejected():
    frame = pd.DataFrame(
        {"Date": ["2024-01-16", "2024-01-16"], "Exchange": ["NYSE", "NYSE"]}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_session_positions(frame)


def test_bootstrap_work_budget_rejected_before_random_allocation(monkeypatch):
    dates = _nyse_dates(26)
    frame = _scores(dates)
    import scripts.walk_forward_evaluation as evaluation

    monkeypatch.setattr(evaluation, "MAX_BOOTSTRAP_DRAW_CELLS", 200)
    with pytest.raises(ValueError, match="workload"):
        paired_block_accuracy_intervals(
            frame, block_sessions=1, bootstrap_samples=100
        )
    assert MAX_BOOTSTRAP_DRAW_CELLS >= 200


def test_drawdown_overflow_is_reported():
    bars = pd.DataFrame(
        {"Date": ["2024-01-02", "2024-01-03"], "Open": [1.0, 1.0], "Close": [1.0, 1.0]}
    )
    env = SimpleTradingEnv(bars)
    env.equity_curve = [0.0, 1e308, -1e308]
    with pytest.raises(ValueError, match="drawdown must remain finite"):
        env.episode_metrics()


def test_reasonable_bootstrap_still_produces_finite_intervals():
    frame = _scores(_nyse_dates(30))
    result = paired_block_accuracy_intervals(frame, block_sessions=5, bootstrap_samples=100)
    assert all(np.isfinite(values).all() for values in result.values())
