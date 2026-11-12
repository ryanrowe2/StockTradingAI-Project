"""Eighth audit: source identity, peer isolation, historical closures and gap policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.data_processing import load_data
from scripts.market_calendar import nyse_open_mask, validate_nyse_open_dates
from scripts.technical_indicators import add_technical_indicators, compute_macd
from scripts.market_calendar import inferred_equity_sessions


def _write_custom_source(tmp_path, names, dates, exchanges=None):
    n = len(names)
    source = pd.DataFrame(
        {
            "Date": dates,
            "Name": names,
            "Open": [10.0] * n,
            "High": [12.0] * n,
            "Low": [9.0] * n,
            "Close": [11.0] * n,
            "Volume": [100] * n,
            **({"Exchange": exchanges} if exchanges is not None else {}),
        }
    )
    path = tmp_path / "custom_equity_prices.csv"
    source.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("symbol", ["aAa", " AAA ", "aAa "])
def test_custom_selected_ticker_normalization_keeps_daily_attestation(tmp_path, symbol):
    path = _write_custom_source(
        tmp_path,
        names=[symbol, symbol, "BBB", "BBB"],
        dates=["2024-01-09", "2024-01-10", "2024-01-09", "2024-01-10"],
        exchanges=["nyse", " NYSE ", "NYSE", "NYSE"],
    )
    selected = load_data(path, tickers=("AAA",))
    assert selected["Name"].tolist() == ["AAA", "AAA"]
    assert selected["Session_Number"].tolist() == [0, 1]


def test_duplicate_unrelated_peer_does_not_block_valid_selected_ticker(tmp_path):
    path = _write_custom_source(
        tmp_path,
        names=["AAA", "AAA", "BBB", " BBB ", "BBB"],
        dates=[
            "2024-01-09",
            "2024-01-11",
            "2024-01-09",
            "2024-01-09",
            "2024-01-10",
        ],
        exchanges=["NYSE", "NYSE", "XNYS", "NYSE", "NYSE"],
    )
    selected = load_data(path, tickers=("AAA",))
    assert selected["Session_Number"].tolist() == [0, 2]
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        load_data(path)


def test_selected_ticker_duplicates_remain_fatal(tmp_path):
    path = _write_custom_source(
        tmp_path,
        names=["AAA", " aaa ", "BBB"],
        dates=["2024-01-09", "2024-01-09", "2024-01-10"],
        exchanges=["NYSE"] * 3,
    )
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        load_data(path, tickers=("AAA",))


@pytest.mark.parametrize(
    "closed",
    [
        "1954-12-24",
        "1958-12-26",
        "1961-05-29",
        "1963-11-25",
        "1963-12-25",
        "1963-11-05",
        "1965-12-24",
        "1966-02-22",
    ],
)
def test_verified_historical_ordinals_cannot_authorize_closed_nyse_days(closed):
    date = pd.DatetimeIndex([closed])
    assert not nyse_open_mask(date, verified_session_ordinals=True).any()
    with pytest.raises(ValueError, match="closed-market date"):
        validate_nyse_open_dates(date, verified_session_ordinals=True)


def test_historical_regular_holidays_respect_pre_1971_rules():
    # February 22 was Washington's Birthday before the Monday observance.
    # February 21, 1966 was an ordinary Monday and remains a candidate
    # (with independent verified historical source-session metadata).
    candidate = pd.DatetimeIndex(["1966-02-21", "1966-02-22"])
    assert nyse_open_mask(candidate, verified_session_ordinals=True).tolist() == [
        True, False
    ]
    assert not nyse_open_mask(candidate).any()


def test_gap_policy_for_observation_windows_and_adjacent_returns():
    sessions = inferred_equity_sessions(
        pd.DatetimeIndex(["2024-01-02", "2024-03-29"])
    )
    dates = sessions.delete(23)
    prices = pd.Series(np.linspace(100.0, 150.0, len(dates)))
    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Name": ["AAA"] * len(dates),
            "Open": prices.to_numpy(),
            "High": prices.to_numpy() + 1,
            "Low": prices.to_numpy() - 1,
            "Close": prices.to_numpy(),
            "Volume": [1000] * len(dates),
        }
    )
    indicators = add_technical_indicators(frame)
    missing_boundary = 23
    assert pd.isna(indicators["Return"].iloc[missing_boundary])
    assert pd.isna(indicators["RSI_14"].iloc[missing_boundary])
    # Longer-horizon observation indicators intentionally carry only past
    # valid prices across a missing session; they are not 20 contiguous days.
    expected_ma = indicators["Close"].rolling(20, min_periods=20).mean()
    pd.testing.assert_series_equal(
        indicators["MA_20"], expected_ma, check_names=False
    )
    expected_macd, _ = compute_macd(indicators)
    pd.testing.assert_series_equal(
        indicators["MACD"], expected_macd, check_names=False
    )
    assert pd.notna(indicators["MA_20"].iloc[missing_boundary])
