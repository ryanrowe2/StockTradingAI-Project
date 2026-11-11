"""Regression coverage for authoritative exchange-aware daily calendar inference."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.data_processing import (
    load_data,
    prepare_indicator_history,
    validate_bars,
)
from scripts.market_calendar import (
    exchange_open_mask,
    inferred_equity_sessions,
)
from scripts.rl_agent import SimpleTradingEnv
from scripts.session_validation import session_positions, validate_session_positions
from scripts.technical_indicators import add_technical_indicators


def _bars(days, *, exchange=None):
    rows = {
        "Name": ["AAPL"] * len(days),
        "Date": days,
        "Open": [10.0] * len(days),
        "High": [13.0] * len(days),
        "Low": [9.0] * len(days),
        "Close": [10.5 + n * 0.4 for n in range(len(days))],
        "Volume": [100] * len(days),
    }
    if exchange is not None:
        rows["Exchange"] = [exchange] * len(days)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("label", ["LSE", "XLON"])
def test_london_boxing_day_is_not_an_inferred_trading_session(label):
    source = _bars(["2024-12-24", "2024-12-27"], exchange=label)
    validated = validate_bars(source)
    assert validated["Session_Number"].tolist() == [0, 1]
    history = prepare_indicator_history(validated, verified_datetime_dates=True)
    assert history["Return"].iloc[1] == pytest.approx(10.9 / 10.5 - 1)
    indicators = add_technical_indicators(source)
    assert indicators["Session_Number"].tolist() == [0, 1]
    assert indicators["Return"].iloc[1] == pytest.approx(10.9 / 10.5 - 1)
    assert len(SimpleTradingEnv(source).data) == 2
    assert inferred_equity_sessions(
        pd.DatetimeIndex(source["Date"]), exchange=label
    ).strftime("%Y-%m-%d").tolist() == ["2024-12-24", "2024-12-27"]


@pytest.mark.parametrize("label", ["LSE", "XLON"])
def test_london_closed_days_cannot_be_fabricated_as_daily_observations(label):
    source = _bars(["2024-12-24", "2024-12-26"], exchange=label)
    with pytest.raises(ValueError, match="closed-market date"):
        validate_bars(source)
    with pytest.raises(ValueError, match="closed-market date"):
        add_technical_indicators(source)
    with pytest.raises(ValueError, match="closed-market date"):
        SimpleTradingEnv(source)
    with pytest.raises(ValueError, match="closed-market date"):
        session_positions(source)
    with pytest.raises(ValueError, match="closed-market date"):
        inferred_equity_sessions(pd.DatetimeIndex(source["Date"]), exchange=label)


@pytest.mark.parametrize("with_ordinals", [False, True])
def test_unlabeled_us_equities_reject_fabricated_mlk_holiday(with_ordinals):
    source = _bars(["2024-01-12", "2024-01-15", "2024-01-16"])
    if with_ordinals:
        source["Session_Number"] = [0, 1, 2]
    with pytest.raises(ValueError, match="closed-market date"):
        validate_bars(source)
    with pytest.raises(ValueError, match="closed-market date"):
        add_technical_indicators(source)
    with pytest.raises(ValueError, match="closed-market date"):
        validate_session_positions(source)
    with pytest.raises(ValueError, match="closed-market date"):
        SimpleTradingEnv(source)


def test_unlabeled_us_equities_reject_fabricated_mlk_holiday_in_loader(tmp_path):
    source = _bars(["2024-01-12", "2024-01-15", "2024-01-16"])
    path = tmp_path / "fabricated_us_holiday.csv"
    source.to_csv(path, index=False)
    with pytest.raises(ValueError, match="closed-market date"):
        load_data(path)


def test_direct_inference_rejects_weekends_and_missing_values():
    for date in ("2024-01-07", "2024-01-06", pd.NaT):
        with pytest.raises(ValueError, match="nonmissing weekday"):
            inferred_equity_sessions(pd.DatetimeIndex([date]))


def test_custom_exchange_remains_available_with_verified_ordinals():
    source = _bars(["2024-01-12", "2024-01-15", "2024-01-16"], exchange="XTEST")
    source["Session_Number"] = [100, 101, 102]
    assert validate_bars(source)["Session_Number"].tolist() == [100, 101, 102]
    assert exchange_open_mask(pd.DatetimeIndex(["2024-01-15"]), "XTEST").all()


def test_invalid_london_peer_holiday_is_not_calendar_evidence(tmp_path):
    source = pd.concat(
        [
            _bars(["2024-12-24", "2024-12-27"], exchange="LSE"),
            _bars(["2024-12-26"], exchange="LSE").assign(Name="OTHER"),
        ],
        ignore_index=True,
    )
    path = tmp_path / "lse_shared_dates.csv"
    source.to_csv(path, index=False)
    selected = load_data(path, tickers=("AAPL",))
    assert selected["Session_Number"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="closed-market date"):
        load_data(path, tickers=("OTHER",))
